#!/usr/bin/env python
"""Iteration 35: UNSUPERVISED manifold alignment ("neural stitching") -- rescue drifted sessions
without labels.

THE GAP: LOG-082 showed label-free calibration (BatchNorm-stat adaptation) gives only +0.02.
But BN only re-fits MEANS and VARIANCES of activations. When electrodes drift, what changes is
WHICH NEURONS each channel samples -- i.e. a ROTATION / re-mixing of the input space. BN cannot
undo a rotation; a subspace alignment can. Our own detector says this is the disease: the failing
sessions retain only 0.16-0.47 channel overlap (LOG-085/086) -- the electrodes moved.

THEORY: population activity lives on a low-D manifold that is preserved across days/years even as
individual electrodes churn (Gallego et al. 2020, Nat Neurosci). Degenhart et al. 2020 (Nat Biomed
Eng) stabilized a BCI for MONTHS with no recalibration by aligning that low-D space.

KEY ENABLER HERE: the N electrodes are the SAME PHYSICAL CHANNELS across sessions, so the pool's
and the session's principal subspaces live in the SAME N-dim space -> a proper orthogonal
Procrustes is well posed (no correspondence problem).

Methods (all LABEL-FREE; fitted on the session's FIRST half = the observation window, scored on
the SECOND half):
  zero_shot : frozen pool model, no adaptation                            (baseline)
  coral     : match full covariance -- M = C_pool^(1/2) @ C_new^(-1/2)    (Sun & Saenko 2016)
  procrustes_k : manifold stitching -- W_pool, W_new = top-k PCs; R = argmin||W_new R - W_pool||
              (orthogonal Procrustes via SVD); M = W_pool @ R.T @ W_new.T
Both produce an NxN linear map applied to the session's z-scored counts BEFORE the normal feature
pipeline (counts + causal EWMA). EWMA is a linear time filter so it commutes with a channel-space
map -- order does not matter.

Usage: py research/iter35_manifold_align.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.harness as H
import research.iter20_multiscale as I20
import research.iter25_causal_smoothing as I25
import research.iter27_fresh_session as I27
import research.iter28_calibration as I28
import research.iter32_forward_split as I32
import models.tcn_gru.evaluate as E

SEED = 42
N_CH = 32
KS = [4, 8, 16, 24]
SOBI_LAG = 1                        # 1 bin = 40 ms; velocity dynamics are strongly autocorrelated                 # latent dims for the stitching
POOL = I32.POOL                     # Sep 15 - Dec 20 2016 (20 sessions)
TARGETS = I27.FRESH + I32.FORWARD   # 4 backward (Apr-Jun) + 4 forward (Jan)
ALPHA = 0.1
WIN = I20.WIN
OUT = ROOT / "results" / "metrics" / "iter35_manifold_align.json"


def zcounts(counts, s, chans):
    """Per-session z-scored raw counts for the selected channels: (N, T)."""
    c = counts[s][0][chans].astype(np.float64)
    mu, sd = c.mean(1, keepdims=True), c.std(1, keepdims=True) + 1e-9
    return (c - mu) / sd


def sqrtm_psd(C, eps=1e-6):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, eps, None)
    return V @ np.diag(np.sqrt(w)) @ V.T


def invsqrtm_psd(C, eps=1e-6):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, eps, None)
    return V @ np.diag(1.0 / np.sqrt(w)) @ V.T


def top_pcs(X, k):
    """X: (N, T) z-scored. Return top-k eigenvectors of the channel covariance: (N, k)."""
    C = np.cov(X)
    w, V = np.linalg.eigh(C)
    return V[:, np.argsort(w)[::-1][:k]]


def sobi(A, k, lag=1):
    """Second-Order Blind Identification: whiten, then diagonalise the TIME-LAGGED covariance.

    Why this and not PCA: cov(X) = B cov(Z) B^T, so with a non-orthogonal mixing B the channel-space
    PCs do NOT correspond to latents -- the i-th PC of two sessions encode different latent mixes,
    and second-order (instantaneous) statistics cannot break that rotation ambiguity. Verified: on
    ideal synthetic data (same latents, drifted mixing) PCA-Procrustes gives ZERO gain and CORAL
    hurts. TIME-LAGGED covariance does break it -- each latent has a characteristic autocorrelation
    timescale, which is a property of the DYNAMICS and survives electrode drift. Synthetic check:
    SOBI recovers |corr|=1.00 vs 0.31 unaligned, with identical autocorr spectra across sessions.

    Returns (unmixing (k,N), autocorr eigenvalues (k,) sorted desc)."""
    C0 = np.cov(A)
    w, V = np.linalg.eigh(C0)
    idx = np.argsort(w)[::-1][:k]
    Wh = (V[:, idx] * (1.0 / np.sqrt(np.clip(w[idx], 1e-9, None)))).T       # (k,N) whitener
    Aw = Wh @ A
    Ct = (Aw[:, :-lag] @ Aw[:, lag:].T) / (A.shape[1] - lag)
    Ct = 0.5 * (Ct + Ct.T)
    ev, U = np.linalg.eigh(Ct)
    o = np.argsort(ev)[::-1]                     # order by autocorrelation = the drift-invariant
    return (U[:, o].T @ Wh), ev[o]


def windows_from_counts(cz, vel, axes):
    """Build the normal feature pipeline (counts + causal EWMA -> z-score -> windows)."""
    f = np.concatenate([cz, I25.ewma(cz.astype(np.float32), ALPHA)], 0)
    mu, sd = f.mean(1, keepdims=True), f.std(1, keepdims=True) + 1e-6
    fz = ((f - mu) / sd).astype(np.float32)
    return [{"e": fz[:, k * WIN:(k + 1) * WIN], "vel": vel[k * WIN:(k + 1) * WIN][:, axes]}
            for k in range(fz.shape[1] // WIN)]


def main():
    import torch
    t0 = time.time()
    counts = I27.load_counts_full(POOL + list(E.EVAL) + list(E.TEST) + TARGETS)
    axes = np.sort(np.argsort(np.mean([counts[s][1].std(0) for s in E.TRAIN], 0))[-2:])
    fr = np.mean([counts[s][0].mean(1) for s in POOL], 0)
    chans = np.sort(np.argsort(fr)[-N_CH:])
    print("=== Iteration 35: unsupervised manifold alignment (neural stitching) ===")
    print(f"    {N_CH} ch; pool {len(POOL)} sessions; targets {len(TARGETS)} (4 backward + 4 forward)")
    print(f"    loaded {len(counts)} sessions [{time.time()-t0:.0f}s]\n", flush=True)

    # --- pool model (frozen) ---
    tr_by = I27.build(counts, chans, axes, POOL)
    data = {"train": [x for s in POOL for x in tr_by[s]],
            "eval": I27.build(counts, chans, axes, list(E.EVAL)),
            "test": I27.build(counts, chans, axes, list(E.TEST))}
    t1 = time.time()
    res = H.run(data, I20.CAUSAL, seeds=(SEED,), ret_net=True)
    net, (ym, ys) = res["net"], res["norm"]
    print(f"  pool model: EVAL {res['eval_r2']:.3f} [{time.time()-t1:.0f}s]\n", flush=True)

    def predict(X):
        net.eval()
        with torch.no_grad():
            return net(torch.tensor(X)).numpy() * ys + ym

    # --- pool reference statistics (z-scored counts pooled across sessions) ---
    Xp = np.concatenate([zcounts(counts, s, chans) for s in POOL], axis=1)   # (N, T_total)
    C_pool = np.cov(Xp)
    C_pool_sqrt = sqrtm_psd(C_pool)
    W_pool_full = {k: top_pcs(Xp, k) for k in KS}
    # SOBI reference: pool unmixing + its mixing (for sign matching) + autocorr spectrum
    sobi_pool = {}
    for k in KS:
        Ux, ex = sobi(Xp, k, lag=SOBI_LAG)
        sobi_pool[k] = (Ux, np.linalg.pinv(Ux), ex)     # unmixing (k,N), mixing (N,k), spectrum
    print("  pool SOBI autocorr spectra (the drift-invariant that gives correspondence):")
    for k in KS:
        print(f"    k={k:2d}: {np.round(sobi_pool[k][2], 3)}")
    print(flush=True)

    rows = {}
    for s in TARGETS:
        cz = zcounts(counts, s, chans)                       # (N, T)
        vel = counts[s][1]
        T_half = (cz.shape[1] // WIN) // 2 * WIN             # observation window = first half
        obs = cz[:, :T_half]
        era = "backward" if s in I27.FRESH else "forward"

        def score_with(cz_mapped):
            w = windows_from_counts(cz_mapped, vel, axes)
            half = len(w) // 2
            Xt, Yt = I28.stack(w[half:])
            return I28.score(Yt.reshape(-1, 2), predict(Xt).reshape(-1, 2))

        out = {"zero_shot": score_with(cz)}
        # CORAL: whiten with the session's covariance, recolour with the pool's
        M_coral = C_pool_sqrt @ invsqrtm_psd(np.cov(obs))
        out["coral"] = score_with(M_coral @ cz)
        # Procrustes stitching at several latent dims (expected to fail -- see sobi() docstring)
        for k in KS:
            W_new = top_pcs(obs, k)
            U, _, Vt = np.linalg.svd(W_new.T @ W_pool_full[k])
            R = U @ Vt                                        # (k,k) orthogonal
            M = W_pool_full[k] @ R.T @ W_new.T                # (N,N)
            out[f"procrustes_k{k}"] = score_with(M @ cz)
        # SOBI temporal stitching: order latents by their autocorrelation (drift-invariant), then
        # sign-match via the mixing vectors. NOTE: unlike the synthetic check we CANNOT correlate
        # latents across sessions (different recordings, no temporal correspondence), so signs come
        # from the channel loadings -- legitimate because the electrodes are the same physical ones.
        for k in KS:
            Ux, Mx, ex = sobi_pool[k]
            Uy, ey = sobi(obs, k, lag=SOBI_LAG)
            My = np.linalg.pinv(Uy)                           # (N,k) session mixing
            sgn = np.sign(np.sum(Mx * My, axis=0))            # per-latent sign from loadings
            sgn[sgn == 0] = 1.0                               # (NB: not `s` -- that is the session name)
            M = Mx @ np.diag(sgn) @ Uy                        # (N,N) session -> pool-aligned
            out[f"sobi_k{k}"] = score_with(M @ cz)
            if k == KS[1]:
                out[f"_spec_match_k{k}"] = (float(np.corrcoef(ex, ey)[0, 1]), 0.0)

        rows[s] = {"era": era, **{k: {"r2": v[0], "r": v[1]} for k, v in out.items()}}
        base = out["zero_shot"][0]
        best = max((k for k in out if k != "zero_shot" and not k.startswith("_")), key=lambda k: out[k][0])
        print(f"  {s:20s} [{era:8s}] zero_shot R2={base:+.3f}")
        for k, v in out.items():
            if k == "zero_shot" or k.startswith("_"):
                continue
            print(f"       {k:<16s} R2={v[0]:+.3f}/r={v[1]:.3f}  ({v[0]-base:+.3f})")
        print(f"       -> best: {best} ({out[best][0]-base:+.3f})\n", flush=True)

    # summary: focus on the sessions that actually failed
    bad = [s for s in TARGETS if rows[s]["zero_shot"]["r2"] < 0.4]
    print(f"  === MEANS over all {len(TARGETS)} targets ===")
    methods = (["zero_shot", "coral"] + [f"procrustes_k{k}" for k in KS]
               + [f"sobi_k{k}" for k in KS])
    for m in methods:
        mu = float(np.mean([rows[s][m]["r2"] for s in TARGETS]))
        mb = float(np.mean([rows[s][m]["r2"] for s in bad])) if bad else float("nan")
        print(f"    {m:<16s} all {mu:+.3f}   |  BAD sessions ({len(bad)}) {mb:+.3f}")
    rows["_bad_sessions"] = bad
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
