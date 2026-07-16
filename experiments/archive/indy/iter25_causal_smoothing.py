#!/usr/bin/env python
"""Iteration 25: LEAKAGE-FREE causal-vs-noncausal input smoothing comparison.

Discovery (2026-07-14): the "strictly causal" pipeline was NOT causal. The input
firing rates are smoothed with scipy `gaussian_filter1d(sigma=1)`, which is
CENTERED/symmetric -- at 40 ms bins it pulls ~30% of its weight from FUTURE bins
(24% from t+1 alone). That future leak is baked into the cached rates and into
every prior "0 ms lookahead" number (0.606/0.618/0.646). This settles the honest,
strictly-causal result and picks the best TRULY causal input.

We never previously tested unsmoothed counts, EWMA on raw counts, or a causal
(one-sided) Gaussian. Here we build every feature from UNSMOOTHED binned spike
counts (RATE_SIGMA=0, read from .mat, bypassing the smoothed cache) and compare:

  causal (deployable):                              non-causal (reference only):
    counts                       (8)                  centered_gauss           (8)
    counts + causal EWMA         (16)                 centered_gauss + EWMA    (16)  <- current pipeline
    causal_gauss (one-sided)     (8)
    causal_gauss + causal EWMA   (16)
    counts + 3 causal EWMAs      (32)

Selection is on EVAL (alpha=0.1, the eval-valid slow scale; LOG-073). 3-seed,
strictly-causal TCN+GRU, 8 ch, 24 sess. Usage: py research/iter25_causal_smoothing.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.harness as H
import research.iter20_multiscale as I20
import models.tcn_gru.evaluate as E

SEEDS = (42, 1, 7)
CH = I20.CHANNELS                 # the 8 deployed firing channels
WIN = I20.WIN
ALPHA = 0.1                       # eval-valid slow EWMA scale (LOG-073)
POOL = list(E.TRAIN) + __import__("research.iter7_final", fromlist=["EXTRA18"]).EXTRA18


# --- feature transforms on UNSMOOTHED counts (8, T) ---
def centered_gauss(r, sigma=1.0):
    """scipy default: symmetric -> uses future (the current, non-causal pipeline)."""
    return gaussian_filter1d(r, sigma, axis=1)


def causal_gauss(r, sigma=1.0, truncate=4.0):
    """One-sided (past-only) half-Gaussian FIR; edge-renormalized. Strictly causal."""
    radius = int(truncate * sigma + 0.5)
    w = np.exp(-(np.arange(radius + 1) ** 2) / (2 * sigma ** 2))
    w = w / w.sum()
    T = r.shape[1]
    out = np.zeros_like(r, dtype=np.float64)
    norm = np.zeros(T)
    for i, wi in enumerate(w):                     # lag i into the PAST only
        out[:, i:] += wi * r[:, :T - i]
        norm[i:] += wi
    return (out / norm).astype(np.float32)


def ewma(r, alpha):
    o = r.astype(np.float64).copy()
    for t in range(1, r.shape[1]):
        o[:, t] = alpha * r[:, t] + (1 - alpha) * o[:, t - 1]
    return o.astype(np.float32)


def cat(*arrs):
    return np.concatenate(arrs, 0)


FEATS = {
    # label:               (causal?, fn(counts 8xT) -> feats DxT)
    "counts":              (True,  lambda r: r.astype(np.float32)),
    "counts+ewma":         (True,  lambda r: cat(r, ewma(r, ALPHA))),
    "causal_gauss":        (True,  lambda r: causal_gauss(r)),
    "causal_gauss+ewma":   (True,  lambda r: cat(causal_gauss(r), ewma(causal_gauss(r), ALPHA))),
    "counts+3ewma":        (True,  lambda r: cat(r, ewma(r, 0.5), ewma(r, 0.2), ewma(r, 0.1))),
    "centered_gauss":      (False, lambda r: centered_gauss(r)),
    "centered_gauss+ewma": (False, lambda r: cat(centered_gauss(r), ewma(centered_gauss(r), ALPHA))),
}
ORDER = ["counts", "counts+ewma", "causal_gauss", "causal_gauss+ewma", "counts+3ewma",
         "centered_gauss", "centered_gauss+ewma"]


def load_counts(names):
    """Unsmoothed binned counts (RATE_SIGMA=0) straight from .mat, bypassing cache."""
    saved = E.RATE_SIGMA
    E.RATE_SIGMA = 0.0
    try:
        cache = {}
        for s in names:
            r, v = E.load_source_electrode(s)      # r = raw counts (no Gaussian)
            cache[s] = (r[CH].astype(np.float32), v)
        return cache
    finally:
        E.RATE_SIGMA = saved


def make_prep(fn, axes, counts):
    def wins(s):
        feat = fn(counts[s][0])                    # (D, T)
        mu, sd = feat.mean(1, keepdims=True), feat.std(1, keepdims=True) + 1e-6
        fz = ((feat - mu) / sd).astype(np.float32)
        v = counts[s][1]
        return [{"e": fz[:, k * WIN:(k + 1) * WIN], "vel": v[k * WIN:(k + 1) * WIN][:, axes]}
                for k in range(fz.shape[1] // WIN)]
    tr = [x for s in POOL for x in wins(s)]
    return {"train": tr, "eval": {s: wins(s) for s in E.EVAL},
            "test": {s: wins(s) for s in E.TEST}}


def main():
    print("=== Iteration 25: leakage-free causal smoothing comparison (3-seed) ===")
    print("    building features from UNSMOOTHED counts; select on EVAL.\n", flush=True)
    t0 = time.time()
    counts = load_counts(POOL + list(E.EVAL) + list(E.TEST))
    axes = np.sort(np.argsort(np.mean([counts[s][1].std(0) for s in E.TRAIN], 0))[-2:])
    print(f"    loaded {len(counts)} sessions of raw counts [{time.time()-t0:.0f}s]\n", flush=True)

    rows = {}
    for name in ORDER:
        causal, fn = FEATS[name]
        t1 = time.time()
        res = H.run(make_prep(fn, axes, counts), I20.CAUSAL, seeds=SEEDS)
        rows[name] = {"eval_r2": res["eval_r2"], "test_r2": res["test_r2"],
                      "causal": causal, "n_feat": res["n_params"]}
        flag = "causal " if causal else "NONcaus"
        print(f"  {name:20s} [{flag}] EVAL R2={res['eval_r2']:.3f}  TEST R2={res['test_r2']:.3f}  "
              f"[{time.time()-t1:.0f}s]", flush=True)

    causal_rows = {k: v for k, v in rows.items() if v["causal"]}
    best = max(causal_rows.values(), key=lambda r: r["eval_r2"])
    best_name = [k for k, v in causal_rows.items() if v is best][0]
    cur = rows["centered_gauss+ewma"]               # current (non-causal) pipeline
    print(f"\n  best TRULY-CAUSAL (by EVAL): {best_name} -> "
          f"EVAL {best['eval_r2']:.3f}, TEST {best['test_r2']:.3f}")
    print(f"  current NON-causal pipeline (centered_gauss+ewma): "
          f"EVAL {cur['eval_r2']:.3f}, TEST {cur['test_r2']:.3f}")
    print(f"  => causality costs {cur['eval_r2']-best['eval_r2']:+.3f} EVAL / "
          f"{cur['test_r2']-best['test_r2']:+.3f} TEST", flush=True)

    out = ROOT / "results" / "metrics" / "iter25_causal_smoothing.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
