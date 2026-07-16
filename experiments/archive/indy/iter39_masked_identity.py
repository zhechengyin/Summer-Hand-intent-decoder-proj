#!/usr/bin/env python
"""Iteration 39: identity-preserving 96-slot MASKED decoder -- reselect channels without retraining.

Executes the RMTBD "Dynamic-Channel Decoder" brief. Question: can a fixed 96-electrode identity
model, trained with realistic 32-channel masks and given a LABEL-FREE session-specific firing-rate
mask, change its active channel set WITHOUT retraining -- matching healthy-session accuracy and
rescuing drift failures?

Background (do not re-derive): LOG-078 (32ch is the big lever), LOG-083 (reselection helps but
feeding reselected electrodes into fixed generic slots hurts -- model depends on identity), LOG-086
(leave-one-month-out: 32ch zero-shot mean ~0.703, ~12% below 0.4), LOG-090 (dropout models ablation
not substitution). This tests the identity-preserving fix.

Four configs, evaluated leave-one-MONTH-out (LOG-086 protocol; single seed, labelled as such):
  1. fixed32          -- current arch, pool top-32 fixed channels (n_ch=64). The baseline.
  2. slot_fixedmask   -- 96-slot masked rep, ALWAYS the pool top-32 mask. Isolates representation cost.
  3. slot_randommask  -- 96-slot, random 32-subset resampled each batch; test on session top-32.
  4. slot_sessionmask -- 96-slot, each training window masked to ITS session's top-32; test on
                         held-out session's top-32 (all firing-based, NO velocity labels).

Channels/masks and all normalization use only training/calibration data of the fold. Held-out mask
= top-32 by firing on the held-out session (label-free). Reports the full stratified table the brief
asks for. Mask-correctness assertions run first. Usage: py research/iter39_masked_identity.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.common.harness as H
import experiments.archive.indy.iter20_multiscale as I20
import experiments.archive.indy.iter27_fresh_session as I27
import experiments.archive.indy.iter25_causal_smoothing as I25
import experiments.archive.indy.iter28_calibration as I28
import experiments.archive.indy.iter34_detector_cv as I34
import experiments.archive.indy.masked_input as MI
import models.tcn_gru.evaluate as E
import models.tcn_gru.best_model as M

SEED = 42
N_ACTIVE = 32
WIN = I20.WIN
CFG = I20.CAUSAL
OUT = ROOT / "results" / "metrics" / "iter39_masked_identity.json"


# ---------------------------------------------------------------- mask-correctness tests (brief 2-4)
def run_mask_tests():
    import torch
    torch.manual_seed(0)
    neural = torch.randn(4, 192, WIN)
    mask = torch.zeros(4, 96); mask[:, [7, 18, 42]] = 1.0          # only 3 electrodes observed
    x = MI.make_masked_input_torch(neural, mask)
    assert x.shape == (4, 288, WIN), x.shape
    # (2) masked electrodes are exactly zero before the network
    keep = torch.cat([mask, mask], 1).bool()
    assert torch.all(x[:, :192][~keep.unsqueeze(-1).expand(-1, -1, WIN)] == 0), "masked neural not 0"
    # (4) identities are NOT compacted: observed slots keep their positions & values
    for e in (7, 18, 42):
        assert torch.allclose(x[:, e], neural[:, e]), f"slot {e} altered"
        assert torch.allclose(x[:, 96 + e], neural[:, 96 + e]), f"ewma slot {e} altered"
    # (3) changing a MASKED electrode's value has no effect on the input tensor
    n2 = neural.clone(); n2[:, 3] = 999.0; n2[:, 96 + 3] = -999.0   # electrode 3 is unobserved
    x2 = MI.make_masked_input_torch(n2, mask)
    assert torch.allclose(x, x2), "masked electrode leaked into input"
    # mask channels present and correct
    assert torch.all(x[:, 192:][mask.bool().unsqueeze(-1).expand(-1, -1, WIN)] == 1.0)
    print("  mask-correctness tests PASS (zeroed / identity-preserved / masked-value-invariant)")


# ---------------------------------------------------------------- data
def neural_windows(counts, s):
    """Full (unmasked) 192-feature windows for one session + its 96-firing vector."""
    n192 = MI.build_neural_192(counts[s][0], ewma_alpha=I27.ALPHA)
    fr = counts[s][0].mean(1)                                      # label-free firing
    axes = AXES
    v = counts[s][1]
    W = WIN
    wins = [(n192[:, k * W:(k + 1) * W], v[k * W:(k + 1) * W][:, axes])
            for k in range(n192.shape[1] // W)]
    return wins, fr


def fixed32_windows(counts, s, chans):
    """Plain 64-feature windows (raw+ewma of the selected 32) for the fixed-32 baseline."""
    c = counts[s][0][chans].astype(np.float32)
    feat = np.concatenate([c, I25.ewma(c, I27.ALPHA)], 0)          # (64, T)
    mu, sd = feat.mean(1, keepdims=True), feat.std(1, keepdims=True) + 1e-6
    fz = ((feat - mu) / sd).astype(np.float32)
    v = counts[s][1]
    return [{"e": fz[:, k * WIN:(k + 1) * WIN], "vel": v[k * WIN:(k + 1) * WIN][:, AXES]}
            for k in range(fz.shape[1] // WIN)]


# ---------------------------------------------------------------- masked training / eval
def train_masked(train_ns, train_masks, cfg, epochs, val_pack, seed=SEED):
    """train_ns: list of (neural192 (192,T), vel (T,2)). train_masks: mask SAMPLER (idx, rng)->96, OR
    a fixed (96,) array. val_pack: (val_ns, val_mask96, ym, ys) for early stopping on eval r."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(4)
    T = min(n.shape[1] for n, _ in train_ns)
    X = np.stack([n[:, :T] for n, _ in train_ns]).astype(np.float32)     # (N,192,T)
    Y = np.stack([v[:T] for _, v in train_ns]).astype(np.float32)
    ym, ys = Y.mean((0, 1)), Y.std((0, 1)) + 1e-6
    Yn = ((Y - ym) / ys).astype(np.float32)
    Xt, Yt = torch.tensor(X), torch.tensor(Yn)
    net = M.build_net({**cfg, "n_out": 2}, MI.IN_DIM)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    mse = nn.MSELoss()
    idx = np.arange(len(Xt)); noise = cfg["noise"]
    fixed = None if callable(train_masks) else torch.tensor(np.asarray(train_masks, np.float32))
    val_ns, val_mask, _, _ = val_pack
    best, best_state = -np.inf, None
    for ep in range(epochs):
        net.train(); np.random.shuffle(idx); rng = np.random.default_rng(1000 + ep)
        for b in range(0, len(idx), cfg["bs"]):
            bi = idx[b:b + cfg["bs"]]
            nb = Xt[bi]
            if noise > 0:
                nb = nb + noise * torch.randn_like(nb)
            if fixed is not None:
                mb = fixed.unsqueeze(0).expand(len(bi), -1)
            else:
                mb = torch.tensor(np.stack([train_masks(j, rng) for j in bi]))
            xb = MI.make_masked_input_torch(nb, mb)
            opt.zero_grad(); mse(net(xb), Yt[bi]).backward(); opt.step()
        sched.step()
        r = eval_masked_r(net, val_ns, val_mask, ym, ys)
        if r > best:
            best, best_state = r, {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net, ym, ys


def eval_masked_r(net, ns, mask96, ym, ys):
    import torch
    net.eval(); rs = []
    with torch.no_grad():
        for n, v in ns:
            x = MI.make_masked_input_torch(torch.tensor(n[None]), torch.tensor(mask96[None]))
            p = net(x).numpy()[0] * ys + ym
            rs.append(M.corr(v.reshape(-1, 2), p.reshape(-1, 2)).mean())
    return float(np.mean(rs))


def score_masked(net, ns, mask96, ym, ys):
    import torch
    net.eval(); Y, P = [], []
    with torch.no_grad():
        for n, v in ns:
            x = MI.make_masked_input_torch(torch.tensor(n[None]), torch.tensor(mask96[None]))
            P.append(net(x).numpy()[0] * ys + ym); Y.append(v)
    Y = np.concatenate([y.reshape(-1, 2) for y in Y]); P = np.concatenate([p.reshape(-1, 2) for p in P])
    return float(M.r2(Y, P).mean()), float(M.corr(Y, P).mean())


AXES = None


def main():
    global AXES
    t0 = time.time()
    alls = list(E.TRAIN) + list(E.EVAL) + list(E.TEST) + __import__(
        "experiments.archive.indy.iter7_final", fromlist=["EXTRA18"]).EXTRA18
    evals = list(E.EVAL)
    folds = {}
    for s in alls:
        if s in evals:
            continue
        folds.setdefault(I34.month_of(s), []).append(s)
    counts = I27.load_counts_full(alls)
    AXES = np.sort(np.argsort(np.mean([counts[s][1].std(0) for s in E.TRAIN], 0))[-2:])
    print("=== Iteration 39: identity-preserving 96-slot masked decoder (single seed) ===")
    run_mask_tests()
    for k in sorted(folds):
        print(f"    fold {k}: {len(folds[k])} sessions")
    print(f"    loaded {len(counts)} sessions [{time.time()-t0:.0f}s]\n", flush=True)

    npar = sum(p.numel() for p in M.build_net({**CFG, "n_out": 2}, MI.IN_DIM).parameters())
    print(f"  96-slot masked model: {npar:,} params (~{npar/1024:.0f} KB int8)\n", flush=True)

    rows = {"_params": npar, "sessions": {}}
    for held, sess in sorted(folds.items()):
        train_s = [s for s in alls if s not in sess and s not in evals]
        fr_pool = np.mean([counts[s][0].mean(1) for s in train_s], 0)
        pool_chans = np.sort(np.argsort(fr_pool)[-N_ACTIVE:])
        pool_mask = np.zeros(96, np.float32); pool_mask[pool_chans] = 1.0

        # precompute neural windows + firing for train/eval/test-of-fold
        tr_ns, tr_sess, tr_masks_sess = [], [], []
        for s in train_s:
            wins, fr = neural_windows(counts, s)
            sm = np.zeros(96, np.float32); sm[np.argsort(fr)[-N_ACTIVE:]] = 1.0
            for w in wins:
                tr_ns.append(w); tr_sess.append(s); tr_masks_sess.append(sm)
        ev_ns, _ = neural_windows(counts, evals[0])
        ev_ym = np.stack([v for _, v in tr_ns[:1]])  # placeholder; real ym set in train_masked

        def rand_sampler(j, rng):
            m = np.zeros(96, np.float32); m[rng.choice(96, N_ACTIVE, replace=False)] = 1.0
            return m
        def sess_sampler(j, rng):
            return tr_masks_sess[j]

        t1 = time.time()
        # --- config 1: fixed-32 baseline (plain, no masking) ---
        b_tr = [x for s in train_s for x in fixed32_windows(counts, s, pool_chans)]
        b_ev = {evals[0]: fixed32_windows(counts, evals[0], pool_chans)}
        b_te = {s: fixed32_windows(counts, s, pool_chans) for s in sess}
        b_res = H.run({"train": b_tr, "eval": b_ev, "test": b_te}, CFG, seeds=(SEED,), ret_preds=True)
        base_r2 = {s: float(M.r2(Ye.reshape(-1, 2), P.reshape(-1, 2)).mean())
                   for s, (Ye, P) in b_res["test_preds"].items()}

        # --- configs 2-4: 96-slot masked ---
        val_pack = (ev_ns, pool_mask, None, None)
        nets = {}
        nets["slot_fixedmask"] = train_masked(tr_ns, pool_mask, CFG, CFG["epochs"], val_pack)
        val_pack_r = (ev_ns, pool_mask, None, None)  # eval uses pool mask (a fixed reference) for stopping
        nets["slot_randommask"] = train_masked(tr_ns, rand_sampler, CFG, CFG["epochs"], val_pack_r)
        nets["slot_sessionmask"] = train_masked(tr_ns, sess_sampler, CFG, CFG["epochs"], val_pack_r)
        print(f"  [fold {held}] trained 4 configs [{time.time()-t1:.0f}s]", flush=True)

        for s in sess:
            te_ns, fr_s = neural_windows(counts, s)
            sess_mask = np.zeros(96, np.float32); sess_mask[np.argsort(fr_s)[-N_ACTIVE:]] = 1.0
            overlap = int((sess_mask * pool_mask).sum())
            rec = {"fold": held, "fixed32": base_r2[s],
                   "overlap_pool_session": overlap,
                   "base_healthy": base_r2[s] >= 0.4}
            # slot_fixedmask tested with the POOL mask (it never learned to adapt)
            net, ym, ys = nets["slot_fixedmask"]
            rec["slot_fixedmask"] = score_masked(net, te_ns, pool_mask, ym, ys)[0]
            # random/session masked tested with the SESSION mask (label-free reselection)
            for cfgname in ("slot_randommask", "slot_sessionmask"):
                net, ym, ys = nets[cfgname]
                rec[cfgname] = score_masked(net, te_ns, sess_mask, ym, ys)[0]
            rows["sessions"][s] = rec
            print(f"      {s:20s} ov{overlap:2d}/32 base(fx32) {rec['fixed32']:+.3f} | "
                  f"slotfix {rec['slot_fixedmask']:+.3f} | rand {rec['slot_randommask']:+.3f} | "
                  f"sess {rec['slot_sessionmask']:+.3f}", flush=True)

    # ---------- aggregate + stratify (brief evaluation protocol) ----------
    S = rows["sessions"]; names = list(S)
    configs = ["fixed32", "slot_fixedmask", "slot_randommask", "slot_sessionmask"]
    def agg(subset):
        return {c: (round(float(np.mean([S[s][c] for s in subset])), 3) if subset else None)
                for c in configs}
    healthy = [s for s in names if S[s]["base_healthy"]]
    failed = [s for s in names if not S[s]["base_healthy"]]
    print("\n  === MEANS ===")
    for label, sub in (("ALL", names), (f"HEALTHY base>=0.4 (n={len(healthy)})", healthy),
                       (f"FAILED base<0.4 (n={len(failed)})", failed)):
        a = agg(sub)
        print(f"    {label:28s} " + " | ".join(f"{c.split('_')[-1]} {a[c]:+.3f}" for c in configs))
    allr2 = {c: [S[s][c] for s in names] for c in configs}
    print("\n  === distribution (the brief: don't hide failures behind the mean) ===")
    for c in configs:
        v = np.array(allr2[c])
        print(f"    {c:18s} mean {v.mean():+.3f}  median {np.median(v):+.3f}  "
              f"min {v.min():+.3f}  max {v.max():+.3f}  <0.4: {int((v<0.4).sum())}/{len(v)}")
    rows["_summary"] = {"all": agg(names), "healthy": agg(healthy), "failed": agg(failed),
                        "n_healthy": len(healthy), "n_failed": len(failed)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
