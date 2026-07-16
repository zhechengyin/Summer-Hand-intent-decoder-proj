#!/usr/bin/env python
"""Iteration 28: WHY does the decoder collapse on fresh sessions -- scale, or representation?

LOG-079: on never-seen sessions the 8ch model collapsed to R2=0.054 (negative on 3 of 4);
32ch fell to 0.305. R2 punishes scale/offset errors; Pearson r does not. Fresh sessions
have visibly different velocity scales (indy_20160407_02 axis std 3.53 vs pool 6.79), and
we un-normalize the output with the POOL's velocity stats -> predictions can be ~2x too
large and score R2<0 while still tracking the movement perfectly.

This separates the two explanations and tests the cheapest fixes, per fresh session, on a
chronological 50/50 split (calibrate on the 1st half, score the 2nd half):

  zero_shot : pool model, pool output scaling      (= the LOG-079 condition)
  affine    : + per-axis gain/offset fit on the calibration half   (2 scalars/axis -- the
              minimal, deployable "recalibration"; leaves the network untouched)
  finetune  : pool model, weights fine-tuned on the calibration half
  scratch   : train only on the calibration half (no pool) -- is the pool helping at all?

Reports Pearson r AND R2. If r is high while zero-shot R2 is negative, the collapse is
SCALE, not representation, and `affine` will recover it. Usage: py research/iter28_calibration.py
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
import research.iter27_fresh_session as I27
import models.tcn_gru.evaluate as E
import models.tcn_gru.best_model as M

SEED = 42
COUNTS = [8, 32]
FRESH = I27.FRESH
POOL = I27.POOL
FT_EPOCHS = 25
FT_LR = 3e-4
OUT = ROOT / "results" / "metrics" / "iter28_calibration.json"


def score(y, p):
    return float(M.r2(y, p).mean()), float(M.corr(y, p).mean())


def stack(trials):
    T = min(t["e"].shape[1] for t in trials)
    X = np.stack([t["e"][:, :T] for t in trials]).astype(np.float32)
    Y = np.stack([t["vel"][:T] for t in trials]).astype(np.float32)
    return X, Y


def main():
    import torch
    import torch.nn as nn
    t0 = time.time()
    counts = I27.load_counts_full(POOL + list(E.EVAL) + list(E.TEST) + FRESH)
    axes = np.sort(np.argsort(np.mean([counts[s][1].std(0) for s in E.TRAIN], 0))[-2:])
    print("=== Iteration 28: fresh-session collapse -- scale or representation? ===")
    print(f"    conditions: zero_shot | affine (gain/offset) | finetune | scratch")
    print(f"    loaded {len(counts)} sessions [{time.time()-t0:.0f}s]\n", flush=True)

    rows = {}
    for n in COUNTS:
        chans = I27.topN_channels(counts, n)
        # --- pretrain on the 24-session pool (single seed for cost) ---
        t1 = time.time()
        tr_by = I27.build(counts, chans, axes, POOL)
        data = {"train": [x for s in POOL for x in tr_by[s]],
                "eval": I27.build(counts, chans, axes, list(E.EVAL)),
                "test": I27.build(counts, chans, axes, list(E.TEST))}
        res = H.run(data, I20.CAUSAL, seeds=(SEED,), ret_net=True)
        pool_net, (ym, ys) = res["net"], res["norm"]
        print(f"  [{n}ch] pool model: EVAL R2={res['eval_r2']:.3f} test1 R2={res['test_r2']:.3f} "
              f"[{time.time()-t1:.0f}s]", flush=True)

        fresh_by = I27.build(counts, chans, axes, FRESH)
        rows[f"{n}ch"] = {}
        for s in FRESH:
            w = fresh_by[s]
            half = len(w) // 2
            Xc, Yc = stack(w[:half])          # calibration half (chronologically first)
            Xt, Yt = stack(w[half:])          # scored half
            yt = Yt.reshape(-1, 2)

            def predict(net, X):
                net.eval()
                with torch.no_grad():
                    return net(torch.tensor(X)).numpy() * ys + ym

            out = {}
            # 1) zero-shot
            p0_c, p0_t = predict(pool_net, Xc), predict(pool_net, Xt)
            out["zero_shot"] = score(yt, p0_t.reshape(-1, 2))

            # 2) affine: per-axis gain/offset fit on the CALIBRATION half only
            pc, yc = p0_c.reshape(-1, 2), Yc.reshape(-1, 2)
            gain = np.zeros(2); off = np.zeros(2)
            for a in range(2):
                A = np.vstack([pc[:, a], np.ones(len(pc))]).T
                gain[a], off[a] = np.linalg.lstsq(A, yc[:, a], rcond=None)[0]
            out["affine"] = score(yt, p0_t.reshape(-1, 2) * gain + off)

            # 3) finetune pool weights on the calibration half
            ft = M.build_net({**I20.CAUSAL, "n_out": 2}, Xc.shape[1])
            ft.load_state_dict(pool_net.state_dict())
            opt = torch.optim.AdamW(ft.parameters(), lr=FT_LR, weight_decay=I20.CAUSAL["wd"])
            mse = nn.MSELoss()
            Xtc = torch.tensor(Xc); Ytc = torch.tensor(((Yc - ym) / ys).astype(np.float32))
            for ep in range(FT_EPOCHS):
                ft.train()
                idx = np.random.permutation(len(Xtc))
                for b in range(0, len(idx), I20.CAUSAL["bs"]):
                    bi = idx[b:b + I20.CAUSAL["bs"]]
                    opt.zero_grad(); mse(ft(Xtc[bi]), Ytc[bi]).backward(); opt.step()
            out["finetune"] = score(yt, predict(ft, Xt).reshape(-1, 2))

            # 4) scratch on the calibration half only (own target norm)
            sc = M.build_net({**I20.CAUSAL, "n_out": 2}, Xc.shape[1])
            cm, cs = Yc.mean((0, 1)), Yc.std((0, 1)) + 1e-6
            opt = torch.optim.AdamW(sc.parameters(), lr=I20.CAUSAL["lr"],
                                    weight_decay=I20.CAUSAL["wd"])
            Ysc = torch.tensor(((Yc - cm) / cs).astype(np.float32))
            for ep in range(I20.CAUSAL["epochs"]):
                sc.train()
                idx = np.random.permutation(len(Xtc))
                for b in range(0, len(idx), I20.CAUSAL["bs"]):
                    bi = idx[b:b + I20.CAUSAL["bs"]]
                    opt.zero_grad(); mse(sc(Xtc[bi]), Ysc[bi]).backward(); opt.step()
            sc.eval()
            with torch.no_grad():
                ps = sc(torch.tensor(Xt)).numpy() * cs + cm
            out["scratch"] = score(yt, ps.reshape(-1, 2))

            rows[f"{n}ch"][s] = {k: {"r2": v[0], "r": v[1]} for k, v in out.items()}
            print(f"    {s}: " + "  ".join(
                f"{k}: R2={v[0]:+.3f}/r={v[1]:.3f}" for k, v in out.items()), flush=True)

        # means
        means = {}
        for cond in ["zero_shot", "affine", "finetune", "scratch"]:
            means[cond] = {
                "r2": float(np.mean([rows[f"{n}ch"][s][cond]["r2"] for s in FRESH])),
                "r": float(np.mean([rows[f"{n}ch"][s][cond]["r"] for s in FRESH]))}
        rows[f"{n}ch"]["mean"] = means
        print(f"  [{n}ch] MEAN over fresh: " + "  ".join(
            f"{c}: R2={m['r2']:+.3f}/r={m['r']:.3f}" for c, m in means.items()), flush=True)
        print(flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
