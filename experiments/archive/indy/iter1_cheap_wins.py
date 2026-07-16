#!/usr/bin/env python
"""Iteration 1: cheap, high-probability R^2 wins on the fixed split (96 ch).

Grounded in standard motor-BCI practice:
  corr_loss   : MSE + lam*(1 - batch Pearson r). MSE alone does not directly
                optimise the correlation/R^2 we report.
  ensemble3   : average 3 seeds' predictions (variance reduction).
  ema_post    : causal exponential smoothing of outputs -- velocity is smooth,
                a light Kalman-like prior at zero model cost.
  combo       : corr_loss + ensemble3 + ema_post.
Compared against the baseline (experiments/common/harness.py) under the identical harness.
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
import models.tcn_gru.evaluate as E


def corr_loss(lam=0.3):
    import torch
    import torch.nn.functional as Fn

    def lf(pred, targ):
        mse = Fn.mse_loss(pred, targ)
        p = pred.reshape(-1, pred.shape[-1]); t = targ.reshape(-1, targ.shape[-1])
        p = p - p.mean(0); t = t - t.mean(0)
        num = (p * t).sum(0)
        den = torch.sqrt((p * p).sum(0) * (t * t).sum(0) + 1e-8)
        cc = (num / den).mean()
        return mse + lam * (1 - cc)
    return lf


def ema_post(alpha=0.7):
    def f(p):                                   # p: (n, T, D)
        out = p.copy()
        for i in range(1, p.shape[1]):
            out[:, i] = alpha * p[:, i] + (1 - alpha) * out[:, i - 1]
        return out
    return f


def main():
    data = H.prep(nch=96)
    cfg = E.CFG
    variants = {
        "baseline":  dict(),
        "corr_loss": dict(loss_fn=corr_loss(0.3)),
        "ensemble3": dict(seeds=(42, 1, 7)),
        "ema_post":  dict(post=ema_post(0.7)),
        "combo":     dict(loss_fn=corr_loss(0.3), seeds=(42, 1, 7), post=ema_post(0.7)),
    }
    rows = {}
    print("=== Iteration 1: cheap wins (96 ch, fixed split) ===\n", flush=True)
    for name, kw in variants.items():
        t0 = time.time()
        res = H.run(data, cfg, **kw)
        rows[name] = res
        print(f"  {name:10s}  EVAL R2={res['eval_r2']:.3f}  "
              f"TEST R2={res['test_r2']:.3f}  (r={res['test_r']:.3f})  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    base = rows["baseline"]["test_r2"]
    print(f"\n--- TEST R2 vs baseline ({base:.3f}) ---")
    for name, r in rows.items():
        d = r["test_r2"] - base
        print(f"  {name:10s}: {r['test_r2']:.3f}  ({d:+.3f})")
    out = ROOT / "results" / "metrics" / "iter1_cheap_wins.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
