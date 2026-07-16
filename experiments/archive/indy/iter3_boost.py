#!/usr/bin/env python
"""Iteration 3: cheap single-model R^2 boosters on the STM32 'small' 8-ch model.

No extra data, no size increase (except the ensemble reference). Levers:
  corr_loss : MSE + lam*(1 - batch Pearson r) -- optimise the metric we report.
  more_aug  : stronger input noise + channel dropout (helps cross-session).
  more_reg  : higher weight decay + dropout (paper uses L2 0.005-0.2, drop 0.3-0.5).
  combo     : the winners together.
  ensemble3 : 3-seed average (REFERENCE only -- 3x compute/size, not STM32-cheap).
Base 'small' (F32/H32/L1/dils[1,2,4,8], ~100 kB) TEST R2 = 0.529.
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
from experiments.archive.indy.iter1_cheap_wins import corr_loss

SMALL = dict(F=32, H=32, L=1, dils=[1, 2, 4, 8])


def main():
    data = H.prep(nch=8)
    base = {**E.CFG, **SMALL}
    variants = {
        "baseline":  (base, dict()),
        "corr_loss": (base, dict(loss_fn=corr_loss(0.3))),
        "more_aug":  ({**base, "noise": 0.2, "chdrop": 0.15}, dict()),
        "more_reg":  ({**base, "wd": 1e-2, "dropout": 0.4}, dict()),
        "ensemble3": (base, dict(seeds=(42, 1, 7))),
    }
    rows = {}
    print("=== Iteration 3: cheap boosters on small 8-ch (base R2=0.529) ===\n",
          flush=True)
    for name, (cfg, kw) in variants.items():
        t0 = time.time()
        res = H.run(data, cfg, **kw)
        rows[name] = res
        print(f"  {name:10s}  EVAL R2={res['eval_r2']:.3f}  TEST R2={res['test_r2']:.3f}"
              f"  (r={res['test_r']:.3f}, {res['kb']:.0f}kB)  [{time.time()-t0:.0f}s]",
              flush=True)

    b = rows["baseline"]["test_r2"]
    print(f"\n--- TEST R2 vs baseline ({b:.3f}) ---")
    for name, r in rows.items():
        print(f"  {name:10s}: {r['test_r2']:.3f}  ({r['test_r2']-b:+.3f})")
    out = ROOT / "results" / "metrics" / "iter3_boost.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
