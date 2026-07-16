#!/usr/bin/env python
"""Iteration 8: does more data (24 sessions) now support a bigger model?

At 6 sessions the 'small' model was best (bigger overfit). With 24 sessions the
data may support more capacity. Sweep architecture size at fixed 24-session data,
8 channels. Report TEST R² and int8 size (must stay STM32-friendly). Ref: small
24-session single = 0.655.
Usage: py experiments/archive/indy/iter8_capacity.py
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
import experiments.archive.indy.iter5_scale as I5
import experiments.archive.indy.iter7_final as I7
import models.tcn_gru.evaluate as E

ARCHS = {
    "small":  dict(F=32, H=32, L=1, dils=[1, 2, 4, 8]),            # ref 0.655
    "medium": dict(F=48, H=48, L=1, dils=[1, 2, 4, 8]),
    "wide":   dict(F=64, H=64, L=1, dils=[1, 2, 4, 8]),
    "deep":   dict(F=48, H=48, L=2, dils=[1, 2, 4, 8, 16]),
}


def main():
    data = I5.prep_more(I7.EXTRA18[:18], reselect=False)          # 24 sessions
    print("=== Iteration 8: capacity sweep at 24 sessions (ref small=0.655) ===\n",
          flush=True)
    rows = {}
    for name, arch in ARCHS.items():
        cfg = {**E.CFG, **arch}
        t0 = time.time()
        res = H.run(data, cfg)
        int8_kb = res["n_params"] / 1024                          # ~1 byte/param
        rows[name] = {**res, "int8_kb": int8_kb}
        print(f"  {name:7s} {res['kb']:6.0f} kB fp32 (~{int8_kb:4.0f} kB int8, "
              f"{res['n_params']:>6,}p)  TEST R2={res['test_r2']:.3f}  "
              f"EVAL R2={res['eval_r2']:.3f}  [{time.time()-t0:.0f}s]", flush=True)

    print("\n--- capacity vs R2 at 24 sessions ---")
    for name, r in rows.items():
        print(f"  {name:7s}: R2={r['test_r2']:.3f}  (~{r['int8_kb']:.0f} kB int8)")
    out = ROOT / "results" / "metrics" / "iter8_capacity.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
