#!/usr/bin/env python
"""Iteration 10: spend the ~400 KB budget -- scale the model up at 24 sessions.

iter8 was still rising at 100 KB int8 (wide=0.677). Budget is ~400 KB, so push
capacity: xwide (F96), xxwide (F128 ~400 KB int8), and a bigger-receptive variant.
Report TEST R² and int8 size. Watch for overfitting (eval vs test) -- if the
bigger models overfit, the fix is more data (iter11).
Usage: py experiments/archive/indy/iter10_bigger.py
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
    "wide":     dict(F=64, H=64, L=1, dils=[1, 2, 4, 8]),          # ref 0.677, ~100 KB
    "xwide":    dict(F=96, H=96, L=1, dils=[1, 2, 4, 8]),          # ~230 KB int8
    "xxwide":   dict(F=128, H=128, L=1, dils=[1, 2, 4, 8]),        # ~400 KB int8
    "xxwide_rf": dict(F=128, H=128, L=1, dils=[1, 2, 4, 8, 16]),   # bigger receptive field
}


def main():
    data = I5.prep_more(I7.EXTRA18[:18], reselect=False)          # 24 sessions
    print("=== Iteration 10: bigger models @ 24 sessions (budget ~400 KB int8) ===\n",
          flush=True)
    DONE = {"wide", "xwide", "xxwide"}         # completed in LOG-059 (0.677/0.679/0.667)
    rows = {}
    for name, arch in ARCHS.items():
        if name in DONE:
            continue
        cfg = {**E.CFG, **arch}
        t0 = time.time()
        res = H.run(data, cfg)
        int8_kb = res["n_params"] / 1024
        rows[name] = {**res, "int8_kb": int8_kb}
        gap = res["eval_r2"] - res["test_r2"]
        print(f"  {name:9s} ~{int8_kb:4.0f} KB int8 ({res['n_params']:>7,}p)  "
              f"TEST R2={res['test_r2']:.3f}  EVAL R2={res['eval_r2']:.3f}  "
              f"(gap {gap:+.3f})  [{time.time()-t0:.0f}s]", flush=True)

    print("\n--- R2 vs size (budget 400 KB int8; ref wide=0.677) ---")
    for name, r in rows.items():
        print(f"  {name:9s}: R2={r['test_r2']:.3f}  (~{r['int8_kb']:.0f} KB int8)")
    out = ROOT / "results" / "metrics" / "iter10_bigger.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
