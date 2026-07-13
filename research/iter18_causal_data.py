#!/usr/bin/env python
"""Iteration 18: does MORE DATA lift the strictly-causal decoder? (ref causal = 0.606)

Data was the biggest lever for the bidir model; test it for the causal one.
Strictly-causal wide TCN+GRU (bidir=False), 24 vs 28 sessions (the 4 extra are
early-2016, temporally distant from the Oct test -- may add drift). 8 fixed ch.
Usage: py research/iter18_causal_data.py
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
import research.iter5_scale as I5
import research.iter7_final as I7
import models.tcn_gru.evaluate as E

CAUSAL = {**E.CFG, "F": 64, "H": 64, "L": 1, "dils": [1, 2, 4, 8], "bidir": False}
EXTRA_28 = I7.EXTRA18 + ["indy_20160624_03", "indy_20160419_01",
                         "indy_20160630_01", "indy_20160407_02"]


def main():
    print("=== Iteration 18: causal + more data (ref causal 24-sess = 0.606) ===\n",
          flush=True)
    rows = {}
    for n_extra, label in [(18, "24_sess"), (22, "28_sess")]:
        data = I5.prep_more(EXTRA_28[:n_extra], reselect=False)
        t0 = time.time()
        res = H.run(data, CAUSAL)
        rows[label] = {"test_r2": res["test_r2"], "eval_r2": res["eval_r2"],
                       "sessions": 6 + n_extra}
        print(f"  {label} ({6+n_extra} sess): TEST R2={res['test_r2']:.3f}  "
              f"EVAL R2={res['eval_r2']:.3f}  [{time.time()-t0:.0f}s]", flush=True)

    print("\n--- causal data scaling (TEST R2) ---")
    for label, r in rows.items():
        print(f"  {label}: {r['test_r2']:.3f}")
    out = ROOT / "results" / "metrics" / "iter18_causal_data.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
