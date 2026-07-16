#!/usr/bin/env python
"""Iteration 7: final push -- 24-session plateau check + ensemble on best data.

Recipe so far: 18 sessions, fixed train1-6 top-8 channels, 100 kB TCN+GRU = 0.628.
Test: (a) 24 sessions (does the plateau hold?), (b) 18 sessions + 3-seed ensemble
(iter3 showed ensembling +0.022; 3x25 kB int8 still fits STM32).
Usage: py experiments/archive/indy/iter7_final.py
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
import models.tcn_gru.evaluate as E

# 18 extra sessions = iter5's 12 + prefetch-2's 6 (24-session pool with the base 6)
EXTRA18 = I5.ALL_EXTRA + ["indy_20161212_02", "indy_20161220_02", "indy_20170123_02",
                          "indy_20170124_01", "indy_20170127_03", "indy_20170131_02"]


def main():
    cfg = {**E.CFG, **I5.SMALL}
    print("=== Iteration 7: final (ref 18-sess single = 0.628) ===\n", flush=True)
    rows = {}
    runs = [
        ("24_single", 18, (42,)),
        ("18_ens3",   12, (42, 1, 7)),
        ("24_ens3",   18, (42, 1, 7)),
    ]
    for name, n_extra, seeds in runs:
        data = I5.prep_more(EXTRA18[:n_extra], reselect=False)
        t0 = time.time()
        res = H.run(data, cfg, seeds=seeds)
        rows[name] = res
        print(f"  {name:10s} ({6+n_extra} sess, {len(seeds)} seed): "
              f"TEST R2={res['test_r2']:.3f}  EVAL R2={res['eval_r2']:.3f}  "
              f"(r={res['test_r']:.3f})  [{time.time()-t0:.0f}s]", flush=True)

    print("\n--- final comparison (TEST R2; 18-single ref = 0.628) ---")
    for name, r in rows.items():
        print(f"  {name:10s}: {r['test_r2']:.3f}")
    out = ROOT / "results" / "metrics" / "iter7_final.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
