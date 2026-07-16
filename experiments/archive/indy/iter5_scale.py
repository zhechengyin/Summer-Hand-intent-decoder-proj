#!/usr/bin/env python
"""Iteration 5: scale training data further (iter4 showed 6->9 = +0.060 R2).

18-session training pool now downloaded. Extend the curve to 15 and 18 sessions,
and test whether re-selecting the 8 channels on the FULL pool (vs fixed to the
original 6) helps -- the deployable device routes 8 fixed electrodes, so a more
robust choice matters.

Base 'small' 100 kB model, fixed eval1/test1. Reference: 6=0.529, 9=0.589.
Usage: py experiments/archive/indy/iter5_scale.py
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

SMALL = dict(F=32, H=32, L=1, dils=[1, 2, 4, 8])
# 12 extra indy sessions (near-window batch first, then wider)
ALL_EXTRA = ["indy_20160927_06", "indy_20160930_02", "indy_20160930_05",
             "indy_20161025_04", "indy_20161026_03", "indy_20161027_03",
             "indy_20160915_01", "indy_20160916_01", "indy_20160921_01",
             "indy_20160927_04", "indy_20161206_02", "indy_20161207_02"]


def prep_more(extra, nch=8, reselect=False):
    names = list(E.TRAIN) + list(extra)
    uniq = list(dict.fromkeys(names + list(E.EVAL) + list(E.TEST)))
    loaded = {s: E.load_electrode(s) for s in uniq}
    var = np.mean([loaded[s][1].std(0) for s in E.TRAIN], 0)
    axes = np.sort(np.argsort(var)[-2:])
    # channel selection: on full training pool if reselect, else original 6
    fr_src = names if reselect else list(E.TRAIN)
    fr = np.mean([loaded[s][0].mean(1) for s in fr_src], 0)
    sel = np.sort(np.argsort(fr)[-nch:])
    tr = []
    for s in names:
        tr += E.windows(loaded[s][0][sel], loaded[s][1], axes)
    ev = {s: E.windows(loaded[s][0][sel], loaded[s][1], axes) for s in E.EVAL}
    te = {s: E.windows(loaded[s][0][sel], loaded[s][1], axes) for s in E.TEST}
    return {"train": tr, "eval": ev, "test": te, "sel": sel}


def main():
    cfg = {**E.CFG, **SMALL}
    print("=== Iteration 5: scale data 15/18 sessions (ref 6=0.529, 9=0.589) ===\n",
          flush=True)
    rows = {}
    runs = [("15_fixed", 9, False), ("18_fixed", 12, False), ("18_reselect", 12, True)]
    for name, n_extra, resel in runs:
        data = prep_more(ALL_EXTRA[:n_extra], reselect=resel)
        t0 = time.time()
        res = H.run(data, cfg)
        rows[name] = {**res, "channels": data["sel"].tolist()}
        print(f"  {name:12s} ({6+n_extra} sess, {len(data['train'])} win): "
              f"TEST R2={res['test_r2']:.3f}  EVAL R2={res['eval_r2']:.3f}  "
              f"(r={res['test_r']:.3f})  [{time.time()-t0:.0f}s]", flush=True)

    print("\n--- data scaling (TEST R2) ---")
    print("   6=0.529  9=0.589  " + "  ".join(
        f"{k.split('_')[0]}({k.split('_')[1]})={r['test_r2']:.3f}"
        for k, r in rows.items()))
    out = ROOT / "results" / "metrics" / "iter5_scale.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
