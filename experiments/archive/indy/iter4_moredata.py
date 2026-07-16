#!/usr/bin/env python
"""Iteration 4: does MORE training data raise 8-ch R^2? (paper: +0.03-0.04)

Add nearby indy sessions to the 6 training files and measure the TEST R^2 lift.
Channel selection (top-8 firing) and movement axes are held FIXED to the original
train1-6, so this isolates the effect of training DATA (not channel/axis choice).
Same 'small' 100 kB model and fixed eval1/test1.

Base 'small', 6 train sessions: TEST R2 = 0.529.
Usage: py research/iter4_moredata.py
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
import models.tcn_gru.evaluate as E

SMALL = dict(F=32, H=32, L=1, dils=[1, 2, 4, 8])
EXTRA = ["indy_20160927_06", "indy_20160930_02", "indy_20160930_05",
         "indy_20161025_04", "indy_20161026_03", "indy_20161027_03"]


def prep_more(extra, nch=8):
    names = list(E.TRAIN) + list(extra)
    uniq = list(dict.fromkeys(names + list(E.EVAL) + list(E.TEST)))
    loaded = {s: E.load_electrode(s) for s in uniq}
    var = np.mean([loaded[s][1].std(0) for s in E.TRAIN], 0)      # axes from base train
    axes = np.sort(np.argsort(var)[-2:])
    fr = np.mean([loaded[s][0].mean(1) for s in E.TRAIN], 0)      # channels from base train
    sel = np.sort(np.argsort(fr)[-nch:])
    tr = []
    for s in names:
        tr += E.windows(loaded[s][0][sel], loaded[s][1], axes)
    ev = {s: E.windows(loaded[s][0][sel], loaded[s][1], axes) for s in E.EVAL}
    te = {s: E.windows(loaded[s][0][sel], loaded[s][1], axes) for s in E.TEST}
    return {"train": tr, "eval": ev, "test": te}


def main():
    cfg = {**E.CFG, **SMALL}
    print("=== Iteration 4: more training data (8-ch small, base 6 sess=0.529) ===\n",
          flush=True)
    rows = {}
    for n_extra in (0, 3, 6):
        data = prep_more(EXTRA[:n_extra])
        t0 = time.time()
        res = H.run(data, cfg)
        rows[str(6 + n_extra)] = res
        print(f"  {6+n_extra:2d} sessions ({len(data['train'])} windows): "
              f"TEST R2={res['test_r2']:.3f}  EVAL R2={res['eval_r2']:.3f}  "
              f"(r={res['test_r']:.3f})  [{time.time()-t0:.0f}s]", flush=True)

    print("\n--- data scaling (TEST R2) ---")
    for k, r in rows.items():
        print(f"  {k:>2s} sessions: {r['test_r2']:.3f}")
    out = ROOT / "results" / "metrics" / "iter4_moredata.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
