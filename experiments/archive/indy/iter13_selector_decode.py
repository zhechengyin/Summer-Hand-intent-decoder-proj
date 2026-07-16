#!/usr/bin/env python
"""Iteration 13: does any channel SCORE beat firing-rate on HELD-OUT decoding?

iter12 showed firing rate picks reliable (not velocity-correlated) channels, yet
generalizes best historically. Here we actually decode with each score's global
top-8 (24-session training, fixed eval1/test1, small model) and compare TEST R².
Scores: firing (ref), lowfreq (0.2-3 Hz power), velcorr, fftweighted.
Usage: py research/iter13_selector_decode.py
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
import research.iter7_final as I7
import research.iter12_channel_scores as SC
import models.tcn_gru.evaluate as E

SMALL = dict(F=32, H=32, L=1, dils=[1, 2, 4, 8])
TRAIN = list(E.TRAIN) + I7.EXTRA18                      # 24 sessions


def prep_sel(sel, loaded, axes):
    tr = []
    for s in TRAIN:
        tr += E.windows(loaded[s][0][sel], loaded[s][1], axes)
    ev = {s: E.windows(loaded[s][0][sel], loaded[s][1], axes) for s in E.EVAL}
    te = {s: E.windows(loaded[s][0][sel], loaded[s][1], axes) for s in E.TEST}
    return {"train": tr, "eval": ev, "test": te}


def main():
    cfg = {**E.CFG, **SMALL}
    loaded = {s: E.load_electrode(s) for s in TRAIN + list(E.EVAL) + list(E.TEST)}
    axes = np.sort(np.argsort(np.mean([loaded[s][1].std(0) for s in E.TRAIN], 0))[-2:])
    per = {s: SC.scores(*loaded[s]) for s in TRAIN}
    # LOG-059 partial done: firing 0.577, lowfreq 0.631, velcorr 0.582. Remaining:
    keys = ["fftweighted"]
    glob = {k: np.mean([per[s][k] for s in TRAIN], 0) for k in keys}
    sels = {k: np.sort(np.argsort(glob[k])[-8:]) for k in keys}

    print("=== Iteration 13: selector decode (8-of-96, 24 sess, small) ===\n", flush=True)
    rows = {}
    for k in keys:
        t0 = time.time()
        res = H.run(prep_sel(sels[k], loaded, axes), cfg)
        rows[k] = {**res, "channels": sels[k].tolist()}
        print(f"  {k:12s} ch={sels[k].tolist()}  TEST R2={res['test_r2']:.3f}  "
              f"EVAL R2={res['eval_r2']:.3f}  [{time.time()-t0:.0f}s]", flush=True)

    ref = rows.get("firing", {}).get("test_r2", 0.577)   # firing from LOG-059 partial
    print(f"\n--- selector decode (TEST R2; firing ref = {ref:.3f}) ---")
    for k, r in rows.items():
        print(f"  {k:12s}: {r['test_r2']:.3f}  ({r['test_r2']-ref:+.3f})")
    out = ROOT / "results" / "metrics" / "iter13_selector_decode.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
