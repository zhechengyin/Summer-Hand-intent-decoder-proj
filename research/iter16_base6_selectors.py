#!/usr/bin/env python
"""Iteration 16: channel selectors on the BASE-6 sessions (the deployment choice).

iter13 selected on 24 sessions; the deployed model selects the 8 channels on the
base 6 (firing-6 -> [26,51,53,66,71,73,75,94], R2~0.655). Does low-freq (0.2-3 Hz)
or fft-weighted selection on the SAME base-6 beat firing-6? If yes, swap the
deployed channels. Decode each on the 24-session pool, small model, test1.
Usage: py research/iter16_base6_selectors.py
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
import research.iter13_selector_decode as I13
import models.tcn_gru.evaluate as E

SMALL = dict(F=32, H=32, L=1, dils=[1, 2, 4, 8])
BASE6 = list(E.TRAIN)                                   # the 6 base training sessions
TRAIN24 = list(E.TRAIN) + I7.EXTRA18


def main():
    cfg = {**E.CFG, **SMALL}
    loaded = {s: E.load_electrode(s) for s in TRAIN24 + list(E.EVAL) + list(E.TEST)}
    axes = np.sort(np.argsort(np.mean([loaded[s][1].std(0) for s in E.TRAIN], 0))[-2:])
    per6 = {s: SC.scores(*loaded[s]) for s in BASE6}     # scores on base-6 only
    keys = ["firing", "lowfreq", "fftweighted"]
    glob = {k: np.mean([per6[s][k] for s in BASE6], 0) for k in keys}
    sels = {k: np.sort(np.argsort(glob[k])[-8:]) for k in keys}

    print("=== Iteration 16: base-6 channel selectors (decode 24 sess, test1) ===")
    print(f"  deployed firing-6 ref = 0.655\n", flush=True)
    rows = {}
    for k in keys:
        t0 = time.time()
        res = H.run(I13.prep_sel(sels[k], loaded, axes), cfg)
        rows[k] = {**res, "channels": sels[k].tolist()}
        print(f"  {k:12s} ch={sels[k].tolist()}  TEST R2={res['test_r2']:.3f}  "
              f"EVAL R2={res['eval_r2']:.3f}  [{time.time()-t0:.0f}s]", flush=True)

    ref = rows["firing"]["test_r2"]
    print(f"\n--- base-6 selectors (TEST R2; firing-6 = {ref:.3f}) ---")
    for k, r in rows.items():
        print(f"  {k:12s}: {r['test_r2']:.3f}  ({r['test_r2']-ref:+.3f})  {r['channels']}")
    out = ROOT / "results" / "metrics" / "iter16_base6_selectors.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
