#!/usr/bin/env python
"""Iteration 9: within-session (calibrated-deployment) R² ceiling at 8 channels.

Our headline 0.668 is CROSS-session (zero calibration -- train on other days, test
on a held-out day). A real implant is usually calibrated on the user's own recent
data. This measures that easier, realistic scenario: per session, temporal split
70% train / 15% eval / 15% test (no leakage), 8 fixed channels, 'small' model.
Comparable to the paper's within-session numbers.
Usage: py research/iter9_within_session.py
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
SESSIONS = ["test1", "eval1", "train1", "train4"]      # representative held-out + train


def within_data(name, sel, axes):
    r, v = E.load_electrode(name)
    W = E.windows(r[sel], v, axes)
    n = len(W); a, b = int(n * 0.70), int(n * 0.85)
    return {"train": W[:a], "eval": {name: W[a:b]}, "test": {name: W[b:]}}


def main():
    cfg = {**E.CFG, **SMALL}
    # fixed 8 channels + axes from the base training sessions (deployment choice)
    fr = np.mean([E.load_electrode(s)[0].mean(1) for s in
                  ["train1", "train2", "train3", "train4", "train5", "train6"]], 0)
    sel = np.sort(np.argsort(fr)[-8:])
    var = np.mean([E.load_electrode(s)[1].std(0) for s in
                   ["train1", "train2", "train3", "train4", "train5", "train6"]], 0)
    axes = np.sort(np.argsort(var)[-2:])
    print("=== Iteration 9: within-session calibrated R2 (8 ch) | cross-session ref "
          "= 0.668 ===\n", flush=True)
    rows = {}
    for s in SESSIONS:
        data = within_data(s, sel, axes)
        t0 = time.time()
        res = H.run(data, cfg)
        rows[s] = res
        print(f"  {s:7s} ({len(data['train'])} train win): TEST R2={res['test_r2']:.3f}"
              f"  (r={res['test_r']:.3f})  [{time.time()-t0:.0f}s]", flush=True)
    mean = float(np.mean([r["test_r2"] for r in rows.values()]))
    print(f"\n  within-session mean R2 = {mean:.3f}  (vs cross-session 0.668)")
    out = ROOT / "results" / "metrics" / "iter9_within_session.json"
    out.write_text(json.dumps({"mean_r2": mean, "per_session": rows}, indent=2),
                   encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
