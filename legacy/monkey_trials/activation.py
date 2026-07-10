#!/usr/bin/env python
"""Which activation is best for the MONKEY velocity decoder (TCN+GRU)?

Small-sample sweep: one indy session, tuned preprocessing (3 Hz vel-LP + sigma1
firing-rate smoothing), TCN+GRU, within-session 5-block CV. Swap the conv/TCN
activation (cfg['act']); everything else fixed. Reports mean r per activation.

Usage: py tools/monkey_activation.py --file data/indy_loco/indy_20161005_06.mat
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import legacy.monkey_trials.velocity as IV
import legacy.monkey_trials.vellp as VL
import models.best_model as R

ACTS = ["gelu", "relu", "elu", "silu", "leaky_relu", "tanh", "mish", "selu"]
BASE = {**R.BASE, "dils": [1, 2, 4, 8, 16], "H": 64, "L": 2, "F": 64,
        "epochs": 60, "noise": 0.1, "chdrop": 0.1, "cosine": True,
        "kfold": IV.KFOLD}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/indy_loco/indy_20161005_06.mat")
    args = ap.parse_args()
    p = ROOT / args.file if not Path(args.file).is_absolute() else Path(args.file)
    rates, vel, nb = VL.load_smoothed(str(p), 3.0)      # 3 Hz vel-LP
    rates = gaussian_filter1d(rates, 1.0, axis=1)       # sigma=1 rate smoothing
    tri = IV.make_trials(rates, vel, nb)
    print(f"=== monkey velocity activation sweep | {p.name} | {rates.shape[0]} "
          f"units, {len(tri)} windows, 5-block CV ===\n", flush=True)

    report = {}
    for act in ACTS:
        t0 = time.time()
        r = R.run_nn(tri, {**BASE, "act": act})
        report[act] = {"r_mean": float(r.mean()), "r": [float(x) for x in r]}
        print(f"{act:12s} r_mean={r.mean():.3f} (a1={r[0]:.3f} a2={r[1]:.3f})   "
              f"[{time.time()-t0:.0f}s]", flush=True)
    best = max(report, key=lambda a: report[a]["r_mean"])
    print(f"\nBEST activation: {best} ({report[best]['r_mean']:.3f})")
    out = ROOT / "results" / "metrics" / "monkey_activation.json"
    out.write_text(json.dumps({"file": p.name, "results": report, "best": best},
                              indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
