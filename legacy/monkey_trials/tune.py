#!/usr/bin/env python
"""Monkey velocity decoder tuning: 3 Hz vel-LP + firing-rate smoothing + causal.

- vel target low-passed at 3 Hz (honest sweet spot, LOG-030).
- input firing-rate Gaussian smoothing (sigma in bins) -- raises input SNR
  WITHOUT touching the target (no over-smoothing concern).
- bidirectional (best/offline) vs causal unidirectional (honest real-time).

Within-session 5-block CV on one session. Usage:
  py tools/indy_tune.py --file data/indy_loco/indy_20161005_06.mat
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import legacy.monkey_trials.velocity as IV
import legacy.monkey_trials.vellp as VL
import models.best_model as R

BASE = {**R.BASE, "dils": [1, 2, 4, 8, 16], "H": 64, "L": 2, "F": 64,
        "act": "relu",                              # ReLU = monkey default (LOG-038)
        "epochs": 60, "noise": 0.1, "chdrop": 0.1, "cosine": True,
        "kfold": IV.KFOLD}


def run(rates, vel, nb, cfg):
    return R.run_nn(IV.make_trials(rates, vel, nb), cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/indy_loco/indy_20161005_06.mat")
    args = ap.parse_args()
    p = ROOT / args.file if not Path(args.file).is_absolute() else Path(args.file)
    rates, vel, nb = VL.load_smoothed(str(p), 3.0)      # 3 Hz vel-LP
    print(f"=== monkey tune | {p.name} | vel-LP 3Hz | {rates.shape[0]} units ===\n")

    def show(tag, r, t0):
        print(f"{tag:34s} r_mean={r.mean():.3f} "
              f"(a1={r[0]:.3f} a2={r[1]:.3f})   [{time.time()-t0:.0f}s]", flush=True)

    best_sig, best_r = 0, -1
    for sig in (0, 1, 2, 3):
        rs = gaussian_filter1d(rates, sig, axis=1) if sig else rates
        t0 = time.time(); r = run(rs, vel, nb, BASE)
        show(f"rate-smooth sigma={sig} bidir", r, t0)
        if r.mean() > best_r:
            best_r, best_sig = r.mean(), sig
    print(f"\nbest rate-smooth sigma = {best_sig} (bidir r={best_r:.3f})")
    print("--- real-time (causal unidirectional) at best sigma ---")
    rs = gaussian_filter1d(rates, best_sig, axis=1) if best_sig else rates
    t0 = time.time()
    show(f"sigma={best_sig} CAUSAL unidir", run(rs, vel, nb, {**BASE, "bidir": False}), t0)
    print("\ndone")


if __name__ == "__main__":
    main()
