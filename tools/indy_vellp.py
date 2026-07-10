#!/usr/bin/env python
"""Velocity-target low-pass sweep on the NHP monkey data (increase decoder r).

We never low-passed the velocity target for the monkey pipeline -- it uses the
raw numerical gradient of finger position, which is noisy. Arm/finger velocity is
band-limited (~<5 Hz), so a low-pass (incl 3 Hz) removes derivative/marker jitter
while keeping real movement. Sweep the cutoff and report Pearson r (TCN+GRU,
within-session 5-block CV). Note: r is for decoding the smoothed velocity;
cutoff is reported transparently.

Usage: py tools/indy_vellp.py --file data/indy_loco/indy_20161005_06.mat
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import h5py
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.indy_velocity as IV
import tools.way_gal_kin_research as R

CUTOFFS = [None, 8.0, 6.0, 4.0, 3.0, 2.0]
CFG = {**R.BASE, "dils": [1, 2, 4, 8, 16], "H": 64, "L": 2, "F": 64,
       "act": "relu",                               # ReLU = monkey default (LOG-038)
       "epochs": 60, "noise": 0.1, "chdrop": 0.1, "cosine": True,
       "kfold": IV.KFOLD}


def load_smoothed(path, lp):
    f = h5py.File(path, "r")
    t = np.array(f["t"]).squeeze()
    fp = np.array(f["finger_pos"])
    sp = f["spikes"]
    edges = np.arange(t[0], t[-1], IV.BIN)
    centers = edges[:-1] + IV.BIN / 2
    rates = []
    for i in range(sp.shape[0]):
        for j in range(sp.shape[1]):
            st = np.array(f[sp[i, j]]).squeeze()
            if st.ndim == 0 or st.size < IV.MIN_SPK:
                continue
            rates.append(np.histogram(st, bins=edges)[0])
    rates = np.asarray(rates, dtype=np.float32)
    pos = np.stack([np.interp(centers, t, fp[a]) for a in range(fp.shape[0])], 1)
    if lp is not None:                        # low-pass position at bin rate
        sos = butter(4, lp / (0.5 / IV.BIN), btype="low", output="sos")
        pos = sosfiltfilt(sos, pos, axis=0)
    vel = np.gradient(pos, IV.BIN, axis=0)
    axes = np.sort(np.argsort(vel.std(0))[-2:])           # top-2 movement axes (2D)
    return rates, vel[:, axes].astype(np.float32), len(edges) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/indy_loco/indy_20161005_06.mat")
    args = ap.parse_args()
    p = ROOT / args.file if not Path(args.file).is_absolute() else Path(args.file)
    print(f"=== velocity low-pass sweep | {p.name} | TCN+GRU, {IV.KFOLD}-block CV ===\n")
    for lp in CUTOFFS:
        rates, vel, nb = load_smoothed(str(p), lp)
        trials = IV.make_trials(rates, vel, nb)
        t0 = time.time()
        r = R.run_nn(trials, CFG)
        tag = "none (raw gradient)" if lp is None else f"{lp:g} Hz"
        print(f"vel-LP {tag:<20} r_mean={r.mean():.3f} "
              f"(axis1={r[0]:.3f} axis2={r[1]:.3f})   [{time.time()-t0:.0f}s]",
              flush=True)
    print("\ndone")


if __name__ == "__main__":
    main()
