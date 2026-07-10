#!/usr/bin/env python
"""Does WITHIN-BIN spike timing carry info beyond the count? (monkey velocity)

Controlled test of temporal vs rate coding. Everything fixed (20 Hz output, 2 s
window, 3 Hz-lowpass velocity target, ReLU TCN+GRU, 5-block CV) EXCEPT the input
representation: each 50 ms decision bin is split into K sub-bins (finer timing),
stacked as extra channels. If timing within the bin matters (e.g. 5 spikes in
10 ms vs spread over 40 ms), K>1 should beat K=1 (the plain count).

  K=1  -> 50 ms count (baseline, rate code)
  K=2  -> two 25 ms sub-bins per unit
  K=5  -> five 10 ms sub-bins per unit
  K=10 -> ten 5 ms sub-bins per unit

NOTE: no firing-rate smoothing here (it would blur the sub-bin timing we test).
Usage: py tools/indy_subbin.py --file data/indy_loco/indy_20161005_06.mat
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import h5py
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import legacy.monkey_trials.velocity as IV
import models.best_model as R

BIN = 0.04          # 40 ms bins -> 25 Hz
MIN_SPK = 10
KS = [1, 2, 5, 10]
CFG = {**R.BASE, "dils": [1, 2, 4, 8, 16], "H": 64, "L": 2, "F": 64,
       "epochs": 60, "noise": 0.1, "chdrop": 0.1, "cosine": True,
       "act": "relu", "kfold": IV.KFOLD}


def load(path, K):
    """rates: (units*K, n_dec) sub-bin counts; vel: (n_dec, 2) at 20 Hz."""
    f = h5py.File(path, "r")
    t = np.array(f["t"]).squeeze()
    fp = np.array(f["finger_pos"])
    sp = f["spikes"]
    coarse = np.arange(t[0], t[-1], BIN)
    n_dec = len(coarse) - 1
    fine_bin = BIN / K
    fine_edges = t[0] + np.arange(n_dec * K + 1) * fine_bin
    unit_rates = []
    for i in range(sp.shape[0]):
        for j in range(sp.shape[1]):
            st = np.array(f[sp[i, j]]).squeeze()
            if st.ndim == 0 or st.size < MIN_SPK:
                continue
            fc = np.histogram(st, bins=fine_edges)[0].reshape(n_dec, K)  # (n_dec,K)
            unit_rates.append(fc)
    U = np.stack(unit_rates, 0).astype(np.float32)          # (units, n_dec, K)
    if K == 1:
        rates = U[:, :, 0]                                  # (units, n_dec)
    else:
        rates = U.transpose(0, 2, 1).reshape(U.shape[0] * K, n_dec)  # (units*K, n_dec)
    centers = coarse[:-1] + BIN / 2
    pos = np.stack([np.interp(centers, t, fp[a]) for a in range(fp.shape[0])], 1)
    sos = butter(4, 3.0 / (0.5 / BIN), btype="low", output="sos")   # 3 Hz vel-LP
    vel = np.gradient(sosfiltfilt(sos, pos, axis=0), BIN, axis=0)
    axes = np.sort(np.argsort(vel.std(0))[-2:])                     # top-2 (2D)
    return rates.astype(np.float32), vel[:, axes].astype(np.float32), n_dec, U.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/indy_loco/indy_20161005_06.mat")
    args = ap.parse_args()
    p = ROOT / args.file if not Path(args.file).is_absolute() else Path(args.file)
    print(f"=== within-bin timing test | {p.name} | ReLU TCN+GRU, 5-block CV, "
          f"3 Hz vel-LP, NO rate smoothing ===\n", flush=True)
    report = {}
    for K in KS:
        rates, vel, nb, n_units = load(str(p), K)
        tri = IV.make_trials(rates, vel, nb)
        t0 = time.time()
        r = R.run_nn(tri, CFG)
        report[f"K{K}"] = {"r_mean": float(r.mean()), "r": [float(x) for x in r],
                           "n_channels": int(rates.shape[0])}
        sub = f"{int(BIN*1000/K)}ms sub-bins" if K > 1 else "50ms count (baseline)"
        print(f"K={K:<2d} ({sub:<18} {n_units} units x{K}={rates.shape[0]} ch): "
              f"r_mean={r.mean():.3f} (a1={r[0]:.3f} a2={r[1]:.3f})  "
              f"[{time.time()-t0:.0f}s]", flush=True)
    best = max(report, key=lambda k: report[k]["r_mean"])
    base = report["K1"]["r_mean"]
    print(f"\nbaseline K=1: {base:.3f} | best: {best} ({report[best]['r_mean']:.3f}, "
          f"delta {report[best]['r_mean']-base:+.3f})")
    out = ROOT / "results" / "metrics" / "indy_subbin.json"
    out.write_text(json.dumps({"file": p.name, "results": report, "best": best},
                              indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
