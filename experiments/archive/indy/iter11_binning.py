#!/usr/bin/env python
"""Iteration 11: binning / input-representation sweep at 8 channels, 24 sessions.

Re-bins from RAW spike times with a CAUSAL window, decoupling the integration
window (how long we sum spikes) from the output stride (how often we predict):
  * boxcar bin-size:  20 / 40 / 80 ms  (integration == stride)
  * overlapping:      80 ms integration @ 40 ms and @ 20 ms stride (paper-style)
  * near-continuous:  10 ms boxcar (approximates "no bins")
Fixed 8 electrodes (checkpoint channels), 2D velocity, 'small' model for speed.
Compares against the 40 ms boxcar internal reference (same causal pipeline).
Usage: py experiments/archive/indy/iter11_binning.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import h5py
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.common.harness as H
import experiments.archive.indy.iter5_scale as I5
import experiments.archive.indy.iter7_final as I7
import models.tcn_gru.evaluate as E

SMALL = dict(F=32, H=32, L=1, dils=[1, 2, 4, 8])
CHANNELS = np.array([26, 51, 53, 66, 71, 73, 75, 94])     # fixed 8 electrodes (checkpoint)
AXES = np.array([1, 2])
VEL_LP, SIGMA_S, WIN_S = 3.0, 0.04, 2.0                    # sigma in seconds (~1 bin @40ms)

# (label, integration_s, stride_s)
CONFIGS = [
    ("10ms_cont", 0.010, 0.010),
    ("20ms",      0.020, 0.020),
    ("40ms_ref",  0.040, 0.040),
    ("80ms",      0.080, 0.080),
    ("80/40_ovl", 0.080, 0.040),
    ("80/20_ovl", 0.080, 0.020),
]

_RAW = {}


def load_raw8(name):
    """Cache the 8 electrodes' sorted spike times + t + finger_pos from raw .mat."""
    if name in _RAW:
        return _RAW[name]
    f = h5py.File(E.fetch(name), "r")
    t = np.array(f["t"]).squeeze()
    fp = np.array(f["finger_pos"])
    sp = f["spikes"]
    st_by = []
    for ch in CHANNELS:
        allst = []
        for u in range(sp.shape[0]):
            st = np.array(f[sp[u, ch]]).squeeze()
            if st.ndim and st.size:
                allst.append(np.atleast_1d(st))
        st_by.append(np.sort(np.concatenate(allst)) if allst else np.zeros(0))
    _RAW[name] = (st_by, t, fp)
    return _RAW[name]


def bin_causal(name, integ_s, stride_s):
    st_by, t, fp = load_raw8(name)
    centers = np.arange(t[0] + integ_s, t[-1], stride_s)        # causal: full window available
    rates = np.zeros((len(st_by), len(centers)), dtype=np.float32)
    for i, st in enumerate(st_by):
        if st.size:
            rates[i] = (np.searchsorted(st, centers, "right")
                        - np.searchsorted(st, centers - integ_s, "right"))
    sig = SIGMA_S / stride_s                                    # keep smoothing ~constant in time
    if sig > 0:
        rates = gaussian_filter1d(rates, sig, axis=1).astype(np.float32)
    pos = np.stack([np.interp(centers, t, fp[a]) for a in range(fp.shape[0])], 1)
    sos = butter(4, VEL_LP / (0.5 / stride_s), btype="low", output="sos")
    pos = sosfiltfilt(sos, pos, axis=0)
    vel = np.gradient(pos, stride_s, axis=0)[:, AXES].astype(np.float32)
    return rates, vel


def windows(rates, vel, stride_s):
    mu, sd = rates.mean(1, keepdims=True), rates.std(1, keepdims=True) + 1e-6
    r = ((rates - mu) / sd).astype(np.float32)
    w = int(round(WIN_S / stride_s))
    return [{"e": r[:, k * w:(k + 1) * w], "vel": vel[k * w:(k + 1) * w]}
            for k in range(r.shape[1] // w)]


def prep(integ_s, stride_s):
    names = list(E.TRAIN) + I7.EXTRA18                          # 24 sessions
    tr = []
    for s in names:
        tr += windows(*bin_causal(s, integ_s, stride_s), stride_s)
    ev = {s: windows(*bin_causal(s, integ_s, stride_s), stride_s) for s in E.EVAL}
    te = {s: windows(*bin_causal(s, integ_s, stride_s), stride_s) for s in E.TEST}
    return {"train": tr, "eval": ev, "test": te}


def main():
    cfg = {**E.CFG, **SMALL}
    print("=== Iteration 11: binning sweep (8 ch, 24 sess, small model) ===")
    print("NOTE: R2 at different output rates is not perfectly comparable (finer =",
          "more, noisier samples); overlap rows share their stride's rate.\n", flush=True)
    DONE = {"10ms_cont"}                       # completed in LOG-059 partial (0.642, slow)
    rows = {}
    for label, integ, stride in CONFIGS:
        if label in DONE:
            continue
        data = prep(integ, stride)
        t0 = time.time()
        res = H.run(data, cfg)
        rows[label] = {**res, "integ_ms": integ * 1000, "stride_ms": stride * 1000,
                       "rate_hz": 1 / stride}
        print(f"  {label:10s} integ {integ*1000:3.0f}ms stride {stride*1000:3.0f}ms "
              f"({1/stride:4.0f}Hz, T={int(WIN_S/stride)}): TEST R2={res['test_r2']:.3f}"
              f"  EVAL R2={res['eval_r2']:.3f}  [{time.time()-t0:.0f}s]", flush=True)

    print("\n--- binning vs R2 (ref 40ms_ref) ---")
    for label, r in rows.items():
        print(f"  {label:10s}: R2={r['test_r2']:.3f}  ({r['rate_hz']:.0f} Hz out)")
    out = ROOT / "results" / "metrics" / "iter11_binning.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
