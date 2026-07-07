#!/usr/bin/env python
"""Decode hand/finger VELOCITY from EEG on WAY-EEG-GAL (continuous regression).

The canonical "decode movement through brain signals" use of this dataset: the
kin block carries 3D positions of 4 markers (Px/Py/Pz 1-4 = hand/thumb/index-
finger/object), on the same 500 Hz grid as the EEG. We differentiate position ->
velocity and regress it from time-lagged, low-frequency EEG (movement-related
cortical potentials), scored by Pearson correlation between predicted and true
velocity under leave-one-series-out.

Usage: py tools/way_gal_kinematics.py --subjects P1
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "way_eeg_gal"
FS = 500.0
DECIM = 20                     # 500 -> 25 Hz (memory-light)
LP_HZ = 4.0                    # EEG low-pass for kinematics (slow potentials)
NLAG = 6                       # +/-lags at 25 Hz (+/-240 ms) around each sample
# position sensor columns in kin (0-based): Px1..4=18..21, Py=22..25, Pz=26..29
PX, PY, PZ = 18, 22, 26


def _series_files(subj):
    return sorted(glob.glob(str(DATA / "**" / f"WS_{subj}_S*.mat"),
                            recursive=True))


def _lowpass_decim_eeg(eeg):
    sos = butter(4, LP_HZ / (FS / 2), btype="low", output="sos")
    f = sosfiltfilt(sos, eeg, axis=1)
    return f[:, ::DECIM]                     # (32, t)


def _sensor_velocity(kin, sensor):
    """3D velocity (smoothed) of position sensor s in {1..4}, decimated."""
    cols = [PX + (sensor - 1), PY + (sensor - 1), PZ + (sensor - 1)]
    pos = kin[:, cols].astype(np.float64)                 # (T, 3)
    if np.isnan(pos).any():
        return None
    sos = butter(4, LP_HZ / (FS / 2), btype="low", output="sos")
    pos = sosfiltfilt(sos, pos, axis=0)[::DECIM]          # (t, 3)
    vel = np.gradient(pos, 1.0 / (FS / DECIM), axis=0)     # cm/s
    return vel


def load_trials(subj, sensor):
    """Return per-trial (eeg_lagged_features, velocity, series)."""
    trials = []
    for series, f in enumerate(_series_files(subj), 1):
        ws = loadmat(f, struct_as_record=False, squeeze_me=True)["ws"]
        for w in np.atleast_1d(ws.win):
            eeg = np.asarray(w.eeg, dtype=np.float64).T          # (32, T)
            kin = np.asarray(w.kin, dtype=np.float64)            # (T, 45)
            vel = _sensor_velocity(kin, sensor)
            if vel is None:
                continue
            e = _lowpass_decim_eeg(eeg)                          # (32, t)
            t = min(e.shape[1], vel.shape[0])
            e, vel = e[:, :t], vel[:t]
            if t <= 2 * NLAG + 5:
                continue
            # lagged design: EEG at offsets -NLAG..+NLAG
            feats = [np.roll(e, s, axis=1) for s in range(-NLAG, NLAG + 1)]
            X = np.concatenate(feats, axis=0).T                  # (t, 32*(2L+1))
            X, vy = X[NLAG:t - NLAG], vel[NLAG:t - NLAG]
            trials.append((X.astype(np.float32), vy.astype(np.float32), series))
    return trials


def corr(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    denom = (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0))
    return (a * b).sum(0) / np.where(denom == 0, 1e-9, denom)     # per axis


def decode(subj, sensor):
    from sklearn.linear_model import Ridge

    trials = load_trials(subj, sensor)
    if len(trials) < 6:
        return None
    series = np.array([s for _, _, s in trials])
    uniq = sorted(set(series.tolist()))
    fold_r = []
    for held in uniq:
        tr = [t for t in trials if t[2] != held]
        te = [t for t in trials if t[2] == held]
        if not te:
            continue
        Xtr = np.vstack([t[0] for t in tr]); Ytr = np.vstack([t[1] for t in tr])
        Xte = np.vstack([t[0] for t in te]); Yte = np.vstack([t[1] for t in te])
        # float32, in-place standardisation (avoids float64 scaler copies -> OOM)
        mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
        Xtr -= mu; Xtr /= sd
        Xte = (Xte - mu) / sd
        reg = Ridge(alpha=1e3).fit(Xtr, Ytr)
        pred = reg.predict(Xte)
        fold_r.append(corr(Yte, pred))                     # per-axis r
    r = np.mean(fold_r, axis=0)                             # (3,)
    return {"r_x": float(r[0]), "r_y": float(r[1]), "r_z": float(r[2]),
            "r_mean": float(r.mean()), "n_trials": len(trials),
            "n_series": len(uniq)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="*", default=["P1"])
    args = ap.parse_args()
    print(f"=== WAY-EEG-GAL EEG->velocity decoding (leave-one-series-out) ===")
    print(f"EEG low-pass {LP_HZ}Hz, decim->{FS/DECIM:.0f}Hz, +/-{NLAG} lags; "
          f"Ridge. Metric = Pearson r (pred vs true velocity).\n")
    report = {}
    for subj in args.subjects:
        report[subj] = {}
        for sensor, tag in [(1, "sensor1"), (2, "sensor2"),
                            (3, "sensor3"), (4, "sensor4")]:
            res = decode(subj, sensor)
            if res is None:
                print(f"{subj} {tag}: skipped"); continue
            report[subj][tag] = res
            print(f"{subj} {tag}: r_mean={res['r_mean']:.3f} "
                  f"(x={res['r_x']:.3f} y={res['r_y']:.3f} z={res['r_z']:.3f}) "
                  f"| {res['n_trials']} trials")
    out = ROOT / "results" / "metrics" / f"way_gal_kin_{'_'.join(args.subjects)}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
