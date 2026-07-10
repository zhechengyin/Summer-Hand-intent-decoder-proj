#!/usr/bin/env python
"""Full-capacity + artifact-controlled EEG->velocity decoding on WAY-EEG-GAL.

Improvements over way_gal_kinematics.py:
  * memory-safe at high capacity: accumulate per-SERIES X^T X / X^T Y (d x d),
    never materialise the full stacked design matrix (no OOM);
  * finer time resolution (50 Hz, +/-240 ms lags);
  * ridge penalty tuned per outer fold via an inner validation series (no leak);
  * two channel sets: ALL (ceiling) vs MOTOR-only (artifact control -- drops
    frontal EOG and temporal EMG channels).

Metric: Pearson r (pred vs true velocity) per axis, leave-one-series-out.
Usage: py tools/way_gal_kin_full.py --subjects P1 P2 P3
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
DECIM = 10                      # -> 50 Hz
LP_HZ = 4.0
NLAG = 12                       # +/-240 ms at 50 Hz
ALPHAS = [1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5]
PX, PY, PZ = 18, 22, 26
MARKERS = [2, 3, 4]             # moving hand/finger markers (sensor 1 ~ static)
# central sensorimotor set (exclude frontal EOG Fp/F7/F8, temporal EMG T7/T8/TP)
MOTOR = {"F3", "Fz", "F4", "FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4",
         "CP5", "CP1", "CP2", "CP6", "P3", "Pz", "P4"}


def _files(subj):
    return sorted(glob.glob(str(DATA / "**" / f"WS_{subj}_S*.mat"), recursive=True))


def _lp_dec(x, axis):
    sos = butter(4, LP_HZ / (FS / 2), btype="low", output="sos")
    return sosfiltfilt(sos, x, axis=axis).take(np.arange(0, x.shape[axis], DECIM),
                                               axis=axis)


def load(subj):
    """Return trials [{e:(nch,t), vel:{m:(t,3)}, series}], eeg channel names."""
    trials, names = [], None
    for series, f in enumerate(_files(subj), 1):
        ws = loadmat(f, struct_as_record=False, squeeze_me=True)["ws"]
        if names is None:
            names = list(ws.names.eeg)
        for w in np.atleast_1d(ws.win):
            eeg = np.asarray(w.eeg, dtype=np.float64).T
            kin = np.asarray(w.kin, dtype=np.float64)
            e = _lp_dec(eeg, 1)                                   # (nch, t)
            vels, ok = {}, True
            for m in MARKERS:
                cols = [PX + m - 1, PY + m - 1, PZ + m - 1]
                pos = kin[:, cols]
                if np.isnan(pos).any():
                    ok = False; break
                pos = _lp_dec(pos, 0)
                vels[m] = np.gradient(pos, 1.0 / (FS / DECIM), axis=0)
            if not ok:
                continue
            t = min(e.shape[1], min(v.shape[0] for v in vels.values()))
            trials.append({"e": e[:, :t], "vel": {m: vels[m][:t] for m in MARKERS},
                           "series": series})
    return trials, names


def _design(e):
    feats = [np.roll(e, s, axis=1) for s in range(-NLAG, NLAG + 1)]
    return np.concatenate(feats, axis=0).T[NLAG:e.shape[1] - NLAG]     # (t', d)


def corr(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    d = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    return (a * b).sum(0) / np.where(d == 0, 1e-9, d)


def decode(trials, ch_idx):
    d = len(ch_idx) * (2 * NLAG + 1)
    # global per-channel z-score (scaling only -> negligible leakage)
    allc = np.concatenate([t["e"][ch_idx] for t in trials], axis=1)
    mu, sd = allc.mean(1, keepdims=True), allc.std(1, keepdims=True)
    sd[sd == 0] = 1.0
    series_ids = sorted({t["series"] for t in trials})
    # per-series accumulators: XtX_s (d,d) shared; XtY_s[m] (d,3); test cache
    XtX = {s: np.zeros((d, d)) for s in series_ids}
    XtY = {s: {m: np.zeros((d, 3)) for m in MARKERS} for s in series_ids}
    cache = {s: [] for s in series_ids}
    for t in trials:
        e = (t["e"][ch_idx] - mu) / sd
        X = _design(e).astype(np.float64)
        XtX[t["series"]] += X.T @ X
        for m in MARKERS:
            Y = t["vel"][m][NLAG:t["e"].shape[1] - NLAG]
            XtY[t["series"]][m] += X.T @ Y
        cache[t["series"]].append((X, {m: t["vel"][m][NLAG:t["e"].shape[1] - NLAG]
                                       for m in MARKERS}))
    I = np.eye(d)
    out = {}
    for m in MARKERS:
        fold_r = []
        for h in series_ids:
            train = [s for s in series_ids if s != h]
            XtX_tr = sum(XtX[s] for s in train)
            XtY_tr = sum(XtY[s][m] for s in train)
            v = train[len(train) // 2]                      # inner val series
            XtX_in, XtY_in = XtX_tr - XtX[v], XtY_tr - XtY[v][m]
            best_a, best_r = ALPHAS[0], -1
            for a in ALPHAS:
                w = np.linalg.solve(XtX_in + a * I, XtY_in)
                yp = np.vstack([X @ w for X, _ in cache[v]])
                yt = np.vstack([vd[m] for _, vd in cache[v]])
                r = corr(yt, yp).mean()
                if r > best_r:
                    best_r, best_a = r, a
            w = np.linalg.solve(XtX_tr + best_a * I, XtY_tr)
            yp = np.vstack([X @ w for X, _ in cache[h]])
            yt = np.vstack([vd[m] for _, vd in cache[h]])
            fold_r.append(corr(yt, yp))
        r = np.mean(fold_r, axis=0)
        out[m] = {"r_mean": float(r.mean()), "r_x": float(r[0]),
                  "r_y": float(r[1]), "r_z": float(r[2])}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="*", default=["P1", "P2", "P3"])
    args = ap.parse_args()
    print("=== WAY-EEG-GAL EEG->velocity: FULL-capacity + artifact-controlled ===")
    print(f"LP {LP_HZ}Hz, {FS/DECIM:.0f}Hz, +/-{NLAG} lags, ridge alpha tuned; "
          f"leave-one-series-out; r = Pearson (pred vs true velocity)\n")
    report = {}
    for subj in args.subjects:
        trials, names = load(subj)
        all_idx = list(range(len(names)))
        motor_idx = [i for i, n in enumerate(names) if n in MOTOR]
        report[subj] = {}
        for tag, idx in [("all_ch", all_idx), ("motor_ch", motor_idx)]:
            res = decode(trials, idx)
            report[subj][tag] = res
            best = max(res.values(), key=lambda r: r["r_mean"])
            avg = float(np.mean([res[m]["r_mean"] for m in MARKERS]))
            print(f"{subj} {tag:9s} ({len(idx)}ch): "
                  f"best marker r_mean={best['r_mean']:.3f} "
                  f"(x={best['r_x']:.3f} y={best['r_y']:.3f} z={best['r_z']:.3f}) "
                  f"| mean over markers={avg:.3f}", flush=True)
    out = ROOT / "results" / "metrics" / f"way_gal_kin_full_{'_'.join(args.subjects)}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
