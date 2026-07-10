#!/usr/bin/env python
"""Apply our best velocity decoder (TCN+GRU) to the NHP reaching dataset.

Dataset: O'Doherty et al. "Nonhuman Primate Reaching with Multichannel
Sensorimotor Cortex Electrophysiology" (Zenodo 3854034). Intracortical spikes
(Utah arrays, M1/S1) + fingertip position @ 250 Hz. NOT scalp EEG -- but the
same seq2seq TCN+GRU applies: spiking units become the input "channels", and we
decode fingertip VELOCITY.

Pipeline: bin spikes -> firing-rate matrix (units x time), window into chunks,
TCN+GRU -> per-timestep velocity, scored by Pearson r under leave-one-
contiguous-block-out CV. Reuses build_net / run_nn from way_gal_kin_research.

Usage: py tools/indy_velocity.py --file data/indy_loco/indy_20161005_06.mat
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import h5py

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import models.best_model as R

BIN = 0.04          # 40 ms bins -> 25 Hz
WIN = 2.0           # 2 s windows (40 bins)
KFOLD = 5           # contiguous time blocks
MIN_SPK = 10        # drop near-empty units


def load(path):
    f = h5py.File(path, "r")
    t = np.array(f["t"]).squeeze()                       # (N,) seconds @250Hz
    fp = np.array(f["finger_pos"])                       # (3, N)
    sp = f["spikes"]                                     # (units, chan) of refs
    edges = np.arange(t[0], t[-1], BIN)
    centers = edges[:-1] + BIN / 2
    nb = len(edges) - 1
    # firing-rate matrix (units x bins)
    rates = []
    for i in range(sp.shape[0]):
        for j in range(sp.shape[1]):
            st = np.array(f[sp[i, j]]).squeeze()
            if st.ndim == 0 or st.size < MIN_SPK:
                continue
            rates.append(np.histogram(st, bins=edges)[0])
    rates = np.asarray(rates, dtype=np.float32)          # (n_units, nb)
    # 2D finger velocity (top-2 movement axes)
    pos_b = np.stack([np.interp(centers, t, fp[a]) for a in range(fp.shape[0])], 1)
    vel = np.gradient(pos_b, BIN, axis=0)                # (nb, 3)
    axes = np.sort(np.argsort(vel.std(0))[-2:])          # top-2 moving axes
    return rates, vel[:, axes].astype(np.float32), nb


def make_trials(rates, vel, nb):
    w = int(round(WIN / BIN))                            # bins per window
    n_win = nb // w
    trials = []
    for k in range(n_win):
        sl = slice(k * w, (k + 1) * w)
        block = int(k * KFOLD / n_win) + 1               # contiguous block id
        trials.append({"e": rates[:, sl], "vel": vel[sl], "series": block})
    return trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/indy_loco/indy_20161005_06.mat")
    args = ap.parse_args()
    rates, vel, nb = load(ROOT / args.file if not Path(args.file).is_absolute()
                          else args.file)
    trials = make_trials(rates, vel, nb)
    print(f"=== NHP finger-velocity decode | {Path(args.file).name} ===")
    print(f"{rates.shape[0]} units, {nb} bins @ {1/BIN:.0f}Hz, "
          f"{len(trials)} windows of {int(WIN/BIN)} bins, {KFOLD}-block CV\n")

    cfg = {**R.BASE, "dils": [1, 2, 4, 8, 16], "H": 64, "L": 2, "F": 64,
           "act": "relu",                           # ReLU = monkey default (LOG-038)
           "epochs": 80, "noise": 0.1, "chdrop": 0.1, "cosine": True,
           "kfold": KFOLD}
    npar = sum(p.numel() for p in R.build_net(cfg, rates.shape[0]).parameters())
    print(f"TCN+GRU: {npar:,} params ({npar*4/1e6:.2f} MB)")

    import time
    t0 = time.time()
    r = R.run_nn(trials, cfg)
    print(f"\nTCN+GRU finger-velocity  r_mean={r.mean():.3f} "
          f"(axis1={r[0]:.3f} axis2={r[1]:.3f})   [{time.time()-t0:.0f}s]")

    # linear reference on same trials
    rl = R.run_linear(trials, nlag=8, kfold=KFOLD)
    print(f"lagged-linear reference  r_mean={rl.mean():.3f} "
          f"(axis1={rl[0]:.3f} axis2={rl[1]:.3f})")

    out = ROOT / "results" / "metrics" / f"indy_velocity_{Path(args.file).stem}.json"
    out.write_text(json.dumps({"file": Path(args.file).name, "n_units": int(rates.shape[0]),
                               "tcn_gru": {"r_mean": float(r.mean()), "r": r.tolist()},
                               "linear": {"r_mean": float(rl.mean()), "r": rl.tolist()},
                               "params": int(npar)}, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
