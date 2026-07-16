#!/usr/bin/env python
"""Causal TCN+GRU benchmark on the U-M Deep Blue finger SBP dataset.

Reuses the Indy/Loco model architecture for a closely related task:

    96-channel intracortical spiking-band power (SBP)
        -> index-finger and MRS-finger-group flexion velocity

Only the manipulandum-control (MC) files are used because their labels are
physical finger measurements. KC/RC labels are decoder-generated feedback, not
physical ground truth. Each monkey is modeled separately because electrode
channels do not correspond across implants.

The released MC ``X`` matrices contain four interleaved historical values per
channel (384 columns). The official ``OlVsClTuning.m`` removes history with
``X(:,1:4:end)``; we do the same and let the causal TCN+GRU learn history itself.

Splits are chronological and trial-disjoint: 70% train, 15% eval, 15% test.
Model/epoch selection uses EVAL only; TEST is reported after selection.

Usage:
    py research/deepblue_tcn_gru.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import models.tcn_gru.evaluate as E
import research.harness as H

DATA = ROOT / "data" / "external" / "umich_deepblue_finger" / "original" / "Data"
FILES = {"MonkeyN": DATA / "MonkeyN_MC.mat", "MonkeyW": DATA / "MonkeyW_MC.mat"}
OUT = ROOT / "results" / "metrics" / "deepblue_tcn_gru.json"

BIN_S = 0.032                    # released MC data are integrated into 32 ms bins
WINDOW_BINS = 32                 # 1.024 s sequences, reset at trial boundaries
SEEDS = (42, 1, 7)
CFG = {
    **E.CFG,
    "F": 64,
    "H": 64,
    "L": 1,
    "dils": [1, 2, 4, 8],
    "bidir": False,
    "n_out": 2,
}


def load_mc(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return current-bin SBP (T,96), physical velocity (T,2), trial IDs."""
    d = loadmat(path, variable_names=["X", "y", "trials"])
    x = np.asarray(d["X"][:, 0::4], dtype=np.float32)  # official history removal
    y = np.asarray(d["y"][:, 2:4], dtype=np.float32)   # index/MRS velocities
    trials = np.asarray(d["trials"]).reshape(-1).astype(np.int64)
    if x.shape[1] != 96 or y.shape[1] != 2 or len(x) != len(y):
        raise ValueError(f"Unexpected MC shapes: X={x.shape}, y={y.shape}")
    return x, y, trials


def chronological_split(trials: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.unique(trials)
    a, b = int(0.70 * len(ids)), int(0.85 * len(ids))
    return ids[:a], ids[a:b], ids[b:]


def window_trials(x: np.ndarray, y: np.ndarray, trial_ids: np.ndarray,
                  selected: np.ndarray) -> list[dict[str, np.ndarray]]:
    """Make non-overlapping sequences without crossing trial boundaries."""
    out = []
    for tr in selected:
        idx = np.flatnonzero(trial_ids == tr)
        for start in range(0, len(idx) - WINDOW_BINS + 1, WINDOW_BINS):
            ii = idx[start:start + WINDOW_BINS]
            out.append({"e": x[ii].T.copy(), "vel": y[ii].copy()})
    return out


def prepare(path: Path) -> tuple[dict, dict]:
    x, y, trials = load_mc(path)
    tr_ids, ev_ids, te_ids = chronological_split(trials)

    # Input normalization is fit on TRAIN only and then frozen for EVAL/TEST.
    tr_mask = np.isin(trials, tr_ids)
    mu = x[tr_mask].mean(0, keepdims=True)
    sd = x[tr_mask].std(0, keepdims=True) + 1e-6
    x = ((x - mu) / sd).astype(np.float32)

    train = window_trials(x, y, trials, tr_ids)
    eval_ = window_trials(x, y, trials, ev_ids)
    test = window_trials(x, y, trials, te_ids)
    if not train or not eval_ or not test:
        raise ValueError("A split produced no full trial-bounded windows")
    data = {"train": train, "eval": {"eval": eval_}, "test": {"test": test}}
    meta = {
        "n_samples": int(len(x)),
        "n_trials": int(len(np.unique(trials))),
        "trial_counts": [int(len(tr_ids)), int(len(ev_ids)), int(len(te_ids))],
        "window_counts": [len(train), len(eval_), len(test)],
        "bin_s": BIN_S,
        "window_bins": WINDOW_BINS,
    }
    return data, meta


def main() -> None:
    print("=== Deep Blue finger SBP: causal TCN+GRU, 3-seed ===", flush=True)
    print("    physical MC labels; chronological trial-disjoint split; select on EVAL\n",
          flush=True)
    rows = {}
    for monkey, path in FILES.items():
        t0 = time.time()
        data, meta = prepare(path)
        print(f"  {monkey}: windows train/eval/test={meta['window_counts']}", flush=True)
        res = H.run(data, CFG, seeds=SEEDS, verbose=True)
        rows[monkey] = {
            **meta,
            "eval_r": res["eval_r"],
            "eval_r2": res["eval_r2"],
            "test_r": res["test_r"],
            "test_r2": res["test_r2"],
            "test_r2_axes": res["test_r2_ax"],
            "n_params": res["n_params"],
            "elapsed_s": time.time() - t0,
        }
        print(f"  {monkey}: EVAL R2={res['eval_r2']:.3f}  "
              f"TEST R2={res['test_r2']:.3f}  axes={res['test_r2_ax']}\n", flush=True)

    rows["mean"] = {
        "eval_r2": float(np.mean([rows[m]["eval_r2"] for m in FILES])),
        "test_r2": float(np.mean([rows[m]["test_r2"] for m in FILES])),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Mean EVAL R2={rows['mean']['eval_r2']:.3f}  "
          f"Mean TEST R2={rows['mean']['test_r2']:.3f}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
