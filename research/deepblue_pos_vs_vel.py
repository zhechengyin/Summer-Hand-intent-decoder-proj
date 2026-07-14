#!/usr/bin/env python
"""Deep Blue: position vs velocity decodability (plain causal TCN+GRU, single-scale).

Clarifies what the model targets. MC `y` columns: 0:2 = finger-group POSITION
(normalized flexion 0..1), 2:4 = finger-group VELOCITY (verified: col2 ~= d(col0)/dt,
r=0.91). The model of record (LOG-075) already decodes VELOCITY (y[:,2:4]) at
mean TEST R2 ~0.408. This A/B adds the POSITION decode (y[:,0:2]) under the exact
same pipeline so the two are directly comparable. Position is smoother (integrated)
and typically decodes higher. Everything else identical to research/deepblue_tcn_gru.py.
Usage: py research/deepblue_pos_vs_vel.py
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

import research.deepblue_tcn_gru as db
import research.harness as H

SEEDS = (42, 1, 7)
TARGETS = {"position": (0, 2), "velocity": (2, 4)}     # y column ranges
OUT = ROOT / "results" / "metrics" / "deepblue_pos_vs_vel.json"


def load_xy(path, cols):
    d = loadmat(path, variable_names=["X", "y", "trials"])
    x = np.asarray(d["X"][:, 0::4], dtype=np.float32)          # history removed
    y = np.asarray(d["y"][:, cols[0]:cols[1]], dtype=np.float32)
    trials = np.asarray(d["trials"]).reshape(-1).astype(np.int64)
    return x, y, trials


def prepare(path, cols):
    x, y, trials = load_xy(path, cols)
    tr_ids, ev_ids, te_ids = db.chronological_split(trials)
    tr_mask = np.isin(trials, tr_ids)
    mu, sd = x[tr_mask].mean(0, keepdims=True), x[tr_mask].std(0, keepdims=True) + 1e-6
    x = ((x - mu) / sd).astype(np.float32)
    train = db.window_trials(x, y, trials, tr_ids)
    eval_ = db.window_trials(x, y, trials, ev_ids)
    test = db.window_trials(x, y, trials, te_ids)
    return {"train": train, "eval": {"eval": eval_}, "test": {"test": test}}


def main():
    print("=== Deep Blue: position vs velocity decode (plain causal TCN+GRU, 3-seed) ===\n",
          flush=True)
    rows = {}
    for monkey, path in db.FILES.items():
        rows[monkey] = {}
        for tname, cols in TARGETS.items():
            t0 = time.time()
            res = H.run(prepare(path, cols), db.CFG, seeds=SEEDS)
            rows[monkey][tname] = {"eval_r2": res["eval_r2"], "test_r2": res["test_r2"],
                                   "test_r": res["test_r"]}
            print(f"  {monkey} {tname:9s} EVAL R2={res['eval_r2']:.3f}  "
                  f"TEST R2={res['test_r2']:.3f}  (r={res['test_r']:.3f})  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        print(flush=True)

    print("  --- mean over monkeys ---")
    for tname in TARGETS:
        ev = float(np.mean([rows[m][tname]["eval_r2"] for m in db.FILES]))
        te = float(np.mean([rows[m][tname]["test_r2"] for m in db.FILES]))
        rows.setdefault("mean", {})[tname] = {"eval_r2": ev, "test_r2": te}
        print(f"    {tname:9s} EVAL {ev:.3f}  TEST {te:.3f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
