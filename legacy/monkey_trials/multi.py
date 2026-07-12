#!/usr/bin/env python
"""Resumable multi-session finger-velocity decode on the NHP reaching dataset.

Processes indy sessions one at a time and writes results to disk AFTER EACH one,
so a shutdown at any moment loses at most the single in-flight session. On
restart it skips sessions already in the results file. Each .mat is downloaded,
decoded, then deleted (unless --keep) to keep disk usage light.

Model: our best TCN+GRU (units-as-channels), 5-block CV, Pearson r vs true
fingertip velocity. Reuses tools/indy_velocity + tools/way_gal_kin_research.

Usage: py tools/indy_multi.py            # process all remaining sessions
       py tools/indy_multi.py --keep     # keep .mat files
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import legacy.monkey_trials.velocity as V
import models.tcn_gru.best_model as R

DATA = ROOT / "data" / "indy_loco"
OUT = ROOT / "results" / "metrics" / "indy_multi.json"
URL = "https://zenodo.org/records/3854034/files/{}?download=1"

SESSIONS = [
    "indy_20160407_02", "indy_20160411_01", "indy_20160411_02",
    "indy_20160418_01", "indy_20160419_01", "indy_20160420_01",
    "indy_20160426_01", "indy_20160622_01", "indy_20160624_03",
    "indy_20160627_01", "indy_20160630_01", "indy_20160915_01",
    "indy_20160916_01", "indy_20160921_01", "indy_20160927_04",
    "indy_20160927_06", "indy_20160930_02", "indy_20160930_05",
    "indy_20161005_06", "indy_20161006_02", "indy_20161007_02",
    "indy_20161011_03", "indy_20161013_03", "indy_20161014_04",
    "indy_20161017_02", "indy_20161024_03", "indy_20161025_04",
    "indy_20161026_03", "indy_20161027_03", "indy_20161206_02",
    "indy_20161207_02", "indy_20161212_02", "indy_20161220_02",
    "indy_20170123_02", "indy_20170124_01", "indy_20170127_03",
    "indy_20170131_02",
]

CFG = {**R.BASE, "dils": [1, 2, 4, 8, 16], "H": 64, "L": 2, "F": 64,
       "act": "relu",                               # ReLU = monkey default (LOG-038)
       "epochs": 60, "noise": 0.1, "chdrop": 0.1, "cosine": True,
       "kfold": V.KFOLD}


def load_results():
    if OUT.exists():
        return json.loads(OUT.read_text())
    return {}


def save_results(res):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(res, indent=2), encoding="utf-8")
    os.replace(tmp, OUT)                       # atomic: never leaves partial file


def process(name, keep):
    path = DATA / f"{name}.mat"
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL.format(f"{name}.mat"), path)
    try:
        rates, vel, nb = V.load(str(path))
        trials = V.make_trials(rates, vel, nb)
        r = R.run_nn(trials, CFG)
        res = {"r_mean": float(r.mean()), "r": [float(x) for x in r],
               "n_units": int(rates.shape[0]), "n_windows": len(trials),
               "n_bins": int(nb)}
    finally:
        if not keep and path.exists():
            path.unlink()                      # free disk; redownload if needed
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="max sessions this run")
    args = ap.parse_args()

    res = load_results()
    todo = [s for s in SESSIONS if s not in res]
    print(f"=== NHP multi-session finger-velocity decode (resumable) ===")
    print(f"{len(res)} done, {len(todo)} remaining of {len(SESSIONS)}\n", flush=True)

    n = 0
    for name in todo:
        if args.limit and n >= args.limit:
            break
        t0 = time.time()
        try:
            r = process(name, args.keep)
            res[name] = r
            save_results(res)                  # checkpoint after EACH session
            done = [v["r_mean"] for v in res.values() if "r_mean" in v]
            print(f"[{len(res)}/{len(SESSIONS)}] {name}: r_mean={r['r_mean']:.3f} "
                  f"({r['n_units']} units)  running mean={np.mean(done):.3f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        except Exception as e:                 # record + continue; stay resumable
            res[name] = {"error": str(e)[:200]}
            save_results(res)
            print(f"[skip] {name}: {e}", flush=True)
        n += 1

    ok = [v["r_mean"] for v in res.values() if "r_mean" in v]
    if ok:
        print(f"\n{len(ok)} sessions done | MEAN r = {np.mean(ok):.3f} "
              f"+/- {np.std(ok):.3f} | min {np.min(ok):.3f} max {np.max(ok):.3f}",
              flush=True)


if __name__ == "__main__":
    main()
