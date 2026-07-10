#!/usr/bin/env python
"""Benchmark the TCN+GRU sequence model on WAY-EEG-GAL (grasp-and-lift).

Runs the raw-EEG TCN+GRU under the same subject-specific K-fold + leave-one-
series-out CV as our other techniques, on both the move-vs-rest positive control
and the hard 3-class weight decode.

Usage: py tools/tcn_gru_bench.py --subjects P1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.next_experiments as nx
import tools.way_gal_probe as wg
from src.config import load_config
from src.tcn_gru import TCNGRUClassifier


def deep_fold(build):
    def fold(etr, ftr, ytr, ete, fte):
        clf = build(); clf.fit(etr, ftr, ytr); return clf.predict(ete, fte)
    return fold


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="*", default=["P1"])
    args = ap.parse_args()
    cfg = load_config(None)
    report = {}
    for task in ("move_rest", "weight"):
        win = 1.5 if task == "move_rest" else 2.5
        eeg = wg.build_epochs(args.subjects, task, win_sec=win)
        nx.CLASSES = eeg.classes
        chance = 1.0 / len(eeg.classes)
        fn0 = np.zeros((eeg.n_trials, 0), dtype=np.float64)
        per = {c: int((eeg.y == i).sum()) for i, c in enumerate(eeg.classes)}
        print(f"\n=== WAY-EEG-GAL {task} | {args.subjects} | {eeg.n_trials} trials, "
              f"{eeg.n_channels}ch x {eeg.n_times} @ {eeg.sfreq:g}Hz | "
              f"chance={chance:.2f} {per} ===")
        t0 = time.time()
        r = nx.run_within_subject(
            eeg, fn0,
            deep_fold(lambda: TCNGRUClassifier(eeg_decimate=5, epochs=80,
                                               patience=15, dropout=0.3)),
            cfg, seed=42)
        report[task] = r
        s, l = r["subject"], r["loro"]
        print(f"TCN+GRU  subj={s['mean_subject_acc']:.3f} "
              f"(bal={s['pooled_bal_acc']:.3f}) | LORO={l['mean_subject_acc']:.3f} "
              f"(bal={l['pooled_bal_acc']:.3f})   [{time.time()-t0:.0f}s]")

    out = ROOT / "results" / "metrics" / f"way_gal_tcngru_{'_'.join(args.subjects)}.json"
    out.write_text(json.dumps({"subjects": args.subjects, "results": report},
                              indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
