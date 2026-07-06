#!/usr/bin/env python
"""Architecture benchmark: our best classical front ends vs deep nets
(EEGNet-style CNN, EEG Conformer transformer) under identical CV.

- ds004362 (PhysioNet left/right hand MI, chance 0.50): EEG-only, all models.
- ds004022 (4-class same-limb, chance 0.25): fused EEG+fNIRS, best-vs-best.

Same subject-specific K-fold + leave-one-run-out protocol for every model, so
differences reflect the MODEL, not the evaluation. Deep nets are deliberately
compact (small data). Run:

    py tools/architecture_bench.py --dataset eegmmidb
    py tools/architecture_bench.py --dataset ds004022
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
from src.config import load_config
from src.conformer import FusedConformerClassifier
from src.temporal_cnn import FusedTemporalCNNClassifier


def deep_fold(build):
    def fold(etr, ftr, ytr, ete, fte):
        clf = build()
        clf.fit(etr, ftr, ytr)
        return clf.predict(ete, fte)
    return fold


def run(eeg, fnirs_X, cfg, models):
    rows = {}
    for tag, fold in models.items():
        t0 = time.time()
        r = nx.run_within_subject(eeg, fnirs_X, fold, cfg, seed=42)
        s, l = r["subject"], r["loro"]
        rows[tag] = r
        print(f"{tag:<34} subj={s['mean_subject_acc']:.4f} "
              f"(bal={s['pooled_bal_acc']:.3f}) | "
              f"LORO={l['mean_subject_acc']:.4f} "
              f"(bal={l['pooled_bal_acc']:.3f})   [{time.time()-t0:.0f}s]", flush=True)
    return rows


def bench_eegmmidb(cfg, n_subjects=5):
    from tools.eegmmidb_probe import SFREQ, build_epochs
    nx.CLASSES = ["left", "right"]
    subjects = [f"sub-{i:03d}" for i in range(1, n_subjects + 1)]
    eeg = build_epochs(subjects)
    fn0 = np.zeros((eeg.n_trials, 0), dtype=np.float64)
    bands = [(8, 13), (13, 30)]
    print(f"\n=== ds004362 left/right MI (EEG-only, chance 0.50) | "
          f"{eeg.n_trials} trials, {eeg.n_channels}ch x {eeg.n_times} @ {SFREQ:g}Hz ===")
    models = {
        "Riemannian tangent + logreg":
            nx.tangent_fold(reg=1e-3, clf="logreg", sfreq=SFREQ),
        "Connectivity(plv+imcoh+wpli)":
            nx.connectivity_fold(SFREQ, bands, ["plv", "imcoh", "wpli"], None,
                                 "logreg"),
        "EEGNet-style CNN":
            deep_fold(lambda: FusedTemporalCNNClassifier(
                eeg_decimate=2, epochs=80, patience=12, dropout=0.4)),
        "EEG Conformer (transformer)":
            deep_fold(lambda: FusedConformerClassifier(
                eeg_decimate=2, epochs=100, patience=15, dropout=0.4)),
    }
    return eeg, run(eeg, fn0, cfg, models)


def bench_ds004022(cfg):
    nx.CLASSES = ["reach", "grasp", "lift", "twist"]
    eeg, fnirs_X = nx.load_aligned(cfg)
    sf = eeg.sfreq
    bands = [(8, 13), (13, 30)]
    print(f"\n=== ds004022 4-class same-limb (fused EEG+fNIRS, chance 0.25) | "
          f"{eeg.n_trials} trials, {eeg.n_channels}ch x {eeg.n_times} @ {sf:g}Hz, "
          f"{fnirs_X.shape[1]} fNIRS feat ===")
    models = {
        "Riemannian tangent + logreg":
            nx.tangent_fold(reg=1e-3, clf="logreg", sfreq=sf),
        "Connectivity -> PLS(8) -> LDA":
            nx.conn_pls_fold(sf, bands, ["plv", "imcoh", "wpli"], None, 8,
                             "lda", use_fnirs=True),
        "EEGNet-style CNN (fused)":
            deep_fold(lambda: FusedTemporalCNNClassifier(
                eeg_decimate=5, epochs=60, patience=10, dropout=0.4)),
        "EEG Conformer (fused)":
            deep_fold(lambda: FusedConformerClassifier(
                eeg_decimate=8, epochs=80, patience=12, dropout=0.5)),
    }
    return eeg, run(eeg, fnirs_X, cfg, models)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["eegmmidb", "ds004022"],
                    required=True)
    ap.add_argument("--subjects", type=int, default=5,
                    help="eegmmidb: number of subjects (sub-001..)")
    args = ap.parse_args()
    cfg = load_config(None)
    if args.dataset == "eegmmidb":
        _, rows = bench_eegmmidb(cfg, n_subjects=args.subjects)
        chance = 0.5
    else:
        _, rows = bench_ds004022(cfg)
        chance = 0.25
    out = ROOT / "results" / "metrics" / f"architecture_bench_{args.dataset}.json"
    out.write_text(json.dumps({"dataset": args.dataset, "chance": chance,
                               "results": rows}, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
