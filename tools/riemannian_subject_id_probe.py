#!/usr/bin/env python
"""Subject-ID probe using Riemannian tangent features + small NN.

This diagnostic asks whether the geometry that failed to separate
Reach/Grasp/Lift/Twist can identify which subject a trial came from.
It is not an N1 hand-intent model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import cfg_get, load_config, resolve_path, seed_everything
from src.containers import TrialEpochs
from src.evaluate import _aligned_eeg_fnirs, save_confusion_matrix
from src.riemannian import build_riemannian_transformer
from src.train_n1 import TorchGeluMLPClassifier


def _load_cached_epochs(cfg):
    cache_dir = resolve_path(cfg, "paths.cache_dir")
    eeg_path = cache_dir / "eeg_epochs.npz"
    fnirs_path = cache_dir / "fnirs_epochs.npz"
    if not eeg_path.exists() or not fnirs_path.exists():
        raise FileNotFoundError(
            f"missing cached epochs under {cache_dir}; run `python main.py preprocess`")
    return TrialEpochs.load(eeg_path), TrialEpochs.load(fnirs_path)


def _subject_labels(subjects):
    names = sorted(set(subjects.tolist()))
    lookup = {name: i for i, name in enumerate(names)}
    return np.asarray([lookup[s] for s in subjects], dtype=int), names


def _build_nn(cfg, seed):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", TorchGeluMLPClassifier(
            hidden_layers=tuple(cfg_get(cfg, "neural_network.hidden_layers",
                                        [64, 32])),
            dropout=float(cfg_get(cfg, "neural_network.dropout", 0.10)),
            lr=float(cfg_get(cfg, "neural_network.lr", 1e-3)),
            weight_decay=float(cfg_get(cfg, "neural_network.weight_decay",
                                       1e-3)),
            batch_size=int(cfg_get(cfg, "neural_network.batch_size", 32)),
            epochs=int(cfg_get(cfg, "neural_network.epochs", 80)),
            patience=int(cfg_get(cfg, "neural_network.patience", 10)),
            validation_fraction=float(cfg_get(
                cfg, "neural_network.validation_fraction", 0.15)),
            random_state=seed,
        )),
    ])


def _metrics(y_true, y_pred, subject_names):
    labels = list(range(len(subject_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels,
                                   average="macro", zero_division=0)),
        "chance_level": 1.0 / len(subject_names),
        "confusion_matrix": cm.tolist(),
        "n": int(len(y_true)),
    }


def _fit_predict_split(eeg_train, eeg_test, fnirs_train, fnirs_test, y_train,
                       cfg, seed, sfreq, include_fnirs):
    riem = build_riemannian_transformer(cfg, sfreq)
    riem.fit(eeg_train, y_train)
    z_train = riem.transform(eeg_train)
    z_test = riem.transform(eeg_test)
    if include_fnirs:
        z_train = np.hstack([z_train, fnirs_train])
        z_test = np.hstack([z_test, fnirs_test])
    clf = _build_nn(cfg, seed)
    clf.fit(z_train, y_train)
    return clf.predict(z_test)


def _evaluate(eeg_X, fnirs_X, y_subject, runs, cfg, seed, sfreq,
              include_fnirs):
    folds = int(cfg_get(cfg, "model.cv_folds", 5))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    y_pred = np.empty_like(y_subject)
    for train_idx, test_idx in skf.split(np.zeros(len(y_subject)), y_subject):
        y_pred[test_idx] = _fit_predict_split(
            eeg_X[train_idx], eeg_X[test_idx],
            fnirs_X[train_idx], fnirs_X[test_idx],
            y_subject[train_idx], cfg, seed, sfreq, include_fnirs)

    logo = LeaveOneGroupOut()
    y_pred_loro = np.empty_like(y_subject)
    for train_idx, test_idx in logo.split(np.zeros(len(y_subject)), y_subject,
                                          groups=runs):
        y_pred_loro[test_idx] = _fit_predict_split(
            eeg_X[train_idx], eeg_X[test_idx],
            fnirs_X[train_idx], fnirs_X[test_idx],
            y_subject[train_idx], cfg, seed, sfreq, include_fnirs)
    return y_pred, y_pred_loro


def run_probe(cfg):
    seed = int(cfg_get(cfg, "seed", 42))
    seed_everything(seed)
    eeg, fnirs = _load_cached_epochs(cfg)
    eeg, fnirs_fs = _aligned_eeg_fnirs(eeg, fnirs, cfg)
    y_subject, subject_names = _subject_labels(eeg.subjects)

    outputs = {
        "task": "subject_id_from_riemannian_features",
        "note": ("Diagnostic only: predicts subject identity, not hand intent. "
                 "Riemannian transformer and scalers are fit inside each fold."),
        "subjects": subject_names,
        "n_trials": int(eeg.n_trials),
        "chance_level": 1.0 / len(subject_names),
        "representations": {},
    }

    figures_dir = resolve_path(cfg, "paths.figures_dir")
    figures_dir.mkdir(parents=True, exist_ok=True)
    for name, include_fnirs in (
            ("eeg_tangent", False),
            ("eeg_tangent_plus_fnirs", True)):
        pred_subject, pred_loro = _evaluate(
            eeg.X, fnirs_fs.X, y_subject, eeg.runs, cfg, seed,
            eeg.sfreq, include_fnirs)
        subject_cv = _metrics(y_subject, pred_subject, subject_names)
        loro = _metrics(y_subject, pred_loro, subject_names)
        outputs["representations"][name] = {
            "subject_stratified_cv": subject_cv,
            "leave_one_run_out": loro,
        }
        save_confusion_matrix(
            subject_cv["confusion_matrix"], subject_names,
            figures_dir / f"subject_id_{name}_subject_cv.png",
            f"Subject ID - {name} - stratified CV")
        save_confusion_matrix(
            loro["confusion_matrix"], subject_names,
            figures_dir / f"subject_id_{name}_loro.png",
            f"Subject ID - {name} - leave-one-run-out")
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    report = run_probe(cfg)
    out = (Path(args.out) if args.out else
           resolve_path(cfg, "paths.metrics_dir") /
           "subject_id_riemannian_nn_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Subject-ID chance: {report['chance_level']:.3f}")
    for name, rep in report["representations"].items():
        cv = rep["subject_stratified_cv"]
        loro = rep["leave_one_run_out"]
        print(f"\n{name}")
        print(f"  stratified CV: acc={cv['accuracy']:.3f} "
              f"bal_acc={cv['balanced_accuracy']:.3f} macro_f1={cv['macro_f1']:.3f}")
        print(f"  leave-one-run-out: acc={loro['accuracy']:.3f} "
              f"bal_acc={loro['balanced_accuracy']:.3f} "
              f"macro_f1={loro['macro_f1']:.3f}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
