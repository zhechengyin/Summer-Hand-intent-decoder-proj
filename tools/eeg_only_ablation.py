#!/usr/bin/env python
"""Diagnostic EEG-only ablations for recent fused model paths.

This is intentionally not an official N1 evaluation path. The project policy is
to keep deployed/reported N1 models fused EEG+fNIRS, but this script answers the
diagnostic question: "Do fNIRS features help or hurt these EEG feature families?"

To make the comparison fair, EEG-only runs are restricted to the same trials that
can be aligned with fNIRS, but the fNIRS feature columns are not used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import cfg_get, load_config, resolve_path, seed_everything
from src.containers import TrialEpochs
from src.evaluate import (cv_leave_one_run_out, cv_subject_specific,
                          evaluate_loro, evaluate_subject_specific)
from src.fusion import align_trials, build_feature_set, metadata_feature_set
from src.load_bids import discover_dataset
from src.preprocess_eeg import build_eeg_epochs
from src.riemannian import build_riemannian_pipeline
from src.symbolic_erd import build_symbolic_erd_feature_set


def _load_cached_epochs(cfg):
    cache_dir = resolve_path(cfg, "paths.cache_dir")
    eeg_path = cache_dir / "eeg_epochs.npz"
    fnirs_path = cache_dir / "fnirs_epochs.npz"
    if not eeg_path.exists() or not fnirs_path.exists():
        raise FileNotFoundError(
            f"missing cached epochs under {cache_dir}; run preprocess first")
    return TrialEpochs.load(eeg_path), TrialEpochs.load(fnirs_path)


def _align_eeg_to_fnirs(eeg: TrialEpochs, fnirs: TrialEpochs) -> TrialEpochs:
    ia, _ = align_trials(metadata_feature_set(eeg), metadata_feature_set(fnirs))
    if ia.size == 0:
        raise ValueError("no aligned EEG/fNIRS trials available")
    return eeg.select(ia)


def _summarize(summary):
    return {
        "mean_accuracy": summary["mean_accuracy"],
        "mean_balanced_accuracy": summary["mean_balanced_accuracy"],
        "pooled_accuracy": summary["pooled"]["accuracy"],
        "pooled_balanced_accuracy": summary["pooled"]["balanced_accuracy"],
        "pooled_macro_f1": summary["pooled"]["macro_f1"],
        "n": summary["pooled"]["n"],
    }


def _eval_feature_set(fs, cfg, classifier: str):
    subject, _, _ = evaluate_subject_specific(fs, cfg, classifier)
    loro, _, _ = evaluate_loro(fs, cfg, classifier)
    return {
        "subject_specific": subject,
        "leave_one_run_out": loro,
    }


def _eval_riemannian(eeg, cfg, classifier: str):
    factory = lambda: build_riemannian_pipeline(  # noqa: E731
        cfg, eeg.sfreq, classifier)
    subject, _, _ = cv_subject_specific(
        eeg.X, eeg.y, eeg.subjects, eeg.classes, cfg, factory,
        "eeg_only_riemannian", classifier)
    loro, _, _ = cv_leave_one_run_out(
        eeg.X, eeg.y, eeg.subjects, eeg.runs, eeg.classes, cfg, factory,
        "eeg_only_riemannian", classifier)
    return {
        "subject_specific": subject,
        "leave_one_run_out": loro,
    }


def _build_wide_symbolic_eeg(cfg, subjects=None):
    root = resolve_path(cfg, "paths.bids_root")
    index = discover_dataset(root)
    tmin = float(cfg_get(cfg, "symbolic_erd.eeg_tmin", -2.0))
    tmax = float(cfg_get(cfg, "symbolic_erd.eeg_tmax", 5.0))
    return build_eeg_epochs(cfg, index, subjects, cache=False,
                            tmin=tmin, tmax=tmax)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg_get(cfg, "seed", 42)))
    eeg, fnirs = _load_cached_epochs(cfg)
    if args.subjects:
        mask_eeg = [str(s) in set(args.subjects) for s in eeg.subjects]
        mask_fnirs = [str(s) in set(args.subjects) for s in fnirs.subjects]
        eeg = eeg.select(mask_eeg)
        fnirs = fnirs.select(mask_fnirs)
    eeg_aligned = _align_eeg_to_fnirs(eeg, fnirs)

    report = {
        "note": ("Diagnostic ablation only. EEG-only models use the same trials "
                 "that align with fNIRS, but fNIRS features are removed."),
        "chance_level": 1.0 / len(eeg.classes),
        "classes": eeg.classes,
        "models": {},
    }

    band_fs = build_feature_set(eeg_aligned, cfg)
    band_fs.modality = "eeg_only_bandpower"
    print("\n=== EEG-only diagnostic: bandpower + LDA ===")
    band = _eval_feature_set(band_fs, cfg, "lda")
    report["models"]["eeg_only_bandpower_lda"] = band
    print("  subject-specific: "
          f"{band['subject_specific']['mean_accuracy']:.4f}")
    print("  leave-one-run-out: "
          f"{band['leave_one_run_out']['mean_accuracy']:.4f}")

    print("\n=== EEG-only diagnostic: Riemannian + Logistic Regression ===")
    riem = _eval_riemannian(eeg_aligned, cfg, "logreg")
    report["models"]["eeg_only_riemannian_logreg"] = riem
    print("  subject-specific: "
          f"{riem['subject_specific']['mean_accuracy']:.4f}")
    print("  leave-one-run-out: "
          f"{riem['leave_one_run_out']['mean_accuracy']:.4f}")

    print("\n=== EEG-only diagnostic: symbolic ERD/ERS + LDA ===")
    wide = _build_wide_symbolic_eeg(cfg, args.subjects)
    wide = _align_eeg_to_fnirs(wide, fnirs)
    sym_fs = build_symbolic_erd_feature_set(wide, cfg)
    sym = _eval_feature_set(sym_fs, cfg, "lda")
    report["models"]["eeg_only_symbolic_erd_lda"] = sym
    print("  subject-specific: "
          f"{sym['subject_specific']['mean_accuracy']:.4f}")
    print("  leave-one-run-out: "
          f"{sym['leave_one_run_out']['mean_accuracy']:.4f}")

    compact = {
        name: {
            "subject_specific": _summarize(vals["subject_specific"]),
            "leave_one_run_out": _summarize(vals["leave_one_run_out"]),
        }
        for name, vals in report["models"].items()
    }
    report["summary"] = compact

    out = (Path(args.out) if args.out else
           resolve_path(cfg, "paths.metrics_dir") / "eeg_only_ablation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
