"""Phase 1d-A: audit FingerMovements data and validation integrity.

This script reads only the official training split. It checks the processed
schema against the canonical TRAIN.ts source, searches for duplicate trials,
audits repeated stratified folds, runs a shuffled-label negative control, and
checks that a small balanced subset can be fitted nearly perfectly.

The official test split is locked and is never opened by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.finger_movements.feature_linear.model import (  # noqa: E402
    CHANNELS,
    TIMEPOINTS,
    fit_preprocessing,
    transform,
)


CASES = 316
CLASS_COUNTS = {0: 159, 1: 157}
SEEDS = (42, 43, 44)
FOLDS = 5
EXPECTED_KEYS = {"x", "y", "source_index", "channel_names"}
EXPECTED_CHANNEL_NAMES = np.asarray(
    [
        "F3", "F1", "Fz", "F2", "F4", "FC5", "FC3", "FC1", "FCz",
        "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4",
        "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
        "O1", "O2",
    ]
)
LABEL_TO_ID = {"left": 0, "right": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/processed/finger_movements/train.npz",
    )
    parser.add_argument(
        "--raw-train",
        type=Path,
        default=ROOT / "data/raw/FingerMovements/FingerMovements_TRAIN.ts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/finger_movements/phase1d_data_sanity_checks",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument(
        "--permutations",
        type=int,
        default=10,
        help="Shuffled-label OOF repetitions per seed.",
    )
    parser.add_argument("--overfit-cases-per-class", type=int, default=16)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.data.name.lower() == "test.npz":
        parser.error("Phase 1d refuses to load test.npz")
    if args.folds < 2 or args.folds > min(CLASS_COUNTS.values()):
        parser.error("--folds is outside the valid range")
    if args.permutations < 1:
        parser.error("--permutations must be positive")
    if not 2 <= args.overfit_cases_per_class <= min(CLASS_COUNTS.values()):
        parser.error("--overfit-cases-per-class is outside the valid range")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds contains duplicates")
    return args


def load_processed_training(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Processed training data not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        if keys != EXPECTED_KEYS:
            raise ValueError(f"Unexpected NPZ keys: {sorted(keys)}")
        raw_x = data["x"]
        raw_y = data["y"]
        raw_source_index = data["source_index"]
        raw_channel_names = data["channel_names"]
        dtypes = {
            "x": str(raw_x.dtype),
            "y": str(raw_y.dtype),
            "source_index": str(raw_source_index.dtype),
            "channel_names": str(raw_channel_names.dtype),
        }
        x = raw_x.astype(np.float32, copy=True)
        y = raw_y.astype(np.int64, copy=True)
        source_index = raw_source_index.astype(np.int64, copy=True)
        channel_names = raw_channel_names.astype(str, copy=True)

    if x.shape != (CASES, CHANNELS, TIMEPOINTS):
        raise ValueError(f"Unexpected x shape: {x.shape}")
    if y.shape != (CASES,):
        raise ValueError(f"Unexpected y shape: {y.shape}")
    if source_index.shape != (CASES,):
        raise ValueError(f"Unexpected source_index shape: {source_index.shape}")
    if channel_names.shape != (CHANNELS,):
        raise ValueError(f"Unexpected channel_names shape: {channel_names.shape}")
    if dtypes["x"] != "float32" or dtypes["y"] != "uint8":
        raise ValueError(f"Unexpected core dtypes: {dtypes}")
    if dtypes["source_index"] != "int32":
        raise ValueError(f"Unexpected source_index dtype: {dtypes['source_index']}")
    if not np.isfinite(x).all():
        raise ValueError("Processed EEG contains NaN or infinity")
    if not np.array_equal(source_index, np.arange(CASES)):
        raise ValueError("source_index is not the canonical TRAIN.ts row order")
    if not np.array_equal(channel_names, EXPECTED_CHANNEL_NAMES):
        raise ValueError("Processed EEG channel names/order are incorrect")
    values, counts = np.unique(y, return_counts=True)
    observed = dict(zip(values.tolist(), counts.tolist(), strict=True))
    if observed != CLASS_COUNTS:
        raise ValueError(f"Unexpected labels or class counts: {observed}")
    return x, y, source_index, channel_names, dtypes


def parse_raw_training(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Independently parse the canonical TRAIN.ts payload for comparison."""
    if not path.is_file():
        raise FileNotFoundError(f"Canonical TRAIN.ts not found: {path}")
    samples: list[list[list[float]]] = []
    labels: list[int] = []
    metadata: dict[str, str] = {}
    in_data = False
    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if not in_data:
                if not line.startswith("@"):
                    raise ValueError(
                        f"{path}:{line_number}: content found before @data"
                    )
                key, _, value = line[1:].partition(" ")
                metadata[key.lower()] = value.strip()
                if key.lower() == "data":
                    in_data = True
                continue
            fields = line.split(":")
            if len(fields) != CHANNELS + 1:
                raise ValueError(
                    f"{path}:{line_number}: expected {CHANNELS + 1} fields, "
                    f"found {len(fields)}"
                )
            label = fields[-1].strip().lower()
            if label not in LABEL_TO_ID:
                raise ValueError(f"{path}:{line_number}: invalid label {label!r}")
            trial = [[float(value) for value in field.split(",")] for field in fields[:-1]]
            if any(len(channel) != TIMEPOINTS for channel in trial):
                raise ValueError(f"{path}:{line_number}: invalid series length")
            samples.append(trial)
            labels.append(LABEL_TO_ID[label])
    expected_metadata = {
        "timestamps": "false",
        "missing": "false",
        "univariate": "false",
        "dimensions": str(CHANNELS),
        "equallength": "true",
        "serieslength": str(TIMEPOINTS),
        "classlabel": "true left right",
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key, "").lower() != expected:
            raise ValueError(
                f"TRAIN.ts @{key}={metadata.get(key)!r}; expected {expected!r}"
            )
    x = np.asarray(samples, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if x.shape != (CASES, CHANNELS, TIMEPOINTS) or y.shape != (CASES,):
        raise ValueError(f"Unexpected TRAIN.ts shapes: x={x.shape}, y={y.shape}")
    return x, y


def stratified_folds(
    y: np.ndarray, fold_count: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Reproduce the fold construction used in Phase 1b/1c."""
    rng = np.random.default_rng(seed)
    pieces: dict[int, list[np.ndarray]] = {}
    for label in sorted(np.unique(y).tolist()):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        pieces[label] = list(np.array_split(indices, fold_count))
    all_indices = np.arange(len(y))
    output = []
    for fold in range(fold_count):
        validation = np.concatenate([pieces[label][fold] for label in pieces])
        training = np.setdiff1d(all_indices, validation, assume_unique=True)
        fold_rng = np.random.default_rng(seed * 10_000 + fold)
        fold_rng.shuffle(training)
        fold_rng.shuffle(validation)
        output.append((training, validation))
    return output


def sample_key(sample: np.ndarray) -> bytes:
    return np.ascontiguousarray(sample, dtype=np.float32).tobytes()


def duplicate_groups(x: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
    by_key: defaultdict[bytes, list[int]] = defaultdict(list)
    for index, sample in enumerate(x):
        by_key[sample_key(sample)].append(index)
    rows = []
    for indices in by_key.values():
        if len(indices) > 1:
            rows.append(
                {
                    "indices": indices,
                    "labels": y[indices].tolist(),
                    "conflicting_labels": len(np.unique(y[indices])) > 1,
                }
            )
    return rows


def most_similar_pairs(x: np.ndarray, y: np.ndarray, limit: int = 20) -> list[dict[str, Any]]:
    flattened = x.reshape(len(x), -1).astype(np.float64)
    flattened -= flattened.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(flattened, axis=1, keepdims=True)
    normalized = flattened / np.maximum(norms, 1e-12)
    correlation = normalized @ normalized.T
    upper_i, upper_j = np.triu_indices(len(x), k=1)
    values = correlation[upper_i, upper_j]
    order = np.argsort(values)[::-1][:limit]
    return [
        {
            "source_index_a": int(upper_i[position]),
            "source_index_b": int(upper_j[position]),
            "label_a": int(y[upper_i[position]]),
            "label_b": int(y[upper_j[position]]),
            "correlation": float(values[position]),
        }
        for position in order
    ]


def audit_folds(
    x: np.ndarray, y: np.ndarray, seeds: list[int], fold_count: int
) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        validation_visits = np.zeros(len(y), dtype=np.int64)
        for fold, (training, validation) in enumerate(
            stratified_folds(y, fold_count, seed), start=1
        ):
            overlap = np.intersect1d(training, validation)
            validation_visits[validation] += 1
            training_keys = {sample_key(sample) for sample in x[training]}
            validation_keys = {sample_key(sample) for sample in x[validation]}
            exact_signal_overlap = len(training_keys & validation_keys)
            if len(overlap) or exact_signal_overlap:
                raise RuntimeError(
                    f"Leakage detected in seed={seed}, fold={fold}: "
                    f"index_overlap={len(overlap)}, signal_overlap={exact_signal_overlap}"
                )
            preprocessing = fit_preprocessing(x[training])
            transformed_training = transform(x[training], preprocessing)
            transformed_validation = transform(x[validation], preprocessing)
            if not np.isfinite(transformed_training).all() or not np.isfinite(
                transformed_validation
            ).all():
                raise RuntimeError("Fold preprocessing produced non-finite values")
            train_counts = np.bincount(y[training], minlength=2)
            validation_counts = np.bincount(y[validation], minlength=2)
            rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "train_cases": len(training),
                    "validation_cases": len(validation),
                    "train_left": int(train_counts[0]),
                    "train_right": int(train_counts[1]),
                    "validation_left": int(validation_counts[0]),
                    "validation_right": int(validation_counts[1]),
                    "index_overlap": 0,
                    "exact_signal_overlap": 0,
                    "preprocessing_fit_cases": len(training),
                }
            )
        if not np.array_equal(validation_visits, np.ones(len(y), dtype=np.int64)):
            raise RuntimeError(
                f"Seed {seed}: each case must appear in validation exactly once"
            )
    return rows


def fit_logistic(
    training_x: np.ndarray,
    training_y: np.ndarray,
    validation_x: np.ndarray,
) -> np.ndarray:
    preprocessing = fit_preprocessing(training_x)
    train_features = transform(training_x, preprocessing)
    validation_features = transform(validation_x, preprocessing)
    classifier = LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=5_000,
    )
    classifier.fit(train_features, training_y)
    return classifier.predict(validation_features).astype(np.int64)


def oof_logistic_score(
    x: np.ndarray, y: np.ndarray, seed: int, fold_count: int
) -> float:
    predictions = np.full(len(y), -1, dtype=np.int64)
    for training, validation in stratified_folds(y, fold_count, seed):
        predictions[validation] = fit_logistic(
            x[training], y[training], x[validation]
        )
    if np.any(predictions < 0):
        raise RuntimeError("Incomplete OOF predictions")
    return float(balanced_accuracy_score(y, predictions))


def shuffled_label_control(
    x: np.ndarray,
    y: np.ndarray,
    seeds: list[int],
    fold_count: int,
    permutations: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    observed_scores = []
    null_scores = []
    for seed in seeds:
        observed = oof_logistic_score(x, y, seed, fold_count)
        observed_scores.append(observed)
        rows.append(
            {
                "kind": "true_labels",
                "seed": seed,
                "permutation": 0,
                "balanced_accuracy": observed,
            }
        )
        print(f"true labels seed={seed}: OOF balanced accuracy={observed:.4f}")
        for permutation in range(1, permutations + 1):
            shuffled = y.copy()
            rng = np.random.default_rng(seed * 1_000_000 + permutation)
            rng.shuffle(shuffled)
            score = oof_logistic_score(x, shuffled, seed, fold_count)
            null_scores.append(score)
            rows.append(
                {
                    "kind": "shuffled_labels",
                    "seed": seed,
                    "permutation": permutation,
                    "balanced_accuracy": score,
                }
            )
        seed_null = [
            row["balanced_accuracy"]
            for row in rows
            if row["kind"] == "shuffled_labels" and row["seed"] == seed
        ]
        print(
            f"shuffled seed={seed}: mean={np.mean(seed_null):.4f} "
            f"range=[{np.min(seed_null):.4f}, {np.max(seed_null):.4f}]"
        )
    observed_mean = float(np.mean(observed_scores))
    null = np.asarray(null_scores, dtype=np.float64)
    empirical_p = float((1 + np.sum(null >= observed_mean)) / (1 + len(null)))
    summary = {
        "observed_balanced_accuracy_mean": observed_mean,
        "observed_balanced_accuracy_by_seed": observed_scores,
        "shuffled_balanced_accuracy_mean": float(null.mean()),
        "shuffled_balanced_accuracy_std": float(null.std(ddof=1)),
        "shuffled_balanced_accuracy_min": float(null.min()),
        "shuffled_balanced_accuracy_max": float(null.max()),
        "empirical_one_sided_p": empirical_p,
        "interpretation": (
            "PASS: labels contain signal above the shuffled-label control"
            if empirical_p <= 0.05 and observed_mean > float(null.mean())
            else "WARNING: true-label performance is not clearly above the null control"
        ),
    }
    return rows, summary


def small_subset_fit_check(
    x: np.ndarray, y: np.ndarray, cases_per_class: int
) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    selected_parts = []
    for label in (0, 1):
        candidates = np.flatnonzero(y == label)
        rng.shuffle(candidates)
        selected_parts.append(candidates[:cases_per_class])
    selected = np.concatenate(selected_parts)
    rng.shuffle(selected)
    preprocessing = fit_preprocessing(x[selected])
    features = transform(x[selected], preprocessing)
    classifier = LogisticRegression(
        C=1e6,
        solver="liblinear",
        max_iter=10_000,
    )
    classifier.fit(features, y[selected])
    prediction = classifier.predict(features)
    accuracy = float(np.mean(prediction == y[selected]))
    return {
        "cases": len(selected),
        "cases_per_class": cases_per_class,
        "source_indices": selected.tolist(),
        "training_accuracy": accuracy,
        "interpretation": (
            "PASS: the feature/classification path can fit the small subset"
            if accuracy >= 0.99
            else "WARNING: the feature/classification path cannot fit the small subset"
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def main() -> None:
    args = parse_args()
    x, y, source_index, channel_names, dtypes = load_processed_training(
        args.data.resolve()
    )
    raw_x, raw_y = parse_raw_training(args.raw_train.resolve())
    raw_signal_match = bool(np.array_equal(x, raw_x))
    raw_label_match = bool(np.array_equal(y, raw_y))
    if not raw_signal_match or not raw_label_match:
        raise RuntimeError(
            "Processed train.npz does not exactly match canonical TRAIN.ts"
        )

    duplicates = duplicate_groups(x, y)
    conflicting_duplicates = [row for row in duplicates if row["conflicting_labels"]]
    if duplicates:
        raise RuntimeError(
            f"Found {len(duplicates)} exact duplicate groups, including "
            f"{len(conflicting_duplicates)} with conflicting labels"
        )
    zero_variance_trials = int(np.sum(x.reshape(len(x), -1).std(axis=1) <= 1e-12))
    zero_variance_channels = int(
        np.sum(x.transpose(1, 0, 2).reshape(CHANNELS, -1).std(axis=1) <= 1e-12)
    )
    if zero_variance_trials or zero_variance_channels:
        raise RuntimeError(
            f"Zero variance found: trials={zero_variance_trials}, "
            f"channels={zero_variance_channels}"
        )

    near_pairs = most_similar_pairs(x, y)
    fold_rows = audit_folds(x, y, args.seeds, args.folds)

    print("=== FingerMovements Phase 1d-A data sanity checks ===")
    print(f"processed={args.data.resolve()}")
    print(f"raw source={args.raw_train.resolve()}")
    print(f"x={x.shape} {x.dtype} | labels={CLASS_COUNTS}")
    print("raw-to-processed signal and labels: EXACT MATCH")
    print("exact duplicates: 0 | fold train/validation overlap: 0")
    print("test: LOCKED AND NOT LOADED")
    print(
        "known limitation: trial-level subject/session IDs are absent, so "
        "same-session mixing across folds cannot be tested"
    )

    if args.validate_only:
        print("validation-only: structural, source, duplicate, and fold checks PASS")
        return

    control_rows, control_summary = shuffled_label_control(
        x, y, args.seeds, args.folds, args.permutations
    )
    overfit_summary = small_subset_fit_check(
        x, y, args.overfit_cases_per_class
    )
    print(
        f"small-subset fit: {overfit_summary['training_accuracy']:.4f} "
        f"on {overfit_summary['cases']} cases"
    )

    suspicious_0999 = sum(row["correlation"] >= 0.999 for row in near_pairs)
    report = {
        "phase": "1d-A",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "official TRAIN split only; official TEST not opened",
        "dataset": {
            "cases": len(y),
            "shape": list(x.shape),
            "dtypes": dtypes,
            "class_counts": CLASS_COUNTS,
            "channel_names": channel_names.tolist(),
            "source_index_min": int(source_index.min()),
            "source_index_max": int(source_index.max()),
            "signal_min": float(x.min()),
            "signal_max": float(x.max()),
            "signal_mean": float(x.mean(dtype=np.float64)),
            "signal_std": float(x.std(dtype=np.float64)),
        },
        "checks": {
            "raw_signal_exact_match": raw_signal_match,
            "raw_labels_exact_match": raw_label_match,
            "finite_values": True,
            "exact_duplicate_groups": len(duplicates),
            "conflicting_duplicate_groups": len(conflicting_duplicates),
            "zero_variance_trials": zero_variance_trials,
            "zero_variance_channels": zero_variance_channels,
            "fold_audits": len(fold_rows),
            "fold_index_or_signal_overlap": 0,
            "high_similarity_pairs_among_top20_at_least_0_999": suspicious_0999,
        },
        "shuffled_label_control": control_summary,
        "small_subset_fit": overfit_summary,
        "limitations": [
            "The archive states that one subject completed three sessions, but "
            "trial-level session IDs are absent. Trial-stratified folds may therefore "
            "mix trials from the same session.",
            "Exact agreement with TRAIN.ts verifies conversion and label transcription, "
            "but cannot independently prove that the source authors assigned every "
            "left/right semantic label correctly.",
            "The official test split remains locked, so this audit intentionally does "
            "not repeat a train/test overlap check.",
        ],
        "verdict": (
            "PASS_WITH_STRUCTURAL_LIMITATION"
            if control_summary["empirical_one_sided_p"] <= 0.05
            and overfit_summary["training_accuracy"] >= 0.99
            else "REVIEW_REQUIRED"
        ),
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "phase1d_fold_integrity.csv", fold_rows)
    write_csv(output_dir / "phase1d_near_duplicate_pairs.csv", near_pairs)
    write_csv(output_dir / "phase1d_shuffled_label_control.csv", control_rows)
    metrics_path = output_dir / "phase1d_data_sanity_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(f"verdict: {report['verdict']}")
    print(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main()
