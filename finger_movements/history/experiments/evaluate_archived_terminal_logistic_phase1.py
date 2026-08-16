"""Re-evaluate the frozen Phase 1 terminal Logistic model on corrected TRAIN.

This is a self-contained reproduction of the archived best Phase 1 pipeline:

* fold-training-only per-channel normalization;
* second-order causal 5 Hz low-pass filtering;
* 252 terminal ABC features;
* fold-training-only feature standardization;
* L2 Logistic Regression with C=1.

Only the official 316-case TRAIN split is accepted. TEST is never loaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)


ROOT = Path(__file__).resolve().parents[2]

CASES = 316
CHANNELS = 28
TIMEPOINTS = 50
SAMPLING_RATE_HZ = 100.0
CLASS_COUNTS = {0: 159, 1: 157}
SEEDS = (42, 43, 44)
FOLDS = 5

LOWPASS_HZ = 5.0
LOWPASS_ORDER = 2
TERMINAL_SAMPLES = 5
TERMINAL_MEAN_WINDOWS = (5, 10, 20)
TERMINAL_SLOPE_WINDOW = 20
FEATURES = CHANNELS * (
    TERMINAL_SAMPLES + len(TERMINAL_MEAN_WINDOWS) + 1
)
LOGISTIC_C = 1.0

LOWPASS_SOS = butter(
    LOWPASS_ORDER,
    LOWPASS_HZ,
    btype="lowpass",
    fs=SAMPLING_RATE_HZ,
    output="sos",
)
LOWPASS_INITIAL = sosfilt_zi(LOWPASS_SOS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/processed/finger_movements/train.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "results/finger_movements/archived_terminal_logistic_official_matlab"
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run one fold without writing result files.",
    )
    args = parser.parse_args()
    if any("test" in part.lower() for part in args.data.parts):
        parser.error("This reproduction refuses to load any TEST path")
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain unique values")
    if not 2 <= args.folds <= min(CLASS_COUNTS.values()):
        parser.error("--folds is outside the valid range")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Training data not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {"x", "y", "source_index"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"Missing arrays: {sorted(missing)}")
        x = data["x"].astype(np.float32, copy=True)
        y = data["y"].astype(np.int64, copy=True)
        source_index = data["source_index"].astype(np.int64, copy=True)

    if x.shape != (CASES, CHANNELS, TIMEPOINTS):
        raise ValueError(f"Unexpected x shape: {x.shape}")
    if y.shape != (CASES,) or source_index.shape != (CASES,):
        raise ValueError("Unexpected label or source-index shape")
    if not np.isfinite(x).all():
        raise ValueError("Input contains non-finite values")
    if dict(Counter(y.tolist())) != CLASS_COUNTS:
        raise ValueError(f"Unexpected class counts: {dict(Counter(y.tolist()))}")
    if not np.array_equal(source_index, np.arange(CASES)):
        raise ValueError("source_index must preserve official TRAIN order")
    if np.all(x[:, :-1, 28:] == x[:, 1:, :22]):
        raise ValueError("Detected the retired UEA sliding-channel layout error")
    return x, y, source_index


def stratified_folds(
    y: np.ndarray, fold_count: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    pieces: dict[int, list[np.ndarray]] = {}
    for label in sorted(np.unique(y).tolist()):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        pieces[label] = list(np.array_split(indices, fold_count))

    all_indices = np.arange(len(y))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(fold_count):
        validation = np.concatenate(
            [pieces[label][fold] for label in sorted(pieces)]
        )
        training = np.setdiff1d(all_indices, validation, assume_unique=True)
        fold_rng = np.random.default_rng(seed * 10_000 + fold)
        fold_rng.shuffle(training)
        fold_rng.shuffle(validation)
        if np.intersect1d(training, validation).size:
            raise RuntimeError("Training/validation fold overlap detected")
        folds.append((training, validation))
    return folds


def causal_lowpass(normalized_x: np.ndarray) -> np.ndarray:
    initial = LOWPASS_INITIAL[:, None, None, :] * normalized_x[
        None, :, :, 0, None
    ]
    filtered, _ = sosfilt(
        LOWPASS_SOS,
        normalized_x.astype(np.float64),
        axis=-1,
        zi=initial,
    )
    return filtered.astype(np.float32)


def terminal_features(normalized_x: np.ndarray) -> np.ndarray:
    filtered = causal_lowpass(normalized_x)
    terminal = filtered[..., -TERMINAL_SAMPLES:].reshape(len(filtered), -1)
    means = [
        filtered[..., -window:].mean(axis=-1)
        for window in TERMINAL_MEAN_WINDOWS
    ]
    time = np.arange(TERMINAL_SLOPE_WINDOW, dtype=np.float64)
    centered = time - time.mean()
    slope = np.tensordot(
        filtered[..., -TERMINAL_SLOPE_WINDOW:],
        centered,
        axes=([-1], [0]),
    ) / np.square(centered).sum()
    features = np.concatenate([terminal, *means, slope], axis=1).astype(np.float32)
    if features.shape != (len(normalized_x), FEATURES):
        raise RuntimeError(f"Unexpected feature shape: {features.shape}")
    return features


def prepare_fold(
    training_x: np.ndarray,
    validation_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    channel_mean = training_x.mean(axis=(0, 2), keepdims=True, dtype=np.float64)
    channel_std = np.maximum(
        training_x.std(axis=(0, 2), keepdims=True, dtype=np.float64),
        1e-6,
    )
    normalized_training = ((training_x - channel_mean) / channel_std).astype(
        np.float32
    )
    normalized_validation = ((validation_x - channel_mean) / channel_std).astype(
        np.float32
    )
    training_features = terminal_features(normalized_training)
    validation_features = terminal_features(normalized_validation)
    feature_mean = training_features.mean(axis=0, keepdims=True, dtype=np.float64)
    feature_std = np.maximum(
        training_features.std(axis=0, keepdims=True, dtype=np.float64),
        1e-6,
    )
    return (
        ((training_features - feature_mean) / feature_std).astype(np.float64),
        ((validation_features - feature_mean) / feature_std).astype(np.float64),
    )


def metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
    probability_right: np.ndarray,
) -> dict[str, Any]:
    probabilities = np.column_stack([1.0 - probability_right, probability_right])
    return {
        "accuracy": float(accuracy_score(actual, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, prediction)),
        "macro_f1": float(f1_score(actual, prediction, average="macro")),
        "mean_log_loss": float(log_loss(actual, probabilities, labels=[0, 1])),
        "confusion_matrix": confusion_matrix(actual, prediction, labels=[0, 1]).tolist(),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def create_figure(seed_rows: list[dict[str, Any]], output_path: Path) -> None:
    seeds = [int(row["seed"]) for row in seed_rows]
    values = 100.0 * np.asarray(
        [float(row["balanced_accuracy"]) for row in seed_rows]
    )
    mean = float(values.mean())
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar([str(seed) for seed in seeds], values, color="#4c72b0")
    ax.axhline(mean, color="#c44e52", linestyle="--", label=f"mean={mean:.2f}%")
    ax.set_ylim(max(0.0, values.min() - 5.0), min(100.0, values.max() + 5.0))
    ax.set_xlabel("Cross-validation seed")
    ax.set_ylabel("OOF balanced accuracy (%)")
    ax.set_title("Archived terminal Logistic on corrected official TRAIN")
    ax.legend()
    for position, value in enumerate(values):
        ax.text(position, value + 0.25, f"{value:.2f}%", ha="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    x, y, source_index = load_training_data(data_path)

    print("=== Archived Phase 1 terminal Logistic on official MATLAB TRAIN ===")
    print(f"cases={CASES} | input={tuple(x.shape)} | features={FEATURES}")
    print(f"seeds={args.seeds} | folds={args.folds} | C={LOGISTIC_C:g}")
    print("preprocessing=fold-training-only | test=REFUSED AND NOT LOADED")

    seeds_to_run = args.seeds[:1] if args.validate_only else args.seeds
    folds_to_run = 1 if args.validate_only else args.folds
    fold_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for seed in seeds_to_run:
        oof_prediction = np.full(CASES, -1, dtype=np.int64)
        oof_probability = np.full(CASES, np.nan, dtype=np.float64)
        fold_definitions = stratified_folds(y, args.folds, seed)
        for fold_index, (training, validation) in enumerate(
            fold_definitions[:folds_to_run], start=1
        ):
            started = perf_counter()
            train_features, validation_features = prepare_fold(
                x[training], x[validation]
            )
            classifier = LogisticRegression(
                C=LOGISTIC_C,
                penalty="l2",
                solver="liblinear",
                max_iter=5_000,
            )
            classifier.fit(train_features, y[training])
            prediction = classifier.predict(validation_features).astype(np.int64)
            probability = classifier.predict_proba(validation_features)[:, 1]
            fold_metric = metrics(y[validation], prediction, probability)
            fold_rows.append(
                {
                    "seed": seed,
                    "fold": fold_index,
                    "training_cases": len(training),
                    "validation_cases": len(validation),
                    "training_accuracy": float(
                        np.mean(classifier.predict(train_features) == y[training])
                    ),
                    **fold_metric,
                    "elapsed_seconds": perf_counter() - started,
                }
            )
            oof_prediction[validation] = prediction
            oof_probability[validation] = probability
            for position, case in enumerate(validation):
                prediction_rows.append(
                    {
                        "seed": seed,
                        "fold": fold_index,
                        "source_index": int(source_index[case]),
                        "true_label": int(y[case]),
                        "predicted_label": int(prediction[position]),
                        "probability_right": float(probability[position]),
                    }
                )
            print(
                f"seed={seed} fold={fold_index}/{args.folds} | "
                f"train={fold_rows[-1]['training_accuracy']:.4f} | "
                f"validation BA={fold_metric['balanced_accuracy']:.4f}"
            )

        if args.validate_only:
            print("validation-only: PASS; no result files written")
            return
        if np.any(oof_prediction < 0) or not np.isfinite(oof_probability).all():
            raise RuntimeError(f"Incomplete OOF predictions for seed {seed}")
        seed_metric = metrics(y, oof_prediction, oof_probability)
        seed_rows.append({"seed": seed, "evaluated_cases": CASES, **seed_metric})
        print(
            f"seed={seed} OOF | accuracy={seed_metric['accuracy']:.4f} | "
            f"BA={seed_metric['balanced_accuracy']:.4f} | "
            f"macro-F1={seed_metric['macro_f1']:.4f}"
        )

    mean_ba = float(np.mean([row["balanced_accuracy"] for row in seed_rows]))
    seed_sd = float(np.std([row["balanced_accuracy"] for row in seed_rows]))
    worst_seed = float(min(row["balanced_accuracy"] for row in seed_rows))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        output_dir / "fold_results.csv",
        fold_rows,
        [
            "seed",
            "fold",
            "training_cases",
            "validation_cases",
            "training_accuracy",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "mean_log_loss",
            "confusion_matrix",
            "elapsed_seconds",
        ],
    )
    write_csv(
        output_dir / "seed_results.csv",
        seed_rows,
        [
            "seed",
            "evaluated_cases",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "mean_log_loss",
            "confusion_matrix",
        ],
    )
    write_csv(
        output_dir / "predictions.csv",
        prediction_rows,
        [
            "seed",
            "fold",
            "source_index",
            "true_label",
            "predicted_label",
            "probability_right",
        ],
    )
    payload = {
        "experiment": "archived_terminal_logistic_official_matlab",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "frozen Phase 1 model on corrected official TRAIN only",
        "data": {
            "path": str(data_path),
            "sha256": sha256(data_path),
            "shape": list(x.shape),
            "test_policy": "official TEST refused and not loaded",
        },
        "pipeline": {
            "channel_normalization": "fold-training-only",
            "lowpass_hz": LOWPASS_HZ,
            "lowpass_order": LOWPASS_ORDER,
            "filter": "causal SOS initialized from each trial's first sample",
            "terminal_samples": TERMINAL_SAMPLES,
            "terminal_mean_windows_samples": list(TERMINAL_MEAN_WINDOWS),
            "terminal_slope_window_samples": TERMINAL_SLOPE_WINDOW,
            "feature_count": FEATURES,
            "feature_standardization": "fold-training-only",
            "classifier": "L2 Logistic Regression",
            "c": LOGISTIC_C,
            "solver": "liblinear",
        },
        "validation": {
            "seeds": [int(seed) for seed in args.seeds],
            "folds": int(args.folds),
            "split": "repeated stratified cross-validation",
        },
        "summary": {
            "mean_oof_balanced_accuracy": mean_ba,
            "balanced_accuracy_seed_sd": seed_sd,
            "worst_seed_balanced_accuracy": worst_seed,
        },
        "seed_results": seed_rows,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    create_figure(seed_rows, output_dir / "balanced_accuracy.png")

    print("\n=== Summary ===")
    print(
        f"mean OOF BA={mean_ba:.4f} | seed SD={seed_sd:.4f} | "
        f"worst seed={worst_seed:.4f}"
    )
    print(f"results={output_dir}")


if __name__ == "__main__":
    main()
