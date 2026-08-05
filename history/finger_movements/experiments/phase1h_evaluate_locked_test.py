"""Phase 1h: one-time pure inference on the official locked test split.

This script accepts only the exact frozen Phase 1h checkpoint. It loads saved
preprocessing parameters and linear weights, applies them to the 100 official
test cases, and writes evaluation metrics and per-case predictions. It contains
no training, parameter selection, or data-derived normalization step.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import sosfilt


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.finger_movements.terminal_logistic.model import (  # noqa: E402
    CHANNELS,
    FEATURES,
    LOGISTIC_C,
    LOWPASS_HZ,
    LOWPASS_ORDER,
    SAMPLING_RATE_HZ,
    TERMINAL_MEAN_WINDOWS,
    TERMINAL_SAMPLES,
    TERMINAL_SLOPE_WINDOW,
    TIMEPOINTS,
    TerminalLogistic,
)


TEST_CASES = 100
TEST_CLASS_COUNTS = {0: 49, 1: 51}
CHECKPOINT_VERSION = "phase1h_terminal_logistic_v1"
FROZEN_CHECKPOINT_SHA256 = (
    "f8fca725c3b638219bbd734257cd958779e595add2fe1118e1e78689bc120047"
)
DEVELOPMENT_OOF_BALANCED_ACCURACY = 0.6888528355299176
WILSON_Z_95 = 1.959963984540054

EXPECTED_CHANNEL_NAMES = np.asarray(
    [
        "F3",
        "F1",
        "Fz",
        "F2",
        "F4",
        "FC5",
        "FC3",
        "FC1",
        "FCz",
        "FC2",
        "FC4",
        "FC6",
        "C5",
        "C3",
        "C1",
        "Cz",
        "C2",
        "C4",
        "C6",
        "CP5",
        "CP3",
        "CP1",
        "CPz",
        "CP2",
        "CP4",
        "CP6",
        "O1",
        "O2",
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            ROOT
            / "models/finger_movements/terminal_logistic/checkpoints"
            / "finger_movements_terminal_logistic_phase1h.npz"
        ),
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=ROOT / "data/processed/finger_movements/test.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/finger_movements/phase1h_official_test",
    )
    args = parser.parse_args()
    if "test" not in args.test_data.name.lower():
        parser.error("--test-data must identify the official test split")
    return args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_scalar(
    checkpoint: np.lib.npyio.NpzFile,
    name: str,
    expected: str | int | float,
) -> None:
    if name not in checkpoint.files:
        raise KeyError(f"Checkpoint is missing scalar: {name}")
    observed = checkpoint[name].item()
    if isinstance(expected, float):
        matches = bool(np.isclose(float(observed), expected, rtol=0.0, atol=1e-12))
    else:
        matches = observed == expected
    if not matches:
        raise ValueError(
            f"Checkpoint {name} mismatch: observed={observed!r}, expected={expected!r}"
        )


def load_frozen_checkpoint(
    path: Path,
) -> tuple[
    TerminalLogistic,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    str,
]:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen checkpoint not found: {path}")
    digest = file_sha256(path)
    if digest != FROZEN_CHECKPOINT_SHA256:
        raise ValueError(
            "Checkpoint SHA-256 does not match the frozen Phase 1h artifact: "
            f"observed={digest}, expected={FROZEN_CHECKPOINT_SHA256}"
        )

    with np.load(path, allow_pickle=False) as checkpoint:
        required = {
            "checkpoint_version",
            "training_split",
            "training_cases",
            "class_ids",
            "class_names",
            "channels",
            "timepoints",
            "sampling_rate_hz",
            "lowpass_hz",
            "lowpass_order",
            "lowpass_sos",
            "lowpass_initial",
            "terminal_samples",
            "terminal_mean_windows",
            "terminal_slope_window",
            "feature_count",
            "logistic_c",
            "channel_mean",
            "channel_std",
            "feature_mean",
            "feature_std",
            "weight",
            "bias",
        }
        missing = required - set(checkpoint.files)
        if missing:
            raise KeyError(f"Checkpoint is missing arrays: {sorted(missing)}")

        require_scalar(checkpoint, "checkpoint_version", CHECKPOINT_VERSION)
        require_scalar(checkpoint, "training_split", "official TRAIN only")
        require_scalar(checkpoint, "training_cases", 316)
        require_scalar(checkpoint, "channels", CHANNELS)
        require_scalar(checkpoint, "timepoints", TIMEPOINTS)
        require_scalar(checkpoint, "sampling_rate_hz", SAMPLING_RATE_HZ)
        require_scalar(checkpoint, "lowpass_hz", LOWPASS_HZ)
        require_scalar(checkpoint, "lowpass_order", LOWPASS_ORDER)
        require_scalar(checkpoint, "terminal_samples", TERMINAL_SAMPLES)
        require_scalar(
            checkpoint, "terminal_slope_window", TERMINAL_SLOPE_WINDOW
        )
        require_scalar(checkpoint, "feature_count", FEATURES)
        require_scalar(checkpoint, "logistic_c", LOGISTIC_C)
        if checkpoint["class_ids"].tolist() != [0, 1]:
            raise ValueError("Checkpoint class IDs must be left=0 and right=1")
        if checkpoint["class_names"].tolist() != ["left", "right"]:
            raise ValueError("Checkpoint class names must be left and right")
        if checkpoint["terminal_mean_windows"].tolist() != list(
            TERMINAL_MEAN_WINDOWS
        ):
            raise ValueError("Checkpoint terminal mean windows changed")

        preprocessing = {
            "channel_mean": checkpoint["channel_mean"].copy(),
            "channel_std": checkpoint["channel_std"].copy(),
            "feature_mean": checkpoint["feature_mean"].copy(),
            "feature_std": checkpoint["feature_std"].copy(),
        }
        filter_parameters = {
            "sos": checkpoint["lowpass_sos"].copy(),
            "initial": checkpoint["lowpass_initial"].copy(),
        }
        model = TerminalLogistic(
            weight=checkpoint["weight"].copy(),
            bias=float(checkpoint["bias"].item()),
        )

    expected_shapes = {
        "channel_mean": (1, CHANNELS, 1),
        "channel_std": (1, CHANNELS, 1),
        "feature_mean": (1, FEATURES),
        "feature_std": (1, FEATURES),
    }
    for name, expected_shape in expected_shapes.items():
        array = preprocessing[name]
        if array.shape != expected_shape:
            raise ValueError(
                f"Checkpoint {name} shape mismatch: "
                f"observed={array.shape}, expected={expected_shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"Checkpoint {name} contains non-finite values")
    if np.any(preprocessing["channel_std"] <= 0.0) or np.any(
        preprocessing["feature_std"] <= 0.0
    ):
        raise ValueError("Checkpoint normalization scale must be positive")

    sos = filter_parameters["sos"]
    initial = filter_parameters["initial"]
    if sos.ndim != 2 or sos.shape[1] != 6:
        raise ValueError(f"Checkpoint SOS shape is invalid: {sos.shape}")
    if initial.shape != (sos.shape[0], 2):
        raise ValueError(
            f"Checkpoint filter-initial shape is invalid: {initial.shape}"
        )
    if not np.isfinite(sos).all() or not np.isfinite(initial).all():
        raise ValueError("Checkpoint filter parameters contain non-finite values")
    return model, preprocessing, filter_parameters, digest


def transform_from_checkpoint(
    x: np.ndarray,
    preprocessing: dict[str, np.ndarray],
    filter_parameters: dict[str, np.ndarray],
) -> np.ndarray:
    """Apply only parameters serialized in the frozen checkpoint."""
    if x.ndim != 3 or x.shape[1:] != (CHANNELS, TIMEPOINTS):
        raise ValueError(
            f"Expected input shape (cases, {CHANNELS}, {TIMEPOINTS}), got {x.shape}"
        )
    if not np.isfinite(x).all():
        raise ValueError("Inference input contains non-finite values")

    normalized = (
        (x - preprocessing["channel_mean"]) / preprocessing["channel_std"]
    ).astype(np.float32)
    sos = filter_parameters["sos"]
    initial = filter_parameters["initial"]
    filter_state = initial[:, None, None, :] * normalized[
        None, :, :, 0, None
    ]
    filtered, _ = sosfilt(
        sos,
        normalized.astype(np.float64),
        axis=-1,
        zi=filter_state,
    )
    filtered = filtered.astype(np.float32)

    terminal = filtered[..., -TERMINAL_SAMPLES:].reshape(len(filtered), -1)
    means = [
        filtered[..., -window:].mean(axis=-1)
        for window in TERMINAL_MEAN_WINDOWS
    ]
    time = np.arange(TERMINAL_SLOPE_WINDOW, dtype=np.float64)
    centered_time = time - time.mean()
    slope = np.tensordot(
        filtered[..., -TERMINAL_SLOPE_WINDOW:],
        centered_time,
        axes=([-1], [0]),
    ) / np.square(centered_time).sum()
    raw_features = np.concatenate([terminal, *means, slope], axis=1).astype(
        np.float32
    )
    if raw_features.shape != (len(x), FEATURES):
        raise RuntimeError(f"Unexpected checkpoint feature shape: {raw_features.shape}")
    return (
        (raw_features - preprocessing["feature_mean"])
        / preprocessing["feature_std"]
    ).astype(np.float32)


def load_official_test(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Official test data not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {"x", "y", "source_index", "channel_names"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"Official test data is missing arrays: {sorted(missing)}")
        x = data["x"].astype(np.float32, copy=True)
        y = data["y"].astype(np.int64, copy=True)
        source_index = data["source_index"].astype(np.int64, copy=True)
        channel_names = data["channel_names"].copy()

    if x.shape != (TEST_CASES, CHANNELS, TIMEPOINTS):
        raise ValueError(f"Unexpected official test input shape: {x.shape}")
    if y.shape != (TEST_CASES,) or source_index.shape != (TEST_CASES,):
        raise ValueError(
            f"Unexpected test metadata shapes: y={y.shape}, "
            f"source_index={source_index.shape}"
        )
    if not np.isfinite(x).all():
        raise ValueError("Official test input contains non-finite values")
    values, counts = np.unique(y, return_counts=True)
    observed = dict(zip(values.tolist(), counts.tolist(), strict=True))
    if observed != TEST_CLASS_COUNTS:
        raise ValueError(f"Unexpected official test class counts: {observed}")
    if not np.array_equal(source_index, np.arange(TEST_CASES)):
        raise ValueError("Test source_index must preserve canonical TEST.ts order")
    if not np.array_equal(channel_names, EXPECTED_CHANNEL_NAMES):
        raise ValueError("Official test channel order does not match the frozen model")
    return x, y, source_index, channel_names


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("Invalid binomial counts for Wilson interval")
    proportion = successes / total
    z_squared = WILSON_Z_95**2
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    radius = (
        WILSON_Z_95
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z_squared / (4.0 * total**2)
        )
        / denominator
    )
    return float(center - radius), float(center + radius)


def classification_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    probability_right: np.ndarray,
) -> dict[str, Any]:
    confusion = np.zeros((2, 2), dtype=np.int64)
    for truth, guess in zip(actual, predicted, strict=True):
        confusion[int(truth), int(guess)] += 1

    recalls = []
    f1_scores = []
    recall_intervals = []
    for label in (0, 1):
        true_positive = int(confusion[label, label])
        actual_count = int(confusion[label].sum())
        predicted_count = int(confusion[:, label].sum())
        recall = true_positive / actual_count
        precision = true_positive / max(predicted_count, 1)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        recalls.append(recall)
        f1_scores.append(f1)
        recall_intervals.append(wilson_interval(true_positive, actual_count))

    correct = int(np.sum(actual == predicted))
    accuracy_interval = wilson_interval(correct, len(actual))
    clipped = np.clip(probability_right, 1e-15, 1.0 - 1e-15)
    log_loss = -np.mean(
        actual * np.log(clipped) + (1 - actual) * np.log(1.0 - clipped)
    )
    return {
        "accuracy": float(correct / len(actual)),
        "accuracy_wilson_95": list(accuracy_interval),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
        "left_recall": float(recalls[0]),
        "left_recall_wilson_95": list(recall_intervals[0]),
        "right_recall": float(recalls[1]),
        "right_recall_wilson_95": list(recall_intervals[1]),
        "mean_log_loss": float(log_loss),
        "confusion_matrix": confusion.tolist(),
        "correct_cases": correct,
        "total_cases": len(actual),
    }


def write_predictions(
    path: Path,
    source_index: np.ndarray,
    y: np.ndarray,
    prediction: np.ndarray,
    score: np.ndarray,
    probability_right: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source_index",
                "true_label",
                "predicted_label",
                "correct",
                "decision_score",
                "probability_right",
            ],
        )
        writer.writeheader()
        for index in range(len(y)):
            writer.writerow(
                {
                    "source_index": int(source_index[index]),
                    "true_label": int(y[index]),
                    "predicted_label": int(prediction[index]),
                    "correct": bool(prediction[index] == y[index]),
                    "decision_score": float(score[index]),
                    "probability_right": float(probability_right[index]),
                }
            )


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    test_path = args.test_data.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            "Official-test output directory already exists; refusing to overwrite "
            f"the one-time evaluation: {output_dir}"
        )

    print("=== FingerMovements Phase 1h official locked-test inference ===")
    print(f"required checkpoint SHA-256={FROZEN_CHECKPOINT_SHA256}")
    print("policy=pure inference only; no fitting or preprocessing estimation")
    print("verifying frozen checkpoint before opening official TEST")

    model, preprocessing, filter_parameters, checkpoint_digest = (
        load_frozen_checkpoint(checkpoint_path)
    )
    print("checkpoint verified; opening official TEST now")
    x, y, source_index, channel_names = load_official_test(test_path)
    features = transform_from_checkpoint(x, preprocessing, filter_parameters)
    if features.shape != (TEST_CASES, FEATURES):
        raise RuntimeError(f"Unexpected transformed test shape: {features.shape}")
    score = model.decision_function(features)
    prediction = model.predict_features(features).astype(np.int64)
    probability_right = model.probability_right(features)
    metrics = classification_metrics(y, prediction, probability_right)

    created_utc = datetime.now(timezone.utc).isoformat()
    report = {
        "phase": "1h",
        "created_utc": created_utc,
        "scope": "one-time pure inference on the 100-case official TEST split",
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(ROOT)),
            "sha256": checkpoint_digest,
            "version": CHECKPOINT_VERSION,
        },
        "test_data": {
            "path": str(test_path.relative_to(ROOT)),
            "cases": TEST_CASES,
            "class_counts": {
                str(key): value for key, value in TEST_CLASS_COUNTS.items()
            },
            "input_shape": [TEST_CASES, CHANNELS, TIMEPOINTS],
            "channel_names": channel_names.tolist(),
        },
        "inference_policy": {
            "training_operations": "none",
            "normalization": "loaded unchanged from frozen checkpoint",
            "filter_coefficients": "loaded unchanged from frozen checkpoint",
            "decision_threshold": 0.0,
            "parameter_selection": "none",
        },
        "development_reference": {
            "metric": "mean OOF balanced accuracy across seeds 42/43/44",
            "value": DEVELOPMENT_OOF_BALANCED_ACCURACY,
        },
        "official_test_metrics": metrics,
        "test_minus_development_oof_balanced_accuracy": float(
            metrics["balanced_accuracy"] - DEVELOPMENT_OOF_BALANCED_ACCURACY
        ),
        "post_test_policy": (
            "Report this result as final locked-test evidence. Do not tune the "
            "frozen pipeline, threshold, or preprocessing from this test result."
        ),
    }

    temporary_dir = output_dir.with_name(output_dir.name + ".tmp")
    if temporary_dir.exists():
        raise FileExistsError(
            f"Temporary official-test output already exists: {temporary_dir}"
        )
    temporary_dir.mkdir(parents=True)
    predictions_path = temporary_dir / "phase1h_official_test_predictions.csv"
    metrics_path = temporary_dir / "phase1h_official_test_metrics.json"
    write_predictions(
        predictions_path,
        source_index,
        y,
        prediction,
        score,
        probability_right,
    )
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    temporary_dir.replace(output_dir)

    accuracy_low, accuracy_high = metrics["accuracy_wilson_95"]
    print("\n=== official TEST result ===")
    print(
        f"accuracy={metrics['accuracy']:.4f} "
        f"(95% Wilson CI {accuracy_low:.4f} to {accuracy_high:.4f})"
    )
    print(f"balanced accuracy={metrics['balanced_accuracy']:.4f}")
    print(f"macro F1={metrics['macro_f1']:.4f}")
    print(
        f"left recall={metrics['left_recall']:.4f} | "
        f"right recall={metrics['right_recall']:.4f}"
    )
    print(f"mean log loss={metrics['mean_log_loss']:.6f}")
    print(f"confusion matrix={metrics['confusion_matrix']}")
    print(
        "test minus development OOF BA="
        f"{100.0 * report['test_minus_development_oof_balanced_accuracy']:+.2f} pp"
    )
    print(f"metrics: {output_dir / metrics_path.name}")
    print(f"predictions: {output_dir / predictions_path.name}")
    print("checkpoint unchanged; no training performed")


if __name__ == "__main__":
    main()
