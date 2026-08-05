"""Phase 1h: fit the frozen model once on all official training cases.

This is final checkpoint training, not another model-selection experiment.
The frozen ABC terminal representation and L2 Logistic Regression with C=1
are fitted on all 316 official training cases. Liblinear performs convex
optimization, so its final converged solution is saved instead of selecting an
epoch checkpoint. The official test split is never loaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.finger_movements.terminal_logistic.model import (  # noqa: E402
    CHANNELS,
    FEATURES,
    LOGISTIC_C,
    LOWPASS_HZ,
    LOWPASS_INITIAL,
    LOWPASS_ORDER,
    LOWPASS_SOS,
    SAMPLING_RATE_HZ,
    TERMINAL_MEAN_WINDOWS,
    TERMINAL_SAMPLES,
    TERMINAL_SLOPE_WINDOW,
    TIMEPOINTS,
    TerminalLogistic,
    fit_preprocessing,
    transform,
)


TRAIN_CASES = 316
CLASS_COUNTS = {0: 159, 1: 157}
SOLVER = "liblinear"
SOLVER_MAX_ITER = 100_000
SOLVER_TOL = 1e-10
SOLVER_RANDOM_STATE = 42
CHECKPOINT_VERSION = "phase1h_terminal_logistic_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-data",
        type=Path,
        default=ROOT / "data/processed/finger_movements/train.npz",
    )
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
        "--result-dir",
        type=Path,
        default=ROOT / "results/finger_movements/phase1h_final_training",
    )
    parser.add_argument(
        "--quiet-solver",
        action="store_true",
        help="Hide liblinear's optimization iterations.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate train data and preprocessing without fitting or writing files.",
    )
    args = parser.parse_args()
    if "test" in args.train_data.name.lower():
        parser.error("Phase 1h refuses to load any file identified as a test split")
    if args.checkpoint.suffix.lower() != ".npz":
        parser.error("--checkpoint must use the .npz extension")
    return args


def load_training_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Training data not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {"x", "y", "source_index"}
        if not required.issubset(data.files):
            raise KeyError(f"Missing arrays: {sorted(required - set(data.files))}")
        x = data["x"].astype(np.float32, copy=True)
        y = data["y"].astype(np.int64, copy=True)
        source_index = data["source_index"].astype(np.int64, copy=True)

    if x.shape != (TRAIN_CASES, CHANNELS, TIMEPOINTS):
        raise ValueError(f"Unexpected training input shape: {x.shape}")
    if y.shape != (TRAIN_CASES,) or source_index.shape != (TRAIN_CASES,):
        raise ValueError(
            f"Unexpected metadata shapes: y={y.shape}, source_index={source_index.shape}"
        )
    if not np.isfinite(x).all():
        raise ValueError("Training input contains non-finite values")
    values, counts = np.unique(y, return_counts=True)
    observed = dict(zip(values.tolist(), counts.tolist(), strict=True))
    if observed != CLASS_COUNTS:
        raise ValueError(f"Unexpected labels or class counts: {observed}")
    if not np.array_equal(source_index, np.arange(TRAIN_CASES)):
        raise ValueError("source_index must preserve canonical TRAIN.ts order")
    return x, y, source_index


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
    for label in (0, 1):
        true_positive = float(confusion[label, label])
        false_negative = float(confusion[label].sum() - true_positive)
        false_positive = float(confusion[:, label].sum() - true_positive)
        recall = true_positive / max(true_positive + false_negative, 1.0)
        precision = true_positive / max(true_positive + false_positive, 1.0)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        recalls.append(recall)
        f1_scores.append(f1)

    clipped = np.clip(probability_right, 1e-15, 1.0 - 1e-15)
    log_loss = -np.mean(
        actual * np.log(clipped) + (1 - actual) * np.log(1.0 - clipped)
    )
    return {
        "accuracy": float(np.mean(actual == predicted)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
        "left_recall": float(recalls[0]),
        "right_recall": float(recalls[1]),
        "mean_log_loss": float(log_loss),
        "confusion_matrix": confusion.tolist(),
    }


def save_checkpoint(
    path: Path,
    model: TerminalLogistic,
    preprocessing: dict[str, np.ndarray],
    solver_iterations: int,
    created_utc: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        checkpoint_version=np.asarray(CHECKPOINT_VERSION),
        created_utc=np.asarray(created_utc),
        training_split=np.asarray("official TRAIN only"),
        training_cases=np.asarray(TRAIN_CASES, dtype=np.int32),
        class_ids=np.asarray([0, 1], dtype=np.int8),
        class_names=np.asarray(["left", "right"]),
        channels=np.asarray(CHANNELS, dtype=np.int32),
        timepoints=np.asarray(TIMEPOINTS, dtype=np.int32),
        sampling_rate_hz=np.asarray(SAMPLING_RATE_HZ, dtype=np.float64),
        lowpass_hz=np.asarray(LOWPASS_HZ, dtype=np.float64),
        lowpass_order=np.asarray(LOWPASS_ORDER, dtype=np.int32),
        lowpass_sos=np.asarray(LOWPASS_SOS, dtype=np.float64),
        lowpass_initial=np.asarray(LOWPASS_INITIAL, dtype=np.float64),
        terminal_samples=np.asarray(TERMINAL_SAMPLES, dtype=np.int32),
        terminal_mean_windows=np.asarray(TERMINAL_MEAN_WINDOWS, dtype=np.int32),
        terminal_slope_window=np.asarray(
            TERMINAL_SLOPE_WINDOW, dtype=np.int32
        ),
        feature_count=np.asarray(FEATURES, dtype=np.int32),
        logistic_c=np.asarray(LOGISTIC_C, dtype=np.float64),
        solver=np.asarray(SOLVER),
        solver_max_iter=np.asarray(SOLVER_MAX_ITER, dtype=np.int32),
        solver_tol=np.asarray(SOLVER_TOL, dtype=np.float64),
        solver_random_state=np.asarray(SOLVER_RANDOM_STATE, dtype=np.int32),
        solver_iterations=np.asarray(solver_iterations, dtype=np.int32),
        channel_mean=np.asarray(preprocessing["channel_mean"], dtype=np.float64),
        channel_std=np.asarray(preprocessing["channel_std"], dtype=np.float64),
        feature_mean=np.asarray(preprocessing["feature_mean"], dtype=np.float64),
        feature_std=np.asarray(preprocessing["feature_std"], dtype=np.float64),
        weight=np.asarray(model.weight, dtype=np.float64),
        bias=np.asarray(model.bias, dtype=np.float64),
    )
    temporary.replace(path)


def verify_checkpoint(
    path: Path,
    x: np.ndarray,
    expected_prediction: np.ndarray,
    expected_score: np.ndarray,
) -> None:
    with np.load(path, allow_pickle=False) as checkpoint:
        required = {
            "checkpoint_version",
            "class_ids",
            "channel_mean",
            "channel_std",
            "feature_mean",
            "feature_std",
            "weight",
            "bias",
        }
        if not required.issubset(checkpoint.files):
            raise RuntimeError(
                f"Checkpoint is missing arrays: {sorted(required - set(checkpoint.files))}"
            )
        if str(checkpoint["checkpoint_version"].item()) != CHECKPOINT_VERSION:
            raise RuntimeError("Checkpoint version changed during serialization")
        if checkpoint["class_ids"].tolist() != [0, 1]:
            raise RuntimeError("Checkpoint class order is not left=0, right=1")
        preprocessing = {
            "channel_mean": checkpoint["channel_mean"].copy(),
            "channel_std": checkpoint["channel_std"].copy(),
            "feature_mean": checkpoint["feature_mean"].copy(),
            "feature_std": checkpoint["feature_std"].copy(),
        }
        restored = TerminalLogistic(
            weight=checkpoint["weight"].copy(),
            bias=float(checkpoint["bias"].item()),
        )

    restored_features = transform(x, preprocessing)
    restored_score = restored.decision_function(restored_features)
    restored_prediction = restored.predict_features(restored_features)
    if not np.array_equal(restored_prediction, expected_prediction):
        raise RuntimeError("Reloaded checkpoint changed training predictions")
    if not np.array_equal(restored_score, expected_score):
        raise RuntimeError("Reloaded checkpoint changed decision scores")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
                    "decision_score": float(score[index]),
                    "probability_right": float(probability_right[index]),
                }
            )


def validate_only(x: np.ndarray, y: np.ndarray) -> None:
    preprocessing = fit_preprocessing(x)
    features = transform(x, preprocessing)
    if features.shape != (TRAIN_CASES, FEATURES):
        raise RuntimeError(f"Unexpected feature shape: {features.shape}")
    if not np.isfinite(features).all():
        raise RuntimeError("Preprocessed training features contain non-finite values")
    print("=== FingerMovements Phase 1h validation-only ===")
    print(f"train cases={len(y)} | input={x.shape} | features={features.shape}")
    print(f"class counts={CLASS_COUNTS}")
    print("model=ABC terminal features + L2 Logistic Regression, C=1")
    print("no fitting performed; no files written")
    print("test: LOCKED AND NOT LOADED")


def main() -> None:
    args = parse_args()
    train_path = args.train_data.resolve()
    x, y, source_index = load_training_data(train_path)
    if args.validate_only:
        validate_only(x, y)
        return

    print("=== FingerMovements Phase 1h final all-training-data fit ===", flush=True)
    print(f"training cases={len(y)} | input={x.shape}", flush=True)
    print(
        f"model=ABC terminal features ({FEATURES}) + "
        f"L2 Logistic Regression | C={LOGISTIC_C:g}",
        flush=True,
    )
    print(
        f"optimizer={SOLVER} | max_iter={SOLVER_MAX_ITER} | "
        f"tol={SOLVER_TOL:g} | deterministic seed={SOLVER_RANDOM_STATE}",
        flush=True,
    )
    print(
        "policy=fit once on official TRAIN; final converged solution is checkpointed",
        flush=True,
    )
    print("test: LOCKED AND NOT LOADED", flush=True)

    preprocessing = fit_preprocessing(x)
    features = transform(x, preprocessing)
    classifier = LogisticRegression(
        C=LOGISTIC_C,
        penalty="l2",
        solver=SOLVER,
        max_iter=SOLVER_MAX_ITER,
        tol=SOLVER_TOL,
        random_state=SOLVER_RANDOM_STATE,
        verbose=0 if args.quiet_solver else 1,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(features, y)
    convergence_messages = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    if convergence_messages:
        raise RuntimeError(
            "Liblinear did not converge; checkpoint was not written: "
            + " | ".join(convergence_messages)
        )
    if classifier.classes_.tolist() != [0, 1]:
        raise RuntimeError(f"Unexpected class order: {classifier.classes_.tolist()}")
    solver_iterations = int(classifier.n_iter_[0])
    if solver_iterations >= SOLVER_MAX_ITER:
        raise RuntimeError("Solver reached max_iter; checkpoint was not written")

    model = TerminalLogistic(
        weight=classifier.coef_[0],
        bias=float(classifier.intercept_[0]),
    )
    score = model.decision_function(features)
    prediction = model.predict_features(features).astype(np.int64)
    sklearn_prediction = classifier.predict(features).astype(np.int64)
    if not np.array_equal(prediction, sklearn_prediction):
        raise RuntimeError("Portable inference disagrees with scikit-learn")
    probability_right = expit(score)
    metrics = classification_metrics(y, prediction, probability_right)

    created_utc = datetime.now(timezone.utc).isoformat()
    checkpoint_path = args.checkpoint.resolve()
    save_checkpoint(
        checkpoint_path,
        model,
        preprocessing,
        solver_iterations,
        created_utc,
    )
    verify_checkpoint(checkpoint_path, x, prediction, score)
    checkpoint_sha256 = file_sha256(checkpoint_path)

    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = result_dir / "phase1h_training_predictions.csv"
    write_predictions(
        predictions_path,
        source_index,
        y,
        prediction,
        score,
        probability_right,
    )
    report = {
        "phase": "1h",
        "created_utc": created_utc,
        "scope": "final fit on all 316 official TRAIN cases; TEST not opened",
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(ROOT)),
            "sha256": checkpoint_sha256,
            "version": CHECKPOINT_VERSION,
            "verified_after_reload": True,
        },
        "model": {
            "input_shape": [CHANNELS, TIMEPOINTS],
            "sampling_rate_hz": SAMPLING_RATE_HZ,
            "feature_representation": "ABC terminal low-pass features",
            "feature_count": FEATURES,
            "classifier": "L2 Logistic Regression",
            "logistic_c": LOGISTIC_C,
            "class_mapping": {"0": "left", "1": "right"},
        },
        "optimization": {
            "solver": SOLVER,
            "max_iter": SOLVER_MAX_ITER,
            "tol": SOLVER_TOL,
            "random_state": SOLVER_RANDOM_STATE,
            "iterations_to_convergence": solver_iterations,
            "selection": "none; saved the final converged convex solution",
        },
        "training_data": {
            "path": str(train_path.relative_to(ROOT)),
            "cases": TRAIN_CASES,
            "class_counts": {str(key): value for key, value in CLASS_COUNTS.items()},
        },
        "apparent_training_metrics_not_generalization_estimates": metrics,
        "test_policy": "official TEST remained locked and was not loaded",
    }
    metrics_path = result_dir / "phase1h_training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")

    print("\n=== converged final solution ===")
    print(f"solver iterations={solver_iterations}/{SOLVER_MAX_ITER}")
    print(
        f"apparent TRAIN | log loss={metrics['mean_log_loss']:.6f} | "
        f"accuracy={metrics['accuracy']:.4f} | "
        f"balanced accuracy={metrics['balanced_accuracy']:.4f} | "
        f"macro F1={metrics['macro_f1']:.4f}"
    )
    print(f"confusion matrix={metrics['confusion_matrix']}")
    print("warning: TRAIN metrics are not held-out performance estimates")
    print(f"checkpoint: {checkpoint_path}")
    print(f"checkpoint SHA-256: {checkpoint_sha256}")
    print(f"metrics: {metrics_path}")
    print(f"predictions: {predictions_path}")
    print("test: LOCKED AND NOT LOADED")


if __name__ == "__main__":
    main()
