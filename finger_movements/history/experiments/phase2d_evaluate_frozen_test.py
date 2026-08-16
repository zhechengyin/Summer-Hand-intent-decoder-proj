"""Phase 2d: pure inference of the frozen Phase 2c model on official TEST.

This retrospective benchmark never fits, recalibrates, selects, or estimates
any parameter from TEST. The official TEST was exposed earlier in the project,
so this result is not presented as a pristine blind-test estimate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARCHIVE_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.finger_movements import FingerMovementsCausalCssdLda
from models.finger_movements.cssd_lda.model import (
    AVAILABLE_INPUT_SAMPLES,
    CHANNELS,
    COLD_START_MS,
    HISTORY_MS,
    UPDATE_MS,
    UPDATE_SAMPLES,
)

TEST_CASES = 100
TEST_CLASS_COUNTS = {0: 49, 1: 51}
FROZEN_CHECKPOINT_SHA256 = (
    "87b84cc2c8baf9efdc1ccf37ad28f5f58ad13c4db2a8f8a273fe73fce9956101"
)
TRAIN_DATA_SHA256 = (
    "a2025f277b5351839554e0ecf3398f1f4fd5151a4fc90f0e25c873734f5a91d1"
)
DEVELOPMENT_MEAN_OOF_BA = 0.8398563206879515
EXPECTED_VARIANT = "causal__window_400ms__bin_50ms"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "models/finger_movements/cssd_lda/checkpoints"
            / "finger_movements_cssd_lda_phase2c_causal_400ms.npz"
        ),
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=PROJECT_ROOT / "data/processed/finger_movements/test.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARCHIVE_ROOT / "results/phase2d_official_test_400ms",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow deterministic rerun over an existing metrics file.",
    )
    args = parser.parse_args()
    if "test" not in args.test_data.name.lower():
        parser.error("--test-data must identify the official TEST split")
    if any("train" in part.lower() for part in args.test_data.parts):
        parser.error("TEST inference refuses a path identified as TRAIN")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_checkpoint(path: Path) -> tuple[FingerMovementsCausalCssdLda, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen checkpoint not found: {path}")
    digest = sha256(path)
    if digest != FROZEN_CHECKPOINT_SHA256:
        raise ValueError(
            "Frozen checkpoint SHA-256 mismatch: "
            f"observed={digest}, expected={FROZEN_CHECKPOINT_SHA256}"
        )
    model = FingerMovementsCausalCssdLda.load(path)
    metadata = model.metadata
    required = {
        "selected_variant": EXPECTED_VARIANT,
        "training_data_sha256": TRAIN_DATA_SHA256,
        "training_cases": 316,
        "official_test_loaded": False,
    }
    for key, expected in required.items():
        observed = metadata.get(key)
        if observed != expected:
            raise ValueError(
                f"Checkpoint metadata mismatch for {key}: "
                f"observed={observed!r}, expected={expected!r}"
            )
    return model, digest


def load_test_data(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Official TEST not found: {path}")
    digest = sha256(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"x", "y", "source_index", "channel_names"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"TEST is missing arrays: {sorted(missing)}")
        x = data["x"].astype(np.float64, copy=True)
        y = data["y"].astype(np.int64, copy=True)
        source_index = data["source_index"].astype(np.int64, copy=True)
        channel_names = data["channel_names"].astype(str, copy=True)
    if x.shape != (TEST_CASES, CHANNELS, AVAILABLE_INPUT_SAMPLES):
        raise ValueError(f"Unexpected TEST x shape: {x.shape}")
    if y.shape != (TEST_CASES,) or source_index.shape != (TEST_CASES,):
        raise ValueError("Unexpected TEST label or source-index shape")
    if channel_names.shape != (CHANNELS,) or len(set(channel_names)) != CHANNELS:
        raise ValueError("Expected 28 unique TEST channel names")
    if dict(Counter(y.tolist())) != TEST_CLASS_COUNTS:
        raise ValueError(f"Unexpected TEST class counts: {dict(Counter(y.tolist()))}")
    if not np.array_equal(source_index, np.arange(TEST_CASES)):
        raise ValueError("TEST source_index must preserve official order")
    if not np.isfinite(x).all():
        raise ValueError("TEST contains non-finite values")
    if np.all(x[:, :-1, 28:] == x[:, 1:, :22]):
        raise ValueError("Detected retired UEA sliding-channel layout in TEST")
    return x, y, source_index, channel_names, digest


def streaming_inference(
    model: FingerMovementsCausalCssdLda, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions: list[int] = []
    probabilities: list[np.ndarray] = []
    scores: list[float] = []
    for case in x:
        state = model.new_stream(case[:, 0])
        result = None
        for start in range(0, AVAILABLE_INPUT_SAMPLES, UPDATE_SAMPLES):
            result = state.push(case[:, start : start + UPDATE_SAMPLES])
        if result is None:
            raise RuntimeError("Streaming model did not emit after cold start")
        prediction, probability, score = result
        predictions.append(prediction)
        probabilities.append(probability)
        scores.append(score)
    return np.asarray(predictions), np.stack(probabilities), np.asarray(scores)


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * np.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return float(center - radius), float(center + radius)


def metric_bundle(
    labels: np.ndarray, prediction: np.ndarray, probability: np.ndarray
) -> dict[str, Any]:
    matrix = confusion_matrix(labels, prediction, labels=[0, 1])
    accuracy = float(accuracy_score(labels, prediction))
    low, high = wilson_interval(int(np.sum(labels == prediction)), len(labels))
    return {
        "accuracy": accuracy,
        "accuracy_wilson_95_low": low,
        "accuracy_wilson_95_high": high,
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "mean_log_loss": float(log_loss(labels, probability, labels=[0, 1])),
        "left_recall": float(matrix[0, 0] / matrix[0].sum()),
        "right_recall": float(matrix[1, 1] / matrix[1].sum()),
        "confusion_matrix": matrix.tolist(),
    }


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    test_path = args.test_data.resolve()
    output_dir = args.output_dir.resolve()
    metrics_path = output_dir / "phase2d_official_test_metrics.json"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Result already exists: {metrics_path}; use --overwrite to rerun"
        )

    print("=== FingerMovements Phase 2d frozen official-TEST inference ===")
    print("policy=pure inference; no fit/recalibration/selection")
    print("status=retrospective benchmark; TEST was exposed earlier")
    model, checkpoint_digest = load_frozen_checkpoint(checkpoint_path)
    print(f"checkpoint SHA-256={checkpoint_digest} | verified=PASS")
    print("opening corrected official TEST now")
    x, y, source_index, channel_names, test_digest = load_test_data(test_path)
    if not np.array_equal(channel_names, model.channel_names):
        raise ValueError("TEST channel names/order differ from checkpoint")

    batch_score = model.decision_function(x, channel_names)
    batch_probability = model.predict_proba(x, channel_names)
    batch_prediction = model.predict(x, channel_names)
    stream_prediction, stream_probability, stream_score = streaming_inference(model, x)
    score_error = float(np.max(np.abs(batch_score - stream_score)))
    probability_error = float(np.max(np.abs(batch_probability - stream_probability)))
    if not np.array_equal(batch_prediction, stream_prediction):
        raise RuntimeError("Batch and streaming TEST predictions differ")
    if max(score_error, probability_error) > 1e-12:
        raise RuntimeError("Batch and streaming TEST inference are inconsistent")

    metrics = metric_bundle(y, batch_prediction, batch_probability)
    rows = [
        {
            "source_index": int(source_index[index]),
            "true_label": int(y[index]),
            "true_class": "left" if y[index] == 0 else "right",
            "predicted_label": int(batch_prediction[index]),
            "predicted_class": (
                "left" if batch_prediction[index] == 0 else "right"
            ),
            "probability_left": float(batch_probability[index, 0]),
            "probability_right": float(batch_probability[index, 1]),
            "decision_score": float(batch_score[index]),
            "correct": bool(batch_prediction[index] == y[index]),
        }
        for index in range(TEST_CASES)
    ]
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "2d",
        "evaluated_model_phase": "2c",
        "scope": "retrospective pure inference on corrected official TEST",
        "interpretation": (
            "not a pristine blind test because official TEST labels/results "
            "were exposed earlier in the project"
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_variant": EXPECTED_VARIANT,
        "development_mean_oof_balanced_accuracy": DEVELOPMENT_MEAN_OOF_BA,
        "test_data": str(test_path),
        "test_data_sha256": test_digest,
        "test_cases": TEST_CASES,
        "class_counts": {"left": 49, "right": 51},
        "inference_contract": {
            "feature_history_ms": HISTORY_MS,
            "cold_start_ms": COLD_START_MS,
            "streaming_update_ms": UPDATE_MS,
            "training_or_test_derived_estimation": False,
        },
        "streaming_equivalence": {
            "predictions_exact": True,
            "maximum_score_absolute_error": score_error,
            "maximum_probability_absolute_error": probability_error,
        },
        "metrics": metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(output_dir / "phase2d_official_test_predictions.csv", rows)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")

    print("batch/streaming equivalence=PASS")
    print(
        f"accuracy={100.0 * metrics['accuracy']:.2f}% | "
        f"balanced accuracy={100.0 * metrics['balanced_accuracy']:.2f}% | "
        f"macro-F1={100.0 * metrics['macro_f1']:.2f}%"
    )
    print(
        "accuracy Wilson 95% CI="
        f"[{100.0 * metrics['accuracy_wilson_95_low']:.2f}%, "
        f"{100.0 * metrics['accuracy_wilson_95_high']:.2f}%]"
    )
    print(f"confusion matrix [left,right]={metrics['confusion_matrix']}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
