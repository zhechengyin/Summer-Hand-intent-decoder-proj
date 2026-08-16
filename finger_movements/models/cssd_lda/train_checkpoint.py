"""Fit and verify the frozen Phase 2c causal model on all official TRAIN cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from model import (
    AVAILABLE_INPUT_SAMPLES,
    CHANNELS,
    CLASS_COUNTS,
    COLD_START_MS,
    HISTORY_MS,
    UPDATE_MS,
    UPDATE_SAMPLES,
    FingerMovementsCausalCssdLda,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data/processed/finger_movements/train.npz"
DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent
    / "checkpoints/finger_movements_cssd_lda_phase2c_causal_400ms.npz"
)
DEVELOPMENT_MEAN_OOF_BA = 0.8398563206879515
DEVELOPMENT_SEED_BA_SD = 0.005394642486715603
DEVELOPMENT_WORST_SEED_BA = 0.8324520290029244


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    if any("test" in part.lower() for part in args.data.parts):
        parser.error("Final fitting refuses to load any TEST path")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_data(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Training data not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {"x", "y", "source_index", "channel_names"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"Missing arrays: {sorted(missing)}")
        x = data["x"].astype(np.float64, copy=True)
        y = data["y"].astype(np.int64, copy=True)
        source_index = data["source_index"].astype(np.int64, copy=True)
        channel_names = data["channel_names"].astype(str, copy=True)
    if x.shape != (316, CHANNELS, AVAILABLE_INPUT_SAMPLES):
        raise ValueError(f"Unexpected x shape: {x.shape}")
    if dict(Counter(y.tolist())) != CLASS_COUNTS:
        raise ValueError(f"Unexpected class counts: {dict(Counter(y.tolist()))}")
    if not np.array_equal(source_index, np.arange(len(y))):
        raise ValueError("source_index must preserve official TRAIN order")
    if np.all(x[:, :-1, 28:] == x[:, 1:, :22]):
        raise ValueError("Detected the retired UEA sliding-channel layout error")
    return x, y, source_index, channel_names


def metric_bundle(
    labels: np.ndarray, prediction: np.ndarray, probability: np.ndarray
) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "mean_log_loss": float(log_loss(labels, probability, labels=[0, 1])),
        "confusion_matrix": confusion_matrix(
            labels, prediction, labels=[0, 1]
        ).tolist(),
    }


def verify_streaming(
    model: FingerMovementsCausalCssdLda,
    x: np.ndarray,
    channel_names: np.ndarray,
) -> dict[str, object]:
    batch_score = model.decision_function(x, channel_names)
    batch_probability = model.predict_proba(x, channel_names)
    batch_prediction = model.predict(x, channel_names)
    stream_scores: list[float] = []
    stream_probabilities: list[np.ndarray] = []
    stream_predictions: list[int] = []
    for case in x:
        state = model.new_stream(case[:, 0])
        result = None
        for start in range(0, AVAILABLE_INPUT_SAMPLES, UPDATE_SAMPLES):
            result = state.push(case[:, start : start + UPDATE_SAMPLES])
        if result is None:
            raise RuntimeError("Streaming state did not emit after 500 ms")
        prediction, probability, score = result
        stream_predictions.append(prediction)
        stream_probabilities.append(probability)
        stream_scores.append(score)
    prediction_array = np.asarray(stream_predictions)
    probability_array = np.stack(stream_probabilities)
    score_array = np.asarray(stream_scores)
    score_error = float(np.max(np.abs(batch_score - score_array)))
    probability_error = float(np.max(np.abs(batch_probability - probability_array)))
    if not np.array_equal(batch_prediction, prediction_array):
        raise RuntimeError("Streaming checkpoint changed predictions")
    if max(score_error, probability_error) > 1e-12:
        raise RuntimeError("Streaming checkpoint is not numerically equivalent")
    return {
        "cases": len(x),
        "update_ms": UPDATE_MS,
        "updates_before_first_output": AVAILABLE_INPUT_SAMPLES // UPDATE_SAMPLES,
        "predictions_exact": True,
        "maximum_score_absolute_error": score_error,
        "maximum_probability_absolute_error": probability_error,
    }


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    checkpoint_path = args.checkpoint.resolve()
    x, y, _, channel_names = load_training_data(data_path)
    data_hash = sha256(data_path)
    created_utc = datetime.now(timezone.utc).isoformat()

    print("=== FingerMovements Phase 2c causal full-TRAIN fit ===")
    print(f"data={data_path}")
    print(
        f"cases={len(y)} | official input={CHANNELS}x{AVAILABLE_INPUT_SAMPLES} | "
        "test=REFUSED"
    )
    print(
        f"feature history={HISTORY_MS} ms | cold start={COLD_START_MS} ms | "
        f"streaming update={UPDATE_MS} ms"
    )
    print("filter=left-to-right causal SOS; future samples forbidden")

    metadata = {
        "created_utc": created_utc,
        "model": "FingerMovements causal CSSD + hierarchical LDA",
        "selection": "Phase 2c official-TRAIN-only repeated stratified CV",
        "selected_variant": "causal__window_400ms__bin_50ms",
        "development_mean_oof_balanced_accuracy": DEVELOPMENT_MEAN_OOF_BA,
        "training_data_path": str(data_path),
        "training_data_sha256": data_hash,
        "training_cases": len(y),
        "official_test_loaded": False,
        "deployment_status": "causal candidate; continuous-stream validation pending",
    }
    model = FingerMovementsCausalCssdLda.fit(x, y, channel_names, metadata=metadata)
    score_before = model.decision_function(x, channel_names)
    probability_before = model.predict_proba(x, channel_names)
    prediction_before = model.predict(x, channel_names)
    training_metrics = metric_bundle(y, prediction_before, probability_before)

    model.save(checkpoint_path)
    restored = FingerMovementsCausalCssdLda.load(checkpoint_path)
    score_after = restored.decision_function(x, channel_names)
    probability_after = restored.predict_proba(x, channel_names)
    prediction_after = restored.predict(x, channel_names)
    if not np.array_equal(prediction_before, prediction_after):
        raise RuntimeError("Reloaded checkpoint changed training predictions")
    score_reload_error = float(np.max(np.abs(score_before - score_after)))
    probability_reload_error = float(
        np.max(np.abs(probability_before - probability_after))
    )
    if max(score_reload_error, probability_reload_error) > 1e-12:
        raise RuntimeError("Reloaded checkpoint changed scores or probabilities")
    streaming = verify_streaming(restored, x, channel_names)

    checkpoint_hash = sha256(checkpoint_path)
    record = {
        "created_utc": created_utc,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "training_data": str(data_path),
        "training_data_sha256": data_hash,
        "configuration": {
            "covariance": "empirical",
            "trial_trace_normalization": True,
            "bp_patterns_per_class": 1,
            "f2_patterns_per_class": 1,
            "fusion": "LDA",
            "temporal_filters": "fourth-order causal Butterworth SOS",
            "history_ms": HISTORY_MS,
            "cold_start_ms": COLD_START_MS,
            "streaming_update_ms": UPDATE_MS,
        },
        "selection_evidence": {
            "mean_oof_balanced_accuracy": DEVELOPMENT_MEAN_OOF_BA,
            "seed_balanced_accuracy_sd": DEVELOPMENT_SEED_BA_SD,
            "worst_seed_balanced_accuracy": DEVELOPMENT_WORST_SEED_BA,
            "seeds": [42, 43, 44],
            "folds_per_seed": 5,
        },
        "apparent_full_training_metrics": training_metrics,
        "reload_verification": {
            "predictions_exact": True,
            "maximum_score_absolute_error": score_reload_error,
            "maximum_probability_absolute_error": probability_reload_error,
        },
        "streaming_equivalence": streaming,
        "test_policy": (
            "official TEST refused and not loaded by this checkpoint-training "
            "entry point; frozen evaluation is a separate Phase 2d record"
        ),
        "deployment_status": (
            "causal firmware candidate; continuous EEG and rest/no-intent "
            "validation are still required"
        ),
    }
    metrics_path = checkpoint_path.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")

    print(
        f"apparent TRAIN accuracy={training_metrics['accuracy']:.4f} | "
        f"BA={training_metrics['balanced_accuracy']:.4f} | "
        f"macro-F1={training_metrics['macro_f1']:.4f}"
    )
    print("reload verification=PASS | streaming equivalence=PASS")
    print(f"checkpoint={checkpoint_path}")
    print(f"checkpoint SHA-256={checkpoint_hash}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
