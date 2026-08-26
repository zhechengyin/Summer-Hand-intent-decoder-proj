#!/usr/bin/env python3
"""Phase 9: replay cold-start deployment policies for the frozen Phase 6 model.

Validation sessions compare two inference-only policies. Strategy A reproduces
the training block contract with a zero-filled future suffix and a reset every
50 bins. Strategy B seeds a continuous rolling window with the final two
calibration seconds. The validation pooled full-session mean R² selects one
policy; only that frozen winner is then evaluated on the registered test split.

No model weights or preprocessing parameters are fitted beyond the required
per-session, first-60-second calibration statistics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import yaml

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from indy_loco.models.midsize.model import load_checkpoint  # noqa: E402

PHASE_NAME: Final = "phase9_deployment_policy_replay"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "midsize" / "checkpoint.pt"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "indy_loco" / "indy"
SESSION_MANIFEST = PROJECT_ROOT / "configs" / "datasets" / "indy_sessions.yaml"
RESULT_DIR = PROJECT_ROOT / "results" / PHASE_NAME
METRICS_PATH = RESULT_DIR / f"{PHASE_NAME}_metrics.json"
VALIDATION_SESSION_PATH = RESULT_DIR / f"{PHASE_NAME}_validation_sessions.csv"
VALIDATION_SUMMARY_PATH = RESULT_DIR / f"{PHASE_NAME}_validation_summary.csv"
VALIDATION_DIFFERENCE_PATH = (
    RESULT_DIR / f"{PHASE_NAME}_validation_prediction_difference.csv"
)
TEST_SESSION_PATH = RESULT_DIR / f"{PHASE_NAME}_locked_test_sessions.csv"
TEST_SUMMARY_PATH = RESULT_DIR / f"{PHASE_NAME}_locked_test_summary.csv"
VALIDATION_TRACE_PATH = RESULT_DIR / f"{PHASE_NAME}_validation_trace.csv"
TEST_TRACE_PATH = RESULT_DIR / f"{PHASE_NAME}_locked_test_trace.csv"
GOLDEN_PATH = RESULT_DIR / f"{PHASE_NAME}_golden_vectors.npz"

PHYSICAL_CHANNELS: Final = 96
FEATURES: Final = 192
ALPHAS: Final = (1.0, 0.1)
CALIBRATION_BINS: Final = 1500
WINDOW_BINS: Final = 50
BIN_SECONDS: Final = 0.040
TARGET_AXES: Final = (0, 1)
INFERENCE_BATCH: Final = 64
TRACE_POST_BINS: Final = 250
A_EQUIVALENCE_TOLERANCE: Final = 2e-5

STRATEGY_A: Final = "A_block_reset"
STRATEGY_B: Final = "B_rolling_calibration_seed"
STRATEGIES: Final = (STRATEGY_A, STRATEGY_B)

REPORT_INTERVALS: Final = {
    "first_10s": 250,
    "first_30s": 750,
    "all_post_calibration": None,
}
DIAGNOSTIC_INTERVALS: Final = {
    "first_2s": (0, 50),
    "stable_after_2s": (50, None),
}


@dataclass
class PreparedSession:
    name: str
    counts: np.ndarray
    velocity: np.ndarray
    features_raw: np.ndarray
    features_normalized: np.ndarray
    feature_mean: np.ndarray
    local_std: np.ndarray
    effective_std: np.ndarray


@dataclass
class StrategyReplay:
    strategy: str
    post_predictions: np.ndarray
    initial_prediction: np.ndarray | None
    a_full_block_max_abs_difference: float | None


def load_session_manifest() -> dict[str, Any]:
    """Load the canonical chronological split used by this replay."""
    with SESSION_MANIFEST.open("r", encoding="utf-8") as source:
        return yaml.safe_load(source)


def load_model_data(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one already-processed session without fitting anything."""
    split = load_session_manifest()["chronological_split"]
    matches = [key for key, sessions in split.items() if name in sessions]
    if len(matches) != 1:
        raise ValueError(f"Expected one registered split for {name}, got {matches}")
    path = PROCESSED_DIR / matches[0] / f"{name}.npz"
    with np.load(path, allow_pickle=False) as artifact:
        return (
            artifact["counts"].astype(np.float32),
            artifact["velocity"].astype(np.float32),
        )


def multiscale_counts(
    counts: np.ndarray,
    alphas: tuple[float, ...] = ALPHAS,
) -> np.ndarray:
    """Build raw-then-causal-EWMA features in the deployment order."""
    blocks = []
    for alpha in alphas:
        if alpha == 1.0:
            blocks.append(counts.astype(np.float32))
            continue
        output = counts.astype(np.float64, copy=True)
        for index in range(1, counts.shape[1]):
            output[:, index] = (
                alpha * counts[:, index] + (1.0 - alpha) * output[:, index - 1]
            )
        blocks.append(output.astype(np.float32))
    return np.concatenate(blocks, axis=0)


def fit_feature_stats(
    features: np.ndarray,
    *,
    observation_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit population statistics from the allowed calibration prefix only."""
    prefix = features[:, :observation_bins]
    return (
        prefix.mean(axis=1, keepdims=True),
        prefix.std(axis=1, ddof=0, keepdims=True) + 1e-6,
    )


def apply_feature_stats(
    features: np.ndarray,
    stats: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Apply already-frozen statistics."""
    mean, std = stats
    return ((features - mean) / std).astype(np.float32)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Auto selects CUDA when available, otherwise CPU. MPS is disabled.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Intentionally replace an existing Phase 9 result directory.",
    )
    parser.add_argument(
        "--protocol-check-only",
        action="store_true",
        help="Validate metadata and a synthetic A-equivalence check without data replay.",
    )
    return parser.parse_args()


def select_device(requested: str) -> Any:
    import torch

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as destination:
            np.savez_compressed(destination, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_arrays(checkpoint: dict[str, Any]) -> tuple[np.ndarray, ...]:
    channels = np.asarray(checkpoint["channels"], dtype=np.int64)
    floor = np.asarray(checkpoint["feature_std_floor"], dtype=np.float32).reshape(
        FEATURES, 1
    )
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32).reshape(2)
    target_std = np.asarray(checkpoint["target_std"], dtype=np.float32).reshape(2)
    if not np.array_equal(channels, np.arange(PHYSICAL_CHANNELS)):
        raise ValueError("Checkpoint channel order is not exactly 0..95")
    if floor.shape != (FEATURES, 1) or np.any(floor <= 0):
        raise ValueError("Invalid checkpoint feature_std_floor")
    if np.any(target_std <= 0):
        raise ValueError("Invalid checkpoint target_std")
    return channels, floor, target_mean, target_std


def validate_registry(checkpoint: dict[str, Any]) -> tuple[list[str], list[str]]:
    manifest = load_session_manifest()
    split = manifest["chronological_split"]
    validation_names = list(split["validation"])
    test_names = list(split["test"])
    if validation_names != list(checkpoint["validation_sessions"]):
        raise ValueError("Checkpoint and manifest validation sessions differ")
    if any(not name.startswith("indy_201612") for name in validation_names):
        raise ValueError("Validation registry is not exclusively December 2016")
    if any(not name.startswith("indy_201701") for name in test_names):
        raise ValueError("Locked test registry is not exclusively January 2017")
    if set(validation_names).intersection(test_names):
        raise ValueError("Validation and test session registries overlap")
    return validation_names, test_names


def prepare_session(
    name: str,
    feature_std_floor: np.ndarray,
) -> PreparedSession:
    counts, velocity = load_model_data(name)
    if counts.shape[0] != PHYSICAL_CHANNELS or velocity.shape[1] < 2:
        raise ValueError(
            f"{name}: unexpected data shapes {counts.shape}, {velocity.shape}"
        )
    if counts.shape[1] != velocity.shape[0] or counts.shape[1] <= CALIBRATION_BINS:
        raise ValueError(f"{name}: invalid aligned timeline")
    features = multiscale_counts(counts, ALPHAS)
    if features.shape != (FEATURES, counts.shape[1]):
        raise ValueError(f"{name}: feature order/shape mismatch {features.shape}")
    feature_mean, local_std = fit_feature_stats(
        features, observation_bins=CALIBRATION_BINS
    )
    # fit_feature_stats uses np.std(ddof=0) + 1e-6 by project definition.
    direct_std = (
        np.std(features[:, :CALIBRATION_BINS], axis=1, ddof=0, keepdims=True) + 1e-6
    ).astype(np.float32)
    if not np.allclose(local_std, direct_std, rtol=0, atol=1e-7):
        raise ValueError(f"{name}: local std is not population std + 1e-6")
    effective_std = np.maximum(local_std, feature_std_floor).astype(np.float32)
    normalized = apply_feature_stats(features, (feature_mean, effective_std))
    return PreparedSession(
        name=name,
        counts=counts,
        velocity=velocity[:, TARGET_AXES].astype(np.float32),
        features_raw=features,
        features_normalized=normalized,
        feature_mean=feature_mean.astype(np.float32),
        local_std=local_std.astype(np.float32),
        effective_std=effective_std,
    )


def model_predict(model: Any, inputs: np.ndarray, device: Any) -> np.ndarray:
    import torch

    model.eval()
    with torch.inference_mode():
        tensor = torch.from_numpy(inputs.astype(np.float32, copy=False)).to(device)
        return model(tensor).cpu().numpy().astype(np.float32)


def predict_strategy_a(
    model: Any,
    normalized_features: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: Any,
) -> StrategyReplay:
    post = normalized_features[:, CALIBRATION_BINS:]
    normalized_predictions: list[np.ndarray] = []
    equivalence_max = 0.0
    for block_start in range(0, post.shape[1], WINDOW_BINS):
        block = post[:, block_start : block_start + WINDOW_BINS]
        length = block.shape[1]
        padded = np.zeros((FEATURES, WINDOW_BINS), dtype=np.float32)
        padded[:, :length] = block
        mask = np.tri(length, WINDOW_BINS, k=0, dtype=np.float32)
        incremental_inputs = padded[None, :, :] * mask[:, None, :]
        incremental_output = model_predict(model, incremental_inputs, device)
        indices = np.arange(length)
        selected = incremental_output[indices, indices]
        normalized_predictions.append(selected)

        full_output = model_predict(model, padded[None, :, :], device)[0, :length]
        equivalence_max = max(
            equivalence_max,
            float(np.max(np.abs(selected - full_output))),
        )
    prediction = np.concatenate(normalized_predictions, axis=0)
    prediction = prediction * target_std + target_mean
    return StrategyReplay(
        strategy=STRATEGY_A,
        post_predictions=prediction.astype(np.float32),
        initial_prediction=None,
        a_full_block_max_abs_difference=equivalence_max,
    )


def rolling_window_batch(
    normalized_features: np.ndarray,
    end_bins: np.ndarray,
) -> np.ndarray:
    windows = np.empty((len(end_bins), FEATURES, WINDOW_BINS), dtype=np.float32)
    for row, end_bin in enumerate(end_bins):
        start = int(end_bin) - WINDOW_BINS + 1
        windows[row] = normalized_features[:, start : int(end_bin) + 1]
    return windows


def predict_strategy_b(
    model: Any,
    normalized_features: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: Any,
) -> StrategyReplay:
    initial_input = normalized_features[
        :, CALIBRATION_BINS - WINDOW_BINS : CALIBRATION_BINS
    ][None]
    initial_normalized = model_predict(model, initial_input, device)[0, -1]
    initial_prediction = initial_normalized * target_std + target_mean

    post_end_bins = np.arange(
        CALIBRATION_BINS,
        normalized_features.shape[1],
        dtype=np.int64,
    )
    chunks: list[np.ndarray] = []
    for start in range(0, len(post_end_bins), INFERENCE_BATCH):
        ends = post_end_bins[start : start + INFERENCE_BATCH]
        inputs = rolling_window_batch(normalized_features, ends)
        chunks.append(model_predict(model, inputs, device)[:, -1])
    normalized_prediction = np.concatenate(chunks, axis=0)
    prediction = normalized_prediction * target_std + target_mean
    return StrategyReplay(
        strategy=STRATEGY_B,
        post_predictions=prediction.astype(np.float32),
        initial_prediction=initial_prediction.astype(np.float32),
        a_full_block_max_abs_difference=None,
    )


def metric_values(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if target.shape != prediction.shape or target.ndim != 2 or target.shape[1] != 2:
        raise ValueError(f"Metric shape mismatch: {target.shape}, {prediction.shape}")
    residual = np.sum((target - prediction) ** 2, axis=0)
    total = np.sum((target - target.mean(axis=0)) ** 2, axis=0)
    scores = 1.0 - residual / np.maximum(total, 1e-12)
    return {
        "r2_x": float(scores[0]),
        "r2_y": float(scores[1]),
        "r2_mean": float(scores.mean()),
        "mse": float(np.mean((target - prediction) ** 2)),
    }


def interval_slice(length: int, limit: int | None) -> slice:
    stop = length if limit is None else min(length, limit)
    return slice(0, stop)


def session_metric_rows(
    split: str,
    session: str,
    target: np.ndarray,
    replay: StrategyReplay,
) -> list[dict[str, Any]]:
    rows = []
    for interval, limit in REPORT_INTERVALS.items():
        selected = interval_slice(len(target), limit)
        metrics = metric_values(target[selected], replay.post_predictions[selected])
        rows.append(
            {
                "split": split,
                "session": session,
                "strategy": replay.strategy,
                "interval": interval,
                "bins": int(len(target[selected])),
                **metrics,
            }
        )
    return rows


def diagnostic_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for label, (start, stop) in DIAGNOSTIC_INTERVALS.items():
        final = len(target) if stop is None else min(len(target), stop)
        if start >= final:
            continue
        selected_target = target[start:final]
        selected_prediction = prediction[start:final]
        output[label] = {
            "bins": int(final - start),
            **metric_values(selected_target, selected_prediction),
        }
    return output


def difference_rows(
    session: str,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for interval, limit in REPORT_INTERVALS.items():
        selected = interval_slice(len(prediction_a), limit)
        difference = np.abs(prediction_a[selected] - prediction_b[selected])
        rows.append(
            {
                "split": "validation",
                "session": session,
                "interval": interval,
                "bins": int(len(prediction_a[selected])),
                "prediction_mae": float(difference.mean()),
                "prediction_max_abs": float(difference.max()),
            }
        )
    return rows


def pooled_summary_rows(
    split: str,
    sessions: list[str],
    targets: dict[str, np.ndarray],
    predictions: dict[str, dict[str, np.ndarray]],
    strategies: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy in strategies:
        for interval, limit in REPORT_INTERVALS.items():
            target_parts = []
            prediction_parts = []
            per_session = []
            total_bins = 0
            for session in sessions:
                selected = interval_slice(len(targets[session]), limit)
                target_part = targets[session][selected]
                prediction_part = predictions[session][strategy][selected]
                target_parts.append(target_part)
                prediction_parts.append(prediction_part)
                per_session.append(metric_values(target_part, prediction_part))
                total_bins += len(target_part)
            pooled = metric_values(
                np.concatenate(target_parts), np.concatenate(prediction_parts)
            )
            macro = {
                key: float(np.mean([metrics[key] for metrics in per_session]))
                for key in ("r2_x", "r2_y", "r2_mean", "mse")
            }
            rows.extend(
                [
                    {
                        "split": split,
                        "aggregation": "pooled",
                        "strategy": strategy,
                        "interval": interval,
                        "sessions": len(sessions),
                        "bins": total_bins,
                        **pooled,
                    },
                    {
                        "split": split,
                        "aggregation": "per_session_macro",
                        "strategy": strategy,
                        "interval": interval,
                        "sessions": len(sessions),
                        "bins": total_bins,
                        **macro,
                    },
                ]
            )
    return rows


def select_validation_winner(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        row
        for row in summary_rows
        if row["aggregation"] == "pooled" and row["interval"] == "all_post_calibration"
    ]
    if {row["strategy"] for row in candidates} != set(STRATEGIES):
        raise ValueError("Validation selection candidates are incomplete")
    ranked = sorted(
        candidates,
        key=lambda row: (row["r2_mean"], -row["mse"]),
        reverse=True,
    )
    return {
        "selected_strategy": ranked[0]["strategy"],
        "primary_metric": "validation pooled all-post-calibration mean R2",
        "tie_breaker": "lower validation pooled all-post-calibration MSE",
        "candidates": candidates,
        "test_used_for_selection": False,
    }


def validation_trace_rows(
    prepared: PreparedSession,
    replay_a: StrategyReplay,
    replay_b: StrategyReplay,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    initial_bin = CALIBRATION_BINS - 1
    rows.append(
        {
            "session": prepared.name,
            "bin_index": initial_bin,
            "bin_end_time_s": (initial_bin + 1) * BIN_SECONDS,
            "phase": "calibration_complete",
            "target_x": float(prepared.velocity[initial_bin, 0]),
            "target_y": float(prepared.velocity[initial_bin, 1]),
            "strategy_a_x": "",
            "strategy_a_y": "",
            "strategy_b_x": float(replay_b.initial_prediction[0]),
            "strategy_b_y": float(replay_b.initial_prediction[1]),
        }
    )
    stop = min(len(replay_a.post_predictions), TRACE_POST_BINS)
    for offset in range(stop):
        bin_index = CALIBRATION_BINS + offset
        rows.append(
            {
                "session": prepared.name,
                "bin_index": bin_index,
                "bin_end_time_s": (bin_index + 1) * BIN_SECONDS,
                "phase": "post_calibration",
                "target_x": float(prepared.velocity[bin_index, 0]),
                "target_y": float(prepared.velocity[bin_index, 1]),
                "strategy_a_x": float(replay_a.post_predictions[offset, 0]),
                "strategy_a_y": float(replay_a.post_predictions[offset, 1]),
                "strategy_b_x": float(replay_b.post_predictions[offset, 0]),
                "strategy_b_y": float(replay_b.post_predictions[offset, 1]),
            }
        )
    return rows


def test_trace_rows(
    prepared: PreparedSession,
    replay: StrategyReplay,
) -> list[dict[str, Any]]:
    stop = min(len(replay.post_predictions), TRACE_POST_BINS)
    return [
        {
            "session": prepared.name,
            "strategy": replay.strategy,
            "bin_index": CALIBRATION_BINS + offset,
            "bin_end_time_s": (CALIBRATION_BINS + offset + 1) * BIN_SECONDS,
            "target_x": float(prepared.velocity[CALIBRATION_BINS + offset, 0]),
            "target_y": float(prepared.velocity[CALIBRATION_BINS + offset, 1]),
            "prediction_x": float(replay.post_predictions[offset, 0]),
            "prediction_y": float(replay.post_predictions[offset, 1]),
        }
        for offset in range(stop)
    ]


def save_golden_vectors(
    prepared: PreparedSession,
    replay_a: StrategyReplay,
    replay_b: StrategyReplay,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    checkpoint_sha256: str,
    model: Any,
    device: Any,
) -> None:
    a_first_input = np.zeros((FEATURES, WINDOW_BINS), dtype=np.float32)
    a_first_input[:] = prepared.features_normalized[
        :, CALIBRATION_BINS : CALIBRATION_BINS + WINDOW_BINS
    ]
    a_first_output = model_predict(model, a_first_input[None], device)[0]
    a_first_prediction = a_first_output * target_std + target_mean

    b_initial_raw = prepared.features_raw[
        :, CALIBRATION_BINS - WINDOW_BINS : CALIBRATION_BINS
    ]
    b_initial_normalized = prepared.features_normalized[
        :, CALIBRATION_BINS - WINDOW_BINS : CALIBRATION_BINS
    ]
    b_end_bins = np.arange(CALIBRATION_BINS, CALIBRATION_BINS + 5)
    b_first_inputs = rolling_window_batch(prepared.features_normalized, b_end_bins)
    b_first_output = model_predict(model, b_first_inputs, device)[:, -1]
    b_first_prediction = b_first_output * target_std + target_mean

    feature_names = np.asarray(
        [f"raw_count_{index}" for index in range(PHYSICAL_CHANNELS)]
        + [f"ewma_alpha_0.1_{index}" for index in range(PHYSICAL_CHANNELS)]
    )
    save_npz_atomic(
        GOLDEN_PATH,
        schema_version=np.asarray("phase9_deployment_replay_golden_v1"),
        checkpoint_sha256=np.asarray(checkpoint_sha256),
        session=np.asarray(prepared.name),
        feature_names=feature_names,
        feature_mean=prepared.feature_mean,
        local_std=prepared.local_std,
        effective_std=prepared.effective_std,
        target_mean=target_mean,
        target_std=target_std,
        strategy_a_first_block_normalized=a_first_input,
        strategy_a_first_block_expected_prediction=a_first_prediction,
        strategy_a_replayed_first_block_prediction=replay_a.post_predictions[
            :WINDOW_BINS
        ],
        strategy_b_initial_unnormalized=b_initial_raw,
        strategy_b_initial_normalized=b_initial_normalized,
        strategy_b_initial_expected_prediction=replay_b.initial_prediction,
        strategy_b_first_five_normalized_windows=b_first_inputs,
        strategy_b_first_five_expected_prediction=b_first_prediction,
        target_first_fifty_post_calibration=prepared.velocity[
            CALIBRATION_BINS : CALIBRATION_BINS + WINDOW_BINS
        ],
    )


def synthetic_protocol_check(model: Any, device: Any) -> float:
    rng = np.random.default_rng(9)
    values = rng.normal(size=(FEATURES, WINDOW_BINS)).astype(np.float32)
    padded_predictions = []
    for timestep in range(WINDOW_BINS):
        incremental = np.zeros_like(values)
        incremental[:, : timestep + 1] = values[:, : timestep + 1]
        output = model_predict(model, incremental[None], device)[0, timestep]
        padded_predictions.append(output)
    full = model_predict(model, values[None], device)[0]
    return float(np.max(np.abs(np.stack(padded_predictions) - full)))


def replay_one(
    model: Any,
    prepared: PreparedSession,
    strategy: str,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: Any,
) -> StrategyReplay:
    if strategy == STRATEGY_A:
        return predict_strategy_a(
            model,
            prepared.features_normalized,
            target_mean,
            target_std,
            device,
        )
    if strategy == STRATEGY_B:
        return predict_strategy_b(
            model,
            prepared.features_normalized,
            target_mean,
            target_std,
            device,
        )
    raise KeyError(strategy)


def main() -> None:
    args = parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    import torch

    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    model, checkpoint = load_checkpoint(CHECKPOINT_PATH, map_location=device)
    model = model.to(device)
    checkpoint_sha = sha256_file(CHECKPOINT_PATH)
    _, feature_std_floor, target_mean, target_std = checkpoint_arrays(checkpoint)
    validation_names, test_names = validate_registry(checkpoint)
    synthetic_difference = synthetic_protocol_check(model, device)
    if synthetic_difference > A_EQUIVALENCE_TOLERANCE:
        raise ValueError(
            f"Synthetic causal equivalence failed: {synthetic_difference:.8g}"
        )

    print("=== Phase 9 deployment-policy replay ===")
    print(f"checkpoint={CHECKPOINT_PATH}")
    print(f"checkpoint SHA-256={checkpoint_sha}")
    print(f"device={device.type} | threads={args.threads}")
    print("channels=0..95 | features=[96 raw, 96 causal EWMA alpha=0.1]")
    print("calibration=first 1500 bins only | stats frozen afterward")
    print(f"synthetic A/full-block max abs difference={synthetic_difference:.3e}")
    print("selection=validation pooled full-post-calibration mean R2")
    print("test=LOCKED until validation winner is frozen")
    if args.protocol_check_only:
        print("protocol check passed; no session arrays loaded; test remained locked")
        return

    if RESULT_DIR.exists() and not args.overwrite:
        raise FileExistsError(f"Result directory exists: {RESULT_DIR}; use --overwrite")
    if RESULT_DIR.exists():
        import shutil

        shutil.rmtree(RESULT_DIR)
    RESULT_DIR.mkdir(parents=True)

    validation_session_rows: list[dict[str, Any]] = []
    validation_difference_rows: list[dict[str, Any]] = []
    validation_targets: dict[str, np.ndarray] = {}
    validation_predictions: dict[str, dict[str, np.ndarray]] = {}
    validation_diagnostics: dict[str, Any] = {}
    a_equivalence_by_session: dict[str, float] = {}
    representative_prepared = None
    representative_a = None
    representative_b = None

    print("\n=== VALIDATION A/B REPLAY (test not loaded) ===", flush=True)
    for name in validation_names:
        print(f"replaying validation: {name}", flush=True)
        prepared = prepare_session(name, feature_std_floor)
        target = prepared.velocity[CALIBRATION_BINS:]
        replay_a = replay_one(
            model, prepared, STRATEGY_A, target_mean, target_std, device
        )
        replay_b = replay_one(
            model, prepared, STRATEGY_B, target_mean, target_std, device
        )
        if len(target) != len(replay_a.post_predictions) or len(target) != len(
            replay_b.post_predictions
        ):
            raise ValueError(f"{name}: prediction/target timeline mismatch")
        equivalence = float(replay_a.a_full_block_max_abs_difference)
        if equivalence > A_EQUIVALENCE_TOLERANCE:
            raise ValueError(f"{name}: A/full-block equivalence failed: {equivalence}")
        a_equivalence_by_session[name] = equivalence
        validation_targets[name] = target
        validation_predictions[name] = {
            STRATEGY_A: replay_a.post_predictions,
            STRATEGY_B: replay_b.post_predictions,
        }
        validation_session_rows.extend(
            session_metric_rows("validation", name, target, replay_a)
        )
        validation_session_rows.extend(
            session_metric_rows("validation", name, target, replay_b)
        )
        validation_difference_rows.extend(
            difference_rows(name, replay_a.post_predictions, replay_b.post_predictions)
        )
        validation_diagnostics[name] = {
            STRATEGY_A: diagnostic_metrics(target, replay_a.post_predictions),
            STRATEGY_B: diagnostic_metrics(target, replay_b.post_predictions),
            "strategy_b_calibration_complete_prediction": replay_b.initial_prediction.tolist(),
            "strategy_a_calibration_complete_prediction": None,
        }
        if representative_prepared is None:
            representative_prepared = prepared
            representative_a = replay_a
            representative_b = replay_b

    validation_summary_rows = pooled_summary_rows(
        "validation",
        validation_names,
        validation_targets,
        validation_predictions,
        STRATEGIES,
    )
    selection = select_validation_winner(validation_summary_rows)
    selected_strategy = selection["selected_strategy"]
    print(f"\nVALIDATION SELECTED: {selected_strategy}", flush=True)
    for candidate in selection["candidates"]:
        print(
            f"{candidate['strategy']}: pooled all R2={candidate['r2_mean']:+.6f} "
            f"MSE={candidate['mse']:.6f}",
            flush=True,
        )

    write_csv(VALIDATION_SESSION_PATH, validation_session_rows)
    write_csv(VALIDATION_SUMMARY_PATH, validation_summary_rows)
    write_csv(VALIDATION_DIFFERENCE_PATH, validation_difference_rows)
    write_csv(
        VALIDATION_TRACE_PATH,
        validation_trace_rows(
            representative_prepared, representative_a, representative_b
        ),
    )
    save_golden_vectors(
        representative_prepared,
        representative_a,
        representative_b,
        target_mean,
        target_std,
        checkpoint_sha,
        model,
        device,
    )

    # Only after validation artifacts and the selection decision are frozen do
    # we load any test session array.
    print(f"\n=== OPENING LOCKED TEST: winner={selected_strategy} ===", flush=True)
    test_session_rows: list[dict[str, Any]] = []
    test_targets: dict[str, np.ndarray] = {}
    test_predictions: dict[str, dict[str, np.ndarray]] = {}
    test_diagnostics: dict[str, Any] = {}
    representative_test_prepared = None
    representative_test_replay = None
    for name in test_names:
        print(f"replaying locked test: {name}", flush=True)
        prepared = prepare_session(name, feature_std_floor)
        target = prepared.velocity[CALIBRATION_BINS:]
        replay = replay_one(
            model, prepared, selected_strategy, target_mean, target_std, device
        )
        if len(target) != len(replay.post_predictions):
            raise ValueError(f"{name}: test prediction/target timeline mismatch")
        test_targets[name] = target
        test_predictions[name] = {selected_strategy: replay.post_predictions}
        test_session_rows.extend(
            session_metric_rows("locked_test", name, target, replay)
        )
        test_diagnostics[name] = {
            selected_strategy: diagnostic_metrics(target, replay.post_predictions),
            "calibration_complete_prediction": (
                None
                if replay.initial_prediction is None
                else replay.initial_prediction.tolist()
            ),
        }
        if representative_test_prepared is None:
            representative_test_prepared = prepared
            representative_test_replay = replay

    test_summary_rows = pooled_summary_rows(
        "locked_test",
        test_names,
        test_targets,
        test_predictions,
        (selected_strategy,),
    )
    write_csv(TEST_SESSION_PATH, test_session_rows)
    write_csv(TEST_SUMMARY_PATH, test_summary_rows)
    write_csv(
        TEST_TRACE_PATH,
        test_trace_rows(representative_test_prepared, representative_test_replay),
    )

    metrics = {
        "phase": PHASE_NAME,
        "completed_at_utc": utc_now(),
        "checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_metadata": {
            "physical_channel_count": checkpoint["physical_channel_count"],
            "input_feature_count": checkpoint["input_feature_count"],
            "parameter_count": checkpoint["parameter_count"],
            "checkpoint_epoch": checkpoint["checkpoint_epoch"],
            "channels": checkpoint["channels"],
            "experiment_config": checkpoint["experiment_config"],
            "target_mean": target_mean.tolist(),
            "target_std": target_std.tolist(),
            "feature_std_floor_shape": list(feature_std_floor.shape),
        },
        "protocol": {
            "bin_seconds": BIN_SECONDS,
            "calibration_bins": CALIBRATION_BINS,
            "calibration_seconds": CALIBRATION_BINS * BIN_SECONDS,
            "window_bins": WINDOW_BINS,
            "window_seconds": WINDOW_BINS * BIN_SECONDS,
            "feature_order": "raw_count[0..95], ewma_alpha_0.1[0..95]",
            "local_std": "population_std_ddof0_plus_1e-6",
            "effective_std": "maximum(local_std, checkpoint_feature_std_floor)",
            "selection_metric": selection["primary_metric"],
            "test_loaded_after_selection": True,
            "future_data_used": False,
        },
        "strategy_a_full_block_equivalence": {
            "tolerance": A_EQUIVALENCE_TOLERANCE,
            "synthetic_max_abs_difference": synthetic_difference,
            "validation_by_session": a_equivalence_by_session,
            "passed": True,
        },
        "validation_session_metrics": validation_session_rows,
        "validation_summary": validation_summary_rows,
        "validation_prediction_difference": validation_difference_rows,
        "validation_cold_start_diagnostics": validation_diagnostics,
        "selection": selection,
        "locked_test_policy": selected_strategy,
        "locked_test_session_metrics": test_session_rows,
        "locked_test_summary": test_summary_rows,
        "locked_test_cold_start_diagnostics": test_diagnostics,
        "representative_trace_sessions": {
            "validation": representative_prepared.name,
            "locked_test": representative_test_prepared.name,
        },
    }
    write_json_atomic(metrics, METRICS_PATH)

    print("\n=== Phase 9 complete ===")
    print(f"selected strategy: {selected_strategy}")
    for row in test_summary_rows:
        if row["aggregation"] == "pooled" and row["interval"] == "all_post_calibration":
            print(f"locked test pooled R2={row['r2_mean']:+.6f} MSE={row['mse']:.6f}")
    print(f"metrics: {METRICS_PATH}")
    print(f"golden vectors: {GOLDEN_PATH}")


if __name__ == "__main__":
    main()
