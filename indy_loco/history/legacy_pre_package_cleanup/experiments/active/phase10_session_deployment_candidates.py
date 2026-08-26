#!/usr/bin/env python3
"""Build and replay six session-specific Phase 7 deployment candidates.

The Phase 7 fold-1 checkpoints remain immutable benchmark artifacts.  This
script derives a training-only 60-second feature-std floor, creates a separate
deployment candidate, and evaluates the exact firmware rolling-window policy
on the same held-out test-reach bins used by the benchmark.

No architecture search or weight fitting happens here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from indy_loco.history.experiments.phase7.phase7_ann_vs_snn_fivefold import (  # noqa: E402
    BIN_SECONDS,
    PHYSICAL_CHANNELS,
    SESSION_BY_NAME,
    SESSIONS,
    aggregate_40ms,
    binned_reach_bounds,
    eligible_reaches,
    load_session,
    make_fold_indices,
    split_fold,
)
from indy_loco.models.midsize.model import MidsizeTCNGRU  # noqa: E402

PHASE_NAME: Final = "phase10_session_deployment_candidates"
MODEL_ROOT = PROJECT_ROOT / "models" / "midsize"
SESSION_MANIFEST = MODEL_ROOT / "session_checkpoints.json"
RESULT_DIR = PROJECT_ROOT / "results" / PHASE_NAME
RESULT_MANIFEST = RESULT_DIR / f"{PHASE_NAME}_metrics.json"

SESSIONS_REQUIRED: Final = tuple(spec.name for spec in SESSIONS)
FEATURES: Final = 192
WINDOW_BINS: Final = 50
CALIBRATION_BINS: Final = 1_500
EWMA_ALPHA: Final = 0.1
STD_FLOOR_PERCENTILE: Final = 10.0
FLOOR_BLOCK_STRIDE: Final = CALIBRATION_BINS
INFERENCE_BATCH: Final = 128
GOLDEN_WINDOW_COUNT: Final = 8
EXPECTED_PARAMETER_COUNT: Final = 86_978


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions",
        nargs="+",
        choices=SESSIONS_REQUIRED,
        default=list(SESSIONS_REQUIRED),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing candidate sidecars and result metadata.",
    )
    parser.add_argument(
        "--protocol-check-only",
        action="store_true",
        help="Validate checkpoints and pure preprocessing contracts without session replay.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def save_torch_atomic(payload: dict[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def select_device(requested: str) -> Any:
    import torch

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(path: Path, device: Any) -> tuple[Any, dict[str, Any]]:
    import torch

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required = {
        "session",
        "fold",
        "model_state",
        "selected_channel_indices",
        "feature_mean",
        "feature_std",
        "target_mean",
        "target_std",
        "model_config",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"{path}: missing checkpoint keys {missing}")
    if checkpoint["fold"] != 1:
        raise ValueError(f"{path}: deployment source must be the registered fold 1")
    channels = np.asarray(checkpoint["selected_channel_indices"], dtype=np.int64)
    if channels.shape != (PHYSICAL_CHANNELS,) or len(np.unique(channels)) != 96:
        raise ValueError(f"{path}: invalid selected-channel mapping")
    model = MidsizeTCNGRU().to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("Unexpected model parameter count")
    return model, checkpoint


def continuous_features(counts: np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != PHYSICAL_CHANNELS:
        raise ValueError(f"Expected 96-by-time counts, received {values.shape}")
    if values.shape[1] == 0 or np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("Counts must be non-empty, finite, and non-negative")
    ewma = values.copy()
    for index in range(1, values.shape[1]):
        ewma[:, index] = (
            EWMA_ALPHA * values[:, index]
            + (1.0 - EWMA_ALPHA) * ewma[:, index - 1]
        )
    return np.concatenate((values, ewma), axis=0).astype(np.float32)


def fit_training_floor(
    counts: np.ndarray,
    bounds: np.ndarray,
    train_reaches: np.ndarray,
    fallback_floor: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a 60-second std floor using fold-1 training reaches only.

    Training reaches are ordered chronologically and concatenated before the
    causal EWMA is built.  Non-overlapping 1500-bin blocks mimic independent
    firmware calibration periods.  The per-feature 10th percentile matches the
    promoted Phase 6 floor statistic while avoiding validation/test reaches.
    """
    ordered = sorted((int(index) for index in train_reaches), key=lambda i: bounds[i, 0])
    segments = [counts[:, bounds[index, 0] : bounds[index, 1]] for index in ordered]
    segments = [segment for segment in segments if segment.shape[1] > 0]
    timeline = np.concatenate(segments, axis=1).astype(np.float32)
    features = continuous_features(timeline)
    starts = list(
        range(0, features.shape[1] - CALIBRATION_BINS + 1, FLOOR_BLOCK_STRIDE)
    )
    if not starts:
        raise ValueError("Training reaches contain less than one 60-second floor block")
    block_stds = np.stack(
        [
            features[:, start : start + CALIBRATION_BINS].std(axis=1, ddof=0)
            + 1e-6
            for start in starts
        ]
    ).astype(np.float32)
    fallback = np.asarray(fallback_floor, dtype=np.float32).reshape(FEATURES)
    if np.any(fallback <= 0) or not np.isfinite(fallback).all():
        raise ValueError("Fallback floor must contain 192 finite positive values")
    floor = np.empty(FEATURES, dtype=np.float32)
    fallback_features: list[int] = []
    for feature in range(FEATURES):
        valid = block_stds[:, feature][block_stds[:, feature] > 1e-4]
        if valid.size == 0:
            floor[feature] = fallback[feature]
            fallback_features.append(feature)
        else:
            floor[feature] = np.percentile(valid, STD_FLOOR_PERCENTILE)
    metadata = {
        "method": "fold1_train_reaches_chronological_60s_blocks",
        "calibration_bins": CALIBRATION_BINS,
        "bin_seconds": BIN_SECONDS,
        "block_stride_bins": FLOOR_BLOCK_STRIDE,
        "block_count": len(starts),
        "training_reach_count": len(ordered),
        "training_bins": int(timeline.shape[1]),
        "percentile": STD_FLOOR_PERCENTILE,
        "silent_threshold": 1e-4,
        "silent_feature_fallback": "promoted_phase6_feature_std_floor",
        "silent_feature_fallback_count": len(fallback_features),
        "silent_feature_fallback_indices": fallback_features,
        "test_or_validation_reaches_used": False,
        "minimum": float(floor.min()),
        "median": float(np.median(floor)),
        "maximum": float(floor.max()),
    }
    return floor, metadata


def model_predict(model: Any, inputs: np.ndarray, device: Any) -> np.ndarray:
    import torch

    with torch.inference_mode():
        tensor = torch.from_numpy(inputs.astype(np.float32, copy=False)).to(device)
        return model(tensor).cpu().numpy().astype(np.float32)


def metric_values(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if target.shape != prediction.shape or target.ndim != 2 or target.shape[1] != 2:
        raise ValueError(f"Metric shape mismatch: {target.shape}, {prediction.shape}")
    residual = np.sum((target - prediction) ** 2, axis=0)
    total = np.sum((target - target.mean(axis=0)) ** 2, axis=0)
    score = 1.0 - residual / np.maximum(total, 1e-12)
    return {
        "r2_x": float(score[0]),
        "r2_y": float(score[1]),
        "r2_mean": float(score.mean()),
        "mse": float(np.mean((target - prediction) ** 2)),
    }


def phase7_reference(
    model: Any,
    counts: np.ndarray,
    velocity: np.ndarray,
    bounds: np.ndarray,
    test_reaches: np.ndarray,
    checkpoint: dict[str, Any],
    device: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32).reshape(FEATURES, 1)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32).reshape(FEATURES, 1)
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32).reshape(2)
    target_std = np.asarray(checkpoint["target_std"], dtype=np.float32).reshape(2)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    ordered = sorted((int(index) for index in test_reaches), key=lambda i: bounds[i, 0])
    for reach in ordered:
        start, stop = (int(value) for value in bounds[reach])
        features = continuous_features(counts[:, start:stop])
        normalized = ((features - mean) / std).astype(np.float32)
        for left in range(0, normalized.shape[1], WINDOW_BINS):
            right = min(left + WINDOW_BINS, normalized.shape[1])
            length = right - left
            inputs = np.zeros((1, FEATURES, WINDOW_BINS), dtype=np.float32)
            inputs[0, :, :length] = normalized[:, left:right]
            output = model_predict(model, inputs, device)[0, :length]
            predictions.append(output * target_std + target_mean)
            targets.append(velocity[start + left : start + right])
            indices.append(np.arange(start + left, start + right, dtype=np.int64))
    return (
        np.concatenate(indices),
        np.concatenate(targets).astype(np.float32),
        np.concatenate(predictions).astype(np.float32),
    )


def rolling_inputs(features: np.ndarray, end_bins: np.ndarray) -> np.ndarray:
    output = np.empty((len(end_bins), FEATURES, WINDOW_BINS), dtype=np.float32)
    for row, end_bin in enumerate(end_bins):
        end = int(end_bin) + 1
        output[row] = features[:, end - WINDOW_BINS : end]
    return output


def firmware_replay(
    model: Any,
    features: np.ndarray,
    velocity: np.ndarray,
    selected_bins: np.ndarray,
    floor: np.ndarray,
    checkpoint: dict[str, Any],
    device: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if features.shape[1] <= CALIBRATION_BINS:
        raise ValueError("Session is shorter than the required 60-second calibration")
    calibration = features[:, :CALIBRATION_BINS]
    mean = calibration.mean(axis=1).astype(np.float32)
    local_std = (calibration.std(axis=1, ddof=0) + 1e-6).astype(np.float32)
    effective_std = np.maximum(local_std, floor).astype(np.float32)
    normalized = ((features - mean[:, None]) / effective_std[:, None]).astype(np.float32)
    valid_bins = np.asarray(selected_bins, dtype=np.int64)
    valid_bins = valid_bins[valid_bins >= CALIBRATION_BINS - 1]
    if valid_bins.size == 0:
        raise ValueError("No held-out test bins remain after calibration")
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32).reshape(2)
    target_std = np.asarray(checkpoint["target_std"], dtype=np.float32).reshape(2)
    normalized_predictions: list[np.ndarray] = []
    for start in range(0, len(valid_bins), INFERENCE_BATCH):
        batch_bins = valid_bins[start : start + INFERENCE_BATCH]
        inputs = rolling_inputs(normalized, batch_bins)
        normalized_predictions.append(model_predict(model, inputs, device)[:, -1])
    prediction = np.concatenate(normalized_predictions) * target_std + target_mean
    stats = {
        "mean": mean,
        "local_std": local_std,
        "effective_std": effective_std,
        "normalized": normalized,
    }
    return valid_bins, velocity[valid_bins], prediction.astype(np.float32), stats


def golden_vectors(
    model: Any,
    normalized: np.ndarray,
    checkpoint: dict[str, Any],
    device: Any,
) -> dict[str, np.ndarray]:
    end_bins = np.arange(
        CALIBRATION_BINS - 1,
        CALIBRATION_BINS - 1 + GOLDEN_WINDOW_COUNT,
        dtype=np.int64,
    )
    inputs = rolling_inputs(normalized, end_bins)
    normalized_output = model_predict(model, inputs, device)
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32).reshape(2)
    target_std = np.asarray(checkpoint["target_std"], dtype=np.float32).reshape(2)
    return {
        "bin_indices": end_bins,
        "model_inputs": inputs,
        "expected_normalized_sequence": normalized_output,
        "expected_velocity_last": normalized_output[:, -1] * target_std + target_mean,
    }


def synthetic_protocol_check() -> None:
    values = np.arange(PHYSICAL_CHANNELS * 3, dtype=np.float32).reshape(96, 3)
    features = continuous_features(values)
    if not np.array_equal(features[:96], values):
        raise ValueError("Raw feature order changed")
    if not np.array_equal(features[96:, 0], values[:, 0]):
        raise ValueError("EWMA first-bin initialization changed")
    expected = EWMA_ALPHA * values[:, 1] + (1.0 - EWMA_ALPHA) * values[:, 0]
    if not np.allclose(features[96:, 1], expected, rtol=0, atol=1e-6):
        raise ValueError("EWMA recurrence changed")
    if not math.isclose(BIN_SECONDS * CALIBRATION_BINS, 60.0):
        raise ValueError("Calibration is not exactly 60 seconds")
    if not math.isclose(BIN_SECONDS * WINDOW_BINS, 2.0):
        raise ValueError("Window is not exactly 2 seconds")


def process_session(name: str, device: Any, overwrite: bool) -> dict[str, Any]:
    import torch

    checkpoint_path = MODEL_ROOT / name / "checkpoint.pt"
    candidate_path = MODEL_ROOT / name / "deployment_candidate.pt"
    constants_path = MODEL_ROOT / name / "deployment_constants.npz"
    replay_path = MODEL_ROOT / name / "deployment_replay.json"
    golden_path = MODEL_ROOT / name / "deployment_golden_vectors.npz"
    outputs = (candidate_path, constants_path, replay_path, golden_path)
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{name}: outputs already exist; use --overwrite: {existing}")

    model, checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint["session"] != name:
        raise ValueError(f"{checkpoint_path}: session identity mismatch")
    spec = SESSION_BY_NAME[name]
    data = load_session(spec)
    counts_all, velocity = aggregate_40ms(data)
    bounds = binned_reach_bounds(data)
    train_reaches, validation_reaches, test_reaches = split_fold(
        make_fold_indices(eligible_reaches(data)), 0
    )
    channels = np.asarray(checkpoint["selected_channel_indices"], dtype=np.int64)
    if channels.max() >= counts_all.shape[0]:
        raise ValueError(f"{name}: mapping exceeds {counts_all.shape[0]} source channels")
    counts = counts_all[channels].astype(np.float32)
    promoted = torch.load(
        MODEL_ROOT / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    promoted_floor = np.asarray(
        promoted["feature_std_floor"], dtype=np.float32
    ).reshape(FEATURES)
    floor, floor_metadata = fit_training_floor(
        counts, bounds, train_reaches, promoted_floor
    )

    reference_bins, reference_target, reference_prediction = phase7_reference(
        model, counts, velocity, bounds, test_reaches, checkpoint, device
    )
    reference_metrics_all = metric_values(reference_target, reference_prediction)
    post_mask = reference_bins >= CALIBRATION_BINS - 1
    reference_metrics_post = metric_values(
        reference_target[post_mask], reference_prediction[post_mask]
    )

    features = continuous_features(counts)
    replay_bins, replay_target, replay_prediction, stats = firmware_replay(
        model,
        features,
        velocity,
        reference_bins,
        floor,
        checkpoint,
        device,
    )
    if not np.array_equal(replay_bins, reference_bins[post_mask]):
        raise ValueError("Firmware replay and Phase 7 reference bins differ")
    if not np.array_equal(replay_target, reference_target[post_mask]):
        raise ValueError("Firmware replay and Phase 7 reference targets differ")
    replay_metrics = metric_values(replay_target, replay_prediction)
    delta = {
        key: replay_metrics[key] - reference_metrics_post[key]
        for key in ("r2_x", "r2_y", "r2_mean")
    }

    source_sha256 = sha256_file(checkpoint_path)
    candidate = dict(checkpoint)
    candidate.update(
        {
            "status": "deployment_candidate_replay_complete",
            "source_checkpoint_status": checkpoint.get("status"),
            "source_checkpoint_sha256": source_sha256,
            "deployment_schema_version": 1,
            "model_id": name,
            "source_channel_count": int(counts_all.shape[0]),
            "physical_channel_count": PHYSICAL_CHANNELS,
            "input_feature_count": FEATURES,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "feature_std_floor": floor,
            "deployment_policy": {
                "bin_ms": int(BIN_SECONDS * 1_000),
                "calibration_bins": CALIBRATION_BINS,
                "window_bins": WINDOW_BINS,
                "ewma_alpha": EWMA_ALPHA,
                "window_order": "oldest_to_newest",
                "output_timestep": WINDOW_BINS - 1,
            },
            "floor_fit": floor_metadata,
            "firmware_style_replay": {
                "held_out_split": "phase7_fold1_test_reaches",
                "test_reaches": int(len(test_reaches)),
                "bins_after_calibration": int(len(replay_bins)),
                "phase7_reference_all": reference_metrics_all,
                "phase7_reference_same_bins": reference_metrics_post,
                "firmware_policy_same_bins": replay_metrics,
                "firmware_minus_reference_r2": delta,
                "promotion_decision": "manual_review_required",
            },
        }
    )
    save_torch_atomic(candidate, candidate_path)
    save_npz_atomic(
        constants_path,
        schema_version=np.asarray("session_deployment_constants_v1"),
        model_id=np.asarray(name),
        source_channel_count=np.asarray(counts_all.shape[0], dtype=np.uint16),
        selected_channel_indices=channels.astype(np.uint16),
        feature_std_floor=floor,
        target_mean=np.asarray(checkpoint["target_mean"], dtype=np.float32),
        target_std=np.asarray(checkpoint["target_std"], dtype=np.float32),
        source_checkpoint_sha256=np.asarray(source_sha256),
    )
    golden = golden_vectors(model, stats["normalized"], checkpoint, device)
    save_npz_atomic(
        golden_path,
        schema_version=np.asarray("session_deployment_golden_v1"),
        model_id=np.asarray(name),
        source_checkpoint_sha256=np.asarray(source_sha256),
        calibration_mean=stats["mean"],
        calibration_local_std=stats["local_std"],
        calibration_effective_std=stats["effective_std"],
        **golden,
    )
    replay_record = {
        "schema_version": 1,
        "phase": PHASE_NAME,
        "created_at_utc": utc_now(),
        "model_id": name,
        "source_checkpoint": checkpoint_path.name,
        "source_checkpoint_sha256": source_sha256,
        "candidate_checkpoint": candidate_path.name,
        "candidate_checkpoint_sha256": sha256_file(candidate_path),
        "constants": constants_path.name,
        "constants_sha256": sha256_file(constants_path),
        "golden_vectors": golden_path.name,
        "golden_vectors_sha256": sha256_file(golden_path),
        "source_channel_count": int(counts_all.shape[0]),
        "selected_channel_indices": channels.tolist(),
        "fold": int(checkpoint["fold"]),
        "reach_counts": {
            "train": int(len(train_reaches)),
            "validation": int(len(validation_reaches)),
            "test": int(len(test_reaches)),
        },
        "floor_fit": floor_metadata,
        "replay": candidate["firmware_style_replay"],
        "candidate_status": candidate["status"],
        "promotion_decision": "manual_review_required",
        "test_targets_used_for_weight_or_floor_fitting": False,
    }
    write_json_atomic(replay_record, replay_path)
    del model, checkpoint, data, counts_all, counts, features
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return replay_record


def update_session_manifest(records: list[dict[str, Any]]) -> None:
    with SESSION_MANIFEST.open("r", encoding="utf-8") as source:
        manifest = json.load(source)
    by_name = {record["model_id"]: record for record in records}
    for session in manifest["sessions"]:
        name = session["session"]
        if name not in by_name:
            continue
        record = by_name[name]
        session["deployment_candidate"] = {
            "status": record["candidate_status"],
            "checkpoint": f"{name}/{record['candidate_checkpoint']}",
            "sha256": record["candidate_checkpoint_sha256"],
            "constants": f"{name}/{record['constants']}",
            "constants_sha256": record["constants_sha256"],
            "golden_vectors": f"{name}/{record['golden_vectors']}",
            "golden_vectors_sha256": record["golden_vectors_sha256"],
            "firmware_replay_r2_mean": record["replay"]["firmware_policy_same_bins"][
                "r2_mean"
            ],
            "phase7_same_bins_r2_mean": record["replay"]["phase7_reference_same_bins"][
                "r2_mean"
            ],
            "promotion_decision": record["promotion_decision"],
        }
    manifest["deployment_validation"] = {
        "phase": PHASE_NAME,
        "schema_version": 1,
        "policy": "continuous EWMA, 60 s calibration, rolling 50-bin window, timestep 49",
        "promotion": "manual review required; this script never silently promotes weights",
    }
    write_json_atomic(manifest, SESSION_MANIFEST)


def main() -> None:
    import torch

    args = parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    synthetic_protocol_check()
    device = select_device(args.device)
    for name in args.sessions:
        load_checkpoint(MODEL_ROOT / name / "checkpoint.pt", device)
    if args.protocol_check_only:
        print("Phase 10 protocol check passed")
        print(f"sessions={len(args.sessions)} parameters={EXPECTED_PARAMETER_COUNT:,}")
        print("floor=train-only 60 s blocks; replay=firmware rolling policy")
        return

    records = []
    for name in args.sessions:
        print(f"\n=== {name} ===", flush=True)
        record = process_session(name, device, args.overwrite)
        records.append(record)
        replay = record["replay"]
        print(
            "Phase7 same-bin R2="
            f"{replay['phase7_reference_same_bins']['r2_mean']:.4f} | "
            "firmware-policy R2="
            f"{replay['firmware_policy_same_bins']['r2_mean']:.4f} | "
            "delta="
            f"{replay['firmware_minus_reference_r2']['r2_mean']:+.4f}",
            flush=True,
        )
    update_session_manifest(records)
    write_json_atomic(
        {
            "schema_version": 1,
            "phase": PHASE_NAME,
            "created_at_utc": utc_now(),
            "device": device.type,
            "sessions": records,
            "promotion_decision": "manual_review_required",
        },
        RESULT_MANIFEST,
    )
    print(f"\nWrote {RESULT_MANIFEST}")


if __name__ == "__main__":
    main()
