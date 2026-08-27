#!/usr/bin/env python3
"""Phase 13 round 3: deployment-aligned five-fold Midsize retraining.

Each fold keeps the Phase-7 reach split and train-only channel selection, but
changes the learning examples to the firmware policy:

* continuous session-level causal EWMA (never reset at a reach boundary),
* a rolling 50-bin past-only window ending at the predicted bin,
* mean/std estimated from the unlabeled first seven minutes, and
* loss applied only to the final timestep used by firmware.

Validation loss selects the checkpoint.  Test targets are not evaluated until
the selected checkpoint has been frozen and saved.  The final report compares
the retrained model with three matched Phase-7 baselines on identical test bins.
Nothing produced here is promoted into ``models/`` automatically.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

HERE = Path(__file__).resolve().parent
INDY_ROOT = HERE.parents[1]
WORKSPACE_ROOT = INDY_ROOT.parent.parent
GUI_ROOT = WORKSPACE_ROOT / "BCI-STM32-Plot" / "data" / "ai_device_sessions"
PHASE7_SCRIPT = (
    INDY_ROOT / "history" / "experiments" / "phase7" / "phase7_ann_vs_snn_fivefold.py"
)
PHASE7_RESULT = (
    INDY_ROOT / "history" / "results" / "indy" / "phase7_ann_vs_snn_fivefold"
)
RESULT_BASE = HERE / "results" / "rolling_retrain"

PHASE_NAME: Final = "phase13_round3_rolling_retrain"
CALIBRATION_MINUTES: Final = 7.0
BIN_SECONDS: Final = 0.04
CALIBRATION_BINS: Final = round(CALIBRATION_MINUTES * 60.0 / BIN_SECONDS)
WINDOW_BINS: Final = 50
FEATURES: Final = 192
FOLD_COUNT: Final = 5
FOLD_SEED: Final = 43
FLOOR_BLOCK_BINS: Final = 1_500
FLOOR_PERCENTILE: Final = 10.0
DEFAULT_EPOCHS: Final = 20
DEFAULT_BATCH_SIZE: Final = 128
DEFAULT_PATIENCE: Final = 6
DEFAULT_WEIGHT_DECAY: Final = 0.025
DEFAULT_GRADIENT_CLIP: Final = 1.0


def import_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PHASE7 = import_file("phase13_round3_phase7", PHASE7_SCRIPT)
SESSIONS: Final = tuple(spec.name for spec in PHASE7.SESSIONS)


@dataclass
class FoldData:
    channels: np.ndarray
    normalized_features: np.ndarray
    calibration_mean: np.ndarray
    calibration_local_std: np.ndarray
    calibration_effective_std: np.ndarray
    feature_std_floor: np.ndarray
    floor_metadata: dict[str, Any]
    velocity: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    train_bins: np.ndarray
    validation_bins: np.ndarray
    test_bins: np.ndarray
    train_reaches: np.ndarray
    validation_reaches: np.ndarray
    test_reaches: np.ndarray
    counts: np.ndarray
    bounds: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", action="append", choices=SESSIONS)
    parser.add_argument(
        "--fold",
        action="append",
        type=int,
        choices=range(1, FOLD_COUNT + 1),
        help="One-based fold; repeat to select several. Default: all five.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument(
        "--init",
        choices=("phase7", "scratch"),
        default="phase7",
        help="Warm-start the matching fold or train fresh weights.",
    )
    parser.add_argument(
        "--train-scope",
        choices=("all", "gru-head"),
        default="all",
        help="Tune all weights or freeze the TCN and tune only GRU/head weights.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Default: 3e-4 for Phase-7 warm-start, 9e-4 from scratch.",
    )
    parser.add_argument(
        "--encoder-lr-scale",
        type=float,
        help="All-scope encoder/TCN LR multiplier; default 0.25 warm-start, 1.0 scratch.",
    )
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--gradient-clip", type=float, default=DEFAULT_GRADIENT_CLIP)
    parser.add_argument(
        "--output-name",
        help="Result subdirectory. Default is derived from init and train scope.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate dependencies, model shape, and checkpoint availability only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare every selected fold and print bin counts without training.",
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


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint_atomic(path: Path, payload: dict[str, Any]) -> None:
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
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        return torch.device("mps")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int, *, mps: bool) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=mps)


def continuous_features(counts: np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != 96:
        raise ValueError(f"Expected 96-by-time counts, received {values.shape}")
    ewma = values.copy()
    for index in range(1, values.shape[1]):
        ewma[:, index] = (
            PHASE7.EWMA_ALPHA * values[:, index]
            + (1.0 - PHASE7.EWMA_ALPHA) * ewma[:, index - 1]
        )
    return np.concatenate((values, ewma), axis=0).astype(np.float32)


def verify_gui_arrays(session: str, counts: np.ndarray, velocity: np.ndarray) -> None:
    path = GUI_ROOT / f"{session}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing GUI deployment dataset: {path}")
    with np.load(path, allow_pickle=False) as archive:
        gui_counts = np.asarray(archive["counts"])
        gui_velocity = np.asarray(archive["velocity"], dtype=np.float32)
    if not np.array_equal(counts, gui_counts):
        raise ValueError(f"{session}: training and GUI 40-ms count arrays differ")
    if not np.allclose(velocity, gui_velocity, rtol=0, atol=1e-6):
        maximum = float(np.max(np.abs(velocity - gui_velocity)))
        raise ValueError(f"{session}: training/GUI velocity difference {maximum}")


def fit_training_floor(
    counts: np.ndarray,
    bounds: np.ndarray,
    train_reaches: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a guard floor from training-reach inputs only.

    Sixty-second blocks are retained for the floor even though the actual
    calibration is seven minutes.  This produces enough independent blocks in
    short sessions; the seven-minute prefix still determines the deployed mean
    and local standard deviation.
    """

    ordered = sorted(
        (int(value) for value in train_reaches), key=lambda i: bounds[i, 0]
    )
    segments = [counts[:, bounds[index, 0] : bounds[index, 1]] for index in ordered]
    timeline = np.concatenate(
        [segment for segment in segments if segment.shape[1]], axis=1
    )
    features = continuous_features(timeline)
    starts = list(range(0, features.shape[1] - FLOOR_BLOCK_BINS + 1, FLOOR_BLOCK_BINS))
    if not starts:
        raise ValueError("Training reaches contain less than one 60-second floor block")
    block_stds = np.stack(
        [
            features[:, start : start + FLOOR_BLOCK_BINS].std(axis=1, ddof=0) + 1e-6
            for start in starts
        ]
    ).astype(np.float32)
    fallback = np.maximum(features.std(axis=1, ddof=0) + 1e-6, 1e-3).astype(np.float32)
    floor = np.empty(FEATURES, dtype=np.float32)
    fallback_features: list[int] = []
    for feature in range(FEATURES):
        valid = block_stds[:, feature][block_stds[:, feature] > 1e-4]
        if valid.size:
            floor[feature] = np.percentile(valid, FLOOR_PERCENTILE)
        else:
            floor[feature] = fallback[feature]
            fallback_features.append(feature)
    return floor, {
        "method": "train_reaches_chronological_60s_blocks",
        "block_bins": FLOOR_BLOCK_BINS,
        "block_seconds": FLOOR_BLOCK_BINS * BIN_SECONDS,
        "block_count": len(starts),
        "percentile": FLOOR_PERCENTILE,
        "training_reaches": len(ordered),
        "training_bins": int(timeline.shape[1]),
        "fallback": "training_timeline_std_min_1e-3",
        "fallback_feature_indices": fallback_features,
        "test_or_validation_reaches_used": False,
        "minimum": float(floor.min()),
        "median": float(np.median(floor)),
        "maximum": float(floor.max()),
    }


def bins_for_reaches(
    bounds: np.ndarray, reaches: np.ndarray, *, minimum_bin: int
) -> np.ndarray:
    pieces = []
    for reach in sorted((int(value) for value in reaches), key=lambda i: bounds[i, 0]):
        start, stop = (int(value) for value in bounds[reach])
        start = max(start, minimum_bin)
        if start < stop:
            pieces.append(np.arange(start, stop, dtype=np.int64))
    if not pieces:
        raise ValueError("No split bins remain after calibration")
    output = np.concatenate(pieces)
    if np.any(np.diff(output) <= 0):
        raise ValueError("Split bins must be unique and chronological")
    return output


def prepare_fold(data: Any, fold: int) -> FoldData:
    eligible = PHASE7.eligible_reaches(data)
    train_reaches, validation_reaches, test_reaches = PHASE7.split_fold(
        PHASE7.make_fold_indices(eligible), fold
    )
    counts_all, velocity = PHASE7.aggregate_40ms(data)
    bounds = PHASE7.binned_reach_bounds(data)
    channels = PHASE7.select_channels(data, counts_all, bounds, train_reaches)
    counts = counts_all[channels].astype(np.float32)
    features = continuous_features(counts)
    if features.shape[1] <= CALIBRATION_BINS:
        raise ValueError(f"{data.spec.name}: session is shorter than seven minutes")
    floor, floor_metadata = fit_training_floor(counts, bounds, train_reaches)
    calibration = features[:, :CALIBRATION_BINS]
    calibration_mean = calibration.mean(axis=1).astype(np.float32)
    calibration_local_std = (calibration.std(axis=1, ddof=0) + 1e-6).astype(np.float32)
    calibration_effective_std = np.maximum(calibration_local_std, floor).astype(
        np.float32
    )
    normalized = (
        (features - calibration_mean[:, None]) / calibration_effective_std[:, None]
    ).astype(np.float32)
    minimum_bin = max(CALIBRATION_BINS - 1, WINDOW_BINS - 1)
    train_bins = bins_for_reaches(bounds, train_reaches, minimum_bin=minimum_bin)
    validation_bins = bins_for_reaches(
        bounds, validation_reaches, minimum_bin=minimum_bin
    )
    test_bins = bins_for_reaches(bounds, test_reaches, minimum_bin=minimum_bin)
    target_mean = velocity[train_bins].mean(axis=0).astype(np.float32)
    target_std = (velocity[train_bins].std(axis=0, ddof=0) + 1e-6).astype(np.float32)
    return FoldData(
        channels=channels,
        normalized_features=normalized,
        calibration_mean=calibration_mean,
        calibration_local_std=calibration_local_std,
        calibration_effective_std=calibration_effective_std,
        feature_std_floor=floor,
        floor_metadata=floor_metadata,
        velocity=velocity,
        target_mean=target_mean,
        target_std=target_std,
        train_bins=train_bins,
        validation_bins=validation_bins,
        test_bins=test_bins,
        train_reaches=train_reaches,
        validation_reaches=validation_reaches,
        test_reaches=test_reaches,
        counts=counts,
        bounds=bounds,
    )


def rolling_batch(features: np.ndarray, end_bins: np.ndarray) -> np.ndarray:
    offsets = np.arange(WINDOW_BINS, dtype=np.int64) - (WINDOW_BINS - 1)
    indices = np.asarray(end_bins, dtype=np.int64)[:, None] + offsets[None, :]
    return np.ascontiguousarray(
        features[:, indices].transpose(1, 0, 2), dtype=np.float32
    )


def predict_last(
    model: Any,
    features: np.ndarray,
    bins: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: Any,
    batch_size: int,
) -> np.ndarray:
    import torch

    model.eval()
    output = []
    with torch.inference_mode():
        for left in range(0, len(bins), batch_size):
            selected = bins[left : left + batch_size]
            inputs = torch.from_numpy(rolling_batch(features, selected)).to(device)
            output.append(model(inputs)[:, -1].cpu().numpy().astype(np.float32))
    normalized = np.concatenate(output)
    return (normalized * target_std + target_mean).astype(np.float32)


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    residual = np.sum((target - prediction) ** 2, axis=0)
    total = np.sum((target - target.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - residual / np.maximum(total, 1e-12)
    return {
        "bins": int(len(target)),
        "mse": float(np.mean((target - prediction) ** 2)),
        "r2_x": float(r2[0]),
        "r2_y": float(r2[1]),
        "r2_mean": float(r2.mean()),
    }


def evaluate_last(
    model: Any,
    features: np.ndarray,
    velocity: np.ndarray,
    bins: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: Any,
    batch_size: int,
) -> dict[str, float | int]:
    prediction = predict_last(
        model, features, bins, target_mean, target_std, device, batch_size
    )
    result = metrics(velocity[bins], prediction)
    normalized_error = (velocity[bins] - prediction) / target_std
    result["normalized_loss"] = float(np.mean(normalized_error**2))
    return result


def phase7_reach_local_reference(
    model: Any,
    checkpoint: dict[str, Any],
    fold_data: FoldData,
    device: Any,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32).reshape(FEATURES, 1)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32).reshape(FEATURES, 1)
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
    target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)
    windows: list[np.ndarray] = []
    lengths: list[int] = []
    bin_parts: list[np.ndarray] = []
    for reach in sorted(
        (int(value) for value in fold_data.test_reaches),
        key=lambda i: fold_data.bounds[i, 0],
    ):
        start, stop = (int(value) for value in fold_data.bounds[reach])
        values = PHASE7.reach_features(fold_data.counts, start, stop)
        normalized = ((values - mean) / std).astype(np.float32)
        for left in range(0, normalized.shape[1], WINDOW_BINS):
            right = min(left + WINDOW_BINS, normalized.shape[1])
            length = right - left
            window = np.zeros((FEATURES, WINDOW_BINS), dtype=np.float32)
            window[:, :length] = normalized[:, left:right]
            windows.append(window)
            lengths.append(length)
            bin_parts.append(np.arange(start + left, start + right, dtype=np.int64))
    stacked = np.stack(windows)
    sequence_predictions = []
    model.eval()
    with torch.inference_mode():
        for left in range(0, len(stacked), batch_size):
            inputs = torch.from_numpy(stacked[left : left + batch_size]).to(device)
            sequence_predictions.append(model(inputs).cpu().numpy().astype(np.float32))
    sequences = np.concatenate(sequence_predictions)
    predictions = [
        sequences[index, :length] * target_std + target_mean
        for index, length in enumerate(lengths)
    ]
    bins = np.concatenate(bin_parts)
    prediction = np.concatenate(predictions).astype(np.float32)
    keep = bins >= CALIBRATION_BINS - 1
    bins = bins[keep]
    prediction = prediction[keep]
    if not np.array_equal(bins, fold_data.test_bins):
        raise ValueError(
            "Phase-7 and rolling references do not use identical test bins"
        )
    return bins, prediction


def source_checkpoint_path(session: str, fold: int) -> Path:
    return PHASE7_RESULT / "checkpoints" / f"{session}_fold{fold + 1}.pt"


def train_fold(
    data: Any,
    fold: int,
    fold_data: FoldData,
    args: argparse.Namespace,
    device: Any,
    run_root: Path,
    learning_rate: float,
    encoder_lr_scale: float,
) -> dict[str, Any]:
    import torch

    seed = FOLD_SEED
    seed_everything(seed, mps=device.type == "mps")
    source_path = source_checkpoint_path(data.spec.name, fold)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    source_channels = np.asarray(source["selected_channel_indices"], dtype=np.int64)
    if not np.array_equal(source_channels, fold_data.channels):
        raise ValueError(f"{data.spec.name} fold {fold + 1}: channel selection changed")
    model = PHASE7.build_model()
    if args.init == "phase7":
        model.load_state_dict(source["model_state"], strict=True)
    model.to(device)

    recurrent_parameters = []
    encoder_parameters = []
    for name, parameter in model.named_parameters():
        recurrent = name.startswith("gru.") or name.startswith("head.")
        parameter.requires_grad = args.train_scope == "all" or recurrent
        if not parameter.requires_grad:
            continue
        (recurrent_parameters if recurrent else encoder_parameters).append(parameter)
    parameter_groups = [
        {"params": recurrent_parameters, "lr": learning_rate, "name": "gru_head"}
    ]
    if encoder_parameters:
        parameter_groups.append(
            {
                "params": encoder_parameters,
                "lr": learning_rate * encoder_lr_scale,
                "name": "encoder_tcn",
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    rng = np.random.default_rng(seed)
    normalized_target = (
        (fold_data.velocity - fold_data.target_mean) / fold_data.target_std
    ).astype(np.float32)
    best_state = None
    best_epoch = 0
    best_validation_loss = math.inf
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    print(
        f"\n=== {data.spec.name} | fold {fold + 1}/{FOLD_COUNT} ===\n"
        f"bins train/validation/test={len(fold_data.train_bins):,}/"
        f"{len(fold_data.validation_bins):,}/{len(fold_data.test_bins):,} | "
        f"init={args.init} | scope={args.train_scope}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(fold_data.train_bins)
        error_sum = 0.0
        value_count = 0
        gradient_sum = 0.0
        gradient_max = 0.0
        batch_count = 0
        for left in range(0, len(order), args.batch_size):
            bins = order[left : left + args.batch_size]
            inputs = torch.from_numpy(
                rolling_batch(fold_data.normalized_features, bins)
            ).to(device)
            targets = torch.from_numpy(normalized_target[bins]).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)[:, -1]
            loss = torch.mean((prediction - targets) ** 2)
            loss.backward()
            gradient = float(
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for group in parameter_groups
                        for parameter in group["params"]
                    ],
                    args.gradient_clip,
                )
            )
            optimizer.step()
            error_sum += float(loss.detach()) * len(bins)
            value_count += len(bins)
            gradient_sum += gradient
            gradient_max = max(gradient_max, gradient)
            batch_count += 1

        validation = evaluate_last(
            model,
            fold_data.normalized_features,
            fold_data.velocity,
            fold_data.validation_bins,
            fold_data.target_mean,
            fold_data.target_std,
            device,
            args.batch_size,
        )
        improved = float(validation["normalized_loss"]) < best_validation_loss
        if improved:
            best_validation_loss = float(validation["normalized_loss"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        row = {
            "session": data.spec.name,
            "subject": data.spec.subject,
            "fold": fold + 1,
            "epoch": epoch,
            "gru_head_lr": float(optimizer.param_groups[0]["lr"]),
            "encoder_tcn_lr": (
                float(optimizer.param_groups[1]["lr"])
                if len(optimizer.param_groups) > 1
                else 0.0
            ),
            "optimization_loss": error_sum / max(value_count, 1),
            "gradient_mean_before_clip": gradient_sum / max(batch_count, 1),
            "gradient_max_before_clip": gradient_max,
            "validation_loss": validation["normalized_loss"],
            "validation_r2": validation["r2_mean"],
            "best": improved,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{args.epochs} | train={row['optimization_loss']:.5f} | "
            f"val={row['validation_loss']:.5f} | val R2={row['validation_r2']:+.4f} | "
            f"grad={row['gradient_mean_before_clip']:.3f}/"
            f"{row['gradient_max_before_clip']:.3f}" + (" *best*" if improved else ""),
            flush=True,
        )
        scheduler.step()
        if args.patience and epochs_without_improvement >= args.patience:
            print(f"early stop after {args.patience} non-improving epochs", flush=True)
            break

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    model.load_state_dict(best_state)
    model.eval()
    checkpoint_path = run_root / "checkpoints" / f"{data.spec.name}_fold{fold + 1}.pt"
    checkpoint = {
        "purpose": PHASE_NAME,
        "status": "experimental_not_promoted",
        "created_at_utc": utc_now(),
        "session": data.spec.name,
        "subject": data.spec.subject,
        "fold": fold + 1,
        "seed": seed,
        "best_epoch": best_epoch,
        "model_state": best_state,
        "selected_channel_indices": fold_data.channels,
        "selected_channel_names": data.channel_names[fold_data.channels],
        "source_channel_count": int(data.spike_presence.shape[0]),
        "physical_channel_count": 96,
        "input_feature_count": FEATURES,
        "parameter_count": PHASE7.parameter_count(),
        "feature_mean": fold_data.calibration_mean[:, None],
        "feature_std": fold_data.calibration_effective_std[:, None],
        "feature_std_floor": fold_data.feature_std_floor,
        "calibration_local_std": fold_data.calibration_local_std,
        "target_mean": fold_data.target_mean,
        "target_std": fold_data.target_std,
        "initialization": args.init,
        "train_scope": args.train_scope,
        "source_phase7_checkpoint": str(source_path.relative_to(INDY_ROOT)),
        "source_phase7_checkpoint_sha256": sha256_file(source_path),
        "selection_policy": "minimum_validation_loss_test_opened_once",
        "test_evaluated_during_training": False,
        "deployment_policy": {
            "bin_ms": round(BIN_SECONDS * 1000),
            "calibration_minutes": CALIBRATION_MINUTES,
            "calibration_bins": CALIBRATION_BINS,
            "window_bins": WINDOW_BINS,
            "window_order": "oldest_to_newest",
            "output_timestep": WINDOW_BINS - 1,
            "ewma_alpha": PHASE7.EWMA_ALPHA,
            "ewma_reset_at_reach": False,
        },
        "floor_fit": fold_data.floor_metadata,
        "reach_counts": {
            "train": int(len(fold_data.train_reaches)),
            "validation": int(len(fold_data.validation_reaches)),
            "test": int(len(fold_data.test_reaches)),
        },
        "bin_counts_after_calibration": {
            "train": int(len(fold_data.train_bins)),
            "validation": int(len(fold_data.validation_bins)),
            "test": int(len(fold_data.test_bins)),
        },
    }
    save_checkpoint_atomic(checkpoint_path, checkpoint)

    # Test labels are opened only after the validation-selected state is frozen.
    retrained_test = evaluate_last(
        model,
        fold_data.normalized_features,
        fold_data.velocity,
        fold_data.test_bins,
        fold_data.target_mean,
        fold_data.target_std,
        device,
        args.batch_size,
    )
    source_model = PHASE7.build_model().to(device)
    source_model.load_state_dict(source["model_state"], strict=True)
    source_model.eval()
    source_target_mean = np.asarray(source["target_mean"], dtype=np.float32)
    source_target_std = np.asarray(source["target_std"], dtype=np.float32)
    _, reach_local_prediction = phase7_reach_local_reference(
        source_model, source, fold_data, device, args.batch_size
    )
    reach_local = metrics(
        fold_data.velocity[fold_data.test_bins], reach_local_prediction
    )
    source_mean = np.asarray(source["feature_mean"], dtype=np.float32).reshape(
        FEATURES, 1
    )
    source_std = np.asarray(source["feature_std"], dtype=np.float32).reshape(
        FEATURES, 1
    )
    continuous_training_norm = (
        (continuous_features(fold_data.counts) - source_mean) / source_std
    ).astype(np.float32)
    source_continuous = evaluate_last(
        source_model,
        continuous_training_norm,
        fold_data.velocity,
        fold_data.test_bins,
        source_target_mean,
        source_target_std,
        device,
        args.batch_size,
    )
    source_calibrated = evaluate_last(
        source_model,
        fold_data.normalized_features,
        fold_data.velocity,
        fold_data.test_bins,
        source_target_mean,
        source_target_std,
        device,
        args.batch_size,
    )
    comparison = {
        "phase7_reach_local_same_bins": reach_local,
        "phase7_weights_continuous_training_norm": source_continuous,
        "phase7_weights_7min_calibration": source_calibrated,
        "retrained_weights_7min_calibration": retrained_test,
        "rolling_only_delta": float(
            source_continuous["r2_mean"] - reach_local["r2_mean"]
        ),
        "seven_minute_calibration_delta_after_rolling": float(
            source_calibrated["r2_mean"] - source_continuous["r2_mean"]
        ),
        "retraining_gain": float(
            retrained_test["r2_mean"] - source_calibrated["r2_mean"]
        ),
        "net_delta_vs_phase7_reach_local": float(
            retrained_test["r2_mean"] - reach_local["r2_mean"]
        ),
        "rolling_gap_recovered_fraction": (
            float(
                (retrained_test["r2_mean"] - source_calibrated["r2_mean"])
                / (reach_local["r2_mean"] - source_continuous["r2_mean"])
            )
            if reach_local["r2_mean"] > source_continuous["r2_mean"]
            else None
        ),
    }
    print(
        f"selected epoch={best_epoch} | old reach-local={reach_local['r2_mean']:+.4f} | "
        f"old rolling+7min={source_calibrated['r2_mean']:+.4f} | "
        f"retrained={retrained_test['r2_mean']:+.4f} | "
        f"gain={comparison['retraining_gain']:+.4f}",
        flush=True,
    )
    return {
        "session": data.spec.name,
        "subject": data.spec.subject,
        "fold": fold + 1,
        "best_epoch": best_epoch,
        "initialization": args.init,
        "train_scope": args.train_scope,
        "learning_rate": learning_rate,
        "encoder_lr_scale": encoder_lr_scale,
        "channels": fold_data.channels.tolist(),
        "bin_counts": checkpoint["bin_counts_after_calibration"],
        "validation": evaluate_last(
            model,
            fold_data.normalized_features,
            fold_data.velocity,
            fold_data.validation_bins,
            fold_data.target_mean,
            fold_data.target_std,
            device,
            args.batch_size,
        ),
        "comparison": comparison,
        "history": history,
        "checkpoint": str(checkpoint_path.relative_to(HERE)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def flatten_fold_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        comparison = result["comparison"]
        rows.append(
            {
                "session": result["session"],
                "subject": result["subject"],
                "fold": result["fold"],
                "best_epoch": result["best_epoch"],
                "phase7_reach_local_r2": comparison["phase7_reach_local_same_bins"][
                    "r2_mean"
                ],
                "phase7_continuous_training_norm_r2": comparison[
                    "phase7_weights_continuous_training_norm"
                ]["r2_mean"],
                "phase7_7min_rolling_r2": comparison["phase7_weights_7min_calibration"][
                    "r2_mean"
                ],
                "retrained_7min_rolling_r2": comparison[
                    "retrained_weights_7min_calibration"
                ]["r2_mean"],
                "rolling_only_delta": comparison["rolling_only_delta"],
                "calibration_delta": comparison[
                    "seven_minute_calibration_delta_after_rolling"
                ],
                "retraining_gain": comparison["retraining_gain"],
                "net_delta_vs_phase7": comparison["net_delta_vs_phase7_reach_local"],
                "rolling_gap_recovered_fraction": comparison[
                    "rolling_gap_recovered_fraction"
                ],
                "test_bins": result["bin_counts"]["test"],
                "checkpoint": result["checkpoint"],
            }
        )
    return rows


def summarize(fold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for session in [*SESSIONS, "overall_fold_macro"]:
        selected = (
            fold_rows
            if session == "overall_fold_macro"
            else [row for row in fold_rows if row["session"] == session]
        )
        if not selected:
            continue
        row: dict[str, Any] = {
            "session": session,
            "subject": (
                "all" if session == "overall_fold_macro" else selected[0]["subject"]
            ),
            "folds": len(selected),
        }
        for field in (
            "phase7_reach_local_r2",
            "phase7_continuous_training_norm_r2",
            "phase7_7min_rolling_r2",
            "retrained_7min_rolling_r2",
            "rolling_only_delta",
            "calibration_delta",
            "retraining_gain",
            "net_delta_vs_phase7",
        ):
            values = np.asarray([item[field] for item in selected], dtype=np.float64)
            row[f"{field}_mean"] = float(values.mean())
            row[f"{field}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return rows


def resolved_hyperparameters(args: argparse.Namespace) -> tuple[float, float]:
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 3e-4 if args.init == "phase7" else PHASE7.LEARNING_RATE
    encoder_lr_scale = args.encoder_lr_scale
    if encoder_lr_scale is None:
        encoder_lr_scale = 0.25 if args.init == "phase7" else 1.0
    if learning_rate <= 0 or encoder_lr_scale < 0:
        raise ValueError(
            "Learning rate must be positive and encoder scale non-negative"
        )
    return float(learning_rate), float(encoder_lr_scale)


def protocol_signature(
    args: argparse.Namespace,
    sessions: list[str],
    folds: list[int],
    device: Any,
    learning_rate: float,
    encoder_lr_scale: float,
) -> str:
    payload = {
        "phase": PHASE_NAME,
        "sessions": sessions,
        "folds": folds,
        "calibration_bins": CALIBRATION_BINS,
        "window_bins": WINDOW_BINS,
        "fold_seed": FOLD_SEED,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "initialization": args.init,
        "train_scope": args.train_scope,
        "learning_rate": learning_rate,
        "encoder_lr_scale": encoder_lr_scale,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "device": device.type,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_protocol(selected_sessions: list[str]) -> None:
    import torch

    if CALIBRATION_BINS != 10_500 or WINDOW_BINS != 50:
        raise ValueError("Seven-minute/50-bin protocol constants changed")
    model = PHASE7.build_model()
    values = torch.zeros(2, FEATURES, WINDOW_BINS)
    if model(values).shape != (2, WINDOW_BINS, 2):
        raise ValueError("Unexpected model output shape")
    if PHASE7.parameter_count() != 86_978:
        raise ValueError("Unexpected Midsize parameter count")
    for session in selected_sessions:
        if not (GUI_ROOT / f"{session}.npz").is_file():
            raise FileNotFoundError(GUI_ROOT / f"{session}.npz")
        for fold in range(FOLD_COUNT):
            path = source_checkpoint_path(session, fold)
            if not path.is_file():
                raise FileNotFoundError(path)


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if min(args.epochs, args.batch_size, args.threads) <= 0 or args.patience < 0:
        raise ValueError("Epochs, batch size, and threads must be positive")
    selected_sessions = list(dict.fromkeys(args.session or SESSIONS))
    selected_folds = sorted(set((value - 1) for value in (args.fold or range(1, 6))))
    learning_rate, encoder_lr_scale = resolved_hyperparameters(args)
    validate_protocol(selected_sessions)
    if args.validate_only:
        print("=== Phase 13 round-3 validation passed ===")
        print(
            f"sessions={len(selected_sessions)} | folds={len(selected_folds)} | "
            f"fits={len(selected_sessions) * len(selected_folds)}"
        )
        print(
            f"calibration={CALIBRATION_MINUTES:g} min/{CALIBRATION_BINS} bins | "
            f"rolling window={WINDOW_BINS} bins | model={PHASE7.parameter_count():,} params"
        )
        return

    import torch

    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    output_name = args.output_name
    if output_name is None:
        output_name = (
            "final_30fold"
            if args.init == "phase7" and args.train_scope == "all"
            else f"{args.init}_{args.train_scope.replace('-', '_')}"
        )
    run_root = RESULT_BASE / output_name
    # Session-input caches are protocol-independent and shared by all variants.
    # Keeping them outside run_root means a dry run never makes the later real
    # run look like an existing/partially completed experiment.
    PHASE7.CACHE_DIR = RESULT_BASE / ".cache"
    signature = protocol_signature(
        args,
        selected_sessions,
        selected_folds,
        device,
        learning_rate,
        encoder_lr_scale,
    )
    state_path = run_root / ".state.json"
    if (
        run_root.exists()
        and not args.resume
        and not args.overwrite
        and not args.dry_run
    ):
        raise FileExistsError(f"Output exists: {run_root}; use --resume or --overwrite")
    if args.overwrite and run_root.exists():
        shutil.rmtree(run_root)
    state: dict[str, Any] = {
        "signature": signature,
        "created_at_utc": utc_now(),
        "completed": {},
    }
    if args.resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"Cannot resume without {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("signature") != signature:
            raise ValueError("Resume state does not match the current protocol")
    elif not args.dry_run:
        write_json_atomic(state_path, state)

    print("=== Phase 13 round 3 · deployment-aligned five-fold retraining ===")
    print(
        f"sessions={len(selected_sessions)} | folds={len(selected_folds)} | "
        f"fits={len(selected_sessions) * len(selected_folds)} | device={device.type}"
    )
    print(
        f"init={args.init} | scope={args.train_scope} | GRU/head LR={learning_rate:g} | "
        f"encoder LR scale={encoder_lr_scale:g}"
    )
    print(
        "test policy=identical post-calibration bins; opened only after validation checkpoint save"
    )

    for session in selected_sessions:
        spec = PHASE7.SESSION_BY_NAME[session]
        PHASE7.validate_source(spec, checksum=False)
        data = PHASE7.load_session(spec)
        counts_all, velocity = PHASE7.aggregate_40ms(data)
        verify_gui_arrays(session, counts_all, velocity)
        print(
            f"\nloaded {session}: bins={counts_all.shape[1]:,} | "
            f"eligible reaches={len(PHASE7.eligible_reaches(data))}",
            flush=True,
        )
        for fold in selected_folds:
            key = f"{session}|fold{fold + 1}"
            if key in state["completed"]:
                print(f"resume: keep completed {key}")
                continue
            fold_data = prepare_fold(data, fold)
            print(
                f"prepared {key}: train/validation/test bins="
                f"{len(fold_data.train_bins):,}/{len(fold_data.validation_bins):,}/"
                f"{len(fold_data.test_bins):,}"
            )
            if args.dry_run:
                continue
            result = train_fold(
                data,
                fold,
                fold_data,
                args,
                device,
                run_root,
                learning_rate,
                encoder_lr_scale,
            )
            state["completed"][key] = result
            write_json_atomic(state_path, state)

    if args.dry_run:
        print("=== Dry run complete; no checkpoints or metrics were written ===")
        return
    ordered_keys = [
        f"{session}|fold{fold + 1}"
        for session in selected_sessions
        for fold in selected_folds
    ]
    results = [state["completed"][key] for key in ordered_keys]
    fold_rows = flatten_fold_rows(results)
    summary = summarize(fold_rows)
    epoch_rows = [epoch for result in results for epoch in result["history"]]
    write_csv(run_root / "phase13_round3_folds.csv", fold_rows)
    write_csv(run_root / "phase13_round3_summary.csv", summary)
    write_csv(run_root / "phase13_round3_epochs.csv", epoch_rows)
    metrics_payload = {
        "phase": PHASE_NAME,
        "status": "complete",
        "completed_at_utc": utc_now(),
        "protocol_signature": signature,
        "sessions": selected_sessions,
        "folds": [value + 1 for value in selected_folds],
        "protocol": {
            "calibration_minutes": CALIBRATION_MINUTES,
            "calibration_bins": CALIBRATION_BINS,
            "calibration_uses_labels": False,
            "calibration_source": "chronological_session_prefix",
            "continuous_session_ewma": True,
            "rolling_window_bins": WINDOW_BINS,
            "loss_timestep": WINDOW_BINS - 1,
            "test_opened_after_validation_checkpoint_save": True,
        },
        "training": {
            "initialization": args.init,
            "train_scope": args.train_scope,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate_gru_head": learning_rate,
            "encoder_lr_scale": encoder_lr_scale,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "device": device.type,
        },
        "reporting_caveat": (
            "Five-fold test means are unbiased with respect to checkpoint selection; "
            "the seven-minute input prefix is an unlabeled deployment calibration and "
            "may contain reaches assigned to any fold."
        ),
        "results": results,
        "summary": summary,
    }
    write_json_atomic(run_root / "phase13_round3_metrics.json", metrics_payload)
    print("\n=== Phase 13 round 3 complete ===")
    for row in summary:
        print(
            f"{row['session']:<24} old7={row['phase7_7min_rolling_r2_mean']:+.4f} | "
            f"new7={row['retrained_7min_rolling_r2_mean']:+.4f} | "
            f"gain={row['retraining_gain_mean']:+.4f} | "
            f"vs reach-local={row['net_delta_vs_phase7_mean']:+.4f}"
        )
    print(f"results: {run_root}")


if __name__ == "__main__":
    main()
