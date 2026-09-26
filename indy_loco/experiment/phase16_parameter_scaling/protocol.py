"""Frozen Phase-13 preprocessing and fitting core for architecture-only scaling.

See protocol_lock.json for source hashes and exact copied function identities.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np

from . import session_data as PHASE7

GUI_ROOT = (
    Path(__file__).resolve().parents[4]
    / "BCI-STM32-Plot"
    / "data"
    / "ai_device_sessions"
)


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


def fit_model(
    model, data, fold, fold_data, args, device, learning_rate, encoder_lr_scale
):
    """Unchanged Phase-13 optimizer, sampling, loss, scheduler and selection loop."""
    import torch

    seed = FOLD_SEED
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
    return best_state, best_epoch, history, best_validation_loss
