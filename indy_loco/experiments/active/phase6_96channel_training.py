#!/usr/bin/env python3
"""Train one strictly causal 96-channel Indy decoder for 20 epochs.

Phase 6 changes only the neural channel count relative to the confirmed Phase 5
baseline.  It keeps the 64/64 TCN+GRU architecture, all 29 chronological train
sessions, session-balanced sampling, and the Phase 5 winning optimization
settings (learning rate 9e-4, weight decay 0.025, dropout 0.10).

The 96 physical channels are used directly; no channel ranking is fitted.
Each session uses its first 60 seconds for causal feature calibration.  Only
train sessions update weights or fit cross-session statistics.  The four
December sessions are inference-only and select the checkpoint.  January is
registered as the locked test split and is never loaded.

The runner uses NVIDIA CUDA when available and otherwise falls back to CPU.
Apple MPS is never selected because previous MPS training produced invalid
gradients for this causal TCN+GRU graph.  It never overwrites retained 32- or
64-channel checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Must be set before importing torch through the model modules. It is required
# for deterministic CUDA matrix multiplication on supported CUDA versions.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from indy_loco.models.indy_32ch.features import multiscale_counts  # noqa: E402
from indy_loco.models.indy_32ch.input_pipeline import (  # noqa: E402
    apply_feature_stats,
    fit_feature_stats,
    load_model_data,
    load_session_manifest,
    processed_session_path,
    window_arrays,
)
from indy_loco.models.indy_32ch.model import build_net, causal_config, r2  # noqa: E402
from indy_loco.models.indy_32ch.sampling import (  # noqa: E402
    draw_session_balanced_indices,
)

PHASE_NAME = "phase6_96channel_training"
RESULT_DIR = PROJECT_ROOT / "results" / PHASE_NAME
METRICS_PATH = RESULT_DIR / f"{PHASE_NAME}_metrics.json"
HISTORY_PATH = RESULT_DIR / f"{PHASE_NAME}_epochs.csv"
FIGURE_PATH = RESULT_DIR / f"{PHASE_NAME}_figure.png"
CHECKPOINT_PATH = RESULT_DIR / "phase6_96channel_checkpoint.pt"

EXPECTED_SPLITS = {"train": 29, "validation": 4, "test": 4}
CHANNEL_COUNT = 96
CHANNELS = np.arange(CHANNEL_COUNT, dtype=np.int64)
ALPHAS = (1.0, 0.1)
INPUT_FEATURES = CHANNEL_COUNT * len(ALPHAS)
BIN_SECONDS = 0.040
OBSERVATION_SECONDS = 60
OBSERVATION_BINS = int(OBSERVATION_SECONDS / BIN_SECONDS)
WINDOW_BINS = 50
TARGET_AXES = (0, 1)
STD_FLOOR_PERCENTILE = 10.0

SEED = 43
TRAIN_EPOCHS = 20
LEARNING_RATE = 9e-4
WEIGHT_DECAY = 0.025
DROPOUT = 0.10
BATCH_SIZE = 32
GRADIENT_CLIP = 1.0
ARCHITECTURE = {"F": 64, "H": 64, "L": 1, "dils": [1, 2, 4, 8]}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint_atomic(payload: Mapping[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def choose_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested, but this PyTorch installation "
                "cannot access an NVIDIA CUDA device. Use --device cpu or auto."
            )
        return torch.device("cuda")
    if requested != "cpu":
        raise ValueError(
            "Phase 6 supports auto, cpu, and cuda only. Apple MPS is disabled "
            "because it previously produced invalid backward gradients."
        )
    return torch.device("cpu")


def seed_everything(seed: int = SEED) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def model_config() -> dict[str, Any]:
    config = causal_config(n_out=2)
    config.update(
        {
            **ARCHITECTURE,
            "bidir": False,
            "dropout": DROPOUT,
            "lr": LEARNING_RATE,
            "wd": WEIGHT_DECAY,
            "epochs": TRAIN_EPOCHS,
            "bs": BATCH_SIZE,
            "noise": 0.0,
            "chdrop": 0.0,
            "cosine": True,
            "act": "relu",
            "gradient_clip": GRADIENT_CLIP,
        }
    )
    return config


def validate_protocol(*, allow_overwrite: bool) -> dict[str, Any]:
    manifest = load_session_manifest()
    split = manifest["chronological_split"]
    split_counts = {name: len(split[name]) for name in EXPECTED_SPLITS}
    if split_counts != EXPECTED_SPLITS:
        raise ValueError(
            f"Expected chronological split {EXPECTED_SPLITS}, found {split_counts}"
        )

    train_names = list(split["train"])
    validation_names = list(split["validation"])
    test_names = list(split["test"])
    if any(name.startswith("indy_201701") for name in train_names + validation_names):
        raise ValueError("January leaked into train or validation")
    if any(not name.startswith("indy_201701") for name in test_names):
        raise ValueError("The locked test registry is not the expected January split")
    for name in train_names + validation_names:
        if not processed_session_path(name).exists():
            raise FileNotFoundError(f"Missing processed session: {name}")

    existing = [
        path
        for path in (CHECKPOINT_PATH, METRICS_PATH, HISTORY_PATH, FIGURE_PATH)
        if path.exists()
    ]
    if existing and not allow_overwrite:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing Phase 6 outputs. Use --overwrite "
            f"only for an intentional rerun: {formatted}"
        )

    parameter_count = sum(
        parameter.numel()
        for parameter in build_net(model_config(), INPUT_FEATURES).parameters()
    )
    return {
        "train_names": train_names,
        "validation_names": validation_names,
        "test_names": test_names,
        "parameter_count": parameter_count,
    }


def fit_feature_std_floor(
    loaded: dict[str, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    session_stds = []
    for counts, _ in loaded.values():
        features = multiscale_counts(counts[CHANNELS], ALPHAS)
        _, std = fit_feature_stats(features, observation_bins=OBSERVATION_BINS)
        session_stds.append(std[:, 0])
    scales = np.stack(session_stds)
    floors = np.empty(scales.shape[1], dtype=np.float32)
    for feature in range(scales.shape[1]):
        valid = scales[:, feature][scales[:, feature] > 1e-4]
        if valid.size == 0:
            raise ValueError(f"Feature {feature} is silent in every train prefix")
        floors[feature] = np.percentile(valid, STD_FLOOR_PERCENTILE)
    return floors[:, None]


def prepare_session(
    data: tuple[np.ndarray, np.ndarray],
    feature_std_floor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    counts, velocity = data
    if counts.shape[0] != CHANNEL_COUNT:
        raise ValueError(
            f"Phase 6 requires exactly {CHANNEL_COUNT} physical channels; "
            f"found {counts.shape[0]}"
        )
    features = multiscale_counts(counts[CHANNELS], ALPHAS)
    if features.shape[0] != INPUT_FEATURES:
        raise AssertionError(
            f"Expected {INPUT_FEATURES} features, found {features.shape[0]}"
        )
    if features.shape[1] < OBSERVATION_BINS + WINDOW_BINS:
        raise ValueError("Session is too short for warm-up plus one window")
    mean, local_std = fit_feature_stats(features, observation_bins=OBSERVATION_BINS)
    normalized = apply_feature_stats(
        features,
        (mean, np.maximum(local_std, feature_std_floor)),
    )
    windows = window_arrays(
        normalized,
        velocity,
        TARGET_AXES,
        window_bins=WINDOW_BINS,
        start_bin=OBSERVATION_BINS,
    )
    return (
        np.stack([window["e"] for window in windows]).astype(np.float32),
        np.stack([window["vel"] for window in windows]).astype(np.float32),
    )


def predict(net, x: np.ndarray, device) -> np.ndarray:
    import torch

    predictions = []
    net.eval()
    with torch.inference_mode():
        for start in range(0, len(x), BATCH_SIZE):
            batch = torch.from_numpy(x[start : start + BATCH_SIZE]).to(device)
            predictions.append(net(batch).cpu().numpy())
    return np.concatenate(predictions, axis=0).astype(np.float32)


def evaluate(
    net,
    x: np.ndarray,
    y: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device,
) -> dict[str, float | int]:
    prediction = predict(net, x, device) * target_std + target_mean
    score = r2(y.reshape(-1, 2), prediction.reshape(-1, 2))
    return {
        "windows": int(len(x)),
        "loss": float(np.mean(((y - prediction) / target_std) ** 2)),
        "r2_x": float(score[0]),
        "r2_y": float(score[1]),
        "r2_mean": float(score.mean()),
    }


def train(
    *,
    train_names: list[str],
    validation_names: list[str],
    parameter_count: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_y_normalized: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    session_lengths: dict[str, int],
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    validation_by_session: dict[str, tuple[np.ndarray, np.ndarray]],
    feature_std_floor: np.ndarray,
    device,
) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    seed_everything()
    config = model_config()
    net = build_net(config, INPUT_FEATURES).to(device)
    actual_parameters = sum(parameter.numel() for parameter in net.parameters())
    if actual_parameters != parameter_count:
        raise AssertionError(
            f"Parameter count changed: {actual_parameters:,} != {parameter_count:,}"
        )

    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, TRAIN_EPOCHS)
    mse = nn.MSELoss()
    rng = np.random.default_rng(SEED)
    history: list[dict[str, Any]] = []
    best_state = None
    best_epoch = None
    best_validation_loss = float("inf")

    print("\n=== Phase 6: 96-channel causal TCN+GRU ===", flush=True)
    print(
        f"parameters={parameter_count:,} | input={INPUT_FEATURES}x{WINDOW_BINS} | "
        f"epochs={TRAIN_EPOCHS} | seed={SEED} | device={device}",
        flush=True,
    )
    for epoch in range(1, TRAIN_EPOCHS + 1):
        indices, session_draws, month_draws = draw_session_balanced_indices(
            train_names,
            session_lengths,
            rng,
        )
        net.train()
        error_sum = 0.0
        value_count = 0
        gradient_sum = 0.0
        gradient_max = 0.0
        batch_count = 0
        for start in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[start : start + BATCH_SIZE]
            x_batch = torch.from_numpy(train_x[batch_indices]).to(device)
            y_batch = torch.from_numpy(train_y_normalized[batch_indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = mse(net(x_batch), y_batch)
            loss.backward()
            gradient = float(
                torch.nn.utils.clip_grad_norm_(net.parameters(), GRADIENT_CLIP)
            )
            optimizer.step()
            error_sum += float(loss.detach().item()) * y_batch.numel()
            value_count += y_batch.numel()
            gradient_sum += gradient
            gradient_max = max(gradient_max, gradient)
            batch_count += 1

        train_metrics = evaluate(
            net,
            train_x,
            train_y,
            target_mean,
            target_std,
            device,
        )
        validation_metrics = evaluate(
            net,
            validation_x,
            validation_y,
            target_mean,
            target_std,
            device,
        )
        improved = validation_metrics["loss"] < best_validation_loss
        if improved:
            best_validation_loss = float(validation_metrics["loss"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in net.state_dict().items()
            }
        row = {
            "epoch": epoch,
            "learning_rate_schedule": float(optimizer.param_groups[0]["lr"]),
            "optimization_loss": error_sum / max(value_count, 1),
            "gradient_norm_mean_before_clip": gradient_sum / max(batch_count, 1),
            "gradient_norm_max_before_clip": gradient_max,
            "train_loss": float(train_metrics["loss"]),
            "train_r2": float(train_metrics["r2_mean"]),
            "validation_loss": float(validation_metrics["loss"]),
            "validation_r2": float(validation_metrics["r2_mean"]),
            "session_draw_min": int(min(session_draws.values())),
            "session_draw_max": int(max(session_draws.values())),
            "month_draws": month_draws,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{TRAIN_EPOCHS} | "
            f"opt={row['optimization_loss']:.5f} | "
            f"loss train={row['train_loss']:.5f} "
            f"validation={row['validation_loss']:.5f} | "
            f"R2 train={row['train_r2']:+.4f} "
            f"validation={row['validation_r2']:+.4f} | "
            f"lr={row['learning_rate_schedule']:.6g} | "
            f"grad={row['gradient_norm_mean_before_clip']:.3f}/"
            f"{row['gradient_norm_max_before_clip']:.3f}"
            + (" *best*" if improved else ""),
            flush=True,
        )
        scheduler.step()

    if best_state is None or best_epoch is None:
        raise RuntimeError("No checkpoint was selected")
    net.load_state_dict(best_state)
    final_train_metrics = evaluate(
        net,
        train_x,
        train_y,
        target_mean,
        target_std,
        device,
    )
    final_validation_metrics = evaluate(
        net,
        validation_x,
        validation_y,
        target_mean,
        target_std,
        device,
    )
    validation_session_metrics = {
        name: evaluate(
            net,
            validation_by_session[name][0],
            validation_by_session[name][1],
            target_mean,
            target_std,
            device,
        )
        for name in validation_names
    }
    validation_macro_r2 = float(
        np.mean([row["r2_mean"] for row in validation_session_metrics.values()])
    )
    validation_worst_r2 = float(
        min(row["r2_mean"] for row in validation_session_metrics.values())
    )

    checkpoint = {
        "purpose": PHASE_NAME,
        "status": "experiment_candidate_not_promoted",
        "created_at_utc": utc_now(),
        "seed": SEED,
        "training_device": device.type,
        "torch_version": torch.__version__,
        "model_state": best_state,
        "config": config,
        "parameter_count": parameter_count,
        "neural_channel_count": CHANNEL_COUNT,
        "input_feature_count": INPUT_FEATURES,
        "channels": CHANNELS.tolist(),
        "channel_selection": "all_96_physical_channels_no_fitted_ranking",
        "feature_std_floor": feature_std_floor[:, 0].tolist(),
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "train_sessions": train_names,
        "validation_sessions": validation_names,
        "test_policy": "January locked and never loaded",
        "january_loaded": False,
        "observation_seconds": OBSERVATION_SECONDS,
        "window_bins": WINDOW_BINS,
        "training_epochs": TRAIN_EPOCHS,
        "checkpoint_epoch": best_epoch,
        "selection_policy": "minimum pooled December validation normalized MSE",
        "best_validation_loss": best_validation_loss,
        "train_metrics": final_train_metrics,
        "validation_metrics": final_validation_metrics,
        "validation_macro_r2_mean": validation_macro_r2,
        "validation_worst_session_r2_mean": validation_worst_r2,
        "validation_by_session": validation_session_metrics,
        "training_history": history,
        "detector_compatibility": (
            "The existing detector was fitted to a different input/checkpoint. "
            "Both detector layers must be refitted before Phase 6 deployment."
        ),
    }
    save_checkpoint_atomic(checkpoint, CHECKPOINT_PATH)
    print(
        f"\nselected epoch={best_epoch:02d}/{TRAIN_EPOCHS} | "
        f"train R2={final_train_metrics['r2_mean']:+.4f} | "
        "validation pooled/macro/worst R2="
        f"{final_validation_metrics['r2_mean']:+.4f}/"
        f"{validation_macro_r2:+.4f}/{validation_worst_r2:+.4f}",
        flush=True,
    )
    print(f"checkpoint: {CHECKPOINT_PATH}", flush=True)
    return {
        "config": config,
        "parameter_count": parameter_count,
        "checkpoint_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "checkpoint_sha256": sha256_file(CHECKPOINT_PATH),
        "checkpoint_size_bytes": CHECKPOINT_PATH.stat().st_size,
        "train_metrics": final_train_metrics,
        "validation_metrics": final_validation_metrics,
        "validation_macro_r2_mean": validation_macro_r2,
        "validation_worst_session_r2_mean": validation_worst_r2,
        "validation_by_session": validation_session_metrics,
        "history": history,
    }


def write_history_csv(history: list[dict[str, Any]]) -> None:
    rows = [
        {key: value for key, value in row.items() if key != "month_draws"}
        for row in history
    ]
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_figure(history: list[dict[str, Any]], checkpoint_epoch: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(
        epochs,
        [row["validation_loss"] for row in history],
        label="validation",
    )
    axes[0].set_title("Normalized loss")
    axes[0].set_ylabel("MSE")
    axes[1].plot(epochs, [row["train_r2"] for row in history], label="train")
    axes[1].plot(
        epochs,
        [row["validation_r2"] for row in history],
        label="validation",
    )
    axes[1].set_title("Pooled R²")
    axes[1].set_ylabel("R²")
    for axis in axes:
        axis.axvline(
            checkpoint_epoch,
            color="#C44E52",
            linestyle="--",
            alpha=0.8,
            label="selected epoch",
        )
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(
        "Phase 6 · 96-channel causal 64/64 TCN+GRU\n"
        "Train updates weights · December selects checkpoint · January not loaded"
    )
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Use CUDA when available, otherwise CPU. Apple MPS is disabled.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate paths, split policy and model shape without loading arrays.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Intentionally replace existing Phase 6 outputs.",
    )
    args = parser.parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")

    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    seed_everything()
    context = validate_protocol(allow_overwrite=args.overwrite or args.validate_only)
    if args.validate_only:
        print("=== Phase 6 validation passed ===")
        print(
            f"architecture=64/64 TCN+GRU | neural channels={CHANNEL_COUNT} | "
            f"input features={INPUT_FEATURES} | parameters="
            f"{context['parameter_count']:,}"
        )
        print(
            f"epochs={TRAIN_EPOCHS} | seed={SEED} | lr={LEARNING_RATE} | "
            f"wd={WEIGHT_DECAY} | dropout={DROPOUT} | selected device={device}"
        )
        print(
            "train=29 updates weights | validation=4 inference/checkpoint only | "
            "January=FORBIDDEN"
        )
        print("no arrays loaded; no output written")
        return

    train_names = context["train_names"]
    validation_names = context["validation_names"]
    print("=== Indy Phase 6: 96-channel training ===", flush=True)
    print(
        f"sessions train={len(train_names)} | validation={len(validation_names)} | "
        "January=FORBIDDEN",
        flush=True,
    )
    print(
        "policy=all 29 train sessions; session-balanced; all train-derived "
        "statistics frozen before December is loaded",
        flush=True,
    )

    train_loaded = {name: load_model_data(name) for name in train_names}
    train_channel_counts = {counts.shape[0] for counts, _ in train_loaded.values()}
    if train_channel_counts != {CHANNEL_COUNT}:
        raise ValueError(
            "All Phase 6 train sessions must contain exactly 96 channels; "
            f"found {sorted(train_channel_counts)}"
        )
    feature_std_floor = fit_feature_std_floor(train_loaded)
    train_prepared = {
        name: prepare_session(train_loaded[name], feature_std_floor)
        for name in train_names
    }
    train_x = np.concatenate([train_prepared[name][0] for name in train_names], axis=0)
    train_y = np.concatenate([train_prepared[name][1] for name in train_names], axis=0)
    target_mean = train_y.mean(axis=(0, 1)).astype(np.float32)
    target_std = (train_y.std(axis=(0, 1)) + 1e-6).astype(np.float32)
    train_y_normalized = ((train_y - target_mean) / target_std).astype(np.float32)
    session_lengths = {name: int(len(train_prepared[name][0])) for name in train_names}
    del train_loaded, train_prepared

    validation_loaded = {name: load_model_data(name) for name in validation_names}
    validation_channel_counts = {
        counts.shape[0] for counts, _ in validation_loaded.values()
    }
    if validation_channel_counts != {CHANNEL_COUNT}:
        raise ValueError(
            "All Phase 6 validation sessions must contain exactly 96 channels; "
            f"found {sorted(validation_channel_counts)}"
        )
    validation_by_session = {
        name: prepare_session(validation_loaded[name], feature_std_floor)
        for name in validation_names
    }
    validation_x = np.concatenate(
        [validation_by_session[name][0] for name in validation_names], axis=0
    )
    validation_y = np.concatenate(
        [validation_by_session[name][1] for name in validation_names], axis=0
    )
    del validation_loaded

    print(
        f"windows train={len(train_x)} validation={len(validation_x)} | "
        f"input per window={tuple(train_x.shape[1:])}",
        flush=True,
    )
    print(
        "feature std floor min/median/max="
        f"{feature_std_floor.min():.5f}/"
        f"{np.median(feature_std_floor):.5f}/"
        f"{feature_std_floor.max():.5f}",
        flush=True,
    )

    result = train(
        train_names=train_names,
        validation_names=validation_names,
        parameter_count=context["parameter_count"],
        train_x=train_x,
        train_y=train_y,
        train_y_normalized=train_y_normalized,
        target_mean=target_mean,
        target_std=target_std,
        session_lengths=session_lengths,
        validation_x=validation_x,
        validation_y=validation_y,
        validation_by_session=validation_by_session,
        feature_std_floor=feature_std_floor,
        device=device,
    )
    write_history_csv(result["history"])
    render_figure(result["history"], result["checkpoint_epoch"])
    metrics = {
        "phase": "6",
        "name": PHASE_NAME,
        "created_at_utc": utc_now(),
        "purpose": "single-seed 96-channel extension of Phase 5 winner",
        "protocol": {
            "architecture": "strictly causal 64/64 TCN+GRU",
            "parameter_count": result["parameter_count"],
            "neural_channels": CHANNEL_COUNT,
            "features": "96 raw counts + 96 causal EWMA",
            "channel_selection": "all 96 physical channels; no fitted ranking",
            "bin_seconds": BIN_SECONDS,
            "window_bins": WINDOW_BINS,
            "observation_seconds": OBSERVATION_SECONDS,
            "sampling": "session-balanced",
            "seed": SEED,
            "epochs": TRAIN_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "checkpoint_selection": (
                "minimum pooled December validation normalized MSE"
            ),
            "device_policy": "CUDA when available, otherwise CPU; MPS disabled",
            "selected_device": device.type,
        },
        "data_policy": {
            "train_sessions_updated_weights": len(train_names),
            "validation_sessions_updated_weights": 0,
            "validation_selected_checkpoint": True,
            "test_sessions_loaded": 0,
            "january_loaded": False,
            "detector_exclusions_applied_to_training": False,
        },
        "checkpoint": str(CHECKPOINT_PATH.relative_to(REPOSITORY_ROOT)),
        "checkpoint_sha256": result["checkpoint_sha256"],
        "checkpoint_size_bytes": result["checkpoint_size_bytes"],
        "checkpoint_epoch": result["checkpoint_epoch"],
        "train_metrics": result["train_metrics"],
        "validation_metrics": result["validation_metrics"],
        "validation_macro_r2_mean": result["validation_macro_r2_mean"],
        "validation_worst_session_r2_mean": result["validation_worst_session_r2_mean"],
        "validation_by_session": result["validation_by_session"],
        "interpretation_boundary": (
            "This is one seed on December validation. Do not promote the "
            "checkpoint or refit the detector until the result is reviewed."
        ),
    }
    write_json_atomic(metrics, METRICS_PATH)
    print(f"metrics: {METRICS_PATH}", flush=True)
    print(f"epochs: {HISTORY_PATH}", flush=True)
    print(f"figure: {FIGURE_PATH}", flush=True)
    print("January: NOT LOADED | retained checkpoints: UNCHANGED", flush=True)


if __name__ == "__main__":
    main()
