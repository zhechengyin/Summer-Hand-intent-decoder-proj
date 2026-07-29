#!/usr/bin/env python3
"""Compare 64x64 and 48x48 causal decoders using 64 neural channels.

Controlled Phase 5a protocol:

- neural input: 64 channels selected only from train-session 60-second prefixes;
- features: 64 raw count streams + 64 causal-EWMA streams;
- architectures: TCN/GRU widths 64/64 and 48/48;
- optimization: CPU-only, identical seed, session-balanced samples and
  30-epoch schedule;
- train: 29 sessions are the only source of fitted statistics and gradients;
- validation: four December sessions are inference-only and select checkpoints;
- test: January session arrays are never loaded;
- outputs: two experiment checkpoints, metrics, epoch table and one figure.

This script never overwrites the retained 32-channel model checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.indy_32ch.features import multiscale_counts
from models.indy_32ch.input_pipeline import (
    apply_feature_stats,
    fit_feature_stats,
    load_model_data,
    load_session_manifest,
    processed_session_path,
    top_firing_channels,
    window_arrays,
)
from models.indy_32ch.model import build_net, causal_config, r2
from models.indy_32ch.sampling import draw_session_balanced_indices


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PHASE_NAME = "phase5a_64channel_width_comparison"
RESULT_DIR = ROOT / "results" / "indy" / PHASE_NAME
METRICS_PATH = RESULT_DIR / f"{PHASE_NAME}_metrics.json"
HISTORY_PATH = RESULT_DIR / f"{PHASE_NAME}_epochs.csv"
FIGURE_PATH = RESULT_DIR / f"{PHASE_NAME}_figure.png"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"

EXPECTED_SPLITS = {"train": 29, "validation": 4, "test": 4}
CHANNEL_COUNT = 64
ALPHAS = (1.0, 0.1)
INPUT_FEATURES = CHANNEL_COUNT * len(ALPHAS)
BIN_SECONDS = 0.040
OBSERVATION_SECONDS = 60
OBSERVATION_BINS = int(OBSERVATION_SECONDS / BIN_SECONDS)
WINDOW_BINS = 50
TARGET_AXES = (0, 1)
STD_FLOOR_PERCENTILE = 10.0

SEED = 43
TRAIN_EPOCHS = 30
LEARNING_RATE = 9e-4
WEIGHT_DECAY = 0.060
DROPOUT = 0.025
BATCH_SIZE = 32
GRADIENT_CLIP = 1.0

ARCHITECTURES = {
    "64ch_64x64": {"F": 64, "H": 64},
    "64ch_48x48": {"F": 48, "H": 48},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint_atomic(payload: dict, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def choose_device(requested: str):
    import torch

    if requested != "cpu":
        raise ValueError(
            "Phase 5a is CPU-only. Apple MPS produced invalid backward "
            "gradients for this causal TCN+GRU graph."
        )
    return torch.device("cpu")


def seed_everything(device, seed: int = SEED) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def architecture_config(widths: dict[str, int]) -> dict:
    config = causal_config(n_out=2)
    config.update(
        {
            "F": widths["F"],
            "H": widths["H"],
            "L": 1,
            "dils": [1, 2, 4, 8],
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


def checkpoint_path(architecture_name: str) -> Path:
    return CHECKPOINT_DIR / f"phase5a_{architecture_name}_checkpoint.pt"


def validate_protocol(
    *,
    allow_overwrite: bool,
    selected_architectures: list[str],
) -> dict:
    manifest = load_session_manifest()
    split = manifest["chronological_split"]
    split_counts = {name: len(split[name]) for name in EXPECTED_SPLITS}
    if split_counts != EXPECTED_SPLITS:
        raise ValueError(
            f"Expected chronological split {EXPECTED_SPLITS}, found {split_counts}"
        )

    train_names = list(split["train"])
    validation_names = list(split["validation"])
    development_names = train_names + validation_names
    if any(name.startswith("indy_201701") for name in development_names):
        raise ValueError("January leaked into train or validation")
    if any(not name.startswith("indy_201701") for name in split["test"]):
        raise ValueError("The locked test registry is not the expected January split")
    for name in development_names:
        if not processed_session_path(name).exists():
            raise FileNotFoundError(f"Missing processed session: {name}")

    if not allow_overwrite:
        existing = [
            str(checkpoint_path(name))
            for name in selected_architectures
            if checkpoint_path(name).exists()
        ]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite Phase-5a checkpoints. Use --overwrite "
                f"only for an intentional rerun: {existing}"
            )

    parameter_counts = {}
    for architecture_name, widths in ARCHITECTURES.items():
        net = build_net(architecture_config(widths), INPUT_FEATURES)
        parameter_counts[architecture_name] = sum(
            parameter.numel() for parameter in net.parameters()
        )

    return {
        "manifest": manifest,
        "train_names": train_names,
        "validation_names": validation_names,
        "parameter_counts": parameter_counts,
    }


def load_existing_result(
    *,
    architecture_name: str,
    widths: dict[str, int],
    parameter_count: int,
    channels: np.ndarray,
) -> dict:
    """Load a completed CPU Phase-5a checkpoint without retraining it."""
    import torch

    path = checkpoint_path(architecture_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing existing {architecture_name} checkpoint: {path}. "
            "Include this architecture in --architectures to train it."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "purpose": "indy_phase5a_64channel_width_comparison",
        "architecture_name": architecture_name,
        "training_device": "cpu",
        "neural_channel_count": CHANNEL_COUNT,
        "input_feature_count": INPUT_FEATURES,
        "training_epochs": TRAIN_EPOCHS,
        "parameter_count": parameter_count,
        "january_loaded": False,
    }
    mismatches = {
        key: {"expected": value, "actual": checkpoint.get(key)}
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Existing {architecture_name} checkpoint is incompatible: {mismatches}"
        )
    config = checkpoint.get("config", {})
    if config.get("F") != widths["F"] or config.get("H") != widths["H"]:
        raise ValueError(
            f"Existing {architecture_name} checkpoint has wrong widths: "
            f"F={config.get('F')} H={config.get('H')}"
        )
    saved_channels = np.asarray(checkpoint.get("channels"), dtype=np.int64)
    if not np.array_equal(saved_channels, channels):
        raise ValueError(
            f"Existing {architecture_name} checkpoint uses a different "
            "64-channel mapping"
        )
    history = checkpoint.get("training_history")
    if not isinstance(history, list) or len(history) != TRAIN_EPOCHS:
        raise ValueError(
            f"Existing {architecture_name} checkpoint lacks 30 history rows"
        )
    return {
        "architecture_name": architecture_name,
        "widths": widths,
        "parameter_count": parameter_count,
        "checkpoint": str(path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_size_bytes": path.stat().st_size,
        "checkpoint_epoch": int(checkpoint["checkpoint_epoch"]),
        "best_validation_loss": float(checkpoint["best_validation_loss"]),
        "train_metrics": checkpoint["train_metrics"],
        "validation_metrics": checkpoint["validation_metrics"],
        "validation_macro_r2_mean": float(
            checkpoint["validation_macro_r2_mean"]
        ),
        "validation_worst_session_r2_mean": float(
            checkpoint["validation_worst_session_r2_mean"]
        ),
        "validation_by_session": checkpoint["validation_by_session"],
        "history": history,
    }


def fit_feature_std_floor(
    loaded: dict[str, tuple[np.ndarray, np.ndarray]],
    channels: np.ndarray,
) -> np.ndarray:
    session_stds = []
    for counts, _ in loaded.values():
        features = multiscale_counts(counts[channels], ALPHAS)
        _, std = fit_feature_stats(
            features,
            observation_bins=OBSERVATION_BINS,
        )
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
    channels: np.ndarray,
    feature_std_floor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    counts, velocity = data
    if counts.shape[0] < CHANNEL_COUNT:
        raise ValueError(
            f"Session has {counts.shape[0]} neural channels; {CHANNEL_COUNT} required"
        )
    features = multiscale_counts(counts[channels], ALPHAS)
    if features.shape[0] != INPUT_FEATURES:
        raise AssertionError(
            f"Expected {INPUT_FEATURES} features, found {features.shape[0]}"
        )
    if features.shape[1] < OBSERVATION_BINS + WINDOW_BINS:
        raise ValueError("Session is too short for warm-up plus one window")
    mean, local_std = fit_feature_stats(
        features,
        observation_bins=OBSERVATION_BINS,
    )
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
) -> dict:
    prediction = predict(net, x, device) * target_std + target_mean
    score = r2(y.reshape(-1, 2), prediction.reshape(-1, 2))
    return {
        "windows": int(len(x)),
        "loss": float(np.mean(((y - prediction) / target_std) ** 2)),
        "r2_x": float(score[0]),
        "r2_y": float(score[1]),
        "r2_mean": float(score.mean()),
    }


def train_architecture(
    *,
    architecture_name: str,
    widths: dict[str, int],
    parameter_count: int,
    train_names: list[str],
    validation_names: list[str],
    channels: np.ndarray,
    feature_std_floor: np.ndarray,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_y_normalized: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    session_lengths: dict[str, int],
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    validation_prepared: dict[str, tuple[np.ndarray, np.ndarray]],
    device,
) -> dict:
    import torch
    import torch.nn as nn

    seed_everything(device)
    config = architecture_config(widths)
    net = build_net(config, INPUT_FEATURES).to(device)
    actual_parameters = sum(parameter.numel() for parameter in net.parameters())
    if actual_parameters != parameter_count:
        raise AssertionError(
            f"{architecture_name} parameter count changed: "
            f"{actual_parameters:,} != {parameter_count:,}"
        )

    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        TRAIN_EPOCHS,
    )
    mse = nn.MSELoss()
    rng = np.random.default_rng(SEED)
    history = []
    best_state = None
    best_epoch = None
    best_validation_loss = float("inf")

    print(f"\n=== {architecture_name} ===", flush=True)
    print(
        f"parameters={parameter_count:,} | input={INPUT_FEATURES}x{WINDOW_BINS} | "
        f"epochs={TRAIN_EPOCHS}",
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
            y_batch = torch.from_numpy(
                train_y_normalized[batch_indices]
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = mse(net(x_batch), y_batch)
            loss.backward()
            gradient = float(
                torch.nn.utils.clip_grad_norm_(
                    net.parameters(),
                    GRADIENT_CLIP,
                )
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
        row = {
            "architecture": architecture_name,
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "optimization_loss": error_sum / max(value_count, 1),
            "gradient_norm_mean_before_clip": gradient_sum / max(batch_count, 1),
            "gradient_norm_max_before_clip": gradient_max,
            "train_loss": train_metrics["loss"],
            "train_r2": train_metrics["r2_mean"],
            "validation_loss": validation_metrics["loss"],
            "validation_r2": validation_metrics["r2_mean"],
            "session_draws": session_draws,
            "month_draws": month_draws,
        }
        improved = validation_metrics["loss"] < best_validation_loss
        if improved:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in net.state_dict().items()
            }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{TRAIN_EPOCHS} | "
            f"opt={row['optimization_loss']:.5f} | "
            f"loss train={train_metrics['loss']:.5f} "
            f"validation={validation_metrics['loss']:.5f} | "
            f"R2 train={train_metrics['r2_mean']:+.4f} "
            f"validation={validation_metrics['r2_mean']:+.4f} | "
            f"lr={row['learning_rate']:.6g} | "
            f"grad={row['gradient_norm_mean_before_clip']:.3f}/"
            f"{row['gradient_norm_max_before_clip']:.3f}"
            + (" *best*" if improved else ""),
            flush=True,
        )
        scheduler.step()

    if best_state is None or best_epoch is None:
        raise RuntimeError(f"No checkpoint selected for {architecture_name}")
    net.load_state_dict(best_state)
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
    validation_by_session = {
        name: evaluate(
            net,
            validation_prepared[name][0],
            validation_prepared[name][1],
            target_mean,
            target_std,
            device,
        )
        for name in validation_names
    }
    validation_macro = float(
        np.mean([row["r2_mean"] for row in validation_by_session.values()])
    )
    validation_worst = float(
        min(row["r2_mean"] for row in validation_by_session.values())
    )

    output_path = checkpoint_path(architecture_name)
    checkpoint = {
        "purpose": "indy_phase5a_64channel_width_comparison",
        "created_at_utc": utc_now(),
        "architecture_name": architecture_name,
        "architecture_status": "experiment_candidate_not_promoted",
        "seed": SEED,
        "training_device": device.type,
        "torch_version": torch.__version__,
        "model_state": {
            key: value.detach().cpu().clone()
            for key, value in net.state_dict().items()
        },
        "config": config,
        "neural_channel_count": CHANNEL_COUNT,
        "input_feature_count": INPUT_FEATURES,
        "channels": channels.tolist(),
        "channel_selection": "top_mean_counts_from_train_60s_prefixes_only",
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "feature_std_floor": feature_std_floor[:, 0].tolist(),
        "train_sessions": train_names,
        "validation_sessions": validation_names,
        "test_policy": "locked_not_loaded",
        "january_loaded": False,
        "observation_seconds": OBSERVATION_SECONDS,
        "training_epochs": TRAIN_EPOCHS,
        "checkpoint_epoch": best_epoch,
        "selection_policy": "minimum_pooled_validation_normalized_mse",
        "best_validation_loss": best_validation_loss,
        "parameter_count": parameter_count,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "validation_macro_r2_mean": validation_macro,
        "validation_worst_session_r2_mean": validation_worst,
        "validation_by_session": validation_by_session,
        "training_history": history,
        "detector_compatibility": (
            "Incompatible with the current 32-channel Layer-1 reference and "
            "existing Layer-2 reference; both layers require refitting before "
            "runtime integration."
        ),
    }
    save_checkpoint_atomic(checkpoint, output_path)
    output_hash = sha256_file(output_path)
    print(
        f"selected epoch={best_epoch}/{TRAIN_EPOCHS} | "
        f"validation loss={validation_metrics['loss']:.5f} | "
        f"pooled/macro/worst R2={validation_metrics['r2_mean']:+.4f}/"
        f"{validation_macro:+.4f}/{validation_worst:+.4f}",
        flush=True,
    )
    print(f"saved: {output_path}", flush=True)

    return {
        "architecture_name": architecture_name,
        "widths": widths,
        "parameter_count": parameter_count,
        "checkpoint": str(output_path.relative_to(ROOT)),
        "checkpoint_sha256": output_hash,
        "checkpoint_size_bytes": output_path.stat().st_size,
        "checkpoint_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "validation_macro_r2_mean": validation_macro,
        "validation_worst_session_r2_mean": validation_worst,
        "validation_by_session": validation_by_session,
        "history": history,
    }


def write_history_csv(results: dict[str, dict]) -> None:
    rows = []
    for result in results.values():
        for row in result["history"]:
            rows.append(
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"session_draws", "month_draws"}
                }
            )
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_figure(results: dict[str, dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"64ch_64x64": "#2458A6", "64ch_48x48": "#C27A16"}
    labels = {"64ch_64x64": "64 channels · 64/64", "64ch_48x48": "64 channels · 48/48"}
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, result in results.items():
        epochs = [row["epoch"] for row in result["history"]]
        color = colors[name]
        label = labels[name]
        axes[0].plot(
            epochs,
            [row["train_loss"] for row in result["history"]],
            color=color,
            linestyle="-",
            label=f"{label} train",
        )
        axes[0].plot(
            epochs,
            [row["validation_loss"] for row in result["history"]],
            color=color,
            linestyle="--",
            label=f"{label} validation",
        )
        axes[1].plot(
            epochs,
            [row["train_r2"] for row in result["history"]],
            color=color,
            linestyle="-",
            label=f"{label} train",
        )
        axes[1].plot(
            epochs,
            [row["validation_r2"] for row in result["history"]],
            color=color,
            linestyle="--",
            label=f"{label} validation",
        )
        axes[0].scatter(
            [result["checkpoint_epoch"]],
            [result["best_validation_loss"]],
            color=color,
            marker="o",
            zorder=5,
        )
    axes[0].set_title("Normalized loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[1].set_title("Pooled R²")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("R²")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(
        "Phase 5a · 64-channel causal TCN+GRU width comparison\n"
        "Train updates weights · December validation selects checkpoints · "
        "January not loaded"
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
        choices=("cpu",),
        default="cpu",
        help="Phase 5a is locked to CPU because MPS backward is invalid.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate paths, split policy and model shapes without loading arrays.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Intentionally replace existing Phase-5a experiment outputs.",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=tuple(ARCHITECTURES),
        default=list(ARCHITECTURES),
        help=(
            "Architectures to train. Omitted architectures are loaded from "
            "their completed CPU Phase-5a checkpoints for final comparison."
        ),
    )
    args = parser.parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")

    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    seed_everything(device)
    selected_architectures = list(dict.fromkeys(args.architectures))
    context = validate_protocol(
        allow_overwrite=args.overwrite or args.validate_only,
        selected_architectures=selected_architectures,
    )
    if args.validate_only:
        print("=== Phase 5a validation passed ===")
        print(
            f"architectures={list(ARCHITECTURES)} | neural channels={CHANNEL_COUNT} | "
            f"input features={INPUT_FEATURES} | epochs each={TRAIN_EPOCHS}"
        )
        print(f"selected for training={selected_architectures}")
        print(f"parameters={context['parameter_counts']}")
        print(
            "train=29 updates weights | validation=4 inference/checkpoint only | "
            "January=FORBIDDEN"
        )
        print("no arrays loaded; no output written")
        return

    train_names = context["train_names"]
    validation_names = context["validation_names"]
    print("=== Phase 5a: 64-channel architecture comparison ===", flush=True)
    print(
        f"train={len(train_names)} | validation={len(validation_names)} "
        "inference/checkpoint only | January=FORBIDDEN",
        flush=True,
    )
    print(
        f"seed={SEED} | epochs each={TRAIN_EPOCHS} | device={device} | "
        "sampling=session-balanced",
        flush=True,
    )
    print(f"selected for training={selected_architectures}", flush=True)

    # Train data alone selects channels and fits every normalization statistic.
    train_loaded = {name: load_model_data(name) for name in train_names}
    channel_counts = {counts.shape[0] for counts, _ in train_loaded.values()}
    if len(channel_counts) != 1 or next(iter(channel_counts)) < CHANNEL_COUNT:
        raise ValueError(f"Inconsistent or insufficient neural channels: {channel_counts}")
    channels = top_firing_channels(
        train_loaded,
        CHANNEL_COUNT,
        observation_bins=OBSERVATION_BINS,
    )
    if channels.shape != (CHANNEL_COUNT,) or len(np.unique(channels)) != CHANNEL_COUNT:
        raise AssertionError("Channel selection did not return 64 unique channels")
    feature_std_floor = fit_feature_std_floor(train_loaded, channels)
    train_prepared = {
        name: prepare_session(
            train_loaded[name],
            channels,
            feature_std_floor,
        )
        for name in train_names
    }
    train_x = np.concatenate(
        [train_prepared[name][0] for name in train_names],
        axis=0,
    )
    train_y = np.concatenate(
        [train_prepared[name][1] for name in train_names],
        axis=0,
    )
    target_mean = train_y.mean(axis=(0, 1))
    target_std = train_y.std(axis=(0, 1)) + 1e-6
    train_y_normalized = ((train_y - target_mean) / target_std).astype(np.float32)
    session_lengths = {
        name: int(len(train_prepared[name][0])) for name in train_names
    }
    del train_loaded, train_prepared

    # December is opened only after all train-derived choices are frozen.
    validation_loaded = {
        name: load_model_data(name) for name in validation_names
    }
    validation_prepared = {
        name: prepare_session(
            validation_loaded[name],
            channels,
            feature_std_floor,
        )
        for name in validation_names
    }
    validation_x = np.concatenate(
        [validation_prepared[name][0] for name in validation_names],
        axis=0,
    )
    validation_y = np.concatenate(
        [validation_prepared[name][1] for name in validation_names],
        axis=0,
    )
    del validation_loaded

    print(f"selected channels (zero-based): {channels.tolist()}", flush=True)
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

    results = {}
    for architecture_name, widths in ARCHITECTURES.items():
        if architecture_name in selected_architectures:
            results[architecture_name] = train_architecture(
                architecture_name=architecture_name,
                widths=widths,
                parameter_count=context["parameter_counts"][architecture_name],
                train_names=train_names,
                validation_names=validation_names,
                channels=channels,
                feature_std_floor=feature_std_floor,
                train_x=train_x,
                train_y=train_y,
                train_y_normalized=train_y_normalized,
                target_mean=target_mean,
                target_std=target_std,
                session_lengths=session_lengths,
                validation_x=validation_x,
                validation_y=validation_y,
                validation_prepared=validation_prepared,
                device=device,
            )
        else:
            results[architecture_name] = load_existing_result(
                architecture_name=architecture_name,
                widths=widths,
                parameter_count=context["parameter_counts"][architecture_name],
                channels=channels,
            )
            print(
                f"\n=== {architecture_name} ===\n"
                "loaded completed CPU checkpoint; no retraining and no overwrite",
                flush=True,
            )

    ranking = sorted(
        results,
        key=lambda name: results[name]["validation_metrics"]["loss"],
    )
    metrics_payload = {
        "phase": "5a",
        "purpose": "compare_64x64_and_48x48_with_64_neural_channels",
        "created_at_utc": utc_now(),
        "protocol": {
            "training_device": device.type,
            "torch_version": torch.__version__,
            "neural_channels": CHANNEL_COUNT,
            "features": ["counts", "causal_ewma"],
            "input_features": INPUT_FEATURES,
            "window_bins": WINDOW_BINS,
            "observation_seconds": OBSERVATION_SECONDS,
            "channel_selection": "train_60s_prefixes_only",
            "sampling": "session_balanced",
            "seed": SEED,
            "epochs_each": TRAIN_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "checkpoint_selection": "minimum_pooled_december_validation_loss",
        },
        "data_policy": {
            "train_sessions_updated_weights": len(train_names),
            "validation_sessions_updated_weights": 0,
            "validation_selected_checkpoint": True,
            "test_sessions_loaded": 0,
            "january_loaded": False,
        },
        "selected_channels_zero_based": channels.tolist(),
        "feature_std_floor": feature_std_floor[:, 0].tolist(),
        "train_windows": int(len(train_x)),
        "validation_windows": int(len(validation_x)),
        "architectures": {
            name: {
                key: value
                for key, value in result.items()
                if key != "history"
            }
            for name, result in results.items()
        },
        "validation_loss_ranking": ranking,
        "provisional_recommendation": ranking[0],
        "recommendation_scope": (
            "Single-seed December validation comparison only; do not promote "
            "without result review and detector refit."
        ),
    }
    write_history_csv(results)
    render_figure(results)
    write_json_atomic(metrics_payload, METRICS_PATH)

    print("\n=== Phase 5a validation-only comparison ===", flush=True)
    for name in ranking:
        row = results[name]
        print(
            f"{name:12s} | params={row['parameter_count']:,} | "
            f"epoch={row['checkpoint_epoch']:02d} | "
            f"train R2={row['train_metrics']['r2_mean']:+.4f} | "
            f"validation loss={row['validation_metrics']['loss']:.5f} "
            f"pooled/macro/worst R2="
            f"{row['validation_metrics']['r2_mean']:+.4f}/"
            f"{row['validation_macro_r2_mean']:+.4f}/"
            f"{row['validation_worst_session_r2_mean']:+.4f}",
            flush=True,
        )
    print(f"provisional recommendation: {ranking[0]}", flush=True)
    print(f"metrics: {METRICS_PATH}", flush=True)
    print(f"figure: {FIGURE_PATH}", flush=True)
    print("January: NOT LOADED | active checkpoints: UNCHANGED", flush=True)


if __name__ == "__main__":
    main()
