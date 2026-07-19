#!/usr/bin/env python
"""Train the 32-channel causal decoder on the fixed 29/4/4 Indy split.

Only the 29 training sessions can update weights or fit shared preprocessing.
The four validation and four test sessions are inference-only diagnostics: they
are evaluated under ``torch.no_grad()`` and never select a checkpoint, stop
training, tune a hyperparameter, or update shared/target statistics. Each
session still performs the deployment-style causal calibration on its own first
60 seconds. The checkpoint is selected using training loss only during a fixed
25-epoch run.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.intent_decoder.data.indy import (
    apply_feature_stats,
    fit_feature_stats,
    load_model_data,
    load_session_manifest,
    top_firing_channels,
    window_arrays,
)
from src.intent_decoder.features.causal import multiscale_counts
from src.intent_decoder.model.tcn_gru import build_net, causal_config, r2

BIN_S = 0.040
WINDOW_BINS = 50
OBSERVATION_SECONDS = 60
OBSERVATION_BINS = int(OBSERVATION_SECONDS / BIN_S)
N_CHANNELS = 32
ALPHAS = (1.0, 0.1)
AXES = np.array([0, 1])
EXPECTED_SPLIT_COUNTS = {"train": 29, "validation": 4, "test": 4}
STD_FLOOR_PERCENTILE = 10.0

METRICS_PATH = ROOT / "results" / "metrics" / "indy_32ch_chronological_baseline.json"
CHECKPOINT_PATH = (
    ROOT / "results" / "large" / "indy_32ch_chronological_baseline_checkpoint.pt"
)
FIGURE_PATH = (
    ROOT / "results" / "figures" / "indy_32ch_chronological_training.png"
)


def stack_windows(windows: list[dict[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    if not windows:
        raise ValueError("No post-observation windows available")
    return (
        np.stack([window["e"] for window in windows]).astype(np.float32),
        np.stack([window["vel"] for window in windows]).astype(np.float32),
    )


def prepare_session(
    loaded: dict[str, tuple[np.ndarray, np.ndarray]],
    session: str,
    channels: np.ndarray,
    feature_std_floor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Create windows after a causal, session-local 60-second calibration."""
    counts, velocity = loaded[session]
    features = multiscale_counts(counts[channels], ALPHAS)
    if features.shape[1] < OBSERVATION_BINS + WINDOW_BINS:
        raise ValueError(f"{session} is too short for the causal observation protocol")

    # This calibration is deployable: only the session's first 60 seconds are
    # observed, the resulting statistics are frozen, and those 60 seconds are
    # excluded from all training and scoring windows.
    mean, local_std = fit_feature_stats(features, observation_bins=OBSERVATION_BINS)
    # A channel can be silent during calibration and become active later. Using
    # std + 1e-6 would then turn one count into a value near 1,000,000. Floors
    # are learned once from training prefixes and prevent that causal edge case.
    safe_std = np.maximum(local_std, feature_std_floor)
    normalized = apply_feature_stats(features, (mean, safe_std))
    windows = window_arrays(
        normalized,
        velocity,
        AXES,
        window_bins=WINDOW_BINS,
        start_bin=OBSERVATION_BINS,
    )
    return stack_windows(windows)


def fit_feature_std_floor(
    training_loaded: dict[str, tuple[np.ndarray, np.ndarray]],
    channels: np.ndarray,
) -> np.ndarray:
    """Fit robust per-feature scale floors from training prefixes only."""
    session_stds = []
    for counts, _ in training_loaded.values():
        features = multiscale_counts(counts[channels], ALPHAS)
        _, std = fit_feature_stats(features, observation_bins=OBSERVATION_BINS)
        session_stds.append(std[:, 0])

    scales = np.stack(session_stds)
    floors = np.empty(scales.shape[1], dtype=np.float32)
    for feature in range(scales.shape[1]):
        valid = scales[:, feature][scales[:, feature] > 1e-4]
        if valid.size == 0:
            raise ValueError(f"Feature {feature} is silent in every training prefix")
        floors[feature] = np.percentile(valid, STD_FLOOR_PERCENTILE)
    return floors[:, None]


def combine_sessions(
    prepared: dict[str, tuple[np.ndarray, np.ndarray]],
    sessions: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([prepared[session][0] for session in sessions], axis=0),
        np.concatenate([prepared[session][1] for session in sessions], axis=0),
    )


def evaluate_split(
    net,
    split: dict[str, np.ndarray],
    target_mean: np.ndarray,
    target_std: np.ndarray,
    batch_size: int,
    device,
) -> tuple[float, np.ndarray]:
    """Return normalized-target MSE and raw-unit R2 for x and y."""
    import torch

    net.eval()
    squared_error = 0.0
    n_values = 0
    predictions = []
    with torch.no_grad():
        for start in range(0, len(split["x"]), batch_size):
            stop = start + batch_size
            x = torch.from_numpy(split["x"][start:stop]).to(device)
            target_norm = torch.from_numpy(split["y_norm"][start:stop]).to(device)
            prediction_norm = net(x)
            squared_error += float(((prediction_norm - target_norm) ** 2).sum().item())
            n_values += target_norm.numel()
            predictions.append(prediction_norm.cpu().numpy() * target_std + target_mean)

    loss = squared_error / max(n_values, 1)
    prediction = np.concatenate(predictions, axis=0)
    score = r2(split["y"].reshape(-1, 2), prediction.reshape(-1, 2))
    return float(loss), score.astype(np.float64)


def plot_history(history: list[dict]) -> None:
    import os

    matplotlib_cache = ROOT / "results" / "large" / ".matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.asarray([row["epoch"] for row in history])
    colors = {"train": "#2368A2", "validation": "#C58B18", "test": "#5B6470"}
    styles = {"train": "-", "validation": "--", "test": ":"}

    fig, (loss_axis, r2_axis) = plt.subplots(1, 2, figsize=(14.0, 5.4), dpi=180)
    for split in ("train", "validation", "test"):
        loss_axis.plot(
            epochs,
            [row[f"{split}_loss"] for row in history],
            color=colors[split],
            linestyle=styles[split],
            linewidth=2.0,
            label=split.title(),
        )
    loss_axis.set_title("Normalized MSE loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.grid(axis="y", color="#D8DDE3", linewidth=0.8, alpha=0.8)
    loss_axis.spines[["top", "right"]].set_visible(False)
    loss_axis.legend(frameon=False)

    for split in ("train", "validation", "test"):
        r2_axis.plot(
            epochs,
            [row[f"{split}_r2_mean"] for row in history],
            color=colors[split],
            linestyle=styles[split],
            linewidth=2.0,
            label=split.title(),
        )
    r2_axis.axhline(0.0, color="#9AA2AC", linewidth=1.0, linestyle="-.")
    r2_axis.set_title("Velocity R2")
    r2_axis.set_xlabel("Epoch")
    r2_axis.set_ylabel("Mean R2 over x and y")
    r2_axis.grid(axis="y", color="#D8DDE3", linewidth=0.8, alpha=0.8)
    r2_axis.spines[["top", "right"]].set_visible(False)
    r2_axis.legend(frameon=False)

    fig.suptitle("Indy 32-channel causal baseline: 29 train / 4 validation / 4 test")
    fig.text(
        0.5,
        0.01,
        "Weights use train only; validation and test are inference-only diagnostics.",
        ha="center",
        color="#5B6470",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--channel-dropout", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Training device. 'auto' prefers CUDA, then Apple MPS, then CPU.",
    )
    return parser.parse_args()


def choose_device(requested: str):
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def main() -> None:
    import torch
    import torch.nn as nn

    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.noise < 0:
        raise ValueError("--noise cannot be negative")
    if not 0 <= args.channel_dropout < 1:
        raise ValueError("--channel-dropout must be in [0, 1)")
    if args.gradient_clip <= 0:
        raise ValueError("--gradient-clip must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    device = choose_device(args.device)
    started = time.time()

    manifest = load_session_manifest()
    split_sessions = {
        name: list(manifest["chronological_split"][name])
        for name in ("train", "validation", "test")
    }
    actual_counts = {name: len(sessions) for name, sessions in split_sessions.items()}
    if actual_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            f"Expected chronological split {EXPECTED_SPLIT_COUNTS}, found {actual_counts}"
        )

    print("=== Indy 32-channel chronological causal baseline ===")
    print(
        f"sessions: train={actual_counts['train']} | "
        f"validation={actual_counts['validation']} | test={actual_counts['test']}"
    )
    print(
        f"epochs={args.epochs} | observation={OBSERVATION_SECONDS}s | "
        f"device={device} | seed={args.seed}"
    )
    print(
        "policy: optimizer=train only; validation/test=inference only; "
        "checkpoint=best train loss"
    )
    print(
        f"optimization: lr={args.learning_rate:g} | noise={args.noise:g} | "
        f"channel_dropout={args.channel_dropout:g} | "
        f"gradient_clip={args.gradient_clip:g}"
    )

    all_sessions = [
        session
        for split in ("train", "validation", "test")
        for session in split_sessions[split]
    ]
    loaded = {session: load_model_data(session) for session in all_sessions}

    # Shared channel selection sees only past prefixes of training sessions.
    training_loaded = {
        session: loaded[session] for session in split_sessions["train"]
    }
    channels = top_firing_channels(
        training_loaded,
        N_CHANNELS,
        observation_bins=OBSERVATION_BINS,
    )
    print(f"selected channels from training prefixes only: {channels.tolist()}")

    feature_std_floor = fit_feature_std_floor(training_loaded, channels)
    print(
        "training-derived feature std floor: "
        f"min={feature_std_floor.min():.5f} | "
        f"median={np.median(feature_std_floor):.5f} | "
        f"max={feature_std_floor.max():.5f}"
    )

    prepared = {
        session: prepare_session(loaded, session, channels, feature_std_floor)
        for session in all_sessions
    }
    combined = {
        split: combine_sessions(prepared, sessions)
        for split, sessions in split_sessions.items()
    }

    train_y = combined["train"][1]
    target_mean = train_y.mean(axis=(0, 1))
    target_std = train_y.std(axis=(0, 1)) + 1e-6

    def pack(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "x": x,
            "y": y,
            "y_norm": ((y - target_mean) / target_std).astype(np.float32),
        }

    splits = {
        split: pack(x, y)
        for split, (x, y) in combined.items()
    }
    print(
        "windows: "
        + " | ".join(f"{name}={len(split['x'])}" for name, split in splits.items())
    )
    print(
        f"input per window: {splits['train']['x'].shape[1:]} "
        f"(32 raw + 32 causal EWMA) | target: {train_y.shape[1:]}"
    )

    config = {
        **causal_config(),
        "epochs": args.epochs,
        "n_out": 2,
        "lr": args.learning_rate,
        "noise": args.noise,
        "chdrop": args.channel_dropout,
        "gradient_clip": args.gradient_clip,
    }
    net = build_net(config, splits["train"]["x"].shape[1]).to(device)
    n_parameters = sum(parameter.numel() for parameter in net.parameters())
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=config["lr"], weight_decay=config["wd"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    mse = nn.MSELoss()
    train_x = torch.from_numpy(splits["train"]["x"])
    train_y_norm = torch.from_numpy(splits["train"]["y_norm"])

    history: list[dict] = []
    best_train_loss = np.inf
    best_epoch = 0
    best_state = None
    print(f"model parameters: {n_parameters:,}\n")

    for epoch in range(1, args.epochs + 1):
        net.train()
        indices = rng.permutation(len(train_x))
        for start in range(0, len(indices), config["bs"]):
            batch_indices = indices[start:start + config["bs"]]
            x_batch = train_x[batch_indices].to(device)
            y_batch = train_y_norm[batch_indices].to(device)
            if config["noise"] > 0:
                x_batch = x_batch + config["noise"] * torch.randn_like(x_batch)
            if config["chdrop"] > 0:
                mask = (
                    torch.rand(
                        x_batch.shape[0], x_batch.shape[1], 1, device=device
                    )
                    > config["chdrop"]
                ).float()
                x_batch = x_batch * mask / (1.0 - config["chdrop"])

            optimizer.zero_grad(set_to_none=True)
            loss = mse(net(x_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.gradient_clip)
            optimizer.step()

        row: dict[str, float | int] = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        for split_name in ("train", "validation", "test"):
            split_loss, split_r2 = evaluate_split(
                net,
                splits[split_name],
                target_mean,
                target_std,
                config["bs"],
                device,
            )
            row[f"{split_name}_loss"] = split_loss
            row[f"{split_name}_r2_x"] = float(split_r2[0])
            row[f"{split_name}_r2_y"] = float(split_r2[1])
            row[f"{split_name}_r2_mean"] = float(split_r2.mean())
        history.append(row)

        improved = row["train_loss"] < best_train_loss
        if improved:
            best_train_loss = float(row["train_loss"])
            best_epoch = epoch
            best_state = copy.deepcopy(net.state_dict())

        print(
            f"epoch {epoch:02d}/{args.epochs} | "
            f"loss train={row['train_loss']:.5f} "
            f"validation={row['validation_loss']:.5f} test={row['test_loss']:.5f} | "
            f"R2 train={row['train_r2_mean']:+.4f} "
            f"validation={row['validation_r2_mean']:+.4f} "
            f"test={row['test_r2_mean']:+.4f}"
            + (" *best train*" if improved else ""),
            flush=True,
        )
        scheduler.step()

    final_epoch_metrics = {
        split: {
            "loss": float(history[-1][f"{split}_loss"]),
            "r2_x": float(history[-1][f"{split}_r2_x"]),
            "r2_y": float(history[-1][f"{split}_r2_y"]),
            "r2_mean": float(history[-1][f"{split}_r2_mean"]),
        }
        for split in ("train", "validation", "test")
    }

    if best_state is None:
        raise RuntimeError("No training checkpoint was selected")
    net.load_state_dict(best_state)
    selected_metrics = {}
    for split in ("train", "validation", "test"):
        split_loss, split_r2 = evaluate_split(
            net,
            splits[split],
            target_mean,
            target_std,
            config["bs"],
            device,
        )
        selected_metrics[split] = {
            "loss": split_loss,
            "r2_x": float(split_r2[0]),
            "r2_y": float(split_r2[1]),
            "r2_mean": float(split_r2.mean()),
        }

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "purpose": "indy_32ch_chronological_causal_baseline",
            "model_state": net.state_dict(),
            "config": config,
            "channels": channels.tolist(),
            "target_mean": target_mean.tolist(),
            "target_std": target_std.tolist(),
            "feature_std_floor": feature_std_floor[:, 0].tolist(),
            "split": split_sessions,
            "observation_seconds": OBSERVATION_SECONDS,
            "checkpoint_epoch": best_epoch,
            "selection_policy": "minimum_train_loss_no_validation_or_test_selection",
        },
        CHECKPOINT_PATH,
    )

    payload = {
        "purpose": "indy_32ch_chronological_causal_baseline",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "config": config,
        "device": str(device),
        "split": split_sessions,
        "data_protocol": {
            "bin_seconds": BIN_S,
            "window_bins": WINDOW_BINS,
            "observation_seconds": OBSERVATION_SECONDS,
            "observation_bins": OBSERVATION_BINS,
            "features": ["counts", "causal_ewma_0.1"],
            "velocity_axes": AXES.tolist(),
            "channel_selection": "top_32_from_train_session_observation_prefixes_only",
            "feature_normalization": "per_session_first_60_seconds_frozen",
            "feature_std_floor": (
                "10th_percentile_positive_scale_from_train_prefixes_only"
            ),
            "target_normalization": "training_windows_only",
            "scored_data": "strictly_after_observation_prefix",
        },
        "training_policy": {
            "gradient_updates": "train_only",
            "checkpoint": "minimum_train_loss",
            "early_stopping": False,
            "validation_and_test": "inference_only_diagnostics",
        },
        "n_parameters": n_parameters,
        "channels": channels.tolist(),
        "feature_std_floor": feature_std_floor[:, 0].tolist(),
        "window_counts": {name: len(split["x"]) for name, split in splits.items()},
        "best_epoch_selected_on_train_loss": best_epoch,
        "best_train_loss": best_train_loss,
        "selected_checkpoint_metrics": selected_metrics,
        "final_epoch_metrics": final_epoch_metrics,
        "history": history,
        "artifacts": {
            "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
            "figure": str(FIGURE_PATH.relative_to(ROOT)),
        },
        "caveat": (
            "Per-epoch test metrics are shown only because this diagnostic run requests "
            "them; do not use their curves to alter the model or training configuration."
        ),
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_history(history)

    print("\n=== train-selected checkpoint result ===")
    print(f"selected epoch: {best_epoch} (train loss={best_train_loss:.5f})")
    for split in ("train", "validation", "test"):
        metric = selected_metrics[split]
        print(
            f"{split:10s} loss={metric['loss']:.5f} | "
            f"R2 x={metric['r2_x']:+.4f} y={metric['r2_y']:+.4f} "
            f"mean={metric['r2_mean']:+.4f}"
        )
    print(f"metrics: {METRICS_PATH}")
    print(f"checkpoint: {CHECKPOINT_PATH}")
    print(f"two-panel figure: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
