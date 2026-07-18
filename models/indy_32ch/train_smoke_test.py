#!/usr/bin/env python
"""Train the 32-channel causal candidate on the eight locally available sessions.

This is a diagnostic smoke test, not a model-promotion workflow. The fixed split
is train1..train6 / eval1 / test1, and test1 has historical reuse. Test metrics
are printed every epoch only to make convergence visible; checkpoint selection
uses eval R² exclusively.
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
AXES = np.array([1, 2])
TRAIN_SESSIONS = ("train1", "train2", "train3", "train4", "train5", "train6")
EVAL_SESSION = "eval1"
TEST_SESSION = "test1"

METRICS_PATH = ROOT / "results" / "metrics" / "indy_32ch_smoke_test.json"
CHECKPOINT_PATH = ROOT / "results" / "large" / "indy_32ch_smoke_test_checkpoint.pt"
LOSS_FIGURE_PATH = ROOT / "results" / "figures" / "indy_32ch_smoke_test_losses.png"
R2_FIGURE_PATH = ROOT / "results" / "figures" / "indy_32ch_smoke_test_test_r2.png"


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
) -> tuple[np.ndarray, np.ndarray]:
    """Return post-60-second windows normalized only from that session's prefix."""
    counts, velocity = loaded[session]
    features = multiscale_counts(counts[channels], ALPHAS)
    if features.shape[1] <= OBSERVATION_BINS + WINDOW_BINS:
        raise ValueError(f"{session} is too short for the causal observation protocol")
    stats = fit_feature_stats(features, observation_bins=OBSERVATION_BINS)
    normalized = apply_feature_stats(features, stats)
    windows = window_arrays(
        normalized,
        velocity,
        AXES,
        window_bins=WINDOW_BINS,
        start_bin=OBSERVATION_BINS,
    )
    return stack_windows(windows)


def evaluate_split(
    net,
    split: dict[str, np.ndarray],
    target_mean: np.ndarray,
    target_std: np.ndarray,
    batch_size: int,
    *,
    compute_r2: bool,
) -> tuple[float, float | None]:
    """Return normalized-target MSE and optional raw-unit R²."""
    import torch

    net.eval()
    squared_error = 0.0
    n_values = 0
    predictions = []
    with torch.no_grad():
        for start in range(0, len(split["x"]), batch_size):
            stop = start + batch_size
            x = torch.from_numpy(split["x"][start:stop])
            target_norm = torch.from_numpy(split["y_norm"][start:stop])
            prediction_norm = net(x)
            squared_error += float(((prediction_norm - target_norm) ** 2).sum())
            n_values += target_norm.numel()
            if compute_r2:
                predictions.append(prediction_norm.numpy() * target_std + target_mean)
    loss = squared_error / max(n_values, 1)
    if not compute_r2:
        return loss, None
    prediction = np.concatenate(predictions, axis=0)
    score = float(r2(split["y"].reshape(-1, 2), prediction.reshape(-1, 2)).mean())
    return loss, score


def plot_history(history: list[dict], best_epoch: int) -> None:
    import os

    matplotlib_cache = ROOT / "results" / "large" / ".matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    LOSS_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.array([row["epoch"] for row in history])

    # Loss chart contract: comparable normalized MSE, one ordered epoch axis,
    # blue/gold/neutral palette plus line-style differences for grayscale use.
    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=180)
    ax.plot(epochs, [row["train_loss"] for row in history], color="#2368A2",
            linewidth=2.0, label="Train", linestyle="-")
    ax.plot(epochs, [row["eval_loss"] for row in history], color="#C58B18",
            linewidth=2.0, label="Eval", linestyle="--")
    ax.plot(epochs, [row["test_loss"] for row in history], color="#5B6470",
            linewidth=2.0, label="Test", linestyle=":")
    ax.axvline(best_epoch, color="#2F343B", linewidth=1.2, linestyle="-.",
               label=f"Best eval epoch: {best_epoch}")
    ax.set_title("Indy 32-channel causal smoke test — loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized target MSE")
    ax.grid(axis="y", color="#D8DDE3", linewidth=0.8, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2)
    fig.text(0.5, 0.01, "6 training sessions · eval1 selection · test1 diagnostic only",
             ha="center", color="#5B6470", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(LOSS_FIGURE_PATH, bbox_inches="tight")
    plt.close(fig)

    # R² chart contract: test trend only, with a zero reference and the eval-selected epoch.
    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=180)
    test_r2 = np.array([row["test_r2"] for row in history])
    ax.plot(epochs, test_r2, color="#2368A2", linewidth=2.2, label="Test R²")
    ax.axhline(0.0, color="#8A939E", linewidth=1.0, linestyle=":")
    ax.axvline(best_epoch, color="#C58B18", linewidth=1.4, linestyle="--",
               label=f"Best eval epoch: {best_epoch}")
    selected_r2 = test_r2[best_epoch - 1]
    ax.scatter([best_epoch], [selected_r2], color="#C58B18", edgecolor="#2F343B",
               linewidth=0.6, zorder=3)
    ax.annotate(f"{selected_r2:.3f}", (best_epoch, selected_r2), xytext=(7, 8),
                textcoords="offset points", color="#2F343B", fontsize=9)
    ax.set_title("Indy 32-channel causal smoke test — test R²")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R² (mean over velocity axes 1 and 2)")
    ax.grid(axis="y", color="#D8DDE3", linewidth=0.8, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.text(0.5, 0.01, "Test is plotted for diagnostics and never used to select the checkpoint",
             ha="center", color="#5B6470", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(R2_FIGURE_PATH, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=causal_config()["epochs"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    import torch
    import torch.nn as nn

    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    started = time.time()

    sessions = (*TRAIN_SESSIONS, EVAL_SESSION, TEST_SESSION)
    print("=== Indy 32-channel fully causal smoke test ===")
    print(f"split: train={list(TRAIN_SESSIONS)} | eval={EVAL_SESSION} | test={TEST_SESSION}")
    print(f"observation={OBSERVATION_SECONDS}s | scored suffix only | seed={args.seed}")
    loaded = {session: load_model_data(session) for session in sessions}
    training_loaded = {session: loaded[session] for session in TRAIN_SESSIONS}
    channels = top_firing_channels(
        training_loaded,
        N_CHANNELS,
        observation_bins=OBSERVATION_BINS,
    )
    print(f"selected channels from training prefixes only: {channels.tolist()}")

    prepared = {
        session: prepare_session(loaded, session, channels) for session in sessions
    }
    train_x = np.concatenate([prepared[s][0] for s in TRAIN_SESSIONS], axis=0)
    train_y = np.concatenate([prepared[s][1] for s in TRAIN_SESSIONS], axis=0)
    eval_x, eval_y = prepared[EVAL_SESSION]
    test_x, test_y = prepared[TEST_SESSION]

    target_mean = train_y.mean(axis=(0, 1))
    target_std = train_y.std(axis=(0, 1)) + 1e-6

    def pack(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "x": x,
            "y": y,
            "y_norm": ((y - target_mean) / target_std).astype(np.float32),
        }

    splits = {
        "train": pack(train_x, train_y),
        "eval": pack(eval_x, eval_y),
        "test": pack(test_x, test_y),
    }
    print(
        "windows: "
        + " | ".join(f"{name}={len(split['x'])}" for name, split in splits.items())
    )
    print(f"input shape per window: {train_x.shape[1:]} | target: {train_y.shape[1:]}")

    config = {**causal_config(), "epochs": args.epochs, "n_out": 2}
    net = build_net(config, train_x.shape[1])
    n_params = sum(parameter.numel() for parameter in net.parameters())
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=config["lr"], weight_decay=config["wd"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    mse = nn.MSELoss()
    x_train = torch.from_numpy(splits["train"]["x"])
    y_train = torch.from_numpy(splits["train"]["y_norm"])

    history: list[dict] = []
    best_eval_r2 = -np.inf
    best_epoch = 0
    best_state = None
    print(f"model parameters: {n_params:,}\n")

    for epoch in range(1, args.epochs + 1):
        net.train()
        indices = rng.permutation(len(x_train))
        for start in range(0, len(indices), config["bs"]):
            batch_indices = indices[start:start + config["bs"]]
            x_batch = x_train[batch_indices]
            if config["noise"] > 0:
                x_batch = x_batch + config["noise"] * torch.randn_like(x_batch)
            if config["chdrop"] > 0:
                mask = (
                    torch.rand(x_batch.shape[0], x_batch.shape[1], 1) > config["chdrop"]
                ).float()
                x_batch = x_batch * mask / (1.0 - config["chdrop"])
            optimizer.zero_grad()
            loss = mse(net(x_batch), y_train[batch_indices])
            loss.backward()
            optimizer.step()

        train_loss, _ = evaluate_split(
            net, splits["train"], target_mean, target_std, config["bs"], compute_r2=False
        )
        eval_loss, eval_r2 = evaluate_split(
            net, splits["eval"], target_mean, target_std, config["bs"], compute_r2=True
        )
        test_loss, test_r2 = evaluate_split(
            net, splits["test"], target_mean, target_std, config["bs"], compute_r2=True
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": float(train_loss),
            "eval_loss": float(eval_loss),
            "test_loss": float(test_loss),
            "eval_r2": float(eval_r2),
            "test_r2": float(test_r2),
        }
        history.append(row)
        improved = eval_r2 > best_eval_r2
        if improved:
            best_eval_r2 = float(eval_r2)
            best_epoch = epoch
            best_state = copy.deepcopy(net.state_dict())
        marker = " *best eval*" if improved else ""
        print(
            f"epoch {epoch:03d}/{args.epochs} | train loss {train_loss:.5f} | "
            f"eval loss {eval_loss:.5f} | test loss {test_loss:.5f} | "
            f"test R2 {test_r2:+.4f}{marker}",
            flush=True,
        )
        scheduler.step()

    if best_state is None:
        raise RuntimeError("No checkpoint was selected")
    net.load_state_dict(best_state)
    selected = {}
    for name, compute_r2 in (("train", False), ("eval", True), ("test", True)):
        loss, score = evaluate_split(
            net, splits[name], target_mean, target_std, config["bs"], compute_r2=compute_r2
        )
        selected[name] = {"loss": float(loss), "r2": None if score is None else float(score)}

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "purpose": "eight_session_causal_smoke_test_not_promoted",
            "model_state": best_state,
            "config": config,
            "channels": channels.tolist(),
            "target_mean": target_mean.tolist(),
            "target_std": target_std.tolist(),
            "split": {
                "train": list(TRAIN_SESSIONS),
                "eval": EVAL_SESSION,
                "test": TEST_SESSION,
            },
            "observation_seconds": OBSERVATION_SECONDS,
            "best_epoch_selected_on_eval_r2": best_epoch,
        },
        CHECKPOINT_PATH,
    )

    payload = {
        "purpose": "eight_session_causal_smoke_test_not_promoted",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "config": config,
        "split": {
            "train": list(TRAIN_SESSIONS),
            "eval": EVAL_SESSION,
            "test": TEST_SESSION,
        },
        "data_protocol": {
            "bin_seconds": BIN_S,
            "window_bins": WINDOW_BINS,
            "observation_seconds": OBSERVATION_SECONDS,
            "observation_bins": OBSERVATION_BINS,
            "features": ["counts", "causal_ewma_0.1"],
            "velocity_axes": AXES.tolist(),
            "feature_normalization": "per_session_first_60_seconds_frozen",
            "scored_data": "strictly_after_observation_prefix",
        },
        "n_parameters": n_params,
        "channels": channels.tolist(),
        "window_counts": {name: len(split["x"]) for name, split in splits.items()},
        "best_epoch_selected_on_eval_r2": best_epoch,
        "best_eval_r2": best_eval_r2,
        "selected_checkpoint_metrics": selected,
        "history": history,
        "artifacts": {
            "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
            "loss_figure": str(LOSS_FIGURE_PATH.relative_to(ROOT)),
            "test_r2_figure": str(R2_FIGURE_PATH.relative_to(ROOT)),
        },
        "caveat": "test1 is historically reused; this is a smoke test, not an unbiased result",
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_history(history, best_epoch)

    print("\n=== eval-selected smoke-test result ===")
    print(f"best epoch: {best_epoch} (eval R2={best_eval_r2:+.4f})")
    print(f"selected test loss: {selected['test']['loss']:.5f}")
    print(f"selected test R2: {selected['test']['r2']:+.4f}")
    print(f"metrics: {METRICS_PATH}")
    print(f"checkpoint: {CHECKPOINT_PATH}")
    print(f"loss figure: {LOSS_FIGURE_PATH}")
    print(f"test R2 figure: {R2_FIGURE_PATH}")


if __name__ == "__main__":
    main()
