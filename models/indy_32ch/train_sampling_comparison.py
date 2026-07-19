#!/usr/bin/env python
"""Sweep causal Indy sampling policies across multiple random seeds.

The default experiment reuses the completed seed-42 result and trains seeds 43
and 44 for three train-only sampling policies. Within each newly trained seed,
all policies share the same initial model state,
processed data, preprocessing, target statistics, optimizer settings, samples
per epoch, and epoch count. Only the training-window sampling distribution
changes:

* ``window``: every available training window appears once per epoch;
* ``session``: every training session contributes equally per epoch;
* ``month``: every training month contributes equally, then sessions are
  balanced within each month.

Checkpoint selection uses minimum pooled validation loss, matching the existing
seed-42 experiment exactly. Session-macro loss and worst-session R² remain
diagnostics in the new runs and cross-seed summary. The locked January test
split is neither loaded nor evaluated.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.indy_32ch.train_chronological_baseline import (
    ALPHAS,
    AXES,
    BIN_S,
    N_CHANNELS,
    OBSERVATION_BINS,
    OBSERVATION_SECONDS,
    WINDOW_BINS,
    combine_sessions,
    evaluate_split,
    fit_feature_std_floor,
    prepare_session,
)
from src.intent_decoder.data.indy import (
    load_model_data,
    load_session_manifest,
    top_firing_channels,
)
from src.intent_decoder.model.tcn_gru import build_net, causal_config

STRATEGIES = ("window", "session", "month")
STRATEGY_LABELS = {
    "window": "Window-weighted",
    "session": "Session-balanced",
    "month": "Month-balanced",
}
DEFAULT_TRAIN_SEEDS = (43, 44)
REUSED_SEED = 42
EXPECTED_SPLIT_COUNTS = {"train": 29, "validation": 4, "test": 4}
SEED42_METRICS_PATH = ROOT / "results" / "metrics" / "indy_32ch_sampling_comparison.json"
METRICS_PATH = ROOT / "results" / "metrics" / "indy_32ch_sampling_seed_sweep.json"
FIGURE_PATH = ROOT / "results" / "figures" / "indy_32ch_sampling_seed_sweep.png"


def session_month(session: str) -> str:
    """Return YYYY-MM from a canonical ``indy_YYYYMMDD_NN`` session name."""
    date = session.split("_")[1]
    return f"{date[:4]}-{date[4:6]}"


def balanced_allocations(
    items: list[str], total: int, rng: np.random.Generator
) -> dict[str, int]:
    """Allocate an exact total as evenly as possible across named items."""
    if not items:
        raise ValueError("Cannot balance an empty item list")
    if total < 0:
        raise ValueError("Allocation total cannot be negative")
    base, remainder = divmod(total, len(items))
    allocation = {item: base for item in items}
    if remainder:
        order = rng.permutation(len(items))[:remainder]
        for index in order:
            allocation[items[int(index)]] += 1
    return allocation


def draw_epoch_indices(
    strategy: str,
    sessions: list[str],
    session_lengths: dict[str, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    """Draw one fixed-size epoch and return its session/month exposure counts."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    if set(sessions) != set(session_lengths):
        raise ValueError("sessions and session_lengths must contain the same names")
    if any(session_lengths[session] <= 0 for session in sessions):
        raise ValueError("Every session must contain at least one training window")

    offsets: dict[str, int] = {}
    cursor = 0
    for session in sessions:
        offsets[session] = cursor
        cursor += session_lengths[session]
    epoch_size = cursor

    if strategy == "window":
        indices = rng.permutation(epoch_size).astype(np.int64)
        session_draws = dict(session_lengths)
    else:
        if strategy == "session":
            session_draws = balanced_allocations(sessions, epoch_size, rng)
        else:
            sessions_by_month: dict[str, list[str]] = {}
            for session in sessions:
                sessions_by_month.setdefault(session_month(session), []).append(session)
            month_draws = balanced_allocations(
                sorted(sessions_by_month), epoch_size, rng
            )
            session_draws = {}
            for month, month_total in month_draws.items():
                session_draws.update(
                    balanced_allocations(sessions_by_month[month], month_total, rng)
                )

        blocks = []
        for session in sessions:
            count = session_draws[session]
            local = rng.integers(0, session_lengths[session], size=count)
            blocks.append(local.astype(np.int64) + offsets[session])
        indices = np.concatenate(blocks)
        rng.shuffle(indices)

    month_counter: Counter[str] = Counter()
    for session, count in session_draws.items():
        month_counter[session_month(session)] += count
    month_draws = dict(sorted(month_counter.items()))

    if len(indices) != epoch_size:
        raise AssertionError("Every strategy must draw the same samples per epoch")
    if sum(session_draws.values()) != epoch_size:
        raise AssertionError("Session exposure counts do not match epoch size")
    if np.any(indices < 0) or np.any(indices >= epoch_size):
        raise AssertionError("Sampler produced an out-of-range window index")
    return indices, session_draws, month_draws


def summarize_values(values: list[float | int]) -> dict[str, float | int | list[float]]:
    """Return inspectable mean/sample-SD/range statistics."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty value list")
    return {
        "n": int(array.size),
        "values": [float(value) for value in array],
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def summarize_seed_sweep(
    per_seed: dict[str, dict],
    seeds: list[int],
    strategies: list[str],
    validation_sessions: list[str],
) -> dict[str, dict]:
    """Aggregate selected-checkpoint metrics across seeds for each strategy."""
    summary: dict[str, dict] = {}
    for strategy in strategies:
        arms = [per_seed[str(seed)]["strategies"][strategy] for seed in seeds]

        def collect(path: tuple[str, ...]) -> list[float]:
            values = []
            for arm in arms:
                value = arm
                for key in path:
                    value = value[key]
                values.append(float(value))
            return values

        by_session = {}
        for session in validation_sessions:
            by_session[session] = {
                "loss": summarize_values(
                    [arm["validation_by_session"][session]["loss"] for arm in arms]
                ),
                "r2_mean": summarize_values(
                    [arm["validation_by_session"][session]["r2_mean"] for arm in arms]
                ),
                "windows": int(arms[0]["validation_by_session"][session]["windows"]),
            }

        train_r2 = collect(("selected_checkpoint_metrics", "train", "r2_mean"))
        validation_r2 = collect(
            ("selected_checkpoint_metrics", "validation", "r2_mean")
        )
        summary[strategy] = {
            "label": STRATEGY_LABELS[strategy],
            "selected_epoch": summarize_values(
                [arm["selected_epoch"] for arm in arms]
            ),
            "selection_score": summarize_values(
                [arm["best_validation_loss"] for arm in arms]
            ),
            "selected_validation_macro_loss": summarize_values(
                [arm["selected_validation_macro_loss"] for arm in arms]
            ),
            "selected_validation_macro_r2_mean": summarize_values(
                [arm["selected_validation_macro_r2_mean"] for arm in arms]
            ),
            "selected_validation_worst_session_r2_mean": summarize_values(
                [arm["selected_validation_worst_session_r2_mean"] for arm in arms]
            ),
            "selected_train_loss": summarize_values(
                collect(("selected_checkpoint_metrics", "train", "loss"))
            ),
            "selected_train_r2_mean": summarize_values(train_r2),
            "selected_validation_loss": summarize_values(
                collect(("selected_checkpoint_metrics", "validation", "loss"))
            ),
            "selected_validation_r2_mean": summarize_values(validation_r2),
            "train_validation_r2_gap": summarize_values(
                [train - validation for train, validation in zip(train_r2, validation_r2)]
            ),
            "validation_by_session": by_session,
        }
    return summary


def load_reused_seed42(
    config: dict,
    strategies: list[str],
    split_sessions: dict[str, list[str]],
) -> dict:
    """Load and normalize the completed compatible seed-42 experiment."""
    if not SEED42_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Cannot reuse seed 42 because {SEED42_METRICS_PATH} is missing. "
            "Pass --no-reuse-seed42 and include 42 in --seeds to retrain it."
        )
    payload = json.loads(SEED42_METRICS_PATH.read_text(encoding="utf-8"))
    if payload.get("purpose") != "indy_32ch_sampling_comparison":
        raise ValueError("The seed-42 artifact has an unexpected purpose")
    if payload.get("device") != "cpu":
        raise ValueError("The reusable seed-42 artifact must be the CPU run")
    if payload.get("split", {}).get("train") != split_sessions["train"]:
        raise ValueError("The seed-42 training split does not match the active split")
    if payload.get("split", {}).get("validation") != split_sessions["validation"]:
        raise ValueError("The seed-42 validation split does not match the active split")

    comparable_fields = (
        "F",
        "H",
        "L",
        "dils",
        "bidir",
        "dropout",
        "lr",
        "wd",
        "epochs",
        "bs",
        "noise",
        "chdrop",
        "act",
        "n_out",
        "gradient_clip",
    )
    old_config = payload.get("config", {})
    mismatches = {
        field: {"existing": old_config.get(field), "requested": config.get(field)}
        for field in comparable_fields
        if old_config.get(field) != config.get(field)
    }
    if mismatches:
        raise ValueError(
            "The requested configuration is not comparable with reused seed 42: "
            f"{mismatches}. Pass --no-reuse-seed42 and train all requested seeds."
        )

    converted = {}
    for strategy in strategies:
        if strategy not in payload.get("strategies", {}):
            raise ValueError(f"Seed-42 artifact does not contain strategy={strategy}")
        old = payload["strategies"][strategy]
        validation_by_session = old["validation_by_session"]
        if set(validation_by_session) != set(split_sessions["validation"]):
            raise ValueError(
                f"Seed-42 validation sessions do not match for strategy={strategy}"
            )
        converted[strategy] = {
            "label": STRATEGY_LABELS[strategy],
            "selected_epoch": old["best_epoch_selected_on_validation_loss"],
            "best_validation_loss": old["best_validation_loss"],
            "selected_validation_macro_loss": float(
                np.mean(
                    [metrics["loss"] for metrics in validation_by_session.values()]
                )
            ),
            "selected_validation_macro_r2_mean": float(
                np.mean(
                    [metrics["r2_mean"] for metrics in validation_by_session.values()]
                )
            ),
            "selected_validation_worst_session_r2_mean": float(
                min(
                    metrics["r2_mean"]
                    for metrics in validation_by_session.values()
                )
            ),
            "selected_checkpoint_metrics": old["selected_checkpoint_metrics"],
            "validation_by_session": validation_by_session,
            "history": old["history"],
            "sampling_exposure": old["sampling_exposure"],
            "checkpoint": old["checkpoint"],
            "provenance": str(SEED42_METRICS_PATH.relative_to(ROOT)),
        }
    winner = min(
        strategies,
        key=lambda strategy: converted[strategy]["best_validation_loss"],
    )
    return {
        "seed": REUSED_SEED,
        "source": str(SEED42_METRICS_PATH.relative_to(ROOT)),
        "strategies": converted,
        "recommended_by_minimum_validation_loss": winner,
    }


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_TRAIN_SEEDS)
    )
    parser.add_argument(
        "--no-reuse-seed42",
        action="store_false",
        dest="reuse_seed42",
        help="Do not import the compatible completed CPU seed-42 experiment.",
    )
    parser.set_defaults(reuse_seed42=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--channel-dropout", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=STRATEGIES,
        default=list(STRATEGIES),
        help="Strategies to run; the default compares all three.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="cpu",
        help="CPU is the validated default; MPS is currently diagnostic only.",
    )
    return parser.parse_args()


def plot_seed_sweep(
    per_seed: dict[str, dict], seeds: list[int], strategies: list[str]
) -> None:
    """Plot mean curves with one-sample-SD bands across seeds."""
    import os

    cache = ROOT / "results" / "large" / ".matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"window": "#2368A2", "session": "#C58B18", "month": "#3A8D5D"}
    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.0), dpi=180)
    panels = (
        ("train_loss", "Full-train normalized MSE", "Loss"),
        ("validation_loss", "Validation normalized MSE", "Loss"),
        ("train_r2_mean", "Full-train mean velocity R²", "R²"),
        ("validation_r2_mean", "Validation mean velocity R²", "R²"),
    )
    for axis, (field, title, ylabel) in zip(axes.ravel(), panels):
        for strategy in strategies:
            histories = [
                per_seed[str(seed)]["strategies"][strategy]["history"]
                for seed in seeds
            ]
            epochs = np.asarray([row["epoch"] for row in histories[0]])
            values = np.asarray(
                [[row[field] for row in history] for history in histories],
                dtype=np.float64,
            )
            mean = values.mean(axis=0)
            spread = values.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(mean)
            axis.plot(
                epochs,
                mean,
                label=STRATEGY_LABELS[strategy],
                color=colors[strategy],
                linewidth=2.0,
            )
            axis.fill_between(
                epochs,
                mean - spread,
                mean + spread,
                color=colors[strategy],
                alpha=0.16,
                linewidth=0,
            )
        if "r2" in field:
            axis.axhline(0.0, color="#9AA2AC", linewidth=1.0, linestyle="--")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D8DDE3", linewidth=0.8, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    figure.suptitle("Indy 32-channel causal decoder: seed × sampling sweep")
    figure.text(
        0.5,
        0.01,
        f"Lines are mean across seeds {seeds}; bands are ±1 sample SD; test is locked.",
        ha="center",
        color="#5B6470",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.96))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    import torch
    import torch.nn as nn

    args = parse_args()
    if args.epochs <= 0 or args.threads <= 0:
        raise ValueError("--epochs and --threads must be positive")
    if args.learning_rate <= 0 or args.gradient_clip <= 0:
        raise ValueError("--learning-rate and --gradient-clip must be positive")
    if args.noise < 0 or not 0 <= args.channel_dropout < 1:
        raise ValueError("Invalid augmentation arguments")
    if len(set(args.strategies)) != len(args.strategies):
        raise ValueError("Do not repeat a sampling strategy")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must contain unique integer seeds")

    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    if device.type == "mps":
        print(
            "WARNING: MPS previously produced non-equivalent exploding gradients; "
            "these results must not be used for model selection.",
            flush=True,
        )
    started = time.time()

    manifest = load_session_manifest()
    split_sessions = {
        name: list(manifest["chronological_split"][name])
        for name in ("train", "validation", "test")
    }
    actual_counts = {name: len(value) for name, value in split_sessions.items()}
    if actual_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            f"Expected chronological split {EXPECTED_SPLIT_COUNTS}, found {actual_counts}"
        )

    print("=== Indy 32-channel causal seed × sampling sweep ===")
    print(
        f"sessions: train={actual_counts['train']} | "
        f"validation={actual_counts['validation']} | test={actual_counts['test']} LOCKED"
    )
    reuse_seed42 = args.reuse_seed42 and REUSED_SEED not in args.seeds
    all_seeds = ([REUSED_SEED] if reuse_seed42 else []) + list(args.seeds)
    print(
        f"train seeds={args.seeds} | reused seeds="
        f"{[REUSED_SEED] if reuse_seed42 else []} | "
        f"strategies={','.join(args.strategies)} | epochs each={args.epochs} | "
        f"new arms={len(args.seeds) * len(args.strategies)} | device={device}"
    )
    print(
        "policy: weights=train only; checkpoint=minimum pooled validation loss; "
        "test=not loaded"
    )

    active_sessions = split_sessions["train"] + split_sessions["validation"]
    loaded = {session: load_model_data(session) for session in active_sessions}
    training_loaded = {
        session: loaded[session] for session in split_sessions["train"]
    }
    channels = top_firing_channels(
        training_loaded,
        N_CHANNELS,
        observation_bins=OBSERVATION_BINS,
    )
    feature_std_floor = fit_feature_std_floor(training_loaded, channels)
    prepared = {
        session: prepare_session(loaded, session, channels, feature_std_floor)
        for session in active_sessions
    }
    combined = {
        split: combine_sessions(prepared, split_sessions[split])
        for split in ("train", "validation")
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

    splits = {name: pack(*arrays) for name, arrays in combined.items()}
    validation_session_splits = {
        session: pack(*prepared[session]) for session in split_sessions["validation"]
    }
    session_lengths = {
        session: len(prepared[session][0]) for session in split_sessions["train"]
    }
    print(f"selected channels: {channels.tolist()}")
    print(
        "training-derived feature std floor: "
        f"min={feature_std_floor.min():.5f} | "
        f"median={np.median(feature_std_floor):.5f} | "
        f"max={feature_std_floor.max():.5f}"
    )
    print(
        f"full windows: train={len(splits['train']['x'])} | "
        f"validation={len(splits['validation']['x'])}"
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
    train_x = torch.from_numpy(splits["train"]["x"])
    train_y_norm = torch.from_numpy(splits["train"]["y_norm"])
    mse = nn.MSELoss()
    per_seed: dict[str, dict] = {}
    if reuse_seed42:
        per_seed[str(REUSED_SEED)] = load_reused_seed42(
            config, args.strategies, split_sessions
        )
        print(
            f"reused seed {REUSED_SEED}: {SEED42_METRICS_PATH.relative_to(ROOT)}"
        )
    n_parameters = 0

    for seed in args.seeds:
        print(f"\n######## seed {seed} ########")
        np.random.seed(seed)
        torch.manual_seed(seed)
        template = build_net(config, splits["train"]["x"].shape[1])
        initial_state = copy.deepcopy(template.state_dict())
        if not n_parameters:
            n_parameters = sum(parameter.numel() for parameter in template.parameters())
            print(f"model parameters: {n_parameters:,}")
        del template
        seed_results: dict[str, dict] = {}

        for strategy in args.strategies:
            print(f"\n=== seed {seed} | {STRATEGY_LABELS[strategy]} ===")
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            net = build_net(config, splits["train"]["x"].shape[1]).to(device)
            net.load_state_dict(initial_state)
            optimizer = torch.optim.AdamW(
                net.parameters(), lr=config["lr"], weight_decay=config["wd"]
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, args.epochs
            )
            history: list[dict] = []
            total_session_draws: Counter[str] = Counter()
            total_month_draws: Counter[str] = Counter()
            best_validation_loss = np.inf
            best_epoch = 0
            best_state = None

            for epoch in range(1, args.epochs + 1):
                indices, session_draws, month_draws = draw_epoch_indices(
                    strategy,
                    split_sessions["train"],
                    session_lengths,
                    rng,
                )
                total_session_draws.update(session_draws)
                total_month_draws.update(month_draws)
                if epoch == 1:
                    shares = {
                        month: count / len(indices)
                        for month, count in month_draws.items()
                    }
                    print(
                        f"epoch samples: {len(indices)} | month shares="
                        + ", ".join(
                            f"{key}:{value:.1%}" for key, value in shares.items()
                        )
                    )

                net.train()
                optimization_error = 0.0
                optimization_values = 0
                gradient_norm_sum = 0.0
                gradient_norm_max = 0.0
                batch_count = 0
                for start in range(0, len(indices), config["bs"]):
                    batch_indices = indices[start : start + config["bs"]]
                    x_batch = train_x[batch_indices].to(device)
                    y_batch = train_y_norm[batch_indices].to(device)
                    if config["noise"] > 0:
                        x_batch = x_batch + config["noise"] * torch.randn_like(x_batch)
                    if config["chdrop"] > 0:
                        mask = (
                            torch.rand(
                                x_batch.shape[0],
                                x_batch.shape[1],
                                1,
                                device=device,
                            )
                            > config["chdrop"]
                        ).float()
                        x_batch = x_batch * mask / (1.0 - config["chdrop"])

                    optimizer.zero_grad(set_to_none=True)
                    loss = mse(net(x_batch), y_batch)
                    loss.backward()
                    gradient_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            net.parameters(), args.gradient_clip
                        ).item()
                    )
                    optimizer.step()
                    optimization_error += float(loss.item()) * y_batch.numel()
                    optimization_values += y_batch.numel()
                    gradient_norm_sum += gradient_norm
                    gradient_norm_max = max(gradient_norm_max, gradient_norm)
                    batch_count += 1

                row: dict[str, float | int | None] = {
                    "epoch": epoch,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "optimization_loss": optimization_error / optimization_values,
                    "gradient_norm_mean_before_clip": gradient_norm_sum / batch_count,
                    "gradient_norm_max_before_clip": gradient_norm_max,
                }
                for split_name in ("train", "validation"):
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

                epoch_session_metrics = {}
                for session, session_split in validation_session_splits.items():
                    session_loss, session_r2 = evaluate_split(
                        net,
                        session_split,
                        target_mean,
                        target_std,
                        config["bs"],
                        device,
                    )
                    epoch_session_metrics[session] = {
                        "loss": session_loss,
                        "r2_mean": float(session_r2.mean()),
                    }
                macro_loss = float(
                    np.mean(
                        [metrics["loss"] for metrics in epoch_session_metrics.values()]
                    )
                )
                macro_r2 = float(
                    np.mean(
                        [
                            metrics["r2_mean"]
                            for metrics in epoch_session_metrics.values()
                        ]
                    )
                )
                worst_session_r2 = float(
                    min(
                        metrics["r2_mean"]
                        for metrics in epoch_session_metrics.values()
                    )
                )
                row["validation_macro_loss"] = macro_loss
                row["validation_macro_r2_mean"] = macro_r2
                row["validation_worst_session_r2_mean"] = worst_session_r2
                row["checkpoint_selection_score"] = row["validation_loss"]
                history.append(row)

                improved = row["validation_loss"] < best_validation_loss
                if improved:
                    best_validation_loss = float(row["validation_loss"])
                    best_epoch = epoch
                    best_state = copy.deepcopy(net.state_dict())
                print(
                    f"epoch {epoch:02d}/{args.epochs} | "
                    f"opt={row['optimization_loss']:.5f} | "
                    f"loss train={row['train_loss']:.5f} "
                    f"validation={row['validation_loss']:.5f} "
                    f"macro={macro_loss:.5f} | "
                    f"R2 train={row['train_r2_mean']:+.4f} "
                    f"validation={row['validation_r2_mean']:+.4f} "
                    f"macro={macro_r2:+.4f} worst={worst_session_r2:+.4f} | "
                    f"select={row['validation_loss']:.5f} | "
                    f"grad={row['gradient_norm_mean_before_clip']:.3f}/"
                    f"{row['gradient_norm_max_before_clip']:.3f}"
                    + (" *best validation*" if improved else ""),
                    flush=True,
                )
                scheduler.step()

            if best_state is None:
                raise RuntimeError(f"No checkpoint selected for seed={seed} {strategy}")
            net.load_state_dict(best_state)
            selected_metrics = {}
            for split_name in ("train", "validation"):
                split_loss, split_r2 = evaluate_split(
                    net,
                    splits[split_name],
                    target_mean,
                    target_std,
                    config["bs"],
                    device,
                )
                selected_metrics[split_name] = {
                    "loss": split_loss,
                    "r2_x": float(split_r2[0]),
                    "r2_y": float(split_r2[1]),
                    "r2_mean": float(split_r2.mean()),
                }

            validation_by_session = {}
            for session, session_split in validation_session_splits.items():
                session_loss, session_r2 = evaluate_split(
                    net,
                    session_split,
                    target_mean,
                    target_std,
                    config["bs"],
                    device,
                )
                validation_by_session[session] = {
                    "windows": len(session_split["x"]),
                    "loss": session_loss,
                    "r2_x": float(session_r2[0]),
                    "r2_y": float(session_r2[1]),
                    "r2_mean": float(session_r2.mean()),
                }
            selected_macro_loss = float(
                np.mean(
                    [metrics["loss"] for metrics in validation_by_session.values()]
                )
            )
            selected_macro_r2 = float(
                np.mean(
                    [metrics["r2_mean"] for metrics in validation_by_session.values()]
                )
            )
            selected_worst_r2 = float(
                min(
                    metrics["r2_mean"]
                    for metrics in validation_by_session.values()
                )
            )

            checkpoint_path = (
                ROOT
                / "results"
                / "large"
                / f"indy_32ch_sampling_seed{seed}_{strategy}_checkpoint.pt"
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "purpose": "indy_32ch_sampling_seed_sweep",
                    "seed": seed,
                    "strategy": strategy,
                    "model_state": net.state_dict(),
                    "config": config,
                    "channels": channels.tolist(),
                    "target_mean": target_mean.tolist(),
                    "target_std": target_std.tolist(),
                    "feature_std_floor": feature_std_floor[:, 0].tolist(),
                    "train_sessions": split_sessions["train"],
                    "validation_sessions": split_sessions["validation"],
                    "test_policy": "locked_not_loaded",
                    "observation_seconds": OBSERVATION_SECONDS,
                    "checkpoint_epoch": best_epoch,
                    "selection_policy": "minimum_pooled_validation_loss",
                    "best_selection_score": best_validation_loss,
                },
                checkpoint_path,
            )

            total_draws = args.epochs * len(train_x)
            seed_results[strategy] = {
                "label": STRATEGY_LABELS[strategy],
                "selected_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "selected_validation_macro_loss": selected_macro_loss,
                "selected_validation_macro_r2_mean": selected_macro_r2,
                "selected_validation_worst_session_r2_mean": selected_worst_r2,
                "selected_checkpoint_metrics": selected_metrics,
                "validation_by_session": validation_by_session,
                "history": history,
                "sampling_exposure": {
                    "total_draws": total_draws,
                    "by_session": dict(total_session_draws),
                    "by_month": dict(sorted(total_month_draws.items())),
                },
                "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            }
            print(
                f"selected epoch={best_epoch} | "
                f"train R2={selected_metrics['train']['r2_mean']:+.4f} | "
                f"validation R2={selected_metrics['validation']['r2_mean']:+.4f} | "
                f"macro loss={selected_macro_loss:.5f} | "
                f"macro/worst R2={selected_macro_r2:+.4f}/{selected_worst_r2:+.4f}"
            )

        seed_winner = min(
            args.strategies,
            key=lambda strategy: seed_results[strategy][
                "best_validation_loss"
            ],
        )
        per_seed[str(seed)] = {
            "seed": seed,
            "strategies": seed_results,
            "recommended_by_minimum_validation_loss": seed_winner,
        }

    strategy_summary = summarize_seed_sweep(
        per_seed,
        all_seeds,
        args.strategies,
        split_sessions["validation"],
    )
    winner = min(
        args.strategies,
        key=lambda strategy: strategy_summary[strategy]["selected_validation_loss"]["mean"],
    )
    wins_by_strategy = Counter(
        per_seed[str(seed)]["recommended_by_minimum_validation_loss"]
        for seed in all_seeds
    )

    payload = {
        "purpose": "indy_32ch_sampling_seed_sweep",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "config": config,
        "device": str(device),
        "seeds": all_seeds,
        "trained_seeds": args.seeds,
        "reused_seeds": [REUSED_SEED] if reuse_seed42 else [],
        "selection": {
            "primary_metric": "pooled_validation_normalized_mse",
            "smoothing": "none",
            "reason": (
                "matches the completed seed-42 protocol so all three seeds are "
                "directly comparable; session-macro and worst-session metrics "
                "are retained as diagnostics"
            ),
        },
        "fair_comparison_controls": {
            "identical_initial_state_within_each_seed": True,
            "independent_initial_state_across_seeds": True,
            "every_strategy_uses_every_seed": True,
            "identical_samples_per_epoch": len(train_x),
            "identical_epochs": args.epochs,
            "identical_optimizer_and_preprocessing": True,
            "only_within_seed_difference": "training_window_sampling_distribution",
        },
        "split": {
            "train": split_sessions["train"],
            "validation": split_sessions["validation"],
            "test": "LOCKED_NOT_LOADED",
        },
        "data_protocol": {
            "bin_seconds": BIN_S,
            "window_bins": WINDOW_BINS,
            "observation_seconds": OBSERVATION_SECONDS,
            "features": ["counts", "causal_ewma_0.1"],
            "velocity_axes": AXES.tolist(),
            "feature_normalization": "per_session_first_60_seconds_frozen",
            "feature_std_floor": "training_prefix_10th_percentile_positive_scale",
            "target_normalization": "training_windows_only",
            "channels": channels.tolist(),
        },
        "n_parameters": n_parameters,
        "window_counts": {
            "train": len(splits["train"]["x"]),
            "validation": len(splits["validation"]["x"]),
        },
        "per_seed": per_seed,
        "strategy_summary": strategy_summary,
        "wins_by_strategy": {
            strategy: int(wins_by_strategy.get(strategy, 0))
            for strategy in args.strategies
        },
        "recommended_by_mean_validation_loss": winner,
        "test_policy": (
            "Do not evaluate January until sampling and all hyperparameters are frozen."
        ),
        "artifacts": {"figure": str(FIGURE_PATH.relative_to(ROOT))},
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_seed_sweep(per_seed, all_seeds, args.strategies)

    print("\n=== multi-seed validation-only comparison ===")
    for strategy in args.strategies:
        result = strategy_summary[strategy]
        val_r2 = result["selected_validation_r2_mean"]
        val_loss = result["selected_validation_loss"]
        macro_loss = result["selected_validation_macro_loss"]
        macro_r2 = result["selected_validation_macro_r2_mean"]
        worst_r2 = result["selected_validation_worst_session_r2_mean"]
        gap = result["train_validation_r2_gap"]
        epochs = result["selected_epoch"]
        print(
            f"{STRATEGY_LABELS[strategy]:18s} | "
            f"epoch={epochs['mean']:.1f}±{epochs['std']:.1f} | "
            f"val loss={val_loss['mean']:.5f}±{val_loss['std']:.5f} | "
            f"val R2={val_r2['mean']:+.4f}±{val_r2['std']:.4f} | "
            f"macro loss={macro_loss['mean']:.5f}±{macro_loss['std']:.5f} | "
            f"macro/worst R2={macro_r2['mean']:+.4f}/"
            f"{worst_r2['mean']:+.4f} | "
            f"gap={gap['mean']:+.4f}"
        )
    print(
        "recommended by mean validation loss: "
        f"{STRATEGY_LABELS[winner]}"
    )
    print(
        "per-seed wins: "
        + ", ".join(
            f"{STRATEGY_LABELS[strategy]}={wins_by_strategy.get(strategy, 0)}"
            for strategy in args.strategies
        )
    )
    print("test: LOCKED and not loaded")
    print(f"metrics: {METRICS_PATH}")
    print(f"figure: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
