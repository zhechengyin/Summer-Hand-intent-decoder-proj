#!/usr/bin/env python
"""Optuna Phase-1 sweep for the session-balanced causal Indy decoder.

This study tunes learning rate, AdamW weight decay, and pre-GRU dropout jointly.
Every trial uses the same model initialization, session-balanced epoch sampler,
training/validation data, causal preprocessing, 20-epoch default budget, cosine
schedule, batch size, and gradient clipping. The January test split is recorded
as locked but is never loaded.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import Counter
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

METRICS_PATH = ROOT / "results" / "metrics" / "indy_32ch_phase1_optuna.json"
FIGURE_PATH = ROOT / "results" / "figures" / "indy_32ch_phase1_optuna.png"
CHECKPOINT_PATH = (
    ROOT / "results" / "large" / "indy_32ch_phase1_best_checkpoint.pt"
)
STORAGE_PATH = ROOT / "results" / "large" / "indy_32ch_phase1_optuna.db"

BASELINE_PARAMS = {
    "learning_rate": 3e-4,
    "weight_decay": 1e-3,
    "dropout": 0.30,
}
SEARCH_SPACE = {
    "learning_rate": {"low": 5e-5, "high": 1e-3, "scale": "log"},
    "weight_decay": {"low": 1e-6, "high": 3e-2, "scale": "log"},
    "dropout": {"low": 0.10, "high": 0.50, "step": 0.05},
}


def stack_windows(
    windows: list[dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Stack model windows from one session."""
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

    mean, local_std = fit_feature_stats(features, observation_bins=OBSERVATION_BINS)
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
    """Fit robust feature-scale floors using training prefixes only."""
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
    """Concatenate complete-session window arrays in manifest order."""
    return (
        np.concatenate([prepared[session][0] for session in sessions], axis=0),
        np.concatenate([prepared[session][1] for session in sessions], axis=0),
    )


def pack_split(
    x: np.ndarray,
    y: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> dict[str, np.ndarray]:
    """Store raw-unit targets and their training-derived normalized form."""
    return {
        "x": x,
        "y": y,
        "y_norm": ((y - target_mean) / target_std).astype(np.float32),
    }


def evaluate_split(
    net,
    split: dict[str, np.ndarray],
    target_mean: np.ndarray,
    target_std: np.ndarray,
    batch_size: int,
    device,
) -> tuple[float, np.ndarray]:
    """Return normalized-target MSE and raw-unit R-squared for x and y."""
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


def draw_session_balanced_indices(
    sessions: list[str],
    session_lengths: dict[str, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    """Draw one fixed-size epoch with equal expected exposure per session."""
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
    session_draws = balanced_allocations(sessions, epoch_size, rng)

    blocks = []
    for session in sessions:
        local = rng.integers(0, session_lengths[session], size=session_draws[session])
        blocks.append(local.astype(np.int64) + offsets[session])
    indices = np.concatenate(blocks)
    rng.shuffle(indices)

    month_draws: Counter[str] = Counter()
    for session, count in session_draws.items():
        date = session.split("_")[1]
        month_draws[f"{date[:4]}-{date[4:6]}"] += count

    if len(indices) != epoch_size or sum(session_draws.values()) != epoch_size:
        raise AssertionError("Session-balanced sampler changed the epoch size")
    if np.any(indices < 0) or np.any(indices >= epoch_size):
        raise AssertionError("Sampler produced an out-of-range window index")
    return indices, session_draws, dict(sorted(month_draws.items()))


def choose_device(requested: str):
    """Choose a validated model-selection device; Apple MPS is excluded."""
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--study-name", default="indy_32ch_phase1")
    parser.add_argument("--storage-path", type=Path, default=STORAGE_PATH)
    parser.add_argument("--timeout-hours", type=float)
    parser.add_argument("--sampler-startup-trials", type=int, default=8)
    parser.add_argument("--pruner-startup-trials", type=int, default=8)
    parser.add_argument("--pruner-warmup-epochs", type=int, default=8)
    parser.add_argument("--no-pruning", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate data preparation and model construction, then exit.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="CPU is the validated Mac path; auto selects CUDA or CPU, never MPS.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value):
    """Convert NumPy/path values into JSON-compatible objects."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def report_path(path: Path) -> str:
    """Prefer a repository-relative artifact path, allowing external overrides."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def trial_to_record(trial) -> dict:
    duration = None
    if trial.datetime_start is not None and trial.datetime_complete is not None:
        duration = (trial.datetime_complete - trial.datetime_start).total_seconds()
    return {
        "number": trial.number,
        "state": trial.state.name,
        "value": trial.value,
        "params": dict(trial.params),
        "datetime_start": (
            trial.datetime_start.isoformat() if trial.datetime_start else None
        ),
        "datetime_complete": (
            trial.datetime_complete.isoformat() if trial.datetime_complete else None
        ),
        "duration_seconds": duration,
        "intermediate_values": {
            str(step): value for step, value in trial.intermediate_values.items()
        },
        "metrics": dict(trial.user_attrs),
    }


def completed_trials(study) -> list:
    import optuna

    return [
        trial
        for trial in study.get_trials(deepcopy=False)
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and math.isfinite(trial.value)
    ]


def write_metrics(study, context: dict, started_at: str) -> None:
    complete = completed_trials(study)
    best = min(complete, key=lambda trial: trial.value) if complete else None
    payload = {
        "purpose": "indy_32ch_phase1_optuna",
        "generated_at_utc": utc_now(),
        "study_started_at_utc": started_at,
        "study_name": study.study_name,
        "direction": "minimize",
        "primary_metric": "pooled_validation_normalized_mse",
        "search_space": SEARCH_SPACE,
        "baseline_params": BASELINE_PARAMS,
        "fixed_protocol": context["fixed_protocol"],
        "split": context["split"],
        "data_protocol": context["data_protocol"],
        "n_parameters": context["n_parameters"],
        "window_counts": context["window_counts"],
        "best_trial": trial_to_record(best) if best else None,
        "trials": [trial_to_record(trial) for trial in study.trials],
        "trial_counts": {
            state: sum(trial.state.name == state for trial in study.trials)
            for state in ("COMPLETE", "PRUNED", "FAIL", "RUNNING", "WAITING")
        },
        "test_policy": "January test is locked and was not loaded.",
        "artifacts": {
            "study_database": report_path(context["storage_path"]),
            "best_checkpoint": report_path(CHECKPOINT_PATH),
            "figure": report_path(FIGURE_PATH),
        },
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = METRICS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    temporary.replace(METRICS_PATH)


def plot_study(study) -> None:
    complete = sorted(completed_trials(study), key=lambda trial: trial.number)
    if not complete:
        return

    import os

    cache = ROOT / "results" / "large" / ".matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    numbers = np.asarray([trial.number for trial in complete])
    values = np.asarray([trial.value for trial in complete], dtype=np.float64)
    running_best = np.minimum.accumulate(values)
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), dpi=180)

    objective_axis = axes[0, 0]
    objective_axis.plot(numbers, values, "o", color="#2368A2", label="Trial")
    objective_axis.plot(
        numbers, running_best, color="#C58B18", linewidth=2.0, label="Running best"
    )
    objective_axis.set_title("Completed-trial validation loss")
    objective_axis.set_xlabel("Trial number")
    objective_axis.set_ylabel("Pooled normalized MSE")
    objective_axis.legend(frameon=False)

    parameter_specs = [
        ("learning_rate", "Learning rate", True),
        ("weight_decay", "AdamW weight decay", True),
        ("dropout", "Pre-GRU dropout", False),
    ]
    for axis, (field, label, logarithmic) in zip(axes.flat[1:], parameter_specs):
        x = np.asarray([trial.params[field] for trial in complete], dtype=np.float64)
        points = axis.scatter(
            x,
            values,
            c=numbers,
            cmap="viridis",
            edgecolor="#263238",
            linewidth=0.4,
        )
        if logarithmic:
            axis.set_xscale("log")
        axis.set_title(f"Validation loss vs {label.lower()}")
        axis.set_xlabel(label)
        axis.set_ylabel("Pooled normalized MSE")
        figure.colorbar(points, ax=axis, label="Trial number")

    for axis in axes.flat:
        axis.grid(axis="y", color="#D8DDE3", linewidth=0.8, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Indy 32-channel Phase-1 Optuna sweep")
    figure.text(
        0.5,
        0.01,
        "Session-balanced training; validation selects trials; January test is locked.",
        ha="center",
        color="#5B6470",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    try:
        import optuna
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Optuna is not installed. Run: pip install -r requirements.txt"
        ) from error
    import torch
    import torch.nn as nn

    args = parse_args()
    if args.n_trials <= 0 or args.epochs <= 0 or args.threads <= 0:
        raise ValueError("--n-trials, --epochs, and --threads must be positive")
    if args.timeout_hours is not None and args.timeout_hours <= 0:
        raise ValueError("--timeout-hours must be positive")
    if not 1 <= args.pruner_warmup_epochs <= args.epochs:
        raise ValueError("--pruner-warmup-epochs must be within the epoch budget")

    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    storage_path = args.storage_path.expanduser().resolve()
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_path.as_posix()}"
    started = time.time()
    started_at = utc_now()

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

    print("=== Indy 32-channel Phase-1 Optuna sweep ===")
    print(
        f"sessions: train={actual_counts['train']} | "
        f"validation={actual_counts['validation']} | test={actual_counts['test']} LOCKED"
    )
    print(
        f"trials to add={args.n_trials} | epochs/trial={args.epochs} | "
        f"seed={args.seed} | device={device} | sampler=session-balanced"
    )
    print(
        "search: learning_rate=5e-5..1e-3 log | "
        "weight_decay=1e-6..3e-2 log | dropout=0.10..0.50 step 0.05"
    )
    print(
        "fixed: AdamW | batch=32 | cosine schedule | gradient_clip=1 | "
        "noise=0 | channel_dropout=0"
    )
    print("policy: train updates weights; validation selects/prunes; test not loaded")

    # Load only train and validation artifacts. Merely reading test session names
    # from the versioned manifest does not load any January arrays.
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
    splits = {
        name: pack_split(x, y, target_mean, target_std)
        for name, (x, y) in combined.items()
    }
    validation_session_splits = {
        session: pack_split(*prepared[session], target_mean, target_std)
        for session in split_sessions["validation"]
    }
    session_lengths = {
        session: len(prepared[session][0]) for session in split_sessions["train"]
    }
    train_x = torch.from_numpy(splits["train"]["x"])
    train_y_norm = torch.from_numpy(splits["train"]["y_norm"])

    fixed_config = {
        **causal_config(),
        "epochs": args.epochs,
        "n_out": 2,
        "bs": 32,
        "noise": 0.0,
        "chdrop": 0.0,
        "gradient_clip": 1.0,
        "cosine": True,
    }
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    template = build_net(fixed_config, splits["train"]["x"].shape[1])
    initial_state = copy.deepcopy(template.state_dict())
    n_parameters = sum(parameter.numel() for parameter in template.parameters())
    del template

    print(f"selected channels: {channels.tolist()}")
    print(
        "training-derived feature std floor: "
        f"min={feature_std_floor.min():.5f} | "
        f"median={np.median(feature_std_floor):.5f} | "
        f"max={feature_std_floor.max():.5f}"
    )
    print(
        f"windows: train={len(splits['train']['x'])} | "
        f"validation={len(splits['validation']['x'])} | "
        f"parameters={n_parameters:,}"
    )
    if args.validate_only:
        print("validation-only check complete; no study or checkpoint was written")
        print("test: LOCKED and not loaded")
        return

    protocol_signature = {
        "phase": 1,
        "seed": args.seed,
        "epochs": args.epochs,
        "sampler": "session_balanced",
        "batch_size": fixed_config["bs"],
        "gradient_clip": fixed_config["gradient_clip"],
        "scheduler": "cosine_annealing",
        "noise": 0.0,
        "channel_dropout": 0.0,
        "train_sessions": split_sessions["train"],
        "validation_sessions": split_sessions["validation"],
        "channels": channels.tolist(),
        "search_space": SEARCH_SPACE,
    }
    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        n_startup_trials=args.sampler_startup_trials,
        multivariate=True,
    )
    pruner = (
        optuna.pruners.NopPruner()
        if args.no_pruning
        else optuna.pruners.MedianPruner(
            n_startup_trials=args.pruner_startup_trials,
            n_warmup_steps=args.pruner_warmup_epochs,
            interval_steps=1,
        )
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    saved_signature = study.user_attrs.get("protocol_signature")
    if saved_signature is not None and saved_signature != protocol_signature:
        raise RuntimeError(
            "The existing Optuna study uses a different data/training protocol. "
            "Choose a new --study-name or --storage-path instead of mixing trials."
        )
    study.set_user_attr("protocol_signature", protocol_signature)
    study.set_user_attr("test_policy", "locked_not_loaded")
    if not study.trials:
        study.enqueue_trial(BASELINE_PARAMS)

    context = {
        "storage_path": storage_path,
        "fixed_protocol": {
            **protocol_signature,
            "device": str(device),
            "optimizer": "AdamW",
            "checkpoint_selection": "minimum_pooled_validation_normalized_mse",
            "pruning": (
                "disabled"
                if args.no_pruning
                else {
                    "type": "MedianPruner",
                    "startup_trials": args.pruner_startup_trials,
                    "warmup_epochs": args.pruner_warmup_epochs,
                }
            ),
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
            "multiscale_alpha": list(ALPHAS),
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
    }
    mse = nn.MSELoss()

    def objective(trial) -> float:
        learning_rate = trial.suggest_float(
            "learning_rate",
            SEARCH_SPACE["learning_rate"]["low"],
            SEARCH_SPACE["learning_rate"]["high"],
            log=True,
        )
        weight_decay = trial.suggest_float(
            "weight_decay",
            SEARCH_SPACE["weight_decay"]["low"],
            SEARCH_SPACE["weight_decay"]["high"],
            log=True,
        )
        dropout = trial.suggest_float(
            "dropout",
            SEARCH_SPACE["dropout"]["low"],
            SEARCH_SPACE["dropout"]["high"],
            step=SEARCH_SPACE["dropout"]["step"],
        )
        config = {
            **fixed_config,
            "lr": learning_rate,
            "wd": weight_decay,
            "dropout": dropout,
        }
        print(
            f"\n=== trial {trial.number:03d} | lr={learning_rate:.3g} | "
            f"wd={weight_decay:.3g} | dropout={dropout:.2f} ===",
            flush=True,
        )

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        rng = np.random.default_rng(args.seed)
        net = build_net(config, splits["train"]["x"].shape[1]).to(device)
        net.load_state_dict(initial_state)
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs
        )
        history = []
        best_validation_loss = np.inf
        best_epoch = 0
        best_state = None

        for epoch in range(1, args.epochs + 1):
            indices, _, month_draws = draw_session_balanced_indices(
                split_sessions["train"], session_lengths, rng
            )
            if epoch == 1:
                shares = {
                    month: count / len(indices) for month, count in month_draws.items()
                }
                print(
                    "month exposure: "
                    + ", ".join(
                        f"{month}:{share:.1%}" for month, share in shares.items()
                    ),
                    flush=True,
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
                optimizer.zero_grad(set_to_none=True)
                loss = mse(net(x_batch), y_batch)
                loss.backward()
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        net.parameters(), config["gradient_clip"]
                    ).item()
                )
                optimizer.step()
                optimization_error += float(loss.item()) * y_batch.numel()
                optimization_values += y_batch.numel()
                gradient_norm_sum += gradient_norm
                gradient_norm_max = max(gradient_norm_max, gradient_norm)
                batch_count += 1

            row = {
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
            row["validation_macro_loss"] = float(
                np.mean([item["loss"] for item in epoch_session_metrics.values()])
            )
            row["validation_macro_r2_mean"] = float(
                np.mean([item["r2_mean"] for item in epoch_session_metrics.values()])
            )
            row["validation_worst_session_r2_mean"] = float(
                min(item["r2_mean"] for item in epoch_session_metrics.values())
            )
            history.append(row)

            improved = row["validation_loss"] < best_validation_loss
            if improved:
                best_validation_loss = float(row["validation_loss"])
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in net.state_dict().items()
                }

            print(
                f"trial {trial.number:03d} | epoch {epoch:02d}/{args.epochs} | "
                f"opt={row['optimization_loss']:.5f} | "
                f"loss train={row['train_loss']:.5f} "
                f"validation={row['validation_loss']:.5f} "
                f"macro={row['validation_macro_loss']:.5f} | "
                f"R2 train={row['train_r2_mean']:+.4f} "
                f"validation={row['validation_r2_mean']:+.4f} "
                f"macro={row['validation_macro_r2_mean']:+.4f} "
                f"worst={row['validation_worst_session_r2_mean']:+.4f} | "
                f"grad={row['gradient_norm_mean_before_clip']:.3f}/"
                f"{row['gradient_norm_max_before_clip']:.3f}"
                + (" *best*" if improved else ""),
                flush=True,
            )

            trial.report(float(row["validation_loss"]), step=epoch)
            scheduler.step()
            if trial.should_prune():
                trial.set_user_attr("best_epoch_before_prune", best_epoch)
                trial.set_user_attr(
                    "best_validation_loss_before_prune", best_validation_loss
                )
                trial.set_user_attr("history", history)
                print(
                    f"trial {trial.number:03d} pruned after epoch {epoch}; "
                    f"best validation loss={best_validation_loss:.5f}",
                    flush=True,
                )
                raise optuna.TrialPruned()

        if best_state is None:
            raise RuntimeError(f"Trial {trial.number} did not select a checkpoint")
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
        macro_loss = float(
            np.mean([item["loss"] for item in validation_by_session.values()])
        )
        macro_r2 = float(
            np.mean([item["r2_mean"] for item in validation_by_session.values()])
        )
        worst_r2 = float(
            min(item["r2_mean"] for item in validation_by_session.values())
        )
        trial.set_user_attr("selected_epoch", best_epoch)
        trial.set_user_attr("selected_checkpoint_metrics", selected_metrics)
        trial.set_user_attr("validation_macro_loss", macro_loss)
        trial.set_user_attr("validation_macro_r2_mean", macro_r2)
        trial.set_user_attr("validation_worst_session_r2_mean", worst_r2)
        trial.set_user_attr("validation_by_session", validation_by_session)
        trial.set_user_attr("history", history)

        previous = completed_trials(study)
        previous_best = min(
            (completed.value for completed in previous), default=np.inf
        )
        if best_validation_loss < previous_best:
            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "purpose": "indy_32ch_phase1_optuna_best",
                    "created_at_utc": utc_now(),
                    "trial_number": trial.number,
                    "model_state": best_state,
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
                    "selection_policy": "minimum_pooled_validation_normalized_mse",
                    "best_validation_loss": best_validation_loss,
                    "validation_macro_r2_mean": macro_r2,
                    "validation_worst_session_r2_mean": worst_r2,
                },
                CHECKPOINT_PATH,
            )

        print(
            f"trial {trial.number:03d} complete | selected epoch={best_epoch} | "
            f"validation loss={best_validation_loss:.5f} | "
            f"R2={selected_metrics['validation']['r2_mean']:+.4f} | "
            f"macro/worst R2={macro_r2:+.4f}/{worst_r2:+.4f}",
            flush=True,
        )
        return best_validation_loss

    def persist_callback(current_study, _trial) -> None:
        write_metrics(current_study, context, started_at)
        plot_study(current_study)

    timeout = args.timeout_hours * 3600 if args.timeout_hours else None
    print(f"study database: {storage_path}")
    print("Ctrl-C is safe: completed/pruned trials persist in SQLite.\n")
    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=timeout,
            n_jobs=1,
            callbacks=[persist_callback],
            gc_after_trial=True,
        )
    finally:
        write_metrics(study, context, started_at)
        plot_study(study)

    complete = sorted(completed_trials(study), key=lambda trial: trial.value)
    print("\n=== Phase-1 validation-only ranking ===")
    for rank, trial in enumerate(complete[:10], start=1):
        print(
            f"{rank:02d}. trial {trial.number:03d} | "
            f"loss={trial.value:.5f} | "
            f"lr={trial.params['learning_rate']:.3g} | "
            f"wd={trial.params['weight_decay']:.3g} | "
            f"dropout={trial.params['dropout']:.2f} | "
            f"epoch={trial.user_attrs.get('selected_epoch', '?')} | "
            f"macro/worst R2="
            f"{trial.user_attrs.get('validation_macro_r2_mean', float('nan')):+.4f}/"
            f"{trial.user_attrs.get('validation_worst_session_r2_mean', float('nan')):+.4f}"
        )
    print(f"elapsed: {(time.time() - started) / 60:.1f} minutes")
    print("test: LOCKED and not loaded")
    print(f"metrics: {METRICS_PATH}")
    print(f"figure: {FIGURE_PATH}")
    print(f"best checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
