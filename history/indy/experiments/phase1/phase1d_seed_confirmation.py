#!/usr/bin/env python
"""Phase-1d multi-seed confirmation for the Indy 32-channel decoder.

Phase-1c bracketed the useful AdamW weight-decay region. This independent
confirmation study fixes learning rate at 9e-4 and pre-GRU dropout at 0.025,
then compares weight decay {0.025, 0.060} on seeds {43, 44}. The matching
seed-42 cells are read from the completed Phase-1c metrics JSON for aggregate
reporting; they are not retrained.

Every new cell receives 20 epochs with session-balanced sampling and no
pruning. The script imports only supported APIs under ``src/`` and does not
import Phase-1/1b/1c scripts, ``common.py``, or anything under ``history/``.
January test session names are recorded as locked, but their arrays are never
loaded.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.indy_32ch.input_pipeline import (
    apply_feature_stats,
    fit_feature_stats,
    load_model_data,
    load_session_manifest,
    top_firing_channels,
    window_arrays,
)
from models.indy_32ch.features import multiscale_counts
from models.indy_32ch.model import build_net, causal_config, r2

BIN_S = 0.040
WINDOW_BINS = 50
OBSERVATION_SECONDS = 60
OBSERVATION_BINS = int(OBSERVATION_SECONDS / BIN_S)
N_CHANNELS = 32
ALPHAS = (1.0, 0.1)
AXES = np.array([0, 1])
EXPECTED_SPLIT_COUNTS = {"train": 29, "validation": 4, "test": 4}
STD_FLOOR_PERCENTILE = 10.0

FIXED_LEARNING_RATE = 9e-4
FIXED_DROPOUT = 0.025
EPOCHS = 20
REFERENCE_SEED = 42
CONFIRMATION_SEEDS = (43, 44)
WEIGHT_DECAYS = (0.025, 0.060)
GRID_SPACE = {
    "seed": list(CONFIRMATION_SEEDS),
    "weight_decay": list(WEIGHT_DECAYS),
}
GRID_SIZE = len(CONFIRMATION_SEEDS) * len(WEIGHT_DECAYS)

PHASE1C_METRICS_PATH = (
    ROOT / "results" / "indy" / "phase1c_wd_upper_grid"
    / "phase1c_wd_upper_grid_metrics.json"
)
RESULT_DIR = ROOT / "results" / "indy" / "phase1d_seed_confirmation"
METRICS_PATH = RESULT_DIR / "phase1d_seed_confirmation_metrics.json"
FIGURE_PATH = RESULT_DIR / "phase1d_seed_confirmation_figure.png"
STORAGE_PATH = RESULT_DIR / "phase1d_seed_confirmation_study.db"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"


def stack_windows(
    windows: list[dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Stack all post-observation windows from one session."""
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
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--study-name", default="phase1d_seed_confirmation")
    parser.add_argument("--storage-path", type=Path, default=STORAGE_PATH)
    parser.add_argument("--timeout-hours", type=float)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate data, seed-42 references, model, and grid without training.",
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
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def completed_trials(study) -> list:
    import optuna

    return [
        trial
        for trial in study.get_trials(deepcopy=False)
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and math.isfinite(trial.value)
    ]


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


def load_seed42_references() -> list[dict]:
    """Load the two comparable seed-42 cells from completed Phase-1c evidence."""
    if not PHASE1C_METRICS_PATH.exists():
        raise FileNotFoundError(
            "Phase-1c metrics are required to reuse seed 42: "
            f"{PHASE1C_METRICS_PATH}"
        )
    payload = json.loads(PHASE1C_METRICS_PATH.read_text(encoding="utf-8"))
    protocol = payload.get("fixed_protocol", {})
    expected = {
        "seed": REFERENCE_SEED,
        "epochs": EPOCHS,
        "sampler": "session_balanced",
        "learning_rate": FIXED_LEARNING_RATE,
    }
    mismatches = {
        key: (protocol.get(key), value)
        for key, value in expected.items()
        if protocol.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Phase-1c reference protocol mismatch: {mismatches}")
    if payload.get("test_policy") != "January test is locked and was not loaded.":
        raise ValueError("Phase-1c reference does not confirm the locked-test policy")

    references = []
    for weight_decay in WEIGHT_DECAYS:
        matches = [
            trial
            for trial in payload.get("trials", [])
            if trial.get("state") == "COMPLETE"
            and math.isclose(
                float(trial.get("params", {}).get("weight_decay", -1)),
                weight_decay,
            )
            and math.isclose(
                float(trial.get("params", {}).get("dropout", -1)),
                FIXED_DROPOUT,
            )
        ]
        if len(matches) != 1:
            raise ValueError(
                "Expected one completed Phase-1c seed-42 cell for "
                f"weight_decay={weight_decay}, dropout={FIXED_DROPOUT}; "
                f"found {len(matches)}"
            )
        trial = matches[0]
        metrics = trial.get("metrics", {})
        selected = metrics.get("selected_checkpoint_metrics", {})
        validation = selected.get("validation", {})
        required = (
            "validation_r2_mean",
            "validation_macro_r2_mean",
            "validation_worst_session_r2_mean",
            "selected_epoch",
        )
        missing = [key for key in required if metrics.get(key) is None]
        if trial.get("value") is None or validation.get("loss") is None or missing:
            raise ValueError(
                f"Phase-1c reference cell wd={weight_decay} lacks {missing or 'loss'}"
            )
        references.append(
            {
                "source": "phase1c_reused_not_retrained",
                "source_trial_number": trial.get("number"),
                "seed": REFERENCE_SEED,
                "weight_decay": weight_decay,
                "dropout": FIXED_DROPOUT,
                "validation_loss": float(trial["value"]),
                "validation_r2_mean": float(metrics["validation_r2_mean"]),
                "validation_macro_r2_mean": float(
                    metrics["validation_macro_r2_mean"]
                ),
                "validation_worst_session_r2_mean": float(
                    metrics["validation_worst_session_r2_mean"]
                ),
                "selected_epoch": int(metrics["selected_epoch"]),
                "validation_by_session": metrics.get("validation_by_session", {}),
            }
        )
    return references


def confirmation_rows(study, seed42_references: list[dict]) -> list[dict]:
    """Return comparable flat rows from reused seed 42 and new trials."""
    rows = [dict(row) for row in seed42_references]
    for trial in completed_trials(study):
        rows.append(
            {
                "source": "phase1d_new_training",
                "source_trial_number": trial.number,
                "seed": int(trial.params["seed"]),
                "weight_decay": float(trial.params["weight_decay"]),
                "dropout": FIXED_DROPOUT,
                "validation_loss": float(trial.value),
                "validation_r2_mean": float(
                    trial.user_attrs["validation_r2_mean"]
                ),
                "validation_macro_r2_mean": float(
                    trial.user_attrs["validation_macro_r2_mean"]
                ),
                "validation_worst_session_r2_mean": float(
                    trial.user_attrs["validation_worst_session_r2_mean"]
                ),
                "selected_epoch": int(trial.user_attrs["selected_epoch"]),
                "validation_by_session": trial.user_attrs.get(
                    "validation_by_session", {}
                ),
            }
        )
    return sorted(rows, key=lambda row: (row["seed"], row["weight_decay"]))


def mean_sd(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": float(statistics.mean(values)),
        "sample_sd": float(statistics.stdev(values)) if len(values) > 1 else None,
    }


def aggregate_candidates(rows: list[dict]) -> dict:
    """Aggregate the paired candidate comparison across available seeds."""
    metric_fields = (
        "validation_loss",
        "validation_r2_mean",
        "validation_macro_r2_mean",
        "validation_worst_session_r2_mean",
    )
    summaries = []
    for weight_decay in WEIGHT_DECAYS:
        candidate_rows = [
            row
            for row in rows
            if math.isclose(float(row["weight_decay"]), weight_decay)
        ]
        summary = {
            "weight_decay": weight_decay,
            "dropout": FIXED_DROPOUT,
            "seeds": [int(row["seed"]) for row in candidate_rows],
            "n_seeds": len(candidate_rows),
            "metrics": {
                field: mean_sd([float(row[field]) for row in candidate_rows])
                for field in metric_fields
            },
        }
        summaries.append(summary)

    paired = []
    expected_seeds = (REFERENCE_SEED, *CONFIRMATION_SEEDS)
    seed_wins = {f"weight_decay_{weight_decay:.3f}": 0 for weight_decay in WEIGHT_DECAYS}
    for seed in expected_seeds:
        by_weight_decay = {
            float(row["weight_decay"]): row for row in rows if row["seed"] == seed
        }
        if not all(weight_decay in by_weight_decay for weight_decay in WEIGHT_DECAYS):
            continue
        lower = by_weight_decay[WEIGHT_DECAYS[0]]
        upper = by_weight_decay[WEIGHT_DECAYS[1]]
        loss_delta = float(upper["validation_loss"] - lower["validation_loss"])
        r2_delta = float(upper["validation_r2_mean"] - lower["validation_r2_mean"])
        paired.append(
            {
                "seed": seed,
                "loss_wd_0.060_minus_0.025": loss_delta,
                "r2_wd_0.060_minus_0.025": r2_delta,
            }
        )
        winner = WEIGHT_DECAYS[0] if loss_delta >= 0 else WEIGHT_DECAYS[1]
        seed_wins[f"weight_decay_{winner:.3f}"] += 1

    complete = len(rows) == len(expected_seeds) * len(WEIGHT_DECAYS)
    recommended = None
    if complete:
        recommended = min(
            summaries,
            key=lambda item: item["metrics"]["validation_loss"]["mean"],
        )["weight_decay"]
    return {
        "complete": complete,
        "expected_seeds": list(expected_seeds),
        "candidate_summaries": summaries,
        "paired_seed_differences": paired,
        "seed_wins_by_validation_loss": seed_wins,
        "recommended_weight_decay_by_mean_validation_loss": recommended,
    }


def checkpoint_path(seed: int, weight_decay: float) -> Path:
    wd_token = f"{weight_decay:.3f}".replace(".", "p")
    return CHECKPOINT_DIR / f"seed{seed}_wd{wd_token}_dropout0p025.pt"


def write_metrics(
    study,
    context: dict,
    seed42_references: list[dict],
    started_at: str,
) -> None:
    rows = confirmation_rows(study, seed42_references)
    payload = {
        "purpose": "phase1d_seed_confirmation",
        "generated_at_utc": utc_now(),
        "run_started_at_utc": started_at,
        "study_name": study.study_name,
        "direction": "minimize",
        "primary_metric": "mean_pooled_validation_normalized_mse_across_seeds",
        "fixed_learning_rate": FIXED_LEARNING_RATE,
        "fixed_dropout": FIXED_DROPOUT,
        "reference_seed": REFERENCE_SEED,
        "new_training_seeds": list(CONFIRMATION_SEEDS),
        "weight_decay_candidates": list(WEIGHT_DECAYS),
        "new_grid_size": GRID_SIZE,
        "fixed_protocol": context["fixed_protocol"],
        "split": context["split"],
        "data_protocol": context["data_protocol"],
        "n_parameters": context["n_parameters"],
        "window_counts": context["window_counts"],
        "seed42_reference": {
            "source": report_path(PHASE1C_METRICS_PATH),
            "policy": "reused_metrics_not_retrained",
            "cells": seed42_references,
        },
        "new_trials": [trial_to_record(trial) for trial in study.trials],
        "new_trial_counts": {
            state: sum(trial.state.name == state for trial in study.trials)
            for state in ("COMPLETE", "PRUNED", "FAIL", "RUNNING", "WAITING")
        },
        "all_seed_rows": rows,
        "aggregate": aggregate_candidates(rows),
        "test_policy": "January test is locked and was not loaded.",
        "artifacts": {
            "study_database": report_path(context["storage_path"]),
            "checkpoint_directory": report_path(CHECKPOINT_DIR),
            "figure": report_path(FIGURE_PATH),
        },
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = METRICS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    temporary.replace(METRICS_PATH)


def plot_confirmation(study, seed42_references: list[dict]) -> None:
    rows = confirmation_rows(study, seed42_references)
    if not rows:
        return

    import os

    cache = Path(tempfile.gettempdir()) / "indy_decoder_matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_panels = (
        ("validation_loss", "Validation normalized MSE", "lower is better"),
        ("validation_r2_mean", "Pooled validation R2", "higher is better"),
        ("validation_macro_r2_mean", "Session-macro validation R2", "higher is better"),
        (
            "validation_worst_session_r2_mean",
            "Worst-session validation R2",
            "higher is better",
        ),
    )
    colors = {WEIGHT_DECAYS[0]: "#1F77B4", WEIGHT_DECAYS[1]: "#E69F00"}
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 7.5), dpi=180)
    seeds = (REFERENCE_SEED, *CONFIRMATION_SEEDS)
    for axis, (field, title, subtitle) in zip(axes.flat, metric_panels):
        for weight_decay in WEIGHT_DECAYS:
            candidate = {
                int(row["seed"]): float(row[field])
                for row in rows
                if math.isclose(float(row["weight_decay"]), weight_decay)
            }
            available_seeds = [seed for seed in seeds if seed in candidate]
            values = [candidate[seed] for seed in available_seeds]
            if not values:
                continue
            mean = statistics.mean(values)
            sd = statistics.stdev(values) if len(values) > 1 else None
            label = f"wd={weight_decay:.3f}; mean={mean:.4f}"
            if sd is not None:
                label += f" +/- {sd:.4f}"
            axis.plot(
                available_seeds,
                values,
                marker="o",
                linewidth=2,
                markersize=5,
                color=colors[weight_decay],
                label=label,
            )
        axis.set_title(title)
        axis.set_xlabel("Training seed")
        axis.set_ylabel(title)
        axis.set_xticks(seeds)
        axis.grid(axis="y", alpha=0.25)
        axis.text(
            0.01,
            0.02,
            subtitle,
            transform=axis.transAxes,
            fontsize=8,
            color="#5B6470",
        )
        axis.legend(fontsize=8)

    figure.suptitle("Indy 32-channel Phase-1d multi-seed confirmation")
    figure.text(
        0.5,
        0.01,
        "lr=9e-4; dropout=0.025; session-balanced; seed 42 reused from "
        "Phase-1c; January test locked.",
        ha="center",
        color="#5B6470",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    try:
        import optuna
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Optuna is not installed. Run: python -m pip install -r requirements.txt"
        ) from error
    import torch
    import torch.nn as nn

    args = parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    if args.timeout_hours is not None and args.timeout_hours <= 0:
        raise ValueError("--timeout-hours must be positive")

    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    storage_path = args.storage_path.expanduser().resolve()
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_path.as_posix()}"
    started = time.time()
    started_at = utc_now()

    seed42_references = load_seed42_references()
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

    print("=== Indy 32-channel Phase-1d multi-seed confirmation ===")
    print(
        f"sessions: train={actual_counts['train']} | "
        f"validation={actual_counts['validation']} | test={actual_counts['test']} LOCKED"
    )
    print(
        f"new grid={GRID_SIZE} complete trials | epochs/trial={EPOCHS} | "
        f"seeds={list(CONFIRMATION_SEEDS)} | device={device} | sampler=session-balanced"
    )
    print(
        f"fixed: lr={FIXED_LEARNING_RATE:.1e} | dropout={FIXED_DROPOUT:.3f} | "
        "AdamW | batch=32 | cosine schedule | gradient_clip=1 | "
        "noise=0 | channel_dropout=0"
    )
    print("weight decay candidates: " + ", ".join(map(str, WEIGHT_DECAYS)))
    print("seed 42: reused from Phase-1c metrics; not retrained")
    print("policy: full budget/no pruning; validation selects; test not loaded")

    # Only train and validation arrays are loaded. Reading test names from the
    # versioned manifest does not inspect any January test data.
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
        "epochs": EPOCHS,
        "n_out": 2,
        "bs": 32,
        "noise": 0.0,
        "chdrop": 0.0,
        "gradient_clip": 1.0,
        "cosine": True,
        "lr": FIXED_LEARNING_RATE,
        "dropout": FIXED_DROPOUT,
    }
    torch.manual_seed(CONFIRMATION_SEEDS[0])
    template = build_net(fixed_config, splits["train"]["x"].shape[1])
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
        f"validation={len(splits['validation']['x'])} | parameters={n_parameters:,}"
    )
    for reference in seed42_references:
        print(
            "seed-42 reference | "
            f"wd={reference['weight_decay']:.3f} | "
            f"loss={reference['validation_loss']:.5f} | "
            f"R2={reference['validation_r2_mean']:+.4f}"
        )
    if args.validate_only:
        print("validation-only check complete; no study or checkpoint was written")
        print("test: LOCKED and not loaded")
        return

    protocol_signature = {
        "phase": "1d_seed_confirmation",
        "reference_seed": REFERENCE_SEED,
        "new_training_seeds": list(CONFIRMATION_SEEDS),
        "epochs": EPOCHS,
        "sampler": "session_balanced",
        "batch_size": fixed_config["bs"],
        "gradient_clip": fixed_config["gradient_clip"],
        "scheduler": "cosine_annealing",
        "noise": 0.0,
        "channel_dropout": 0.0,
        "learning_rate": FIXED_LEARNING_RATE,
        "dropout": FIXED_DROPOUT,
        "weight_decay_candidates": list(WEIGHT_DECAYS),
        "train_sessions": split_sessions["train"],
        "validation_sessions": split_sessions["validation"],
        "channels": channels.tolist(),
        "device": str(device),
        "seed42_reference": report_path(PHASE1C_METRICS_PATH),
    }
    sampler = optuna.samplers.GridSampler(GRID_SPACE, seed=REFERENCE_SEED)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction="minimize",
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True,
    )
    saved_signature = study.user_attrs.get("protocol_signature")
    if saved_signature is not None and saved_signature != protocol_signature:
        raise RuntimeError(
            "The existing Phase-1d study uses a different protocol. Choose a new "
            "--study-name and --storage-path instead of mixing trials."
        )
    study.set_user_attr("protocol_signature", protocol_signature)
    study.set_user_attr("test_policy", "locked_not_loaded")
    study.set_user_attr("pruning", "disabled_for_equal_full_budget_comparison")
    study.set_user_attr("seed42_policy", "reused_phase1c_metrics_not_retrained")

    context = {
        "storage_path": storage_path,
        "fixed_protocol": {
            **protocol_signature,
            "optimizer": "AdamW",
            "checkpoint_selection": "minimum_pooled_validation_normalized_mse",
            "aggregate_selection": "minimum_mean_validation_loss_across_seeds_42_43_44",
            "pruning": "disabled",
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
        seed = int(trial.suggest_categorical("seed", CONFIRMATION_SEEDS))
        weight_decay = float(
            trial.suggest_categorical("weight_decay", WEIGHT_DECAYS)
        )
        config = {**fixed_config, "wd": weight_decay}
        print(
            f"\n=== trial {trial.number:03d}/{GRID_SIZE - 1:03d} | "
            f"seed={seed} | lr={FIXED_LEARNING_RATE:.1e} | "
            f"wd={weight_decay:.3f} | dropout={FIXED_DROPOUT:.3f} ===",
            flush=True,
        )

        # Resetting both generators per cell gives the two weight decays the
        # same initialization and session-balanced sample stream within a seed.
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        net = build_net(config, splits["train"]["x"].shape[1]).to(device)
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=FIXED_LEARNING_RATE, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
        history = []
        best_validation_loss = np.inf
        best_epoch = 0
        best_state = None

        for epoch in range(1, EPOCHS + 1):
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
                f"trial {trial.number:03d} | seed={seed} | "
                f"epoch {epoch:02d}/{EPOCHS} | "
                f"opt={row['optimization_loss']:.5f} | "
                f"loss train={row['train_loss']:.5f} "
                f"validation={row['validation_loss']:.5f} | "
                f"R2 train={row['train_r2_mean']:+.4f} "
                f"validation={row['validation_r2_mean']:+.4f} | "
                f"grad={row['gradient_norm_mean_before_clip']:.3f}/"
                f"{row['gradient_norm_max_before_clip']:.3f}"
                + (" *best*" if improved else ""),
                flush=True,
            )
            trial.report(float(row["validation_loss"]), step=epoch)
            scheduler.step()

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
        cell_checkpoint = checkpoint_path(seed, weight_decay)
        trial.set_user_attr("fixed_learning_rate", FIXED_LEARNING_RATE)
        trial.set_user_attr("fixed_dropout", FIXED_DROPOUT)
        trial.set_user_attr("selected_epoch", best_epoch)
        trial.set_user_attr("selected_checkpoint_metrics", selected_metrics)
        trial.set_user_attr(
            "validation_r2_mean", selected_metrics["validation"]["r2_mean"]
        )
        trial.set_user_attr("validation_macro_loss", macro_loss)
        trial.set_user_attr("validation_macro_r2_mean", macro_r2)
        trial.set_user_attr("validation_worst_session_r2_mean", worst_r2)
        trial.set_user_attr("validation_by_session", validation_by_session)
        trial.set_user_attr("history", history)
        trial.set_user_attr("checkpoint", report_path(cell_checkpoint))

        cell_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "purpose": "phase1d_confirmation_cell",
                "created_at_utc": utc_now(),
                "trial_number": trial.number,
                "seed": seed,
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
                "validation_r2_mean": selected_metrics["validation"]["r2_mean"],
                "validation_macro_r2_mean": macro_r2,
                "validation_worst_session_r2_mean": worst_r2,
            },
            cell_checkpoint,
        )

        print(
            f"trial {trial.number:03d} complete | seed={seed} | "
            f"selected epoch={best_epoch} | validation loss={best_validation_loss:.5f} | "
            f"R2={selected_metrics['validation']['r2_mean']:+.4f} | "
            f"macro/worst R2={macro_r2:+.4f}/{worst_r2:+.4f}",
            flush=True,
        )
        return best_validation_loss

    def persist_callback(current_study, _trial) -> None:
        write_metrics(current_study, context, seed42_references, started_at)
        plot_confirmation(current_study, seed42_references)

    existing_complete = len(completed_trials(study))
    remaining = max(GRID_SIZE - existing_complete, 0)
    print(f"study database: {storage_path}")
    print(f"existing complete={existing_complete} | new cells remaining<={remaining}")
    print("Ctrl-C is safe: each completed trial persists in SQLite.\n")
    if remaining:
        timeout = args.timeout_hours * 3600 if args.timeout_hours else None
        try:
            study.optimize(
                objective,
                n_trials=remaining,
                timeout=timeout,
                n_jobs=1,
                callbacks=[persist_callback],
                gc_after_trial=True,
            )
        finally:
            write_metrics(study, context, seed42_references, started_at)
            plot_confirmation(study, seed42_references)
    else:
        write_metrics(study, context, seed42_references, started_at)
        plot_confirmation(study, seed42_references)

    rows = confirmation_rows(study, seed42_references)
    aggregate = aggregate_candidates(rows)
    print("\n=== Phase-1d all-seed validation summary ===")
    for summary in aggregate["candidate_summaries"]:
        loss = summary["metrics"]["validation_loss"]
        pooled = summary["metrics"]["validation_r2_mean"]
        macro = summary["metrics"]["validation_macro_r2_mean"]
        worst = summary["metrics"]["validation_worst_session_r2_mean"]
        sd = lambda metric: (
            f"{metric['sample_sd']:.5f}" if metric["sample_sd"] is not None else "n/a"
        )
        print(
            f"wd={summary['weight_decay']:.3f} | seeds={summary['seeds']} | "
            f"loss={loss['mean']:.5f} +/- {sd(loss)} | "
            f"R2={pooled['mean']:+.4f} +/- {sd(pooled)} | "
            f"macro={macro['mean']:+.4f} +/- {sd(macro)} | "
            f"worst={worst['mean']:+.4f} +/- {sd(worst)}"
        )
    if aggregate["complete"]:
        print(
            "recommended by mean validation loss: "
            f"weight_decay={aggregate['recommended_weight_decay_by_mean_validation_loss']:.3f}"
        )
        print(
            "seed wins: "
            + ", ".join(
                f"{name}={wins}"
                for name, wins in aggregate["seed_wins_by_validation_loss"].items()
            )
        )
    else:
        print("aggregate incomplete: rerun the same command to finish remaining cells")
    print(f"elapsed this invocation: {(time.time() - started) / 60:.1f} minutes")
    print("test: LOCKED and not loaded")
    print(f"metrics: {METRICS_PATH}")
    print(f"figure: {FIGURE_PATH}")
    print(f"cell checkpoints: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
