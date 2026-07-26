#!/usr/bin/env python3
"""Archived Phase 3b: strict pre-January leave-one-month-out evaluation.

For each pre-January month, this script:

1. fits the label-free detector using all other pre-January months;
2. trains a temporary decoder using only those other months;
3. stops at the frozen epoch-7 checkpoint without inspecting the held month;
4. loads held-month velocity labels for the first time and evaluates R²;
5. relates label-free detector scores to genuinely out-of-month decoder R².

January is structurally excluded.  This is an evaluation experiment, not model
selection: architecture, channels, sampler, seed, optimizer, hyperparameters,
checkpoint epoch and scheduler trajectory are fixed before any fold is run.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.indy_32ch.drift_detector import (  # noqa: E402
    DetectorConfig,
    DriftDetector,
    assert_pre_january,
    session_month,
)
from models.indy_32ch.features import multiscale_counts  # noqa: E402
from models.indy_32ch.input_pipeline import (  # noqa: E402
    apply_feature_stats,
    fit_feature_stats,
    load_model_data,
    load_session_manifest,
    processed_session_path,
    window_arrays,
)
from models.indy_32ch.model import build_net, causal_config, r2  # noqa: E402
from models.indy_32ch.sampling import draw_session_balanced_indices  # noqa: E402

MODEL_CONFIG_PATH = ROOT / "configs" / "indy_32ch.yaml"
DETECTOR_CONFIG_PATH = ROOT / "configs" / "indy_32ch_detector.yaml"
RESULT_DIR = ROOT / "results" / "indy" / "phase3b_leave_one_month_out"
METRICS_PATH = RESULT_DIR / "phase3b_leave_one_month_out_metrics.json"
SESSION_CSV_PATH = RESULT_DIR / "phase3b_leave_one_month_out_sessions.csv"
FOLD_CSV_PATH = RESULT_DIR / "phase3b_leave_one_month_out_folds.csv"
FIGURE_PATH = RESULT_DIR / "phase3b_leave_one_month_out_figure.png"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"

BIN_SECONDS = 0.040
OBSERVATION_SECONDS = 60
OBSERVATION_BINS = int(OBSERVATION_SECONDS / BIN_SECONDS)
WINDOW_BINS = 50
ALPHAS = (1.0, 0.1)
TARGET_AXES = np.asarray([0, 1])
STD_FLOOR_PERCENTILE = 10.0

FROZEN_SEED = 43
CHECKPOINT_EPOCH = 7
SCHEDULER_EPOCH_BUDGET = 20
EXPECTED_DEVELOPMENT_SESSIONS = 33
EXPECTED_MONTHS = ("2016-04", "2016-06", "2016-09", "2016-10", "2016-12")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto selects CUDA when available, otherwise CPU; MPS is excluded.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--folds",
        nargs="+",
        choices=EXPECTED_MONTHS,
        default=list(EXPECTED_MONTHS),
        help="Run a subset for debugging. Run all five for authoritative results.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate manifests, fold isolation, data and model shapes; do not train.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value):
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


def choose_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def load_frozen_protocol() -> tuple[np.ndarray, dict, DetectorConfig]:
    with MODEL_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        model_yaml = yaml.safe_load(handle)
    with DETECTOR_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        detector_yaml = yaml.safe_load(handle)

    channels = np.asarray(
        model_yaml["input"]["selected_zero_based"], dtype=np.int64
    )
    if channels.shape != (32,) or np.unique(channels).size != 32:
        raise ValueError("Frozen model must define exactly 32 unique channels")
    training = model_yaml["training"]
    expected = {
        "sampler": "session_balanced",
        "seed": FROZEN_SEED,
        "checkpoint_epoch": CHECKPOINT_EPOCH,
        "epoch_budget": SCHEDULER_EPOCH_BUDGET,
        "batch_size": 32,
        "learning_rate": 9e-4,
        "weight_decay": 0.060,
        "dropout": 0.025,
        "gradient_clip": 1.0,
        "noise": 0.0,
        "channel_dropout": 0.0,
    }
    for key, value in expected.items():
        if training.get(key) != value:
            raise ValueError(
                f"Frozen protocol mismatch for {key}: "
                f"expected {value!r}, found {training.get(key)!r}"
            )

    detector_config = DetectorConfig(
        observation_bins=int(detector_yaml["data_policy"]["observation_bins"]),
        bin_seconds=BIN_SECONDS,
        n_components=int(
            detector_yaml["mindful_inspired_detector"]["dimensions"]
        ),
        warning_quantile=float(detector_yaml["thresholds"]["quantile"]),
    )
    return channels, model_yaml, detector_config


def development_sessions() -> tuple[list[str], dict[str, list[str]]]:
    manifest = load_session_manifest()
    chronological = manifest["chronological_split"]
    names = list(chronological["train"]) + list(chronological["validation"])
    assert_pre_january(names)
    if len(names) != EXPECTED_DEVELOPMENT_SESSIONS:
        raise ValueError(
            f"Expected {EXPECTED_DEVELOPMENT_SESSIONS} development sessions, "
            f"found {len(names)}"
        )
    if set(names) & set(chronological["test"]):
        raise RuntimeError("Pre-January development pool intersects the test split")
    by_month = {
        month: [name for name in names if session_month(name) == month]
        for month in EXPECTED_MONTHS
    }
    if any(not sessions for sessions in by_month.values()):
        raise ValueError(f"Missing expected month in development pool: {by_month}")
    if set(name for sessions in by_month.values() for name in sessions) != set(names):
        raise ValueError("Unexpected month found in pre-January development pool")
    return names, by_month


def load_counts_only(name: str, channels: np.ndarray) -> np.ndarray:
    """Read detector input without touching the held session's velocity labels."""
    assert_pre_january([name])
    artifact = processed_session_path(name)
    if artifact.parent.name == "test":
        raise RuntimeError(f"Refusing to read a test artifact: {artifact}")
    with np.load(artifact, allow_pickle=False) as data:
        return data["counts"][channels].astype(np.float32)


def partition_fold(
    all_names: list[str],
    by_month: dict[str, list[str]],
    held_month: str,
) -> tuple[list[str], list[str]]:
    held_names = list(by_month[held_month])
    training_names = [name for name in all_names if name not in held_names]
    assert_pre_january(training_names + held_names)
    if set(training_names) & set(held_names):
        raise AssertionError("Training and held-month sessions overlap")
    if {session_month(name) for name in held_names} != {held_month}:
        raise AssertionError("Held fold contains an unexpected month")
    if held_month in {session_month(name) for name in training_names}:
        raise AssertionError("Held month leaked into decoder training")
    return training_names, held_names


def stack_windows(
    windows: list[dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    if not windows:
        raise ValueError("No post-observation windows available")
    return (
        np.stack([window["e"] for window in windows]).astype(np.float32),
        np.stack([window["vel"] for window in windows]).astype(np.float32),
    )


def fit_feature_std_floor(
    loaded_training: dict[str, tuple[np.ndarray, np.ndarray]],
    channels: np.ndarray,
) -> np.ndarray:
    session_stds = []
    for counts, _ in loaded_training.values():
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


def prepare_session(
    loaded: tuple[np.ndarray, np.ndarray],
    channels: np.ndarray,
    feature_std_floor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    counts, velocity = loaded
    features = multiscale_counts(counts[channels], ALPHAS)
    if features.shape[1] < OBSERVATION_BINS + WINDOW_BINS:
        raise ValueError("Session is too short for the frozen observation protocol")
    mean, local_std = fit_feature_stats(
        features, observation_bins=OBSERVATION_BINS
    )
    normalized = apply_feature_stats(
        features, (mean, np.maximum(local_std, feature_std_floor))
    )
    return stack_windows(
        window_arrays(
            normalized,
            velocity,
            TARGET_AXES,
            window_bins=WINDOW_BINS,
            start_bin=OBSERVATION_BINS,
        )
    )


def combine_prepared(
    prepared: dict[str, tuple[np.ndarray, np.ndarray]], names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([prepared[name][0] for name in names], axis=0),
        np.concatenate([prepared[name][1] for name in names], axis=0),
    )


def pack_split(
    x: np.ndarray,
    y: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> dict[str, np.ndarray]:
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
) -> tuple[float, np.ndarray, np.ndarray]:
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
            squared_error += float(((prediction_norm - target_norm) ** 2).sum())
            n_values += target_norm.numel()
            predictions.append(
                prediction_norm.cpu().numpy() * target_std + target_mean
            )
    prediction = np.concatenate(predictions, axis=0).astype(np.float32)
    score = r2(split["y"].reshape(-1, 2), prediction.reshape(-1, 2))
    return (
        squared_error / max(n_values, 1),
        score.astype(np.float64),
        prediction,
    )


def fixed_model_config(model_yaml: dict) -> dict:
    training = model_yaml["training"]
    config = {
        **causal_config(),
        "epochs": SCHEDULER_EPOCH_BUDGET,
        "bs": int(training["batch_size"]),
        "lr": float(training["learning_rate"]),
        "wd": float(training["weight_decay"]),
        "dropout": float(training["dropout"]),
        "gradient_clip": float(training["gradient_clip"]),
        "noise": float(training["noise"]),
        "chdrop": float(training["channel_dropout"]),
        "cosine": True,
        "n_out": 2,
    }
    return config


def train_fixed_epoch_decoder(
    train_split: dict[str, np.ndarray],
    training_names: list[str],
    session_lengths: dict[str, int],
    config: dict,
    device,
) -> tuple[object, list[dict]]:
    import torch
    import torch.nn as nn

    torch.manual_seed(FROZEN_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FROZEN_SEED)
    np.random.seed(FROZEN_SEED)
    rng = np.random.default_rng(FROZEN_SEED)

    net = build_net(config, train_split["x"].shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=config["lr"], weight_decay=config["wd"]
    )
    # The frozen epoch-7 checkpoint came from a 20-epoch cosine schedule.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, SCHEDULER_EPOCH_BUDGET
    )
    mse = nn.MSELoss()
    train_x = torch.from_numpy(train_split["x"])
    train_y = torch.from_numpy(train_split["y_norm"])
    history = []

    for epoch in range(1, CHECKPOINT_EPOCH + 1):
        indices, _, month_draws = draw_session_balanced_indices(
            training_names, session_lengths, rng
        )
        net.train()
        error_sum = 0.0
        value_count = 0
        gradient_sum = 0.0
        gradient_max = 0.0
        batch_count = 0
        for start in range(0, len(indices), config["bs"]):
            batch_indices = indices[start : start + config["bs"]]
            x_batch = train_x[batch_indices].to(device)
            y_batch = train_y[batch_indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = mse(net(x_batch), y_batch)
            loss.backward()
            gradient = float(
                torch.nn.utils.clip_grad_norm_(
                    net.parameters(), config["gradient_clip"]
                )
            )
            optimizer.step()
            error_sum += float(loss) * y_batch.numel()
            value_count += y_batch.numel()
            gradient_sum += gradient
            gradient_max = max(gradient_max, gradient)
            batch_count += 1

        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "optimization_loss": error_sum / value_count,
            "gradient_norm_mean_before_clip": gradient_sum / batch_count,
            "gradient_norm_max_before_clip": gradient_max,
            "month_draws": month_draws,
        }
        history.append(row)
        shares = ", ".join(
            f"{month}:{count / len(indices):.1%}"
            for month, count in month_draws.items()
        )
        print(
            f"  epoch {epoch:02d}/{CHECKPOINT_EPOCH} | "
            f"train opt loss={row['optimization_loss']:.5f} | "
            f"lr={row['learning_rate']:.6g} | "
            f"grad mean/max={row['gradient_norm_mean_before_clip']:.3f}/"
            f"{row['gradient_norm_max_before_clip']:.3f} | {shares}",
            flush=True,
        )
        scheduler.step()
    return net, history


def detector_rows_for_fold(
    detector: DriftDetector,
    held_names: list[str],
    held_counts: dict[str, np.ndarray],
    held_month: str,
) -> dict[str, dict]:
    rows = {}
    for name in held_names:
        score = detector.score(held_counts[name])
        rows[name] = {
            "session": name,
            "held_month": held_month,
            **asdict(score),
        }
    return rows


def save_checkpoint_atomic(payload: dict, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run_fold(
    held_month: str,
    all_names: list[str],
    by_month: dict[str, list[str]],
    channels: np.ndarray,
    model_yaml: dict,
    detector_config: DetectorConfig,
    device,
    *,
    validate_only: bool,
) -> tuple[dict | None, list[dict], list[np.ndarray], list[np.ndarray]]:
    import torch

    training_names, held_names = partition_fold(all_names, by_month, held_month)

    print(
        f"\n=== held month {held_month} | "
        f"train sessions={len(training_names)} | held sessions={len(held_names)} ===",
        flush=True,
    )

    # Detector may inspect held neural counts, but never held velocity.
    all_counts = {name: load_counts_only(name, channels) for name in all_names}
    detector = DriftDetector(detector_config).fit(
        {name: all_counts[name] for name in training_names}
    )
    detector_rows = detector_rows_for_fold(
        detector,
        held_names,
        {name: all_counts[name] for name in held_names},
        held_month,
    )

    # Only reference-month labels are loaded before optimization.
    loaded_training = {name: load_model_data(name) for name in training_names}
    feature_std_floor = fit_feature_std_floor(loaded_training, channels)
    prepared_training = {
        name: prepare_session(loaded_training[name], channels, feature_std_floor)
        for name in training_names
    }
    train_x, train_y = combine_prepared(prepared_training, training_names)
    target_mean = train_y.mean(axis=(0, 1))
    target_std = train_y.std(axis=(0, 1)) + 1e-6
    train_split = pack_split(train_x, train_y, target_mean, target_std)
    session_lengths = {
        name: len(prepared_training[name][0]) for name in training_names
    }
    config = fixed_model_config(model_yaml)

    torch.manual_seed(FROZEN_SEED)
    template = build_net(config, train_x.shape[1])
    parameter_count = sum(parameter.numel() for parameter in template.parameters())
    del template
    print(
        f"  windows train={len(train_x)} | parameters={parameter_count:,} | "
        f"feature floor min/median/max={feature_std_floor.min():.4f}/"
        f"{np.median(feature_std_floor):.4f}/{feature_std_floor.max():.4f}"
    )
    print(
        "  held velocity policy: NOT LOADED before the fixed epoch-7 checkpoint"
    )

    if validate_only:
        print("  validation-only: fold isolation and training tensors passed")
        return None, list(detector_rows.values()), [], []

    net, history = train_fixed_epoch_decoder(
        train_split,
        training_names,
        session_lengths,
        config,
        device,
    )

    # The optimizer is finished. Only now may held-month labels be loaded.
    loaded_held = {name: load_model_data(name) for name in held_names}
    prepared_held = {
        name: prepare_session(loaded_held[name], channels, feature_std_floor)
        for name in held_names
    }
    held_x, held_y = combine_prepared(prepared_held, held_names)
    held_split = pack_split(held_x, held_y, target_mean, target_std)
    train_loss, train_r2, _ = evaluate_split(
        net,
        train_split,
        target_mean,
        target_std,
        config["bs"],
        device,
    )
    held_loss, held_r2, held_prediction = evaluate_split(
        net,
        held_split,
        target_mean,
        target_std,
        config["bs"],
        device,
    )

    session_rows = []
    actual_blocks = []
    prediction_blocks = []
    for name in held_names:
        x, y = prepared_held[name]
        split = pack_split(x, y, target_mean, target_std)
        loss, score, prediction = evaluate_split(
            net,
            split,
            target_mean,
            target_std,
            config["bs"],
            device,
        )
        detector_row = detector_rows[name]
        row = {
            **detector_row,
            "windows": len(x),
            "decoder_loss": loss,
            "decoder_r2_x": float(score[0]),
            "decoder_r2_y": float(score[1]),
            "decoder_r2_mean": float(score.mean()),
        }
        session_rows.append(row)
        actual_blocks.append(y.reshape(-1, 2))
        prediction_blocks.append(prediction.reshape(-1, 2))
        print(
            f"  held {name} | loss={loss:.5f} | "
            f"R2={score.mean():+.4f} | detector={row['combined_decision']}",
            flush=True,
        )

    checkpoint_path = CHECKPOINT_DIR / f"held_{held_month}.pt"
    checkpoint_payload = {
        "purpose": "phase3b_temporary_leave_one_month_out_decoder",
        "created_at_utc": utc_now(),
        "held_month": held_month,
        "training_sessions": training_names,
        "held_sessions": held_names,
        "january_policy": "forbidden_not_loaded",
        "selection_policy": "fixed_epoch_7_no_held_month_selection",
        "seed": FROZEN_SEED,
        "checkpoint_epoch": CHECKPOINT_EPOCH,
        "scheduler_epoch_budget": SCHEDULER_EPOCH_BUDGET,
        "channels": channels.tolist(),
        "feature_std_floor": feature_std_floor,
        "target_mean": target_mean,
        "target_std": target_std,
        "config": config,
        "model_state": copy.deepcopy(net.state_dict()),
    }
    save_checkpoint_atomic(checkpoint_payload, checkpoint_path)

    fold_row = {
        "held_month": held_month,
        "training_sessions": len(training_names),
        "held_sessions": len(held_names),
        "training_windows": len(train_x),
        "held_windows": len(held_x),
        "train_loss": train_loss,
        "train_r2_x": float(train_r2[0]),
        "train_r2_y": float(train_r2[1]),
        "train_r2_mean": float(train_r2.mean()),
        "held_loss": held_loss,
        "held_r2_x": float(held_r2[0]),
        "held_r2_y": float(held_r2[1]),
        "held_r2_mean": float(held_r2.mean()),
        "held_macro_r2_mean": float(
            np.mean([row["decoder_r2_mean"] for row in session_rows])
        ),
        "held_worst_session_r2_mean": float(
            np.min([row["decoder_r2_mean"] for row in session_rows])
        ),
        "history": history,
        "checkpoint": str(checkpoint_path),
        "target_mean": target_mean,
        "target_std": target_std,
        "feature_std_floor_min": float(feature_std_floor.min()),
        "feature_std_floor_median": float(np.median(feature_std_floor)),
        "feature_std_floor_max": float(feature_std_floor.max()),
    }
    print(
        f"  fold result | held pooled R2={held_r2.mean():+.4f} | "
        f"macro R2={fold_row['held_macro_r2_mean']:+.4f} | "
        f"worst={fold_row['held_worst_session_r2_mean']:+.4f}",
        flush=True,
    )
    del net, loaded_training, loaded_held, prepared_training, prepared_held
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return fold_row, session_rows, actual_blocks, prediction_blocks


def correlation_rows(session_frame: pd.DataFrame) -> list[dict]:
    score_fields = (
        "pattern_distance",
        "robust_rate_distance",
        "absolute_log_rate_ratio",
        "unexpected_silent_channels",
        "global_mindful_kld",
        "multi_reference_mindful_kld",
        "simple_evidence_count",
        "combined_evidence_count",
    )
    rows = []
    for field in score_fields:
        statistic, p_value = spearmanr(
            session_frame[field], session_frame["decoder_r2_mean"]
        )
        rows.append(
            {
                "detector_metric": field,
                "spearman_rho_vs_decoder_r2": (
                    None if not np.isfinite(statistic) else float(statistic)
                ),
                "p_value_descriptive_only": (
                    None if not np.isfinite(p_value) else float(p_value)
                ),
                "sessions": len(session_frame),
            }
        )
    return rows


def decision_summary(session_frame: pd.DataFrame) -> list[dict]:
    rows = []
    for decision in ("pass", "warning", "abstain"):
        selected = session_frame[session_frame["combined_decision"] == decision]
        rows.append(
            {
                "decision": decision,
                "sessions": len(selected),
                "coverage": len(selected) / len(session_frame),
                "mean_decoder_r2": (
                    None if selected.empty else float(selected["decoder_r2_mean"].mean())
                ),
                "median_decoder_r2": (
                    None
                    if selected.empty
                    else float(selected["decoder_r2_mean"].median())
                ),
                "worst_decoder_r2": (
                    None if selected.empty else float(selected["decoder_r2_mean"].min())
                ),
            }
        )
    accepted = session_frame[session_frame["combined_decision"] != "abstain"]
    rows.append(
        {
            "decision": "not_abstained",
            "sessions": len(accepted),
            "coverage": len(accepted) / len(session_frame),
            "mean_decoder_r2": (
                None if accepted.empty else float(accepted["decoder_r2_mean"].mean())
            ),
            "median_decoder_r2": (
                None
                if accepted.empty
                else float(accepted["decoder_r2_mean"].median())
            ),
            "worst_decoder_r2": (
                None if accepted.empty else float(accepted["decoder_r2_mean"].min())
            ),
        }
    )
    return rows


def save_figure(
    session_frame: pd.DataFrame,
    fold_frame: pd.DataFrame,
    correlations: list[dict],
) -> None:
    cache = Path(tempfile.gettempdir()) / "indy_decoder_matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "pass": "#2CA02C",
        "warning": "#FFB000",
        "abstain": "#D62728",
    }
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=180)
    for decision, group in session_frame.groupby("combined_decision"):
        axes[0].scatter(
            group["multi_reference_mindful_kld"],
            group["decoder_r2_mean"],
            label=decision,
            color=colors[decision],
            s=48,
            alpha=0.85,
        )
    axes[0].axhline(0, color="black", linewidth=1, linestyle=":")
    axes[0].set(
        xlabel="Multi-reference KLD",
        ylabel="Strict out-of-month decoder R²",
        title="Detector score versus decoder performance",
    )
    axes[0].legend()

    axes[1].bar(
        fold_frame["held_month"],
        fold_frame["held_macro_r2_mean"],
        color="#4C78A8",
    )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(
        xlabel="Held month",
        ylabel="Session-macro R²",
        title="Frozen protocol across unseen months",
    )
    axes[1].tick_params(axis="x", rotation=35)

    correlation_frame = pd.DataFrame(correlations).dropna(
        subset=["spearman_rho_vs_decoder_r2"]
    )
    axes[2].barh(
        correlation_frame["detector_metric"],
        correlation_frame["spearman_rho_vs_decoder_r2"],
        color="#7A5195",
    )
    axes[2].axvline(0, color="black", linewidth=1)
    axes[2].set(
        xlabel="Spearman rho with R²",
        title="Continuous detector relationships",
        xlim=(-1, 1),
    )
    figure.suptitle(
        "Phase 3b — strict pre-January leave-one-month-out evaluation"
    )
    figure.tight_layout()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.close(figure)


def write_json_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    import torch

    args = parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    if len(set(args.folds)) != len(args.folds):
        raise ValueError("--folds cannot contain duplicates")
    authoritative = tuple(args.folds) == EXPECTED_MONTHS
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    channels, model_yaml, detector_config = load_frozen_protocol()
    all_names, by_month = development_sessions()

    config = fixed_model_config(model_yaml)
    torch.manual_seed(FROZEN_SEED)
    template = build_net(config, len(channels) * len(ALPHAS))
    parameters = sum(parameter.numel() for parameter in template.parameters())
    del template

    print("=== Phase 3b: strict pre-January leave-one-month-out ===")
    print(
        f"sessions={len(all_names)} | folds={', '.join(args.folds)} | "
        f"device={device} | seed={FROZEN_SEED}"
    )
    print(
        f"frozen protocol: session-balanced | epoch={CHECKPOINT_EPOCH} from "
        f"{SCHEDULER_EPOCH_BUDGET}-epoch cosine trajectory | "
        f"lr={config['lr']:.4g} | wd={config['wd']:.3f} | "
        f"dropout={config['dropout']:.3f} | parameters={parameters:,}"
    )
    print("January policy: FORBIDDEN and not loaded")
    print(
        "held-month policy: counts may be scored label-free; velocity remains "
        "unloaded until optimizer and fixed checkpoint are complete"
    )
    if not authoritative:
        print("WARNING: subset run is diagnostic and cannot produce final Phase 3b claims")

    started_at = utc_now()
    started = time.time()
    fold_rows = []
    session_rows = []
    actual_blocks = []
    prediction_blocks = []
    for held_month in args.folds:
        fold, sessions, actual, predicted = run_fold(
            held_month,
            all_names,
            by_month,
            channels,
            model_yaml,
            detector_config,
            device,
            validate_only=args.validate_only,
        )
        if fold is not None:
            fold_rows.append(fold)
            session_rows.extend(sessions)
            actual_blocks.extend(actual)
            prediction_blocks.extend(predicted)

    if args.validate_only:
        print("\nvalidate-only complete: no optimizer, checkpoint or result was written")
        print("January: FORBIDDEN and not loaded")
        return

    session_frame = pd.DataFrame(session_rows)
    fold_frame = pd.DataFrame(
        [
            {key: value for key, value in row.items() if key != "history"}
            for row in fold_rows
        ]
    )
    correlations = correlation_rows(session_frame)
    decisions = decision_summary(session_frame)
    pooled_actual = np.concatenate(actual_blocks, axis=0)
    pooled_prediction = np.concatenate(prediction_blocks, axis=0)
    pooled_r2 = r2(pooled_actual, pooled_prediction)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    session_frame.to_csv(SESSION_CSV_PATH, index=False)
    fold_frame.to_csv(FOLD_CSV_PATH, index=False)
    save_figure(session_frame, fold_frame, correlations)

    metrics = {
        "phase": "3b",
        "status": (
            "complete_all_pre_january_months"
            if authoritative
            else "diagnostic_subset_not_authoritative"
        ),
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "elapsed_seconds": time.time() - started,
        "january_loaded": False,
        "held_velocity_policy": (
            "loaded_only_after_optimizer_finished_and_fixed_epoch_7_reached"
        ),
        "folds": args.folds,
        "sessions": len(session_frame),
        "device": str(device),
        "fixed_protocol": {
            "seed": FROZEN_SEED,
            "checkpoint_epoch": CHECKPOINT_EPOCH,
            "scheduler_epoch_budget": SCHEDULER_EPOCH_BUDGET,
            "checkpoint_selection": "none_fixed_epoch",
            "channels": channels.tolist(),
            "channel_selection_policy": (
                "frozen_deployment_mapping_not_refit_per_fold"
            ),
            "sampler": "session_balanced",
            "config": config,
            "feature_normalization": (
                "per_session_first_60_seconds_then_frozen"
            ),
            "feature_std_floor": (
                "outer_training_prefix_positive_10th_percentile"
            ),
            "target_normalization": "outer_training_windows_only",
        },
        "detector_protocol": {
            "config": asdict(detector_config),
            "reference": "outer_training_months_only",
            "threshold_calibration": (
                "inner_leave_one_complete_month_out_on_outer_training_pool"
            ),
            "held_inputs": "first_60_seconds_counts_only",
        },
        "fold_results": fold_rows,
        "session_results": session_rows,
        "aggregate": {
            "pooled_r2_x": float(pooled_r2[0]),
            "pooled_r2_y": float(pooled_r2[1]),
            "pooled_r2_mean": float(pooled_r2.mean()),
            "session_macro_r2_mean": float(
                session_frame["decoder_r2_mean"].mean()
            ),
            "session_median_r2_mean": float(
                session_frame["decoder_r2_mean"].median()
            ),
            "worst_session_r2_mean": float(
                session_frame["decoder_r2_mean"].min()
            ),
            "fold_macro_r2_mean": float(fold_frame["held_macro_r2_mean"].mean()),
            "detector_correlations": correlations,
            "decision_summary": decisions,
        },
        "interpretation_guardrails": [
            "No January data participated.",
            "No held-month metric selected a checkpoint or changed training.",
            "P-values are descriptive only because sessions are few and related.",
            "Do not tune the detector on these outer-fold outcomes and report the same run as validation.",
            "Temporary fold checkpoints are evaluation artifacts, not deployment candidates.",
        ],
        "artifacts": {
            "metrics": str(METRICS_PATH),
            "session_csv": str(SESSION_CSV_PATH),
            "fold_csv": str(FOLD_CSV_PATH),
            "figure": str(FIGURE_PATH),
            "checkpoint_directory": str(CHECKPOINT_DIR),
        },
    }
    write_json_atomic(metrics, METRICS_PATH)

    print("\n=== Phase 3b aggregate ===")
    print(
        f"pooled R2={pooled_r2.mean():+.4f} | "
        f"session macro={metrics['aggregate']['session_macro_r2_mean']:+.4f} | "
        f"worst session={metrics['aggregate']['worst_session_r2_mean']:+.4f}"
    )
    print("\nDetector score relationships:")
    for row in correlations:
        rho = row["spearman_rho_vs_decoder_r2"]
        print(
            f"  {row['detector_metric']}: "
            + ("constant/undefined" if rho is None else f"rho={rho:+.3f}")
        )
    print("\nDecision groups:")
    for row in decisions:
        mean = row["mean_decoder_r2"]
        print(
            f"  {row['decision']:13s} | sessions={row['sessions']:2d} | "
            f"coverage={row['coverage']:.1%} | "
            f"mean R2={'n/a' if mean is None else f'{mean:+.4f}'}"
        )
    print(f"\nmetrics: {METRICS_PATH}")
    print(f"figure:  {FIGURE_PATH}")
    print("January: FORBIDDEN and not loaded")


if __name__ == "__main__":
    main()
