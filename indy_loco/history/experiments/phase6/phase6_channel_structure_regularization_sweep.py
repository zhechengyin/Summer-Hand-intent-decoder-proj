#!/usr/bin/env python3
"""Phase 6 controlled channel, structure, and channel-dropout ablation.

The 64/64 TCN+GRU width, chronological data split, causal preprocessing,
session-balanced sampling, and Phase 5 optimizer settings remain frozen.  The
experiment changes one of the following at a time:

1. channel count/ranking: activity Top-64, stability-aware Top-64/72/80/88,
   or all 96 physical channels;
2. temporal structure on all 96 channels: kernel size 2 or three TCN blocks;
3. paired physical-channel dropout on all 96 channels: 0.10 or 0.20.

Seed 43 screens every configuration.  The best channel configuration, best
all-96 structure/regularization configuration, activity Top-64 reference, and
all-96 baseline are then confirmed with seeds 42 and 44.  This avoids a full
Cartesian sweep while retaining three-seed evidence for every final candidate.

Channel ranking uses only the first 60 seconds of the 29 training sessions.
December is inference-only and may select configurations/checkpoints. January
is registered as locked and is never loaded. CUDA is used when available,
otherwise CPU; Apple MPS is disabled.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from indy_loco.models.indy_32ch.features import multiscale_counts  # noqa: E402
from indy_loco.models.indy_32ch.input_pipeline import (  # noqa: E402
    apply_feature_stats,
    fit_feature_stats,
    load_model_data,
    load_session_manifest,
    processed_session_path,
    top_firing_channels,
    window_arrays,
)
from indy_loco.models.indy_32ch.model import causal_config, r2  # noqa: E402
from indy_loco.models.indy_32ch.sampling import (  # noqa: E402
    draw_session_balanced_indices,
)

PHASE_NAME = "phase6_channel_structure_regularization_sweep"
RESULT_DIR = PROJECT_ROOT / "results" / PHASE_NAME
STATE_PATH = RESULT_DIR / ".cache" / f"{PHASE_NAME}_state.json"
METRICS_PATH = RESULT_DIR / f"{PHASE_NAME}_metrics.json"
TRIALS_PATH = RESULT_DIR / f"{PHASE_NAME}_trials.csv"
EPOCHS_PATH = RESULT_DIR / f"{PHASE_NAME}_epochs.csv"
SUMMARY_PATH = RESULT_DIR / f"{PHASE_NAME}_summary.csv"
RANKING_PATH = RESULT_DIR / f"{PHASE_NAME}_channel_ranking.csv"
FIGURE_PATH = RESULT_DIR / f"{PHASE_NAME}_comparison.png"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"

EXPECTED_SPLITS = {"train": 29, "validation": 4, "test": 4}
PHYSICAL_CHANNELS = 96
ALPHAS = (1.0, 0.1)
BIN_SECONDS = 0.040
OBSERVATION_SECONDS = 60
OBSERVATION_BINS = int(OBSERVATION_SECONDS / BIN_SECONDS)
WINDOW_BINS = 50
TARGET_AXES = (0, 1)
STD_FLOOR_PERCENTILE = 10.0

WIDTH = 64
GRU_WIDTH = 64
GRU_LAYERS = 1
BASE_DILATIONS = (1, 2, 4, 8)
BASE_KERNEL = 3
MODEL_DROPOUT = 0.10
LEARNING_RATE = 9e-4
WEIGHT_DECAY = 0.025
BATCH_SIZE = 32
GRADIENT_CLIP = 1.0
SCREEN_SEED = 43
CONFIRMATION_SEEDS = (42, 44)
DEFAULT_EPOCHS = 20

SILENT_RATE_THRESHOLD = 0.01
RANKING_WEIGHTS = {
    "activity": 0.35,
    "cross_session_stability": 0.25,
    "availability": 0.20,
    "cross_month_stability": 0.20,
}


@dataclass(frozen=True)
class ExperimentConfig:
    label: str
    channel_key: str
    kernel_size: int = BASE_KERNEL
    dilations: tuple[int, ...] = BASE_DILATIONS
    channel_dropout: float = 0.0
    category: str = "channel"

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "channel_key": self.channel_key,
            "kernel_size": self.kernel_size,
            "dilations": list(self.dilations),
            "channel_dropout": self.channel_dropout,
            "category": self.category,
        }


SCREEN_CONFIGS = (
    ExperimentConfig("activity64", "activity64"),
    ExperimentConfig("stable64", "stable64"),
    ExperimentConfig("stable72", "stable72"),
    ExperimentConfig("stable80", "stable80"),
    ExperimentConfig("stable88", "stable88"),
    ExperimentConfig("all96_baseline", "all96", category="all96"),
    ExperimentConfig(
        "all96_kernel2",
        "all96",
        kernel_size=2,
        category="all96_structure",
    ),
    ExperimentConfig(
        "all96_3blocks",
        "all96",
        dilations=(1, 2, 4),
        category="all96_structure",
    ),
    ExperimentConfig(
        "all96_chdrop010",
        "all96",
        channel_dropout=0.10,
        category="all96_regularization",
    ),
    ExperimentConfig(
        "all96_chdrop020",
        "all96",
        channel_dropout=0.20,
        category="all96_regularization",
    ),
)
CONFIG_BY_LABEL = {config.label: config for config in SCREEN_CONFIGS}
CHANNEL_COMPARISON_LABELS = (
    "activity64",
    "stable64",
    "stable72",
    "stable80",
    "stable88",
    "all96_baseline",
)
ALL96_COMPARISON_LABELS = (
    "all96_baseline",
    "all96_kernel2",
    "all96_3blocks",
    "all96_chdrop010",
    "all96_chdrop020",
)


@dataclass
class PreparedData:
    channel_key: str
    channels: np.ndarray
    feature_std_floor: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    train_x: np.ndarray
    train_y: np.ndarray
    train_y_normalized: np.ndarray
    session_lengths: dict[str, int]
    validation_x: np.ndarray
    validation_y: np.ndarray
    validation_by_session: dict[str, tuple[np.ndarray, np.ndarray]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint_atomic(payload: dict[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def choose_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested, but CUDA is unavailable. "
                "Install a CUDA-enabled PyTorch build or use auto/cpu."
            )
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("Only auto, cpu, and cuda are supported; MPS is disabled")


def seed_everything(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def model_config(config: ExperimentConfig, epochs: int) -> dict[str, Any]:
    output = causal_config(n_out=2)
    output.update(
        {
            "F": WIDTH,
            "H": GRU_WIDTH,
            "L": GRU_LAYERS,
            "dils": list(config.dilations),
            "kernel_size": config.kernel_size,
            "channel_dropout": config.channel_dropout,
            "bidir": False,
            "dropout": MODEL_DROPOUT,
            "lr": LEARNING_RATE,
            "wd": WEIGHT_DECAY,
            "epochs": epochs,
            "bs": BATCH_SIZE,
            "noise": 0.0,
            "chdrop": 0.0,
            "cosine": True,
            "act": "relu",
            "gradient_clip": GRADIENT_CLIP,
        }
    )
    return output


def build_experiment_net(config: ExperimentConfig, channel_count: int):
    """Build a local configurable model without modifying the frozen module."""
    import torch
    import torch.nn as nn

    class PointwiseLayerNorm(nn.Module):
        def __init__(self, features: int) -> None:
            super().__init__()
            self.normalization = nn.LayerNorm(features)

        def forward(self, values):
            return self.normalization(values.transpose(1, 2)).transpose(1, 2)

    class PairedChannelDropout(nn.Module):
        def __init__(self, physical_channels: int, probability: float) -> None:
            super().__init__()
            self.physical_channels = physical_channels
            self.probability = probability

        def forward(self, values):
            if not self.training or self.probability == 0:
                return values
            keep_probability = 1.0 - self.probability
            mask = torch.empty(
                values.shape[0],
                self.physical_channels,
                1,
                device=values.device,
                dtype=values.dtype,
            ).bernoulli_(keep_probability)
            mask = mask / keep_probability
            return values * torch.cat((mask, mask), dim=1)

    class ConfigurableCausalTCNGRU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            input_features = channel_count * len(ALPHAS)
            self.channel_dropout = PairedChannelDropout(
                channel_count, config.channel_dropout
            )
            self.spatial = nn.Sequential(
                nn.Conv1d(input_features, WIDTH, 1),
                PointwiseLayerNorm(WIDTH),
                nn.ReLU(),
            )
            self.convolutions = nn.ModuleList(
                [
                    nn.Conv1d(
                        WIDTH,
                        WIDTH,
                        config.kernel_size,
                        padding=(config.kernel_size - 1) * dilation,
                        dilation=dilation,
                    )
                    for dilation in config.dilations
                ]
            )
            self.padding = [
                (config.kernel_size - 1) * dilation for dilation in config.dilations
            ]
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(MODEL_DROPOUT)
            self.gru = nn.GRU(
                WIDTH,
                GRU_WIDTH,
                GRU_LAYERS,
                batch_first=True,
                bidirectional=False,
            )
            self.head = nn.Linear(GRU_WIDTH, 2)

        def forward(self, values):
            encoded = self.spatial(self.channel_dropout(values))
            for convolution, padding in zip(
                self.convolutions, self.padding, strict=True
            ):
                convolved = convolution(encoded)
                if padding:
                    convolved = convolved[:, :, :-padding]
                encoded = self.activation(convolved + encoded)
            encoded, _ = self.gru(self.dropout(encoded).transpose(1, 2))
            return self.head(encoded)

    return ConfigurableCausalTCNGRU()


def parameter_count(config: ExperimentConfig, channel_count: int) -> int:
    return sum(
        parameter.numel()
        for parameter in build_experiment_net(config, channel_count).parameters()
    )


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    if len(values) > 1:
        ranks /= len(values) - 1
    return ranks


def channel_ranking(
    train_names: list[str],
    train_loaded: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    rates = np.stack(
        [train_loaded[name][0][:, :OBSERVATION_BINS].mean(1) for name in train_names]
    )
    activity = rates.mean(0)
    cross_session_cv = rates.std(0) / (activity + 1e-6)
    silent_fraction = (rates < SILENT_RATE_THRESHOLD).mean(0)

    month_rows: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    for name, rate in zip(train_names, rates, strict=True):
        date = name.split("_")[1]
        month_rows[f"{date[:4]}-{date[4:6]}"].append(rate)
    month_medians = np.stack(
        [np.median(np.stack(rows), axis=0) for rows in month_rows.values()]
    )
    global_median = np.median(rates, axis=0)
    month_drift = np.max(
        np.abs(np.log((month_medians + 1e-3) / (global_median[None, :] + 1e-3))),
        axis=0,
    )

    activity_score = percentile_ranks(activity)
    stability_score = 1.0 - percentile_ranks(cross_session_cv)
    availability_score = 1.0 - percentile_ranks(silent_fraction)
    month_score = 1.0 - percentile_ranks(month_drift)
    score = (
        RANKING_WEIGHTS["activity"] * activity_score
        + RANKING_WEIGHTS["cross_session_stability"] * stability_score
        + RANKING_WEIGHTS["availability"] * availability_score
        + RANKING_WEIGHTS["cross_month_stability"] * month_score
    )
    stable_order = np.lexsort((np.arange(PHYSICAL_CHANNELS), -score))
    activity64 = top_firing_channels(
        train_loaded,
        64,
        observation_bins=OBSERVATION_BINS,
    )
    mappings = {
        "activity64": activity64.astype(np.int64),
        "stable64": np.sort(stable_order[:64]).astype(np.int64),
        "stable72": np.sort(stable_order[:72]).astype(np.int64),
        "stable80": np.sort(stable_order[:80]).astype(np.int64),
        "stable88": np.sort(stable_order[:88]).astype(np.int64),
        "all96": np.arange(PHYSICAL_CHANNELS, dtype=np.int64),
    }
    ranking_rows = []
    stable_rank_position = np.empty(PHYSICAL_CHANNELS, dtype=np.int64)
    stable_rank_position[stable_order] = np.arange(1, PHYSICAL_CHANNELS + 1)
    for channel in range(PHYSICAL_CHANNELS):
        ranking_rows.append(
            {
                "channel_zero_based": channel,
                "activity_mean_counts_per_bin": float(activity[channel]),
                "cross_session_cv": float(cross_session_cv[channel]),
                "silent_session_fraction": float(silent_fraction[channel]),
                "max_train_month_log_shift": float(month_drift[channel]),
                "composite_score": float(score[channel]),
                "composite_rank": int(stable_rank_position[channel]),
                "selected_activity64": channel in set(activity64.tolist()),
                "selected_stable64": channel in set(mappings["stable64"].tolist()),
                "selected_stable72": channel in set(mappings["stable72"].tolist()),
                "selected_stable80": channel in set(mappings["stable80"].tolist()),
                "selected_stable88": channel in set(mappings["stable88"].tolist()),
            }
        )
    return mappings, ranking_rows


def validate_protocol(epochs: int, device_type: str) -> dict[str, Any]:
    if epochs <= 0:
        raise ValueError("--epochs must be positive")
    manifest = load_session_manifest()
    split = manifest["chronological_split"]
    counts = {name: len(split[name]) for name in EXPECTED_SPLITS}
    if counts != EXPECTED_SPLITS:
        raise ValueError(f"Expected split {EXPECTED_SPLITS}, found {counts}")
    train_names = list(split["train"])
    validation_names = list(split["validation"])
    test_names = list(split["test"])
    if any(name.startswith("indy_201701") for name in train_names + validation_names):
        raise ValueError("January leaked into train or validation")
    if any(not name.startswith("indy_201701") for name in test_names):
        raise ValueError("Locked test registry is not the expected January split")
    for name in train_names + validation_names:
        if not processed_session_path(name).exists():
            raise FileNotFoundError(f"Missing processed session: {name}")
    signature_payload = {
        "phase": PHASE_NAME,
        "configs": [config.as_dict() for config in SCREEN_CONFIGS],
        "ranking_weights": RANKING_WEIGHTS,
        "silent_rate_threshold": SILENT_RATE_THRESHOLD,
        "model_width": WIDTH,
        "gru_width": GRU_WIDTH,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "model_dropout": MODEL_DROPOUT,
        "epochs": epochs,
        "screen_seed": SCREEN_SEED,
        "confirmation_seeds": CONFIRMATION_SEEDS,
        "device": device_type,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    counts_by_config = {
        config.label: parameter_count(
            config,
            96 if config.channel_key == "all96" else int(config.channel_key[-2:]),
        )
        for config in SCREEN_CONFIGS
    }
    return {
        "train_names": train_names,
        "validation_names": validation_names,
        "test_names": test_names,
        "signature": signature,
        "signature_payload": signature_payload,
        "parameter_counts": counts_by_config,
    }


def fit_feature_std_floor(
    train_loaded: dict[str, tuple[np.ndarray, np.ndarray]],
    channels: np.ndarray,
) -> np.ndarray:
    session_stds = []
    for counts, _ in train_loaded.values():
        features = multiscale_counts(counts[channels], ALPHAS)
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
    channels: np.ndarray,
    feature_std_floor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    counts, velocity = data
    features = multiscale_counts(counts[channels], ALPHAS)
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
    if not windows:
        raise ValueError("Session is too short after the causal warm-up")
    return (
        np.stack([window["e"] for window in windows]).astype(np.float32),
        np.stack([window["vel"] for window in windows]).astype(np.float32),
    )


def prepare_data(
    *,
    channel_key: str,
    channels: np.ndarray,
    train_names: list[str],
    validation_names: list[str],
    train_loaded: dict[str, tuple[np.ndarray, np.ndarray]],
    validation_loaded: dict[str, tuple[np.ndarray, np.ndarray]],
) -> PreparedData:
    print(
        f"\nPreparing {channel_key}: {len(channels)} physical channels",
        flush=True,
    )
    feature_std_floor = fit_feature_std_floor(train_loaded, channels)
    train_prepared = {
        name: prepare_session(train_loaded[name], channels, feature_std_floor)
        for name in train_names
    }
    train_x = np.concatenate([train_prepared[name][0] for name in train_names], axis=0)
    train_y = np.concatenate([train_prepared[name][1] for name in train_names], axis=0)
    target_mean = train_y.mean(axis=(0, 1)).astype(np.float32)
    target_std = (train_y.std(axis=(0, 1)) + 1e-6).astype(np.float32)
    train_y_normalized = ((train_y - target_mean) / target_std).astype(np.float32)
    session_lengths = {name: int(len(train_prepared[name][0])) for name in train_names}
    del train_prepared

    validation_by_session = {
        name: prepare_session(validation_loaded[name], channels, feature_std_floor)
        for name in validation_names
    }
    validation_x = np.concatenate(
        [validation_by_session[name][0] for name in validation_names], axis=0
    )
    validation_y = np.concatenate(
        [validation_by_session[name][1] for name in validation_names], axis=0
    )
    print(
        f"windows train={len(train_x)} validation={len(validation_x)} | "
        f"input={tuple(train_x.shape[1:])}",
        flush=True,
    )
    return PreparedData(
        channel_key=channel_key,
        channels=channels,
        feature_std_floor=feature_std_floor,
        target_mean=target_mean,
        target_std=target_std,
        train_x=train_x,
        train_y=train_y,
        train_y_normalized=train_y_normalized,
        session_lengths=session_lengths,
        validation_x=validation_x,
        validation_y=validation_y,
        validation_by_session=validation_by_session,
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


def run_key(config: ExperimentConfig, seed: int) -> str:
    return f"{config.label}|seed{seed}"


def checkpoint_path(config: ExperimentConfig, seed: int) -> Path:
    return CHECKPOINT_DIR / f"{config.label}_seed{seed}.pt"


def train_one(
    *,
    config: ExperimentConfig,
    seed: int,
    stage: str,
    epochs: int,
    prepared: PreparedData,
    train_names: list[str],
    validation_names: list[str],
    device,
) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    seed_everything(seed)
    net = build_experiment_net(config, len(prepared.channels)).to(device)
    parameters = sum(parameter.numel() for parameter in net.parameters())
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    mse = nn.MSELoss()
    rng = np.random.default_rng(seed)
    best_state = None
    best_epoch = None
    best_validation_loss = float("inf")
    history = []

    print(
        f"\n=== {stage} | {config.label} | seed={seed} ===\n"
        f"channels={len(prepared.channels)} | parameters={parameters:,} | "
        f"kernel={config.kernel_size} | blocks={len(config.dilations)} | "
        f"channel_dropout={config.channel_dropout}",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        indices, session_draws, month_draws = draw_session_balanced_indices(
            train_names,
            prepared.session_lengths,
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
            x_batch = torch.from_numpy(prepared.train_x[batch_indices]).to(device)
            y_batch = torch.from_numpy(prepared.train_y_normalized[batch_indices]).to(
                device
            )
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
            prepared.train_x,
            prepared.train_y,
            prepared.target_mean,
            prepared.target_std,
            device,
        )
        validation_metrics = evaluate(
            net,
            prepared.validation_x,
            prepared.validation_y,
            prepared.target_mean,
            prepared.target_std,
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
            f"epoch {epoch:02d}/{epochs} | opt={row['optimization_loss']:.5f} | "
            f"loss train={row['train_loss']:.5f} "
            f"validation={row['validation_loss']:.5f} | "
            f"R2 train={row['train_r2']:+.4f} "
            f"validation={row['validation_r2']:+.4f} | "
            f"grad={row['gradient_norm_mean_before_clip']:.3f}/"
            f"{row['gradient_norm_max_before_clip']:.3f}"
            + (" *best*" if improved else ""),
            flush=True,
        )
        scheduler.step()

    if best_state is None or best_epoch is None:
        raise RuntimeError(f"No checkpoint selected for {config.label}")
    net.load_state_dict(best_state)
    train_metrics = evaluate(
        net,
        prepared.train_x,
        prepared.train_y,
        prepared.target_mean,
        prepared.target_std,
        device,
    )
    validation_metrics = evaluate(
        net,
        prepared.validation_x,
        prepared.validation_y,
        prepared.target_mean,
        prepared.target_std,
        device,
    )
    validation_by_session = {
        name: evaluate(
            net,
            prepared.validation_by_session[name][0],
            prepared.validation_by_session[name][1],
            prepared.target_mean,
            prepared.target_std,
            device,
        )
        for name in validation_names
    }
    macro_r2 = float(
        np.mean([row["r2_mean"] for row in validation_by_session.values()])
    )
    worst_r2 = float(min(row["r2_mean"] for row in validation_by_session.values()))
    output_path = checkpoint_path(config, seed)
    checkpoint = {
        "purpose": PHASE_NAME,
        "status": "experiment_only_not_promoted",
        "created_at_utc": utc_now(),
        "stage": stage,
        "experiment_config": config.as_dict(),
        "seed": seed,
        "training_device": device.type,
        "model_state": best_state,
        "model_config": model_config(config, epochs),
        "parameter_count": parameters,
        "physical_channel_count": len(prepared.channels),
        "input_feature_count": len(prepared.channels) * len(ALPHAS),
        "channels": prepared.channels.tolist(),
        "channel_selection": config.channel_key,
        "feature_std_floor": prepared.feature_std_floor[:, 0].tolist(),
        "target_mean": prepared.target_mean.tolist(),
        "target_std": prepared.target_std.tolist(),
        "train_sessions": train_names,
        "validation_sessions": validation_names,
        "january_loaded": False,
        "checkpoint_epoch": best_epoch,
        "selection_policy": "minimum pooled December validation normalized MSE",
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "validation_macro_r2_mean": macro_r2,
        "validation_worst_session_r2_mean": worst_r2,
        "validation_by_session": validation_by_session,
        "training_history": history,
    }
    save_checkpoint_atomic(checkpoint, output_path)
    print(
        f"selected epoch={best_epoch:02d} | train R2={train_metrics['r2_mean']:+.4f} | "
        "validation pooled/macro/worst R2="
        f"{validation_metrics['r2_mean']:+.4f}/{macro_r2:+.4f}/{worst_r2:+.4f}",
        flush=True,
    )
    return {
        "key": run_key(config, seed),
        "stage": stage,
        "config": config.as_dict(),
        "seed": seed,
        "training_device": device.type,
        "parameter_count": parameters,
        "channels": prepared.channels.tolist(),
        "checkpoint": output_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "checkpoint_sha256": sha256_file(output_path),
        "checkpoint_epoch": best_epoch,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "validation_macro_r2_mean": macro_r2,
        "validation_worst_session_r2_mean": worst_r2,
        "validation_by_session": validation_by_session,
        "history": history,
    }


def initial_state(signature: str, signature_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": PHASE_NAME,
        "protocol_signature": signature,
        "protocol": signature_payload,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "runs": {},
    }


def load_or_create_state(
    *,
    signature: str,
    signature_payload: dict[str, Any],
    resume: bool,
    overwrite: bool,
) -> dict[str, Any]:
    if resume and overwrite:
        raise ValueError("Use only one of --resume and --overwrite")
    if overwrite and RESULT_DIR.exists():
        shutil.rmtree(RESULT_DIR)
    if STATE_PATH.exists():
        if not resume:
            raise FileExistsError(
                f"Existing Phase 6 sweep state: {STATE_PATH}. "
                "Use --resume or --overwrite."
            )
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if state.get("protocol_signature") != signature:
            raise ValueError(
                "Existing state uses a different protocol/device. "
                "Use --overwrite only for an intentional replacement."
            )
        return state
    state = initial_state(signature, signature_payload)
    write_json_atomic(state, STATE_PATH)
    return state


def run_if_needed(
    *,
    state: dict[str, Any],
    config: ExperimentConfig,
    seed: int,
    stage: str,
    epochs: int,
    prepared: PreparedData,
    train_names: list[str],
    validation_names: list[str],
    device,
) -> dict[str, Any]:
    key = run_key(config, seed)
    existing = state["runs"].get(key)
    if existing is not None:
        path = REPOSITORY_ROOT / existing["checkpoint"]
        if not path.exists() or sha256_file(path) != existing["checkpoint_sha256"]:
            raise ValueError(f"Completed run checkpoint is missing/changed: {key}")
        print(f"\nSKIP completed: {key}", flush=True)
        return existing
    result = train_one(
        config=config,
        seed=seed,
        stage=stage,
        epochs=epochs,
        prepared=prepared,
        train_names=train_names,
        validation_names=validation_names,
        device=device,
    )
    state["runs"][key] = result
    state["updated_at_utc"] = utc_now()
    write_json_atomic(state, STATE_PATH)
    return result


def screen_winners(state: dict[str, Any]) -> tuple[str, str, list[str]]:
    channel_winner = min(
        CHANNEL_COMPARISON_LABELS,
        key=lambda label: state["runs"][f"{label}|seed{SCREEN_SEED}"][
            "validation_metrics"
        ]["loss"],
    )
    all96_winner = min(
        ALL96_COMPARISON_LABELS,
        key=lambda label: state["runs"][f"{label}|seed{SCREEN_SEED}"][
            "validation_metrics"
        ]["loss"],
    )
    candidates = list(
        dict.fromkeys(("activity64", "all96_baseline", channel_winner, all96_winner))
    )
    return channel_winner, all96_winner, candidates


def aggregate_results(
    state: dict[str, Any], candidate_labels: list[str]
) -> list[dict[str, Any]]:
    seeds = (SCREEN_SEED, *CONFIRMATION_SEEDS)
    rows = []
    for label in candidate_labels:
        results = [state["runs"][f"{label}|seed{seed}"] for seed in seeds]
        rows.append(
            {
                "config_label": label,
                "channel_count": len(results[0]["channels"]),
                "kernel_size": results[0]["config"]["kernel_size"],
                "tcn_blocks": len(results[0]["config"]["dilations"]),
                "channel_dropout": results[0]["config"]["channel_dropout"],
                "parameter_count": results[0]["parameter_count"],
                "validation_loss_mean": float(
                    np.mean([row["validation_metrics"]["loss"] for row in results])
                ),
                "validation_loss_std": float(
                    np.std([row["validation_metrics"]["loss"] for row in results])
                ),
                "validation_pooled_r2_mean": float(
                    np.mean([row["validation_metrics"]["r2_mean"] for row in results])
                ),
                "validation_pooled_r2_std": float(
                    np.std([row["validation_metrics"]["r2_mean"] for row in results])
                ),
                "validation_macro_r2_mean": float(
                    np.mean([row["validation_macro_r2_mean"] for row in results])
                ),
                "validation_macro_r2_std": float(
                    np.std([row["validation_macro_r2_mean"] for row in results])
                ),
                "validation_worst_r2_mean": float(
                    np.mean(
                        [row["validation_worst_session_r2_mean"] for row in results]
                    )
                ),
                "validation_worst_r2_min": float(
                    min(row["validation_worst_session_r2_mean"] for row in results)
                ),
                "train_validation_gap_mean": float(
                    np.mean(
                        [
                            row["train_metrics"]["r2_mean"]
                            - row["validation_metrics"]["r2_mean"]
                            for row in results
                        ]
                    )
                ),
                "selected_epoch_mean": float(
                    np.mean([row["checkpoint_epoch"] for row in results])
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    *,
    state: dict[str, Any],
    ranking_rows: list[dict[str, Any]],
    channel_winner: str,
    all96_winner: str,
    candidate_labels: list[str],
    epochs: int,
) -> None:
    runs = list(state["runs"].values())
    trial_rows = []
    epoch_rows = []
    for result in runs:
        trial_rows.append(
            {
                "stage": result["stage"],
                "config_label": result["config"]["label"],
                "seed": result["seed"],
                "training_device": result["training_device"],
                "channel_count": len(result["channels"]),
                "kernel_size": result["config"]["kernel_size"],
                "tcn_blocks": len(result["config"]["dilations"]),
                "channel_dropout": result["config"]["channel_dropout"],
                "parameter_count": result["parameter_count"],
                "checkpoint_epoch": result["checkpoint_epoch"],
                "train_r2": result["train_metrics"]["r2_mean"],
                "validation_loss": result["validation_metrics"]["loss"],
                "validation_pooled_r2": result["validation_metrics"]["r2_mean"],
                "validation_macro_r2": result["validation_macro_r2_mean"],
                "validation_worst_r2": result["validation_worst_session_r2_mean"],
                "train_validation_gap": (
                    result["train_metrics"]["r2_mean"]
                    - result["validation_metrics"]["r2_mean"]
                ),
                "checkpoint": result["checkpoint"],
            }
        )
        for row in result["history"]:
            epoch_rows.append(
                {
                    "config_label": result["config"]["label"],
                    "seed": result["seed"],
                    "epoch": row["epoch"],
                    "optimization_loss": row["optimization_loss"],
                    "train_loss": row["train_loss"],
                    "train_r2": row["train_r2"],
                    "validation_loss": row["validation_loss"],
                    "validation_r2": row["validation_r2"],
                }
            )
    trial_rows.sort(key=lambda row: (row["config_label"], row["seed"]))
    epoch_rows.sort(key=lambda row: (row["config_label"], row["seed"], row["epoch"]))
    aggregate_rows = aggregate_results(state, candidate_labels)
    aggregate_rows.sort(key=lambda row: row["validation_loss_mean"])
    write_csv(TRIALS_PATH, trial_rows)
    write_csv(EPOCHS_PATH, epoch_rows)
    write_csv(SUMMARY_PATH, aggregate_rows)
    write_csv(RANKING_PATH, ranking_rows)

    payload = {
        "phase": "6",
        "name": PHASE_NAME,
        "created_at_utc": utc_now(),
        "protocol_signature": state["protocol_signature"],
        "purpose": (
            "isolate channel count/ranking, temporal structure, and paired "
            "channel dropout while freezing 64/64 width"
        ),
        "protocol": state["protocol"],
        "data_policy": {
            "train_sessions_updated_weights": 29,
            "validation_sessions_updated_weights": 0,
            "validation_selected_configs_and_checkpoints": True,
            "test_sessions_loaded": 0,
            "january_loaded": False,
        },
        "screen_channel_winner": channel_winner,
        "screen_all96_structure_or_regularization_winner": all96_winner,
        "confirmed_candidates": candidate_labels,
        "aggregate_results": aggregate_rows,
        "recommended_by_mean_validation_loss": aggregate_rows[0]["config_label"],
        "interpretation_boundary": (
            "Seed 43 screened configurations; seeds 42 and 44 confirm only "
            "the selected candidates and fixed references. December is reused "
            "for tuning; January remains unopened."
        ),
        "epochs_each": epochs,
    }
    write_json_atomic(payload, METRICS_PATH)
    render_figure(aggregate_rows)


def render_figure(rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["config_label"] for row in rows]
    positions = np.arange(len(rows))
    means = [row["validation_pooled_r2_mean"] for row in rows]
    errors = [row["validation_pooled_r2_std"] for row in rows]
    gaps = [row["train_validation_gap_mean"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(positions, means, yerr=errors, capsize=4, color="#3A6EA5")
    axes[0].set_title("Three-seed December validation R²")
    axes[0].set_ylabel("Pooled R² mean ± SD")
    axes[1].bar(positions, gaps, color="#C57B57")
    axes[1].set_title("Train–validation R² gap")
    axes[1].set_ylabel("Mean gap across seeds")
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Phase 6 · Channel/structure/regularization confirmation\n"
        "64/64 width frozen · December validation only · January not loaded"
    )
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)


def grouped_configs(
    configs: list[ExperimentConfig],
) -> dict[str, list[ExperimentConfig]]:
    groups: defaultdict[str, list[ExperimentConfig]] = defaultdict(list)
    for config in configs:
        groups[config.channel_key].append(config)
    return dict(groups)


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")

    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    seed_everything(SCREEN_SEED)
    context = validate_protocol(args.epochs, device.type)
    if args.validate_only:
        print("=== Phase 6 controlled sweep validation passed ===")
        print(f"device={device} | epochs each={args.epochs}")
        print(f"seed-43 screen configs={[c.label for c in SCREEN_CONFIGS]}")
        print(
            "confirmation seeds=[42, 44] for activity64, all96 baseline, "
            "channel winner, and all96 structure/regularization winner"
        )
        print(f"parameter counts={context['parameter_counts']}")
        print("train=29 | validation=4 inference/tuning only | January=FORBIDDEN")
        print("no arrays loaded; no output written")
        return

    state = load_or_create_state(
        signature=context["signature"],
        signature_payload=context["signature_payload"],
        resume=args.resume,
        overwrite=args.overwrite,
    )
    train_names = context["train_names"]
    validation_names = context["validation_names"]
    print("=== Indy Phase 6 controlled sweep ===", flush=True)
    print(
        f"device={device} | epochs={args.epochs} | screen seed={SCREEN_SEED} | "
        f"confirmation seeds={list(CONFIRMATION_SEEDS)}",
        flush=True,
    )
    print(
        "64/64 width, LR, WD, dropout, split, causal processing and sampler frozen",
        flush=True,
    )

    train_loaded = {name: load_model_data(name) for name in train_names}
    validation_loaded = {name: load_model_data(name) for name in validation_names}
    channel_counts = {
        counts.shape[0]
        for counts, _ in (*train_loaded.values(), *validation_loaded.values())
    }
    if channel_counts != {PHYSICAL_CHANNELS}:
        raise ValueError(f"Expected exactly 96 channels, found {channel_counts}")
    mappings, ranking_rows = channel_ranking(train_names, train_loaded)
    write_csv(RANKING_PATH, ranking_rows)
    print(
        "channel ranking fitted from train 60-second prefixes only; "
        f"stable64={mappings['stable64'].tolist()}",
        flush=True,
    )

    # Stage 1: screen every isolated configuration at seed 43.
    for channel_key, configs in grouped_configs(list(SCREEN_CONFIGS)).items():
        prepared = prepare_data(
            channel_key=channel_key,
            channels=mappings[channel_key],
            train_names=train_names,
            validation_names=validation_names,
            train_loaded=train_loaded,
            validation_loaded=validation_loaded,
        )
        for config in configs:
            run_if_needed(
                state=state,
                config=config,
                seed=SCREEN_SEED,
                stage="seed43_screen",
                epochs=args.epochs,
                prepared=prepared,
                train_names=train_names,
                validation_names=validation_names,
                device=device,
            )
        del prepared

    channel_winner, all96_winner, candidate_labels = screen_winners(state)
    print(
        f"\nscreen winners: channel={channel_winner} | "
        f"all96 structure/regularization={all96_winner}",
        flush=True,
    )
    print(f"confirmation candidates={candidate_labels}", flush=True)

    # Stage 2: confirm selected candidates with seeds 42 and 44.
    confirmation_configs = [CONFIG_BY_LABEL[label] for label in candidate_labels]
    for channel_key, configs in grouped_configs(confirmation_configs).items():
        prepared = prepare_data(
            channel_key=channel_key,
            channels=mappings[channel_key],
            train_names=train_names,
            validation_names=validation_names,
            train_loaded=train_loaded,
            validation_loaded=validation_loaded,
        )
        for config in configs:
            for seed in CONFIRMATION_SEEDS:
                run_if_needed(
                    state=state,
                    config=config,
                    seed=seed,
                    stage="three_seed_confirmation",
                    epochs=args.epochs,
                    prepared=prepared,
                    train_names=train_names,
                    validation_names=validation_names,
                    device=device,
                )
        del prepared

    write_outputs(
        state=state,
        ranking_rows=ranking_rows,
        channel_winner=channel_winner,
        all96_winner=all96_winner,
        candidate_labels=candidate_labels,
        epochs=args.epochs,
    )
    print("\n=== Phase 6 sweep complete ===", flush=True)
    print(f"metrics: {METRICS_PATH}", flush=True)
    print(f"summary: {SUMMARY_PATH}", flush=True)
    print(f"figure: {FIGURE_PATH}", flush=True)
    print("January: NOT LOADED | retained checkpoints: UNCHANGED", flush=True)


if __name__ == "__main__":
    main()
