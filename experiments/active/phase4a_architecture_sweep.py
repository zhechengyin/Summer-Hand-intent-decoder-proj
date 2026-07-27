#!/usr/bin/env python3
"""Phase 4a: protected architecture-only sweep for the Indy decoder.

This file is deliberately self-contained. It does not import the frozen model,
training helpers, sampler, or archived experiments. The retained baseline
checkpoint is read only through a SHA-256 integrity check and is never loaded
or written.

Every candidate keeps the data, causal preprocessing and optimization protocol
fixed. Optuna changes exactly five architecture fields:

1. TCN filter width;
2. GRU hidden width;
3. number of power-of-two dilation blocks;
4. temporal kernel size;
5. number of GRU layers.

All five complete pre-January months are used as deterministic leave-one-month-
out folds. January is forbidden in code. Trial pruning occurs only after a
candidate has finished training and evaluation for a complete held-month fold;
held labels never select an epoch or update a weight.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG_PATH = ROOT / "configs" / "indy_32ch.yaml"
SESSION_MANIFEST_PATH = ROOT / "configs" / "datasets" / "indy_sessions.yaml"
BASELINE_CHECKPOINT_PATH = ROOT / "models" / "indy_32ch" / "checkpoint.pt"
PROCESSED_DIR = ROOT / "data" / "processed" / "indy_loco" / "indy"

RESULT_DIR = ROOT / "results" / "indy" / "phase4a_architecture_sweep"
CACHE_DIR = RESULT_DIR / ".cache"
STORAGE_PATH = RESULT_DIR / "phase4a_architecture_sweep.db"
METRICS_PATH = RESULT_DIR / "phase4a_architecture_sweep_metrics.json"
TRIALS_PATH = RESULT_DIR / "phase4a_architecture_sweep_trials.csv"
FIGURE_PATH = RESULT_DIR / "phase4a_architecture_sweep_figure.png"

MODEL_READY_SCHEMA = "indy_counts_velocity_v2"
EXPECTED_SPLITS = {"train": 29, "validation": 4, "test": 4}
EXPECTED_MONTHS = ("2016-04", "2016-06", "2016-09", "2016-10", "2016-12")
# Hard folds first so pruning spends less time on clearly weak candidates.
FOLD_ORDER = ("2016-06", "2016-10", "2016-12", "2016-04", "2016-09")

BIN_SECONDS = 0.04
OBSERVATION_SECONDS = 60
OBSERVATION_BINS = 1500
WINDOW_BINS = 50
TARGET_AXES = (0, 1)
ALPHAS = (1.0, 0.1)
STD_FLOOR_PERCENTILE = 10.0

FROZEN_SEED = 43
TRAIN_EPOCHS = 7
SCHEDULER_EPOCH_BUDGET = 20
LEARNING_RATE = 9e-4
WEIGHT_DECAY = 0.060
DROPOUT = 0.025
BATCH_SIZE = 32
GRADIENT_CLIP = 1.0
SAMPLER_NAME = "session_balanced"

SEARCH_SPACE = {
    "tcn_filters": [32, 48, 64, 96],
    "gru_hidden": [32, 48, 64, 96],
    "tcn_blocks": [2, 3, 4],
    "kernel_size": [2, 3, 4],
    "gru_layers": [1, 2],
}
BASELINE_ARCHITECTURE = {
    "tcn_filters": 64,
    "gru_hidden": 64,
    "tcn_blocks": 4,
    "kernel_size": 3,
    "gru_layers": 1,
}
SELECTION_WEIGHTS = {"session_macro_r2": 0.75, "session_q10_r2": 0.25}
OPTUNA_SAMPLER_SEED = 20260726
EXPECTED_BASELINE_PARAMETERS = 78_786


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


def write_json_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def save_npy_atomic(values: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def session_month(name: str) -> str:
    parts = name.split("_")
    if len(parts) != 3 or len(parts[1]) != 8 or not parts[1].isdigit():
        raise ValueError(f"Invalid Indy session name: {name!r}")
    return f"{parts[1][:4]}-{parts[1][4:6]}"


def validate_frozen_protocol() -> dict:
    """Validate every fixed choice and capture the protected checkpoint hash."""
    model_yaml = load_yaml(MODEL_CONFIG_PATH)
    manifest = load_yaml(SESSION_MANIFEST_PATH)
    expected_hash = str(model_yaml["artifact"]["sha256"])
    actual_hash = sha256_file(BASELINE_CHECKPOINT_PATH)
    if actual_hash != expected_hash:
        raise ValueError(
            "Frozen baseline checkpoint SHA-256 mismatch: "
            f"expected {expected_hash}, found {actual_hash}"
        )

    channels = np.asarray(
        model_yaml["input"]["selected_zero_based"], dtype=np.int64
    )
    if channels.shape != (32,) or len(np.unique(channels)) != 32:
        raise ValueError("Frozen configuration must contain 32 unique channels")
    training = model_yaml["training"]
    expected_training = {
        "sampler": SAMPLER_NAME,
        "seed": FROZEN_SEED,
        "checkpoint_epoch": TRAIN_EPOCHS,
        "epoch_budget": SCHEDULER_EPOCH_BUDGET,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
        "gradient_clip": GRADIENT_CLIP,
        "noise": 0.0,
        "channel_dropout": 0.0,
    }
    mismatches = {
        key: {"expected": expected, "actual": training.get(key)}
        for key, expected in expected_training.items()
        if training.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Frozen training protocol changed: {mismatches}")
    model = model_yaml["model"]
    expected_model = {
        "family": "tcn_gru",
        "filters": BASELINE_ARCHITECTURE["tcn_filters"],
        "gru_hidden": BASELINE_ARCHITECTURE["gru_hidden"],
        "gru_layers": BASELINE_ARCHITECTURE["gru_layers"],
        "dilations": [1, 2, 4, 8],
        "bidirectional": False,
    }
    model_mismatches = {
        key: {"expected": expected, "actual": model.get(key)}
        for key, expected in expected_model.items()
        if model.get(key) != expected
    }
    if model_mismatches:
        raise ValueError(
            f"Frozen baseline architecture changed: {model_mismatches}"
        )

    split = manifest["chronological_split"]
    split_counts = {name: len(split[name]) for name in EXPECTED_SPLITS}
    if split_counts != EXPECTED_SPLITS:
        raise ValueError(
            f"Expected chronological split {EXPECTED_SPLITS}, found {split_counts}"
        )
    development = list(split["train"]) + list(split["validation"])
    january = list(split["test"])
    if any(session_month(name) >= "2017-01" for name in development):
        raise ValueError("January-or-later session leaked into development")
    if any(session_month(name) != "2017-01" for name in january):
        raise ValueError("The locked test split is no longer January 2017")
    if set(development) & set(january):
        raise ValueError("Development and locked test sessions overlap")
    by_month = {
        month: [name for name in development if session_month(name) == month]
        for month in EXPECTED_MONTHS
    }
    if any(not names for names in by_month.values()):
        raise ValueError(f"Development pool is missing a fixed fold: {by_month}")
    if set(name for names in by_month.values() for name in names) != set(
        development
    ):
        raise ValueError("Unexpected month found in the development pool")

    baseline_model = build_candidate_net(BASELINE_ARCHITECTURE, 64, DROPOUT)
    baseline_parameters = sum(
        parameter.numel() for parameter in baseline_model.parameters()
    )
    if baseline_parameters != EXPECTED_BASELINE_PARAMETERS:
        raise ValueError(
            "Independent baseline copy no longer matches the protected model: "
            f"{baseline_parameters:,} != {EXPECTED_BASELINE_PARAMETERS:,}"
        )
    del baseline_model
    return {
        "model_yaml": model_yaml,
        "manifest": manifest,
        "channels": channels,
        "development_sessions": development,
        "by_month": by_month,
        "january_sessions": january,
        "baseline_checkpoint_sha256": actual_hash,
        "baseline_parameters": baseline_parameters,
    }


def assert_baseline_untouched(expected_hash: str) -> None:
    current_hash = sha256_file(BASELINE_CHECKPOINT_PATH)
    if current_hash != expected_hash:
        raise RuntimeError(
            "Protected baseline checkpoint changed during Phase 4a; stopping"
        )


def processed_session_path(name: str, manifest: dict) -> Path:
    matches = [
        split
        for split, names in manifest["chronological_split"].items()
        if name in names
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one split for {name}, found {matches}")
    if matches[0] == "test":
        raise RuntimeError(f"Phase 4a refuses to load locked test session {name}")
    return PROCESSED_DIR / matches[0] / f"{name}.npz"


def load_model_data(name: str, manifest: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load one pre-January artifact and validate its causal target metadata."""
    path = processed_session_path(name, manifest)
    if not path.exists():
        raise FileNotFoundError(f"Missing processed session: {path}")
    with np.load(path, allow_pickle=False) as artifact:
        schema = str(np.asarray(artifact["schema_version"]).item())
        filter_name = str(np.asarray(artifact["velocity_filter"]).item())
        difference = str(np.asarray(artifact["velocity_difference"]).item())
        sampling = str(np.asarray(artifact["kinematic_sampling"]).item())
        if schema != MODEL_READY_SCHEMA:
            raise ValueError(f"Unsupported schema {schema!r} in {path}")
        if (
            filter_name != "causal_forward_butterworth"
            or difference != "backward"
            or sampling != "causal_latest_sample_at_bin_end"
        ):
            raise ValueError(f"Non-causal or unsupported target metadata in {path}")
        counts = artifact["counts"].astype(np.float32)
        velocity = artifact["velocity"].astype(np.float32)
    if counts.ndim != 2 or counts.shape[0] <= 94:
        raise ValueError(f"Unexpected count shape for {name}: {counts.shape}")
    if velocity.ndim != 2 or velocity.shape[1] < 2:
        raise ValueError(f"Unexpected velocity shape for {name}: {velocity.shape}")
    if counts.shape[1] != velocity.shape[0]:
        raise ValueError(f"Count/velocity timeline mismatch for {name}")
    return counts, velocity


def causal_ewma(values: np.ndarray, alpha: float) -> np.ndarray:
    if not 0 < alpha <= 1:
        raise ValueError("EWMA alpha must be in (0, 1]")
    output = values.astype(np.float64, copy=True)
    for index in range(1, values.shape[-1]):
        output[..., index] = (
            alpha * values[..., index]
            + (1.0 - alpha) * output[..., index - 1]
        )
    return output.astype(np.float32)


def multiscale_counts(counts: np.ndarray) -> np.ndarray:
    blocks = [
        counts.astype(np.float32) if alpha == 1.0 else causal_ewma(counts, alpha)
        for alpha in ALPHAS
    ]
    return np.concatenate(blocks, axis=0)


def fit_feature_stats(
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    observation = features[:, :OBSERVATION_BINS]
    return (
        observation.mean(axis=1, keepdims=True),
        observation.std(axis=1, keepdims=True) + 1e-6,
    )


def fit_feature_std_floor(
    loaded: dict[str, tuple[np.ndarray, np.ndarray]],
    channels: np.ndarray,
) -> np.ndarray:
    session_stds = []
    for counts, _ in loaded.values():
        features = multiscale_counts(counts[channels])
        _, std = fit_feature_stats(features)
        session_stds.append(std[:, 0])
    scales = np.stack(session_stds)
    floors = np.empty(scales.shape[1], dtype=np.float32)
    for feature in range(scales.shape[1]):
        valid = scales[:, feature][scales[:, feature] > 1e-4]
        if valid.size == 0:
            raise ValueError(f"Feature {feature} is silent in every prefix")
        floors[feature] = np.percentile(valid, STD_FLOOR_PERCENTILE)
    return floors[:, None]


def prepare_session(
    loaded: tuple[np.ndarray, np.ndarray],
    channels: np.ndarray,
    feature_std_floor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    counts, velocity = loaded
    features = multiscale_counts(counts[channels])
    if features.shape[1] < OBSERVATION_BINS + WINDOW_BINS:
        raise ValueError("Session is too short for the fixed causal protocol")
    mean, local_std = fit_feature_stats(features)
    normalized = (
        (features - mean) / np.maximum(local_std, feature_std_floor)
    ).astype(np.float32)
    usable = (normalized.shape[1] - OBSERVATION_BINS) // WINDOW_BINS
    x = np.stack(
        [
            normalized[
                :,
                OBSERVATION_BINS
                + index * WINDOW_BINS : OBSERVATION_BINS
                + (index + 1) * WINDOW_BINS,
            ]
            for index in range(usable)
        ]
    ).astype(np.float32)
    y = np.stack(
        [
            velocity[
                OBSERVATION_BINS
                + index * WINDOW_BINS : OBSERVATION_BINS
                + (index + 1) * WINDOW_BINS
            ][:, TARGET_AXES]
            for index in range(usable)
        ]
    ).astype(np.float32)
    return x, y


def fold_partition(
    development: list[str],
    by_month: dict[str, list[str]],
    held_month: str,
) -> tuple[list[str], list[str]]:
    held = list(by_month[held_month])
    training = [name for name in development if name not in held]
    if set(training) & set(held):
        raise AssertionError("Training and held sessions overlap")
    if {session_month(name) for name in held} != {held_month}:
        raise AssertionError("Held fold contains another month")
    if held_month in {session_month(name) for name in training}:
        raise AssertionError("Held month leaked into training")
    return training, held


def cache_signature(
    held_month: str,
    training_names: list[str],
    held_names: list[str],
    channels: np.ndarray,
) -> str:
    payload = {
        "schema": "phase4a_fold_cache_v1",
        "held_month": held_month,
        "training_names": training_names,
        "held_names": held_names,
        "channels": channels.tolist(),
        "observation_bins": OBSERVATION_BINS,
        "window_bins": WINDOW_BINS,
        "alphas": ALPHAS,
        "target_axes": TARGET_AXES,
        "std_floor_percentile": STD_FLOOR_PERCENTILE,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def train_cache_paths(held_month: str) -> dict[str, Path]:
    folder = CACHE_DIR / f"held_{held_month}"
    return {
        "folder": folder,
        "metadata": folder / "training_metadata.json",
        "x": folder / "training_x.npy",
        "y_norm": folder / "training_y_normalized.npy",
    }


def held_cache_paths(held_month: str) -> dict[str, Path]:
    folder = CACHE_DIR / f"held_{held_month}"
    return {
        "folder": folder,
        "metadata": folder / "held_metadata.json",
        "x": folder / "held_x.npy",
        "y": folder / "held_y.npy",
    }


def cache_is_valid(paths: dict[str, Path], expected_signature: str) -> bool:
    required = [path for name, path in paths.items() if name != "folder"]
    if not all(path.exists() for path in required):
        return False
    try:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return metadata.get("cache_signature") == expected_signature


def build_training_cache(
    held_month: str,
    training_names: list[str],
    held_names: list[str],
    manifest: dict,
    channels: np.ndarray,
    *,
    rebuild: bool,
) -> dict:
    paths = train_cache_paths(held_month)
    signature = cache_signature(
        held_month, training_names, held_names, channels
    )
    if not rebuild and cache_is_valid(paths, signature):
        return json.loads(paths["metadata"].read_text(encoding="utf-8"))

    print(
        f"building training cache for held {held_month} "
        f"({len(training_names)} sessions)",
        flush=True,
    )
    loaded = {
        name: load_model_data(name, manifest) for name in training_names
    }
    feature_std_floor = fit_feature_std_floor(loaded, channels)
    prepared = {
        name: prepare_session(loaded[name], channels, feature_std_floor)
        for name in training_names
    }
    train_x = np.concatenate(
        [prepared[name][0] for name in training_names], axis=0
    )
    train_y = np.concatenate(
        [prepared[name][1] for name in training_names], axis=0
    )
    target_mean = train_y.mean(axis=(0, 1))
    target_std = train_y.std(axis=(0, 1)) + 1e-6
    train_y_normalized = ((train_y - target_mean) / target_std).astype(
        np.float32
    )
    session_lengths = {
        name: int(len(prepared[name][0])) for name in training_names
    }
    metadata = {
        "cache_signature": signature,
        "held_month": held_month,
        "training_sessions": training_names,
        "held_sessions": held_names,
        "training_windows": int(len(train_x)),
        "session_lengths": session_lengths,
        "feature_std_floor": feature_std_floor[:, 0].tolist(),
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "january_loaded": False,
    }
    save_npy_atomic(train_x, paths["x"])
    save_npy_atomic(train_y_normalized, paths["y_norm"])
    write_json_atomic(metadata, paths["metadata"])
    del loaded, prepared, train_x, train_y, train_y_normalized
    return metadata


def build_held_cache(
    held_month: str,
    training_metadata: dict,
    manifest: dict,
    channels: np.ndarray,
    *,
    rebuild: bool,
) -> dict:
    """Prepare held arrays only after the candidate fold has finished training."""
    paths = held_cache_paths(held_month)
    signature = training_metadata["cache_signature"]
    if not rebuild and cache_is_valid(paths, signature):
        return json.loads(paths["metadata"].read_text(encoding="utf-8"))

    held_names = list(training_metadata["held_sessions"])
    print(
        f"  preparing held cache for {held_month} after optimization",
        flush=True,
    )
    loaded = {name: load_model_data(name, manifest) for name in held_names}
    floor = np.asarray(
        training_metadata["feature_std_floor"], dtype=np.float32
    )[:, None]
    prepared = {
        name: prepare_session(loaded[name], channels, floor)
        for name in held_names
    }
    held_x = np.concatenate([prepared[name][0] for name in held_names], axis=0)
    held_y = np.concatenate([prepared[name][1] for name in held_names], axis=0)
    slices = {}
    cursor = 0
    for name in held_names:
        stop = cursor + len(prepared[name][0])
        slices[name] = [cursor, stop]
        cursor = stop
    metadata = {
        "cache_signature": signature,
        "held_month": held_month,
        "held_sessions": held_names,
        "held_windows": int(len(held_x)),
        "session_slices": slices,
        "january_loaded": False,
        "held_labels_available_to_optimizer": False,
    }
    save_npy_atomic(held_x, paths["x"])
    save_npy_atomic(held_y, paths["y"])
    write_json_atomic(metadata, paths["metadata"])
    del loaded, prepared, held_x, held_y
    return metadata


def balanced_allocations(
    items: list[str], total: int, rng: np.random.Generator
) -> dict[str, int]:
    if not items:
        raise ValueError("Cannot balance an empty session list")
    base, remainder = divmod(total, len(items))
    allocation = {item: base for item in items}
    if remainder:
        for index in rng.permutation(len(items))[:remainder]:
            allocation[items[int(index)]] += 1
    return allocation


def draw_session_balanced_indices(
    sessions: list[str],
    session_lengths: dict[str, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, int]]:
    if set(sessions) != set(session_lengths):
        raise ValueError("Session lengths do not match the training fold")
    offsets = {}
    cursor = 0
    for name in sessions:
        offsets[name] = cursor
        cursor += int(session_lengths[name])
    allocation = balanced_allocations(sessions, cursor, rng)
    blocks = [
        rng.integers(0, session_lengths[name], size=allocation[name])
        + offsets[name]
        for name in sessions
    ]
    indices = np.concatenate(blocks).astype(np.int64)
    rng.shuffle(indices)
    month_draws: Counter[str] = Counter()
    for name, count in allocation.items():
        month_draws[session_month(name)] += count
    return indices, dict(sorted(month_draws.items()))


def receptive_field(architecture: dict) -> int:
    dilations = [2**index for index in range(architecture["tcn_blocks"])]
    return 1 + (architecture["kernel_size"] - 1) * sum(dilations)


def build_candidate_net(
    architecture: dict,
    n_features: int,
    dropout: float,
):
    """Independent generalized copy of the causal TCN+GRU baseline."""
    import torch.nn as nn

    filters = int(architecture["tcn_filters"])
    hidden = int(architecture["gru_hidden"])
    layers = int(architecture["gru_layers"])
    kernel_size = int(architecture["kernel_size"])
    dilations = [
        2**index for index in range(int(architecture["tcn_blocks"]))
    ]

    class PointwiseLayerNorm(nn.Module):
        def __init__(self, features: int) -> None:
            super().__init__()
            self.normalization = nn.LayerNorm(features)

        def forward(self, values):
            return self.normalization(values.transpose(1, 2)).transpose(1, 2)

    class CandidateCausalTCNGRU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spatial = nn.Sequential(
                nn.Conv1d(n_features, filters, 1),
                PointwiseLayerNorm(filters),
                nn.ReLU(),
            )
            self.convolutions = nn.ModuleList()
            self.right_crops = []
            for dilation in dilations:
                crop = (kernel_size - 1) * dilation
                self.convolutions.append(
                    nn.Conv1d(
                        filters,
                        filters,
                        kernel_size,
                        padding=crop,
                        dilation=dilation,
                    )
                )
                self.right_crops.append(crop)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.gru = nn.GRU(
                filters,
                hidden,
                layers,
                batch_first=True,
                bidirectional=False,
                dropout=dropout if layers > 1 else 0.0,
            )
            self.head = nn.Linear(hidden, 2)

        def forward(self, values):
            encoded = self.spatial(values)
            for convolution, crop in zip(
                self.convolutions, self.right_crops
            ):
                update = convolution(encoded)
                if crop:
                    update = update[:, :, :-crop]
                encoded = self.activation(update + encoded)
            encoded, _ = self.gru(
                self.dropout(encoded).transpose(1, 2)
            )
            return self.head(encoded)

    return CandidateCausalTCNGRU()


def r2(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    residual = ((actual - predicted) ** 2).sum(axis=0)
    total = ((actual - actual.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - residual / np.where(total == 0, 1e-9, total)


def choose_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def train_fold(
    architecture: dict,
    training_metadata: dict,
    device,
) -> tuple[object, list[dict]]:
    import torch
    import torch.nn as nn

    torch.manual_seed(FROZEN_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FROZEN_SEED)
    np.random.seed(FROZEN_SEED)
    rng = np.random.default_rng(FROZEN_SEED)

    paths = train_cache_paths(training_metadata["held_month"])
    train_x = np.load(paths["x"], mmap_mode="r")
    train_y = np.load(paths["y_norm"], mmap_mode="r")
    sessions = list(training_metadata["training_sessions"])
    session_lengths = {
        name: int(value)
        for name, value in training_metadata["session_lengths"].items()
    }
    net = build_candidate_net(architecture, train_x.shape[1], DROPOUT).to(
        device
    )
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, SCHEDULER_EPOCH_BUDGET
    )
    mse = nn.MSELoss()
    history = []
    for epoch in range(1, TRAIN_EPOCHS + 1):
        indices, month_draws = draw_session_balanced_indices(
            sessions, session_lengths, rng
        )
        net.train()
        error_sum = 0.0
        value_count = 0
        gradient_sum = 0.0
        gradient_max = 0.0
        batches = 0
        for start in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[start : start + BATCH_SIZE]
            x_batch = torch.from_numpy(
                np.asarray(train_x[batch_indices])
            ).to(device)
            y_batch = torch.from_numpy(
                np.asarray(train_y[batch_indices])
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = mse(net(x_batch), y_batch)
            loss.backward()
            gradient = float(
                torch.nn.utils.clip_grad_norm_(
                    net.parameters(), GRADIENT_CLIP
                )
            )
            optimizer.step()
            error_sum += float(loss.detach().item()) * y_batch.numel()
            value_count += y_batch.numel()
            gradient_sum += gradient
            gradient_max = max(gradient_max, gradient)
            batches += 1
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "optimization_loss": error_sum / max(value_count, 1),
            "gradient_norm_mean_before_clip": gradient_sum / max(batches, 1),
            "gradient_norm_max_before_clip": gradient_max,
            "month_draws": month_draws,
        }
        history.append(row)
        shares = ", ".join(
            f"{month}:{count / len(indices):.1%}"
            for month, count in month_draws.items()
        )
        print(
            f"    epoch {epoch:02d}/{TRAIN_EPOCHS} | "
            f"opt={row['optimization_loss']:.5f} | "
            f"lr={row['learning_rate']:.6g} | "
            f"grad={row['gradient_norm_mean_before_clip']:.3f}/"
            f"{row['gradient_norm_max_before_clip']:.3f} | {shares}",
            flush=True,
        )
        scheduler.step()
    del train_x, train_y
    return net, history


def predict(net, x: np.ndarray, device) -> np.ndarray:
    import torch

    predictions = []
    net.eval()
    with torch.inference_mode():
        for start in range(0, len(x), BATCH_SIZE):
            batch = torch.from_numpy(
                np.array(x[start : start + BATCH_SIZE], copy=True)
            ).to(device)
            predictions.append(net(batch).cpu().numpy())
    return np.concatenate(predictions, axis=0).astype(np.float32)


def normalized_loss(
    actual: np.ndarray,
    predicted: np.ndarray,
    target_std: np.ndarray,
) -> float:
    return float(np.mean(((actual - predicted) / target_std) ** 2))


def evaluate_fold(
    net,
    held_month: str,
    held_metadata: dict,
    training_metadata: dict,
    device,
) -> tuple[dict, list[dict]]:
    paths = held_cache_paths(held_month)
    held_x = np.load(paths["x"], mmap_mode="r")
    held_y = np.load(paths["y"], mmap_mode="r")
    target_mean = np.asarray(
        training_metadata["target_mean"], dtype=np.float32
    )
    target_std = np.asarray(
        training_metadata["target_std"], dtype=np.float32
    )
    prediction_normalized = predict(net, held_x, device)
    prediction = prediction_normalized * target_std + target_mean
    pooled_score = r2(
        np.asarray(held_y).reshape(-1, 2),
        prediction.reshape(-1, 2),
    )
    session_rows = []
    for name in held_metadata["held_sessions"]:
        start, stop = held_metadata["session_slices"][name]
        actual = np.asarray(held_y[start:stop])
        estimate = prediction[start:stop]
        score = r2(actual.reshape(-1, 2), estimate.reshape(-1, 2))
        session_rows.append(
            {
                "session": name,
                "held_month": held_month,
                "windows": int(stop - start),
                "loss": normalized_loss(actual, estimate, target_std),
                "r2_x": float(score[0]),
                "r2_y": float(score[1]),
                "r2_mean": float(score.mean()),
            }
        )
    fold_row = {
        "held_month": held_month,
        "training_sessions": len(training_metadata["training_sessions"]),
        "held_sessions": len(held_metadata["held_sessions"]),
        "training_windows": int(training_metadata["training_windows"]),
        "held_windows": int(held_metadata["held_windows"]),
        "pooled_loss": normalized_loss(
            np.asarray(held_y), prediction, target_std
        ),
        "pooled_r2_x": float(pooled_score[0]),
        "pooled_r2_y": float(pooled_score[1]),
        "pooled_r2_mean": float(pooled_score.mean()),
        "session_macro_r2": float(
            np.mean([row["r2_mean"] for row in session_rows])
        ),
        "worst_session_r2": float(
            np.min([row["r2_mean"] for row in session_rows])
        ),
    }
    del held_x, held_y, prediction_normalized, prediction
    return fold_row, session_rows


def selection_metrics(session_rows: list[dict]) -> dict:
    scores = np.asarray(
        [row["r2_mean"] for row in session_rows], dtype=np.float64
    )
    macro = float(scores.mean())
    q10 = float(np.quantile(scores, 0.10))
    worst = float(scores.min())
    score = (
        SELECTION_WEIGHTS["session_macro_r2"] * macro
        + SELECTION_WEIGHTS["session_q10_r2"] * q10
    )
    return {
        "selection_score": float(score),
        "session_macro_r2": macro,
        "session_q10_r2": q10,
        "worst_session_r2": worst,
        "sessions_evaluated": int(len(scores)),
    }


def architecture_from_trial(trial) -> dict:
    return {
        name: trial.suggest_categorical(name, values)
        for name, values in SEARCH_SPACE.items()
    }


def trial_record(trial) -> dict:
    duration = None
    if trial.datetime_start is not None and trial.datetime_complete is not None:
        duration = (
            trial.datetime_complete - trial.datetime_start
        ).total_seconds()
    return {
        "number": trial.number,
        "state": trial.state.name,
        "value": trial.value,
        "params": dict(trial.params),
        "user_attrs": dict(trial.user_attrs),
        "intermediate_values": {
            str(step): value
            for step, value in trial.intermediate_values.items()
        },
        "datetime_start": (
            trial.datetime_start.isoformat()
            if trial.datetime_start is not None
            else None
        ),
        "datetime_complete": (
            trial.datetime_complete.isoformat()
            if trial.datetime_complete is not None
            else None
        ),
        "duration_seconds": duration,
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


def write_trials_csv(study) -> None:
    fields = [
        "trial",
        "state",
        "selection_score",
        *SEARCH_SPACE,
        "parameters",
        "receptive_field_bins",
        "session_macro_r2",
        "session_q10_r2",
        "worst_session_r2",
        "folds_completed",
        "duration_seconds",
    ]
    TRIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = TRIALS_PATH.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for trial in study.trials:
            record = trial_record(trial)
            attrs = record["user_attrs"]
            writer.writerow(
                {
                    "trial": trial.number,
                    "state": trial.state.name,
                    "selection_score": trial.value,
                    **{
                        name: trial.params.get(name)
                        for name in SEARCH_SPACE
                    },
                    "parameters": attrs.get("parameter_count"),
                    "receptive_field_bins": attrs.get(
                        "receptive_field_bins"
                    ),
                    "session_macro_r2": attrs.get("session_macro_r2"),
                    "session_q10_r2": attrs.get("session_q10_r2"),
                    "worst_session_r2": attrs.get("worst_session_r2"),
                    "folds_completed": attrs.get("folds_completed"),
                    "duration_seconds": record["duration_seconds"],
                }
            )
    temporary.replace(TRIALS_PATH)


def plot_trials(study) -> None:
    complete = sorted(completed_trials(study), key=lambda trial: trial.number)
    if not complete:
        return
    cache = Path(tempfile.gettempdir()) / "indy_phase4a_matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=170)
    fields = list(SEARCH_SPACE)
    scores = np.asarray([trial.value for trial in complete])
    numbers = np.asarray([trial.number for trial in complete])
    baseline_numbers = [
        trial.number
        for trial in complete
        if trial.params == BASELINE_ARCHITECTURE
    ]
    for axis, field in zip(axes.flat[:5], fields):
        values = np.asarray([trial.params[field] for trial in complete])
        points = axis.scatter(
            values,
            scores,
            c=numbers,
            cmap="viridis",
            edgecolor="#263238",
            linewidth=0.4,
        )
        axis.set(
            xlabel=field.replace("_", " ").title(),
            ylabel="Selection score",
            title=f"Score vs {field.replace('_', ' ')}",
        )
        figure.colorbar(points, ax=axis, label="Trial")
    parameters = np.asarray(
        [trial.user_attrs["parameter_count"] for trial in complete]
    )
    axes[1, 2].scatter(
        parameters,
        scores,
        c=numbers,
        cmap="viridis",
        edgecolor="#263238",
        linewidth=0.4,
    )
    if baseline_numbers:
        baseline = next(
            trial for trial in complete if trial.number == baseline_numbers[0]
        )
        axes[1, 2].scatter(
            [baseline.user_attrs["parameter_count"]],
            [baseline.value],
            marker="*",
            s=180,
            color="#D62728",
            label="Protected baseline copy",
        )
        axes[1, 2].legend(frameon=False)
    axes[1, 2].set(
        xlabel="Parameters",
        ylabel="Selection score",
        title="Accuracy–size trade-off",
    )
    figure.suptitle(
        "Phase 4a architecture-only sweep "
        "(75% macro R² + 25% session-q10 R²)"
    )
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.close(figure)


def protocol_signature(max_parameters: int) -> str:
    payload = {
        "search_space": SEARCH_SPACE,
        "fold_order": FOLD_ORDER,
        "fixed_seed": FROZEN_SEED,
        "train_epochs": TRAIN_EPOCHS,
        "scheduler_epoch_budget": SCHEDULER_EPOCH_BUDGET,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "gradient_clip": GRADIENT_CLIP,
        "sampler": SAMPLER_NAME,
        "observation_bins": OBSERVATION_BINS,
        "window_bins": WINDOW_BINS,
        "alphas": ALPHAS,
        "selection_weights": SELECTION_WEIGHTS,
        "max_parameters": max_parameters,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def write_outputs(
    study,
    context: dict,
    started_at: str,
    max_parameters: int,
) -> None:
    assert_baseline_untouched(context["baseline_checkpoint_sha256"])
    complete = completed_trials(study)
    best = max(complete, key=lambda trial: trial.value) if complete else None
    payload = {
        "phase": "4a",
        "purpose": "architecture_only_sweep_protected_baseline",
        "updated_at_utc": utc_now(),
        "study_started_at_utc": started_at,
        "study_name": study.study_name,
        "direction": "maximize",
        "objective": {
            "selection_score": (
                "0.75 * held-session macro R2 + "
                "0.25 * held-session 10th-percentile R2"
            ),
            "weights": SELECTION_WEIGHTS,
            "worst_session_r2": "reported guardrail, not optimized directly",
        },
        "search_space": SEARCH_SPACE,
        "baseline_architecture": BASELINE_ARCHITECTURE,
        "fixed_protocol": {
            "seed": FROZEN_SEED,
            "train_epochs": TRAIN_EPOCHS,
            "scheduler_epoch_budget": SCHEDULER_EPOCH_BUDGET,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "gradient_clip": GRADIENT_CLIP,
            "sampler": SAMPLER_NAME,
            "channels": context["channels"].tolist(),
            "observation_seconds": OBSERVATION_SECONDS,
            "window_bins": WINDOW_BINS,
            "fold_order": FOLD_ORDER,
            "max_parameters": max_parameters,
        },
        "data_policy": {
            "development_sessions": 33,
            "folds": list(EXPECTED_MONTHS),
            "fold_unit": "complete_month",
            "january_loaded": False,
            "locked_test_used_for_selection": False,
            "held_labels_available_to_optimizer": False,
            "held_metric_selects_epoch": False,
        },
        "baseline_protection": {
            "path": str(BASELINE_CHECKPOINT_PATH.relative_to(ROOT)),
            "expected_sha256": context["baseline_checkpoint_sha256"],
            "current_sha256": sha256_file(BASELINE_CHECKPOINT_PATH),
            "checkpoint_written_by_phase4a": False,
            "candidate_checkpoint_saved": False,
        },
        "best_trial": trial_record(best) if best is not None else None,
        "trials": [trial_record(trial) for trial in study.trials],
        "trial_counts": {
            state: sum(trial.state.name == state for trial in study.trials)
            for state in ("COMPLETE", "PRUNED", "FAIL", "RUNNING", "WAITING")
        },
        "artifacts": {
            "study_database": (
                str(context["storage_path"].relative_to(ROOT))
                if context["storage_path"].is_relative_to(ROOT)
                else str(context["storage_path"])
            ),
            "trial_table": str(TRIALS_PATH.relative_to(ROOT)),
            "figure": str(FIGURE_PATH.relative_to(ROOT)),
            "fold_cache": str(CACHE_DIR.relative_to(ROOT)),
            "checkpoint": None,
        },
        "next_gate": (
            "Phase 4b must confirm finalists across multiple seeds before any "
            "new model can be considered; January remains unavailable."
        ),
    }
    write_json_atomic(payload, METRICS_PATH)
    write_trials_csv(study)
    plot_trials(study)


def validate_study_protocol(study, signature: str) -> None:
    existing = study.user_attrs.get("protocol_signature")
    if existing is None:
        study.set_user_attr("protocol_signature", signature)
        study.set_user_attr("created_at_utc", utc_now())
        study.set_user_attr("january_policy", "forbidden_not_loaded")
    elif existing != signature:
        raise ValueError(
            "Existing Optuna study uses a different Phase-4a protocol. "
            "Use a new --study-name or --storage-path."
        )


def validate_only(context: dict, max_parameters: int) -> None:
    import optuna  # noqa: F401
    import torch

    first = context["development_sessions"][0]
    counts, velocity = load_model_data(first, context["manifest"])
    if counts.shape[1] != velocity.shape[0]:
        raise AssertionError("Validation session timeline changed")

    minimum_parameters = None
    maximum_parameters = None
    valid_under_cap = 0
    for filters in SEARCH_SPACE["tcn_filters"]:
        for hidden in SEARCH_SPACE["gru_hidden"]:
            for blocks in SEARCH_SPACE["tcn_blocks"]:
                for kernel in SEARCH_SPACE["kernel_size"]:
                    for layers in SEARCH_SPACE["gru_layers"]:
                        architecture = {
                            "tcn_filters": filters,
                            "gru_hidden": hidden,
                            "tcn_blocks": blocks,
                            "kernel_size": kernel,
                            "gru_layers": layers,
                        }
                        net = build_candidate_net(architecture, 64, DROPOUT)
                        parameters = sum(
                            value.numel() for value in net.parameters()
                        )
                        minimum_parameters = (
                            parameters
                            if minimum_parameters is None
                            else min(minimum_parameters, parameters)
                        )
                        maximum_parameters = (
                            parameters
                            if maximum_parameters is None
                            else max(maximum_parameters, parameters)
                        )
                        valid_under_cap += int(parameters <= max_parameters)
                        net.eval()
                        sample = torch.randn(1, 64, WINDOW_BINS)
                        changed = sample.clone()
                        changed[:, :, 25:] += 100.0
                        with torch.inference_mode():
                            original_output = net(sample)
                            changed_output = net(changed)
                        if not torch.equal(
                            original_output[:, :25], changed_output[:, :25]
                        ):
                            raise AssertionError(
                                f"Candidate architecture is non-causal: {architecture}"
                            )
                        del net

    assert_baseline_untouched(context["baseline_checkpoint_sha256"])
    total = math.prod(len(values) for values in SEARCH_SPACE.values())
    print("=== Phase 4a validation passed ===")
    print(
        f"development sessions=33 | folds={len(EXPECTED_MONTHS)} | "
        "January=FORBIDDEN"
    )
    print(
        f"architecture combinations={total} | under parameter cap="
        f"{valid_under_cap} | parameter range={minimum_parameters:,}--"
        f"{maximum_parameters:,}"
    )
    print(
        f"protected baseline={context['baseline_parameters']:,} parameters | "
        f"sha256={context['baseline_checkpoint_sha256']}"
    )
    print("no cache, Optuna study, result or checkpoint was written")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Target total trials, including the enqueued baseline trial.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="CPU is deterministic on Mac; auto selects CUDA or CPU, never MPS.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout-hours", type=float)
    parser.add_argument("--study-name", default="phase4a_architecture_sweep")
    parser.add_argument("--storage-path", type=Path, default=STORAGE_PATH)
    parser.add_argument("--sampler-startup-trials", type=int, default=8)
    parser.add_argument("--pruner-startup-trials", type=int, default=8)
    parser.add_argument(
        "--pruner-min-folds",
        type=int,
        default=2,
        help="No pruning before this many complete held-month folds.",
    )
    parser.add_argument(
        "--max-parameter-multiplier",
        type=float,
        default=2.0,
        help="Candidate parameter cap relative to the 78,786-parameter baseline.",
    )
    parser.add_argument("--no-pruning", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check isolation, causality and architecture construction; write nothing.",
    )
    return parser.parse_args()


def main() -> None:
    import optuna
    import torch

    args = parse_args()
    if args.n_trials <= 0:
        raise ValueError("--n-trials must be positive")
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    if args.sampler_startup_trials < 0 or args.pruner_startup_trials < 0:
        raise ValueError("Pruner/sampler startup trials cannot be negative")
    if not 1 <= args.pruner_min_folds <= len(FOLD_ORDER):
        raise ValueError("--pruner-min-folds must be between 1 and 5")
    if args.max_parameter_multiplier < 1.0:
        raise ValueError("--max-parameter-multiplier must be at least 1.0")

    torch.set_num_threads(args.threads)
    context = validate_frozen_protocol()
    max_parameters = int(
        math.floor(
            context["baseline_parameters"] * args.max_parameter_multiplier
        )
    )
    if args.validate_only:
        validate_only(context, max_parameters)
        return

    device = choose_device(args.device)
    started_at = utc_now()
    started = time.time()
    print("=== Phase 4a protected architecture-only sweep ===")
    print(
        f"target trials={args.n_trials} | folds={','.join(FOLD_ORDER)} | "
        f"device={device} | seed={FROZEN_SEED}"
    )
    print(
        f"fixed: lr={LEARNING_RATE} | wd={WEIGHT_DECAY} | "
        f"dropout={DROPOUT} | sampler={SAMPLER_NAME} | "
        f"epochs={TRAIN_EPOCHS}/{SCHEDULER_EPOCH_BUDGET}-epoch schedule"
    )
    print(
        f"parameter cap={max_parameters:,} "
        f"({args.max_parameter_multiplier:.2f}x baseline)"
    )
    print("January: FORBIDDEN | baseline checkpoint: read-only SHA check")

    fold_training_metadata = {}
    for held_month in FOLD_ORDER:
        training_names, held_names = fold_partition(
            context["development_sessions"],
            context["by_month"],
            held_month,
        )
        fold_training_metadata[held_month] = build_training_cache(
            held_month,
            training_names,
            held_names,
            context["manifest"],
            context["channels"],
            rebuild=args.rebuild_cache,
        )
    assert_baseline_untouched(context["baseline_checkpoint_sha256"])

    storage_path = args.storage_path
    if not storage_path.is_absolute():
        storage_path = (ROOT / storage_path).resolve()
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    context["storage_path"] = storage_path
    storage_url = f"sqlite:///{storage_path}"
    sampler = optuna.samplers.TPESampler(
        seed=OPTUNA_SAMPLER_SEED,
        n_startup_trials=args.sampler_startup_trials,
    )
    pruner = (
        optuna.pruners.NopPruner()
        if args.no_pruning
        else optuna.pruners.MedianPruner(
            n_startup_trials=args.pruner_startup_trials,
            n_warmup_steps=args.pruner_min_folds - 1,
            interval_steps=1,
        )
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )
    signature = protocol_signature(max_parameters)
    validate_study_protocol(study, signature)
    study.enqueue_trial(BASELINE_ARCHITECTURE, skip_if_exists=True)
    rebuilt_held_caches: set[str] = set()

    def objective(trial):
        architecture = architecture_from_trial(trial)
        net_template = build_candidate_net(architecture, 64, DROPOUT)
        parameter_count = sum(
            parameter.numel() for parameter in net_template.parameters()
        )
        del net_template
        field = receptive_field(architecture)
        trial.set_user_attr("architecture", architecture)
        trial.set_user_attr("parameter_count", parameter_count)
        trial.set_user_attr("parameter_multiplier", parameter_count / context["baseline_parameters"])
        trial.set_user_attr("receptive_field_bins", field)
        trial.set_user_attr(
            "receptive_field_seconds", field * BIN_SECONDS
        )
        trial.set_user_attr("fold_order", list(FOLD_ORDER))
        trial.set_user_attr("january_loaded", False)
        if parameter_count > max_parameters:
            trial.set_user_attr("prune_reason", "parameter_cap")
            raise optuna.TrialPruned(
                f"{parameter_count:,} parameters exceeds cap {max_parameters:,}"
            )

        print(
            f"\n=== trial {trial.number:03d} | {architecture} | "
            f"parameters={parameter_count:,} | RF={field} bins ===",
            flush=True,
        )
        fold_rows = []
        session_rows = []
        for fold_index, held_month in enumerate(FOLD_ORDER):
            metadata = fold_training_metadata[held_month]
            print(
                f"  fold {fold_index + 1}/{len(FOLD_ORDER)} "
                f"held={held_month}",
                flush=True,
            )
            net, history = train_fold(architecture, metadata, device)
            # Held arrays and labels become accessible only after optimization.
            held_metadata = build_held_cache(
                held_month,
                metadata,
                context["manifest"],
                context["channels"],
                rebuild=(
                    args.rebuild_cache
                    and held_month not in rebuilt_held_caches
                ),
            )
            rebuilt_held_caches.add(held_month)
            fold_row, held_session_rows = evaluate_fold(
                net,
                held_month,
                held_metadata,
                metadata,
                device,
            )
            fold_row["history"] = history
            fold_rows.append(fold_row)
            session_rows.extend(held_session_rows)
            partial = selection_metrics(session_rows)
            trial.set_user_attr("folds", fold_rows)
            trial.set_user_attr("sessions", session_rows)
            trial.set_user_attr("folds_completed", fold_index + 1)
            trial.set_user_attr("partial_metrics", partial)
            trial.report(partial["selection_score"], step=fold_index)
            print(
                f"  held {held_month} | pooled R2="
                f"{fold_row['pooled_r2_mean']:+.4f} | macro="
                f"{fold_row['session_macro_r2']:+.4f} | worst="
                f"{fold_row['worst_session_r2']:+.4f} | "
                f"partial score={partial['selection_score']:+.4f}",
                flush=True,
            )
            del net
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if (
                fold_index + 1 >= args.pruner_min_folds
                and trial.should_prune()
            ):
                trial.set_user_attr("prune_reason", "median_after_complete_fold")
                raise optuna.TrialPruned(
                    f"Pruned after {fold_index + 1} complete folds"
                )

        metrics = selection_metrics(session_rows)
        for name, value in metrics.items():
            trial.set_user_attr(name, value)
        trial.set_user_attr("folds", fold_rows)
        trial.set_user_attr("sessions", session_rows)
        trial.set_user_attr("folds_completed", len(FOLD_ORDER))
        print(
            f"trial {trial.number:03d} complete | score="
            f"{metrics['selection_score']:+.4f} | macro="
            f"{metrics['session_macro_r2']:+.4f} | q10="
            f"{metrics['session_q10_r2']:+.4f} | worst="
            f"{metrics['worst_session_r2']:+.4f}",
            flush=True,
        )
        assert_baseline_untouched(context["baseline_checkpoint_sha256"])
        return metrics["selection_score"]

    def callback(current_study, _trial):
        write_outputs(
            current_study,
            context,
            started_at,
            max_parameters,
        )

    finished = sum(
        trial.state.name in {"COMPLETE", "PRUNED", "FAIL"}
        for trial in study.trials
    )
    remaining = max(args.n_trials - finished, 0)
    if remaining:
        study.optimize(
            objective,
            n_trials=remaining,
            timeout=(
                None
                if args.timeout_hours is None
                else args.timeout_hours * 3600
            ),
            callbacks=[callback],
            gc_after_trial=True,
        )
    write_outputs(study, context, started_at, max_parameters)
    assert_baseline_untouched(context["baseline_checkpoint_sha256"])

    complete = completed_trials(study)
    print("\n=== Phase 4a summary ===")
    print(
        f"elapsed={(time.time() - started) / 3600:.2f} h | "
        f"complete={len(complete)} | total records={len(study.trials)}"
    )
    if complete:
        best = max(complete, key=lambda trial: trial.value)
        print(
            f"best trial={best.number:03d} | score={best.value:+.4f} | "
            f"architecture={best.params}"
        )
        print(
            f"macro={best.user_attrs['session_macro_r2']:+.4f} | "
            f"q10={best.user_attrs['session_q10_r2']:+.4f} | "
            f"worst={best.user_attrs['worst_session_r2']:+.4f} | "
            f"parameters={best.user_attrs['parameter_count']:,}"
        )
    print(f"metrics: {METRICS_PATH}")
    print(f"trials: {TRIALS_PATH}")
    print(f"figure: {FIGURE_PATH}")
    print("checkpoint saved: NO | protected baseline modified: NO")
    print("January: FORBIDDEN and not loaded")


if __name__ == "__main__":
    main()
