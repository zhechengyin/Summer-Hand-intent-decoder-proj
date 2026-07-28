#!/usr/bin/env python3
"""Phase 4b: five-seed confirmation of the 64/64 and 48/48 architectures.

This runner is self-contained and does not import Phase 4a, archived training
code, or a shared ``common`` module. It compares exactly two causal TCN+GRU
architectures over seeds 42--46 and five complete pre-January held-month folds.

All data processing, optimizer settings, sampling rules and epoch budgets are
fixed. January arrays are forbidden. Held-month labels are opened only after
that fold has finished optimization, and they never update a weight or select
an epoch. The protected baseline checkpoint is checksum-verified but is never
loaded, modified or overwritten. No candidate checkpoint is saved.

Completed folds are stored under the Phase-4b cache so an interrupted run can
resume without retraining finished cells.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG_PATH = ROOT / "configs" / "indy_32ch.yaml"
SESSION_MANIFEST_PATH = ROOT / "configs" / "datasets" / "indy_sessions.yaml"
BASELINE_CHECKPOINT_PATH = ROOT / "models" / "indy_32ch" / "64x64checkpoint.pt"
PROCESSED_DIR = ROOT / "data" / "processed" / "indy_loco" / "indy"

RESULT_DIR = ROOT / "results" / "indy" / "phase4b_five_seed_confirmation"
CACHE_DIR = RESULT_DIR / ".cache"
PROGRESS_DIR = CACHE_DIR / "progress"
METRICS_PATH = RESULT_DIR / "phase4b_five_seed_confirmation_metrics.json"
CELLS_PATH = RESULT_DIR / "phase4b_five_seed_confirmation_cells.csv"
FOLDS_PATH = RESULT_DIR / "phase4b_five_seed_confirmation_folds.csv"
SESSIONS_PATH = RESULT_DIR / "phase4b_five_seed_confirmation_sessions.csv"
FIGURE_PATH = RESULT_DIR / "phase4b_five_seed_confirmation_figure.png"

MODEL_READY_SCHEMA = "indy_counts_velocity_v2"
EXPECTED_SPLITS = {"train": 29, "validation": 4, "test": 4}
EXPECTED_MONTHS = ("2016-04", "2016-06", "2016-09", "2016-10", "2016-12")
FOLD_ORDER = ("2016-06", "2016-10", "2016-12", "2016-04", "2016-09")
SEEDS = (42, 43, 44, 45, 46)

BIN_SECONDS = 0.04
OBSERVATION_SECONDS = 60
OBSERVATION_BINS = 1500
WINDOW_BINS = 50
TARGET_AXES = (0, 1)
ALPHAS = (1.0, 0.1)
STD_FLOOR_PERCENTILE = 10.0

TRAIN_EPOCHS = 7
SCHEDULER_EPOCH_BUDGET = 20
LEARNING_RATE = 9e-4
WEIGHT_DECAY = 0.060
DROPOUT = 0.025
BATCH_SIZE = 32
GRADIENT_CLIP = 1.0
SAMPLER_NAME = "session_balanced"

ARCHITECTURES = {
    "baseline_64x64": {
        "tcn_filters": 64,
        "gru_hidden": 64,
        "tcn_blocks": 4,
        "kernel_size": 3,
        "gru_layers": 1,
    },
    "candidate_48x48": {
        "tcn_filters": 48,
        "gru_hidden": 48,
        "tcn_blocks": 4,
        "kernel_size": 3,
        "gru_layers": 1,
    },
}
EXPECTED_PARAMETERS = {
    "baseline_64x64": 78_786,
    "candidate_48x48": 45_266,
}
SELECTION_WEIGHTS = {"session_macro_r2": 0.75, "session_q10_r2": 0.25}

# Predeclared deployment-oriented non-inferiority guardrails. The smaller
# candidate is promoted only if every guardrail passes across all five seeds.
NONINFERIORITY_LIMITS = {
    "session_macro_r2_mean_delta_min": -0.010,
    "session_q10_r2_mean_delta_min": -0.020,
    "worst_session_r2_mean_delta_min": -0.020,
    "worst_month_macro_r2_mean_delta_min": -0.020,
}


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
    """Validate all fixed choices and capture the protected checkpoint hash."""
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
    baseline = ARCHITECTURES["baseline_64x64"]
    expected_model = {
        "family": "tcn_gru",
        "filters": baseline["tcn_filters"],
        "gru_hidden": baseline["gru_hidden"],
        "gru_layers": baseline["gru_layers"],
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
            f"Expected chronological split {EXPECTED_SPLITS}, found "
            f"{split_counts}"
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

    parameter_counts = {}
    for name, architecture in ARCHITECTURES.items():
        net = build_candidate_net(architecture, 64, DROPOUT)
        parameter_counts[name] = sum(
            parameter.numel() for parameter in net.parameters()
        )
        if parameter_counts[name] != EXPECTED_PARAMETERS[name]:
            raise ValueError(
                f"{name} parameter count changed: {parameter_counts[name]:,} "
                f"!= {EXPECTED_PARAMETERS[name]:,}"
            )
        del net

    return {
        "model_yaml": model_yaml,
        "manifest": manifest,
        "channels": channels,
        "development_sessions": development,
        "by_month": by_month,
        "january_sessions": january,
        "baseline_checkpoint_sha256": actual_hash,
        "parameter_counts": parameter_counts,
    }


def assert_baseline_untouched(expected_hash: str) -> None:
    current_hash = sha256_file(BASELINE_CHECKPOINT_PATH)
    if current_hash != expected_hash:
        raise RuntimeError(
            "Protected baseline checkpoint changed during Phase 4b; stopping"
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
        raise RuntimeError(f"Phase 4b refuses to load locked test session {name}")
    return PROCESSED_DIR / matches[0] / f"{name}.npz"


def load_model_data(name: str, manifest: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load one pre-January session and validate causal-target metadata."""
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
            raise ValueError(
                f"Non-causal or unsupported target metadata in {path}"
            )
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


def fold_cache_signature(
    held_month: str,
    training_names: list[str],
    held_names: list[str],
    channels: np.ndarray,
) -> str:
    payload = {
        "schema": "phase4b_fold_cache_v1",
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


def fold_cache_paths(held_month: str) -> dict[str, Path]:
    folder = CACHE_DIR / f"held_{held_month}"
    return {
        "folder": folder,
        "training_metadata": folder / "training_metadata.json",
        "training_x": folder / "training_x.npy",
        "training_y_norm": folder / "training_y_normalized.npy",
        "held_metadata": folder / "held_metadata.json",
        "held_x": folder / "held_x.npy",
        "held_y": folder / "held_y.npy",
    }


def cache_metadata_valid(
    metadata_path: Path,
    required_paths: list[Path],
    expected_signature: str,
) -> bool:
    if not metadata_path.exists() or not all(path.exists() for path in required_paths):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
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
    paths = fold_cache_paths(held_month)
    signature = fold_cache_signature(
        held_month, training_names, held_names, channels
    )
    valid = cache_metadata_valid(
        paths["training_metadata"],
        [paths["training_x"], paths["training_y_norm"]],
        signature,
    )
    if valid and not rebuild:
        return json.loads(
            paths["training_metadata"].read_text(encoding="utf-8")
        )

    print(
        f"building Phase-4b training cache for held {held_month} "
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
    metadata = {
        "cache_signature": signature,
        "held_month": held_month,
        "training_sessions": training_names,
        "held_sessions": held_names,
        "training_windows": int(len(train_x)),
        "session_lengths": {
            name: int(len(prepared[name][0])) for name in training_names
        },
        "feature_std_floor": feature_std_floor[:, 0].tolist(),
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "january_loaded": False,
    }
    save_npy_atomic(train_x, paths["training_x"])
    save_npy_atomic(train_y_normalized, paths["training_y_norm"])
    write_json_atomic(metadata, paths["training_metadata"])
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
    """Prepare held arrays only after the current fold finishes optimization."""
    paths = fold_cache_paths(held_month)
    signature = training_metadata["cache_signature"]
    valid = cache_metadata_valid(
        paths["held_metadata"],
        [paths["held_x"], paths["held_y"]],
        signature,
    )
    if valid and not rebuild:
        return json.loads(paths["held_metadata"].read_text(encoding="utf-8"))

    held_names = list(training_metadata["held_sessions"])
    print(
        f"    preparing held cache for {held_month} after optimization",
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
    save_npy_atomic(held_x, paths["held_x"])
    save_npy_atomic(held_y, paths["held_y"])
    write_json_atomic(metadata, paths["held_metadata"])
    del loaded, prepared, held_x, held_y
    return metadata


def balanced_allocations(
    items: list[str],
    total: int,
    rng: np.random.Generator,
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
    """Independent generalized copy of the protected causal TCN+GRU."""
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


def configure_determinism(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def train_fold(
    architecture: dict,
    seed: int,
    training_metadata: dict,
    device,
) -> tuple[object, list[dict]]:
    import torch
    import torch.nn as nn

    configure_determinism(seed)
    rng = np.random.default_rng(seed)
    paths = fold_cache_paths(training_metadata["held_month"])
    train_x = np.load(paths["training_x"], mmap_mode="r")
    train_y = np.load(paths["training_y_norm"], mmap_mode="r")
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
            f"      epoch {epoch:02d}/{TRAIN_EPOCHS} | "
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
    paths = fold_cache_paths(held_month)
    held_x = np.load(paths["held_x"], mmap_mode="r")
    held_y = np.load(paths["held_y"], mmap_mode="r")
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
    selection_score = (
        SELECTION_WEIGHTS["session_macro_r2"] * macro
        + SELECTION_WEIGHTS["session_q10_r2"] * q10
    )
    return {
        "selection_score": float(selection_score),
        "session_macro_r2": macro,
        "session_q10_r2": q10,
        "worst_session_r2": worst,
        "sessions_evaluated": int(len(scores)),
    }


def runtime_signature(device, threads: int, checkpoint_hash: str) -> str:
    import torch

    payload = {
        "phase": "4b",
        "schema": "five_seed_architecture_confirmation_v1",
        "architectures": ARCHITECTURES,
        "seeds": SEEDS,
        "fold_order": FOLD_ORDER,
        "selection_weights": SELECTION_WEIGHTS,
        "noninferiority_limits": NONINFERIORITY_LIMITS,
        "checkpoint_hash": checkpoint_hash,
        "training": {
            "epochs": TRAIN_EPOCHS,
            "scheduler_epoch_budget": SCHEDULER_EPOCH_BUDGET,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "gradient_clip": GRADIENT_CLIP,
            "sampler": SAMPLER_NAME,
        },
        "data": {
            "observation_bins": OBSERVATION_BINS,
            "window_bins": WINDOW_BINS,
            "alphas": ALPHAS,
            "target_axes": TARGET_AXES,
            "std_floor_percentile": STD_FLOOR_PERCENTILE,
        },
        "runtime": {
            "device_type": device.type,
            "torch_version": torch.__version__,
            "threads": threads,
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def progress_path(
    architecture_name: str,
    seed: int,
    held_month: str,
) -> Path:
    return (
        PROGRESS_DIR
        / architecture_name
        / f"seed_{seed}"
        / f"held_{held_month}.json"
    )


def load_completed_fold(
    architecture_name: str,
    seed: int,
    held_month: str,
    signature: str,
) -> dict | None:
    path = progress_path(architecture_name, seed, held_month)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "protocol_signature": signature,
        "architecture_name": architecture_name,
        "seed": seed,
        "held_month": held_month,
        "complete": True,
        "january_loaded": False,
        "checkpoint_saved": False,
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Incompatible Phase-4b progress file {path}: {mismatches}"
        )
    return payload


def run_or_resume_fold(
    architecture_name: str,
    architecture: dict,
    seed: int,
    held_month: str,
    training_metadata: dict,
    context: dict,
    device,
    signature: str,
    rebuild_cache: bool,
) -> dict:
    completed = load_completed_fold(
        architecture_name, seed, held_month, signature
    )
    if completed is not None:
        print(
            f"    held {held_month}: complete progress found; skipping",
            flush=True,
        )
        return completed

    print(f"    held {held_month}: training", flush=True)
    started_at = utc_now()
    started = time.time()
    net, history = train_fold(
        architecture,
        seed,
        training_metadata,
        device,
    )
    assert_baseline_untouched(context["baseline_checkpoint_sha256"])
    rebuild_held = (
        rebuild_cache
        and held_month not in context["rebuilt_held_caches"]
    )
    held_metadata = build_held_cache(
        held_month,
        training_metadata,
        context["manifest"],
        context["channels"],
        rebuild=rebuild_held,
    )
    context["rebuilt_held_caches"].add(held_month)
    fold_row, session_rows = evaluate_fold(
        net,
        held_month,
        held_metadata,
        training_metadata,
        device,
    )
    payload = {
        "phase": "4b",
        "protocol_signature": signature,
        "architecture_name": architecture_name,
        "architecture": architecture,
        "parameter_count": context["parameter_counts"][architecture_name],
        "receptive_field_bins": receptive_field(architecture),
        "seed": seed,
        "held_month": held_month,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "duration_seconds": time.time() - started,
        "training_history": history,
        "fold": fold_row,
        "sessions": session_rows,
        "complete": True,
        "january_loaded": False,
        "held_labels_available_to_optimizer": False,
        "held_metric_selected_epoch": False,
        "checkpoint_saved": False,
    }
    write_json_atomic(
        payload,
        progress_path(architecture_name, seed, held_month),
    )
    assert_baseline_untouched(context["baseline_checkpoint_sha256"])
    print(
        f"    held {held_month}: macro R2="
        f"{fold_row['session_macro_r2']:+.4f} | worst="
        f"{fold_row['worst_session_r2']:+.4f}",
        flush=True,
    )
    del net
    return payload


def run_cell(
    architecture_name: str,
    architecture: dict,
    seed: int,
    training_metadata_by_fold: dict[str, dict],
    context: dict,
    device,
    signature: str,
    rebuild_cache: bool,
) -> dict:
    print(
        f"\n=== {architecture_name} | seed {seed} | "
        f"parameters={context['parameter_counts'][architecture_name]:,} ===",
        flush=True,
    )
    fold_payloads = []
    for held_month in FOLD_ORDER:
        fold_payloads.append(
            run_or_resume_fold(
                architecture_name,
                architecture,
                seed,
                held_month,
                training_metadata_by_fold[held_month],
                context,
                device,
                signature,
                rebuild_cache,
            )
        )
    fold_rows = [
        {
            "architecture_name": architecture_name,
            "seed": seed,
            **payload["fold"],
        }
        for payload in fold_payloads
    ]
    session_rows = [
        {
            "architecture_name": architecture_name,
            "seed": seed,
            **row,
        }
        for payload in fold_payloads
        for row in payload["sessions"]
    ]
    if len(session_rows) != 33:
        raise AssertionError(
            f"Expected 33 held sessions, found {len(session_rows)}"
        )
    metrics = selection_metrics(session_rows)
    cell = {
        "architecture_name": architecture_name,
        "architecture": architecture,
        "parameter_count": context["parameter_counts"][architecture_name],
        "receptive_field_bins": receptive_field(architecture),
        "seed": seed,
        **metrics,
        "folds": fold_rows,
        "sessions": session_rows,
    }
    print(
        f"=== {architecture_name} seed {seed} complete | "
        f"score={metrics['selection_score']:+.4f} | "
        f"macro={metrics['session_macro_r2']:+.4f} | "
        f"q10={metrics['session_q10_r2']:+.4f} | "
        f"worst={metrics['worst_session_r2']:+.4f} ===",
        flush=True,
    )
    return cell


def mean_sd(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def approximate_window_multiplies(architecture: dict) -> int:
    filters = int(architecture["tcn_filters"])
    hidden = int(architecture["gru_hidden"])
    blocks = int(architecture["tcn_blocks"])
    kernel = int(architecture["kernel_size"])
    layers = int(architecture["gru_layers"])
    spatial = WINDOW_BINS * 64 * filters
    tcn = WINDOW_BINS * blocks * filters * filters * kernel
    gru = WINDOW_BINS * 3 * (filters * hidden + hidden * hidden)
    if layers > 1:
        gru += (
            WINDOW_BINS
            * (layers - 1)
            * 3
            * (hidden * hidden + hidden * hidden)
        )
    head = WINDOW_BINS * hidden * 2
    return int(spatial + tcn + gru + head)


def aggregate_results(cells: list[dict]) -> dict:
    metrics = (
        "selection_score",
        "session_macro_r2",
        "session_q10_r2",
        "worst_session_r2",
    )
    by_cell = {
        (cell["architecture_name"], cell["seed"]): cell for cell in cells
    }
    if set(by_cell) != {
        (architecture_name, seed)
        for architecture_name in ARCHITECTURES
        for seed in SEEDS
    }:
        raise AssertionError("Phase-4b result grid is incomplete")

    architecture_summary = {}
    for architecture_name, architecture in ARCHITECTURES.items():
        selected = [
            by_cell[(architecture_name, seed)] for seed in SEEDS
        ]
        month_summary = []
        for month in EXPECTED_MONTHS:
            per_seed = []
            for cell in selected:
                rows = [
                    row["r2_mean"]
                    for row in cell["sessions"]
                    if row["held_month"] == month
                ]
                per_seed.append(float(np.mean(rows)))
            month_summary.append(
                {
                    "held_month": month,
                    "macro_r2_across_seeds": mean_sd(per_seed),
                    "per_seed": {
                        str(seed): value
                        for seed, value in zip(SEEDS, per_seed)
                    },
                }
            )
        architecture_summary[architecture_name] = {
            "architecture": architecture,
            "parameter_count": selected[0]["parameter_count"],
            "receptive_field_bins": selected[0]["receptive_field_bins"],
            "approximate_multiplies_per_50_bin_window": (
                approximate_window_multiplies(architecture)
            ),
            "metrics_across_seeds": {
                metric: mean_sd(
                    [float(cell[metric]) for cell in selected]
                )
                for metric in metrics
            },
            "held_months": month_summary,
        }

    paired_seed_deltas = []
    for seed in SEEDS:
        baseline = by_cell[("baseline_64x64", seed)]
        candidate = by_cell[("candidate_48x48", seed)]
        paired_seed_deltas.append(
            {
                "seed": seed,
                **{
                    f"{metric}_delta": float(
                        candidate[metric] - baseline[metric]
                    )
                    for metric in metrics
                },
            }
        )

    month_deltas = []
    for month in EXPECTED_MONTHS:
        per_seed = []
        for seed in SEEDS:
            baseline_rows = [
                row["r2_mean"]
                for row in by_cell[("baseline_64x64", seed)]["sessions"]
                if row["held_month"] == month
            ]
            candidate_rows = [
                row["r2_mean"]
                for row in by_cell[("candidate_48x48", seed)]["sessions"]
                if row["held_month"] == month
            ]
            per_seed.append(
                float(np.mean(candidate_rows) - np.mean(baseline_rows))
            )
        month_deltas.append(
            {
                "held_month": month,
                "candidate_minus_baseline_macro_r2": mean_sd(per_seed),
                "per_seed": {
                    str(seed): value
                    for seed, value in zip(SEEDS, per_seed)
                },
            }
        )

    mean_deltas = {
        metric: float(
            np.mean([row[f"{metric}_delta"] for row in paired_seed_deltas])
        )
        for metric in metrics
    }
    worst_month_delta = min(
        row["candidate_minus_baseline_macro_r2"]["mean"]
        for row in month_deltas
    )
    checks = {
        "session_macro_r2": {
            "observed": mean_deltas["session_macro_r2"],
            "minimum": NONINFERIORITY_LIMITS[
                "session_macro_r2_mean_delta_min"
            ],
        },
        "session_q10_r2": {
            "observed": mean_deltas["session_q10_r2"],
            "minimum": NONINFERIORITY_LIMITS[
                "session_q10_r2_mean_delta_min"
            ],
        },
        "worst_session_r2": {
            "observed": mean_deltas["worst_session_r2"],
            "minimum": NONINFERIORITY_LIMITS[
                "worst_session_r2_mean_delta_min"
            ],
        },
        "worst_month_macro_r2": {
            "observed": worst_month_delta,
            "minimum": NONINFERIORITY_LIMITS[
                "worst_month_macro_r2_mean_delta_min"
            ],
        },
    }
    for check in checks.values():
        check["passed"] = bool(check["observed"] >= check["minimum"])
    passed = all(check["passed"] for check in checks.values())

    baseline_parameters = EXPECTED_PARAMETERS["baseline_64x64"]
    candidate_parameters = EXPECTED_PARAMETERS["candidate_48x48"]
    baseline_multiplies = approximate_window_multiplies(
        ARCHITECTURES["baseline_64x64"]
    )
    candidate_multiplies = approximate_window_multiplies(
        ARCHITECTURES["candidate_48x48"]
    )
    return {
        "architecture_summary": architecture_summary,
        "paired_seed_deltas": paired_seed_deltas,
        "paired_mean_deltas": mean_deltas,
        "held_month_deltas": month_deltas,
        "noninferiority": {
            "purpose": (
                "Promote the smaller candidate only when all predeclared "
                "accuracy and tail guardrails pass."
            ),
            "checks": checks,
            "all_checks_passed": passed,
        },
        "recommendation": {
            "architecture_name": (
                "candidate_48x48" if passed else "baseline_64x64"
            ),
            "reason": (
                "48/48 passed every predeclared non-inferiority guardrail "
                "and is preferred for firmware efficiency."
                if passed
                else "48/48 failed at least one predeclared guardrail; retain "
                "the protected 64/64 baseline."
            ),
            "checkpoint_promoted": False,
            "january_used": False,
        },
        "deployment_comparison": {
            "baseline_parameters": baseline_parameters,
            "candidate_parameters": candidate_parameters,
            "parameter_reduction_fraction": float(
                1.0 - candidate_parameters / baseline_parameters
            ),
            "baseline_approximate_multiplies_per_50_bin_window": (
                baseline_multiplies
            ),
            "candidate_approximate_multiplies_per_50_bin_window": (
                candidate_multiplies
            ),
            "multiply_reduction_fraction": float(
                1.0 - candidate_multiplies / baseline_multiplies
            ),
            "estimate_caveat": (
                "Bias, activation, memory-transfer and detector costs are "
                "excluded; measure the actual firmware target."
            ),
        },
    }


def write_csv_atomic(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def plot_results(cells: list[dict], aggregate: dict) -> None:
    cache = Path(tempfile.gettempdir()) / "indy_phase4b_matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "baseline_64x64": "#2368A2",
        "candidate_48x48": "#C58B18",
    }
    labels = {
        "baseline_64x64": "Baseline 64/64",
        "candidate_48x48": "Candidate 48/48",
    }
    ink = "#263238"
    grid = "#D9DEE3"
    metrics = (
        "selection_score",
        "session_macro_r2",
        "session_q10_r2",
        "worst_session_r2",
    )
    metric_labels = ("Score", "Macro R²", "Q10 R²", "Worst R²")
    by_cell = {
        (cell["architecture_name"], cell["seed"]): cell for cell in cells
    }
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=180)

    positions = np.arange(len(metrics))
    width = 0.34
    for offset, architecture_name in zip(
        (-width / 2, width / 2), ARCHITECTURES
    ):
        summary = aggregate["architecture_summary"][architecture_name]
        means = [
            summary["metrics_across_seeds"][metric]["mean"]
            for metric in metrics
        ]
        sds = [
            summary["metrics_across_seeds"][metric]["sd"]
            for metric in metrics
        ]
        axes[0, 0].bar(
            positions + offset,
            means,
            width,
            yerr=sds,
            capsize=4,
            color=colors[architecture_name],
            edgecolor=ink,
            linewidth=0.5,
            label=labels[architecture_name],
            alpha=0.9,
        )
        for metric_index, metric in enumerate(metrics):
            seed_values = [
                by_cell[(architecture_name, seed)][metric] for seed in SEEDS
            ]
            jitter = np.linspace(-0.06, 0.06, len(SEEDS))
            axes[0, 0].scatter(
                metric_index + offset + jitter,
                seed_values,
                color=ink,
                s=15,
                zorder=4,
            )
    axes[0, 0].axhline(0, color=ink, linewidth=0.8)
    axes[0, 0].set_xticks(positions, metric_labels)
    axes[0, 0].set(
        ylabel="Mean across five seeds",
        title="Architecture metrics (error bars: seed SD)",
    )
    axes[0, 0].legend(frameon=False)

    delta_fields = [
        f"{metric}_delta" for metric in metrics
    ]
    for index, row in enumerate(aggregate["paired_seed_deltas"]):
        axes[0, 1].plot(
            positions,
            [row[field] for field in delta_fields],
            marker="o",
            linewidth=1.2,
            label=f"Seed {row['seed']}",
        )
    axes[0, 1].axhline(0, color=ink, linewidth=0.9)
    axes[0, 1].set_xticks(positions, metric_labels)
    axes[0, 1].set(
        ylabel="Candidate − baseline",
        title="Paired seed differences",
    )
    axes[0, 1].legend(frameon=False, ncol=2)

    month_rows = aggregate["held_month_deltas"]
    month_means = [
        row["candidate_minus_baseline_macro_r2"]["mean"]
        for row in month_rows
    ]
    month_sds = [
        row["candidate_minus_baseline_macro_r2"]["sd"]
        for row in month_rows
    ]
    axes[1, 0].bar(
        [row["held_month"] for row in month_rows],
        month_means,
        yerr=month_sds,
        capsize=4,
        color=[
            colors["candidate_48x48"] if value >= 0 else "#F4E7BE"
            for value in month_means
        ],
        edgecolor=ink,
        linewidth=0.6,
    )
    axes[1, 0].axhline(0, color=ink, linewidth=0.9)
    axes[1, 0].axhline(
        NONINFERIORITY_LIMITS["worst_month_macro_r2_mean_delta_min"],
        color="#B43C3C",
        linestyle="--",
        linewidth=1,
        label="Month guardrail",
    )
    axes[1, 0].set(
        xlabel="Held month",
        ylabel="Candidate − baseline macro R²",
        title="Held-month differences across seeds",
    )
    axes[1, 0].legend(frameon=False)

    for architecture_name, architecture in ARCHITECTURES.items():
        summary = aggregate["architecture_summary"][architecture_name]
        axes[1, 1].scatter(
            summary["parameter_count"],
            summary["metrics_across_seeds"]["selection_score"]["mean"],
            s=150,
            color=colors[architecture_name],
            edgecolor=ink,
            linewidth=0.7,
            label=labels[architecture_name],
        )
        axes[1, 1].annotate(
            (
                f"{summary['parameter_count']:,} params\n"
                f"{summary['approximate_multiplies_per_50_bin_window'] / 1e6:.2f}M mult"
            ),
            (
                summary["parameter_count"],
                summary["metrics_across_seeds"]["selection_score"]["mean"],
            ),
            xytext=(8, 7),
            textcoords="offset points",
            fontsize=9,
            color=ink,
        )
    axes[1, 1].set(
        xlabel="Parameters",
        ylabel="Mean selection score",
        title="Accuracy–firmware-size trade-off",
    )
    axes[1, 1].legend(frameon=False)

    for axis in axes.flat:
        axis.grid(axis="y", color=grid, linewidth=0.7, alpha=0.7)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.suptitle("Phase 4b five-seed architecture confirmation", fontsize=16)
    figure.text(
        0.5,
        0.955,
        "Seeds 42–46 · five complete pre-January held-month folds · no checkpoint",
        ha="center",
        color=ink,
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.close(figure)


def write_outputs(
    cells: list[dict],
    context: dict,
    signature: str,
    started_at: str,
    duration_seconds: float,
    device,
    threads: int,
) -> dict:
    aggregate = aggregate_results(cells)
    cell_rows = [
        {
            "architecture_name": cell["architecture_name"],
            "seed": cell["seed"],
            "parameter_count": cell["parameter_count"],
            "receptive_field_bins": cell["receptive_field_bins"],
            "selection_score": cell["selection_score"],
            "session_macro_r2": cell["session_macro_r2"],
            "session_q10_r2": cell["session_q10_r2"],
            "worst_session_r2": cell["worst_session_r2"],
        }
        for cell in cells
    ]
    fold_rows = [row for cell in cells for row in cell["folds"]]
    session_rows = [row for cell in cells for row in cell["sessions"]]
    write_csv_atomic(
        CELLS_PATH,
        cell_rows,
        [
            "architecture_name",
            "seed",
            "parameter_count",
            "receptive_field_bins",
            "selection_score",
            "session_macro_r2",
            "session_q10_r2",
            "worst_session_r2",
        ],
    )
    write_csv_atomic(
        FOLDS_PATH,
        fold_rows,
        [
            "architecture_name",
            "seed",
            "held_month",
            "training_sessions",
            "held_sessions",
            "training_windows",
            "held_windows",
            "pooled_loss",
            "pooled_r2_x",
            "pooled_r2_y",
            "pooled_r2_mean",
            "session_macro_r2",
            "worst_session_r2",
        ],
    )
    write_csv_atomic(
        SESSIONS_PATH,
        session_rows,
        [
            "architecture_name",
            "seed",
            "session",
            "held_month",
            "windows",
            "loss",
            "r2_x",
            "r2_y",
            "r2_mean",
        ],
    )
    plot_results(cells, aggregate)
    payload = {
        "phase": "4b",
        "purpose": "five_seed_architecture_noninferiority_confirmation",
        "updated_at_utc": utc_now(),
        "study_started_at_utc": started_at,
        "duration_seconds_this_invocation": duration_seconds,
        "protocol_signature": signature,
        "architectures": ARCHITECTURES,
        "seeds": SEEDS,
        "fixed_protocol": {
            "fold_order": FOLD_ORDER,
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
            "selection_weights": SELECTION_WEIGHTS,
            "noninferiority_limits": NONINFERIORITY_LIMITS,
        },
        "runtime": {
            "device": str(device),
            "threads": threads,
            "python": platform.python_version(),
        },
        "data_policy": {
            "development_sessions": 33,
            "complete_held_month_folds": 5,
            "january_loaded": False,
            "locked_test_used_for_selection": False,
            "held_labels_available_to_optimizer": False,
            "held_metric_selected_epoch": False,
        },
        "baseline_protection": {
            "path": str(BASELINE_CHECKPOINT_PATH.relative_to(ROOT)),
            "expected_sha256": context["baseline_checkpoint_sha256"],
            "current_sha256": sha256_file(BASELINE_CHECKPOINT_PATH),
            "checkpoint_written_by_phase4b": False,
            "candidate_checkpoint_saved": False,
        },
        "aggregate": aggregate,
        "cells": cell_rows,
        "artifacts": {
            "cell_table": str(CELLS_PATH.relative_to(ROOT)),
            "fold_table": str(FOLDS_PATH.relative_to(ROOT)),
            "session_table": str(SESSIONS_PATH.relative_to(ROOT)),
            "figure": str(FIGURE_PATH.relative_to(ROOT)),
            "resume_cache": str(PROGRESS_DIR.relative_to(ROOT)),
            "checkpoint": None,
        },
        "next_gate": (
            "Review all non-inferiority checks before creating any candidate "
            "checkpoint. January remains unavailable."
        ),
    }
    write_json_atomic(payload, METRICS_PATH)
    return payload


def validate_only(context: dict) -> None:
    import torch

    for architecture_name, architecture in ARCHITECTURES.items():
        net = build_candidate_net(architecture, 64, DROPOUT)
        parameters = sum(value.numel() for value in net.parameters())
        sample = torch.randn(1, 64, WINDOW_BINS)
        changed = sample.clone()
        changed[:, :, 25:] += 100.0
        net.eval()
        with torch.inference_mode():
            original_output = net(sample)
            changed_output = net(changed)
        if not torch.equal(
            original_output[:, :25], changed_output[:, :25]
        ):
            raise AssertionError(
                f"{architecture_name} is not strictly causal"
            )
        if parameters != EXPECTED_PARAMETERS[architecture_name]:
            raise AssertionError(
                f"{architecture_name} parameter count changed"
            )
    assert_baseline_untouched(context["baseline_checkpoint_sha256"])
    print("=== Phase 4b validation passed ===")
    print("grid=2 architectures x 5 seeds x 5 held months = 50 fold fits")
    print("January=FORBIDDEN | checkpoints=NONE")
    print(
        "parameters: "
        + ", ".join(
            f"{name}={count:,}"
            for name, count in context["parameter_counts"].items()
        )
    )
    print("no cache, result or checkpoint was written")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="CPU is deterministic on Mac; auto selects CUDA or CPU, never MPS.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild Phase-4b fold arrays; completed fold progress still resumes.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check protocol, causality and parameter counts without training.",
    )
    return parser.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    context = validate_frozen_protocol()
    context["rebuilt_held_caches"] = set()
    if args.validate_only:
        validate_only(context)
        return

    device = choose_device(args.device)
    signature = runtime_signature(
        device,
        args.threads,
        context["baseline_checkpoint_sha256"],
    )
    started_at = utc_now()
    started = time.time()
    print("=== Phase 4b five-seed architecture confirmation ===")
    print(
        f"architectures={','.join(ARCHITECTURES)} | "
        f"seeds={','.join(str(seed) for seed in SEEDS)} | "
        f"folds={','.join(FOLD_ORDER)}"
    )
    print(
        f"device={device} | threads={args.threads} | "
        f"fits={len(ARCHITECTURES) * len(SEEDS) * len(FOLD_ORDER)}"
    )
    print(
        f"fixed: lr={LEARNING_RATE} | wd={WEIGHT_DECAY} | "
        f"dropout={DROPOUT} | sampler={SAMPLER_NAME} | "
        f"epochs={TRAIN_EPOCHS}/{SCHEDULER_EPOCH_BUDGET}-epoch schedule"
    )
    print("January: FORBIDDEN | baseline checkpoint: read-only SHA check")
    print("completed fold progress: automatically resumed")

    training_metadata_by_fold = {}
    for held_month in FOLD_ORDER:
        training_names, held_names = fold_partition(
            context["development_sessions"],
            context["by_month"],
            held_month,
        )
        training_metadata_by_fold[held_month] = build_training_cache(
            held_month,
            training_names,
            held_names,
            context["manifest"],
            context["channels"],
            rebuild=args.rebuild_cache,
        )
    assert_baseline_untouched(context["baseline_checkpoint_sha256"])

    cells = []
    for seed in SEEDS:
        for architecture_name, architecture in ARCHITECTURES.items():
            cells.append(
                run_cell(
                    architecture_name,
                    architecture,
                    seed,
                    training_metadata_by_fold,
                    context,
                    device,
                    signature,
                    args.rebuild_cache,
                )
            )
    assert_baseline_untouched(context["baseline_checkpoint_sha256"])
    payload = write_outputs(
        cells,
        context,
        signature,
        started_at,
        time.time() - started,
        device,
        args.threads,
    )
    assert_baseline_untouched(context["baseline_checkpoint_sha256"])

    aggregate = payload["aggregate"]
    print("\n=== Phase 4b five-seed summary ===")
    for architecture_name in ARCHITECTURES:
        summary = aggregate["architecture_summary"][architecture_name]
        metrics = summary["metrics_across_seeds"]
        print(
            f"{architecture_name:18s} | "
            f"score={metrics['selection_score']['mean']:+.4f} "
            f"+/- {metrics['selection_score']['sd']:.4f} | "
            f"macro={metrics['session_macro_r2']['mean']:+.4f} "
            f"+/- {metrics['session_macro_r2']['sd']:.4f} | "
            f"q10={metrics['session_q10_r2']['mean']:+.4f} | "
            f"worst={metrics['worst_session_r2']['mean']:+.4f}"
        )
    for name, check in aggregate["noninferiority"]["checks"].items():
        print(
            f"guardrail {name}: observed={check['observed']:+.4f} | "
            f"minimum={check['minimum']:+.4f} | "
            f"{'PASS' if check['passed'] else 'FAIL'}"
        )
    recommendation = aggregate["recommendation"]
    print(
        f"recommendation: {recommendation['architecture_name']} | "
        f"{recommendation['reason']}"
    )
    print(f"metrics: {METRICS_PATH}")
    print(f"figure:  {FIGURE_PATH}")
    print("checkpoint: NONE | January: NOT LOADED")


if __name__ == "__main__":
    main()
