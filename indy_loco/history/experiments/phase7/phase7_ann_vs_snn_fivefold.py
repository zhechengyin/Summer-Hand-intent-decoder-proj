#!/usr/bin/env python3
"""Phase 7: paper-aligned five-fold Indy/Loco TCN+GRU benchmark.

This experiment reuses the promoted Phase 6 *architecture and hyperparameters*,
not its learned weights.  A fresh model is trained for every session and fold.
All channel selection, feature normalization, and target normalization are fit
only on that fold's training reaches.  Validation selects the checkpoint; the
held-out test reaches are evaluated once after checkpoint selection.

Indy supplies all 96 physical channels.  Loco has 192 physical channels, so a
training-reach-only reliability ranking selects 96 channels in every fold.  A
paired channel-dropout mask always removes a raw-count stream and its matching
causal-EWMA stream together.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "indy_loco"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "indy_loco"
PHASE_NAME = "phase7_ann_vs_snn_fivefold"
RESULT_DIR = PROJECT_ROOT / "results" / PHASE_NAME
CACHE_DIR = RESULT_DIR / ".cache"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"
STATE_PATH = CACHE_DIR / "state.json"
METRICS_PATH = RESULT_DIR / f"{PHASE_NAME}_metrics.json"
FOLDS_PATH = RESULT_DIR / f"{PHASE_NAME}_folds.csv"
EPOCHS_PATH = RESULT_DIR / f"{PHASE_NAME}_epochs.csv"
SUMMARY_PATH = RESULT_DIR / f"{PHASE_NAME}_summary.csv"
FIGURE_PATH = RESULT_DIR / f"{PHASE_NAME}_comparison.png"

SAMPLE_INTERVAL_S: Final = 0.004
BIN_SAMPLES: Final = 10
BIN_SECONDS: Final = SAMPLE_INTERVAL_S * BIN_SAMPLES
WINDOW_BINS: Final = 50
MAX_REACH_SECONDS: Final = 8.0
MAX_REACH_SAMPLES: Final = round(MAX_REACH_SECONDS / SAMPLE_INTERVAL_S)
PHYSICAL_CHANNELS: Final = 96
EWMA_ALPHA: Final = 0.1
WIDTH: Final = 64
GRU_WIDTH: Final = 64
DILATIONS: Final = (1, 2, 4, 8)
KERNEL_SIZE: Final = 3
MODEL_DROPOUT: Final = 0.10
CHANNEL_DROPOUT: Final = 0.20
LEARNING_RATE: Final = 9e-4
WEIGHT_DECAY: Final = 0.025
BATCH_SIZE: Final = 32
GRADIENT_CLIP: Final = 1.0
DEFAULT_EPOCHS: Final = 20
FOLD_SEED: Final = 43
FOLD_COUNT: Final = 5

PAPER_REFERENCES: Final = {
    "ANN": 0.6186,
    "ANN3D": 0.6467,
    "SNN3D": 0.6661,
}


@dataclass(frozen=True)
class SessionSpec:
    name: str
    subject: str
    source_md5: str
    paper_reaches: int


SESSIONS: Final = (
    SessionSpec(
        "indy_20160622_01",
        "indy",
        "c33d5fff31320d709d23fe445561fb6e",
        970,
    ),
    SessionSpec(
        "indy_20160630_01",
        "indy",
        "197413a5339630ea926cbd22b8b43338",
        1023,
    ),
    SessionSpec(
        "indy_20170131_02",
        "indy",
        "2790b1c869564afaa7772dbf9e42d784",
        635,
    ),
    # The paper table labels this row 20170131_02, but the published 587-reach
    # Loco benchmark session is 20170210_03.
    SessionSpec(
        "loco_20170210_03",
        "loco",
        "4cae63b58c4cb9c8abd44929216c703b",
        587,
    ),
    SessionSpec(
        "loco_20170215_02",
        "loco",
        "739b70762d838f3a1f358733c426bb02",
        409,
    ),
    SessionSpec(
        "loco_20170301_05",
        "loco",
        "47342da09f9c950050c9213c3df38ea3",
        472,
    ),
)
SESSION_BY_NAME = {spec.name: spec for spec in SESSIONS}


@dataclass
class SessionData:
    spec: SessionSpec
    spike_presence: np.ndarray
    velocity: np.ndarray
    reach_bounds: np.ndarray
    channel_names: np.ndarray


@dataclass
class FoldArrays:
    channels: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    train_x: np.ndarray
    train_y: np.ndarray
    train_mask: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    validation_mask: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    test_mask: np.ndarray
    reach_counts: dict[str, int]
    window_counts: dict[str, int]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Auto selects CUDA when available and otherwise CPU; MPS is disabled.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--session",
        action="append",
        choices=sorted(SESSION_BY_NAME),
        help="Run one named session; repeat to select several. Default: all six.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate sources, protocol constants, and model construction only.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from completed fold records in the Phase 7 state file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Intentionally replace existing Phase 7 outputs.",
    )
    return parser.parse_args()


def md5sum(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - published dataset integrity hash
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
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


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as destination:
            np.savez_compressed(destination, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    if len(values) > 1:
        ranks /= len(values) - 1
    return ranks


def reach_bounds(target_position: np.ndarray) -> np.ndarray:
    target = np.asarray(target_position, dtype=np.float32).T
    target_diff = np.diff(target, axis=1, append=target[:, -1:].copy())
    transitions = np.nonzero(np.sum(np.abs(target_diff), axis=0))[0]
    boundaries = np.concatenate(([0], transitions, [target.shape[1]]))
    bounds = np.column_stack((boundaries[:-1], boundaries[1:])).astype(np.int64)
    if bounds.size == 0 or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Invalid reach segmentation")
    return bounds


def event_upper_edge_indices(events: np.ndarray, edges: np.ndarray) -> np.ndarray:
    events = np.asarray(events, dtype=np.float64).reshape(-1)
    if events.size == 0:
        return np.empty(0, dtype=np.int64)
    indices = np.searchsorted(edges, events, side="right") - 1
    indices[events == edges[-1]] = edges.size - 2
    valid = (events >= edges[0]) & (events <= edges[-1])
    valid &= (indices >= 0) & (indices < edges.size - 1)
    return np.unique(indices[valid] + 1)


def build_spike_presence(file: Any, edges: np.ndarray) -> np.ndarray:
    references = np.asarray(file["spikes"])
    if references.ndim != 2:
        raise ValueError(f"spikes must be 2-D, got {references.shape}")
    unit_count, channel_count = references.shape
    presence = np.zeros((channel_count, edges.size), dtype=np.uint8)
    for unit_index in range(unit_count):
        for channel_index in range(channel_count):
            reference = references[unit_index, channel_index]
            if not reference:
                continue
            cell = file[reference]
            if bool(cell.attrs.get("MATLAB_empty", 0)):
                continue
            presence[channel_index, event_upper_edge_indices(cell, edges)] = 1
    return presence


def decode_matlab_text(dataset: Any) -> str:
    return "".join(
        chr(int(value)) for value in np.asarray(dataset).reshape(-1) if value
    )


def read_channel_names(file: Any, count: int) -> np.ndarray:
    if "chan_names" not in file:
        return np.asarray([f"channel_{index + 1:03d}" for index in range(count)])
    references = np.asarray(file["chan_names"]).reshape(-1)
    if references.size != count:
        raise ValueError(f"Found {references.size} channel names, expected {count}")
    return np.asarray([decode_matlab_text(file[reference]) for reference in references])


def raw_indy_path(spec: SessionSpec) -> Path:
    return RAW_ROOT / "indy" / f"{spec.name}.mat"


def processed_loco_path(spec: SessionSpec) -> Path:
    return PROCESSED_ROOT / "loco" / f"{spec.name}.npz"


def indy_cache_path(spec: SessionSpec) -> Path:
    return CACHE_DIR / "session_inputs" / f"{spec.name}_4ms.npz"


def validate_complete_reaches(spec: SessionSpec, bounds: np.ndarray) -> np.ndarray:
    if bounds.ndim != 2 or bounds.shape[1] != 2 or len(bounds) < 3:
        raise ValueError(f"{spec.name}: invalid reach bounds {bounds.shape}")
    complete = bounds[1:-1]
    if len(complete) != spec.paper_reaches:
        raise ValueError(
            f"{spec.name}: found {len(complete)} complete reaches; "
            f"paper reports {spec.paper_reaches}"
        )
    return complete


def validate_source(spec: SessionSpec, *, checksum: bool) -> None:
    if spec.subject == "indy":
        path = raw_indy_path(spec)
        if not path.is_file():
            raise FileNotFoundError(f"Missing raw Indy session: {path}")
        if checksum and md5sum(path) != spec.source_md5:
            raise ValueError(f"{spec.name}: raw MD5 mismatch")
        import h5py

        with h5py.File(path, "r") as file:
            required = {"t", "spikes", "cursor_pos", "target_pos"}
            missing = required.difference(file.keys())
            if missing:
                raise ValueError(f"{spec.name}: missing raw fields {sorted(missing)}")
            target = np.asarray(file["target_pos"], dtype=np.float32).T
            validate_complete_reaches(spec, reach_bounds(target))
        return

    path = processed_loco_path(spec)
    if not path.is_file():
        raise FileNotFoundError(f"Missing processed Loco session: {path}")
    with np.load(path, allow_pickle=False) as data:
        if str(data["session"].item()) != spec.name:
            raise ValueError(f"{path}: session metadata mismatch")
        if str(data["source_md5"].item()) != spec.source_md5:
            raise ValueError(f"{path}: source MD5 metadata mismatch")
        if data["spike_presence"].shape[0] != 192:
            raise ValueError(f"{spec.name}: Loco must contain 192 source channels")
        validate_complete_reaches(spec, data["reach_bounds"])


def build_indy_cache(spec: SessionSpec) -> Path:
    cache_path = indy_cache_path(spec)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as data:
            if (
                str(data["session"].item()) == spec.name
                and str(data["source_md5"].item()) == spec.source_md5
                and data["spike_presence"].shape[0] == 96
            ):
                validate_complete_reaches(spec, data["reach_bounds"])
                return cache_path
        cache_path.unlink()

    import h5py

    path = raw_indy_path(spec)
    print(f"building one-time 4 ms cache: {spec.name}", flush=True)
    with h5py.File(path, "r") as file:
        timestamps = np.asarray(file["t"], dtype=np.float64).reshape(-1)
        edges = np.arange(
            timestamps[0] - SAMPLE_INTERVAL_S,
            timestamps[-1],
            SAMPLE_INTERVAL_S,
            dtype=np.float64,
        )
        presence = build_spike_presence(file, edges)[:, : timestamps.size]
        cursor = np.asarray(file["cursor_pos"], dtype=np.float32).T
        target = np.asarray(file["target_pos"], dtype=np.float32).T
        channel_names = read_channel_names(file, presence.shape[0])
    if presence.shape != (96, timestamps.size):
        raise ValueError(f"{spec.name}: unexpected spike shape {presence.shape}")
    if cursor.shape != (timestamps.size, 2):
        raise ValueError(f"{spec.name}: unexpected cursor shape {cursor.shape}")
    bounds = reach_bounds(target)
    validate_complete_reaches(spec, bounds)
    save_npz_atomic(
        cache_path,
        session=np.asarray(spec.name),
        source_md5=np.asarray(spec.source_md5),
        spike_presence=presence,
        velocity_per_sample=np.gradient(cursor, axis=0).astype(np.float32),
        reach_bounds=bounds,
        channel_names=channel_names,
    )
    return cache_path


def load_session(spec: SessionSpec) -> SessionData:
    path = (
        build_indy_cache(spec) if spec.subject == "indy" else processed_loco_path(spec)
    )
    with np.load(path, allow_pickle=False) as data:
        return SessionData(
            spec=spec,
            spike_presence=data["spike_presence"].astype(np.uint8),
            velocity=data["velocity_per_sample"].astype(np.float32),
            reach_bounds=validate_complete_reaches(spec, data["reach_bounds"]).copy(),
            channel_names=data["channel_names"].copy(),
        )


def eligible_reaches(data: SessionData) -> np.ndarray:
    durations = data.reach_bounds[:, 1] - data.reach_bounds[:, 0]
    bounds = binned_reach_bounds(data)
    binned_lengths = bounds[:, 1] - bounds[:, 0]
    eligible = np.nonzero((durations <= MAX_REACH_SAMPLES) & (binned_lengths > 0))[0]
    if eligible.size < FOLD_COUNT * 2:
        raise ValueError(
            f"{data.spec.name}: too few eligible reaches ({eligible.size})"
        )
    return eligible


def make_fold_indices(indices: np.ndarray) -> list[np.ndarray]:
    shuffled = indices.copy()
    np.random.default_rng(FOLD_SEED).shuffle(shuffled)
    return [part.astype(np.int64) for part in np.array_split(shuffled, FOLD_COUNT)]


def split_fold(parts: list[np.ndarray], fold: int) -> tuple[np.ndarray, ...]:
    held_out = parts[fold]
    validation = held_out[::2]
    test = held_out[1::2]
    train = np.concatenate([part for index, part in enumerate(parts) if index != fold])
    if not len(train) or not len(validation) or not len(test):
        raise ValueError(f"Fold {fold + 1} produced an empty split")
    return train, validation, test


def aggregate_40ms(data: SessionData) -> tuple[np.ndarray, np.ndarray]:
    usable = (data.spike_presence.shape[1] // BIN_SAMPLES) * BIN_SAMPLES
    counts = (
        data.spike_presence[:, :usable]
        .reshape(data.spike_presence.shape[0], -1, BIN_SAMPLES)
        .sum(axis=2, dtype=np.uint16)
    )
    velocity = data.velocity[:usable].reshape(-1, BIN_SAMPLES, 2).mean(axis=1)
    return counts.astype(np.float32), velocity.astype(np.float32)


def binned_reach_bounds(data: SessionData) -> np.ndarray:
    starts = np.ceil(data.reach_bounds[:, 0] / BIN_SAMPLES).astype(np.int64)
    stops = np.floor(data.reach_bounds[:, 1] / BIN_SAMPLES).astype(np.int64)
    return np.column_stack((starts, stops))


def select_channels(
    data: SessionData,
    counts: np.ndarray,
    bounds: np.ndarray,
    train_reaches: np.ndarray,
) -> np.ndarray:
    if counts.shape[0] == PHYSICAL_CHANNELS:
        return np.arange(PHYSICAL_CHANNELS, dtype=np.int64)
    if counts.shape[0] != 192:
        raise ValueError(f"Unsupported source channel count: {counts.shape[0]}")

    rates = []
    for reach in train_reaches:
        start, stop = bounds[reach]
        rates.append(counts[:, start:stop].mean(axis=1))
    reach_rates = np.stack(rates)
    activity = reach_rates.mean(axis=0)
    availability = (reach_rates > 0.01).mean(axis=0)
    coefficient_of_variation = reach_rates.std(axis=0) / (activity + 1e-6)
    score = (
        0.50 * percentile_ranks(activity)
        + 0.25 * percentile_ranks(availability)
        + 0.25 * percentile_ranks(-coefficient_of_variation)
    )
    selected = np.argsort(score, kind="stable")[-PHYSICAL_CHANNELS:]
    return np.sort(selected.astype(np.int64))


def causal_ewma(values: np.ndarray, alpha: float = EWMA_ALPHA) -> np.ndarray:
    output = values.astype(np.float32, copy=True)
    for index in range(1, output.shape[1]):
        output[:, index] = alpha * values[:, index] + (1 - alpha) * output[:, index - 1]
    return output


def reach_features(counts: np.ndarray, start: int, stop: int) -> np.ndarray:
    raw = counts[:, start:stop].astype(np.float32)
    return np.concatenate((raw, causal_ewma(raw)), axis=0)


def collect_features(
    counts: np.ndarray,
    bounds: np.ndarray,
    channels: np.ndarray,
    reaches: Iterable[int],
) -> list[np.ndarray]:
    return [
        reach_features(counts[channels], int(bounds[index, 0]), int(bounds[index, 1]))
        for index in reaches
    ]


def fit_feature_stats(features: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    timeline = np.concatenate(features, axis=1)
    mean = timeline.mean(axis=1, keepdims=True).astype(np.float32)
    std = (timeline.std(axis=1, keepdims=True) + 1e-6).astype(np.float32)
    return mean, std


def windows_from_reaches(
    features: list[np.ndarray],
    velocity: np.ndarray,
    bounds: np.ndarray,
    reaches: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_windows: list[np.ndarray] = []
    y_windows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for values, reach in zip(features, reaches, strict=True):
        normalized = ((values - mean) / std).astype(np.float32)
        start, stop = bounds[reach]
        target = velocity[start:stop]
        length = min(normalized.shape[1], target.shape[0])
        window_count = math.ceil(length / WINDOW_BINS)
        for window in range(window_count):
            left = window * WINDOW_BINS
            right = min(left + WINDOW_BINS, length)
            valid = right - left
            x_window = np.zeros((normalized.shape[0], WINDOW_BINS), dtype=np.float32)
            y_window = np.zeros((WINDOW_BINS, 2), dtype=np.float32)
            mask = np.zeros(WINDOW_BINS, dtype=bool)
            x_window[:, :valid] = normalized[:, left:right]
            y_window[:valid] = target[left:right]
            mask[:valid] = True
            x_windows.append(x_window)
            y_windows.append(y_window)
            masks.append(mask)
    if not x_windows:
        raise ValueError("No valid causal windows in split")
    return (
        np.stack(x_windows).astype(np.float32),
        np.stack(y_windows).astype(np.float32),
        np.stack(masks),
    )


def prepare_fold(data: SessionData, fold: int) -> FoldArrays:
    eligible = eligible_reaches(data)
    train_reaches, validation_reaches, test_reaches = split_fold(
        make_fold_indices(eligible), fold
    )
    counts, velocity = aggregate_40ms(data)
    bounds = binned_reach_bounds(data)
    channels = select_channels(data, counts, bounds, train_reaches)

    train_features = collect_features(counts, bounds, channels, train_reaches)
    validation_features = collect_features(counts, bounds, channels, validation_reaches)
    test_features = collect_features(counts, bounds, channels, test_reaches)
    feature_mean, feature_std = fit_feature_stats(train_features)
    train_x, train_y, train_mask = windows_from_reaches(
        train_features,
        velocity,
        bounds,
        train_reaches,
        feature_mean,
        feature_std,
    )
    validation_x, validation_y, validation_mask = windows_from_reaches(
        validation_features,
        velocity,
        bounds,
        validation_reaches,
        feature_mean,
        feature_std,
    )
    test_x, test_y, test_mask = windows_from_reaches(
        test_features,
        velocity,
        bounds,
        test_reaches,
        feature_mean,
        feature_std,
    )
    train_target_values = train_y[train_mask]
    target_mean = train_target_values.mean(axis=0).astype(np.float32)
    target_std = (train_target_values.std(axis=0) + 1e-6).astype(np.float32)
    return FoldArrays(
        channels=channels,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
        train_x=train_x,
        train_y=train_y,
        train_mask=train_mask,
        validation_x=validation_x,
        validation_y=validation_y,
        validation_mask=validation_mask,
        test_x=test_x,
        test_y=test_y,
        test_mask=test_mask,
        reach_counts={
            "train": int(len(train_reaches)),
            "validation": int(len(validation_reaches)),
            "test": int(len(test_reaches)),
            "paper_complete": data.spec.paper_reaches,
            "eligible": int(len(eligible)),
            "excluded": int(data.spec.paper_reaches - len(eligible)),
        },
        window_counts={
            "train": int(len(train_x)),
            "validation": int(len(validation_x)),
            "test": int(len(test_x)),
        },
    )


def build_model() -> Any:
    import torch
    import torch.nn as nn

    class PointwiseLayerNorm(nn.Module):
        def __init__(self, features: int) -> None:
            super().__init__()
            self.normalization = nn.LayerNorm(features)

        def forward(self, values: Any) -> Any:
            return self.normalization(values.transpose(1, 2)).transpose(1, 2)

    class PairedChannelDropout(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.probability = CHANNEL_DROPOUT

        def forward(self, values: Any) -> Any:
            if not self.training:
                return values
            keep_probability = 1.0 - self.probability
            mask = torch.empty(
                values.shape[0],
                PHYSICAL_CHANNELS,
                1,
                device=values.device,
                dtype=values.dtype,
            ).bernoulli_(keep_probability)
            mask /= keep_probability
            return values * torch.cat((mask, mask), dim=1)

    class Phase7TCNGRU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.channel_dropout = PairedChannelDropout()
            self.spatial = nn.Sequential(
                nn.Conv1d(PHYSICAL_CHANNELS * 2, WIDTH, 1),
                PointwiseLayerNorm(WIDTH),
                nn.ReLU(),
            )
            self.convolutions = nn.ModuleList(
                [
                    nn.Conv1d(
                        WIDTH,
                        WIDTH,
                        KERNEL_SIZE,
                        padding=(KERNEL_SIZE - 1) * dilation,
                        dilation=dilation,
                    )
                    for dilation in DILATIONS
                ]
            )
            self.paddings = [(KERNEL_SIZE - 1) * d for d in DILATIONS]
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(MODEL_DROPOUT)
            self.gru = nn.GRU(WIDTH, GRU_WIDTH, batch_first=True)
            self.head = nn.Linear(GRU_WIDTH, 2)

        def forward(self, values: Any) -> Any:
            encoded = self.spatial(self.channel_dropout(values))
            for convolution, padding in zip(
                self.convolutions, self.paddings, strict=True
            ):
                convolved = convolution(encoded)
                if padding:
                    convolved = convolved[:, :, :-padding]
                encoded = self.activation(convolved + encoded)
            encoded, _ = self.gru(self.dropout(encoded).transpose(1, 2))
            return self.head(encoded)

    return Phase7TCNGRU()


def parameter_count() -> int:
    return sum(parameter.numel() for parameter in build_model().parameters())


def select_device(requested: str) -> Any:
    import torch

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def r2_axes(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    residual = np.sum((target - prediction) ** 2, axis=0)
    total = np.sum((target - target.mean(axis=0)) ** 2, axis=0)
    return 1.0 - residual / np.maximum(total, 1e-12)


def predict(model: Any, values: np.ndarray, device: Any) -> np.ndarray:
    import torch

    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(values), BATCH_SIZE):
            batch = torch.from_numpy(values[start : start + BATCH_SIZE]).to(device)
            output.append(model(batch).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def evaluate(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: Any,
) -> dict[str, float | int]:
    normalized_prediction = predict(model, x, device)
    prediction = normalized_prediction * target_std + target_mean
    target_flat = y[mask]
    prediction_flat = prediction[mask]
    score = r2_axes(target_flat, prediction_flat)
    return {
        "windows": int(len(x)),
        "valid_bins": int(mask.sum()),
        "loss": float(np.mean(((target_flat - prediction_flat) / target_std) ** 2)),
        "r2_x": float(score[0]),
        "r2_y": float(score[1]),
        "r2_mean": float(score.mean()),
    }


def checkpoint_path(session: str, fold: int) -> Path:
    return CHECKPOINT_DIR / f"{session}_fold{fold + 1}.pt"


def train_fold(
    data: SessionData,
    fold: int,
    arrays: FoldArrays,
    epochs: int,
    device: Any,
) -> dict[str, Any]:
    import torch

    seed = FOLD_SEED
    seed_everything(seed)
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    normalized_y = ((arrays.train_y - arrays.target_mean) / arrays.target_std).astype(
        np.float32
    )
    rng = np.random.default_rng(seed)
    best_state = None
    best_epoch = 0
    best_validation_loss = math.inf
    history: list[dict[str, Any]] = []

    print(
        f"\n=== {data.spec.name} | fold {fold + 1}/{FOLD_COUNT} ===\n"
        f"channels={len(arrays.channels)} | reaches={arrays.reach_counts} | "
        f"windows={arrays.window_counts}",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(arrays.train_x))
        error_sum = 0.0
        value_count = 0
        gradient_sum = 0.0
        gradient_max = 0.0
        batch_count = 0
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE]
            x_batch = torch.from_numpy(arrays.train_x[indices]).to(device)
            y_batch = torch.from_numpy(normalized_y[indices]).to(device)
            mask_batch = torch.from_numpy(arrays.train_mask[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            squared_error = (model(x_batch) - y_batch) ** 2
            valid_mask = mask_batch.unsqueeze(-1).expand_as(squared_error)
            loss = squared_error[valid_mask].mean()
            loss.backward()
            gradient = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            )
            optimizer.step()
            valid_values = int(valid_mask.sum())
            error_sum += float(loss.detach()) * valid_values
            value_count += valid_values
            gradient_sum += gradient
            gradient_max = max(gradient_max, gradient)
            batch_count += 1

        train_metrics = evaluate(
            model,
            arrays.train_x,
            arrays.train_y,
            arrays.train_mask,
            arrays.target_mean,
            arrays.target_std,
            device,
        )
        validation_metrics = evaluate(
            model,
            arrays.validation_x,
            arrays.validation_y,
            arrays.validation_mask,
            arrays.target_mean,
            arrays.target_std,
            device,
        )
        improved = validation_metrics["loss"] < best_validation_loss
        if improved:
            best_validation_loss = float(validation_metrics["loss"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        row = {
            "session": data.spec.name,
            "subject": data.spec.subject,
            "fold": fold + 1,
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "optimization_loss": error_sum / max(value_count, 1),
            "gradient_mean_before_clip": gradient_sum / max(batch_count, 1),
            "gradient_max_before_clip": gradient_max,
            "train_loss": train_metrics["loss"],
            "train_r2": train_metrics["r2_mean"],
            "validation_loss": validation_metrics["loss"],
            "validation_r2": validation_metrics["r2_mean"],
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{epochs} | opt={row['optimization_loss']:.5f} | "
            f"loss train={row['train_loss']:.5f} validation={row['validation_loss']:.5f} | "
            f"R2 train={row['train_r2']:+.4f} "
            f"validation={row['validation_r2']:+.4f} | "
            f"grad={row['gradient_mean_before_clip']:.3f}/"
            f"{row['gradient_max_before_clip']:.3f}" + (" *best*" if improved else ""),
            flush=True,
        )
        scheduler.step()

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    model.load_state_dict(best_state)
    train_metrics = evaluate(
        model,
        arrays.train_x,
        arrays.train_y,
        arrays.train_mask,
        arrays.target_mean,
        arrays.target_std,
        device,
    )
    validation_metrics = evaluate(
        model,
        arrays.validation_x,
        arrays.validation_y,
        arrays.validation_mask,
        arrays.target_mean,
        arrays.target_std,
        device,
    )

    output_path = checkpoint_path(data.spec.name, fold)
    checkpoint = {
        "purpose": PHASE_NAME,
        "status": "benchmark_only_not_promoted",
        "created_at_utc": utc_now(),
        "session": data.spec.name,
        "subject": data.spec.subject,
        "fold": fold + 1,
        "seed": seed,
        "best_epoch": best_epoch,
        "model_state": best_state,
        "selected_channel_indices": arrays.channels,
        "selected_channel_names": data.channel_names[arrays.channels],
        "feature_mean": arrays.feature_mean,
        "feature_std": arrays.feature_std,
        "target_mean": arrays.target_mean,
        "target_std": arrays.target_std,
        "model_config": model_config(epochs),
        "reach_counts": arrays.reach_counts,
        "window_counts": arrays.window_counts,
        "training_device": device.type,
        "test_evaluated_during_training": False,
    }
    save_checkpoint_atomic(checkpoint, output_path)

    # The test split is opened only after validation checkpoint selection and save.
    test_metrics = evaluate(
        model,
        arrays.test_x,
        arrays.test_y,
        arrays.test_mask,
        arrays.target_mean,
        arrays.target_std,
        device,
    )
    print(
        f"selected epoch={best_epoch:02d} | train R2={train_metrics['r2_mean']:+.4f} | "
        f"validation R2={validation_metrics['r2_mean']:+.4f} | "
        f"test R2={test_metrics['r2_mean']:+.4f}",
        flush=True,
    )
    return {
        "session": data.spec.name,
        "subject": data.spec.subject,
        "fold": fold + 1,
        "seed": seed,
        "best_epoch": best_epoch,
        "selected_channels": arrays.channels.tolist(),
        "reach_counts": arrays.reach_counts,
        "window_counts": arrays.window_counts,
        "train": train_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
        "history": history,
        "checkpoint": str(output_path.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": sha256_file(output_path),
    }


def model_config(epochs: int) -> dict[str, Any]:
    return {
        "physical_channels": PHYSICAL_CHANNELS,
        "input_features": PHYSICAL_CHANNELS * 2,
        "input_bin_ms": round(BIN_SECONDS * 1000),
        "window_bins": WINDOW_BINS,
        "window_seconds": BIN_SECONDS * WINDOW_BINS,
        "causal_ewma_alpha": EWMA_ALPHA,
        "tcn_width": WIDTH,
        "gru_width": GRU_WIDTH,
        "kernel_size": KERNEL_SIZE,
        "dilations": list(DILATIONS),
        "model_dropout": MODEL_DROPOUT,
        "paired_channel_dropout": CHANNEL_DROPOUT,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "gradient_clip": GRADIENT_CLIP,
        "epochs": epochs,
        "parameter_count": parameter_count(),
    }


def protocol_signature(session_names: list[str], epochs: int, device: str) -> str:
    payload = {
        "phase": PHASE_NAME,
        "sessions": session_names,
        "model": model_config(epochs),
        "fold_count": FOLD_COUNT,
        "fold_seed": FOLD_SEED,
        "max_reach_seconds": MAX_REACH_SECONDS,
        "device": device,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for spec in SESSIONS:
        session_results = [row for row in results if row["session"] == spec.name]
        if not session_results:
            continue
        values = np.asarray([row["test"]["r2_mean"] for row in session_results])
        rows.append(
            {
                "session": spec.name,
                "subject": spec.subject,
                "folds": len(values),
                "test_r2_mean": float(values.mean()),
                "test_r2_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "test_r2_min": float(values.min()),
                "test_r2_max": float(values.max()),
            }
        )
    if results:
        values = np.asarray([row["test"]["r2_mean"] for row in results])
        rows.append(
            {
                "session": "overall_fold_macro",
                "subject": "all",
                "folds": len(values),
                "test_r2_mean": float(values.mean()),
                "test_r2_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "test_r2_min": float(values.min()),
                "test_r2_max": float(values.max()),
            }
        )
    return rows


def make_figure(summary: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping figure")
        return
    sessions = [row for row in summary if row["session"] != "overall_fold_macro"]
    if not sessions:
        return
    labels = [
        row["session"].replace("_2017", "\n2017").replace("_2016", "\n2016")
        for row in sessions
    ]
    means = [row["test_r2_mean"] for row in sessions]
    errors = [row["test_r2_std"] for row in sessions]
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.bar(np.arange(len(labels)), means, yerr=errors, capsize=4, color="#3B82F6")
    for name, score in PAPER_REFERENCES.items():
        axis.axhline(
            score, linestyle="--", linewidth=1.2, label=f"Paper {name} {score:.4f}"
        )
    axis.set_xticks(np.arange(len(labels)), labels, fontsize=8)
    axis.set_ylabel("Held-out test R²")
    axis.set_title("Phase 7 · Session-local 5-fold benchmark")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=3, fontsize=8)
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def flatten_fold_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "session": row["session"],
            "subject": row["subject"],
            "fold": row["fold"],
            "seed": row["seed"],
            "best_epoch": row["best_epoch"],
            "train_r2": row["train"]["r2_mean"],
            "validation_r2": row["validation"]["r2_mean"],
            "test_r2": row["test"]["r2_mean"],
            "test_r2_x": row["test"]["r2_x"],
            "test_r2_y": row["test"]["r2_y"],
            "train_windows": row["window_counts"]["train"],
            "validation_windows": row["window_counts"]["validation"],
            "test_windows": row["window_counts"]["test"],
            "checkpoint": row["checkpoint"],
        }
        for row in results
    ]


def validate_model() -> None:
    import torch

    model = build_model()
    model.train()
    values = torch.zeros(2, PHYSICAL_CHANNELS * 2, WINDOW_BINS)
    output = model(values)
    if output.shape != (2, WINDOW_BINS, 2):
        raise ValueError(f"Unexpected model output shape: {tuple(output.shape)}")
    if parameter_count() != 86_978:
        raise ValueError(f"Expected 86,978 parameters, found {parameter_count():,}")


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.threads <= 0:
        raise ValueError("--epochs and --threads must be positive")
    selected_names = list(
        dict.fromkeys(args.session or [spec.name for spec in SESSIONS])
    )
    selected_specs = [SESSION_BY_NAME[name] for name in selected_names]
    for spec in selected_specs:
        print(f"validating source: {spec.name}", flush=True)
        validate_source(spec, checksum=True)
    validate_model()
    if args.validate_only:
        print("=== Phase 7 validation passed ===")
        print(
            f"sessions={len(selected_specs)} | folds={FOLD_COUNT} | fits={len(selected_specs) * FOLD_COUNT}"
        )
        print(f"model parameters={parameter_count():,} | input=96 raw + 96 causal EWMA")
        print("Loco policy=train-fold-only 192-to-96 reliability selection")
        print("test policy=opened only after validation checkpoint selection")
        return

    import torch

    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    signature = protocol_signature(selected_names, args.epochs, device.type)
    if RESULT_DIR.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(
            f"Phase 7 output already exists: {RESULT_DIR}. Use --resume or --overwrite."
        )
    if args.overwrite and RESULT_DIR.exists():
        shutil.rmtree(RESULT_DIR)
    state: dict[str, Any] = {
        "signature": signature,
        "created_at_utc": utc_now(),
        "completed": {},
    }
    if args.resume:
        if not STATE_PATH.is_file():
            raise FileNotFoundError(f"Cannot resume without {STATE_PATH}")
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if state.get("signature") != signature:
            raise ValueError(
                "Resume state does not match sessions, protocol, epochs, or device"
            )
    else:
        write_json_atomic(state, STATE_PATH)

    print("=== Phase 7 ANN-vs-SNN paper-aligned benchmark ===")
    print(
        f"sessions={len(selected_specs)} | folds={FOLD_COUNT} | fits={len(selected_specs) * FOLD_COUNT}"
    )
    print(f"epochs={args.epochs} | device={device.type} | threads={args.threads}")
    print(
        f"model parameters={parameter_count():,} | paired channel dropout={CHANNEL_DROPOUT}"
    )
    print("policy=fresh weights per fold; train-only preprocessing/channel selection")
    print("test=not evaluated until best validation checkpoint is frozen")

    for spec in selected_specs:
        data = load_session(spec)
        print(
            f"\nloaded {spec.name}: source channels={data.spike_presence.shape[0]} | "
            f"paper complete reaches={spec.paper_reaches} | eligible={len(eligible_reaches(data))}",
            flush=True,
        )
        for fold in range(FOLD_COUNT):
            key = f"{spec.name}|fold{fold + 1}"
            if key in state["completed"]:
                print(f"resume: keep completed {key}")
                continue
            arrays = prepare_fold(data, fold)
            result = train_fold(data, fold, arrays, args.epochs, device)
            state["completed"][key] = result
            write_json_atomic(state, STATE_PATH)

    ordered_keys = [
        f"{spec.name}|fold{fold + 1}"
        for spec in selected_specs
        for fold in range(FOLD_COUNT)
    ]
    results = [state["completed"][key] for key in ordered_keys]
    summary = summarize(results)
    fold_rows = flatten_fold_rows(results)
    epoch_rows = [epoch for row in results for epoch in row["history"]]
    write_csv(FOLDS_PATH, fold_rows)
    write_csv(EPOCHS_PATH, epoch_rows)
    write_csv(SUMMARY_PATH, summary)
    make_figure(summary)
    metrics = {
        "phase": PHASE_NAME,
        "completed_at_utc": utc_now(),
        "protocol_signature": signature,
        "model_config": model_config(args.epochs),
        "sessions": selected_names,
        "fold_seed": FOLD_SEED,
        "fold_count": FOLD_COUNT,
        "paper_reference_r2": PAPER_REFERENCES,
        "comparison_caveat": (
            "Phase 7 shares the six sessions and reach-level five-fold structure "
            "with the paper but evaluates the project's 40 ms causal TCN+GRU."
        ),
        "results": results,
        "summary": summary,
    }
    write_json_atomic(metrics, METRICS_PATH)
    print("\n=== Phase 7 complete ===")
    for row in summary:
        print(
            f"{row['session']:<24} test R2={row['test_r2_mean']:+.4f} "
            f"± {row['test_r2_std']:.4f}"
        )
    print(f"metrics: {METRICS_PATH}")
    print(f"figure: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
