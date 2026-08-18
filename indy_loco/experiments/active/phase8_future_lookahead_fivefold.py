#!/usr/bin/env python3
"""Phase 8: compare nominal 50 ms and exact 100 ms neural lookahead.

The dataset has 4 ms temporal resolution, so the conservative representation
of the permitted 50 ms condition is 48 ms (12 samples); the script never reads
the next 4 ms sample that would exceed 50 ms.  The 100 ms condition is exact
(25 samples).  Each condition trains a fresh model for five reach-level folds
in each of the three Indy sessions used by the ANN-vs-SNN benchmark.

The promoted Phase 6 architecture and hyperparameters are copied here, but no
old experiment module or learned checkpoint is imported.  Every fold fits
weights, feature statistics, and target statistics from training reaches only.
Validation selects a checkpoint; test is evaluated only after selection.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "indy_loco" / "indy"
PHASE_NAME = "phase8_future_lookahead_fivefold"
RESULT_DIR = PROJECT_ROOT / "results" / PHASE_NAME
CACHE_DIR = RESULT_DIR / ".cache"
SESSION_CACHE_DIR = CACHE_DIR / "session_inputs"
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
MAX_REACH_SAMPLES: Final = round(8.0 / SAMPLE_INTERVAL_S)
PHYSICAL_CHANNELS: Final = 96
EWMA_ALPHA: Final = 0.1
WIDTH: Final = 64
GRU_WIDTH: Final = 64
DILATIONS: Final = (1, 2, 4, 8)
KERNEL_SIZE: Final = 3
MODEL_DROPOUT: Final = 0.10
PAIRED_CHANNEL_DROPOUT: Final = 0.20
LEARNING_RATE: Final = 9e-4
WEIGHT_DECAY: Final = 0.025
BATCH_SIZE: Final = 32
GRADIENT_CLIP: Final = 1.0
DEFAULT_EPOCHS: Final = 20
FOLD_SEED: Final = 43
FOLD_COUNT: Final = 5


@dataclass(frozen=True)
class SessionSpec:
    name: str
    source_md5: str
    paper_reaches: int


@dataclass(frozen=True)
class LookaheadConfig:
    key: str
    nominal_ms: int
    samples: int

    @property
    def realized_ms(self) -> int:
        return round(self.samples * SAMPLE_INTERVAL_S * 1000)


SESSIONS: Final = (
    SessionSpec(
        "indy_20160622_01",
        "c33d5fff31320d709d23fe445561fb6e",
        970,
    ),
    SessionSpec(
        "indy_20160630_01",
        "197413a5339630ea926cbd22b8b43338",
        1023,
    ),
    SessionSpec(
        "indy_20170131_02",
        "2790b1c869564afaa7772dbf9e42d784",
        635,
    ),
)
SESSION_BY_NAME = {spec.name: spec for spec in SESSIONS}

LOOKAHEADS: Final = (
    LookaheadConfig("future50ms", nominal_ms=50, samples=12),
    LookaheadConfig("future100ms", nominal_ms=100, samples=25),
)
LOOKAHEAD_BY_KEY = {config.key: config for config in LOOKAHEADS}


@dataclass
class SessionData:
    spec: SessionSpec
    spike_presence: np.ndarray
    velocity_per_sample: np.ndarray
    reach_bounds: np.ndarray
    channel_names: np.ndarray


@dataclass
class PackedSplit:
    x: np.ndarray
    y: np.ndarray
    mask: np.ndarray


@dataclass
class FoldArrays:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    train: PackedSplit
    validation: PackedSplit
    test: PackedSplit
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
        help="Auto selects CUDA when available, otherwise CPU. MPS is disabled.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--session",
        action="append",
        choices=sorted(SESSION_BY_NAME),
        help="Run one session; repeat to select several. Default: all three.",
    )
    parser.add_argument(
        "--lookahead",
        action="append",
        choices=sorted(LOOKAHEAD_BY_KEY),
        help="Run one lookahead; repeat to select both. Default: both.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate sources, reach counts, alignment, and model graph only.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def md5sum(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - published dataset integrity checksum
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


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as destination:
            np.savez_compressed(destination, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint_atomic(payload: dict[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def raw_path(spec: SessionSpec) -> Path:
    return RAW_DIR / f"{spec.name}.mat"


def session_cache_path(spec: SessionSpec) -> Path:
    return SESSION_CACHE_DIR / f"{spec.name}_4ms.npz"


def reach_bounds(target_position: np.ndarray) -> np.ndarray:
    target = np.asarray(target_position, dtype=np.float32).T
    differences = np.diff(target, axis=1, append=target[:, -1:].copy())
    transitions = np.nonzero(np.sum(np.abs(differences), axis=0))[0]
    boundaries = np.concatenate(([0], transitions, [target.shape[1]]))
    bounds = np.column_stack((boundaries[:-1], boundaries[1:])).astype(np.int64)
    if bounds.size == 0 or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Invalid reach segmentation")
    return bounds


def complete_reaches(spec: SessionSpec, bounds: np.ndarray) -> np.ndarray:
    if bounds.ndim != 2 or bounds.shape[1] != 2 or len(bounds) < 3:
        raise ValueError(f"{spec.name}: invalid reach bounds {bounds.shape}")
    complete = bounds[1:-1]
    if len(complete) != spec.paper_reaches:
        raise ValueError(
            f"{spec.name}: found {len(complete)} complete reaches; "
            f"expected {spec.paper_reaches}"
        )
    return complete


def event_upper_edge_indices(events: np.ndarray, edges: np.ndarray) -> np.ndarray:
    values = np.asarray(events, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.empty(0, dtype=np.int64)
    indices = np.searchsorted(edges, values, side="right") - 1
    indices[values == edges[-1]] = edges.size - 2
    valid = (values >= edges[0]) & (values <= edges[-1])
    valid &= (indices >= 0) & (indices < edges.size - 1)
    return np.unique(indices[valid] + 1)


def build_spike_presence(file: Any, edges: np.ndarray) -> np.ndarray:
    references = np.asarray(file["spikes"])
    if references.ndim != 2 or references.shape[1] != PHYSICAL_CHANNELS:
        raise ValueError(f"Unexpected spike cell shape: {references.shape}")
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
            indices = event_upper_edge_indices(np.asarray(cell), edges)
            presence[channel_index, indices] = 1
    return presence


def decode_matlab_text(dataset: Any) -> str:
    return "".join(
        chr(int(value)) for value in np.asarray(dataset).reshape(-1) if value
    )


def read_channel_names(file: Any) -> np.ndarray:
    if "chan_names" not in file:
        return np.asarray(
            [f"channel_{index + 1:03d}" for index in range(PHYSICAL_CHANNELS)]
        )
    references = np.asarray(file["chan_names"]).reshape(-1)
    if references.size != PHYSICAL_CHANNELS:
        raise ValueError(f"Expected 96 channel names, found {references.size}")
    return np.asarray([decode_matlab_text(file[reference]) for reference in references])


def validate_source(spec: SessionSpec, *, checksum: bool) -> None:
    import h5py

    path = raw_path(spec)
    if not path.is_file():
        raise FileNotFoundError(f"Missing raw session: {path}")
    if checksum and md5sum(path) != spec.source_md5:
        raise ValueError(f"{spec.name}: source MD5 mismatch")
    with h5py.File(path, "r") as file:
        required = {"t", "spikes", "cursor_pos", "target_pos"}
        missing = required.difference(file.keys())
        if missing:
            raise ValueError(f"{spec.name}: missing fields {sorted(missing)}")
        target = np.asarray(file["target_pos"], dtype=np.float32).T
        complete_reaches(spec, reach_bounds(target))


def build_session_cache(spec: SessionSpec) -> Path:
    path = session_cache_path(spec)
    if path.is_file():
        with np.load(path, allow_pickle=False) as cached:
            valid = (
                str(cached["session"].item()) == spec.name
                and str(cached["source_md5"].item()) == spec.source_md5
                and cached["spike_presence"].shape[0] == PHYSICAL_CHANNELS
            )
            if valid:
                complete_reaches(spec, cached["reach_bounds"])
                return path
        path.unlink()

    import h5py

    print(f"building one-time native 4 ms cache: {spec.name}", flush=True)
    with h5py.File(raw_path(spec), "r") as file:
        timestamps = np.asarray(file["t"], dtype=np.float64).reshape(-1)
        edges = np.arange(
            timestamps[0] - SAMPLE_INTERVAL_S,
            timestamps[-1],
            SAMPLE_INTERVAL_S,
            dtype=np.float64,
        )
        spikes = build_spike_presence(file, edges)
        if spikes.shape[1] < timestamps.size:
            raise ValueError(f"{spec.name}: spike input is shorter than kinematics")
        spikes = spikes[:, : timestamps.size]
        cursor = np.asarray(file["cursor_pos"], dtype=np.float32).T
        target = np.asarray(file["target_pos"], dtype=np.float32).T
        channel_names = read_channel_names(file)
    if cursor.shape != (timestamps.size, 2):
        raise ValueError(f"{spec.name}: unexpected cursor shape {cursor.shape}")
    bounds = reach_bounds(target)
    complete_reaches(spec, bounds)
    save_npz_atomic(
        path,
        session=np.asarray(spec.name),
        source_md5=np.asarray(spec.source_md5),
        spike_presence=spikes,
        velocity_per_sample=np.gradient(cursor, axis=0).astype(np.float32),
        reach_bounds=bounds,
        channel_names=channel_names,
    )
    return path


def load_session(spec: SessionSpec) -> SessionData:
    with np.load(build_session_cache(spec), allow_pickle=False) as data:
        return SessionData(
            spec=spec,
            spike_presence=data["spike_presence"].astype(np.uint8),
            velocity_per_sample=data["velocity_per_sample"].astype(np.float32),
            reach_bounds=complete_reaches(spec, data["reach_bounds"]).copy(),
            channel_names=data["channel_names"].copy(),
        )


def aligned_bin_starts(
    reach_start: int,
    reach_stop: int,
    lookahead_samples: int,
) -> np.ndarray:
    first = math.ceil((reach_start + lookahead_samples) / BIN_SAMPLES) * BIN_SAMPLES
    final = reach_stop - BIN_SAMPLES
    if first > final:
        return np.empty(0, dtype=np.int64)
    starts = np.arange(first, final + 1, BIN_SAMPLES, dtype=np.int64)
    target_starts = starts - lookahead_samples
    valid = target_starts >= reach_start
    valid &= target_starts + BIN_SAMPLES <= reach_stop
    return starts[valid]


def eligible_reaches(data: SessionData) -> np.ndarray:
    maximum_lookahead = max(config.samples for config in LOOKAHEADS)
    eligible = []
    for index, (start, stop) in enumerate(data.reach_bounds):
        if stop - start > MAX_REACH_SAMPLES:
            continue
        if aligned_bin_starts(int(start), int(stop), maximum_lookahead).size:
            eligible.append(index)
    if len(eligible) < FOLD_COUNT * 2:
        raise ValueError(f"{data.spec.name}: only {len(eligible)} eligible reaches")
    return np.asarray(eligible, dtype=np.int64)


def fold_parts(indices: np.ndarray) -> list[np.ndarray]:
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


def causal_ewma(values: np.ndarray) -> np.ndarray:
    output = values.astype(np.float32, copy=True)
    for index in range(1, output.shape[1]):
        output[:, index] = (
            EWMA_ALPHA * values[:, index] + (1 - EWMA_ALPHA) * output[:, index - 1]
        )
    return output


def aligned_reach(
    data: SessionData,
    reach_index: int,
    lookahead: LookaheadConfig,
) -> tuple[np.ndarray, np.ndarray]:
    reach_start, reach_stop = data.reach_bounds[reach_index]
    neural_starts = aligned_bin_starts(
        int(reach_start), int(reach_stop), lookahead.samples
    )
    if not neural_starts.size:
        raise ValueError(f"Reach {reach_index} has no aligned bins")
    counts = np.stack(
        [
            data.spike_presence[:, start : start + BIN_SAMPLES].sum(
                axis=1, dtype=np.uint16
            )
            for start in neural_starts
        ],
        axis=1,
    ).astype(np.float32)
    target_starts = neural_starts - lookahead.samples
    targets = np.stack(
        [
            data.velocity_per_sample[start : start + BIN_SAMPLES].mean(axis=0)
            for start in target_starts
        ]
    ).astype(np.float32)
    features = np.concatenate((counts, causal_ewma(counts)), axis=0)
    return features, targets


def collect_reaches(
    data: SessionData,
    indices: np.ndarray,
    lookahead: LookaheadConfig,
) -> list[tuple[np.ndarray, np.ndarray]]:
    return [aligned_reach(data, int(index), lookahead) for index in indices]


def fit_feature_stats(
    reaches: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    timeline = np.concatenate([features for features, _ in reaches], axis=1)
    mean = timeline.mean(axis=1, keepdims=True).astype(np.float32)
    std = (timeline.std(axis=1, keepdims=True) + 1e-6).astype(np.float32)
    return mean, std


def pack_reaches(
    reaches: list[tuple[np.ndarray, np.ndarray]],
    mean: np.ndarray,
    std: np.ndarray,
) -> PackedSplit:
    x_windows: list[np.ndarray] = []
    y_windows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for features, target in reaches:
        normalized = ((features - mean) / std).astype(np.float32)
        window_count = math.ceil(normalized.shape[1] / WINDOW_BINS)
        for window in range(window_count):
            left = window * WINDOW_BINS
            right = min(left + WINDOW_BINS, normalized.shape[1])
            valid = right - left
            x_values = np.zeros((PHYSICAL_CHANNELS * 2, WINDOW_BINS), np.float32)
            y_values = np.zeros((WINDOW_BINS, 2), np.float32)
            mask = np.zeros(WINDOW_BINS, dtype=bool)
            x_values[:, :valid] = normalized[:, left:right]
            y_values[:valid] = target[left:right]
            mask[:valid] = True
            x_windows.append(x_values)
            y_windows.append(y_values)
            masks.append(mask)
    if not x_windows:
        raise ValueError("Split has no valid windows")
    return PackedSplit(np.stack(x_windows), np.stack(y_windows), np.stack(masks))


def prepare_fold(
    data: SessionData,
    lookahead: LookaheadConfig,
    fold: int,
) -> FoldArrays:
    eligible = eligible_reaches(data)
    train_indices, validation_indices, test_indices = split_fold(
        fold_parts(eligible), fold
    )
    train_reaches = collect_reaches(data, train_indices, lookahead)
    validation_reaches = collect_reaches(data, validation_indices, lookahead)
    test_reaches = collect_reaches(data, test_indices, lookahead)
    feature_mean, feature_std = fit_feature_stats(train_reaches)
    train = pack_reaches(train_reaches, feature_mean, feature_std)
    validation = pack_reaches(validation_reaches, feature_mean, feature_std)
    test = pack_reaches(test_reaches, feature_mean, feature_std)
    target_values = train.y[train.mask]
    target_mean = target_values.mean(axis=0).astype(np.float32)
    target_std = (target_values.std(axis=0) + 1e-6).astype(np.float32)
    return FoldArrays(
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
        train=train,
        validation=validation,
        test=test,
        reach_counts={
            "paper_complete": data.spec.paper_reaches,
            "eligible_common": int(len(eligible)),
            "excluded": int(data.spec.paper_reaches - len(eligible)),
            "train": int(len(train_indices)),
            "validation": int(len(validation_indices)),
            "test": int(len(test_indices)),
        },
        window_counts={
            "train": int(len(train.x)),
            "validation": int(len(validation.x)),
            "test": int(len(test.x)),
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
        def forward(self, values: Any) -> Any:
            if not self.training:
                return values
            keep = 1.0 - PAIRED_CHANNEL_DROPOUT
            mask = torch.empty(
                values.shape[0],
                PHYSICAL_CHANNELS,
                1,
                device=values.device,
                dtype=values.dtype,
            ).bernoulli_(keep)
            mask /= keep
            return values * torch.cat((mask, mask), dim=1)

    class Phase8TCNGRU(nn.Module):
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

    return Phase8TCNGRU()


def parameter_count() -> int:
    return sum(parameter.numel() for parameter in build_model().parameters())


def validate_model() -> None:
    import torch

    model = build_model()
    output = model(torch.zeros(2, PHYSICAL_CHANNELS * 2, WINDOW_BINS))
    if output.shape != (2, WINDOW_BINS, 2):
        raise ValueError(f"Unexpected model output: {tuple(output.shape)}")
    if parameter_count() != 86_978:
        raise ValueError(f"Expected 86,978 parameters, found {parameter_count():,}")


def select_device(requested: str) -> Any:
    import torch

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
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


def predict(model: Any, x: np.ndarray, device: Any) -> np.ndarray:
    import torch

    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), BATCH_SIZE):
            batch = torch.from_numpy(x[start : start + BATCH_SIZE]).to(device)
            predictions.append(model(batch).cpu().numpy())
    return np.concatenate(predictions).astype(np.float32)


def evaluate(
    model: Any,
    split: PackedSplit,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: Any,
) -> dict[str, float | int]:
    normalized_prediction = predict(model, split.x, device)
    prediction = normalized_prediction * target_std + target_mean
    target_values = split.y[split.mask]
    prediction_values = prediction[split.mask]
    scores = r2_axes(target_values, prediction_values)
    return {
        "windows": int(len(split.x)),
        "valid_bins": int(split.mask.sum()),
        "loss": float(np.mean(((target_values - prediction_values) / target_std) ** 2)),
        "r2_x": float(scores[0]),
        "r2_y": float(scores[1]),
        "r2_mean": float(scores.mean()),
    }


def model_config(epochs: int) -> dict[str, Any]:
    return {
        "physical_channels": PHYSICAL_CHANNELS,
        "input_features": PHYSICAL_CHANNELS * 2,
        "bin_ms": round(BIN_SECONDS * 1000),
        "window_bins": WINDOW_BINS,
        "window_seconds": BIN_SECONDS * WINDOW_BINS,
        "ewma_alpha": EWMA_ALPHA,
        "tcn_width": WIDTH,
        "gru_width": GRU_WIDTH,
        "kernel_size": KERNEL_SIZE,
        "dilations": list(DILATIONS),
        "model_dropout": MODEL_DROPOUT,
        "paired_channel_dropout": PAIRED_CHANNEL_DROPOUT,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "gradient_clip": GRADIENT_CLIP,
        "epochs": epochs,
        "parameter_count": parameter_count(),
    }


def checkpoint_path(config: LookaheadConfig, session: str, fold: int) -> Path:
    return CHECKPOINT_DIR / config.key / f"{session}_fold{fold + 1}.pt"


def train_fold(
    data: SessionData,
    config: LookaheadConfig,
    fold: int,
    arrays: FoldArrays,
    epochs: int,
    device: Any,
) -> dict[str, Any]:
    import torch

    seed_everything(FOLD_SEED)
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    normalized_y = ((arrays.train.y - arrays.target_mean) / arrays.target_std).astype(
        np.float32
    )
    rng = np.random.default_rng(FOLD_SEED)
    best_state = None
    best_epoch = 0
    best_validation_loss = math.inf
    history: list[dict[str, Any]] = []

    print(
        f"\n=== {config.key} ({config.realized_ms} ms) | "
        f"{data.spec.name} | fold {fold + 1}/{FOLD_COUNT} ===\n"
        f"reaches={arrays.reach_counts} | windows={arrays.window_counts}",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(arrays.train.x))
        error_sum = 0.0
        valid_value_count = 0
        gradient_sum = 0.0
        gradient_max = 0.0
        batch_count = 0
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE]
            x_batch = torch.from_numpy(arrays.train.x[indices]).to(device)
            y_batch = torch.from_numpy(normalized_y[indices]).to(device)
            mask_batch = torch.from_numpy(arrays.train.mask[indices]).to(device)
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
            valid_value_count += valid_values
            gradient_sum += gradient
            gradient_max = max(gradient_max, gradient)
            batch_count += 1

        train_metrics = evaluate(
            model,
            arrays.train,
            arrays.target_mean,
            arrays.target_std,
            device,
        )
        validation_metrics = evaluate(
            model,
            arrays.validation,
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
            "lookahead": config.key,
            "nominal_ms": config.nominal_ms,
            "realized_ms": config.realized_ms,
            "session": data.spec.name,
            "fold": fold + 1,
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "optimization_loss": error_sum / max(valid_value_count, 1),
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
            f"loss train={row['train_loss']:.5f} "
            f"validation={row['validation_loss']:.5f} | "
            f"R2 train={row['train_r2']:+.4f} "
            f"validation={row['validation_r2']:+.4f} | "
            f"grad={row['gradient_mean_before_clip']:.3f}/"
            f"{row['gradient_max_before_clip']:.3f}" + (" *best*" if improved else ""),
            flush=True,
        )
        scheduler.step()

    if best_state is None:
        raise RuntimeError("No validation checkpoint selected")
    model.load_state_dict(best_state)
    train_metrics = evaluate(
        model, arrays.train, arrays.target_mean, arrays.target_std, device
    )
    validation_metrics = evaluate(
        model, arrays.validation, arrays.target_mean, arrays.target_std, device
    )
    output_path = checkpoint_path(config, data.spec.name, fold)
    checkpoint = {
        "purpose": PHASE_NAME,
        "status": "experiment_only_not_promoted",
        "created_at_utc": utc_now(),
        "lookahead_key": config.key,
        "nominal_lookahead_ms": config.nominal_ms,
        "realized_lookahead_ms": config.realized_ms,
        "lookahead_samples": config.samples,
        "session": data.spec.name,
        "fold": fold + 1,
        "seed": FOLD_SEED,
        "best_epoch": best_epoch,
        "model_state": best_state,
        "feature_mean": arrays.feature_mean,
        "feature_std": arrays.feature_std,
        "target_mean": arrays.target_mean,
        "target_std": arrays.target_std,
        "channel_names": data.channel_names,
        "model_config": model_config(epochs),
        "reach_counts": arrays.reach_counts,
        "window_counts": arrays.window_counts,
        "training_device": device.type,
        "test_evaluated_during_training": False,
    }
    save_checkpoint_atomic(checkpoint, output_path)

    test_metrics = evaluate(
        model, arrays.test, arrays.target_mean, arrays.target_std, device
    )
    print(
        f"selected epoch={best_epoch:02d} | train R2={train_metrics['r2_mean']:+.4f} | "
        f"validation R2={validation_metrics['r2_mean']:+.4f} | "
        f"test R2={test_metrics['r2_mean']:+.4f}",
        flush=True,
    )
    return {
        "lookahead": config.key,
        "nominal_ms": config.nominal_ms,
        "realized_ms": config.realized_ms,
        "lookahead_samples": config.samples,
        "session": data.spec.name,
        "fold": fold + 1,
        "seed": FOLD_SEED,
        "best_epoch": best_epoch,
        "reach_counts": arrays.reach_counts,
        "window_counts": arrays.window_counts,
        "train": train_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
        "history": history,
        "checkpoint": str(output_path.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": sha256_file(output_path),
    }


def protocol_signature(
    sessions: list[str],
    lookaheads: list[str],
    epochs: int,
    device: str,
) -> str:
    payload = {
        "phase": PHASE_NAME,
        "sessions": sessions,
        "lookaheads": lookaheads,
        "epochs": epochs,
        "device": device,
        "fold_seed": FOLD_SEED,
        "fold_count": FOLD_COUNT,
        "model": model_config(epochs),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flatten_fold_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "lookahead": row["lookahead"],
            "nominal_ms": row["nominal_ms"],
            "realized_ms": row["realized_ms"],
            "session": row["session"],
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


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config in LOOKAHEADS:
        config_results = [row for row in results if row["lookahead"] == config.key]
        if not config_results:
            continue
        for spec in SESSIONS:
            session_results = [
                row for row in config_results if row["session"] == spec.name
            ]
            if not session_results:
                continue
            values = np.asarray([row["test"]["r2_mean"] for row in session_results])
            rows.append(
                {
                    "lookahead": config.key,
                    "nominal_ms": config.nominal_ms,
                    "realized_ms": config.realized_ms,
                    "session": spec.name,
                    "folds": len(values),
                    "test_r2_mean": float(values.mean()),
                    "test_r2_std": float(values.std(ddof=1)),
                    "test_r2_min": float(values.min()),
                    "test_r2_max": float(values.max()),
                }
            )
        values = np.asarray([row["test"]["r2_mean"] for row in config_results])
        rows.append(
            {
                "lookahead": config.key,
                "nominal_ms": config.nominal_ms,
                "realized_ms": config.realized_ms,
                "session": "overall_fold_macro",
                "folds": len(values),
                "test_r2_mean": float(values.mean()),
                "test_r2_std": float(values.std(ddof=1)),
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
    available_sessions = {row["session"] for row in summary}
    sessions = [spec.name for spec in SESSIONS if spec.name in available_sessions] + [
        "overall_fold_macro"
    ]
    x = np.arange(len(sessions))
    width = 0.36
    figure, axis = plt.subplots(figsize=(11, 5.5))
    available_configs = [
        config
        for config in LOOKAHEADS
        if any(row["lookahead"] == config.key for row in summary)
    ]
    for offset, config in enumerate(available_configs):
        rows = {
            row["session"]: row for row in summary if row["lookahead"] == config.key
        }
        if not rows:
            continue
        means = [rows[session]["test_r2_mean"] for session in sessions]
        errors = [rows[session]["test_r2_std"] for session in sessions]
        center = (len(available_configs) - 1) / 2
        positions = x + (offset - center) * width
        axis.bar(
            positions,
            means,
            width,
            yerr=errors,
            capsize=3,
            label=f"nominal {config.nominal_ms} ms (actual {config.realized_ms} ms)",
        )
    labels = [session.replace("indy_", "").replace("_", "\n") for session in sessions]
    labels[-1] = "overall"
    axis.set_xticks(x, labels)
    axis.set_ylabel("Held-out test R²")
    axis.set_title("Phase 8 · Permitted neural lookahead comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def validate_alignment() -> None:
    start, stop = 103, 1003
    for config in LOOKAHEADS:
        starts = aligned_bin_starts(start, stop, config.samples)
        if not starts.size:
            raise ValueError(f"No alignment for {config.key}")
        target_starts = starts - config.samples
        observed = starts - target_starts
        if not np.all(observed == config.samples):
            raise ValueError(f"Incorrect alignment for {config.key}")
        if np.any(target_starts < start) or np.any(starts + BIN_SAMPLES > stop):
            raise ValueError(f"Alignment crossed reach boundary for {config.key}")
    if LOOKAHEAD_BY_KEY["future50ms"].realized_ms > 50:
        raise ValueError(
            "Conservative 50 ms condition exceeds user-permitted lookahead"
        )
    if LOOKAHEAD_BY_KEY["future100ms"].realized_ms != 100:
        raise ValueError("100 ms condition must be represented exactly")


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.threads <= 0:
        raise ValueError("--epochs and --threads must be positive")
    session_names = list(
        dict.fromkeys(args.session or [spec.name for spec in SESSIONS])
    )
    lookahead_keys = list(
        dict.fromkeys(args.lookahead or [config.key for config in LOOKAHEADS])
    )
    selected_sessions = [SESSION_BY_NAME[name] for name in session_names]
    selected_lookaheads = [LOOKAHEAD_BY_KEY[key] for key in lookahead_keys]

    for spec in selected_sessions:
        print(f"validating source: {spec.name}", flush=True)
        validate_source(spec, checksum=True)
    validate_alignment()
    validate_model()
    if args.validate_only:
        print("=== Phase 8 validation passed ===")
        print(
            f"sessions={len(selected_sessions)} | lookaheads={len(selected_lookaheads)} | "
            f"folds={FOLD_COUNT} | fits="
            f"{len(selected_sessions) * len(selected_lookaheads) * FOLD_COUNT}"
        )
        for config in selected_lookaheads:
            print(
                f"{config.key}: nominal={config.nominal_ms} ms | "
                f"realized={config.realized_ms} ms | samples={config.samples}"
            )
        print(f"model parameters={parameter_count():,}")
        print("test policy=opened only after validation checkpoint selection")
        return

    import torch

    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    signature = protocol_signature(
        session_names, lookahead_keys, args.epochs, device.type
    )
    if RESULT_DIR.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(
            f"Phase 8 outputs exist: {RESULT_DIR}. Use --resume or --overwrite."
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
            raise ValueError("Resume state does not match the requested protocol")
    else:
        write_json_atomic(state, STATE_PATH)

    print("=== Phase 8 future-lookahead five-fold benchmark ===")
    print(
        f"sessions={len(selected_sessions)} | lookaheads={len(selected_lookaheads)} | "
        f"folds={FOLD_COUNT} | fits="
        f"{len(selected_sessions) * len(selected_lookaheads) * FOLD_COUNT}"
    )
    print(f"epochs={args.epochs} | device={device.type} | threads={args.threads}")
    print(f"model parameters={parameter_count():,}")
    print("weights/statistics=train only; validation=checkpoint selection")
    print("test=not evaluated until checkpoint is frozen")

    loaded = {spec.name: load_session(spec) for spec in selected_sessions}
    for config in selected_lookaheads:
        print(
            f"\n--- {config.key}: nominal {config.nominal_ms} ms, "
            f"realized {config.realized_ms} ms ---"
        )
        for spec in selected_sessions:
            data = loaded[spec.name]
            print(
                f"loaded {spec.name}: complete reaches={spec.paper_reaches} | "
                f"eligible common={len(eligible_reaches(data))}",
                flush=True,
            )
            for fold in range(FOLD_COUNT):
                key = f"{config.key}|{spec.name}|fold{fold + 1}"
                if key in state["completed"]:
                    print(f"resume: keep completed {key}")
                    continue
                arrays = prepare_fold(data, config, fold)
                result = train_fold(data, config, fold, arrays, args.epochs, device)
                state["completed"][key] = result
                write_json_atomic(state, STATE_PATH)

    ordered_keys = [
        f"{config.key}|{spec.name}|fold{fold + 1}"
        for config in selected_lookaheads
        for spec in selected_sessions
        for fold in range(FOLD_COUNT)
    ]
    results = [state["completed"][key] for key in ordered_keys]
    summary = summarize(results)
    write_csv(FOLDS_PATH, flatten_fold_rows(results))
    write_csv(EPOCHS_PATH, [epoch for row in results for epoch in row["history"]])
    write_csv(SUMMARY_PATH, summary)
    make_figure(summary)
    metrics = {
        "phase": PHASE_NAME,
        "completed_at_utc": utc_now(),
        "protocol_signature": signature,
        "sessions": session_names,
        "lookaheads": [
            {
                "key": config.key,
                "nominal_ms": config.nominal_ms,
                "realized_ms": config.realized_ms,
                "samples": config.samples,
            }
            for config in selected_lookaheads
        ],
        "model_config": model_config(args.epochs),
        "results": results,
        "summary": summary,
    }
    write_json_atomic(metrics, METRICS_PATH)
    print("\n=== Phase 8 complete ===")
    for row in summary:
        print(
            f"{row['lookahead']:<12} {row['session']:<24} "
            f"test R2={row['test_r2_mean']:+.4f} ± {row['test_r2_std']:.4f}"
        )
    print(f"metrics: {METRICS_PATH}")
    print(f"figure: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
