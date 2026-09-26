"""Frozen data/split implementation used by Phase 13; no history code imports.

Bodies are copied verbatim from its Phase-7 dependency. See protocol_lock.json.
Only cache location changes; raw/processed inputs are read without modification.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "indy_loco"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "indy_loco"
CACHE_DIR = Path(__file__).resolve().parent / "results" / ".cache"


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

FOLD_SEED: Final = 43

FOLD_COUNT: Final = 5


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


def md5sum(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - published dataset integrity hash
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
