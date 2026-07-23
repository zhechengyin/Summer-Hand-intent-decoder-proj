"""Canonical Indy raw and model-ready data loader.

Raw MAT files live in ``data/raw/indy_loco/indy``. Sessions use their canonical
Zenodo names (``indy_YYYYMMDD_NN``), and the fixed chronological split lives in
``configs/datasets/indy_sessions.yaml``.

This loader returns unsmoothed spike counts.  Causal temporal features are owned
by :mod:`src.intent_decoder.features.causal`; non-causal Gaussian smoothing is
deliberately not part of the stable data API.
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import yaml

from src.intent_decoder.features.causal import causal_sample_hold, causal_velocity
from src.intent_decoder.paths import DATASET_CONFIG_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

RAW_DIR = RAW_DATA_DIR / "indy_loco" / "indy"
PROCESSED_DIR = PROCESSED_DATA_DIR / "indy_loco" / "indy"
MANIFEST = DATASET_CONFIG_DIR / "indy_sessions.yaml"
URL = "https://zenodo.org/records/3854034/files/{}?download=1"
DEFAULT_BIN_S = 0.040
DEFAULT_VELOCITY_LOWPASS_HZ = 3.0
DEFAULT_N_CHANNELS = 96
MODEL_READY_SCHEMA = "indy_counts_velocity_v2"
_ORIGINAL_NAME = re.compile(r"^indy_\d{8}_\d{2}$")


def load_session_manifest(path: Path = MANIFEST) -> dict:
    """Load the canonical chronological session registry."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_source_name(name: str) -> str:
    """Validate and return a canonical Zenodo session stem."""
    if _ORIGINAL_NAME.fullmatch(name):
        return name
    raise KeyError(f"Unknown Indy/Loco session name: {name!r}")


def session_path(name: str, raw_dir: Path = RAW_DIR) -> Path:
    """Return the canonical source-named raw MAT path."""
    source = resolve_source_name(name)
    return raw_dir / f"{source}.mat"


def processed_session_path(name: str, processed_dir: Path = PROCESSED_DIR) -> Path:
    """Return the fixed chronological-split artifact path for a session."""
    manifest = load_session_manifest()
    source = resolve_source_name(name)
    matches = [
        split
        for split, sessions in manifest["chronological_split"].items()
        if source in sessions
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one chronological split for {source}, found {matches}")
    return processed_dir / matches[0] / f"{source}.npz"


def fetch_session(name: str, raw_dir: Path = RAW_DIR) -> Path:
    """Download one missing public session into the immutable raw-data area."""
    path = session_path(name, raw_dir)
    if path.exists():
        return path
    source = resolve_source_name(name)
    raw_dir.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL.format(f"{source}.mat"), path)
    return path


def _decode_matlab_text(dataset: h5py.Dataset) -> str:
    """Decode a MATLAB char array stored behind an HDF5 object reference."""
    values = np.asarray(dataset).reshape(-1)
    return "".join(chr(int(value)) for value in values if int(value))


def _channel_names(file: h5py.File) -> list[str]:
    return [_decode_matlab_text(file[ref]) for ref in np.asarray(file["chan_names"]).reshape(-1)]


def _m1_channel_indices(file: h5py.File, limit: int) -> np.ndarray:
    """Return stable M1 indices using channel metadata, not array position alone."""
    names = _channel_names(file)
    indices = np.asarray(
        [index for index, channel_name in enumerate(names) if channel_name.startswith("M1 ")],
        dtype=np.int64,
    )
    if indices.size < limit:
        raise ValueError(f"Expected at least {limit} M1 channels, found {indices.size}")
    return indices[:limit]


def load_counts_velocity(
    name: str,
    *,
    bin_s: float = DEFAULT_BIN_S,
    velocity_lowpass_hz: float = DEFAULT_VELOCITY_LOWPASS_HZ,
    n_channels: int = DEFAULT_N_CHANNELS,
    download: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unsmoothed M1 counts and two-axis primary-fingertip velocity.

    Returns ``counts`` with shape ``(channels, time_bins)`` and ``velocity`` with
    shape ``(time_bins, 2)`` for the ``(-x, -y)`` axes. Sessions containing two
    tracked markers are reduced to the primary marker's first three coordinates
    before the causal velocity calculation.
    """
    path = fetch_session(name) if download else session_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing raw session {name!r}: {path}. Place it under {RAW_DIR} or "
            "call load_counts_velocity(..., download=True)."
        )

    with h5py.File(path, "r") as f:
        t = np.asarray(f["t"]).squeeze()
        finger_pos = np.asarray(f["finger_pos"])[:3]
        spikes = f["spikes"]
        if t.ndim != 1 or t.size < 2 or not np.all(np.diff(t) > 0):
            raise ValueError(f"Invalid timestamps in {path}")
        n_bins = int(np.floor((t[-1] - t[0]) / bin_s))
        edges = t[0] + np.arange(n_bins + 1, dtype=np.float64) * bin_s
        bin_end_time = edges[1:]
        selected_channels = _m1_channel_indices(f, min(n_channels, spikes.shape[1]))
        counts = np.zeros((len(selected_channels), n_bins), dtype=np.float32)
        for output_channel, source_channel in enumerate(selected_channels):
            for unit in range(spikes.shape[0]):
                unit_data = f[spikes[unit, source_channel]]
                if bool(unit_data.attrs.get("MATLAB_empty", 0)):
                    continue
                events = np.asarray(unit_data).reshape(-1)
                if events.size:
                    counts[output_channel] += np.histogram(events, bins=edges)[0]

    # At each completed neural bin, use only the latest kinematic sample whose
    # timestamp is not later than that bin end. Linear interpolation would read
    # the next 250 Hz sample and therefore introduce a small future-data leak.
    position = causal_sample_hold(t, finger_pos.T, bin_end_time)
    velocity = causal_velocity(position, bin_s, velocity_lowpass_hz)[:, 1:3]
    return counts, velocity


def load_model_data(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the supported causal artifact, falling back to causal raw processing."""
    artifact = processed_session_path(name)
    if artifact.exists():
        with np.load(artifact, allow_pickle=False) as data:
            schema = str(np.asarray(data["schema_version"]).item())
            filter_name = str(np.asarray(data["velocity_filter"]).item())
            difference = str(np.asarray(data["velocity_difference"]).item())
            sampling = str(np.asarray(data["kinematic_sampling"]).item())
            if schema != MODEL_READY_SCHEMA:
                raise ValueError(f"Unsupported schema {schema!r} in {artifact}")
            if (
                filter_name != "causal_forward_butterworth"
                or difference != "backward"
                or sampling != "causal_latest_sample_at_bin_end"
            ):
                raise ValueError(f"Unsupported target metadata in {artifact}")
            counts = data["counts"]
            velocity = data["velocity"]
            return counts.astype(np.float32), velocity.astype(np.float32)
    return load_counts_velocity(name)


def top_firing_channels(
    sessions: dict[str, tuple[np.ndarray, np.ndarray]],
    n: int,
    *,
    observation_bins: int,
) -> np.ndarray:
    """Select top-N channels from each session's allowed past prefix only."""
    if observation_bins <= 0:
        raise ValueError("observation_bins must be positive")
    firing = np.mean(
        [counts[:, :observation_bins].mean(1) for counts, _ in sessions.values()],
        axis=0,
    )
    return np.sort(np.argsort(firing)[-n:])


def fit_feature_stats(
    features: np.ndarray, *, observation_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-feature statistics from an explicit past observation prefix."""
    if observation_bins <= 0 or observation_bins > features.shape[1]:
        raise ValueError("observation_bins must be within the feature timeline")
    observation = features[:, :observation_bins]
    return (
        observation.mean(1, keepdims=True),
        observation.std(1, keepdims=True) + 1e-6,
    )


def apply_feature_stats(
    features: np.ndarray, stats: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """Apply previously fitted statistics without inspecting future samples."""
    mean, std = stats
    return ((features - mean) / std).astype(np.float32)


def window_arrays(
    features: np.ndarray,
    velocity: np.ndarray,
    axes: np.ndarray | tuple[int, int] = (0, 1),
    *,
    window_bins: int = 50,
    start_bin: int = 0,
) -> list[dict[str, np.ndarray]]:
    """Create non-overlapping windows without crossing a requested start boundary."""
    usable = (features.shape[1] - start_bin) // window_bins
    out = []
    for index in range(usable):
        start = start_bin + index * window_bins
        stop = start + window_bins
        out.append({"e": features[:, start:stop], "vel": velocity[start:stop][:, axes]})
    return out
