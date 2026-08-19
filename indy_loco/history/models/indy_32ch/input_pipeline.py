"""Prepare fixed Indy processed artifacts for the frozen 32-channel model."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed" / "indy_loco" / "indy"
MANIFEST = ROOT / "configs" / "datasets" / "indy_sessions.yaml"
MODEL_READY_SCHEMA = "indy_counts_velocity_v2"
_SESSION_NAME = re.compile(r"^indy_\d{8}_\d{2}$")


def load_session_manifest(path: Path = MANIFEST) -> dict:
    """Load the canonical session registry and chronological split."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_source_name(name: str) -> str:
    """Validate a canonical Zenodo session stem."""
    if _SESSION_NAME.fullmatch(name):
        return name
    raise KeyError(f"Unknown Indy/Loco session name: {name!r}")


def processed_session_path(name: str, processed_dir: Path = PROCESSED_DIR) -> Path:
    """Return the fixed train/validation/test artifact path for one session."""
    manifest = load_session_manifest()
    source = resolve_source_name(name)
    matches = [
        split
        for split, sessions in manifest["chronological_split"].items()
        if source in sessions
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one chronological split for {source}, found {matches}"
        )
    return processed_dir / matches[0] / f"{source}.npz"


def load_model_data(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one supported causal processed artifact."""
    artifact = processed_session_path(name)
    if not artifact.exists():
        raise FileNotFoundError(
            f"Missing processed session {name!r}: {artifact}. "
            "Run data/processing/indy_loco/indy/prepare_indy_model_ready.ipynb."
        )
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
        return data["counts"].astype(np.float32), data["velocity"].astype(np.float32)


def top_firing_channels(
    sessions: dict[str, tuple[np.ndarray, np.ndarray]],
    n: int,
    *,
    observation_bins: int,
) -> np.ndarray:
    """Select top-N channels using only each session's allowed past prefix."""
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
    """Fit per-feature normalization from an explicit past prefix."""
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
    """Apply already-fitted statistics without inspecting future samples."""
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
    """Create non-overlapping windows without crossing the start boundary."""
    usable = (features.shape[1] - start_bin) // window_bins
    output = []
    for index in range(usable):
        start = start_bin + index * window_bins
        stop = start + window_bins
        output.append(
            {"e": features[:, start:stop], "vel": velocity[start:stop][:, axes]}
        )
    return output
