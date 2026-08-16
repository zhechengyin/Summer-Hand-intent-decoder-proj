"""Frozen Feature + Linear model and preprocessing contract."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from torch import Tensor, nn


CHANNELS = 28
TIMEPOINTS = 50
FEATURES = 196
DROPOUT = 0.25
SAMPLING_INTERVAL_SECONDS = 0.01
BANDS_HZ = ((1, 4), (4, 8), (8, 13), (13, 30))


class FeatureLinear(nn.Module):
    """Dropout-regularized binary linear classifier over 196 EEG features."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(FEATURES, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


def _validate_raw(x: np.ndarray) -> None:
    if x.ndim != 3 or x.shape[1:] != (CHANNELS, TIMEPOINTS):
        raise ValueError(
            f"Expected (cases, {CHANNELS}, {TIMEPOINTS}), received {x.shape}"
        )
    if not np.isfinite(x).all():
        raise ValueError("EEG input contains non-finite values")


def handcrafted_features(normalized_x: np.ndarray) -> np.ndarray:
    """Create seven deterministic features per EEG channel."""
    _validate_raw(normalized_x)
    blocks = [
        normalized_x.mean(axis=-1),
        normalized_x.std(axis=-1),
        np.square(normalized_x).mean(axis=-1),
    ]
    spectrum = (
        np.square(np.abs(np.fft.rfft(normalized_x, axis=-1)))
        / normalized_x.shape[-1]
    )
    frequencies = np.fft.rfftfreq(
        normalized_x.shape[-1], d=SAMPLING_INTERVAL_SECONDS
    )
    for low, high in BANDS_HZ:
        mask = (frequencies >= low) & (frequencies < high)
        blocks.append(spectrum[..., mask].mean(axis=-1))
    output = np.concatenate(blocks, axis=1).astype(np.float32)
    if output.shape[1] != FEATURES:
        raise RuntimeError(f"Unexpected feature shape: {output.shape}")
    return output


def fit_preprocessing(training_x: np.ndarray) -> dict[str, np.ndarray]:
    """Fit all required preprocessing from training cases only."""
    _validate_raw(training_x)
    channel_mean = training_x.mean(
        axis=(0, 2), keepdims=True, dtype=np.float64
    )
    channel_std = np.maximum(
        training_x.std(axis=(0, 2), keepdims=True, dtype=np.float64), 1e-6
    )
    normalized = ((training_x - channel_mean) / channel_std).astype(np.float32)
    features = handcrafted_features(normalized)
    feature_mean = features.mean(axis=0, keepdims=True, dtype=np.float64)
    feature_std = np.maximum(
        features.std(axis=0, keepdims=True, dtype=np.float64), 1e-6
    )
    return {
        "channel_mean": channel_mean,
        "channel_std": channel_std,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
    }


def transform(
    x: np.ndarray, preprocessing: Mapping[str, np.ndarray | Tensor]
) -> np.ndarray:
    """Apply frozen training-derived preprocessing to raw EEG cases."""
    _validate_raw(x)
    required = {
        "channel_mean": (1, CHANNELS, 1),
        "channel_std": (1, CHANNELS, 1),
        "feature_mean": (1, FEATURES),
        "feature_std": (1, FEATURES),
    }
    arrays: dict[str, np.ndarray] = {}
    for name, shape in required.items():
        if name not in preprocessing:
            raise KeyError(f"Missing preprocessing array: {name}")
        value = preprocessing[name]
        if isinstance(value, Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if value.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}, received {value.shape}"
            )
        arrays[name] = value
    normalized = (
        (x - arrays["channel_mean"]) / arrays["channel_std"]
    ).astype(np.float32)
    features = handcrafted_features(normalized)
    return (
        (features - arrays["feature_mean"])
        / arrays["feature_std"]
    ).astype(np.float32)
