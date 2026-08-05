"""Frozen FingerMovements terminal low-pass Logistic Regression model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


CHANNELS = 28
TIMEPOINTS = 50
SAMPLING_RATE_HZ = 100.0

LOWPASS_HZ = 5.0
LOWPASS_ORDER = 2
TERMINAL_SAMPLES = 5
TERMINAL_MEAN_WINDOWS = (5, 10, 20)
TERMINAL_SLOPE_WINDOW = 20
FEATURES = CHANNELS * (
    TERMINAL_SAMPLES + len(TERMINAL_MEAN_WINDOWS) + 1
)
LOGISTIC_C = 1.0

LOWPASS_SOS = butter(
    LOWPASS_ORDER,
    LOWPASS_HZ,
    btype="lowpass",
    fs=SAMPLING_RATE_HZ,
    output="sos",
)
LOWPASS_INITIAL = sosfilt_zi(LOWPASS_SOS)


def _validate_raw(x: np.ndarray) -> None:
    if x.ndim != 3 or x.shape[1:] != (CHANNELS, TIMEPOINTS):
        raise ValueError(
            f"Expected (cases, {CHANNELS}, {TIMEPOINTS}), received {x.shape}"
        )
    if not np.isfinite(x).all():
        raise ValueError("EEG input contains non-finite values")


def _validate_labels(y: np.ndarray, cases: int) -> np.ndarray:
    labels = np.asarray(y, dtype=np.int64)
    if labels.shape != (cases,):
        raise ValueError(f"Expected labels with shape ({cases},), received {labels.shape}")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Training labels must contain left=0 and right=1")
    return labels


def causal_lowpass(normalized_x: np.ndarray) -> np.ndarray:
    """Apply the frozen causal IIR, initialized from each trial's first sample."""
    _validate_raw(normalized_x)
    initial = LOWPASS_INITIAL[:, None, None, :] * normalized_x[
        None, :, :, 0, None
    ]
    filtered, _ = sosfilt(
        LOWPASS_SOS,
        normalized_x.astype(np.float64),
        axis=-1,
        zi=initial,
    )
    return filtered.astype(np.float32)


def terminal_features(normalized_x: np.ndarray) -> np.ndarray:
    """Extract the frozen 252-dimensional terminal low-frequency representation.

    Feature order is:
    1. five terminal samples per channel (140 values, channel-major);
    2. per-channel mean over the final 5, 10, and 20 samples (84 values);
    3. per-channel least-squares slope over the final 20 samples (28 values).
    """
    filtered = causal_lowpass(normalized_x)
    terminal = filtered[..., -TERMINAL_SAMPLES:].reshape(len(filtered), -1)
    means = [
        filtered[..., -window:].mean(axis=-1)
        for window in TERMINAL_MEAN_WINDOWS
    ]
    time = np.arange(TERMINAL_SLOPE_WINDOW, dtype=np.float64)
    centered_time = time - time.mean()
    slope = np.tensordot(
        filtered[..., -TERMINAL_SLOPE_WINDOW:],
        centered_time,
        axes=([-1], [0]),
    ) / np.square(centered_time).sum()
    output = np.concatenate([terminal, *means, slope], axis=1).astype(np.float32)
    if output.shape[1] != FEATURES:
        raise RuntimeError(f"Unexpected terminal feature shape: {output.shape}")
    return output


def fit_preprocessing(training_x: np.ndarray) -> dict[str, np.ndarray]:
    """Fit normalization parameters from training cases only."""
    _validate_raw(training_x)
    channel_mean = training_x.mean(
        axis=(0, 2), keepdims=True, dtype=np.float64
    )
    channel_std = np.maximum(
        training_x.std(axis=(0, 2), keepdims=True, dtype=np.float64), 1e-6
    )
    normalized = ((training_x - channel_mean) / channel_std).astype(np.float32)
    features = terminal_features(normalized)
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
    x: np.ndarray, preprocessing: Mapping[str, np.ndarray]
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
        value = np.asarray(preprocessing[name])
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, received {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
        arrays[name] = value
    normalized = (
        (x - arrays["channel_mean"]) / arrays["channel_std"]
    ).astype(np.float32)
    features = terminal_features(normalized)
    return (
        (features - arrays["feature_mean"]) / arrays["feature_std"]
    ).astype(np.float32)


@dataclass(frozen=True)
class TerminalLogistic:
    """Framework-independent binary linear inference parameters."""

    weight: np.ndarray
    bias: float

    def __post_init__(self) -> None:
        weight = np.asarray(self.weight, dtype=np.float64)
        if weight.shape != (FEATURES,):
            raise ValueError(
                f"Expected logistic weight shape ({FEATURES},), received {weight.shape}"
            )
        if not np.isfinite(weight).all() or not np.isfinite(self.bias):
            raise ValueError("Logistic parameters contain non-finite values")
        object.__setattr__(self, "weight", weight.copy())
        object.__setattr__(self, "bias", float(self.bias))

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != FEATURES:
            raise ValueError(
                f"Expected feature shape (cases, {FEATURES}), received {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("Features contain non-finite values")
        return values @ self.weight + self.bias

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        """Return class IDs ordered as left=0 and right=1."""
        return (self.decision_function(features) >= 0.0).astype(np.uint8)

    def probability_right(self, features: np.ndarray) -> np.ndarray:
        score = np.clip(self.decision_function(features), -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-score))

    def predict_raw(
        self, x: np.ndarray, preprocessing: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        return self.predict_features(transform(x, preprocessing))


def fit_logistic(
    training_x: np.ndarray,
    training_y: np.ndarray,
    *,
    c: float = LOGISTIC_C,
) -> tuple[TerminalLogistic, dict[str, np.ndarray]]:
    """Fit the frozen model with training-only preprocessing."""
    if c <= 0:
        raise ValueError("Logistic C must be positive")
    labels = _validate_labels(training_y, len(training_x))
    preprocessing = fit_preprocessing(training_x)
    features = transform(training_x, preprocessing)

    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression(C=c, solver="liblinear", max_iter=5_000)
    classifier.fit(features, labels)
    if classifier.classes_.tolist() != [0, 1]:
        raise RuntimeError(f"Unexpected class order: {classifier.classes_.tolist()}")
    model = TerminalLogistic(
        weight=classifier.coef_[0],
        bias=float(classifier.intercept_[0]),
    )
    if not np.array_equal(
        model.predict_features(features), classifier.predict(features)
    ):
        raise RuntimeError("Framework-independent inference disagrees with scikit-learn")
    return model, preprocessing
