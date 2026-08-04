"""Active FingerMovements handcrafted-feature Logistic Regression model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


CHANNELS = 28
TIMEPOINTS = 50
FEATURES = 196
SAMPLING_INTERVAL_SECONDS = 0.01
BANDS_HZ = ((1, 4), (4, 8), (8, 13), (13, 30))
CURRENT_C = 1.0


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
    """Fit all preprocessing parameters from training cases only."""
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
    features = handcrafted_features(normalized)
    return (
        (features - arrays["feature_mean"]) / arrays["feature_std"]
    ).astype(np.float32)


@dataclass(frozen=True)
class FeatureLogistic:
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
    c: float = CURRENT_C,
) -> tuple[FeatureLogistic, dict[str, np.ndarray]]:
    """Fit the current candidate with training-only preprocessing.

    C=1 is the Phase 1d candidate value, not yet a final frozen value. Phase 1e
    must confirm regularization before a final all-training-data checkpoint is
    created.
    """
    if c <= 0:
        raise ValueError("Logistic C must be positive")
    labels = _validate_labels(training_y, len(training_x))
    preprocessing = fit_preprocessing(training_x)
    features = transform(training_x, preprocessing)

    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression(
        C=c,
        solver="liblinear",
        max_iter=5_000,
    )
    classifier.fit(features, labels)
    if classifier.classes_.tolist() != [0, 1]:
        raise RuntimeError(f"Unexpected class order: {classifier.classes_.tolist()}")
    model = FeatureLogistic(
        weight=classifier.coef_[0],
        bias=float(classifier.intercept_[0]),
    )
    if not np.array_equal(model.predict_features(features), classifier.predict(features)):
        raise RuntimeError("Framework-independent inference disagrees with scikit-learn")
    return model, preprocessing
