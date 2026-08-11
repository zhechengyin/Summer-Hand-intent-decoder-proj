"""Strictly causal FingerMovements CSSD + hierarchical LDA model.

Each prediction uses the 400 ms interval ending at the current point. Temporal
filters run left-to-right and the streaming interface carries both IIR state
and a rolling history buffer between updates. Official 500 ms epochs provide
100 ms of causal filter pre-roll before the selected 400 ms feature window.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfilt, sosfilt_zi
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CHANNELS = 28
AVAILABLE_INPUT_SAMPLES = 50
HISTORY_SAMPLES = 40
FILTER_PREROLL_SAMPLES = AVAILABLE_INPUT_SAMPLES - HISTORY_SAMPLES
SAMPLING_RATE_HZ = 100.0
HISTORY_MS = 400
COLD_START_MS = 500
UPDATE_SAMPLES = 5
UPDATE_MS = 50
CLASS_COUNTS = {0: 159, 1: 157}

FILTER_ORDER = 4
BP_LOW_PASS_HZ = 7.0
ERD_BAND_HZ = (10.0, 33.0)
CSSD_RIDGE = 1e-6
BP_PATTERNS_PER_CLASS = 1
ERD_PATTERNS_PER_CLASS = 1

BP_RECENT_SAMPLES = 4
ERD_RECENT_SAMPLES = 32
ERD_POOL_SAMPLES = 8
TREND_OLDEST_SAMPLES = 8
TREND_RECENT_SAMPLES = 10

REJECTED_TREND_CHANNELS = (
    "F3",
    "F1",
    "F4",
    "FC5",
    "FC3",
    "C5",
    "C3",
    "CP5",
    "CP3",
)


def _sigmoid(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    probability = np.empty_like(score)
    positive = score >= 0.0
    probability[positive] = 1.0 / (1.0 + np.exp(-score[positive]))
    exponent = np.exp(score[~positive])
    probability[~positive] = exponent / (1.0 + exponent)
    return probability


@dataclass(frozen=True)
class LinearDiscriminantState:
    """Inference-only StandardScaler followed by binary LDA."""

    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.mean.size:
            raise ValueError(
                f"Expected (*, {self.mean.size}) features, got {features.shape}"
            )
        standardized = (features - self.mean) / self.scale
        return standardized @ self.coefficient + self.intercept


def _fit_lda(features: np.ndarray, labels: np.ndarray) -> LinearDiscriminantState:
    pipeline = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="svd"))
    pipeline.fit(features, labels)
    scaler = pipeline.named_steps["standardscaler"]
    lda = pipeline.named_steps["lineardiscriminantanalysis"]
    state = LinearDiscriminantState(
        mean=np.asarray(scaler.mean_, dtype=np.float64),
        scale=np.asarray(scaler.scale_, dtype=np.float64),
        coefficient=np.asarray(lda.coef_[0], dtype=np.float64),
        intercept=float(lda.intercept_[0]),
    )
    expected = np.asarray(pipeline.decision_function(features)).reshape(-1)
    actual = state.decision_function(features)
    if not np.allclose(actual, expected, rtol=1e-12, atol=1e-12):
        raise RuntimeError("Serialized LDA state does not reproduce sklearn scores")
    return state


def _initial_filter_state(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    return sosfilt_zi(sos)[:, None, None, :] * x[None, :, :, 0, None]


def _causal_filter(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    filtered, _ = sosfilt(sos, x, axis=-1, zi=_initial_filter_state(x, sos))
    if not np.isfinite(filtered).all():
        raise FloatingPointError("Causal filtering produced non-finite values")
    return filtered


def _trace_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    trace = float(np.trace(matrix))
    if not np.isfinite(trace) or trace <= 1e-12:
        raise ValueError("Degenerate spatial matrix")
    return matrix / trace


def _class_spatial_matrix(class_x: np.ndarray) -> np.ndarray:
    moments = np.einsum("nct,ndt->ncd", class_x, class_x, optimize=True)
    traces = np.trace(moments, axis1=1, axis2=2)
    if not np.isfinite(traces).all() or np.any(traces <= 1e-12):
        raise ValueError("Trial normalization received a degenerate window")
    normalized = class_x / np.sqrt(traces)[:, None, None]
    samples = normalized.transpose(0, 2, 1).reshape(-1, CHANNELS)
    matrix = _trace_normalize(samples.T @ samples / len(samples))
    scale = float(np.trace(matrix) / CHANNELS)
    matrix = (1.0 - CSSD_RIDGE) * matrix + CSSD_RIDGE * scale * np.eye(CHANNELS)
    return 0.5 * (matrix + matrix.T)


def _fit_cssd_filters(
    windowed_x: np.ndarray, labels: np.ndarray, patterns_per_class: int
) -> np.ndarray:
    left = _class_spatial_matrix(windowed_x[labels == 0])
    right = _class_spatial_matrix(windowed_x[labels == 1])
    eigenvalues, eigenvectors = eigh(left, left + right, check_finite=True)
    order = np.argsort(eigenvalues)
    rows: list[np.ndarray] = []
    for offset in range(patterns_per_class):
        rows.extend(
            [
                eigenvectors[:, int(order[-1 - offset])],
                eigenvectors[:, int(order[offset])],
            ]
        )
    filters = np.stack(rows)
    for row in filters:
        anchor = int(np.argmax(np.abs(row)))
        if row[anchor] < 0.0:
            row *= -1.0
    if not np.isfinite(filters).all():
        raise FloatingPointError("CSSD produced non-finite filters")
    return filters


def _project(x: np.ndarray, filters: np.ndarray) -> np.ndarray:
    return np.einsum("kc,nct->nkt", filters, x, optimize=True)


def _bp_features(bp: np.ndarray, filters: np.ndarray) -> np.ndarray:
    features = _project(bp[..., -BP_RECENT_SAMPLES:], filters).reshape(len(bp), -1)
    if features.shape != (len(bp), 8):
        raise RuntimeError(f"Unexpected BP feature shape: {features.shape}")
    return features


def _erd_features(erd: np.ndarray, filters: np.ndarray) -> np.ndarray:
    projected = _project(erd[..., -ERD_RECENT_SAMPLES:], filters)
    pooled = (
        np.abs(projected)
        .reshape(
            len(erd),
            projected.shape[1],
            ERD_RECENT_SAMPLES // ERD_POOL_SAMPLES,
            ERD_POOL_SAMPLES,
        )
        .mean(axis=-1)
    )
    features = pooled.reshape(len(erd), -1)
    if features.shape != (len(erd), 8):
        raise RuntimeError(f"Unexpected ERD feature shape: {features.shape}")
    return features


def _trend_features(bp: np.ndarray, retained_indices: np.ndarray) -> np.ndarray:
    selected = bp[:, retained_indices]
    oldest = selected[..., :TREND_OLDEST_SAMPLES].mean(axis=-1)
    recent = selected[..., -TREND_RECENT_SAMPLES:].mean(axis=-1)
    features = np.stack([oldest, recent], axis=-1).reshape(len(bp), -1)
    if features.shape != (len(bp), 38):
        raise RuntimeError(f"Unexpected trend feature shape: {features.shape}")
    return features


@dataclass(frozen=True)
class FingerMovementsCausalCssdLda:
    """Frozen Phase 2c causal model for 400 ms past-context windows."""

    channel_names: np.ndarray
    trend_indices: np.ndarray
    bp_sos: np.ndarray
    erd_sos: np.ndarray
    bp_filters: np.ndarray
    erd_filters: np.ndarray
    bp_branch: LinearDiscriminantState
    erd_branch: LinearDiscriminantState
    trend_branch: LinearDiscriminantState
    fusion: LinearDiscriminantState
    metadata: dict[str, Any]

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        labels: np.ndarray,
        channel_names: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> FingerMovementsCausalCssdLda:
        x, labels, channel_names = _validate_inputs(x, labels, channel_names)
        bp_sos = butter(
            FILTER_ORDER,
            BP_LOW_PASS_HZ,
            btype="lowpass",
            fs=SAMPLING_RATE_HZ,
            output="sos",
        )
        erd_sos = butter(
            FILTER_ORDER,
            ERD_BAND_HZ,
            btype="bandpass",
            fs=SAMPLING_RATE_HZ,
            output="sos",
        )
        bp = _causal_filter(x, bp_sos)[..., -HISTORY_SAMPLES:]
        bp = bp - bp[..., :1]
        erd = _causal_filter(x, erd_sos)[..., -HISTORY_SAMPLES:]
        bp_filters = _fit_cssd_filters(
            bp[..., -BP_RECENT_SAMPLES:], labels, BP_PATTERNS_PER_CLASS
        )
        erd_filters = _fit_cssd_filters(
            erd[..., -ERD_RECENT_SAMPLES:], labels, ERD_PATTERNS_PER_CLASS
        )
        rejected = set(REJECTED_TREND_CHANNELS)
        trend_indices = np.asarray(
            [
                index
                for index, name in enumerate(channel_names)
                if name not in rejected
            ],
            dtype=np.int64,
        )
        if trend_indices.size != 19:
            raise ValueError(
                f"Expected 19 retained trend channels, got {trend_indices.size}"
            )
        features = (
            _bp_features(bp, bp_filters),
            _erd_features(erd, erd_filters),
            _trend_features(bp, trend_indices),
        )
        branches = tuple(_fit_lda(values, labels) for values in features)
        branch_scores = np.column_stack(
            [
                branch.decision_function(values)
                for branch, values in zip(branches, features)
            ]
        )
        fusion = _fit_lda(branch_scores, labels)
        return cls(
            channel_names=channel_names,
            trend_indices=trend_indices,
            bp_sos=bp_sos,
            erd_sos=erd_sos,
            bp_filters=bp_filters,
            erd_filters=erd_filters,
            bp_branch=branches[0],
            erd_branch=branches[1],
            trend_branch=branches[2],
            fusion=fusion,
            metadata=dict(metadata or {}),
        )

    def _scores_from_filtered(
        self, bp: np.ndarray, erd: np.ndarray
    ) -> np.ndarray:
        features = (
            _bp_features(bp, self.bp_filters),
            _erd_features(erd, self.erd_filters),
            _trend_features(bp, self.trend_indices),
        )
        branches = (self.bp_branch, self.erd_branch, self.trend_branch)
        branch_scores = np.column_stack(
            [
                branch.decision_function(values)
                for branch, values in zip(branches, features)
            ]
        )
        return self.fusion.decision_function(branch_scores)

    def _filtered_windows(
        self, x: np.ndarray, channel_names: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        channel_names = np.asarray(channel_names).astype(str)
        if x.ndim != 3 or x.shape[1:] != (CHANNELS, AVAILABLE_INPUT_SAMPLES):
            raise ValueError(
                f"Expected (cases, {CHANNELS}, {AVAILABLE_INPUT_SAMPLES}), "
                f"got {x.shape}"
            )
        if not np.isfinite(x).all():
            raise ValueError("Input contains non-finite values")
        if not np.array_equal(channel_names, self.channel_names):
            raise ValueError("Channel names/order differ from the frozen checkpoint")
        bp = _causal_filter(x, self.bp_sos)[..., -HISTORY_SAMPLES:]
        bp = bp - bp[..., :1]
        erd = _causal_filter(x, self.erd_sos)[..., -HISTORY_SAMPLES:]
        return bp, erd

    def decision_function(self, x: np.ndarray, channel_names: np.ndarray) -> np.ndarray:
        bp, erd = self._filtered_windows(x, channel_names)
        return self._scores_from_filtered(bp, erd)

    def predict_proba(self, x: np.ndarray, channel_names: np.ndarray) -> np.ndarray:
        right = _sigmoid(self.decision_function(x, channel_names))
        return np.column_stack([1.0 - right, right])

    def predict(self, x: np.ndarray, channel_names: np.ndarray) -> np.ndarray:
        return (self.decision_function(x, channel_names) >= 0.0).astype(np.int64)

    def new_stream(self, first_sample: np.ndarray) -> CausalStreamingState:
        return CausalStreamingState(self, first_sample)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {
            "format_version": np.asarray("finger_movements_causal_cssd_lda_v2"),
            "metadata_json": np.asarray(json.dumps(self.metadata, sort_keys=True)),
            "channel_names": self.channel_names,
            "trend_indices": self.trend_indices,
            "bp_sos": self.bp_sos,
            "erd_sos": self.erd_sos,
            "bp_filters": self.bp_filters,
            "erd_filters": self.erd_filters,
        }
        for prefix, state in (
            ("bp_branch", self.bp_branch),
            ("erd_branch", self.erd_branch),
            ("trend_branch", self.trend_branch),
            ("fusion", self.fusion),
        ):
            arrays[f"{prefix}_mean"] = state.mean
            arrays[f"{prefix}_scale"] = state.scale
            arrays[f"{prefix}_coefficient"] = state.coefficient
            arrays[f"{prefix}_intercept"] = np.asarray(state.intercept)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> FingerMovementsCausalCssdLda:
        path = Path(path)
        with np.load(path, allow_pickle=False) as checkpoint:
            if str(checkpoint["format_version"]) != (
                "finger_movements_causal_cssd_lda_v2"
            ):
                raise ValueError("Unsupported checkpoint format")

            def state(prefix: str) -> LinearDiscriminantState:
                return LinearDiscriminantState(
                    mean=checkpoint[f"{prefix}_mean"].astype(np.float64),
                    scale=checkpoint[f"{prefix}_scale"].astype(np.float64),
                    coefficient=checkpoint[f"{prefix}_coefficient"].astype(np.float64),
                    intercept=float(checkpoint[f"{prefix}_intercept"]),
                )

            return cls(
                channel_names=checkpoint["channel_names"].astype(str),
                trend_indices=checkpoint["trend_indices"].astype(np.int64),
                bp_sos=checkpoint["bp_sos"].astype(np.float64),
                erd_sos=checkpoint["erd_sos"].astype(np.float64),
                bp_filters=checkpoint["bp_filters"].astype(np.float64),
                erd_filters=checkpoint["erd_filters"].astype(np.float64),
                bp_branch=state("bp_branch"),
                erd_branch=state("erd_branch"),
                trend_branch=state("trend_branch"),
                fusion=state("fusion"),
                metadata=json.loads(str(checkpoint["metadata_json"])),
            )


class CausalStreamingState:
    """Mutable IIR and 400 ms ring state for one continuous EEG stream."""

    def __init__(
        self, model: FingerMovementsCausalCssdLda, first_sample: np.ndarray
    ) -> None:
        first_sample = np.asarray(first_sample, dtype=np.float64)
        if first_sample.shape != (CHANNELS,) or not np.isfinite(first_sample).all():
            raise ValueError(f"Expected one finite ({CHANNELS},) sample")
        self.model = model
        self.bp_state = (
            sosfilt_zi(model.bp_sos)[:, None, :] * first_sample[None, :, None]
        )
        self.erd_state = (
            sosfilt_zi(model.erd_sos)[:, None, :] * first_sample[None, :, None]
        )
        self.bp_ring = np.empty((CHANNELS, 0), dtype=np.float64)
        self.erd_ring = np.empty((CHANNELS, 0), dtype=np.float64)
        self.samples_seen = 0

    def push(
        self, samples: np.ndarray
    ) -> tuple[int, np.ndarray, float] | None:
        """Consume one causal chunk and return prediction/probability/score when warm."""
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[0] != CHANNELS or samples.shape[1] < 1:
            raise ValueError(f"Expected ({CHANNELS}, positive samples), got {samples.shape}")
        if not np.isfinite(samples).all():
            raise ValueError("Streaming input contains non-finite values")
        bp, self.bp_state = sosfilt(
            self.model.bp_sos, samples, axis=-1, zi=self.bp_state
        )
        erd, self.erd_state = sosfilt(
            self.model.erd_sos, samples, axis=-1, zi=self.erd_state
        )
        self.samples_seen += samples.shape[-1]
        self.bp_ring = np.concatenate([self.bp_ring, bp], axis=-1)[
            ..., -HISTORY_SAMPLES:
        ]
        self.erd_ring = np.concatenate([self.erd_ring, erd], axis=-1)[
            ..., -HISTORY_SAMPLES:
        ]
        if (
            self.bp_ring.shape[-1] < HISTORY_SAMPLES
            or self.samples_seen < AVAILABLE_INPUT_SAMPLES
        ):
            return None
        bp_window = self.bp_ring[None] - self.bp_ring[None, ..., :1]
        erd_window = self.erd_ring[None]
        score = float(self.model._scores_from_filtered(bp_window, erd_window)[0])
        probability_right = float(_sigmoid(np.asarray([score]))[0])
        probability = np.asarray([1.0 - probability_right, probability_right])
        return int(score >= 0.0), probability, score


def _validate_inputs(
    x: np.ndarray, labels: np.ndarray, channel_names: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    channel_names = np.asarray(channel_names).astype(str)
    if x.ndim != 3 or x.shape[1:] != (CHANNELS, AVAILABLE_INPUT_SAMPLES):
        raise ValueError(
            f"Expected (cases, {CHANNELS}, {AVAILABLE_INPUT_SAMPLES}), got {x.shape}"
        )
    if labels.shape != (len(x),):
        raise ValueError("Labels do not match the number of cases")
    if channel_names.shape != (CHANNELS,) or len(set(channel_names)) != CHANNELS:
        raise ValueError("Expected 28 unique channel names")
    if not np.isfinite(x).all():
        raise ValueError("Input contains non-finite values")
    observed = dict(zip(*np.unique(labels, return_counts=True), strict=True))
    if set(observed) != {0, 1}:
        raise ValueError(f"Expected binary labels 0/1, got {observed}")
    if not set(REJECTED_TREND_CHANNELS).issubset(set(channel_names.tolist())):
        raise ValueError("One or more required channel names are missing")
    return x, labels, channel_names
