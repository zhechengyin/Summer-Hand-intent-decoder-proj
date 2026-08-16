"""Phase 2f paired TRAIN-only low-dimensional Riemannian evaluation.

The frozen Phase 2c CSSD + hierarchical LDA is reproduced as the baseline.
The candidate uses fold-training-only four-dimensional CSSD projections in
the BP and ERD bands, regularized 4x4 trial covariance matrices, affine-
invariant Riemannian means, tangent-space features, and hierarchical LDA.
The existing BP-trend branch is retained so the candidate does not discard
signed movement-potential information. Official TEST is refused.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfilt, sosfilt_zi
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]

CASES = 316
CHANNELS = 28
AVAILABLE_SAMPLES = 50
HISTORY_SAMPLES = 40
SAMPLING_RATE_HZ = 100.0
HISTORY_MS = 400
UPDATE_MS = 50
CLASS_COUNTS = {0: 159, 1: 157}
SEEDS = (42, 43, 44)
FOLDS = 5

FILTER_ORDER = 4
BP_LOW_PASS_HZ = 7.0
ERD_BAND_HZ = (10.0, 33.0)
CSSD_RIDGE = 1e-6
BASELINE_PATTERNS_PER_CLASS = 1
RIEMANNIAN_PATTERNS_PER_CLASS = 2
RIEMANNIAN_DIMENSION = 4
RIEMANNIAN_COVARIANCE_RIDGE = 1e-3
RIEMANNIAN_MEAN_TOLERANCE = 1e-10
RIEMANNIAN_MEAN_MAX_ITERATIONS = 50

BP_RECENT_SAMPLES = 4
ERD_RECENT_SAMPLES = 32
ERD_POOL_SAMPLES = 8
TREND_OLDEST_SAMPLES = 8
TREND_RECENT_SAMPLES = 10

BASELINE_NAME = "baseline_cssd_lda"
RIEMANNIAN_NAME = "low_dimensional_riemannian_tslda"

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

BP_SOS = butter(
    FILTER_ORDER,
    BP_LOW_PASS_HZ,
    btype="lowpass",
    fs=SAMPLING_RATE_HZ,
    output="sos",
)
ERD_SOS = butter(
    FILTER_ORDER,
    ERD_BAND_HZ,
    btype="bandpass",
    fs=SAMPLING_RATE_HZ,
    output="sos",
)


@dataclass(frozen=True)
class LinearState:
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float

    def decision(self, features: np.ndarray) -> np.ndarray:
        standardized = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        return standardized @ self.coefficient + self.intercept

    def decision_float32(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        standardized = (
            values - self.mean.astype(np.float32)
        ) / self.scale.astype(np.float32)
        return (
            np.sum(
                standardized * self.coefficient.astype(np.float32),
                axis=1,
                dtype=np.float32,
            )
            + np.float32(self.intercept)
        )


@dataclass(frozen=True)
class BaselineModel:
    bp_filters: np.ndarray
    erd_filters: np.ndarray
    bp_branch: LinearState
    erd_branch: LinearState
    trend_branch: LinearState
    fusion: LinearState


@dataclass(frozen=True)
class RiemannianModel:
    bp_filters: np.ndarray
    erd_filters: np.ndarray
    bp_reference_inverse_sqrt: np.ndarray
    erd_reference_inverse_sqrt: np.ndarray
    bp_branch: LinearState
    erd_branch: LinearState
    trend_branch: LinearState
    fusion: LinearState
    bp_mean_iterations: int
    erd_mean_iterations: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/processed/finger_movements/train.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/finger_movements/phase2f_riemannian",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run baseline and candidate on seed 42 fold 1; write no files.",
    )
    args = parser.parse_args()
    if any("test" in part.lower() for part in args.data.parts):
        parser.error("Phase 2f refuses any data path identified as TEST")
    if args.seeds != list(SEEDS):
        parser.error("Phase 2f freezes seeds to exactly 42 43 44")
    if args.folds != FOLDS:
        parser.error("Phase 2f freezes the fold count to exactly five")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_data(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Training data not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {"x", "y", "source_index", "channel_names"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"Missing arrays: {sorted(missing)}")
        x = data["x"].astype(np.float64, copy=True)
        y = data["y"].astype(np.int64, copy=True)
        source_index = data["source_index"].astype(np.int64, copy=True)
        channel_names = data["channel_names"].astype(str, copy=True)
    if x.shape != (CASES, CHANNELS, AVAILABLE_SAMPLES):
        raise ValueError(f"Unexpected x shape: {x.shape}")
    if dict(Counter(y.tolist())) != CLASS_COUNTS:
        raise ValueError(f"Unexpected class counts: {dict(Counter(y.tolist()))}")
    if not np.array_equal(source_index, np.arange(CASES)):
        raise ValueError("source_index must preserve official TRAIN order")
    if channel_names.shape != (CHANNELS,) or len(set(channel_names)) != CHANNELS:
        raise ValueError("Expected 28 unique channel names")
    if not np.isfinite(x).all():
        raise ValueError("TRAIN contains non-finite values")
    if np.all(x[:, :-1, 28:] == x[:, 1:, :22]):
        raise ValueError("Detected retired UEA sliding-channel layout")
    return x, y, source_index, channel_names


def stratified_folds(
    y: np.ndarray, fold_count: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Reproduce the exact deterministic Phase 2c fold construction."""
    rng = np.random.default_rng(seed)
    pieces: dict[int, list[np.ndarray]] = {}
    for label in sorted(np.unique(y).tolist()):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        pieces[label] = list(np.array_split(indices, fold_count))
    all_indices = np.arange(len(y))
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_index in range(fold_count):
        validation = np.concatenate(
            [pieces[label][fold_index] for label in sorted(pieces)]
        )
        training = np.setdiff1d(all_indices, validation, assume_unique=True)
        fold_rng = np.random.default_rng(seed * 10_000 + fold_index)
        fold_rng.shuffle(training)
        fold_rng.shuffle(validation)
        if np.intersect1d(training, validation).size:
            raise RuntimeError("Training/validation fold overlap detected")
        result.append((training, validation))
    return result


def initial_filter_state(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    return sosfilt_zi(sos)[:, None, None, :] * x[None, :, :, 0, None]


def causal_filter(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    filtered, _ = sosfilt(sos, x, axis=-1, zi=initial_filter_state(x, sos))
    if not np.isfinite(filtered).all():
        raise FloatingPointError("Causal filtering produced non-finite values")
    return filtered


def prepare_windows(
    x: np.ndarray, dtype: np.dtype[Any]
) -> tuple[np.ndarray, np.ndarray]:
    values = x.astype(dtype, copy=False)
    bp = causal_filter(values, BP_SOS.astype(dtype))[..., -HISTORY_SAMPLES:]
    bp = bp - bp[..., :1]
    erd = causal_filter(values, ERD_SOS.astype(dtype))[..., -HISTORY_SAMPLES:]
    return bp, erd


def trace_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    trace = float(np.trace(matrix))
    if not np.isfinite(trace) or trace <= 1e-12:
        raise ValueError("Degenerate spatial matrix")
    return matrix / trace


def class_spatial_matrix(class_x: np.ndarray) -> np.ndarray:
    moments = np.einsum("nct,ndt->ncd", class_x, class_x, optimize=True)
    traces = np.trace(moments, axis1=1, axis2=2)
    if not np.isfinite(traces).all() or np.any(traces <= 1e-12):
        raise ValueError("CSSD received a degenerate trial")
    normalized = class_x / np.sqrt(traces)[:, None, None]
    samples = normalized.transpose(0, 2, 1).reshape(-1, CHANNELS)
    matrix = trace_normalize(samples.T @ samples / len(samples))
    scale = float(np.trace(matrix) / CHANNELS)
    matrix = (1.0 - CSSD_RIDGE) * matrix + CSSD_RIDGE * scale * np.eye(CHANNELS)
    return 0.5 * (matrix + matrix.T)


def fit_cssd_filters(
    windowed_x: np.ndarray, labels: np.ndarray, patterns_per_class: int
) -> np.ndarray:
    left = class_spatial_matrix(windowed_x[labels == 0])
    right = class_spatial_matrix(windowed_x[labels == 1])
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


def project(x: np.ndarray, filters: np.ndarray) -> np.ndarray:
    return np.einsum("kc,nct->nkt", filters, x, optimize=True)


def trend_features(bp: np.ndarray, trend_indices: np.ndarray) -> np.ndarray:
    selected = bp[:, trend_indices]
    oldest = selected[..., :TREND_OLDEST_SAMPLES].mean(axis=-1)
    recent = selected[..., -TREND_RECENT_SAMPLES:].mean(axis=-1)
    result = np.stack([oldest, recent], axis=-1).reshape(len(bp), -1)
    if result.shape[1] != 38:
        raise RuntimeError("Unexpected trend feature shape")
    return result


def baseline_features(
    bp: np.ndarray,
    erd: np.ndarray,
    bp_filters: np.ndarray,
    erd_filters: np.ndarray,
    trend_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bp_values = project(bp[..., -BP_RECENT_SAMPLES:], bp_filters).reshape(len(bp), -1)
    erd_projected = project(erd[..., -ERD_RECENT_SAMPLES:], erd_filters)
    erd_values = (
        np.abs(erd_projected)
        .reshape(len(erd), 2, ERD_RECENT_SAMPLES // ERD_POOL_SAMPLES, ERD_POOL_SAMPLES)
        .mean(axis=-1)
        .reshape(len(erd), -1)
    )
    if bp_values.shape[1] != 8 or erd_values.shape[1] != 8:
        raise RuntimeError("Unexpected baseline branch feature shape")
    return bp_values, erd_values, trend_features(bp, trend_indices)


def fit_lda(features: np.ndarray, labels: np.ndarray) -> LinearState:
    pipeline = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="svd"))
    pipeline.fit(features, labels)
    scaler = pipeline.named_steps["standardscaler"]
    lda = pipeline.named_steps["lineardiscriminantanalysis"]
    state = LinearState(
        mean=np.asarray(scaler.mean_, dtype=np.float64),
        scale=np.asarray(scaler.scale_, dtype=np.float64),
        coefficient=np.asarray(lda.coef_[0], dtype=np.float64),
        intercept=float(lda.intercept_[0]),
    )
    expected = np.asarray(pipeline.decision_function(features)).reshape(-1)
    if not np.allclose(state.decision(features), expected, rtol=1e-11, atol=1e-11):
        raise RuntimeError("Serialized LDA state differs from sklearn")
    return state


def fit_baseline(
    bp: np.ndarray,
    erd: np.ndarray,
    labels: np.ndarray,
    trend_indices: np.ndarray,
) -> BaselineModel:
    bp_filters = fit_cssd_filters(
        bp[..., -BP_RECENT_SAMPLES:], labels, BASELINE_PATTERNS_PER_CLASS
    )
    erd_filters = fit_cssd_filters(
        erd[..., -ERD_RECENT_SAMPLES:], labels, BASELINE_PATTERNS_PER_CLASS
    )
    features = baseline_features(
        bp, erd, bp_filters, erd_filters, trend_indices
    )
    branches = tuple(fit_lda(values, labels) for values in features)
    scores = np.column_stack(
        [
            branch.decision(values)
            for branch, values in zip(branches, features, strict=True)
        ]
    )
    fusion = fit_lda(scores, labels)
    return BaselineModel(
        bp_filters,
        erd_filters,
        branches[0],
        branches[1],
        branches[2],
        fusion,
    )


def symmetric_matrix_function(
    matrix: np.ndarray, function: Any, eigenvalue_floor: float = 1e-10
) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    floor = np.asarray(eigenvalue_floor, dtype=matrix.dtype)
    eigenvalues = np.maximum(eigenvalues, floor)
    transformed = function(eigenvalues)
    result = (eigenvectors * transformed) @ eigenvectors.T
    return 0.5 * (result + result.T)


def matrix_log_spd(matrix: np.ndarray) -> np.ndarray:
    return symmetric_matrix_function(matrix, np.log)


def matrix_exp_symmetric(matrix: np.ndarray) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    result = (eigenvectors * np.exp(eigenvalues)) @ eigenvectors.T
    return 0.5 * (result + result.T)


def matrix_power_spd(matrix: np.ndarray, power: float) -> np.ndarray:
    return symmetric_matrix_function(matrix, lambda values: values**power)


def trial_covariances(projected: np.ndarray) -> np.ndarray:
    centered = projected - projected.mean(axis=-1, keepdims=True)
    covariance = np.einsum("nkt,nlt->nkl", centered, centered, optimize=True)
    covariance /= max(projected.shape[-1] - 1, 1)
    dimension = covariance.shape[-1]
    traces = np.trace(covariance, axis1=1, axis2=2)
    scales = np.maximum(traces / dimension, np.finfo(covariance.dtype).eps)
    identity = np.eye(dimension, dtype=covariance.dtype)[None]
    covariance = covariance + (
        np.asarray(RIEMANNIAN_COVARIANCE_RIDGE, dtype=covariance.dtype)
        * scales[:, None, None]
        * identity
    )
    if not np.isfinite(covariance).all():
        raise FloatingPointError("Trial covariance contains non-finite values")
    return 0.5 * (covariance + covariance.transpose(0, 2, 1))


def affine_invariant_mean(covariances: np.ndarray) -> tuple[np.ndarray, int]:
    logs = np.stack([matrix_log_spd(matrix) for matrix in covariances])
    mean = matrix_exp_symmetric(logs.mean(axis=0))
    for iteration in range(1, RIEMANNIAN_MEAN_MAX_ITERATIONS + 1):
        square_root = matrix_power_spd(mean, 0.5)
        inverse_square_root = matrix_power_spd(mean, -0.5)
        updates = np.stack(
            [
                matrix_log_spd(inverse_square_root @ matrix @ inverse_square_root)
                for matrix in covariances
            ]
        )
        tangent_update = updates.mean(axis=0)
        mean = square_root @ matrix_exp_symmetric(tangent_update) @ square_root
        mean = 0.5 * (mean + mean.T)
        if float(np.linalg.norm(tangent_update, ord="fro")) < RIEMANNIAN_MEAN_TOLERANCE:
            return mean, iteration
    raise RuntimeError("Affine-invariant Riemannian mean did not converge")


def tangent_vector(matrix: np.ndarray) -> np.ndarray:
    dimension = matrix.shape[0]
    rows, columns = np.triu_indices(dimension)
    vector = matrix[rows, columns].copy()
    vector[rows != columns] *= np.asarray(np.sqrt(2.0), dtype=matrix.dtype)
    return vector


def tangent_features(
    covariances: np.ndarray, reference_inverse_sqrt: np.ndarray
) -> np.ndarray:
    features = []
    for covariance in covariances:
        whitened = reference_inverse_sqrt @ covariance @ reference_inverse_sqrt
        features.append(tangent_vector(matrix_log_spd(whitened)))
    result = np.stack(features)
    expected = RIEMANNIAN_DIMENSION * (RIEMANNIAN_DIMENSION + 1) // 2
    if result.shape != (len(covariances), expected):
        raise RuntimeError("Unexpected tangent feature shape")
    return result


def riemannian_features(
    bp: np.ndarray,
    erd: np.ndarray,
    bp_filters: np.ndarray,
    erd_filters: np.ndarray,
    bp_reference_inverse_sqrt: np.ndarray,
    erd_reference_inverse_sqrt: np.ndarray,
    trend_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bp_covariances = trial_covariances(project(bp, bp_filters))
    erd_covariances = trial_covariances(
        project(erd[..., -ERD_RECENT_SAMPLES:], erd_filters)
    )
    return (
        tangent_features(bp_covariances, bp_reference_inverse_sqrt),
        tangent_features(erd_covariances, erd_reference_inverse_sqrt),
        trend_features(bp, trend_indices),
    )


def fit_riemannian(
    bp: np.ndarray,
    erd: np.ndarray,
    labels: np.ndarray,
    trend_indices: np.ndarray,
) -> RiemannianModel:
    bp_filters = fit_cssd_filters(
        bp, labels, RIEMANNIAN_PATTERNS_PER_CLASS
    )
    erd_filters = fit_cssd_filters(
        erd[..., -ERD_RECENT_SAMPLES:], labels, RIEMANNIAN_PATTERNS_PER_CLASS
    )
    bp_covariances = trial_covariances(project(bp, bp_filters))
    erd_covariances = trial_covariances(
        project(erd[..., -ERD_RECENT_SAMPLES:], erd_filters)
    )
    bp_mean, bp_iterations = affine_invariant_mean(bp_covariances)
    erd_mean, erd_iterations = affine_invariant_mean(erd_covariances)
    bp_inverse_sqrt = matrix_power_spd(bp_mean, -0.5)
    erd_inverse_sqrt = matrix_power_spd(erd_mean, -0.5)
    features = (
        tangent_features(bp_covariances, bp_inverse_sqrt),
        tangent_features(erd_covariances, erd_inverse_sqrt),
        trend_features(bp, trend_indices),
    )
    branches = tuple(fit_lda(values, labels) for values in features)
    scores = np.column_stack(
        [
            branch.decision(values)
            for branch, values in zip(branches, features, strict=True)
        ]
    )
    fusion = fit_lda(scores, labels)
    return RiemannianModel(
        bp_filters,
        erd_filters,
        bp_inverse_sqrt,
        erd_inverse_sqrt,
        branches[0],
        branches[1],
        branches[2],
        fusion,
        bp_iterations,
        erd_iterations,
    )


def sigmoid(score: np.ndarray) -> np.ndarray:
    result = np.empty_like(score)
    positive = score >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-score[positive]))
    exponent = np.exp(score[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def decisions_from_features(
    features: tuple[np.ndarray, np.ndarray, np.ndarray],
    branches: tuple[LinearState, LinearState, LinearState],
    fusion: LinearState,
    float32: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if float32:
        scores = np.column_stack(
            [
                branch.decision_float32(values)
                for branch, values in zip(branches, features, strict=True)
            ]
        ).astype(np.float32)
        decision = fusion.decision_float32(scores)
    else:
        scores = np.column_stack(
            [
                branch.decision(values)
                for branch, values in zip(branches, features, strict=True)
            ]
        )
        decision = fusion.decision(scores)
    probability = sigmoid(decision)
    prediction = (decision >= 0.0).astype(np.int64)
    return prediction, probability, decision


def predict_baseline(
    model: BaselineModel,
    bp: np.ndarray,
    erd: np.ndarray,
    trend_indices: np.ndarray,
    float32: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dtype = np.float32 if float32 else np.float64
    features = baseline_features(
        bp.astype(dtype, copy=False),
        erd.astype(dtype, copy=False),
        model.bp_filters.astype(dtype),
        model.erd_filters.astype(dtype),
        trend_indices,
    )
    branches = (model.bp_branch, model.erd_branch, model.trend_branch)
    return decisions_from_features(features, branches, model.fusion, float32)


def predict_riemannian(
    model: RiemannianModel,
    bp: np.ndarray,
    erd: np.ndarray,
    trend_indices: np.ndarray,
    float32: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dtype = np.float32 if float32 else np.float64
    features = riemannian_features(
        bp.astype(dtype, copy=False),
        erd.astype(dtype, copy=False),
        model.bp_filters.astype(dtype),
        model.erd_filters.astype(dtype),
        model.bp_reference_inverse_sqrt.astype(dtype),
        model.erd_reference_inverse_sqrt.astype(dtype),
        trend_indices,
    )
    branches = (model.bp_branch, model.erd_branch, model.trend_branch)
    return decisions_from_features(features, branches, model.fusion, float32)


def metric_bundle(
    labels: np.ndarray, prediction: np.ndarray, probability: np.ndarray
) -> dict[str, Any]:
    probability = np.clip(probability, 1e-8, 1.0 - 1e-8)
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "mean_log_loss": float(
            log_loss(
                labels,
                np.column_stack([1.0 - probability, probability]),
                labels=[0, 1],
            )
        ),
        "confusion_matrix": confusion_matrix(
            labels, prediction, labels=[0, 1]
        ).tolist(),
    }


def resource_records() -> list[dict[str, Any]]:
    baseline_floats = 323
    tangent_dimension = RIEMANNIAN_DIMENSION * (RIEMANNIAN_DIMENSION + 1) // 2
    riemannian_floats = (
        BP_SOS.size
        + ERD_SOS.size
        + 2 * RIEMANNIAN_DIMENSION * CHANNELS
        + 2 * RIEMANNIAN_DIMENSION**2
        + 2 * (3 * tangent_dimension + 1)
        + (3 * 38 + 1)
        + (3 * 3 + 1)
    )
    ring_bytes = 2 * CHANNELS * HISTORY_SAMPLES * 4
    iir_state_bytes = (2 * 2 * CHANNELS + 4 * 2 * CHANNELS) * 4
    base_workspace = CHANNELS * 5 * 4 * 3 + (8 + 8 + 38 + 6) * 4
    baseline_feature_buffer = 2 * ERD_RECENT_SAMPLES * 4
    baseline_ram = (
        ring_bytes + iir_state_bytes + base_workspace + baseline_feature_buffer
    )
    riemannian_workspace = (
        2 * RIEMANNIAN_DIMENSION * HISTORY_SAMPLES
        + 8 * RIEMANNIAN_DIMENSION**2
        + 2 * tangent_dimension
    ) * 4
    return [
        {
            "variant": BASELINE_NAME,
            "deployment_float_parameters": baseline_floats,
            "deployment_parameter_bytes_float32": baseline_floats * 4 + 19,
            "deployment_parameter_kb_float32": (baseline_floats * 4 + 19)
            / 1024.0,
            "estimated_runtime_ram_kb_float32": baseline_ram / 1024.0,
            "inference_matrix_eigendecompositions_per_update": 0,
        },
        {
            "variant": RIEMANNIAN_NAME,
            "deployment_float_parameters": int(riemannian_floats),
            "deployment_parameter_bytes_float32": int(riemannian_floats * 4 + 19),
            "deployment_parameter_kb_float32": (riemannian_floats * 4 + 19)
            / 1024.0,
            "estimated_runtime_ram_kb_float32": (
                baseline_ram + riemannian_workspace
            )
            / 1024.0,
            "inference_matrix_eigendecompositions_per_update": 2,
        },
    ]


def summarize(
    fold_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    variants = (BASELINE_NAME, RIEMANNIAN_NAME)
    seed_rows: list[dict[str, Any]] = []
    for variant in variants:
        for seed in SEEDS:
            selected = sorted(
                (
                    row
                    for row in prediction_rows
                    if row["variant"] == variant and row["seed"] == seed
                ),
                key=lambda row: row["source_index"],
            )
            if len(selected) != CASES:
                raise RuntimeError("Incomplete OOF coverage")
            indices = np.asarray([row["source_index"] for row in selected])
            if not np.array_equal(indices, np.arange(CASES)):
                raise RuntimeError("OOF indices are incomplete or duplicated")
            labels = np.asarray([row["true_label"] for row in selected])
            prediction = np.asarray([row["predicted_label"] for row in selected])
            probability = np.asarray([row["probability_right"] for row in selected])
            metrics = metric_bundle(labels, prediction, probability)
            seed_rows.append(
                {
                    "variant": variant,
                    "seed": int(seed),
                    **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
                    "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
                }
            )
    resources = {row["variant"]: row for row in resource_records()}
    summary_rows: list[dict[str, Any]] = []
    for variant in variants:
        selected_seeds = [row for row in seed_rows if row["variant"] == variant]
        balanced = np.asarray([row["balanced_accuracy"] for row in selected_seeds])
        selected_folds = [row for row in fold_rows if row["variant"] == variant]
        fold_balanced = np.asarray(
            [row["balanced_accuracy"] for row in selected_folds]
        )
        summary_rows.append(
            {
                "variant": variant,
                "mean_balanced_accuracy": float(balanced.mean()),
                "balanced_accuracy_seed_sd": float(balanced.std(ddof=0)),
                "worst_seed_balanced_accuracy": float(balanced.min()),
                "best_seed_balanced_accuracy": float(balanced.max()),
                "fold_balanced_accuracy_sd": float(fold_balanced.std(ddof=0)),
                "mean_accuracy": float(np.mean([row["accuracy"] for row in selected_seeds])),
                "mean_macro_f1": float(np.mean([row["macro_f1"] for row in selected_seeds])),
                "mean_log_loss": float(np.mean([row["mean_log_loss"] for row in selected_seeds])),
                "float32_all_predictions_exact": bool(
                    all(row["float32_predictions_exact"] for row in selected_folds)
                ),
                "float32_max_score_error": float(
                    max(row["float32_max_score_error"] for row in selected_folds)
                ),
                "float32_max_probability_error": float(
                    max(row["float32_max_probability_error"] for row in selected_folds)
                ),
                **{
                    key: value
                    for key, value in resources[variant].items()
                    if key != "variant"
                },
            }
        )
    baseline_seed = {
        row["seed"]: row
        for row in seed_rows
        if row["variant"] == BASELINE_NAME
    }
    candidate_seed = [row for row in seed_rows if row["variant"] == RIEMANNIAN_NAME]
    deltas = [
        row["balanced_accuracy"] - baseline_seed[row["seed"]]["balanced_accuracy"]
        for row in candidate_seed
    ]
    baseline_summary = next(row for row in summary_rows if row["variant"] == BASELINE_NAME)
    candidate_summary = next(row for row in summary_rows if row["variant"] == RIEMANNIAN_NAME)
    promotion = {
        "mean_ba_improved": bool(
            candidate_summary["mean_balanced_accuracy"]
            > baseline_summary["mean_balanced_accuracy"]
        ),
        "all_three_seeds_improved": bool(all(delta > 0.0 for delta in deltas)),
        "worst_seed_improved": bool(
            candidate_summary["worst_seed_balanced_accuracy"]
            > baseline_summary["worst_seed_balanced_accuracy"]
        ),
        "fold_variability_not_worse": bool(
            candidate_summary["fold_balanced_accuracy_sd"]
            <= baseline_summary["fold_balanced_accuracy_sd"]
        ),
        "float32_predictions_exact": bool(
            candidate_summary["float32_all_predictions_exact"]
        ),
        "seed_ba_deltas_vs_baseline": deltas,
    }
    promotion["passed"] = bool(
        all(value for key, value in promotion.items() if key != "seed_ba_deltas_vs_baseline")
    )
    return seed_rows, summary_rows, promotion


def complementarity(prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        baseline = sorted(
            (
                row
                for row in prediction_rows
                if row["variant"] == BASELINE_NAME and row["seed"] == seed
            ),
            key=lambda row: row["source_index"],
        )
        candidate = sorted(
            (
                row
                for row in prediction_rows
                if row["variant"] == RIEMANNIAN_NAME and row["seed"] == seed
            ),
            key=lambda row: row["source_index"],
        )
        labels = np.asarray([row["true_label"] for row in baseline])
        base_prediction = np.asarray([row["predicted_label"] for row in baseline])
        candidate_prediction = np.asarray(
            [row["predicted_label"] for row in candidate]
        )
        base_error = base_prediction != labels
        candidate_error = candidate_prediction != labels
        corrected = base_error & ~candidate_error
        introduced = ~base_error & candidate_error
        overlap = base_error & candidate_error
        union = base_error | candidate_error
        rows.append(
            {
                "seed": int(seed),
                "baseline_errors": int(base_error.sum()),
                "riemannian_errors": int(candidate_error.sum()),
                "baseline_errors_corrected": int(corrected.sum()),
                "new_errors_introduced": int(introduced.sum()),
                "corrected_baseline_error_fraction": float(
                    corrected.sum() / max(base_error.sum(), 1)
                ),
                "error_jaccard": float(overlap.sum() / max(union.sum(), 1)),
                "prediction_disagreement_fraction": float(
                    np.mean(base_prediction != candidate_prediction)
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def create_figure(summary_rows: list[dict[str, Any]], path: Path) -> None:
    names = ["Phase 2c\nbaseline", "Low-dimensional\nRiemannian"]
    mean_ba = 100.0 * np.asarray(
        [row["mean_balanced_accuracy"] for row in summary_rows]
    )
    worst_ba = 100.0 * np.asarray(
        [row["worst_seed_balanced_accuracy"] for row in summary_rows]
    )
    fold_sd = 100.0 * np.asarray(
        [row["fold_balanced_accuracy_sd"] for row in summary_rows]
    )
    x_axis = np.arange(2)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.8), constrained_layout=True)
    axes[0].bar(x_axis - 0.18, mean_ba, 0.36, label="Mean OOF BA")
    axes[0].bar(x_axis + 0.18, worst_ba, 0.36, label="Worst-seed BA")
    axes[0].set_xticks(x_axis, names)
    axes[0].set_ylim(65.0, max(90.0, float(mean_ba.max()) + 2.0))
    axes[0].set_ylabel("Balanced accuracy (%)")
    axes[0].legend()
    axes[1].bar(x_axis, fold_sd, color=["tab:blue", "tab:orange"])
    axes[1].set_xticks(x_axis, names)
    axes[1].set_ylabel("Fold BA standard deviation (pp)")
    axes[1].set_title("Lower is better")
    fig.suptitle("Phase 2f TRAIN-only Riemannian comparison")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    x, y, source_index, channel_names = load_training_data(data_path)
    bp64, erd64 = prepare_windows(x, np.dtype(np.float64))
    bp32, erd32 = prepare_windows(x, np.dtype(np.float32))
    trend_indices = np.asarray(
        [
            index
            for index, name in enumerate(channel_names.tolist())
            if name not in REJECTED_TREND_CHANNELS
        ],
        dtype=np.int64,
    )
    if trend_indices.size != 19:
        raise RuntimeError(f"Expected 19 retained trend channels, got {trend_indices.size}")

    print("=== FingerMovements Phase 2f low-dimensional Riemannian test ===")
    print(f"data={data_path}")
    print(f"seeds={list(SEEDS)} | folds={FOLDS} | window={HISTORY_MS} ms")
    print("candidate=4D CSSD -> 4x4 SPD covariance -> tangent-space LDA")
    print("policy=TRAIN only; official TEST refused; exact Phase 2c folds")
    print("terminal rule=if promotion criteria fail, stop model exploration")

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    seeds = SEEDS[:1] if args.validate_only else SEEDS
    for seed in seeds:
        folds = stratified_folds(y, FOLDS, seed)
        if args.validate_only:
            folds = folds[:1]
        for fold_number, (training, validation) in enumerate(folds, start=1):
            for variant in (BASELINE_NAME, RIEMANNIAN_NAME):
                started = perf_counter()
                if variant == BASELINE_NAME:
                    model = fit_baseline(
                        bp64[training], erd64[training], y[training], trend_indices
                    )
                    prediction, probability, score = predict_baseline(
                        model, bp64[validation], erd64[validation], trend_indices, False
                    )
                    prediction32, probability32, score32 = predict_baseline(
                        model, bp32[validation], erd32[validation], trend_indices, True
                    )
                    bp_iterations = 0
                    erd_iterations = 0
                else:
                    model = fit_riemannian(
                        bp64[training], erd64[training], y[training], trend_indices
                    )
                    prediction, probability, score = predict_riemannian(
                        model, bp64[validation], erd64[validation], trend_indices, False
                    )
                    prediction32, probability32, score32 = predict_riemannian(
                        model, bp32[validation], erd32[validation], trend_indices, True
                    )
                    bp_iterations = model.bp_mean_iterations
                    erd_iterations = model.erd_mean_iterations
                metrics = metric_bundle(y[validation], prediction, probability)
                exact = bool(np.array_equal(prediction, prediction32))
                fold_rows.append(
                    {
                        "variant": variant,
                        "seed": int(seed),
                        "fold": int(fold_number),
                        "training_cases": len(training),
                        "validation_cases": len(validation),
                        **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
                        "fit_and_inference_seconds": perf_counter() - started,
                        "riemannian_bp_mean_iterations": bp_iterations,
                        "riemannian_erd_mean_iterations": erd_iterations,
                        "float32_predictions_exact": exact,
                        "float32_max_score_error": float(np.max(np.abs(score - score32))),
                        "float32_max_probability_error": float(
                            np.max(np.abs(probability - probability32))
                        ),
                    }
                )
                for local_index, case_index in enumerate(validation):
                    prediction_rows.append(
                        {
                            "variant": variant,
                            "seed": int(seed),
                            "fold": int(fold_number),
                            "source_index": int(source_index[case_index]),
                            "true_label": int(y[case_index]),
                            "predicted_label": int(prediction[local_index]),
                            "probability_right": float(probability[local_index]),
                            "decision_score": float(score[local_index]),
                            "float32_predicted_label": int(prediction32[local_index]),
                            "float32_probability_right": float(probability32[local_index]),
                            "float32_decision_score": float(score32[local_index]),
                        }
                    )
                print(
                    f"seed={seed} fold={fold_number}/{FOLDS} | {variant} | "
                    f"BA={100.0 * metrics['balanced_accuracy']:.2f}% | "
                    f"float32={'PASS' if exact else 'FAIL'}"
                )

    if args.validate_only:
        if not all(row["float32_predictions_exact"] for row in fold_rows):
            raise RuntimeError("Float32 changed one or more smoke-test predictions")
        print("validate-only=PASS | baseline and Riemannian candidate fitted")
        print("no result files written")
        return

    seed_rows, summary_rows, promotion = summarize(fold_rows, prediction_rows)
    complementarity_rows = complementarity(prediction_rows)
    resources = resource_records()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "phase2f_fold_results.csv", fold_rows)
    write_csv(output_dir / "phase2f_seed_results.csv", seed_rows)
    write_csv(output_dir / "phase2f_summary.csv", summary_rows)
    write_csv(output_dir / "phase2f_oof_predictions.csv", prediction_rows)
    write_csv(output_dir / "phase2f_error_complementarity.csv", complementarity_rows)
    write_csv(output_dir / "phase2f_resource_estimates.csv", resources)
    create_figure(summary_rows, output_dir / "phase2f_summary.png")
    payload = {
        "phase": "2f",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "final lightweight low-dimensional Riemannian candidate",
        "data": {
            "path": str(data_path),
            "sha256": sha256(data_path),
            "cases": CASES,
            "class_counts": {"left": 159, "right": 157},
            "test_policy": "official TEST refused and not loaded",
        },
        "frozen_protocol": {
            "seeds": list(SEEDS),
            "folds": FOLDS,
            "window_ms": HISTORY_MS,
            "update_ms": UPDATE_MS,
            "fold_construction": "exact Phase 2c deterministic stratified folds",
            "all_learned_steps": "fitted inside each training fold only",
        },
        "candidate": {
            "spatial_reduction": "four fold-trained CSSD projections per band",
            "covariance": (
                "4x4 centered trial covariance plus fixed 1e-3 trace-scaled ridge"
            ),
            "geometry": "affine-invariant Riemannian mean and tangent mapping",
            "features": "10 BP tangent + 10 ERD tangent + 38 BP trend",
            "classifier": "three SVD-LDA branches plus SVD-LDA fusion",
        },
        "promotion_rule": {
            "requirements": [
                "mean OOF BA improves",
                "all three seed OOF BAs improve",
                "worst-seed OOF BA improves",
                "fold BA variability does not increase",
                "all float32 OOF predictions match float64",
            ],
            **promotion,
        },
        "terminal_decision": (
            "promote Riemannian candidate for checkpoint confirmation"
            if promotion["passed"]
            else "retain Phase 2c checkpoint and stop further model exploration"
        ),
        "summary": summary_rows,
        "error_complementarity": complementarity_rows,
        "resources": resources,
    }
    with (output_dir / "phase2f_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print("\n=== Phase 2f summary ===")
    for row in summary_rows:
        print(
            f"{row['variant']:<38} | mean BA="
            f"{100.0 * row['mean_balanced_accuracy']:.2f}% | "
            f"seed SD={100.0 * row['balanced_accuracy_seed_sd']:.2f} pp | "
            f"worst={100.0 * row['worst_seed_balanced_accuracy']:.2f}% | "
            f"fold SD={100.0 * row['fold_balanced_accuracy_sd']:.2f} pp | "
            f"float32={'PASS' if row['float32_all_predictions_exact'] else 'FAIL'}"
        )
    print(f"promotion rule={'PASS' if promotion['passed'] else 'FAIL'}")
    print(
        "next="
        + (
            "confirm Riemannian all-TRAIN checkpoint"
            if promotion["passed"]
            else "stop model exploration and return to firmware deployment"
        )
    )
    print(f"metrics={output_dir / 'phase2f_metrics.json'}")


if __name__ == "__main__":
    main()
