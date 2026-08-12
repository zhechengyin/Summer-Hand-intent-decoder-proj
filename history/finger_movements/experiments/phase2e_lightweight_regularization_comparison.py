"""Phase 2e paired TRAIN-only comparison of lightweight CSSD/LDA variants.

Every method uses the exact Phase 2c seeds, stratified folds, causal filters,
400 ms feature ring, and feature definitions. Official TEST is refused.

ToeplitzLDA here is an explicitly documented lightweight adaptation: each
branch keeps its channel-major spatiotemporal feature layout, estimates a
block-Toeplitz pooled within-class covariance by lag-diagonal averaging, and
applies fold-training-only OAS shrinkage before solving the LDA system. The
three scalar branch scores are fused with shrinkage LDA because they do not
have a temporal block structure.
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
from sklearn.covariance import OAS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE_ROOT = Path(__file__).resolve().parents[1]

CASES = 316
CHANNELS = 28
AVAILABLE_SAMPLES = 50
HISTORY_SAMPLES = 40
SAMPLING_RATE_HZ = 100.0
SAMPLE_INTERVAL_MS = 10
HISTORY_MS = 400
UPDATE_MS = 50
UPDATE_SAMPLES = 5
CLASS_COUNTS = {0: 159, 1: 157}
SEEDS = (42, 43, 44)
FOLDS = 5
INNER_FOLDS = 4

FILTER_ORDER = 4
BP_LOW_PASS_HZ = 7.0
ERD_BAND_HZ = (10.0, 33.0)
CSSD_RIDGE = 1e-6
BP_RECENT_SAMPLES = 4
ERD_RECENT_SAMPLES = 32
ERD_POOL_SAMPLES = 8
TREND_OLDEST_SAMPLES = 8
TREND_RECENT_SAMPLES = 10

FUSION_MIN_CORRECTED_BASELINE_ERROR_FRACTION = 0.10

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
class Variant:
    name: str
    regularized_cssd: bool
    lda_kind: str
    description: str


VARIANTS = (
    Variant(
        "baseline_cssd_lda",
        False,
        "svd",
        "Current empirical CSSD plus StandardScaler and SVD LDA",
    ),
    Variant(
        "regularized_cssd",
        True,
        "svd",
        "Fold-training-only OAS CSSD covariance plus current SVD LDA",
    ),
    Variant(
        "shrinkage_lda",
        False,
        "shrinkage",
        "Current CSSD plus analytical Ledoit-Wolf shrinkage LDA",
    ),
    Variant(
        "regularized_cssd_shrinkage_lda",
        True,
        "shrinkage",
        "OAS-regularized CSSD plus analytical shrinkage LDA",
    ),
    Variant(
        "toeplitz_lda",
        False,
        "toeplitz",
        "Current CSSD plus block-Toeplitz OAS-shrunk branch LDA",
    ),
)


@dataclass(frozen=True)
class LinearState:
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    covariance_shrinkage: float | None = None

    def decision(self, features: np.ndarray) -> np.ndarray:
        standardized = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        return standardized @ self.coefficient + self.intercept

    def decision_float32(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        mean = self.mean.astype(np.float32)
        scale = self.scale.astype(np.float32)
        coefficient = self.coefficient.astype(np.float32)
        standardized = (values - mean) / scale
        return (
            np.sum(standardized * coefficient, axis=1, dtype=np.float32)
            + np.float32(self.intercept)
        )


@dataclass(frozen=True)
class FoldModel:
    variant: Variant
    bp_filters: np.ndarray
    erd_filters: np.ndarray
    bp_branch: LinearState
    erd_branch: LinearState
    trend_branch: LinearState
    fusion: LinearState
    cssd_shrinkages: dict[str, float]


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
        default=ARCHIVE_ROOT / "results/phase2e_lightweight_comparison",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Fit all five variants on the first fold only; write no results.",
    )
    args = parser.parse_args()
    if any("test" in part.lower() for part in args.data.parts):
        parser.error("Phase 2e refuses any data path identified as TEST")
    if args.seeds != list(SEEDS):
        parser.error("Phase 2e freezes seeds to exactly 42 43 44")
    if args.folds != FOLDS:
        parser.error("Phase 2e freezes the fold count to exactly five")
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
    """Reproduce the exact Phase 2c deterministic fold construction."""
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


def _initial_filter_state(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    return sosfilt_zi(sos)[:, None, None, :] * x[None, :, :, 0, None]


def causal_filter(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    filtered, _ = sosfilt(sos, x, axis=-1, zi=_initial_filter_state(x, sos))
    if not np.isfinite(filtered).all():
        raise FloatingPointError("Causal filtering produced non-finite values")
    return filtered


def prepare_windows(
    x: np.ndarray, dtype: np.dtype[Any]
) -> tuple[np.ndarray, np.ndarray]:
    values = x.astype(dtype, copy=False)
    bp_sos = BP_SOS.astype(dtype)
    erd_sos = ERD_SOS.astype(dtype)
    bp = causal_filter(values, bp_sos)[..., -HISTORY_SAMPLES:]
    bp = bp - bp[..., :1]
    erd = causal_filter(values, erd_sos)[..., -HISTORY_SAMPLES:]
    return bp, erd


def trace_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    trace = float(np.trace(matrix))
    if not np.isfinite(trace) or trace <= 1e-12:
        raise ValueError("Degenerate spatial matrix")
    return matrix / trace


def class_spatial_matrix(
    class_x: np.ndarray, regularized: bool
) -> tuple[np.ndarray, float]:
    moments = np.einsum("nct,ndt->ncd", class_x, class_x, optimize=True)
    traces = np.trace(moments, axis1=1, axis2=2)
    if not np.isfinite(traces).all() or np.any(traces <= 1e-12):
        raise ValueError("CSSD received a degenerate trial")
    normalized = class_x / np.sqrt(traces)[:, None, None]
    samples = normalized.transpose(0, 2, 1).reshape(-1, CHANNELS)
    if regularized:
        estimator = OAS(assume_centered=True).fit(samples)
        matrix = trace_normalize(estimator.covariance_)
        shrinkage = float(estimator.shrinkage_)
    else:
        matrix = trace_normalize(samples.T @ samples / len(samples))
        shrinkage = 0.0
    scale = float(np.trace(matrix) / CHANNELS)
    matrix = (1.0 - CSSD_RIDGE) * matrix + CSSD_RIDGE * scale * np.eye(CHANNELS)
    return 0.5 * (matrix + matrix.T), shrinkage


def fit_cssd_filters(
    windowed_x: np.ndarray, labels: np.ndarray, regularized: bool
) -> tuple[np.ndarray, dict[str, float]]:
    left, left_shrinkage = class_spatial_matrix(
        windowed_x[labels == 0], regularized
    )
    right, right_shrinkage = class_spatial_matrix(
        windowed_x[labels == 1], regularized
    )
    eigenvalues, eigenvectors = eigh(left, left + right, check_finite=True)
    order = np.argsort(eigenvalues)
    filters = np.stack([eigenvectors[:, order[-1]], eigenvectors[:, order[0]]])
    for row in filters:
        anchor = int(np.argmax(np.abs(row)))
        if row[anchor] < 0.0:
            row *= -1.0
    if not np.isfinite(filters).all():
        raise FloatingPointError("CSSD produced non-finite filters")
    return filters, {"left": left_shrinkage, "right": right_shrinkage}


def project(x: np.ndarray, filters: np.ndarray) -> np.ndarray:
    return np.einsum("kc,nct->nkt", filters, x, optimize=True)


def extract_features(
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
    selected = bp[:, trend_indices]
    oldest = selected[..., :TREND_OLDEST_SAMPLES].mean(axis=-1)
    recent = selected[..., -TREND_RECENT_SAMPLES:].mean(axis=-1)
    trend_values = np.stack([oldest, recent], axis=-1).reshape(len(bp), -1)
    if bp_values.shape[1] != 8 or erd_values.shape[1] != 8:
        raise RuntimeError("Unexpected CSSD branch feature shape")
    if trend_values.shape[1] != 38:
        raise RuntimeError("Unexpected trend feature shape")
    return bp_values, erd_values, trend_values


def fit_sklearn_lda(
    features: np.ndarray, labels: np.ndarray, shrinkage: bool
) -> LinearState:
    scaler = StandardScaler().fit(features)
    standardized = scaler.transform(features)
    if shrinkage:
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    else:
        lda = LinearDiscriminantAnalysis(solver="svd")
    lda.fit(standardized, labels)
    state = LinearState(
        mean=np.asarray(scaler.mean_, dtype=np.float64),
        scale=np.asarray(scaler.scale_, dtype=np.float64),
        coefficient=np.asarray(lda.coef_[0], dtype=np.float64),
        intercept=float(lda.intercept_[0]),
    )
    expected = np.asarray(lda.decision_function(standardized)).reshape(-1)
    if not np.allclose(state.decision(features), expected, rtol=1e-11, atol=1e-11):
        raise RuntimeError("Extracted LDA state differs from sklearn")
    return state


def block_toeplitz_covariance(
    covariance: np.ndarray, spatial_features: int, temporal_features: int
) -> np.ndarray:
    expected = spatial_features * temporal_features
    if covariance.shape != (expected, expected):
        raise ValueError("Toeplitz feature layout does not match covariance")
    structured = np.empty_like(covariance)
    for first in range(spatial_features):
        first_slice = slice(first * temporal_features, (first + 1) * temporal_features)
        for second in range(spatial_features):
            second_slice = slice(
                second * temporal_features, (second + 1) * temporal_features
            )
            block = covariance[first_slice, second_slice]
            rebuilt = np.empty_like(block)
            for lag in range(-(temporal_features - 1), temporal_features):
                values = np.diag(block, k=lag)
                rows, columns = np.where(
                    np.subtract.outer(
                        np.arange(temporal_features), np.arange(temporal_features)
                    )
                    == -lag
                )
                rebuilt[rows, columns] = values.mean()
            structured[first_slice, second_slice] = rebuilt
    return 0.5 * (structured + structured.T)


def fit_toeplitz_lda(
    features: np.ndarray,
    labels: np.ndarray,
    layout: tuple[int, int],
) -> LinearState:
    scaler = StandardScaler().fit(features)
    standardized = scaler.transform(features)
    class_zero = standardized[labels == 0]
    class_one = standardized[labels == 1]
    mean_zero = class_zero.mean(axis=0)
    mean_one = class_one.mean(axis=0)
    centered = np.concatenate(
        [class_zero - mean_zero, class_one - mean_one], axis=0
    )
    covariance = centered.T @ centered / max(len(centered) - 2, 1)
    structured = block_toeplitz_covariance(covariance, *layout)
    shrinkage = float(OAS(assume_centered=True).fit(centered).shrinkage_)
    target_scale = float(np.trace(structured) / structured.shape[0])
    regularized = (
        (1.0 - shrinkage) * structured
        + shrinkage * target_scale * np.eye(structured.shape[0])
    )
    ridge = max(target_scale, 1.0) * 1e-10
    regularized = 0.5 * (regularized + regularized.T) + ridge * np.eye(
        regularized.shape[0]
    )
    coefficient = np.linalg.solve(regularized, mean_one - mean_zero)
    priors = np.bincount(labels, minlength=2).astype(np.float64) / len(labels)
    intercept = float(
        -0.5 * (mean_one + mean_zero) @ coefficient
        + np.log(priors[1] / priors[0])
    )
    return LinearState(
        mean=np.asarray(scaler.mean_, dtype=np.float64),
        scale=np.asarray(scaler.scale_, dtype=np.float64),
        coefficient=np.asarray(coefficient, dtype=np.float64),
        intercept=intercept,
        covariance_shrinkage=shrinkage,
    )


def fit_linear_state(
    features: np.ndarray,
    labels: np.ndarray,
    kind: str,
    layout: tuple[int, int] | None = None,
) -> LinearState:
    if kind == "svd":
        return fit_sklearn_lda(features, labels, shrinkage=False)
    if kind == "shrinkage":
        return fit_sklearn_lda(features, labels, shrinkage=True)
    if kind == "toeplitz":
        if layout is None:
            raise ValueError("ToeplitzLDA requires a spatial/temporal layout")
        return fit_toeplitz_lda(features, labels, layout)
    raise ValueError(f"Unknown LDA kind: {kind}")


def fit_fold_model(
    bp: np.ndarray,
    erd: np.ndarray,
    labels: np.ndarray,
    trend_indices: np.ndarray,
    variant: Variant,
) -> FoldModel:
    bp_filters, bp_shrinkages = fit_cssd_filters(
        bp[..., -BP_RECENT_SAMPLES:], labels, variant.regularized_cssd
    )
    erd_filters, erd_shrinkages = fit_cssd_filters(
        erd[..., -ERD_RECENT_SAMPLES:], labels, variant.regularized_cssd
    )
    features = extract_features(
        bp, erd, bp_filters, erd_filters, trend_indices
    )
    layouts = ((2, 4), (2, 4), (19, 2))
    branches = tuple(
        fit_linear_state(values, labels, variant.lda_kind, layout)
        for values, layout in zip(features, layouts, strict=True)
    )
    branch_scores = np.column_stack(
        [
            branch.decision(values)
            for branch, values in zip(branches, features, strict=True)
        ]
    )
    fusion_kind = "shrinkage" if variant.lda_kind in {"shrinkage", "toeplitz"} else "svd"
    fusion = fit_linear_state(branch_scores, labels, fusion_kind)
    return FoldModel(
        variant=variant,
        bp_filters=bp_filters,
        erd_filters=erd_filters,
        bp_branch=branches[0],
        erd_branch=branches[1],
        trend_branch=branches[2],
        fusion=fusion,
        cssd_shrinkages={
            "bp_left": bp_shrinkages["left"],
            "bp_right": bp_shrinkages["right"],
            "erd_left": erd_shrinkages["left"],
            "erd_right": erd_shrinkages["right"],
        },
    )


def sigmoid(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score)
    result = np.empty_like(score)
    positive = score >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-score[positive]))
    exponent = np.exp(score[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def predict_model(
    model: FoldModel,
    bp: np.ndarray,
    erd: np.ndarray,
    trend_indices: np.ndarray,
    float32: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if float32:
        features = extract_features(
            bp.astype(np.float32, copy=False),
            erd.astype(np.float32, copy=False),
            model.bp_filters.astype(np.float32),
            model.erd_filters.astype(np.float32),
            trend_indices,
        )
        branches = (model.bp_branch, model.erd_branch, model.trend_branch)
        branch_scores = np.column_stack(
            [
                branch.decision_float32(values)
                for branch, values in zip(branches, features, strict=True)
            ]
        ).astype(np.float32)
        score = model.fusion.decision_float32(branch_scores)
    else:
        features = extract_features(
            bp, erd, model.bp_filters, model.erd_filters, trend_indices
        )
        branches = (model.bp_branch, model.erd_branch, model.trend_branch)
        branch_scores = np.column_stack(
            [
                branch.decision(values)
                for branch, values in zip(branches, features, strict=True)
            ]
        )
        score = model.fusion.decision(branch_scores)
    probability = sigmoid(score)
    prediction = (score >= 0.0).astype(np.int64)
    return prediction, probability, score


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


def model_resource_record(name: str, fusion: bool = False) -> dict[str, Any]:
    filter_floats = BP_SOS.size + ERD_SOS.size
    cssd_floats = 2 * CHANNELS * 2
    classifier_floats = (3 * 8 + 1) * 2 + (3 * 38 + 1) + (3 * 3 + 1)
    if fusion:
        classifier_floats = 2 * classifier_floats + (3 * 2 + 1)
    total_floats = filter_floats + cssd_floats + classifier_floats
    parameter_bytes = total_floats * 4 + 19
    ring_bytes = 2 * CHANNELS * HISTORY_SAMPLES * 4
    iir_state_bytes = (2 * 2 * CHANNELS + 4 * 2 * CHANNELS) * 4
    input_chunk_bytes = CHANNELS * UPDATE_SAMPLES * 4
    filtered_chunk_bytes = 2 * input_chunk_bytes
    feature_workspace_bytes = (8 + 8 + 38 + 2 * ERD_RECENT_SAMPLES + 6) * 4
    runtime_ram_bytes = (
        ring_bytes
        + iir_state_bytes
        + input_chunk_bytes
        + filtered_chunk_bytes
        + feature_workspace_bytes
    )
    return {
        "variant": name,
        "deployment_float_parameters": int(total_floats),
        "deployment_parameter_bytes_float32": int(parameter_bytes),
        "deployment_parameter_kb_float32": float(parameter_bytes / 1024.0),
        "estimated_runtime_ram_bytes_float32": int(runtime_ram_bytes),
        "estimated_runtime_ram_kb_float32": float(runtime_ram_bytes / 1024.0),
        "note": (
            "Training covariance matrices are excluded because firmware stores "
            "only frozen linear weights; RAM is a conservative workspace estimate"
        ),
    }


def complementarity_rows(
    predictions: list[dict[str, Any]], candidate: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        baseline = sorted(
            (
                row
                for row in predictions
                if row["variant"] == "baseline_cssd_lda" and row["seed"] == seed
            ),
            key=lambda row: row["source_index"],
        )
        alternative = sorted(
            (
                row
                for row in predictions
                if row["variant"] == candidate and row["seed"] == seed
            ),
            key=lambda row: row["source_index"],
        )
        if len(baseline) != CASES or len(alternative) != CASES:
            raise RuntimeError("Incomplete OOF predictions for complementarity")
        labels = np.asarray([row["true_label"] for row in baseline])
        base_pred = np.asarray([row["predicted_label"] for row in baseline])
        alt_pred = np.asarray([row["predicted_label"] for row in alternative])
        base_error = base_pred != labels
        alt_error = alt_pred != labels
        corrected = base_error & ~alt_error
        introduced = ~base_error & alt_error
        overlap = base_error & alt_error
        union = base_error | alt_error
        disagreement = base_pred != alt_pred
        rows.append(
            {
                "candidate": candidate,
                "seed": int(seed),
                "baseline_errors": int(base_error.sum()),
                "candidate_errors": int(alt_error.sum()),
                "baseline_errors_corrected": int(corrected.sum()),
                "new_errors_introduced": int(introduced.sum()),
                "corrected_baseline_error_fraction": float(
                    corrected.sum() / max(base_error.sum(), 1)
                ),
                "error_jaccard": float(overlap.sum() / max(union.sum(), 1)),
                "prediction_disagreement_fraction": float(disagreement.mean()),
                "candidate_accuracy_on_disagreements": float(
                    np.mean(alt_pred[disagreement] == labels[disagreement])
                    if disagreement.any()
                    else 0.0
                ),
            }
        )
    return rows


def fit_nested_fusion(
    y: np.ndarray,
    bp64: np.ndarray,
    erd64: np.ndarray,
    bp32: np.ndarray,
    erd32: np.ndarray,
    trend_indices: np.ndarray,
    outer_cache: dict[tuple[int, int, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline = next(item for item in VARIANTS if item.name == "baseline_cssd_lda")
    toeplitz = next(item for item in VARIANTS if item.name == "toeplitz_lda")
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for fold_number, (training, validation) in enumerate(
            stratified_folds(y, FOLDS, seed), start=1
        ):
            inner_scores = np.empty((len(training), 2), dtype=np.float64)
            inner_labels = y[training]
            inner_seed = seed * 100 + fold_number
            for inner_training, inner_validation in stratified_folds(
                inner_labels, INNER_FOLDS, inner_seed
            ):
                absolute_training = training[inner_training]
                absolute_validation = training[inner_validation]
                for column, variant in enumerate((baseline, toeplitz)):
                    model = fit_fold_model(
                        bp64[absolute_training],
                        erd64[absolute_training],
                        y[absolute_training],
                        trend_indices,
                        variant,
                    )
                    _, _, score = predict_model(
                        model,
                        bp64[absolute_validation],
                        erd64[absolute_validation],
                        trend_indices,
                    )
                    inner_scores[inner_validation, column] = score
            meta = fit_linear_state(inner_scores, inner_labels, "shrinkage")
            baseline_cache = outer_cache[(seed, fold_number, baseline.name)]
            toeplitz_cache = outer_cache[(seed, fold_number, toeplitz.name)]
            outer_scores = np.column_stack(
                [baseline_cache["score"], toeplitz_cache["score"]]
            )
            outer_scores32 = np.column_stack(
                [baseline_cache["score32"], toeplitz_cache["score32"]]
            ).astype(np.float32)
            score = meta.decision(outer_scores)
            score32 = meta.decision_float32(outer_scores32)
            probability = sigmoid(score)
            probability32 = sigmoid(score32)
            prediction = (score >= 0.0).astype(np.int64)
            prediction32 = (score32 >= 0.0).astype(np.int64)
            metrics = metric_bundle(y[validation], prediction, probability)
            fold_rows.append(
                {
                    "variant": "baseline_toeplitz_nested_fusion",
                    "seed": int(seed),
                    "fold": int(fold_number),
                    "training_cases": len(training),
                    "validation_cases": len(validation),
                    **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
                    "float32_predictions_exact": bool(
                        np.array_equal(prediction, prediction32)
                    ),
                    "float32_max_score_error": float(np.max(np.abs(score - score32))),
                    "float32_max_probability_error": float(
                        np.max(np.abs(probability - probability32))
                    ),
                }
            )
            for local_index, source_index in enumerate(validation):
                prediction_rows.append(
                    {
                        "variant": "baseline_toeplitz_nested_fusion",
                        "seed": int(seed),
                        "fold": int(fold_number),
                        "source_index": int(source_index),
                        "true_label": int(y[source_index]),
                        "predicted_label": int(prediction[local_index]),
                        "probability_right": float(probability[local_index]),
                        "decision_score": float(score[local_index]),
                        "float32_predicted_label": int(prediction32[local_index]),
                        "float32_probability_right": float(probability32[local_index]),
                        "float32_decision_score": float(score32[local_index]),
                    }
                )
            print(
                f"fusion seed={seed} fold={fold_number}/{FOLDS} | "
                f"BA={100.0 * metrics['balanced_accuracy']:.2f}%"
            )
    return fold_rows, prediction_rows


def summarize(
    fold_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    variants = list(dict.fromkeys(row["variant"] for row in prediction_rows))
    seed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
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
                raise RuntimeError(f"Incomplete OOF coverage for {variant}, seed={seed}")
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
        variant_seed_rows = [row for row in seed_rows if row["variant"] == variant]
        balanced = np.asarray(
            [row["balanced_accuracy"] for row in variant_seed_rows]
        )
        variant_folds = [row for row in fold_rows if row["variant"] == variant]
        fold_balanced = np.asarray(
            [row["balanced_accuracy"] for row in variant_folds]
        )
        float32_exact = all(row["float32_predictions_exact"] for row in variant_folds)
        summary_rows.append(
            {
                "variant": variant,
                "mean_balanced_accuracy": float(balanced.mean()),
                "balanced_accuracy_seed_sd": float(balanced.std(ddof=0)),
                "worst_seed_balanced_accuracy": float(balanced.min()),
                "best_seed_balanced_accuracy": float(balanced.max()),
                "fold_balanced_accuracy_sd": float(fold_balanced.std(ddof=0)),
                "mean_accuracy": float(np.mean([row["accuracy"] for row in variant_seed_rows])),
                "mean_macro_f1": float(np.mean([row["macro_f1"] for row in variant_seed_rows])),
                "mean_log_loss": float(np.mean([row["mean_log_loss"] for row in variant_seed_rows])),
                "float32_all_predictions_exact": bool(float32_exact),
                "float32_max_score_error": float(
                    max(row["float32_max_score_error"] for row in variant_folds)
                ),
                "float32_max_probability_error": float(
                    max(row["float32_max_probability_error"] for row in variant_folds)
                ),
            }
        )
    baseline_by_seed = {
        row["seed"]: row
        for row in seed_rows
        if row["variant"] == "baseline_cssd_lda"
    }
    for row in summary_rows:
        variant_seed_rows = [item for item in seed_rows if item["variant"] == row["variant"]]
        deltas = [
            item["balanced_accuracy"]
            - baseline_by_seed[item["seed"]]["balanced_accuracy"]
            for item in variant_seed_rows
        ]
        row["seed_ba_deltas_vs_baseline"] = json.dumps(deltas)
        row["seeds_improved_vs_baseline"] = int(sum(delta > 0.0 for delta in deltas))
        row["all_three_seeds_improved"] = bool(all(delta > 0.0 for delta in deltas))
    return seed_rows, summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def create_figure(summary: list[dict[str, Any]], path: Path) -> None:
    names = [row["variant"].replace("_", "\n") for row in summary]
    mean_ba = 100.0 * np.asarray([row["mean_balanced_accuracy"] for row in summary])
    worst_ba = 100.0 * np.asarray(
        [row["worst_seed_balanced_accuracy"] for row in summary]
    )
    fold_sd = 100.0 * np.asarray(
        [row["fold_balanced_accuracy_sd"] for row in summary]
    )
    x_axis = np.arange(len(summary))
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.0), constrained_layout=True)
    axes[0].bar(x_axis - 0.18, mean_ba, width=0.36, label="Mean seed OOF BA")
    axes[0].bar(x_axis + 0.18, worst_ba, width=0.36, label="Worst-seed OOF BA")
    axes[0].axhline(mean_ba[0], color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Balanced accuracy (%)")
    axes[0].set_ylim(65.0, max(90.0, float(mean_ba.max()) + 2.0))
    axes[0].legend()
    axes[1].bar(x_axis, fold_sd, color="tab:orange")
    axes[1].set_ylabel("Fold BA standard deviation (pp)")
    axes[1].set_xticks(x_axis, names)
    axes[1].set_title("Lower fold variability is better")
    fig.suptitle("Phase 2e TRAIN-only lightweight CSSD/LDA comparison")
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

    print("=== FingerMovements Phase 2e lightweight regularization comparison ===")
    print(f"data={data_path}")
    print(f"variants={[variant.name for variant in VARIANTS]}")
    print(f"seeds={list(SEEDS)} | folds={FOLDS} | window={HISTORY_MS} ms")
    print("policy=TRAIN only; official TEST refused; identical Phase 2c folds")
    print("float32 check=entire causal filter + feature + classifier inference")

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    outer_cache: dict[tuple[int, int, str], dict[str, Any]] = {}
    seeds = SEEDS[:1] if args.validate_only else SEEDS
    for seed in seeds:
        folds = stratified_folds(y, FOLDS, seed)
        if args.validate_only:
            folds = folds[:1]
        for fold_number, (training, validation) in enumerate(folds, start=1):
            for variant in VARIANTS:
                started = perf_counter()
                model = fit_fold_model(
                    bp64[training],
                    erd64[training],
                    y[training],
                    trend_indices,
                    variant,
                )
                prediction, probability, score = predict_model(
                    model, bp64[validation], erd64[validation], trend_indices
                )
                prediction32, probability32, score32 = predict_model(
                    model,
                    bp32[validation],
                    erd32[validation],
                    trend_indices,
                    float32=True,
                )
                metrics = metric_bundle(y[validation], prediction, probability)
                elapsed = perf_counter() - started
                exact = bool(np.array_equal(prediction, prediction32))
                score_error = float(np.max(np.abs(score - score32)))
                probability_error = float(
                    np.max(np.abs(probability - probability32))
                )
                fold_rows.append(
                    {
                        "variant": variant.name,
                        "seed": int(seed),
                        "fold": int(fold_number),
                        "training_cases": len(training),
                        "validation_cases": len(validation),
                        **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
                        "fit_and_inference_seconds": elapsed,
                        "float32_predictions_exact": exact,
                        "float32_max_score_error": score_error,
                        "float32_max_probability_error": probability_error,
                        **{
                            f"cssd_oas_{key}": value
                            for key, value in model.cssd_shrinkages.items()
                        },
                    }
                )
                for local_index, case_index in enumerate(validation):
                    prediction_rows.append(
                        {
                            "variant": variant.name,
                            "seed": int(seed),
                            "fold": int(fold_number),
                            "source_index": int(source_index[case_index]),
                            "true_label": int(y[case_index]),
                            "predicted_label": int(prediction[local_index]),
                            "probability_right": float(probability[local_index]),
                            "decision_score": float(score[local_index]),
                            "float32_predicted_label": int(prediction32[local_index]),
                            "float32_probability_right": float(
                                probability32[local_index]
                            ),
                            "float32_decision_score": float(score32[local_index]),
                        }
                    )
                outer_cache[(seed, fold_number, variant.name)] = {
                    "score": score,
                    "score32": score32,
                }
                print(
                    f"seed={seed} fold={fold_number}/{FOLDS} | "
                    f"{variant.name} | BA={100.0 * metrics['balanced_accuracy']:.2f}% | "
                    f"float32={'PASS' if exact else 'FAIL'}"
                )

    if args.validate_only:
        if not all(row["float32_predictions_exact"] for row in fold_rows):
            raise RuntimeError("One or more float32 validation predictions changed")
        print("validate-only=PASS | five variants fitted | no files written")
        return

    complementarity: list[dict[str, Any]] = []
    for variant in VARIANTS[1:]:
        complementarity.extend(complementarity_rows(prediction_rows, variant.name))
    toeplitz_complementarity = [
        row for row in complementarity if row["candidate"] == "toeplitz_lda"
    ]
    fusion_gate_passed = all(
        row["corrected_baseline_error_fraction"]
        >= FUSION_MIN_CORRECTED_BASELINE_ERROR_FRACTION
        for row in toeplitz_complementarity
    )
    if fusion_gate_passed:
        print("Toeplitz complementarity gate=PASS; running nested OOF fusion")
        fusion_folds, fusion_predictions = fit_nested_fusion(
            y, bp64, erd64, bp32, erd32, trend_indices, outer_cache
        )
        fold_rows.extend(fusion_folds)
        prediction_rows.extend(fusion_predictions)
        complementarity.extend(
            complementarity_rows(
                prediction_rows, "baseline_toeplitz_nested_fusion"
            )
        )
    else:
        print("Toeplitz complementarity gate=FAIL; fusion skipped as predeclared")

    seed_rows, summary_rows = summarize(fold_rows, prediction_rows)
    resources = [model_resource_record(variant.name) for variant in VARIANTS]
    if fusion_gate_passed:
        resources.append(
            model_resource_record("baseline_toeplitz_nested_fusion", fusion=True)
        )
    resources_by_name = {row["variant"]: row for row in resources}
    for row in summary_rows:
        row.update(
            {
                key: value
                for key, value in resources_by_name[row["variant"]].items()
                if key != "variant" and key != "note"
            }
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "phase2e_fold_results.csv", fold_rows)
    write_csv(output_dir / "phase2e_seed_results.csv", seed_rows)
    write_csv(output_dir / "phase2e_summary.csv", summary_rows)
    write_csv(output_dir / "phase2e_oof_predictions.csv", prediction_rows)
    write_csv(output_dir / "phase2e_error_complementarity.csv", complementarity)
    write_csv(output_dir / "phase2e_resource_estimates.csv", resources)
    create_figure(summary_rows, output_dir / "phase2e_summary.png")

    selected = max(
        summary_rows,
        key=lambda row: (
            row["mean_balanced_accuracy"],
            row["worst_seed_balanced_accuracy"],
            -row["fold_balanced_accuracy_sd"],
        ),
    )
    record = {
        "phase": "2e",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "paired lightweight CSSD/LDA regularization comparison",
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
            "all_learned_steps": "fitted within each training fold only",
        },
        "variants": [variant.__dict__ for variant in VARIANTS],
        "toeplitz_definition": (
            "channel-major block-Toeplitz pooled within-class covariance via "
            "lag-diagonal averaging, followed by fold-training-only OAS shrinkage"
        ),
        "fusion_gate": {
            "minimum_corrected_baseline_error_fraction_in_every_seed": (
                FUSION_MIN_CORRECTED_BASELINE_ERROR_FRACTION
            ),
            "passed": fusion_gate_passed,
            "method_if_passed": (
                "outer-fold evaluation with four-fold inner-OOF shrinkage-LDA "
                "stacking of baseline and Toeplitz decision scores"
            ),
        },
        "selection_rule": (
            "judge mean BA, consistency across all three seeds, worst-seed BA, "
            "fold BA variability, error complementarity, float32 equivalence, "
            "and deployment resources; do not select from official TEST"
        ),
        "selected_by_mean_then_worst_then_fold_sd": selected["variant"],
        "summary": summary_rows,
        "resources": resources,
        "complementarity": complementarity,
    }
    with (output_dir / "phase2e_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")

    print("\n=== Phase 2e summary ===")
    for row in summary_rows:
        print(
            f"{row['variant']:<40} | mean BA="
            f"{100.0 * row['mean_balanced_accuracy']:.2f}% | "
            f"seed SD={100.0 * row['balanced_accuracy_seed_sd']:.2f} pp | "
            f"worst={100.0 * row['worst_seed_balanced_accuracy']:.2f}% | "
            f"fold SD={100.0 * row['fold_balanced_accuracy_sd']:.2f} pp | "
            f"float32={'PASS' if row['float32_all_predictions_exact'] else 'FAIL'}"
        )
    print(f"fusion gate={'PASS' if fusion_gate_passed else 'SKIPPED'}")
    print(f"provisional ranking winner={selected['variant']}")
    print(f"metrics={output_dir / 'phase2e_metrics.json'}")


if __name__ == "__main__":
    main()
