"""Phase 2b: full combination ablation of CSSD stabilization techniques.

This runner crosses the stabilization levels originally selected for study.
Results from the retired UEA conversion are not treated as evidence that any
level is useful; every level is re-evaluated on the corrected official MATLAB
data with its corresponding baseline:

    3 covariance levels x 2 trial levels x 3 F2 levels x 2 fusion levels
    = 36 compatible combinations.

Mutually exclusive choices are represented as levels of one factor rather
than being combined illegally. Every learned quantity is fit inside the
current outer training fold. The official TEST split is refused.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh, subspace_angles
from scipy.signal import butter, sosfiltfilt
from scipy.stats import binomtest
from sklearn.covariance import ledoit_wolf, oas
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
TIMEPOINTS = 50
SAMPLING_RATE_HZ = 100.0
CLASS_COUNTS = {0: 159, 1: 157}
LABEL_NAMES = {0: "left", 1: "right"}
SEEDS = (42, 43, 44)
FOLDS = 5

FILTER_ORDER = 4
BP_LOW_PASS_HZ = 7.0
ERD_BAND_HZ = (10.0, 33.0)
BASELINE_RIDGE = 1e-6

BP_WINDOW = slice(43, 47)
ERD_WINDOW = slice(18, 50)
TREND_START_WINDOW = slice(0, 8)
TREND_END_WINDOW = slice(40, 50)
BP_PATTERNS_PER_CLASS = 1
BASELINE_ERD_PATTERNS_PER_CLASS = 3
ERD_POOL_SAMPLES = 8

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
    family: str = "combination_ablation"
    covariance: str = "empirical"
    covariance_ridge: float = BASELINE_RIDGE
    trial_trace_normalization: bool = False
    erd_patterns_per_class: int = BASELINE_ERD_PATTERNS_PER_CLASS
    fusion: str = "lda"
    description: str = ""


COVARIANCE_LEVELS = ("empirical", "ledoit_wolf", "oas")
TRIAL_LEVELS = (False, True)
F2_LEVELS = (3, 2, 1)
FUSION_LEVELS = ("lda", "soft_vote")


def variant_name(
    covariance: str,
    trial_trace_normalization: bool,
    erd_patterns_per_class: int,
    fusion: str,
) -> str:
    trial = "on" if trial_trace_normalization else "off"
    return (
        f"cov_{covariance}__trial_{trial}__f2_{erd_patterns_per_class}__fusion_{fusion}"
    )


VARIANTS = tuple(
    Variant(
        name=variant_name(covariance, trial, f2_components, fusion),
        covariance=covariance,
        trial_trace_normalization=trial,
        erd_patterns_per_class=f2_components,
        fusion=fusion,
        description=(
            f"covariance={covariance}; trial_trace_normalization={trial}; "
            f"F2 components/class={f2_components}; fusion={fusion}"
        ),
    )
    for covariance in COVARIANCE_LEVELS
    for trial in TRIAL_LEVELS
    for f2_components in F2_LEVELS
    for fusion in FUSION_LEVELS
)
VARIANT_BY_NAME = {variant.name: variant for variant in VARIANTS}
BASELINE_NAME = variant_name("empirical", False, 3, "lda")


def factor_values(variant: Variant) -> dict[str, Any]:
    return {
        "covariance": variant.covariance,
        "trial_trace_normalization": int(variant.trial_trace_normalization),
        "f2_components_per_class": int(variant.erd_patterns_per_class),
        "fusion": variant.fusion,
    }


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
        default=ROOT / "results/finger_movements/phase2b_combination_ablation",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANT_BY_NAME),
        default=list(VARIANT_BY_NAME),
        help="Run selected factorial combinations; the reference is added automatically.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run one outer fold for every selected combination and write nothing.",
    )
    args = parser.parse_args()

    if any("test" in part.lower() for part in args.data.parts):
        parser.error("Phase 2b refuses to load any path identified as TEST")
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain unique values")
    if not 2 <= args.folds <= min(CLASS_COUNTS.values()):
        parser.error("--folds is outside the valid range")
    selected = list(dict.fromkeys(args.variants))
    if BASELINE_NAME not in selected:
        selected.insert(0, BASELINE_NAME)
    args.variants = selected
    return args


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

    if x.shape != (CASES, CHANNELS, TIMEPOINTS):
        raise ValueError(f"Unexpected input shape: {x.shape}")
    if y.shape != (CASES,) or source_index.shape != (CASES,):
        raise ValueError("Unexpected label or source-index shape")
    if channel_names.shape != (CHANNELS,):
        raise ValueError(f"Unexpected channel_names shape: {channel_names.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Input contains non-finite values")
    labels, counts = np.unique(y, return_counts=True)
    observed = dict(zip(labels.tolist(), counts.tolist(), strict=True))
    if observed != CLASS_COUNTS:
        raise ValueError(f"Unexpected labels or class counts: {observed}")
    if not np.array_equal(source_index, np.arange(CASES)):
        raise ValueError("source_index must preserve canonical TRAIN.ts order")
    if len(set(channel_names.tolist())) != CHANNELS:
        raise ValueError("channel_names contains duplicates")
    if not set(REJECTED_TREND_CHANNELS).issubset(set(channel_names.tolist())):
        raise ValueError("One or more paper-rejected trend channels are missing")
    return x, y, source_index, channel_names


def stratified_folds(
    y: np.ndarray, fold_count: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    pieces: dict[int, list[np.ndarray]] = {}
    for label in sorted(np.unique(y).tolist()):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        pieces[label] = list(np.array_split(indices, fold_count))

    all_indices = np.arange(len(y))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_index in range(fold_count):
        validation = np.concatenate(
            [pieces[label][fold_index] for label in sorted(pieces)]
        )
        training = np.setdiff1d(all_indices, validation, assume_unique=True)
        fold_rng = np.random.default_rng(seed * 10_000 + fold_index)
        fold_rng.shuffle(training)
        fold_rng.shuffle(validation)
        if np.intersect1d(training, validation).size:
            raise RuntimeError("Cross-validation fold overlap detected")
        folds.append((training, validation))
    return folds


def temporal_filter(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bp = sosfiltfilt(BP_SOS, x, axis=-1)
    bp = bp - bp[..., :1]
    erd = sosfiltfilt(ERD_SOS, x, axis=-1)
    if not np.isfinite(bp).all() or not np.isfinite(erd).all():
        raise FloatingPointError("Temporal filtering produced non-finite values")
    return bp, erd


def _trace_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    trace = float(np.trace(matrix))
    if not np.isfinite(trace) or trace <= 1e-12:
        raise ValueError("Degenerate spatial matrix")
    return matrix / trace


def class_spatial_matrix(
    class_x: np.ndarray,
    covariance: str,
    ridge: float,
    trial_trace_normalization: bool,
) -> tuple[np.ndarray, float | None]:
    """Estimate one class spatial matrix using training trials only."""
    if class_x.ndim != 3 or class_x.shape[1] != CHANNELS:
        raise ValueError(f"Expected (trials, {CHANNELS}, samples), got {class_x.shape}")
    working_x = class_x
    if trial_trace_normalization:
        moments = np.einsum("nct,ndt->ncd", working_x, working_x, optimize=True)
        traces = np.trace(moments, axis1=1, axis2=2)
        if not np.isfinite(traces).all() or np.any(traces <= 1e-12):
            raise ValueError("Trial normalization received a degenerate trial")
        working_x = working_x / np.sqrt(traces)[:, None, None]

    samples = working_x.transpose(0, 2, 1).reshape(-1, CHANNELS)
    if covariance == "empirical":
        matrix = samples.T @ samples / len(samples)
        shrinkage: float | None = None
    elif covariance == "ledoit_wolf":
        matrix, shrinkage_value = ledoit_wolf(samples, assume_centered=True)
        shrinkage = float(shrinkage_value)
    elif covariance == "oas":
        matrix, shrinkage_value = oas(samples, assume_centered=True)
        shrinkage = float(shrinkage_value)
    else:
        raise ValueError(f"Unknown covariance estimator: {covariance}")

    matrix = _trace_normalize(matrix)
    scale = float(np.trace(matrix) / CHANNELS)
    matrix = (1.0 - ridge) * matrix + ridge * scale * np.eye(CHANNELS)
    return 0.5 * (matrix + matrix.T), shrinkage


def fit_cssd_filters(
    windowed_x: np.ndarray,
    y: np.ndarray,
    patterns_per_class: int,
    variant: Variant,
) -> tuple[np.ndarray, dict[str, Any]]:
    left, left_shrinkage = class_spatial_matrix(
        windowed_x[y == 0],
        variant.covariance,
        variant.covariance_ridge,
        variant.trial_trace_normalization,
    )
    right, right_shrinkage = class_spatial_matrix(
        windowed_x[y == 1],
        variant.covariance,
        variant.covariance_ridge,
        variant.trial_trace_normalization,
    )
    composite = left + right
    eigenvalues, eigenvectors = eigh(left, composite, check_finite=True)
    order = np.argsort(eigenvalues)
    rows: list[np.ndarray] = []
    selected_left: list[float] = []
    selected_right: list[float] = []
    for offset in range(patterns_per_class):
        left_index = int(order[-1 - offset])
        right_index = int(order[offset])
        rows.extend([eigenvectors[:, left_index], eigenvectors[:, right_index]])
        selected_left.append(float(eigenvalues[left_index]))
        selected_right.append(float(eigenvalues[right_index]))
    filters = np.stack(rows)
    for row in filters:
        anchor = int(np.argmax(np.abs(row)))
        if row[anchor] < 0:
            row *= -1.0
    if not np.isfinite(filters).all():
        raise FloatingPointError("CSSD produced non-finite filters")
    return filters, {
        "condition_number": float(np.linalg.cond(composite)),
        "left_shrinkage": left_shrinkage,
        "right_shrinkage": right_shrinkage,
        "left_eigenvalues": selected_left,
        "right_eigenvalues": selected_right,
    }


def project_cssd(x: np.ndarray, filters: np.ndarray) -> np.ndarray:
    return np.einsum("kc,nct->nkt", filters, x, optimize=True)


def bp_cssd_features(bp: np.ndarray, filters: np.ndarray) -> np.ndarray:
    features = project_cssd(bp[..., BP_WINDOW], filters).reshape(len(bp), -1)
    expected = 2 * BP_PATTERNS_PER_CLASS * 4
    if features.shape != (len(bp), expected):
        raise RuntimeError(f"Unexpected F1 shape: {features.shape}")
    return features


def erd_cssd_features(
    erd: np.ndarray, filters: np.ndarray, patterns_per_class: int
) -> np.ndarray:
    projected = project_cssd(erd[..., ERD_WINDOW], filters)
    pooled = (
        np.abs(projected)
        .reshape(
            len(erd),
            projected.shape[1],
            projected.shape[2] // ERD_POOL_SAMPLES,
            ERD_POOL_SAMPLES,
        )
        .mean(axis=-1)
    )
    features = pooled.reshape(len(erd), -1)
    expected = 2 * patterns_per_class * 4
    if features.shape != (len(erd), expected):
        raise RuntimeError(f"Unexpected F2 shape: {features.shape}")
    return features


def trend_features(bp: np.ndarray, channel_names: np.ndarray) -> np.ndarray:
    retained = [
        index
        for index, name in enumerate(channel_names.tolist())
        if name not in REJECTED_TREND_CHANNELS
    ]
    if len(retained) != 19:
        raise RuntimeError(f"Expected 19 retained trend channels, got {len(retained)}")
    selected = bp[:, retained]
    start = selected[..., TREND_START_WINDOW].mean(axis=-1)
    end = selected[..., TREND_END_WINDOW].mean(axis=-1)
    features = np.stack([start, end], axis=-1).reshape(len(bp), -1)
    if features.shape != (len(bp), 38):
        raise RuntimeError(f"Unexpected F3 shape: {features.shape}")
    return features


def make_lda() -> Any:
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="svd"))


def fit_branch(
    training_features: np.ndarray,
    validation_features: np.ndarray,
    training_y: np.ndarray,
) -> dict[str, np.ndarray]:
    model = make_lda()
    model.fit(training_features, training_y)
    return {
        "training_score": np.asarray(
            model.decision_function(training_features)
        ).reshape(-1),
        "validation_score": np.asarray(
            model.decision_function(validation_features)
        ).reshape(-1),
        "validation_probability": model.predict_proba(validation_features)[:, 1],
        "validation_prediction": model.predict(validation_features).astype(np.int64),
    }


def combine_branches(
    branches: list[dict[str, np.ndarray]], training_y: np.ndarray, fusion: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    training_scores = np.column_stack([branch["training_score"] for branch in branches])
    validation_scores = np.column_stack(
        [branch["validation_score"] for branch in branches]
    )
    if fusion == "lda":
        model = make_lda()
        model.fit(training_scores, training_y)
        prediction = model.predict(validation_scores).astype(np.int64)
        probability = model.predict_proba(validation_scores)[:, 1]
        score = np.asarray(model.decision_function(validation_scores)).reshape(-1)
    elif fusion == "hard_vote":
        votes = np.column_stack(
            [branch["validation_prediction"] for branch in branches]
        )
        probability = votes.mean(axis=1)
        prediction = (votes.sum(axis=1) >= 2).astype(np.int64)
        score = votes.sum(axis=1).astype(np.float64) - 1.5
    elif fusion == "soft_vote":
        probabilities = np.column_stack(
            [branch["validation_probability"] for branch in branches]
        )
        probability = probabilities.mean(axis=1)
        prediction = (probability >= 0.5).astype(np.int64)
        clipped = np.clip(probability, 1e-8, 1.0 - 1e-8)
        score = np.log(clipped / (1.0 - clipped))
    else:
        raise ValueError(f"Unknown fusion: {fusion}")
    return prediction, probability, score


def metric_bundle(
    y_true: np.ndarray, prediction: np.ndarray, probability: np.ndarray
) -> dict[str, Any]:
    clipped = np.clip(probability, 1e-8, 1.0 - 1e-8)
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro")),
        "mean_log_loss": float(
            log_loss(y_true, np.column_stack([1.0 - clipped, clipped]), labels=[0, 1])
        ),
        "confusion_matrix": confusion_matrix(
            y_true, prediction, labels=[0, 1]
        ).tolist(),
    }


def run_fold(
    bp: np.ndarray,
    erd: np.ndarray,
    y: np.ndarray,
    channel_names: np.ndarray,
    training_indices: np.ndarray,
    validation_indices: np.ndarray,
    variant: Variant,
) -> dict[str, Any]:
    training_y = y[training_indices]
    validation_y = y[validation_indices]
    training_bp, validation_bp = bp[training_indices], bp[validation_indices]
    training_erd, validation_erd = erd[training_indices], erd[validation_indices]

    bp_filters, bp_diagnostics = fit_cssd_filters(
        training_bp[..., BP_WINDOW], training_y, BP_PATTERNS_PER_CLASS, variant
    )
    erd_filters, erd_diagnostics = fit_cssd_filters(
        training_erd[..., ERD_WINDOW],
        training_y,
        variant.erd_patterns_per_class,
        variant,
    )

    branch_features = (
        (
            bp_cssd_features(training_bp, bp_filters),
            bp_cssd_features(validation_bp, bp_filters),
        ),
        (
            erd_cssd_features(
                training_erd, erd_filters, variant.erd_patterns_per_class
            ),
            erd_cssd_features(
                validation_erd, erd_filters, variant.erd_patterns_per_class
            ),
        ),
        (
            trend_features(training_bp, channel_names),
            trend_features(validation_bp, channel_names),
        ),
    )
    branches = [
        fit_branch(training, validation, training_y)
        for training, validation in branch_features
    ]
    prediction, probability, score = combine_branches(
        branches, training_y, variant.fusion
    )
    if prediction.shape != validation_y.shape or not np.isfinite(probability).all():
        raise RuntimeError("Invalid Phase 2b fold output")
    return {
        "prediction": prediction,
        "probability": probability,
        "score": score,
        "metrics": metric_bundle(validation_y, prediction, probability),
        "bp_filters": bp_filters,
        "erd_filters": erd_filters,
        "bp_diagnostics": bp_diagnostics,
        "erd_diagnostics": erd_diagnostics,
        "feature_dimensions": [
            int(branch_features[0][0].shape[1]),
            int(branch_features[1][0].shape[1]),
            int(branch_features[2][0].shape[1]),
        ],
    }


def subspace_stability(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    angles = subspace_angles(first.T, second.T)
    cosines = np.abs(np.cos(angles))
    return {
        "mean_abs_cosine": float(cosines.mean()),
        "minimum_abs_cosine": float(cosines.min()),
        "mean_principal_angle_deg": float(np.degrees(angles).mean()),
        "maximum_principal_angle_deg": float(np.degrees(angles).max()),
    }


def filter_stability_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant_name in sorted({record["variant"] for record in records}):
        for seed in sorted(
            {record["seed"] for record in records if record["variant"] == variant_name}
        ):
            selected = [
                record
                for record in records
                if record["variant"] == variant_name and record["seed"] == seed
            ]
            for first, second in combinations(selected, 2):
                definitions = (
                    ("bp", "left", first["bp_filters"][[0]], second["bp_filters"][[0]]),
                    (
                        "bp",
                        "right",
                        first["bp_filters"][[1]],
                        second["bp_filters"][[1]],
                    ),
                    (
                        "erd",
                        "left",
                        first["erd_filters"][0::2],
                        second["erd_filters"][0::2],
                    ),
                    (
                        "erd",
                        "right",
                        first["erd_filters"][1::2],
                        second["erd_filters"][1::2],
                    ),
                )
                for branch, side, first_filters, second_filters in definitions:
                    rows.append(
                        {
                            "variant": variant_name,
                            "seed": int(seed),
                            "fold_a": int(first["fold"]),
                            "fold_b": int(second["fold"]),
                            "branch": branch,
                            "side": side,
                            **subspace_stability(first_filters, second_filters),
                        }
                    )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


FACTOR_SPECS: tuple[tuple[str, tuple[Any, ...], Any], ...] = (
    ("covariance", COVARIANCE_LEVELS, "empirical"),
    ("trial_trace_normalization", (0, 1), 0),
    ("f2_components_per_class", F2_LEVELS, 3),
    ("fusion", FUSION_LEVELS, "lda"),
)
FACTOR_FIELDS = tuple(spec[0] for spec in FACTOR_SPECS)


def factorial_effects(
    seed_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return matched main effects and pairwise difference-in-differences."""
    lookup = {
        (row["seed"], *(row[field] for field in FACTOR_FIELDS)): row
        for row in seed_rows
    }
    if len(lookup) != len(seed_rows):
        raise RuntimeError("Factorial seed-result keys are not unique")

    main_effect_rows: list[dict[str, Any]] = []
    for factor, levels, reference in FACTOR_SPECS:
        factor_index = FACTOR_FIELDS.index(factor)
        for level in levels:
            if level == reference:
                continue
            deltas: list[float] = []
            for row in seed_rows:
                if row[factor] != level:
                    continue
                reference_levels = [row[field] for field in FACTOR_FIELDS]
                reference_levels[factor_index] = reference
                reference_row = lookup[(row["seed"], *reference_levels)]
                deltas.append(
                    row["balanced_accuracy"] - reference_row["balanced_accuracy"]
                )
            values = np.asarray(deltas, dtype=np.float64)
            main_effect_rows.append(
                {
                    "factor": factor,
                    "level": level,
                    "reference_level": reference,
                    "matched_contexts": len(values),
                    "mean_paired_delta_vs_reference_level": float(values.mean()),
                    "paired_delta_sd": float(values.std(ddof=0)),
                    "minimum_paired_delta": float(values.min()),
                    "maximum_paired_delta": float(values.max()),
                    "positive_context_count": int(np.sum(values > 0.0)),
                    "tied_context_count": int(np.sum(np.isclose(values, 0.0))),
                }
            )

    interaction_rows: list[dict[str, Any]] = []
    for first_index, second_index in combinations(range(len(FACTOR_SPECS)), 2):
        first_factor, first_levels, first_reference = FACTOR_SPECS[first_index]
        second_factor, second_levels, second_reference = FACTOR_SPECS[second_index]
        for first_level in first_levels:
            if first_level == first_reference:
                continue
            for second_level in second_levels:
                if second_level == second_reference:
                    continue
                interactions: list[float] = []
                for row in seed_rows:
                    if (
                        row[first_factor] != first_reference
                        or row[second_factor] != second_reference
                    ):
                        continue
                    base_levels = [row[field] for field in FACTOR_FIELDS]
                    first_only = list(base_levels)
                    second_only = list(base_levels)
                    both = list(base_levels)
                    first_only[first_index] = first_level
                    second_only[second_index] = second_level
                    both[first_index] = first_level
                    both[second_index] = second_level
                    y00 = row["balanced_accuracy"]
                    y10 = lookup[(row["seed"], *first_only)]["balanced_accuracy"]
                    y01 = lookup[(row["seed"], *second_only)]["balanced_accuracy"]
                    y11 = lookup[(row["seed"], *both)]["balanced_accuracy"]
                    interactions.append((y11 - y10) - (y01 - y00))
                values = np.asarray(interactions, dtype=np.float64)
                interaction_rows.append(
                    {
                        "factor_a": first_factor,
                        "level_a": first_level,
                        "reference_a": first_reference,
                        "factor_b": second_factor,
                        "level_b": second_level,
                        "reference_b": second_reference,
                        "matched_contexts": len(values),
                        "mean_difference_in_differences": float(values.mean()),
                        "interaction_sd": float(values.std(ddof=0)),
                        "minimum_interaction": float(values.min()),
                        "maximum_interaction": float(values.max()),
                        "positive_context_count": int(np.sum(values > 0.0)),
                        "tied_context_count": int(np.sum(np.isclose(values, 0.0))),
                    }
                )
    return main_effect_rows, interaction_rows


def create_figure(
    summary: list[dict[str, Any]],
    main_effects: list[dict[str, Any]],
    output_path: Path,
) -> None:
    ordered = sorted(
        summary, key=lambda row: row["mean_balanced_accuracy"], reverse=True
    )[:12]
    labels = [row["variant"] for row in ordered]
    values = 100.0 * np.array([row["mean_balanced_accuracy"] for row in ordered])
    errors = 100.0 * np.array([row["balanced_accuracy_seed_sd"] for row in ordered])
    colors = ["#4c72b0" if name == BASELINE_NAME else "#55a868" for name in labels]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    positions = np.arange(len(labels))
    axes[0].barh(positions, values, xerr=errors, color=colors, capsize=3)
    axes[0].set_yticks(positions, labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Mean OOF balanced accuracy (%) ± seed SD")
    axes[0].set_title("Top 12 combinations")
    for position, value, error in zip(positions, values, errors, strict=True):
        axes[0].text(
            value + error + 0.35,
            position,
            f"{value:.2f}%",
            va="center",
            fontsize=9,
        )

    effect_labels = [f"{row['factor']}={row['level']}" for row in main_effects]
    effect_values = 100.0 * np.array(
        [row["mean_paired_delta_vs_reference_level"] for row in main_effects]
    )
    effect_colors = ["#55a868" if value >= 0 else "#c44e52" for value in effect_values]
    effect_positions = np.arange(len(main_effects))
    axes[1].barh(effect_positions, effect_values, color=effect_colors)
    axes[1].axvline(0.0, color="black", linewidth=1)
    axes[1].set_yticks(effect_positions, effect_labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean paired BA effect across matched contexts (pp)")
    axes[1].set_title("Factor main effects")
    for position, value in zip(effect_positions, effect_values, strict=True):
        axes[1].text(
            value + 0.05,
            position,
            f"{value:+.2f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    fig.suptitle("FingerMovements Phase 2b: combination ablation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    x, y, source_index, channel_names = load_training_data(args.data)
    bp, erd = temporal_filter(x)
    variants = [VARIANT_BY_NAME[name] for name in args.variants]

    print("=== FingerMovements Phase 2b: CSSD combination ablation ===")
    print(f"data={args.data}")
    print(f"cases={len(y)} | input={CHANNELS}x{TIMEPOINTS} @ {SAMPLING_RATE_HZ:g} Hz")
    print(f"seeds={args.seeds} | folds={args.folds} | variants={len(variants)}")
    print("policy=TRAIN only; TEST refused; every learned operation is fold-local")
    print("factorial=3 covariance x 2 trial x 3 F2 x 2 fusion = 36 combinations")

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    filter_records: list[dict[str, Any]] = []
    seeds_to_run = args.seeds[:1] if args.validate_only else args.seeds

    for seed in seeds_to_run:
        folds = stratified_folds(y, args.folds, seed)
        if args.validate_only:
            folds = folds[:1]
        for fold_number, (training_indices, validation_indices) in enumerate(
            folds, start=1
        ):
            print(f"\nseed {seed} | fold {fold_number}/{args.folds}")
            for variant in variants:
                started = perf_counter()
                result = run_fold(
                    bp,
                    erd,
                    y,
                    channel_names,
                    training_indices,
                    validation_indices,
                    variant,
                )
                elapsed = perf_counter() - started
                metrics = result["metrics"]
                fold_rows.append(
                    {
                        "variant": variant.name,
                        "family": variant.family,
                        **factor_values(variant),
                        "seed": int(seed),
                        "fold": int(fold_number),
                        "training_cases": len(training_indices),
                        "validation_cases": len(validation_indices),
                        "accuracy": metrics["accuracy"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "mean_log_loss": metrics["mean_log_loss"],
                        "bp_condition_number": result["bp_diagnostics"][
                            "condition_number"
                        ],
                        "erd_condition_number": result["erd_diagnostics"][
                            "condition_number"
                        ],
                        "bp_left_shrinkage": result["bp_diagnostics"]["left_shrinkage"],
                        "bp_right_shrinkage": result["bp_diagnostics"][
                            "right_shrinkage"
                        ],
                        "erd_left_shrinkage": result["erd_diagnostics"][
                            "left_shrinkage"
                        ],
                        "erd_right_shrinkage": result["erd_diagnostics"][
                            "right_shrinkage"
                        ],
                        "f1_features": result["feature_dimensions"][0],
                        "f2_features": result["feature_dimensions"][1],
                        "f3_features": result["feature_dimensions"][2],
                        "elapsed_seconds": float(elapsed),
                    }
                )
                for local_index, global_index in enumerate(validation_indices):
                    prediction_rows.append(
                        {
                            "variant": variant.name,
                            "family": variant.family,
                            **factor_values(variant),
                            "seed": int(seed),
                            "fold": int(fold_number),
                            "source_index": int(source_index[global_index]),
                            "true_label": int(y[global_index]),
                            "predicted_label": int(result["prediction"][local_index]),
                            "probability_right": float(
                                result["probability"][local_index]
                            ),
                            "score": float(result["score"][local_index]),
                        }
                    )
                filter_records.append(
                    {
                        "variant": variant.name,
                        "seed": int(seed),
                        "fold": int(fold_number),
                        "bp_filters": result["bp_filters"],
                        "erd_filters": result["erd_filters"],
                    }
                )
                print(
                    f"  {variant.name:<26} | "
                    f"BA={100.0 * metrics['balanced_accuracy']:.2f}% | "
                    f"F2={result['feature_dimensions'][1]:02d} | "
                    f"cond BP/ERD={result['bp_diagnostics']['condition_number']:.1f}/"
                    f"{result['erd_diagnostics']['condition_number']:.1f} | "
                    f"{elapsed:.2f}s"
                )

    if args.validate_only:
        print("\nvalidate-only complete; all variants ran one fold; no files written")
        return

    seed_rows: list[dict[str, Any]] = []
    for variant in variants:
        for seed in args.seeds:
            selected = [
                row
                for row in prediction_rows
                if row["variant"] == variant.name and row["seed"] == seed
            ]
            selected.sort(key=lambda row: row["source_index"])
            if len(selected) != CASES:
                raise RuntimeError(
                    f"{variant.name} seed {seed} has {len(selected)} OOF cases, expected {CASES}"
                )
            indices = np.array([row["source_index"] for row in selected])
            if not np.array_equal(indices, np.arange(CASES)):
                raise RuntimeError(
                    f"{variant.name} seed {seed} OOF coverage is invalid"
                )
            labels = np.array([row["true_label"] for row in selected], dtype=np.int64)
            predictions = np.array(
                [row["predicted_label"] for row in selected], dtype=np.int64
            )
            probabilities = np.array(
                [row["probability_right"] for row in selected], dtype=np.float64
            )
            seed_rows.append(
                {
                    "variant": variant.name,
                    "family": variant.family,
                    **factor_values(variant),
                    "seed": int(seed),
                    "evaluated_cases": CASES,
                    **metric_bundle(labels, predictions, probabilities),
                }
            )

    summary_rows: list[dict[str, Any]] = []
    baseline_seed = {
        row["seed"]: row for row in seed_rows if row["variant"] == BASELINE_NAME
    }
    for variant in variants:
        selected = [row for row in seed_rows if row["variant"] == variant.name]
        balanced = np.array([row["balanced_accuracy"] for row in selected])
        deltas = np.array(
            [
                row["balanced_accuracy"]
                - baseline_seed[row["seed"]]["balanced_accuracy"]
                for row in selected
            ]
        )
        summary_rows.append(
            {
                "variant": variant.name,
                "family": variant.family,
                **factor_values(variant),
                "mean_accuracy": float(np.mean([row["accuracy"] for row in selected])),
                "mean_balanced_accuracy": float(balanced.mean()),
                "balanced_accuracy_seed_sd": float(balanced.std(ddof=0)),
                "worst_seed_balanced_accuracy": float(balanced.min()),
                "mean_macro_f1": float(np.mean([row["macro_f1"] for row in selected])),
                "mean_ba_delta_vs_baseline": float(deltas.mean()),
                "improved_seed_count": int(np.sum(deltas > 0.0)),
                "tied_seed_count": int(np.sum(np.isclose(deltas, 0.0))),
            }
        )

    complete_factorial = set(args.variants) == set(VARIANT_BY_NAME)
    if complete_factorial:
        main_effect_rows, interaction_rows = factorial_effects(seed_rows)
    else:
        main_effect_rows, interaction_rows = [], []

    paired_rows: list[dict[str, Any]] = []
    for variant in variants:
        if variant.name == BASELINE_NAME:
            continue
        for seed in args.seeds:
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
                    if row["variant"] == variant.name and row["seed"] == seed
                ),
                key=lambda row: row["source_index"],
            )
            base_correct = np.array(
                [row["predicted_label"] == row["true_label"] for row in baseline]
            )
            candidate_correct = np.array(
                [row["predicted_label"] == row["true_label"] for row in candidate]
            )
            gained = int(np.sum(~base_correct & candidate_correct))
            lost = int(np.sum(base_correct & ~candidate_correct))
            discordant = gained + lost
            p_value = (
                float(binomtest(min(gained, lost), discordant, 0.5).pvalue)
                if discordant
                else 1.0
            )
            paired_rows.append(
                {
                    "variant": variant.name,
                    "family": variant.family,
                    **factor_values(variant),
                    "seed": int(seed),
                    "baseline_wrong_candidate_right": gained,
                    "baseline_right_candidate_wrong": lost,
                    "discordant_cases": discordant,
                    "exact_mcnemar_p": p_value,
                }
            )

    stability = filter_stability_rows(filter_records)
    stability_summary: list[dict[str, Any]] = []
    for variant in variants:
        for branch in ("bp", "erd"):
            for side in ("left", "right"):
                selected = [
                    row
                    for row in stability
                    if row["variant"] == variant.name
                    and row["branch"] == branch
                    and row["side"] == side
                ]
                stability_summary.append(
                    {
                        "variant": variant.name,
                        "family": variant.family,
                        **factor_values(variant),
                        "branch": branch,
                        "side": side,
                        "comparisons": len(selected),
                        "mean_abs_cosine": float(
                            np.mean([row["mean_abs_cosine"] for row in selected])
                        ),
                        "minimum_abs_cosine": float(
                            np.min([row["minimum_abs_cosine"] for row in selected])
                        ),
                        "mean_principal_angle_deg": float(
                            np.mean(
                                [row["mean_principal_angle_deg"] for row in selected]
                            )
                        ),
                        "maximum_principal_angle_deg": float(
                            np.max(
                                [row["maximum_principal_angle_deg"] for row in selected]
                            )
                        ),
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "phase2b_combination_fold_results.csv",
        fold_rows,
        list(fold_rows[0]),
    )
    serializable_seed_rows: list[dict[str, Any]] = []
    for row in seed_rows:
        serializable = dict(row)
        serializable["confusion_matrix"] = json.dumps(row["confusion_matrix"])
        serializable_seed_rows.append(serializable)
    write_csv(
        args.output_dir / "phase2b_combination_seed_results.csv",
        serializable_seed_rows,
        list(serializable_seed_rows[0]),
    )
    write_csv(
        args.output_dir / "phase2b_combination_summary.csv",
        summary_rows,
        list(summary_rows[0]),
    )
    write_csv(
        args.output_dir / "phase2b_combination_paired_comparisons.csv",
        paired_rows,
        [
            "variant",
            "family",
            "covariance",
            "trial_trace_normalization",
            "f2_components_per_class",
            "fusion",
            "seed",
            "baseline_wrong_candidate_right",
            "baseline_right_candidate_wrong",
            "discordant_cases",
            "exact_mcnemar_p",
        ],
    )
    write_csv(
        args.output_dir / "phase2b_combination_predictions.csv",
        prediction_rows,
        list(prediction_rows[0]),
    )
    write_csv(
        args.output_dir / "phase2b_combination_filter_stability.csv",
        stability,
        list(stability[0]),
    )
    write_csv(
        args.output_dir / "phase2b_combination_filter_stability_summary.csv",
        stability_summary,
        list(stability_summary[0]),
    )
    write_csv(
        args.output_dir / "phase2b_combination_main_effects.csv",
        main_effect_rows,
        [
            "factor",
            "level",
            "reference_level",
            "matched_contexts",
            "mean_paired_delta_vs_reference_level",
            "paired_delta_sd",
            "minimum_paired_delta",
            "maximum_paired_delta",
            "positive_context_count",
            "tied_context_count",
        ],
    )
    write_csv(
        args.output_dir / "phase2b_combination_pairwise_interactions.csv",
        interaction_rows,
        [
            "factor_a",
            "level_a",
            "reference_a",
            "factor_b",
            "level_b",
            "reference_b",
            "matched_contexts",
            "mean_difference_in_differences",
            "interaction_sd",
            "minimum_interaction",
            "maximum_interaction",
            "positive_context_count",
            "tied_context_count",
        ],
    )

    payload = {
        "phase": "2b",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "full combination ablation of useful Phase 2b settings on official TRAIN only",
        "data": {
            "path": str(args.data),
            "cases": CASES,
            "shape": [CASES, CHANNELS, TIMEPOINTS],
            "test_policy": "official TEST refused and not loaded",
        },
        "validation": {
            "seeds": [int(seed) for seed in args.seeds],
            "folds": int(args.folds),
            "split": "repeated stratified cross-validation",
            "paired_cases_per_seed": CASES,
        },
        "factorial_design": {
            "complete": complete_factorial,
            "covariance_levels": list(COVARIANCE_LEVELS),
            "trial_trace_normalization_levels": list(TRIAL_LEVELS),
            "f2_components_per_class_levels": list(F2_LEVELS),
            "fusion_levels": list(FUSION_LEVELS),
            "combination_count": len(variants),
            "mutually_exclusive_levels_are_not_illegally_combined": True,
        },
        "variants": [asdict(variant) for variant in variants],
        "reference_policy": (
            "all deltas use the baseline recomputed on the same official-MATLAB "
            "data and paired folds; metrics from the retired UEA conversion are "
            "not comparable"
        ),
        "seed_results": seed_rows,
        "summary": summary_rows,
        "paired_comparisons": paired_rows,
        "main_effects": main_effect_rows,
        "pairwise_interactions": interaction_rows,
        "filter_stability_summary": stability_summary,
    }
    with (args.output_dir / "phase2b_combination_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    create_figure(
        summary_rows,
        main_effect_rows,
        args.output_dir / "phase2b_combination_ablation.png",
    )

    print("\n=== Phase 2b combination summary (top 10) ===")
    for row in sorted(
        summary_rows, key=lambda item: item["mean_balanced_accuracy"], reverse=True
    )[:10]:
        print(
            f"{row['variant']} | "
            f"BA={100.0 * row['mean_balanced_accuracy']:.2f}% "
            f"±{100.0 * row['balanced_accuracy_seed_sd']:.2f}pp | "
            f"worst={100.0 * row['worst_seed_balanced_accuracy']:.2f}% | "
            f"delta={100.0 * row['mean_ba_delta_vs_baseline']:+.2f}pp | "
            f"improved seeds={row['improved_seed_count']}/{len(args.seeds)}"
        )
    if main_effect_rows:
        print("\n=== Matched factor main effects ===")
        for row in main_effect_rows:
            print(
                f"{row['factor']}={row['level']} vs {row['reference_level']} | "
                f"delta={100.0 * row['mean_paired_delta_vs_reference_level']:+.2f}pp | "
                f"positive contexts={row['positive_context_count']}/{row['matched_contexts']}"
            )
    print(f"metrics={args.output_dir / 'phase2b_combination_metrics.json'}")
    print(f"figure={args.output_dir / 'phase2b_combination_ablation.png'}")


if __name__ == "__main__":
    main()
