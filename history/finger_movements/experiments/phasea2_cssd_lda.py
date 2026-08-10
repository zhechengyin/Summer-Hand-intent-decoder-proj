"""Phase A2: paper-style CSSD + hierarchical LDA on FingerMovements.

This experiment follows the three-branch representation in Wang et al. (2004):

1. Bereitschaftspotential (BP): zero-phase 0--7 Hz, points 44--47,
   one left-specific and one right-specific CSSD spatial filter;
2. event-related desynchronization (ERD): zero-phase 10--33 Hz, points 19--50,
   three left/right CSSD filter pairs and eight-sample absolute pooling;
3. BP trend: zero-phase 0--7 Hz, points 1--8 and 41--50 on the 19 retained
   electrodes from the paper.

Each branch is projected to one Fisher/LDA score. A final LDA combines the
three scores. The paper used a perceptron for this final step; Phase A2 uses
LDA as explicitly requested. All CSSD filters, scalers, and LDA models are
fitted from the current training fold only. The official TEST split is never
loaded.

The published article does not specify its temporal-filter family/order or a
numerical regularizer for simultaneous diagonalization. This implementation
freezes fourth-order Butterworth filters and a small 1e-6 covariance ridge.
It is therefore a transparent paper-style reproduction, not a claim of
bit-exact reproduction of the authors' unavailable original code.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh, subspace_angles
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# This experiment is archived three directories below the repository root:
# history/finger_movements/experiments/phasea2_cssd_lda.py.
ROOT = Path(__file__).resolve().parents[3]

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
CSSD_RIDGE = 1e-6

# Paper sample indices are one-based and inclusive.
BP_WINDOW = slice(43, 47)  # points 44--47, four samples
ERD_WINDOW = slice(18, 50)  # points 19--50, 32 samples
TREND_START_WINDOW = slice(0, 8)  # points 1--8
TREND_END_WINDOW = slice(40, 50)  # points 41--50

BP_PATTERNS_PER_CLASS = 1
ERD_PATTERNS_PER_CLASS = 3
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

ARCHIVED_BASELINE_OOF_BA = 0.6888528355299176
ARTICLE_URL = "https://doi.org/10.1109/TBME.2004.826697"

BRANCH_NAMES = ("f1_bp_cssd", "f2_erd_cssd", "f3_bp_trend")
BRANCH_SUBSETS = (
    ("f1", (0,)),
    ("f2", (1,)),
    ("f3", (2,)),
    ("f1_f2", (0, 1)),
    ("f1_f3", (0, 2)),
    ("f2_f3", (1, 2)),
    ("f1_f2_f3", (0, 1, 2)),
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
        default=ROOT / "results/finger_movements/phasea2_cssd_lda",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run one fold, print checks, and write no result files.",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help=(
            "Add layer train/validation gaps, CSSD filter stability, all branch "
            "ablations, and inner-OOF fusion diagnostics."
        ),
    )
    args = parser.parse_args()

    if "test" in args.data.name.lower():
        parser.error("Phase A2 refuses to load any file identified as TEST")
    if len(args.seeds) == 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain unique values")
    if not 2 <= args.folds <= min(CLASS_COUNTS.values()):
        parser.error("--folds is outside the valid range")
    if args.validate_only and args.diagnostics:
        parser.error("--validate-only and --diagnostics cannot be combined")
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
    values, counts = np.unique(y, return_counts=True)
    observed = dict(zip(values.tolist(), counts.tolist(), strict=True))
    if observed != CLASS_COUNTS:
        raise ValueError(f"Unexpected labels or class counts: {observed}")
    if not np.array_equal(source_index, np.arange(CASES)):
        raise ValueError("source_index must preserve canonical TRAIN.ts order")
    if len(set(channel_names.tolist())) != CHANNELS:
        raise ValueError("channel_names contains duplicates")
    missing_rejected = sorted(set(REJECTED_TREND_CHANNELS) - set(channel_names))
    if missing_rejected:
        raise ValueError(f"Paper-rejected channels are missing: {missing_rejected}")
    return x, y, source_index, channel_names


def stratified_folds(
    y: np.ndarray, fold_count: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    class_pieces: dict[int, list[np.ndarray]] = {}
    for label in sorted(np.unique(y).tolist()):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        class_pieces[label] = list(np.array_split(indices, fold_count))

    all_indices = np.arange(len(y))
    output: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(fold_count):
        validation = np.concatenate(
            [class_pieces[label][fold] for label in sorted(class_pieces)]
        )
        training = np.setdiff1d(all_indices, validation, assume_unique=True)
        fold_rng = np.random.default_rng(seed * 10_000 + fold)
        fold_rng.shuffle(training)
        fold_rng.shuffle(validation)
        if np.intersect1d(training, validation).size:
            raise RuntimeError("Cross-validation fold overlap detected")
        output.append((training, validation))
    return output


def temporal_filter(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return paper-style zero-phase BP and ERD signals."""
    bp = sosfiltfilt(BP_SOS, x, axis=-1)
    # The paper normalizes the beginning of each epoch to zero for BP.
    bp = bp - bp[..., :1]
    erd = sosfiltfilt(ERD_SOS, x, axis=-1)
    if not np.isfinite(bp).all() or not np.isfinite(erd).all():
        raise FloatingPointError("Temporal filtering produced non-finite values")
    return bp, erd


def _class_spatial_second_moment(x: np.ndarray) -> np.ndarray:
    """Estimate a unit-trace channel second-moment matrix."""
    if x.ndim != 3 or x.shape[1] != CHANNELS:
        raise ValueError(f"CSSD expects (cases, {CHANNELS}, samples), got {x.shape}")
    samples = x.transpose(1, 0, 2).reshape(CHANNELS, -1)
    moment = samples @ samples.T / samples.shape[1]
    moment = 0.5 * (moment + moment.T)
    trace = float(np.trace(moment))
    if not np.isfinite(trace) or trace <= 1e-12:
        raise ValueError("CSSD received a degenerate class second moment")
    moment /= trace
    scale = float(np.trace(moment) / CHANNELS)
    moment = (1.0 - CSSD_RIDGE) * moment + CSSD_RIDGE * scale * np.eye(CHANNELS)
    return 0.5 * (moment + moment.T)


def fit_cssd_filters(
    windowed_x: np.ndarray,
    y: np.ndarray,
    patterns_per_class: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit left/right filters by simultaneous covariance diagonalization.

    Generalized eigendecomposition of R_left against R_left + R_right is the
    two-condition simultaneous-diagonalization step underlying CSSD/CSP. Large
    eigenvalues are left-specific; small eigenvalues are right-specific.
    """
    left_moment = _class_spatial_second_moment(windowed_x[y == 0])
    right_moment = _class_spatial_second_moment(windowed_x[y == 1])
    composite = left_moment + right_moment
    eigenvalues, eigenvectors = eigh(left_moment, composite, check_finite=True)
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

    filters = np.stack(rows, axis=0)
    # Resolve eigenvector sign ambiguity for reproducible artifacts.
    for row in filters:
        anchor = int(np.argmax(np.abs(row)))
        if row[anchor] < 0:
            row *= -1.0

    projected_composite = filters @ composite @ filters.T
    if not np.isfinite(filters).all():
        raise FloatingPointError("CSSD produced non-finite filters")
    diagnostics = {
        "left_specific_eigenvalues": selected_left,
        "right_specific_eigenvalues": selected_right,
        "composite_condition_number": float(np.linalg.cond(composite)),
        "selected_filter_orthogonality_error": float(
            np.max(np.abs(projected_composite - np.eye(len(filters))))
        ),
    }
    return filters, diagnostics


def project_cssd(x: np.ndarray, filters: np.ndarray) -> np.ndarray:
    return np.einsum("kc,nct->nkt", filters, x, optimize=True)


def bp_cssd_features(bp: np.ndarray, filters: np.ndarray) -> np.ndarray:
    projected = project_cssd(bp[..., BP_WINDOW], filters)
    output = projected.reshape(len(bp), -1)
    expected = 2 * BP_PATTERNS_PER_CLASS * 4
    if output.shape != (len(bp), expected):
        raise RuntimeError(f"Unexpected BP feature shape: {output.shape}")
    return output


def erd_cssd_features(erd: np.ndarray, filters: np.ndarray) -> np.ndarray:
    projected = project_cssd(erd[..., ERD_WINDOW], filters)
    if projected.shape[-1] % ERD_POOL_SAMPLES:
        raise RuntimeError("ERD window is not divisible by the pooling length")
    pooled = np.abs(projected).reshape(
        len(erd),
        projected.shape[1],
        projected.shape[2] // ERD_POOL_SAMPLES,
        ERD_POOL_SAMPLES,
    ).mean(axis=-1)
    output = pooled.reshape(len(erd), -1)
    expected = 2 * ERD_PATTERNS_PER_CLASS * 4
    if output.shape != (len(erd), expected):
        raise RuntimeError(f"Unexpected ERD feature shape: {output.shape}")
    return output


def trend_features(
    bp: np.ndarray, channel_names: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    retained = [
        index
        for index, name in enumerate(channel_names.tolist())
        if name not in REJECTED_TREND_CHANNELS
    ]
    retained_names = channel_names[retained].tolist()
    if len(retained) != 19:
        raise RuntimeError(f"Expected 19 retained trend channels, got {len(retained)}")
    selected = bp[:, retained, :]
    k1 = selected[..., TREND_START_WINDOW].mean(axis=-1)
    k2 = selected[..., TREND_END_WINDOW].mean(axis=-1)
    # Paper order: [k11, k12, ..., ki1, ki2] for i=1..19.
    output = np.stack([k1, k2], axis=-1).reshape(len(bp), -1)
    if output.shape != (len(bp), 38):
        raise RuntimeError(f"Unexpected trend feature shape: {output.shape}")
    return output, retained_names


def make_fisher_pipeline() -> Any:
    return make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="svd"),
    )


def fit_branch_fisher(
    training_features: np.ndarray,
    validation_features: np.ndarray,
    training_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Any]:
    model = make_fisher_pipeline()
    model.fit(training_features, training_y)
    training_score = np.asarray(model.decision_function(training_features)).reshape(-1)
    validation_score = np.asarray(model.decision_function(validation_features)).reshape(-1)
    return training_score, validation_score, model


def metric_bundle(
    y_true: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro")),
        "mean_log_loss": float(
            log_loss(y_true, np.column_stack([1.0 - probability, probability]), labels=[0, 1])
        ),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=[0, 1]).tolist(),
    }


def score_metric_bundle(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    prediction = (score >= 0.0).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro")),
        "roc_auc": float(roc_auc_score(y_true, score)),
    }


def fit_fusion_lda(
    training_scores: np.ndarray,
    validation_scores: np.ndarray,
    training_y: np.ndarray,
    columns: tuple[int, ...],
) -> dict[str, np.ndarray]:
    model = make_fisher_pipeline()
    model.fit(training_scores[:, columns], training_y)
    prediction = model.predict(validation_scores[:, columns]).astype(np.int64)
    probability = model.predict_proba(validation_scores[:, columns])[:, 1]
    score = np.asarray(
        model.decision_function(validation_scores[:, columns])
    ).reshape(-1)
    return {
        "prediction": prediction,
        "probability": probability,
        "score": score,
    }


def run_fold(
    x: np.ndarray,
    y: np.ndarray,
    channel_names: np.ndarray,
    training_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> dict[str, Any]:
    training_x = x[training_indices]
    validation_x = x[validation_indices]
    training_y = y[training_indices]
    validation_y = y[validation_indices]

    training_bp, training_erd = temporal_filter(training_x)
    validation_bp, validation_erd = temporal_filter(validation_x)

    bp_filters, bp_diagnostics = fit_cssd_filters(
        training_bp[..., BP_WINDOW], training_y, BP_PATTERNS_PER_CLASS
    )
    erd_filters, erd_diagnostics = fit_cssd_filters(
        training_erd[..., ERD_WINDOW], training_y, ERD_PATTERNS_PER_CLASS
    )

    training_f1_raw = bp_cssd_features(training_bp, bp_filters)
    validation_f1_raw = bp_cssd_features(validation_bp, bp_filters)
    training_f2_raw = erd_cssd_features(training_erd, erd_filters)
    validation_f2_raw = erd_cssd_features(validation_erd, erd_filters)
    training_f3_raw, retained_channels = trend_features(training_bp, channel_names)
    validation_f3_raw, retained_validation = trend_features(
        validation_bp, channel_names
    )
    if retained_channels != retained_validation:
        raise RuntimeError("Trend-channel selection changed between train and validation")

    training_f1, validation_f1, _ = fit_branch_fisher(
        training_f1_raw, validation_f1_raw, training_y
    )
    training_f2, validation_f2, _ = fit_branch_fisher(
        training_f2_raw, validation_f2_raw, training_y
    )
    training_f3, validation_f3, _ = fit_branch_fisher(
        training_f3_raw, validation_f3_raw, training_y
    )

    training_fusion = np.column_stack([training_f1, training_f2, training_f3])
    validation_fusion = np.column_stack([validation_f1, validation_f2, validation_f3])
    final_lda = make_fisher_pipeline()
    final_lda.fit(training_fusion, training_y)
    training_prediction = final_lda.predict(training_fusion).astype(np.int64)
    training_probability = final_lda.predict_proba(training_fusion)[:, 1]
    training_score = np.asarray(
        final_lda.decision_function(training_fusion)
    ).reshape(-1)
    prediction = final_lda.predict(validation_fusion).astype(np.int64)
    probability = final_lda.predict_proba(validation_fusion)[:, 1]
    score = np.asarray(final_lda.decision_function(validation_fusion)).reshape(-1)

    if prediction.shape != validation_y.shape or not np.isfinite(probability).all():
        raise RuntimeError("Invalid Phase A2 validation output")

    return {
        "prediction": prediction,
        "probability": probability,
        "score": score,
        "branch_scores": validation_fusion,
        "training_branch_scores": training_fusion,
        "metrics": metric_bundle(validation_y, prediction, probability),
        "training_metrics": metric_bundle(
            training_y, training_prediction, training_probability
        ),
        "training_score": training_score,
        "branch_training_metrics": {
            name: score_metric_bundle(training_y, training_fusion[:, index])
            for index, name in enumerate(BRANCH_NAMES)
        },
        "branch_validation_metrics": {
            name: score_metric_bundle(validation_y, validation_fusion[:, index])
            for index, name in enumerate(BRANCH_NAMES)
        },
        "bp_filters": bp_filters,
        "erd_filters": erd_filters,
        "bp_cssd": bp_diagnostics,
        "erd_cssd": erd_diagnostics,
        "feature_shapes": {
            "f1_bp_cssd": int(training_f1_raw.shape[1]),
            "f2_erd_cssd": int(training_f2_raw.shape[1]),
            "f3_bp_trend": int(training_f3_raw.shape[1]),
            "fusion": int(training_fusion.shape[1]),
        },
        "retained_trend_channels": retained_channels,
    }


def inner_oof_branch_scores(
    training_x: np.ndarray,
    training_y: np.ndarray,
    channel_names: np.ndarray,
    fold_count: int,
    seed: int,
) -> np.ndarray:
    """Create branch scores unseen by their own branch models.

    These scores are used only to train a diagnostic cross-fitted fusion LDA.
    The outer validation fold remains completely untouched.
    """
    scores = np.full((len(training_y), len(BRANCH_NAMES)), np.nan, dtype=np.float64)
    for inner_training, inner_validation in stratified_folds(
        training_y, fold_count, seed
    ):
        inner_result = run_fold(
            training_x,
            training_y,
            channel_names,
            inner_training,
            inner_validation,
        )
        scores[inner_validation] = inner_result["branch_scores"]
    if not np.isfinite(scores).all():
        raise RuntimeError("Inner-OOF branch scores are incomplete")
    return scores


def _subspace_stability(
    first: np.ndarray, second: np.ndarray
) -> dict[str, float]:
    first_basis = np.linalg.qr(first.T, mode="reduced")[0]
    second_basis = np.linalg.qr(second.T, mode="reduced")[0]
    angles = subspace_angles(first_basis, second_basis)
    cosines = np.cos(angles)
    return {
        "mean_abs_cosine": float(np.mean(np.abs(cosines))),
        "minimum_abs_cosine": float(np.min(np.abs(cosines))),
        "mean_principal_angle_deg": float(np.degrees(angles).mean()),
        "maximum_principal_angle_deg": float(np.degrees(angles).max()),
    }


def cssd_filter_stability_rows(
    filter_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in sorted({int(record["seed"]) for record in filter_records}):
        records = [record for record in filter_records if int(record["seed"]) == seed]
        for first, second in combinations(records, 2):
            definitions = (
                ("bp", "left", first["bp_filters"][[0]], second["bp_filters"][[0]]),
                ("bp", "right", first["bp_filters"][[1]], second["bp_filters"][[1]]),
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
                        "seed": seed,
                        "fold_a": int(first["fold"]),
                        "fold_b": int(second["fold"]),
                        "branch": branch,
                        "class_specific_side": side,
                        **_subspace_stability(first_filters, second_filters),
                    }
                )
    return rows


def create_diagnostic_figure(
    layer_summary: list[dict[str, Any]],
    ablation_summary: list[dict[str, Any]],
    filter_summary: list[dict[str, Any]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    layer_order = [*BRANCH_NAMES, "final_in_sample_fusion"]
    x = np.arange(len(layer_order))
    width = 0.36
    training_auc = [
        100.0
        * next(
            row["mean_roc_auc"]
            for row in layer_summary
            if row["layer"] == layer and row["split"] == "training"
        )
        for layer in layer_order
    ]
    validation_auc = [
        100.0
        * next(
            row["mean_roc_auc"]
            for row in layer_summary
            if row["layer"] == layer and row["split"] == "validation"
        )
        for layer in layer_order
    ]
    axes[0].bar(x - width / 2, training_auc, width, label="Training")
    axes[0].bar(x + width / 2, validation_auc, width, label="Validation")
    axes[0].axhline(50.0, color="black", linestyle=":", linewidth=1)
    axes[0].set_xticks(x, ["F1 BP", "F2 ERD", "F3 trend", "Final"])
    axes[0].set_ylabel("Mean ROC AUC (%)")
    axes[0].set_ylim(40.0, 100.0)
    axes[0].set_title("Layer generalization gap")
    axes[0].legend()

    subset_order = [name for name, _ in BRANCH_SUBSETS]
    x = np.arange(len(subset_order))
    standard = [
        100.0
        * next(
            row["mean_balanced_accuracy"]
            for row in ablation_summary
            if row["fusion_training"] == "in_sample"
            and row["subset"] == subset
        )
        for subset in subset_order
    ]
    cross_fitted = [
        100.0
        * next(
            row["mean_balanced_accuracy"]
            for row in ablation_summary
            if row["fusion_training"] == "inner_oof"
            and row["subset"] == subset
        )
        for subset in subset_order
    ]
    axes[1].plot(x, standard, marker="o", label="In-sample fusion")
    axes[1].plot(x, cross_fitted, marker="o", label="Inner-OOF fusion")
    axes[1].axhline(
        100.0 * ARCHIVED_BASELINE_OOF_BA,
        color="#c44e52",
        linestyle="--",
        label="Archived baseline",
    )
    axes[1].set_xticks(x, subset_order, rotation=35, ha="right")
    axes[1].set_ylabel("Mean OOF balanced accuracy (%)")
    axes[1].set_ylim(40.0, 75.0)
    axes[1].set_title("Branch ablation and fusion")
    axes[1].legend(fontsize=8)

    stability_labels = ["BP-left", "BP-right", "ERD-left", "ERD-right"]
    stability_values = []
    for label in stability_labels:
        branch, side = label.lower().split("-")
        stability_values.append(
            100.0
            * next(
                row["mean_abs_cosine"]
                for row in filter_summary
                if row["branch"] == branch
                and row["class_specific_side"] == side
            )
        )
    axes[2].bar(stability_labels, stability_values, color="#55a868")
    axes[2].set_ylabel("Mean cross-fold subspace cosine (%)")
    axes[2].set_ylim(0.0, 100.0)
    axes[2].set_title("CSSD spatial stability")
    axes[2].tick_params(axis="x", rotation=25)
    for index, value in enumerate(stability_values):
        axes[2].text(index, value + 1.5, f"{value:.1f}", ha="center", fontsize=9)

    fig.suptitle("FingerMovements Phase A2 generalization diagnostics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_figure(
    seed_rows: list[dict[str, Any]],
    aggregate_confusion: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    labels = [str(row["seed"]) for row in seed_rows]
    values = [100.0 * float(row["balanced_accuracy"]) for row in seed_rows]
    bars = axes[0].bar(labels, values, color="#3976af")
    axes[0].axhline(
        100.0 * ARCHIVED_BASELINE_OOF_BA,
        color="#c44e52",
        linestyle="--",
        linewidth=1.5,
        label="Archived terminal Logistic OOF mean",
    )
    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("OOF balanced accuracy (%)")
    axes[0].set_title("Phase A2 seed stability")
    axes[0].set_ylim(40.0, 100.0)
    axes[0].legend(fontsize=8)
    for bar, value in zip(bars, values, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.0,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    image = axes[1].imshow(aggregate_confusion, cmap="Blues")
    axes[1].set_xticks([0, 1], ["left", "right"])
    axes[1].set_yticks([0, 1], ["left", "right"])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("Aggregated OOF confusion")
    for row in range(2):
        for column in range(2):
            axes[1].text(
                column,
                row,
                str(int(aggregate_confusion[row, column])),
                ha="center",
                va="center",
                color="white"
                if aggregate_confusion[row, column] > aggregate_confusion.max() / 2
                else "black",
            )
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle("FingerMovements Phase A2: paper-style CSSD + LDA")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    x, y, source_index, channel_names = load_training_data(args.data)

    print("=== FingerMovements Phase A2: paper-style CSSD + hierarchical LDA ===")
    print(f"data={args.data}")
    print(f"cases={len(y)} | input={CHANNELS}x{TIMEPOINTS} @ {SAMPLING_RATE_HZ:g} Hz")
    print(f"seeds={args.seeds} | folds={args.folds}")
    print("policy=TRAIN only; TEST refused; CSSD/scalers/LDA fit per training fold")
    print("filters=zero-phase Butterworth order 4 (offline reproduction)")
    print("branches=f1 BP-CSSD(8) | f2 ERD-CSSD(24) | f3 BP-trend(38) -> LDA(3)")

    all_fold_rows: list[dict[str, Any]] = []
    all_prediction_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    first_fold_contract: dict[str, Any] | None = None
    diagnostic_layer_rows: list[dict[str, Any]] = []
    diagnostic_ablation_fold_rows: list[dict[str, Any]] = []
    diagnostic_prediction_rows: list[dict[str, Any]] = []
    filter_records: list[dict[str, Any]] = []

    seeds_to_run = args.seeds[:1] if args.validate_only else args.seeds
    for seed in seeds_to_run:
        folds = stratified_folds(y, args.folds, seed)
        if args.validate_only:
            folds = folds[:1]

        seed_prediction = np.full(len(y), -1, dtype=np.int64)
        seed_probability = np.full(len(y), np.nan, dtype=np.float64)
        seed_score = np.full(len(y), np.nan, dtype=np.float64)
        seed_branches = np.full((len(y), 3), np.nan, dtype=np.float64)

        for fold_number, (training_indices, validation_indices) in enumerate(
            folds, start=1
        ):
            started = perf_counter()
            result = run_fold(
                x,
                y,
                channel_names,
                training_indices,
                validation_indices,
            )
            elapsed = perf_counter() - started
            metrics = result["metrics"]
            seed_prediction[validation_indices] = result["prediction"]
            seed_probability[validation_indices] = result["probability"]
            seed_score[validation_indices] = result["score"]
            seed_branches[validation_indices] = result["branch_scores"]

            fold_row = {
                "seed": int(seed),
                "fold": int(fold_number),
                "training_cases": int(len(training_indices)),
                "validation_cases": int(len(validation_indices)),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "mean_log_loss": metrics["mean_log_loss"],
                "elapsed_seconds": float(elapsed),
                "bp_left_eigenvalue": result["bp_cssd"][
                    "left_specific_eigenvalues"
                ][0],
                "bp_right_eigenvalue": result["bp_cssd"][
                    "right_specific_eigenvalues"
                ][0],
                "bp_condition_number": result["bp_cssd"][
                    "composite_condition_number"
                ],
                "erd_condition_number": result["erd_cssd"][
                    "composite_condition_number"
                ],
            }
            all_fold_rows.append(fold_row)
            if first_fold_contract is None:
                first_fold_contract = {
                    "feature_shapes": result["feature_shapes"],
                    "retained_trend_channels": result["retained_trend_channels"],
                }

            if args.diagnostics:
                training_y = y[training_indices]
                validation_y = y[validation_indices]
                for layer in BRANCH_NAMES:
                    for split, cases, values in (
                        (
                            "training",
                            len(training_indices),
                            result["branch_training_metrics"][layer],
                        ),
                        (
                            "validation",
                            len(validation_indices),
                            result["branch_validation_metrics"][layer],
                        ),
                    ):
                        diagnostic_layer_rows.append(
                            {
                                "seed": int(seed),
                                "fold": int(fold_number),
                                "layer": layer,
                                "split": split,
                                "cases": int(cases),
                                **values,
                            }
                        )
                for split, cases, labels, scores in (
                    (
                        "training",
                        len(training_indices),
                        training_y,
                        result["training_score"],
                    ),
                    (
                        "validation",
                        len(validation_indices),
                        validation_y,
                        result["score"],
                    ),
                ):
                    diagnostic_layer_rows.append(
                        {
                            "seed": int(seed),
                            "fold": int(fold_number),
                            "layer": "final_in_sample_fusion",
                            "split": split,
                            "cases": int(cases),
                            **score_metric_bundle(labels, scores),
                        }
                    )

                filter_records.append(
                    {
                        "seed": int(seed),
                        "fold": int(fold_number),
                        "bp_filters": result["bp_filters"],
                        "erd_filters": result["erd_filters"],
                    }
                )

                inner_seed = int(seed * 1000 + fold_number)
                inner_scores = inner_oof_branch_scores(
                    x[training_indices],
                    training_y,
                    channel_names,
                    args.folds,
                    inner_seed,
                )
                training_score_sets = {
                    "in_sample": result["training_branch_scores"],
                    "inner_oof": inner_scores,
                }
                for fusion_training, training_scores in training_score_sets.items():
                    for subset_name, columns in BRANCH_SUBSETS:
                        fusion = fit_fusion_lda(
                            training_scores,
                            result["branch_scores"],
                            training_y,
                            columns,
                        )
                        fusion_metrics = metric_bundle(
                            validation_y,
                            fusion["prediction"],
                            fusion["probability"],
                        )
                        diagnostic_ablation_fold_rows.append(
                            {
                                "seed": int(seed),
                                "fold": int(fold_number),
                                "fusion_training": fusion_training,
                                "subset": subset_name,
                                "validation_cases": int(len(validation_indices)),
                                "accuracy": fusion_metrics["accuracy"],
                                "balanced_accuracy": fusion_metrics[
                                    "balanced_accuracy"
                                ],
                                "macro_f1": fusion_metrics["macro_f1"],
                                "mean_log_loss": fusion_metrics["mean_log_loss"],
                            }
                        )
                        for local_index, global_index in enumerate(
                            validation_indices
                        ):
                            diagnostic_prediction_rows.append(
                                {
                                    "seed": int(seed),
                                    "fold": int(fold_number),
                                    "source_index": int(source_index[global_index]),
                                    "true_label": int(y[global_index]),
                                    "fusion_training": fusion_training,
                                    "subset": subset_name,
                                    "predicted_label": int(
                                        fusion["prediction"][local_index]
                                    ),
                                    "probability_right": float(
                                        fusion["probability"][local_index]
                                    ),
                                    "score": float(fusion["score"][local_index]),
                                }
                            )

            print(
                f"seed {seed} | fold {fold_number}/{args.folds} | "
                f"accuracy={100.0 * metrics['accuracy']:.2f}% | "
                f"BA={100.0 * metrics['balanced_accuracy']:.2f}% | "
                f"macro-F1={100.0 * metrics['macro_f1']:.2f}% | "
                f"time={elapsed:.2f}s"
            )

        evaluated = seed_prediction >= 0
        seed_metrics = metric_bundle(
            y[evaluated],
            seed_prediction[evaluated],
            seed_probability[evaluated],
        )
        seed_row = {
            "seed": int(seed),
            "evaluated_cases": int(evaluated.sum()),
            **seed_metrics,
        }
        seed_rows.append(seed_row)
        print(
            f"seed {seed} OOF | cases={evaluated.sum()} | "
            f"accuracy={100.0 * seed_metrics['accuracy']:.2f}% | "
            f"BA={100.0 * seed_metrics['balanced_accuracy']:.2f}% | "
            f"macro-F1={100.0 * seed_metrics['macro_f1']:.2f}%"
        )

        for index in np.flatnonzero(evaluated):
            all_prediction_rows.append(
                {
                    "seed": int(seed),
                    "source_index": int(source_index[index]),
                    "true_label": int(y[index]),
                    "true_name": LABEL_NAMES[int(y[index])],
                    "predicted_label": int(seed_prediction[index]),
                    "predicted_name": LABEL_NAMES[int(seed_prediction[index])],
                    "probability_right": float(seed_probability[index]),
                    "final_lda_score": float(seed_score[index]),
                    "f1_bp_cssd_score": float(seed_branches[index, 0]),
                    "f2_erd_cssd_score": float(seed_branches[index, 1]),
                    "f3_bp_trend_score": float(seed_branches[index, 2]),
                }
            )

    if args.validate_only:
        if first_fold_contract is None:
            raise RuntimeError("Validation did not execute a fold")
        print(f"feature contract={first_fold_contract['feature_shapes']}")
        print(
            "retained trend channels="
            + ", ".join(first_fold_contract["retained_trend_channels"])
        )
        print("validate-only complete; no files written")
        return

    balanced_values = np.array(
        [row["balanced_accuracy"] for row in seed_rows], dtype=np.float64
    )
    accuracy_values = np.array(
        [row["accuracy"] for row in seed_rows], dtype=np.float64
    )
    macro_f1_values = np.array(
        [row["macro_f1"] for row in seed_rows], dtype=np.float64
    )
    aggregate_confusion = np.sum(
        [np.asarray(row["confusion_matrix"], dtype=np.int64) for row in seed_rows],
        axis=0,
    )
    aggregate = {
        "mean_accuracy": float(accuracy_values.mean()),
        "mean_balanced_accuracy": float(balanced_values.mean()),
        "balanced_accuracy_seed_sd": float(balanced_values.std(ddof=0)),
        "worst_seed_balanced_accuracy": float(balanced_values.min()),
        "mean_macro_f1": float(macro_f1_values.mean()),
        "mean_ba_minus_archived_baseline": float(
            balanced_values.mean() - ARCHIVED_BASELINE_OOF_BA
        ),
        "aggregate_confusion_matrix": aggregate_confusion.tolist(),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fold_fields = [
        "seed",
        "fold",
        "training_cases",
        "validation_cases",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "mean_log_loss",
        "elapsed_seconds",
        "bp_left_eigenvalue",
        "bp_right_eigenvalue",
        "bp_condition_number",
        "erd_condition_number",
    ]
    seed_fields = [
        "seed",
        "evaluated_cases",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "mean_log_loss",
        "confusion_matrix",
    ]
    prediction_fields = [
        "seed",
        "source_index",
        "true_label",
        "true_name",
        "predicted_label",
        "predicted_name",
        "probability_right",
        "final_lda_score",
        "f1_bp_cssd_score",
        "f2_erd_cssd_score",
        "f3_bp_trend_score",
    ]
    write_csv(args.output_dir / "phasea2_fold_results.csv", all_fold_rows, fold_fields)
    serializable_seed_rows = []
    for row in seed_rows:
        serializable = dict(row)
        serializable["confusion_matrix"] = json.dumps(row["confusion_matrix"])
        serializable_seed_rows.append(serializable)
    write_csv(
        args.output_dir / "phasea2_seed_results.csv",
        serializable_seed_rows,
        seed_fields,
    )
    write_csv(
        args.output_dir / "phasea2_predictions.csv",
        all_prediction_rows,
        prediction_fields,
    )

    diagnostic_payload: dict[str, Any] | None = None
    if args.diagnostics:
        layer_fields = [
            "seed",
            "fold",
            "layer",
            "split",
            "cases",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "roc_auc",
        ]
        write_csv(
            args.output_dir / "phasea2_diagnostic_layer_metrics.csv",
            diagnostic_layer_rows,
            layer_fields,
        )

        layer_summary: list[dict[str, Any]] = []
        for layer in [*BRANCH_NAMES, "final_in_sample_fusion"]:
            for split in ("training", "validation"):
                selected = [
                    row
                    for row in diagnostic_layer_rows
                    if row["layer"] == layer and row["split"] == split
                ]
                weights = np.array([row["cases"] for row in selected], dtype=float)
                layer_summary.append(
                    {
                        "layer": layer,
                        "split": split,
                        "mean_accuracy": float(
                            np.average([row["accuracy"] for row in selected], weights=weights)
                        ),
                        "mean_balanced_accuracy": float(
                            np.average(
                                [row["balanced_accuracy"] for row in selected],
                                weights=weights,
                            )
                        ),
                        "mean_macro_f1": float(
                            np.average([row["macro_f1"] for row in selected], weights=weights)
                        ),
                        "mean_roc_auc": float(
                            np.average([row["roc_auc"] for row in selected], weights=weights)
                        ),
                    }
                )
        write_csv(
            args.output_dir / "phasea2_diagnostic_layer_summary.csv",
            layer_summary,
            [
                "layer",
                "split",
                "mean_accuracy",
                "mean_balanced_accuracy",
                "mean_macro_f1",
                "mean_roc_auc",
            ],
        )

        write_csv(
            args.output_dir / "phasea2_diagnostic_ablation_fold_results.csv",
            diagnostic_ablation_fold_rows,
            [
                "seed",
                "fold",
                "fusion_training",
                "subset",
                "validation_cases",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "mean_log_loss",
            ],
        )
        write_csv(
            args.output_dir / "phasea2_diagnostic_ablation_predictions.csv",
            diagnostic_prediction_rows,
            [
                "seed",
                "fold",
                "source_index",
                "true_label",
                "fusion_training",
                "subset",
                "predicted_label",
                "probability_right",
                "score",
            ],
        )

        ablation_seed_rows: list[dict[str, Any]] = []
        for fusion_training in ("in_sample", "inner_oof"):
            for subset_name, _ in BRANCH_SUBSETS:
                for seed in args.seeds:
                    selected = [
                        row
                        for row in diagnostic_prediction_rows
                        if row["fusion_training"] == fusion_training
                        and row["subset"] == subset_name
                        and int(row["seed"]) == int(seed)
                    ]
                    if len(selected) != CASES:
                        raise RuntimeError(
                            "Diagnostic ablation did not cover every outer OOF case"
                        )
                    selected.sort(key=lambda row: int(row["source_index"]))
                    labels = np.array(
                        [row["true_label"] for row in selected], dtype=np.int64
                    )
                    predictions = np.array(
                        [row["predicted_label"] for row in selected], dtype=np.int64
                    )
                    probabilities = np.array(
                        [row["probability_right"] for row in selected], dtype=float
                    )
                    metrics = metric_bundle(labels, predictions, probabilities)
                    ablation_seed_rows.append(
                        {
                            "fusion_training": fusion_training,
                            "subset": subset_name,
                            "seed": int(seed),
                            "accuracy": metrics["accuracy"],
                            "balanced_accuracy": metrics["balanced_accuracy"],
                            "macro_f1": metrics["macro_f1"],
                            "mean_log_loss": metrics["mean_log_loss"],
                            "confusion_matrix": json.dumps(
                                metrics["confusion_matrix"]
                            ),
                        }
                    )
        write_csv(
            args.output_dir / "phasea2_diagnostic_ablation_seed_results.csv",
            ablation_seed_rows,
            [
                "fusion_training",
                "subset",
                "seed",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "mean_log_loss",
                "confusion_matrix",
            ],
        )

        ablation_summary: list[dict[str, Any]] = []
        for fusion_training in ("in_sample", "inner_oof"):
            for subset_name, _ in BRANCH_SUBSETS:
                selected = [
                    row
                    for row in ablation_seed_rows
                    if row["fusion_training"] == fusion_training
                    and row["subset"] == subset_name
                ]
                balanced = np.array(
                    [row["balanced_accuracy"] for row in selected], dtype=float
                )
                ablation_summary.append(
                    {
                        "fusion_training": fusion_training,
                        "subset": subset_name,
                        "mean_accuracy": float(
                            np.mean([row["accuracy"] for row in selected])
                        ),
                        "mean_balanced_accuracy": float(balanced.mean()),
                        "balanced_accuracy_seed_sd": float(balanced.std(ddof=0)),
                        "worst_seed_balanced_accuracy": float(balanced.min()),
                        "mean_macro_f1": float(
                            np.mean([row["macro_f1"] for row in selected])
                        ),
                    }
                )
        write_csv(
            args.output_dir / "phasea2_diagnostic_ablation_summary.csv",
            ablation_summary,
            [
                "fusion_training",
                "subset",
                "mean_accuracy",
                "mean_balanced_accuracy",
                "balanced_accuracy_seed_sd",
                "worst_seed_balanced_accuracy",
                "mean_macro_f1",
            ],
        )

        filter_rows = cssd_filter_stability_rows(filter_records)
        write_csv(
            args.output_dir / "phasea2_diagnostic_filter_stability.csv",
            filter_rows,
            [
                "seed",
                "fold_a",
                "fold_b",
                "branch",
                "class_specific_side",
                "mean_abs_cosine",
                "minimum_abs_cosine",
                "mean_principal_angle_deg",
                "maximum_principal_angle_deg",
            ],
        )
        filter_summary: list[dict[str, Any]] = []
        for branch in ("bp", "erd"):
            for side in ("left", "right"):
                selected = [
                    row
                    for row in filter_rows
                    if row["branch"] == branch
                    and row["class_specific_side"] == side
                ]
                filter_summary.append(
                    {
                        "branch": branch,
                        "class_specific_side": side,
                        "comparisons": int(len(selected)),
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
        write_csv(
            args.output_dir / "phasea2_diagnostic_filter_stability_summary.csv",
            filter_summary,
            [
                "branch",
                "class_specific_side",
                "comparisons",
                "mean_abs_cosine",
                "minimum_abs_cosine",
                "mean_principal_angle_deg",
                "maximum_principal_angle_deg",
            ],
        )

        case_rows: list[dict[str, Any]] = []
        for case_index in range(CASES):
            selected = [
                row
                for row in all_prediction_rows
                if int(row["source_index"]) == case_index
            ]
            if len(selected) != len(args.seeds):
                raise RuntimeError("Case stability is missing one or more seeds")
            predictions = np.array(
                [row["predicted_label"] for row in selected], dtype=np.int64
            )
            true_label = int(selected[0]["true_label"])
            case_rows.append(
                {
                    "source_index": case_index,
                    "true_label": true_label,
                    "correct_seed_count": int(np.sum(predictions == true_label)),
                    "predicted_right_seed_count": int(predictions.sum()),
                    "unanimous_prediction": int(len(set(predictions.tolist())) == 1),
                    "final_score_seed_sd": float(
                        np.std([row["final_lda_score"] for row in selected], ddof=0)
                    ),
                    "f1_score_seed_sd": float(
                        np.std([row["f1_bp_cssd_score"] for row in selected], ddof=0)
                    ),
                    "f2_score_seed_sd": float(
                        np.std([row["f2_erd_cssd_score"] for row in selected], ddof=0)
                    ),
                    "f3_score_seed_sd": float(
                        np.std([row["f3_bp_trend_score"] for row in selected], ddof=0)
                    ),
                }
            )
        write_csv(
            args.output_dir / "phasea2_diagnostic_case_stability.csv",
            case_rows,
            [
                "source_index",
                "true_label",
                "correct_seed_count",
                "predicted_right_seed_count",
                "unanimous_prediction",
                "final_score_seed_sd",
                "f1_score_seed_sd",
                "f2_score_seed_sd",
                "f3_score_seed_sd",
            ],
        )

        case_summary = {
            "unanimous_prediction_cases": int(
                sum(row["unanimous_prediction"] for row in case_rows)
            ),
            "always_correct_cases": int(
                sum(row["correct_seed_count"] == len(args.seeds) for row in case_rows)
            ),
            "always_wrong_cases": int(
                sum(row["correct_seed_count"] == 0 for row in case_rows)
            ),
            "seed_sensitive_cases": int(
                sum(row["unanimous_prediction"] == 0 for row in case_rows)
            ),
        }

        diagnostic_payload = {
            "purpose": "locate Phase A2 cross-fold generalization failure without changing model hyperparameters",
            "fusion_protocols": {
                "in_sample": "fusion LDA trained on branch scores from branch models fitted on the same outer-training cases",
                "inner_oof": "fusion LDA trained on five-fold inner-OOF branch scores inside each outer-training fold",
            },
            "layer_summary": layer_summary,
            "ablation_summary": ablation_summary,
            "filter_stability_summary": filter_summary,
            "case_stability_summary": case_summary,
            "test_policy": "official TEST refused and not loaded",
        }
        with (args.output_dir / "phasea2_diagnostics.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(diagnostic_payload, handle, indent=2)
            handle.write("\n")
        create_diagnostic_figure(
            layer_summary,
            ablation_summary,
            filter_summary,
            args.output_dir / "phasea2_generalization_diagnostics.png",
        )

    payload = {
        "phase": "a2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "paper-style CSSD + hierarchical LDA on official TRAIN only",
        "article": {
            "title": (
                "BCI Competition 2003--Data Set IV: An Algorithm Based on "
                "CSSD and FDA for Classifying Single-Trial EEG"
            ),
            "doi_url": ARTICLE_URL,
            "published_test_accuracy_reference": 0.84,
        },
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
            "learned_operations": "CSSD, scaling, branch LDA, and fusion LDA fit on training fold only",
        },
        "implementation": {
            "temporal_filter": "fourth-order zero-phase Butterworth",
            "bp_band_hz": [0.0, BP_LOW_PASS_HZ],
            "bp_window_one_based_inclusive": [44, 47],
            "bp_patterns_per_class": BP_PATTERNS_PER_CLASS,
            "erd_band_hz": list(ERD_BAND_HZ),
            "erd_window_one_based_inclusive": [19, 50],
            "erd_patterns_per_class": ERD_PATTERNS_PER_CLASS,
            "erd_absolute_pool_samples": ERD_POOL_SAMPLES,
            "trend_windows_one_based_inclusive": [[1, 8], [41, 50]],
            "rejected_trend_channels": list(REJECTED_TREND_CHANNELS),
            "cssd": "regularized two-condition simultaneous covariance diagonalization",
            "cssd_ridge": CSSD_RIDGE,
            "branch_classifier": "standardized Fisher/LDA (svd)",
            "fusion_classifier": "standardized Fisher/LDA (svd)",
            "feature_contract": first_fold_contract,
            "paper_difference": "paper uses a perceptron for final fusion; Phase A2 uses LDA",
            "causality": "zero-phase filtering is an offline reference and is not a causal firmware pipeline",
        },
        "archived_baseline_reference": {
            "name": "terminal low-pass + Logistic",
            "mean_oof_balanced_accuracy": ARCHIVED_BASELINE_OOF_BA,
        },
        "seed_results": seed_rows,
        "aggregate": aggregate,
        "diagnostics": diagnostic_payload,
    }
    with (args.output_dir / "phasea2_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    create_figure(
        seed_rows,
        aggregate_confusion,
        args.output_dir / "phasea2_cssd_lda.png",
    )

    print("\n=== Phase A2 aggregate ===")
    print(
        f"mean accuracy={100.0 * aggregate['mean_accuracy']:.2f}% | "
        f"mean BA={100.0 * aggregate['mean_balanced_accuracy']:.2f}% | "
        f"seed SD={100.0 * aggregate['balanced_accuracy_seed_sd']:.2f} pp | "
        f"worst seed={100.0 * aggregate['worst_seed_balanced_accuracy']:.2f}%"
    )
    print(
        "difference from archived terminal-Logistic OOF mean="
        f"{100.0 * aggregate['mean_ba_minus_archived_baseline']:+.2f} pp"
    )
    print(f"metrics={args.output_dir / 'phasea2_metrics.json'}")
    print(f"figure={args.output_dir / 'phasea2_cssd_lda.png'}")
    if diagnostic_payload is not None:
        print("\n=== Phase A2 generalization diagnostics ===")
        for row in diagnostic_payload["layer_summary"]:
            if row["split"] == "validation":
                training = next(
                    candidate
                    for candidate in diagnostic_payload["layer_summary"]
                    if candidate["layer"] == row["layer"]
                    and candidate["split"] == "training"
                )
                print(
                    f"{row['layer']}: AUC train="
                    f"{100.0 * training['mean_roc_auc']:.2f}% "
                    f"validation={100.0 * row['mean_roc_auc']:.2f}% "
                    f"gap={100.0 * (training['mean_roc_auc'] - row['mean_roc_auc']):+.2f} pp"
                )
        best_in_sample = max(
            (
                row
                for row in diagnostic_payload["ablation_summary"]
                if row["fusion_training"] == "in_sample"
            ),
            key=lambda row: row["mean_balanced_accuracy"],
        )
        best_inner_oof = max(
            (
                row
                for row in diagnostic_payload["ablation_summary"]
                if row["fusion_training"] == "inner_oof"
            ),
            key=lambda row: row["mean_balanced_accuracy"],
        )
        print(
            f"best in-sample fusion={best_in_sample['subset']} "
            f"BA={100.0 * best_in_sample['mean_balanced_accuracy']:.2f}%"
        )
        print(
            f"best inner-OOF fusion={best_inner_oof['subset']} "
            f"BA={100.0 * best_inner_oof['mean_balanced_accuracy']:.2f}%"
        )
        print(
            f"diagnostics={args.output_dir / 'phasea2_diagnostics.json'}"
        )
        print(
            "diagnostic figure="
            f"{args.output_dir / 'phasea2_generalization_diagnostics.png'}"
        )
    print("official TEST: NOT LOADED")


if __name__ == "__main__":
    main()
