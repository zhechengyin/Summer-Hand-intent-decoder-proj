"""Phase 2c diagnostic: causal horizon evaluation for CSSD + LDA.

One prediction is evaluated at the end of every 50 ms bin.  A prediction at
time t may use x[0:t] including the current sample and may never use x[t+1:].
Every learned quantity is fitted inside the current training fold.  Official
TEST is refused.
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARCHIVE_ROOT = Path(__file__).resolve().parents[1]

CASES = 316
CHANNELS = 28
TIMEPOINTS = 50
SAMPLING_RATE_HZ = 100.0
SAMPLE_INTERVAL_MS = 10
BIN_MS = 50
SAMPLES_PER_BIN = BIN_MS // SAMPLE_INTERVAL_MS
BIN_COUNT = TIMEPOINTS // SAMPLES_PER_BIN
CLASS_COUNTS = {0: 159, 1: 157}
SEEDS = (42, 43, 44)
FOLDS = 5

FILTER_ORDER = 4
BP_LOW_PASS_HZ = 7.0
ERD_BAND_HZ = (10.0, 33.0)
CSSD_RIDGE = 1e-6
BP_PATTERNS_PER_CLASS = 1
ERD_PATTERNS_PER_CLASS = 1

BP_FEATURE_SAMPLES = 4
ERD_HISTORY_SAMPLES = 32
ERD_POOL_SAMPLES = 8
TREND_START_SAMPLES = 8
TREND_RECENT_SAMPLES = 10

OFFLINE_REFERENCE_MEAN_BA = 0.8672168142183766

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
class HorizonModel:
    bp_filters: np.ndarray
    erd_filters: np.ndarray
    bp_branch: Any
    erd_branch: Any
    trend_branch: Any
    fusion: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data/processed/finger_movements/train.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARCHIVE_ROOT / "results/phase2c_horizon_diagnostic",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run all ten causal horizons for one fold and write nothing.",
    )
    args = parser.parse_args()
    if any("test" in part.lower() for part in args.data.parts):
        parser.error("Phase 2c refuses to load any path identified as TEST")
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain unique values")
    if not 2 <= args.folds <= min(CLASS_COUNTS.values()):
        parser.error("--folds is outside the valid range")
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
        raise ValueError(f"Unexpected x shape: {x.shape}")
    if y.shape != (CASES,) or source_index.shape != (CASES,):
        raise ValueError("Unexpected label or source-index shape")
    if channel_names.shape != (CHANNELS,):
        raise ValueError(f"Unexpected channel-name shape: {channel_names.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Input contains non-finite values")
    if dict(Counter(y.tolist())) != CLASS_COUNTS:
        raise ValueError(f"Unexpected class counts: {dict(Counter(y.tolist()))}")
    if not np.array_equal(source_index, np.arange(CASES)):
        raise ValueError("source_index must preserve official TRAIN order")
    if len(set(channel_names.tolist())) != CHANNELS:
        raise ValueError("channel_names contains duplicates")
    if not set(REJECTED_TREND_CHANNELS).issubset(set(channel_names.tolist())):
        raise ValueError("Required trend-channel names are missing")
    if np.all(x[:, :-1, 28:] == x[:, 1:, :22]):
        raise ValueError("Detected the retired UEA sliding-channel layout error")
    return x, y, source_index, channel_names


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def causal_sos_filter(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Filter each trial left-to-right with first-sample causal initialization."""
    initial = sosfilt_zi(sos)[:, None, None, :] * x[None, :, :, 0, None]
    filtered, _ = sosfilt(sos, x, axis=-1, zi=initial)
    if not np.isfinite(filtered).all():
        raise FloatingPointError("Causal filtering produced non-finite values")
    return filtered


def temporal_filter(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bp = causal_sos_filter(x, BP_SOS)
    bp = bp - bp[..., :1]
    erd = causal_sos_filter(x, ERD_SOS)
    return bp, erd


def _trace_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    trace = float(np.trace(matrix))
    if not np.isfinite(trace) or trace <= 1e-12:
        raise ValueError("Degenerate spatial matrix")
    return matrix / trace


def class_spatial_matrix(class_x: np.ndarray) -> np.ndarray:
    """Frozen empirical covariance with per-trial trace normalization."""
    moments = np.einsum("nct,ndt->ncd", class_x, class_x, optimize=True)
    traces = np.trace(moments, axis1=1, axis2=2)
    if not np.isfinite(traces).all() or np.any(traces <= 1e-12):
        raise ValueError("Trial normalization received a degenerate prefix")
    normalized = class_x / np.sqrt(traces)[:, None, None]
    samples = normalized.transpose(0, 2, 1).reshape(-1, CHANNELS)
    matrix = _trace_normalize(samples.T @ samples / len(samples))
    scale = float(np.trace(matrix) / CHANNELS)
    matrix = (1.0 - CSSD_RIDGE) * matrix + CSSD_RIDGE * scale * np.eye(CHANNELS)
    return 0.5 * (matrix + matrix.T)


def fit_cssd_filters(
    windowed_x: np.ndarray, y: np.ndarray, patterns_per_class: int
) -> np.ndarray:
    left = class_spatial_matrix(windowed_x[y == 0])
    right = class_spatial_matrix(windowed_x[y == 1])
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


def project_cssd(x: np.ndarray, filters: np.ndarray) -> np.ndarray:
    return np.einsum("kc,nct->nkt", filters, x, optimize=True)


def horizon_windows(stop: int) -> tuple[slice, slice, slice, slice]:
    if stop < SAMPLES_PER_BIN or stop > TIMEPOINTS:
        raise ValueError(f"Invalid causal horizon: {stop}")
    bp_window = slice(max(0, stop - BP_FEATURE_SAMPLES), stop)
    erd_window = slice(max(0, stop - ERD_HISTORY_SAMPLES), stop)
    trend_start = slice(0, min(TREND_START_SAMPLES, stop))
    trend_recent = slice(max(0, stop - TREND_RECENT_SAMPLES), stop)
    return bp_window, erd_window, trend_start, trend_recent


def bp_features(bp: np.ndarray, filters: np.ndarray, stop: int) -> np.ndarray:
    bp_window, _, _, _ = horizon_windows(stop)
    features = project_cssd(bp[..., bp_window], filters).reshape(len(bp), -1)
    expected = 2 * BP_PATTERNS_PER_CLASS * BP_FEATURE_SAMPLES
    if features.shape != (len(bp), expected):
        raise RuntimeError(f"Unexpected BP feature shape: {features.shape}")
    return features


def erd_features(erd: np.ndarray, filters: np.ndarray, stop: int) -> np.ndarray:
    _, erd_window, _, _ = horizon_windows(stop)
    projected = project_cssd(erd[..., erd_window], filters)
    missing = ERD_HISTORY_SAMPLES - projected.shape[-1]
    if missing:
        projected = np.pad(projected, ((0, 0), (0, 0), (missing, 0)))
    pooled = (
        np.abs(projected)
        .reshape(
            len(erd),
            projected.shape[1],
            ERD_HISTORY_SAMPLES // ERD_POOL_SAMPLES,
            ERD_POOL_SAMPLES,
        )
        .mean(axis=-1)
    )
    features = pooled.reshape(len(erd), -1)
    expected = 2 * ERD_PATTERNS_PER_CLASS * (ERD_HISTORY_SAMPLES // ERD_POOL_SAMPLES)
    if features.shape != (len(erd), expected):
        raise RuntimeError(f"Unexpected ERD feature shape: {features.shape}")
    return features


def trend_features(
    bp: np.ndarray, retained_indices: np.ndarray, stop: int
) -> np.ndarray:
    _, _, start_window, recent_window = horizon_windows(stop)
    selected = bp[:, retained_indices]
    start = selected[..., start_window].mean(axis=-1)
    recent = selected[..., recent_window].mean(axis=-1)
    features = np.stack([start, recent], axis=-1).reshape(len(bp), -1)
    if features.shape != (len(bp), 38):
        raise RuntimeError(f"Unexpected trend feature shape: {features.shape}")
    return features


def make_lda() -> Any:
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="svd"))


def fit_horizon_model(
    training_bp: np.ndarray,
    training_erd: np.ndarray,
    training_y: np.ndarray,
    retained_indices: np.ndarray,
    stop: int,
) -> HorizonModel:
    bp_window, erd_window, _, _ = horizon_windows(stop)
    bp_filters = fit_cssd_filters(
        training_bp[..., bp_window], training_y, BP_PATTERNS_PER_CLASS
    )
    erd_filters = fit_cssd_filters(
        training_erd[..., erd_window], training_y, ERD_PATTERNS_PER_CLASS
    )
    features = (
        bp_features(training_bp, bp_filters, stop),
        erd_features(training_erd, erd_filters, stop),
        trend_features(training_bp, retained_indices, stop),
    )
    branches = tuple(make_lda() for _ in range(3))
    training_scores: list[np.ndarray] = []
    for model, branch_features in zip(branches, features):
        model.fit(branch_features, training_y)
        training_scores.append(
            np.asarray(model.decision_function(branch_features)).reshape(-1)
        )
    fusion = make_lda()
    fusion.fit(np.column_stack(training_scores), training_y)
    return HorizonModel(
        bp_filters=bp_filters,
        erd_filters=erd_filters,
        bp_branch=branches[0],
        erd_branch=branches[1],
        trend_branch=branches[2],
        fusion=fusion,
    )


def predict_horizon(
    model: HorizonModel,
    bp: np.ndarray,
    erd: np.ndarray,
    retained_indices: np.ndarray,
    stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = (
        bp_features(bp, model.bp_filters, stop),
        erd_features(erd, model.erd_filters, stop),
        trend_features(bp, retained_indices, stop),
    )
    branches = (model.bp_branch, model.erd_branch, model.trend_branch)
    scores = np.column_stack(
        [
            np.asarray(branch.decision_function(branch_features)).reshape(-1)
            for branch, branch_features in zip(branches, features)
        ]
    )
    prediction = model.fusion.predict(scores).astype(np.int64)
    probability = model.fusion.predict_proba(scores)[:, 1]
    decision = np.asarray(model.fusion.decision_function(scores)).reshape(-1)
    return prediction, probability, decision


def metric_bundle(
    y_true: np.ndarray, prediction: np.ndarray, probability: np.ndarray
) -> dict[str, Any]:
    probability = np.clip(probability, 1e-8, 1.0 - 1e-8)
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro")),
        "mean_log_loss": float(
            log_loss(
                y_true,
                np.column_stack([1.0 - probability, probability]),
                labels=[0, 1],
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_true, prediction, labels=[0, 1]
        ).tolist(),
    }


def assert_future_invariance(
    raw_x: np.ndarray,
    model: HorizonModel,
    retained_indices: np.ndarray,
    stop: int,
) -> float:
    """Prove that replacing every future sample leaves current scores unchanged."""
    if stop == TIMEPOINTS:
        return 0.0
    selected = raw_x.copy()
    altered = selected.copy()
    rng = np.random.default_rng(91_000 + stop)
    altered[..., stop:] = rng.normal(loc=1e6, scale=1e5, size=altered[..., stop:].shape)
    bp_original, erd_original = temporal_filter(selected)
    bp_altered, erd_altered = temporal_filter(altered)
    _, probability_original, score_original = predict_horizon(
        model, bp_original, erd_original, retained_indices, stop
    )
    _, probability_altered, score_altered = predict_horizon(
        model, bp_altered, erd_altered, retained_indices, stop
    )
    maximum_error = float(
        max(
            np.max(np.abs(score_original - score_altered)),
            np.max(np.abs(probability_original - probability_altered)),
        )
    )
    if maximum_error != 0.0:
        raise RuntimeError(
            f"Future-data invariance failed at {stop * SAMPLE_INTERVAL_MS} ms: "
            f"maximum error={maximum_error}"
        )
    return maximum_error


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def create_figure(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    ordered = sorted(summary_rows, key=lambda row: row["prediction_time_ms"])
    time_ms = np.asarray([row["prediction_time_ms"] for row in ordered])
    mean = 100.0 * np.asarray([row["mean_balanced_accuracy"] for row in ordered])
    low = 100.0 * np.asarray([row["worst_seed_balanced_accuracy"] for row in ordered])
    high = 100.0 * np.asarray([row["best_seed_balanced_accuracy"] for row in ordered])
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(time_ms, mean, marker="o", color="#4c72b0", label="strict-causal mean")
    ax.fill_between(time_ms, low, high, color="#4c72b0", alpha=0.18, label="seed range")
    ax.axhline(
        100.0 * OFFLINE_REFERENCE_MEAN_BA,
        color="#c44e52",
        linestyle="--",
        label="offline zero-phase reference",
    )
    ax.axhline(50.0, color="#777777", linestyle=":", label="chance")
    ax.set_xticks(time_ms)
    ax.set_xlabel("Prediction time from trial start (ms)")
    ax.set_ylabel("OOF balanced accuracy (%)")
    ax.set_title("Phase 2c causal horizon diagnostic")
    ax.set_ylim(40.0, 100.0)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    x, y, source_index, channel_names = load_training_data(args.data)
    bp, erd = temporal_filter(x)
    retained_indices = np.asarray(
        [
            index
            for index, name in enumerate(channel_names.tolist())
            if name not in REJECTED_TREND_CHANNELS
        ],
        dtype=np.int64,
    )
    if retained_indices.size != 19:
        raise RuntimeError(
            f"Expected 19 retained channels, got {retained_indices.size}"
        )

    print("=== FingerMovements Phase 2c causal horizon diagnostic ===")
    print(f"data={args.data}")
    print(f"cases={CASES} | input={CHANNELS}x{TIMEPOINTS} @ {SAMPLING_RATE_HZ:g} Hz")
    print(f"bin={BIN_MS} ms ({SAMPLES_PER_BIN} samples) | horizons={BIN_COUNT}")
    print(f"seeds={args.seeds} | folds={args.folds}")
    print("policy=TRAIN only; TEST refused; current/past samples only")
    print("filter=sosfilt left-to-right; zero-phase filtering forbidden")

    seeds_to_run = args.seeds[:1] if args.validate_only else args.seeds
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    invariance_rows: list[dict[str, Any]] = []

    for seed in seeds_to_run:
        folds = stratified_folds(y, args.folds, seed)
        if args.validate_only:
            folds = folds[:1]
        for fold_number, (training, validation) in enumerate(folds, start=1):
            print(f"\nseed={seed} fold={fold_number}/{args.folds}")
            for bin_number in range(1, BIN_COUNT + 1):
                stop = bin_number * SAMPLES_PER_BIN
                started = perf_counter()
                model = fit_horizon_model(
                    bp[training],
                    erd[training],
                    y[training],
                    retained_indices,
                    stop,
                )
                prediction, probability, decision = predict_horizon(
                    model,
                    bp[validation],
                    erd[validation],
                    retained_indices,
                    stop,
                )
                metrics = metric_bundle(y[validation], prediction, probability)
                fold_rows.append(
                    {
                        "seed": int(seed),
                        "fold": int(fold_number),
                        "bin_number": int(bin_number),
                        "prediction_time_ms": int(stop * SAMPLE_INTERVAL_MS),
                        "training_cases": len(training),
                        "validation_cases": len(validation),
                        "accuracy": metrics["accuracy"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "mean_log_loss": metrics["mean_log_loss"],
                        "elapsed_seconds": perf_counter() - started,
                    }
                )
                for local_index, case_index in enumerate(validation):
                    prediction_rows.append(
                        {
                            "seed": int(seed),
                            "fold": int(fold_number),
                            "bin_number": int(bin_number),
                            "prediction_time_ms": int(stop * SAMPLE_INTERVAL_MS),
                            "source_index": int(source_index[case_index]),
                            "true_label": int(y[case_index]),
                            "predicted_label": int(prediction[local_index]),
                            "probability_right": float(probability[local_index]),
                            "decision_score": float(decision[local_index]),
                        }
                    )
                if seed == seeds_to_run[0] and fold_number == 1:
                    maximum_error = assert_future_invariance(
                        x[validation], model, retained_indices, stop
                    )
                    invariance_rows.append(
                        {
                            "bin_number": int(bin_number),
                            "prediction_time_ms": int(stop * SAMPLE_INTERVAL_MS),
                            "future_samples_replaced": int(TIMEPOINTS - stop),
                            "checked_cases": len(validation),
                            "maximum_score_or_probability_error": maximum_error,
                            "passed": True,
                        }
                    )
                print(
                    f"  t={stop * SAMPLE_INTERVAL_MS:03d} ms | "
                    f"BA={100.0 * metrics['balanced_accuracy']:.2f}% | "
                    f"loss={metrics['mean_log_loss']:.4f}"
                )

    if args.validate_only:
        if len(invariance_rows) != BIN_COUNT or not all(
            row["passed"] for row in invariance_rows
        ):
            raise RuntimeError("Incomplete future-invariance validation")
        print("\nvalidate-only=PASS | all 10 horizons | future invariance=PASS")
        print("no result files written")
        return

    seed_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        for bin_number in range(1, BIN_COUNT + 1):
            selected = sorted(
                (
                    row
                    for row in prediction_rows
                    if row["seed"] == seed and row["bin_number"] == bin_number
                ),
                key=lambda row: row["source_index"],
            )
            if len(selected) != CASES:
                raise RuntimeError(
                    f"seed={seed} bin={bin_number} has {len(selected)} OOF cases"
                )
            indices = np.asarray([row["source_index"] for row in selected])
            if not np.array_equal(indices, np.arange(CASES)):
                raise RuntimeError("OOF coverage is incomplete or duplicated")
            labels = np.asarray([row["true_label"] for row in selected])
            predictions = np.asarray([row["predicted_label"] for row in selected])
            probabilities = np.asarray([row["probability_right"] for row in selected])
            metrics = metric_bundle(labels, predictions, probabilities)
            seed_rows.append(
                {
                    "seed": int(seed),
                    "bin_number": int(bin_number),
                    "prediction_time_ms": int(bin_number * BIN_MS),
                    "evaluated_cases": CASES,
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "mean_log_loss": metrics["mean_log_loss"],
                    "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
                }
            )

    summary_rows: list[dict[str, Any]] = []
    for bin_number in range(1, BIN_COUNT + 1):
        selected = [row for row in seed_rows if row["bin_number"] == bin_number]
        balanced = np.asarray([row["balanced_accuracy"] for row in selected])
        summary_rows.append(
            {
                "bin_number": int(bin_number),
                "prediction_time_ms": int(bin_number * BIN_MS),
                "mean_accuracy": float(np.mean([row["accuracy"] for row in selected])),
                "mean_balanced_accuracy": float(balanced.mean()),
                "balanced_accuracy_seed_sd": float(balanced.std(ddof=0)),
                "worst_seed_balanced_accuracy": float(balanced.min()),
                "best_seed_balanced_accuracy": float(balanced.max()),
                "mean_macro_f1": float(np.mean([row["macro_f1"] for row in selected])),
                "delta_vs_offline_reference": float(
                    balanced.mean() - OFFLINE_REFERENCE_MEAN_BA
                ),
            }
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "phase2c_horizon_fold_results.csv", fold_rows)
    write_csv(output_dir / "phase2c_horizon_seed_results.csv", seed_rows)
    write_csv(output_dir / "phase2c_horizon_summary.csv", summary_rows)
    write_csv(output_dir / "phase2c_horizon_predictions.csv", prediction_rows)
    write_csv(output_dir / "phase2c_horizon_future_invariance.csv", invariance_rows)
    create_figure(summary_rows, output_dir / "phase2c_horizon_accuracy.png")

    final = summary_rows[-1]
    payload = {
        "phase": "2c",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "initial causal horizon diagnostic superseded by the rolling Phase 2c runner",
        "data": {
            "path": str(args.data.resolve()),
            "sha256": sha256(args.data.resolve()),
            "cases": CASES,
            "shape": [CASES, CHANNELS, TIMEPOINTS],
            "test_policy": "official TEST refused and not loaded",
        },
        "causality_contract": {
            "sample_rule": "prediction at t uses samples <= t only",
            "bin_ms": BIN_MS,
            "samples_per_bin": SAMPLES_PER_BIN,
            "prediction_horizons_ms": [
                bin_number * BIN_MS for bin_number in range(1, BIN_COUNT + 1)
            ],
            "temporal_filter": "left-to-right scipy.signal.sosfilt",
            "initialization": "first current sample only",
            "forbidden_operations": [
                "sosfiltfilt",
                "filtfilt",
                "centered rolling windows",
                "whole-trial inference normalization",
            ],
            "horizon_specific_models": True,
            "future_replacement_checks": invariance_rows,
        },
        "validation": {
            "seeds": [int(seed) for seed in args.seeds],
            "folds": int(args.folds),
            "split_unit": "whole trial/case",
            "all_learned_operations": "outer-training-fold only",
        },
        "frozen_model_family": {
            "covariance": "empirical",
            "trial_trace_normalization": True,
            "bp_patterns_per_class": BP_PATTERNS_PER_CLASS,
            "erd_patterns_per_class": ERD_PATTERNS_PER_CLASS,
            "fusion": "hierarchical LDA",
        },
        "feature_windows": {
            "bp_recent_samples": BP_FEATURE_SAMPLES,
            "erd_causal_history_samples": ERD_HISTORY_SAMPLES,
            "erd_pool_samples": ERD_POOL_SAMPLES,
            "trend_start_samples": TREND_START_SAMPLES,
            "trend_recent_samples": TREND_RECENT_SAMPLES,
        },
        "offline_reference_mean_oof_balanced_accuracy": OFFLINE_REFERENCE_MEAN_BA,
        "final_horizon_summary": final,
        "seed_results": seed_rows,
        "horizon_summary": summary_rows,
    }
    with (output_dir / "phase2c_horizon_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print("\n=== Phase 2c horizon diagnostic summary ===")
    for row in summary_rows:
        print(
            f"t={row['prediction_time_ms']:03d} ms | "
            f"mean BA={100.0 * row['mean_balanced_accuracy']:.2f}% | "
            f"seed SD={100.0 * row['balanced_accuracy_seed_sd']:.2f} pp | "
            f"worst={100.0 * row['worst_seed_balanced_accuracy']:.2f}%"
        )
    print("future invariance=PASS at all 10 horizons")
    print(f"metrics={output_dir / 'phase2c_horizon_metrics.json'}")


if __name__ == "__main__":
    main()
