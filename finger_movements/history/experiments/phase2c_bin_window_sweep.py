"""Phase 2c TRAIN-only sweep of causal history window and streaming bin size.

The history window changes the samples used by the classifier and is refitted
inside every fold. The bin size changes only how the same causal stream is
chunked; it must not change endpoint predictions. Official TEST is refused.
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
AVAILABLE_SAMPLES = 50
SAMPLING_RATE_HZ = 100.0
SAMPLE_INTERVAL_MS = 10
CLASS_COUNTS = {0: 159, 1: 157}
SEEDS = (42, 43, 44)
FOLDS = 5
DEFAULT_BINS_MS = (10, 20, 50, 100)
DEFAULT_WINDOWS_MS = (200, 300, 400, 500)

FILTER_ORDER = 4
BP_LOW_PASS_HZ = 7.0
ERD_BAND_HZ = (10.0, 33.0)
CSSD_RIDGE = 1e-6
BP_PATTERNS_PER_CLASS = 1
ERD_PATTERNS_PER_CLASS = 1
BP_RECENT_SAMPLES = 4
ERD_MAX_RECENT_SAMPLES = 32
ERD_POOL_COUNT = 4
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
class FoldModel:
    bp_filters: np.ndarray
    erd_filters: np.ndarray
    bp_branch: Any
    erd_branch: Any
    trend_branch: Any
    fusion: Any
    window_samples: int


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
        default=ARCHIVE_ROOT / "results/phase2c_bin_window_sweep",
    )
    parser.add_argument("--bins-ms", nargs="+", type=int, default=list(DEFAULT_BINS_MS))
    parser.add_argument(
        "--windows-ms", nargs="+", type=int, default=list(DEFAULT_WINDOWS_MS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run one fold for every window plus all bin-equivalence checks.",
    )
    args = parser.parse_args()
    if any("test" in part.lower() for part in args.data.parts):
        parser.error("Phase 2c refuses to load any path identified as TEST")
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain unique values")
    if not 2 <= args.folds <= min(CLASS_COUNTS.values()):
        parser.error("--folds is outside the valid range")
    if len(args.bins_ms) != len(set(args.bins_ms)):
        parser.error("--bins-ms must contain unique values")
    if len(args.windows_ms) != len(set(args.windows_ms)):
        parser.error("--windows-ms must contain unique values")
    for value in (*args.bins_ms, *args.windows_ms):
        if value <= 0 or value % SAMPLE_INTERVAL_MS:
            parser.error("Every bin/window must be a positive multiple of 10 ms")
    if max(args.windows_ms) > AVAILABLE_SAMPLES * SAMPLE_INTERVAL_MS:
        parser.error("The official epochs provide at most 500 ms")
    if min(args.windows_ms) < 200:
        parser.error("This frozen feature family requires windows >= 200 ms")
    incompatible = [
        (bin_ms, window_ms)
        for bin_ms in args.bins_ms
        for window_ms in args.windows_ms
        if window_ms % bin_ms
    ]
    if incompatible:
        parser.error(f"Every window must contain whole bins; invalid={incompatible}")
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
        raise ValueError("Input contains non-finite values")
    if np.all(x[:, :-1, 28:] == x[:, 1:, :22]):
        raise ValueError("Detected the retired UEA sliding-channel layout error")
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


def causal_filter_full(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    filtered, _ = sosfilt(sos, x, axis=-1, zi=_initial_filter_state(x, sos))
    if not np.isfinite(filtered).all():
        raise FloatingPointError("Causal filtering produced non-finite values")
    return filtered


def causal_filter_chunked(
    x: np.ndarray, sos: np.ndarray, chunk_samples: int
) -> np.ndarray:
    state = _initial_filter_state(x, sos)
    chunks: list[np.ndarray] = []
    for start in range(0, x.shape[-1], chunk_samples):
        filtered, state = sosfilt(
            sos,
            x[..., start : start + chunk_samples],
            axis=-1,
            zi=state,
        )
        chunks.append(filtered)
    return np.concatenate(chunks, axis=-1)


def temporal_filter(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bp = causal_filter_full(x, BP_SOS)
    erd = causal_filter_full(x, ERD_SOS)
    return bp, erd


def _trace_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    trace = float(np.trace(matrix))
    if not np.isfinite(trace) or trace <= 1e-12:
        raise ValueError("Degenerate spatial matrix")
    return matrix / trace


def class_spatial_matrix(class_x: np.ndarray) -> np.ndarray:
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
    return filters


def project_cssd(x: np.ndarray, filters: np.ndarray) -> np.ndarray:
    return np.einsum("kc,nct->nkt", filters, x, optimize=True)


def bp_features(bp: np.ndarray, filters: np.ndarray) -> np.ndarray:
    return project_cssd(bp[..., -BP_RECENT_SAMPLES:], filters).reshape(len(bp), -1)


def erd_feature_window(window_samples: int) -> int:
    return min(ERD_MAX_RECENT_SAMPLES, window_samples)


def erd_features(erd: np.ndarray, filters: np.ndarray) -> np.ndarray:
    projected = np.abs(project_cssd(erd, filters))
    pools = [part.mean(axis=-1) for part in np.array_split(projected, ERD_POOL_COUNT, axis=-1)]
    return np.stack(pools, axis=-1).reshape(len(erd), -1)


def trend_features(bp: np.ndarray, retained_indices: np.ndarray) -> np.ndarray:
    selected = bp[:, retained_indices]
    oldest = selected[..., :TREND_OLDEST_SAMPLES].mean(axis=-1)
    recent = selected[..., -TREND_RECENT_SAMPLES:].mean(axis=-1)
    return np.stack([oldest, recent], axis=-1).reshape(len(bp), -1)


def make_lda() -> Any:
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="svd"))


def fit_fold_model(
    bp_full: np.ndarray,
    erd_full: np.ndarray,
    y: np.ndarray,
    retained_indices: np.ndarray,
    window_samples: int,
) -> FoldModel:
    bp = bp_full[..., -window_samples:]
    erd = erd_full[..., -window_samples:]
    erd_samples = erd_feature_window(window_samples)
    bp_filters = fit_cssd_filters(
        bp[..., -BP_RECENT_SAMPLES:], y, BP_PATTERNS_PER_CLASS
    )
    erd_filters = fit_cssd_filters(
        erd[..., -erd_samples:], y, ERD_PATTERNS_PER_CLASS
    )
    features = (
        bp_features(bp, bp_filters),
        erd_features(erd[..., -erd_samples:], erd_filters),
        trend_features(bp, retained_indices),
    )
    branches = tuple(make_lda() for _ in range(3))
    branch_scores: list[np.ndarray] = []
    for branch, values in zip(branches, features):
        branch.fit(values, y)
        branch_scores.append(np.asarray(branch.decision_function(values)).reshape(-1))
    fusion = make_lda()
    fusion.fit(np.column_stack(branch_scores), y)
    return FoldModel(
        bp_filters=bp_filters,
        erd_filters=erd_filters,
        bp_branch=branches[0],
        erd_branch=branches[1],
        trend_branch=branches[2],
        fusion=fusion,
        window_samples=window_samples,
    )


def predict(
    model: FoldModel,
    bp_full: np.ndarray,
    erd_full: np.ndarray,
    retained_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bp = bp_full[..., -model.window_samples:]
    erd = erd_full[..., -model.window_samples:]
    erd_samples = erd_feature_window(model.window_samples)
    features = (
        bp_features(bp, model.bp_filters),
        erd_features(erd[..., -erd_samples:], model.erd_filters),
        trend_features(bp, retained_indices),
    )
    branches = (model.bp_branch, model.erd_branch, model.trend_branch)
    scores = np.column_stack(
        [
            np.asarray(branch.decision_function(values)).reshape(-1)
            for branch, values in zip(branches, features)
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


def verify_bin_equivalence(x: np.ndarray, bins_ms: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bp_full = causal_filter_full(x, BP_SOS)
    erd_full = causal_filter_full(x, ERD_SOS)
    for bin_ms in bins_ms:
        chunk_samples = bin_ms // SAMPLE_INTERVAL_MS
        bp_chunked = causal_filter_chunked(x, BP_SOS, chunk_samples)
        erd_chunked = causal_filter_chunked(x, ERD_SOS, chunk_samples)
        error = float(
            max(
                np.max(np.abs(bp_full - bp_chunked)),
                np.max(np.abs(erd_full - erd_chunked)),
            )
        )
        if error > 1e-12:
            raise RuntimeError(f"{bin_ms} ms bin changed causal filtering")
        rows.append(
            {
                "bin_ms": bin_ms,
                "samples_per_bin": chunk_samples,
                "checked_cases": len(x),
                "maximum_filtered_signal_error": error,
                "passed": True,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def create_figure(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    bins = sorted({int(row["bin_ms"]) for row in summary_rows})
    windows = sorted({int(row["window_ms"]) for row in summary_rows})
    matrix = np.empty((len(bins), len(windows)), dtype=np.float64)
    for bin_index, bin_ms in enumerate(bins):
        for window_index, window_ms in enumerate(windows):
            row = next(
                item
                for item in summary_rows
                if item["bin_ms"] == bin_ms and item["window_ms"] == window_ms
            )
            matrix[bin_index, window_index] = 100.0 * row["mean_balanced_accuracy"]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    image = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=50.0, vmax=90.0)
    for row_index in range(len(bins)):
        for column_index in range(len(windows)):
            ax.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                color="white" if matrix[row_index, column_index] < 72 else "black",
            )
    ax.set_xticks(range(len(windows)), [str(value) for value in windows])
    ax.set_yticks(range(len(bins)), [str(value) for value in bins])
    ax.set_xlabel("Past-context window (ms)")
    ax.set_ylabel("Streaming bin (ms)")
    ax.set_title("Phase 2c TRAIN-only mean OOF balanced accuracy (%)")
    fig.colorbar(image, ax=ax, label="Balanced accuracy (%)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    x, y, source_index, channel_names = load_training_data(data_path)
    bp_filtered, erd_filtered = temporal_filter(x)
    retained_indices = np.asarray(
        [
            index
            for index, name in enumerate(channel_names.tolist())
            if name not in REJECTED_TREND_CHANNELS
        ],
        dtype=np.int64,
    )
    if retained_indices.size != 19:
        raise RuntimeError(f"Expected 19 retained channels, got {retained_indices.size}")

    print("=== FingerMovements Phase 2c causal bin/window sweep ===")
    print(f"data={data_path}")
    print(f"bins_ms={args.bins_ms} | windows_ms={args.windows_ms}")
    print(f"seeds={args.seeds} | folds={args.folds}")
    print("policy=TRAIN only; TEST refused; windows end at current point A")
    print("note=bin size changes streaming chunking, not endpoint information")

    bin_checks = verify_bin_equivalence(x[:64], args.bins_ms)
    seeds_to_run = args.seeds[:1] if args.validate_only else args.seeds
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for seed in seeds_to_run:
        folds = stratified_folds(y, args.folds, seed)
        if args.validate_only:
            folds = folds[:1]
        for fold_number, (training, validation) in enumerate(folds, start=1):
            for window_ms in args.windows_ms:
                window_samples = window_ms // SAMPLE_INTERVAL_MS
                bp_window = bp_filtered[..., -window_samples:]
                bp_window = bp_window - bp_window[..., :1]
                erd_window = erd_filtered[..., -window_samples:]
                started = perf_counter()
                model = fit_fold_model(
                    bp_window[training],
                    erd_window[training],
                    y[training],
                    retained_indices,
                    window_samples,
                )
                prediction, probability, decision = predict(
                    model,
                    bp_window[validation],
                    erd_window[validation],
                    retained_indices,
                )
                metrics = metric_bundle(y[validation], prediction, probability)
                elapsed = perf_counter() - started
                for bin_ms in args.bins_ms:
                    fold_rows.append(
                        {
                            "seed": int(seed),
                            "fold": int(fold_number),
                            "bin_ms": int(bin_ms),
                            "window_ms": int(window_ms),
                            "samples_per_bin": int(bin_ms // SAMPLE_INTERVAL_MS),
                            "bins_per_window": int(window_ms // bin_ms),
                            "training_cases": len(training),
                            "validation_cases": len(validation),
                            "accuracy": metrics["accuracy"],
                            "balanced_accuracy": metrics["balanced_accuracy"],
                            "macro_f1": metrics["macro_f1"],
                            "mean_log_loss": metrics["mean_log_loss"],
                            "model_fit_seconds": elapsed,
                        }
                    )
                    for local_index, case_index in enumerate(validation):
                        prediction_rows.append(
                            {
                                "seed": int(seed),
                                "fold": int(fold_number),
                                "bin_ms": int(bin_ms),
                                "window_ms": int(window_ms),
                                "source_index": int(source_index[case_index]),
                                "true_label": int(y[case_index]),
                                "predicted_label": int(prediction[local_index]),
                                "probability_right": float(probability[local_index]),
                                "decision_score": float(decision[local_index]),
                            }
                        )
                print(
                    f"seed={seed} fold={fold_number}/{args.folds} | "
                    f"window={window_ms} ms | BA={100.0 * metrics['balanced_accuracy']:.2f}%"
                )

    if args.validate_only:
        print("validate-only=PASS | all windows fitted for one fold")
        print("all requested bin chunking checks=PASS")
        print("no result files written")
        return

    seed_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        for bin_ms in args.bins_ms:
            for window_ms in args.windows_ms:
                selected = sorted(
                    (
                        row
                        for row in prediction_rows
                        if row["seed"] == seed
                        and row["bin_ms"] == bin_ms
                        and row["window_ms"] == window_ms
                    ),
                    key=lambda row: row["source_index"],
                )
                if len(selected) != CASES:
                    raise RuntimeError("OOF coverage is incomplete")
                indices = np.asarray([row["source_index"] for row in selected])
                if not np.array_equal(indices, np.arange(CASES)):
                    raise RuntimeError("OOF source indices are incomplete or duplicated")
                labels = np.asarray([row["true_label"] for row in selected])
                predictions = np.asarray(
                    [row["predicted_label"] for row in selected]
                )
                probabilities = np.asarray(
                    [row["probability_right"] for row in selected]
                )
                metrics = metric_bundle(labels, predictions, probabilities)
                seed_rows.append(
                    {
                        "seed": int(seed),
                        "bin_ms": int(bin_ms),
                        "window_ms": int(window_ms),
                        "accuracy": metrics["accuracy"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "mean_log_loss": metrics["mean_log_loss"],
                        "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
                    }
                )

    summary_rows: list[dict[str, Any]] = []
    for bin_ms in args.bins_ms:
        for window_ms in args.windows_ms:
            selected = [
                row
                for row in seed_rows
                if row["bin_ms"] == bin_ms and row["window_ms"] == window_ms
            ]
            balanced = np.asarray([row["balanced_accuracy"] for row in selected])
            summary_rows.append(
                {
                    "bin_ms": int(bin_ms),
                    "window_ms": int(window_ms),
                    "samples_per_bin": int(bin_ms // SAMPLE_INTERVAL_MS),
                    "bins_per_window": int(window_ms // bin_ms),
                    "mean_accuracy": float(
                        np.mean([row["accuracy"] for row in selected])
                    ),
                    "mean_balanced_accuracy": float(balanced.mean()),
                    "balanced_accuracy_seed_sd": float(balanced.std(ddof=0)),
                    "worst_seed_balanced_accuracy": float(balanced.min()),
                    "best_seed_balanced_accuracy": float(balanced.max()),
                    "mean_macro_f1": float(
                        np.mean([row["macro_f1"] for row in selected])
                    ),
                    "mean_log_loss": float(
                        np.mean([row["mean_log_loss"] for row in selected])
                    ),
                }
            )
    best_window_row = max(
        (row for row in summary_rows if row["bin_ms"] == args.bins_ms[0]),
        key=lambda row: (
            row["mean_balanced_accuracy"],
            row["worst_seed_balanced_accuracy"],
            -row["window_ms"],
        ),
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "phase2c_bin_window_fold_results.csv", fold_rows)
    write_csv(output_dir / "phase2c_bin_window_seed_results.csv", seed_rows)
    write_csv(output_dir / "phase2c_bin_window_summary.csv", summary_rows)
    write_csv(output_dir / "phase2c_bin_window_predictions.csv", prediction_rows)
    write_csv(output_dir / "phase2c_bin_equivalence_checks.csv", bin_checks)
    create_figure(summary_rows, output_dir / "phase2c_bin_window_sweep.png")
    payload = {
        "phase": "2c",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "causal past-context window and streaming-bin sweep",
        "data": {
            "path": str(data_path),
            "sha256": sha256(data_path),
            "cases": CASES,
            "shape": [CASES, CHANNELS, AVAILABLE_SAMPLES],
            "test_policy": "official TEST refused and not loaded",
        },
        "sweep": {
            "bins_ms": [int(value) for value in args.bins_ms],
            "windows_ms": [int(value) for value in args.windows_ms],
            "seeds": [int(value) for value in args.seeds],
            "folds": int(args.folds),
            "bin_interpretation": (
                "streaming chunk size only; endpoint features and predictions "
                "must remain identical for a fixed window"
            ),
            "window_interpretation": "past context ending at current point A",
        },
        "feature_policy": {
            "bp_recent_ms": BP_RECENT_SAMPLES * SAMPLE_INTERVAL_MS,
            "erd_recent_ms": (
                "min(320 ms, selected history window), split into four pools"
            ),
            "trend": "oldest 80 ms and newest 100 ms inside selected window",
            "causal_filter_state": (
                "carried from all available past samples; BP is causally "
                "re-referenced to the oldest sample in the selected ring"
            ),
        },
        "bin_equivalence_checks": bin_checks,
        "selection_rule": (
            "select history window by mean BA, then worst-seed BA, then shorter "
            "window; select bin separately from firmware latency constraints"
        ),
        "selected_window_ms": int(best_window_row["window_ms"]),
        "summary": summary_rows,
    }
    with (output_dir / "phase2c_bin_window_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print("\n=== Phase 2c sweep summary ===")
    for window_ms in args.windows_ms:
        row = next(
            item
            for item in summary_rows
            if item["bin_ms"] == args.bins_ms[0]
            and item["window_ms"] == window_ms
        )
        print(
            f"window={window_ms} ms | mean BA="
            f"{100.0 * row['mean_balanced_accuracy']:.2f}% | "
            f"seed SD={100.0 * row['balanced_accuracy_seed_sd']:.2f} pp | "
            f"worst={100.0 * row['worst_seed_balanced_accuracy']:.2f}%"
        )
    print(f"selected window={best_window_row['window_ms']} ms")
    print("bin-equivalence checks=PASS; choose bin from latency/firmware constraints")
    print(f"metrics={output_dir / 'phase2c_bin_window_metrics.json'}")


if __name__ == "__main__":
    main()
