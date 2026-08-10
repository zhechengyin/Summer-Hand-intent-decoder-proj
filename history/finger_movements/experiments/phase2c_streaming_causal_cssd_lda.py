"""Phase 2c: past-only rolling-window streaming CSSD + LDA evaluation.

For a prediction at point A, the model consumes only [A-500 ms, A].  The
500 ms window is historical context, not a post-A wait.  Firmware-style input
arrives in 50 ms bins with causal filter state carried between bins.  After one
startup warm-up, a prediction can be updated every 50 ms.  Official TEST is
refused.
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
HISTORY_SAMPLES = 50
SAMPLING_RATE_HZ = 100.0
SAMPLE_INTERVAL_MS = 10
HISTORY_MS = HISTORY_SAMPLES * SAMPLE_INTERVAL_MS
UPDATE_MS = 50
SAMPLES_PER_UPDATE = UPDATE_MS // SAMPLE_INTERVAL_MS
UPDATES_PER_WARMUP = HISTORY_SAMPLES // SAMPLES_PER_UPDATE
CLASS_COUNTS = {0: 159, 1: 157}
SEEDS = (42, 43, 44)
FOLDS = 5

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

OFFLINE_REFERENCE_MEAN_BA = 0.8672168142183766
PHASE2C_HORIZON_ENDPOINT_MEAN_BA = 0.8293073749148739

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


class StreamingCausalWindow:
    """Carry causal filter state and a 500 ms ring across 50 ms updates."""

    def __init__(self, first_sample: np.ndarray) -> None:
        first_sample = np.asarray(first_sample, dtype=np.float64)
        if first_sample.shape != (CHANNELS,):
            raise ValueError(f"Expected ({CHANNELS},) first sample")
        self.bp_state = sosfilt_zi(BP_SOS)[:, None, :] * first_sample[None, :, None]
        self.erd_state = sosfilt_zi(ERD_SOS)[:, None, :] * first_sample[None, :, None]
        self.bp_ring = np.empty((CHANNELS, 0), dtype=np.float64)
        self.erd_ring = np.empty((CHANNELS, 0), dtype=np.float64)

    def push(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        samples = np.asarray(samples, dtype=np.float64)
        if samples.shape != (CHANNELS, SAMPLES_PER_UPDATE):
            raise ValueError(
                f"Expected ({CHANNELS}, {SAMPLES_PER_UPDATE}) update, "
                f"got {samples.shape}"
            )
        bp, self.bp_state = sosfilt(BP_SOS, samples, axis=-1, zi=self.bp_state)
        erd, self.erd_state = sosfilt(ERD_SOS, samples, axis=-1, zi=self.erd_state)
        self.bp_ring = np.concatenate([self.bp_ring, bp], axis=-1)[
            ..., -HISTORY_SAMPLES:
        ]
        self.erd_ring = np.concatenate([self.erd_ring, erd], axis=-1)[
            ..., -HISTORY_SAMPLES:
        ]
        if self.bp_ring.shape[-1] < HISTORY_SAMPLES:
            return None
        bp_window = self.bp_ring - self.bp_ring[..., :1]
        return bp_window.copy(), self.erd_ring.copy()


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
        default=ARCHIVE_ROOT / "results/phase2c_streaming_causal",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run one fold plus streaming/causality checks and write nothing.",
    )
    args = parser.parse_args()
    if any("test" in part.lower() for part in args.data.parts):
        parser.error("Phase 2c refuses to load any path identified as TEST")
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain unique values")
    if not 2 <= args.folds <= min(CLASS_COUNTS.values()):
        parser.error("--folds is outside the valid range")
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

    if x.shape != (CASES, CHANNELS, HISTORY_SAMPLES):
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
    filtered, _ = sosfilt(
        sos,
        x,
        axis=-1,
        zi=_initial_filter_state(x, sos),
    )
    if not np.isfinite(filtered).all():
        raise FloatingPointError("Causal filtering produced non-finite values")
    return filtered


def causal_filter_streaming(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Process five samples at a time while carrying the IIR state."""
    state = _initial_filter_state(x, sos)
    chunks: list[np.ndarray] = []
    for start in range(0, x.shape[-1], SAMPLES_PER_UPDATE):
        stop = min(start + SAMPLES_PER_UPDATE, x.shape[-1])
        filtered, state = sosfilt(
            sos,
            x[..., start:stop],
            axis=-1,
            zi=state,
        )
        chunks.append(filtered)
    result = np.concatenate(chunks, axis=-1)
    if result.shape != x.shape or not np.isfinite(result).all():
        raise RuntimeError("Streaming filter output is invalid")
    return result


def temporal_filter_full(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bp = causal_filter_full(x, BP_SOS)
    bp = bp - bp[..., :1]
    erd = causal_filter_full(x, ERD_SOS)
    return bp, erd


def temporal_filter_streaming(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bp = causal_filter_streaming(x, BP_SOS)
    bp = bp - bp[..., :1]
    erd = causal_filter_streaming(x, ERD_SOS)
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
    if not np.isfinite(filters).all():
        raise FloatingPointError("CSSD produced non-finite filters")
    return filters


def project_cssd(x: np.ndarray, filters: np.ndarray) -> np.ndarray:
    return np.einsum("kc,nct->nkt", filters, x, optimize=True)


def bp_features(bp: np.ndarray, filters: np.ndarray) -> np.ndarray:
    features = project_cssd(bp[..., -BP_RECENT_SAMPLES:], filters).reshape(len(bp), -1)
    if features.shape != (len(bp), 8):
        raise RuntimeError(f"Unexpected BP feature shape: {features.shape}")
    return features


def erd_features(erd: np.ndarray, filters: np.ndarray) -> np.ndarray:
    projected = project_cssd(erd[..., -ERD_RECENT_SAMPLES:], filters)
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


def trend_features(bp: np.ndarray, retained_indices: np.ndarray) -> np.ndarray:
    selected = bp[:, retained_indices]
    oldest = selected[..., :TREND_OLDEST_SAMPLES].mean(axis=-1)
    recent = selected[..., -TREND_RECENT_SAMPLES:].mean(axis=-1)
    features = np.stack([oldest, recent], axis=-1).reshape(len(bp), -1)
    if features.shape != (len(bp), 38):
        raise RuntimeError(f"Unexpected trend feature shape: {features.shape}")
    return features


def make_lda() -> Any:
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="svd"))


def fit_fold_model(
    bp: np.ndarray,
    erd: np.ndarray,
    y: np.ndarray,
    retained_indices: np.ndarray,
) -> FoldModel:
    bp_filters = fit_cssd_filters(
        bp[..., -BP_RECENT_SAMPLES:], y, BP_PATTERNS_PER_CLASS
    )
    erd_filters = fit_cssd_filters(
        erd[..., -ERD_RECENT_SAMPLES:], y, ERD_PATTERNS_PER_CLASS
    )
    features = (
        bp_features(bp, bp_filters),
        erd_features(erd, erd_filters),
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
    )


def predict(
    model: FoldModel,
    bp: np.ndarray,
    erd: np.ndarray,
    retained_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = (
        bp_features(bp, model.bp_filters),
        erd_features(erd, model.erd_filters),
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


def verify_streaming_equivalence(
    raw_x: np.ndarray,
    model: FoldModel,
    retained_indices: np.ndarray,
) -> dict[str, Any]:
    bp_full, erd_full = temporal_filter_full(raw_x)
    bp_stream, erd_stream = temporal_filter_streaming(raw_x)
    signal_error = float(
        max(
            np.max(np.abs(bp_full - bp_stream)),
            np.max(np.abs(erd_full - erd_stream)),
        )
    )
    full_prediction, full_probability, full_score = predict(
        model, bp_full, erd_full, retained_indices
    )
    stream_prediction, stream_probability, stream_score = predict(
        model, bp_stream, erd_stream, retained_indices
    )
    score_error = float(np.max(np.abs(full_score - stream_score)))
    probability_error = float(np.max(np.abs(full_probability - stream_probability)))
    if not np.array_equal(full_prediction, stream_prediction):
        raise RuntimeError("Chunked streaming changed endpoint predictions")
    if max(signal_error, score_error, probability_error) > 1e-12:
        raise RuntimeError("Chunked streaming is not numerically equivalent")

    rolling_bp: list[np.ndarray] = []
    rolling_erd: list[np.ndarray] = []
    second_update_count = 0
    rng = np.random.default_rng(202_050)
    for case in raw_x:
        state = StreamingCausalWindow(case[:, 0])
        window: tuple[np.ndarray, np.ndarray] | None = None
        for update_number, start in enumerate(
            range(0, HISTORY_SAMPLES, SAMPLES_PER_UPDATE), start=1
        ):
            window = state.push(case[:, start : start + SAMPLES_PER_UPDATE])
            if update_number < UPDATES_PER_WARMUP and window is not None:
                raise RuntimeError("Rolling decoder emitted before warm-up")
        if window is None:
            raise RuntimeError("Rolling decoder did not emit at point A")
        rolling_bp.append(window[0])
        rolling_erd.append(window[1])
        synthetic_next_bin = rng.normal(size=(CHANNELS, SAMPLES_PER_UPDATE))
        if state.push(synthetic_next_bin) is not None:
            second_update_count += 1

    rolling_bp_array = np.stack(rolling_bp)
    rolling_erd_array = np.stack(rolling_erd)
    ring_error = float(
        max(
            np.max(np.abs(bp_full - rolling_bp_array)),
            np.max(np.abs(erd_full - rolling_erd_array)),
        )
    )
    if ring_error > 1e-12 or second_update_count != len(raw_x):
        raise RuntimeError("Rolling-buffer update behavior is invalid")
    return {
        "checked_cases": len(raw_x),
        "bins_per_window": UPDATES_PER_WARMUP,
        "maximum_filtered_signal_error": signal_error,
        "maximum_decision_score_error": score_error,
        "maximum_probability_error": probability_error,
        "maximum_ring_buffer_error": ring_error,
        "predictions_exact": True,
        "first_output_after_bins": UPDATES_PER_WARMUP,
        "next_output_after_one_more_bin": second_update_count == len(raw_x),
        "passed": True,
    }


def verify_post_a_invariance(
    raw_x: np.ndarray,
    model: FoldModel,
    retained_indices: np.ndarray,
) -> dict[str, Any]:
    rng = np.random.default_rng(202_000)
    future = rng.normal(loc=1e6, scale=1e5, size=(*raw_x.shape[:-1], 10))
    extended = np.concatenate([raw_x, future], axis=-1)
    bp_original, erd_original = temporal_filter_full(raw_x)
    bp_extended, erd_extended = temporal_filter_full(extended)
    bp_extended = bp_extended[..., :HISTORY_SAMPLES]
    erd_extended = erd_extended[..., :HISTORY_SAMPLES]
    original_prediction, original_probability, original_score = predict(
        model, bp_original, erd_original, retained_indices
    )
    extended_prediction, extended_probability, extended_score = predict(
        model, bp_extended, erd_extended, retained_indices
    )
    signal_error = float(
        max(
            np.max(np.abs(bp_original - bp_extended)),
            np.max(np.abs(erd_original - erd_extended)),
        )
    )
    score_error = float(np.max(np.abs(original_score - extended_score)))
    probability_error = float(
        np.max(np.abs(original_probability - extended_probability))
    )
    if not np.array_equal(original_prediction, extended_prediction):
        raise RuntimeError("Samples after point A changed predictions at A")
    if max(signal_error, score_error, probability_error) != 0.0:
        raise RuntimeError("Samples after point A influenced inference at A")
    return {
        "checked_cases": len(raw_x),
        "synthetic_future_samples_after_a": future.shape[-1],
        "maximum_filtered_signal_error": signal_error,
        "maximum_decision_score_error": score_error,
        "maximum_probability_error": probability_error,
        "predictions_exact": True,
        "passed": True,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def create_figure(seed_rows: list[dict[str, Any]], output_path: Path) -> None:
    seeds = [str(row["seed"]) for row in seed_rows]
    values = 100.0 * np.asarray([row["balanced_accuracy"] for row in seed_rows])
    mean = float(values.mean())
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(seeds, values, color="#4c72b0", label="streaming causal")
    ax.axhline(mean, color="#4c72b0", linestyle="--", label=f"mean={mean:.2f}%")
    ax.axhline(
        100.0 * OFFLINE_REFERENCE_MEAN_BA,
        color="#c44e52",
        linestyle=":",
        label="offline zero-phase reference",
    )
    ax.set_ylim(max(0.0, values.min() - 5.0), 100.0)
    ax.set_xlabel("Cross-validation seed")
    ax.set_ylabel("OOF balanced accuracy (%)")
    ax.set_title("Phase 2c: 500 ms past context, 50 ms streaming update")
    ax.legend()
    for position, value in enumerate(values):
        ax.text(position, value + 0.35, f"{value:.2f}%", ha="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    x, y, source_index, channel_names = load_training_data(data_path)
    bp, erd = temporal_filter_full(x)
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

    print("=== FingerMovements Phase 2c past-only streaming causal model ===")
    print(f"data={data_path}")
    print(f"cases={CASES} | input={CHANNELS}x{HISTORY_SAMPLES} @ 100 Hz")
    print(f"prediction point=A | context=[A-{HISTORY_MS} ms, A]")
    print(
        f"stream update={UPDATE_MS} ms ({SAMPLES_PER_UPDATE} samples) | "
        f"startup warm-up={UPDATES_PER_WARMUP} bins once"
    )
    print(f"seeds={args.seeds} | folds={args.folds}")
    print("policy=TRAIN only; TEST refused; no sample after A is consumed")

    seeds_to_run = args.seeds[:1] if args.validate_only else args.seeds
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    streaming_check: dict[str, Any] | None = None
    post_a_check: dict[str, Any] | None = None

    for seed in seeds_to_run:
        folds = stratified_folds(y, args.folds, seed)
        if args.validate_only:
            folds = folds[:1]
        for fold_number, (training, validation) in enumerate(folds, start=1):
            started = perf_counter()
            model = fit_fold_model(
                bp[training], erd[training], y[training], retained_indices
            )
            prediction, probability, decision = predict(
                model, bp[validation], erd[validation], retained_indices
            )
            metrics = metric_bundle(y[validation], prediction, probability)
            fold_rows.append(
                {
                    "seed": int(seed),
                    "fold": int(fold_number),
                    "training_cases": len(training),
                    "validation_cases": len(validation),
                    "history_ms": HISTORY_MS,
                    "update_ms": UPDATE_MS,
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
                        "source_index": int(source_index[case_index]),
                        "true_label": int(y[case_index]),
                        "predicted_label": int(prediction[local_index]),
                        "probability_right": float(probability[local_index]),
                        "decision_score": float(decision[local_index]),
                    }
                )
            if seed == seeds_to_run[0] and fold_number == 1:
                streaming_check = verify_streaming_equivalence(
                    x[validation], model, retained_indices
                )
                post_a_check = verify_post_a_invariance(
                    x[validation], model, retained_indices
                )
            print(
                f"seed={seed} fold={fold_number}/{args.folds} | "
                f"BA={100.0 * metrics['balanced_accuracy']:.2f}% | "
                f"loss={metrics['mean_log_loss']:.4f}"
            )

    if streaming_check is None or post_a_check is None:
        raise RuntimeError("Streaming or causality verification did not run")
    if args.validate_only:
        print("validation-only=PASS")
        print("chunked streaming equivalence=PASS")
        print("post-A future invariance=PASS")
        print("no result files written")
        return

    seed_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        selected = sorted(
            (row for row in prediction_rows if row["seed"] == seed),
            key=lambda row: row["source_index"],
        )
        if len(selected) != CASES:
            raise RuntimeError(f"seed={seed} has {len(selected)} OOF cases")
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
                "evaluated_cases": CASES,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "mean_log_loss": metrics["mean_log_loss"],
                "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
            }
        )

    balanced = np.asarray([row["balanced_accuracy"] for row in seed_rows])
    summary = {
        "mean_accuracy": float(np.mean([row["accuracy"] for row in seed_rows])),
        "mean_balanced_accuracy": float(balanced.mean()),
        "balanced_accuracy_seed_sd": float(balanced.std(ddof=0)),
        "worst_seed_balanced_accuracy": float(balanced.min()),
        "best_seed_balanced_accuracy": float(balanced.max()),
        "mean_macro_f1": float(np.mean([row["macro_f1"] for row in seed_rows])),
        "delta_vs_offline_reference": float(
            balanced.mean() - OFFLINE_REFERENCE_MEAN_BA
        ),
        "delta_vs_phase2c_horizon_endpoint": float(
            balanced.mean() - PHASE2C_HORIZON_ENDPOINT_MEAN_BA
        ),
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "phase2c_fold_results.csv", fold_rows)
    write_csv(output_dir / "phase2c_seed_results.csv", seed_rows)
    write_csv(output_dir / "phase2c_predictions.csv", prediction_rows)
    write_csv(
        output_dir / "phase2c_streaming_checks.csv",
        [
            {"check": "chunked_streaming_equivalence", **streaming_check},
            {"check": "post_a_future_invariance", **post_a_check},
        ],
    )
    create_figure(seed_rows, output_dir / "phase2c_streaming_causal.png")

    payload = {
        "phase": "2c",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "500 ms past-only rolling context with 50 ms streaming updates",
        "data": {
            "path": str(data_path),
            "sha256": sha256(data_path),
            "cases": CASES,
            "shape": [CASES, CHANNELS, HISTORY_SAMPLES],
            "test_policy": "official TEST refused and not loaded",
        },
        "timing_contract": {
            "prediction_point": "A",
            "input_interval": "[A-500 ms, A]",
            "future_after_a_consumed": False,
            "history_ms": HISTORY_MS,
            "update_ms": UPDATE_MS,
            "samples_per_update": SAMPLES_PER_UPDATE,
            "startup_warmup_bins": UPDATES_PER_WARMUP,
            "steady_state_output": "one update every 50 ms after warm-up",
        },
        "filter_state": {
            "implementation": "causal SOS IIR state carried across 50 ms bins",
            "evaluation_initialization": (
                "first sample of each isolated official epoch; the dataset does "
                "not include preceding continuous EEG"
            ),
            "deployment_requirement": (
                "carry state continuously; validate with continuous recordings"
            ),
        },
        "validation": {
            "seeds": [int(seed) for seed in args.seeds],
            "folds": int(args.folds),
            "split_unit": "whole trial/case",
            "all_learned_operations": "outer-training-fold only",
        },
        "model": {
            "covariance": "empirical",
            "trial_trace_normalization": True,
            "bp_patterns_per_class": BP_PATTERNS_PER_CLASS,
            "erd_patterns_per_class": ERD_PATTERNS_PER_CLASS,
            "fusion": "hierarchical LDA",
        },
        "streaming_equivalence_check": streaming_check,
        "post_a_future_invariance_check": post_a_check,
        "offline_reference_mean_oof_balanced_accuracy": OFFLINE_REFERENCE_MEAN_BA,
        "phase2c_horizon_endpoint_mean_oof_balanced_accuracy": (
            PHASE2C_HORIZON_ENDPOINT_MEAN_BA
        ),
        "seed_results": seed_rows,
        "summary": summary,
    }
    with (output_dir / "phase2c_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print("\n=== Phase 2c summary ===")
    print(
        f"mean OOF BA={100.0 * summary['mean_balanced_accuracy']:.2f}% | "
        f"seed SD={100.0 * summary['balanced_accuracy_seed_sd']:.2f} pp | "
        f"worst={100.0 * summary['worst_seed_balanced_accuracy']:.2f}%"
    )
    print(
        f"delta vs offline={100.0 * summary['delta_vs_offline_reference']:+.2f} pp | "
        "delta vs initial Phase 2c horizon endpoint="
        f"{100.0 * summary['delta_vs_phase2c_horizon_endpoint']:+.2f} pp"
    )
    print("chunked streaming equivalence=PASS")
    print("post-A future invariance=PASS")
    print(f"metrics={output_dir / 'phase2c_metrics.json'}")


if __name__ == "__main__":
    main()
