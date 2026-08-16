"""Phase 1f: factorial comparison of EEG representations and linear classifiers.

Four representations are crossed with two classifiers for eight pipelines:

1. current 196 handcrafted features;
2. causal 5 Hz low-pass terminal features;
3. current and terminal features concatenated;
4. 0--14 Hz Fourier coefficients reduced by fold-training-only PCA.

Each representation is evaluated with fixed-C L2 Logistic Regression and
shrinkage Fisher/LDA. Every learned preprocessing step is fitted only from the
current training fold. The official test split remains locked and is never
opened.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
from scipy.stats import binomtest
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[2]

CASES = 316
CHANNELS = 28
TIMEPOINTS = 50
SAMPLING_RATE_HZ = 100.0
CLASS_COUNTS = {0: 159, 1: 157}

SEEDS = (42, 43, 44)
FOLDS = 5
LOGISTIC_C = 1.0

LOWPASS_HZ = 5.0
LOWPASS_ORDER = 2
TERMINAL_SAMPLES = 5
TERMINAL_MEAN_WINDOWS = (5, 10, 20)
TERMINAL_SLOPE_WINDOW = 20
FOURIER_MAX_HZ = 14.0
PCA_COMPONENTS = 20

REPRESENTATIONS = ("current", "terminal", "combined", "fourier_pca")
CLASSIFIERS = ("logistic", "fisher")
PIPELINES = tuple(
    f"{representation}__{classifier}"
    for representation in REPRESENTATIONS
    for classifier in CLASSIFIERS
)
BASELINE_PIPELINE = "current__logistic"

REPRESENTATION_LABELS = {
    "current": "Current 196",
    "terminal": "Terminal low-pass",
    "combined": "Current + terminal",
    "fourier_pca": "Fourier + PCA",
}
CLASSIFIER_LABELS = {
    "logistic": "Logistic",
    "fisher": "Shrinkage Fisher",
}

BANDS_HZ = ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0))

FOURIER_FREQUENCIES = np.fft.rfftfreq(TIMEPOINTS, d=1.0 / SAMPLING_RATE_HZ)
FOURIER_MASK = FOURIER_FREQUENCIES <= FOURIER_MAX_HZ
FOURIER_BINS = int(FOURIER_MASK.sum())
CURRENT_FEATURES = CHANNELS * (3 + len(BANDS_HZ))
TERMINAL_FEATURES = CHANNELS * (
    TERMINAL_SAMPLES + len(TERMINAL_MEAN_WINDOWS) + 1
)
FOURIER_FEATURES = CHANNELS * (1 + 2 * (FOURIER_BINS - 1))
LOWPASS_SOS = butter(
    LOWPASS_ORDER,
    LOWPASS_HZ,
    btype="lowpass",
    fs=SAMPLING_RATE_HZ,
    output="sos",
)
LOWPASS_INITIAL = sosfilt_zi(LOWPASS_SOS)


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
        default=ROOT / "results/finger_movements/phase1f_low_frequency_factorial",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--pca-components", type=int, default=PCA_COMPONENTS)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if "test" in args.data.name.lower():
        parser.error("Phase 1f refuses to load any file identified as a test split")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds contains duplicates")
    if args.folds < 2 or args.folds > min(CLASS_COUNTS.values()):
        parser.error("--folds is outside the valid range")
    max_training_cases = CASES - CASES // args.folds
    if not 1 <= args.pca_components <= min(max_training_cases, FOURIER_FEATURES):
        parser.error("--pca-components is outside the valid range")
    return args


def load_training_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Training data not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {"x", "y", "source_index"}
        if not required.issubset(data.files):
            raise KeyError(f"Missing arrays: {sorted(required - set(data.files))}")
        x = data["x"].astype(np.float32, copy=True)
        y = data["y"].astype(np.int64, copy=True)
        source_index = data["source_index"].astype(np.int64, copy=True)
    if x.shape != (CASES, CHANNELS, TIMEPOINTS) or y.shape != (CASES,):
        raise ValueError(f"Unexpected data shapes: x={x.shape}, y={y.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Input contains non-finite values")
    values, counts = np.unique(y, return_counts=True)
    observed = dict(zip(values.tolist(), counts.tolist(), strict=True))
    if observed != CLASS_COUNTS:
        raise ValueError(f"Unexpected labels or class counts: {observed}")
    if not np.array_equal(source_index, np.arange(CASES)):
        raise ValueError("source_index must preserve canonical TRAIN.ts order")
    return x, y, source_index


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
    output = []
    for fold in range(fold_count):
        validation = np.concatenate([pieces[label][fold] for label in pieces])
        training = np.setdiff1d(all_indices, validation, assume_unique=True)
        fold_rng = np.random.default_rng(seed * 10_000 + fold)
        fold_rng.shuffle(training)
        fold_rng.shuffle(validation)
        output.append((training, validation))
    return output


def fit_channel_normalization(training_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = training_x.mean(axis=(0, 2), keepdims=True, dtype=np.float64)
    std = np.maximum(
        training_x.std(axis=(0, 2), keepdims=True, dtype=np.float64), 1e-6
    )
    return mean, std


def apply_channel_normalization(
    x: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def standardize_pair(
    training: np.ndarray, validation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = training.mean(axis=0, keepdims=True, dtype=np.float64)
    std = np.maximum(training.std(axis=0, keepdims=True, dtype=np.float64), 1e-6)
    return (
        ((training - mean) / std).astype(np.float32),
        ((validation - mean) / std).astype(np.float32),
    )


def current_features(normalized_x: np.ndarray) -> np.ndarray:
    blocks = [
        normalized_x.mean(axis=-1),
        normalized_x.std(axis=-1),
        np.square(normalized_x).mean(axis=-1),
    ]
    spectrum = np.square(np.abs(np.fft.rfft(normalized_x, axis=-1))) / TIMEPOINTS
    frequencies = np.fft.rfftfreq(TIMEPOINTS, d=1.0 / SAMPLING_RATE_HZ)
    for low, high in BANDS_HZ:
        mask = (frequencies >= low) & (frequencies < high)
        blocks.append(spectrum[..., mask].mean(axis=-1))
    output = np.concatenate(blocks, axis=1).astype(np.float32)
    if output.shape[1] != CURRENT_FEATURES:
        raise RuntimeError(f"Unexpected current feature shape: {output.shape}")
    return output


def causal_lowpass(normalized_x: np.ndarray) -> np.ndarray:
    """Apply a causal filter, initialized from each trial's first sample."""
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
    filtered = causal_lowpass(normalized_x)
    last_values = filtered[..., -TERMINAL_SAMPLES:].reshape(len(filtered), -1)
    means = [filtered[..., -window:].mean(axis=-1) for window in TERMINAL_MEAN_WINDOWS]
    time = np.arange(TERMINAL_SLOPE_WINDOW, dtype=np.float64)
    centered_time = time - time.mean()
    slope = np.tensordot(
        filtered[..., -TERMINAL_SLOPE_WINDOW:],
        centered_time,
        axes=([-1], [0]),
    ) / np.square(centered_time).sum()
    output = np.concatenate([last_values, *means, slope], axis=1).astype(np.float32)
    if output.shape[1] != TERMINAL_FEATURES:
        raise RuntimeError(f"Unexpected terminal feature shape: {output.shape}")
    return output


def fourier_features(normalized_x: np.ndarray) -> np.ndarray:
    coefficients = np.fft.rfft(normalized_x, axis=-1) / TIMEPOINTS
    selected = coefficients[..., FOURIER_MASK]
    output = np.concatenate(
        [
            selected[..., :1].real,
            selected[..., 1:].real,
            selected[..., 1:].imag,
        ],
        axis=-1,
    ).reshape(len(normalized_x), -1)
    output = output.astype(np.float32)
    if output.shape[1] != FOURIER_FEATURES:
        raise RuntimeError(f"Unexpected Fourier feature shape: {output.shape}")
    return output


def prepare_representations(
    training_x: np.ndarray,
    validation_x: np.ndarray,
    pca_components: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    channel_mean, channel_std = fit_channel_normalization(training_x)
    normalized_training = apply_channel_normalization(
        training_x, channel_mean, channel_std
    )
    normalized_validation = apply_channel_normalization(
        validation_x, channel_mean, channel_std
    )

    current_training = current_features(normalized_training)
    current_validation = current_features(normalized_validation)
    terminal_training = terminal_features(normalized_training)
    terminal_validation = terminal_features(normalized_validation)

    raw_representations = {
        "current": (current_training, current_validation),
        "terminal": (terminal_training, terminal_validation),
        "combined": (
            np.concatenate([current_training, terminal_training], axis=1),
            np.concatenate([current_validation, terminal_validation], axis=1),
        ),
    }
    output = {
        name: standardize_pair(training, validation)
        for name, (training, validation) in raw_representations.items()
    }

    fourier_training = fourier_features(normalized_training)
    fourier_validation = fourier_features(normalized_validation)
    pca = PCA(n_components=pca_components, svd_solver="full")
    pca_training = pca.fit_transform(fourier_training)
    pca_validation = pca.transform(fourier_validation)
    output["fourier_pca"] = standardize_pair(pca_training, pca_validation)
    return output


def fit_classifier(
    classifier_name: str,
    training_features: np.ndarray,
    training_y: np.ndarray,
    validation_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    if classifier_name == "logistic":
        classifier: Any = LogisticRegression(
            C=LOGISTIC_C,
            solver="liblinear",
            max_iter=5_000,
        )
    elif classifier_name == "fisher":
        classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")
    started = perf_counter()
    classifier.fit(training_features, training_y)
    fit_seconds = perf_counter() - started
    prediction = classifier.predict(validation_features).astype(np.int64)
    score = np.asarray(classifier.decision_function(validation_features)).reshape(-1)
    training_accuracy = float(
        np.mean(classifier.predict(training_features) == training_y)
    )
    if not np.isfinite(score).all():
        raise RuntimeError(f"{classifier_name} produced non-finite scores")
    return prediction, score.astype(np.float64), training_accuracy, fit_seconds


def classification_metrics(
    actual: np.ndarray, predicted: np.ndarray
) -> dict[str, Any]:
    confusion = np.zeros((2, 2), dtype=np.int64)
    for truth, guess in zip(actual, predicted, strict=True):
        confusion[int(truth), int(guess)] += 1
    recalls = []
    f1_scores = []
    for label in (0, 1):
        true_positive = float(confusion[label, label])
        false_negative = float(confusion[label].sum() - true_positive)
        false_positive = float(confusion[:, label].sum() - true_positive)
        recall = true_positive / max(true_positive + false_negative, 1.0)
        precision = true_positive / max(true_positive + false_positive, 1.0)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        recalls.append(recall)
        f1_scores.append(f1)
    return {
        "accuracy": float(np.mean(actual == predicted)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
        "confusion_matrix": confusion.tolist(),
    }


def summarize_pipelines(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for pipeline in PIPELINES:
        rows = [row for row in seed_rows if row["pipeline"] == pipeline]
        representation, classifier = pipeline.split("__")
        summary: dict[str, Any] = {
            "pipeline": pipeline,
            "representation": representation,
            "classifier": classifier,
            "feature_count": rows[0]["feature_count"],
            "seeds": len(rows),
        }
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=1))
            summary[f"{metric}_min"] = float(values.min())
            summary[f"{metric}_max"] = float(values.max())
        summaries.append(summary)
    return summaries


def paired_comparison(
    actual: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    seed: int,
    candidate_pipeline: str,
) -> dict[str, Any]:
    reference_correct = reference == actual
    candidate_correct = candidate == actual
    reference_only = int(np.sum(reference_correct & ~candidate_correct))
    candidate_only = int(np.sum(~reference_correct & candidate_correct))
    discordant = reference_only + candidate_only
    p_value = (
        float(
            binomtest(
                min(reference_only, candidate_only),
                discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
        if discordant
        else 1.0
    )
    return {
        "seed": seed,
        "reference_pipeline": BASELINE_PIPELINE,
        "candidate_pipeline": candidate_pipeline,
        "candidate_minus_reference_accuracy": float(
            candidate_correct.mean() - reference_correct.mean()
        ),
        "reference_only_correct": reference_only,
        "candidate_only_correct": candidate_only,
        "both_correct": int(np.sum(reference_correct & candidate_correct)),
        "both_wrong": int(np.sum(~reference_correct & ~candidate_correct)),
        "mcnemar_exact_p": p_value,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(path: Path, summaries: list[dict[str, Any]]) -> None:
    ordered = sorted(
        summaries,
        key=lambda row: (-row["balanced_accuracy_mean"], row["pipeline"]),
    )
    labels = [
        f"{REPRESENTATION_LABELS[row['representation']]}\n"
        f"{CLASSIFIER_LABELS[row['classifier']]}"
        for row in ordered
    ]
    means = 100.0 * np.asarray(
        [row["balanced_accuracy_mean"] for row in ordered], dtype=np.float64
    )
    deviations = 100.0 * np.asarray(
        [row["balanced_accuracy_std"] for row in ordered], dtype=np.float64
    )
    worst = 100.0 * np.asarray(
        [row["balanced_accuracy_min"] for row in ordered], dtype=np.float64
    )

    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = [
        "#4C78A8" if row["classifier"] == "logistic" else "#F58518"
        for row in ordered
    ]
    positions = np.arange(len(ordered))
    axes[0].bar(positions, means, yerr=deviations, capsize=4, color=colors)
    axes[0].scatter(positions, worst, marker="v", color="black", label="Worst seed")
    axes[0].set_xticks(positions, labels, rotation=35, ha="right")
    axes[0].set_ylabel("OOF balanced accuracy (%)")
    axes[0].set_title("Pipeline ranking: mean ± seed SD")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    matrix = np.zeros((len(REPRESENTATIONS), len(CLASSIFIERS)), dtype=np.float64)
    for row in summaries:
        representation_index = REPRESENTATIONS.index(row["representation"])
        classifier_index = CLASSIFIERS.index(row["classifier"])
        matrix[representation_index, classifier_index] = (
            100.0 * row["balanced_accuracy_mean"]
        )
    image = axes[1].imshow(matrix, cmap="viridis", aspect="auto")
    axes[1].set_xticks(
        np.arange(len(CLASSIFIERS)),
        [CLASSIFIER_LABELS[name] for name in CLASSIFIERS],
    )
    axes[1].set_yticks(
        np.arange(len(REPRESENTATIONS)),
        [REPRESENTATION_LABELS[name] for name in REPRESENTATIONS],
    )
    axes[1].set_title("Mean OOF balanced accuracy (%)")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes[1].text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] < matrix.mean() else "black",
            )
    figure.colorbar(image, ax=axes[1], shrink=0.8)
    figure.suptitle("FingerMovements Phase 1f: representation × classifier")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def validate_only(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    folds: int,
    pca_components: int,
) -> None:
    training, validation = stratified_folds(y, folds, seed)[0]
    representations = prepare_representations(
        x[training], x[validation], pca_components
    )
    print("=== FingerMovements Phase 1f validation-only ===")
    for representation in REPRESENTATIONS:
        training_features, validation_features = representations[representation]
        for classifier in CLASSIFIERS:
            prediction, score, _, _ = fit_classifier(
                classifier,
                training_features,
                y[training],
                validation_features,
            )
            if prediction.shape != (len(validation),) or score.shape != (
                len(validation),
            ):
                raise RuntimeError(
                    f"Validation failed for {representation}__{classifier}"
                )
        print(
            f"{representation}: train={training_features.shape} | "
            f"validation={validation_features.shape} | classifiers=2 OK"
        )
    print("test: LOCKED AND NOT LOADED")


def main() -> None:
    args = parse_args()
    x, y, source_index = load_training_data(args.data.resolve())
    if args.validate_only:
        validate_only(x, y, args.seeds[0], args.folds, args.pca_components)
        return

    print("=== FingerMovements Phase 1f low-frequency factorial ===")
    print(f"cases={len(y)} | seeds={args.seeds} | folds={args.folds}")
    print(f"representations={REPRESENTATIONS}")
    print(f"classifiers={CLASSIFIERS} | pipelines={len(PIPELINES)}")
    print(
        f"low-pass={LOWPASS_ORDER}th-order causal {LOWPASS_HZ:g} Hz | "
        f"Fourier=0-{FOURIER_MAX_HZ:g} Hz | PCA={args.pca_components}"
    )
    print("all preprocessing: fitted from each training fold only")
    print("test: LOCKED AND NOT LOADED")

    fold_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []

    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        oof_predictions = {
            pipeline: np.full(len(y), -1, dtype=np.int64) for pipeline in PIPELINES
        }
        oof_scores = {
            pipeline: np.full(len(y), np.nan, dtype=np.float64)
            for pipeline in PIPELINES
        }
        feature_counts: dict[str, int] = {}

        for fold, (training, validation) in enumerate(
            stratified_folds(y, args.folds, seed), start=1
        ):
            representations = prepare_representations(
                x[training], x[validation], args.pca_components
            )
            fold_text = []
            for representation in REPRESENTATIONS:
                training_features, validation_features = representations[
                    representation
                ]
                feature_counts[representation] = training_features.shape[1]
                for classifier in CLASSIFIERS:
                    pipeline = f"{representation}__{classifier}"
                    prediction, score, training_accuracy, fit_seconds = fit_classifier(
                        classifier,
                        training_features,
                        y[training],
                        validation_features,
                    )
                    oof_predictions[pipeline][validation] = prediction
                    oof_scores[pipeline][validation] = score
                    metrics = classification_metrics(y[validation], prediction)
                    fold_rows.append(
                        {
                            "pipeline": pipeline,
                            "representation": representation,
                            "classifier": classifier,
                            "seed": seed,
                            "fold": fold,
                            "training_cases": len(training),
                            "validation_cases": len(validation),
                            "feature_count": training_features.shape[1],
                            "training_accuracy": training_accuracy,
                            "fit_seconds": fit_seconds,
                            **metrics,
                        }
                    )
                    fold_text.append(
                        f"{pipeline}={metrics['balanced_accuracy']:.3f}"
                    )
            print(f"fold {fold}/{args.folds} | " + " | ".join(fold_text))

        for pipeline in PIPELINES:
            if np.any(oof_predictions[pipeline] < 0) or not np.isfinite(
                oof_scores[pipeline]
            ).all():
                raise RuntimeError(f"Incomplete OOF results for {pipeline}, seed={seed}")
            representation, classifier = pipeline.split("__")
            metrics = classification_metrics(y, oof_predictions[pipeline])
            seed_rows.append(
                {
                    "pipeline": pipeline,
                    "representation": representation,
                    "classifier": classifier,
                    "seed": seed,
                    "folds": args.folds,
                    "feature_count": feature_counts[representation],
                    **metrics,
                }
            )
            for index in range(len(y)):
                prediction_rows.append(
                    {
                        "pipeline": pipeline,
                        "seed": seed,
                        "source_index": int(source_index[index]),
                        "true_label": int(y[index]),
                        "predicted_label": int(oof_predictions[pipeline][index]),
                        "score": float(oof_scores[pipeline][index]),
                    }
                )
            print(
                f"seed summary | {pipeline} | "
                f"BA={metrics['balanced_accuracy']:.4f} | "
                f"accuracy={metrics['accuracy']:.4f}"
            )

        reference = oof_predictions[BASELINE_PIPELINE]
        for pipeline in PIPELINES:
            if pipeline == BASELINE_PIPELINE:
                continue
            paired_rows.append(
                paired_comparison(
                    y,
                    reference,
                    oof_predictions[pipeline],
                    seed,
                    pipeline,
                )
            )

    pipeline_summary = summarize_pipelines(seed_rows)
    ranking = sorted(
        pipeline_summary,
        key=lambda row: (
            -row["balanced_accuracy_mean"],
            -row["balanced_accuracy_min"],
            row["feature_count"],
            row["pipeline"],
        ),
    )
    print("\n=== pipeline ranking ===")
    for rank, row in enumerate(ranking, start=1):
        print(
            f"{rank}. {row['pipeline']} | features={row['feature_count']} | "
            f"BA={row['balanced_accuracy_mean']:.4f} ± "
            f"{row['balanced_accuracy_std']:.4f} | "
            f"worst={row['balanced_accuracy_min']:.4f}"
        )
    print("decision: review stability and paired results before changing the baseline")

    report = {
        "phase": "1f",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "official TRAIN split only; official TEST not opened",
        "protocol": {
            "cases": len(y),
            "seeds": args.seeds,
            "folds": args.folds,
            "representations": list(REPRESENTATIONS),
            "classifiers": list(CLASSIFIERS),
            "pipelines": list(PIPELINES),
            "baseline_pipeline": BASELINE_PIPELINE,
            "logistic_c": LOGISTIC_C,
            "fisher": "LinearDiscriminantAnalysis lsqr with automatic shrinkage",
            "lowpass_hz": LOWPASS_HZ,
            "lowpass_order": LOWPASS_ORDER,
            "lowpass_policy": "causal; initialized from first sample",
            "terminal_samples": TERMINAL_SAMPLES,
            "terminal_mean_windows": list(TERMINAL_MEAN_WINDOWS),
            "terminal_slope_window": TERMINAL_SLOPE_WINDOW,
            "fourier_max_hz": FOURIER_MAX_HZ,
            "pca_components": args.pca_components,
            "fold_training_only_preprocessing": True,
            "selection_metric": "mean OOF balanced accuracy across seeds",
            "tie_breakers": [
                "higher worst-seed balanced accuracy",
                "fewer features",
            ],
        },
        "feature_counts": {
            "current": CURRENT_FEATURES,
            "terminal": TERMINAL_FEATURES,
            "combined": CURRENT_FEATURES + TERMINAL_FEATURES,
            "fourier_pca": args.pca_components,
        },
        "pipeline_summary": pipeline_summary,
        "ranking": [row["pipeline"] for row in ranking],
        "paired_comparisons": paired_rows,
        "decision_policy": (
            "Attribute representation gains only when both classifier and paired "
            "results are reviewed. Prefer a replacement over the current Logistic "
            "baseline only for meaningful, stable development-set improvement."
        ),
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "phase1f_fold_results.csv", fold_rows)
    write_csv(output_dir / "phase1f_seed_results.csv", seed_rows)
    write_csv(output_dir / "phase1f_pipeline_summary.csv", pipeline_summary)
    write_csv(output_dir / "phase1f_predictions.csv", prediction_rows)
    write_csv(output_dir / "phase1f_paired_comparisons.csv", paired_rows)
    metrics_path = output_dir / "phase1f_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    figure_path = output_dir / "phase1f_factorial_comparison.png"
    save_figure(figure_path, pipeline_summary)
    print(f"metrics: {metrics_path}")
    print(f"figure: {figure_path}")

if __name__ == "__main__":
    main()
