"""Phase 1e upper refinement: fixed-C Logistic Regression from C=1 to C=5.

Every candidate C is evaluated on exactly the same repeated stratified folds.
This second Phase 1e check uses one fixed C across all folds, matching the
eventual deployment contract, after the broad nested-CV sweep showed unstable
per-fold selection. The official test split remains locked and is never opened.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.finger_movements.feature_logistic.model import (  # noqa: E402
    CURRENT_C,
    fit_preprocessing,
    transform,
)


CASES = 316
CHANNELS = 28
TIMEPOINTS = 50
CLASS_COUNTS = {0: 159, 1: 157}
SEEDS = (42, 43, 44)
FOLDS = 5
C_VALUES = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)


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
        default=ROOT / "results/finger_movements/phase1e_logistic_upper_refinement",
    )
    parser.add_argument("--c-values", nargs="+", type=float, default=list(C_VALUES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.data.name.lower() == "test.npz":
        parser.error("Phase 1e upper refinement refuses to load test.npz")
    if any(value <= 0 for value in args.c_values):
        parser.error("Every --c-values entry must be positive")
    if len(args.c_values) != len(set(args.c_values)):
        parser.error("--c-values contains duplicates")
    if CURRENT_C not in args.c_values:
        parser.error(f"--c-values must include the current baseline C={CURRENT_C:g}")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds contains duplicates")
    if args.folds < 2 or args.folds > min(CLASS_COUNTS.values()):
        parser.error("--folds is outside the valid range")
    args.c_values = sorted(args.c_values)
    return args


def load_training_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Training data not found: {path}")
    with np.load(path, allow_pickle=False) as data:
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
        raise ValueError("source_index must preserve the canonical TRAIN.ts order")
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


def prepare_fold(
    training_x: np.ndarray, validation_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    preprocessing = fit_preprocessing(training_x)
    return transform(training_x, preprocessing), transform(
        validation_x, preprocessing
    )


def fit_candidate(
    training_features: np.ndarray,
    training_y: np.ndarray,
    validation_features: np.ndarray,
    c: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    classifier = LogisticRegression(
        C=c,
        solver="liblinear",
        max_iter=5_000,
    )
    classifier.fit(training_features, training_y)
    prediction = classifier.predict(validation_features).astype(np.int64)
    score = classifier.decision_function(validation_features).astype(np.float64)
    training_accuracy = float(
        np.mean(classifier.predict(training_features) == training_y)
    )
    return prediction, score, training_accuracy


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


def paired_comparison(
    actual: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    seed: int,
    c: float,
) -> dict[str, Any]:
    baseline_correct = baseline == actual
    candidate_correct = candidate == actual
    baseline_only = int(np.sum(baseline_correct & ~candidate_correct))
    candidate_only = int(np.sum(~baseline_correct & candidate_correct))
    discordant = baseline_only + candidate_only
    p_value = (
        float(
            binomtest(
                min(baseline_only, candidate_only),
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
        "baseline_c": CURRENT_C,
        "candidate_c": c,
        "candidate_minus_baseline_accuracy": float(
            candidate_correct.mean() - baseline_correct.mean()
        ),
        "baseline_only_correct": baseline_only,
        "candidate_only_correct": candidate_only,
        "both_correct": int(np.sum(baseline_correct & candidate_correct)),
        "both_wrong": int(np.sum(~baseline_correct & ~candidate_correct)),
        "mcnemar_exact_p": p_value,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def summarize_candidates(
    seed_rows: list[dict[str, Any]], c_values: list[float]
) -> list[dict[str, Any]]:
    output = []
    for c in c_values:
        rows = [row for row in seed_rows if row["c"] == c]
        summary: dict[str, Any] = {"c": c, "seeds": len(rows)}
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            values = np.asarray([row[metric] for row in rows], dtype=float)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=1))
            summary[f"{metric}_min"] = float(values.min())
            summary[f"{metric}_max"] = float(values.max())
        output.append(summary)
    return output


def save_figure(path: Path, summaries: list[dict[str, Any]]) -> None:
    c_values = np.asarray([row["c"] for row in summaries], dtype=float)
    means = 100.0 * np.asarray([row["balanced_accuracy_mean"] for row in summaries])
    standard_deviations = 100.0 * np.asarray(
        [row["balanced_accuracy_std"] for row in summaries]
    )
    worst = 100.0 * np.asarray([row["balanced_accuracy_min"] for row in summaries])
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.errorbar(
        c_values,
        means,
        yerr=standard_deviations,
        marker="o",
        capsize=5,
        color="#4C78A8",
        label="Mean ± seed SD",
    )
    axis.plot(c_values, worst, marker="s", color="#E45756", label="Worst seed")
    axis.axvline(CURRENT_C, color="black", linestyle="--", label="Current C=1")
    axis.set_xlabel("Fixed Logistic Regression C")
    axis.set_ylabel("OOF balanced accuracy (%)")
    axis.set_title("FingerMovements Phase 1e fixed-C upper refinement")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def validate_only(
    x: np.ndarray, y: np.ndarray, seed: int, folds: int, c_values: list[float]
) -> None:
    training, validation = stratified_folds(y, folds, seed)[0]
    training_features, validation_features = prepare_fold(
        x[training], x[validation]
    )
    for c in (c_values[0], c_values[-1]):
        prediction, score, _ = fit_candidate(
            training_features, y[training], validation_features, c
        )
        if prediction.shape != (len(validation),) or not np.isfinite(score).all():
            raise RuntimeError(f"Candidate validation failed for C={c:g}")
    print("=== FingerMovements Phase 1e upper-refinement validation-only ===")
    print(
        f"train features={training_features.shape} | "
        f"validation={validation_features.shape}"
    )
    print(f"fixed-C endpoints validated: {c_values[0]:g}, {c_values[-1]:g}")
    print("test: LOCKED AND NOT LOADED")


def main() -> None:
    args = parse_args()
    x, y, source_index = load_training_data(args.data.resolve())
    if args.validate_only:
        validate_only(x, y, args.seeds[0], args.folds, args.c_values)
        return

    print("=== FingerMovements Phase 1e fixed-C upper refinement ===")
    print(f"cases={len(y)} | seeds={args.seeds} | folds={args.folds}")
    print(f"fixed C values={args.c_values}")
    print("preprocessing: fitted from each fold's training cases only")
    print("test: LOCKED AND NOT LOADED")

    fold_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    predictions_by_seed: dict[int, dict[float, np.ndarray]] = {}

    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        oof_predictions = {
            c: np.full(len(y), -1, dtype=np.int64) for c in args.c_values
        }
        oof_scores = {
            c: np.full(len(y), np.nan, dtype=np.float64) for c in args.c_values
        }
        for fold, (training, validation) in enumerate(
            stratified_folds(y, args.folds, seed), start=1
        ):
            training_features, validation_features = prepare_fold(
                x[training], x[validation]
            )
            fold_text = []
            for c in args.c_values:
                prediction, score, training_accuracy = fit_candidate(
                    training_features,
                    y[training],
                    validation_features,
                    c,
                )
                oof_predictions[c][validation] = prediction
                oof_scores[c][validation] = score
                metrics = classification_metrics(y[validation], prediction)
                fold_rows.append(
                    {
                        "c": c,
                        "seed": seed,
                        "fold": fold,
                        "training_cases": len(training),
                        "validation_cases": len(validation),
                        "training_accuracy": training_accuracy,
                        **metrics,
                    }
                )
                fold_text.append(f"C={c:g}:{metrics['balanced_accuracy']:.3f}")
            print(f"fold {fold}/{args.folds} | " + " | ".join(fold_text))

        predictions_by_seed[seed] = oof_predictions
        for c in args.c_values:
            if np.any(oof_predictions[c] < 0) or not np.isfinite(oof_scores[c]).all():
                raise RuntimeError(f"Incomplete OOF results for seed={seed}, C={c:g}")
            metrics = classification_metrics(y, oof_predictions[c])
            seed_rows.append({"c": c, "seed": seed, **metrics})
            for index in range(len(y)):
                prediction_rows.append(
                    {
                        "c": c,
                        "seed": seed,
                        "source_index": int(source_index[index]),
                        "true_label": int(y[index]),
                        "predicted_label": int(oof_predictions[c][index]),
                        "score": float(oof_scores[c][index]),
                    }
                )
            print(
                f"seed summary | C={c:g} | accuracy={metrics['accuracy']:.4f} | "
                f"balanced accuracy={metrics['balanced_accuracy']:.4f} | "
                f"macro F1={metrics['macro_f1']:.4f}"
            )
        baseline_prediction = oof_predictions[CURRENT_C]
        for c in args.c_values:
            if c == CURRENT_C:
                continue
            paired_rows.append(
                paired_comparison(y, baseline_prediction, oof_predictions[c], seed, c)
            )

    candidate_summary = summarize_candidates(seed_rows, args.c_values)
    ranking = sorted(
        candidate_summary,
        key=lambda row: (
            -row["balanced_accuracy_mean"],
            -row["balanced_accuracy_min"],
            row["c"],
        ),
    )
    print("\n=== fixed-C ranking ===")
    for rank, row in enumerate(ranking, start=1):
        print(
            f"{rank}. C={row['c']:g} | BA={row['balanced_accuracy_mean']:.4f} ± "
            f"{row['balanced_accuracy_std']:.4f} | "
            f"worst={row['balanced_accuracy_min']:.4f}"
        )
    print("decision: review paired and stability results before freezing C")

    report = {
        "phase": "1e",
        "stage": "upper_refinement",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "official TRAIN split only; official TEST not opened",
        "protocol": {
            "cases": len(y),
            "seeds": args.seeds,
            "folds": args.folds,
            "c_values": args.c_values,
            "fixed_c_per_candidate": True,
            "fold_training_only_preprocessing": True,
            "baseline_c": CURRENT_C,
            "selection_metric": "mean OOF balanced accuracy across seeds",
            "tie_breakers": ["higher worst-seed BA", "smaller C"],
        },
        "candidate_summary": candidate_summary,
        "ranking": [row["c"] for row in ranking],
        "paired_comparisons": paired_rows,
        "decision_policy": (
            "Prefer a new C only when its mean and worst-seed balanced accuracy "
            "provide a meaningful, consistent improvement over C=1. The official "
            "test remains the final unbiased evaluation."
        ),
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "phase1e_upper_fold_results.csv", fold_rows)
    write_csv(output_dir / "phase1e_upper_seed_results.csv", seed_rows)
    write_csv(output_dir / "phase1e_upper_candidate_summary.csv", candidate_summary)
    write_csv(output_dir / "phase1e_upper_predictions.csv", prediction_rows)
    write_csv(output_dir / "phase1e_upper_paired_comparisons.csv", paired_rows)
    metrics_path = output_dir / "phase1e_upper_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    figure_path = output_dir / "phase1e_upper_regularization_sweep.png"
    save_figure(figure_path, candidate_summary)
    print(f"metrics: {metrics_path}")
    print(f"figure: {figure_path}")


if __name__ == "__main__":
    main()
