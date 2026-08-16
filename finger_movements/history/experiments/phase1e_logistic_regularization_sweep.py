"""Phase 1e: nested-CV sweep of Logistic Regression regularization C.

Each outer validation fold is isolated before C is selected. Candidate values
are compared only through inner folds built from the outer training subset.
The nested-selected policy is then evaluated on the untouched outer validation
fold and compared with the Phase 1d fixed C=1 baseline.

The official test split is locked and is never opened by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
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
    fit_logistic,
    fit_preprocessing,
    transform,
)


CASES = 316
CHANNELS = 28
TIMEPOINTS = 50
CLASS_COUNTS = {0: 159, 1: 157}
OUTER_SEEDS = (42, 43, 44)
OUTER_FOLDS = 5
INNER_FOLDS = 4
C_VALUES = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
PIPELINES = ("nested_selected", "fixed_c_1")
PIPELINE_LABELS = {
    "nested_selected": "Nested-selected C",
    "fixed_c_1": "Fixed C=1",
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
        default=ROOT / "results/finger_movements/phase1e_logistic_regularization",
    )
    parser.add_argument(
        "--c-values", nargs="+", type=float, default=list(C_VALUES)
    )
    parser.add_argument(
        "--outer-seeds", nargs="+", type=int, default=list(OUTER_SEEDS)
    )
    parser.add_argument("--outer-folds", type=int, default=OUTER_FOLDS)
    parser.add_argument("--inner-folds", type=int, default=INNER_FOLDS)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.data.name.lower() == "test.npz":
        parser.error("Phase 1e refuses to load test.npz")
    if any(value <= 0 for value in args.c_values):
        parser.error("Every --c-values entry must be positive")
    if len(args.c_values) != len(set(args.c_values)):
        parser.error("--c-values contains duplicates")
    if len(args.outer_seeds) != len(set(args.outer_seeds)):
        parser.error("--outer-seeds contains duplicates")
    if args.outer_folds < 2 or args.outer_folds > min(CLASS_COUNTS.values()):
        parser.error("--outer-folds is outside the valid range")
    if args.inner_folds < 2:
        parser.error("--inner-folds must be at least 2")
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
    """Reproduce the stratified fold construction used in prior phases."""
    minimum_class_count = int(np.bincount(y, minlength=2).min())
    if fold_count > minimum_class_count:
        raise ValueError(
            f"Cannot create {fold_count} folds with minimum class count "
            f"{minimum_class_count}"
        )
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


def fit_feature_classifier(
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


def prepare_inner_folds(
    outer_training_x: np.ndarray,
    outer_training_y: np.ndarray,
    fold_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Cache leakage-safe features once per inner split."""
    prepared = []
    validation_visits = np.zeros(len(outer_training_y), dtype=np.int64)
    for fold, (training, validation) in enumerate(
        stratified_folds(outer_training_y, fold_count, seed), start=1
    ):
        if np.intersect1d(training, validation).size:
            raise RuntimeError("Inner training/validation indices overlap")
        validation_visits[validation] += 1
        preprocessing = fit_preprocessing(outer_training_x[training])
        prepared.append(
            {
                "fold": fold,
                "training": training,
                "validation": validation,
                "training_features": transform(
                    outer_training_x[training], preprocessing
                ),
                "validation_features": transform(
                    outer_training_x[validation], preprocessing
                ),
            }
        )
    if not np.array_equal(validation_visits, np.ones(len(outer_training_y), dtype=int)):
        raise RuntimeError("Every outer-training case must be inner validation once")
    return prepared


def evaluate_inner_candidate(
    prepared_folds: list[dict[str, Any]],
    outer_training_y: np.ndarray,
    c: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions = np.full(len(outer_training_y), -1, dtype=np.int64)
    fold_rows = []
    for prepared in prepared_folds:
        training = prepared["training"]
        validation = prepared["validation"]
        prediction, _, training_accuracy = fit_feature_classifier(
            prepared["training_features"],
            outer_training_y[training],
            prepared["validation_features"],
            c,
        )
        predictions[validation] = prediction
        metrics = classification_metrics(outer_training_y[validation], prediction)
        fold_rows.append(
            {
                "inner_fold": prepared["fold"],
                "c": c,
                "training_accuracy": training_accuracy,
                **metrics,
            }
        )
    if np.any(predictions < 0):
        raise RuntimeError("Incomplete inner OOF predictions")
    oof_metrics = classification_metrics(outer_training_y, predictions)
    summary = {
        "c": c,
        "inner_oof_accuracy": oof_metrics["accuracy"],
        "inner_oof_balanced_accuracy": oof_metrics["balanced_accuracy"],
        "inner_oof_macro_f1": oof_metrics["macro_f1"],
        "inner_fold_balanced_accuracy_mean": float(
            np.mean([row["balanced_accuracy"] for row in fold_rows])
        ),
        "inner_fold_balanced_accuracy_std": float(
            np.std([row["balanced_accuracy"] for row in fold_rows], ddof=1)
        ),
        "inner_training_accuracy_mean": float(
            np.mean([row["training_accuracy"] for row in fold_rows])
        ),
    }
    return summary, fold_rows


def choose_c(candidate_rows: list[dict[str, Any]]) -> float:
    """Maximize inner OOF BA; exact ties favor stronger regularization."""
    ordered = sorted(
        candidate_rows,
        key=lambda row: (-row["inner_oof_balanced_accuracy"], row["c"]),
    )
    return float(ordered[0]["c"])


def fit_outer_model(
    training_x: np.ndarray,
    training_y: np.ndarray,
    validation_x: np.ndarray,
    c: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    model, preprocessing = fit_logistic(training_x, training_y, c=c)
    training_prediction = model.predict_raw(training_x, preprocessing)
    validation_features = transform(validation_x, preprocessing)
    validation_prediction = model.predict_features(validation_features).astype(np.int64)
    validation_score = model.decision_function(validation_features)
    training_accuracy = float(np.mean(training_prediction == training_y))
    return validation_prediction, validation_score, training_accuracy


def paired_comparison(
    actual: np.ndarray,
    fixed_prediction: np.ndarray,
    nested_prediction: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    fixed_correct = fixed_prediction == actual
    nested_correct = nested_prediction == actual
    fixed_only = int(np.sum(fixed_correct & ~nested_correct))
    nested_only = int(np.sum(~fixed_correct & nested_correct))
    discordant = fixed_only + nested_only
    p_value = (
        float(
            binomtest(
                min(fixed_only, nested_only),
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
        "nested_minus_fixed_accuracy": float(
            nested_correct.mean() - fixed_correct.mean()
        ),
        "fixed_only_correct": fixed_only,
        "nested_only_correct": nested_only,
        "both_correct": int(np.sum(fixed_correct & nested_correct)),
        "both_wrong": int(np.sum(~fixed_correct & ~nested_correct)),
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
    candidate_rows: list[dict[str, Any]], c_values: list[float]
) -> tuple[list[dict[str, Any]], float, float]:
    selected_by_context: dict[tuple[int, int], float] = {}
    for row in candidate_rows:
        context = (row["outer_seed"], row["outer_fold"])
        selected_c = float(row["selected_c"])
        if context in selected_by_context and selected_by_context[context] != selected_c:
            raise RuntimeError(f"Inconsistent selected C for outer context {context}")
        selected_by_context[context] = selected_c
    selected_counts = Counter(selected_by_context.values())
    summaries = []
    for c in c_values:
        rows = [row for row in candidate_rows if row["c"] == c]
        values = np.asarray(
            [row["inner_oof_balanced_accuracy"] for row in rows], dtype=float
        )
        summaries.append(
            {
                "c": c,
                "outer_contexts": len(rows),
                "inner_oof_balanced_accuracy_mean": float(values.mean()),
                "inner_oof_balanced_accuracy_std": float(values.std(ddof=1)),
                "inner_oof_balanced_accuracy_min": float(values.min()),
                "inner_oof_balanced_accuracy_max": float(values.max()),
                "selected_count": int(selected_counts[c]),
            }
        )
    best = sorted(
        summaries,
        key=lambda row: (-row["inner_oof_balanced_accuracy_mean"], row["c"]),
    )[0]
    best_standard_error = best["inner_oof_balanced_accuracy_std"] / np.sqrt(
        best["outer_contexts"]
    )
    eligible = [
        row
        for row in summaries
        if row["inner_oof_balanced_accuracy_mean"]
        >= best["inner_oof_balanced_accuracy_mean"] - best_standard_error
    ]
    one_standard_error_c = float(min(row["c"] for row in eligible))
    return summaries, float(best["c"]), one_standard_error_c


def summarize_outer_seed_results(
    seed_rows: list[dict[str, Any]], pipeline: str
) -> dict[str, Any]:
    rows = [row for row in seed_rows if row["pipeline"] == pipeline]
    summary: dict[str, Any] = {"pipeline": pipeline}
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        values = np.asarray([row[metric] for row in rows], dtype=float)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=1))
        summary[f"{metric}_min"] = float(values.min())
        summary[f"{metric}_max"] = float(values.max())
    return summary


def save_figure(
    path: Path,
    candidate_summary: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    c_values = np.asarray([row["c"] for row in candidate_summary], dtype=float)
    means = 100.0 * np.asarray(
        [row["inner_oof_balanced_accuracy_mean"] for row in candidate_summary]
    )
    standard_deviations = 100.0 * np.asarray(
        [row["inner_oof_balanced_accuracy_std"] for row in candidate_summary]
    )
    axes[0].errorbar(
        c_values,
        means,
        yerr=standard_deviations,
        marker="o",
        capsize=4,
        color="#4C78A8",
    )
    axes[0].set_xscale("log")
    axes[0].axvline(CURRENT_C, color="#E45756", linestyle="--", label="Current C=1")
    axes[0].set_xlabel("C (smaller = stronger regularization)")
    axes[0].set_ylabel("Inner OOF balanced accuracy (%)")
    axes[0].set_title("Training-only regularization profile")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    positions = np.arange(len(PIPELINES))
    for position, pipeline in enumerate(PIPELINES):
        rows = [row for row in seed_rows if row["pipeline"] == pipeline]
        values = 100.0 * np.asarray([row["balanced_accuracy"] for row in rows])
        axes[1].scatter(np.full(len(values), position), values, s=52, color="#4C78A8")
        axes[1].errorbar(
            position,
            values.mean(),
            yerr=values.std(ddof=1),
            fmt="o",
            capsize=5,
            color="#E45756",
        )
    axes[1].set_xticks(positions, [PIPELINE_LABELS[name] for name in PIPELINES])
    axes[1].set_ylabel("Outer OOF balanced accuracy (%)")
    axes[1].set_title("Untouched outer-fold performance")
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("FingerMovements Phase 1e Logistic regularization sweep")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def validate_only(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    outer_folds: int,
    inner_folds: int,
    c: float,
) -> None:
    outer_training, outer_validation = stratified_folds(y, outer_folds, seed)[0]
    prepared = prepare_inner_folds(
        x[outer_training],
        y[outer_training],
        inner_folds,
        seed * 1_000,
    )
    summary, _ = evaluate_inner_candidate(prepared, y[outer_training], c)
    prediction, score, _ = fit_outer_model(
        x[outer_training], y[outer_training], x[outer_validation], c
    )
    if prediction.shape != (len(outer_validation),) or not np.isfinite(score).all():
        raise RuntimeError("Outer model validation failed")
    print("=== FingerMovements Phase 1e validation-only ===")
    print(
        f"outer train={len(outer_training)} validation={len(outer_validation)} | "
        f"inner folds={inner_folds} | C={c:g}"
    )
    print(
        f"inner OOF balanced accuracy={summary['inner_oof_balanced_accuracy']:.4f}"
    )
    print("nested preprocessing/model paths: PASS")
    print("test: LOCKED AND NOT LOADED")


def main() -> None:
    args = parse_args()
    x, y, source_index = load_training_data(args.data.resolve())
    if args.validate_only:
        validate_only(
            x,
            y,
            args.outer_seeds[0],
            args.outer_folds,
            args.inner_folds,
            args.c_values[0],
        )
        return

    print("=== FingerMovements Phase 1e Logistic regularization sweep ===")
    print(
        f"cases={len(y)} | outer seeds={args.outer_seeds} | "
        f"outer folds={args.outer_folds} | inner folds={args.inner_folds}"
    )
    print(f"C values={args.c_values}")
    print("selection: inner OOF balanced accuracy; exact ties favor smaller C")
    print("outer validation: isolated until C has been selected")
    print("test: LOCKED AND NOT LOADED")

    inner_candidate_rows: list[dict[str, Any]] = []
    inner_fold_rows: list[dict[str, Any]] = []
    outer_fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []

    for outer_seed in args.outer_seeds:
        print(f"\n=== outer seed {outer_seed} ===")
        oof_predictions = {
            pipeline: np.full(len(y), -1, dtype=np.int64)
            for pipeline in PIPELINES
        }
        oof_scores = {
            pipeline: np.full(len(y), np.nan, dtype=np.float64)
            for pipeline in PIPELINES
        }
        for outer_fold, (outer_training, outer_validation) in enumerate(
            stratified_folds(y, args.outer_folds, outer_seed), start=1
        ):
            inner_seed = outer_seed * 1_000 + outer_fold
            prepared = prepare_inner_folds(
                x[outer_training],
                y[outer_training],
                args.inner_folds,
                inner_seed,
            )
            context_candidates = []
            for c in args.c_values:
                summary, fold_rows = evaluate_inner_candidate(
                    prepared, y[outer_training], c
                )
                context_candidates.append(summary)
                for row in fold_rows:
                    inner_fold_rows.append(
                        {
                            "outer_seed": outer_seed,
                            "outer_fold": outer_fold,
                            "inner_seed": inner_seed,
                            **row,
                        }
                    )
            selected_c = choose_c(context_candidates)
            for row in context_candidates:
                inner_candidate_rows.append(
                    {
                        "outer_seed": outer_seed,
                        "outer_fold": outer_fold,
                        "inner_seed": inner_seed,
                        "selected_c": selected_c,
                        **row,
                    }
                )

            for pipeline, c in (
                ("nested_selected", selected_c),
                ("fixed_c_1", CURRENT_C),
            ):
                prediction, score, training_accuracy = fit_outer_model(
                    x[outer_training],
                    y[outer_training],
                    x[outer_validation],
                    c,
                )
                oof_predictions[pipeline][outer_validation] = prediction
                oof_scores[pipeline][outer_validation] = score
                metrics = classification_metrics(y[outer_validation], prediction)
                outer_fold_rows.append(
                    {
                        "pipeline": pipeline,
                        "outer_seed": outer_seed,
                        "outer_fold": outer_fold,
                        "c": c,
                        "selected_c": selected_c,
                        "training_cases": len(outer_training),
                        "validation_cases": len(outer_validation),
                        "training_accuracy": training_accuracy,
                        **metrics,
                    }
                )
            selected_metrics = outer_fold_rows[-2]
            fixed_metrics = outer_fold_rows[-1]
            best_inner = max(
                row["inner_oof_balanced_accuracy"] for row in context_candidates
            )
            print(
                f"outer fold {outer_fold}/{args.outer_folds} | selected C={selected_c:g} "
                f"(inner BA={best_inner:.4f}) | outer BA selected="
                f"{selected_metrics['balanced_accuracy']:.4f} fixed C=1="
                f"{fixed_metrics['balanced_accuracy']:.4f}"
            )

        for pipeline in PIPELINES:
            if np.any(oof_predictions[pipeline] < 0) or not np.isfinite(
                oof_scores[pipeline]
            ).all():
                raise RuntimeError(
                    f"Incomplete outer OOF results: seed={outer_seed}, {pipeline}"
                )
            metrics = classification_metrics(y, oof_predictions[pipeline])
            seed_rows.append(
                {"pipeline": pipeline, "outer_seed": outer_seed, **metrics}
            )
            for index in range(len(y)):
                prediction_rows.append(
                    {
                        "pipeline": pipeline,
                        "outer_seed": outer_seed,
                        "source_index": int(source_index[index]),
                        "true_label": int(y[index]),
                        "predicted_label": int(oof_predictions[pipeline][index]),
                        "score": float(oof_scores[pipeline][index]),
                    }
                )
            print(
                f"seed summary | {PIPELINE_LABELS[pipeline]} | "
                f"accuracy={metrics['accuracy']:.4f} | "
                f"balanced accuracy={metrics['balanced_accuracy']:.4f} | "
                f"macro F1={metrics['macro_f1']:.4f}"
            )
        paired_rows.append(
            paired_comparison(
                y,
                oof_predictions["fixed_c_1"],
                oof_predictions["nested_selected"],
                outer_seed,
            )
        )

    candidate_summary, best_mean_c, one_standard_error_c = summarize_candidates(
        inner_candidate_rows, args.c_values
    )
    outer_summary = [
        summarize_outer_seed_results(seed_rows, pipeline) for pipeline in PIPELINES
    ]
    selected_c_counts = Counter(
        row["selected_c"]
        for row in inner_candidate_rows
        if row["c"] == args.c_values[0]
    )

    print("\n=== training-only candidate profile ===")
    for row in sorted(
        candidate_summary,
        key=lambda item: item["inner_oof_balanced_accuracy_mean"],
        reverse=True,
    ):
        print(
            f"C={row['c']:g} | inner BA="
            f"{row['inner_oof_balanced_accuracy_mean']:.4f} ± "
            f"{row['inner_oof_balanced_accuracy_std']:.4f} | "
            f"selected {row['selected_count']}/{row['outer_contexts']} contexts"
        )
    print(f"best mean inner-CV C: {best_mean_c:g}")
    print(f"one-standard-error conservative C: {one_standard_error_c:g}")

    print("\n=== untouched outer-fold policy comparison ===")
    for row in outer_summary:
        print(
            f"{PIPELINE_LABELS[row['pipeline']]} | BA="
            f"{row['balanced_accuracy_mean']:.4f} ± "
            f"{row['balanced_accuracy_std']:.4f} | "
            f"worst={row['balanced_accuracy_min']:.4f}"
        )
    print("decision: review results before freezing C; test remains locked")

    report = {
        "phase": "1e",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "official TRAIN split only; official TEST not opened",
        "protocol": {
            "cases": len(y),
            "outer_seeds": args.outer_seeds,
            "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds,
            "c_values": args.c_values,
            "selection_metric": "inner OOF balanced accuracy",
            "tie_policy": "smaller C (stronger regularization)",
            "fold_training_only_preprocessing": True,
            "fixed_baseline_c": CURRENT_C,
        },
        "selected_c_counts": {
            str(c): int(count) for c, count in sorted(selected_c_counts.items())
        },
        "candidate_summary": candidate_summary,
        "best_mean_inner_cv_c": best_mean_c,
        "one_standard_error_c": one_standard_error_c,
        "outer_policy_summary": outer_summary,
        "paired_comparisons": paired_rows,
        "decision_policy": (
            "Do not freeze C from the inner profile alone. Confirm that the nested-"
            "selected policy is not worse than fixed C=1 on untouched outer folds, "
            "then review selection frequency, stability, and regularization strength."
        ),
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "phase1e_inner_fold_results.csv", inner_fold_rows)
    write_csv(output_dir / "phase1e_inner_candidate_results.csv", inner_candidate_rows)
    write_csv(output_dir / "phase1e_candidate_summary.csv", candidate_summary)
    write_csv(output_dir / "phase1e_outer_fold_results.csv", outer_fold_rows)
    write_csv(output_dir / "phase1e_seed_results.csv", seed_rows)
    write_csv(output_dir / "phase1e_predictions.csv", prediction_rows)
    write_csv(output_dir / "phase1e_paired_comparisons.csv", paired_rows)
    metrics_path = output_dir / "phase1e_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    figure_path = output_dir / "phase1e_regularization_sweep.png"
    save_figure(figure_path, candidate_summary, seed_rows)
    print(f"metrics: {metrics_path}")
    print(f"figure: {figure_path}")


if __name__ == "__main__":
    main()
