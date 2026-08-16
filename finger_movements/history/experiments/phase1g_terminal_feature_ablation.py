"""Phase 1g: contribution analysis for the frozen terminal feature groups.

The frozen 252-dimensional representation is divided into three groups:

* A: five terminal low-pass samples per channel (140 features);
* B: final 50/100/200 ms low-pass means per channel (84 features);
* C: final 200 ms low-pass slope per channel (28 features).

All eight subsets of A/B/C are evaluated with the frozen Logistic Regression
configuration (C=1) using seeds 42/43/44 and five-fold out-of-fold validation.
The script reports standalone performance, leave-one-group-out degradation,
and exact three-player Shapley contributions. The official test split remains
locked and is never loaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.finger_movements.terminal_logistic.model import (  # noqa: E402
    CHANNELS,
    FEATURES,
    LOGISTIC_C,
    TERMINAL_MEAN_WINDOWS,
    TERMINAL_SAMPLES,
    TERMINAL_SLOPE_WINDOW,
    TIMEPOINTS,
    terminal_features,
)


CASES = 316
CLASS_COUNTS = {0: 159, 1: 157}
SEEDS = (42, 43, 44)
FOLDS = 5

GROUPS = ("A", "B", "C")
GROUP_LABELS = {
    "A": "A: final 5 samples",
    "B": "B: 50/100/200 ms means",
    "C": "C: final 200 ms slope",
}
GROUP_DESCRIPTIONS = {
    "A": "five final causal-low-pass samples per channel",
    "B": "causal-low-pass means over the final 50, 100, and 200 ms",
    "C": "causal-low-pass least-squares slope over the final 200 ms",
}

A_FEATURES = CHANNELS * TERMINAL_SAMPLES
B_FEATURES = CHANNELS * len(TERMINAL_MEAN_WINDOWS)
C_FEATURES = CHANNELS
if A_FEATURES + B_FEATURES + C_FEATURES != FEATURES:
    raise RuntimeError("Feature-group dimensions disagree with the frozen model")

GROUP_INDICES = {
    "A": np.arange(0, A_FEATURES, dtype=np.int64),
    "B": np.arange(A_FEATURES, A_FEATURES + B_FEATURES, dtype=np.int64),
    "C": np.arange(A_FEATURES + B_FEATURES, FEATURES, dtype=np.int64),
}

SUBSET_GROUPS = {
    "none": (),
    "A": ("A",),
    "B": ("B",),
    "C": ("C",),
    "AB": ("A", "B"),
    "AC": ("A", "C"),
    "BC": ("B", "C"),
    "ABC": ("A", "B", "C"),
}
SUBSET_LABELS = {
    "none": "No EEG features",
    "A": "A only",
    "B": "B only",
    "C": "C only",
    "AB": "A + B",
    "AC": "A + C",
    "BC": "B + C",
    "ABC": "A + B + C (frozen)",
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
        default=ROOT / "results/finger_movements/phase1g_terminal_feature_ablation",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if "test" in args.data.name.lower():
        parser.error("Phase 1g refuses to load any file identified as a test split")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds contains duplicates")
    if args.folds < 2 or args.folds > min(CLASS_COUNTS.values()):
        parser.error("--folds is outside the valid range")
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
    """Reproduce the custom folds used by the preceding experiments."""
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


def normalized_terminal_pair(
    training_x: np.ndarray,
    validation_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build frozen terminal features with fold-training-only preprocessing."""
    channel_mean = training_x.mean(axis=(0, 2), keepdims=True, dtype=np.float64)
    channel_std = np.maximum(
        training_x.std(axis=(0, 2), keepdims=True, dtype=np.float64), 1e-6
    )
    normalized_training = ((training_x - channel_mean) / channel_std).astype(
        np.float32
    )
    normalized_validation = ((validation_x - channel_mean) / channel_std).astype(
        np.float32
    )
    raw_training = terminal_features(normalized_training)
    raw_validation = terminal_features(normalized_validation)

    feature_mean = raw_training.mean(axis=0, keepdims=True, dtype=np.float64)
    feature_std = np.maximum(
        raw_training.std(axis=0, keepdims=True, dtype=np.float64), 1e-6
    )
    training = ((raw_training - feature_mean) / feature_std).astype(np.float32)
    validation = ((raw_validation - feature_mean) / feature_std).astype(np.float32)
    return training, validation


def subset_indices(groups: Iterable[str]) -> np.ndarray:
    selected = tuple(groups)
    if not selected:
        return np.empty(0, dtype=np.int64)
    return np.concatenate([GROUP_INDICES[group] for group in selected])


def majority_prediction(training_y: np.ndarray, cases: int) -> np.ndarray:
    """Intercept-only reference; its balanced accuracy is exactly chance (0.5)."""
    counts = np.bincount(training_y, minlength=2)
    label = int(np.flatnonzero(counts == counts.max())[0])
    return np.full(cases, label, dtype=np.int64)


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


def summarize_subsets(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for subset, groups in SUBSET_GROUPS.items():
        rows = [row for row in seed_rows if row["subset"] == subset]
        summary: dict[str, Any] = {
            "subset": subset,
            "groups": "+".join(groups) if groups else "none",
            "feature_count": rows[0]["feature_count"],
            "seeds": len(rows),
        }
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=1))
            summary[f"{metric}_min"] = float(values.min())
            summary[f"{metric}_max"] = float(values.max())
        output.append(summary)
    return output


def subset_name(groups: frozenset[str]) -> str:
    for name, members in SUBSET_GROUPS.items():
        if frozenset(members) == groups:
            return name
    raise KeyError(f"No configured subset for groups: {sorted(groups)}")


def exact_shapley_rows(
    seed_rows: list[dict[str, Any]], seeds: list[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute exact Shapley values for the three predefined feature groups."""
    detail_rows: list[dict[str, Any]] = []
    seed_contributions: list[dict[str, Any]] = []
    group_set = frozenset(GROUPS)
    denominator = math.factorial(len(GROUPS))

    for seed in seeds:
        values = {
            row["subset"]: float(row["balanced_accuracy"])
            for row in seed_rows
            if row["seed"] == seed
        }
        for group in GROUPS:
            contribution = 0.0
            other_groups = sorted(group_set - {group})
            for subset_size in range(len(other_groups) + 1):
                for members in combinations(other_groups, subset_size):
                    without = frozenset(members)
                    with_group = without | {group}
                    weight = (
                        math.factorial(subset_size)
                        * math.factorial(len(GROUPS) - subset_size - 1)
                        / denominator
                    )
                    marginal = (
                        values[subset_name(with_group)]
                        - values[subset_name(without)]
                    )
                    weighted = weight * marginal
                    contribution += weighted
                    detail_rows.append(
                        {
                            "seed": seed,
                            "group": group,
                            "context_without_group": subset_name(without),
                            "context_with_group": subset_name(with_group),
                            "shapley_weight": weight,
                            "marginal_balanced_accuracy": marginal,
                            "weighted_contribution": weighted,
                        }
                    )
            seed_contributions.append(
                {
                    "seed": seed,
                    "group": group,
                    "shapley_balanced_accuracy": contribution,
                    "shapley_percentage_points": 100.0 * contribution,
                }
            )

        observed_sum = sum(
            row["shapley_balanced_accuracy"]
            for row in seed_contributions
            if row["seed"] == seed
        )
        expected_sum = values["ABC"] - values["none"]
        if not np.isclose(observed_sum, expected_sum, atol=1e-12):
            raise RuntimeError(
                f"Shapley additivity failed for seed {seed}: "
                f"observed={observed_sum}, expected={expected_sum}"
            )
    return seed_contributions, detail_rows


def summarize_group_contributions(
    seed_rows: list[dict[str, Any]],
    shapley_by_seed: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    values_by_seed = {
        (row["seed"], row["subset"]): float(row["balanced_accuracy"])
        for row in seed_rows
    }
    seeds = sorted({int(row["seed"]) for row in seed_rows})
    output = []
    for group in GROUPS:
        without = "".join(member for member in GROUPS if member != group)
        standalone = np.asarray(
            [values_by_seed[(seed, group)] for seed in seeds], dtype=np.float64
        )
        leave_out_drop = np.asarray(
            [
                values_by_seed[(seed, "ABC")] - values_by_seed[(seed, without)]
                for seed in seeds
            ],
            dtype=np.float64,
        )
        shapley = np.asarray(
            [
                row["shapley_balanced_accuracy"]
                for row in shapley_by_seed
                if row["group"] == group
            ],
            dtype=np.float64,
        )
        output.append(
            {
                "group": group,
                "description": GROUP_DESCRIPTIONS[group],
                "feature_count": len(GROUP_INDICES[group]),
                "standalone_balanced_accuracy_mean": float(standalone.mean()),
                "standalone_balanced_accuracy_std": float(standalone.std(ddof=1)),
                "leave_one_out_drop_mean": float(leave_out_drop.mean()),
                "leave_one_out_drop_std": float(leave_out_drop.std(ddof=1)),
                "shapley_mean": float(shapley.mean()),
                "shapley_std": float(shapley.std(ddof=1)),
                "shapley_percentage_points_mean": float(100.0 * shapley.mean()),
                "shapley_percentage_points_std": float(100.0 * shapley.std(ddof=1)),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(
    path: Path,
    subset_summaries: list[dict[str, Any]],
    contribution_summaries: list[dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    labels = list(SUBSET_GROUPS)
    means = 100.0 * np.asarray(
        [
            next(
                row["balanced_accuracy_mean"]
                for row in subset_summaries
                if row["subset"] == subset
            )
            for subset in labels
        ],
        dtype=np.float64,
    )
    deviations = 100.0 * np.asarray(
        [
            next(
                row["balanced_accuracy_std"]
                for row in subset_summaries
                if row["subset"] == subset
            )
            for subset in labels
        ],
        dtype=np.float64,
    )
    positions = np.arange(len(labels))
    axes[0].bar(positions, means, yerr=deviations, capsize=4, color="#4C78A8")
    axes[0].axhline(50.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylabel("OOF balanced accuracy (%)")
    axes[0].set_title("All feature-group subsets")
    axes[0].grid(axis="y", alpha=0.25)

    group_positions = np.arange(len(GROUPS))
    shapley_means = np.asarray(
        [
            next(
                row["shapley_percentage_points_mean"]
                for row in contribution_summaries
                if row["group"] == group
            )
            for group in GROUPS
        ]
    )
    shapley_std = np.asarray(
        [
            next(
                row["shapley_percentage_points_std"]
                for row in contribution_summaries
                if row["group"] == group
            )
            for group in GROUPS
        ]
    )
    axes[1].bar(
        group_positions,
        shapley_means,
        yerr=shapley_std,
        capsize=4,
        color="#59A14F",
    )
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xticks(group_positions, GROUPS)
    axes[1].set_ylabel("Exact Shapley contribution (BA points)")
    axes[1].set_title("Average contribution across all contexts")
    axes[1].grid(axis="y", alpha=0.25)

    leave_out_means = 100.0 * np.asarray(
        [
            next(
                row["leave_one_out_drop_mean"]
                for row in contribution_summaries
                if row["group"] == group
            )
            for group in GROUPS
        ]
    )
    leave_out_std = 100.0 * np.asarray(
        [
            next(
                row["leave_one_out_drop_std"]
                for row in contribution_summaries
                if row["group"] == group
            )
            for group in GROUPS
        ]
    )
    axes[2].bar(
        group_positions,
        leave_out_means,
        yerr=leave_out_std,
        capsize=4,
        color="#F28E2B",
    )
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_xticks(group_positions, GROUPS)
    axes[2].set_ylabel("Full-model BA drop (points)")
    axes[2].set_title("Remove one group from A+B+C")
    axes[2].grid(axis="y", alpha=0.25)

    figure.suptitle("FingerMovements Phase 1g: terminal feature contribution")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def validate_only(x: np.ndarray, y: np.ndarray, seed: int, folds: int) -> None:
    training, validation = stratified_folds(y, folds, seed)[0]
    training_features, validation_features = normalized_terminal_pair(
        x[training], x[validation]
    )
    print("=== FingerMovements Phase 1g validation-only ===")
    print(
        f"full features: train={training_features.shape} | "
        f"validation={validation_features.shape}"
    )
    for subset, groups in SUBSET_GROUPS.items():
        indices = subset_indices(groups)
        if len(indices):
            classifier = LogisticRegression(
                C=LOGISTIC_C,
                solver="liblinear",
                max_iter=5_000,
            )
            classifier.fit(training_features[:, indices], y[training])
            prediction = classifier.predict(validation_features[:, indices])
        else:
            prediction = majority_prediction(y[training], len(validation))
        if prediction.shape != (len(validation),):
            raise RuntimeError(f"Validation failed for subset {subset}")
        print(f"{subset}: groups={groups or ('none',)} | features={len(indices)} OK")
    print("test: LOCKED AND NOT LOADED")


def main() -> None:
    args = parse_args()
    x, y, source_index = load_training_data(args.data.resolve())
    if args.validate_only:
        validate_only(x, y, args.seeds[0], args.folds)
        return

    print("=== FingerMovements Phase 1g terminal feature ablation ===")
    print(f"cases={len(y)} | seeds={args.seeds} | folds={args.folds}")
    print(f"classifier=L2 Logistic Regression | C={LOGISTIC_C:g}")
    for group in GROUPS:
        print(
            f"{group}: features={len(GROUP_INDICES[group])} | "
            f"{GROUP_DESCRIPTIONS[group]}"
        )
    print("all normalization and standardization: fitted from each training fold only")
    print("test: LOCKED AND NOT LOADED")

    fold_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        oof_predictions = {
            subset: np.full(len(y), -1, dtype=np.int64)
            for subset in SUBSET_GROUPS
        }

        for fold, (training, validation) in enumerate(
            stratified_folds(y, args.folds, seed), start=1
        ):
            full_training, full_validation = normalized_terminal_pair(
                x[training], x[validation]
            )
            fold_text = []
            for subset, groups in SUBSET_GROUPS.items():
                indices = subset_indices(groups)
                if len(indices):
                    classifier = LogisticRegression(
                        C=LOGISTIC_C,
                        solver="liblinear",
                        max_iter=5_000,
                    )
                    classifier.fit(full_training[:, indices], y[training])
                    training_prediction = classifier.predict(
                        full_training[:, indices]
                    ).astype(np.int64)
                    prediction = classifier.predict(
                        full_validation[:, indices]
                    ).astype(np.int64)
                else:
                    training_prediction = majority_prediction(
                        y[training], len(training)
                    )
                    prediction = majority_prediction(y[training], len(validation))

                oof_predictions[subset][validation] = prediction
                metrics = classification_metrics(y[validation], prediction)
                fold_rows.append(
                    {
                        "subset": subset,
                        "groups": "+".join(groups) if groups else "none",
                        "seed": seed,
                        "fold": fold,
                        "training_cases": len(training),
                        "validation_cases": len(validation),
                        "feature_count": len(indices),
                        "training_accuracy": float(
                            np.mean(training_prediction == y[training])
                        ),
                        **metrics,
                    }
                )
                fold_text.append(f"{subset}={metrics['balanced_accuracy']:.3f}")
            print(f"fold {fold}/{args.folds} | " + " | ".join(fold_text))

        for subset, groups in SUBSET_GROUPS.items():
            prediction = oof_predictions[subset]
            if np.any(prediction < 0):
                raise RuntimeError(f"Incomplete OOF predictions: {subset}, seed={seed}")
            metrics = classification_metrics(y, prediction)
            feature_count = len(subset_indices(groups))
            seed_rows.append(
                {
                    "subset": subset,
                    "groups": "+".join(groups) if groups else "none",
                    "seed": seed,
                    "folds": args.folds,
                    "feature_count": feature_count,
                    **metrics,
                }
            )
            for index in range(len(y)):
                prediction_rows.append(
                    {
                        "subset": subset,
                        "seed": seed,
                        "source_index": int(source_index[index]),
                        "true_label": int(y[index]),
                        "predicted_label": int(prediction[index]),
                    }
                )
            print(
                f"seed summary | {subset:>4} | features={feature_count:>3} | "
                f"BA={metrics['balanced_accuracy']:.4f} | "
                f"accuracy={metrics['accuracy']:.4f}"
            )

    subset_summaries = summarize_subsets(seed_rows)
    shapley_by_seed, shapley_detail = exact_shapley_rows(seed_rows, args.seeds)
    contribution_summaries = summarize_group_contributions(
        seed_rows, shapley_by_seed
    )
    subset_ranking = sorted(
        subset_summaries,
        key=lambda row: (
            -row["balanced_accuracy_mean"],
            -row["balanced_accuracy_min"],
            row["feature_count"],
        ),
    )
    contribution_ranking = sorted(
        contribution_summaries,
        key=lambda row: -row["shapley_mean"],
    )

    print("\n=== subset ranking ===")
    for rank, row in enumerate(subset_ranking, start=1):
        print(
            f"{rank}. {row['subset']:>4} | features={row['feature_count']:>3} | "
            f"BA={row['balanced_accuracy_mean']:.4f} ± "
            f"{row['balanced_accuracy_std']:.4f} | "
            f"worst={row['balanced_accuracy_min']:.4f}"
        )

    print("\n=== group contribution ranking ===")
    for rank, row in enumerate(contribution_ranking, start=1):
        print(
            f"{rank}. {row['group']} | "
            f"Shapley={row['shapley_percentage_points_mean']:+.3f} ± "
            f"{row['shapley_percentage_points_std']:.3f} BA points | "
            f"leave-out drop={100.0 * row['leave_one_out_drop_mean']:+.3f} points | "
            f"standalone BA={row['standalone_balanced_accuracy_mean']:.4f}"
        )
    print(
        "interpretation: use Shapley as the primary shared-contribution estimate; "
        "standalone and leave-out results reveal redundancy and interactions"
    )
    print("decision: this diagnostic does not modify the frozen model")

    report = {
        "phase": "1g",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "official TRAIN split only; official TEST not opened",
        "question": "Which frozen terminal feature group contributes most?",
        "protocol": {
            "cases": len(y),
            "seeds": args.seeds,
            "folds": args.folds,
            "classifier": "L2 Logistic Regression with liblinear",
            "logistic_c": LOGISTIC_C,
            "subsets": {name: list(groups) for name, groups in SUBSET_GROUPS.items()},
            "fold_training_only_preprocessing": True,
            "empty_subset": "training-fold majority-class intercept-only reference",
            "primary_contribution_method": "exact three-player Shapley value on OOF balanced accuracy",
            "secondary_checks": [
                "standalone group OOF balanced accuracy",
                "full-model OOF balanced-accuracy drop after removing one group",
            ],
        },
        "feature_groups": {
            group: {
                "description": GROUP_DESCRIPTIONS[group],
                "feature_count": len(GROUP_INDICES[group]),
            }
            for group in GROUPS
        },
        "subset_summary": subset_summaries,
        "subset_ranking": [row["subset"] for row in subset_ranking],
        "group_contribution_summary": contribution_summaries,
        "group_contribution_ranking": [
            row["group"] for row in contribution_ranking
        ],
        "decision_policy": (
            "Do not change the frozen model from this diagnostic alone. Review "
            "mean, seed variability, redundancy, and interactions before proposing "
            "a smaller feature set."
        ),
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "phase1g_fold_results.csv", fold_rows)
    write_csv(output_dir / "phase1g_seed_results.csv", seed_rows)
    write_csv(output_dir / "phase1g_subset_summary.csv", subset_summaries)
    write_csv(output_dir / "phase1g_predictions.csv", prediction_rows)
    write_csv(output_dir / "phase1g_shapley_by_seed.csv", shapley_by_seed)
    write_csv(output_dir / "phase1g_shapley_detail.csv", shapley_detail)
    write_csv(
        output_dir / "phase1g_group_contribution_summary.csv",
        contribution_summaries,
    )
    metrics_path = output_dir / "phase1g_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    figure_path = output_dir / "phase1g_terminal_feature_contribution.png"
    save_figure(figure_path, subset_summaries, contribution_summaries)
    print(f"metrics: {metrics_path}")
    print(f"figure: {figure_path}")


if __name__ == "__main__":
    main()
