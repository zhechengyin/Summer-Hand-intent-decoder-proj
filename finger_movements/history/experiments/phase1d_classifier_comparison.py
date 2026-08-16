"""Phase 1d-B: compare linear classifiers on the frozen EEG features.

All four classifiers receive the same 196 handcrafted features. Channel and
feature normalization are fitted independently inside every training fold.
Repeated stratified folds use seeds 42, 43, and 44, matching prior phases.

The official test split is locked and is never opened by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.finger_movements.feature_linear.model import (  # noqa: E402
    FeatureLinear,
    fit_preprocessing,
    transform,
)


MODELS = ("adamw_linear", "logistic_l2", "ridge", "linear_svm")
MODEL_LABELS = {
    "adamw_linear": "AdamW + Dropout Linear",
    "logistic_l2": "L2 Logistic Regression",
    "ridge": "Ridge Classifier",
    "linear_svm": "Linear SVM",
}
SEEDS = (42, 43, 44)
FOLDS = 5
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
CASES = 316
CHANNELS = 28
TIMEPOINTS = 50
CLASS_COUNTS = {0: 159, 1: 157}


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
        default=ROOT / "results/finger_movements/phase1d_classifier_comparison",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument(
        "--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu"
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.data.name.lower() == "test.npz":
        parser.error("Phase 1d refuses to load test.npz")
    if args.folds < 2 or args.folds > min(CLASS_COUNTS.values()):
        parser.error("--folds is outside the valid range")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds contains duplicates")
    return args


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


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
    """Reproduce the fold construction used in Phase 1b/1c."""
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


def batches(size: int, rng: np.random.Generator | None = None) -> list[np.ndarray]:
    order = np.arange(size)
    if rng is not None:
        rng.shuffle(order)
    return [order[start : start + BATCH_SIZE] for start in range(0, size, BATCH_SIZE)]


def train_adamw_linear(
    training_x: np.ndarray,
    training_y: np.ndarray,
    validation_x: np.ndarray,
    seed: int,
    fold: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    run_seed = seed * 1_000 + fold
    set_seed(run_seed)
    model = FeatureLinear().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    tx = torch.from_numpy(training_x).to(device)
    ty = torch.from_numpy(training_y).to(device)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        rng = np.random.default_rng(run_seed * 100 + epoch)
        for indices in batches(len(ty), rng):
            index = torch.as_tensor(indices, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(tx[index]), ty[index])
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        training_prediction = model(tx).argmax(dim=1).cpu().numpy()
        logits = model(torch.from_numpy(validation_x).to(device))
        validation_prediction = logits.argmax(dim=1).cpu().numpy()
        score = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    training_accuracy = float(np.mean(training_prediction == training_y))
    return validation_prediction, score, training_accuracy


def fit_classical_classifier(
    model_name: str,
    training_x: np.ndarray,
    training_y: np.ndarray,
    validation_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    if model_name == "logistic_l2":
        classifier = LogisticRegression(
            C=1.0, solver="liblinear", max_iter=5_000
        )
    elif model_name == "ridge":
        classifier = RidgeClassifier(alpha=1.0)
    elif model_name == "linear_svm":
        classifier = LinearSVC(C=1.0, dual=False, max_iter=10_000)
    else:
        raise ValueError(f"Unsupported classical classifier: {model_name}")
    classifier.fit(training_x, training_y)
    prediction = classifier.predict(validation_x).astype(np.int64)
    score = np.asarray(classifier.decision_function(validation_x), dtype=np.float64)
    training_accuracy = float(np.mean(classifier.predict(training_x) == training_y))
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
    candidate_name: str,
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
        "baseline": "adamw_linear",
        "candidate": candidate_name,
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


def save_figure(path: Path, summaries: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(MODELS))
    for position, model_name in enumerate(MODELS):
        rows = [row for row in summaries if row["model"] == model_name]
        values = 100.0 * np.asarray([row["balanced_accuracy"] for row in rows])
        axis.scatter(
            np.full(len(values), position),
            values,
            s=48,
            color="#4C78A8",
            zorder=3,
            label="Seed result" if position == 0 else None,
        )
        axis.errorbar(
            position,
            values.mean(),
            yerr=values.std(ddof=1),
            fmt="o",
            color="#E45756",
            capsize=5,
            markersize=7,
            label="Mean ± seed SD" if position == 0 else None,
        )
    axis.axhline(50.0, color="black", linestyle="--", linewidth=1, label="Chance")
    axis.set_xticks(positions, [MODEL_LABELS[name] for name in MODELS], rotation=12)
    axis.set_ylabel("OOF balanced accuracy (%)")
    axis.set_title("FingerMovements Phase 1d classifier comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def validate_only(
    x: np.ndarray, y: np.ndarray, seed: int, fold_count: int, device: torch.device
) -> None:
    training, validation = stratified_folds(y, fold_count, seed)[0]
    train_features, validation_features = prepare_fold(x[training], x[validation])
    if train_features.shape[1] != 196 or validation_features.shape[1] != 196:
        raise RuntimeError("Unexpected handcrafted feature count")
    set_seed(seed * 1_000)
    model = FeatureLinear().to(device).eval()
    with torch.inference_mode():
        output = model(torch.from_numpy(validation_features[:4]).to(device))
    if output.shape != (4, 2) or not torch.isfinite(output).all():
        raise RuntimeError("AdamW linear model validation failed")
    for model_name in MODELS[1:]:
        prediction, score, _ = fit_classical_classifier(
            model_name,
            train_features,
            y[training],
            validation_features,
        )
        if prediction.shape != (len(validation),) or not np.isfinite(score).all():
            raise RuntimeError(f"{model_name} validation failed")
    print("=== FingerMovements Phase 1d-B validation-only ===")
    print(f"train features={train_features.shape} | validation={validation_features.shape}")
    print("all four classifier paths: PASS")
    print("test: LOCKED AND NOT LOADED")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    x, y, source_index = load_training_data(args.data.resolve())
    if args.validate_only:
        validate_only(x, y, args.seeds[0], args.folds, device)
        return

    print("=== FingerMovements Phase 1d-B classifier comparison ===")
    print(
        f"cases={len(y)} | seeds={args.seeds} | folds={args.folds} | "
        f"AdamW epochs={EPOCHS} | device={device}"
    )
    print("features: identical 196D handcrafted representation for every classifier")
    print("preprocessing: fitted from each fold's training cases only")
    print("test: LOCKED AND NOT LOADED")

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    predictions_by_seed: dict[int, dict[str, np.ndarray]] = {}

    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        oof = {
            model_name: np.full(len(y), -1, dtype=np.int64)
            for model_name in MODELS
        }
        scores = {
            model_name: np.full(len(y), np.nan, dtype=np.float64)
            for model_name in MODELS
        }
        for fold, (training, validation) in enumerate(
            stratified_folds(y, args.folds, seed), start=1
        ):
            train_features, validation_features = prepare_fold(
                x[training], x[validation]
            )
            for model_name in MODELS:
                if model_name == "adamw_linear":
                    prediction, score, training_accuracy = train_adamw_linear(
                        train_features,
                        y[training],
                        validation_features,
                        seed,
                        fold - 1,
                        device,
                    )
                else:
                    prediction, score, training_accuracy = fit_classical_classifier(
                        model_name,
                        train_features,
                        y[training],
                        validation_features,
                    )
                oof[model_name][validation] = prediction
                scores[model_name][validation] = score
                metrics = classification_metrics(y[validation], prediction)
                fold_rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "fold": fold,
                        "train_cases": len(training),
                        "validation_cases": len(validation),
                        "training_accuracy": training_accuracy,
                        **metrics,
                    }
                )
                print(
                    f"fold {fold}/{args.folds} | {MODEL_LABELS[model_name]:>23} | "
                    f"train acc={training_accuracy:.4f} | "
                    f"validation BA={metrics['balanced_accuracy']:.4f}"
                )

        predictions_by_seed[seed] = oof
        for model_name in MODELS:
            if np.any(oof[model_name] < 0) or not np.isfinite(scores[model_name]).all():
                raise RuntimeError(f"Incomplete OOF results for {model_name}, seed={seed}")
            metrics = classification_metrics(y, oof[model_name])
            seed_rows.append({"model": model_name, "seed": seed, **metrics})
            for index in range(len(y)):
                prediction_rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "source_index": int(source_index[index]),
                        "true_label": int(y[index]),
                        "predicted_label": int(oof[model_name][index]),
                        "score": float(scores[model_name][index]),
                    }
                )
            print(
                f"seed summary | {MODEL_LABELS[model_name]:>23} | "
                f"accuracy={metrics['accuracy']:.4f} | "
                f"balanced accuracy={metrics['balanced_accuracy']:.4f} | "
                f"macro F1={metrics['macro_f1']:.4f}"
            )

    aggregate_rows = []
    for model_name in MODELS:
        model_seed_rows = [row for row in seed_rows if row["model"] == model_name]
        aggregate: dict[str, Any] = {"model": model_name}
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            values = np.asarray([row[metric] for row in model_seed_rows], dtype=float)
            aggregate[f"{metric}_mean"] = float(values.mean())
            aggregate[f"{metric}_std"] = float(values.std(ddof=1))
            aggregate[f"{metric}_min"] = float(values.min())
            aggregate[f"{metric}_max"] = float(values.max())
        aggregate_rows.append(aggregate)

    paired_rows = []
    for seed in args.seeds:
        for candidate in MODELS[1:]:
            paired_rows.append(
                paired_comparison(
                    y,
                    predictions_by_seed[seed]["adamw_linear"],
                    predictions_by_seed[seed][candidate],
                    seed,
                    candidate,
                )
            )

    ranking = sorted(
        aggregate_rows,
        key=lambda row: row["balanced_accuracy_mean"],
        reverse=True,
    )
    print("\n=== aggregate ranking by mean OOF balanced accuracy ===")
    for rank, row in enumerate(ranking, start=1):
        print(
            f"{rank}. {MODEL_LABELS[row['model']]}: "
            f"{row['balanced_accuracy_mean']:.4f} ± "
            f"{row['balanced_accuracy_std']:.4f}; "
            f"worst={row['balanced_accuracy_min']:.4f}"
        )
    print("interpretation: diagnostic comparison only; official test remains locked")

    report = {
        "phase": "1d-B",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "official TRAIN split only; official TEST not opened",
        "protocol": {
            "cases": len(y),
            "seeds": args.seeds,
            "folds": args.folds,
            "features": "196D active handcrafted Feature + Linear representation",
            "fold_training_only_preprocessing": True,
            "adamw_linear": {
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "dropout": 0.25,
            },
            "logistic_l2": {"C": 1.0, "solver": "liblinear"},
            "ridge": {"alpha": 1.0},
            "linear_svm": {"C": 1.0, "dual": False},
        },
        "aggregate_results": aggregate_rows,
        "ranking": [row["model"] for row in ranking],
        "selection_policy": (
            "Do not select from mean alone. Review seed stability, worst seed, "
            "paired disagreements, and Phase 1d-A sanity results first."
        ),
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "phase1d_classifier_fold_results.csv", fold_rows)
    write_csv(output_dir / "phase1d_classifier_seed_results.csv", seed_rows)
    write_csv(output_dir / "phase1d_classifier_predictions.csv", prediction_rows)
    write_csv(output_dir / "phase1d_classifier_paired_comparisons.csv", paired_rows)
    metrics_path = output_dir / "phase1d_classifier_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    figure_path = output_dir / "phase1d_classifier_comparison.png"
    save_figure(figure_path, seed_rows)
    print(f"metrics: {metrics_path}")
    print(f"figure: {figure_path}")


if __name__ == "__main__":
    main()
