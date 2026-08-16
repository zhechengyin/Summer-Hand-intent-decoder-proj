"""Phase 1c: extend Feature + Linear to 50 epochs.

Only training duration changes. The model, features, optimizer, learning rate,
weight decay, dropout, batch size, seeds, and stratified folds are identical to
the original Phase 1c run. The official test split is never loaded.

Epoch 20 must reproduce the original Feature + Linear OOF predictions exactly.
The epoch-50 result is paired with the completed Tiny EEGNet epoch-50 result.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import binomtest
from torch import nn
from torch.nn import functional as F


LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.25
BATCH_SIZE = 32
EPOCHS = 50
MILESTONES = (20, 30, 40, 50)
SEEDS = (42, 43, 44)
FOLDS = 5

CASES = 316
CHANNELS = 28
TIMEPOINTS = 50
FEATURES = 196
CLASS_COUNTS = {0: 159, 1: 157}


class FeatureLinear(nn.Module):
    """The unchanged Phase 1c handcrafted-feature linear classifier."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(FEATURES, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "data/processed/finger_movements/train.npz",
    )
    parser.add_argument(
        "--phase1c-predictions",
        type=Path,
        default=(
            root
            / "results/finger_movements/phase1c_representation_comparison"
            / "phase1c_predictions.csv"
        ),
    )
    parser.add_argument(
        "--eegnet-milestone-predictions",
        type=Path,
        default=(
            root
            / "results/finger_movements/phase1c_eegnet_epoch_check"
            / "phase1c_eegnet_milestone_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results/finger_movements/phase1c_feature_linear_50epoch",
    )
    parser.add_argument(
        "--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu"
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


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
    if path.name.lower() == "test.npz":
        raise ValueError("Phase 1c refuses to load test.npz")
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
        raise ValueError(f"Unexpected labels or counts: {observed}")
    if not np.array_equal(source_index, np.arange(CASES)):
        raise ValueError("source_index must preserve the official row order")
    return x, y, source_index


def load_original_feature_predictions(
    path: Path, y: np.ndarray
) -> dict[int, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Original Phase 1c predictions not found: {path}")
    output = {
        seed: np.full(len(y), -1, dtype=np.int64)
        for seed in SEEDS
    }
    seen: defaultdict[int, set[int]] = defaultdict(set)
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row["model"] != "feature_linear":
                continue
            seed = int(row["seed"])
            if seed not in SEEDS:
                continue
            index = int(row["source_index"])
            if index in seen[seed]:
                raise ValueError(f"Duplicate original prediction for seed {seed}")
            if int(row["true_label"]) != int(y[index]):
                raise ValueError("Original prediction labels do not match train.npz")
            output[seed][index] = int(row["predicted_label"])
            seen[seed].add(index)
    for seed, predictions in output.items():
        if len(seen[seed]) != len(y) or np.any(predictions < 0):
            raise ValueError(f"Incomplete original predictions for seed {seed}")
    return output


def load_eegnet_epoch50_predictions(
    path: Path, y: np.ndarray
) -> dict[int, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"EEGNet milestone predictions not found: {path}; run the Phase 1c "
            "EEGNet epoch check first"
        )
    output = {
        seed: np.full(len(y), -1, dtype=np.int64)
        for seed in SEEDS
    }
    seen: defaultdict[int, set[int]] = defaultdict(set)
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if int(row["epoch"]) != 50:
                continue
            seed = int(row["seed"])
            if seed not in SEEDS:
                continue
            index = int(row["source_index"])
            if index in seen[seed]:
                raise ValueError(f"Duplicate EEGNet prediction for seed {seed}")
            if int(row["true_label"]) != int(y[index]):
                raise ValueError("EEGNet prediction labels do not match train.npz")
            output[seed][index] = int(row["predicted_label"])
            seen[seed].add(index)
    for seed, predictions in output.items():
        if len(seen[seed]) != len(y) or np.any(predictions < 0):
            raise ValueError(f"Incomplete EEGNet epoch-50 predictions for seed {seed}")
    return output


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


def normalize_channels(
    training: np.ndarray, validation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = training.mean(axis=(0, 2), keepdims=True, dtype=np.float64)
    std = np.maximum(
        training.std(axis=(0, 2), keepdims=True, dtype=np.float64), 1e-6
    )
    return (
        ((training - mean) / std).astype(np.float32),
        ((validation - mean) / std).astype(np.float32),
    )


def handcrafted_features(x: np.ndarray) -> np.ndarray:
    blocks = [x.mean(-1), x.std(-1), np.square(x).mean(-1)]
    spectrum = np.square(np.abs(np.fft.rfft(x, axis=-1))) / x.shape[-1]
    frequencies = np.fft.rfftfreq(x.shape[-1], d=0.01)
    for low, high in ((1, 4), (4, 8), (8, 13), (13, 30)):
        mask = (frequencies >= low) & (frequencies < high)
        blocks.append(spectrum[..., mask].mean(-1))
    output = np.concatenate(blocks, axis=1).astype(np.float32)
    if output.shape[1] != FEATURES:
        raise RuntimeError(f"Unexpected feature count: {output.shape[1]}")
    return output


def prepare_fold(
    training: np.ndarray, validation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    training, validation = normalize_channels(training, validation)
    training = handcrafted_features(training)
    validation = handcrafted_features(validation)
    mean = training.mean(axis=0, keepdims=True, dtype=np.float64)
    std = np.maximum(
        training.std(axis=0, keepdims=True, dtype=np.float64), 1e-6
    )
    return (
        ((training - mean) / std).astype(np.float32),
        ((validation - mean) / std).astype(np.float32),
    )


def batches(
    size: int, batch_size: int, rng: np.random.Generator | None = None
) -> list[np.ndarray]:
    order = np.arange(size)
    if rng is not None:
        rng.shuffle(order)
    return [order[start : start + batch_size] for start in range(0, size, batch_size)]


def evaluate(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    loss_sum = 0.0
    predictions = []
    probabilities = []
    with torch.inference_mode():
        for indices in batches(len(y), BATCH_SIZE):
            index = torch.as_tensor(indices, device=y.device)
            logits = model(x[index])
            loss_sum += float(
                F.cross_entropy(logits, y[index], reduction="sum").item()
            )
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return (
        loss_sum / len(y),
        np.concatenate(predictions),
        np.concatenate(probabilities),
    )


def classification_metrics(
    actual: np.ndarray, predicted: np.ndarray
) -> dict[str, Any]:
    confusion = np.zeros((2, 2), dtype=np.int64)
    for truth, guess in zip(actual, predicted, strict=True):
        confusion[int(truth), int(guess)] += 1
    recalls, f1_scores = [], []
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


def binary_cross_entropy(y: np.ndarray, probability: np.ndarray) -> float:
    probability = np.clip(probability, 1e-7, 1.0 - 1e-7)
    return float(
        -np.mean(y * np.log(probability) + (1 - y) * np.log(1 - probability))
    )


def train_fold(
    seed: int,
    fold: int,
    training_x: np.ndarray,
    training_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[int, tuple[np.ndarray, np.ndarray]]]:
    run_seed = seed * 1_000 + fold
    set_seed(run_seed)
    training_x, validation_x = prepare_fold(training_x, validation_x)
    model = FeatureLinear().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    tx = torch.from_numpy(training_x).to(device)
    ty = torch.from_numpy(training_y).to(device)
    vx = torch.from_numpy(validation_x).to(device)
    vy = torch.from_numpy(validation_y).to(device)
    history: list[dict[str, Any]] = []
    validation_outputs: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_rng = np.random.default_rng(run_seed * 100 + epoch)
        optimization_loss = 0.0
        for indices in batches(len(ty), BATCH_SIZE, epoch_rng):
            index = torch.as_tensor(indices, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(tx[index]), ty[index])
            loss.backward()
            optimizer.step()
            optimization_loss += float(loss.item()) * len(indices)

        train_loss, train_prediction, _ = evaluate(model, tx, ty)
        validation_loss, validation_prediction, validation_probability = evaluate(
            model, vx, vy
        )
        train_accuracy = float(np.mean(train_prediction == training_y))
        validation_accuracy = float(
            np.mean(validation_prediction == validation_y)
        )
        history.append(
            {
                "seed": seed,
                "fold": fold + 1,
                "epoch": epoch,
                "optimization_loss": optimization_loss / len(training_y),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        validation_outputs[epoch] = (
            validation_prediction,
            validation_probability,
        )
        marker = " milestone" if epoch in MILESTONES else ""
        print(
            f"epoch {epoch:02d}/{EPOCHS} | "
            f"train loss={train_loss:.5f} acc={train_accuracy:.4f} | "
            f"validation loss={validation_loss:.5f} "
            f"acc={validation_accuracy:.4f}{marker}"
        )
    return history, validation_outputs


def summarize_epochs(seed_epoch_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for epoch in range(1, EPOCHS + 1):
        rows = [row for row in seed_epoch_rows if row["epoch"] == epoch]
        summary: dict[str, Any] = {"epoch": epoch}
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "validation_loss",
        ):
            values = np.asarray([row[metric] for row in rows], dtype=float)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=1))
            summary[f"{metric}_min"] = float(values.min())
            summary[f"{metric}_max"] = float(values.max())
        output.append(summary)
    return output


def add_training_summary(
    fold_history: list[dict[str, Any]], epoch_summaries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    for summary in epoch_summaries:
        rows = [row for row in fold_history if row["epoch"] == summary["epoch"]]
        output.append(
            {
                **summary,
                "train_loss_mean": float(
                    np.mean([row["train_loss"] for row in rows])
                ),
                "train_accuracy_mean": float(
                    np.mean([row["train_accuracy"] for row in rows])
                ),
            }
        )
    return output


def paired_comparison(
    actual: np.ndarray,
    feature: np.ndarray,
    eegnet: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    feature_correct = feature == actual
    eegnet_correct = eegnet == actual
    feature_only = int(np.sum(feature_correct & ~eegnet_correct))
    eegnet_only = int(np.sum(eegnet_correct & ~feature_correct))
    discordant = feature_only + eegnet_only
    p_value = (
        float(
            binomtest(
                min(feature_only, eegnet_only),
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
        "feature_50_minus_eegnet_50_accuracy": float(
            feature_correct.mean() - eegnet_correct.mean()
        ),
        "feature_only_correct": feature_only,
        "eegnet_only_correct": eegnet_only,
        "both_correct": int(np.sum(feature_correct & eegnet_correct)),
        "both_wrong": int(np.sum(~feature_correct & ~eegnet_correct)),
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


def save_figure(
    path: Path,
    epoch_summaries: list[dict[str, Any]],
    eegnet_epoch50_balanced: float,
) -> None:
    epochs = np.asarray([row["epoch"] for row in epoch_summaries])
    train_accuracy = 100.0 * np.asarray(
        [row["train_accuracy_mean"] for row in epoch_summaries]
    )
    validation_accuracy = 100.0 * np.asarray(
        [row["balanced_accuracy_mean"] for row in epoch_summaries]
    )
    validation_std = 100.0 * np.asarray(
        [row["balanced_accuracy_std"] for row in epoch_summaries]
    )
    validation_loss = np.asarray(
        [row["validation_loss_mean"] for row in epoch_summaries]
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(epochs, train_accuracy, label="Train accuracy", color="#4C78A8")
    axes[0].plot(
        epochs, validation_accuracy, label="OOF balanced accuracy", color="#F58518"
    )
    axes[0].fill_between(
        epochs,
        validation_accuracy - validation_std,
        validation_accuracy + validation_std,
        color="#F58518",
        alpha=0.18,
        label="OOF ±1 seed SD",
    )
    axes[0].axhline(
        100.0 * eegnet_epoch50_balanced,
        color="#54A24B",
        linestyle="--",
        label="Tiny EEGNet epoch 50",
    )
    axes[0].axvline(50, color="black", linestyle=":")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Feature + Linear trajectory")
    axes[0].legend(fontsize=8)

    axes[1].plot(epochs, validation_loss, color="#E45756")
    for milestone in MILESTONES:
        axes[1].axvline(milestone, color="gray", linewidth=0.7, alpha=0.35)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("OOF cross-entropy")
    axes[1].set_title("Validation loss trajectory")
    figure.suptitle("FingerMovements Phase 1c Feature + Linear 50-epoch check")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def validate_only(
    x: np.ndarray,
    y: np.ndarray,
    original: dict[int, np.ndarray],
    eegnet: dict[int, np.ndarray],
    device: torch.device,
) -> None:
    folds = stratified_folds(y, FOLDS, SEEDS[0])
    held_out = np.concatenate([validation for _, validation in folds])
    if not np.array_equal(np.sort(held_out), np.arange(len(y))):
        raise RuntimeError("Fold coverage check failed")
    training, validation = folds[0]
    _, validation_x = prepare_fold(x[training], x[validation])
    model = FeatureLinear().to(device)
    with torch.inference_mode():
        output = model(torch.from_numpy(validation_x[:4]).to(device))
    if output.shape != (4, 2) or not torch.isfinite(output).all():
        raise RuntimeError("Feature + Linear forward check failed")
    for seed in SEEDS:
        if np.any(original[seed] < 0) or np.any(eegnet[seed] < 0):
            raise RuntimeError("Reference prediction validation failed")
    print("=== Phase 1c Feature + Linear 50-epoch validation-only ===")
    print(f"data: x={x.shape} y={y.shape}")
    print(f"features: input={validation_x[:4].shape} output={tuple(output.shape)}")
    print("fold coverage: PASS")
    print("original Feature and EEGNet epoch-50 predictions: PASS")
    print("test: LOCKED AND NOT LOADED")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    x, y, source_index = load_training_data(args.data.resolve())
    original = load_original_feature_predictions(
        args.phase1c_predictions.resolve(), y
    )
    eegnet = load_eegnet_epoch50_predictions(
        args.eegnet_milestone_predictions.resolve(), y
    )
    if args.validate_only:
        validate_only(x, y, original, eegnet, device)
        return

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print("=== FingerMovements Phase 1c Feature + Linear 50-epoch check ===")
    print(f"official train: x={x.shape} y={y.shape} | test=LOCKED AND NOT LOADED")
    print(
        f"epochs={EPOCHS} | milestones={MILESTONES} | seeds={SEEDS} | "
        f"folds={FOLDS} | device={device}"
    )
    print(
        f"fixed: AdamW lr={LEARNING_RATE:g} wd={WEIGHT_DECAY:g} "
        f"dropout={DROPOUT:g} batch={BATCH_SIZE}"
    )
    print("comparison: Feature + Linear epoch 50 versus Tiny EEGNet epoch 50")
    print("policy: fold-training-only preprocessing; no checkpoint; test not loaded")

    fold_history: list[dict[str, Any]] = []
    seed_epoch_rows: list[dict[str, Any]] = []
    milestone_prediction_rows: list[dict[str, Any]] = []
    feature_epoch50: dict[int, np.ndarray] = {}
    reproduction = []

    for seed in SEEDS:
        oof_predictions = {
            epoch: np.full(len(y), -1, dtype=np.int64)
            for epoch in range(1, EPOCHS + 1)
        }
        oof_probabilities = {
            epoch: np.full(len(y), np.nan, dtype=np.float64)
            for epoch in range(1, EPOCHS + 1)
        }
        print(f"\n=== Feature + Linear | seed={seed} ===")
        for fold, (training, validation) in enumerate(
            stratified_folds(y, FOLDS, seed)
        ):
            print(
                f"\n--- fold {fold + 1}/{FOLDS} | "
                f"train={len(training)} validation={len(validation)} ---"
            )
            history, outputs = train_fold(
                seed,
                fold,
                x[training],
                y[training],
                x[validation],
                y[validation],
                device,
            )
            fold_history.extend(history)
            for epoch, (predicted, probability) in outputs.items():
                oof_predictions[epoch][validation] = predicted
                oof_probabilities[epoch][validation] = probability

        for epoch in range(1, EPOCHS + 1):
            predictions = oof_predictions[epoch]
            probabilities = oof_probabilities[epoch]
            if np.any(predictions < 0) or not np.isfinite(probabilities).all():
                raise RuntimeError("Incomplete OOF prediction coverage")
            metrics = classification_metrics(y, predictions)
            metrics.update(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "validation_loss": binary_cross_entropy(y, probabilities),
                }
            )
            seed_epoch_rows.append(metrics)
            if epoch in MILESTONES:
                for index in range(len(y)):
                    milestone_prediction_rows.append(
                        {
                            "seed": seed,
                            "epoch": epoch,
                            "source_index": int(source_index[index]),
                            "true_label": int(y[index]),
                            "predicted_label": int(predictions[index]),
                            "probability_right": float(probabilities[index]),
                            "correct": int(predictions[index] == y[index]),
                        }
                    )
        feature_epoch50[seed] = oof_predictions[50].copy()
        reproduction.append(
            {
                "seed": seed,
                "epoch_20_prediction_mismatches": int(
                    np.sum(oof_predictions[20] != original[seed])
                ),
            }
        )
        print(
            "seed milestones | "
            + " | ".join(
                f"e{epoch}={classification_metrics(y, oof_predictions[epoch])['balanced_accuracy']:.4f}"
                for epoch in MILESTONES
            )
        )

    epoch_summaries = add_training_summary(
        fold_history, summarize_epochs(seed_epoch_rows)
    )
    milestone_summaries = [
        row for row in epoch_summaries if row["epoch"] in MILESTONES
    ]
    feature_50_summary = next(row for row in epoch_summaries if row["epoch"] == 50)
    eegnet_seed_metrics = []
    for seed in SEEDS:
        metrics = classification_metrics(y, eegnet[seed])
        metrics["seed"] = seed
        eegnet_seed_metrics.append(metrics)
    eegnet_50_balanced = float(
        np.mean([row["balanced_accuracy"] for row in eegnet_seed_metrics])
    )
    feature_50_balanced = float(feature_50_summary["balanced_accuracy_mean"])
    paired_rows = [
        paired_comparison(y, feature_epoch50[seed], eegnet[seed], seed)
        for seed in SEEDS
    ]
    recommended_pipeline = (
        "feature_linear_epoch_50"
        if feature_50_balanced > eegnet_50_balanced
        else "tiny_eegnet_epoch_50"
    )

    elapsed = time.monotonic() - started
    print("\n=== Registered milestone summary ===")
    for row in milestone_summaries:
        print(
            f"epoch {row['epoch']:02d} | "
            f"balanced={row['balanced_accuracy_mean']:.4f}"
            f"±{row['balanced_accuracy_std']:.4f} | "
            f"accuracy={row['accuracy_mean']:.4f} | "
            f"loss={row['validation_loss_mean']:.5f}"
        )
    print(f"Feature + Linear epoch 50 balanced={feature_50_balanced:.4f}")
    print(f"Tiny EEGNet epoch 50 balanced={eegnet_50_balanced:.4f}")
    print(f"provisional Phase 1c recommendation: {recommended_pipeline}")
    print(
        "epoch-20 reproduction mismatches: "
        + ", ".join(
            f"seed {row['seed']}={row['epoch_20_prediction_mismatches']}"
            for row in reproduction
        )
    )

    report = {
        "experiment": "phase1c_feature_linear_50epoch_check",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data": {
            "path": str(args.data.resolve()),
            "shape": list(x.shape),
            "class_counts": CLASS_COUNTS,
            "test_policy": "locked and not loaded",
        },
        "protocol": {
            "model": "feature_linear",
            "epochs": EPOCHS,
            "registered_milestones": list(MILESTONES),
            "seeds": list(SEEDS),
            "folds": FOLDS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "preprocessing": "fit on each fold training subset only",
            "comparison": "Feature + Linear epoch 50 versus Tiny EEGNet epoch 50",
            "checkpoint": "none",
            "device": str(device),
        },
        "feature_linear_epoch_50_balanced_accuracy_mean": feature_50_balanced,
        "tiny_eegnet_epoch_50_balanced_accuracy_mean": eegnet_50_balanced,
        "provisional_recommendation": recommended_pipeline,
        "epoch_20_reproduction": reproduction,
        "milestone_summaries": milestone_summaries,
        "epoch_summaries": epoch_summaries,
        "eegnet_epoch_50_seed_metrics": eegnet_seed_metrics,
        "paired_comparisons": paired_rows,
        "elapsed_seconds": elapsed,
    }
    (output_dir / "phase1c_feature_linear_50epoch_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_csv(output_dir / "phase1c_feature_linear_fold_epoch_history.csv", fold_history)
    write_csv(output_dir / "phase1c_feature_linear_seed_epoch_results.csv", seed_epoch_rows)
    write_csv(output_dir / "phase1c_feature_linear_epoch_summary.csv", epoch_summaries)
    write_csv(
        output_dir / "phase1c_feature_linear_milestone_predictions.csv",
        milestone_prediction_rows,
    )
    write_csv(output_dir / "phase1c_feature_vs_eegnet_epoch50.csv", paired_rows)
    save_figure(
        output_dir / "phase1c_feature_linear_50epoch.png",
        epoch_summaries,
        eegnet_50_balanced,
    )
    print(f"\nelapsed: {elapsed / 60.0:.1f} minutes")
    print(f"metrics: {output_dir / 'phase1c_feature_linear_50epoch_metrics.json'}")
    print(f"figure: {output_dir / 'phase1c_feature_linear_50epoch.png'}")
    print("test: LOCKED AND NOT LOADED")


if __name__ == "__main__":
    main()
