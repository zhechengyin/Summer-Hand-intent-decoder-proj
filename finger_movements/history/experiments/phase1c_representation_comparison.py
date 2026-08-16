"""Phase 1c: compare three FingerMovements representations fairly.

The official train split is evaluated with repeated stratified cross-validation.
The official test split is never loaded. Every learned preprocessing operation
is fitted from the current fold's training subset only.

Compared pipelines:
1. handcrafted per-channel features + linear classifier;
2. Tiny EEGNet on normalized EEG;
3. regularized CSP log-variance features + shrinkage LDA.

The file is self-contained and does not import Phase 1b or retired code.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import eigh
from scipy.stats import binomtest
from torch import nn
from torch.nn import functional as F


MODELS = ("feature_linear", "tiny_eegnet", "csp_lda")
MODEL_LABELS = {
    "feature_linear": "Feature + Linear",
    "tiny_eegnet": "Tiny EEGNet",
    "csp_lda": "Regularized CSP + LDA",
}

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.25
EPOCHS = 20
BATCH_SIZE = 32
SEEDS = (42, 43, 44)
FOLDS = 5

CSP_COMPONENTS = 6
CSP_SHRINKAGE = 0.1
LDA_SHRINKAGE = 0.1

CASES = 316
CHANNELS = 28
TIMEPOINTS = 50
CLASS_COUNTS = {0: 159, 1: 157}


class FeatureLinear(nn.Module):
    def __init__(self, feature_count: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(feature_count, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TinyEEGNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 8, (1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(8),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(8, 16, (CHANNELS, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(DROPOUT),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(16, 16, (1, 7), padding=(0, 3), groups=16, bias=False),
            nn.Conv2d(16, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(DROPOUT),
        )
        self.output = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x.unsqueeze(1))
        x = self.spatial(x)
        x = self.separable(x)
        return self.output(x.flatten(1))


class RegularizedCSPLDA:
    """Six-component regularized CSP followed by shrinkage binary LDA."""

    def __init__(self) -> None:
        self.channel_mean: np.ndarray | None = None
        self.channel_std: np.ndarray | None = None
        self.filters: np.ndarray | None = None
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.lda_weight: np.ndarray | None = None
        self.lda_bias: float | None = None

    @staticmethod
    def _trial_covariances(x: np.ndarray) -> np.ndarray:
        centered = x - x.mean(axis=2, keepdims=True)
        covariance = np.einsum("nct,ndt->ncd", centered, centered)
        traces = np.trace(covariance, axis1=1, axis2=2)
        if np.any(traces <= 1e-12):
            raise ValueError("CSP received a zero-variance trial")
        return covariance / traces[:, None, None]

    @staticmethod
    def _regularize_covariance(covariance: np.ndarray, alpha: float) -> np.ndarray:
        scale = float(np.trace(covariance) / covariance.shape[0])
        return (
            (1.0 - alpha) * covariance
            + alpha * scale * np.eye(covariance.shape[0])
        )

    @staticmethod
    def _csp_features(x: np.ndarray, filters: np.ndarray) -> np.ndarray:
        centered = x - x.mean(axis=2, keepdims=True)
        projected = np.einsum("kc,nct->nkt", filters, centered)
        variance = projected.var(axis=2, ddof=1)
        relative_variance = variance / np.maximum(
            variance.sum(axis=1, keepdims=True), 1e-12
        )
        return np.log(np.maximum(relative_variance, 1e-12))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RegularizedCSPLDA":
        self.channel_mean, self.channel_std = fit_channel_normalization(x)
        normalized = apply_channel_normalization(
            x, self.channel_mean, self.channel_std
        )

        trial_covariances = self._trial_covariances(normalized)
        class_covariances = []
        for label in (0, 1):
            covariance = trial_covariances[y == label].mean(axis=0)
            class_covariances.append(
                self._regularize_covariance(covariance, CSP_SHRINKAGE)
            )
        covariance_0, covariance_1 = class_covariances
        eigenvalues, eigenvectors = eigh(
            covariance_1, covariance_0 + covariance_1, check_finite=True
        )
        order = np.argsort(eigenvalues)
        pair_count = CSP_COMPONENTS // 2
        selected = []
        for offset in range(pair_count):
            selected.extend([order[-1 - offset], order[offset]])
        self.filters = eigenvectors[:, selected].T.astype(np.float64)

        csp_features = self._csp_features(normalized, self.filters)
        self.feature_mean = csp_features.mean(axis=0, keepdims=True)
        self.feature_std = np.maximum(
            csp_features.std(axis=0, keepdims=True), 1e-8
        )
        standardized = (csp_features - self.feature_mean) / self.feature_std

        class_means = [standardized[y == label].mean(axis=0) for label in (0, 1)]
        residuals = np.concatenate(
            [standardized[y == label] - class_means[label] for label in (0, 1)],
            axis=0,
        )
        pooled_covariance = residuals.T @ residuals / max(len(y) - 2, 1)
        pooled_covariance = self._regularize_covariance(
            pooled_covariance, LDA_SHRINKAGE
        )
        mean_0, mean_1 = class_means
        self.lda_weight = np.linalg.solve(pooled_covariance, mean_1 - mean_0)
        prior_0 = float(np.mean(y == 0))
        prior_1 = float(np.mean(y == 1))
        self.lda_bias = float(
            -0.5 * (mean_1 + mean_0) @ self.lda_weight
            + np.log(prior_1 / prior_0)
        )
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        required = (
            self.channel_mean,
            self.channel_std,
            self.filters,
            self.feature_mean,
            self.feature_std,
            self.lda_weight,
            self.lda_bias,
        )
        if any(value is None for value in required):
            raise RuntimeError("RegularizedCSPLDA must be fitted before prediction")
        normalized = apply_channel_normalization(
            x, self.channel_mean, self.channel_std
        )
        csp_features = self._csp_features(normalized, self.filters)
        standardized = (csp_features - self.feature_mean) / self.feature_std
        return standardized @ self.lda_weight + self.lda_bias

    def predict_probability(self, x: np.ndarray) -> np.ndarray:
        score = np.clip(self.decision_function(x), -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-score))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "data/processed/finger_movements/train.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results/finger_movements/phase1c_representation_comparison",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu"
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not 2 <= args.folds <= min(CLASS_COUNTS.values()):
        parser.error("--folds is outside the valid range")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if len(set(args.seeds)) != len(args.seeds):
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
    if path.name.lower() == "test.npz":
        raise ValueError("Phase 1c refuses to load test.npz")
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run the FingerMovements converter first"
        )
    with np.load(path, allow_pickle=False) as data:
        x = data["x"].astype(np.float32, copy=True)
        y = data["y"].astype(np.int64, copy=True)
        source_index = data["source_index"].astype(np.int64, copy=True)
    if x.shape != (CASES, CHANNELS, TIMEPOINTS) or y.shape != (CASES,):
        raise ValueError(f"Unexpected data shapes: x={x.shape}, y={y.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Input contains non-finite values")
    values, counts = np.unique(y, return_counts=True)
    if dict(zip(values.tolist(), counts.tolist(), strict=True)) != CLASS_COUNTS:
        raise ValueError("Unexpected class labels or counts")
    if not np.array_equal(source_index, np.arange(CASES)):
        raise ValueError("source_index must preserve the official row order")
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


def fit_channel_normalization(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=(0, 2), keepdims=True, dtype=np.float64)
    std = np.maximum(
        x.std(axis=(0, 2), keepdims=True, dtype=np.float64), 1e-6
    )
    return mean, std


def apply_channel_normalization(
    x: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return ((x - mean) / std).astype(np.float64, copy=False)


def handcrafted_features(x: np.ndarray) -> np.ndarray:
    feature_blocks = [x.mean(-1), x.std(-1), np.square(x).mean(-1)]
    spectrum = np.square(np.abs(np.fft.rfft(x, axis=-1))) / x.shape[-1]
    frequencies = np.fft.rfftfreq(x.shape[-1], d=0.01)
    for low, high in ((1, 4), (4, 8), (8, 13), (13, 30)):
        mask = (frequencies >= low) & (frequencies < high)
        feature_blocks.append(spectrum[..., mask].mean(-1))
    return np.concatenate(feature_blocks, axis=1).astype(np.float32)


def prepare_torch_inputs(
    model_name: str, training: np.ndarray, validation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    channel_mean, channel_std = fit_channel_normalization(training)
    training = apply_channel_normalization(
        training, channel_mean, channel_std
    ).astype(np.float32)
    validation = apply_channel_normalization(
        validation, channel_mean, channel_std
    ).astype(np.float32)
    if model_name == "feature_linear":
        training = handcrafted_features(training)
        validation = handcrafted_features(validation)
        feature_mean = training.mean(axis=0, keepdims=True, dtype=np.float64)
        feature_std = np.maximum(
            training.std(axis=0, keepdims=True, dtype=np.float64), 1e-6
        )
        training = ((training - feature_mean) / feature_std).astype(np.float32)
        validation = ((validation - feature_mean) / feature_std).astype(np.float32)
    return training, validation


def build_torch_model(model_name: str, input_shape: tuple[int, ...]) -> nn.Module:
    if model_name == "feature_linear":
        return FeatureLinear(input_shape[0])
    if model_name == "tiny_eegnet":
        return TinyEEGNet()
    raise ValueError(f"Not a Torch model: {model_name}")


def batches(
    size: int, batch_size: int, rng: np.random.Generator | None = None
) -> list[np.ndarray]:
    order = np.arange(size)
    if rng is not None:
        rng.shuffle(order)
    return [order[start : start + batch_size] for start in range(0, size, batch_size)]


def evaluate_torch(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    loss_sum = 0.0
    predictions = []
    probabilities = []
    with torch.inference_mode():
        for indices in batches(len(y), batch_size):
            index = torch.as_tensor(indices, device=y.device)
            logits = model(x[index])
            loss_sum += float(
                F.cross_entropy(logits, y[index], reduction="sum").item()
            )
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    probability = np.concatenate(probabilities)
    return loss_sum / len(y), np.concatenate(predictions), probability


def binary_cross_entropy(y: np.ndarray, probability: np.ndarray) -> float:
    probability = np.clip(probability, 1e-7, 1.0 - 1e-7)
    return float(
        -np.mean(y * np.log(probability) + (1 - y) * np.log(1 - probability))
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


def train_torch_fold(
    model_name: str,
    seed: int,
    fold: int,
    training_x: np.ndarray,
    training_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray, np.ndarray]:
    run_seed = seed * 1_000 + fold
    set_seed(run_seed)
    training_x, validation_x = prepare_torch_inputs(
        model_name, training_x, validation_x
    )
    model = build_torch_model(model_name, training_x.shape[1:]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    tx = torch.from_numpy(training_x).to(device)
    ty = torch.from_numpy(training_y).to(device)
    vx = torch.from_numpy(validation_x).to(device)
    vy = torch.from_numpy(validation_y).to(device)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_rng = np.random.default_rng(run_seed * 100 + epoch)
        optimization_loss = 0.0
        for indices in batches(len(ty), batch_size, epoch_rng):
            index = torch.as_tensor(indices, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(tx[index]), ty[index])
            loss.backward()
            optimizer.step()
            optimization_loss += float(loss.item()) * len(indices)
        train_loss, train_prediction, _ = evaluate_torch(model, tx, ty, batch_size)
        validation_loss, validation_prediction, validation_probability = (
            evaluate_torch(model, vx, vy, batch_size)
        )
        train_accuracy = float(np.mean(train_prediction == training_y))
        validation_accuracy = float(
            np.mean(validation_prediction == validation_y)
        )
        history.append(
            {
                "model": model_name,
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
        print(
            f"epoch {epoch:02d}/{epochs} | "
            f"train loss={train_loss:.5f} acc={train_accuracy:.4f} | "
            f"validation loss={validation_loss:.5f} acc={validation_accuracy:.4f}"
        )
    result = classification_metrics(validation_y, validation_prediction)
    result.update(
        {
            "model": model_name,
            "seed": seed,
            "fold": fold + 1,
            "training_cases": len(training_y),
            "validation_cases": len(validation_y),
            "train_loss": history[-1]["train_loss"],
            "train_accuracy": history[-1]["train_accuracy"],
            "validation_loss": history[-1]["validation_loss"],
        }
    )
    return history, result, validation_prediction, validation_probability


def train_csp_fold(
    seed: int,
    fold: int,
    training_x: np.ndarray,
    training_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model = RegularizedCSPLDA().fit(training_x, training_y)
    train_probability = model.predict_probability(training_x)
    validation_probability = model.predict_probability(validation_x)
    train_prediction = (train_probability >= 0.5).astype(np.int64)
    validation_prediction = (validation_probability >= 0.5).astype(np.int64)
    result = classification_metrics(validation_y, validation_prediction)
    result.update(
        {
            "model": "csp_lda",
            "seed": seed,
            "fold": fold + 1,
            "training_cases": len(training_y),
            "validation_cases": len(validation_y),
            "train_loss": binary_cross_entropy(training_y, train_probability),
            "train_accuracy": float(np.mean(train_prediction == training_y)),
            "validation_loss": binary_cross_entropy(
                validation_y, validation_probability
            ),
        }
    )
    return result, validation_prediction, validation_probability


def summarize(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for model_name in MODELS:
        rows = [row for row in seed_rows if row["model"] == model_name]
        summary: dict[str, Any] = {
            "model": model_name,
            "seeds": [row["seed"] for row in rows],
        }
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            values = np.asarray([row[metric] for row in rows], dtype=float)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
            summary[f"{metric}_min"] = float(values.min())
            summary[f"{metric}_max"] = float(values.max())
        output.append(summary)
    return output


def paired_rows(
    y: np.ndarray, predictions: dict[tuple[str, int], np.ndarray], seeds: list[int]
) -> list[dict[str, Any]]:
    rows = []
    for reference, candidate in combinations(MODELS, 2):
        for seed in seeds:
            reference_correct = predictions[(reference, seed)] == y
            candidate_correct = predictions[(candidate, seed)] == y
            reference_only = int(np.sum(reference_correct & ~candidate_correct))
            candidate_only = int(np.sum(candidate_correct & ~reference_correct))
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
            rows.append(
                {
                    "reference_model": reference,
                    "candidate_model": candidate,
                    "seed": seed,
                    "candidate_minus_reference_accuracy": float(
                        candidate_correct.mean() - reference_correct.mean()
                    ),
                    "reference_only_correct": reference_only,
                    "candidate_only_correct": candidate_only,
                    "both_correct": int(np.sum(reference_correct & candidate_correct)),
                    "both_wrong": int(np.sum(~reference_correct & ~candidate_correct)),
                    "mcnemar_exact_p": p_value,
                }
            )
    return rows


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
    path: Path, seed_rows: list[dict[str, Any]], summaries: list[dict[str, Any]]
) -> None:
    labels = [MODEL_LABELS[name] for name in MODELS]
    seeds = sorted({int(row["seed"]) for row in seed_rows})
    seed_values = {
        seed: [
            100.0
            * next(
                row["accuracy"]
                for row in seed_rows
                if row["model"] == model_name and row["seed"] == seed
            )
            for model_name in MODELS
        ]
        for seed in seeds
    }
    means = [100.0 * row["accuracy_mean"] for row in summaries]
    standard_deviations = [100.0 * row["accuracy_std"] for row in summaries]
    x = np.arange(len(MODELS))
    width = min(0.72 / len(seeds), 0.24)
    colors = plt.cm.Set2(np.linspace(0.0, 1.0, max(len(seeds), len(MODELS))))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for offset, (seed, values) in enumerate(seed_values.items()):
        axes[0].bar(
            x + (offset - (len(seeds) - 1) / 2.0) * width,
            values,
            width,
            label=f"seed {seed}",
            color=colors[offset],
        )
    axes[0].set_xticks(x, labels, rotation=18)
    axes[0].set_ylabel("OOF accuracy (%)")
    axes[0].set_title("Accuracy for each seed")
    axes[0].set_ylim(0, 100)
    axes[0].axhline(50.32, color="black", linestyle="--", linewidth=1)
    axes[0].legend()

    axes[1].bar(labels, means, yerr=standard_deviations, capsize=5, color=colors)
    axes[1].set_ylabel("OOF accuracy (%)")
    axes[1].set_title("Mean ± SD across seeds")
    axes[1].set_ylim(0, 100)
    axes[1].axhline(50.32, color="black", linestyle="--", linewidth=1)
    axes[1].tick_params(axis="x", rotation=18)
    figure.suptitle("FingerMovements Phase 1c representation comparison")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def validate_only(x: np.ndarray, y: np.ndarray, device: torch.device) -> None:
    training, validation = stratified_folds(y, FOLDS, 42)[0]
    print("=== Phase 1c validation-only ===")
    for model_name in ("feature_linear", "tiny_eegnet"):
        train_x, validation_x = prepare_torch_inputs(
            model_name, x[training], x[validation]
        )
        model = build_torch_model(model_name, train_x.shape[1:]).to(device)
        with torch.inference_mode():
            output = model(torch.from_numpy(validation_x[:4]).to(device))
        if output.shape != (4, 2) or not torch.isfinite(output).all():
            raise RuntimeError(f"{model_name}: invalid forward output")
        print(
            f"{model_name:>20}: input={validation_x[:4].shape} "
            f"output={tuple(output.shape)}"
        )
    csp = RegularizedCSPLDA().fit(x[training], y[training])
    probability = csp.predict_probability(x[validation])
    if probability.shape != (len(validation),) or not np.isfinite(probability).all():
        raise RuntimeError("csp_lda: invalid probability output")
    print(
        f"{'csp_lda':>20}: input={x[validation].shape} "
        f"output={probability.shape} range=[{probability.min():.4f}, "
        f"{probability.max():.4f}]"
    )
    print("validation-only checks: PASS")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    x, y, source_index = load_training_data(args.data.resolve())
    if args.validate_only:
        validate_only(x, y, device)
        return

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print("=== FingerMovements Phase 1c representation comparison ===")
    print(f"official train: x={x.shape} y={y.shape} | test=LOCKED AND NOT LOADED")
    print(
        f"models={list(MODELS)} | seeds={args.seeds} | folds={args.folds} | "
        f"epochs={args.epochs} | device={device}"
    )
    print(
        f"torch fixed: AdamW lr={LEARNING_RATE:g} wd={WEIGHT_DECAY:g} "
        f"dropout={DROPOUT:g} batch={args.batch_size}"
    )
    print(
        f"CSP fixed: components={CSP_COMPONENTS} "
        f"csp_shrinkage={CSP_SHRINKAGE:g} lda_shrinkage={LDA_SHRINKAGE:g}"
    )
    print("policy: fold-training-only preprocessing; no checkpoint; test not loaded")

    epoch_history: list[dict[str, Any]] = []
    fold_results: list[dict[str, Any]] = []
    seed_results: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    oof_predictions: dict[tuple[str, int], np.ndarray] = {}

    for model_name in MODELS:
        for seed in args.seeds:
            predictions = np.full(len(y), -1, dtype=np.int64)
            probabilities = np.full(len(y), np.nan, dtype=np.float64)
            print(f"\n=== {MODEL_LABELS[model_name]} | seed={seed} ===")
            for fold, (training, validation) in enumerate(
                stratified_folds(y, args.folds, seed)
            ):
                print(
                    f"\n--- fold {fold + 1}/{args.folds} | "
                    f"train={len(training)} validation={len(validation)} ---"
                )
                if model_name == "csp_lda":
                    result, predicted, probability = train_csp_fold(
                        seed,
                        fold,
                        x[training],
                        y[training],
                        x[validation],
                        y[validation],
                    )
                else:
                    history, result, predicted, probability = train_torch_fold(
                        model_name,
                        seed,
                        fold,
                        x[training],
                        y[training],
                        x[validation],
                        y[validation],
                        args.epochs,
                        args.batch_size,
                        device,
                    )
                    epoch_history.extend(history)
                fold_results.append(result)
                predictions[validation] = predicted
                probabilities[validation] = probability
                print(
                    f"fold result | accuracy={result['accuracy']:.4f} "
                    f"balanced={result['balanced_accuracy']:.4f} "
                    f"macro_f1={result['macro_f1']:.4f}"
                )
                for local_index, dataset_index in enumerate(validation):
                    prediction_rows.append(
                        {
                            "model": model_name,
                            "seed": seed,
                            "fold": fold + 1,
                            "source_index": int(source_index[dataset_index]),
                            "true_label": int(y[dataset_index]),
                            "predicted_label": int(predicted[local_index]),
                            "probability_right": float(probability[local_index]),
                            "correct": int(predicted[local_index] == y[dataset_index]),
                        }
                    )
            if np.any(predictions < 0) or not np.isfinite(probabilities).all():
                raise RuntimeError("Incomplete out-of-fold prediction coverage")
            metrics = classification_metrics(y, predictions)
            metrics.update({"model": model_name, "seed": seed, "folds": args.folds})
            seed_results.append(metrics)
            oof_predictions[(model_name, seed)] = predictions
            print(
                f"seed result | OOF accuracy={metrics['accuracy']:.4f} "
                f"balanced={metrics['balanced_accuracy']:.4f} "
                f"macro_f1={metrics['macro_f1']:.4f}"
            )

    summaries = summarize(seed_results)
    comparisons = paired_rows(y, oof_predictions, args.seeds)
    elapsed = time.monotonic() - started
    print("\n=== Cross-seed summary ===")
    for row in summaries:
        print(
            f"{MODEL_LABELS[row['model']]:>24} | "
            f"accuracy={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f} | "
            f"balanced={row['balanced_accuracy_mean']:.4f}"
            f"±{row['balanced_accuracy_std']:.4f} | "
            f"worst={row['accuracy_min']:.4f}"
        )

    report = {
        "experiment": "phase1c_finger_movements_representation_comparison",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data": {
            "path": str(args.data.resolve()),
            "shape": list(x.shape),
            "class_counts": CLASS_COUNTS,
            "test_policy": "locked and not loaded",
        },
        "protocol": {
            "models": list(MODELS),
            "seeds": args.seeds,
            "folds": args.folds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "csp_components": CSP_COMPONENTS,
            "csp_shrinkage": CSP_SHRINKAGE,
            "lda_shrinkage": LDA_SHRINKAGE,
            "normalization": "fit on each fold training subset only",
            "checkpoint": "none; fixed final epoch for Torch models",
            "augmentation": "none",
            "device": str(device),
        },
        "seed_results": seed_results,
        "summary": summaries,
        "paired_comparisons": comparisons,
        "elapsed_seconds": elapsed,
    }
    (output_dir / "phase1c_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_csv(output_dir / "phase1c_epoch_history.csv", epoch_history)
    write_csv(output_dir / "phase1c_fold_results.csv", fold_results)
    write_csv(output_dir / "phase1c_seed_results.csv", seed_results)
    write_csv(output_dir / "phase1c_predictions.csv", prediction_rows)
    write_csv(output_dir / "phase1c_paired_comparisons.csv", comparisons)
    save_figure(output_dir / "phase1c_comparison.png", seed_results, summaries)
    print(f"\nelapsed: {elapsed / 60.0:.1f} minutes")
    print(f"metrics: {output_dir / 'phase1c_metrics.json'}")
    print(f"predictions: {output_dir / 'phase1c_predictions.csv'}")
    print(f"figure: {output_dir / 'phase1c_comparison.png'}")
    print("test: LOCKED AND NOT LOADED")


if __name__ == "__main__":
    main()
