"""Phase 1b: repeated stratified CV for four FingerMovements baselines.

Only train.npz is loaded. Normalization is fitted independently inside each
fold, and the official test set is never used. This script is self-contained:
it does not import shared or retired Indy code.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


MODELS = (
    "feature_linear",
    "tiny_mlp",
    "tiny_eegnet",
    "tiny_multiscale_cnn",
)
LABELS = {
    "feature_linear": "Feature + Linear",
    "tiny_mlp": "Tiny MLP",
    "tiny_eegnet": "Tiny EEGNet",
    "tiny_multiscale_cnn": "Tiny Multi-scale CNN",
}

# Phase-1b constants: model families are compared without a hyperparameter sweep.
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.25
EPOCHS = 20
BATCH_SIZE = 32
SEEDS = (42, 43, 44)
FOLDS = 5

CASES = 316
CHANNELS = 28
TIMEPOINTS = 50
CLASS_COUNTS = {0: 159, 1: 157}


class FeatureLinear(nn.Module):
    def __init__(self, feature_count: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Dropout(DROPOUT), nn.Linear(feature_count, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(CHANNELS * 10, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(16, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.avg_pool1d(x, kernel_size=5, stride=5)
        return self.layers(x.flatten(1))


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
            nn.Conv2d(
                16, 16, (1, 7), padding=(0, 3), groups=16, bias=False
            ),
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


class TinyMultiScaleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(CHANNELS, 8, kernel, padding=kernel // 2, bias=False),
                    nn.BatchNorm1d(8),
                    nn.ReLU(),
                )
                for kernel in (3, 7, 15)
            ]
        )
        self.mixer = nn.Sequential(
            nn.Conv1d(24, 16, 1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Conv1d(16, 16, 3, padding=1, groups=16, bias=False),
            nn.Conv1d(16, 16, 1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.output = nn.Sequential(
            nn.Flatten(), nn.Dropout(DROPOUT), nn.Linear(16, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.output(self.mixer(x))


def arguments() -> argparse.Namespace:
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
        default=root / "results/finger_movements/phase1b_baseline_comparison",
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
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


def device_from(name: str) -> torch.device:
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_train(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.name.lower() == "test.npz":
        raise ValueError("Phase 1b refuses to load test.npz")
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run the FingerMovements converter first"
        )
    with np.load(path, allow_pickle=False) as data:
        x = data["x"].astype(np.float32, copy=True)
        y = data["y"].astype(np.int64, copy=True)
    if x.shape != (CASES, CHANNELS, TIMEPOINTS) or y.shape != (CASES,):
        raise ValueError(f"Unexpected data shapes: x={x.shape}, y={y.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Input contains non-finite values")
    values, counts = np.unique(y, return_counts=True)
    if dict(zip(values.tolist(), counts.tolist(), strict=True)) != CLASS_COUNTS:
        raise ValueError("Unexpected class labels or counts")
    return x, y


def stratified_folds(
    y: np.ndarray, fold_count: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    pieces: dict[int, list[np.ndarray]] = {}
    for label in sorted(np.unique(y).tolist()):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        pieces[label] = list(np.array_split(indices, fold_count))

    result = []
    all_indices = np.arange(len(y))
    for fold in range(fold_count):
        validation = np.concatenate([pieces[label][fold] for label in pieces])
        training = np.setdiff1d(all_indices, validation, assume_unique=True)
        fold_rng = np.random.default_rng(seed * 10_000 + fold)
        fold_rng.shuffle(training)
        fold_rng.shuffle(validation)
        result.append((training, validation))
    return result


def channel_normalize(
    training: np.ndarray, validation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = training.mean((0, 2), keepdims=True, dtype=np.float64)
    std = training.std((0, 2), keepdims=True, dtype=np.float64)
    std = np.maximum(std, 1e-6)
    return (
        ((training - mean) / std).astype(np.float32),
        ((validation - mean) / std).astype(np.float32),
    )


def features(x: np.ndarray) -> np.ndarray:
    statistics = [x.mean(-1), x.std(-1), np.square(x).mean(-1)]
    spectrum = np.square(np.abs(np.fft.rfft(x, axis=-1))) / x.shape[-1]
    frequencies = np.fft.rfftfreq(x.shape[-1], d=0.01)
    for low, high in ((1, 4), (4, 8), (8, 13), (13, 30)):
        mask = (frequencies >= low) & (frequencies < high)
        statistics.append(spectrum[..., mask].mean(-1))
    return np.concatenate(statistics, axis=1).astype(np.float32)


def prepare_inputs(
    model_name: str, training: np.ndarray, validation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    training, validation = channel_normalize(training, validation)
    if model_name == "feature_linear":
        training, validation = features(training), features(validation)
        mean = training.mean(0, keepdims=True, dtype=np.float64)
        std = np.maximum(training.std(0, keepdims=True, dtype=np.float64), 1e-6)
        training = ((training - mean) / std).astype(np.float32)
        validation = ((validation - mean) / std).astype(np.float32)
    return training, validation


def model_for(name: str, input_shape: tuple[int, ...]) -> nn.Module:
    if name == "feature_linear":
        return FeatureLinear(input_shape[0])
    if name == "tiny_mlp":
        return TinyMLP()
    if name == "tiny_eegnet":
        return TinyEEGNet()
    if name == "tiny_multiscale_cnn":
        return TinyMultiScaleCNN()
    raise ValueError(name)


def parameters(model: nn.Module) -> int:
    return sum(value.numel() for value in model.parameters() if value.requires_grad)


def batches(
    size: int, batch_size: int, rng: np.random.Generator | None = None
) -> list[np.ndarray]:
    order = np.arange(size)
    if rng is not None:
        rng.shuffle(order)
    return [order[start : start + batch_size] for start in range(0, size, batch_size)]


def evaluate(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int
) -> tuple[float, np.ndarray]:
    model.eval()
    loss_sum = 0.0
    predicted = []
    with torch.inference_mode():
        for indices in batches(len(y), batch_size):
            index = torch.as_tensor(indices, device=y.device)
            logits = model(x[index])
            loss_sum += float(
                F.cross_entropy(logits, y[index], reduction="sum").item()
            )
            predicted.append(logits.argmax(1).cpu().numpy())
    return loss_sum / len(y), np.concatenate(predicted)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    confusion = np.zeros((2, 2), dtype=np.int64)
    for truth, guess in zip(actual, predicted, strict=True):
        confusion[int(truth), int(guess)] += 1
    recalls, f1s = [], []
    for label in (0, 1):
        true_positive = float(confusion[label, label])
        false_negative = float(confusion[label].sum() - true_positive)
        false_positive = float(confusion[:, label].sum() - true_positive)
        recall = true_positive / max(true_positive + false_negative, 1.0)
        precision = true_positive / max(true_positive + false_positive, 1.0)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        recalls.append(recall)
        f1s.append(f1)
    return {
        "accuracy": float(np.mean(actual == predicted)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "confusion_matrix": confusion.tolist(),
    }


def train_fold(
    name: str,
    seed: int,
    fold: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray]:
    run_seed = seed * 1_000 + fold
    seed_everything(run_seed)
    train_x, validation_x = prepare_inputs(name, train_x, validation_x)
    model = model_for(name, train_x.shape[1:]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    tx = torch.from_numpy(train_x).to(device)
    ty = torch.from_numpy(train_y).to(device)
    vx = torch.from_numpy(validation_x).to(device)
    vy = torch.from_numpy(validation_y).to(device)
    history: list[dict[str, Any]] = []
    validation_predictions = np.empty(0, dtype=np.int64)

    for epoch in range(1, epochs + 1):
        model.train()
        rng = np.random.default_rng(run_seed * 100 + epoch)
        optimization_loss = 0.0
        for indices in batches(len(ty), batch_size, rng):
            index = torch.as_tensor(indices, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(tx[index]), ty[index])
            loss.backward()
            optimizer.step()
            optimization_loss += float(loss.item()) * len(indices)

        train_loss, train_predictions = evaluate(model, tx, ty, batch_size)
        validation_loss, validation_predictions = evaluate(model, vx, vy, batch_size)
        train_accuracy = float(np.mean(train_predictions == train_y))
        validation_accuracy = float(np.mean(validation_predictions == validation_y))
        row = {
            "model": name,
            "seed": seed,
            "fold": fold + 1,
            "epoch": epoch,
            "optimization_loss": optimization_loss / len(ty),
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{epochs} | "
            f"train loss={train_loss:.5f} acc={train_accuracy:.4f} | "
            f"validation loss={validation_loss:.5f} acc={validation_accuracy:.4f}"
        )

    result = metrics(validation_y, validation_predictions)
    result.update(
        {
            "model": name,
            "seed": seed,
            "fold": fold + 1,
            "training_cases": len(train_y),
            "validation_cases": len(validation_y),
            "parameters": parameters(model),
            "final_train_loss": history[-1]["train_loss"],
            "final_train_accuracy": history[-1]["train_accuracy"],
            "final_validation_loss": history[-1]["validation_loss"],
        }
    )
    return history, result, validation_predictions


def summarize(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for name in MODELS:
        rows = [row for row in seed_rows if row["model"] == name]
        if not rows:
            continue
        summary: dict[str, Any] = {
            "model": name,
            "parameters": rows[0]["parameters"],
            "seeds": [row["seed"] for row in rows],
        }
        for key in ("accuracy", "balanced_accuracy", "macro_f1"):
            values = np.asarray([row[key] for row in rows], dtype=float)
            summary[f"{key}_mean"] = float(values.mean())
            summary[f"{key}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
            summary[f"{key}_min"] = float(values.min())
            summary[f"{key}_max"] = float(values.max())
        output.append(summary)
    return output


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


def plot_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    names = [LABELS[row["model"]] for row in rows]
    accuracy = [100 * row["accuracy_mean"] for row in rows]
    deviations = [100 * row["accuracy_std"] for row in rows]
    counts = [row["parameters"] for row in rows]
    colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756")[: len(rows)]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(names, accuracy, yerr=deviations, capsize=5, color=colors)
    axes[0].axhline(50, color="black", linestyle="--", linewidth=1)
    axes[0].set(title="Mean ± SD across seeds", ylabel="5-fold OOF accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[1].bar(names, counts, color=colors)
    axes[1].set_yscale("log")
    axes[1].set(title="Model size", ylabel="Trainable parameters (log scale)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
    for index, count in enumerate(counts):
        axes[1].text(index, count * 1.08, f"{count:,}", ha="center")
    figure.suptitle("FingerMovements Phase 1b baseline comparison")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def validate_only(
    names: list[str], x: np.ndarray, y: np.ndarray, device: torch.device
) -> None:
    training, validation = stratified_folds(y, FOLDS, 42)[0]
    print("=== Phase 1b validation-only ===")
    for name in names:
        train_x, validation_x = prepare_inputs(
            name, x[training], x[validation]
        )
        model = model_for(name, train_x.shape[1:]).to(device)
        with torch.inference_mode():
            output = model(torch.from_numpy(validation_x[:4]).to(device))
        if output.shape != (4, 2) or not torch.isfinite(output).all():
            raise RuntimeError(f"{name}: bad forward output")
        print(
            f"{name:>20}: input={validation_x[:4].shape} "
            f"output={tuple(output.shape)} parameters={parameters(model):,}"
        )
    print("validation-only checks: PASS")


def main() -> None:
    args = arguments()
    device = device_from(args.device)
    x, y = load_train(args.data.resolve())
    if args.validate_only:
        validate_only(args.models, x, y, device)
        return

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print("=== FingerMovements Phase 1b baseline comparison ===")
    print(f"official train: x={x.shape} y={y.shape} | test=LOCKED AND NOT LOADED")
    print(
        f"models={args.models} | seeds={args.seeds} | folds={args.folds} | "
        f"epochs={args.epochs} | device={device}"
    )
    print(
        f"fixed: AdamW lr={LR:g} wd={WEIGHT_DECAY:g} dropout={DROPOUT:g} "
        f"batch={args.batch_size}"
    )
    print("policy: fold-train normalization; fixed final epoch; no checkpoint")

    histories: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for name in args.models:
        for seed in args.seeds:
            oof = np.full(len(y), -1, dtype=np.int64)
            parameter_count = 0
            print(f"\n=== {LABELS[name]} | seed={seed} ===")
            for fold, (training, validation) in enumerate(
                stratified_folds(y, args.folds, seed)
            ):
                print(
                    f"\n--- fold {fold + 1}/{args.folds} | "
                    f"train={len(training)} validation={len(validation)} ---"
                )
                history, result, predicted = train_fold(
                    name,
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
                histories.extend(history)
                fold_rows.append(result)
                oof[validation] = predicted
                parameter_count = result["parameters"]
                print(
                    f"fold result | accuracy={result['accuracy']:.4f} "
                    f"balanced={result['balanced_accuracy']:.4f} "
                    f"macro_f1={result['macro_f1']:.4f}"
                )
            if np.any(oof < 0):
                raise RuntimeError("Incomplete out-of-fold coverage")
            row = metrics(y, oof)
            row.update(
                {
                    "model": name,
                    "seed": seed,
                    "folds": args.folds,
                    "parameters": parameter_count,
                }
            )
            seed_rows.append(row)
            print(
                f"seed result | OOF accuracy={row['accuracy']:.4f} "
                f"balanced={row['balanced_accuracy']:.4f} "
                f"macro_f1={row['macro_f1']:.4f}"
            )

    summaries = summarize(seed_rows)
    print("\n=== Cross-seed summary ===")
    for row in summaries:
        print(
            f"{LABELS[row['model']]:>20} | "
            f"accuracy={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f} | "
            f"balanced={row['balanced_accuracy_mean']:.4f}"
            f"±{row['balanced_accuracy_std']:.4f} | "
            f"parameters={row['parameters']:,}"
        )

    report = {
        "experiment": "phase1b_finger_movements_baseline_comparison",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data": {
            "path": str(args.data.resolve()),
            "shape": list(x.shape),
            "class_counts": CLASS_COUNTS,
            "test_policy": "locked and not loaded",
        },
        "protocol": {
            "models": args.models,
            "seeds": args.seeds,
            "folds": args.folds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "AdamW",
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "normalization": "fit on each fold training subset only",
            "checkpoint": "none; fixed final epoch",
            "augmentation": "none",
            "device": str(device),
        },
        "seed_results": seed_rows,
        "summary": summaries,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output_dir / "phase1b_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_csv(output_dir / "phase1b_epoch_history.csv", histories)
    write_csv(output_dir / "phase1b_fold_results.csv", fold_rows)
    write_csv(output_dir / "phase1b_seed_results.csv", seed_rows)
    plot_summary(output_dir / "phase1b_comparison.png", summaries)
    print(f"\nmetrics: {output_dir / 'phase1b_metrics.json'}")
    print(f"figure:  {output_dir / 'phase1b_comparison.png'}")
    print("test: LOCKED AND NOT LOADED")


if __name__ == "__main__":
    main()
