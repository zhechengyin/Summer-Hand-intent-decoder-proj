"""Train the frozen FingerMovements Feature + Linear pipeline on all train cases.

This entry point never loads the official test split. It produces the single
checkpoint later used for locked-test evaluation and firmware export.
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

if __package__:
    from .model import FeatureLinear, fit_preprocessing, transform
else:
    from model import FeatureLinear, fit_preprocessing, transform


SEED = 42
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
CASES = 316
CLASS_COUNTS = {0: 159, 1: 157}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "data/processed/finger_movements/train.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            root
            / "models/finger_movements/feature_linear/checkpoints"
            / "feature_linear_seed42_epoch50.pt"
        ),
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


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_training_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.name.lower() == "test.npz":
        raise ValueError("Final training refuses to load test.npz")
    if not path.is_file():
        raise FileNotFoundError(f"Training data not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        x = data["x"].astype(np.float32, copy=True)
        y = data["y"].astype(np.int64, copy=True)
        source_index = data["source_index"].astype(np.int64, copy=True)
    if x.shape != (CASES, 28, 50) or y.shape != (CASES,):
        raise ValueError(f"Unexpected data shapes: x={x.shape}, y={y.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Input contains non-finite values")
    values, counts = np.unique(y, return_counts=True)
    observed = dict(zip(values.tolist(), counts.tolist(), strict=True))
    if observed != CLASS_COUNTS:
        raise ValueError(f"Unexpected labels or counts: {observed}")
    if not np.array_equal(source_index, np.arange(CASES)):
        raise ValueError("source_index must preserve the official train order")
    return x, y


def batch_indices(size: int, epoch: int) -> list[np.ndarray]:
    order = np.arange(size)
    np.random.default_rng(SEED * 100 + epoch).shuffle(order)
    return [
        order[start : start + BATCH_SIZE]
        for start in range(0, size, BATCH_SIZE)
    ]


def evaluate(
    model: FeatureLinear, x: torch.Tensor, y: torch.Tensor
) -> tuple[float, float]:
    model.eval()
    with torch.inference_mode():
        logits = model(x)
        loss = float(F.cross_entropy(logits, y).item())
        accuracy = float((logits.argmax(dim=1) == y).float().mean().item())
    return loss, accuracy


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    raw_x, labels = load_training_data(args.data.resolve())
    preprocessing = fit_preprocessing(raw_x)
    features = transform(raw_x, preprocessing)
    model = FeatureLinear().to(device)
    if args.validate_only:
        with torch.inference_mode():
            output = model(torch.from_numpy(features[:4]).to(device))
        if output.shape != (4, 2) or not torch.isfinite(output).all():
            raise RuntimeError("Active model validation failed")
        print("=== Feature + Linear final-training validation-only ===")
        print(f"raw={raw_x.shape} features={features.shape} output={tuple(output.shape)}")
        print("test: LOCKED AND NOT LOADED")
        return

    set_seed()
    model = FeatureLinear().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    x = torch.from_numpy(features).to(device)
    y = torch.from_numpy(labels).to(device)
    print("=== FingerMovements final Feature + Linear training ===")
    print(
        f"cases={len(y)} | epochs={EPOCHS} | seed={SEED} | device={device} | "
        "test=LOCKED AND NOT LOADED"
    )

    final_loss = float("nan")
    final_accuracy = float("nan")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimization_loss = 0.0
        for indices in batch_indices(len(y), epoch):
            index = torch.as_tensor(indices, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[index]), y[index])
            loss.backward()
            optimizer.step()
            optimization_loss += float(loss.item()) * len(indices)
        final_loss, final_accuracy = evaluate(model, x, y)
        print(
            f"epoch {epoch:02d}/{EPOCHS} | "
            f"opt={optimization_loss / len(y):.5f} | "
            f"train loss={final_loss:.5f} accuracy={final_accuracy:.4f}"
        )

    cpu_state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }
    checkpoint = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": "FingerMovements left-versus-right classification",
        "input_contract": {
            "shape": ["cases", 28, 50],
            "dtype": "float32",
            "sampling_rate_hz": 100,
            "labels": {0: "left", 1: "right"},
        },
        "model": "feature_linear",
        "model_state_dict": cpu_state,
        "preprocessing": {
            name: torch.from_numpy(value.copy())
            for name, value in preprocessing.items()
        },
        "training": {
            "cases": CASES,
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dropout": 0.25,
            "final_training_loss": final_loss,
            "final_training_accuracy": final_accuracy,
            "test_loaded": False,
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"checkpoint: {output}")
    print("test: LOCKED AND NOT LOADED")


if __name__ == "__main__":
    main()
