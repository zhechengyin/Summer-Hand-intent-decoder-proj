#!/usr/bin/env python3
"""Phase 12: held-out external-memory query representation ablation.

The experiment keeps the promoted Midsize model frozen and compares four
causal query representations on Phase-7 fold 1 of the first Indy benchmark
session.  Memory entries, PCA fits, and residuals are train-reach-only;
hyperparameters are selected on validation reaches; test reaches are opened
once for the final comparison.

The generated ``*.memlib`` files use the self-describing
``phase12_pc_memlib_v1`` NumPy archive schema.  They are PC experiment
artifacts, not drop-in firmware-v1 BCIMEM binaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = REPOSITORY_ROOT / "indy_loco"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from indy_loco.history.experiments.phase7.phase7_ann_vs_snn_fivefold import (  # noqa: E402
    SESSION_BY_NAME,
    aggregate_40ms,
    binned_reach_bounds,
    eligible_reaches,
    load_session,
    make_fold_indices,
    split_fold,
)
from indy_loco.models.midsize.model import MidsizeTCNGRU  # noqa: E402

PHASE_NAME: Final = "phase12_external_memory_representation_ablation"
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULT_DIR = EXPERIMENT_DIR / "results"
MODEL_ROOT = PROJECT_ROOT / "models" / "midsize"
LARGE_ROOT = PROJECT_ROOT / "models" / "large"

DEFAULT_SESSION: Final = "indy_20160622_01"
CHECKPOINT_NAME: Final = "checkpoint.pt"
WINDOW_BINS: Final = 50
FEATURES: Final = 192
TCN_WIDTH: Final = 64
GRU_WIDTH: Final = 64
REPRESENTATION_DIMS: Final = 32
CONTEXT_SOURCE_DIMS: Final = FEATURES * 3
CONTEXT_DIMS: Final = 32
KEY_DIMS: Final = REPRESENTATION_DIMS + CONTEXT_DIMS
FAST_CONTEXT_ALPHA: Final = 0.02
SLOW_CONTEXT_ALPHA: Final = 0.005
CONTEXT_WEIGHT: Final = 0.5
MAX_NEIGHBOURS: Final = 128
NEIGHBOURS: Final = (8, 16, 32, 64, 128)
TEMPERATURES: Final = (0.02, 0.05, 0.10, 0.20)
BLENDS: Final = (0.25, 0.50, 0.75, 1.00)
BOOTSTRAP_REPETITIONS: Final = 1_000
PCA_SEED: Final = 1212
EXPECTED_PARAMETER_COUNT: Final = 86_978
MCU_ENTRY_STRIDE: Final = 96
MCU_CLUSTER_COUNT: Final = 256
MCU_PROBES: Final = 16

REPRESENTATIONS: Final = (
    "encoder_49",
    "gru_hidden_49",
    "encoder_gru_49",
    "encoder_50step_mean",
)


@dataclass(frozen=True)
class SplitData:
    representation: dict[str, np.ndarray]
    context: np.ndarray
    prediction: np.ndarray
    target: np.ndarray
    residual: np.ndarray
    reach_id: np.ndarray
    bin_id: np.ndarray


@dataclass(frozen=True)
class PCAFit:
    centre: np.ndarray
    basis: np.ndarray
    scale: np.ndarray
    explained_variance_ratio: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session",
        choices=sorted(SESSION_BY_NAME),
        default=DEFAULT_SESSION,
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--protocol-check-only",
        action="store_true",
        help="Validate checkpoint, split, representation shapes, and leakage guards.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("Cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as destination:
            np.savez_compressed(destination, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def select_device(requested: str) -> Any:
    import torch

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(path: Path, device: Any, session_name: str) -> tuple[Any, dict[str, Any]]:
    import torch

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required = {
        "session",
        "fold",
        "model_state",
        "selected_channel_indices",
        "feature_mean",
        "feature_std",
        "target_mean",
        "target_std",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint is missing keys: {missing}")
    if checkpoint["session"] != session_name or int(checkpoint["fold"]) != 1:
        raise ValueError("Phase 12 requires the registered fold-1 session checkpoint")
    model = MidsizeTCNGRU().to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("Unexpected model parameter count")
    return model, checkpoint


def causal_ewma(values: np.ndarray, alpha: float) -> np.ndarray:
    output = np.asarray(values, dtype=np.float32).copy()
    for index in range(1, output.shape[1]):
        output[:, index] = alpha * values[:, index] + (1.0 - alpha) * output[:, index - 1]
    return output


def normalized_reach_features(
    counts: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(counts, dtype=np.float32)
    features = np.concatenate((raw, causal_ewma(raw, 0.1)), axis=0)
    return ((features - mean) / std).astype(np.float32)


def long_context(normalized: np.ndarray) -> np.ndarray:
    fast = causal_ewma(normalized, FAST_CONTEXT_ALPHA)
    slow = causal_ewma(normalized, SLOW_CONTEXT_ALPHA)
    return np.concatenate((fast, slow, normalized - fast), axis=0).T.astype(np.float32)


def rolling_windows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.pad(values, ((0, 0), (WINDOW_BINS - 1, 0)))
    view = np.lib.stride_tricks.sliding_window_view(
        padded, window_shape=WINDOW_BINS, axis=1
    )
    windows = np.ascontiguousarray(view.transpose(1, 0, 2), dtype=np.float32)
    valid_lengths = np.minimum(np.arange(1, values.shape[1] + 1), WINDOW_BINS)
    return windows, valid_lengths.astype(np.int64)


def model_intermediates(model: Any, inputs: Any) -> tuple[Any, Any, Any]:
    encoded = model.spatial(inputs)
    for convolution, padding in zip(model.convolutions, model.padding, strict=True):
        encoded = model.activation(convolution(encoded)[:, :, :-padding] + encoded)
    states, _ = model.gru(encoded.transpose(1, 2))
    prediction = model.head(states)
    return encoded.transpose(1, 2), states, prediction


def extract_split(
    model: Any,
    checkpoint: dict[str, Any],
    counts: np.ndarray,
    velocity: np.ndarray,
    bounds: np.ndarray,
    reaches: np.ndarray,
    device: Any,
    batch_size: int,
) -> SplitData:
    import torch

    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32).reshape(FEATURES, 1)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32).reshape(FEATURES, 1)
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32).reshape(2)
    target_std = np.asarray(checkpoint["target_std"], dtype=np.float32).reshape(2)
    representation: dict[str, list[np.ndarray]] = {name: [] for name in REPRESENTATIONS}
    contexts: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    reach_ids: list[np.ndarray] = []
    bin_ids: list[np.ndarray] = []

    for reach in sorted((int(value) for value in reaches), key=lambda i: bounds[i, 0]):
        start, stop = (int(value) for value in bounds[reach])
        normalized = normalized_reach_features(counts[:, start:stop], mean, std)
        windows, valid_lengths = rolling_windows(normalized)
        contexts.append(long_context(normalized))
        targets.append(velocity[start:stop])
        reach_ids.append(np.full(stop - start, reach, dtype=np.int32))
        bin_ids.append(np.arange(start, stop, dtype=np.int64))
        for left in range(0, len(windows), batch_size):
            right = min(left + batch_size, len(windows))
            tensor = torch.from_numpy(windows[left:right]).to(device)
            with torch.inference_mode():
                encoded, states, output = model_intermediates(model, tensor)
            encoded_np = encoded.cpu().numpy().astype(np.float32)
            states_np = states.cpu().numpy().astype(np.float32)
            representation["encoder_49"].append(encoded_np[:, -1])
            representation["gru_hidden_49"].append(states_np[:, -1])
            representation["encoder_gru_49"].append(
                np.concatenate((encoded_np[:, -1], states_np[:, -1]), axis=1)
            )
            pooled = np.empty((right - left, TCN_WIDTH), dtype=np.float32)
            for row, length in enumerate(valid_lengths[left:right]):
                pooled[row] = encoded_np[row, -int(length) :].mean(axis=0)
            representation["encoder_50step_mean"].append(pooled)
            normalized_prediction = output[:, -1].cpu().numpy().astype(np.float32)
            predictions.append(normalized_prediction * target_std + target_mean)

    final_prediction = np.concatenate(predictions).astype(np.float32)
    final_target = np.concatenate(targets).astype(np.float32)
    return SplitData(
        representation={
            name: np.concatenate(parts).astype(np.float32)
            for name, parts in representation.items()
        },
        context=np.concatenate(contexts).astype(np.float32),
        prediction=final_prediction,
        target=final_target,
        residual=(final_target - final_prediction).astype(np.float32),
        reach_id=np.concatenate(reach_ids),
        bin_id=np.concatenate(bin_ids),
    )


def fit_pca(values: np.ndarray, dimensions: int, seed: int) -> PCAFit:
    import torch

    source = np.asarray(values, dtype=np.float32)
    centre = source.mean(axis=0).astype(np.float32)
    tensor = torch.from_numpy(source - centre)
    torch.manual_seed(seed)
    _, singular, basis = torch.pca_lowrank(tensor, q=dimensions, center=False, niter=4)
    transformed = tensor.matmul(basis).numpy()
    scale = (transformed.std(axis=0, ddof=0) + 1e-6).astype(np.float32)
    variance = singular.numpy() ** 2 / max(len(source) - 1, 1)
    total_variance = np.var(source, axis=0, ddof=1).sum()
    explained = (variance / max(float(total_variance), 1e-12)).astype(np.float32)
    return PCAFit(
        centre=centre,
        basis=basis.numpy().astype(np.float32),
        scale=scale,
        explained_variance_ratio=explained,
    )


def transform_pca(values: np.ndarray, fit: PCAFit) -> np.ndarray:
    return ((values - fit.centre) @ fit.basis / fit.scale).astype(np.float32)


def row_normalize(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return (values / np.maximum(norm, 1e-8)).astype(np.float32)


def make_keys(representation: np.ndarray, context: np.ndarray) -> np.ndarray:
    rep = row_normalize(representation)
    ctx = row_normalize(context)
    return row_normalize(np.concatenate((rep, CONTEXT_WEIGHT * ctx), axis=1))


def metric_values(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = np.sum((target - prediction) ** 2, axis=0)
    total = np.sum((target - target.mean(axis=0)) ** 2, axis=0)
    score = 1.0 - residual / np.maximum(total, 1e-12)
    return {
        "r2_x": float(score[0]),
        "r2_y": float(score[1]),
        "r2_mean": float(score.mean()),
        "mse": float(np.mean((target - prediction) ** 2)),
    }


def residual_consistency(
    true_residual: np.ndarray,
    estimate: np.ndarray,
    neighbours: np.ndarray,
) -> dict[str, float]:
    baseline = np.sum(true_residual**2, axis=0)
    error = np.sum((true_residual - estimate) ** 2, axis=0)
    score = 1.0 - error / np.maximum(baseline, 1e-12)
    dispersion = np.mean(np.var(neighbours, axis=1))
    return {
        "residual_r2_x": float(score[0]),
        "residual_r2_y": float(score[1]),
        "residual_r2_mean": float(score.mean()),
        "neighbour_residual_dispersion": float(dispersion),
    }


def query_tree(tree: Any, keys: np.ndarray, workers: int) -> tuple[np.ndarray, np.ndarray, float]:
    started = time.perf_counter()
    distance, index = tree.query(keys, k=MAX_NEIGHBOURS, workers=workers)
    elapsed = time.perf_counter() - started
    return distance.astype(np.float32), index.astype(np.int64), elapsed


def weighted_residual(
    distance: np.ndarray,
    index: np.ndarray,
    bank_residual: np.ndarray,
    neighbours: int,
    temperature: float,
) -> np.ndarray:
    selected_distance = distance[:, :neighbours]
    selected = bank_residual[index[:, :neighbours]]
    logits = -(selected_distance**2) / max(temperature, 1e-8)
    logits -= logits.max(axis=1, keepdims=True)
    weight = np.exp(logits).astype(np.float32)
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-12)
    return np.sum(selected * weight[:, :, None], axis=1).astype(np.float32)


def tune_validation(
    validation: SplitData,
    distance: np.ndarray,
    index: np.ndarray,
    bank_residual: np.ndarray,
) -> tuple[dict[str, float | int], np.ndarray]:
    best: dict[str, float | int] | None = None
    best_estimate: np.ndarray | None = None
    for neighbours in NEIGHBOURS:
        for temperature in TEMPERATURES:
            estimate = weighted_residual(
                distance, index, bank_residual, neighbours, temperature
            )
            for blend in BLENDS:
                metrics = metric_values(
                    validation.target, validation.prediction + blend * estimate
                )
                candidate: dict[str, float | int] = {
                    "neighbours": neighbours,
                    "temperature": temperature,
                    "blend": blend,
                    "validation_r2_mean": metrics["r2_mean"],
                }
                if best is None or candidate["validation_r2_mean"] > best["validation_r2_mean"]:
                    best = candidate
                    best_estimate = estimate
    assert best is not None and best_estimate is not None
    return best, best_estimate


def bootstrap_r2_delta(
    target: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    reach_id: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    unique = np.unique(reach_id)
    by_reach = {value: np.flatnonzero(reach_id == value) for value in unique}
    deltas = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([by_reach[value] for value in sampled])
        deltas[repetition] = (
            metric_values(target[rows], first[rows])["r2_mean"]
            - metric_values(target[rows], second[rows])["r2_mean"]
        )
    return {
        "mean": float(deltas.mean()),
        "ci95_low": float(np.percentile(deltas, 2.5)),
        "ci95_high": float(np.percentile(deltas, 97.5)),
        "probability_positive": float(np.mean(deltas > 0)),
        "repetitions": repetitions,
        "unit": "reach",
    }


def projected_mcu_cost(entry_count: int, source_dims: int) -> dict[str, int | float]:
    candidates = math.ceil(entry_count * MCU_PROBES / MCU_CLUSTER_COUNT)
    centroid_macs = MCU_CLUSTER_COUNT * KEY_DIMS
    candidate_macs = candidates * KEY_DIMS
    projection_bytes = (
        source_dims * REPRESENTATION_DIMS
        + source_dims
        + CONTEXT_SOURCE_DIMS * CONTEXT_DIMS
        + CONTEXT_SOURCE_DIMS
    ) * 4
    total_bytes = (
        256
        + projection_bytes
        + MCU_CLUSTER_COUNT * KEY_DIMS
        + (MCU_CLUSTER_COUNT + 1) * 4
        + entry_count * MCU_ENTRY_STRIDE
    )
    return {
        "entry_count": entry_count,
        "entry_stride_bytes": MCU_ENTRY_STRIDE,
        "projected_bank_bytes": total_bytes,
        "projected_bank_mib": total_bytes / (1024 * 1024),
        "ivf_clusters": MCU_CLUSTER_COUNT,
        "ivf_probes": MCU_PROBES,
        "expected_candidate_entries_uniform": candidates,
        "estimated_int8_mac_per_query": centroid_macs + candidate_macs,
        "estimated_key_bytes_read_per_query": centroid_macs + candidate_macs,
    }


def export_memlib(
    path: Path,
    name: str,
    checkpoint_sha256: str,
    rep_fit: PCAFit,
    context_fit: PCAFit,
    keys: np.ndarray,
    residual: np.ndarray,
) -> None:
    quantized = np.clip(np.rint(keys * 127.0), -127, 127).astype(np.int8)
    save_npz(
        path,
        schema=np.asarray("phase12_pc_memlib_v1"),
        representation=np.asarray(name),
        checkpoint_sha256=np.asarray(checkpoint_sha256),
        key_scale=np.asarray(127.0, dtype=np.float32),
        context_weight=np.asarray(CONTEXT_WEIGHT, dtype=np.float32),
        representation_centre=rep_fit.centre,
        representation_basis=rep_fit.basis,
        representation_scale=rep_fit.scale,
        context_centre=context_fit.centre,
        context_basis=context_fit.basis,
        context_scale=context_fit.scale,
        keys_int8=quantized,
        residual_fp16=residual.astype(np.float16),
    )


def protocol_check(
    checkpoint: dict[str, Any],
    train_reaches: np.ndarray,
    validation_reaches: np.ndarray,
    test_reaches: np.ndarray,
) -> None:
    if len(set(train_reaches) & set(validation_reaches)):
        raise ValueError("Train/validation reach leakage")
    if len(set(train_reaches) & set(test_reaches)):
        raise ValueError("Train/test reach leakage")
    if len(set(validation_reaches) & set(test_reaches)):
        raise ValueError("Validation/test reach leakage")
    if np.asarray(checkpoint["feature_mean"]).shape != (FEATURES, 1):
        raise ValueError("Feature mean shape mismatch")
    synthetic = np.arange(FEATURES * 7, dtype=np.float32).reshape(FEATURES, 7)
    windows, lengths = rolling_windows(synthetic)
    if windows.shape != (7, FEATURES, WINDOW_BINS):
        raise ValueError("Rolling window shape contract changed")
    if not np.array_equal(windows[-1, :, -7:], synthetic):
        raise ValueError("Rolling window is not oldest-to-newest causal data")
    if not np.array_equal(lengths, np.arange(1, 8)):
        raise ValueError("Masked pooling lengths are incorrect")


def main() -> None:
    from scipy.spatial import cKDTree
    import torch

    args = parse_args()
    if args.threads < 1 or args.batch_size < 1:
        raise ValueError("threads and batch-size must be positive")
    torch.set_num_threads(args.threads)
    np.random.seed(PCA_SEED)
    device = select_device(args.device)
    session_name = args.session
    checkpoint_path = MODEL_ROOT / session_name / CHECKPOINT_NAME
    model, checkpoint = load_model(checkpoint_path, device, session_name)
    session = load_session(SESSION_BY_NAME[session_name])
    counts_all, velocity = aggregate_40ms(session)
    bounds = binned_reach_bounds(session)
    train_reaches, validation_reaches, test_reaches = split_fold(
        make_fold_indices(eligible_reaches(session)), 0
    )
    channels = np.asarray(checkpoint["selected_channel_indices"], dtype=np.int64)
    counts = counts_all[channels].astype(np.float32)
    protocol_check(checkpoint, train_reaches, validation_reaches, test_reaches)
    if args.protocol_check_only:
        print(
            json.dumps(
                {
                    "status": "protocol_check_passed",
                    "session": session_name,
                    "checkpoint_fold": int(checkpoint["fold"]),
                    "split_reaches": {
                        "train": len(train_reaches),
                        "validation": len(validation_reaches),
                        "test": len(test_reaches),
                    },
                },
                indent=2,
            )
        )
        return

    session_result_dir = RESULT_DIR / "by_session" / session_name
    metrics_path = session_result_dir / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metrics_path} exists; use --overwrite")
    session_result_dir.mkdir(parents=True, exist_ok=True)
    large_session_dir = LARGE_ROOT / session_name
    large_session_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_sha256 = sha256_file(checkpoint_path)

    extracted: dict[str, SplitData] = {}
    extraction_seconds: dict[str, float] = {}
    for split_name, reaches in (
        ("train", train_reaches),
        ("validation", validation_reaches),
        ("test", test_reaches),
    ):
        started = time.perf_counter()
        extracted[split_name] = extract_split(
            model, checkpoint, counts, velocity, bounds, reaches, device, args.batch_size
        )
        extraction_seconds[split_name] = time.perf_counter() - started
        print(f"extracted {split_name}: {len(extracted[split_name].target)} bins")

    train = extracted["train"]
    validation = extracted["validation"]
    test = extracted["test"]
    context_fit = fit_pca(train.context, CONTEXT_DIMS, PCA_SEED + 1)
    transformed_context = {
        name: transform_pca(split.context, context_fit)
        for name, split in extracted.items()
    }
    base_validation = metric_values(validation.target, validation.prediction)
    base_test = metric_values(test.target, test.prediction)
    records: list[dict[str, Any]] = []
    detailed: dict[str, Any] = {}
    corrected_test_predictions: dict[str, np.ndarray] = {}

    for rep_index, name in enumerate(REPRESENTATIONS):
        print(f"evaluating {name}")
        rep_fit = fit_pca(train.representation[name], REPRESENTATION_DIMS, PCA_SEED + 10 + rep_index)
        transformed_rep = {
            split_name: transform_pca(split.representation[name], rep_fit)
            for split_name, split in extracted.items()
        }
        keys = {
            split_name: make_keys(transformed_rep[split_name], transformed_context[split_name])
            for split_name in extracted
        }
        tree_started = time.perf_counter()
        tree = cKDTree(keys["train"])
        build_seconds = time.perf_counter() - tree_started
        val_distance, val_index, val_search_seconds = query_tree(
            tree, keys["validation"], args.threads
        )
        tuning, _ = tune_validation(validation, val_distance, val_index, train.residual)
        test_distance, test_index, test_search_seconds = query_tree(
            tree, keys["test"], args.threads
        )
        estimate = weighted_residual(
            test_distance,
            test_index,
            train.residual,
            int(tuning["neighbours"]),
            float(tuning["temperature"]),
        )
        corrected = test.prediction + float(tuning["blend"]) * estimate
        corrected_test_predictions[name] = corrected
        corrected_metrics = metric_values(test.target, corrected)
        consistency = residual_consistency(
            test.residual,
            estimate,
            train.residual[test_index[:, : int(tuning["neighbours"])]],
        )
        bootstrap_base = bootstrap_r2_delta(
            test.target,
            corrected,
            test.prediction,
            test.reach_id,
            BOOTSTRAP_REPETITIONS,
            PCA_SEED + 100 + rep_index,
        )
        mcu = projected_mcu_cost(len(train.target), train.representation[name].shape[1])
        memlib_path = large_session_dir / f"phase12_{name}.memlib"
        export_memlib(
            memlib_path,
            name,
            checkpoint_sha256,
            rep_fit,
            context_fit,
            keys["train"],
            train.residual,
        )
        actual_bytes = memlib_path.stat().st_size
        record = {
            "representation": name,
            "source_dims": train.representation[name].shape[1],
            "pca_dims": REPRESENTATION_DIMS,
            "context_dims": CONTEXT_DIMS,
            "key_dims": KEY_DIMS,
            "validation_r2": float(tuning["validation_r2_mean"]),
            "test_base_r2": base_test["r2_mean"],
            "test_corrected_r2": corrected_metrics["r2_mean"],
            "test_delta_r2": corrected_metrics["r2_mean"] - base_test["r2_mean"],
            "delta_ci95_low": bootstrap_base["ci95_low"],
            "delta_ci95_high": bootstrap_base["ci95_high"],
            "residual_r2": consistency["residual_r2_mean"],
            "neighbour_residual_dispersion": consistency["neighbour_residual_dispersion"],
            "neighbours": int(tuning["neighbours"]),
            "temperature": float(tuning["temperature"]),
            "blend": float(tuning["blend"]),
            "bank_entries": len(train.target),
            "pc_memlib_bytes": actual_bytes,
            "projected_mcu_bank_bytes": int(mcu["projected_bank_bytes"]),
            "pc_exact_search_us_per_query": 1e6 * test_search_seconds / len(test.target),
            "mcu_int8_mac_per_query_proxy": int(mcu["estimated_int8_mac_per_query"]),
        }
        records.append(record)
        detailed[name] = {
            "pca_explained_variance_ratio_sum": float(rep_fit.explained_variance_ratio.sum()),
            "tuning": tuning,
            "test_corrected": corrected_metrics,
            "test_delta_vs_base": bootstrap_base,
            "nearest_neighbour_residual_consistency": consistency,
            "pc_search": {
                "index": "scipy.spatial.cKDTree exact Euclidean over normalized keys",
                "build_seconds": build_seconds,
                "validation_search_seconds": val_search_seconds,
                "test_search_seconds": test_search_seconds,
                "test_us_per_query": 1e6 * test_search_seconds / len(test.target),
            },
            "mcu_cost_proxy": mcu,
            "memlib": {
                "path": str(memlib_path.relative_to(REPOSITORY_ROOT)),
                "schema": "phase12_pc_memlib_v1",
                "bytes": actual_bytes,
                "firmware_v1_compatible": False,
            },
        }

    winner = max(records, key=lambda row: row["validation_r2"])["representation"]
    for rep_index, name in enumerate(REPRESENTATIONS):
        detailed[name]["test_delta_vs_encoder_49"] = bootstrap_r2_delta(
            test.target,
            corrected_test_predictions[name],
            corrected_test_predictions["encoder_49"],
            test.reach_id,
            BOOTSTRAP_REPETITIONS,
            PCA_SEED + 200 + rep_index,
        )
    write_csv(session_result_dir / "representation_comparison.csv", records)
    result = {
        "schema_version": 1,
        "phase": PHASE_NAME,
        "created_at_utc": utc_now(),
        "status": "complete",
        "session": session_name,
        "subject": SESSION_BY_NAME[session_name].subject,
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(REPOSITORY_ROOT)),
            "sha256": checkpoint_sha256,
            "fold": int(checkpoint["fold"]),
            "best_epoch": int(checkpoint["best_epoch"]),
            "why_not_best_fold_checkpoint": "best_fold_checkpoint.pt was selected using test R2",
        },
        "protocol": {
            "split_unit": "reach",
            "split_reaches": {
                "train": len(train_reaches),
                "validation": len(validation_reaches),
                "test": len(test_reaches),
            },
            "split_bins": {
                "train": len(train.target),
                "validation": len(validation.target),
                "test": len(test.target),
            },
            "pca_fit": "train only",
            "bank_entries": "train only",
            "hyperparameter_selection": "validation only, independently per representation",
            "final_evaluation": "held-out test reaches",
            "bootstrap_unit": "reach",
            "rolling_window_bins": WINDOW_BINS,
            "early_reach_policy": "left-zero-pad; pooling masks padding",
            "long_context": {
                "source": "normalized 192D reach features",
                "fast_ewma_alpha": FAST_CONTEXT_ALPHA,
                "slow_ewma_alpha": SLOW_CONTEXT_ALPHA,
                "layout": "[fast, slow, current-fast]",
                "reset": "each reach",
                "source_dims": CONTEXT_SOURCE_DIMS,
                "pca_dims": CONTEXT_DIMS,
                "weight": CONTEXT_WEIGHT,
            },
        },
        "base_metrics": {"validation": base_validation, "test": base_test},
        "selection": {
            "criterion": "highest validation corrected R2",
            "winner": winner,
            "winner_test_metrics_are_reported_but_not_used_for_selection": True,
        },
        "representations": detailed,
        "extraction_seconds": extraction_seconds,
        "caveats": [
            "This is a one-session result and is not evidence of cross-session generalization.",
            "The gain is relative to the identical fold-1 rolling-window baseline; Phase-7 chunked inference and the test-selected best-fold score are not apples-to-apples baselines.",
            "PC cKDTree timing and MCU MAC/byte counts are cost proxies, not measured STM32 latency.",
            "The Phase-12 memlib archives are experimental PC artifacts, not BCIMEM firmware-v1 binaries.",
            "GRU/pooled/concatenated queries require exposing the matching intermediate state in the MCU graph.",
        ],
    }
    write_json(metrics_path, result)
    print(
        json.dumps(
            {
                "status": "complete",
                "session": session_name,
                "winner": winner,
                "results": str(metrics_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
