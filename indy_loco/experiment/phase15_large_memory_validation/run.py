#!/usr/bin/env python3
"""Phase 15: 30-fold PC validation of Large GRU-hidden residual memory.

Large uses the frozen Phase-13 neural checkpoints and adds one fold-specific
external residual bank.  Every bank and both PCA projections are fit from that
fold's training bins, retrieval hyperparameters are selected on validation
bins, and test bins are evaluated once.  The PC replay uses exact KNN over the
same INT8 key / FP16 residual numeric path intended for the MCU; it does not
claim firmware IVF latency or recall parity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import binomtest, wilcoxon

HERE = Path(__file__).resolve().parent
INDY_ROOT = HERE.parents[1]
REPOSITORY_ROOT = INDY_ROOT.parent
PHASE13_SCRIPT = INDY_ROOT / "experiment" / "phase13_deployment_validation" / "run_rolling_retrain.py"
MODEL_SCRIPT = INDY_ROOT / "models" / "midsize" / "model.py"
LARGE_ROOT = INDY_ROOT / "models" / "large"
RESULT_ROOT = HERE / "results"
FOLD_REFERENCE = (
    INDY_ROOT
    / "experiment"
    / "phase13_deployment_validation"
    / "results"
    / "rolling_retrain"
    / "final_30fold"
    / "phase13_round3_folds.csv"
)

SESSIONS: Final = (
    "indy_20160622_01",
    "indy_20160630_01",
    "indy_20170131_02",
    "loco_20170210_03",
    "loco_20170215_02",
    "loco_20170301_05",
)
FOLDS: Final = (1, 2, 3, 4, 5)
REPRESENTATION_DIMS: Final = 32
CONTEXT_DIMS: Final = 32
KEY_DIMS: Final = REPRESENTATION_DIMS + CONTEXT_DIMS
FAST_CONTEXT_ALPHA: Final = 0.02
SLOW_CONTEXT_ALPHA: Final = 0.005
CONTEXT_WEIGHT: Final = 0.5
NEIGHBOURS: Final = (8, 16, 32, 64, 128)
TEMPERATURES: Final = (0.02, 0.05, 0.10, 0.20)
BLENDS: Final = (0.25, 0.50, 0.75, 1.00)
PCA_SEED: Final = 15_013
BOOTSTRAP_REPETITIONS: Final = 1_000
SESSION_BOOTSTRAP_REPETITIONS: Final = 100_000


def import_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PHASE13 = import_file("phase15_phase13", PHASE13_SCRIPT)
MODEL = import_file("phase15_model", MODEL_SCRIPT)
PHASE7 = PHASE13.PHASE7


@dataclass(frozen=True)
class Split:
    hidden: np.ndarray
    context: np.ndarray
    prediction: np.ndarray
    target: np.ndarray
    residual: np.ndarray
    reach_id: np.ndarray
    bins: np.ndarray


@dataclass(frozen=True)
class PCAFit:
    centre: np.ndarray
    basis: np.ndarray
    scale: np.ndarray
    explained_variance_ratio: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", action="append", choices=SESSIONS)
    parser.add_argument("--fold", action="append", type=int, choices=FOLDS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def select_device(requested: str) -> Any:
    import torch

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        return torch.device("mps")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def checkpoint_path(session: str, fold: int) -> Path:
    candidates = sorted((LARGE_ROOT / session).glob(f"fold-{fold}*.pt"))
    if len(candidates) != 1:
        raise ValueError(f"{session} fold {fold}: expected one checkpoint, found {candidates}")
    return candidates[0]


def load_fold_reference() -> dict[tuple[str, int], float]:
    with FOLD_REFERENCE.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    reference = {
        (row["session"], int(row["fold"])): float(row["retrained_7min_rolling_r2"])
        for row in rows
    }
    if set(reference) != {(session, fold) for session in SESSIONS for fold in FOLDS}:
        raise ValueError("Phase-13 fold reference is not the complete 6 x 5 grid")
    return reference


def causal_ewma(values: np.ndarray, alpha: float) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    output = source.copy()
    for index in range(1, source.shape[1]):
        output[:, index] = alpha * source[:, index] + (1.0 - alpha) * output[:, index - 1]
    return output


def long_context(normalized: np.ndarray) -> np.ndarray:
    fast = causal_ewma(normalized, FAST_CONTEXT_ALPHA)
    slow = causal_ewma(normalized, SLOW_CONTEXT_ALPHA)
    return np.concatenate((fast, slow, normalized - fast), axis=0).T.astype(np.float32)


def model_intermediates(model: Any, inputs: Any) -> tuple[Any, Any]:
    encoded = model.spatial(inputs)
    for convolution, padding in zip(model.convolutions, model.padding, strict=True):
        encoded = model.activation(convolution(encoded)[:, :, :-padding] + encoded)
    states, _ = model.gru(encoded.transpose(1, 2))
    return states, model.head(states)


def reach_ids_for_bins(bounds: np.ndarray, reaches: np.ndarray, bins: np.ndarray) -> np.ndarray:
    output = np.full(len(bins), -1, dtype=np.int32)
    for reach in reaches:
        start, stop = (int(value) for value in bounds[int(reach)])
        output[(bins >= start) & (bins < stop)] = int(reach)
    if np.any(output < 0):
        raise ValueError("A selected bin is outside its split reaches")
    return output


def extract_split(
    model: Any,
    fold_data: Any,
    context_all: np.ndarray,
    bins: np.ndarray,
    reaches: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: Any,
    batch_size: int,
) -> Split:
    import torch

    hidden_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for left in range(0, len(bins), batch_size):
            selected = bins[left : left + batch_size]
            inputs = torch.from_numpy(
                PHASE13.rolling_batch(fold_data.normalized_features, selected)
            ).to(device)
            states, output = model_intermediates(model, inputs)
            hidden_parts.append(states[:, -1].cpu().numpy().astype(np.float32))
            normalized_prediction = output[:, -1].cpu().numpy().astype(np.float32)
            prediction_parts.append(normalized_prediction * target_std + target_mean)
    prediction = np.concatenate(prediction_parts).astype(np.float32)
    target = fold_data.velocity[bins].astype(np.float32)
    return Split(
        hidden=np.concatenate(hidden_parts).astype(np.float32),
        context=context_all[bins].astype(np.float32),
        prediction=prediction,
        target=target,
        residual=(target - prediction).astype(np.float32),
        reach_id=reach_ids_for_bins(fold_data.bounds, reaches, bins),
        bins=bins,
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
    return PCAFit(centre, basis.numpy().astype(np.float32), scale, explained)


def transform_pca(values: np.ndarray, fit: PCAFit) -> np.ndarray:
    return ((values - fit.centre) @ fit.basis / fit.scale).astype(np.float32)


def row_normalize(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return (values / np.maximum(norm, 1e-8)).astype(np.float32)


def make_keys(hidden: np.ndarray, context: np.ndarray) -> np.ndarray:
    hidden_unit = row_normalize(hidden)
    context_unit = row_normalize(context)
    return row_normalize(
        np.concatenate((hidden_unit, CONTEXT_WEIGHT * context_unit), axis=1)
    )


def quantize_keys(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    integer = np.clip(np.rint(values * 127.0), -127, 127).astype(np.int8)
    return integer, (integer.astype(np.float32) / 127.0).astype(np.float32)


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    residual = np.sum((target - prediction) ** 2, axis=0)
    total = np.sum((target - target.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - residual / np.maximum(total, 1e-12)
    return {
        "bins": int(len(target)),
        "mse": float(np.mean((target - prediction) ** 2)),
        "r2_x": float(r2[0]),
        "r2_y": float(r2[1]),
        "r2_mean": float(r2.mean()),
    }


def residual_estimate(
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


def tune(
    validation: Split,
    distance: np.ndarray,
    index: np.ndarray,
    bank_residual: np.ndarray,
) -> dict[str, float | int]:
    best: dict[str, float | int] | None = None
    for neighbours in NEIGHBOURS:
        for temperature in TEMPERATURES:
            estimate = residual_estimate(
                distance, index, bank_residual, neighbours, temperature
            )
            for blend in BLENDS:
                score = metrics(
                    validation.target, validation.prediction + blend * estimate
                )["r2_mean"]
                candidate: dict[str, float | int] = {
                    "neighbours": neighbours,
                    "temperature": temperature,
                    "blend": blend,
                    "validation_r2_mean": float(score),
                }
                if best is None or score > best["validation_r2_mean"]:
                    best = candidate
    if best is None:
        raise RuntimeError("No retrieval hyperparameters were selected")
    return best


def bootstrap_delta(
    target: np.ndarray,
    ready: np.ndarray,
    absent: np.ndarray,
    reach_id: np.ndarray,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    unique = np.unique(reach_id)
    rows_by_reach = {reach: np.flatnonzero(reach_id == reach) for reach in unique}
    deltas = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    for repetition in range(BOOTSTRAP_REPETITIONS):
        sample = rng.choice(unique, len(unique), replace=True)
        rows = np.concatenate([rows_by_reach[reach] for reach in sample])
        deltas[repetition] = (
            metrics(target[rows], ready[rows])["r2_mean"]
            - metrics(target[rows], absent[rows])["r2_mean"]
        )
    return {
        "repetitions": BOOTSTRAP_REPETITIONS,
        "unit": "reach",
        "mean": float(deltas.mean()),
        "ci95_low": float(np.percentile(deltas, 2.5)),
        "ci95_high": float(np.percentile(deltas, 97.5)),
        "probability_positive": float(np.mean(deltas > 0)),
    }


def export_memlib(
    path: Path,
    checkpoint_sha256: str,
    hidden_fit: PCAFit,
    context_fit: PCAFit,
    keys_int8: np.ndarray,
    residual_fp16: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as destination:
            np.savez_compressed(
                destination,
                schema=np.asarray("phase15_pc_memlib_v1"),
                representation=np.asarray("gru_hidden_49_plus_long_context"),
                checkpoint_sha256=np.asarray(checkpoint_sha256),
                key_scale=np.asarray(127.0, dtype=np.float32),
                context_weight=np.asarray(CONTEXT_WEIGHT, dtype=np.float32),
                hidden_centre=hidden_fit.centre,
                hidden_basis=hidden_fit.basis,
                hidden_scale=hidden_fit.scale,
                context_centre=context_fit.centre,
                context_basis=context_fit.basis,
                context_scale=context_fit.scale,
                keys_int8=keys_int8,
                residual_fp16=residual_fp16,
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_checkpoint(checkpoint: dict[str, Any], fold_data: Any, session: str, fold: int) -> None:
    if checkpoint.get("session") != session or int(checkpoint.get("fold", -1)) != fold:
        raise ValueError(f"{session} fold {fold}: checkpoint identity mismatch")
    if checkpoint.get("selection_policy") != "minimum_validation_loss_test_opened_once":
        raise ValueError(f"{session} fold {fold}: invalid selection policy")
    if checkpoint.get("test_evaluated_during_training") is not False:
        raise ValueError(f"{session} fold {fold}: test used during checkpoint selection")
    expected_arrays = {
        "selected_channel_indices": fold_data.channels,
        "feature_mean": fold_data.calibration_mean[:, None],
        "feature_std": fold_data.calibration_effective_std[:, None],
        "target_mean": fold_data.target_mean,
        "target_std": fold_data.target_std,
    }
    for key, expected in expected_arrays.items():
        actual = np.asarray(checkpoint[key])
        if not np.allclose(actual, expected, rtol=0, atol=1e-6):
            raise ValueError(f"{session} fold {fold}: {key} differs from Phase-13 replay")
    expected_counts = {
        "train": len(fold_data.train_bins),
        "validation": len(fold_data.validation_bins),
        "test": len(fold_data.test_bins),
    }
    if checkpoint.get("bin_counts_after_calibration") != expected_counts:
        raise ValueError(f"{session} fold {fold}: split-bin counts changed")


def process_fold(
    session: str,
    fold: int,
    reference_r2: float,
    device: Any,
    batch_size: int,
    threads: int,
    validate_only: bool,
) -> dict[str, Any]:
    import torch

    path = checkpoint_path(session, fold)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    data = PHASE7.load_session(PHASE7.SESSION_BY_NAME[session])
    fold_data = PHASE13.prepare_fold(data, fold - 1)
    validate_checkpoint(checkpoint, fold_data, session, fold)
    if validate_only:
        return {
            "session": session,
            "subject": data.spec.subject,
            "fold": fold,
            "status": "validated",
            "checkpoint": str(path.relative_to(REPOSITORY_ROOT)),
        }

    model = MODEL.MidsizeTCNGRU()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    context_all = long_context(fold_data.normalized_features)
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
    target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)
    extracted = {
        "train": extract_split(
            model,
            fold_data,
            context_all,
            fold_data.train_bins,
            fold_data.train_reaches,
            target_mean,
            target_std,
            device,
            batch_size,
        ),
        "validation": extract_split(
            model,
            fold_data,
            context_all,
            fold_data.validation_bins,
            fold_data.validation_reaches,
            target_mean,
            target_std,
            device,
            batch_size,
        ),
        "test": extract_split(
            model,
            fold_data,
            context_all,
            fold_data.test_bins,
            fold_data.test_reaches,
            target_mean,
            target_std,
            device,
            batch_size,
        ),
    }
    del context_all
    train, validation, test = extracted["train"], extracted["validation"], extracted["test"]
    absent = metrics(test.target, test.prediction)
    if not np.isclose(absent["r2_mean"], reference_r2, rtol=0, atol=2e-6):
        raise ValueError(
            f"{session} fold {fold}: Phase-13 R2 mismatch "
            f"{absent['r2_mean']} vs {reference_r2}"
        )

    hidden_fit = fit_pca(train.hidden, REPRESENTATION_DIMS, PCA_SEED + fold)
    context_fit = fit_pca(train.context, CONTEXT_DIMS, PCA_SEED + 100 + fold)
    keys_int8: dict[str, np.ndarray] = {}
    keys_float: dict[str, np.ndarray] = {}
    for name, split in extracted.items():
        combined = make_keys(
            transform_pca(split.hidden, hidden_fit),
            transform_pca(split.context, context_fit),
        )
        keys_int8[name], keys_float[name] = quantize_keys(combined)
    bank_residual = train.residual.astype(np.float16).astype(np.float32)
    tree = cKDTree(keys_float["train"])
    validation_distance, validation_index = tree.query(
        keys_float["validation"], k=max(NEIGHBOURS), workers=threads
    )
    tuning = tune(validation, validation_distance, validation_index, bank_residual)
    search_started = time.perf_counter()
    test_distance, test_index = tree.query(
        keys_float["test"], k=max(NEIGHBOURS), workers=threads
    )
    search_seconds = time.perf_counter() - search_started
    estimate = residual_estimate(
        test_distance,
        test_index,
        bank_residual,
        int(tuning["neighbours"]),
        float(tuning["temperature"]),
    )
    ready_prediction = test.prediction + float(tuning["blend"]) * estimate
    ready = metrics(test.target, ready_prediction)
    delta = float(ready["r2_mean"] - absent["r2_mean"])
    checkpoint_sha256 = sha256_file(path)
    memlib_path = RESULT_ROOT / "memlibs" / session / f"fold-{fold}.memlib"
    export_memlib(
        memlib_path,
        checkpoint_sha256,
        hidden_fit,
        context_fit,
        keys_int8["train"],
        train.residual.astype(np.float16),
    )
    return {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "status": "complete",
        "session": session,
        "subject": data.spec.subject,
        "fold": fold,
        "checkpoint": str(path.relative_to(REPOSITORY_ROOT)),
        "checkpoint_sha256": checkpoint_sha256,
        "protocol": {
            "phase13_checkpoint_frozen": True,
            "calibration_minutes": 7.0,
            "continuous_session_ewma": True,
            "rolling_window_bins": 50,
            "query": "GRU hidden[49] 64D -> train PCA 32D + long context 576D -> train PCA 32D",
            "bank_and_pca": "train bins only",
            "retrieval_tuning": "validation bins only",
            "test_use": "one final evaluation after tuning",
            "key_numeric_path": "row-normalized, INT8 rounded, exact PC KNN",
            "residual_numeric_path": "FP16 rounded",
            "search_caveat": "exact PC KNN; not firmware IVF recall or latency",
        },
        "bin_counts": {
            "train": len(train.bins),
            "validation": len(validation.bins),
            "test": len(test.bins),
        },
        "bank_absent": absent,
        "bank_ready": ready,
        "ready_minus_absent_r2": delta,
        "reach_bootstrap_delta": bootstrap_delta(
            test.target,
            ready_prediction,
            test.prediction,
            test.reach_id,
            PCA_SEED + 1_000 + 10 * fold,
        ),
        "retrieval": {
            "tuning": tuning,
            "bank_entries": len(train.bins),
            "key_dimensions": KEY_DIMS,
            "test_exact_search_us_per_query": 1e6 * search_seconds / len(test.bins),
            "hidden_pca_explained_variance": float(
                hidden_fit.explained_variance_ratio.sum()
            ),
            "context_pca_explained_variance": float(
                context_fit.explained_variance_ratio.sum()
            ),
        },
        "memlib": {
            "path": str(memlib_path.relative_to(REPOSITORY_ROOT)),
            "schema": "phase15_pc_memlib_v1",
            "bytes": memlib_path.stat().st_size,
            "firmware_bcimem_compatible": False,
        },
    }


def fold_row(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "session": result["session"],
        "subject": result["subject"],
        "fold": result["fold"],
        "absent_r2": result["bank_absent"]["r2_mean"],
        "ready_r2": result["bank_ready"]["r2_mean"],
        "delta_r2": result["ready_minus_absent_r2"],
        "delta_ci95_low": result["reach_bootstrap_delta"]["ci95_low"],
        "delta_ci95_high": result["reach_bootstrap_delta"]["ci95_high"],
        "validation_r2": result["retrieval"]["tuning"]["validation_r2_mean"],
        "neighbours": result["retrieval"]["tuning"]["neighbours"],
        "temperature": result["retrieval"]["tuning"]["temperature"],
        "blend": result["retrieval"]["tuning"]["blend"],
        "bank_entries": result["retrieval"]["bank_entries"],
        "test_bins": result["bank_absent"]["bins"],
        "memlib_bytes": result["memlib"]["bytes"],
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [fold_row(result) for result in results]
    by_session = []
    for session in SESSIONS:
        selected = [row for row in rows if row["session"] == session]
        absent = np.asarray([row["absent_r2"] for row in selected])
        ready = np.asarray([row["ready_r2"] for row in selected])
        by_session.append(
            {
                "session": session,
                "subject": selected[0]["subject"],
                "folds": len(selected),
                "absent_r2_mean": float(absent.mean()),
                "ready_r2_mean": float(ready.mean()),
                "delta_r2_mean": float((ready - absent).mean()),
            }
        )
    absent = np.asarray([row["absent_r2"] for row in rows])
    ready = np.asarray([row["ready_r2"] for row in rows])
    deltas = ready - absent
    session_deltas = np.asarray([row["delta_r2_mean"] for row in by_session])
    rng = np.random.default_rng(PCA_SEED + 9_999)
    sampled = rng.integers(
        0,
        len(session_deltas),
        size=(SESSION_BOOTSTRAP_REPETITIONS, len(session_deltas)),
    )
    bootstrap = session_deltas[sampled].mean(axis=1)
    wilcoxon_one = wilcoxon(session_deltas, alternative="greater", method="exact")
    wilcoxon_two = wilcoxon(session_deltas, alternative="two-sided", method="exact")
    positive_sessions = int(np.sum(session_deltas > 0))
    sign = binomtest(positive_sessions, len(session_deltas), 0.5, alternative="greater")
    summary = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "status": "complete",
        "question": "What is the 30-fold test R2 of Phase-13 Large with fold-specific GRU-hidden external residual memory?",
        "reporting_unit": "unweighted macro average over 30 validation-selected folds",
        "folds": len(rows),
        "sessions": len(by_session),
        "bank_absent_r2_mean": float(absent.mean()),
        "bank_absent_r2_std_across_folds": float(absent.std(ddof=1)),
        "bank_ready_r2_mean": float(ready.mean()),
        "bank_ready_r2_std_across_folds": float(ready.std(ddof=1)),
        "ready_minus_absent_r2_mean": float(deltas.mean()),
        "ready_minus_absent_r2_median": float(np.median(deltas)),
        "positive_folds": int(np.sum(deltas > 0)),
        "positive_sessions": positive_sessions,
        "session_bootstrap_delta_ci95": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "exact_one_sided_wilcoxon_session_p": float(wilcoxon_one.pvalue),
        "exact_two_sided_wilcoxon_session_p": float(wilcoxon_two.pvalue),
        "exact_one_sided_sign_test_session_p": float(sign.pvalue),
        "indy_15fold": {
            "absent_r2_mean": float(absent[:15].mean()),
            "ready_r2_mean": float(ready[:15].mean()),
            "delta_r2_mean": float(deltas[:15].mean()),
        },
        "loco_15fold": {
            "absent_r2_mean": float(absent[15:].mean()),
            "ready_r2_mean": float(ready[15:].mean()),
            "delta_r2_mean": float(deltas[15:].mean()),
        },
        "session_results": by_session,
        "caveats": [
            "This is exact PC KNN over INT8-rounded keys, not firmware IVF search.",
            "The seven-minute calibration and continuous causal context use no labels but are transductive within each session.",
            "Folds from one session are correlated; session-level paired inference is primary.",
        ],
    }
    write_csv(RESULT_ROOT / "phase15_large_memory_folds.csv", rows)
    write_csv(RESULT_ROOT / "phase15_large_memory_sessions.csv", by_session)
    write_json(RESULT_ROOT / "phase15_large_memory_summary.json", summary)
    return summary


def main() -> None:
    import torch

    args = parse_args()
    if args.threads < 1 or args.batch_size < 1:
        raise ValueError("threads and batch size must be positive")
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    sessions = tuple(args.session or SESSIONS)
    folds = tuple(args.fold or FOLDS)
    references = load_fold_reference()
    work = [(session, fold) for session in sessions for fold in folds]
    results: list[dict[str, Any]] = []
    for index, (session, fold) in enumerate(work, start=1):
        result_path = RESULT_ROOT / "by_fold" / session / f"fold-{fold}.json"
        if args.resume and result_path.is_file() and not args.validate_only:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            print(f"[{index}/{len(work)}] {session} fold {fold}: resumed", flush=True)
        else:
            if result_path.exists() and not (args.overwrite or args.validate_only):
                raise FileExistsError(f"{result_path} exists; use --resume or --overwrite")
            started = time.perf_counter()
            print(f"[{index}/{len(work)}] {session} fold {fold}: running", flush=True)
            result = process_fold(
                session,
                fold,
                references[(session, fold)],
                device,
                args.batch_size,
                args.threads,
                args.validate_only,
            )
            result["elapsed_seconds"] = time.perf_counter() - started
            if not args.validate_only:
                write_json(result_path, result)
            print(
                f"[{index}/{len(work)}] {session} fold {fold}: "
                + (
                    "validated"
                    if args.validate_only
                    else f"{result['bank_absent']['r2_mean']:.4f} -> "
                    f"{result['bank_ready']['r2_mean']:.4f} "
                    f"({result['ready_minus_absent_r2']:+.4f})"
                ),
                flush=True,
            )
        results.append(result)
    if args.validate_only:
        print(json.dumps({"status": "validated", "folds": len(results)}, indent=2))
        return
    if set(work) == {(session, fold) for session in SESSIONS for fold in FOLDS}:
        print(json.dumps(aggregate(results), indent=2), flush=True)
    else:
        print(json.dumps({"status": "partial", "folds": len(results)}, indent=2))


if __name__ == "__main__":
    main()
