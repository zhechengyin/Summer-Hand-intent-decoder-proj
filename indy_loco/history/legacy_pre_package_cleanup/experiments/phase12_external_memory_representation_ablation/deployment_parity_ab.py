#!/usr/bin/env python3
"""Best-fold deployment-preprocessing A/B: bank ABSENT vs GRU bank READY.

This replay uses each GUI-selected deployment candidate, its selected-fold
train/validation/test reaches, the existing 60-second calibration and std
floor, continuous session EWMA, and 50-bin rolling model input.  GRU keys are
PCA-compressed with train bins only, int8-quantized, and searched exactly on
PC; stored residuals are rounded through FP16.  Exact KNN isolates memory
quality but is not an IVF latency/recall measurement.
"""

from __future__ import annotations

import argparse
import csv
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

ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run as phase12  # noqa: E402
from indy_loco.experiments.active import phase10_session_deployment_candidates as p10  # noqa: E402
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

MODEL_ROOT = REPOSITORY_ROOT / "indy_loco" / "models" / "midsize"
LARGE_ROOT = REPOSITORY_ROOT / "indy_loco" / "models" / "large"
RESULT_ROOT = ROOT / "results" / "deployment_parity"
SESSIONS: Final = (
    "indy_20160622_01",
    "indy_20160630_01",
    "indy_20170131_02",
    "loco_20170210_03",
    "loco_20170215_02",
    "loco_20170301_05",
)
BOOTSTRAP_REPETITIONS: Final = 1_000
SESSION_BOOTSTRAP_REPETITIONS: Final = 100_000
SEED: Final = 12_121_111


@dataclass(frozen=True)
class ReplaySplit:
    representation: np.ndarray
    context: np.ndarray
    prediction: np.ndarray
    target: np.ndarray
    residual: np.ndarray
    reach_id: np.ndarray
    bin_id: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--protocol-check-only", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_candidate(session: str, device: Any) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    import torch

    candidate_path = MODEL_ROOT / session / "deployment_candidate.pt"
    replay_path = MODEL_ROOT / session / "deployment_replay.json"
    candidate = torch.load(candidate_path, map_location=device, weights_only=False)
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    fold = int(candidate["fold"])
    if candidate["session"] != session or fold != int(replay["selected_fold"]):
        raise ValueError(f"{session}: candidate/replay identity mismatch")
    if candidate.get("selection_policy") != "highest_phase7_test_r2_fold":
        raise ValueError(f"{session}: expected GUI best-fold selection policy")
    model = MidsizeTCNGRU().to(device)
    model.load_state_dict(candidate["model_state"], strict=True)
    model.eval()
    return model, candidate, replay


def calibrated_session_features(
    counts: np.ndarray, candidate: dict[str, Any]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    features = p10.continuous_features(counts)
    calibration = features[:, : p10.CALIBRATION_BINS]
    mean = calibration.mean(axis=1).astype(np.float32)
    local_std = (calibration.std(axis=1, ddof=0) + 1e-6).astype(np.float32)
    floor = np.asarray(candidate["feature_std_floor"], dtype=np.float32).reshape(p10.FEATURES)
    effective_std = np.maximum(local_std, floor).astype(np.float32)
    normalized = ((features - mean[:, None]) / effective_std[:, None]).astype(np.float32)
    return normalized, {"mean": mean, "local_std": local_std, "effective_std": effective_std}


def bins_for_reaches(
    bounds: np.ndarray, reaches: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    bins: list[np.ndarray] = []
    reach_ids: list[np.ndarray] = []
    for reach in sorted((int(value) for value in reaches), key=lambda value: bounds[value, 0]):
        start, stop = (int(value) for value in bounds[reach])
        selected = np.arange(start, stop, dtype=np.int64)
        selected = selected[selected >= p10.CALIBRATION_BINS - 1]
        if selected.size:
            bins.append(selected)
            reach_ids.append(np.full(len(selected), reach, dtype=np.int32))
    if not bins:
        raise ValueError("No bins remain after deployment calibration")
    return np.concatenate(bins), np.concatenate(reach_ids)


def extract_split(
    model: Any,
    candidate: dict[str, Any],
    normalized: np.ndarray,
    context_all: np.ndarray,
    velocity: np.ndarray,
    bins: np.ndarray,
    reach_ids: np.ndarray,
    device: Any,
    batch_size: int,
) -> ReplaySplit:
    import torch

    target_mean = np.asarray(candidate["target_mean"], dtype=np.float32).reshape(2)
    target_std = np.asarray(candidate["target_std"], dtype=np.float32).reshape(2)
    representations: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for left in range(0, len(bins), batch_size):
        batch_bins = bins[left : left + batch_size]
        inputs = p10.rolling_inputs(normalized, batch_bins)
        tensor = torch.from_numpy(inputs).to(device)
        with torch.inference_mode():
            _, states, output = phase12.model_intermediates(model, tensor)
        representations.append(states[:, -1].cpu().numpy().astype(np.float32))
        normalized_prediction = output[:, -1].cpu().numpy().astype(np.float32)
        predictions.append(normalized_prediction * target_std + target_mean)
    prediction = np.concatenate(predictions).astype(np.float32)
    target = velocity[bins].astype(np.float32)
    return ReplaySplit(
        representation=np.concatenate(representations).astype(np.float32),
        context=context_all[bins].astype(np.float32),
        prediction=prediction,
        target=target,
        residual=(target - prediction).astype(np.float32),
        reach_id=reach_ids,
        bin_id=bins,
    )


def quantized_keys(values: np.ndarray) -> np.ndarray:
    quantized = np.clip(np.rint(values * 127.0), -127, 127).astype(np.int8)
    return (quantized.astype(np.float32) / 127.0).astype(np.float32)


def tune(
    split: ReplaySplit,
    distance: np.ndarray,
    index: np.ndarray,
    bank_residual: np.ndarray,
) -> dict[str, float | int]:
    best: dict[str, float | int] | None = None
    for neighbours in phase12.NEIGHBOURS:
        for temperature in phase12.TEMPERATURES:
            estimate = phase12.weighted_residual(
                distance, index, bank_residual, neighbours, temperature
            )
            for blend in phase12.BLENDS:
                score = phase12.metric_values(
                    split.target, split.prediction + blend * estimate
                )["r2_mean"]
                candidate: dict[str, float | int] = {
                    "neighbours": neighbours,
                    "temperature": temperature,
                    "blend": blend,
                    "validation_r2_mean": score,
                }
                if best is None or score > best["validation_r2_mean"]:
                    best = candidate
    assert best is not None
    return best


def process_session(
    session: str,
    device: Any,
    threads: int,
    batch_size: int,
    overwrite: bool,
    protocol_check_only: bool,
) -> dict[str, Any]:
    model, candidate, existing_replay = load_candidate(session, device)
    data = load_session(SESSION_BY_NAME[session])
    counts_all, velocity = aggregate_40ms(data)
    bounds = binned_reach_bounds(data)
    fold = int(candidate["fold"])
    train_reaches, validation_reaches, test_reaches = split_fold(
        make_fold_indices(eligible_reaches(data)), fold - 1
    )
    channels = np.asarray(candidate["selected_channel_indices"], dtype=np.int64)
    counts = counts_all[channels].astype(np.float32)
    normalized, calibration = calibrated_session_features(counts, candidate)
    context_all = phase12.long_context(normalized)
    split_bins: dict[str, np.ndarray] = {}
    split_reach_ids: dict[str, np.ndarray] = {}
    for name, reaches in (
        ("train", train_reaches),
        ("validation", validation_reaches),
        ("test", test_reaches),
    ):
        split_bins[name], split_reach_ids[name] = bins_for_reaches(bounds, reaches)
    if len(set(split_bins["train"]) & set(split_bins["validation"])):
        raise ValueError(f"{session}: train/validation bin leakage")
    if len(set(split_bins["train"]) & set(split_bins["test"])):
        raise ValueError(f"{session}: train/test bin leakage")
    if len(split_bins["test"]) != existing_replay["replay"]["bins_after_calibration"]:
        raise ValueError(f"{session}: selected-fold test-bin count mismatch")
    if protocol_check_only:
        return {
            "session": session,
            "fold": fold,
            "status": "protocol_check_passed",
            "bins": {name: len(values) for name, values in split_bins.items()},
        }

    output_dir = RESULT_ROOT / "by_session" / session
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists() and not overwrite:
        raise FileExistsError(f"{metrics_path} exists; use --overwrite")
    extracted = {
        name: extract_split(
            model,
            candidate,
            normalized,
            context_all,
            velocity,
            split_bins[name],
            split_reach_ids[name],
            device,
            batch_size,
        )
        for name in ("train", "validation", "test")
    }
    train, validation, test = (
        extracted["train"],
        extracted["validation"],
        extracted["test"],
    )
    absent = phase12.metric_values(test.target, test.prediction)
    expected_absent = existing_replay["replay"]["firmware_policy_same_bins"]
    for key in ("r2_x", "r2_y", "r2_mean", "mse"):
        if not np.isclose(absent[key], expected_absent[key], rtol=0, atol=2e-6):
            raise ValueError(
                f"{session}: ABSENT replay mismatch for {key}: "
                f"{absent[key]} vs {expected_absent[key]}"
            )

    rep_fit = phase12.fit_pca(train.representation, phase12.REPRESENTATION_DIMS, SEED + fold)
    context_fit = phase12.fit_pca(train.context, phase12.CONTEXT_DIMS, SEED + 100 + fold)
    keys = {}
    for name, split in extracted.items():
        rep = phase12.transform_pca(split.representation, rep_fit)
        context = phase12.transform_pca(split.context, context_fit)
        keys[name] = quantized_keys(phase12.make_keys(rep, context))
    bank_residual = train.residual.astype(np.float16).astype(np.float32)
    tree = cKDTree(keys["train"])
    validation_distance, validation_index, _ = phase12.query_tree(
        tree, keys["validation"], threads
    )
    tuning = tune(validation, validation_distance, validation_index, bank_residual)
    started = time.perf_counter()
    test_distance, test_index, _ = phase12.query_tree(tree, keys["test"], threads)
    search_seconds = time.perf_counter() - started
    estimate = phase12.weighted_residual(
        test_distance,
        test_index,
        bank_residual,
        int(tuning["neighbours"]),
        float(tuning["temperature"]),
    )
    ready_prediction = test.prediction + float(tuning["blend"]) * estimate
    ready = phase12.metric_values(test.target, ready_prediction)
    uplift = phase12.bootstrap_r2_delta(
        test.target,
        ready_prediction,
        test.prediction,
        test.reach_id,
        BOOTSTRAP_REPETITIONS,
        SEED + fold * 10,
    )
    consistency = phase12.residual_consistency(
        test.residual,
        estimate,
        bank_residual[test_index[:, : int(tuning["neighbours"])]],
    )
    memlib_path = LARGE_ROOT / session / "phase12_deployment_parity_gru_hidden_49.memlib"
    phase12.export_memlib(
        memlib_path,
        "deployment_parity_gru_hidden_49",
        phase12.sha256_file(MODEL_ROOT / session / "deployment_candidate.pt"),
        rep_fit,
        context_fit,
        keys["train"],
        bank_residual,
    )
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "session": session,
        "subject": SESSION_BY_NAME[session].subject,
        "status": "complete",
        "selected_fold": fold,
        "selection_policy": candidate["selection_policy"],
        "selection_test_r2_mean": float(existing_replay["selection_test_r2_mean"]),
        "protocol": {
            "candidate": str(
                (MODEL_ROOT / session / "deployment_candidate.pt").relative_to(
                    REPOSITORY_ROOT
                )
            ),
            "calibration_bins": p10.CALIBRATION_BINS,
            "continuous_session_ewma": True,
            "rolling_window_bins": p10.WINDOW_BINS,
            "train_only_bank_and_pca": True,
            "validation_only_retrieval_tuning": True,
            "test_bins_match_existing_deployment_replay": True,
            "key_numeric_path": "32D GRU PCA + 32D context PCA; int8 quantized",
            "residual_storage_path": "FP16 rounded",
            "search": "PC exact cKDTree over quantized keys; not firmware IVF",
            "split_reaches": {
                "train": len(train_reaches),
                "validation": len(validation_reaches),
                "test": len(test_reaches),
            },
            "split_bins_after_calibration": {
                name: len(values) for name, values in split_bins.items()
            },
        },
        "calibration": {
            "mean_min": float(calibration["mean"].min()),
            "mean_max": float(calibration["mean"].max()),
            "effective_std_min": float(calibration["effective_std"].min()),
            "effective_std_max": float(calibration["effective_std"].max()),
        },
        "bank_absent": absent,
        "bank_ready_gru": ready,
        "ready_minus_absent": {
            "r2_mean": ready["r2_mean"] - absent["r2_mean"],
            "mse": ready["mse"] - absent["mse"],
            "reach_bootstrap": uplift,
        },
        "retrieval": {
            "tuning": tuning,
            "residual_consistency": consistency,
            "bank_entries": len(train.target),
            "pc_exact_search_us_per_query": 1e6 * search_seconds / len(test.target),
        },
        "memlib": {
            "path": str(memlib_path.relative_to(REPOSITORY_ROOT)),
            "bytes": memlib_path.stat().st_size,
            "schema": "phase12_pc_memlib_v1",
            "firmware_bcimem_compatible": False,
        },
        "caveats": [
            "The checkpoint/fold was selected using test R2; absolute accuracy is selection-biased.",
            "Calibration and continuous rolling windows intentionally match deployment replay but allow unsupervised temporal context across reach splits.",
            "Search is exact PC KNN over int8 keys, not firmware IVF; this estimates memory quality, not final recall or latency.",
        ],
    }
    write_json(metrics_path, result)
    return result


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for result in results:
        rows.append(
            {
                "session": result["session"],
                "subject": result["subject"],
                "selected_fold": result["selected_fold"],
                "selection_test_r2_mean": result["selection_test_r2_mean"],
                "absent_r2": result["bank_absent"]["r2_mean"],
                "ready_gru_r2": result["bank_ready_gru"]["r2_mean"],
                "ready_minus_absent_r2": result["ready_minus_absent"]["r2_mean"],
                "ci95_low": result["ready_minus_absent"]["reach_bootstrap"]["ci95_low"],
                "ci95_high": result["ready_minus_absent"]["reach_bootstrap"]["ci95_high"],
                "individual_ci_excludes_zero": bool(
                    result["ready_minus_absent"]["reach_bootstrap"]["ci95_low"] > 0
                ),
                "bank_entries": result["retrieval"]["bank_entries"],
            }
        )
    deltas = np.asarray([row["ready_minus_absent_r2"] for row in rows])
    rng = np.random.default_rng(SEED + 999)
    sampled = rng.integers(
        0, len(deltas), size=(SESSION_BOOTSTRAP_REPETITIONS, len(deltas))
    )
    bootstrap_means = deltas[sampled].mean(axis=1)
    wilcoxon_result = wilcoxon(deltas, alternative="greater", method="exact")
    two_sided = wilcoxon(deltas, alternative="two-sided", method="exact")
    positive = int(np.sum(deltas > 0))
    sign = binomtest(positive, len(deltas), p=0.5, alternative="greater")
    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": "What is the GUI-selected deployment uplift from GRU memory READY versus ABSENT?",
        "sessions": len(rows),
        "mean_selection_test_r2": float(
            np.mean([row["selection_test_r2_mean"] for row in rows])
        ),
        "mean_absent_r2": float(np.mean([row["absent_r2"] for row in rows])),
        "mean_ready_gru_r2": float(np.mean([row["ready_gru_r2"] for row in rows])),
        "mean_ready_minus_absent_r2": float(deltas.mean()),
        "median_ready_minus_absent_r2": float(np.median(deltas)),
        "session_bootstrap_ci95": [
            float(np.percentile(bootstrap_means, 2.5)),
            float(np.percentile(bootstrap_means, 97.5)),
        ],
        "positive_sessions": positive,
        "individually_significant_sessions": int(
            sum(row["individual_ci_excludes_zero"] for row in rows)
        ),
        "exact_one_sided_wilcoxon_p": float(wilcoxon_result.pvalue),
        "exact_two_sided_wilcoxon_p": float(two_sided.pvalue),
        "exact_one_sided_sign_test_p": float(sign.pvalue),
        "interpretation": (
            "READY versus ABSENT is a paired deployment-preprocessing comparison. "
            "The 0.746 selection score remains a different chunked-inference metric."
        ),
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(RESULT_ROOT / "deployment_parity_ab.csv", rows)
    write_json(RESULT_ROOT / "summary.json", summary)
    return summary


def main() -> None:
    import torch

    args = parse_args()
    torch.set_num_threads(args.threads)
    device = phase12.select_device(args.device)
    results = []
    for index, session in enumerate(SESSIONS, start=1):
        print(f"[{index}/{len(SESSIONS)}] {session}", flush=True)
        result = process_session(
            session,
            device,
            args.threads,
            args.batch_size,
            args.overwrite,
            args.protocol_check_only,
        )
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
    if not args.protocol_check_only:
        print(json.dumps(aggregate(results), indent=2))


if __name__ == "__main__":
    main()
