#!/usr/bin/env python3
"""Phase 13 round 2: sweep causal prefix-calibration duration.

No labels, targets, future bins, model weights, or external-memory values are
used during calibration.  For fair duration comparisons, the primary metric
uses the same best-fold test bins that occur after the longest calibration in
the sweep.  A secondary metric reports every test bin available after each
duration and is explicitly treated as a changing-population diagnostic.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
RESULT_ROOT = HERE / "results" / "calibration_sweep"
DEFAULT_MINUTES = (
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
)
BIN_SECONDS = 0.04


def import_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = import_file("phase13_base", HERE / "run.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--minutes", nargs="+", type=float, default=list(DEFAULT_MINUTES)
    )
    parser.add_argument("--session", action="append", choices=BASE.SESSIONS)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--cubeai-minutes",
        nargs="*",
        type=float,
        default=[],
        help="Optional durations to confirm through generated X-CUBE-AI host graphs.",
    )
    parser.add_argument("--stedgeai", type=Path)
    parser.add_argument(
        "--output-name",
        default="primary",
        help="Use a named subdirectory for sensitivity runs; default writes primary outputs.",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path(tempfile.gettempdir()) / "phase13_cubeai_host",
    )
    return parser.parse_args()


def calibration_bins(minutes: float) -> int:
    bins = int(round(minutes * 60.0 / BIN_SECONDS))
    if bins < 50:
        raise ValueError(f"calibration duration {minutes} min is shorter than 50 bins")
    return bins


def causal_calibration(
    builder: Any, counts: np.ndarray, std_floor: np.ndarray, bins: int
) -> tuple[np.ndarray, dict[str, float]]:
    raw = np.asarray(counts, dtype=np.float32)
    features = np.concatenate(
        (raw, builder.causal_ewma(raw, builder.EWMA_ALPHA)), axis=0
    )
    if bins > features.shape[1]:
        raise ValueError(
            f"calibration requires {bins} bins; session has {features.shape[1]}"
        )
    calibration = features[:, :bins]
    sums = calibration.astype(np.float64).sum(axis=1)
    squares = np.square(calibration.astype(np.float64)).sum(axis=1)
    mean = (sums / bins).astype(np.float32)
    variance = squares / bins - mean.astype(np.float64) ** 2
    local_std = (np.sqrt(np.maximum(variance, 0.0)) + 1.0e-6).astype(np.float32)
    floor = np.asarray(std_floor, dtype=np.float32).reshape(192)
    effective_std = np.maximum(local_std, floor).astype(np.float32)
    normalized = ((features - mean[:, None]) / effective_std[:, None]).astype(
        np.float32
    )
    return normalized, {
        "mean_min": float(mean.min()),
        "mean_max": float(mean.max()),
        "effective_std_min": float(effective_std.min()),
        "effective_std_max": float(effective_std.max()),
    }


def select_tradeoff(duration_rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in duration_rows if row["minutes"] >= 1.0]
    candidates = sorted(candidates, key=lambda row: row["minutes"])
    baseline = min(candidates, key=lambda row: abs(row["minutes"] - 1.0))
    best = max(candidates, key=lambda row: row["common_mask_r2_mean"])
    total_gain = best["common_mask_r2_mean"] - baseline["common_mask_r2_mean"]
    if total_gain <= 0:
        gain95 = baseline
        threshold = baseline["common_mask_r2_mean"]
    else:
        threshold = baseline["common_mask_r2_mean"] + 0.95 * total_gain
        gain95 = next(
            row for row in candidates if row["common_mask_r2_mean"] >= threshold
        )
    plateau_threshold = 0.005
    slopes = [
        (right["common_mask_r2_mean"] - left["common_mask_r2_mean"])
        / (right["minutes"] - left["minutes"])
        for left, right in zip(candidates, candidates[1:], strict=False)
    ]
    plateau = best
    for index, candidate in enumerate(candidates[:-1]):
        if all(slope <= plateau_threshold for slope in slopes[index:]):
            plateau = candidate
            break
    return {
        "recommended_rule": (
            "earliest duration after which every observed marginal gain is at most "
            "0.005 R2 per additional minute"
        ),
        "baseline_minutes": baseline["minutes"],
        "baseline_r2_mean": baseline["common_mask_r2_mean"],
        "best_observed_minutes": best["minutes"],
        "best_observed_r2_mean": best["common_mask_r2_mean"],
        "total_observed_gain": total_gain,
        "recommended_minutes": plateau["minutes"],
        "recommended_r2_mean": plateau["common_mask_r2_mean"],
        "recommended_gain_fraction": (
            (plateau["common_mask_r2_mean"] - baseline["common_mask_r2_mean"])
            / total_gain
            if total_gain > 0
            else 0.0
        ),
        "post_recommendation_marginal_threshold_r2_per_minute": plateau_threshold,
        "gain95_rule": "earliest duration reaching 95% of observed gain over 1 minute",
        "gain95_threshold_r2_mean": threshold,
        "gain95_minutes": gain95["minutes"],
        "gain95_r2_mean": gain95["common_mask_r2_mean"],
    }


def main() -> None:
    args = parse_args()
    minutes = sorted(set(float(value) for value in args.minutes))
    if not minutes or any(value <= 0 for value in minutes):
        raise ValueError("--minutes must contain positive durations")
    cubeai_minutes = sorted(set(float(value) for value in args.cubeai_minutes))
    unknown = set(cubeai_minutes) - set(minutes)
    if unknown:
        raise ValueError(f"--cubeai-minutes must be in --minutes: {sorted(unknown)}")
    sessions = list(dict.fromkeys(args.session or BASE.SESSIONS))
    builder = import_file(
        "phase13_firmware_builder_sweep",
        BASE.FIRMWARE_ROOT / "tools" / "build_gru_hidden_bcimem.py",
    )
    import torch

    longest_bins = calibration_bins(max(minutes))
    stedgeai = None
    gru = None
    encoders: dict[str, Any] = {}
    if cubeai_minutes:
        stedgeai = BASE.discover_stedgeai(args.stedgeai)
        gru = BASE.prepare_gru_host(builder, stedgeai, args.work_root, False)

    session_rows: list[dict[str, Any]] = []
    for session in sessions:
        print(f"calibration sweep: {session}", flush=True)
        checkpoint_path = (
            BASE.INDY_ROOT / "models" / "midsize" / session / "checkpoint.pt"
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        with np.load(
            BASE.GUI_ROOT / "data" / "ai_device_sessions" / f"{session}.npz",
            allow_pickle=False,
        ) as archive:
            counts_all = np.asarray(archive["counts"])
            target_all = np.asarray(archive["velocity"], dtype=np.float32)
        with np.load(
            BASE.GUI_ROOT
            / "data"
            / "ai_device_sessions"
            / f"{session}_best_fold_test_bins.npz",
            allow_pickle=False,
        ) as archive:
            test_bins = np.asarray(archive["bin_indices"], dtype=np.int64)
        common_bins = test_bins[test_bins >= longest_bins - 1]
        if len(common_bins) < 300:
            raise ValueError(
                f"{session}: only {len(common_bins)} common test bins remain after "
                f"{max(minutes)} minutes"
            )
        channels = np.asarray(checkpoint["selected_channel_indices"], dtype=np.int64)
        counts = counts_all[channels]
        if cubeai_minutes:
            encoders[session], _ = BASE.prepare_encoder_host(
                builder, session, stedgeai, args.work_root, False
            )
        for duration in minutes:
            duration_bins = calibration_bins(duration)
            available_bins = test_bins[test_bins >= duration_bins - 1]
            normalized, calibration = causal_calibration(
                builder, counts, checkpoint["feature_std_floor"], duration_bins
            )
            common_prediction, _ = BASE.fp32_inference(
                checkpoint_path, normalized, common_bins, args.batch_size
            )
            available_prediction, _ = BASE.fp32_inference(
                checkpoint_path, normalized, available_bins, args.batch_size
            )
            common_metric = builder.metrics(target_all[common_bins], common_prediction)
            available_metric = builder.metrics(
                target_all[available_bins], available_prediction
            )
            cubeai_r2 = None
            if duration in cubeai_minutes:
                normalized_prediction, _ = builder.cubeai_inference(
                    normalized=normalized,
                    bins=common_bins,
                    encoder=encoders[session],
                    gru=gru,
                    gru_weights=(
                        BASE.ARCHIVE
                        / "models"
                        / "midsize"
                        / session
                        / "cubeai_int8"
                        / "gru_head.weights.bin"
                    ),
                    scratch=args.work_root / "sweep_scratch" / session / str(duration),
                    batch_size=args.batch_size,
                )
                target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
                target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)
                cubeai_prediction = normalized_prediction * target_std + target_mean
                cubeai_r2 = builder.metrics(target_all[common_bins], cubeai_prediction)[
                    "r2_mean"
                ]
            row = {
                "session": session,
                "subject": session.split("_", 1)[0],
                "selected_fold": int(checkpoint["fold"]),
                "minutes": duration,
                "calibration_bins": duration_bins,
                "common_mask_bins": int(len(common_bins)),
                "available_mask_bins": int(len(available_bins)),
                "common_mask_fp32_r2_mean": float(common_metric["r2_mean"]),
                "available_mask_fp32_r2_mean": float(available_metric["r2_mean"]),
                "common_mask_cubeai_r2_mean": (
                    float(cubeai_r2) if cubeai_r2 is not None else None
                ),
                **calibration,
            }
            session_rows.append(row)
            print(
                f"  {duration:>4g} min: common R2={common_metric['r2_mean']:.6f}; "
                f"available R2={available_metric['r2_mean']:.6f}",
                flush=True,
            )

    one_minute = {
        row["session"]: row for row in session_rows if np.isclose(row["minutes"], 1.0)
    }
    duration_rows: list[dict[str, Any]] = []
    for duration in minutes:
        selected = [row for row in session_rows if row["minutes"] == duration]
        common = np.asarray(
            [row["common_mask_fp32_r2_mean"] for row in selected], dtype=np.float64
        )
        available = np.asarray(
            [row["available_mask_fp32_r2_mean"] for row in selected], dtype=np.float64
        )
        delta = np.asarray(
            [
                row["common_mask_fp32_r2_mean"]
                - one_minute[row["session"]]["common_mask_fp32_r2_mean"]
                for row in selected
            ],
            dtype=np.float64,
        )
        cubeai_values = [
            row["common_mask_cubeai_r2_mean"]
            for row in selected
            if row["common_mask_cubeai_r2_mean"] is not None
        ]
        duration_rows.append(
            {
                "minutes": duration,
                "sessions": len(selected),
                "common_mask_r2_mean": float(common.mean()),
                "common_mask_r2_std_sample": float(common.std(ddof=1)),
                "available_mask_r2_mean": float(available.mean()),
                "common_mask_delta_vs_1min_mean": float(delta.mean()),
                "sessions_improved_vs_1min": int((delta > 0).sum()),
                "common_mask_cubeai_r2_mean": (
                    float(np.mean(cubeai_values)) if cubeai_values else None
                ),
                "minimum_common_bins_per_session": min(
                    row["common_mask_bins"] for row in selected
                ),
                "minimum_available_bins_per_session": min(
                    row["available_mask_bins"] for row in selected
                ),
            }
        )
    if not any(np.isclose(row["minutes"], 1.0) for row in duration_rows):
        raise ValueError("The default tradeoff comparison requires a 1-minute point")
    for index, row in enumerate(duration_rows):
        if index == 0:
            row["marginal_r2_gain_per_minute"] = None
            continue
        prior = duration_rows[index - 1]
        row["marginal_r2_gain_per_minute"] = float(
            (row["common_mask_r2_mean"] - prior["common_mask_r2_mean"])
            / (row["minutes"] - prior["minutes"])
        )
    tradeoff = select_tradeoff(duration_rows)
    output = {
        "phase": "phase13_round2_calibration_duration_sweep",
        "status": "complete",
        "model_tier": "midsize bank ABSENT",
        "sessions": sessions,
        "durations_minutes": minutes,
        "primary_metric": (
            "session-macro mean R2 on each session's same best-fold test bins "
            f"occurring after the longest calibration ({max(minutes)} minutes)"
        ),
        "secondary_metric": (
            "session-macro mean R2 on all best-fold test bins available after each "
            "duration; population changes and this metric is not used to select duration"
        ),
        "calibration_contract": (
            "causal session prefix; neural counts only; no velocity labels, future bins, "
            "weight updates, or external memory"
        ),
        "numeric_path": "FP32 PyTorch sweep with optional X-CUBE-AI confirmation",
        "duration_summary": duration_rows,
        "session_results": session_rows,
        "tradeoff": tradeoff,
        "limitations": [
            "Six sessions are a small paired sample.",
            "The 10-minute common mask leaves relatively few bins for the shortest session.",
            "This sweep changes calibration duration only; firmware remains hard-coded to 1 minute.",
            "The chosen fold remains the best-test-fold demonstration and retains selection bias.",
        ],
        "stedgeai": str(stedgeai) if stedgeai is not None else None,
    }
    output_root = (
        RESULT_ROOT if args.output_name == "primary" else RESULT_ROOT / args.output_name
    )
    output_root.mkdir(parents=True, exist_ok=True)
    BASE.write_json(output_root / "calibration_sweep.json", output)
    BASE.write_csv(output_root / "calibration_sweep_by_session.csv", session_rows)
    BASE.write_csv(output_root / "calibration_sweep_summary.csv", duration_rows)
    print(json.dumps(tradeoff, indent=2))
    print(f"Calibration sweep complete: {output_root}")


if __name__ == "__main__":
    main()
