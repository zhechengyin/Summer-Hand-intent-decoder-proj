#!/usr/bin/env python
"""One-shot, inference-only evaluation on the four locked January sessions.

This script evaluates the already-trained seed-43 Phase-1d checkpoint selected
before January was opened. It never constructs an optimizer, calls backward,
updates model weights, selects a checkpoint, or changes a hyperparameter.

The only test-session calibration is the frozen causal protocol: each session's
first 60 seconds of neural counts fit feature mean/std, those bins are discarded,
and every scored prediction uses only past and present input. Target labels are
used only after inference to calculate the final locked-test metrics.

Run ``--validate-only`` first to verify the frozen protocol without loading any
January arrays. Run ``--confirm-locked-test`` once when ready to open the test.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.intent_decoder.data.indy import (
    apply_feature_stats,
    fit_feature_stats,
    load_model_data,
    load_session_manifest,
    window_arrays,
)
from src.intent_decoder.features.causal import multiscale_counts
from src.intent_decoder.model.tcn_gru import build_net, corr, r2

BIN_S = 0.040
WINDOW_BINS = 50
OBSERVATION_SECONDS = 60
OBSERVATION_BINS = int(OBSERVATION_SECONDS / BIN_S)
ALPHAS = (1.0, 0.1)
AXES = np.array([0, 1])
BATCH_SIZE = 32
EXPECTED_SPLIT_COUNTS = {"train": 29, "validation": 4, "test": 4}

FROZEN_SEED = 43
FROZEN_LEARNING_RATE = 9e-4
FROZEN_DROPOUT = 0.025
FROZEN_WEIGHT_DECAY = 0.060
FROZEN_EPOCH_BUDGET = 20
FROZEN_CHECKPOINT_EPOCH = 7
FROZEN_SAMPLER = "session_balanced"

CHECKPOINT_PATH = ROOT / "models" / "indy_32ch" / "checkpoint.pt"
PHASE1E_METRICS_PATH = (
    ROOT / "history" / "phase1" / "results" / "phase1e_seed_crosscheck.json"
)
OUTPUT_PATH = (
    ROOT / "results" / "metrics" / "indy_32ch_locked_january_seed43.json"
)
FIGURE_PATH = (
    ROOT / "results" / "figures" / "indy_32ch_locked_january_seed43.png"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def report_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def choose_device(requested: str):
    """Choose a deterministic inference device; Apple MPS is excluded."""
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the frozen checkpoint/protocol without loading January arrays.",
    )
    mode.add_argument(
        "--confirm-locked-test",
        action="store_true",
        help="Acknowledge that this command opens and scores the locked January test.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="CPU is the frozen local path; auto selects CUDA or CPU, never MPS.",
    )
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def _close(actual, expected: float) -> bool:
    return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)


def validate_frozen_protocol(device="cpu") -> dict:
    """Validate all pre-test decisions without loading a January data array."""
    import torch

    checkpoint_path = CHECKPOINT_PATH.resolve()
    metrics_path = PHASE1E_METRICS_PATH.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Frozen checkpoint is missing: {checkpoint_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Phase-1e metrics are missing: {metrics_path}")

    manifest = load_session_manifest()
    split_sessions = {
        name: list(manifest["chronological_split"][name])
        for name in ("train", "validation", "test")
    }
    counts = {name: len(sessions) for name, sessions in split_sessions.items()}
    if counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"Expected split {EXPECTED_SPLIT_COUNTS}, found {counts}")
    if any(not session.startswith("indy_201701") for session in split_sessions["test"]):
        raise ValueError("Locked test must contain only January 2017 Indy sessions")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    expected_fields = {
        "seed": (checkpoint.get("seed"), FROZEN_SEED),
        "learning_rate": (config.get("lr"), FROZEN_LEARNING_RATE),
        "dropout": (config.get("dropout"), FROZEN_DROPOUT),
        "weight_decay": (config.get("wd"), FROZEN_WEIGHT_DECAY),
        "epoch_budget": (config.get("epochs"), FROZEN_EPOCH_BUDGET),
        "checkpoint_epoch": (
            checkpoint.get("checkpoint_epoch"),
            FROZEN_CHECKPOINT_EPOCH,
        ),
        "observation_seconds": (
            checkpoint.get("observation_seconds"),
            OBSERVATION_SECONDS,
        ),
    }
    mismatches = {}
    for name, (actual, expected) in expected_fields.items():
        matches = _close(actual, expected) if isinstance(expected, float) else actual == expected
        if not matches:
            mismatches[name] = {"actual": actual, "expected": expected}
    if mismatches:
        raise ValueError(f"Frozen checkpoint mismatch: {mismatches}")
    if checkpoint.get("test_policy") != "locked_not_loaded":
        raise ValueError("Checkpoint does not carry the locked-test policy")
    if config.get("bidir") is not False:
        raise ValueError("Frozen checkpoint is not strictly unidirectional")
    if list(checkpoint.get("train_sessions", [])) != split_sessions["train"]:
        raise ValueError("Checkpoint training sessions do not match the manifest")
    if list(checkpoint.get("validation_sessions", [])) != split_sessions["validation"]:
        raise ValueError("Checkpoint validation sessions do not match the manifest")

    phase1e = json.loads(metrics_path.read_text(encoding="utf-8"))
    decision = phase1e.get("aggregate", {}).get("stop_rule_decision", {})
    if decision.get("status") != "selected_by_preregistered_rule":
        raise ValueError("Phase-1e did not complete the preregistered selection rule")
    if not _close(decision.get("recommended_weight_decay"), FROZEN_WEIGHT_DECAY):
        raise ValueError("Phase-1e did not select the frozen weight decay")
    if phase1e.get("test_policy") != "January test is locked and was not loaded.":
        raise ValueError("Phase-1e metrics do not preserve the locked-test policy")

    candidate_rows = [
        row
        for row in phase1e.get("all_seed_rows", [])
        if _close(row.get("weight_decay"), FROZEN_WEIGHT_DECAY)
    ]
    if {int(row["seed"]) for row in candidate_rows} != {42, 43, 44, 45, 46}:
        raise ValueError("Phase-1e does not contain all five frozen-candidate seeds")
    best_loss = min(candidate_rows, key=lambda row: row["validation_loss"])
    best_r2 = max(candidate_rows, key=lambda row: row["validation_r2_mean"])
    if int(best_loss["seed"]) != FROZEN_SEED or int(best_r2["seed"]) != FROZEN_SEED:
        raise ValueError("Seed 43 is not best by both frozen validation metrics")

    channels = np.asarray(checkpoint.get("channels"), dtype=np.int64)
    target_mean = np.asarray(checkpoint.get("target_mean"), dtype=np.float32)
    target_std = np.asarray(checkpoint.get("target_std"), dtype=np.float32)
    feature_std_floor = np.asarray(
        checkpoint.get("feature_std_floor"), dtype=np.float32
    )[:, None]
    if channels.shape != (32,) or len(np.unique(channels)) != 32:
        raise ValueError("Frozen checkpoint must contain 32 unique raw channels")
    if target_mean.shape != (2,) or target_std.shape != (2,):
        raise ValueError("Frozen target normalization must have exactly two axes")
    if np.any(target_std <= 0):
        raise ValueError("Frozen target std must be positive")
    if feature_std_floor.shape != (len(channels) * len(ALPHAS), 1):
        raise ValueError("Frozen feature std floor has the wrong shape")

    net = build_net(config, feature_std_floor.shape[0]).to(device)
    net.load_state_dict(checkpoint["model_state"], strict=True)
    net.eval()
    n_parameters = sum(parameter.numel() for parameter in net.parameters())
    if n_parameters != 78_786:
        raise ValueError(f"Expected 78,786 parameters, found {n_parameters:,}")

    return {
        "checkpoint": checkpoint,
        "config": config,
        "net": net,
        "channels": channels,
        "target_mean": target_mean,
        "target_std": target_std,
        "feature_std_floor": feature_std_floor,
        "split_sessions": split_sessions,
        "n_parameters": n_parameters,
        "phase1e_decision": decision,
        "selection_row": best_loss,
    }


def prepare_locked_session(
    session: str,
    channels: np.ndarray,
    feature_std_floor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the frozen causal 60-second calibration and return scored windows."""
    counts, velocity = load_model_data(session)
    features = multiscale_counts(counts[channels], ALPHAS)
    if features.shape[1] < OBSERVATION_BINS + WINDOW_BINS:
        raise ValueError(f"{session} is too short for the frozen protocol")
    mean, local_std = fit_feature_stats(features, observation_bins=OBSERVATION_BINS)
    safe_std = np.maximum(local_std, feature_std_floor)
    normalized = apply_feature_stats(features, (mean, safe_std))
    windows = window_arrays(
        normalized,
        velocity,
        AXES,
        window_bins=WINDOW_BINS,
        start_bin=OBSERVATION_BINS,
    )
    if not windows:
        raise ValueError(f"{session} has no windows after the observation prefix")
    x = np.stack([window["e"] for window in windows]).astype(np.float32)
    y = np.stack([window["vel"] for window in windows]).astype(np.float32)
    return x, y


def predict_only(net, x: np.ndarray, batch_size: int, device) -> np.ndarray:
    """Run forward inference without receiving or inspecting target labels."""
    import torch

    predictions = []
    net.eval()
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            predictions.append(net(batch).cpu().numpy())
    return np.concatenate(predictions, axis=0).astype(np.float32)


def score_predictions(
    prediction_norm: np.ndarray,
    target: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> dict:
    """Use locked labels only here, after predictions have already been made."""
    target_norm = ((target - target_mean) / target_std).astype(np.float32)
    prediction = prediction_norm * target_std + target_mean
    flat_target = target.reshape(-1, 2)
    flat_prediction = prediction.reshape(-1, 2)
    axis_r2 = r2(flat_target, flat_prediction).astype(np.float64)
    axis_corr = corr(flat_target, flat_prediction).astype(np.float64)
    return {
        "loss": float(np.mean((prediction_norm - target_norm) ** 2)),
        "r2_x": float(axis_r2[0]),
        "r2_y": float(axis_r2[1]),
        "r2_mean": float(axis_r2.mean()),
        "pearson_r_x": float(axis_corr[0]),
        "pearson_r_y": float(axis_corr[1]),
        "pearson_r_mean": float(axis_corr.mean()),
        "prediction": prediction,
        "prediction_norm": prediction_norm,
        "target": target,
        "target_norm": target_norm,
    }


def aggregate_results(by_session: dict[str, dict]) -> dict:
    """Aggregate the four predeclared sessions without selecting or excluding one."""
    prediction = np.concatenate(
        [item["prediction"].reshape(-1, 2) for item in by_session.values()], axis=0
    )
    target = np.concatenate(
        [item["target"].reshape(-1, 2) for item in by_session.values()], axis=0
    )
    prediction_norm = np.concatenate(
        [item["prediction_norm"].reshape(-1, 2) for item in by_session.values()],
        axis=0,
    )
    target_norm = np.concatenate(
        [item["target_norm"].reshape(-1, 2) for item in by_session.values()], axis=0
    )
    pooled_r2 = r2(target, prediction).astype(np.float64)
    pooled_corr = corr(target, prediction).astype(np.float64)
    session_metrics = [item["metrics"] for item in by_session.values()]
    worst_name, worst = min(
        ((name, item["metrics"]) for name, item in by_session.items()),
        key=lambda pair: pair[1]["r2_mean"],
    )
    return {
        "pooled": {
            "loss": float(np.mean((prediction_norm - target_norm) ** 2)),
            "r2_x": float(pooled_r2[0]),
            "r2_y": float(pooled_r2[1]),
            "r2_mean": float(pooled_r2.mean()),
            "pearson_r_x": float(pooled_corr[0]),
            "pearson_r_y": float(pooled_corr[1]),
            "pearson_r_mean": float(pooled_corr.mean()),
        },
        "session_macro": {
            key: float(np.mean([metrics[key] for metrics in session_metrics]))
            for key in (
                "loss",
                "r2_x",
                "r2_y",
                "r2_mean",
                "pearson_r_x",
                "pearson_r_y",
                "pearson_r_mean",
            )
        },
        "worst_session": {
            "session": worst_name,
            "r2_mean": float(worst["r2_mean"]),
            "loss": float(worst["loss"]),
        },
    }


def plot_results(payload: dict) -> None:
    import os

    cache = ROOT / "results" / "large" / ".matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_session = payload["results"]["by_session"]
    sessions = list(by_session)
    labels = [session.replace("indy_2017", "") for session in sessions]
    x = np.arange(len(sessions))
    r2_x = [by_session[name]["r2_x"] for name in sessions]
    r2_y = [by_session[name]["r2_y"] for name in sessions]
    r2_mean = [by_session[name]["r2_mean"] for name in sessions]
    losses = [by_session[name]["loss"] for name in sessions]

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=180)
    width = 0.25
    axes[0].bar(x - width, r2_x, width, label="R2 x", color="#4477AA")
    axes[0].bar(x, r2_y, width, label="R2 y", color="#EE6677")
    axes[0].bar(x + width, r2_mean, width, label="Mean R2", color="#228833")
    axes[0].axhline(0.0, color="#333333", linewidth=0.8)
    axes[0].set_title("Locked January R2 by session")
    axes[0].set_ylabel("R2")
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, losses, color="#CCBB44", edgecolor="#665D22")
    axes[1].set_title("Locked January normalized MSE by session")
    axes[1].set_ylabel("Normalized MSE")
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    for index, value in enumerate(losses):
        axes[1].text(index, value, f"{value:.3f}", ha="center", va="bottom")

    figure.suptitle("Indy 32-channel frozen seed-43 locked-test evaluation")
    figure.text(
        0.5,
        0.01,
        "Inference only; January labels used for scoring only; no parameter update or selection.",
        ha="center",
        color="#5B6470",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.close(figure)


def run_locked_test(context: dict, device) -> dict:
    """Open all four locked sessions exactly once and run pure inference."""
    test_sessions = context["split_sessions"]["test"]
    by_session_internal = {}
    for session in test_sessions:
        x, target = prepare_locked_session(
            session,
            context["channels"],
            context["feature_std_floor"],
        )
        prediction_norm = predict_only(context["net"], x, BATCH_SIZE, device)
        scored = score_predictions(
            prediction_norm,
            target,
            context["target_mean"],
            context["target_std"],
        )
        metrics = {
            key: value
            for key, value in scored.items()
            if key not in {"prediction", "prediction_norm", "target", "target_norm"}
        }
        metrics.update(
            {
                "windows": int(len(x)),
                "scored_bins": int(len(x) * WINDOW_BINS),
                "scored_seconds": float(len(x) * WINDOW_BINS * BIN_S),
            }
        )
        by_session_internal[session] = {"metrics": metrics, **scored}
        print(
            f"{session} | windows={len(x)} | loss={metrics['loss']:.5f} | "
            f"R2 x/y/mean={metrics['r2_x']:+.4f}/"
            f"{metrics['r2_y']:+.4f}/{metrics['r2_mean']:+.4f} | "
            f"r={metrics['pearson_r_mean']:+.4f}",
            flush=True,
        )

    aggregate = aggregate_results(by_session_internal)
    validation_reference = {
        "loss": float(context["selection_row"]["validation_loss"]),
        "pooled_r2": float(context["selection_row"]["validation_r2_mean"]),
        "macro_r2": float(context["selection_row"]["validation_macro_r2_mean"]),
        "worst_session_r2": float(
            context["selection_row"]["validation_worst_session_r2_mean"]
        ),
    }
    comparison_to_validation = {
        "pooled_loss_delta_test_minus_validation": float(
            aggregate["pooled"]["loss"] - validation_reference["loss"]
        ),
        "pooled_r2_delta_test_minus_validation": float(
            aggregate["pooled"]["r2_mean"] - validation_reference["pooled_r2"]
        ),
        "macro_r2_delta_test_minus_validation": float(
            aggregate["session_macro"]["r2_mean"]
            - validation_reference["macro_r2"]
        ),
        "worst_r2_delta_test_minus_validation": float(
            aggregate["worst_session"]["r2_mean"]
            - validation_reference["worst_session_r2"]
        ),
    }
    by_session = {
        name: item["metrics"] for name, item in by_session_internal.items()
    }
    payload = {
        "purpose": "indy_32ch_locked_january_seed43_inference_only",
        "generated_at_utc": utc_now(),
        "locked_test_opened": True,
        "model_selection_complete_before_test": True,
        "frozen_checkpoint": report_path(CHECKPOINT_PATH),
        "frozen_protocol": {
            "seed": FROZEN_SEED,
            "learning_rate": FROZEN_LEARNING_RATE,
            "dropout": FROZEN_DROPOUT,
            "weight_decay": FROZEN_WEIGHT_DECAY,
            "epoch_budget": FROZEN_EPOCH_BUDGET,
            "checkpoint_epoch": FROZEN_CHECKPOINT_EPOCH,
            "sampler": FROZEN_SAMPLER,
            "batch_size_inference_only": BATCH_SIZE,
            "observation_seconds": OBSERVATION_SECONDS,
            "features": ["raw_counts", "causal_ewma_0.1"],
            "channels": context["channels"].tolist(),
            "n_parameters": context["n_parameters"],
        },
        "selection_evidence": {
            "phase1e_metrics": report_path(PHASE1E_METRICS_PATH),
            "rule_status": context["phase1e_decision"]["status"],
            "selected_weight_decay": context["phase1e_decision"][
                "recommended_weight_decay"
            ],
            "seed_rule": "best validation loss and pooled R2 within frozen WD=0.060",
            "seed43_validation_loss": context["selection_row"]["validation_loss"],
            "seed43_validation_r2": context["selection_row"]["validation_r2_mean"],
            "seed43_validation_macro_r2": context["selection_row"][
                "validation_macro_r2_mean"
            ],
            "seed43_validation_worst_session_r2": context["selection_row"][
                "validation_worst_session_r2_mean"
            ],
        },
        "test_sessions": test_sessions,
        "safeguards": {
            "optimizer_created": False,
            "backward_called": False,
            "model_weights_updated": False,
            "test_used_for_checkpoint_selection": False,
            "test_used_for_seed_selection": False,
            "test_used_for_hyperparameter_selection": False,
            "test_labels_used_only_after_inference": True,
            "test_observation_prefix": (
                "first 60 seconds of neural counts fit session-local feature "
                "normalization; prefix outputs and labels are discarded"
            ),
        },
        "results": {
            "by_session": by_session,
            **aggregate,
            "frozen_validation_reference": validation_reference,
            "comparison_to_frozen_validation": comparison_to_validation,
        },
        "artifacts": {
            "metrics": report_path(OUTPUT_PATH),
            "figure": report_path(FIGURE_PATH),
            "new_checkpoint_written": False,
        },
    }
    return payload


def main() -> None:
    import torch

    args = parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    context = validate_frozen_protocol(device)

    print("=== Indy 32-channel frozen locked-test protocol ===")
    print(
        f"checkpoint: seed={FROZEN_SEED} | epoch={FROZEN_CHECKPOINT_EPOCH} | "
        f"lr={FROZEN_LEARNING_RATE:.1e} | dropout={FROZEN_DROPOUT:.3f} | "
        f"wd={FROZEN_WEIGHT_DECAY:.3f}"
    )
    print(
        f"sessions: train=29 used for weights | validation=4 used for selection | "
        f"test=4 January LOCKED"
    )
    print(
        "policy: inference only; no optimizer/backward/update; "
        "no test-driven selection or tuning"
    )
    print(f"device={device} | parameters={context['n_parameters']:,}")

    if args.validate_only:
        print("validation-only complete: January arrays were not loaded")
        print("run the locked test only when ready:")
        print(
            "python history/locked_test/evaluate_locked_january.py "
            "--confirm-locked-test"
        )
        return

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            "Locked-test metrics already exist. Refusing to rerun or overwrite: "
            f"{OUTPUT_PATH}"
        )

    print("\nOPENING LOCKED JANUARY TEST: all four sessions will be reported.\n")
    payload = run_locked_test(context, device)
    plot_results(payload)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2), encoding="utf-8"
    )
    temporary.replace(OUTPUT_PATH)

    pooled = payload["results"]["pooled"]
    macro = payload["results"]["session_macro"]
    worst = payload["results"]["worst_session"]
    delta = payload["results"]["comparison_to_frozen_validation"]
    print("\n=== FINAL LOCKED JANUARY TEST ===")
    print(
        f"pooled | loss={pooled['loss']:.5f} | "
        f"R2 x/y/mean={pooled['r2_x']:+.4f}/"
        f"{pooled['r2_y']:+.4f}/{pooled['r2_mean']:+.4f} | "
        f"r={pooled['pearson_r_mean']:+.4f}"
    )
    print(
        f"session macro | loss={macro['loss']:.5f} | "
        f"R2={macro['r2_mean']:+.4f} | r={macro['pearson_r_mean']:+.4f}"
    )
    print(
        f"worst session | {worst['session']} | "
        f"loss={worst['loss']:.5f} | R2={worst['r2_mean']:+.4f}"
    )
    print(
        "test minus frozen validation | "
        f"loss={delta['pooled_loss_delta_test_minus_validation']:+.5f} | "
        f"pooled R2={delta['pooled_r2_delta_test_minus_validation']:+.4f} | "
        f"macro R2={delta['macro_r2_delta_test_minus_validation']:+.4f} | "
        f"worst R2={delta['worst_r2_delta_test_minus_validation']:+.4f}"
    )
    print("weights updated: NO | checkpoint selected from test: NO")
    print(f"metrics: {OUTPUT_PATH}")
    print(f"figure: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
