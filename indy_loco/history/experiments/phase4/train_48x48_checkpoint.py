#!/usr/bin/env python3
"""Train the confirmed 48/48 firmware candidate and save its checkpoint.

This script performs one fixed build, not another sweep:

- architecture: 48 TCN filters, 48 GRU hidden units, four causal blocks;
- optimization: seed 43, full 20-epoch cosine trajectory;
- data: 29 training sessions update weights;
- validation: four December sessions are inference-only and select the
  minimum-validation-loss checkpoint, matching the 64/64 protocol;
- test: January is never opened;
- output: models/indy_32ch/48x48checkpoint.pt.

The protected 64x64checkpoint.pt is checksum-verified and never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.indy_32ch.features import multiscale_counts
from models.indy_32ch.input_pipeline import (
    apply_feature_stats,
    fit_feature_stats,
    load_model_data,
    load_session_manifest,
    processed_session_path,
    window_arrays,
)
from models.indy_32ch.model import build_net, causal_config, r2
from models.indy_32ch.sampling import draw_session_balanced_indices


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

MODEL_CONFIG_PATH = ROOT / "configs" / "indy_32ch.yaml"
BASELINE_PATH = ROOT / "models" / "indy_32ch" / "64x64checkpoint.pt"
OUTPUT_PATH = ROOT / "models" / "indy_32ch" / "48x48checkpoint.pt"
PHASE4B_METRICS_PATH = (
    ROOT
    / "results"
    / "indy"
    / "phase4b_five_seed_confirmation"
    / "phase4b_five_seed_confirmation_metrics.json"
)
PHASE4B_REFERENCE_PATH = (
    ROOT
    / "results"
    / "indy"
    / "phase4b_five_seed_confirmation"
    / ".cache"
    / "progress"
    / "candidate_48x48"
    / "seed_43"
    / "held_2016-12.json"
)
BUILD_METRICS_PATH = (
    ROOT
    / "results"
    / "indy"
    / "phase4b_five_seed_confirmation"
    / "48x48checkpoint_build_metrics.json"
)

EXPECTED_BASELINE_SHA256 = (
    "2ee52c426ee43ba88cebe7c85dd8392f40f9e75748abe9bbf4e94093556363a5"
)
EXPECTED_PARAMETERS = 45_266
EXPECTED_SPLITS = {"train": 29, "validation": 4, "test": 4}

BIN_SECONDS = 0.040
OBSERVATION_SECONDS = 60
OBSERVATION_BINS = int(OBSERVATION_SECONDS / BIN_SECONDS)
WINDOW_BINS = 50
ALPHAS = (1.0, 0.1)
TARGET_AXES = (0, 1)
STD_FLOOR_PERCENTILE = 10.0

SEED = 43
TRAIN_EPOCHS = 20
SCHEDULER_EPOCH_BUDGET = 20
LEARNING_RATE = 9e-4
WEIGHT_DECAY = 0.060
DROPOUT = 0.025
BATCH_SIZE = 32
GRADIENT_CLIP = 1.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint_atomic(payload: dict, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def choose_device(requested: str):
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def configure_determinism(device, threads: int) -> None:
    import torch

    torch.set_num_threads(threads)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(SEED)
    if device.type == "mps":
        print(
            "warning: MPS may not reproduce the CPU Phase-4b values exactly",
            flush=True,
        )


def candidate_config() -> dict:
    config = causal_config(n_out=2)
    config.update(
        {
            "F": 48,
            "H": 48,
            "L": 1,
            "dils": [1, 2, 4, 8],
            "dropout": DROPOUT,
            "lr": LEARNING_RATE,
            "wd": WEIGHT_DECAY,
            "epochs": SCHEDULER_EPOCH_BUDGET,
            "bs": BATCH_SIZE,
            "gradient_clip": GRADIENT_CLIP,
        }
    )
    return config


def validate_protocol(*, require_output_absent: bool) -> dict:
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Missing protected baseline: {BASELINE_PATH}")
    baseline_hash = sha256_file(BASELINE_PATH)
    if baseline_hash != EXPECTED_BASELINE_SHA256:
        raise ValueError(
            "Protected 64/64 checkpoint changed: "
            f"expected {EXPECTED_BASELINE_SHA256}, found {baseline_hash}"
        )
    if require_output_absent and OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing candidate: {OUTPUT_PATH}"
        )

    manifest = load_session_manifest()
    split = manifest["chronological_split"]
    split_counts = {name: len(split[name]) for name in EXPECTED_SPLITS}
    if split_counts != EXPECTED_SPLITS:
        raise ValueError(
            f"Expected chronological split {EXPECTED_SPLITS}, found {split_counts}"
        )
    development_names = list(split["train"]) + list(split["validation"])
    if any(name.startswith("indy_201701") for name in development_names):
        raise ValueError("January leaked into train or validation")
    for name in development_names:
        if not processed_session_path(name).exists():
            raise FileNotFoundError(f"Missing processed session: {name}")

    model_yaml = yaml.safe_load(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    channels = np.asarray(
        model_yaml["input"]["selected_zero_based"],
        dtype=np.int64,
    )
    if channels.shape != (32,) or len(np.unique(channels)) != 32:
        raise ValueError("Expected exactly 32 unique frozen channels")

    phase4b = json.loads(PHASE4B_METRICS_PATH.read_text(encoding="utf-8"))
    aggregate = phase4b["aggregate"]
    if not aggregate["noninferiority"]["all_checks_passed"]:
        raise ValueError("Phase 4b did not pass every non-inferiority check")
    if (
        aggregate["recommendation"]["architecture_name"]
        != "candidate_48x48"
    ):
        raise ValueError("Phase 4b did not nominate candidate_48x48")
    if phase4b["data_policy"]["january_loaded"]:
        raise ValueError("Phase 4b metadata reports January access")

    reference = json.loads(PHASE4B_REFERENCE_PATH.read_text(encoding="utf-8"))
    expected_reference = {
        "architecture_name": "candidate_48x48",
        "seed": SEED,
        "held_month": "2016-12",
        "complete": True,
        "january_loaded": False,
        "checkpoint_saved": False,
    }
    mismatches = {
        key: {"expected": value, "actual": reference.get(key)}
        for key, value in expected_reference.items()
        if reference.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Phase 4b reference mismatch: {mismatches}")

    return {
        "manifest": manifest,
        "channels": channels,
        "phase4b": phase4b,
        "reference": reference,
        "baseline_sha256": baseline_hash,
    }


def fit_feature_std_floor(
    loaded: dict[str, tuple[np.ndarray, np.ndarray]],
    channels: np.ndarray,
) -> np.ndarray:
    session_stds = []
    for counts, _ in loaded.values():
        features = multiscale_counts(counts[channels], ALPHAS)
        _, std = fit_feature_stats(
            features,
            observation_bins=OBSERVATION_BINS,
        )
        session_stds.append(std[:, 0])
    scales = np.stack(session_stds)
    floors = np.empty(scales.shape[1], dtype=np.float32)
    for feature in range(scales.shape[1]):
        valid = scales[:, feature][scales[:, feature] > 1e-4]
        if valid.size == 0:
            raise ValueError(f"Feature {feature} is silent in every train prefix")
        floors[feature] = np.percentile(valid, STD_FLOOR_PERCENTILE)
    return floors[:, None]


def prepare_session(
    data: tuple[np.ndarray, np.ndarray],
    channels: np.ndarray,
    feature_std_floor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    counts, velocity = data
    features = multiscale_counts(counts[channels], ALPHAS)
    if features.shape[1] < OBSERVATION_BINS + WINDOW_BINS:
        raise ValueError("Session is too short for warm-up plus one window")
    mean, local_std = fit_feature_stats(
        features,
        observation_bins=OBSERVATION_BINS,
    )
    normalized = apply_feature_stats(
        features,
        (mean, np.maximum(local_std, feature_std_floor)),
    )
    windows = window_arrays(
        normalized,
        velocity,
        TARGET_AXES,
        window_bins=WINDOW_BINS,
        start_bin=OBSERVATION_BINS,
    )
    return (
        np.stack([window["e"] for window in windows]).astype(np.float32),
        np.stack([window["vel"] for window in windows]).astype(np.float32),
    )


def predict(net, x: np.ndarray, device) -> np.ndarray:
    import torch

    predictions = []
    net.eval()
    with torch.inference_mode():
        for start in range(0, len(x), BATCH_SIZE):
            batch = torch.from_numpy(x[start : start + BATCH_SIZE]).to(device)
            predictions.append(net(batch).cpu().numpy())
    return np.concatenate(predictions, axis=0).astype(np.float32)


def evaluate(
    net,
    x: np.ndarray,
    y: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device,
) -> dict:
    prediction = predict(net, x, device) * target_std + target_mean
    score = r2(y.reshape(-1, 2), prediction.reshape(-1, 2))
    return {
        "windows": int(len(x)),
        "loss": float(np.mean(((y - prediction) / target_std) ** 2)),
        "r2_x": float(score[0]),
        "r2_y": float(score[1]),
        "r2_mean": float(score.mean()),
    }


def main() -> None:
    import torch
    import torch.nn as nn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda", "auto"),
        default="cpu",
        help="CPU is the default because Phase 4b was confirmed on CPU.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check protocol and architecture without loading arrays or training.",
    )
    args = parser.parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")

    context = validate_protocol(require_output_absent=not args.validate_only)
    device = choose_device(args.device)
    configure_determinism(device, args.threads)
    config = candidate_config()
    net = build_net(config, 64).to(device)
    parameter_count = sum(parameter.numel() for parameter in net.parameters())
    if parameter_count != EXPECTED_PARAMETERS:
        raise ValueError(
            f"Expected {EXPECTED_PARAMETERS:,} parameters, found {parameter_count:,}"
        )
    if args.validate_only:
        print("=== 48/48 checkpoint build validation passed ===")
        print(f"output: {OUTPUT_PATH}")
        print(f"parameters: {parameter_count:,}")
        print(
            f"seed={SEED} | epochs={TRAIN_EPOCHS} | "
            "checkpoint=min validation loss | train=29 | "
            "validation=4 inference-only | January=FORBIDDEN"
        )
        print("no arrays loaded; no checkpoint written")
        return

    split = context["manifest"]["chronological_split"]
    train_names = list(split["train"])
    validation_names = list(split["validation"])
    print("=== Build 48x48 firmware checkpoint ===", flush=True)
    print(
        f"train={len(train_names)} | validation={len(validation_names)} "
        "inference-only | January=FORBIDDEN",
        flush=True,
    )
    print(
        f"seed={SEED} | epochs={TRAIN_EPOCHS} | "
        "checkpoint=min validation loss | "
        f"device={device} | output={OUTPUT_PATH.name}",
        flush=True,
    )

    # Only the 29 training sessions determine channels, normalization floors,
    # target normalization, samples and gradients.
    train_loaded = {name: load_model_data(name) for name in train_names}
    feature_std_floor = fit_feature_std_floor(
        train_loaded,
        context["channels"],
    )
    train_prepared = {
        name: prepare_session(
            train_loaded[name],
            context["channels"],
            feature_std_floor,
        )
        for name in train_names
    }
    train_x = np.concatenate(
        [train_prepared[name][0] for name in train_names],
        axis=0,
    )
    train_y = np.concatenate(
        [train_prepared[name][1] for name in train_names],
        axis=0,
    )
    target_mean = train_y.mean(axis=(0, 1))
    target_std = train_y.std(axis=(0, 1)) + 1e-6
    train_y_normalized = ((train_y - target_mean) / target_std).astype(np.float32)
    session_lengths = {
        name: int(len(train_prepared[name][0])) for name in train_names
    }
    del train_loaded, train_prepared

    # Match the 64/64 protocol: December is available for inference-only
    # checkpoint selection but never enters the optimizer or fitted statistics.
    validation_loaded = {
        name: load_model_data(name) for name in validation_names
    }
    validation_prepared = {
        name: prepare_session(
            validation_loaded[name],
            context["channels"],
            feature_std_floor,
        )
        for name in validation_names
    }
    validation_x = np.concatenate(
        [validation_prepared[name][0] for name in validation_names],
        axis=0,
    )
    validation_y = np.concatenate(
        [validation_prepared[name][1] for name in validation_names],
        axis=0,
    )
    del validation_loaded

    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        SCHEDULER_EPOCH_BUDGET,
    )
    mse = nn.MSELoss()
    rng = np.random.default_rng(SEED)
    history = []
    best_state = None
    best_epoch = None
    best_validation_loss = float("inf")
    phase4b_epoch7_reproduction = None
    for epoch in range(1, TRAIN_EPOCHS + 1):
        indices, session_draws, month_draws = draw_session_balanced_indices(
            train_names,
            session_lengths,
            rng,
        )
        net.train()
        error_sum = 0.0
        value_count = 0
        gradient_sum = 0.0
        gradient_max = 0.0
        batch_count = 0
        for start in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[start : start + BATCH_SIZE]
            x_batch = torch.from_numpy(train_x[batch_indices]).to(device)
            y_batch = torch.from_numpy(
                train_y_normalized[batch_indices]
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = mse(net(x_batch), y_batch)
            loss.backward()
            gradient = float(
                torch.nn.utils.clip_grad_norm_(
                    net.parameters(),
                    GRADIENT_CLIP,
                )
            )
            optimizer.step()
            error_sum += float(loss.detach().item()) * y_batch.numel()
            value_count += y_batch.numel()
            gradient_sum += gradient
            gradient_max = max(gradient_max, gradient)
            batch_count += 1
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "optimization_loss": error_sum / max(value_count, 1),
            "gradient_norm_mean_before_clip": (
                gradient_sum / max(batch_count, 1)
            ),
            "gradient_norm_max_before_clip": gradient_max,
            "session_draws": session_draws,
            "month_draws": month_draws,
        }
        epoch_train_metrics = evaluate(
            net,
            train_x,
            train_y,
            target_mean,
            target_std,
            device,
        )
        epoch_validation_metrics = evaluate(
            net,
            validation_x,
            validation_y,
            target_mean,
            target_std,
            device,
        )
        row["train_metrics"] = epoch_train_metrics
        row["validation_metrics"] = epoch_validation_metrics

        if epoch == 7:
            reference_fold = context["reference"]["fold"]
            phase4b_epoch7_reproduction = {
                "pooled_loss_delta": (
                    epoch_validation_metrics["loss"]
                    - reference_fold["pooled_loss"]
                ),
                "pooled_r2_delta": (
                    epoch_validation_metrics["r2_mean"]
                    - reference_fold["pooled_r2_mean"]
                ),
            }
            if device.type == "cpu" and args.threads == 4:
                largest_delta = max(
                    abs(value)
                    for value in phase4b_epoch7_reproduction.values()
                )
                if largest_delta > 1e-5:
                    raise RuntimeError(
                        "Epoch 7 did not reproduce the Phase-4b seed-43 "
                        f"December cell; largest metric delta={largest_delta:.3g}. "
                        "Checkpoint was not written."
                    )

        improved = epoch_validation_metrics["loss"] < best_validation_loss
        if improved:
            best_validation_loss = epoch_validation_metrics["loss"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in net.state_dict().items()
            }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{TRAIN_EPOCHS} | "
            f"opt={row['optimization_loss']:.5f} | "
            f"loss train={epoch_train_metrics['loss']:.5f} "
            f"validation={epoch_validation_metrics['loss']:.5f} | "
            f"R2 train={epoch_train_metrics['r2_mean']:+.4f} "
            f"validation={epoch_validation_metrics['r2_mean']:+.4f} | "
            f"lr={row['learning_rate']:.6g} | "
            f"grad={row['gradient_norm_mean_before_clip']:.3f}/"
            f"{row['gradient_norm_max_before_clip']:.3f}"
            + (" *best*" if improved else ""),
            flush=True,
        )
        scheduler.step()

    if best_state is None or best_epoch is None:
        raise RuntimeError("No validation checkpoint was selected")
    net.load_state_dict(best_state)
    train_metrics = evaluate(
        net,
        train_x,
        train_y,
        target_mean,
        target_std,
        device,
    )
    validation_by_session = {
        name: evaluate(
            net,
            validation_prepared[name][0],
            validation_prepared[name][1],
            target_mean,
            target_std,
            device,
        )
        for name in validation_names
    }
    validation_metrics = evaluate(
        net,
        validation_x,
        validation_y,
        target_mean,
        target_std,
        device,
    )
    validation_macro = float(
        np.mean([row["r2_mean"] for row in validation_by_session.values()])
    )
    validation_worst = float(
        min(row["r2_mean"] for row in validation_by_session.values())
    )
    if phase4b_epoch7_reproduction is None:
        raise RuntimeError("Epoch-7 Phase-4b reproduction check did not run")

    checkpoint = {
        "purpose": "indy_32ch_phase4b_confirmed_48x48_firmware_candidate",
        "created_at_utc": utc_now(),
        "architecture_status": "firmware_candidate_detector_refit_pending",
        "seed": SEED,
        "model_state": {
            key: value.detach().cpu().clone()
            for key, value in net.state_dict().items()
        },
        "config": config,
        "channels": context["channels"].tolist(),
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "feature_std_floor": feature_std_floor[:, 0].tolist(),
        "train_sessions": train_names,
        "validation_sessions": validation_names,
        "test_policy": "locked_not_loaded",
        "observation_seconds": OBSERVATION_SECONDS,
        "training_epochs": TRAIN_EPOCHS,
        "checkpoint_epoch": best_epoch,
        "selection_policy": "minimum_pooled_validation_normalized_mse",
        "best_validation_loss": best_validation_loss,
        "parameter_count": parameter_count,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "validation_macro_r2_mean": validation_macro,
        "validation_worst_session_r2_mean": validation_worst,
        "validation_by_session": validation_by_session,
        "training_history": history,
        "phase4b_protocol_signature": context["phase4b"]["protocol_signature"],
        "phase4b_epoch7_cpu_reproduction_deltas": (
            phase4b_epoch7_reproduction
        ),
        "baseline_checkpoint": "models/indy_32ch/64x64checkpoint.pt",
        "baseline_sha256": context["baseline_sha256"],
        "detector_compatibility": (
            "Layer 1 is channel-compatible; Layer 2 must be refitted for "
            "48-dimensional GRU states before integrated runtime use."
        ),
    }
    save_checkpoint_atomic(checkpoint, OUTPUT_PATH)
    candidate_hash = sha256_file(OUTPUT_PATH)
    build_metrics = {
        "purpose": "48x48_firmware_checkpoint_build",
        "created_at_utc": utc_now(),
        "checkpoint": str(OUTPUT_PATH.relative_to(ROOT)),
        "sha256": candidate_hash,
        "size_bytes": OUTPUT_PATH.stat().st_size,
        "parameter_count": parameter_count,
        "seed": SEED,
        "training_epochs": TRAIN_EPOCHS,
        "checkpoint_epoch": best_epoch,
        "selection_policy": "minimum_pooled_validation_normalized_mse",
        "best_validation_loss": best_validation_loss,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "validation_macro_r2_mean": validation_macro,
        "validation_worst_session_r2_mean": validation_worst,
        "validation_by_session": validation_by_session,
        "phase4b_epoch7_cpu_reproduction_deltas": (
            phase4b_epoch7_reproduction
        ),
        "data_policy": {
            "train_sessions_updated_weights": len(train_names),
            "validation_sessions_updated_weights": 0,
            "january_loaded": False,
            "validation_selected_epoch": True,
        },
        "baseline_protection": {
            "checkpoint": "models/indy_32ch/64x64checkpoint.pt",
            "sha256": sha256_file(BASELINE_PATH),
            "unchanged": sha256_file(BASELINE_PATH)
            == EXPECTED_BASELINE_SHA256,
        },
    }
    write_json_atomic(build_metrics, BUILD_METRICS_PATH)

    print(
        f"selected epoch={best_epoch}/{TRAIN_EPOCHS} | "
        "validation | "
        f"loss={validation_metrics['loss']:.5f} | "
        f"pooled R2={validation_metrics['r2_mean']:+.4f} | "
        f"macro/worst={validation_macro:+.4f}/{validation_worst:+.4f}",
        flush=True,
    )
    print(f"saved: {OUTPUT_PATH}", flush=True)
    print(f"sha256: {candidate_hash}", flush=True)
    print(
        "64x64checkpoint: UNCHANGED | January: NOT LOADED | "
        "validation updated weights: NO",
        flush=True,
    )


if __name__ == "__main__":
    main()
