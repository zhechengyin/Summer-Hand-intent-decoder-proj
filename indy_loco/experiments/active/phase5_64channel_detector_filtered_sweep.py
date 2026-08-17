#!/usr/bin/env python3
"""Sweep a 64-channel Indy decoder with a paired detector-filter ablation.

Phase 5 protocol
----------------

The model architecture, causal features, chronological split, optimizer,
session-balanced sampler and checkpoint rule are fixed.  The experiment varies
learning rate, weight decay and dropout, then compares two training policies:

``baseline``
    All 29 canonical training sessions may update weights.

``detector_filtered``
    The two retrospective Phase-3c detector failures are excluded from every
    train-derived operation and from weight updates.  The excluded sessions are
    ``indy_20160630_01`` and ``indy_20161013_03``.

The first stage evaluates the same eight configurations under both policies at
seed 43.  The second stage takes the union of the two policy winners and checks
each candidate under both policies at seeds 42 and 44.  Seed 43 is reused, so a
complete run performs at most 24 fits.  Both policies draw the baseline number
of windows per epoch, keeping optimizer update counts equal.

December validation is inference-only but selects epochs and hyperparameters.
January is registered as the locked test month and is never loaded.  This is a
retrospective data-exclusion ablation, not prospective validation of an online
detector or permission to delete the excluded source data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from indy_loco.models.indy_32ch.features import multiscale_counts  # noqa: E402
from indy_loco.models.indy_32ch.input_pipeline import (  # noqa: E402
    apply_feature_stats,
    fit_feature_stats,
    load_model_data,
    load_session_manifest,
    processed_session_path,
    top_firing_channels,
    window_arrays,
)
from indy_loco.models.indy_32ch.model import build_net, causal_config, r2  # noqa: E402
from indy_loco.models.indy_32ch.sampling import balanced_allocations  # noqa: E402

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PHASE_NAME = "phase5_64channel_detector_filtered_sweep"
RESULT_DIR = PROJECT_ROOT / "results" / PHASE_NAME
STATE_PATH = RESULT_DIR / ".cache" / f"{PHASE_NAME}_state.json"
METRICS_PATH = RESULT_DIR / f"{PHASE_NAME}_metrics.json"
TRIALS_PATH = RESULT_DIR / f"{PHASE_NAME}_trials.csv"
EPOCHS_PATH = RESULT_DIR / f"{PHASE_NAME}_epochs.csv"
SUMMARY_PATH = RESULT_DIR / f"{PHASE_NAME}_summary.csv"
FIGURE_PATH = RESULT_DIR / f"{PHASE_NAME}_comparison.png"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"

DETECTOR_EVIDENCE_PATH = (
    PROJECT_ROOT
    / "history"
    / "results"
    / "indy"
    / "phase3c_decoder_state_detector"
    / "phase3c_decoder_state_detector_sessions.csv"
)
EXPECTED_EXCLUSIONS = (
    "indy_20160630_01",
    "indy_20161013_03",
)
EXPECTED_SPLITS = {"train": 29, "validation": 4, "test": 4}

CHANNEL_COUNT = 64
ALPHAS = (1.0, 0.1)
INPUT_FEATURES = CHANNEL_COUNT * len(ALPHAS)
BIN_SECONDS = 0.040
OBSERVATION_SECONDS = 60
OBSERVATION_BINS = int(OBSERVATION_SECONDS / BIN_SECONDS)
WINDOW_BINS = 50
TARGET_AXES = (0, 1)
STD_FLOOR_PERCENTILE = 10.0

ARCHITECTURE = {"F": 64, "H": 64, "L": 1, "dils": [1, 2, 4, 8]}
BATCH_SIZE = 32
GRADIENT_CLIP = 1.0
SWEEP_SEED = 43
CONFIRMATION_SEEDS = (42, 44)

# Includes the previous Phase-5a setting: lr=9e-4, wd=0.060, dropout=0.025.
SEARCH_SPACE = {
    "learning_rate": (3e-4, 9e-4),
    "weight_decay": (0.025, 0.060),
    "dropout": (0.025, 0.100),
}

POLICY_EXCLUSIONS = {
    "baseline": (),
    "detector_filtered": EXPECTED_EXCLUSIONS,
}


@dataclass(frozen=True)
class Hyperparameters:
    learning_rate: float
    weight_decay: float
    dropout: float

    @property
    def label(self) -> str:
        return (
            f"lr{self.learning_rate:.0e}_wd{self.weight_decay:.3f}_do{self.dropout:.3f}"
        ).replace("+", "")

    def as_dict(self) -> dict[str, float]:
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
        }


@dataclass
class PreparedPolicy:
    policy: str
    train_names: list[str]
    validation_names: list[str]
    excluded_sessions: list[str]
    channels: np.ndarray
    feature_std_floor: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    train_x: np.ndarray
    train_y: np.ndarray
    train_y_normalized: np.ndarray
    session_lengths: dict[str, int]
    available_train_windows: int
    epoch_samples: int
    validation_x: np.ndarray
    validation_y: np.ndarray
    validation_by_session: dict[str, tuple[np.ndarray, np.ndarray]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint_atomic(payload: dict[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def bool_from_csv(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Expected CSV boolean, found {value!r}")


def audit_detector_evidence() -> dict[str, Any]:
    """Verify the immutable Phase-3c evidence behind the exclusion policy."""
    if not DETECTOR_EVIDENCE_PATH.exists():
        raise FileNotFoundError(
            f"Detector evidence is missing: {DETECTOR_EVIDENCE_PATH}"
        )
    with DETECTOR_EVIDENCE_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    flagged = []
    evidence = {}
    for row in rows:
        known_failure = bool_from_csv(row["known_failure"])
        layer2_flag = bool_from_csv(row["layer2_flag"])
        if known_failure and layer2_flag:
            session = row["session"]
            flagged.append(session)
            evidence[session] = {
                "known_failure": known_failure,
                "layer2_flag": layer2_flag,
                "proposed_combined_decision": row["proposed_combined_decision"],
                "decoder_r2_mean": float(row["decoder_r2_mean"]),
            }
    if tuple(flagged) != EXPECTED_EXCLUSIONS:
        raise ValueError(
            "Phase-3c detector exclusions changed: "
            f"expected {EXPECTED_EXCLUSIONS}, found {tuple(flagged)}"
        )
    if any(
        evidence[name]["proposed_combined_decision"] != "abstain"
        for name in EXPECTED_EXCLUSIONS
    ):
        raise ValueError("A detector-filtered session is no longer marked abstain")
    return {
        "path": str(DETECTOR_EVIDENCE_PATH.relative_to(REPOSITORY_ROOT)),
        "sha256": sha256_file(DETECTOR_EVIDENCE_PATH),
        "flagged_sessions": list(EXPECTED_EXCLUSIONS),
        "rows": evidence,
        "interpretation": (
            "Retrospective Phase-3c evidence; valid for this exclusion ablation "
            "only, not prospective detector performance."
        ),
    }


def all_hyperparameters() -> list[Hyperparameters]:
    return [
        Hyperparameters(*values)
        for values in itertools.product(
            SEARCH_SPACE["learning_rate"],
            SEARCH_SPACE["weight_decay"],
            SEARCH_SPACE["dropout"],
        )
    ]


def model_config(hyperparameters: Hyperparameters, epochs: int) -> dict[str, Any]:
    config = causal_config(n_out=2)
    config.update(
        {
            **ARCHITECTURE,
            "dropout": hyperparameters.dropout,
            "lr": hyperparameters.learning_rate,
            "wd": hyperparameters.weight_decay,
            "epochs": epochs,
            "bs": BATCH_SIZE,
            "noise": 0.0,
            "chdrop": 0.0,
            "cosine": True,
            "act": "relu",
            "gradient_clip": GRADIENT_CLIP,
            "bidir": False,
        }
    )
    return config


def protocol_signature(detector_audit: dict[str, Any], epochs: int) -> str:
    protocol = {
        "phase": PHASE_NAME,
        "detector_sha256": detector_audit["sha256"],
        "expected_exclusions": EXPECTED_EXCLUSIONS,
        "search_space": SEARCH_SPACE,
        "architecture": ARCHITECTURE,
        "channel_count": CHANNEL_COUNT,
        "alphas": ALPHAS,
        "observation_seconds": OBSERVATION_SECONDS,
        "window_bins": WINDOW_BINS,
        "epochs": epochs,
        "sweep_seed": SWEEP_SEED,
        "confirmation_seeds": CONFIRMATION_SEEDS,
    }
    encoded = json.dumps(protocol, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def choose_device(requested: str):
    import torch

    if requested != "cpu":
        raise ValueError(
            "This experiment is CPU-only. Previous MPS runs produced invalid "
            "backward gradients for this causal TCN+GRU graph."
        )
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate_protocol(epochs: int) -> dict[str, Any]:
    if epochs <= 0:
        raise ValueError("--epochs must be positive")
    detector_audit = audit_detector_evidence()
    manifest = load_session_manifest()
    split = manifest["chronological_split"]
    counts = {name: len(split[name]) for name in EXPECTED_SPLITS}
    if counts != EXPECTED_SPLITS:
        raise ValueError(
            f"Expected chronological split {EXPECTED_SPLITS}, found {counts}"
        )
    train_names = list(split["train"])
    validation_names = list(split["validation"])
    if not set(EXPECTED_EXCLUSIONS).issubset(train_names):
        raise ValueError("Detector exclusions must both belong to the train split")
    if any(name.startswith("indy_201701") for name in train_names + validation_names):
        raise ValueError("January leaked into train or validation")
    if any(not name.startswith("indy_201701") for name in split["test"]):
        raise ValueError("Locked test registry is not the expected January month")
    for name in train_names + validation_names:
        if not processed_session_path(name).exists():
            raise FileNotFoundError(f"Missing processed session: {name}")
    parameter_count = sum(
        parameter.numel()
        for parameter in build_net(
            model_config(all_hyperparameters()[0], epochs), INPUT_FEATURES
        ).parameters()
    )
    if parameter_count != 82_882:
        raise ValueError(
            f"64-channel 64/64 parameter count changed: {parameter_count:,}"
        )
    return {
        "manifest": manifest,
        "train_names": train_names,
        "validation_names": validation_names,
        "detector_audit": detector_audit,
        "parameter_count": parameter_count,
        "signature": protocol_signature(detector_audit, epochs),
    }


def fit_feature_std_floor(
    loaded: dict[str, tuple[np.ndarray, np.ndarray]], channels: np.ndarray
) -> np.ndarray:
    session_stds = []
    for counts, _ in loaded.values():
        features = multiscale_counts(counts[channels], ALPHAS)
        _, std = fit_feature_stats(features, observation_bins=OBSERVATION_BINS)
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
    if counts.shape[0] < CHANNEL_COUNT:
        raise ValueError(
            f"Session has {counts.shape[0]} channels; {CHANNEL_COUNT} required"
        )
    features = multiscale_counts(counts[channels], ALPHAS)
    if features.shape[0] != INPUT_FEATURES:
        raise AssertionError(
            f"Expected {INPUT_FEATURES} features, found {features.shape[0]}"
        )
    mean, local_std = fit_feature_stats(features, observation_bins=OBSERVATION_BINS)
    normalized = apply_feature_stats(
        features, (mean, np.maximum(local_std, feature_std_floor))
    )
    windows = window_arrays(
        normalized,
        velocity,
        TARGET_AXES,
        window_bins=WINDOW_BINS,
        start_bin=OBSERVATION_BINS,
    )
    if not windows:
        raise ValueError("Session is too short after the 60-second warm-up")
    return (
        np.stack([window["e"] for window in windows]).astype(np.float32),
        np.stack([window["vel"] for window in windows]).astype(np.float32),
    )


def prepare_policy(
    policy: str,
    canonical_train_names: list[str],
    validation_names: list[str],
    epoch_samples: int | None = None,
) -> PreparedPolicy:
    excluded = list(POLICY_EXCLUSIONS[policy])
    train_names = [name for name in canonical_train_names if name not in excluded]
    expected_count = 29 if policy == "baseline" else 27
    if len(train_names) != expected_count:
        raise ValueError(
            f"{policy} should use {expected_count} sessions, found {len(train_names)}"
        )

    print(f"\nPreparing {policy}: train={len(train_names)}", flush=True)
    if excluded:
        print(f"excluded from every fitted operation: {excluded}", flush=True)
    train_loaded = {name: load_model_data(name) for name in train_names}
    channels = top_firing_channels(
        train_loaded, CHANNEL_COUNT, observation_bins=OBSERVATION_BINS
    )
    if channels.shape != (CHANNEL_COUNT,) or len(np.unique(channels)) != CHANNEL_COUNT:
        raise AssertionError("Channel selection did not return 64 unique channels")
    feature_std_floor = fit_feature_std_floor(train_loaded, channels)
    train_prepared = {
        name: prepare_session(train_loaded[name], channels, feature_std_floor)
        for name in train_names
    }
    train_x = np.concatenate([train_prepared[name][0] for name in train_names], axis=0)
    train_y = np.concatenate([train_prepared[name][1] for name in train_names], axis=0)
    target_mean = train_y.mean(axis=(0, 1)).astype(np.float32)
    target_std = (train_y.std(axis=(0, 1)) + 1e-6).astype(np.float32)
    train_y_normalized = ((train_y - target_mean) / target_std).astype(np.float32)
    session_lengths = {name: int(len(train_prepared[name][0])) for name in train_names}
    available_train_windows = int(sum(session_lengths.values()))
    if epoch_samples is None:
        epoch_samples = available_train_windows
    if epoch_samples <= 0:
        raise ValueError("The shared epoch sample count must be positive")
    del train_loaded, train_prepared

    # December is opened only after this policy's train-derived choices freeze.
    validation_loaded = {name: load_model_data(name) for name in validation_names}
    validation_by_session = {
        name: prepare_session(validation_loaded[name], channels, feature_std_floor)
        for name in validation_names
    }
    validation_x = np.concatenate(
        [validation_by_session[name][0] for name in validation_names], axis=0
    )
    validation_y = np.concatenate(
        [validation_by_session[name][1] for name in validation_names], axis=0
    )
    del validation_loaded
    print(f"channels (zero-based): {channels.tolist()}", flush=True)
    print(
        f"available windows train={len(train_x)} | epoch samples={epoch_samples} | "
        f"validation windows={len(validation_x)} | "
        f"input={tuple(train_x.shape[1:])}",
        flush=True,
    )
    return PreparedPolicy(
        policy=policy,
        train_names=train_names,
        validation_names=validation_names,
        excluded_sessions=excluded,
        channels=channels,
        feature_std_floor=feature_std_floor,
        target_mean=target_mean,
        target_std=target_std,
        train_x=train_x,
        train_y=train_y,
        train_y_normalized=train_y_normalized,
        session_lengths=session_lengths,
        available_train_windows=available_train_windows,
        epoch_samples=epoch_samples,
        validation_x=validation_x,
        validation_y=validation_y,
        validation_by_session=validation_by_session,
    )


def draw_fixed_size_session_balanced_indices(
    sessions: list[str],
    session_lengths: dict[str, int],
    epoch_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    """Draw equal-sized epochs while preserving equal session exposure."""
    if set(sessions) != set(session_lengths):
        raise ValueError("sessions and session_lengths contain different names")
    if any(session_lengths[name] <= 0 for name in sessions):
        raise ValueError("Every allowed training session needs at least one window")
    session_draws = balanced_allocations(sessions, epoch_samples, rng)
    offsets = {}
    cursor = 0
    for session in sessions:
        offsets[session] = cursor
        cursor += session_lengths[session]
    blocks = []
    for session in sessions:
        local = rng.integers(0, session_lengths[session], size=session_draws[session])
        blocks.append(local.astype(np.int64) + offsets[session])
    indices = np.concatenate(blocks)
    rng.shuffle(indices)
    month_draws: Counter[str] = Counter()
    for session, count in session_draws.items():
        date = session.split("_")[1]
        month_draws[f"{date[:4]}-{date[4:6]}"] += count
    if len(indices) != epoch_samples:
        raise AssertionError("Balanced sampler changed the shared epoch size")
    if np.any(indices < 0) or np.any(indices >= cursor):
        raise AssertionError("Balanced sampler produced an invalid window index")
    return indices, session_draws, dict(sorted(month_draws.items()))


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
) -> dict[str, float | int]:
    prediction = predict(net, x, device) * target_std + target_mean
    score = r2(y.reshape(-1, 2), prediction.reshape(-1, 2))
    return {
        "windows": int(len(x)),
        "loss": float(np.mean(((y - prediction) / target_std) ** 2)),
        "r2_x": float(score[0]),
        "r2_y": float(score[1]),
        "r2_mean": float(score.mean()),
    }


def run_key(policy: str, seed: int, hyperparameters: Hyperparameters) -> str:
    return f"{policy}|seed{seed}|{hyperparameters.label}"


def checkpoint_path(policy: str, seed: int, hyperparameters: Hyperparameters) -> Path:
    return CHECKPOINT_DIR / f"{policy}_seed{seed}_{hyperparameters.label}.pt"


def train_one(
    *,
    prepared: PreparedPolicy,
    hyperparameters: Hyperparameters,
    seed: int,
    epochs: int,
    stage: str,
    device,
    parameter_count: int,
    detector_audit: dict[str, Any],
) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    seed_everything(seed)
    config = model_config(hyperparameters, epochs)
    net = build_net(config, INPUT_FEATURES).to(device)
    actual_parameters = sum(parameter.numel() for parameter in net.parameters())
    if actual_parameters != parameter_count:
        raise AssertionError(
            f"Parameter count changed: {actual_parameters:,} != {parameter_count:,}"
        )
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    mse = nn.MSELoss()
    rng = np.random.default_rng(seed)
    best_state = None
    best_epoch = None
    best_validation_loss = float("inf")
    history = []

    print(
        f"\n=== {stage} | {prepared.policy} | seed={seed} | "
        f"{hyperparameters.label} ===",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        indices, session_draws, month_draws = draw_fixed_size_session_balanced_indices(
            prepared.train_names,
            prepared.session_lengths,
            prepared.epoch_samples,
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
            x_batch = torch.from_numpy(prepared.train_x[batch_indices]).to(device)
            y_batch = torch.from_numpy(prepared.train_y_normalized[batch_indices]).to(
                device
            )
            optimizer.zero_grad(set_to_none=True)
            loss = mse(net(x_batch), y_batch)
            loss.backward()
            gradient = float(
                torch.nn.utils.clip_grad_norm_(net.parameters(), GRADIENT_CLIP)
            )
            optimizer.step()
            error_sum += float(loss.detach().item()) * y_batch.numel()
            value_count += y_batch.numel()
            gradient_sum += gradient
            gradient_max = max(gradient_max, gradient)
            batch_count += 1

        validation_metrics = evaluate(
            net,
            prepared.validation_x,
            prepared.validation_y,
            prepared.target_mean,
            prepared.target_std,
            device,
        )
        improved = validation_metrics["loss"] < best_validation_loss
        if improved:
            best_validation_loss = float(validation_metrics["loss"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in net.state_dict().items()
            }
        row = {
            "epoch": epoch,
            "learning_rate_schedule": float(optimizer.param_groups[0]["lr"]),
            "optimization_loss": error_sum / max(value_count, 1),
            "gradient_norm_mean_before_clip": gradient_sum / max(batch_count, 1),
            "gradient_norm_max_before_clip": gradient_max,
            "validation_loss": float(validation_metrics["loss"]),
            "validation_r2": float(validation_metrics["r2_mean"]),
            "month_draws": month_draws,
            "session_draw_min": int(min(session_draws.values())),
            "session_draw_max": int(max(session_draws.values())),
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{epochs} | opt={row['optimization_loss']:.5f} | "
            f"validation loss={row['validation_loss']:.5f} "
            f"R2={row['validation_r2']:+.4f} | "
            f"lr={row['learning_rate_schedule']:.6g} | "
            f"grad={row['gradient_norm_mean_before_clip']:.3f}/"
            f"{row['gradient_norm_max_before_clip']:.3f}"
            + (" *best*" if improved else ""),
            flush=True,
        )
        scheduler.step()

    if best_state is None or best_epoch is None:
        raise RuntimeError("No checkpoint was selected")
    net.load_state_dict(best_state)
    train_metrics = evaluate(
        net,
        prepared.train_x,
        prepared.train_y,
        prepared.target_mean,
        prepared.target_std,
        device,
    )
    validation_metrics = evaluate(
        net,
        prepared.validation_x,
        prepared.validation_y,
        prepared.target_mean,
        prepared.target_std,
        device,
    )
    validation_by_session = {
        name: evaluate(
            net,
            prepared.validation_by_session[name][0],
            prepared.validation_by_session[name][1],
            prepared.target_mean,
            prepared.target_std,
            device,
        )
        for name in prepared.validation_names
    }
    macro_r2 = float(
        np.mean([row["r2_mean"] for row in validation_by_session.values()])
    )
    worst_r2 = float(min(row["r2_mean"] for row in validation_by_session.values()))
    output_path = checkpoint_path(prepared.policy, seed, hyperparameters)
    checkpoint = {
        "purpose": PHASE_NAME,
        "status": "experiment_only_not_promoted",
        "created_at_utc": utc_now(),
        "stage": stage,
        "policy": prepared.policy,
        "excluded_sessions": prepared.excluded_sessions,
        "detector_evidence_sha256": detector_audit["sha256"],
        "detector_evidence_scope": detector_audit["interpretation"],
        "seed": seed,
        "training_device": device.type,
        "model_state": best_state,
        "config": config,
        "parameter_count": parameter_count,
        "neural_channel_count": CHANNEL_COUNT,
        "input_feature_count": INPUT_FEATURES,
        "channels": prepared.channels.tolist(),
        "feature_std_floor": prepared.feature_std_floor[:, 0].tolist(),
        "target_mean": prepared.target_mean.tolist(),
        "target_std": prepared.target_std.tolist(),
        "train_sessions": prepared.train_names,
        "available_train_windows": prepared.available_train_windows,
        "epoch_samples": prepared.epoch_samples,
        "validation_sessions": prepared.validation_names,
        "test_policy": "January locked and never loaded",
        "january_loaded": False,
        "checkpoint_epoch": best_epoch,
        "selection_policy": "minimum pooled December validation normalized MSE",
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "validation_macro_r2_mean": macro_r2,
        "validation_worst_session_r2_mean": worst_r2,
        "validation_by_session": validation_by_session,
        "training_history": history,
    }
    save_checkpoint_atomic(checkpoint, output_path)
    print(
        f"selected epoch={best_epoch:02d} | train R2={train_metrics['r2_mean']:+.4f} | "
        f"validation pooled/macro/worst R2="
        f"{validation_metrics['r2_mean']:+.4f}/{macro_r2:+.4f}/{worst_r2:+.4f}",
        flush=True,
    )
    return {
        "key": run_key(prepared.policy, seed, hyperparameters),
        "stage": stage,
        "policy": prepared.policy,
        "seed": seed,
        "hyperparameters": hyperparameters.as_dict(),
        "config_label": hyperparameters.label,
        "train_session_count": len(prepared.train_names),
        "excluded_sessions": prepared.excluded_sessions,
        "selected_channels_zero_based": prepared.channels.tolist(),
        "available_train_windows": prepared.available_train_windows,
        "epoch_samples": prepared.epoch_samples,
        "validation_windows": int(len(prepared.validation_x)),
        "checkpoint": str(output_path.relative_to(REPOSITORY_ROOT)),
        "checkpoint_sha256": sha256_file(output_path),
        "checkpoint_epoch": best_epoch,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "validation_macro_r2_mean": macro_r2,
        "validation_worst_session_r2_mean": worst_r2,
        "validation_by_session": validation_by_session,
        "history": history,
    }


def initial_state(signature: str, detector_audit: dict[str, Any], epochs: int) -> dict:
    return {
        "phase": PHASE_NAME,
        "protocol_signature": signature,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "epochs": epochs,
        "detector_audit": detector_audit,
        "runs": {},
    }


def load_or_create_state(
    *,
    signature: str,
    detector_audit: dict[str, Any],
    epochs: int,
    resume: bool,
    overwrite: bool,
) -> dict[str, Any]:
    if resume and overwrite:
        raise ValueError("Use only one of --resume and --overwrite")
    if overwrite and RESULT_DIR.exists():
        shutil.rmtree(RESULT_DIR)
    if STATE_PATH.exists():
        if not resume:
            raise FileExistsError(
                f"Existing Phase 5 state found at {STATE_PATH}. "
                "Use --resume or --overwrite."
            )
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if state.get("protocol_signature") != signature:
            raise ValueError(
                "Existing state belongs to a different protocol. "
                "Use --overwrite only if replacement is intentional."
            )
        return state
    state = initial_state(signature, detector_audit, epochs)
    write_json_atomic(state, STATE_PATH)
    return state


def persist_run(state: dict[str, Any], result: dict[str, Any]) -> None:
    state["runs"][result["key"]] = result
    state["updated_at_utc"] = utc_now()
    write_json_atomic(state, STATE_PATH)


def config_from_result(result: dict[str, Any]) -> Hyperparameters:
    values = result["hyperparameters"]
    return Hyperparameters(
        learning_rate=float(values["learning_rate"]),
        weight_decay=float(values["weight_decay"]),
        dropout=float(values["dropout"]),
    )


def run_if_needed(
    *,
    state: dict[str, Any],
    prepared: PreparedPolicy,
    hyperparameters: Hyperparameters,
    seed: int,
    epochs: int,
    stage: str,
    device,
    parameter_count: int,
    detector_audit: dict[str, Any],
) -> dict[str, Any]:
    key = run_key(prepared.policy, seed, hyperparameters)
    existing = state["runs"].get(key)
    if existing is not None:
        existing_checkpoint = REPOSITORY_ROOT / existing["checkpoint"]
        if not existing_checkpoint.exists():
            raise FileNotFoundError(
                f"Completed record {key} has no checkpoint: {existing_checkpoint}"
            )
        if sha256_file(existing_checkpoint) != existing["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint digest changed for completed record {key}")
        print(f"\nSKIP completed: {key}", flush=True)
        return existing
    result = train_one(
        prepared=prepared,
        hyperparameters=hyperparameters,
        seed=seed,
        epochs=epochs,
        stage=stage,
        device=device,
        parameter_count=parameter_count,
        detector_audit=detector_audit,
    )
    persist_run(state, result)
    return result


def select_seed43_winners(
    state: dict[str, Any], hyperparameters: list[Hyperparameters]
) -> dict[str, Hyperparameters]:
    winners = {}
    for policy in POLICY_EXCLUSIONS:
        results = [
            state["runs"][run_key(policy, SWEEP_SEED, config)]
            for config in hyperparameters
        ]
        winner = min(results, key=lambda row: row["validation_metrics"]["loss"])
        winners[policy] = config_from_result(winner)
    return winners


def aggregate_candidate_results(
    state: dict[str, Any], candidates: list[Hyperparameters]
) -> list[dict[str, Any]]:
    seeds = (SWEEP_SEED, *CONFIRMATION_SEEDS)
    rows = []
    for config in candidates:
        for policy in POLICY_EXCLUSIONS:
            results = [state["runs"][run_key(policy, seed, config)] for seed in seeds]
            rows.append(
                {
                    "policy": policy,
                    "config_label": config.label,
                    **config.as_dict(),
                    "seeds": list(seeds),
                    "validation_loss_mean": float(
                        np.mean([r["validation_metrics"]["loss"] for r in results])
                    ),
                    "validation_loss_std": float(
                        np.std([r["validation_metrics"]["loss"] for r in results])
                    ),
                    "validation_pooled_r2_mean": float(
                        np.mean([r["validation_metrics"]["r2_mean"] for r in results])
                    ),
                    "validation_pooled_r2_std": float(
                        np.std([r["validation_metrics"]["r2_mean"] for r in results])
                    ),
                    "validation_macro_r2_mean": float(
                        np.mean([r["validation_macro_r2_mean"] for r in results])
                    ),
                    "validation_macro_r2_std": float(
                        np.std([r["validation_macro_r2_mean"] for r in results])
                    ),
                    "validation_worst_r2_mean": float(
                        np.mean(
                            [r["validation_worst_session_r2_mean"] for r in results]
                        )
                    ),
                    "validation_worst_r2_min": float(
                        min(r["validation_worst_session_r2_mean"] for r in results)
                    ),
                    "selected_epoch_mean": float(
                        np.mean([r["checkpoint_epoch"] for r in results])
                    ),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    *,
    state: dict[str, Any],
    detector_audit: dict[str, Any],
    parameter_count: int,
    epochs: int,
    seed43_winners: dict[str, Hyperparameters],
    candidates: list[Hyperparameters],
) -> dict[str, Any]:
    runs = list(state["runs"].values())
    trial_rows = []
    epoch_rows = []
    for result in runs:
        trial_rows.append(
            {
                "stage": result["stage"],
                "policy": result["policy"],
                "seed": result["seed"],
                "config_label": result["config_label"],
                **result["hyperparameters"],
                "train_session_count": result["train_session_count"],
                "excluded_sessions": ";".join(result["excluded_sessions"]),
                "available_train_windows": result["available_train_windows"],
                "epoch_samples": result["epoch_samples"],
                "checkpoint_epoch": result["checkpoint_epoch"],
                "train_loss": result["train_metrics"]["loss"],
                "train_r2": result["train_metrics"]["r2_mean"],
                "validation_loss": result["validation_metrics"]["loss"],
                "validation_pooled_r2": result["validation_metrics"]["r2_mean"],
                "validation_macro_r2": result["validation_macro_r2_mean"],
                "validation_worst_session_r2": result[
                    "validation_worst_session_r2_mean"
                ],
                "checkpoint": result["checkpoint"],
            }
        )
        for row in result["history"]:
            epoch_rows.append(
                {
                    "stage": result["stage"],
                    "policy": result["policy"],
                    "seed": result["seed"],
                    "config_label": result["config_label"],
                    "epoch": row["epoch"],
                    "learning_rate_schedule": row["learning_rate_schedule"],
                    "optimization_loss": row["optimization_loss"],
                    "gradient_norm_mean_before_clip": row[
                        "gradient_norm_mean_before_clip"
                    ],
                    "gradient_norm_max_before_clip": row[
                        "gradient_norm_max_before_clip"
                    ],
                    "validation_loss": row["validation_loss"],
                    "validation_r2": row["validation_r2"],
                }
            )
    trial_rows.sort(key=lambda row: (row["policy"], row["seed"], row["config_label"]))
    epoch_rows.sort(
        key=lambda row: (
            row["policy"],
            row["seed"],
            row["config_label"],
            row["epoch"],
        )
    )
    aggregate_rows = aggregate_candidate_results(state, candidates)
    write_csv(TRIALS_PATH, trial_rows)
    write_csv(EPOCHS_PATH, epoch_rows)
    write_csv(SUMMARY_PATH, aggregate_rows)

    aggregate_winners = {}
    for policy in POLICY_EXCLUSIONS:
        policy_rows = [row for row in aggregate_rows if row["policy"] == policy]
        aggregate_winners[policy] = min(
            policy_rows, key=lambda row: row["validation_loss_mean"]
        )
    paired_deltas = []
    for config in candidates:
        baseline = next(
            row
            for row in aggregate_rows
            if row["policy"] == "baseline" and row["config_label"] == config.label
        )
        filtered = next(
            row
            for row in aggregate_rows
            if row["policy"] == "detector_filtered"
            and row["config_label"] == config.label
        )
        paired_deltas.append(
            {
                "config_label": config.label,
                "filtered_minus_baseline_pooled_r2": (
                    filtered["validation_pooled_r2_mean"]
                    - baseline["validation_pooled_r2_mean"]
                ),
                "filtered_minus_baseline_macro_r2": (
                    filtered["validation_macro_r2_mean"]
                    - baseline["validation_macro_r2_mean"]
                ),
                "filtered_minus_baseline_worst_r2": (
                    filtered["validation_worst_r2_mean"]
                    - baseline["validation_worst_r2_mean"]
                ),
            }
        )
    payload = {
        "phase": "5",
        "name": PHASE_NAME,
        "created_at_utc": utc_now(),
        "protocol_signature": state["protocol_signature"],
        "purpose": (
            "64-channel hyperparameter sweep and paired retrospective "
            "detector-filter exclusion ablation"
        ),
        "protocol": {
            "model": "strictly causal 64-channel 64/64 TCN+GRU",
            "parameter_count": parameter_count,
            "features": "64 raw counts + 64 causal EWMA",
            "sampling": "session-balanced",
            "epoch_size_policy": (
                "both policies use the baseline available-window count so "
                "optimizer update counts match"
            ),
            "epochs_each": epochs,
            "sweep_seed": SWEEP_SEED,
            "confirmation_seeds": list(CONFIRMATION_SEEDS),
            "search_space": SEARCH_SPACE,
            "checkpoint_selection": (
                "minimum pooled December validation normalized MSE"
            ),
            "device": "cpu only",
        },
        "data_policy": {
            "baseline_train_sessions": 29,
            "detector_filtered_train_sessions": 27,
            "validation_sessions": 4,
            "validation_updates_weights": False,
            "test_sessions_loaded": 0,
            "january_loaded": False,
            "excluded_source_data_deleted": False,
        },
        "detector_audit": detector_audit,
        "seed43_winners": {
            policy: config.as_dict() for policy, config in seed43_winners.items()
        },
        "confirmation_candidates": [config.as_dict() for config in candidates],
        "aggregate_results": aggregate_rows,
        "aggregate_winners": aggregate_winners,
        "paired_deltas": paired_deltas,
        "interpretation_boundary": (
            "December validation may compare policies and tune hyperparameters. "
            "January remains unopened. Detector filtering is retrospective and "
            "does not establish prospective detector reliability."
        ),
    }
    write_json_atomic(payload, METRICS_PATH)
    return payload


def render_figure(
    state: dict[str, Any],
    payload: dict[str, Any],
    candidates: list[Hyperparameters],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference = Hyperparameters(9e-4, 0.060, 0.025)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    colors = {"baseline": "#2458A6", "detector_filtered": "#C4771A"}
    for policy in POLICY_EXCLUSIONS:
        result = state["runs"][run_key(policy, SWEEP_SEED, reference)]
        axes[0].plot(
            [row["epoch"] for row in result["history"]],
            [row["validation_r2"] for row in result["history"]],
            label=policy.replace("_", " "),
            color=colors[policy],
            linewidth=2,
        )
        axes[0].scatter(
            [result["checkpoint_epoch"]],
            [result["validation_metrics"]["r2_mean"]],
            color=colors[policy],
            zorder=4,
        )
    axes[0].set_title("Previous Phase-5a hyperparameters · seed 43")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("December pooled R²")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    aggregate = payload["aggregate_results"]
    labels = [config.label for config in candidates]
    x = np.arange(len(labels))
    width = 0.36
    for offset, policy in zip((-width / 2, width / 2), POLICY_EXCLUSIONS, strict=True):
        rows = [
            next(
                row
                for row in aggregate
                if row["policy"] == policy and row["config_label"] == label
            )
            for label in labels
        ]
        axes[1].bar(
            x + offset,
            [row["validation_pooled_r2_mean"] for row in rows],
            width,
            yerr=[row["validation_pooled_r2_std"] for row in rows],
            capsize=4,
            label=policy.replace("_", " "),
            color=colors[policy],
            alpha=0.88,
        )
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_ylabel("December pooled R² · mean ± SD (3 seeds)")
    axes[1].set_title("Paired confirmation candidates")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    figure.suptitle(
        "Indy Loco Phase 5 · 64-channel hyperparameter and detector-filter ablation\n"
        "December selects models · January never loaded"
    )
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Epochs per fit; 30 preserves the original Phase-5a schedule.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check the protocol without loading arrays or writing results.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a matching interrupted run from completed fit records.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and replace existing Phase 5 outputs intentionally.",
    )
    args = parser.parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    context = validate_protocol(args.epochs)
    hyperparameters = all_hyperparameters()

    if args.validate_only:
        print("=== Indy Loco Phase 5 protocol validation passed ===")
        print(
            f"model=64 channels, 64/64 TCN+GRU | "
            f"parameters={context['parameter_count']:,} | epochs={args.epochs}"
        )
        print(f"grid configurations={len(hyperparameters)}")
        for config in hyperparameters:
            print(f"  {config.label}: {config.as_dict()}")
        print("baseline train sessions=29")
        print(
            "detector-filtered train sessions=27 | excluded="
            f"{list(EXPECTED_EXCLUSIONS)}"
        )
        print(
            "stage 1: both policies × 8 configs × seed 43 = 16 fits\n"
            "stage 2: both policies × up to 2 winners × seeds 42/44 "
            "= up to 8 fits"
        )
        print("December=validation only | January=LOCKED AND NOT LOADED")
        print("no arrays loaded; no outputs written")
        return

    state = load_or_create_state(
        signature=context["signature"],
        detector_audit=context["detector_audit"],
        epochs=args.epochs,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print("=== Indy Loco Phase 5 ===", flush=True)
    print(
        "metric=velocity decoding R² (not classification accuracy) | "
        f"device={device} | epochs={args.epochs}",
        flush=True,
    )
    print(
        "train policies: baseline=29 sessions | detector_filtered=27 sessions",
        flush=True,
    )
    print(
        f"retrospective exclusions={list(EXPECTED_EXCLUSIONS)} | "
        "January=LOCKED AND NOT LOADED",
        flush=True,
    )

    # Stage 1: identical grid under both policies at seed 43.
    shared_epoch_samples = None
    for policy in POLICY_EXCLUSIONS:
        prepared = prepare_policy(
            policy,
            context["train_names"],
            context["validation_names"],
            epoch_samples=shared_epoch_samples,
        )
        if shared_epoch_samples is None:
            shared_epoch_samples = prepared.available_train_windows
        for config in hyperparameters:
            run_if_needed(
                state=state,
                prepared=prepared,
                hyperparameters=config,
                seed=SWEEP_SEED,
                epochs=args.epochs,
                stage="seed43_grid",
                device=device,
                parameter_count=context["parameter_count"],
                detector_audit=context["detector_audit"],
            )
        del prepared

    seed43_winners = select_seed43_winners(state, hyperparameters)
    candidates = sorted(
        set(seed43_winners.values()),
        key=lambda item: (item.learning_rate, item.weight_decay, item.dropout),
    )
    print("\n=== seed 43 winners ===", flush=True)
    for policy, config in seed43_winners.items():
        print(f"{policy}: {config.label}", flush=True)
    print(
        f"paired confirmation candidates={[config.label for config in candidates]}",
        flush=True,
    )

    # Stage 2: both policies see both candidate configs at seeds 42 and 44.
    confirmation_epoch_samples = None
    for policy in POLICY_EXCLUSIONS:
        prepared = prepare_policy(
            policy,
            context["train_names"],
            context["validation_names"],
            epoch_samples=confirmation_epoch_samples,
        )
        if confirmation_epoch_samples is None:
            confirmation_epoch_samples = prepared.available_train_windows
        if confirmation_epoch_samples != shared_epoch_samples:
            raise AssertionError("Shared epoch sample count changed between stages")
        for config in candidates:
            for seed in CONFIRMATION_SEEDS:
                run_if_needed(
                    state=state,
                    prepared=prepared,
                    hyperparameters=config,
                    seed=seed,
                    epochs=args.epochs,
                    stage="paired_seed_confirmation",
                    device=device,
                    parameter_count=context["parameter_count"],
                    detector_audit=context["detector_audit"],
                )
        del prepared

    payload = write_outputs(
        state=state,
        detector_audit=context["detector_audit"],
        parameter_count=context["parameter_count"],
        epochs=args.epochs,
        seed43_winners=seed43_winners,
        candidates=candidates,
    )
    render_figure(state, payload, candidates)

    print("\n=== three-seed paired summary ===", flush=True)
    for row in payload["aggregate_results"]:
        print(
            f"{row['policy']:19s} | {row['config_label']:27s} | "
            f"loss={row['validation_loss_mean']:.5f}±"
            f"{row['validation_loss_std']:.5f} | pooled R2="
            f"{row['validation_pooled_r2_mean']:+.4f}±"
            f"{row['validation_pooled_r2_std']:.4f} | macro/worst="
            f"{row['validation_macro_r2_mean']:+.4f}/"
            f"{row['validation_worst_r2_mean']:+.4f}",
            flush=True,
        )
    print("\nPaired filtered-minus-baseline deltas:", flush=True)
    for row in payload["paired_deltas"]:
        print(
            f"{row['config_label']:27s} | pooled/macro/worst="
            f"{row['filtered_minus_baseline_pooled_r2']:+.4f}/"
            f"{row['filtered_minus_baseline_macro_r2']:+.4f}/"
            f"{row['filtered_minus_baseline_worst_r2']:+.4f}",
            flush=True,
        )
    print(f"metrics: {METRICS_PATH}", flush=True)
    print(f"trials: {TRIALS_PATH}", flush=True)
    print(f"figure: {FIGURE_PATH}", flush=True)
    print("January: NOT LOADED | active checkpoints: UNCHANGED", flush=True)


if __name__ == "__main__":
    main()
