#!/usr/bin/env python3
"""Build deployment candidates from each session's highest test-R2 Phase-7 fold.

This is an explicitly best-test-fold-selected demonstration policy, not an
unbiased five-fold estimate.  No weights are retrained by this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from indy_loco.experiments.active import phase10_session_deployment_candidates as p10
from indy_loco.history.experiments.phase7.phase7_ann_vs_snn_fivefold import (
    SESSION_BY_NAME,
    aggregate_40ms,
    binned_reach_bounds,
    eligible_reaches,
    load_session,
    make_fold_indices,
    split_fold,
)
from indy_loco.models.midsize.model import MidsizeTCNGRU


PHASE7_ROOT = (
    PROJECT_ROOT
    / "history"
    / "results"
    / "indy"
    / "phase7_ann_vs_snn_fivefold"
)
METRICS_PATH = PHASE7_ROOT / "phase7_ann_vs_snn_fivefold_metrics.json"
CHECKPOINT_ROOT = PHASE7_ROOT / "checkpoints"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--sessions", nargs="*")
    parser.add_argument("--registry-only", action="store_true")
    return parser.parse_args()


def best_rows() -> dict[str, dict]:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    selected = {}
    for session in metrics["sessions"]:
        rows = [row for row in metrics["results"] if row["session"] == session]
        selected[session] = max(rows, key=lambda row: float(row["test"]["r2_mean"]))
    return selected


def load_model(path: Path, device):
    import torch

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = MidsizeTCNGRU().to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint


def process(session_id: str, row: dict, device) -> None:
    import torch

    fold = int(row["fold"])
    source_path = CHECKPOINT_ROOT / f"{session_id}_fold{fold}.pt"
    if p10.sha256_file(source_path) != row["checkpoint_sha256"]:
        raise ValueError(f"{session_id}: Phase-7 checkpoint SHA-256 mismatch")
    model, checkpoint = load_model(source_path, device)
    if checkpoint["session"] != session_id or int(checkpoint["fold"]) != fold:
        raise ValueError(f"{session_id}: best-fold checkpoint identity mismatch")

    data = load_session(SESSION_BY_NAME[session_id])
    counts_all, velocity = aggregate_40ms(data)
    bounds = binned_reach_bounds(data)
    train_reaches, validation_reaches, test_reaches = split_fold(
        make_fold_indices(eligible_reaches(data)), fold - 1
    )
    channels = np.asarray(checkpoint["selected_channel_indices"], dtype=np.int64)
    counts = counts_all[channels].astype(np.float32)
    promoted = torch.load(
        p10.MODEL_ROOT / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    fallback_floor = np.asarray(promoted["feature_std_floor"], dtype=np.float32)
    floor, floor_metadata = p10.fit_training_floor(
        counts, bounds, train_reaches, fallback_floor
    )

    reference_bins, reference_target, reference_prediction = p10.phase7_reference(
        model, counts, velocity, bounds, test_reaches, checkpoint, device
    )
    post_mask = reference_bins >= p10.CALIBRATION_BINS - 1
    features = p10.continuous_features(counts)
    replay_bins, replay_target, replay_prediction, stats = p10.firmware_replay(
        model, features, velocity, reference_bins, floor, checkpoint, device
    )
    replay_metrics = p10.metric_values(replay_target, replay_prediction)
    phase7_metrics = p10.metric_values(
        reference_target[post_mask], reference_prediction[post_mask]
    )
    if not np.array_equal(replay_bins, reference_bins[post_mask]):
        raise ValueError(f"{session_id}: replay/test-bin mismatch")

    output_root = p10.MODEL_ROOT / session_id
    candidate_path = output_root / "deployment_candidate.pt"
    constants_path = output_root / "deployment_constants.npz"
    golden_path = output_root / "deployment_golden_vectors.npz"
    replay_path = output_root / "deployment_replay.json"
    source_copy = output_root / "best_fold_checkpoint.pt"
    source_copy.write_bytes(source_path.read_bytes())

    source_sha = p10.sha256_file(source_path)
    candidate = dict(checkpoint)
    candidate.update(
        {
            "status": "deployment_candidate_replay_complete",
            "source_checkpoint_status": checkpoint.get("status"),
            "source_checkpoint_sha256": source_sha,
            "deployment_schema_version": 2,
            "model_id": session_id,
            "source_channel_count": int(counts_all.shape[0]),
            "physical_channel_count": p10.PHYSICAL_CHANNELS,
            "input_feature_count": p10.FEATURES,
            "parameter_count": p10.EXPECTED_PARAMETER_COUNT,
            "feature_std_floor": floor,
            "selection_policy": "highest_phase7_test_r2_fold",
            "selected_fold": fold,
            "selection_test_r2_mean": float(row["test"]["r2_mean"]),
            "deployment_policy": {
                "bin_ms": int(p10.BIN_SECONDS * 1000),
                "calibration_bins": p10.CALIBRATION_BINS,
                "window_bins": p10.WINDOW_BINS,
                "ewma_alpha": p10.EWMA_ALPHA,
                "window_order": "oldest_to_newest",
                "output_timestep": p10.WINDOW_BINS - 1,
            },
            "floor_fit": floor_metadata,
            "firmware_style_replay": {
                "held_out_split": f"phase7_fold{fold}_test_reaches",
                "test_reaches": int(len(test_reaches)),
                "bins_after_calibration": int(len(replay_bins)),
                "phase7_reference_same_bins": phase7_metrics,
                "firmware_policy_same_bins": replay_metrics,
                "selection_policy": "highest_phase7_test_r2_fold",
                "reporting_caveat": "best-fold result; not a five-fold mean",
            },
        }
    )
    p10.save_torch_atomic(candidate, candidate_path)
    p10.save_npz_atomic(
        constants_path,
        schema_version=np.asarray("session_deployment_constants_v2"),
        model_id=np.asarray(session_id),
        selected_fold=np.asarray(fold, dtype=np.uint8),
        source_channel_count=np.asarray(counts_all.shape[0], dtype=np.uint16),
        selected_channel_indices=channels.astype(np.uint16),
        feature_std_floor=floor,
        target_mean=np.asarray(checkpoint["target_mean"], dtype=np.float32),
        target_std=np.asarray(checkpoint["target_std"], dtype=np.float32),
        source_checkpoint_sha256=np.asarray(source_sha),
    )
    golden = p10.golden_vectors(model, stats["normalized"], checkpoint, device)
    p10.save_npz_atomic(
        golden_path,
        schema_version=np.asarray("session_deployment_golden_v2"),
        model_id=np.asarray(session_id),
        selected_fold=np.asarray(fold, dtype=np.uint8),
        source_checkpoint_sha256=np.asarray(source_sha),
        calibration_mean=stats["mean"],
        calibration_local_std=stats["local_std"],
        calibration_effective_std=stats["effective_std"],
        **golden,
    )
    record = {
        "schema_version": 2,
        "phase": "phase11_best_fold_deployment_candidates",
        "model_id": session_id,
        "selection_policy": "highest_phase7_test_r2_fold",
        "selected_fold": fold,
        "selection_test_r2_mean": float(row["test"]["r2_mean"]),
        "source_checkpoint": source_copy.name,
        "source_checkpoint_sha256": source_sha,
        "candidate_checkpoint_sha256": p10.sha256_file(candidate_path),
        "constants_sha256": p10.sha256_file(constants_path),
        "golden_vectors_sha256": p10.sha256_file(golden_path),
        "source_channel_count": int(counts_all.shape[0]),
        "selected_channel_indices": channels.tolist(),
        "reach_counts": {
            "train": int(len(train_reaches)),
            "validation": int(len(validation_reaches)),
            "test": int(len(test_reaches)),
        },
        "floor_fit": floor_metadata,
        "replay": candidate["firmware_style_replay"],
        "reporting_caveat": "best-fold result; not a five-fold mean",
    }
    p10.write_json_atomic(record, replay_path)
    print(
        f"{session_id}: fold {fold}, Phase-7 test R2={row['test']['r2_mean']:.6f}, "
        f"firmware-policy R2={replay_metrics['r2_mean']:.6f}, "
        f"held-out bins={len(replay_bins):,}"
    )


def main() -> None:
    import torch

    args = parse_args()
    torch.set_num_threads(args.threads)
    device = p10.select_device(args.device)
    rows = best_rows()
    sessions = tuple(args.sessions or rows)
    unknown = set(sessions) - set(rows)
    if unknown:
        raise ValueError(f"Unknown sessions: {sorted(unknown)}")
    if not args.registry_only:
        for session_id in sessions:
            process(session_id, rows[session_id], device)
    if set(sessions) == set(rows):
        registry_path = p10.MODEL_ROOT / "session_checkpoints.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        registry["deployment_selection_policy"] = "highest_phase7_test_r2_fold"
        registry["reporting_caveat"] = "best-fold result; not a five-fold mean"
        for item in registry["sessions"]:
            session_id = item["session"]
            row = rows[session_id]
            root = p10.MODEL_ROOT / session_id
            replay = json.loads((root / "deployment_replay.json").read_text())
            item["deployment_candidate"] = {
                "status": "deployment_candidate_replay_complete",
                "selection_policy": "highest_phase7_test_r2_fold",
                "selected_fold": int(row["fold"]),
                "selection_test_r2_mean": float(row["test"]["r2_mean"]),
                "source_checkpoint": f"{session_id}/best_fold_checkpoint.pt",
                "source_checkpoint_sha256": row["checkpoint_sha256"],
                "checkpoint": f"{session_id}/deployment_candidate.pt",
                "sha256": p10.sha256_file(root / "deployment_candidate.pt"),
                "constants": f"{session_id}/deployment_constants.npz",
                "constants_sha256": p10.sha256_file(root / "deployment_constants.npz"),
                "golden_vectors": f"{session_id}/deployment_golden_vectors.npz",
                "golden_vectors_sha256": p10.sha256_file(
                    root / "deployment_golden_vectors.npz"
                ),
                "firmware_replay_r2_mean": replay["replay"][
                    "firmware_policy_same_bins"
                ]["r2_mean"],
                "phase7_same_bins_r2_mean": replay["replay"][
                    "phase7_reference_same_bins"
                ]["r2_mean"],
                "reporting_caveat": "best-fold result; not a five-fold mean",
            }
        p10.write_json_atomic(registry, registry_path)


if __name__ == "__main__":
    main()
