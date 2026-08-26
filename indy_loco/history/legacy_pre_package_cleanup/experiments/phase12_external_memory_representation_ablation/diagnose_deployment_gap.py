#!/usr/bin/env python3
"""Sequentially decompose the Indy best-fold Phase-7/deployment replay gap."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
sys.path[:0] = [str(REPO), str(ROOT)]

import deployment_parity_ab as parity  # noqa: E402
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

SESSION = "indy_20160622_01"
OUTPUT = ROOT / "results" / "deployment_parity" / "deployment_gap_diagnosis.json"


def rolling_with_reach_resets(
    model: object,
    counts: np.ndarray,
    velocity: np.ndarray,
    bounds: np.ndarray,
    reaches: np.ndarray,
    selected: set[int],
    checkpoint: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32).reshape(p10.FEATURES, 1)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32).reshape(p10.FEATURES, 1)
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32).reshape(2)
    target_std = np.asarray(checkpoint["target_std"], dtype=np.float32).reshape(2)
    inputs, targets = [], []
    for reach in sorted((int(value) for value in reaches), key=lambda value: bounds[value, 0]):
        start, stop = (int(value) for value in bounds[reach])
        normalized = ((p10.continuous_features(counts[:, start:stop]) - mean) / std).astype(np.float32)
        for local in range(stop - start):
            if start + local not in selected:
                continue
            window = np.zeros((p10.FEATURES, p10.WINDOW_BINS), dtype=np.float32)
            left = max(0, local - p10.WINDOW_BINS + 1)
            valid = normalized[:, left : local + 1]
            window[:, -valid.shape[1] :] = valid
            inputs.append(window)
            targets.append(velocity[start + local])
    prediction = []
    stacked = np.stack(inputs)
    for left in range(0, len(stacked), p10.INFERENCE_BATCH):
        output = p10.model_predict(model, stacked[left : left + p10.INFERENCE_BATCH], device)
        prediction.append(output[:, -1] * target_std + target_mean)
    return np.stack(targets).astype(np.float32), np.concatenate(prediction).astype(np.float32)


def main() -> None:
    torch.set_num_threads(4)
    device = torch.device("cpu")
    model, checkpoint, replay = parity.load_candidate(SESSION, device)
    data = load_session(SESSION_BY_NAME[SESSION])
    counts_all, velocity = aggregate_40ms(data)
    bounds = binned_reach_bounds(data)
    _, _, test_reaches = split_fold(
        make_fold_indices(eligible_reaches(data)), int(checkpoint["fold"]) - 1
    )
    channels = np.asarray(checkpoint["selected_channel_indices"], dtype=np.int64)
    counts = counts_all[channels].astype(np.float32)

    indices, target, reference_prediction = p10.phase7_reference(
        model, counts, velocity, bounds, test_reaches, checkpoint, device
    )
    keep = indices >= p10.CALIBRATION_BINS - 1
    indices, target, reference_prediction = indices[keep], target[keep], reference_prediction[keep]
    selected = set(int(value) for value in indices)

    reach_target, reach_rolling_prediction = rolling_with_reach_resets(
        model, counts, velocity, bounds, test_reaches, selected, checkpoint, device
    )
    continuous = p10.continuous_features(counts)
    train_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32).reshape(p10.FEATURES, 1)
    train_std = np.asarray(checkpoint["feature_std"], dtype=np.float32).reshape(p10.FEATURES, 1)
    normalized = ((continuous - train_mean) / train_std).astype(np.float32)
    continuous_prediction = []
    for left in range(0, len(indices), p10.INFERENCE_BATCH):
        inputs = p10.rolling_inputs(normalized, indices[left : left + p10.INFERENCE_BATCH])
        output = p10.model_predict(model, inputs, device)[:, -1]
        target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32).reshape(2)
        target_std = np.asarray(checkpoint["target_std"], dtype=np.float32).reshape(2)
        continuous_prediction.append(output * target_std + target_mean)
    continuous_prediction = np.concatenate(continuous_prediction).astype(np.float32)

    floor = np.asarray(checkpoint["feature_std_floor"], dtype=np.float32)
    deployment_indices, deployment_target, deployment_prediction, _ = p10.firmware_replay(
        model, continuous, velocity, indices, floor, checkpoint, device
    )
    assert np.array_equal(indices, deployment_indices)
    assert np.allclose(target, deployment_target)

    variants = [
        ("phase7_chunked_reach_reset_train_stats", target, reference_prediction),
        ("rolling_reach_reset_train_stats", reach_target, reach_rolling_prediction),
        ("rolling_continuous_ewma_train_stats", target, continuous_prediction),
        ("rolling_continuous_ewma_60s_calibration", target, deployment_prediction),
    ]
    rows = []
    previous = None
    for name, truth, prediction in variants:
        metrics = phase12.metric_values(truth, prediction)
        rows.append(
            {
                "variant": name,
                **metrics,
                "sequential_delta_r2_mean": None if previous is None else metrics["r2_mean"] - previous,
            }
        )
        previous = metrics["r2_mean"]

    assert np.isclose(rows[0]["r2_mean"], replay["replay"]["phase7_reference_same_bins"]["r2_mean"], atol=2e-6)
    assert np.isclose(rows[-1]["r2_mean"], replay["replay"]["firmware_policy_same_bins"]["r2_mean"], atol=2e-6)
    payload = {
        "session": SESSION,
        "selected_fold": int(checkpoint["fold"]),
        "test_bins": len(indices),
        "decomposition_note": "Sequential/order-dependent counterfactuals; deltas are not causal Shapley effects.",
        "variants": rows,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
