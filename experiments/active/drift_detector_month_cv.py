#!/usr/bin/env python
"""Nested month-level drift-detector evaluation with streaming-safe inputs.

Differences from archived ``iter34_detector_cv.py``:

* a fixed 60-second observation prefix fits every session's normalization;
* the scored suffix never contributes to its own normalization;
* no output is considered valid during the 60-second normalization warm-up;
* outer-month sessions never participate in epoch selection;
* a candidate drift threshold is selected from inner validation sessions only;
* movement axes are fixed from the dataset coordinate definition, not held-out labels;
* shared data/feature/drift code comes from ``src/intent_decoder``.

This script writes a small versionable JSON result. It does not promote a model.
"""
from __future__ import annotations

import json
import re
import sys
import time
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
    resolve_source_name,
    top_firing_channels,
    window_arrays,
)
from src.intent_decoder.drift import prediction_std_ratio, top_channel_overlap
from src.intent_decoder.features.causal import multiscale_counts
from src.intent_decoder.model.tcn_gru import causal_config, r2
from src.intent_decoder.training import run

BIN_S = 0.040
WINDOW_BINS = 50
OBSERVATION_SECONDS = 60
OBSERVATION_BINS = int(OBSERVATION_SECONDS / BIN_S)
N_CHANNELS = 32
ALPHAS = (1.0, 0.1)
AXES = np.array([1, 2])
SEED = 42
BAD_R2 = 0.4
OUT = ROOT / "results" / "metrics" / "drift_detector_month_cv.json"


def month_of(session: str) -> str:
    source = resolve_source_name(session)
    match = re.search(r"(\d{4})(\d{2})\d{2}", source)
    if not match:
        raise ValueError(f"No date in session name: {session}")
    return f"{match.group(1)}{match.group(2)}"


def stack(trials: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    if not trials:
        raise ValueError("No complete windows available")
    return (
        np.stack([trial["e"] for trial in trials]).astype(np.float32),
        np.stack([trial["vel"] for trial in trials]).astype(np.float32),
    )


def session_features(counts: np.ndarray, channels: np.ndarray) -> np.ndarray:
    return multiscale_counts(counts[channels], ALPHAS)


def prefix_parts(
    loaded: dict[str, tuple[np.ndarray, np.ndarray]], session: str, channels: np.ndarray
) -> tuple[list[dict], list[dict]]:
    """Observation and scored windows normalized from observation samples only."""
    features = session_features(loaded[session][0], channels)
    if features.shape[1] <= OBSERVATION_BINS + WINDOW_BINS:
        raise ValueError(f"{session} is too short for a {OBSERVATION_SECONDS}s prefix")
    stats = fit_feature_stats(features[:, :OBSERVATION_BINS])
    normalized = apply_feature_stats(features, stats)
    observation = window_arrays(
        normalized[:, :OBSERVATION_BINS],
        loaded[session][1][:OBSERVATION_BINS],
        AXES,
        window_bins=WINDOW_BINS,
    )
    scored = window_arrays(
        normalized,
        loaded[session][1],
        AXES,
        window_bins=WINDOW_BINS,
        start_bin=OBSERVATION_BINS,
    )
    return observation, scored


def predict(net, trials: list[dict], target_norm: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    import torch

    x, y = stack(trials)
    mean, std = target_norm
    net.eval()
    with torch.no_grad():
        prediction = net(torch.tensor(x)).numpy() * std + mean
    return y.reshape(-1, y.shape[-1]), prediction.reshape(-1, prediction.shape[-1])


def balanced_accuracy(truth_bad: np.ndarray, predicted_bad: np.ndarray) -> float:
    positives = truth_bad.sum()
    negatives = (~truth_bad).sum()
    if positives == 0 or negatives == 0:
        return float("nan")
    sensitivity = (predicted_bad & truth_bad).sum() / positives
    specificity = ((~predicted_bad) & (~truth_bad)).sum() / negatives
    return float((sensitivity + specificity) / 2)


def select_threshold(records: list[dict]) -> float | None:
    """Select pred-std threshold on inner validation records only."""
    values = np.array([record["pred_std_ratio"] for record in records])
    truth = np.array([record["r2"] < BAD_R2 for record in records])
    if len(np.unique(truth)) < 2:
        return None
    candidates = np.r_[-np.inf, (np.sort(values)[:-1] + np.sort(values)[1:]) / 2, np.inf]
    scores = [balanced_accuracy(truth, values < threshold) for threshold in candidates]
    return float(candidates[int(np.nanargmax(scores))])


def score_session(net, target_norm, loaded, session, channels) -> dict:
    observation, scored = prefix_parts(loaded, session, channels)
    _, observation_prediction = predict(net, observation, target_norm)
    target, prediction = predict(net, scored, target_norm)
    observation_bins = loaded[session][0][:, :OBSERVATION_BINS]
    return {
        "session": session,
        "r2": float(r2(target, prediction).mean()),
        "pred_std_ratio": prediction_std_ratio(observation_prediction, target_norm[1]),
        "overlap_topN": top_channel_overlap(observation_bins, channels, len(channels)),
    }


def main() -> None:
    manifest = load_session_manifest()
    sessions = manifest["experiment_pool"]
    print(f"Loading {len(sessions)} sessions from data/raw/indy_loco ...", flush=True)
    loaded = {session: load_model_data(session) for session in sessions}
    folds: dict[str, list[str]] = {}
    for session in sessions:
        folds.setdefault(month_of(session), []).append(session)

    output = {"config": {"observation_seconds": OBSERVATION_SECONDS,
                         "n_channels": N_CHANNELS, "seed": SEED}, "folds": {}}
    all_outer_records = []
    for held_month, outer_sessions in sorted(folds.items()):
        started = time.time()
        candidates = [session for session in sessions if session not in outer_sessions]
        by_month: dict[str, list[str]] = {}
        for session in candidates:
            by_month.setdefault(month_of(session), []).append(session)
        # One deterministic validation session per available training month.
        validation_sessions = [sorted(group)[-1] for group in by_month.values()]
        training_sessions = [session for session in candidates if session not in validation_sessions]
        training_loaded = {session: loaded[session] for session in training_sessions}
        channels = top_firing_channels(training_loaded, N_CHANNELS)

        # Training follows the same deployment rule: collect a past-only prefix,
        # freeze its statistics, discard warm-up outputs, then train on the suffix.
        train_trials = [trial for session in training_sessions
                        for trial in prefix_parts(loaded, session, channels)[1]]
        eval_by = {session: prefix_parts(loaded, session, channels)[1]
                   for session in validation_sessions}
        test_by = {session: prefix_parts(loaded, session, channels)[1]
                   for session in outer_sessions}
        result = run(
            {"train": train_trials, "eval": eval_by, "test": test_by},
            causal_config(),
            seeds=(SEED,),
            ret_net=True,
        )
        net, target_norm = result["net"], result["norm"]
        inner_records = [score_session(net, target_norm, loaded, s, channels)
                         for s in validation_sessions]
        threshold = select_threshold(inner_records)
        outer_records = [score_session(net, target_norm, loaded, s, channels)
                         for s in outer_sessions]
        for record in outer_records:
            record["predicted_bad"] = None if threshold is None else record["pred_std_ratio"] < threshold
            record["actually_bad"] = record["r2"] < BAD_R2
        all_outer_records.extend(outer_records)
        output["folds"][held_month] = {
            "training_sessions": training_sessions,
            "validation_sessions": validation_sessions,
            "outer_sessions": outer_sessions,
            "channels": channels.tolist(),
            "threshold_from_inner_validation": threshold,
            "inner_records": inner_records,
            "outer_records": outer_records,
            "elapsed_s": time.time() - started,
        }
        print(f"{held_month}: {len(outer_sessions)} outer sessions; threshold={threshold}", flush=True)

    valid = [record for record in all_outer_records if record["predicted_bad"] is not None]
    if valid:
        truth = np.array([record["actually_bad"] for record in valid])
        prediction = np.array([record["predicted_bad"] for record in valid])
        output["summary"] = {
            "sessions_with_inner_threshold": len(valid),
            "balanced_accuracy": balanced_accuracy(truth, prediction),
            "false_positives": int((prediction & ~truth).sum()),
            "false_negatives": int((~prediction & truth).sum()),
            "mean_r2": float(np.mean([record["r2"] for record in all_outer_records])),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
