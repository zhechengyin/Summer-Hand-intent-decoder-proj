#!/usr/bin/env python3
"""Independently validate the six-session deployment-parity artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
RESULTS = ROOT / "results" / "deployment_parity"
MODEL_ROOT = REPO / "indy_loco" / "models" / "midsize"
LARGE_ROOT = REPO / "indy_loco" / "models" / "large"


def main() -> None:
    with (RESULTS / "deployment_parity_ab.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    assert len(rows) == summary["sessions"] == 6

    absent, ready, deltas, selection = [], [], [], []
    for row in rows:
        session = row["session"]
        metrics = json.loads(
            (RESULTS / "by_session" / session / "metrics.json").read_text(encoding="utf-8")
        )
        replay = json.loads(
            (MODEL_ROOT / session / "deployment_replay.json").read_text(encoding="utf-8")
        )
        expected = replay["replay"]["firmware_policy_same_bins"]
        for key in ("r2_x", "r2_y", "r2_mean", "mse"):
            assert np.isclose(metrics["bank_absent"][key], expected[key], rtol=0, atol=2e-6)
        assert metrics["protocol"]["train_only_bank_and_pca"]
        assert metrics["protocol"]["validation_only_retrieval_tuning"]
        assert metrics["ready_minus_absent"]["reach_bootstrap"]["ci95_low"] > 0

        memlib = LARGE_ROOT / session / "phase12_deployment_parity_gru_hidden_49.memlib"
        assert memlib.stat().st_size == metrics["memlib"]["bytes"]
        with np.load(memlib, allow_pickle=False) as archive:
            assert str(archive["schema"]) == "phase12_pc_memlib_v1"
            assert str(archive["representation"]) == "deployment_parity_gru_hidden_49"
            assert archive["keys_int8"].dtype == np.int8
            assert archive["keys_int8"].shape == (metrics["retrieval"]["bank_entries"], 64)
            assert archive["residual_fp16"].dtype == np.float16
            assert archive["residual_fp16"].shape == (metrics["retrieval"]["bank_entries"], 2)
            digest = hashlib.sha256((MODEL_ROOT / session / "deployment_candidate.pt").read_bytes()).hexdigest()
            assert str(archive["checkpoint_sha256"]) == digest

        absent.append(float(row["absent_r2"]))
        ready.append(float(row["ready_gru_r2"]))
        deltas.append(float(row["ready_minus_absent_r2"]))
        selection.append(float(row["selection_test_r2_mean"]))

    recomputed = {
        "mean_selection_test_r2": np.mean(selection),
        "mean_absent_r2": np.mean(absent),
        "mean_ready_gru_r2": np.mean(ready),
        "mean_ready_minus_absent_r2": np.mean(deltas),
    }
    for key, value in recomputed.items():
        assert np.isclose(value, summary[key], rtol=0, atol=1e-12)
    assert np.allclose(np.asarray(ready) - np.asarray(absent), deltas, rtol=0, atol=1e-12)
    assert summary["positive_sessions"] == 6
    assert summary["individually_significant_sessions"] == 6
    print(json.dumps({"status": "passed", "sessions": 6, **recomputed}, indent=2))


if __name__ == "__main__":
    main()
