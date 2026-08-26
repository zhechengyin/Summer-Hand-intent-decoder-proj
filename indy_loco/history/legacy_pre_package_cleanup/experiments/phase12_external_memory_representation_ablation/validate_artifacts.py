#!/usr/bin/env python3
"""Validate Phase-12 result and memory-library artifact contracts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[2]
RESULT_DIR = ROOT / "results"
MODEL_DIR = REPOSITORY_ROOT / "indy_loco" / "models" / "large" / "indy_20160622_01"
BY_SESSION = RESULT_DIR / "by_session"
CROSS_SESSION = RESULT_DIR / "cross_session"

SESSIONS = (
    "indy_20160622_01",
    "indy_20160630_01",
    "indy_20170131_02",
    "loco_20170210_03",
    "loco_20170215_02",
    "loco_20170301_05",
)

REPRESENTATIONS = (
    "encoder_49",
    "gru_hidden_49",
    "encoder_gru_49",
    "encoder_50step_mean",
)


def main() -> None:
    deltas: list[float] = []
    for session in SESSIONS:
        session_results = BY_SESSION / session
        metrics = json.loads((session_results / "metrics.json").read_text(encoding="utf-8"))
        with (session_results / "representation_comparison.csv").open(
            newline="", encoding="utf-8"
        ) as source:
            rows = list(csv.DictReader(source))
        assert metrics["status"] == "complete"
        assert metrics["session"] == session
        assert metrics["checkpoint"]["fold"] == 1
        assert metrics["protocol"]["pca_fit"] == "train only"
        assert metrics["protocol"]["bank_entries"] == "train only"
        assert len(rows) == len(REPRESENTATIONS)
        assert {row["representation"] for row in rows} == set(REPRESENTATIONS)
        winner = max(rows, key=lambda row: float(row["validation_r2"]))["representation"]
        assert winner == metrics["selection"]["winner"]
        entry_count = metrics["protocol"]["split_bins"]["train"]
        model_dir = MODEL_DIR.parent / session
        for name in REPRESENTATIONS:
            with np.load(model_dir / f"phase12_{name}.memlib", allow_pickle=False) as bank:
                assert bank["schema"].item() == "phase12_pc_memlib_v1"
                assert bank["representation"].item() == name
                assert bank["keys_int8"].shape == (entry_count, 64)
                assert bank["keys_int8"].dtype == np.int8
                assert bank["residual_fp16"].shape == (entry_count, 2)
                assert bank["residual_fp16"].dtype == np.float16
                assert bank["context_basis"].shape == (576, 32)
                expected_source = 128 if name == "encoder_gru_49" else 64
                assert bank["representation_basis"].shape == (expected_source, 32)
                for key in bank.files:
                    if bank[key].dtype.kind in "fc":
                        assert np.isfinite(bank[key]).all()
        encoder = metrics["representations"]["encoder_49"]["test_corrected"]["r2_mean"]
        gru = metrics["representations"]["gru_hidden_49"]["test_corrected"]["r2_mean"]
        deltas.append(float(gru - encoder))

    cross = json.loads((CROSS_SESSION / "summary.json").read_text(encoding="utf-8"))
    assert cross["session_count"] == len(SESSIONS)
    assert cross["positive_sessions"] == sum(value > 0 for value in deltas)
    assert np.isclose(cross["session_bootstrap"]["mean_delta_r2"], np.mean(deltas))
    assert cross["exact_one_sided_wilcoxon"]["p_value"] < 0.05
    assert cross["exact_two_sided_wilcoxon_sensitivity"]["p_value"] < 0.05
    assert (CROSS_SESSION / "gru_vs_encoder_cross_session.png").stat().st_size > 10_000
    assert (CROSS_SESSION / "report.md").stat().st_size > 1_000
    print("Phase-12 validation passed: 6 sessions, 24 train-only memlibs, aggregate statistics, chart, and report")


if __name__ == "__main__":
    main()
