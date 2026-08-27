#!/usr/bin/env python3
"""Audit Phase 13 round-3 fold metrics and parameter changes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
INDY_ROOT = HERE.parents[1]
DEFAULT_RUN_ROOT = HERE / "results" / "rolling_retrain" / "final_30fold"
PHASE7_CHECKPOINTS = (
    INDY_ROOT
    / "history"
    / "results"
    / "indy"
    / "phase7_ann_vs_snn_fivefold"
    / "checkpoints"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    return parser.parse_args()


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distribution(values: np.ndarray) -> dict[str, Any]:
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std_sample": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "positive_count": int(np.sum(values > 0)),
        "negative_count": int(np.sum(values < 0)),
        "zero_count": int(np.sum(values == 0)),
    }


def parameter_group(name: str) -> str:
    if name.startswith("gru."):
        return "gru"
    if name.startswith("head."):
        return "head"
    return "encoder_tcn"


def main() -> None:
    import torch

    args = parse_args()
    run_root = args.run_root.resolve()
    fold_path = run_root / "phase13_round3_folds.csv"
    summary_path = run_root / "phase13_round3_summary.csv"
    if not fold_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"Incomplete round-3 output: {run_root}")
    with fold_path.open(newline="", encoding="utf-8") as source:
        fold_rows = list(csv.DictReader(source))
    with summary_path.open(newline="", encoding="utf-8") as source:
        summary_rows = list(csv.DictReader(source))
    if len(fold_rows) != 30:
        raise ValueError(f"Expected 30 folds, found {len(fold_rows)}")

    weight_rows: list[dict[str, Any]] = []
    for row in fold_rows:
        session = row["session"]
        fold = int(row["fold"])
        old_path = PHASE7_CHECKPOINTS / f"{session}_fold{fold}.pt"
        new_path = run_root / "checkpoints" / f"{session}_fold{fold}.pt"
        old = torch.load(old_path, map_location="cpu", weights_only=False)[
            "model_state"
        ]
        new = torch.load(new_path, map_location="cpu", weights_only=False)[
            "model_state"
        ]
        accumulators = {
            group: {"delta_squared": 0.0, "base_squared": 0.0}
            for group in ("encoder_tcn", "gru", "head")
        }
        for name, old_value in old.items():
            if name not in new or old_value.shape != new[name].shape:
                raise ValueError(f"{session} fold {fold}: state mismatch at {name}")
            group = parameter_group(name)
            base = old_value.double()
            delta = new[name].double() - base
            accumulators[group]["delta_squared"] += float(torch.sum(delta**2))
            accumulators[group]["base_squared"] += float(torch.sum(base**2))
        weight_rows.append(
            {
                "session": session,
                "subject": row["subject"],
                "fold": fold,
                **{
                    f"{group}_relative_l2_change": (
                        values["delta_squared"] / values["base_squared"]
                    )
                    ** 0.5
                    for group, values in accumulators.items()
                },
            }
        )

    numeric_fields = (
        "phase7_reach_local_r2",
        "phase7_continuous_training_norm_r2",
        "phase7_7min_rolling_r2",
        "retrained_7min_rolling_r2",
        "rolling_only_delta",
        "calibration_delta",
        "retraining_gain",
        "net_delta_vs_phase7",
    )
    fold_distributions = {
        field: distribution(
            np.asarray([float(row[field]) for row in fold_rows], dtype=np.float64)
        )
        for field in numeric_fields
    }
    session_rows = [
        row for row in summary_rows if row["session"] != "overall_fold_macro"
    ]
    session_tests = {}
    for field in ("retraining_gain_mean", "net_delta_vs_phase7_mean"):
        values = np.asarray(
            [float(row[field]) for row in session_rows], dtype=np.float64
        )
        test = wilcoxon(values, alternative="two-sided", method="exact")
        session_tests[field] = {
            **distribution(values),
            "wilcoxon_statistic": float(test.statistic),
            "wilcoxon_exact_two_sided_p": float(test.pvalue),
            "unit": "session five-fold mean",
        }
    weight_distributions = {}
    for group in ("encoder_tcn", "gru", "head"):
        field = f"{group}_relative_l2_change"
        weight_distributions[group] = distribution(
            np.asarray([float(row[field]) for row in weight_rows])
        )

    payload = {
        "phase": "phase13_round3_rolling_retrain_analysis",
        "status": "complete",
        "run_root": str(run_root),
        "fold_count": len(fold_rows),
        "session_count": len(session_rows),
        "fold_distributions": fold_distributions,
        "primary_session_level_tests": session_tests,
        "weight_relative_l2_change": weight_distributions,
        "interpretation": {
            "retraining_gain_positive_folds": fold_distributions["retraining_gain"][
                "positive_count"
            ],
            "retrained_above_reach_local_folds": fold_distributions[
                "net_delta_vs_phase7"
            ]["positive_count"],
            "independence_caveat": (
                "Folds within a session are correlated; session-level paired tests "
                "are primary and fold counts describe consistency only."
            ),
        },
    }
    write_csv(run_root / "phase13_round3_weight_audit.csv", weight_rows)
    write_json_atomic(run_root / "phase13_round3_analysis.json", payload)
    print("=== Phase 13 round-3 analysis complete ===")
    print(
        f"retraining gain={fold_distributions['retraining_gain']['mean']:+.6f}; "
        f"positive folds={fold_distributions['retraining_gain']['positive_count']}/30"
    )
    print(
        "session-level p="
        f"{session_tests['retraining_gain_mean']['wilcoxon_exact_two_sided_p']:.5f}"
    )
    print(f"outputs: {run_root}")


if __name__ == "__main__":
    main()
