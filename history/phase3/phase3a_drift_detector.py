#!/usr/bin/env python3
"""Archived Phase 3a: pre-January drift-detector development.

This experiment intentionally does not load velocity labels or the January test
split.  It measures held-month false alarms and sensitivity to explicitly
synthetic recording faults.  Synthetic faults are engineering stress tests,
not substitutes for future independently collected failed sessions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.indy_32ch.drift_detector import (  # noqa: E402
    DetectorConfig,
    DriftDetector,
    assert_pre_january,
    session_month,
)
from models.indy_32ch.input_pipeline import (  # noqa: E402
    load_session_manifest,
    processed_session_path,
)

MODEL_CONFIG = ROOT / "configs" / "indy_32ch.yaml"
DEFAULT_OUTPUT = ROOT / "results" / "indy" / "phase3a_drift_detector"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--warning-quantile", type=float, default=0.99)
    parser.add_argument(
        "--skip-stress-tests",
        action="store_true",
        help="Only score intact held-month sessions.",
    )
    return parser.parse_args()


def load_selected_channels() -> np.ndarray:
    with MODEL_CONFIG.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    channels = np.asarray(config["input"]["selected_zero_based"], dtype=np.int64)
    if channels.shape != (32,) or np.unique(channels).size != 32:
        raise ValueError("Frozen model configuration must contain 32 unique channels")
    return channels


def load_pre_january_counts(channels: np.ndarray) -> dict[str, np.ndarray]:
    """Load counts only from train and validation; never resolve a test path."""
    manifest = load_session_manifest()
    split = manifest["chronological_split"]
    names = list(split["train"]) + list(split["validation"])
    assert_pre_january(names)
    if len(names) != 33 or set(names) & set(split["test"]):
        raise RuntimeError("Expected exactly 33 pre-January development sessions")

    sessions: dict[str, np.ndarray] = {}
    for name in names:
        artifact = processed_session_path(name)
        if artifact.parent.name == "test":
            raise RuntimeError(f"Refusing to load test artifact: {artifact}")
        with np.load(artifact, allow_pickle=False) as data:
            sessions[name] = data["counts"][channels].astype(np.float32)
    return sessions


def stable_rng(seed: int, session_name: str, condition: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{session_name}:{condition}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def synthetic_fault(
    counts: np.ndarray,
    condition: str,
    *,
    seed: int,
    session_name: str,
) -> np.ndarray:
    """Apply a known recording fault without changing the held-month split."""
    rng = stable_rng(seed, session_name, condition)
    values = np.asarray(counts, dtype=np.int64).copy()
    if condition == "spike_thinning_50pct":
        return rng.binomial(values, 0.50).astype(np.float32)
    if condition == "spike_thinning_75pct":
        return rng.binomial(values, 0.25).astype(np.float32)
    if condition == "channel_dropout_25pct":
        dropped = rng.choice(values.shape[0], size=values.shape[0] // 4, replace=False)
        values[dropped] = 0
        return values.astype(np.float32)
    if condition == "channel_permutation":
        return values[rng.permutation(values.shape[0])].astype(np.float32)
    if condition == "thinning_65pct_plus_dropout_25pct":
        thinned = rng.binomial(values, 0.35)
        dropped = rng.choice(values.shape[0], size=values.shape[0] // 4, replace=False)
        thinned[dropped] = 0
        return thinned.astype(np.float32)
    raise KeyError(condition)


def score_row(
    detector: DriftDetector,
    session_name: str,
    condition: str,
    counts: np.ndarray,
    held_month: str,
) -> dict:
    score = detector.score(counts)
    row = {
        "session": session_name,
        "held_month": held_month,
        "condition": condition,
        "is_synthetic_fault": condition != "normal",
        **asdict(score),
    }
    for metric in (
        "pattern_distance",
        "robust_rate_distance",
        "absolute_log_rate_ratio",
        "unexpected_silent_channels",
        "global_mindful_kld",
        "multi_reference_mindful_kld",
    ):
        threshold = detector.thresholds[metric]
        value = float(row[metric])
        row[f"{metric}_threshold"] = threshold
        row[f"{metric}_to_threshold"] = (
            value / threshold if threshold > 0 else float(value > 0)
        )
    return row


def summarize(rows: pd.DataFrame) -> list[dict]:
    baselines = {
        "A_simple_multi_reference": "simple_decision",
        "B_global_mindful": "global_mindful_decision",
        "C_multi_reference_mindful": "multi_mindful_decision",
        "combined_A_plus_C": "combined_decision",
    }
    output = []
    for condition, condition_rows in rows.groupby("condition", sort=False):
        for baseline, column in baselines.items():
            decisions = condition_rows[column]
            output.append(
                {
                    "condition": condition,
                    "baseline": baseline,
                    "sessions": int(len(condition_rows)),
                    "flag_rate": float((decisions != "pass").mean()),
                    "abstain_rate": float((decisions == "abstain").mean()),
                }
            )
    return output


def save_figure(rows: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    normal = rows[rows["condition"] == "normal"].copy()
    conditions = list(summary["condition"].drop_duplicates())
    baselines = list(summary["baseline"].drop_duplicates())

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for month, group in normal.groupby("held_month"):
        axes[0].scatter(
            group["multi_reference_mindful_kld_to_threshold"],
            group["combined_evidence_count"],
            label=month,
            s=48,
            alpha=0.85,
        )
    axes[0].axvline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].axhline(2.0, color="black", linestyle=":", linewidth=1)
    axes[0].set(
        xlabel="Multi-reference KLD / fold threshold",
        ylabel="Combined abnormal evidence count",
        title="Intact sessions held out by month",
    )
    axes[0].legend(fontsize=8, ncol=2)

    width = 0.18
    x = np.arange(len(conditions))
    for index, baseline in enumerate(baselines):
        values = (
            summary[summary["baseline"] == baseline]
            .set_index("condition")
            .loc[conditions, "flag_rate"]
            .to_numpy()
        )
        axes[1].bar(
            x + (index - 1.5) * width,
            values,
            width=width,
            label=baseline,
        )
    axes[1].set_xticks(x, [value.replace("_", "\n") for value in conditions])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Flag rate")
    axes[1].set_title("Held-month normal data and synthetic stress tests")
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not 0.5 < args.warning_quantile < 1.0:
        raise ValueError("--warning-quantile must be between 0.5 and 1.0")
    detector_config = DetectorConfig(
        n_components=args.n_components,
        warning_quantile=args.warning_quantile,
    )
    channels = load_selected_channels()
    sessions = load_pre_january_counts(channels)
    months = sorted({session_month(name) for name in sessions})
    if len(months) < 3:
        raise RuntimeError("Leave-one-month-out requires at least three months")

    print("=== Phase 3a: pre-January label-free drift detector ===")
    print(f"sessions={len(sessions)} | months={', '.join(months)}")
    print("January policy: FORBIDDEN and not loaded")
    print(
        "baselines: A=simple multi-reference | B=global KLD | "
        "C=multi-reference KLD | combined=A+C"
    )
    print(
        f"prefix={detector_config.observation_bins} bins / "
        f"{detector_config.observation_bins * detector_config.bin_seconds:.0f}s | "
        f"PCA={detector_config.n_components}D | "
        f"threshold quantile={detector_config.warning_quantile:.2f}"
    )

    conditions = ["normal"]
    if not args.skip_stress_tests:
        conditions += [
            "spike_thinning_50pct",
            "spike_thinning_75pct",
            "channel_dropout_25pct",
            "channel_permutation",
            "thinning_65pct_plus_dropout_25pct",
        ]

    rows: list[dict] = []
    for held_month in months:
        reference = {
            name: counts
            for name, counts in sessions.items()
            if session_month(name) != held_month
        }
        held = {
            name: counts
            for name, counts in sessions.items()
            if session_month(name) == held_month
        }
        detector = DriftDetector(detector_config).fit(reference)
        print(
            f"\nheld month={held_month} | reference={len(reference)} | held={len(held)}"
        )
        for name, counts in held.items():
            for condition in conditions:
                candidate = (
                    counts
                    if condition == "normal"
                    else synthetic_fault(
                        counts,
                        condition,
                        seed=args.seed,
                        session_name=name,
                    )
                )
                row = score_row(detector, name, condition, candidate, held_month)
                rows.append(row)
            normal = rows[-len(conditions)]
            print(
                f"  {name} | simple={normal['simple_decision']:7s} "
                f"multi-KLD={normal['multi_mindful_decision']:7s} "
                f"combined={normal['combined_decision']:7s} | "
                f"simple-ref={normal['matched_simple_reference']} "
                f"KLD-ref={normal['matched_mindful_reference']}"
            )

    rows_frame = pd.DataFrame(rows)
    summary_rows = summarize(rows_frame)
    summary_frame = pd.DataFrame(summary_rows)

    # Fit the deployable candidate only after all outer folds are complete.
    # Its thresholds still come exclusively from leave-one-out pre-January scores.
    final_detector = DriftDetector(detector_config).fit(sessions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "phase3a_drift_detector_scores.csv"
    summary_path = args.output_dir / "phase3a_drift_detector_summary.csv"
    metrics_path = args.output_dir / "phase3a_drift_detector_metrics.json"
    figure_path = args.output_dir / "phase3a_drift_detector_figure.png"
    artifact_path = args.output_dir / "phase3a_drift_detector_reference.npz"
    rows_frame.to_csv(rows_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    final_detector.save(artifact_path, selected_channels=channels)
    save_figure(rows_frame, summary_frame, figure_path)

    metrics = {
        "phase": "3a",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "development_only_not_january_validated",
        "january_loaded": False,
        "velocity_labels_loaded": False,
        "sessions": len(sessions),
        "months": months,
        "selected_channels_zero_based": channels.tolist(),
        "outer_validation": "leave_one_complete_month_out",
        "threshold_calibration": (
            "leave_one_complete_month_out within each outer-fold reference pool"
        ),
        "detector": final_detector.metadata(),
        "baseline_summary": summary_rows,
        "limitations": [
            "Synthetic faults test known perturbations but are not real failures.",
            "No decoder R2 is used in Phase 3a.",
            "January was previously observed and cannot tune this detector.",
            "Final thresholds require frozen prospective sessions.",
        ],
        "files": {
            "scores_csv": str(rows_path),
            "summary_csv": str(summary_path),
            "figure": str(figure_path),
            "reference_artifact": str(artifact_path),
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n=== Held-month summary ===")
    print(summary_frame.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nmetrics: {metrics_path}")
    print(f"figure:  {figure_path}")
    print(f"reference artifact: {artifact_path}")
    print("Interpretation: use this run to check false alarms and fault sensitivity.")
    print("Do not claim real failure prediction until prospective sessions exist.")


if __name__ == "__main__":
    main()
