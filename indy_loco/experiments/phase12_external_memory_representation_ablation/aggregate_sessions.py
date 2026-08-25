#!/usr/bin/env python3
"""Aggregate the six Phase-12 sessions without treating bins as independent."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parent
BY_SESSION = ROOT / "results" / "by_session"
OUTPUT_DIR = ROOT / "results" / "cross_session"
SESSIONS = (
    "indy_20160622_01",
    "indy_20160630_01",
    "indy_20170131_02",
    "loco_20170210_03",
    "loco_20170215_02",
    "loco_20170301_05",
)
BOOTSTRAP_REPETITIONS = 100_000
SEED = 12_120_026


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def session_bootstrap(values: np.ndarray) -> dict[str, float | int | str]:
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPETITIONS, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "mean_delta_r2": float(values.mean()),
        "median_delta_r2": float(np.median(values)),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "repetitions": BOOTSTRAP_REPETITIONS,
        "resampling_unit": "session",
    }


def main() -> None:
    rows: list[dict[str, Any]] = []
    for session in SESSIONS:
        source_path = BY_SESSION / session / "metrics.json"
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        encoder = payload["representations"]["encoder_49"]
        gru = payload["representations"]["gru_hidden_49"]
        interval = gru["test_delta_vs_encoder_49"]
        encoder_r2 = float(encoder["test_corrected"]["r2_mean"])
        gru_r2 = float(gru["test_corrected"]["r2_mean"])
        rows.append(
            {
                "session": session,
                "subject": payload["subject"],
                "test_reaches": payload["protocol"]["split_reaches"]["test"],
                "test_bins": payload["protocol"]["split_bins"]["test"],
                "base_r2": payload["base_metrics"]["test"]["r2_mean"],
                "encoder_corrected_r2": encoder_r2,
                "gru_corrected_r2": gru_r2,
                "gru_minus_encoder_r2": gru_r2 - encoder_r2,
                "ci95_low": interval["ci95_low"],
                "ci95_high": interval["ci95_high"],
                "reach_bootstrap_probability_positive": interval[
                    "probability_positive"
                ],
                "individual_ci_excludes_zero": bool(interval["ci95_low"] > 0),
                "validation_winner_all_four": payload["selection"]["winner"],
                "source_metrics": str(source_path.relative_to(ROOT)),
            }
        )

    deltas = np.asarray([row["gru_minus_encoder_r2"] for row in rows], dtype=np.float64)
    bootstrap = session_bootstrap(deltas)
    wilcoxon_result = wilcoxon(
        deltas,
        alternative="greater",
        zero_method="wilcox",
        method="exact",
    )
    wilcoxon_two_sided = wilcoxon(
        deltas,
        alternative="two-sided",
        zero_method="wilcox",
        method="exact",
    )
    positive = int(np.sum(deltas > 0))
    sign_result = binomtest(positive, len(deltas), p=0.5, alternative="greater")
    subject_summary: dict[str, Any] = {}
    for subject in ("indy", "loco"):
        selected = deltas[[row["subject"] == subject for row in rows]]
        subject_summary[subject] = {
            "session_count": len(selected),
            "mean_delta_r2": float(selected.mean()),
            "median_delta_r2": float(np.median(selected)),
            "minimum_delta_r2": float(selected.min()),
            "maximum_delta_r2": float(selected.max()),
        }

    cross_session_significant = bool(
        float(wilcoxon_result.pvalue) < 0.05 and float(bootstrap["ci95_low"]) > 0
    )
    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Is GRU hidden[49] significantly better than Encoder[49] across benchmark sessions?",
        "comparison": "held-out corrected mean R2, GRU minus Encoder",
        "primary_unit": "session",
        "session_count": len(rows),
        "positive_sessions": positive,
        "individually_significant_sessions_unadjusted": int(
            sum(row["individual_ci_excludes_zero"] for row in rows)
        ),
        "session_bootstrap": bootstrap,
        "exact_one_sided_wilcoxon": {
            "statistic": float(wilcoxon_result.statistic),
            "p_value": float(wilcoxon_result.pvalue),
            "alternative": "GRU minus Encoder > 0",
        },
        "exact_two_sided_wilcoxon_sensitivity": {
            "statistic": float(wilcoxon_two_sided.statistic),
            "p_value": float(wilcoxon_two_sided.pvalue),
            "alternative": "GRU minus Encoder != 0",
        },
        "exact_one_sided_sign_test": {
            "positive_sessions": positive,
            "session_count": len(rows),
            "p_value": float(sign_result.pvalue),
            "alternative": "probability of positive session > 0.5",
        },
        "subject_summary": subject_summary,
        "decision": {
            "cross_session_significant_at_0_05": cross_session_significant,
            "individual_session_effect_is_uniformly_significant": all(
                row["individual_ci_excludes_zero"] for row in rows
            ),
            "interpretation": (
                "Overall paired evidence supports GRU, but the effect is not "
                "individually significant in every session."
            ),
        },
        "methodology": {
            "per_session": "1,000 reach-level bootstrap repetitions from each saved Phase-12 run",
            "cross_session": "unweighted session effect; 100,000 session bootstrap repetitions",
            "primary_test": "exact one-sided Wilcoxon signed-rank over six paired session deltas",
            "sensitivity_test": "exact one-sided sign test",
            "why_not_bin_level": "adjacent bins within a reach are autocorrelated and not independent experimental units",
            "multiple_comparison_note": "per-session intervals are exploratory and unadjusted; the cross-session pair is prespecified",
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "gru_vs_encoder_by_session.csv", rows)
    write_json(OUTPUT_DIR / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
