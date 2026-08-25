#!/usr/bin/env python3
"""Render the session-level GRU-versus-Encoder interval plot."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "cross_session"


def main() -> None:
    with (RESULT_DIR / "gru_vs_encoder_by_session.csv").open(
        newline="", encoding="utf-8"
    ) as source:
        rows = list(csv.DictReader(source))
    summary = json.loads((RESULT_DIR / "summary.json").read_text(encoding="utf-8"))
    rows = rows[::-1]
    labels = [row["session"] for row in rows]
    delta = np.asarray([float(row["gru_minus_encoder_r2"]) for row in rows])
    low = np.asarray([float(row["ci95_low"]) for row in rows])
    high = np.asarray([float(row["ci95_high"]) for row in rows])
    colors = ["#4C78A8" if row["subject"] == "indy" else "#F58518" for row in rows]
    y = np.arange(len(rows))

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    for position, value, lower, upper, color in zip(
        y, delta, low, high, colors, strict=True
    ):
        axis.errorbar(
            value,
            position,
            xerr=[[value - lower], [upper - value]],
            fmt="o",
            color=color,
            ecolor=color,
            markersize=8,
            capsize=4,
        )
    mean = float(summary["session_bootstrap"]["mean_delta_r2"])
    mean_low = float(summary["session_bootstrap"]["ci95_low"])
    mean_high = float(summary["session_bootstrap"]["ci95_high"])
    axis.errorbar(
        mean,
        len(rows) + 0.25,
        xerr=[[mean - mean_low], [mean_high - mean]],
        fmt="D",
        color="#222222",
        ecolor="#222222",
        markersize=7,
        capsize=5,
    )
    axis.axvline(0, color="#666666", linewidth=1.2)
    axis.set_yticks(
        np.append(y, len(rows) + 0.25), labels + ["Unweighted session mean"]
    )
    axis.set_xlabel("Corrected R² difference: GRU hidden[49] − Encoder[49]")
    axis.set_title("Phase 12 across six held-out benchmark sessions")
    axis.text(
        0.02,
        0.97,
        f"Exact one-sided Wilcoxon p={summary['exact_one_sided_wilcoxon']['p_value']:.4f}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )
    axis.scatter([], [], color="#4C78A8", label="Indy")
    axis.scatter([], [], color="#F58518", label="Loco")
    axis.legend(frameon=False, loc="upper right")
    axis.set_ylim(-0.75, len(rows) + 1.0)
    figure.savefig(RESULT_DIR / "gru_vs_encoder_cross_session.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
