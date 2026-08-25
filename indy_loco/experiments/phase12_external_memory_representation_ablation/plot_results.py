#!/usr/bin/env python3
"""Render the Phase-12 comparison figure from the saved CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "results" / "representation_comparison.csv"
OUTPUT_PATH = ROOT / "results" / "representation_comparison.png"

LABELS = {
    "encoder_49": "Encoder[49]",
    "gru_hidden_49": "GRU hidden[49]",
    "encoder_gru_49": "Encoder + GRU",
    "encoder_50step_mean": "Encoder mean(50)",
}


def main() -> None:
    with CSV_PATH.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    names = [LABELS[row["representation"]] for row in rows]
    base = np.array([float(row["test_base_r2"]) for row in rows])
    corrected = np.array([float(row["test_corrected_r2"]) for row in rows])
    delta = corrected - base
    low = np.array([float(row["delta_ci95_low"]) for row in rows])
    high = np.array([float(row["delta_ci95_high"]) for row in rows])
    residual_r2 = np.array([float(row["residual_r2"]) for row in rows])

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    x = np.arange(len(rows))
    axes[0].axhline(base[0], color="#555555", linestyle="--", linewidth=1.5,
                    label=f"Frozen Midsize: {base[0]:.3f}")
    axes[0].bar(x, corrected, color=["#4C78A8", "#F58518", "#72B7B2", "#54A24B"])
    axes[0].set_ylim(0.76, 0.83)
    axes[0].set_ylabel("Held-out test R² (mean x/y)")
    axes[0].set_xticks(x, names, rotation=18, ha="right")
    axes[0].legend(frameon=False, loc="lower right")
    for index, value in enumerate(corrected):
        axes[0].text(index, value + 0.0012, f"{value:.3f}", ha="center", fontsize=9)

    error = np.vstack((delta - low, high - delta))
    axes[1].errorbar(
        x,
        delta,
        yerr=error,
        fmt="o",
        markersize=8,
        capsize=4,
        color="#E45756",
        ecolor="#E45756",
    )
    axes[1].axhline(0.0, color="#555555", linewidth=1)
    axes[1].set_ylabel("ΔR² vs frozen Midsize (95% reach bootstrap CI)")
    axes[1].set_xticks(x, names, rotation=18, ha="right")
    for index, (change, consistency) in enumerate(zip(delta, residual_r2, strict=True)):
        axes[1].text(index, high[index] + 0.002, f"resid. R²={consistency:.3f}",
                     ha="center", fontsize=8)

    figure.suptitle("Phase 12 — query representation ablation | indy_20160622_01")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
