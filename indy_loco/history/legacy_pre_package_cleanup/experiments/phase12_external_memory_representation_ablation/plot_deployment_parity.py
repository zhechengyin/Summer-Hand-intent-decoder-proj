#!/usr/bin/env python3
"""Plot paired ABSENT/READY deployment replay and session-level uplift."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "deployment_parity"


def main() -> None:
    with (RESULTS / "deployment_parity_ab.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    labels = [row["session"].replace("_", "\n", 1) for row in rows]
    absent = np.asarray([float(row["absent_r2"]) for row in rows])
    ready = np.asarray([float(row["ready_gru_r2"]) for row in rows])
    delta = ready - absent
    low = np.asarray([float(row["ci95_low"]) for row in rows])
    high = np.asarray([float(row["ci95_high"]) for row in rows])

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    x = np.arange(len(rows))
    axes[0].plot(x, absent, "o", color="#6b7280", label="Bank ABSENT", markersize=7)
    axes[0].plot(x, ready, "o", color="#0f766e", label="GRU bank READY", markersize=7)
    for index in x:
        axes[0].plot([index, index], [absent[index], ready[index]], color="#99a1aa", linewidth=1.5)
    axes[0].set_xticks(x, labels, fontsize=8)
    axes[0].set_ylabel("Held-out rolling replay mean $R^2$")
    axes[0].set_title("Matched deployment-preprocessing A/B")
    axes[0].legend(frameon=False)

    y = np.arange(len(rows))
    axes[1].errorbar(
        delta,
        y,
        xerr=np.vstack([delta - low, high - delta]),
        fmt="o",
        color="#0f766e",
        ecolor="#5eead4",
        capsize=3,
    )
    axes[1].axvline(0, color="#374151", linewidth=1)
    axes[1].axvline(delta.mean(), color="#f59e0b", linestyle="--", linewidth=1.5, label=f"Mean +{delta.mean():.3f}")
    axes[1].set_yticks(y, [row["session"] for row in rows], fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("READY minus ABSENT mean $R^2$")
    axes[1].set_title("Reach-bootstrap 95% intervals")
    axes[1].legend(frameon=False)
    figure.suptitle("Phase 12 GRU external memory: six-session deployment replay", fontsize=14)
    figure.savefig(RESULTS / "deployment_parity_ab.png", dpi=180)
    figure.savefig(RESULTS / "deployment_parity_ab.pdf")


if __name__ == "__main__":
    main()
