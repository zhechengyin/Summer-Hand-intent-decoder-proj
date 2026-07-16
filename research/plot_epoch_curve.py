#!/usr/bin/env python
"""Render epoch training curves (from the cached JSON) to a PNG figure.

Reads results/metrics/epoch_loss_curve.json (written by research/epoch_loss_curve.py) and
saves results/figures/epoch_loss_curve.png:
  top    -- train / validation(eval) / test loss (all MSE, same scale)
  bottom -- validation(eval) R2 vs test R2  ("accuracy" analog)
with the eval-selected epoch marked. No training; instant, regenerable from cache.

Reusable: `render(json_path, out_path)` is called automatically at the end of
epoch_loss_curve.py, or run this file standalone: py research/plot_epoch_curve.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "metrics" / "epoch_loss_curve.json"
OUT = ROOT / "results" / "figures" / "epoch_loss_curve.png"


def render(json_path=SRC, out_path=OUT,
           title="8-channel causal-multiscale decoder — training curves (seed 42, 24 sessions)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.loads(Path(json_path).read_text())
    ep = d["epoch"]
    sel = d.get("best_epoch")
    has_test = "test_r2" in d and d["test_r2"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7.2), sharex=True,
                                   gridspec_kw={"height_ratios": [1.1, 1]})
    fig.suptitle(title, fontsize=13, fontweight="medium")

    # ---- top: losses (all MSE, same normalized scale) ----
    ax1.plot(ep, d["train_loss"], color="#2a78d6", lw=2, label="train loss")
    ax1.plot(ep, d["val_loss"], color="#e34948", lw=2, label="validation loss (eval)")
    if has_test:
        ax1.plot(ep, d["test_loss"], color="#eda100", lw=2, label="test loss")
    if sel:
        ax1.axvline(sel, color="#888781", ls="--", lw=1.2, label=f"selected epoch {sel} (by eval)")
    ax1.set_ylabel("loss (MSE, normalized)")
    ax1.set_title("train keeps falling; held-out (eval/test) flattens = overfitting",
                  fontsize=10, color="#52514e")
    ax1.legend(frameon=False, fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.25)

    # ---- bottom: R2 (accuracy analog), eval vs test ----
    ax2.plot(ep, d["eval_r2"], color="#e34948", lw=2, label="validation R² (eval, tuned-on)")
    if has_test:
        ax2.plot(ep, d["test_r2"], color="#1baf7a", lw=2, label="test R² (untouched by tuning)")
    if sel:
        ax2.axvline(sel, color="#888781", ls="--", lw=1.2)
        ax2.scatter([sel], [d["test_r2"][sel - 1] if has_test else d["eval_r2"][sel - 1]],
                    color="#0f6e56" if has_test else "#a32d2d", zorder=5, s=30)
    ax2.set_ylabel("R²  ('accuracy')")
    ax2.set_xlabel("epoch")
    ax2.set_title("R² per epoch — does the eval-selected epoch also hold on test?",
                  fontsize=10, color="#52514e")
    ax2.legend(frameon=False, fontsize=9, loc="lower right")
    ax2.grid(True, alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if has_test and sel:
        print(f"Wrote {out_path}  ({out_path.stat().st_size/1024:.0f} KB) | "
              f"epoch {sel}: eval R²={d['eval_r2'][sel-1]:.3f}, test R²={d['test_r2'][sel-1]:.3f}")
    else:
        print(f"Wrote {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")
    return out_path


if __name__ == "__main__":
    render()
