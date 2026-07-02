"""Epoch / analysis-window sweep.

Motivation
----------
The best decoding window is not obvious and differs by modality:

* **EEG** carries mu/beta (de)synchronisation *during* imagery, so windows like
  0-5 s or 0.5-4.5 s after imagery onset are typical.
* **fNIRS** is hemodynamic and *lags* neural activity: the response starts ~2 s
  after onset and peaks ~5-6 s, so windows like 2-7 s, 3-8 s or 4-10 s (relative
  to imagery onset) usually beat an EEG-style 0-5 s window.

Rather than assume, we sweep. To stay cheap we epoch ONE wide window per
modality (load + filter once), then crop each candidate window, extract features,
and score it with subject-specific cross-validation.
"""
from __future__ import annotations

import csv

import numpy as np

from .config import cfg_get, resolve_path
from .containers import TrialEpochs
from .evaluate import evaluate_subject_specific
from .fusion import build_feature_set
from .load_bids import BidsIndex, discover_dataset
from .preprocess_eeg import build_eeg_epochs
from .preprocess_fnirs import build_fnirs_epochs


def _score_windows(wide: TrialEpochs, windows, cfg, name):
    rows = []
    for a, b in windows:
        try:
            cropped = wide.crop(float(a), float(b))
        except ValueError as e:
            print(f"  [{a}-{b}s] skipped: {e}")
            continue
        fs = build_feature_set(cropped, cfg)
        summary, _, _ = evaluate_subject_specific(fs, cfg, name)
        acc = summary["mean_accuracy"]
        bacc = summary["mean_balanced_accuracy"]
        f1 = summary["pooled"]["macro_f1"]
        rows.append({"tmin": float(a), "tmax": float(b), "accuracy": acc,
                     "balanced_accuracy": bacc, "macro_f1": f1,
                     "n_samples": cropped.n_times})
        print(f"  window {a:>4.1f}-{b:<4.1f}s : acc={acc:.3f} bal={bacc:.3f} "
              f"f1={f1:.3f}  ({cropped.n_times} samp)")
    return rows


def _save(rows, modality, cfg):
    if not rows:
        return None
    metrics_dir = resolve_path(cfg, "paths.metrics_dir")
    figures_dir = resolve_path(cfg, "paths.figures_dir")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = metrics_dir / f"window_sweep_{modality}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [f"{r['tmin']:g}-{r['tmax']:g}" for r in rows]
        accs = [r["accuracy"] for r in rows]
        best = int(np.argmax(accs))
        colors = ["#5b8def" if i == best else "#c9d6ea" for i in range(len(rows))]
        fig, ax = plt.subplots(figsize=(max(5, 1.1 * len(rows)), 4))
        ax.bar(labels, accs, color=colors)
        ax.axhline(1 / 4, ls="--", lw=0.8, color="#999", label="chance")
        ax.set_ylabel("subject-specific accuracy")
        ax.set_xlabel("analysis window (s from imagery onset)")
        ax.set_title(f"{modality.upper()} window sweep")
        ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / f"window_sweep_{modality}.png", dpi=130)
        plt.close(fig)
    except Exception as e:  # pragma: no cover
        print(f"  (plot skipped: {e})")

    best_row = max(rows, key=lambda r: r["accuracy"])
    print(f"\nBest {modality} window: {best_row['tmin']:g}-{best_row['tmax']:g}s "
          f"(acc={best_row['accuracy']:.3f}). Sweep -> {csv_path}")
    return best_row


def sweep_eeg(cfg: dict, index: BidsIndex | None = None,
              subjects: list[str] | None = None, windows=None, name=None):
    windows = windows or cfg_get(cfg, "sweep.eeg_windows",
                                 [[0, 5], [0.5, 4.5], [0, 4], [1, 5], [2, 5]])
    lo = min(w[0] for w in windows)
    hi = max(w[1] for w in windows)
    print(f"[sweep-eeg] building wide epochs {lo}-{hi}s (once) ...")
    wide = build_eeg_epochs(cfg, index, subjects, cache=False, tmin=lo, tmax=hi)
    return _save(_score_windows(wide, windows, cfg, name), "eeg", cfg)


def sweep_fnirs(cfg: dict, index: BidsIndex | None = None,
                subjects: list[str] | None = None, windows=None, name=None):
    windows = windows or cfg_get(cfg, "sweep.fnirs_windows",
                                 [[0, 10], [2, 7], [3, 8], [4, 10], [5, 12]])
    print("[sweep-fnirs] building wide epochs (config fnirs window) ...")
    wide = build_fnirs_epochs(cfg, index, subjects, cache=False)
    if wide is None:
        print("[sweep-fnirs] no fNIRS signal available (convert first). Skipping.")
        return None
    return _save(_score_windows(wide, windows, cfg, name), "fnirs", cfg)


def run_sweep(cfg: dict, modality: str, index: BidsIndex | None = None,
              subjects: list[str] | None = None, name: str | None = None):
    if index is None:
        index = discover_dataset(resolve_path(cfg, "paths.bids_root"))
    if modality in ("eeg", "both"):
        sweep_eeg(cfg, index, subjects, name=name)
    if modality in ("fnirs", "both"):
        sweep_fnirs(cfg, index, subjects, name=name)
