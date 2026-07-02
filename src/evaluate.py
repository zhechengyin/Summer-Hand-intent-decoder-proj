"""Stage 7 -- rigorous evaluation.

Two protocols:
* subject-specific  : stratified K-fold *within* each participant, then aggregate
                      (the realistic first milestone for a small dataset).
* leave-one-subject-out (LOSO): train on n-1 subjects, test on the held-out one
                      (the harder cross-subject stretch goal).

Leakage guards: features come from non-overlapping trial epochs, the scaler lives
inside the Pipeline (fit on train folds only), and LOSO groups by subject so no
subject appears in both train and test. Chance level = 1 / n_classes.

Also compares raw N1 argmax commands vs state-aware N1+N2 commands.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import cfg_get, resolve_path
from .fusion import FeatureSet


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, classes: list[str]) -> dict:
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, f1_score)

    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None,
                            zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels,
                                   average="macro", zero_division=0)),
        "per_class_f1": {c: float(per_class_f1[i]) for i, c in enumerate(classes)},
        "confusion_matrix": cm.tolist(),
        "chance_level": 1.0 / len(classes),
        "n": int(len(y_true)),
    }


def save_confusion_matrix(cm, classes: list[str], path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    cm_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1e-9, None)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-validation protocols
# ---------------------------------------------------------------------------
def _safe_folds(y: np.ndarray, requested: int) -> int:
    _, counts = np.unique(y, return_counts=True)
    return int(max(2, min(requested, counts.min())))


def evaluate_subject_specific(fs: FeatureSet, cfg: dict, name: str | None = None):
    """Stratified K-fold within each subject; returns (summary, y_true, y_pred)."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    from .train_n1 import build_pipeline

    folds = int(cfg_get(cfg, "model.cv_folds", 5))
    seed = int(cfg_get(cfg, "seed", 42))
    per_subject = {}
    all_true, all_pred = [], []

    for subj in sorted(set(fs.subjects.tolist())):
        m = fs.subjects == subj
        Xs, ys = fs.X[m], fs.y[m]
        if len(np.unique(ys)) < 2:
            continue
        k = _safe_folds(ys, folds)
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        pipe = build_pipeline(cfg, name)
        yp = cross_val_predict(pipe, Xs, ys, cv=skf)
        per_subject[subj] = compute_metrics(ys, yp, fs.classes)
        all_true.extend(ys.tolist())
        all_pred.extend(yp.tolist())

    pooled = compute_metrics(np.array(all_true), np.array(all_pred), fs.classes)
    accs = [v["accuracy"] for v in per_subject.values()]
    baccs = [v["balanced_accuracy"] for v in per_subject.values()]
    summary = {
        "protocol": "subject_specific_kfold",
        "modality": fs.modality,
        "classifier": name or cfg_get(cfg, "model.classifier", "lda"),
        "pooled": pooled,
        "per_subject": per_subject,
        "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "std_accuracy": float(np.std(accs)) if accs else 0.0,
        "mean_balanced_accuracy": float(np.mean(baccs)) if baccs else 0.0,
    }
    return summary, np.array(all_true), np.array(all_pred)


def evaluate_loso(fs: FeatureSet, cfg: dict, name: str | None = None):
    """Leave-one-subject-out cross-subject evaluation."""
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

    from .train_n1 import build_pipeline

    subjects = fs.subjects
    if len(set(subjects.tolist())) < 2:
        return None, None, None
    logo = LeaveOneGroupOut()
    pipe = build_pipeline(cfg, name)
    yp = cross_val_predict(pipe, fs.X, fs.y, groups=subjects, cv=logo)

    per_subject = {}
    for subj in sorted(set(subjects.tolist())):
        m = subjects == subj
        per_subject[subj] = compute_metrics(fs.y[m], yp[m], fs.classes)
    pooled = compute_metrics(fs.y, yp, fs.classes)
    accs = [v["accuracy"] for v in per_subject.values()]
    summary = {
        "protocol": "leave_one_subject_out",
        "modality": fs.modality,
        "classifier": name or cfg_get(cfg, "model.classifier", "lda"),
        "pooled": pooled,
        "per_subject": per_subject,
        "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "std_accuracy": float(np.std(accs)) if accs else 0.0,
    }
    return summary, fs.y.copy(), yp


# ---------------------------------------------------------------------------
# N1-only vs N1+N2 behavioural comparison
# ---------------------------------------------------------------------------
def compare_n1_vs_n2(fs: FeatureSet, cfg: dict, name: str | None = None) -> dict:
    """Fit N1 on a train split, then over a held-out sequence tally how N2's
    state-aware layer changes the raw argmax command stream (deferrals, safety
    holds, state-modified commands)."""
    from sklearn.model_selection import train_test_split

    from .mini_ai_spine_n2 import N2Interpreter
    from .state import ProstheticState
    from .train_n1 import N1Decoder

    seed = int(cfg_get(cfg, "seed", 42))
    idx = np.arange(fs.n_trials)
    tr, te = train_test_split(idx, test_size=0.3, random_state=seed,
                              stratify=fs.y)
    n1 = N1Decoder.train(fs.select(tr), cfg, name)

    # Isolate the confidence-gate + state-aware effects here by disabling
    # temporal smoothing (each test trial is an independent decision, not a
    # continuous stream). Smoothing is demonstrated in the replay demo instead.
    from collections import deque

    n2 = N2Interpreter(cfg)
    n2.win, n2.min_agree = 1, 1
    n2.history = deque(maxlen=1)
    state = ProstheticState()
    raw_cmds, n2_cmds, deferrals, modified = [], [], 0, 0
    for i in te:
        out = n1.predict_one(fs.X[i])
        raw = n2.class_to_command(out.intent)           # naive mapping
        result = n2.step(out.probabilities, state)
        raw_cmds.append(raw)
        n2_cmds.append(result.prosthetic_action)
        if result.prosthetic_action in ("no_action", "hold_state",
                                        "request_more_evidence"):
            deferrals += 1
        elif result.prosthetic_action != raw:
            modified += 1
        state = result.next_state

    n = len(te)
    return {
        "n_test": int(n),
        "deferral_rate": deferrals / n if n else 0.0,
        "state_modified_rate": modified / n if n else 0.0,
        "note": ("N2 changes commands based on prosthetic STATE and confidence, "
                 "not classification accuracy; these rates quantify its safety "
                 "behaviour vs a naive class->command map."),
    }


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------
def evaluate_all(feature_sets: dict[str, FeatureSet], cfg: dict,
                 name: str | None = None, loso: bool = True) -> dict:
    """Evaluate every available modality and write metrics + figures."""
    metrics_dir = resolve_path(cfg, "paths.metrics_dir")
    figures_dir = resolve_path(cfg, "paths.figures_dir")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, dict] = {}
    comparison_rows = []
    for mode, fs in feature_sets.items():
        print(f"\n=== Evaluating {mode} ({fs.n_trials} trials, "
              f"{fs.n_features} features) ===")
        summary, yt, yp = evaluate_subject_specific(fs, cfg, name)
        print(f"  subject-specific: acc={summary['mean_accuracy']:.3f} "
              f"bal_acc={summary['mean_balanced_accuracy']:.3f} "
              f"(chance={summary['pooled']['chance_level']:.2f})")
        save_confusion_matrix(summary["pooled"]["confusion_matrix"], fs.classes,
                              figures_dir / f"confusion_{mode}_subject.png",
                              f"{mode} - subject-specific")
        report[f"{mode}_subject_specific"] = summary
        comparison_rows.append((mode, "subject", summary["mean_accuracy"],
                                summary["mean_balanced_accuracy"],
                                summary["pooled"]["macro_f1"]))

        if loso:
            lsummary, lyt, lyp = evaluate_loso(fs, cfg, name)
            if lsummary:
                print(f"  LOSO:             acc={lsummary['mean_accuracy']:.3f}")
                save_confusion_matrix(lsummary["pooled"]["confusion_matrix"],
                                      fs.classes,
                                      figures_dir / f"confusion_{mode}_loso.png",
                                      f"{mode} - LOSO")
                report[f"{mode}_loso"] = lsummary
                comparison_rows.append((mode, "loso", lsummary["mean_accuracy"],
                                        lsummary["pooled"]["balanced_accuracy"],
                                        lsummary["pooled"]["macro_f1"]))

        report[f"{mode}_n1_vs_n2"] = compare_n1_vs_n2(fs, cfg, name)

    with open(metrics_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    _write_comparison_csv(comparison_rows, metrics_dir / "comparison.csv")
    print(f"\nMetrics -> {metrics_dir/'metrics.json'}")
    print(f"Figures -> {figures_dir}")
    return report


def _write_comparison_csv(rows, path: Path) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["modality", "protocol", "accuracy", "balanced_accuracy",
                    "macro_f1"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.4f}", f"{r[4]:.4f}"])
