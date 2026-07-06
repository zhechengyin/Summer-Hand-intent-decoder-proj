"""Stage 7 -- rigorous evaluation.

Two protocols:
* subject-specific  : stratified K-fold *within* each participant, then aggregate
                      (the realistic first milestone for a small dataset).
* leave-one-run-out : within each participant, train on two runs and test on the
                      held-out run.

Leakage guards: features come from non-overlapping trial epochs, the scaler lives
inside the Pipeline (fit on train folds only), and run-held-out testing groups by
recording run within each subject. Chance level = 1 / n_classes.

Also compares raw N1 argmax commands vs state-aware N1+N2 commands.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import cfg_get, resolve_path
from .fusion import (FeatureSet, align_trials, build_feature_set,
                     metadata_feature_set)


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


# --- generic CV cores (shared by bandpower FeatureSets and FBCSP epochs) ---
def cv_subject_specific(X, y, subjects, classes, cfg, estimator_factory,
                        modality: str, clf_label: str):
    """Stratified K-fold within each subject. ``estimator_factory`` returns a
    fresh unfitted estimator so nothing leaks across folds. ``X`` may be a 2-D
    feature matrix or a 3-D epoch array (n, ch, time)."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    y = np.asarray(y)
    subjects = np.asarray(subjects)
    folds = int(cfg_get(cfg, "model.cv_folds", 5))
    seed = int(cfg_get(cfg, "seed", 42))
    per_subject, all_true, all_pred = {}, [], []

    for subj in sorted(set(subjects.tolist())):
        m = subjects == subj
        Xs, ys = X[m], y[m]
        if len(np.unique(ys)) < 2:
            continue
        k = _safe_folds(ys, folds)
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        yp = cross_val_predict(estimator_factory(), Xs, ys, cv=skf)
        per_subject[subj] = compute_metrics(ys, yp, classes)
        all_true.extend(ys.tolist())
        all_pred.extend(yp.tolist())

    pooled = compute_metrics(np.array(all_true), np.array(all_pred), classes)
    accs = [v["accuracy"] for v in per_subject.values()]
    baccs = [v["balanced_accuracy"] for v in per_subject.values()]
    summary = {
        "protocol": "subject_specific_kfold", "modality": modality,
        "classifier": clf_label, "pooled": pooled, "per_subject": per_subject,
        "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "std_accuracy": float(np.std(accs)) if accs else 0.0,
        "mean_balanced_accuracy": float(np.mean(baccs)) if baccs else 0.0,
    }
    return summary, np.array(all_true), np.array(all_pred)


def evaluate_subject_specific(fs: FeatureSet, cfg: dict, name: str | None = None):
    """Bandpower-FeatureSet wrapper around :func:`cv_subject_specific`."""
    from .train_n1 import build_pipeline

    clf = name or cfg_get(cfg, "model.classifier", "lda")
    return cv_subject_specific(fs.X, fs.y, fs.subjects, fs.classes, cfg,
                               lambda: build_pipeline(cfg, name), fs.modality, clf)


def cv_leave_one_run_out(X, y, subjects, runs, classes, cfg, estimator_factory,
                         modality: str, clf_label: str):
    """Within each subject, hold out one whole RUN as test and train on the
    others (rotating). This is the most intuitive "keep a session for testing"
    protocol and, because train/test come from different runs, it is immune to
    within-run/adjacent-window leakage."""
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

    y = np.asarray(y)
    subjects = np.asarray(subjects)
    runs = np.asarray(runs)
    per_subject, all_true, all_pred = {}, [], []
    for subj in sorted(set(subjects.tolist())):
        m = subjects == subj
        Xs, ys, rs = X[m], y[m], runs[m]
        if len(set(rs.tolist())) < 2:      # need >=2 runs to hold one out
            continue
        yp = cross_val_predict(estimator_factory(), Xs, ys, groups=rs,
                               cv=LeaveOneGroupOut())
        per_subject[subj] = compute_metrics(ys, yp, classes)
        all_true.extend(ys.tolist())
        all_pred.extend(yp.tolist())

    if not all_true:
        return None, None, None
    pooled = compute_metrics(np.array(all_true), np.array(all_pred), classes)
    accs = [v["accuracy"] for v in per_subject.values()]
    baccs = [v["balanced_accuracy"] for v in per_subject.values()]
    summary = {
        "protocol": "leave_one_run_out", "modality": modality,
        "classifier": clf_label, "pooled": pooled, "per_subject": per_subject,
        "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "std_accuracy": float(np.std(accs)) if accs else 0.0,
        "mean_balanced_accuracy": float(np.mean(baccs)) if baccs else 0.0,
    }
    return summary, np.array(all_true), np.array(all_pred)


def evaluate_loro(fs: FeatureSet, cfg: dict, name: str | None = None):
    """Bandpower-FeatureSet wrapper around :func:`cv_leave_one_run_out`."""
    from .train_n1 import build_pipeline

    clf = name or cfg_get(cfg, "model.classifier", "lda")
    return cv_leave_one_run_out(fs.X, fs.y, fs.subjects, fs.runs, fs.classes, cfg,
                                lambda: build_pipeline(cfg, name), fs.modality, clf)


def _fit_predict_eeg_fnirs(eeg_train, eeg_test, fnirs_train, fnirs_test,
                           y_train, transformer_factory, cfg, clf_label):
    """Fit EEG-front-end + fNIRS feature fusion for one CV split."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from .train_n1 import build_classifier

    seed = int(cfg_get(cfg, "seed", 42))
    transformer = transformer_factory()
    transformer.fit(eeg_train, y_train)
    z_train = np.hstack([transformer.transform(eeg_train), fnirs_train])
    z_test = np.hstack([transformer.transform(eeg_test), fnirs_test])
    clf = Pipeline([("scaler", StandardScaler()),
                    ("clf", build_classifier(clf_label, seed, cfg))])
    clf.fit(z_train, y_train)
    return clf.predict(z_test)


def cv_subject_specific_eeg_fnirs(eeg_X, fnirs_X, y, subjects, classes, cfg,
                                  transformer_factory, modality: str,
                                  clf_label: str):
    """Subject-specific CV for an EEG front-end fused with fNIRS features."""
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y)
    subjects = np.asarray(subjects)
    folds = int(cfg_get(cfg, "model.cv_folds", 5))
    seed = int(cfg_get(cfg, "seed", 42))
    per_subject, all_true, all_pred = {}, [], []

    for subj in sorted(set(subjects.tolist())):
        m = subjects == subj
        Xs_eeg, Xs_fnirs, ys = eeg_X[m], fnirs_X[m], y[m]
        if len(np.unique(ys)) < 2:
            continue
        k = _safe_folds(ys, folds)
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        yp = np.empty_like(ys)
        for train_idx, test_idx in skf.split(np.zeros(len(ys)), ys):
            yp[test_idx] = _fit_predict_eeg_fnirs(
                Xs_eeg[train_idx], Xs_eeg[test_idx],
                Xs_fnirs[train_idx], Xs_fnirs[test_idx],
                ys[train_idx], transformer_factory, cfg, clf_label)
        per_subject[subj] = compute_metrics(ys, yp, classes)
        all_true.extend(ys.tolist())
        all_pred.extend(yp.tolist())

    pooled = compute_metrics(np.array(all_true), np.array(all_pred), classes)
    accs = [v["accuracy"] for v in per_subject.values()]
    baccs = [v["balanced_accuracy"] for v in per_subject.values()]
    summary = {
        "protocol": "subject_specific_kfold", "modality": modality,
        "classifier": clf_label, "pooled": pooled, "per_subject": per_subject,
        "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "std_accuracy": float(np.std(accs)) if accs else 0.0,
        "mean_balanced_accuracy": float(np.mean(baccs)) if baccs else 0.0,
    }
    return summary, np.array(all_true), np.array(all_pred)


def cv_leave_one_run_out_eeg_fnirs(eeg_X, fnirs_X, y, subjects, runs, classes,
                                   cfg, transformer_factory, modality: str,
                                   clf_label: str):
    """Run-held-out CV for an EEG front-end fused with fNIRS features."""
    from sklearn.model_selection import LeaveOneGroupOut

    y = np.asarray(y)
    subjects = np.asarray(subjects)
    runs = np.asarray(runs)
    per_subject, all_true, all_pred = {}, [], []
    for subj in sorted(set(subjects.tolist())):
        m = subjects == subj
        Xs_eeg, Xs_fnirs, ys, rs = eeg_X[m], fnirs_X[m], y[m], runs[m]
        if len(set(rs.tolist())) < 2:
            continue
        yp = np.empty_like(ys)
        logo = LeaveOneGroupOut()
        for train_idx, test_idx in logo.split(np.zeros(len(ys)), ys, groups=rs):
            yp[test_idx] = _fit_predict_eeg_fnirs(
                Xs_eeg[train_idx], Xs_eeg[test_idx],
                Xs_fnirs[train_idx], Xs_fnirs[test_idx],
                ys[train_idx], transformer_factory, cfg, clf_label)
        per_subject[subj] = compute_metrics(ys, yp, classes)
        all_true.extend(ys.tolist())
        all_pred.extend(yp.tolist())

    if not all_true:
        return None, None, None
    pooled = compute_metrics(np.array(all_true), np.array(all_pred), classes)
    accs = [v["accuracy"] for v in per_subject.values()]
    baccs = [v["balanced_accuracy"] for v in per_subject.values()]
    summary = {
        "protocol": "leave_one_run_out", "modality": modality,
        "classifier": clf_label, "pooled": pooled, "per_subject": per_subject,
        "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "std_accuracy": float(np.std(accs)) if accs else 0.0,
        "mean_balanced_accuracy": float(np.mean(baccs)) if baccs else 0.0,
    }
    return summary, np.array(all_true), np.array(all_pred)


def _aligned_eeg_fnirs(eeg_epochs, fnirs_epochs, cfg: dict):
    if eeg_epochs is None or fnirs_epochs is None:
        raise ValueError("Both EEG and fNIRS epochs are required for all models")
    eeg_meta = metadata_feature_set(eeg_epochs)
    fnirs_fs = build_feature_set(fnirs_epochs, cfg)
    ia, ib = align_trials(eeg_meta, fnirs_fs)
    if ia.size == 0:
        raise ValueError("no aligned EEG+fNIRS trials available")
    if not np.array_equal(eeg_epochs.y[ia], fnirs_fs.y[ib]):
        raise ValueError("label mismatch after EEG+fNIRS alignment")
    return eeg_epochs.select(ia), fnirs_fs.select(ib)


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
                 name: str | None = None, loro: bool = True) -> dict:
    """Evaluate supplied fused feature sets and write metrics + figures."""
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

        if loro:
            rsummary, _, _ = evaluate_loro(fs, cfg, name)
            if rsummary:
                print(f"  leave-one-run-out: acc={rsummary['mean_accuracy']:.3f} "
                      f"bal_acc={rsummary['mean_balanced_accuracy']:.3f}")
                save_confusion_matrix(rsummary["pooled"]["confusion_matrix"],
                                      fs.classes,
                                      figures_dir / f"confusion_{mode}_loro.png",
                                      f"{mode} - leave-one-run-out")
                report[f"{mode}_leave_one_run_out"] = rsummary
                comparison_rows.append((mode, "loro", rsummary["mean_accuracy"],
                                        rsummary["mean_balanced_accuracy"],
                                        rsummary["pooled"]["macro_f1"]))

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


# ---------------------------------------------------------------------------
# FBCSP + fNIRS evaluation
# ---------------------------------------------------------------------------
def evaluate_fbcsp(eeg_epochs, fnirs_epochs, cfg: dict,
                   name: str | None = None, loro: bool = True) -> dict:
    """Evaluate FBCSP EEG features fused with fNIRS features.

    FBCSP is fit *inside* every CV fold (it is supervised and cross-trial), so
    the EEG branch is fit only on training trials. fNIRS features are concatenated
    after alignment and scaled inside the same training fold.
    """
    from .fbcsp import build_fbcsp_transformer

    eeg_epochs, fnirs_fs = _aligned_eeg_fnirs(eeg_epochs, fnirs_epochs, cfg)
    if eeg_epochs.modality != "eeg":
        raise ValueError("FBCSP is defined for EEG epochs only")
    metrics_dir = resolve_path(cfg, "paths.metrics_dir")
    figures_dir = resolve_path(cfg, "paths.figures_dir")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    clf = name or cfg_get(cfg, "model.classifier", "lda")
    factory = lambda: build_fbcsp_transformer(cfg, eeg_epochs.sfreq)  # noqa: E731
    print(f"\n=== Evaluating EEG+fNIRS FBCSP ({eeg_epochs.n_trials} trials, "
          f"{eeg_epochs.n_channels} EEG ch x {eeg_epochs.n_times} samp, "
          f"{fnirs_fs.n_features} fNIRS features, clf={clf}) ===")

    summary, _, _ = cv_subject_specific_eeg_fnirs(
        eeg_epochs.X, fnirs_fs.X, eeg_epochs.y, eeg_epochs.subjects,
        eeg_epochs.classes, cfg, factory, "fused_fbcsp", clf)
    print(f"  subject-specific: acc={summary['mean_accuracy']:.3f} "
          f"bal_acc={summary['mean_balanced_accuracy']:.3f} "
          f"(chance={summary['pooled']['chance_level']:.2f})")
    save_confusion_matrix(summary["pooled"]["confusion_matrix"], eeg_epochs.classes,
                          figures_dir / "confusion_fused_fbcsp_subject.png",
                          "EEG+fNIRS FBCSP - subject-specific")
    report = {"fused_fbcsp_subject_specific": summary}

    if loro:
        rsummary, _, _ = cv_leave_one_run_out_eeg_fnirs(
            eeg_epochs.X, fnirs_fs.X, eeg_epochs.y, eeg_epochs.subjects,
            eeg_epochs.runs, eeg_epochs.classes, cfg, factory, "fused_fbcsp", clf)
        if rsummary:
            print(f"  leave-one-run-out: acc={rsummary['mean_accuracy']:.3f}")
            save_confusion_matrix(rsummary["pooled"]["confusion_matrix"],
                                  eeg_epochs.classes,
                                  figures_dir / "confusion_fused_fbcsp_loro.png",
                                  "EEG+fNIRS FBCSP - leave-one-run-out")
            report["fused_fbcsp_leave_one_run_out"] = rsummary

    with open(metrics_dir / "metrics_fbcsp.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Metrics -> {metrics_dir/'metrics_fbcsp.json'}")
    return report


# ---------------------------------------------------------------------------
# Riemannian + fNIRS evaluation
# ---------------------------------------------------------------------------
def evaluate_riemannian(eeg_epochs, fnirs_epochs, cfg: dict,
                        name: str | None = None, loro: bool = True) -> dict:
    """Evaluate EEG tangent-space covariance features fused with fNIRS."""
    from .riemannian import build_riemannian_transformer

    eeg_epochs, fnirs_fs = _aligned_eeg_fnirs(eeg_epochs, fnirs_epochs, cfg)
    if eeg_epochs.modality != "eeg":
        raise ValueError(
            "Riemannian tangent-space features are defined for EEG epochs only")
    metrics_dir = resolve_path(cfg, "paths.metrics_dir")
    figures_dir = resolve_path(cfg, "paths.figures_dir")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    clf = name or cfg_get(cfg, "model.classifier", "lda")
    factory = lambda: build_riemannian_transformer(cfg, eeg_epochs.sfreq)  # noqa: E731
    print(f"\n=== Evaluating EEG+fNIRS Riemannian tangent space "
          f"({eeg_epochs.n_trials} trials, {eeg_epochs.n_channels} EEG ch x "
          f"{eeg_epochs.n_times} samp, {fnirs_fs.n_features} fNIRS features, "
          f"clf={clf}) ===")

    summary, _, _ = cv_subject_specific_eeg_fnirs(
        eeg_epochs.X, fnirs_fs.X, eeg_epochs.y, eeg_epochs.subjects,
        eeg_epochs.classes, cfg, factory, "fused_riemannian", clf)
    print(f"  subject-specific: acc={summary['mean_accuracy']:.3f} "
          f"bal_acc={summary['mean_balanced_accuracy']:.3f} "
          f"(chance={summary['pooled']['chance_level']:.2f})")
    save_confusion_matrix(summary["pooled"]["confusion_matrix"], eeg_epochs.classes,
                          figures_dir / "confusion_fused_riemannian_subject.png",
                          "EEG+fNIRS Riemannian - subject-specific")
    report = {"fused_riemannian_subject_specific": summary}

    if loro:
        rsummary, _, _ = cv_leave_one_run_out_eeg_fnirs(
            eeg_epochs.X, fnirs_fs.X, eeg_epochs.y, eeg_epochs.subjects,
            eeg_epochs.runs, eeg_epochs.classes, cfg, factory,
            "fused_riemannian", clf)
        if rsummary:
            print(f"  leave-one-run-out: acc={rsummary['mean_accuracy']:.3f}")
            save_confusion_matrix(
                rsummary["pooled"]["confusion_matrix"], eeg_epochs.classes,
                figures_dir / "confusion_fused_riemannian_loro.png",
                "EEG+fNIRS Riemannian - leave-one-run-out")
            report["fused_riemannian_leave_one_run_out"] = rsummary

    with open(metrics_dir / "metrics_riemannian.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Metrics -> {metrics_dir/'metrics_riemannian.json'}")
    return report


# ---------------------------------------------------------------------------
# Symbolic ERD/ERS + fNIRS evaluation
# ---------------------------------------------------------------------------
def evaluate_symbolic_erd(eeg_epochs, fnirs_epochs, cfg: dict,
                          name: str | None = None,
                          loro: bool = True) -> dict:
    """Evaluate symbolic EEG ERD/ERS timing features fused with fNIRS."""
    from .symbolic_erd import build_fused_symbolic_erd_feature_set

    if eeg_epochs is None or fnirs_epochs is None:
        raise ValueError("symbolic ERD evaluation requires EEG+fNIRS epochs")
    fs = build_fused_symbolic_erd_feature_set(eeg_epochs, fnirs_epochs, cfg)
    metrics_dir = resolve_path(cfg, "paths.metrics_dir")
    figures_dir = resolve_path(cfg, "paths.figures_dir")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    clf = name or cfg_get(cfg, "model.classifier", "lda")
    print(f"\n=== Evaluating EEG+fNIRS symbolic ERD/ERS "
          f"({fs.n_trials} trials, {fs.n_features} features, clf={clf}) ===")
    print(f"  eeg_epoch={cfg_get(cfg, 'symbolic_erd.eeg_tmin', -2.0)}.."
          f"{cfg_get(cfg, 'symbolic_erd.eeg_tmax', 5.0)}s "
          f"baseline={cfg_get(cfg, 'symbolic_erd.baseline_window', [-2.0, 0.0])} "
          f"analysis={cfg_get(cfg, 'symbolic_erd.analysis_window', [0.0, 5.0])} "
          f"moving_average_window="
          f"{cfg_get(cfg, 'symbolic_erd.moving_average_window', 10)}")

    summary, _, _ = evaluate_subject_specific(fs, cfg, clf)
    print(f"  subject-specific: acc={summary['mean_accuracy']:.3f} "
          f"bal_acc={summary['mean_balanced_accuracy']:.3f} "
          f"(chance={summary['pooled']['chance_level']:.2f})")
    save_confusion_matrix(summary["pooled"]["confusion_matrix"], fs.classes,
                          figures_dir / "confusion_fused_symbolic_erd_subject.png",
                          "EEG+fNIRS symbolic ERD/ERS - subject-specific")
    report = {"fused_symbolic_erd_subject_specific": summary}

    if loro:
        rsummary, _, _ = evaluate_loro(fs, cfg, clf)
        if rsummary:
            print(f"  leave-one-run-out: acc={rsummary['mean_accuracy']:.3f} "
                  f"bal_acc={rsummary['mean_balanced_accuracy']:.3f}")
            save_confusion_matrix(
                rsummary["pooled"]["confusion_matrix"], fs.classes,
                figures_dir / "confusion_fused_symbolic_erd_loro.png",
                "EEG+fNIRS symbolic ERD/ERS - leave-one-run-out")
            report["fused_symbolic_erd_leave_one_run_out"] = rsummary

    out = metrics_dir / "metrics_symbolic_erd.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Metrics -> {out}")
    return report


# ---------------------------------------------------------------------------
# Temporal CNN + fNIRS evaluation
# ---------------------------------------------------------------------------
def _fit_predict_temporal_cnn(eeg_train, eeg_test, fnirs_train, fnirs_test,
                              y_train, cfg, prefix: str = "temporal_cnn"):
    """Fit one fused temporal CNN for a single CV split."""
    from .temporal_cnn import build_temporal_cnn_classifier

    clf = build_temporal_cnn_classifier(cfg, prefix)
    clf.fit(eeg_train, fnirs_train, y_train)
    return clf.predict(eeg_test, fnirs_test)


def cv_subject_specific_temporal_cnn(eeg_X, fnirs_X, y, subjects, classes, cfg,
                                     modality: str = "fused_temporal_cnn",
                                     prefix: str = "temporal_cnn",
                                     classifier_label: str = "temporal_cnn"):
    """Subject-specific CV for fused EEG temporal CNN + fNIRS features."""
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y)
    subjects = np.asarray(subjects)
    folds = int(cfg_get(cfg, "model.cv_folds", 5))
    seed = int(cfg_get(cfg, "seed", 42))
    per_subject, all_true, all_pred = {}, [], []

    for subj in sorted(set(subjects.tolist())):
        m = subjects == subj
        Xs_eeg, Xs_fnirs, ys = eeg_X[m], fnirs_X[m], y[m]
        if len(np.unique(ys)) < 2:
            continue
        k = _safe_folds(ys, folds)
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        yp = np.empty_like(ys)
        for train_idx, test_idx in skf.split(np.zeros(len(ys)), ys):
            yp[test_idx] = _fit_predict_temporal_cnn(
                Xs_eeg[train_idx], Xs_eeg[test_idx],
                Xs_fnirs[train_idx], Xs_fnirs[test_idx],
                ys[train_idx], cfg, prefix)
        per_subject[subj] = compute_metrics(ys, yp, classes)
        all_true.extend(ys.tolist())
        all_pred.extend(yp.tolist())

    pooled = compute_metrics(np.array(all_true), np.array(all_pred), classes)
    accs = [v["accuracy"] for v in per_subject.values()]
    baccs = [v["balanced_accuracy"] for v in per_subject.values()]
    return {
        "protocol": "subject_specific_kfold",
        "modality": modality,
        "classifier": classifier_label,
        "pooled": pooled,
        "per_subject": per_subject,
        "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "std_accuracy": float(np.std(accs)) if accs else 0.0,
        "mean_balanced_accuracy": float(np.mean(baccs)) if baccs else 0.0,
    }, np.array(all_true), np.array(all_pred)


def cv_leave_one_run_out_temporal_cnn(eeg_X, fnirs_X, y, subjects, runs,
                                      classes, cfg,
                                      modality: str = "fused_temporal_cnn",
                                      prefix: str = "temporal_cnn",
                                      classifier_label: str = "temporal_cnn"):
    """Run-held-out CV for fused EEG temporal CNN + fNIRS features."""
    from sklearn.model_selection import LeaveOneGroupOut

    y = np.asarray(y)
    subjects = np.asarray(subjects)
    runs = np.asarray(runs)
    per_subject, all_true, all_pred = {}, [], []
    for subj in sorted(set(subjects.tolist())):
        m = subjects == subj
        Xs_eeg, Xs_fnirs, ys, rs = eeg_X[m], fnirs_X[m], y[m], runs[m]
        if len(set(rs.tolist())) < 2:
            continue
        yp = np.empty_like(ys)
        logo = LeaveOneGroupOut()
        for train_idx, test_idx in logo.split(np.zeros(len(ys)), ys, groups=rs):
            yp[test_idx] = _fit_predict_temporal_cnn(
                Xs_eeg[train_idx], Xs_eeg[test_idx],
                Xs_fnirs[train_idx], Xs_fnirs[test_idx],
                ys[train_idx], cfg, prefix)
        per_subject[subj] = compute_metrics(ys, yp, classes)
        all_true.extend(ys.tolist())
        all_pred.extend(yp.tolist())

    if not all_true:
        return None, None, None
    pooled = compute_metrics(np.array(all_true), np.array(all_pred), classes)
    accs = [v["accuracy"] for v in per_subject.values()]
    baccs = [v["balanced_accuracy"] for v in per_subject.values()]
    return {
        "protocol": "leave_one_run_out",
        "modality": modality,
        "classifier": classifier_label,
        "pooled": pooled,
        "per_subject": per_subject,
        "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "std_accuracy": float(np.std(accs)) if accs else 0.0,
        "mean_balanced_accuracy": float(np.mean(baccs)) if baccs else 0.0,
    }, np.array(all_true), np.array(all_pred)


def evaluate_temporal_cnn(eeg_epochs, fnirs_epochs, cfg: dict,
                          loro: bool = True, prefix: str = "temporal_cnn",
                          stem: str = "temporal_cnn",
                          label: str = "temporal CNN") -> dict:
    """Evaluate raw-EEG temporal CNN fused with fNIRS hemodynamic features."""
    eeg_epochs, fnirs_fs = _aligned_eeg_fnirs(eeg_epochs, fnirs_epochs, cfg)
    if eeg_epochs.modality != "eeg":
        raise ValueError("Temporal CNN is defined for EEG epochs only")
    metrics_dir = resolve_path(cfg, "paths.metrics_dir")
    figures_dir = resolve_path(cfg, "paths.figures_dir")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    modality = f"fused_{stem}"
    print(f"\n=== Evaluating EEG+fNIRS {label} ({eeg_epochs.n_trials} "
          f"trials, {eeg_epochs.n_channels} EEG ch x {eeg_epochs.n_times} "
          f"samp, {fnirs_fs.n_features} fNIRS features) ===")
    print(f"  eeg_decimate={cfg_get(cfg, f'{prefix}.eeg_decimate', 5)} "
          f"moving_average_window={cfg_get(cfg, f'{prefix}.moving_average_window', 1)} "
          f"epochs={cfg_get(cfg, f'{prefix}.epochs', 40)} "
          f"patience={cfg_get(cfg, f'{prefix}.patience', 8)}")

    summary, _, _ = cv_subject_specific_temporal_cnn(
        eeg_epochs.X, fnirs_fs.X, eeg_epochs.y, eeg_epochs.subjects,
        eeg_epochs.classes, cfg, modality=modality, prefix=prefix,
        classifier_label=stem)
    print(f"  subject-specific: acc={summary['mean_accuracy']:.3f} "
          f"bal_acc={summary['mean_balanced_accuracy']:.3f} "
          f"(chance={summary['pooled']['chance_level']:.2f})")
    save_confusion_matrix(summary["pooled"]["confusion_matrix"], eeg_epochs.classes,
                          figures_dir / f"confusion_fused_{stem}_subject.png",
                          f"EEG+fNIRS {label} - subject-specific")
    report = {f"fused_{stem}_subject_specific": summary}

    if loro:
        rsummary, _, _ = cv_leave_one_run_out_temporal_cnn(
            eeg_epochs.X, fnirs_fs.X, eeg_epochs.y, eeg_epochs.subjects,
            eeg_epochs.runs, eeg_epochs.classes, cfg, modality=modality,
            prefix=prefix, classifier_label=stem)
        if rsummary:
            print(f"  leave-one-run-out: acc={rsummary['mean_accuracy']:.3f} "
                  f"bal_acc={rsummary['mean_balanced_accuracy']:.3f}")
            save_confusion_matrix(
                rsummary["pooled"]["confusion_matrix"], eeg_epochs.classes,
                figures_dir / f"confusion_fused_{stem}_loro.png",
                f"EEG+fNIRS {label} - leave-one-run-out")
            report[f"fused_{stem}_leave_one_run_out"] = rsummary

    with open(metrics_dir / f"metrics_{stem}.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Metrics -> {metrics_dir / f'metrics_{stem}.json'}")
    return report


# ---------------------------------------------------------------------------
# Bare SNN + fNIRS evaluation
# ---------------------------------------------------------------------------
def _fit_predict_snn(eeg_train, eeg_test, fnirs_train, fnirs_test,
                     y_train, cfg):
    """Fit one fused bare SNN for a single CV split."""
    from .snn import build_snn_classifier

    clf = build_snn_classifier(cfg)
    clf.fit(eeg_train, fnirs_train, y_train)
    return clf.predict(eeg_test, fnirs_test)


def cv_subject_specific_snn(eeg_X, fnirs_X, y, subjects, classes, cfg,
                            modality: str = "fused_snn"):
    """Subject-specific CV for fused bare SNN + fNIRS features."""
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y)
    subjects = np.asarray(subjects)
    folds = int(cfg_get(cfg, "model.cv_folds", 5))
    seed = int(cfg_get(cfg, "seed", 42))
    per_subject, all_true, all_pred = {}, [], []

    for subj in sorted(set(subjects.tolist())):
        m = subjects == subj
        Xs_eeg, Xs_fnirs, ys = eeg_X[m], fnirs_X[m], y[m]
        if len(np.unique(ys)) < 2:
            continue
        k = _safe_folds(ys, folds)
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        yp = np.empty_like(ys)
        for train_idx, test_idx in skf.split(np.zeros(len(ys)), ys):
            yp[test_idx] = _fit_predict_snn(
                Xs_eeg[train_idx], Xs_eeg[test_idx],
                Xs_fnirs[train_idx], Xs_fnirs[test_idx],
                ys[train_idx], cfg)
        per_subject[subj] = compute_metrics(ys, yp, classes)
        all_true.extend(ys.tolist())
        all_pred.extend(yp.tolist())

    pooled = compute_metrics(np.array(all_true), np.array(all_pred), classes)
    accs = [v["accuracy"] for v in per_subject.values()]
    baccs = [v["balanced_accuracy"] for v in per_subject.values()]
    return {
        "protocol": "subject_specific_kfold",
        "modality": modality,
        "classifier": "bare_snn",
        "pooled": pooled,
        "per_subject": per_subject,
        "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "std_accuracy": float(np.std(accs)) if accs else 0.0,
        "mean_balanced_accuracy": float(np.mean(baccs)) if baccs else 0.0,
    }, np.array(all_true), np.array(all_pred)


def cv_leave_one_run_out_snn(eeg_X, fnirs_X, y, subjects, runs, classes, cfg,
                             modality: str = "fused_snn"):
    """Run-held-out CV for fused bare SNN + fNIRS features."""
    from sklearn.model_selection import LeaveOneGroupOut

    y = np.asarray(y)
    subjects = np.asarray(subjects)
    runs = np.asarray(runs)
    per_subject, all_true, all_pred = {}, [], []
    for subj in sorted(set(subjects.tolist())):
        m = subjects == subj
        Xs_eeg, Xs_fnirs, ys, rs = eeg_X[m], fnirs_X[m], y[m], runs[m]
        if len(set(rs.tolist())) < 2:
            continue
        yp = np.empty_like(ys)
        logo = LeaveOneGroupOut()
        for train_idx, test_idx in logo.split(np.zeros(len(ys)), ys, groups=rs):
            yp[test_idx] = _fit_predict_snn(
                Xs_eeg[train_idx], Xs_eeg[test_idx],
                Xs_fnirs[train_idx], Xs_fnirs[test_idx],
                ys[train_idx], cfg)
        per_subject[subj] = compute_metrics(ys, yp, classes)
        all_true.extend(ys.tolist())
        all_pred.extend(yp.tolist())

    if not all_true:
        return None, None, None
    pooled = compute_metrics(np.array(all_true), np.array(all_pred), classes)
    accs = [v["accuracy"] for v in per_subject.values()]
    baccs = [v["balanced_accuracy"] for v in per_subject.values()]
    return {
        "protocol": "leave_one_run_out",
        "modality": modality,
        "classifier": "bare_snn",
        "pooled": pooled,
        "per_subject": per_subject,
        "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "std_accuracy": float(np.std(accs)) if accs else 0.0,
        "mean_balanced_accuracy": float(np.mean(baccs)) if baccs else 0.0,
    }, np.array(all_true), np.array(all_pred)


def evaluate_snn(eeg_epochs, fnirs_epochs, cfg: dict,
                 loro: bool = True) -> dict:
    """Evaluate bare SNN on raw EEG time steps fused with fNIRS features."""
    eeg_epochs, fnirs_fs = _aligned_eeg_fnirs(eeg_epochs, fnirs_epochs, cfg)
    if eeg_epochs.modality != "eeg":
        raise ValueError("SNN is defined for EEG epochs only")
    metrics_dir = resolve_path(cfg, "paths.metrics_dir")
    figures_dir = resolve_path(cfg, "paths.figures_dir")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Evaluating EEG+fNIRS bare SNN ({eeg_epochs.n_trials} "
          f"trials, {eeg_epochs.n_channels} EEG ch x {eeg_epochs.n_times} "
          f"samp, {fnirs_fs.n_features} fNIRS features) ===")
    print(f"  eeg_decimate={cfg_get(cfg, 'snn.eeg_decimate', 5)} "
          f"time_bins={cfg_get(cfg, 'snn.time_bins', 128)} "
          f"hidden={cfg_get(cfg, 'snn.hidden_size', 64)} "
          f"epochs={cfg_get(cfg, 'snn.epochs', 40)} "
          f"patience={cfg_get(cfg, 'snn.patience', 8)}")

    summary, _, _ = cv_subject_specific_snn(
        eeg_epochs.X, fnirs_fs.X, eeg_epochs.y, eeg_epochs.subjects,
        eeg_epochs.classes, cfg)
    print(f"  subject-specific: acc={summary['mean_accuracy']:.3f} "
          f"bal_acc={summary['mean_balanced_accuracy']:.3f} "
          f"(chance={summary['pooled']['chance_level']:.2f})")
    save_confusion_matrix(summary["pooled"]["confusion_matrix"], eeg_epochs.classes,
                          figures_dir / "confusion_fused_snn_subject.png",
                          "EEG+fNIRS bare SNN - subject-specific")
    report = {"fused_snn_subject_specific": summary}

    if loro:
        rsummary, _, _ = cv_leave_one_run_out_snn(
            eeg_epochs.X, fnirs_fs.X, eeg_epochs.y, eeg_epochs.subjects,
            eeg_epochs.runs, eeg_epochs.classes, cfg)
        if rsummary:
            print(f"  leave-one-run-out: acc={rsummary['mean_accuracy']:.3f} "
                  f"bal_acc={rsummary['mean_balanced_accuracy']:.3f}")
            save_confusion_matrix(
                rsummary["pooled"]["confusion_matrix"], eeg_epochs.classes,
                figures_dir / "confusion_fused_snn_loro.png",
                "EEG+fNIRS bare SNN - leave-one-run-out")
            report["fused_snn_leave_one_run_out"] = rsummary

    with open(metrics_dir / "metrics_snn.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Metrics -> {metrics_dir/'metrics_snn.json'}")
    return report


# ---------------------------------------------------------------------------
# Riemannian SNN + fNIRS evaluation
# ---------------------------------------------------------------------------
def _fit_predict_riemannian_snn(eeg_train, eeg_test, fnirs_train, fnirs_test,
                                y_train, cfg, sfreq: float, windowed: bool):
    """Fit static or windowed Riemannian features, then a feature SNN."""
    from .riemannian import (build_riemannian_transformer,
                             build_windowed_riemannian_transformer)
    from .snn import build_feature_snn_classifier

    if windowed:
        transformer = build_windowed_riemannian_transformer(cfg, sfreq)
        prefix = "windowed_riemannian_snn"
    else:
        transformer = build_riemannian_transformer(cfg, sfreq)
        prefix = "riemannian_snn"

    transformer.fit(eeg_train, y_train)
    z_train = transformer.transform(eeg_train)
    z_test = transformer.transform(eeg_test)

    if windowed:
        n_train, n_win, _ = z_train.shape
        n_test = z_test.shape[0]
        fnirs_train_seq = np.repeat(fnirs_train[:, None, :], n_win, axis=1)
        fnirs_test_seq = np.repeat(fnirs_test[:, None, :], n_win, axis=1)
        z_train = np.concatenate([z_train, fnirs_train_seq], axis=2)
        z_test = np.concatenate([z_test, fnirs_test_seq], axis=2)
        if z_train.shape[0] != n_train or z_test.shape[0] != n_test:
            raise ValueError("windowed Riemannian/fNIRS shape mismatch")
    else:
        z_train = np.hstack([z_train, fnirs_train])
        z_test = np.hstack([z_test, fnirs_test])

    clf = build_feature_snn_classifier(cfg, prefix)
    clf.fit(z_train, y_train)
    return clf.predict(z_test)


def cv_subject_specific_riemannian_snn(
        eeg_X, fnirs_X, y, subjects, classes, cfg, sfreq: float,
        windowed: bool = False):
    """Subject-specific CV for static/windowed Riemannian SNN + fNIRS."""
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y)
    subjects = np.asarray(subjects)
    folds = int(cfg_get(cfg, "model.cv_folds", 5))
    seed = int(cfg_get(cfg, "seed", 42))
    per_subject, all_true, all_pred = {}, [], []
    modality = ("fused_windowed_riemannian_snn" if windowed
                else "fused_riemannian_snn")
    classifier = "windowed_riemannian_snn" if windowed else "riemannian_snn"

    for subj in sorted(set(subjects.tolist())):
        m = subjects == subj
        Xs_eeg, Xs_fnirs, ys = eeg_X[m], fnirs_X[m], y[m]
        if len(np.unique(ys)) < 2:
            continue
        k = _safe_folds(ys, folds)
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        yp = np.empty_like(ys)
        for train_idx, test_idx in skf.split(np.zeros(len(ys)), ys):
            yp[test_idx] = _fit_predict_riemannian_snn(
                Xs_eeg[train_idx], Xs_eeg[test_idx],
                Xs_fnirs[train_idx], Xs_fnirs[test_idx],
                ys[train_idx], cfg, sfreq, windowed)
        per_subject[subj] = compute_metrics(ys, yp, classes)
        all_true.extend(ys.tolist())
        all_pred.extend(yp.tolist())

    pooled = compute_metrics(np.array(all_true), np.array(all_pred), classes)
    accs = [v["accuracy"] for v in per_subject.values()]
    baccs = [v["balanced_accuracy"] for v in per_subject.values()]
    return {
        "protocol": "subject_specific_kfold",
        "modality": modality,
        "classifier": classifier,
        "pooled": pooled,
        "per_subject": per_subject,
        "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "std_accuracy": float(np.std(accs)) if accs else 0.0,
        "mean_balanced_accuracy": float(np.mean(baccs)) if baccs else 0.0,
    }, np.array(all_true), np.array(all_pred)


def cv_leave_one_run_out_riemannian_snn(
        eeg_X, fnirs_X, y, subjects, runs, classes, cfg, sfreq: float,
        windowed: bool = False):
    """Run-held-out CV for static/windowed Riemannian SNN + fNIRS."""
    from sklearn.model_selection import LeaveOneGroupOut

    y = np.asarray(y)
    subjects = np.asarray(subjects)
    runs = np.asarray(runs)
    per_subject, all_true, all_pred = {}, [], []
    modality = ("fused_windowed_riemannian_snn" if windowed
                else "fused_riemannian_snn")
    classifier = "windowed_riemannian_snn" if windowed else "riemannian_snn"

    for subj in sorted(set(subjects.tolist())):
        m = subjects == subj
        Xs_eeg, Xs_fnirs, ys, rs = eeg_X[m], fnirs_X[m], y[m], runs[m]
        if len(set(rs.tolist())) < 2:
            continue
        yp = np.empty_like(ys)
        logo = LeaveOneGroupOut()
        for train_idx, test_idx in logo.split(np.zeros(len(ys)), ys, groups=rs):
            yp[test_idx] = _fit_predict_riemannian_snn(
                Xs_eeg[train_idx], Xs_eeg[test_idx],
                Xs_fnirs[train_idx], Xs_fnirs[test_idx],
                ys[train_idx], cfg, sfreq, windowed)
        per_subject[subj] = compute_metrics(ys, yp, classes)
        all_true.extend(ys.tolist())
        all_pred.extend(yp.tolist())

    if not all_true:
        return None, None, None
    pooled = compute_metrics(np.array(all_true), np.array(all_pred), classes)
    accs = [v["accuracy"] for v in per_subject.values()]
    baccs = [v["balanced_accuracy"] for v in per_subject.values()]
    return {
        "protocol": "leave_one_run_out",
        "modality": modality,
        "classifier": classifier,
        "pooled": pooled,
        "per_subject": per_subject,
        "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "std_accuracy": float(np.std(accs)) if accs else 0.0,
        "mean_balanced_accuracy": float(np.mean(baccs)) if baccs else 0.0,
    }, np.array(all_true), np.array(all_pred)


def evaluate_riemannian_snn(eeg_epochs, fnirs_epochs, cfg: dict,
                            loro: bool = True, windowed: bool = False) -> dict:
    """Evaluate static/windowed Riemannian tangent features with an SNN head."""
    eeg_epochs, fnirs_fs = _aligned_eeg_fnirs(eeg_epochs, fnirs_epochs, cfg)
    if eeg_epochs.modality != "eeg":
        raise ValueError("Riemannian SNN is defined for EEG epochs only")
    metrics_dir = resolve_path(cfg, "paths.metrics_dir")
    figures_dir = resolve_path(cfg, "paths.figures_dir")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    label = "windowed Riemannian SNN" if windowed else "Riemannian SNN"
    stem = "windowed_riemannian_snn" if windowed else "riemannian_snn"
    print(f"\n=== Evaluating EEG+fNIRS {label} ({eeg_epochs.n_trials} "
          f"trials, {eeg_epochs.n_channels} EEG ch x {eeg_epochs.n_times} "
          f"samp, {fnirs_fs.n_features} fNIRS features) ===")
    if windowed:
        print(f"  window_sec={cfg_get(cfg, 'windowed_riemannian_snn.window_sec', 1.0)} "
              f"overlap={cfg_get(cfg, 'windowed_riemannian_snn.window_overlap', 0.5)} "
              f"hidden={cfg_get(cfg, 'windowed_riemannian_snn.hidden_size', 64)}")
    else:
        print(f"  time_steps={cfg_get(cfg, 'riemannian_snn.time_steps', 32)} "
              f"hidden={cfg_get(cfg, 'riemannian_snn.hidden_size', 64)}")

    summary, _, _ = cv_subject_specific_riemannian_snn(
        eeg_epochs.X, fnirs_fs.X, eeg_epochs.y, eeg_epochs.subjects,
        eeg_epochs.classes, cfg, eeg_epochs.sfreq, windowed)
    print(f"  subject-specific: acc={summary['mean_accuracy']:.3f} "
          f"bal_acc={summary['mean_balanced_accuracy']:.3f} "
          f"(chance={summary['pooled']['chance_level']:.2f})")
    save_confusion_matrix(
        summary["pooled"]["confusion_matrix"], eeg_epochs.classes,
        figures_dir / f"confusion_fused_{stem}_subject.png",
        f"EEG+fNIRS {label} - subject-specific")
    report = {f"fused_{stem}_subject_specific": summary}

    if loro:
        rsummary, _, _ = cv_leave_one_run_out_riemannian_snn(
            eeg_epochs.X, fnirs_fs.X, eeg_epochs.y, eeg_epochs.subjects,
            eeg_epochs.runs, eeg_epochs.classes, cfg, eeg_epochs.sfreq,
            windowed)
        if rsummary:
            print(f"  leave-one-run-out: acc={rsummary['mean_accuracy']:.3f} "
                  f"bal_acc={rsummary['mean_balanced_accuracy']:.3f}")
            save_confusion_matrix(
                rsummary["pooled"]["confusion_matrix"], eeg_epochs.classes,
                figures_dir / f"confusion_fused_{stem}_loro.png",
                f"EEG+fNIRS {label} - leave-one-run-out")
            report[f"fused_{stem}_leave_one_run_out"] = rsummary

    out = metrics_dir / f"metrics_{stem}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Metrics -> {out}")
    return report
