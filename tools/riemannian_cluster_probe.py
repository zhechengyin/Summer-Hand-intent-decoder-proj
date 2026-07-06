#!/usr/bin/env python
"""K-means sanity probe for Riemannian tangent-space structure.

This is not a classifier evaluation. It fits the unsupervised Riemannian
reference covariance on all aligned trials, then asks whether K-means sees
cluster structure that resembles the four task classes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import cfg_get, load_config, resolve_path, seed_everything
from src.containers import TrialEpochs
from src.evaluate import _aligned_eeg_fnirs
from src.riemannian import build_riemannian_transformer


def _load_cached_epochs(cfg):
    cache_dir = resolve_path(cfg, "paths.cache_dir")
    eeg_path = cache_dir / "eeg_epochs.npz"
    fnirs_path = cache_dir / "fnirs_epochs.npz"
    if not eeg_path.exists() or not fnirs_path.exists():
        raise FileNotFoundError(
            f"missing cached epochs under {cache_dir}; run `python main.py preprocess`")
    return TrialEpochs.load(eeg_path), TrialEpochs.load(fnirs_path)


def _as_int_labels(values):
    uniq = {v: i for i, v in enumerate(sorted(set(values)))}
    return np.asarray([uniq[v] for v in values], dtype=int)


def _cluster_scores(X, y_class, y_subject, y_subject_run, max_k, seed):
    Z = StandardScaler().fit_transform(np.asarray(X, dtype=np.float32))
    rows = []
    max_k = min(max_k, max(2, Z.shape[0] - 1))
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, n_init=50, random_state=seed)
        pred = km.fit_predict(Z)
        rows.append({
            "k": int(k),
            "inertia": float(km.inertia_),
            "silhouette": float(silhouette_score(Z, pred)),
            "calinski_harabasz": float(calinski_harabasz_score(Z, pred)),
            "davies_bouldin": float(davies_bouldin_score(Z, pred)),
            "class_ari": float(adjusted_rand_score(y_class, pred)),
            "class_nmi": float(normalized_mutual_info_score(y_class, pred)),
            "subject_ari": float(adjusted_rand_score(y_subject, pred)),
            "subject_nmi": float(normalized_mutual_info_score(y_subject, pred)),
            "subject_run_ari": float(adjusted_rand_score(y_subject_run, pred)),
            "subject_run_nmi": float(normalized_mutual_info_score(y_subject_run, pred)),
        })
    best = max(rows, key=lambda r: r["silhouette"])
    return rows, best


def _composition(X, y, classes, k, seed):
    Z = StandardScaler().fit_transform(np.asarray(X, dtype=np.float32))
    pred = KMeans(n_clusters=k, n_init=50, random_state=seed).fit_predict(Z)
    rows = []
    for c in range(k):
        mask = pred == c
        counts = {name: int((y[mask] == i).sum())
                  for i, name in enumerate(classes)}
        rows.append({
            "cluster": int(c),
            "n": int(mask.sum()),
            "class_counts": counts,
            "majority_class": max(counts, key=counts.get),
            "majority_fraction": (
                float(max(counts.values()) / max(1, int(mask.sum())))),
        })
    return rows


def _per_subject_best(X, y, subjects, max_k, seed):
    rows = []
    for subj in sorted(set(subjects.tolist())):
        mask = subjects == subj
        if mask.sum() < 12 or len(np.unique(y[mask])) < 2:
            continue
        y_subj = np.zeros(mask.sum(), dtype=int)
        y_run = np.zeros(mask.sum(), dtype=int)
        scores, best = _cluster_scores(
            X[mask], y[mask], y_subj, y_run, min(max_k, mask.sum() - 1), seed)
        rows.append({
            "subject": str(subj),
            "n": int(mask.sum()),
            "best_k_by_silhouette": int(best["k"]),
            "best_silhouette": float(best["silhouette"]),
            "best_class_ari": float(best["class_ari"]),
            "best_class_nmi": float(best["class_nmi"]),
            "k4_silhouette": next((float(r["silhouette"]) for r in scores
                                   if r["k"] == 4), None),
            "k4_class_ari": next((float(r["class_ari"]) for r in scores
                                  if r["k"] == 4), None),
            "k4_class_nmi": next((float(r["class_nmi"]) for r in scores
                                  if r["k"] == 4), None),
        })
    return rows


def run_probe(cfg, max_k: int):
    seed = int(cfg_get(cfg, "seed", 42))
    seed_everything(seed)
    eeg, fnirs = _load_cached_epochs(cfg)
    eeg, fnirs_fs = _aligned_eeg_fnirs(eeg, fnirs, cfg)

    riem = build_riemannian_transformer(cfg, eeg.sfreq)
    tangent = riem.fit_transform(eeg.X)
    fused = np.hstack([tangent, fnirs_fs.X]).astype(np.float32)

    y = eeg.y
    subjects = eeg.subjects
    subject_labels = _as_int_labels(subjects.tolist())
    subject_run_labels = _as_int_labels(
        [f"{s}_run-{int(r)}" for s, r in zip(eeg.subjects, eeg.runs)])

    outputs = {
        "n_trials": int(eeg.n_trials),
        "classes": eeg.classes,
        "chance_level": 1.0 / len(eeg.classes),
        "note": ("K-means is unsupervised and descriptive. The Riemannian "
                 "reference covariance is fit on all aligned trials because this "
                 "probe asks about geometry, not held-out classification."),
        "representations": {},
    }

    for name, X in (("eeg_tangent", tangent), ("eeg_tangent_plus_fnirs", fused)):
        scores, best = _cluster_scores(
            X, y, subject_labels, subject_run_labels, max_k, seed)
        outputs["representations"][name] = {
            "n_features": int(X.shape[1]),
            "scores_by_k": scores,
            "best_k_by_silhouette": int(best["k"]),
            "best_silhouette": float(best["silhouette"]),
            "best_class_ari": float(best["class_ari"]),
            "best_class_nmi": float(best["class_nmi"]),
            "k4_composition": _composition(X, y, eeg.classes, 4, seed),
            "best_k_composition": _composition(X, y, eeg.classes, int(best["k"]), seed),
            "per_subject_best_k": _per_subject_best(X, y, subjects, max_k, seed),
        }
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    report = run_probe(cfg, args.max_k)
    out = (Path(args.out) if args.out else
           resolve_path(cfg, "paths.metrics_dir") / "riemannian_cluster_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Trials: {report['n_trials']} | classes={report['classes']}")
    for name, rep in report["representations"].items():
        print(f"\n{name}: {rep['n_features']} features")
        print(f"  best k by silhouette: {rep['best_k_by_silhouette']} "
              f"(sil={rep['best_silhouette']:.4f}, "
              f"class ARI={rep['best_class_ari']:.4f}, "
              f"class NMI={rep['best_class_nmi']:.4f})")
        k4 = next(r for r in rep["scores_by_k"] if r["k"] == 4)
        print(f"  k=4: sil={k4['silhouette']:.4f}, "
              f"class ARI={k4['class_ari']:.4f}, "
              f"class NMI={k4['class_nmi']:.4f}, "
              f"subject NMI={k4['subject_nmi']:.4f}")
        print("  k=4 cluster class composition:")
        for row in rep["k4_composition"]:
            print(f"    c{row['cluster']} n={row['n']} "
                  f"majority={row['majority_class']} "
                  f"frac={row['majority_fraction']:.2f} "
                  f"counts={row['class_counts']}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
