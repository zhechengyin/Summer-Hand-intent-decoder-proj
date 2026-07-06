#!/usr/bin/env python
"""Visualize subject-aligned Riemannian tangent features.

This diagnostic removes each subject's covariance baseline:

    C_aligned = R_subject^{-1/2} C_trial R_subject^{-1/2}

where R_subject is the log-Euclidean mean covariance for that subject. The
aligned covariances are then projected to the tangent space at identity.

This script is for geometry inspection, not held-out classification.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import cfg_get, load_config, resolve_path, seed_everything
from src.containers import TrialEpochs
from src.evaluate import _aligned_eeg_fnirs
from src.riemannian import (_cov, _expm_sym, _invsqrtm_spd, _logm_spd,
                            _upper_triangular_features,
                            build_riemannian_transformer)


def _load_cached_epochs(cfg):
    cache_dir = resolve_path(cfg, "paths.cache_dir")
    eeg_path = cache_dir / "eeg_epochs.npz"
    fnirs_path = cache_dir / "fnirs_epochs.npz"
    if not eeg_path.exists() or not fnirs_path.exists():
        raise FileNotFoundError(
            f"missing cached epochs under {cache_dir}; run `python main.py preprocess`")
    return TrialEpochs.load(eeg_path), TrialEpochs.load(fnirs_path)


def _causal_moving_average(X, window):
    """Causal moving average over time: y[t] uses only samples <= t."""
    X = np.asarray(X, dtype=np.float32)
    window = int(window)
    if window <= 1:
        return X
    csum = np.cumsum(X, axis=2, dtype=np.float64)
    csum = np.concatenate([np.zeros((*X.shape[:2], 1)), csum], axis=2)
    t = np.arange(X.shape[2])
    starts = np.maximum(0, t + 1 - window)
    sums = csum[:, :, t + 1] - csum[:, :, starts]
    counts = (t + 1 - starts).astype(np.float64)
    return (sums / counts[None, None, :]).astype(np.float32)


def _subject_aligned_tangent(eeg, cfg, moving_average_window=1):
    """Return tangent features after subject covariance recentering."""
    eps = float(cfg_get(cfg, "riemannian.eps", 1e-7))
    reg = float(cfg_get(cfg, "riemannian.reg", 1e-3))
    riem = build_riemannian_transformer(cfg, eeg.sfreq)
    X = _causal_moving_average(eeg.X, moving_average_window)
    Xf = riem._filter(X)
    covs = np.stack([_cov(epoch, reg, eps) for epoch in Xf], axis=0)

    aligned = []
    subject_refs = {}
    for subj in sorted(set(eeg.subjects.tolist())):
        idx = np.where(eeg.subjects == subj)[0]
        logs = np.stack([_logm_spd(covs[i], eps) for i in idx], axis=0)
        ref = _expm_sym(logs.mean(axis=0))
        ref_inv_sqrt = _invsqrtm_spd(ref, eps)
        subject_refs[subj] = ref
        for i in idx:
            c_aligned = ref_inv_sqrt @ covs[i] @ ref_inv_sqrt
            tangent = _logm_spd(c_aligned, eps)
            aligned.append((i, _upper_triangular_features(tangent)))

    aligned.sort(key=lambda pair: pair[0])
    features = np.vstack([feat for _, feat in aligned]).astype(np.float32)
    return features, subject_refs


def _best_k(Z, max_k, seed):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    best = None
    max_k = min(max_k, Z.shape[0] - 1)
    for k in range(2, max_k + 1):
        pred = KMeans(n_clusters=k, n_init=50, random_state=seed).fit_predict(Z)
        sil = silhouette_score(Z, pred)
        if best is None or sil > best[0]:
            best = (float(sil), int(k), pred)
    return best


def _plot_pooled(features, y, subjects, classes, cfg, seed, max_k,
                 suffix="", title_extra=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import (adjusted_rand_score,
                                 normalized_mutual_info_score)
    from sklearn.preprocessing import StandardScaler

    Z = StandardScaler().fit_transform(features)
    coords = PCA(n_components=2, random_state=seed).fit_transform(Z)
    sil, best_k, pred = _best_k(Z, max_k, seed)

    class_colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    class_cmap = ListedColormap(class_colors)
    class_norm = BoundaryNorm(np.arange(len(classes) + 1) - 0.5, len(classes))
    subject_ids = {s: i for i, s in enumerate(sorted(set(subjects.tolist())))}
    subject_y = np.asarray([subject_ids[s] for s in subjects], dtype=int)
    subject_colors = [plt.get_cmap("tab10")(i) for i in range(len(subject_ids))]
    subject_cmap = ListedColormap(subject_colors)
    subject_norm = BoundaryNorm(
        np.arange(len(subject_ids) + 1) - 0.5, len(subject_ids))
    cluster_colors = [plt.get_cmap("tab10")(i) for i in range(best_k)]
    cluster_cmap = ListedColormap(cluster_colors)
    cluster_norm = BoundaryNorm(np.arange(best_k + 1) - 0.5, best_k)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    panels = [
        ("K-means clusters", pred, cluster_cmap, cluster_norm, "cluster",
         [f"c{i}" for i in range(best_k)], cluster_colors),
        ("True class labels", y, class_cmap, class_norm, "class",
         classes, class_colors),
        ("Subject labels", subject_y, subject_cmap, subject_norm, "subject",
         list(subject_ids.keys()), subject_colors),
    ]
    for ax, (title, color_y, cmap, norm, legend_title, labels, colors) in zip(
            axes, panels):
        ax.scatter(coords[:, 0], coords[:, 1], c=color_y, s=18, alpha=0.78,
                   cmap=cmap, norm=norm, linewidths=0)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        handles = [
            plt.Line2D([], [], marker="o", linestyle="", markersize=6,
                       color=colors[i], label=labels[i])
            for i in range(len(labels))
        ]
        ax.legend(handles=handles, title=legend_title, fontsize=8,
                  title_fontsize=8, loc="best", frameon=True)
        ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.7)

    class_ari = float(adjusted_rand_score(y, pred))
    class_nmi = float(normalized_mutual_info_score(y, pred))
    subject_nmi = float(normalized_mutual_info_score(subject_y, pred))
    title = "Subject-aligned Riemannian tangent space"
    if title_extra:
        title = f"{title} - {title_extra}"
    fig.suptitle(
        f"{title} (best k={best_k}, sil={sil:.3f}, "
        f"class NMI={class_nmi:.3f})",
        fontsize=14)

    out = resolve_path(cfg, "paths.figures_dir") / \
        f"aligned_riemannian_tangent_pooled{suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return {
        "path": str(out),
        "best_k": best_k,
        "silhouette": sil,
        "class_ari": class_ari,
        "class_nmi": class_nmi,
        "subject_nmi": subject_nmi,
    }


def _plot_per_subject(features, y, subjects, classes, cfg, seed, max_k,
                      suffix="", title_extra=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from sklearn.decomposition import PCA
    from sklearn.metrics import (adjusted_rand_score,
                                 normalized_mutual_info_score)
    from sklearn.preprocessing import StandardScaler

    class_colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    class_cmap = ListedColormap(class_colors)
    class_norm = BoundaryNorm(np.arange(len(classes) + 1) - 0.5, len(classes))
    cluster_base = plt.get_cmap("tab10")
    subject_list = sorted(set(subjects.tolist()))
    fig, axes = plt.subplots(len(subject_list), 2,
                             figsize=(10.5, 3.0 * len(subject_list)),
                             constrained_layout=True)
    summaries = []

    for row, subj in enumerate(subject_list):
        mask = subjects == subj
        Z = StandardScaler().fit_transform(features[mask])
        coords = PCA(n_components=2, random_state=seed).fit_transform(Z)
        sil, best_k, pred = _best_k(Z, min(max_k, mask.sum() - 1), seed)
        ys = y[mask]
        class_ari = float(adjusted_rand_score(ys, pred))
        class_nmi = float(normalized_mutual_info_score(ys, pred))
        summaries.append({
            "subject": str(subj),
            "best_k": best_k,
            "silhouette": sil,
            "class_ari": class_ari,
            "class_nmi": class_nmi,
        })

        cluster_colors = [cluster_base(i) for i in range(best_k)]
        cluster_cmap = ListedColormap(cluster_colors)
        cluster_norm = BoundaryNorm(np.arange(best_k + 1) - 0.5, best_k)
        ax = axes[row, 0]
        ax.scatter(coords[:, 0], coords[:, 1], c=pred, s=24, alpha=0.82,
                   cmap=cluster_cmap, norm=cluster_norm, linewidths=0)
        ax.set_title(f"{subj} | best k={best_k} | sil={sil:.3f}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        handles = [
            plt.Line2D([], [], marker="o", linestyle="", markersize=6,
                       color=cluster_colors[i], label=f"c{i}")
            for i in range(best_k)
        ]
        ax.legend(handles=handles, title="cluster", fontsize=7,
                  title_fontsize=7, loc="best", frameon=True)

        ax = axes[row, 1]
        ax.scatter(coords[:, 0], coords[:, 1], c=ys, s=24, alpha=0.82,
                   cmap=class_cmap, norm=class_norm, linewidths=0)
        ax.set_title(f"{subj} | true class labels | class NMI={class_nmi:.3f}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        handles = [
            plt.Line2D([], [], marker="o", linestyle="", markersize=6,
                       color=class_colors[i], label=cls)
            for i, cls in enumerate(classes)
        ]
        ax.legend(handles=handles, title="class", fontsize=7,
                  title_fontsize=7, loc="best", frameon=True)
        for ax in axes[row, :]:
            ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.7)

    title = "Per-subject subject-aligned Riemannian tangent space"
    if title_extra:
        title = f"{title} - {title_extra}"
    fig.suptitle(title, fontsize=16)
    out = resolve_path(cfg, "paths.figures_dir") / \
        f"aligned_riemannian_tangent_per_subject{suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return {"path": str(out), "subjects": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--out", default=None)
    parser.add_argument("--moving-average-window", type=int, default=1,
                        help=("smooth EEG with a causal moving average before "
                              "Riemannian covariance estimation"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg_get(cfg, "seed", 42))
    seed_everything(seed)
    eeg, fnirs = _load_cached_epochs(cfg)
    eeg, _ = _aligned_eeg_fnirs(eeg, fnirs, cfg)
    ma_window = int(args.moving_average_window)
    suffix = f"_ma{ma_window}" if ma_window > 1 else ""
    title_extra = (f"causal {ma_window}-sample moving average"
                   if ma_window > 1 else "")
    features, _ = _subject_aligned_tangent(eeg, cfg, ma_window)
    report = {
        "note": ("Visual diagnostic only. Subject means are estimated from all "
                 "aligned trials to remove subject covariance baselines before "
                 "plotting."),
        "moving_average_window": ma_window,
        "n_trials": int(eeg.n_trials),
        "n_features": int(features.shape[1]),
        "classes": eeg.classes,
        "pooled": _plot_pooled(features, eeg.y, eeg.subjects, eeg.classes,
                               cfg, seed, args.max_k, suffix, title_extra),
        "per_subject": _plot_per_subject(features, eeg.y, eeg.subjects,
                                         eeg.classes, cfg, seed, args.max_k,
                                         suffix, title_extra),
    }
    out = (Path(args.out) if args.out else
           resolve_path(cfg, "paths.metrics_dir") /
           f"aligned_riemannian_tangent_plot{suffix}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["pooled"], indent=2))
    print(f"per_subject_plot: {report['per_subject']['path']}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
