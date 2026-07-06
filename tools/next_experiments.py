#!/usr/bin/env python
"""Next-round experiments on the fused EEG+fNIRS 4-class same-limb decoder.

This is a self-contained experiment harness that reuses the project's cached,
already-aligned EEG+fNIRS epochs and the same two evaluation protocols used
everywhere else (subject-specific stratified K-fold, and leave-one-run-out).

Every fold fits *all* trainable state (covariance reference, PLS directions,
scalers, classifiers, neural nets) on the training split only, so the numbers
are directly comparable to the existing metrics ledger. Chance = 0.25.

Experiments (select with --exp, default: all cheap ones):

    baseline    reproduce Riemannian tangent + fNIRS + logreg (LORO ~0.2715)
    shrinkage   covariance shrinkage sweep (fixed reg + Ledoit-Wolf + OAS)
    pls         tangent(+fNIRS) -> PLS-DA latent -> LDA, sweep n_components
    connectivity PLV / imag-coherence / wPLI functional-connectivity + fNIRS
    multiview   tangent + connectivity + fNIRS (early concat)
    adversarial cross-subject (LOSO) subject-adversarial MLP  [DIAGNOSTIC]

Run:  py tools/next_experiments.py --exp baseline shrinkage pls
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import cfg_get, load_config, resolve_path, seed_everything
from src.containers import TrialEpochs
from src.fusion import align_trials, build_feature_set, metadata_feature_set
from src.riemannian import (RiemannianTangentSpace, _expm_sym, _invsqrtm_spd,
                            _logm_spd, _sym, _upper_triangular_features)

CLASSES = ["reach", "grasp", "lift", "twist"]


# ---------------------------------------------------------------------------
# data loading + alignment
# ---------------------------------------------------------------------------
def load_aligned(cfg):
    """Return (eeg_epochs_aligned, fnirs_feature_matrix) sharing the same trials.

    EEG stays as raw epochs (n, ch, time) so each experiment can build its own
    front end; fNIRS is reduced to the standard hemodynamic feature matrix.
    """
    cache_dir = resolve_path(cfg, "paths.cache_dir")
    eeg = TrialEpochs.load(cache_dir / "eeg_epochs.npz")
    fnirs = TrialEpochs.load(cache_dir / "fnirs_epochs.npz")
    eeg_meta = metadata_feature_set(eeg)
    fnirs_fs = build_feature_set(fnirs, cfg)
    ia, ib = align_trials(eeg_meta, fnirs_fs)
    if ia.size == 0:
        raise RuntimeError("no aligned EEG+fNIRS trials")
    eeg = eeg.select(ia)
    fnirs_fs = fnirs_fs.select(ib)
    assert np.array_equal(eeg.y, fnirs_fs.y)
    return eeg, np.asarray(fnirs_fs.X, dtype=np.float64)


# ---------------------------------------------------------------------------
# generic within-subject CV runner
# ---------------------------------------------------------------------------
def _metrics(y_true, y_pred):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score)
    labels = list(range(len(CLASSES)))
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels,
                                   average="macro", zero_division=0)),
    }


def run_within_subject(eeg, fnirs_X, fold_fn, cfg, seed=42, folds=5):
    """Run subject-specific K-fold and leave-one-run-out for a fold model.

    fold_fn(eeg_tr, fnirs_tr, y_tr, eeg_te, fnirs_te) -> y_pred
    eeg_tr/eeg_te are raw epoch arrays (n, ch, time); fnirs_* are feature rows.
    Returns dict with per-protocol mean subject accuracy + pooled metrics.
    """
    from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

    y = eeg.y
    subjects = eeg.subjects
    runs = eeg.runs
    X = np.asarray(eeg.X)

    out = {}
    for proto in ("subject", "loro"):
        subj_accs, all_true, all_pred = [], [], []
        for subj in sorted(set(subjects.tolist())):
            m = subjects == subj
            Xs, fs, ys, rs = X[m], fnirs_X[m], y[m], runs[m]
            if len(np.unique(ys)) < 2:
                continue
            yp = np.empty_like(ys)
            if proto == "subject":
                k = int(max(2, min(folds, np.bincount(ys).min())))
                splitter = StratifiedKFold(n_splits=k, shuffle=True,
                                           random_state=seed)
                split_iter = splitter.split(np.zeros(len(ys)), ys)
            else:
                if len(set(rs.tolist())) < 2:
                    continue
                split_iter = LeaveOneGroupOut().split(np.zeros(len(ys)), ys,
                                                      groups=rs)
            for tr, te in split_iter:
                yp[te] = fold_fn(Xs[tr], fs[tr], ys[tr], Xs[te], fs[te])
            subj_accs.append(float((yp == ys).mean()))
            all_true.extend(ys.tolist())
            all_pred.extend(yp.tolist())
        pooled = _metrics(np.array(all_true), np.array(all_pred))
        out[proto] = {
            "mean_subject_acc": float(np.mean(subj_accs)) if subj_accs else 0.0,
            "pooled_acc": pooled["acc"],
            "pooled_bal_acc": pooled["bal_acc"],
            "pooled_macro_f1": pooled["macro_f1"],
            "n_subjects": len(subj_accs),
        }
    return out


# ---------------------------------------------------------------------------
# EEG front ends (all fit-on-train-only, sklearn-style)
# ---------------------------------------------------------------------------
def make_classifier(name, seed=42):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    if name == "lda":
        return LinearDiscriminantAnalysis()
    if name == "svm":
        return SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed)
    return LogisticRegression(max_iter=2000, C=1.0, random_state=seed)


def tangent_fold(l_freq=8.0, h_freq=30.0, reg=1e-3, clf="logreg", seed=42,
                 sfreq=500.0):
    """Baseline-style fold: Riemannian tangent(EEG) + fNIRS -> scaler -> clf."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def fold(eeg_tr, fnirs_tr, y_tr, eeg_te, fnirs_te):
        ts = RiemannianTangentSpace(sfreq=sfreq, l_freq=l_freq, h_freq=h_freq,
                                    reg=reg)
        ts.fit(eeg_tr, y_tr)
        ztr = np.hstack([ts.transform(eeg_tr), fnirs_tr])
        zte = np.hstack([ts.transform(eeg_te), fnirs_te])
        pipe = Pipeline([("sc", StandardScaler()),
                         ("clf", make_classifier(clf, seed))])
        pipe.fit(ztr, y_tr)
        return pipe.predict(zte)
    return fold


# ---- shrinkage-covariance tangent transformer -----------------------------
class ShrinkTangentSpace(RiemannianTangentSpace):
    """Tangent space where the per-trial covariance uses an sklearn shrinkage
    estimator (Ledoit-Wolf / OAS) instead of a fixed regularisation."""

    def __init__(self, method="ledoit_wolf", **kw):
        super().__init__(**kw)
        self.method = method

    def _one_cov(self, epoch):
        from sklearn.covariance import OAS, LedoitWolf
        x = epoch - epoch.mean(axis=1, keepdims=True)
        est = LedoitWolf() if self.method == "ledoit_wolf" else OAS()
        est.fit(x.T)                    # samples = time, features = channels
        c = est.covariance_
        tr = float(np.trace(c))
        if tr > 0:
            c = c / tr
        n = c.shape[0]
        return _sym(c + float(self.eps) * np.eye(n))

    def _covariances(self, X):
        Xf = self._filter(X)
        return np.stack([self._one_cov(ep) for ep in Xf], axis=0)


def shrink_tangent_fold(method="ledoit_wolf", clf="logreg", seed=42,
                        sfreq=500.0, l_freq=8.0, h_freq=30.0):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def fold(eeg_tr, fnirs_tr, y_tr, eeg_te, fnirs_te):
        ts = ShrinkTangentSpace(method=method, sfreq=sfreq, l_freq=l_freq,
                                h_freq=h_freq)
        ts.fit(eeg_tr, y_tr)
        ztr = np.hstack([ts.transform(eeg_tr), fnirs_tr])
        zte = np.hstack([ts.transform(eeg_te), fnirs_te])
        pipe = Pipeline([("sc", StandardScaler()),
                         ("clf", make_classifier(clf, seed))])
        pipe.fit(ztr, y_tr)
        return pipe.predict(zte)
    return fold


# ---- tangent + PLS-DA -----------------------------------------------------
def pls_fold(n_components=8, reg=1e-3, clf="lda", use_fnirs=True, seed=42,
             sfreq=500.0, l_freq=8.0, h_freq=30.0):
    """Tangent(+fNIRS) -> StandardScaler -> PLS-DA latent -> classifier."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import StandardScaler

    def onehot(y):
        Y = np.zeros((len(y), len(CLASSES)), dtype=float)
        Y[np.arange(len(y)), y] = 1.0
        return Y

    def fold(eeg_tr, fnirs_tr, y_tr, eeg_te, fnirs_te):
        ts = RiemannianTangentSpace(sfreq=sfreq, l_freq=l_freq, h_freq=h_freq,
                                    reg=reg)
        ts.fit(eeg_tr, y_tr)
        if use_fnirs:
            ztr = np.hstack([ts.transform(eeg_tr), fnirs_tr])
            zte = np.hstack([ts.transform(eeg_te), fnirs_te])
        else:
            ztr, zte = ts.transform(eeg_tr), ts.transform(eeg_te)
        sc = StandardScaler().fit(ztr)
        ztr, zte = sc.transform(ztr), sc.transform(zte)
        k = int(min(n_components, ztr.shape[1], ztr.shape[0] - 1))
        pls = PLSRegression(n_components=max(1, k))
        pls.fit(ztr, onehot(y_tr))
        ltr, lte = pls.transform(ztr), pls.transform(zte)
        cl = make_classifier(clf, seed)
        cl.fit(ltr, y_tr)
        return cl.predict(lte)
    return fold


# ---------------------------------------------------------------------------
# Functional connectivity front end (fit-free -> no leakage)
# ---------------------------------------------------------------------------
def _pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def connectivity_batch(X, sfreq, bands, metrics, ch_idx=None, chunk=32):
    """Per-trial functional-connectivity features.

    X: (n, ch, T). Returns (n, n_features). Metrics in {plv, imcoh, wpli}.
    Computed independently per trial (fit-free, leakage-safe). Processed in
    small trial chunks with complex64 so peak memory stays bounded on
    RAM-tight machines regardless of the number of trials.
    """
    from scipy.signal import butter, hilbert, sosfiltfilt

    X = np.asarray(X, dtype=np.float32)
    if ch_idx is not None:
        X = X[:, ch_idx, :]
    n, nch, T = X.shape
    pairs = _pairs(nch)
    nyq = sfreq / 2.0
    sos_list = [butter(4, [lo / nyq, min(hi, nyq * 0.99) / nyq], btype="band",
                       output="sos") for (lo, hi) in bands]
    out_rows = []
    for start in range(0, n, chunk):
        Xc = X[start:start + chunk]
        feats = []
        for sos in sos_list:
            filt = sosfiltfilt(sos, Xc, axis=2)
            A = hilbert(filt, axis=2).astype(np.complex64)   # (c, ch, T)
            pw = np.mean(np.abs(A) ** 2, axis=2)             # (c, ch)
            for (i, j) in pairs:
                z = A[:, i, :] * np.conj(A[:, j, :])         # (c, T)
                if "plv" in metrics:
                    feats.append(np.abs(np.mean(z / (np.abs(z) + 1e-20),
                                                axis=1)))
                if "imcoh" in metrics:
                    denom = np.sqrt(pw[:, i] * pw[:, j]) + 1e-20
                    feats.append(np.abs(np.imag(np.mean(z, axis=1) / denom)))
                if "wpli" in metrics:
                    imz = np.imag(z)
                    feats.append(np.abs(np.mean(imz, axis=1)) /
                                 (np.mean(np.abs(imz), axis=1) + 1e-20))
            del filt, A
        out_rows.append(np.vstack(feats).T.astype(np.float64))
    return np.vstack(out_rows)                          # (n, n_features)


def connectivity_fold(sfreq, bands, metrics, ch_idx=None, clf="logreg",
                      use_fnirs=True, seed=42):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def fold(eeg_tr, fnirs_tr, y_tr, eeg_te, fnirs_te):
        ctr = connectivity_batch(eeg_tr, sfreq, bands, metrics, ch_idx)
        cte = connectivity_batch(eeg_te, sfreq, bands, metrics, ch_idx)
        if use_fnirs:
            ctr = np.hstack([ctr, fnirs_tr])
            cte = np.hstack([cte, fnirs_te])
        pipe = Pipeline([("sc", StandardScaler()),
                         ("clf", make_classifier(clf, seed))])
        pipe.fit(ctr, y_tr)
        return pipe.predict(cte)
    return fold


# ---- multiview: tangent + connectivity + fNIRS ----------------------------
def multiview_fold(sfreq, bands, metrics, ch_idx=None, reg=1e-3, clf="logreg",
                   seed=42, l_freq=8.0, h_freq=30.0):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def fold(eeg_tr, fnirs_tr, y_tr, eeg_te, fnirs_te):
        ts = RiemannianTangentSpace(sfreq=sfreq, l_freq=l_freq, h_freq=h_freq,
                                    reg=reg)
        ts.fit(eeg_tr, y_tr)
        ctr = connectivity_batch(eeg_tr, sfreq, bands, metrics, ch_idx)
        cte = connectivity_batch(eeg_te, sfreq, bands, metrics, ch_idx)
        ztr = np.hstack([ts.transform(eeg_tr), ctr, fnirs_tr])
        zte = np.hstack([ts.transform(eeg_te), cte, fnirs_te])
        pipe = Pipeline([("sc", StandardScaler()),
                         ("clf", make_classifier(clf, seed))])
        pipe.fit(ztr, y_tr)
        return pipe.predict(zte)
    return fold


# ---------------------------------------------------------------------------
# COMBINATIONS of the winning ingredients
# ---------------------------------------------------------------------------
def _onehot(y):
    Y = np.zeros((len(y), len(CLASSES)), dtype=float)
    Y[np.arange(len(y)), y] = 1.0
    return Y


def shrinkage_lda():
    """LDA with automatic (Ledoit-Wolf) shrinkage of the class covariance."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def conn_pls_fold(sfreq, bands, metrics, ch_idx=None, n_components=16,
                  clf="lda", use_fnirs=True, seed=42):
    """Combination 1: connectivity(+fNIRS) -> PLS-DA compression -> classifier."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import StandardScaler

    def fold(eeg_tr, fnirs_tr, y_tr, eeg_te, fnirs_te):
        ctr = connectivity_batch(eeg_tr, sfreq, bands, metrics, ch_idx)
        cte = connectivity_batch(eeg_te, sfreq, bands, metrics, ch_idx)
        if use_fnirs:
            ctr, cte = np.hstack([ctr, fnirs_tr]), np.hstack([cte, fnirs_te])
        sc = StandardScaler().fit(ctr)
        ctr, cte = sc.transform(ctr), sc.transform(cte)
        k = int(min(n_components, ctr.shape[1], ctr.shape[0] - 1))
        pls = PLSRegression(n_components=max(1, k)).fit(ctr, _onehot(y_tr))
        cl = shrinkage_lda() if clf == "slda" else make_classifier(clf, seed)
        cl.fit(pls.transform(ctr), y_tr)
        return cl.predict(pls.transform(cte))
    return fold


def conn_reg_fold(sfreq, bands, metrics, ch_idx=None, clf="slda",
                  use_fnirs=True, seed=42):
    """Combination 3: connectivity(+fNIRS) -> regularized classifier."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def fold(eeg_tr, fnirs_tr, y_tr, eeg_te, fnirs_te):
        ctr = connectivity_batch(eeg_tr, sfreq, bands, metrics, ch_idx)
        cte = connectivity_batch(eeg_te, sfreq, bands, metrics, ch_idx)
        if use_fnirs:
            ctr, cte = np.hstack([ctr, fnirs_tr]), np.hstack([cte, fnirs_te])
        cl = shrinkage_lda() if clf == "slda" else make_classifier(clf, seed)
        pipe = Pipeline([("sc", StandardScaler()), ("clf", cl)])
        pipe.fit(ctr, y_tr)
        return pipe.predict(cte)
    return fold


def late_fusion_fold(sfreq, bands, metrics, ch_idx=None, reg=1e-3,
                     w_conn=0.5, seed=42, l_freq=8.0, h_freq=30.0):
    """Combination 2: probability-level (late) fusion of a connectivity model
    and a Riemannian tangent model, each with fNIRS. Avoids the dimensionality
    dilution seen with early concatenation (multiview)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def _lr():
        return LogisticRegression(max_iter=2000, C=1.0, random_state=seed)

    def fold(eeg_tr, fnirs_tr, y_tr, eeg_te, fnirs_te):
        # connectivity branch
        ctr = np.hstack([connectivity_batch(eeg_tr, sfreq, bands, metrics,
                                            ch_idx), fnirs_tr])
        cte = np.hstack([connectivity_batch(eeg_te, sfreq, bands, metrics,
                                            ch_idx), fnirs_te])
        m1 = Pipeline([("sc", StandardScaler()), ("clf", _lr())]).fit(ctr, y_tr)
        # tangent branch
        ts = RiemannianTangentSpace(sfreq=sfreq, l_freq=l_freq, h_freq=h_freq,
                                    reg=reg)
        ts.fit(eeg_tr, y_tr)
        ttr = np.hstack([ts.transform(eeg_tr), fnirs_tr])
        tte = np.hstack([ts.transform(eeg_te), fnirs_te])
        m2 = Pipeline([("sc", StandardScaler()), ("clf", _lr())]).fit(ttr, y_tr)
        p = w_conn * m1.predict_proba(cte) + (1 - w_conn) * m2.predict_proba(tte)
        return p.argmax(1)
    return fold


# ---------------------------------------------------------------------------
# Cross-subject subject-adversarial MLP (DIAGNOSTIC, leave-one-subject-out)
# ---------------------------------------------------------------------------
def run_adversarial(eeg, fnirs_X, cfg, lambda_adv=1.0, epochs=200, seed=42):
    """Leave-one-subject-out: does gradient-reversal subject suppression help a
    model trained across subjects generalise to a held-out subject?

    Features = Riemannian tangent (fit on train subjects only) + fNIRS. For each
    held-out subject we train (a) a plain class MLP and (b) the same MLP plus an
    adversarial subject-ID head via gradient reversal, then compare held-out
    class accuracy. Cross-subject DIAGNOSTIC, not an official within-subject N1.
    """
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(seed)
    np.random.seed(seed)
    y = eeg.y
    subjects = eeg.subjects
    X = np.asarray(eeg.X)
    subj_list = sorted(set(subjects.tolist()))

    class GradReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lamb):
            ctx.lamb = lamb
            return x.view_as(x)

        @staticmethod
        def backward(ctx, g):
            return -ctx.lamb * g, None

    class Net(nn.Module):
        def __init__(self, d_in, n_subj, adv):
            super().__init__()
            self.adv = adv
            self.enc = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(),
                                     nn.Dropout(0.3), nn.Linear(64, 32),
                                     nn.ReLU())
            self.cls = nn.Linear(32, len(CLASSES))
            self.sub = nn.Sequential(nn.Linear(32, 32), nn.ReLU(),
                                     nn.Linear(32, n_subj))

        def forward(self, x, lamb=0.0):
            h = self.enc(x)
            sl = self.sub(GradReverse.apply(h, lamb)) if self.adv else None
            return self.cls(h), sl

    results = {}
    for adv in (False, True):
        all_true, all_pred, subj_accs = [], [], []
        for held in subj_list:
            tr = subjects != held
            te = subjects == held
            ts = RiemannianTangentSpace(
                sfreq=eeg.sfreq,
                l_freq=float(cfg_get(cfg, "riemannian.l_freq", 8.0)),
                h_freq=float(cfg_get(cfg, "riemannian.h_freq", 30.0)),
                reg=float(cfg_get(cfg, "riemannian.reg", 1e-3)))
            ts.fit(X[tr], y[tr])
            ztr = np.hstack([ts.transform(X[tr]), fnirs_X[tr]])
            zte = np.hstack([ts.transform(X[te]), fnirs_X[te]])
            sc = StandardScaler().fit(ztr)
            ztr, zte = sc.transform(ztr), sc.transform(zte)
            uniq = {s: i for i, s in enumerate(sorted(set(subjects[tr].tolist())))}
            s_tr = np.array([uniq[s] for s in subjects[tr]])

            Xt = torch.tensor(ztr, dtype=torch.float32)
            yt = torch.tensor(y[tr], dtype=torch.long)
            st = torch.tensor(s_tr, dtype=torch.long)
            net = Net(ztr.shape[1], len(uniq), adv)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
            ce = nn.CrossEntropyLoss()
            net.train()
            for ep in range(epochs):
                lamb = lambda_adv * (2.0 / (1.0 + np.exp(-5.0 * ep / epochs)) - 1.0)
                opt.zero_grad()
                cl, sl = net(Xt, lamb)
                loss = ce(cl, yt)
                if adv:
                    loss = loss + ce(sl, st)
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                pred = net(torch.tensor(zte, dtype=torch.float32))[0].argmax(1).numpy()
            all_true.extend(y[te].tolist())
            all_pred.extend(pred.tolist())
            subj_accs.append(float((pred == y[te]).mean()))
        pooled = _metrics(np.array(all_true), np.array(all_pred))
        results["adversarial" if adv else "plain"] = {
            "mean_subject_acc": float(np.mean(subj_accs)),
            "pooled_acc": pooled["acc"],
            "pooled_bal_acc": pooled["bal_acc"],
            "pooled_macro_f1": pooled["macro_f1"],
            "protocol": "leave_one_subject_out",
        }
    return results


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _fmt(tag, r):
    s, l = r["subject"], r["loro"]
    return (f"{tag:<44} subj acc={s['mean_subject_acc']:.4f} "
            f"(bal={s['pooled_bal_acc']:.3f} f1={s['pooled_macro_f1']:.3f}) | "
            f"LORO acc={l['mean_subject_acc']:.4f} "
            f"(bal={l['pooled_bal_acc']:.3f} f1={l['pooled_macro_f1']:.3f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--exp", nargs="*", default=["baseline", "shrinkage",
                                                 "pls", "connectivity",
                                                 "multiview"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg_get(cfg, "seed", 42))
    seed_everything(seed)
    eeg, fnirs_X = load_aligned(cfg)
    sfreq = eeg.sfreq
    print(f"Loaded {eeg.n_trials} aligned trials, "
          f"{eeg.n_channels} EEG ch x {eeg.n_times} samp @ {sfreq:g}Hz, "
          f"{fnirs_X.shape[1]} fNIRS features, "
          f"subjects={sorted(set(eeg.subjects.tolist()))}")
    print(f"channels: {eeg.ch_names}")

    report = {"chance": 0.25, "n_trials": int(eeg.n_trials), "results": {}}
    R = report["results"]

    def do(tag, fold_fn):
        t0 = time.time()
        r = run_within_subject(eeg, fnirs_X, fold_fn, cfg, seed)
        R[tag] = r
        print(_fmt(tag, r) + f"   [{time.time()-t0:.0f}s]")

    if "baseline" in args.exp:
        print("\n--- E0 baseline reproduction ---")
        do("riemann+fnirs+logreg (reg=1e-3)",
           tangent_fold(reg=1e-3, clf="logreg", sfreq=sfreq))

    if "shrinkage" in args.exp:
        print("\n--- E1 covariance shrinkage sweep ---")
        for reg in (1e-4, 1e-3, 1e-2, 5e-2, 1e-1):
            do(f"tangent reg={reg:g} + fnirs + logreg",
               tangent_fold(reg=reg, clf="logreg", sfreq=sfreq))
        do("tangent Ledoit-Wolf + fnirs + logreg",
           shrink_tangent_fold("ledoit_wolf", "logreg", sfreq=sfreq))
        do("tangent OAS + fnirs + logreg",
           shrink_tangent_fold("oas", "logreg", sfreq=sfreq))

    if "pls" in args.exp:
        print("\n--- E2 tangent + PLS-DA ---")
        for k in (2, 4, 8, 16, 32):
            do(f"tangent+fnirs PLS({k})->LDA",
               pls_fold(n_components=k, clf="lda", use_fnirs=True, sfreq=sfreq))

    if "connectivity" in args.exp:
        print("\n--- E3 functional connectivity ---")
        bands = [(8, 13), (13, 30)]
        motor = ["FC5", "FC1", "C3", "CP5", "CP1", "Cz",
                 "FC2", "FC6", "C4", "CP2", "CP6"]
        low = {c.lower(): i for i, c in enumerate(eeg.ch_names)}
        ch_idx = [low[c.lower()] for c in motor if c.lower() in low]
        for mets in (["plv"], ["imcoh"], ["wpli"], ["plv", "imcoh", "wpli"]):
            do(f"conn[{'+'.join(mets)}] allch + fnirs + logreg",
               connectivity_fold(sfreq, bands, mets, None, "logreg"))
        do("conn[plv+imcoh+wpli] motor + fnirs + logreg",
           connectivity_fold(sfreq, bands, ["plv", "imcoh", "wpli"], ch_idx,
                             "logreg"))

    if "multiview" in args.exp:
        print("\n--- E4 multiview tangent + connectivity + fNIRS ---")
        bands = [(8, 13), (13, 30)]
        do("tangent + conn[plv+imcoh+wpli] + fnirs + logreg",
           multiview_fold(sfreq, bands, ["plv", "imcoh", "wpli"], None,
                          clf="logreg"))

    if "combos" in args.exp:
        print("\n--- E6 combinations of winning ingredients ---")
        bands = [(8, 13), (13, 30)]
        mets = ["plv", "imcoh", "wpli"]
        # 0. re-check the plain connectivity winner under the chunked impl
        do("conn (all) + fnirs + logreg [recheck]",
           connectivity_fold(sfreq, bands, mets, None, "logreg"))
        # 1. connectivity -> PLS-DA -> LDA (sweep components)
        for k in (8, 16, 32, 64):
            do(f"conn -> PLS({k}) -> LDA + fnirs",
               conn_pls_fold(sfreq, bands, mets, None, k, "lda"))
        # 3. connectivity -> shrinkage LDA / low-C logreg
        do("conn -> shrinkage-LDA + fnirs",
           conn_reg_fold(sfreq, bands, mets, None, "slda"))
        # 2. late (probability) fusion of connectivity + tangent
        for w in (0.5, 0.65, 0.8):
            do(f"late-fusion conn({w:g})+tangent + fnirs",
               late_fusion_fold(sfreq, bands, mets, None, w_conn=w))

    if "adversarial" in args.exp:
        print("\n--- E5 subject-adversarial (LOSO diagnostic) ---")
        adv = run_adversarial(eeg, fnirs_X, cfg, lambda_adv=1.0, epochs=200,
                              seed=seed)
        R["adversarial_loso"] = adv
        for k in ("plain", "adversarial"):
            v = adv[k]
            print(f"  LOSO {k:<12} acc={v['pooled_acc']:.4f} "
                  f"bal={v['pooled_bal_acc']:.3f} f1={v['pooled_macro_f1']:.3f}")

    out = (Path(args.out) if args.out else
           resolve_path(cfg, "paths.metrics_dir") / "next_experiments.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
