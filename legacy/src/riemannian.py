"""Riemannian tangent-space EEG features for the fused EEG+fNIRS N1 decoder.

Pipeline:

    EEG epoch (n_channels x n_times)
      -> optional MI-band filter
      -> trace-normalised regularised covariance
      -> tangent-space projection at the training covariance mean
      -> LDA / SVM / logistic regression / GELU MLP

The transformer is intentionally sklearn-compatible so the covariance mean and
projection are fit inside each cross-validation fold.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.base import BaseEstimator, TransformerMixin

from .config import cfg_get


def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def _eigh_spd(A: np.ndarray, eps: float):
    vals, vecs = np.linalg.eigh(_sym(A))
    vals = np.clip(vals, eps, None)
    return vals, vecs


def _logm_spd(A: np.ndarray, eps: float) -> np.ndarray:
    vals, vecs = _eigh_spd(A, eps)
    return _sym((vecs * np.log(vals)) @ vecs.T)


def _expm_sym(A: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(_sym(A))
    return _sym((vecs * np.exp(vals)) @ vecs.T)


def _invsqrtm_spd(A: np.ndarray, eps: float) -> np.ndarray:
    vals, vecs = _eigh_spd(A, eps)
    return _sym((vecs * (1.0 / np.sqrt(vals))) @ vecs.T)


def _cov(epoch: np.ndarray, reg: float, eps: float) -> np.ndarray:
    x = epoch - epoch.mean(axis=1, keepdims=True)
    c = x @ x.T / max(1, x.shape[1] - 1)
    tr = float(np.trace(c))
    if tr > 0:
        c = c / tr
    n = c.shape[0]
    c = (1.0 - reg) * c + reg * np.eye(n) / n
    return _sym(c + eps * np.eye(n))


def _upper_triangular_features(M: np.ndarray) -> np.ndarray:
    """Vectorise a symmetric tangent matrix with sqrt(2) off-diagonal scaling."""
    n = M.shape[0]
    feats = []
    root2 = np.sqrt(2.0)
    for i in range(n):
        for j in range(i, n):
            feats.append(float(M[i, j] if i == j else root2 * M[i, j]))
    return np.asarray(feats, dtype=np.float32)


class RiemannianTangentSpace(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq: float = 500.0, l_freq: float | None = 8.0,
                 h_freq: float | None = 30.0, reg: float = 1e-3,
                 eps: float = 1e-7, filter_order: int = 4):
        self.sfreq = sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.reg = reg
        self.eps = eps
        self.filter_order = filter_order

    def _filter(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.l_freq is None and self.h_freq is None:
            return X
        nyq = self.sfreq / 2.0
        lo = None if self.l_freq is None else float(self.l_freq) / nyq
        hi = (None if self.h_freq is None
              else min(float(self.h_freq), nyq * 0.99) / nyq)
        if lo is not None and hi is not None:
            btype, wn = "band", [lo, hi]
        elif lo is not None:
            btype, wn = "highpass", lo
        else:
            btype, wn = "lowpass", hi
        b, a = butter(int(self.filter_order), wn, btype=btype)
        padlen = 3 * max(len(a), len(b))
        if X.shape[2] <= padlen:
            return X
        out = np.empty_like(X, dtype=np.float32)
        for i in range(X.shape[0]):
            out[i] = filtfilt(b, a, X[i], axis=1).astype(np.float32)
        return out

    def _covariances(self, X):
        Xf = self._filter(X)
        return np.stack([_cov(epoch, float(self.reg), float(self.eps))
                         for epoch in Xf], axis=0)

    def fit(self, X, y=None):
        covs = self._covariances(X)
        logs = np.stack([_logm_spd(c, float(self.eps)) for c in covs], axis=0)
        self.reference_ = _expm_sym(logs.mean(axis=0))
        self.reference_invsqrt_ = _invsqrtm_spd(self.reference_, float(self.eps))
        self.n_features_in_ = covs.shape[1]
        self.n_features_out_ = self.n_features_in_ * (self.n_features_in_ + 1) // 2
        return self

    def transform(self, X):
        covs = self._covariances(X)
        W = self.reference_invsqrt_
        feats = []
        for c in covs:
            tangent = _logm_spd(W @ c @ W, float(self.eps))
            feats.append(_upper_triangular_features(tangent))
        return np.vstack(feats).astype(np.float32)


class WindowedRiemannianTangentSpace(BaseEstimator, TransformerMixin):
    """Windowed tangent-space features for sequence models.

    Each trial becomes ``(n_windows, n_tangent_features)``. The reference
    covariance is fit from all training-fold windows only.
    """

    def __init__(self, sfreq: float = 500.0, l_freq: float | None = 8.0,
                 h_freq: float | None = 30.0, reg: float = 1e-3,
                 eps: float = 1e-7, filter_order: int = 4,
                 window_sec: float = 1.0, window_overlap: float = 0.5):
        self.sfreq = sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.reg = reg
        self.eps = eps
        self.filter_order = filter_order
        self.window_sec = window_sec
        self.window_overlap = window_overlap

    def _filter(self, X):
        return RiemannianTangentSpace(
            sfreq=self.sfreq,
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            reg=self.reg,
            eps=self.eps,
            filter_order=self.filter_order,
        )._filter(X)

    def _window_slices(self, n_times: int):
        win = max(2, int(round(float(self.window_sec) * float(self.sfreq))))
        win = min(win, n_times)
        overlap = min(max(float(self.window_overlap), 0.0), 0.95)
        step = max(1, int(round(win * (1.0 - overlap))))
        starts = list(range(0, max(1, n_times - win + 1), step))
        if starts[-1] != n_times - win:
            starts.append(n_times - win)
        return [slice(s, s + win) for s in starts]

    def _window_covariances(self, X):
        Xf = self._filter(X)
        if not hasattr(self, "window_slices_"):
            self.window_slices_ = self._window_slices(Xf.shape[2])
        covs = []
        for epoch in Xf:
            trial_covs = [_cov(epoch[:, sl], float(self.reg), float(self.eps))
                          for sl in self.window_slices_]
            covs.append(trial_covs)
        return np.asarray(covs, dtype=np.float32)

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        self.window_slices_ = self._window_slices(X.shape[2])
        covs = self._window_covariances(X)
        flat = covs.reshape((-1, covs.shape[-2], covs.shape[-1]))
        logs = np.stack([_logm_spd(c, float(self.eps)) for c in flat], axis=0)
        self.reference_ = _expm_sym(logs.mean(axis=0))
        self.reference_invsqrt_ = _invsqrtm_spd(self.reference_, float(self.eps))
        self.n_windows_ = covs.shape[1]
        self.n_features_in_ = covs.shape[2]
        self.n_features_out_ = self.n_features_in_ * (self.n_features_in_ + 1) // 2
        return self

    def transform(self, X):
        covs = self._window_covariances(X)
        W = self.reference_invsqrt_
        out = []
        for trial_covs in covs:
            trial_feats = []
            for c in trial_covs:
                tangent = _logm_spd(W @ c @ W, float(self.eps))
                trial_feats.append(_upper_triangular_features(tangent))
            out.append(trial_feats)
        return np.asarray(out, dtype=np.float32)


def build_riemannian_transformer(cfg: dict, sfreq: float) -> RiemannianTangentSpace:
    """Build the tangent-space transformer from config."""
    return RiemannianTangentSpace(
        sfreq=sfreq,
        l_freq=cfg_get(cfg, "riemannian.l_freq", 8.0),
        h_freq=cfg_get(cfg, "riemannian.h_freq", 30.0),
        reg=float(cfg_get(cfg, "riemannian.reg", 1e-3)),
        eps=float(cfg_get(cfg, "riemannian.eps", 1e-7)),
        filter_order=int(cfg_get(cfg, "riemannian.filter_order", 4)),
    )


def build_windowed_riemannian_transformer(
        cfg: dict, sfreq: float) -> WindowedRiemannianTangentSpace:
    """Build a windowed tangent-space transformer from config."""
    return WindowedRiemannianTangentSpace(
        sfreq=sfreq,
        l_freq=cfg_get(cfg, "windowed_riemannian_snn.l_freq",
                       cfg_get(cfg, "riemannian.l_freq", 8.0)),
        h_freq=cfg_get(cfg, "windowed_riemannian_snn.h_freq",
                       cfg_get(cfg, "riemannian.h_freq", 30.0)),
        reg=float(cfg_get(cfg, "windowed_riemannian_snn.reg",
                          cfg_get(cfg, "riemannian.reg", 1e-3))),
        eps=float(cfg_get(cfg, "windowed_riemannian_snn.eps",
                          cfg_get(cfg, "riemannian.eps", 1e-7))),
        filter_order=int(cfg_get(cfg, "windowed_riemannian_snn.filter_order",
                                 cfg_get(cfg, "riemannian.filter_order", 4))),
        window_sec=float(cfg_get(cfg, "windowed_riemannian_snn.window_sec", 1.0)),
        window_overlap=float(cfg_get(cfg, "windowed_riemannian_snn.window_overlap",
                                     0.5)),
    )


def build_riemannian_pipeline(cfg: dict, sfreq: float, name: str | None = None):
    """Tangent-space covariance features -> scaler -> classifier Pipeline."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from .train_n1 import build_classifier

    name = name or cfg_get(cfg, "model.classifier", "lda")
    seed = int(cfg_get(cfg, "seed", 42))
    riem = build_riemannian_transformer(cfg, sfreq)
    return Pipeline([("riemannian", riem), ("scaler", StandardScaler()),
                     ("clf", build_classifier(name, seed, cfg))])
