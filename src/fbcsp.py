"""Filter Bank Common Spatial Pattern (FBCSP) for the N1 EEG decoder.

Pipeline (Ang et al., 2008/2009 style, multiclass one-vs-rest):

    EEG epoch (n_channels x n_times)
      -> filter bank: 8-12, 12-16, 16-20, 20-24, 24-30 Hz
      -> CSP per band (one-vs-rest for the 4 classes)
      -> log-variance features
      -> mutual-information feature selection (MIBIF-style, simplified)
      -> LDA / SVM / logistic regression

CSP is *supervised* and fit across trials, so -- unlike per-trial bandpower --
it MUST be fit inside each cross-validation fold. This module therefore exposes
an sklearn transformer (:class:`FBCSP`) that operates on 3-D epoch arrays
(n_trials, n_channels, n_times); dropping it into a Pipeline makes
``cross_val_predict`` refit it per fold automatically (no leakage).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, filtfilt
from sklearn.base import BaseEstimator, TransformerMixin

from .config import cfg_get


def _csp_filters(cov_a: np.ndarray, cov_b: np.ndarray, m: int) -> np.ndarray:
    """Binary CSP: m filters maximising class A / minimising class B variance,
    plus m the other way. Returns (2m, n_channels)."""
    evals, evecs = eigh(cov_a, cov_a + cov_b)      # ascending eigenvalues
    order = np.argsort(evals)
    idx = np.concatenate([order[-m:], order[:m]])  # high-A and high-B extremes
    return evecs[:, idx].T


def _trial_cov(epoch: np.ndarray, reg: float) -> np.ndarray:
    """Trace-normalised (optionally regularised) covariance of one epoch."""
    e = epoch - epoch.mean(axis=1, keepdims=True)
    c = e @ e.T
    tr = np.trace(c)
    c = c / tr if tr > 0 else c
    if reg:
        c = c + reg * np.eye(c.shape[0])
    return c


class FBCSP(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq: float = 500.0,
                 bands=((8, 12), (12, 16), (16, 20), (20, 24), (24, 30)),
                 n_components: int = 2, n_select: int = 20, reg: float = 1e-5,
                 filter_order: int = 4):
        self.sfreq = sfreq
        self.bands = bands
        self.n_components = n_components
        self.n_select = n_select
        self.reg = reg
        self.filter_order = filter_order

    # -- helpers -----------------------------------------------------------
    def _design(self):
        nyq = self.sfreq / 2.0
        filters = []
        for lo, hi in self.bands:
            hi = min(hi, nyq * 0.99)
            b, a = butter(self.filter_order, [lo / nyq, hi / nyq], btype="band")
            filters.append((b, a))
        return filters

    def _bandpass(self, X, ba):
        b, a = ba
        padlen = 3 * max(len(a), len(b))
        if X.shape[2] <= padlen:
            return X
        return filtfilt(b, a, X, axis=2)

    def _features_for_band(self, Xb, filters_per_class):
        """log-variance CSP features for all trials in one band."""
        feats = []
        for W in filters_per_class:                # one W per class (OVR)
            Z = np.einsum("fc,ncs->nfs", W, Xb)    # (n_trials, 2m, n_times)
            var = Z.var(axis=2)                    # (n_trials, 2m)
            var = var / (var.sum(axis=1, keepdims=True) + 1e-12)
            feats.append(np.log(var + 1e-12))
        return np.hstack(feats)                     # (n_trials, n_classes*2m)

    # -- sklearn API -------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self._filters = self._design()
        m = self.n_components

        # Per band: OVR CSP filters + per-band training features.
        self.band_filters_ = []                    # [ [W_class0, W_class1, ...], ... ]
        band_feats = []
        for ba in self._filters:
            Xb = self._bandpass(X, ba)
            covs = {c: np.mean([_trial_cov(Xb[i], self.reg)
                                for i in np.where(y == c)[0]], axis=0)
                    for c in self.classes_}
            filters_per_class = []
            for c in self.classes_:
                cov_c = covs[c]
                cov_rest = np.mean([_trial_cov(Xb[i], self.reg)
                                    for i in np.where(y != c)[0]], axis=0)
                filters_per_class.append(_csp_filters(cov_c, cov_rest, m))
            self.band_filters_.append(filters_per_class)
            band_feats.append(self._features_for_band(Xb, filters_per_class))

        F = np.hstack(band_feats)                  # (n_trials, n_bands*n_classes*2m)

        # Mutual-information feature selection (simplified MIBIF).
        k = min(self.n_select, F.shape[1])
        from sklearn.feature_selection import mutual_info_classif

        mi = mutual_info_classif(F, y, random_state=0)
        self.selected_ = np.argsort(mi)[::-1][:k]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        band_feats = []
        for ba, filters_per_class in zip(self._filters, self.band_filters_):
            Xb = self._bandpass(X, ba)
            band_feats.append(self._features_for_band(Xb, filters_per_class))
        F = np.hstack(band_feats)
        return F[:, self.selected_]


def build_fbcsp_pipeline(cfg: dict, sfreq: float, name: str | None = None):
    """FBCSP -> StandardScaler -> classifier Pipeline (fit per CV fold)."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from .train_n1 import build_classifier

    name = name or cfg_get(cfg, "model.classifier", "lda")
    seed = int(cfg_get(cfg, "seed", 42))
    fbcsp = FBCSP(
        sfreq=sfreq,
        bands=[tuple(b) for b in cfg_get(cfg, "fbcsp.bands",
                                         [[8, 12], [12, 16], [16, 20],
                                          [20, 24], [24, 30]])],
        n_components=int(cfg_get(cfg, "fbcsp.n_components", 2)),
        n_select=int(cfg_get(cfg, "fbcsp.n_select", 20)),
        reg=float(cfg_get(cfg, "fbcsp.reg", 1e-5)),
        filter_order=int(cfg_get(cfg, "fbcsp.filter_order", 4)),
    )
    return Pipeline([("fbcsp", fbcsp), ("scaler", StandardScaler()),
                     ("clf", build_classifier(name, seed))])
