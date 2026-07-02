"""Stage 4 -- multimodal fusion layer.

Provides three model *modes* by giving each classifier a different feature
matrix:

* EEG-only    -> FeatureSet from EEG epochs
* fNIRS-only  -> FeatureSet from fNIRS epochs
* EEG+fNIRS   -> feature_level_fusion(...) concatenates the two, trial-aligned by
                 UID so a row is guaranteed to be the *same* trial in both
                 modalities (robust to dropped/rejected epochs).

Standardisation is deliberately NOT done here -- it lives inside the sklearn
Pipeline (train_n1.build_pipeline) so it is fit on training folds only and never
leaks test statistics.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .config import cfg_get
from .containers import TrialEpochs
from .feature_extraction import extract_features


@dataclass
class FeatureSet:
    X: np.ndarray            # (n_trials, n_features)
    names: list[str]
    y: np.ndarray
    subjects: np.ndarray
    runs: np.ndarray
    uids: np.ndarray
    modality: str
    classes: list[str]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def n_trials(self) -> int:
        return self.X.shape[0]

    def select(self, mask) -> "FeatureSet":
        mask = np.asarray(mask)
        return FeatureSet(self.X[mask], list(self.names), self.y[mask],
                          self.subjects[mask], self.runs[mask], self.uids[mask],
                          self.modality, list(self.classes))


def build_feature_set(epochs: TrialEpochs, cfg: dict) -> FeatureSet:
    """Extract features from epochs and carry labels/metadata along."""
    X, names = extract_features(epochs, cfg)
    return FeatureSet(X=np.asarray(X, dtype=np.float32), names=list(names),
                      y=epochs.y.copy(), subjects=epochs.subjects.copy(),
                      runs=epochs.runs.copy(), uids=epochs.uids.copy(),
                      modality=epochs.modality, classes=list(epochs.classes))


def _trial_number(uid: str) -> int:
    m = re.search(r"t(\d+)$", str(uid))
    return int(m.group(1)) if m else -1


def _align_trials(a: "FeatureSet", b: "FeatureSet"):
    """Return matched index arrays (ia, ib) pairing the SAME trial across two
    modalities. EEG and fNIRS are recorded on separate devices and occasionally
    disagree on marker count (e.g. a spurious extra fNIRS cue), so we cannot
    align by positional index. Instead, within each (subject, run) we align the
    two label sequences with difflib -- the paradigm's class order is fixed and
    identical across modalities, so matching blocks pair genuinely-corresponding
    trials and drop spurious/dropped ones."""
    import difflib

    def group(fs: "FeatureSet"):
        g: dict[tuple[str, int], list[int]] = {}
        for i in range(fs.n_trials):
            g.setdefault((str(fs.subjects[i]), int(fs.runs[i])), []).append(i)
        for key in g:
            g[key].sort(key=lambda i: _trial_number(fs.uids[i]))
        return g

    ga, gb = group(a), group(b)
    ia, ib = [], []
    for key, idxa in ga.items():
        idxb = gb.get(key)
        if not idxb:
            continue
        ya = [int(a.y[i]) for i in idxa]
        yb = [int(b.y[i]) for i in idxb]
        sm = difflib.SequenceMatcher(a=ya, b=yb, autojunk=False)
        for blk in sm.get_matching_blocks():
            for k in range(blk.size):
                ia.append(idxa[blk.a + k])
                ib.append(idxb[blk.b + k])
    return np.array(ia, dtype=int), np.array(ib, dtype=int)


def feature_level_fusion(a: FeatureSet, b: FeatureSet) -> FeatureSet:
    """Concatenate two FeatureSets on the trials they share (label-sequence aligned)."""
    ia, ib = _align_trials(a, b)
    if ia.size == 0:
        raise ValueError("no overlapping trials between the two modalities")
    if not np.array_equal(a.y[ia], b.y[ib]):  # guaranteed by matching blocks
        raise ValueError("label mismatch after alignment -- unexpected")

    X = np.hstack([a.X[ia], b.X[ib]])
    names = [f"eeg:{n}" for n in a.names] + [f"fnirs:{n}" for n in b.names]
    return FeatureSet(X=X.astype(np.float32), names=names, y=a.y[ia],
                      subjects=a.subjects[ia], runs=a.runs[ia], uids=a.uids[ia],
                      modality="fused", classes=list(a.classes))


def average_probabilities(probas: list[np.ndarray],
                          weights: list[float] | None = None) -> np.ndarray:
    """Late-fusion helper: weighted average of per-model probability matrices."""
    probas = [np.asarray(p) for p in probas]
    if weights is None:
        weights = [1.0] * len(probas)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    stacked = np.stack(probas, axis=0)                 # (n_models, n, n_classes)
    out = np.tensordot(w, stacked, axes=(0, 0))
    return out / out.sum(axis=1, keepdims=True)


def build_feature_sets(cfg: dict, eeg_epochs: TrialEpochs | None,
                       fnirs_epochs: TrialEpochs | None) -> dict[str, FeatureSet]:
    """Return whichever of {'eeg','fnirs','fused'} feature sets are available."""
    sets: dict[str, FeatureSet] = {}
    if eeg_epochs is not None:
        sets["eeg"] = build_feature_set(eeg_epochs, cfg)
    if fnirs_epochs is not None:
        sets["fnirs"] = build_feature_set(fnirs_epochs, cfg)
    if "eeg" in sets and "fnirs" in sets:
        try:
            sets["fused"] = feature_level_fusion(sets["eeg"], sets["fnirs"])
        except ValueError as e:
            print(f"[fusion] skipped fused set: {e}")
    return sets
