"""Stage 3 -- classical feature extraction (before any deep learning).

EEG  : time-windowed log-bandpower in the mu (8-13 Hz) and beta (13-30 Hz)
       bands per channel. Sliding sub-windows retain the temporal ERD/ERS
       dynamics that distinguish sustained motor imagery.
fNIRS: per-channel hemodynamic descriptors (mean, peak, slope, area-under-curve)
       computed over N equal time windows of the response.

Both return a 2-D design matrix ``X`` (n_trials, n_features) plus feature names,
ready for a scikit-learn pipeline.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

from .config import cfg_get
from .containers import TrialEpochs

# np.trapz was renamed np.trapezoid in NumPy 2.0; support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz"))


# ---------------------------------------------------------------------------
# EEG
# ---------------------------------------------------------------------------
def _bandpower(sig: np.ndarray, fs: float, band: tuple[float, float]) -> float:
    nperseg = int(min(len(sig), max(64, fs)))  # ~1 s segments, capped by length
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        return 0.0
    return float(_trapz(psd[mask], freqs[mask]))


def _window_bounds(n_times: int, fs: float, win_sec: float, overlap: float):
    win = max(1, int(round(win_sec * fs)))
    step = max(1, int(round(win * (1.0 - overlap))))
    starts = list(range(0, max(1, n_times - win + 1), step))
    return [(s, s + win) for s in starts] or [(0, n_times)]


def eeg_features(epochs: TrialEpochs, cfg: dict):
    bands = cfg_get(cfg, "features.eeg_bands", {"mu": [8, 13], "beta": [13, 30]})
    win_sec = float(cfg_get(cfg, "features.eeg_window_sec", 1.0))
    overlap = float(cfg_get(cfg, "features.eeg_window_overlap", 0.5))
    log_power = bool(cfg_get(cfg, "features.eeg_log_power", True))

    fs = epochs.sfreq
    bounds = _window_bounds(epochs.n_times, fs, win_sec, overlap)
    band_items = list(bands.items())

    feats = np.empty((epochs.n_trials,
                      len(bounds) * epochs.n_channels * len(band_items)),
                     dtype=np.float32)
    names: list[str] = []
    for wi, (a, b) in enumerate(bounds):
        for ch_i, ch in enumerate(epochs.ch_names):
            for bname, brange in band_items:
                names.append(f"{ch}|{bname}|w{wi}")
    # fill
    for t in range(epochs.n_trials):
        col = 0
        for (a, b) in bounds:
            seg = epochs.X[t, :, a:b]
            for ch_i in range(epochs.n_channels):
                for _, brange in band_items:
                    bp = _bandpower(seg[ch_i], fs, tuple(brange))
                    feats[t, col] = np.log(bp + 1e-12) if log_power else bp
                    col += 1
    return feats, names


# ---------------------------------------------------------------------------
# fNIRS
# ---------------------------------------------------------------------------
def _win_stats(sig: np.ndarray, t: np.ndarray, stats: list[str]) -> list[float]:
    out = []
    for s in stats:
        if s == "mean":
            out.append(float(np.mean(sig)))
        elif s == "peak":
            out.append(float(sig[np.argmax(np.abs(sig))]))
        elif s == "slope":
            out.append(float(np.polyfit(t, sig, 1)[0]) if len(t) > 1 else 0.0)
        elif s == "auc":
            out.append(float(_trapz(sig, t)))
        else:
            raise ValueError(f"unknown fNIRS stat '{s}'")
    return out


def fnirs_features(epochs: TrialEpochs, cfg: dict):
    n_win = int(cfg_get(cfg, "features.fnirs_windows", 3))
    stats = list(cfg_get(cfg, "features.fnirs_stats", ["mean", "peak", "slope", "auc"]))

    # Use the post-onset portion of the response (times >= 0) if available.
    post = epochs.times >= 0
    idx = np.where(post)[0]
    if idx.size < n_win:
        idx = np.arange(epochs.n_times)
    splits = np.array_split(idx, n_win)

    names: list[str] = []
    for ch in epochs.ch_names:
        for wi in range(n_win):
            for s in stats:
                names.append(f"{ch}|w{wi}|{s}")

    feats = np.empty((epochs.n_trials, len(names)), dtype=np.float32)
    for tr in range(epochs.n_trials):
        col = 0
        for ch_i in range(epochs.n_channels):
            for sp in splits:
                sig = epochs.X[tr, ch_i, sp]
                tt = epochs.times[sp]
                for v in _win_stats(sig, tt, stats):
                    feats[tr, col] = v
                    col += 1
    return feats, names


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def extract_features(epochs: TrialEpochs, cfg: dict):
    """Return (X, feature_names) for the given modality's epochs."""
    if epochs.modality == "eeg":
        return eeg_features(epochs, cfg)
    if epochs.modality == "fnirs":
        return fnirs_features(epochs, cfg)
    raise ValueError(f"unknown modality '{epochs.modality}'")
