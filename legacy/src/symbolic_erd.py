"""Symbolic ERD/ERS timing features for same-limb motor imagery.

This is a deliberately interpretable EEG front end:

    wide EEG epoch (-2..5 s)
      -> light causal moving average
      -> mu / beta analytic power envelopes
      -> per-trial baseline normalization from -2..0 s
      -> post-onset sliding-window ERD/ERS state tokens

The output is still a normal 2-D feature matrix, so it can be fused with fNIRS
and evaluated with the same subject-specific and leave-one-run-out protocols as
the other N1 models.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from .config import cfg_get
from .containers import TrialEpochs
from .fusion import FeatureSet, align_trials, build_feature_set, metadata_feature_set


REGIONS = {
    "left_motor": ["FC5", "FC1", "C3", "CP5", "CP1"],
    "mid_motor": ["Cz"],
    "right_motor": ["FC2", "FC6", "C4", "CP2", "CP6"],
}


def causal_moving_average(X: np.ndarray, window: int) -> np.ndarray:
    """Apply a causal moving average over the time axis."""
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


def _region_indices(ch_names: list[str]) -> list[tuple[str, list[int]]]:
    out = []
    lower_to_idx = {ch.lower(): i for i, ch in enumerate(ch_names)}
    for name, channels in REGIONS.items():
        idx = [lower_to_idx[ch.lower()] for ch in channels
               if ch.lower() in lower_to_idx]
        if idx:
            out.append((name, idx))
    if not out:
        out = [(ch, [i]) for i, ch in enumerate(ch_names)]
    return out


def _window_bounds(times: np.ndarray, win_sec: float, overlap: float):
    dt = float(np.median(np.diff(times)))
    fs = 1.0 / dt
    win = max(2, int(round(float(win_sec) * fs)))
    step = max(1, int(round(win * (1.0 - float(overlap)))))
    starts = list(range(0, max(1, len(times) - win + 1), step))
    if starts and starts[-1] != len(times) - win:
        starts.append(len(times) - win)
    return [(s, s + win) for s in starts]


def _log_band_power(X: np.ndarray, sfreq: float, band, order: int,
                    eps: float = 1e-20) -> np.ndarray:
    lo, hi = float(band[0]), float(band[1])
    nyq = float(sfreq) / 2.0
    hi = min(hi, nyq * 0.99)
    sos = butter(int(order), [lo / nyq, hi / nyq], btype="band",
                 output="sos")
    filtered = sosfiltfilt(sos, X, axis=2).astype(np.float32)
    analytic = hilbert(filtered, axis=2)
    power = np.abs(analytic) ** 2
    return np.log(power + eps).astype(np.float32)


def _state_labels(values: np.ndarray, erd_threshold: float,
                  ers_threshold: float) -> np.ndarray:
    states = np.zeros(values.shape, dtype=np.int8)
    states[values <= float(erd_threshold)] = -1
    states[values >= float(ers_threshold)] = 1
    return states


def symbolic_erd_features(epochs: TrialEpochs, cfg: dict):
    """Return symbolic ERD/ERS EEG features and feature names."""
    baseline_window = cfg_get(cfg, "symbolic_erd.baseline_window", [-2.0, 0.0])
    analysis_window = cfg_get(cfg, "symbolic_erd.analysis_window", [0.0, 5.0])
    ma_window = int(cfg_get(cfg, "symbolic_erd.moving_average_window", 10))
    win_sec = float(cfg_get(cfg, "symbolic_erd.window_sec", 0.5))
    overlap = float(cfg_get(cfg, "symbolic_erd.window_overlap", 0.5))
    order = int(cfg_get(cfg, "symbolic_erd.filter_order", 4))
    bands = cfg_get(cfg, "symbolic_erd.bands", {
        "mu": [8, 13],
        "low_beta": [13, 20],
        "high_beta": [20, 30],
    })
    erd_threshold = float(cfg_get(cfg, "symbolic_erd.erd_threshold", -0.10))
    ers_threshold = float(cfg_get(cfg, "symbolic_erd.ers_threshold", 0.10))
    include_cont = bool(cfg_get(cfg, "symbolic_erd.include_continuous", True))
    include_onehot = bool(cfg_get(cfg, "symbolic_erd.include_state_onehot", True))
    include_counts = bool(cfg_get(cfg, "symbolic_erd.include_state_counts", True))
    include_trans = bool(cfg_get(cfg, "symbolic_erd.include_transitions", True))

    times = np.asarray(epochs.times, dtype=float)
    base = ((times >= float(baseline_window[0])) &
            (times < float(baseline_window[1])))
    analysis = ((times >= float(analysis_window[0])) &
                (times <= float(analysis_window[1])))
    if base.sum() < 4:
        raise ValueError(
            "symbolic ERD needs pre-onset EEG baseline samples; build EEG "
            "epochs with symbolic_erd.eeg_tmin < 0")
    if analysis.sum() < 4:
        raise ValueError("symbolic ERD analysis window has too few EEG samples")

    X = causal_moving_average(epochs.X, ma_window)
    regions = _region_indices(epochs.ch_names)
    analysis_times = times[analysis]
    bounds = _window_bounds(analysis_times, win_sec, overlap)

    cols: list[np.ndarray] = []
    names: list[str] = []
    state_values = (-1, 0, 1)
    state_names = {-1: "erd", 0: "flat", 1: "ers"}

    for band_name, band_range in bands.items():
        log_power = _log_band_power(X, epochs.sfreq, band_range, order)
        baseline_mean = log_power[:, :, base].mean(axis=2, keepdims=True)
        rel = log_power - baseline_mean
        rel_analysis = rel[:, :, analysis]

        for region_name, idx in regions:
            region = rel_analysis[:, idx, :].mean(axis=1)
            window_means = []
            for wi, (a, b) in enumerate(bounds):
                seg = region[:, a:b]
                mean = seg.mean(axis=1)
                window_means.append(mean)
                if include_cont:
                    tt = analysis_times[a:b]
                    tt = tt - tt.mean()
                    denom = float(np.sum(tt ** 2))
                    if denom > 0:
                        slope = (seg @ tt) / denom
                    else:
                        slope = np.zeros_like(mean)
                    cols.extend([mean.astype(np.float32), slope.astype(np.float32)])
                    names.extend([
                        f"{region_name}|{band_name}|w{wi}|rel_log_power",
                        f"{region_name}|{band_name}|w{wi}|slope",
                    ])

            means = np.stack(window_means, axis=1)
            states = _state_labels(means, erd_threshold, ers_threshold)
            if include_onehot:
                for wi in range(states.shape[1]):
                    for state in state_values:
                        cols.append((states[:, wi] == state).astype(np.float32))
                        names.append(
                            f"{region_name}|{band_name}|w{wi}|state_"
                            f"{state_names[state]}")
            if include_counts:
                for state in state_values:
                    cols.append((states == state).mean(axis=1).astype(np.float32))
                    names.append(
                        f"{region_name}|{band_name}|state_fraction_"
                        f"{state_names[state]}")
            if include_trans and states.shape[1] > 1:
                prev = states[:, :-1]
                nxt = states[:, 1:]
                for a_state in state_values:
                    for b_state in state_values:
                        trans = ((prev == a_state) & (nxt == b_state))
                        cols.append(trans.mean(axis=1).astype(np.float32))
                        names.append(
                            f"{region_name}|{band_name}|transition_"
                            f"{state_names[a_state]}_to_{state_names[b_state]}")

    X_feat = np.column_stack(cols).astype(np.float32)
    return X_feat, names


def build_symbolic_erd_feature_set(epochs: TrialEpochs, cfg: dict) -> FeatureSet:
    """Extract symbolic ERD features and wrap them in a FeatureSet."""
    X, names = symbolic_erd_features(epochs, cfg)
    return FeatureSet(X=X, names=names, y=epochs.y.copy(),
                      subjects=epochs.subjects.copy(), runs=epochs.runs.copy(),
                      uids=epochs.uids.copy(), modality="eeg_symbolic_erd",
                      classes=list(epochs.classes))


def build_fused_symbolic_erd_feature_set(
        eeg_epochs: TrialEpochs, fnirs_epochs: TrialEpochs,
        cfg: dict) -> FeatureSet:
    """Build EEG symbolic ERD features concatenated with fNIRS features."""
    eeg_fs = build_symbolic_erd_feature_set(eeg_epochs, cfg)
    fnirs_fs = build_feature_set(fnirs_epochs, cfg)
    ia, ib = align_trials(metadata_feature_set(eeg_epochs), fnirs_fs)
    if ia.size == 0:
        raise ValueError("no aligned EEG+fNIRS trials available")
    if not np.array_equal(eeg_fs.y[ia], fnirs_fs.y[ib]):
        raise ValueError("label mismatch after symbolic ERD/fNIRS alignment")
    X = np.hstack([eeg_fs.X[ia], fnirs_fs.X[ib]]).astype(np.float32)
    names = [f"eeg_symbolic_erd:{n}" for n in eeg_fs.names] + [
        f"fnirs:{n}" for n in fnirs_fs.names]
    return FeatureSet(X=X, names=names, y=eeg_fs.y[ia],
                      subjects=eeg_fs.subjects[ia], runs=eeg_fs.runs[ia],
                      uids=eeg_fs.uids[ia], modality="fused_symbolic_erd",
                      classes=list(eeg_fs.classes))
