"""Stage 2 -- fNIRS preprocessing + epoching.

Raw intensity -> optical density -> HbO/HbR (modified Beer-Lambert) ->
low-pass (hemodynamic band) -> epoch a long, baseline-corrected window.

NOTE (baseline implementation): the fNIRS raw signal in ds004022 is a MATLAB
table object; this module therefore runs on either (a) converted real data (see
tools/convert_fnirs_octave.m) or (b) synthetic HbO/HbR. The Beer-Lambert step
uses a compact extinction table and natural-log optical density -- a standard but
*simplified* conversion, adequate as a first-version baseline.
"""
from __future__ import annotations

import re

import numpy as np
from scipy.signal import butter, filtfilt

from .config import cfg_get, resolve_path
from .containers import TrialEpochs
from .load_bids import (BidsIndex, FnirsRun, discover_dataset,
                        fnirs_trial_onsets, load_fnirs_bbci)

# Molar extinction coefficients [1/(M*cm)] at the NIRScout wavelengths, from the
# widely used Homer2 table. Rows = wavelength, cols = [HbO, HbR].
_EXTINCTION = {
    760: (1486.59, 3843.71),
    850: (2526.39, 1798.64),
}

_CH_RE = re.compile(r"S(?P<s>\d+)[_ ]?D(?P<d>\d+)\s*(?P<wl>\d+)", re.IGNORECASE)


def parse_fnirs_channels(ch_names: list[str]):
    """Group 'S{s}_D{d} {wl}' labels into ordered SD pairs and per-wl columns.

    Returns (pairs, wl_cols) where pairs is a list of (s, d) and wl_cols maps
    wavelength -> array of column indices aligned with `pairs`.
    """
    parsed = []
    for i, name in enumerate(ch_names):
        m = _CH_RE.search(name)
        if m:
            parsed.append((i, int(m["s"]), int(m["d"]), int(m["wl"])))
    pairs = sorted({(s, d) for _, s, d, _ in parsed})
    pair_index = {sd: k for k, sd in enumerate(pairs)}
    wls = sorted({wl for *_, wl in parsed})
    wl_cols = {wl: np.full(len(pairs), -1, dtype=int) for wl in wls}
    for col, s, d, wl in parsed:
        wl_cols[wl][pair_index[(s, d)]] = col
    return pairs, wl_cols


def intensity_to_hb(data: np.ndarray, ch_names: list[str], cfg: dict):
    """Convert (n_samples, n_channels) intensity to HbO/HbR concentration change.

    Returns (hb (2*n_pairs, n_samples), hb_names) with HbO pairs first, then HbR.
    """
    pairs, wl_cols = parse_fnirs_channels(ch_names)
    wls = sorted(wl_cols)
    if len(wls) < 2 or not pairs:
        raise ValueError("need >=2 wavelengths and >=1 source-detector pair")
    wl1, wl2 = wls[0], wls[1]

    dist = float(cfg_get(cfg, "fnirs.source_detector_distance_cm", 3.0))
    ppf = float(cfg_get(cfg, "fnirs.ppf", 6.0))
    path_len = dist * ppf

    e1, e2 = _EXTINCTION.get(wl1), _EXTINCTION.get(wl2)
    if e1 is None or e2 is None:
        raise ValueError(f"no extinction coefficients for {wls}")
    E = np.array([e1, e2], dtype=float)          # [[HbO,HbR]@wl1, ...@wl2]
    E_inv = np.linalg.inv(E)

    # Auto-detect the input type. Raw intensity is strictly positive -> convert
    # to optical density via -log(I/mean). ds004022's recovered signal is already
    # AC-coupled/optical-density-like (contains negatives), so we treat it as
    # delta-OD and skip the log.
    is_intensity = bool(np.all(data > 0))

    def od(col_idx):
        sig = data[:, col_idx].astype(float)
        if is_intensity:
            mean = np.maximum(np.mean(sig, axis=0, keepdims=True), 1e-12)
            return -np.log(np.clip(sig, 1e-12, None) / mean)
        return sig - np.mean(sig, axis=0, keepdims=True)   # already delta-OD

    od1, od2 = od(wl_cols[wl1]), od(wl_cols[wl2])            # each (T, n_pairs)
    # concentration = E_inv @ [od1; od2] / path_len, per pair
    hbo = (E_inv[0, 0] * od1 + E_inv[0, 1] * od2) / path_len
    hbr = (E_inv[1, 0] * od1 + E_inv[1, 1] * od2) / path_len
    hb = np.concatenate([hbo, hbr], axis=1).T               # (2*n_pairs, T)
    names = ([f"S{s}_D{d} HbO" for s, d in pairs]
             + [f"S{s}_D{d} HbR" for s, d in pairs])
    return hb, names


def filter_hemo(hb: np.ndarray, fs: float, cfg: dict) -> np.ndarray:
    """Zero-phase band-pass for the hemodynamic band (default 0.01-0.2 Hz)."""
    l = float(cfg_get(cfg, "fnirs.l_freq", 0.01))
    h = float(cfg_get(cfg, "fnirs.h_freq", 0.2))
    nyq = fs / 2.0
    h = min(h, nyq * 0.99)
    b, a = butter(3, [max(l / nyq, 1e-4), h / nyq], btype="band")
    padlen = 3 * max(len(a), len(b))
    if hb.shape[1] <= padlen:
        return hb  # too short to filter safely; leave as-is
    return filtfilt(b, a, hb, axis=1)


def epoch_fnirs_run(hb: np.ndarray, fs: float, hb_names: list[str],
                    onsets_sec: np.ndarray, labels: np.ndarray, cfg: dict,
                    subject: str, run: int) -> TrialEpochs | None:
    if len(labels) == 0:
        return None
    classes = cfg_get(cfg, "dataset.classes")
    tmin = float(cfg_get(cfg, "fnirs.tmin", -2.0))
    tmax = float(cfg_get(cfg, "fnirs.tmax", 15.0))
    baseline = cfg_get(cfg, "fnirs.baseline", None)

    n_pre = int(round(tmin * fs))
    n_post = int(round(tmax * fs))
    times = np.arange(n_pre, n_post) / fs
    T = hb.shape[1]

    X, y, uids = [], [], []
    for idx, (on, lab) in enumerate(zip(onsets_sec, labels)):
        c = int(round(on * fs))
        a, b = c + n_pre, c + n_post
        if a < 0 or b > T:
            continue
        seg = hb[:, a:b].copy()                     # (ch, win)
        if baseline:
            bl0 = int(round((baseline[0] - tmin) * fs))
            bl1 = int(round((baseline[1] - tmin) * fs))
            bl0, bl1 = max(bl0, 0), max(bl1, 1)
            seg -= seg[:, bl0:bl1].mean(axis=1, keepdims=True)
        X.append(seg)
        y.append(lab)
        uids.append(f"{subject}_run-{run}_t{idx:02d}")

    if not X:
        return None
    X = np.stack(X, axis=0)
    return TrialEpochs(X=X, y=np.array(y), sfreq=fs, ch_names=list(hb_names),
                       times=times, subjects=np.array([subject] * len(y)),
                       runs=np.array([run] * len(y)), uids=np.array(uids),
                       modality="fnirs", classes=list(classes))


def preprocess_fnirs_run(run: FnirsRun, cfg: dict, subject: str,
                         run_no: int) -> TrialEpochs | None:
    """Full single-run fNIRS pipeline; returns None if the signal is unavailable."""
    if run.data is None:
        return None
    hb, hb_names = intensity_to_hb(run.data, run.ch_names, cfg)
    hb = filter_hemo(hb, run.fs, cfg)
    onsets, labels = fnirs_trial_onsets(run, cfg)
    return epoch_fnirs_run(hb, run.fs, hb_names, onsets, labels, cfg,
                           subject, run_no)


def build_fnirs_epochs(cfg: dict, index: BidsIndex | None = None,
                       subjects: list[str] | None = None,
                       cache: bool = True) -> TrialEpochs | None:
    """Build fNIRS epochs for all runs whose signal is loadable.

    Returns None (with a message) if no fNIRS signal is available -- e.g. the
    intensity is still a MATLAB table and has not been converted.
    """
    # Only cache full-dataset builds (see the note in preprocess_eeg).
    cache = cache and subjects is None
    cache_path = resolve_path(cfg, "paths.cache_dir") / "fnirs_epochs.npz"
    if cache and cache_path.exists():
        print(f"[fnirs] loading cached epochs from {cache_path}")
        return TrialEpochs.load(cache_path)

    if index is None:
        index = discover_dataset(resolve_path(cfg, "paths.bids_root"))

    parts: list[TrialEpochs] = []
    n_unavailable = 0
    for rf in index.runs:
        if rf.fnirs is None:
            continue
        if subjects and rf.subject not in subjects:
            continue
        run = load_fnirs_bbci(rf.fnirs)
        te = preprocess_fnirs_run(run, cfg, rf.subject, rf.run)
        if te is None:
            n_unavailable += 1
            continue
        parts.append(te)
        print(f"[fnirs] {rf.subject} run-{rf.run}: {te.n_trials} epochs "
              f"({run.data_source})")

    if not parts:
        if n_unavailable:
            print(f"[fnirs] {n_unavailable} run(s) have an unconvertible MATLAB-table "
                  "signal -- run tools/convert_fnirs_octave.m. Skipping fNIRS.")
        else:
            print("[fnirs] no fNIRS files found for the selected subjects. "
                  "Skipping fNIRS.")
        return None
    epochs = TrialEpochs.concatenate(parts)
    if cache:
        epochs.save(cache_path)
        print(f"[fnirs] cached -> {cache_path}")
    return epochs
