"""Stage 2 -- EEG preprocessing + epoching.

Pipeline per run: notch (line noise) -> band-pass (motor-imagery band) ->
optional resample -> epoch around the imagery-onset marker -> TrialEpochs.

Time-domain structure is preserved: each epoch keeps its full (channels x time)
array so later stages can either compute bandpower or replay sub-windows.
"""
from __future__ import annotations

import warnings

import numpy as np

from .config import cfg_get, resolve_path
from .containers import TrialEpochs
from .load_bids import (BidsIndex, discover_dataset, eeg_trial_onsets,
                        load_eeg_raw)


def preprocess_raw_eeg(raw, cfg: dict):
    """Filter a Raw EEG in place-ish (returns the modified Raw)."""
    l_freq = cfg_get(cfg, "eeg.l_freq", 1.0)
    h_freq = cfg_get(cfg, "eeg.h_freq", 40.0)
    notch = cfg_get(cfg, "eeg.notch", None)
    resample = cfg_get(cfg, "eeg.resample", None)
    picks = cfg_get(cfg, "eeg.picks", None)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if picks:
            keep = [ch for ch in picks if ch in raw.ch_names]
            if keep:
                raw.pick(keep)
        # Attach a standard montage where channel names allow (best-effort).
        try:
            import mne

            raw.set_montage("standard_1020", on_missing="ignore", match_case=False)
        except Exception:
            pass
        if notch:
            notch = [f for f in np.atleast_1d(notch) if f < raw.info["sfreq"] / 2]
            if notch:
                raw.notch_filter(freqs=notch, verbose="ERROR")
        raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin",
                   verbose="ERROR")
        if resample:
            raw.resample(float(resample), verbose="ERROR")
    return raw


def epoch_eeg_run(raw, onsets_sec: np.ndarray, labels: np.ndarray,
                  cfg: dict, subject: str, run: int) -> TrialEpochs | None:
    """Cut labelled epochs from one preprocessed run."""
    import mne

    if len(labels) == 0:
        return None
    classes = cfg_get(cfg, "dataset.classes")
    tmin = float(cfg_get(cfg, "eeg.tmin", 0.0))
    tmax = float(cfg_get(cfg, "eeg.tmax", 5.0))
    baseline = cfg_get(cfg, "eeg.baseline", None)
    baseline = tuple(baseline) if baseline else None
    reject_uv = cfg_get(cfg, "eeg.reject_uv", None)
    reject = {"eeg": reject_uv * 1e-6} if reject_uv else None

    sfreq = raw.info["sfreq"]
    samples = np.round(onsets_sec * sfreq).astype(int) + raw.first_samp
    codes = labels.astype(int) + 1  # MNE prefers non-zero event codes
    events = np.column_stack([samples, np.zeros_like(samples), codes])
    event_id = {c: i + 1 for i, c in enumerate(classes)}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                            baseline=baseline, preload=True, reject=reject,
                            on_missing="ignore", verbose="ERROR")

    X = epochs.get_data(copy=True)                     # (n, ch, time)
    y = epochs.events[:, 2] - 1                         # back to 0..3
    kept = epochs.selection                             # indices into `events`
    uids = np.array([f"{subject}_run-{run}_t{idx:02d}" for idx in kept])
    subs = np.array([subject] * len(y))
    runs = np.array([run] * len(y))
    return TrialEpochs(X=X, y=y, sfreq=sfreq, ch_names=list(epochs.ch_names),
                       times=epochs.times, subjects=subs, runs=runs, uids=uids,
                       modality="eeg", classes=list(classes))


def build_eeg_epochs(cfg: dict, index: BidsIndex | None = None,
                     subjects: list[str] | None = None,
                     cache: bool = True) -> TrialEpochs:
    """Load + preprocess + epoch every (subject, run) into one TrialEpochs.

    Results are cached to ``data/cache/eeg_epochs.npz`` and reused unless
    ``cache=False``.
    """
    # Only cache full-dataset builds; a subject-filtered build must not read or
    # write the shared cache (it would otherwise serve stale/partial epochs).
    cache = cache and subjects is None
    cache_path = resolve_path(cfg, "paths.cache_dir") / "eeg_epochs.npz"
    if cache and cache_path.exists():
        print(f"[eeg] loading cached epochs from {cache_path}")
        return TrialEpochs.load(cache_path)

    if index is None:
        index = discover_dataset(resolve_path(cfg, "paths.bids_root"))

    parts: list[TrialEpochs] = []
    for rf in index.runs:
        if rf.eeg is None:
            continue
        if subjects and rf.subject not in subjects:
            continue
        raw = load_eeg_raw(rf.eeg)
        raw = preprocess_raw_eeg(raw, cfg)
        onsets, labels = eeg_trial_onsets(raw, cfg)
        te = epoch_eeg_run(raw, onsets, labels, cfg, rf.subject, rf.run)
        if te is not None:
            parts.append(te)
            print(f"[eeg] {rf.subject} run-{rf.run}: {te.n_trials} epochs")

    if not parts:
        raise RuntimeError("No EEG epochs produced -- is the dataset downloaded?")
    epochs = TrialEpochs.concatenate(parts)
    if cache:
        epochs.save(cache_path)
        print(f"[eeg] cached -> {cache_path}")
    return epochs
