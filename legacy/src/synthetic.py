"""Synthetic, class-structured epochs for smoke-testing the whole pipeline.

This lets `python main.py smoke` exercise features -> fusion -> N1 -> evaluation
-> N2 -> replay end-to-end WITHOUT the (large) real dataset, and gives the
classifier genuine (but noisy) structure so accuracy lands above chance.

EEG  : class-specific oscillations in mu/beta on sensorimotor channels + noise.
fNIRS: class-specific canonical hemodynamic responses (already HbO/HbR, so the
       Beer-Lambert step is bypassed) + noise.

The EEG and fNIRS generators share the same per-trial labels and UIDs so the
fusion path aligns correctly.
"""
from __future__ import annotations

import numpy as np

from .config import cfg_get
from .containers import TrialEpochs

EEG_CH = ["C3", "C4", "Cz", "CP1", "CP2", "CP5", "CP6", "FC1", "FC2", "FC5",
          "FC6", "Fp2", "O1", "O2", "Oz", "P3", "P4", "Pz"]

# class -> [(channel, freq_hz, amplitude), ...]
CLASS_EEG = {
    0: [("C3", 10.0, 1.3), ("CP1", 10.0, 0.7)],    # reach  -> left mu
    1: [("C4", 22.0, 1.3), ("CP2", 22.0, 0.7)],    # grasp  -> right beta
    2: [("Cz", 11.0, 1.3), ("FC1", 18.0, 0.6)],    # lift   -> central
    3: [("C4", 26.0, 1.2), ("FC2", 26.0, 0.7)],    # twist  -> right high-beta
}
# class -> fNIRS channel-pair indices that activate
CLASS_FNIRS = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8], 3: [9, 10, 11]}


def _trial_plan(n_subjects, n_runs, per_class, n_classes, seed):
    """Deterministic balanced, shuffled labels per (subject, run)."""
    rng = np.random.default_rng(seed)
    plan = []
    for s in range(1, n_subjects + 1):
        subj = f"sub-{s:02d}"
        for r in range(1, n_runs + 1):
            labels = np.repeat(np.arange(n_classes), per_class)
            rng.shuffle(labels)
            for t, lab in enumerate(labels):
                plan.append((subj, r, t, int(lab)))
    return plan


def synthetic_eeg_epochs(cfg, n_subjects=3, n_runs=2, per_class=8,
                         fs=128.0, dur=4.0, noise=1.0, seed=0) -> TrialEpochs:
    classes = cfg_get(cfg, "dataset.classes")
    rng = np.random.default_rng(seed + 1)
    plan = _trial_plan(n_subjects, n_runs, per_class, len(classes), seed)
    n_times = int(fs * dur)
    t = np.arange(n_times) / fs
    ch_idx = {c: i for i, c in enumerate(EEG_CH)}
    # ERD/ERS-like envelope: suppression early, rebound late
    env = 1.0 + 0.4 * np.tanh((t - dur / 2) / 0.6)

    X = np.empty((len(plan), len(EEG_CH), n_times), dtype=np.float32)
    y, subs, runs, uids = [], [], [], []
    for k, (subj, r, tt, lab) in enumerate(plan):
        x = rng.standard_normal((len(EEG_CH), n_times)) * noise
        for ch, freq, amp in CLASS_EEG[lab]:
            phase = rng.uniform(0, 2 * np.pi)
            x[ch_idx[ch]] += amp * env * np.sin(2 * np.pi * freq * t + phase)
        X[k] = x
        y.append(lab); subs.append(subj); runs.append(r)
        uids.append(f"{subj}_run-{r}_t{tt:02d}")
    return TrialEpochs(X=X, y=np.array(y), sfreq=fs, ch_names=list(EEG_CH),
                       times=t, subjects=np.array(subs), runs=np.array(runs),
                       uids=np.array(uids), modality="eeg", classes=list(classes))


def _hrf(t):
    """Canonical-ish hemodynamic response (peak ~6 s), zero before onset."""
    h = np.where(t > 0, (t ** 2.0) * np.exp(-t / 1.6), 0.0)
    return h / (h.max() + 1e-9)


def synthetic_fnirs_epochs(cfg, n_subjects=3, n_runs=2, per_class=8,
                           fs=7.8125, n_pairs=12, noise=0.35, seed=0) -> TrialEpochs:
    classes = cfg_get(cfg, "dataset.classes")
    tmin = float(cfg_get(cfg, "fnirs.tmin", -2.0))
    tmax = float(cfg_get(cfg, "fnirs.tmax", 15.0))
    rng = np.random.default_rng(seed + 2)
    plan = _trial_plan(n_subjects, n_runs, per_class, len(classes), seed)
    n_times = int(round((tmax - tmin) * fs))
    times = np.arange(n_times) / fs + tmin
    hrf = _hrf(times)
    ch_names = ([f"pair{p:02d}_HbO" for p in range(n_pairs)]
                + [f"pair{p:02d}_HbR" for p in range(n_pairs)])

    X = np.empty((len(plan), 2 * n_pairs, n_times), dtype=np.float32)
    y, subs, runs, uids = [], [], [], []
    for k, (subj, r, tt, lab) in enumerate(plan):
        x = rng.standard_normal((2 * n_pairs, n_times)) * noise
        # slow drift
        x += 0.2 * np.sin(2 * np.pi * 0.02 * times + rng.uniform(0, 6.28))
        for p in CLASS_FNIRS[lab]:
            amp = 1.0 + 0.3 * rng.standard_normal()
            x[p] += amp * hrf                 # HbO increases
            x[n_pairs + p] += -0.4 * amp * hrf  # HbR decreases
        X[k] = x
        y.append(lab); subs.append(subj); runs.append(r)
        uids.append(f"{subj}_run-{r}_t{tt:02d}")
    return TrialEpochs(X=X, y=np.array(y), sfreq=fs, ch_names=ch_names,
                       times=times, subjects=np.array(subs), runs=np.array(runs),
                       uids=np.array(uids), modality="fnirs", classes=list(classes))


def make_synthetic(cfg, **kw):
    """Return (eeg_epochs, fnirs_epochs) sharing labels/UIDs for fusion."""
    seed = int(cfg_get(cfg, "seed", 42))
    eeg = synthetic_eeg_epochs(cfg, seed=seed, **kw)
    fnirs = synthetic_fnirs_epochs(cfg, seed=seed, **kw)
    return eeg, fnirs
