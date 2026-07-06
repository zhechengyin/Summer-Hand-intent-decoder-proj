"""TrialEpochs -- the array container passed between pipeline stages.

A ``TrialEpochs`` holds time-domain epochs for ONE modality (EEG or fNIRS):

    X        (n_trials, n_channels, n_times)  float32
    y        (n_trials,)                       int    class ids (0..3)
    times    (n_times,)                        float  seconds relative to onset
    subjects (n_trials,)                       str    e.g. 'sub-01'
    runs     (n_trials,)                       int
    uids     (n_trials,)                       str    'sub-01_run-1_t03' (for
                                                      trial alignment across
                                                      modalities during fusion)

Keeping the raw 3-D array (not just a feature matrix) is what lets Stage 8 replay
each trial as a stream of sub-windows -- the "time-domain" requirement.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class TrialEpochs:
    X: np.ndarray
    y: np.ndarray
    sfreq: float
    ch_names: list[str]
    times: np.ndarray
    subjects: np.ndarray
    runs: np.ndarray
    uids: np.ndarray
    modality: str
    classes: list[str]

    # -- basic shape helpers ------------------------------------------------
    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=int)
        self.times = np.asarray(self.times, dtype=float)
        self.subjects = np.asarray(self.subjects)
        self.runs = np.asarray(self.runs)
        self.uids = np.asarray(self.uids)
        n = self.X.shape[0]
        for name, arr in (("y", self.y), ("subjects", self.subjects),
                          ("runs", self.runs), ("uids", self.uids)):
            if arr.shape[0] != n:
                raise ValueError(f"{name} length {arr.shape[0]} != n_trials {n}")
        if self.X.shape[2] != self.times.shape[0]:
            raise ValueError("times length must match X's time axis")

    @property
    def n_trials(self) -> int:
        return self.X.shape[0]

    @property
    def n_channels(self) -> int:
        return self.X.shape[1]

    @property
    def n_times(self) -> int:
        return self.X.shape[2]

    # -- selection / combination -------------------------------------------
    def select(self, mask: np.ndarray) -> "TrialEpochs":
        """Return a new TrialEpochs with the boolean/index-selected trials."""
        mask = np.asarray(mask)
        return TrialEpochs(
            X=self.X[mask], y=self.y[mask], sfreq=self.sfreq,
            ch_names=list(self.ch_names), times=self.times.copy(),
            subjects=self.subjects[mask], runs=self.runs[mask],
            uids=self.uids[mask], modality=self.modality,
            classes=list(self.classes),
        )

    def crop(self, tmin: float, tmax: float) -> "TrialEpochs":
        """Return a copy keeping only samples with tmin <= time <= tmax.

        Used by the window sweep to test different analysis windows without
        re-loading or re-filtering the raw recording.
        """
        mask = (self.times >= tmin - 1e-9) & (self.times <= tmax + 1e-9)
        if not mask.any():
            raise ValueError(f"crop [{tmin},{tmax}] outside epoch "
                             f"[{self.times[0]:.2f},{self.times[-1]:.2f}]")
        return TrialEpochs(
            X=self.X[:, :, mask], y=self.y.copy(), sfreq=self.sfreq,
            ch_names=list(self.ch_names), times=self.times[mask],
            subjects=self.subjects.copy(), runs=self.runs.copy(),
            uids=self.uids.copy(), modality=self.modality,
            classes=list(self.classes),
        )

    @staticmethod
    def concatenate(parts: Sequence["TrialEpochs"]) -> "TrialEpochs":
        parts = [p for p in parts if p is not None and p.n_trials > 0]
        if not parts:
            raise ValueError("nothing to concatenate")
        ref = parts[0]
        for p in parts[1:]:
            if p.ch_names != ref.ch_names:
                raise ValueError("channel mismatch during concatenate")
            if p.n_times != ref.n_times:
                raise ValueError("time-length mismatch during concatenate")
        return TrialEpochs(
            X=np.concatenate([p.X for p in parts], axis=0),
            y=np.concatenate([p.y for p in parts], axis=0),
            sfreq=ref.sfreq, ch_names=list(ref.ch_names), times=ref.times.copy(),
            subjects=np.concatenate([p.subjects for p in parts]),
            runs=np.concatenate([p.runs for p in parts]),
            uids=np.concatenate([p.uids for p in parts]),
            modality=ref.modality, classes=list(ref.classes),
        )

    # -- caching ------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, X=self.X, y=self.y, sfreq=self.sfreq,
            ch_names=np.array(self.ch_names, dtype=object), times=self.times,
            subjects=self.subjects, runs=self.runs, uids=self.uids,
            modality=self.modality, classes=np.array(self.classes, dtype=object),
        )

    def save_array_cache(self, path: str | Path) -> None:
        """Save a memmap-friendly cache directory next to the compressed cache."""
        path = Path(path)
        cache_dir = path.with_name(f"{path.stem}_arrays")
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / "X.npy", self.X)
        np.savez(
            cache_dir / "meta.npz", y=self.y, sfreq=self.sfreq,
            ch_names=np.array(self.ch_names, dtype=object), times=self.times,
            subjects=self.subjects, runs=self.runs, uids=self.uids,
            modality=self.modality, classes=np.array(self.classes, dtype=object),
        )

    @staticmethod
    def load(path: str | Path) -> "TrialEpochs":
        path = Path(path)
        cache_dir = path.with_name(f"{path.stem}_arrays")
        if cache_dir.exists():
            d = np.load(cache_dir / "meta.npz", allow_pickle=True)
            return TrialEpochs(
                X=np.load(cache_dir / "X.npy", mmap_mode="r"), y=d["y"],
                sfreq=float(d["sfreq"]), ch_names=list(d["ch_names"]),
                times=d["times"], subjects=d["subjects"], runs=d["runs"],
                uids=d["uids"], modality=str(d["modality"]),
                classes=list(d["classes"]),
            )
        d = np.load(path, allow_pickle=True)
        return TrialEpochs(
            X=d["X"], y=d["y"], sfreq=float(d["sfreq"]),
            ch_names=list(d["ch_names"]), times=d["times"],
            subjects=d["subjects"], runs=d["runs"], uids=d["uids"],
            modality=str(d["modality"]), classes=list(d["classes"]),
        )

    # -- reporting ----------------------------------------------------------
    def summary(self) -> str:
        counts = {c: int((self.y == i).sum()) for i, c in enumerate(self.classes)}
        subs = sorted(set(self.subjects.tolist()))
        return (f"[{self.modality}] {self.n_trials} trials, "
                f"{self.n_channels} ch x {self.n_times} samp "
                f"({self.times[0]:.1f}..{self.times[-1]:.1f}s @ {self.sfreq:g}Hz) | "
                f"subjects={len(subs)} | per-class={counts}")
