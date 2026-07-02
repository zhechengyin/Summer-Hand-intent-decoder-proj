"""Stage 1 -- dataset loading / discovery layer for OpenNeuro ds004022.

Design principles (per the project brief):
* Do NOT hard-code filenames. Discover subjects/runs by globbing BIDS-style
  paths and parsing entities with a regex.
* ds004022 has NO events.tsv. EEG labels come from the EEGLAB ``.set`` event
  struct (BrainVision markers), read via MNE annotations. fNIRS labels come from
  the BBCI ``.mat`` marker struct (``mrk.toe``).
* fNIRS raw intensity (``cnt.x``) is a MATLAB ``table`` (MCOS object) that Python
  cannot deserialise -> we detect this and look for a converted sibling file
  (see tools/convert_fnirs_octave.m). Everything degrades gracefully.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.io as sio

from .config import cfg_get, resolve_path

# sub-01_task-motorimagery_run-1_eeg.set  ->  ('01', '1')
_ENTITY_RE = re.compile(r"sub-(?P<sub>[A-Za-z0-9]+).*?run-(?P<run>\d+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
@dataclass
class RunFiles:
    subject: str          # 'sub-01'
    run: int              # 1
    eeg: Path | None = None
    fnirs: Path | None = None


@dataclass
class BidsIndex:
    root: Path
    runs: list[RunFiles] = field(default_factory=list)

    @property
    def subjects(self) -> list[str]:
        return sorted({r.subject for r in self.runs})

    def for_subject(self, subject: str) -> list[RunFiles]:
        return [r for r in self.runs if r.subject == subject]


def _parse_entities(path: Path) -> tuple[str, int] | None:
    m = _ENTITY_RE.search(path.name)
    if not m:
        return None
    return f"sub-{m.group('sub')}", int(m.group("run"))


def discover_dataset(bids_root: str | Path) -> BidsIndex:
    """Glob EEG (.set) and fNIRS (.mat) files and group them by (subject, run).

    Works with plain filesystem globbing (no mne-bids dependency). If mne-bids is
    installed you could instead build a BIDSPath, but globbing is robust to the
    quirks of this particular dataset.
    """
    root = Path(bids_root)
    index = BidsIndex(root=root)
    table: dict[tuple[str, int], RunFiles] = {}

    for setf in sorted(root.glob("sub-*/eeg/*_eeg.set")):
        ent = _parse_entities(setf)
        if ent is None:
            continue
        rf = table.setdefault(ent, RunFiles(subject=ent[0], run=ent[1]))
        rf.eeg = setf

    for matf in sorted(root.glob("sub-*/fnirs/*_nirs.mat")):
        if matf.name.endswith("_nirs_converted.mat"):
            continue
        ent = _parse_entities(matf)
        if ent is None:
            continue
        rf = table.setdefault(ent, RunFiles(subject=ent[0], run=ent[1]))
        rf.fnirs = matf

    index.runs = [table[k] for k in sorted(table)]
    return index


# ---------------------------------------------------------------------------
# EEG
# ---------------------------------------------------------------------------
def load_eeg_raw(path: str | Path):
    """Load an EEGLAB .set (+.fdt) file as an MNE Raw object (preloaded)."""
    import mne

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_eeglab(str(path), preload=True, verbose="ERROR")
    # EEGLAB channel labels arrive space-padded ('C3          '); tidy them.
    raw.rename_channels({ch: ch.strip() for ch in raw.ch_names})
    return raw


def _norm_marker(s: str) -> str:
    """Whitespace-insensitive marker key: 'S  3' -> 'S3', ' 3 ' -> '3'."""
    return re.sub(r"\s+", "", str(s))


def _marker_lookup(cfg: dict) -> tuple[dict[str, int], set[str]]:
    """Return (normalized-marker -> class_id, {normalized MI-onset markers})."""
    classes = cfg_get(cfg, "dataset.classes")
    class_markers = cfg_get(cfg, "dataset.eeg_class_markers", {})
    marker_to_id: dict[str, int] = {}
    for cid, cname in enumerate(classes):
        for alias in class_markers.get(cname, []):
            marker_to_id[_norm_marker(alias)] = cid
    onset_markers = {_norm_marker(m)
                     for m in cfg_get(cfg, "dataset.eeg_mi_onset_markers", [])}
    return marker_to_id, onset_markers


def eeg_trial_onsets(raw, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Pair each class-cue marker with the imagery-onset that follows it.

    Returns
    -------
    onsets_sec : (n_trials,) float   imagery-window anchor, seconds from start
    labels     : (n_trials,) int     class id 0..3
    """
    marker_to_id, onset_markers = _marker_lookup(cfg)
    fallback = float(cfg_get(cfg, "eeg.fallback_onset_offset", 10.0))

    ann = raw.annotations
    ev = sorted(zip(ann.onset, ann.description), key=lambda t: t[0])
    cues = [(on, marker_to_id[_norm_marker(d)]) for on, d in ev
            if _norm_marker(d) in marker_to_id]
    onsets_mi = [on for on, d in ev if _norm_marker(d) in onset_markers]

    onsets, labels = [], []
    for i, (cue_on, cid) in enumerate(cues):
        next_cue = cues[i + 1][0] if i + 1 < len(cues) else np.inf
        following = [o for o in onsets_mi if cue_on < o < next_cue]
        anchor = following[0] if following else cue_on + fallback
        onsets.append(anchor)
        labels.append(cid)
    return np.asarray(onsets, dtype=float), np.asarray(labels, dtype=int)


def eeg_event_inventory(raw) -> dict[str, int]:
    """Count unique annotation descriptions (for the `inspect` command)."""
    from collections import Counter

    return dict(sorted(Counter(map(str, raw.annotations.description)).items()))


# ---------------------------------------------------------------------------
# fNIRS (BBCI toolbox .mat)
# ---------------------------------------------------------------------------
@dataclass
class FnirsRun:
    fs: float
    ch_names: list[str]                 # e.g. 'S1_D1 760'
    wavelengths: np.ndarray             # e.g. [760, 850]
    mrk_pos: np.ndarray                 # marker sample indices
    mrk_toe: np.ndarray                 # marker type codes (1..)
    data: np.ndarray | None             # (n_samples, n_channels) or None
    data_source: str                    # 'mat' | 'converted' | 'unavailable'


def _unwrap(obj, name: str):
    """ds004022 double-wraps structs: nirs_data.cnt.cnt, nirs_data.mrk.mrk."""
    while hasattr(obj, "_fieldnames") and obj._fieldnames == [name]:
        obj = getattr(obj, name)
    return obj


def _is_mcos_table(x) -> bool:
    return isinstance(x, np.ndarray) and x.dtype.names is not None and x.dtype != float


def load_fnirs_bbci(path: str | Path) -> FnirsRun:
    """Load a ds004022 fNIRS ``*_nirs.mat`` (BBCI format).

    Markers and montage always load. The intensity matrix loads only if a
    converted sibling (``*_nirs_converted.mat``, produced by
    tools/convert_fnirs_octave.m) exists, because the original ``cnt.x`` is a
    MATLAB table object Python cannot read.
    """
    path = Path(path)
    m = sio.loadmat(path, struct_as_record=False, squeeze_me=True)["nirs_data"]
    cnt = _unwrap(m.cnt, "cnt")
    mrk = _unwrap(m.mrk, "mrk")

    fs = float(getattr(cnt, "fs"))
    clab = [str(c).strip() for c in np.atleast_1d(getattr(cnt, "clab"))]
    wl = np.atleast_1d(getattr(cnt, "wavelengths")).astype(float)
    mrk_pos = np.atleast_1d(getattr(mrk, "pos")).astype(int).ravel()
    mrk_toe = np.atleast_1d(getattr(mrk, "toe")).astype(int).ravel()

    data, source = None, "unavailable"
    converted = path.with_name(path.name.replace("_nirs.mat", "_nirs_converted.mat"))
    if converted.exists():
        conv = sio.loadmat(converted, squeeze_me=True)
        data = np.asarray(conv["X"], dtype=float)
        source = "converted"
        if "clab" in conv:
            clab = [str(c).strip() for c in np.atleast_1d(conv["clab"])]
        if "mrk_pos" in conv:
            mrk_pos = np.atleast_1d(conv["mrk_pos"]).astype(int).ravel()
            mrk_toe = np.atleast_1d(conv["mrk_toe"]).astype(int).ravel()
    else:
        x = getattr(cnt, "x")
        if _is_mcos_table(x):
            warnings.warn(
                f"{path.name}: fNIRS intensity is a MATLAB table (MCOS) and cannot "
                "be read in Python. Run tools/convert_fnirs_octave.m to enable the "
                "real-data fNIRS branch. Skipping this run's signal.",
                RuntimeWarning,
            )
        elif isinstance(x, np.ndarray) and x.dtype != object:
            data = np.asarray(x, dtype=float)
            source = "mat"

    return FnirsRun(fs=fs, ch_names=clab, wavelengths=wl, mrk_pos=mrk_pos,
                    mrk_toe=mrk_toe, data=data, data_source=source)


def fnirs_trial_onsets(run: FnirsRun, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (onset_sec, class_id) trials from the fNIRS marker stream."""
    toe_map = cfg_get(cfg, "dataset.fnirs_class_toe", {})
    classes = cfg_get(cfg, "dataset.classes")
    onset_toe = int(cfg_get(cfg, "dataset.fnirs_mi_onset_toe", 8))
    code_to_id = {int(toe_map[c]): i for i, c in enumerate(classes) if c in toe_map}

    pos_sec = run.mrk_pos / run.fs
    cues = [(pos_sec[i], code_to_id[t]) for i, t in enumerate(run.mrk_toe)
            if int(t) in code_to_id]
    mi = [pos_sec[i] for i, t in enumerate(run.mrk_toe) if int(t) == onset_toe]

    onsets, labels = [], []
    for i, (cue_on, cid) in enumerate(cues):
        next_cue = cues[i + 1][0] if i + 1 < len(cues) else np.inf
        following = [o for o in mi if cue_on < o < next_cue]
        onsets.append(following[0] if following else cue_on)
        labels.append(cid)
    return np.asarray(onsets, dtype=float), np.asarray(labels, dtype=int)


# ---------------------------------------------------------------------------
# Inspector (backs `python main.py inspect`)
# ---------------------------------------------------------------------------
def inspect_dataset(cfg: dict, max_runs: int = 2) -> None:
    root = resolve_path(cfg, "paths.bids_root")
    print(f"BIDS root: {root}")
    if not root.exists():
        print("  (not found -- see data/README.md to download ds004022)")
        return
    index = discover_dataset(root)
    print(f"Subjects: {index.subjects}")
    print(f"Runs discovered: {len(index.runs)}")

    shown = 0
    for rf in index.runs:
        if shown >= max_runs:
            break
        print(f"\n--- {rf.subject} run-{rf.run} ---")
        print(f"  eeg:   {rf.eeg.name if rf.eeg else 'MISSING'}")
        print(f"  fnirs: {rf.fnirs.name if rf.fnirs else 'MISSING'}")
        if rf.eeg:
            try:
                raw = load_eeg_raw(rf.eeg)
                print(f"  EEG: {len(raw.ch_names)} ch @ {raw.info['sfreq']:g} Hz, "
                      f"{raw.n_times / raw.info['sfreq']:.0f}s")
                print(f"  EEG markers: {eeg_event_inventory(raw)}")
                on, lab = eeg_trial_onsets(raw, cfg)
                print(f"  EEG trials: {len(lab)} "
                      f"(per-class {np.bincount(lab, minlength=4).tolist()})")
            except Exception as e:  # pragma: no cover
                print(f"  EEG load error: {e}")
        if rf.fnirs:
            try:
                run = load_fnirs_bbci(rf.fnirs)
                print(f"  fNIRS: {len(run.ch_names)} ch @ {run.fs:g} Hz, "
                      f"wl={run.wavelengths.tolist()}, data_source={run.data_source}")
                on, lab = fnirs_trial_onsets(run, cfg)
                print(f"  fNIRS trials: {len(lab)} "
                      f"(per-class {np.bincount(lab, minlength=4).tolist()})")
            except Exception as e:  # pragma: no cover
                print(f"  fNIRS load error: {e}")
        shown += 1
