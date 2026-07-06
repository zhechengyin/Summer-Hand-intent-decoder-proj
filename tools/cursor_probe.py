#!/usr/bin/env python
"""Test our best pipeline on ALL subjects in safetyai/CursorSelectionData.

Unlike the root recordings (one session per class), this folder has MULTIPLE
recordings per class per subject (background/blink/left/right x5), across 4
subjects (Alex, BX, Ethan, XQ). That lets us treat each recording as a "run"
and do proper LEAVE-ONE-RECORDING-OUT -- no single-session leakage.

Reuses the project's subject-specific + leave-one-run-out CV harness and our
best EEG front ends (Riemannian tangent, tangent+connectivity multiview).
Chance = 0.25 (4 classes).

Usage: py tools/cursor_probe.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.next_experiments as nx
from src.config import load_config
from src.containers import TrialEpochs
from tools.muse_probe import WIN_SEC, load_recording

DATA = Path(r"C:\Users\angel\safetyai\CursorSelectionData")
CLASSES = ["background", "blink", "left", "right"]


def _label(name: str):
    s = name.lower()
    if "background" in s:
        return 0
    if "blink" in s:
        return 1
    if "left" in s:                 # left / glanceleft
        return 2
    if "right" in s:                # right / glanceright
        return 3
    return None


def _rec_index(name: str):
    m = re.search(r"(\d+)\s*\.csv$", name)
    return int(m.group(1)) if m else 1


def build_epochs():
    Xs, y, subs, runs, fss = [], [], [], [], []
    for subj_dir in sorted(p for p in DATA.iterdir() if p.is_dir()):
        subj = subj_dir.name
        for f in sorted(subj_dir.glob("*.csv")):
            lab = _label(f.name)
            if lab is None:
                continue
            rec = _rec_index(f.name)
            try:
                sig, fs = load_recording(f)
            except Exception as e:
                print(f"  [skip {subj}/{f.name}] {e}")
                continue
            fss.append(fs)
            win = int(round(WIN_SEC * fs))
            if win < 8 or sig.shape[1] < win:
                continue
            n_win = sig.shape[1] // win
            for w in range(n_win):
                Xs.append(sig[:, w * win:(w + 1) * win])
                y.append(lab)
                subs.append(subj)
                runs.append(rec)
    L = min(x.shape[1] for x in Xs)
    X = np.stack([x[:, :L] for x in Xs], axis=0).astype(np.float32)
    fs = float(np.median(fss))
    times = np.arange(L) / fs
    return TrialEpochs(X=X, y=np.array(y), sfreq=fs,
                       ch_names=["TP9", "AF7", "AF8", "TP10"], times=times,
                       subjects=np.array(subs), runs=np.array(runs),
                       uids=np.array([f"{s}_r{r}_t{i}" for i, (s, r)
                                      in enumerate(zip(subs, runs))]),
                       modality="eeg", classes=CLASSES)


def main():
    nx.CLASSES = CLASSES
    eeg = build_epochs()
    fs = eeg.sfreq
    fn0 = np.zeros((eeg.n_trials, 0), dtype=np.float64)
    bands = [(1, 7), (8, 30)]
    per_subj = {s: int((eeg.subjects == s).sum())
                for s in sorted(set(eeg.subjects.tolist()))}
    print("=== safetyai/CursorSelectionData -- all subjects ===")
    print(f"{eeg.n_trials} windows ({WIN_SEC:g}s), 4ch x {eeg.n_times} @ ~{fs:.0f}Hz "
          f"| subjects={per_subj} | chance=0.25")
    print("LORO = leave-one-recording-out (5 recordings/class => leakage-free)\n")

    models = {
        "Riemannian tangent(1-40Hz) + logreg":
            nx.tangent_fold(l_freq=1.0, h_freq=40.0, reg=1e-2, clf="logreg",
                            sfreq=fs),
        "Connectivity[1-7,8-30] plv+imcoh+wpli + logreg":
            nx.connectivity_fold(fs, bands, ["plv", "imcoh", "wpli"], None,
                                 "logreg"),
        "Tangent + Connectivity[plv+imcoh+wpli] + logreg":
            nx.multiview_fold(fs, bands, ["plv", "imcoh", "wpli"], None,
                              reg=1e-2, clf="logreg", l_freq=1.0, h_freq=40.0),
    }
    report = {}
    for name, fold in models.items():
        r = nx.run_within_subject(eeg, fn0, fold, load_config(None), seed=42)
        report[name] = r
        s, l = r["subject"], r["loro"]
        print(f"{name:<50} subj={s['mean_subject_acc']:.3f} "
              f"(bal={s['pooled_bal_acc']:.3f}) | LORO={l['mean_subject_acc']:.3f} "
              f"(bal={l['pooled_bal_acc']:.3f})")

    out = ROOT / "results" / "metrics" / "cursor_probe_all.json"
    out.write_text(json.dumps({"conditions": CLASSES, "chance": 0.25,
                               "per_subject_windows": per_subj,
                               "results": report}, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
