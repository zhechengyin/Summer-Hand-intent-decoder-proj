#!/usr/bin/env python
"""Generalization probe: apply our best ds004022 models to a SECOND OpenNeuro
motor-imagery dataset with a KNOWN decodable contrast.

Dataset: OpenNeuro ds004362 (PhysioNet EEG Motor Movement/Imagery, eegmmidb),
64-ch EEGLAB .set @ 160 Hz. Runs 4/8/12 = imagined fist movement; event value
...T1 = LEFT fist, ...T2 = RIGHT fist. We decode LEFT vs RIGHT hand motor
imagery (classic, well-separated MI). Chance = 0.50.

This validates that connectivity->PLS and Riemannian tangent (our best ds004022
front ends) extract real signal when a real contrast exists -- confirming the
near-chance ds004022 result is task difficulty, not a pipeline bug.

Downloads only what it needs (a few small .set files) over public HTTPS.
"""
from __future__ import annotations

import csv
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.next_experiments as nx
from src.config import load_config
from src.containers import TrialEpochs

DS = "ds004362"
S3 = f"https://s3.amazonaws.com/openneuro.org/{DS}"
IMAGERY_FIST_RUNS = [4, 8, 12]           # imagined left/right fist
MOTOR = ["Fc5", "Fc3", "Fc1", "Fcz", "Fc2", "Fc4", "Fc6",
         "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
         "Cp5", "Cp3", "Cp1", "Cpz", "Cp2", "Cp4", "Cp6"]
SFREQ = 160.0
TMIN, TMAX = 0.5, 3.5                     # window after cue onset (s)


def _fetch(subj, run, kind, ext):
    rel = f"{subj}/eeg/{subj}_task-motion_run-{run}_{kind}.{ext}"
    dst = ROOT / "data" / DS / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        url = f"{S3}/{rel}"
        print(f"  downloading {rel} ...", flush=True)
        urllib.request.urlretrieve(url, dst)
    return dst


def _load_run(subj, run):
    import mne
    set_path = _fetch(subj, run, "eeg", "set")
    ev_path = _fetch(subj, run, "events", "tsv")
    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose="ERROR")
    sf = float(raw.info["sfreq"])
    names = raw.ch_names
    low = {n.lower(): i for i, n in enumerate(names)}
    idx = [low[c.lower()] for c in MOTOR if c.lower() in low]
    picked = [names[i] for i in idx]
    data = raw.get_data()[idx]                       # (ch, time)
    win = int(round((TMAX - TMIN) * sf))
    X, y, uids = [], [], []
    with open(ev_path, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            val = row["value"].strip()
            if val.endswith("T1"):
                lab = 0                               # left fist
            elif val.endswith("T2"):
                lab = 1                               # right fist
            else:
                continue
            onset = float(row["onset"])
            s0 = int(round((onset + TMIN) * sf))
            s1 = s0 + win
            if s1 > data.shape[1]:
                continue
            X.append(data[:, s0:s1])
            y.append(lab)
            uids.append(f"{subj}_run-{run}_t{len(uids):02d}")
    X = np.asarray(X, dtype=np.float32)
    return X, np.asarray(y, int), picked, sf, uids


def build_epochs(subjects):
    Xs, ys, subs, runs, uids = [], [], [], [], []
    ref_ch = None
    for subj in subjects:
        try:
            loaded = [(_load_run(subj, run), run) for run in IMAGERY_FIST_RUNS]
        except Exception as e:                       # skip subjects with bad runs
            print(f"  [skip {subj}] {e}", flush=True)
            continue
        for (X, y, ch, sf, uid), run in loaded:
            if ref_ch is None:
                ref_ch = ch
            if ch != ref_ch or abs(sf - SFREQ) > 1e-6:
                print(f"  [skip {subj} run-{run}] channel/sfreq mismatch",
                      flush=True)
                continue
            Xs.append(X)
            ys.append(y)
            subs.extend([subj] * len(y))
            runs.extend([run] * len(y))
            uids.extend(uid)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    times = np.arange(X.shape[2]) / SFREQ + TMIN
    return TrialEpochs(X=X, y=y, sfreq=SFREQ, ch_names=ref_ch, times=times,
                       subjects=np.array(subs), runs=np.array(runs),
                       uids=np.array(uids), modality="eeg",
                       classes=["left", "right"])


def main():
    subjects = [f"sub-{i:03d}" for i in range(1, 6)]   # 5 subjects
    print(f"=== ds004362 left/right hand MI probe: {subjects} ===")
    nx.CLASSES = ["left", "right"]                      # 2-class metrics/onehot
    eeg = build_epochs(subjects)
    n_per = {c: int((eeg.y == i).sum()) for i, c in enumerate(eeg.classes)}
    print(f"Loaded {eeg.n_trials} trials, {eeg.n_channels} motor ch x "
          f"{eeg.n_times} samp @ {SFREQ:g}Hz | per-class={n_per} | "
          f"subjects={len(subjects)}, runs={IMAGERY_FIST_RUNS}")

    cfg = load_config(None)
    fn0 = np.zeros((eeg.n_trials, 0), dtype=np.float64)  # EEG-only (no fNIRS)
    bands = [(8, 13), (13, 30)]

    def do(tag, fold):
        import time
        t0 = time.time()
        r = nx.run_within_subject(eeg, fn0, fold, cfg, seed=42)
        s, l = r["subject"], r["loro"]
        print(f"{tag:<40} subj acc={s['mean_subject_acc']:.4f} "
              f"(bal={s['pooled_bal_acc']:.3f}) | LORO acc={l['mean_subject_acc']:.4f} "
              f"(bal={l['pooled_bal_acc']:.3f})   [{time.time()-t0:.0f}s]")
        return r

    report = {}
    print("\n--- our best ds004022 front ends on a decodable contrast (chance=0.50) ---")
    report["riemann"] = do("Riemannian tangent (logreg)",
                           nx.tangent_fold(reg=1e-3, clf="logreg", sfreq=SFREQ))
    report["conn"] = do("connectivity plv+imcoh+wpli (logreg)",
                        nx.connectivity_fold(SFREQ, bands,
                                             ["plv", "imcoh", "wpli"], None,
                                             "logreg"))
    report["conn_pls"] = do("connectivity -> PLS(8) -> LDA",
                            nx.conn_pls_fold(SFREQ, bands,
                                             ["plv", "imcoh", "wpli"], None, 8,
                                             "lda"))

    import json
    out = ROOT / "results" / "metrics" / "eegmmidb_probe.json"
    out.write_text(json.dumps({"dataset": DS, "subjects": subjects,
                               "contrast": "left_vs_right_hand_MI",
                               "chance": 0.5, "n_trials": int(eeg.n_trials),
                               "per_class": n_per, "results": report},
                              indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
