#!/usr/bin/env python
"""Test our best EEG model pipeline on the StreamAdapt consumer-EEG data
(Muse2, 4 channels TP9/AF7/AF8/TP10) from the safetyai repo.

This is NOT motor imagery -- each recording is a continuous session of one
condition (Background / Blink / GlanceLeft / GlanceRight / Jaw / Motion). We
window each recording and classify the condition, applying the SAME geometric /
connectivity front ends used on the MI datasets.

Two honesty notes baked in:
  * The discriminative signal for glances/blinks is low-frequency EOG, so the
    tangent front end uses a broadband 1-40 Hz filter (the 8-30 Hz MI band would
    delete it).
  * There is only ONE recording per class, so random CV can exploit
    session-level drift/impedance (optimistic). We therefore also report a
    chronological 70/30 split (train early, test late) which removes
    adjacent-window leakage and is the fair number.

Usage: py tools/muse_probe.py --subject Alex
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.riemannian import RiemannianTangentSpace
from tools.next_experiments import connectivity_batch

REPO = Path(r"C:\Users\angel\safetyai")
CH = ["TP9_RAW", "AF7_RAW", "AF8_RAW", "TP10_RAW"]
# 4-class eye/idle task (chance 0.25); files are "<Subject><Cond>1.csv"
CONDITIONS = ["Background", "Blink", "GlanceLeft", "GlanceRight"]
WIN_SEC = 2.0


def load_recording(path):
    df = pd.read_csv(path, low_memory=False)
    df = df[df["PACKET_TYPE"] == "EEG"]
    raw = df[CH].apply(pd.to_numeric, errors="coerce").dropna()
    ms = pd.to_numeric(df.loc[raw.index, "ms_ELAPSED"], errors="coerce")
    span = float(ms.max() - ms.min()) / 1000.0
    fs = len(raw) / span if span > 0 else 256.0
    return raw.to_numpy(dtype=np.float64).T, fs          # (4, n_samples), fs


def build_epochs(subject):
    Xs, y, tpos, fss = [], [], [], []
    for ci, cond in enumerate(CONDITIONS):
        path = REPO / f"{subject}{cond}1.csv"
        if not path.exists():
            print(f"  [missing] {path.name}")
            continue
        sig, fs = load_recording(path)
        fss.append(fs)
        win = int(round(WIN_SEC * fs))
        n_win = sig.shape[1] // win
        for w in range(n_win):
            Xs.append(sig[:, w * win:(w + 1) * win])
            y.append(ci)
            tpos.append(w / max(1, n_win - 1))           # 0..1 within recording
    # pad/truncate windows to a common length (fs jitter across files)
    L = min(x.shape[1] for x in Xs)
    X = np.stack([x[:, :L] for x in Xs], axis=0).astype(np.float32)
    return X, np.array(y), np.array(tpos), float(np.median(fss))


# ---- feature front ends (fit on train only) -------------------------------
def _onehot(y, k):
    Y = np.zeros((len(y), k)); Y[np.arange(len(y)), y] = 1.0
    return Y


def tangent_fit_predict(Xtr, ytr, Xte, fs, clf):
    ts = RiemannianTangentSpace(sfreq=fs, l_freq=1.0, h_freq=40.0, reg=1e-2)
    ts.fit(Xtr, ytr)
    ztr, zte = ts.transform(Xtr), ts.transform(Xte)
    pipe = Pipeline([("sc", StandardScaler()), ("clf", clf)]).fit(ztr, ytr)
    return pipe.predict(zte)


def conn_fit_predict(Xtr, ytr, Xte, fs, bands, mets, pls_k=None):
    ctr = connectivity_batch(Xtr, fs, bands, mets, None)
    cte = connectivity_batch(Xte, fs, bands, mets, None)
    sc = StandardScaler().fit(ctr)
    ctr, cte = sc.transform(ctr), sc.transform(cte)
    if pls_k:
        k = min(pls_k, ctr.shape[1], ctr.shape[0] - 1)
        pls = PLSRegression(n_components=max(1, k)).fit(ctr, _onehot(ytr, len(CONDITIONS)))
        cl = LinearDiscriminantAnalysis().fit(pls.transform(ctr), ytr)
        return cl.predict(pls.transform(cte))
    cl = LogisticRegression(max_iter=2000).fit(ctr, ytr)
    return cl.predict(cte)


def multiview_fit_predict(Xtr, ytr, Xte, fs, bands, mets, clf):
    """Early-concat multiview: tangent(1-40Hz) + connectivity[plv+imcoh+wpli]."""
    ts = RiemannianTangentSpace(sfreq=fs, l_freq=1.0, h_freq=40.0, reg=1e-2)
    ts.fit(Xtr, ytr)
    ztr = np.hstack([ts.transform(Xtr),
                     connectivity_batch(Xtr, fs, bands, mets, None)])
    zte = np.hstack([ts.transform(Xte),
                     connectivity_batch(Xte, fs, bands, mets, None)])
    pipe = Pipeline([("sc", StandardScaler()), ("clf", clf)]).fit(ztr, ytr)
    return pipe.predict(zte)


def logvar_fit_predict(Xtr, ytr, Xte, clf):
    ftr = np.log(np.var(Xtr, axis=2) + 1e-8)
    fte = np.log(np.var(Xte, axis=2) + 1e-8)
    pipe = Pipeline([("sc", StandardScaler()), ("clf", clf)]).fit(ftr, ytr)
    return pipe.predict(fte)


MODELS = {
    "log-variance + LDA":
        lambda Xtr, ytr, Xte, fs: logvar_fit_predict(
            Xtr, ytr, Xte, LinearDiscriminantAnalysis()),
    "Riemannian tangent(1-40Hz) + LDA":
        lambda Xtr, ytr, Xte, fs: tangent_fit_predict(
            Xtr, ytr, Xte, fs, LinearDiscriminantAnalysis()),
    "Riemannian tangent(1-40Hz) + logreg":
        lambda Xtr, ytr, Xte, fs: tangent_fit_predict(
            Xtr, ytr, Xte, fs, LogisticRegression(max_iter=2000)),
    "Connectivity[1-7,8-30] plv+imcoh+wpli + logreg":
        lambda Xtr, ytr, Xte, fs: conn_fit_predict(
            Xtr, ytr, Xte, fs, [(1, 7), (8, 30)], ["plv", "imcoh", "wpli"]),
    "Connectivity -> PLS(6) -> LDA":
        lambda Xtr, ytr, Xte, fs: conn_fit_predict(
            Xtr, ytr, Xte, fs, [(1, 7), (8, 30)], ["plv", "imcoh", "wpli"], 6),
    "Tangent(1-40) + Connectivity[plv+imcoh+wpli] + logreg":
        lambda Xtr, ytr, Xte, fs: multiview_fit_predict(
            Xtr, ytr, Xte, fs, [(1, 7), (8, 30)], ["plv", "imcoh", "wpli"],
            LogisticRegression(max_iter=2000)),
}


def score(y_true, y_pred):
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
    }


def eval_kfold(X, y, fs, fn, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    yp = np.empty_like(y)
    for tr, te in skf.split(X, y):
        yp[te] = fn(X[tr], y[tr], X[te], fs)
    return score(y, yp)


def eval_chrono(X, y, tpos, fs, fn, frac=0.7):
    tr = tpos <= frac
    te = ~tr
    yp = fn(X[tr], y[tr], X[te], fs)
    return score(y[te], yp)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", default="Alex")
    args = ap.parse_args()

    X, y, tpos, fs = build_epochs(args.subject)
    counts = {CONDITIONS[i]: int((y == i).sum()) for i in np.unique(y)}
    print(f"=== StreamAdapt Muse probe: subject {args.subject} ===")
    print(f"{len(y)} windows ({WIN_SEC:g}s), {X.shape[1]}ch x {X.shape[2]} samp @ "
          f"~{fs:.0f}Hz | classes(chance={1/len(CONDITIONS):.2f})={counts}")
    print("\nProtocols: [kfold] = stratified 5-fold (OPTIMISTIC, single recording "
          "per class);\n           [chrono] = train first 70% / test last 30% per "
          "recording (FAIR).\n")

    report = {}
    for name, fn in MODELS.items():
        kf = eval_kfold(X, y, fs, fn)
        ch = eval_chrono(X, y, tpos, fs, fn)
        report[name] = {"kfold": kf, "chrono": ch}
        print(f"{name:<48} kfold acc={kf['acc']:.3f} (f1={kf['macro_f1']:.3f}) | "
              f"chrono acc={ch['acc']:.3f} (f1={ch['macro_f1']:.3f})")

    out = ROOT / "results" / "metrics" / f"muse_probe_{args.subject}.json"
    out.write_text(json.dumps({"subject": args.subject, "conditions": CONDITIONS,
                               "chance": 1 / len(CONDITIONS), "n_windows": len(y),
                               "per_class": counts, "results": report}, indent=2),
                   encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
