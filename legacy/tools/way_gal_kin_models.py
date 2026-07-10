#!/usr/bin/env python
"""Compare velocity decoders on WAY-EEG-GAL: lagged-linear vs sliding-window
Riemannian tangent vs seq2seq TCN+GRU.

Target: 3D velocity of the best hand/finger marker (sensor 4). EEG low-passed to
4 Hz, decimated to 50 Hz. Metric: Pearson r (pred vs true velocity), per axis.

- lagged-linear : +/-240 ms EEG lags -> Ridge (our baseline)          [LOSO]
- tangent       : per-window regularised covariance -> tangent space
                  -> Ridge to the window-centre velocity               [LOSO]
- tcn_gru       : seq2seq TCN + bidirectional GRU -> per-timestep 3D   [3-fold]

Usage: py tools/way_gal_kin_models.py --subject P1
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.riemannian import RiemannianTangentSpace

DATA = ROOT / "data" / "way_eeg_gal"
FS = 500.0
DECIM = 10
LP = 4.0
NLAG = 12
MARK = 4
PX, PY, PZ = 18, 22, 26


def _lpdec(x, axis):
    sos = butter(4, LP / (FS / 2), btype="low", output="sos")
    return sosfiltfilt(sos, x, axis=axis).take(np.arange(0, x.shape[axis], DECIM),
                                               axis=axis)


def load(subj):
    trials = []
    for series, f in enumerate(sorted(glob.glob(str(DATA / "**" / f"WS_{subj}_S*.mat"),
                                                 recursive=True)), 1):
        ws = loadmat(f, struct_as_record=False, squeeze_me=True)["ws"]
        for w in np.atleast_1d(ws.win):
            eeg = np.asarray(w.eeg, dtype=np.float64).T
            kin = np.asarray(w.kin, dtype=np.float64)
            pos = kin[:, [PX + MARK - 1, PY + MARK - 1, PZ + MARK - 1]]
            if np.isnan(pos).any():
                continue
            e = _lpdec(eeg, 1)
            vel = np.gradient(_lpdec(pos, 0), 1.0 / (FS / DECIM), axis=0)
            t = min(e.shape[1], vel.shape[0])
            trials.append({"e": e[:, :t].astype(np.float32),
                           "vel": vel[:t].astype(np.float32), "series": series})
    return trials


def corr(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    d = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    return (a * b).sum(0) / np.where(d == 0, 1e-9, d)


def _zc(trials, idx):
    allc = np.concatenate([trials[i]["e"] for i in idx], axis=1)
    mu, sd = allc.mean(1, keepdims=True), allc.std(1, keepdims=True)
    sd[sd == 0] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def series_folds(trials):
    s = sorted({t["series"] for t in trials})
    return [(h, [i for i, t in enumerate(trials) if t["series"] != h],
             [i for i, t in enumerate(trials) if t["series"] == h]) for h in s]


# ---- 1. lagged-linear Ridge ------------------------------------------------
def _design(e):
    feats = [np.roll(e, k, axis=1) for k in range(-NLAG, NLAG + 1)]
    return np.concatenate(feats, axis=0).T[NLAG:e.shape[1] - NLAG]


def run_linear(trials):
    from numpy.linalg import solve
    rs = []
    for h, tr, te in series_folds(trials):
        mu, sd = _zc(trials, tr)
        d = 32 * (2 * NLAG + 1)
        XtX = np.zeros((d, d)); XtY = np.zeros((d, 3))
        for i in tr:
            X = _design((trials[i]["e"] - mu) / sd).astype(np.float64)
            Y = trials[i]["vel"][NLAG:trials[i]["e"].shape[1] - NLAG]
            XtX += X.T @ X; XtY += X.T @ Y
        w = solve(XtX + 1e3 * np.eye(d), XtY)
        yp = np.vstack([_design((trials[i]["e"] - mu) / sd) @ w for i in te])
        yt = np.vstack([trials[i]["vel"][NLAG:trials[i]["e"].shape[1] - NLAG] for i in te])
        rs.append(corr(yt, yp))
    return np.mean(rs, axis=0)


# ---- 2. sliding-window Riemannian tangent -> Ridge -------------------------
def _windows(e, vel, win, stride):
    n = (e.shape[1] - win) // stride + 1
    W = np.stack([e[:, k * stride:k * stride + win] for k in range(n)])   # (n,ch,win)
    yc = np.stack([vel[k * stride + win // 2] for k in range(n)])          # (n,3)
    return W.astype(np.float32), yc


def run_tangent(trials, win=50, stride=5, reg=0.3):
    from numpy.linalg import solve
    per = [_windows(t["e"], t["vel"], win, stride) for t in trials]
    rs = []
    for h, tr, te in series_folds(trials):
        ts = RiemannianTangentSpace(sfreq=FS / DECIM, l_freq=None, h_freq=None,
                                    reg=reg)
        Wtr = np.concatenate([per[i][0] for i in tr], axis=0)
        ts.fit(Wtr)
        Ztr = ts.transform(Wtr).astype(np.float64)
        Ytr = np.concatenate([per[i][1] for i in tr], axis=0)
        mu, sd = Ztr.mean(0), Ztr.std(0); sd[sd == 0] = 1
        Ztr = (Ztr - mu) / sd
        d = Ztr.shape[1]
        w = solve(Ztr.T @ Ztr + 1e2 * np.eye(d), Ztr.T @ Ytr)
        yp, yt = [], []
        for i in te:
            Z = (ts.transform(per[i][0]).astype(np.float64) - mu) / sd
            yp.append(Z @ w); yt.append(per[i][1])
        rs.append(corr(np.vstack(yt), np.vstack(yp)))
    return np.mean(rs, axis=0)


# ---- 3. seq2seq TCN + GRU --------------------------------------------------
def run_tcn_gru(trials, epochs=80):
    import torch
    import torch.nn as nn
    torch.manual_seed(42); np.random.seed(42)
    T = min(t["e"].shape[1] for t in trials)
    F = 32

    class Seq(nn.Module):
        def __init__(self):
            super().__init__()
            self.sp = nn.Sequential(nn.Conv1d(32, F, 1), nn.BatchNorm1d(F), nn.GELU())
            blocks = []
            for dl in (1, 2, 4, 8):
                pad = (3 - 1) * dl
                blocks += [nn.Conv1d(F, F, 3, padding=pad, dilation=dl)]
            self.tcn = nn.ModuleList(blocks)
            self.pads = [(3 - 1) * dl for dl in (1, 2, 4, 8)]
            self.act = nn.GELU(); self.drop = nn.Dropout(0.3)
            self.gru = nn.GRU(F, F, batch_first=True, bidirectional=True)
            self.head = nn.Linear(2 * F, 3)

        def forward(self, x):
            z = self.sp(x)
            for c, p in zip(self.tcn, self.pads):
                z = self.act(c(z)[:, :, :-p] + z)
            z = self.drop(z).transpose(1, 2)
            z, _ = self.gru(z)
            return self.head(z)

    series = sorted({t["series"] for t in trials})
    groups = [series[0::3], series[1::3], series[2::3]]
    rs = []
    for g in groups:
        tr = [t for t in trials if t["series"] not in g]
        te = [t for t in trials if t["series"] in g]
        Xtr = np.stack([t["e"][:, :T] for t in tr]); Ytr = np.stack([t["vel"][:T] for t in tr])
        Xte = np.stack([t["e"][:, :T] for t in te]); Yte = np.stack([t["vel"][:T] for t in te])
        cm, cs = Xtr.mean((0, 2), keepdims=True), Xtr.std((0, 2), keepdims=True) + 1e-6
        ym, ys = Ytr.mean((0, 1)), Ytr.std((0, 1)) + 1e-6
        Xtr = ((Xtr - cm) / cs).astype(np.float32); Xte = ((Xte - cm) / cs).astype(np.float32)
        Ytn = ((Ytr - ym) / ys).astype(np.float32)
        net = Seq(); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
        mse = nn.MSELoss()
        Xt = torch.tensor(Xtr); Yt = torch.tensor(Ytn)
        idx = np.arange(len(Xt))
        for ep in range(epochs):
            net.train(); np.random.shuffle(idx)
            for b in range(0, len(idx), 16):
                bi = idx[b:b + 16]
                opt.zero_grad(); mse(net(Xt[bi]), Yt[bi]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(torch.tensor(Xte)).numpy() * ys + ym
        rs.append(corr(Yte.reshape(-1, 3), pred.reshape(-1, 3)))
    return np.mean(rs, axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", default="P1")
    args = ap.parse_args()
    trials = load(args.subject)
    print(f"=== EEG->velocity decoders | {args.subject} marker {MARK} | "
          f"{len(trials)} trials @ {FS/DECIM:.0f}Hz | r = Pearson (per axis) ===\n")
    out = {}
    for name, fn in [("lagged-linear Ridge  [LOSO]", run_linear),
                     ("Riemannian tangent   [LOSO]", run_tangent),
                     ("seq2seq TCN+GRU      [3-fold]", run_tcn_gru)]:
        r = fn(trials)
        out[name] = {"r_mean": float(r.mean()), "r_x": float(r[0]),
                     "r_y": float(r[1]), "r_z": float(r[2])}
        print(f"{name:32s} r_mean={r.mean():.3f} "
              f"(x={r[0]:.3f} y={r[1]:.3f} z={r[2]:.3f})", flush=True)
    p = ROOT / "results" / "metrics" / f"way_gal_kin_models_{args.subject}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
