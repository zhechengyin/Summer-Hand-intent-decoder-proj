#!/usr/bin/env python
"""Research harness to push EEG->velocity correlation on WAY-EEG-GAL.

Configurable seq2seq TCN+GRU + lagged-linear reference + ensemble, on the best
hand/finger marker (sensor 4). Consistent 3-fold-over-series evaluation for every
config so numbers are comparable. Trials cached per (lp, decim) so band sweeps
are cheap.

Usage: py tools/way_gal_kin_research.py --subject P1 --stage band
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "way_eeg_gal"
FS = 500.0
MARK = 4
PX, PY, PZ = 18, 22, 26
_CACHE = {}


def _sos(lp):
    return butter(4, lp / (FS / 2), btype="low", output="sos")


CROP = (1.5, 7.0)      # movement window (s); LED onset ~2.0 s. Cuts GRU length.


def load(subj, lp, decim):
    key = (subj, lp, decim)
    if key in _CACHE:
        return _CACHE[key]
    sos = _sos(lp)
    fsd = FS / decim
    lo, hi = int(CROP[0] * fsd), int(CROP[1] * fsd)
    trials = []
    for series, f in enumerate(sorted(glob.glob(
            str(DATA / "**" / f"WS_{subj}_S*.mat"), recursive=True)), 1):
        ws = loadmat(f, struct_as_record=False, squeeze_me=True)["ws"]
        for w in np.atleast_1d(ws.win):
            eeg = np.asarray(w.eeg, dtype=np.float64).T
            kin = np.asarray(w.kin, dtype=np.float64)
            pos = kin[:, [PX + MARK - 1, PY + MARK - 1, PZ + MARK - 1]]
            if np.isnan(pos).any():
                continue
            e = sosfiltfilt(sos, eeg, axis=1)[:, ::decim]
            p = sosfiltfilt(sos, pos, axis=0)[::decim]
            vel = np.gradient(p, decim / FS, axis=0)
            t = min(e.shape[1], vel.shape[0])
            if t < hi:                                   # need the full window
                continue
            trials.append({"e": e[:, lo:hi].astype(np.float32),
                           "vel": vel[lo:hi].astype(np.float32), "series": series})
    _CACHE[key] = trials
    return trials


def corr(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    d = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    return (a * b).sum(0) / np.where(d == 0, 1e-9, d)


def series_groups(trials, k=3):
    s = sorted({t["series"] for t in trials})
    return [s[i::k] for i in range(k)]


def build_net(cfg, n_ch):
    import torch.nn as nn

    class Seq(nn.Module):
        def __init__(self):
            super().__init__()
            F = cfg["F"]
            self.sp = nn.Sequential(nn.Conv1d(n_ch, F, 1), nn.BatchNorm1d(F),
                                    nn.GELU())
            self.convs = nn.ModuleList(
                [nn.Conv1d(F, F, 3, padding=(3 - 1) * d, dilation=d)
                 for d in cfg["dils"]])
            self.pads = [(3 - 1) * d for d in cfg["dils"]]
            self.act = nn.GELU(); self.drop = nn.Dropout(cfg["dropout"])
            self.gru = nn.GRU(F, cfg["H"], cfg["L"], batch_first=True,
                              bidirectional=cfg["bidir"],
                              dropout=cfg["dropout"] if cfg["L"] > 1 else 0.0)
            self.head = nn.Linear(cfg["H"] * (2 if cfg["bidir"] else 1), 3)

        def forward(self, x):
            z = self.sp(x)
            for c, p in zip(self.convs, self.pads):
                z = self.act(c(z)[:, :, :-p] + z)
            z, _ = self.gru(self.drop(z).transpose(1, 2))
            return self.head(z)

    return Seq()


def run_nn(trials, cfg, ret_preds=False):
    import torch
    import torch.nn as nn
    torch.set_num_threads(4); torch.manual_seed(42); np.random.seed(42)
    n_ch = trials[0]["e"].shape[0]
    T = min(t["e"].shape[1] for t in trials)
    rs, preds = [], {}
    for g in series_groups(trials, cfg.get("kfold", 3)):
        tr = [t for t in trials if t["series"] not in g]
        te = [t for t in trials if t["series"] in g]
        Xtr = np.stack([t["e"][:, :T] for t in tr]); Ytr = np.stack([t["vel"][:T] for t in tr])
        Xte = np.stack([t["e"][:, :T] for t in te]); Yte = np.stack([t["vel"][:T] for t in te])
        cm, cs = Xtr.mean((0, 2), keepdims=True), Xtr.std((0, 2), keepdims=True) + 1e-6
        ym, ys = Ytr.mean((0, 1)), Ytr.std((0, 1)) + 1e-6
        Xtr = ((Xtr - cm) / cs).astype(np.float32); Xte = ((Xte - cm) / cs).astype(np.float32)
        Ytn = ((Ytr - ym) / ys).astype(np.float32)
        net = build_net(cfg, n_ch)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
        mse = nn.MSELoss()
        Xt, Yt = torch.tensor(Xtr), torch.tensor(Ytn)
        idx = np.arange(len(Xt))
        for ep in range(cfg["epochs"]):
            net.train(); np.random.shuffle(idx)
            for b in range(0, len(idx), cfg["bs"]):
                bi = idx[b:b + cfg["bs"]]
                opt.zero_grad(); mse(net(Xt[bi]), Yt[bi]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pr = net(torch.tensor(Xte)).numpy() * ys + ym
        rs.append(corr(Yte.reshape(-1, 3), pr.reshape(-1, 3)))
        for t, p in zip(te, pr):
            preds[id(t)] = p
    r = np.mean(rs, axis=0)
    return (r, preds) if ret_preds else r


def run_linear(trials, nlag=12, kfold=3, ret_preds=False):
    from numpy.linalg import solve
    n_ch = trials[0]["e"].shape[0]
    d = n_ch * (2 * nlag + 1)

    def design(e):
        return np.concatenate([np.roll(e, k, axis=1) for k in
                               range(-nlag, nlag + 1)], axis=0).T[nlag:e.shape[1] - nlag]
    rs, preds = [], {}
    for g in series_groups(trials, kfold):
        tr = [t for t in trials if t["series"] not in g]
        te = [t for t in trials if t["series"] in g]
        allc = np.concatenate([t["e"] for t in tr], axis=1)
        mu, sd = allc.mean(1, keepdims=True), allc.std(1, keepdims=True); sd[sd == 0] = 1
        XtX = np.zeros((d, d)); XtY = np.zeros((d, 3))
        for t in tr:
            X = design((t["e"] - mu) / sd).astype(np.float64)
            XtX += X.T @ X; XtY += X.T @ t["vel"][nlag:t["e"].shape[1] - nlag]
        w = solve(XtX + 1e3 * np.eye(d), XtY)
        yp, yt = [], []
        for t in te:
            p = design((t["e"] - mu) / sd) @ w
            yp.append(p); yt.append(t["vel"][nlag:t["e"].shape[1] - nlag])
            preds[id(t)] = p
        rs.append(corr(np.vstack(yt), np.vstack(yp)))
    r = np.mean(rs, axis=0)
    return (r, preds) if ret_preds else r


BASE = dict(F=32, dils=[1, 2, 4, 8], H=32, L=1, bidir=True, dropout=0.3,
            lr=1e-3, wd=1e-3, epochs=30, bs=32, kfold=3)
DEC = 20        # 25 Hz -- fast (short GRU sequences), plenty for <4 Hz velocity


def _zscore_subject(trials):
    """Per-subject channel z-score (comparable scale before pooling)."""
    allc = np.concatenate([t["e"] for t in trials], axis=1)
    mu, sd = allc.mean(1, keepdims=True), allc.std(1, keepdims=True)
    sd[sd == 0] = 1.0
    return [{**t, "e": ((t["e"] - mu) / sd).astype(np.float32)} for t in trials]


def run_pool(subjects, cfg, lp, decim):
    """Cross-subject pooled TCN+GRU. Train on all subjects' series != h, test
    each subject's held-out series h. Returns per-subject mean-r."""
    import torch
    import torch.nn as nn
    torch.set_num_threads(4); torch.manual_seed(42); np.random.seed(42)
    per = {s: _zscore_subject(load(s, lp, decim)) for s in subjects}
    for s in per:
        for t in per[s]:
            t["subj"] = s
    n_ch = per[subjects[0]][0]["e"].shape[0]
    T = min(t["e"].shape[1] for s in per for t in per[s])
    series = sorted({t["series"] for s in per for t in per[s]})
    groups = [series[i::cfg["kfold"]] for i in range(cfg["kfold"])]
    acc = {s: [] for s in subjects}
    for g in groups:
        tr = [t for s in per for t in per[s] if t["series"] not in g]
        Xtr = np.stack([t["e"][:, :T] for t in tr])
        Ytr = np.stack([t["vel"][:T] for t in tr])
        ym, ys = Ytr.mean((0, 1)), Ytr.std((0, 1)) + 1e-6
        Ytn = ((Ytr - ym) / ys).astype(np.float32)
        net = build_net(cfg, n_ch)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
        mse = nn.MSELoss()
        Xt, Yt = torch.tensor(Xtr.astype(np.float32)), torch.tensor(Ytn)
        idx = np.arange(len(Xt))
        for ep in range(cfg["epochs"]):
            net.train(); np.random.shuffle(idx)
            for b in range(0, len(idx), cfg["bs"]):
                bi = idx[b:b + cfg["bs"]]
                opt.zero_grad(); mse(net(Xt[bi]), Yt[bi]).backward(); opt.step()
        net.eval()
        for s in subjects:
            te = [t for t in per[s] if t["series"] in g]
            if not te:
                continue
            Xte = np.stack([t["e"][:, :T] for t in te]).astype(np.float32)
            Yte = np.stack([t["vel"][:T] for t in te])
            with torch.no_grad():
                pr = net(torch.tensor(Xte)).numpy() * ys + ym
            acc[s].append(corr(Yte.reshape(-1, 3), pr.reshape(-1, 3)))
    return {s: np.mean(acc[s], axis=0) for s in subjects if acc[s]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="P1")
    ap.add_argument("--stage", default="band",
                    choices=["band", "arch", "ensemble", "pool", "quick",
                             "final"])
    args = ap.parse_args()
    subj = args.subject

    def show(tag, r, t0=None):
        extra = f"  [{time.time()-t0:.0f}s]" if t0 else ""
        print(f"{tag:38s} r_mean={r.mean():.3f} (x={r[0]:.3f} y={r[1]:.3f} "
              f"z={r[2]:.3f}){extra}", flush=True)

    BIG = {**BASE, "dils": [1, 2, 4, 8, 16], "H": 64, "L": 2, "F": 64,
           "epochs": 100}

    print(f"=== velocity research | {subj} marker {MARK} | 3-fold ===\n")
    if args.stage == "final":
        rs = []
        for s in ("P1", "P2", "P3"):
            tr = load(s, 2.0, DEC)
            t0 = time.time()
            r = run_nn(tr, BIG)
            rs.append(r.mean())
            show(f"BIG {s}", r, t0)
        print(f"\nBIG 3-subject MEAN r = {np.mean(rs):.3f}", flush=True)
    elif args.stage == "band":
        for lp in (2.0, 4.0, 8.0, 12.0):
            tr = load(subj, lp, DEC)
            t0 = time.time()
            show(f"TCN+GRU lp={lp}Hz decim=50Hz", run_nn(tr, BASE), t0)
    elif args.stage == "arch":
        tr = load(subj, 2.0, DEC)               # best band from stage 1
        show("baseline (lp2 cropped)", run_nn(tr, BASE))
        for name, upd in [("+dil16 (context)", {"dils": [1, 2, 4, 8, 16]}),
                          ("+GRU H64 L2", {"H": 64, "L": 2}),
                          ("+F64", {"F": 64}),
                          ("big (all)", {"dils": [1, 2, 4, 8, 16], "H": 64,
                                         "L": 2, "F": 64, "epochs": 100})]:
            cfg = {**BASE, **upd}
            t0 = time.time(); show(name, run_nn(tr, cfg), t0)
    elif args.stage == "quick":
        # fast within-subject check that cropping+50ep doesn't regress
        for lp in (2.0, 4.0):
            tr = load(subj, lp, DEC)
            t0 = time.time()
            show(f"P1 cropped lp={lp} 50ep", run_nn(tr, BASE), t0)
    elif args.stage == "pool":
        subs = ["P1", "P2", "P3"]
        print("cross-subject POOLED TCN+GRU (train all subjects, test held series)")
        for lp in (2.0,):
            t0 = time.time()
            res = run_pool(subs, BASE, lp, DEC)
            for s in subs:
                if s in res:
                    r = res[s]
                    print(f"  pooled lp={lp} {s}: r_mean={r.mean():.3f} "
                          f"(x={r[0]:.3f} y={r[1]:.3f} z={r[2]:.3f})", flush=True)
            mean = np.mean([res[s].mean() for s in subs if s in res])
            print(f"  pooled lp={lp} MEAN r={mean:.3f}  [{time.time()-t0:.0f}s]",
                  flush=True)
    print("\ndone")


if __name__ == "__main__":
    main()
