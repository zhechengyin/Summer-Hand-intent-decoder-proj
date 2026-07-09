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
# central sensorimotor channels (drop frontal EOG Fp/F7/F8, temporal EMG T7/T8/TP)
MOTOR = {"F3", "Fz", "F4", "FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4",
         "CP5", "CP1", "CP2", "CP6", "P3", "Pz", "P4"}
CH_IDX = None          # if set (list), restrict EEG to these channel indices


def motor_idx():
    f = sorted(glob.glob(str(DATA / "**" / "WS_P1_S*.mat"), recursive=True))[0]
    names = list(loadmat(f, struct_as_record=False, squeeze_me=True)["ws"].names.eeg)
    return [i for i, n in enumerate(names) if n in MOTOR]


def load(subj, lp, decim):
    key = (subj, lp, decim, tuple(CH_IDX) if CH_IDX is not None else None)
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
            if CH_IDX is not None:
                e = e[CH_IDX]
            p = sosfiltfilt(sos, pos, axis=0)[::decim]
            vel = np.gradient(p, decim / FS, axis=0)
            t = min(e.shape[1], vel.shape[0])
            if t < hi:                                   # need the full window
                continue
            trials.append({"e": e[:, lo:hi].astype(np.float32),
                           "vel": vel[lo:hi].astype(np.float32), "series": series})
    _CACHE[key] = trials
    return trials


def load_mt(subj, lp, decim, markers=(2, 3, 4)):
    """Multi-task loader: target = velocity of several markers stacked
    (t, 3*len(markers)). Marker 4 (the eval target) is placed LAST."""
    key = (subj, lp, decim, "mt", markers, tuple(CH_IDX) if CH_IDX else None)
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
            cols = []
            for m in markers:
                cols += [PX + m - 1, PY + m - 1, PZ + m - 1]
            pos = kin[:, cols]
            if np.isnan(pos).any():
                continue
            e = sosfiltfilt(sos, eeg, axis=1)[:, ::decim]
            if CH_IDX is not None:
                e = e[CH_IDX]
            p = sosfiltfilt(sos, pos, axis=0)[::decim]
            vel = np.gradient(p, decim / FS, axis=0)          # (t, 3*n_markers)
            t = min(e.shape[1], vel.shape[0])
            if t < hi:
                continue
            trials.append({"e": e[:, lo:hi].astype(np.float32),
                           "vel": vel[lo:hi].astype(np.float32), "series": series})
    _CACHE[key] = trials
    return trials


def _band_feature(eeg, lo, hi, mode, decim):
    """One band -> (32, t) at the decimated rate. mode 'raw' = filtered signal
    (slow movement potential); 'env' = band-pass -> Hilbert amplitude envelope
    (band-power time series, e.g. mu/beta ERD-ERS), smoothed then decimated."""
    from scipy.signal import hilbert
    nyq = FS / 2
    if mode == "raw":
        if lo is None:
            sos = butter(4, hi / nyq, btype="low", output="sos")
        else:
            sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
        return sosfiltfilt(sos, eeg, axis=1)[:, ::decim].astype(np.float32)
    sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, eeg, axis=1), axis=1))
    slow = butter(4, 5.0 / nyq, btype="low", output="sos")     # smooth envelope
    return sosfiltfilt(slow, env, axis=1)[:, ::decim].astype(np.float32)


# band presets: list of (lo, hi, mode). 'raw' low band + 'env' rhythm bands.
BANDSETS = {
    "lp2": [(None, 2.0, "raw")],                                  # baseline
    "lp2+mu": [(None, 2.0, "raw"), (8.0, 12.0, "env")],
    "lp2+mu(8-10)": [(None, 2.0, "raw"), (8.0, 10.0, "env")],
    "lp2+mu+beta": [(None, 2.0, "raw"), (8.0, 12.0, "env"), (13.0, 30.0, "env")],
    "lp2+mu+beta+lowgamma": [(None, 2.0, "raw"), (8.0, 12.0, "env"),
                             (13.0, 30.0, "env"), (30.0, 45.0, "env")],
    "lp4+mu+beta": [(None, 4.0, "raw"), (8.0, 12.0, "env"), (13.0, 30.0, "env")],
}


def load_mb(subj, bands, decim):
    """Multi-band loader: stack per-band 32-ch features -> (32*nbands, t)."""
    key = (subj, "mb", tuple(bands), decim, tuple(CH_IDX) if CH_IDX else None)
    if key in _CACHE:
        return _CACHE[key]
    fsd = FS / decim
    lo_i, hi_i = int(CROP[0] * fsd), int(CROP[1] * fsd)
    lp = _sos(2.0)                                     # for smoothing the target
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
            e = np.vstack([_band_feature(eeg, lo, hi, m, decim)
                           for (lo, hi, m) in bands])
            if CH_IDX is not None:                    # (not used for multi-band)
                pass
            p = sosfiltfilt(lp, pos, axis=0)[::decim]
            vel = np.gradient(p, decim / FS, axis=0)
            t = min(e.shape[1], vel.shape[0])
            if t < hi_i:
                continue
            trials.append({"e": e[:, lo_i:hi_i].astype(np.float32),
                           "vel": vel[lo_i:hi_i].astype(np.float32),
                           "series": series})
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
    import torch
    import torch.nn as nn

    class BandGate(nn.Module):
        """Learned gate over frequency bands. 'static' = one weight per band
        (which bands help overall). 'dynamic' = per-band, per-timestep gate from
        a small conv over the input (which bands help WHEN). Stores last_gate for
        inspection (reveals the learned 'law': when each band is up/down)."""
        def __init__(self, n_ch, nbands, mode):
            super().__init__()
            self.nb, self.cpb, self.mode = nbands, n_ch // nbands, mode
            if mode == "static":
                self.g = nn.Parameter(torch.zeros(nbands))       # sigmoid(0)=.5
            else:
                self.net = nn.Conv1d(n_ch, nbands, 5, padding=2)
            self.last_gate = None

        def forward(self, x):
            B, C, T = x.shape
            xb = x.view(B, self.nb, self.cpb, T)
            if self.mode == "static":
                g = torch.sigmoid(self.g).view(1, self.nb, 1, 1)
            else:
                g = torch.sigmoid(self.net(x)).view(B, self.nb, 1, T)
            self.last_gate = g.detach()
            return (xb * g).reshape(B, C, T)

    class Seq(nn.Module):
        def __init__(self):
            super().__init__()
            F = cfg["F"]
            bg = cfg.get("band_gate")
            self.gate = BandGate(n_ch, cfg["nbands"], bg) if bg else None
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
            self.head = nn.Linear(cfg["H"] * (2 if cfg["bidir"] else 1),
                                   cfg.get("n_out", 3))

        def forward(self, x):
            if self.gate is not None:
                x = self.gate(x)
            z = self.sp(x)
            for c, p in zip(self.convs, self.pads):
                z = self.act(c(z)[:, :, :-p] + z)
            z, _ = self.gru(self.drop(z).transpose(1, 2))
            return self.head(z)

    return Seq()


def run_nn(trials, cfg, ret_preds=False, ret_gate=False):
    import torch
    import torch.nn as nn
    torch.set_num_threads(4); torch.manual_seed(42); np.random.seed(42)
    n_ch = trials[0]["e"].shape[0]
    n_out = trials[0]["vel"].shape[-1]
    cfg = {**cfg, "n_out": n_out}
    T = min(t["e"].shape[1] for t in trials)
    esl = cfg.get("eval_slice")               # (start,end) columns to score, or None
    rs, preds, gate_acc = [], {}, []
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
        sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
                 if cfg.get("cosine") else None)
        mse = nn.MSELoss()
        Xt, Yt = torch.tensor(Xtr), torch.tensor(Ytn)
        noise = float(cfg.get("noise", 0.0)); chdrop = float(cfg.get("chdrop", 0.0))
        idx = np.arange(len(Xt))
        for ep in range(cfg["epochs"]):
            net.train(); np.random.shuffle(idx)
            for b in range(0, len(idx), cfg["bs"]):
                bi = idx[b:b + cfg["bs"]]
                xb = Xt[bi]
                if noise > 0:                            # additive Gaussian aug
                    xb = xb + noise * torch.randn_like(xb)
                if chdrop > 0:                           # per-sample channel dropout
                    m = (torch.rand(xb.shape[0], xb.shape[1], 1) > chdrop).float()
                    xb = xb * m / (1 - chdrop)
                opt.zero_grad(); mse(net(xb), Yt[bi]).backward(); opt.step()
            if sched:
                sched.step()
        net.eval()
        with torch.no_grad():
            pr = net(torch.tensor(Xte)).numpy() * ys + ym
            if ret_gate and net.gate is not None:      # (nb,1) static / (nb,T) dyn
                gv = net.gate.last_gate.numpy()        # (B,nb,1,1) or (B,nb,1,T)
                gate_acc.append(gv.mean(0).reshape(gv.shape[1], -1))
        yt = Yte.reshape(-1, n_out); yp = pr.reshape(-1, n_out)
        if esl:                                   # score only these columns
            yt, yp = yt[:, esl[0]:esl[1]], yp[:, esl[0]:esl[1]]
        rs.append(corr(yt, yp))
        for t, p in zip(te, pr):
            preds[id(t)] = p
    r = np.mean(rs, axis=0)
    if ret_gate:
        gate = np.mean(gate_acc, axis=0) if gate_acc else None   # (nb,) or (nb,T)
        return r, gate
    return (r, preds) if ret_preds else r


def run_linear(trials, nlag=12, kfold=3, ret_preds=False):
    from numpy.linalg import solve
    n_ch = trials[0]["e"].shape[0]
    n_out = trials[0]["vel"].shape[-1]
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
        XtX = np.zeros((d, d)); XtY = np.zeros((d, n_out))
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
                             "final", "final_motor", "improve",
                             "final_improved", "mt", "final_mt",
                             "mband", "final_mband", "gate"])
    ap.add_argument("--bandset", default="lp2+mu+beta")
    args = ap.parse_args()
    subj = args.subject

    def show(tag, r, t0=None):
        extra = f"  [{time.time()-t0:.0f}s]" if t0 else ""
        print(f"{tag:38s} r_mean={r.mean():.3f} (x={r[0]:.3f} y={r[1]:.3f} "
              f"z={r[2]:.3f}){extra}", flush=True)

    BIG = {**BASE, "dils": [1, 2, 4, 8, 16], "H": 64, "L": 2, "F": 64,
           "epochs": 100}
    # improved recipe: longer context + augmentation + cosine schedule (cheap on
    # params -- only +1 conv block). Keeps model < 1 MB.
    BIGP = {**BIG, "dils": [1, 2, 4, 8, 16, 32], "noise": 0.1, "chdrop": 0.1,
            "cosine": True, "epochs": 150}

    print(f"=== velocity research | {subj} marker {MARK} | 3-fold ===\n")
    if args.stage == "improve":
        for name, cfg in [("BIG ref", BIG), ("BIG+aug+cos+dil32", BIGP)]:
            import torch
            net = build_net(cfg, load(subj, 2.0, DEC)[0]["e"].shape[0])
            npar = sum(p.numel() for p in net.parameters())
            t0 = time.time()
            show(f"{name} ({npar/1e3:.0f}k par)", run_nn(load(subj, 2.0, DEC), cfg), t0)
    elif args.stage == "final_improved":
        rs = []
        for s in ("P1", "P2", "P3"):
            t0 = time.time()
            r = run_nn(load(s, 2.0, DEC), BIGP)
            rs.append(r.mean()); show(f"BIGP {s}", r, t0)
        print(f"\nBIGP 3-subject MEAN r = {np.mean(rs):.3f}", flush=True)
    elif args.stage == "mband":
        # multi-band input sweep on one subject (filter-bank: raw low + rhythm
        # envelopes). BIGP arch, fewer epochs for speed.
        cfg = {**BIGP, "epochs": 60}
        for name, bands in BANDSETS.items():
            tr = load_mb(subj, bands, DEC)
            nch = tr[0]["e"].shape[0]
            t0 = time.time()
            show(f"{name} ({nch}ch)", run_nn(tr, cfg), t0)
    elif args.stage == "final_mband":
        bands = BANDSETS[args.bandset]
        pth = (ROOT / "results" / "metrics" /
               f"mband_{args.bandset.replace('+', '_').replace('(', '').replace(')', '').replace('-', '')}.json")
        res = json.loads(pth.read_text()) if pth.exists() else {}
        for s in ("P1", "P2", "P3"):
            if s in res:                                  # resume: skip done subj
                show(f"mband[{args.bandset}] {s} (cached)", np.array(res[s]))
                continue
            t0 = time.time()
            r = run_nn(load_mb(s, bands, DEC), BIGP)
            res[s] = [float(x) for x in r]                # checkpoint per subject
            pth.parent.mkdir(parents=True, exist_ok=True)
            pth.write_text(json.dumps(res, indent=2))
            show(f"mband[{args.bandset}] {s}", r, t0)
        means = [np.mean(v) for v in res.values()]
        print(f"\nmband[{args.bandset}] 3-subject MEAN r = {np.mean(means):.3f}",
              flush=True)
    elif args.stage == "gate":
        # learned band-gating: does adaptively weighting bands beat concat, and
        # what pattern does it learn (which band, when)?
        bands = BANDSETS["lp2+mu+beta"]
        bn = ["delta<2", "mu8-12", "beta13-30"]
        tr = load_mb(subj, bands, DEC)
        base = {**BIGP, "epochs": 60}
        show("concat (no gate)", run_nn(tr, base))
        for mode in ("static", "dynamic"):
            cfg = {**base, "band_gate": mode, "nbands": len(bands)}
            t0 = time.time()
            r, gate = run_nn(tr, cfg, ret_gate=True)
            show(f"band-gate [{mode}]", r, t0)
            if mode == "static":
                print("  learned band weights: " +
                      ", ".join(f"{bn[i]}={float(gate[i, 0]):.3f}"
                                for i in range(len(bands))), flush=True)
            else:
                fsd = FS / DEC
                onset = int((2.0 - CROP[0]) * fsd)     # LED onset bin (~2.0 s)
                for i in range(len(bands)):
                    p = gate[i]
                    print(f"  {bn[i]}: mean={p.mean():.3f} "
                          f"pre-onset={p[:onset].mean():.3f} "
                          f"post-onset={p[onset:].mean():.3f}", flush=True)
                (ROOT / "results" / "metrics" / f"band_gate_profile_{subj}.json"
                 ).write_text(json.dumps({bn[i]: gate[i].tolist()
                                          for i in range(len(bands))}, indent=2))
    elif args.stage in ("mt", "final_mt"):
        # multi-task: predict markers (2,3,4); score marker 4 (last 3 cols)
        mt = {**BIGP, "eval_slice": (6, 9)}
        subs = ("P1", "P2", "P3") if args.stage == "final_mt" else (subj,)
        rs = []
        for s in subs:
            t0 = time.time()
            r = run_nn(load_mt(s, 2.0, DEC, markers=(2, 3, 4)), mt)
            rs.append(r.mean()); show(f"BIGP-multitask {s}", r, t0)
        if len(subs) > 1:
            print(f"\nBIGP-multitask 3-subject MEAN r = {np.mean(rs):.3f}",
                  flush=True)
    if args.stage in ("final", "final_motor"):
        global CH_IDX
        motor = args.stage == "final_motor"
        if motor:
            CH_IDX = motor_idx()
            print(f"MOTOR-ONLY: {len(CH_IDX)} central sensorimotor channels\n")
        rs = []
        for s in ("P1", "P2", "P3"):
            tr = load(s, 2.0, DEC)
            t0 = time.time()
            r = run_nn(tr, BIG)
            rs.append(r.mean())
            show(f"BIG{'-motor' if motor else ''} {s}", r, t0)
        print(f"\nBIG{'-motor' if motor else ''} 3-subject MEAN r = "
              f"{np.mean(rs):.3f}", flush=True)
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
