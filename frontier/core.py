#!/usr/bin/env python
"""Shared model core for the active frontier (monkey intracortical decoding).

The architecture and training/eval primitives every frontier decoder imports:
  build_net(cfg, n_ch) -- dilated causal TCN + (bi)GRU + per-timestep linear head
  run_nn / run_linear  -- series-grouped CV train/eval (used by the sweep tools)
  corr                 -- per-column Pearson r
  BASE                 -- default hyperparameters

Extracted from the original WAY-EEG-GAL velocity script
(legacy/tools/way_gal_kin_research.py), which developed this architecture; the
monkey work is now its primary user, so the shared code lives here. Depends only
on numpy (+ torch, imported lazily inside the functions).
"""
from __future__ import annotations

import numpy as np


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

    _ACTS = {"gelu": nn.GELU, "relu": nn.ReLU, "elu": nn.ELU, "silu": nn.SiLU,
             "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh, "mish": nn.Mish,
             "selu": nn.SELU}
    Act = _ACTS.get(cfg.get("act", "gelu"), nn.GELU)   # conv/TCN activation

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
            sp = [nn.Conv1d(n_ch, F, 1), nn.BatchNorm1d(F)]
            if cfg.get("sp_act", True):        # activation after spatial-mix BN
                sp.append(Act())
            self.sp = nn.Sequential(*sp)
            self.convs = nn.ModuleList(
                [nn.Conv1d(F, F, 3, padding=(3 - 1) * d, dilation=d)
                 for d in cfg["dils"]])
            self.pads = [(3 - 1) * d for d in cfg["dils"]]
            self.act = Act(); self.drop = nn.Dropout(cfg["dropout"])
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
