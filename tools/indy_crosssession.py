#!/usr/bin/env python
"""Cross-session held-out test on the NHP reaching dataset.

Unlike indy_velocity (sorted units, within-session CV), this uses PER-ELECTRODE
multiunit spike counts (96 fixed electrodes on indy's array -> a channel space
consistent across sessions). That lets us POOL several sessions to TRAIN one
model and evaluate on ENTIRELY HELD-OUT SESSIONS the model never trained on --
a true generalisation test (the standard, hard cross-session BCI setting).

Model: our best TCN+GRU (0.8 MB). Target: 2D fingertip velocity (top-2 movement
axes). Metric: Pearson r on held-out test sessions (per axis + mean).

Usage: py tools/indy_crosssession.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import h5py
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.way_gal_kin_research as R

DATA = ROOT / "data" / "indy_loco"
URL = "https://zenodo.org/records/3854034/files/{}?download=1"
BIN = 0.04          # 40 ms bins -> 25 Hz
WIN = 2.0

TRAIN = ["indy_20161005_06", "indy_20161006_02", "indy_20161007_02",
         "indy_20161011_03", "indy_20161013_03", "indy_20161014_04"]
# TEST: held-out INDY sessions (same subject, never in training).
TEST = ["indy_20161017_02", "indy_20161024_03"]
NCH = 96                                            # match indy M1 array size
VEL_LP = 3.0                                        # velocity target low-pass (LOG-030)
RATE_SIGMA = 1.0                                    # firing-rate smoothing (LOG-032)

CFG = {**R.BASE, "dils": [1, 2, 4, 8, 16], "H": 64, "L": 2, "F": 64,
       "epochs": 60, "noise": 0.1, "chdrop": 0.1, "cosine": True,
       "act": "relu"}                               # ReLU = monkey default (LOG-037/038)
# NOTE: sp_act=False (drop spatial-mix GELU) looked +0.007 within-session but was
# a wash on the held-out test (0.868 vs 0.870), so kept the GELU. See LOG-035/036.


def fetch(name):
    p = DATA / f"{name}.mat"
    if not p.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {name} ...", flush=True)
        urllib.request.urlretrieve(URL.format(f"{name}.mat"), p)
    return p


def load_electrode(name):
    """Per-electrode multiunit rates (96 x bins) + fingertip velocity (bins x 3)."""
    f = h5py.File(fetch(name), "r")
    t = np.array(f["t"]).squeeze()
    fp = np.array(f["finger_pos"])                       # (3, N)
    sp = f["spikes"]                                      # (units, chan)
    edges = np.arange(t[0], t[-1], BIN)
    centers = edges[:-1] + BIN / 2
    n_chan = sp.shape[1]
    rates = np.zeros((n_chan, len(edges) - 1), dtype=np.float32)
    for ch in range(n_chan):
        allst = []
        for u in range(sp.shape[0]):                     # sum all units on electrode
            st = np.array(f[sp[u, ch]]).squeeze()
            if st.ndim and st.size:
                allst.append(np.atleast_1d(st))
        if allst:
            rates[ch] = np.histogram(np.concatenate(allst), bins=edges)[0]
    if rates.shape[0] > NCH:            # loco has M1+S1 (192): keep first NCH (M1)
        rates = rates[:NCH]
    if RATE_SIGMA:                      # input firing-rate smoothing (raises SNR)
        rates = gaussian_filter1d(rates, RATE_SIGMA, axis=1).astype(np.float32)
    pos_b = np.stack([np.interp(centers, t, fp[a]) for a in range(fp.shape[0])], 1)
    if VEL_LP:                          # low-pass position before differentiating
        sos = butter(4, VEL_LP / (0.5 / BIN), btype="low", output="sos")
        pos_b = sosfiltfilt(sos, pos_b, axis=0)
    vel = np.gradient(pos_b, BIN, axis=0)                 # (bins, 3)
    return rates, vel.astype(np.float32)


def windows(rates, vel, axes):
    # per-session channel z-score (comparable scale across sessions)
    mu, sd = rates.mean(1, keepdims=True), rates.std(1, keepdims=True) + 1e-6
    r = ((rates - mu) / sd).astype(np.float32)
    w = int(round(WIN / BIN))
    out = []
    for k in range(r.shape[1] // w):
        sl = slice(k * w, (k + 1) * w)
        out.append({"e": r[:, sl], "vel": vel[sl][:, axes]})
    return out


def train_eval(train_trials, test_by_session, cfg):
    import torch
    import torch.nn as nn
    torch.set_num_threads(4); torch.manual_seed(42); np.random.seed(42)
    n_ch = train_trials[0]["e"].shape[0]
    T = min(t["e"].shape[1] for t in train_trials)
    Xtr = np.stack([t["e"][:, :T] for t in train_trials])
    Ytr = np.stack([t["vel"][:T] for t in train_trials])
    ym, ys = Ytr.mean((0, 1)), Ytr.std((0, 1)) + 1e-6
    Ytn = ((Ytr - ym) / ys).astype(np.float32)
    cfg = {**cfg, "n_out": Ytr.shape[-1]}
    net = R.build_net(cfg, n_ch)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    mse = nn.MSELoss()
    Xt, Yt = torch.tensor(Xtr), torch.tensor(Ytn)
    idx = np.arange(len(Xt)); noise = cfg["noise"]; chd = cfg["chdrop"]
    for ep in range(cfg["epochs"]):
        net.train(); np.random.shuffle(idx)
        for b in range(0, len(idx), cfg["bs"]):
            xb = Xt[idx[b:b + cfg["bs"]]]
            if noise > 0:
                xb = xb + noise * torch.randn_like(xb)
            if chd > 0:
                m = (torch.rand(xb.shape[0], xb.shape[1], 1) > chd).float()
                xb = xb * m / (1 - chd)
            opt.zero_grad(); mse(net(xb), Yt[idx[b:b + cfg["bs"]]]).backward(); opt.step()
        sched.step()
    net.eval()
    res = {}
    for name, tri in test_by_session.items():
        Tt = min(t["e"].shape[1] for t in tri)
        Xe = np.stack([t["e"][:, :Tt] for t in tri]).astype(np.float32)
        Ye = np.stack([t["vel"][:Tt] for t in tri])
        with torch.no_grad():
            pr = net(torch.tensor(Xe)).numpy() * ys + ym
        res[name] = R.corr(Ye.reshape(-1, Ye.shape[-1]), pr.reshape(-1, Ye.shape[-1]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    print("=== NHP cross-session held-out test (per-electrode, 96 ch) ===")
    print(f"TRAIN sessions ({len(TRAIN)}): {[s[5:] for s in TRAIN]}")
    print(f"TEST  sessions ({len(TEST)}, held out): {[s[5:] for s in TEST]}\n", flush=True)

    loaded = {}
    for s in TRAIN + TEST:
        loaded[s] = load_electrode(s)
        print(f"  loaded {s[5:]}: {loaded[s][0].shape[0]} electrodes, "
              f"{loaded[s][0].shape[1]} bins", flush=True)
    # fixed movement axes = top-2 velocity-variance axes averaged over TRAIN
    var = np.mean([loaded[s][1].std(0) for s in TRAIN], 0)
    axes = np.sort(np.argsort(var)[-2:])                    # top-2 (2D)
    print(f"\ndecoding 2D finger velocity, axes {axes.tolist()} "
          f"(var {var.round(2).tolist()})", flush=True)

    train_trials = []
    for s in TRAIN:
        train_trials += windows(*loaded[s], axes)
    test_by = {s: windows(*loaded[s], axes) for s in TEST}
    npar = sum(p.numel() for p in R.build_net({**CFG, "n_out": len(axes)},
                                              train_trials[0]["e"].shape[0]).parameters())
    print(f"pooled train windows: {len(train_trials)} | model {npar:,} params "
          f"({npar*4/1e6:.2f} MB)\n", flush=True)

    t0 = time.time()
    res = train_eval(train_trials, test_by, CFG)
    print(f"--- HELD-OUT TEST results ({time.time()-t0:.0f}s) ---", flush=True)
    means = []
    for s, r in res.items():
        means.append(r.mean())
        per_ax = " ".join(f"ax{i}={r[i]:.3f}" for i in range(len(r)))
        print(f"  {s[5:]}: r_mean={r.mean():.3f} ({per_ax})", flush=True)
    print(f"\nHELD-OUT-SESSION mean r (3D) = {np.mean(means):.3f}", flush=True)

    out = ROOT / "results" / "metrics" / "indy_crosssession.json"
    out.write_text(json.dumps({"train": TRAIN, "test": TEST,
                               "axes": axes.tolist(), "params": int(npar),
                               "held_out": {s: [float(x) for x in r]
                                            for s, r in res.items()},
                               "mean_r": float(np.mean(means))}, indent=2),
                   encoding="utf-8")
    if not args.keep:
        for s in TRAIN + TEST:
            p = DATA / f"{s}.mat"
            if p.exists():
                p.unlink()
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
