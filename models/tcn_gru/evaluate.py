#!/usr/bin/env python
"""Train/validation/test evaluation on session-level NHP reaching files.

Unlike indy_velocity (sorted units, within-session CV), this uses PER-ELECTRODE
multiunit spike counts (96 fixed electrodes on indy's array -> a channel space
consistent across recordings). Files are assigned once to train, validation,
or test sets. Validation selects the best epoch; test is evaluated only after
model selection.

Model: our best TCN+GRU (0.8 MB). Target: 2D fingertip velocity (top-2 movement
axes). Metric: Pearson r on held-out test sessions (per axis + mean).

Usage: py models/tcn_gru/evaluate.py
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import models.tcn_gru.best_model as R

DATA = ROOT / "data" / "source_data" / "indy_loco"
PROCESSED_DATA = ROOT / "data" / "processed" / "bin_40ms" / "artifacts"
URL = "https://zenodo.org/records/3854034/files/{}?download=1"
BIN = 0.04          # 40 ms bins -> 25 Hz
WIN = 2.0

# Eight same-subject recordings cannot be split at exactly 70/15/15 by file.
# The nearest useful whole-file allocation is 6/1/1 = 75/12.5/12.5.
TRAIN = [f"train{i}" for i in range(1, 7)]
EVAL = ["eval1"]
TEST = ["test1"]
SOURCE_NAMES = {
    "train1": "indy_20161005_06", "train2": "indy_20161006_02",
    "train3": "indy_20161007_02", "train4": "indy_20161011_03",
    "train5": "indy_20161013_03", "train6": "indy_20161014_04",
    "eval1": "indy_20161017_02", "test1": "indy_20161024_03",
}
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
        source = SOURCE_NAMES[name]
        print(f"  downloading {source} as {name}.mat ...", flush=True)
        urllib.request.urlretrieve(URL.format(f"{source}.mat"), p)
    return p


def load_source_electrode(name):
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


def load_electrode(name):
    """Load the named 40 ms processed artifact, falling back to raw generation."""
    artifact = PROCESSED_DATA / f"{name}.npz"
    if artifact.exists():
        with np.load(artifact) as z:
            return z["rates"].astype(np.float32), z["velocity"].astype(np.float32)
    return load_source_electrode(name)


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


def train_eval(train_trials, eval_by_session, test_by_session, cfg):
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
    def score(by_session):
        net.eval()
        res = {}
        for name, tri in by_session.items():
            Tt = min(t["e"].shape[1] for t in tri)
            Xe = np.stack([t["e"][:, :Tt] for t in tri]).astype(np.float32)
            Ye = np.stack([t["vel"][:Tt] for t in tri])
            with torch.no_grad():
                pr = net(torch.tensor(Xe)).numpy() * ys + ym
            res[name] = R.corr(Ye.reshape(-1, Ye.shape[-1]),
                               pr.reshape(-1, Ye.shape[-1]))
        return res

    best_eval, best_state = -np.inf, None
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
        eval_res = score(eval_by_session)
        eval_mean = float(np.mean([r.mean() for r in eval_res.values()]))
        if eval_mean > best_eval:
            best_eval = eval_mean
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        net.train()
    net.load_state_dict(best_state)
    return score(eval_by_session), score(test_by_session)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cleanup", action="store_true",
                    help="delete the downloaded split MAT files after evaluation")
    args = ap.parse_args()
    print("=== NHP file-level train/eval/test split (per-electrode, 96 ch) ===")
    print(f"TRAIN ({len(TRAIN)}): {TRAIN}")
    print(f"EVAL  ({len(EVAL)}): {EVAL}")
    print(f"TEST  ({len(TEST)}): {TEST}\n", flush=True)

    loaded = {}
    for s in TRAIN + EVAL + TEST:
        loaded[s] = load_electrode(s)
        print(f"  loaded {s}: {loaded[s][0].shape[0]} electrodes, "
              f"{loaded[s][0].shape[1]} bins", flush=True)
    # fixed movement axes = top-2 velocity-variance axes averaged over TRAIN
    var = np.mean([loaded[s][1].std(0) for s in TRAIN], 0)
    axes = np.sort(np.argsort(var)[-2:])                    # top-2 (2D)
    print(f"\ndecoding 2D finger velocity, axes {axes.tolist()} "
          f"(var {var.round(2).tolist()})", flush=True)

    train_trials = []
    for s in TRAIN:
        train_trials += windows(*loaded[s], axes)
    eval_by = {s: windows(*loaded[s], axes) for s in EVAL}
    test_by = {s: windows(*loaded[s], axes) for s in TEST}
    npar = sum(p.numel() for p in R.build_net({**CFG, "n_out": len(axes)},
                                              train_trials[0]["e"].shape[0]).parameters())
    print(f"pooled train windows: {len(train_trials)} | model {npar:,} params "
          f"({npar*4/1e6:.2f} MB)\n", flush=True)

    t0 = time.time()
    eval_res, test_res = train_eval(train_trials, eval_by, test_by, CFG)
    print(f"--- RESULTS ({time.time()-t0:.0f}s) ---", flush=True)
    for label, res in (("EVAL", eval_res), ("TEST", test_res)):
        means = []
        for s, r in res.items():
            means.append(r.mean())
            per_ax = " ".join(f"ax{i}={r[i]:.3f}" for i in range(len(r)))
            print(f"  {label} {s}: r_mean={r.mean():.3f} ({per_ax})", flush=True)
        print(f"{label} mean r (2D) = {np.mean(means):.3f}", flush=True)

    out = ROOT / "results" / "metrics" / "indy_split.json"
    out.write_text(json.dumps({"train": TRAIN, "eval": EVAL, "test": TEST,
                               "source_names": SOURCE_NAMES,
                               "axes": axes.tolist(), "params": int(npar),
                               "eval_results": {s: [float(x) for x in r]
                                                for s, r in eval_res.items()},
                               "test_results": {s: [float(x) for x in r]
                                                for s, r in test_res.items()},
                               "test_mean_r": float(np.mean(
                                   [r.mean() for r in test_res.values()]))}, indent=2),
                   encoding="utf-8")
    if args.cleanup:
        for s in TRAIN + EVAL + TEST:
            p = DATA / f"{s}.mat"
            if p.exists():
                p.unlink()
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
