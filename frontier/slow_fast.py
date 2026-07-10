#!/usr/bin/env python
"""Slow/fast velocity decomposition test (monkey NHP pipeline only).

raw_velocity = slow_velocity + fast_residual, where slow = velocity of 3 Hz
low-passed finger position, fast = raw - slow. Train separate TCN+GRU decoders on
slow and fast, and test whether pred_slow + alpha*pred_fast reconstructs raw
velocity better than a direct raw decoder. If the fast residual is mostly
marker/derivative noise, the fast decoder is ~0 and best alpha -> 0.

5-block contiguous CV; Pearson r per axis + mean. Sorted-unit spikes, 50 ms bins.
Usage: py tools/indy_slow_fast.py --file data/indy_loco/indy_20161005_06.mat
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import h5py
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import frontier.core as R

BIN = 0.04          # 40 ms bins -> 25 Hz
WIN = 2.0
KFOLD = 5
MIN_SPK = 10
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
CFG = {**R.BASE, "dils": [1, 2, 4, 8, 16], "H": 64, "L": 2, "F": 64,
       "epochs": 60, "noise": 0.1, "chdrop": 0.1, "cosine": True,
       "act": "relu"}                               # ReLU = monkey default (LOG-038)


def load(path):
    f = h5py.File(path, "r")
    t = np.array(f["t"]).squeeze()
    fp = np.array(f["finger_pos"])
    sp = f["spikes"]
    edges = np.arange(t[0], t[-1], BIN)
    centers = edges[:-1] + BIN / 2
    rates = []
    for i in range(sp.shape[0]):
        for j in range(sp.shape[1]):
            st = np.array(f[sp[i, j]]).squeeze()
            if st.ndim == 0 or st.size < MIN_SPK:
                continue
            rates.append(np.histogram(st, bins=edges)[0])
    rates = np.asarray(rates, dtype=np.float32)                  # (units, nb)
    pos_b = np.stack([np.interp(centers, t, fp[a]) for a in range(fp.shape[0])], 1)
    raw3 = np.gradient(pos_b, BIN, axis=0)                       # raw velocity 3D
    sos = butter(4, 3.0 / (0.5 / BIN), btype="low", output="sos")
    slow3 = np.gradient(sosfiltfilt(sos, pos_b, axis=0), BIN, axis=0)
    axes = np.sort(np.argsort(raw3.std(0))[-2:])                 # top-2 movement axes (2D)
    raw = raw3[:, axes].astype(np.float32)
    slow = slow3[:, axes].astype(np.float32)
    fast = (raw - slow).astype(np.float32)
    return rates, raw, slow, fast, axes, rates.shape[1]


def windows(rates, raw, slow, fast, nb):
    w = int(round(WIN / BIN))
    n = nb // w
    tri = []
    for k in range(n):
        sl = slice(k * w, (k + 1) * w)
        tri.append({"e": rates[:, sl], "raw": raw[sl], "slow": slow[sl],
                    "fast": fast[sl], "series": int(k * KFOLD / n) + 1})
    return tri


def train_predict(tr, te, cfg, key):
    """Train TCN+GRU on target `key`; return test predictions in REAL units,
    flattened (n_te*T, 2)."""
    import torch
    import torch.nn as nn
    torch.set_num_threads(4); torch.manual_seed(42); np.random.seed(42)
    n_ch = tr[0]["e"].shape[0]
    T = min(t["e"].shape[1] for t in tr + te)
    Xtr = np.stack([t["e"][:, :T] for t in tr]); Ytr = np.stack([t[key][:T] for t in tr])
    Xte = np.stack([t["e"][:, :T] for t in te])
    cm, cs = Xtr.mean((0, 2), keepdims=True), Xtr.std((0, 2), keepdims=True) + 1e-6
    ym, ys = Ytr.mean((0, 1)), Ytr.std((0, 1)) + 1e-6
    Xtr = ((Xtr - cm) / cs).astype(np.float32); Xte = ((Xte - cm) / cs).astype(np.float32)
    Ytn = ((Ytr - ym) / ys).astype(np.float32)
    net = R.build_net({**cfg, "n_out": 2}, n_ch)
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
    with torch.no_grad():
        pr = net(torch.tensor(Xte)).numpy() * ys + ym          # (n_te, T, 2) real
    return pr.reshape(-1, 2), T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/indy_loco/indy_20161005_06.mat")
    args = ap.parse_args()
    p = ROOT / args.file if not Path(args.file).is_absolute() else Path(args.file)
    rates, raw, slow, fast, axes, nb = load(str(p))
    tri = windows(rates, raw, slow, fast, nb)
    print(f"=== slow/fast decomposition | {p.name} | {rates.shape[0]} units, "
          f"{nb} bins, {len(tri)} windows, axes {axes.tolist()} ===\n", flush=True)

    blocks = sorted({t["series"] for t in tri})
    acc = {k: [] for k in ("raw_direct", "slow_slow", "slow_raw", "fast_fast")}
    add_acc = {a: [] for a in ALPHAS}
    for h in blocks:
        tr = [t for t in tri if t["series"] != h]
        te = [t for t in tri if t["series"] == h]
        t0 = time.time()
        pr_raw, T = train_predict(tr, te, CFG, "raw")
        pr_slow, _ = train_predict(tr, te, CFG, "slow")
        pr_fast, _ = train_predict(tr, te, CFG, "fast")
        y_raw = np.stack([t["raw"][:T] for t in te]).reshape(-1, 2)
        y_slow = np.stack([t["slow"][:T] for t in te]).reshape(-1, 2)
        y_fast = np.stack([t["fast"][:T] for t in te]).reshape(-1, 2)
        acc["raw_direct"].append(R.corr(y_raw, pr_raw))
        acc["slow_slow"].append(R.corr(y_slow, pr_slow))
        acc["slow_raw"].append(R.corr(y_raw, pr_slow))
        acc["fast_fast"].append(R.corr(y_fast, pr_fast))
        for a in ALPHAS:
            add_acc[a].append(R.corr(y_raw, pr_slow + a * pr_fast))
        print(f"  block {h}/{len(blocks)} done [{time.time()-t0:.0f}s]", flush=True)

    def m(v):
        return np.mean(v, axis=0)
    res = {k: m(v) for k, v in acc.items()}
    add = {a: float(m(v).mean()) for a, v in add_acc.items()}
    best_a = max(add, key=add.get)
    print("\n--- RESULTS (mean r over 5 folds) ---")
    print(f"  raw_direct   (pred_raw  vs raw ): {res['raw_direct'].mean():.3f} "
          f"{res['raw_direct'].round(3).tolist()}")
    print(f"  slow_vs_slow (pred_slow vs slow): {res['slow_slow'].mean():.3f}")
    print(f"  slow_vs_raw  (pred_slow vs raw ): {res['slow_raw'].mean():.3f}")
    print(f"  fast_vs_fast (pred_fast vs fast): {res['fast_fast'].mean():.3f}")
    print("  additive (pred_slow + a*pred_fast vs raw):")
    for a in ALPHAS:
        print(f"      alpha={a}: {add[a]:.3f}")
    print(f"  best_alpha={best_a}  best_additive_r={add[best_a]:.3f}  "
          f"(raw_direct={res['raw_direct'].mean():.3f})")

    out = ROOT / "results" / "metrics" / "indy_slow_fast.json"
    out.write_text(json.dumps({
        "file": p.name, "n_units": int(rates.shape[0]), "n_bins": int(nb),
        "n_windows": len(tri), "axes": axes.tolist(),
        "raw_direct_r": float(res["raw_direct"].mean()),
        "slow_vs_slow_r": float(res["slow_slow"].mean()),
        "slow_vs_raw_r": float(res["slow_raw"].mean()),
        "fast_vs_fast_r": float(res["fast_fast"].mean()),
        "additive_by_alpha": add, "best_alpha": best_a,
        "best_additive_r": add[best_a]}, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
