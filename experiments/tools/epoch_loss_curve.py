#!/usr/bin/env python
"""Record a training curve (epoch vs train loss + eval R2) for the current-best 8ch model.

Same pipeline as the model of record (strictly-causal wide TCN+GRU + multiscale
raw+EWMA(0.2), 8 firing channels, 24 sessions, 40 ms bins), single seed, 60 epochs.
Logs mean train MSE (on normalized targets) and eval mean r / R2 each epoch so we can
see convergence and where early-stopping picks the best epoch.
Usage: py research/epoch_loss_curve.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.iter20_multiscale as I20
import models.tcn_gru.evaluate as E
import models.tcn_gru.best_model as M

SEED = 42
ALPHAS = [1.0, 0.2]
WIN = I20.WIN
OUT = ROOT / "results" / "metrics" / "epoch_loss_curve.json"


def prep(chans, axes):
    def wins(s):
        r, v = E.load_electrode(s)
        feat = I20.ewma_feats(r[chans], ALPHAS)
        mu, sd = feat.mean(1, keepdims=True), feat.std(1, keepdims=True) + 1e-6
        fz = ((feat - mu) / sd).astype(np.float32)
        return [{"e": fz[:, k * WIN:(k + 1) * WIN], "vel": v[k * WIN:(k + 1) * WIN][:, axes]}
                for k in range(fz.shape[1] // WIN)]
    tr = [x for s in I20.TRAIN for x in wins(s)]
    return {"train": tr, "eval": {s: wins(s) for s in E.EVAL},
            "test": {s: wins(s) for s in E.TEST}}


def main():
    import torch
    import torch.nn as nn
    chans = I20.CHANNELS
    axes = np.sort(np.argsort(np.mean([E.load_electrode(s)[1].std(0)
                                       for s in E.TRAIN], 0))[-2:])
    data = prep(chans, axes)
    cfg = {**I20.CAUSAL, "n_out": 2}
    n_ch = data["train"][0]["e"].shape[0]

    Xtr = np.stack([t["e"][:, :min(t["e"].shape[1] for t in data["train"])]
                    for t in data["train"]]).astype(np.float32)
    Ytr = np.stack([t["vel"][:Xtr.shape[2]] for t in data["train"]]).astype(np.float32)
    ym, ys = Ytr.mean((0, 1)), Ytr.std((0, 1)) + 1e-6
    Ytn = ((Ytr - ym) / ys).astype(np.float32)

    def prep_eval(by):
        out = {}
        for name, tri in by.items():
            T = min(t["e"].shape[1] for t in tri)
            Xe = np.stack([t["e"][:, :T] for t in tri]).astype(np.float32)
            Ye = np.stack([t["vel"][:T] for t in tri]).astype(np.float32)
            out[name] = (Xe, Ye)
        return out
    ev_p = prep_eval(data["eval"])

    torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(4)
    net = M.build_net(cfg, n_ch)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    mse = nn.MSELoss()
    Xt, Yt = torch.tensor(Xtr), torch.tensor(Ytn)
    idx = np.arange(len(Xt)); noise = cfg["noise"]; chd = cfg["chdrop"]

    hist = {"epoch": [], "train_loss": [], "eval_r": [], "eval_r2": [], "lr": []}
    best, best_ep = -np.inf, 0
    print(f"=== epoch-loss curve: 8ch causal multiscale, seed {SEED}, {cfg['epochs']} epochs ===",
          flush=True)
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        net.train(); np.random.shuffle(idx); losses = []
        for b in range(0, len(idx), cfg["bs"]):
            xb = Xt[idx[b:b + cfg["bs"]]]
            if noise > 0:
                xb = xb + noise * torch.randn_like(xb)
            if chd > 0:
                m = (torch.rand(xb.shape[0], xb.shape[1], 1) > chd).float()
                xb = xb * m / (1 - chd)
            opt.zero_grad()
            loss = mse(net(xb), Yt[idx[b:b + cfg["bs"]]])
            loss.backward(); opt.step()
            losses.append(float(loss))
        sched.step()
        # eval
        net.eval()
        rs, r2s = [], []
        with torch.no_grad():
            for name, (Xe, Ye) in ev_p.items():
                p = net(torch.tensor(Xe)).numpy() * ys + ym
                rs.append(M.corr(Ye.reshape(-1, 2), p.reshape(-1, 2)).mean())
                r2s.append(M.r2(Ye.reshape(-1, 2), p.reshape(-1, 2)).mean())
        er, er2 = float(np.mean(rs)), float(np.mean(r2s))
        tl = float(np.mean(losses))
        hist["epoch"].append(ep + 1); hist["train_loss"].append(tl)
        hist["eval_r"].append(er); hist["eval_r2"].append(er2)
        hist["lr"].append(float(sched.get_last_lr()[0]))
        if er > best:
            best, best_ep = er, ep + 1
        print(f"  ep {ep+1:2d}  train_loss={tl:.4f}  eval_r={er:.3f}  eval_R2={er2:.3f}",
              flush=True)

    hist["best_epoch"] = best_ep
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(hist, indent=2), encoding="utf-8")
    print(f"\nBest epoch (by eval r) = {best_ep}  [{time.time()-t0:.0f}s]")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
