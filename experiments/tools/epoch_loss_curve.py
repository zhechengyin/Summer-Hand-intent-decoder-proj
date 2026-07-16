#!/usr/bin/env python
"""Train the current-best 8ch model and record + PLOT the epoch curves. One script does it all.

Same pipeline as the model of record (strictly-causal wide TCN+GRU + multiscale
raw+EWMA(0.2), 8 firing channels, 24 sessions, 40 ms bins), single seed, 60 epochs.
Each epoch it records train loss, validation(eval) loss+R2, and TEST loss+R2. TEST is
recorded for the plot ONLY -- the epoch is still selected on EVAL, never on test.
CHANNEL COUNT is a command-line number (default 8): the top-N electrodes by firing rate.
    py experiments/tools/epoch_loss_curve.py         # 8 channels
    py experiments/tools/epoch_loss_curve.py 32      # 32 channels
    py experiments/tools/epoch_loss_curve.py 16      # 16, etc.
Outputs are suffixed by N so they never clash:
    results/metrics/epoch_loss_curve_{N}ch.json  +  results/figures/epoch_loss_curve_{N}ch.png
To only redraw a figure from a cached JSON (no retraining):
    py experiments/tools/plot_epoch_curve.py   (or import plot_epoch_curve.render with a json/png path)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.archive.indy.iter20_multiscale as I20
import models.tcn_gru.evaluate as E
import models.tcn_gru.best_model as M

SEED = 42
ALPHAS = [1.0, 0.2]
ETA_MIN = 0.0      # cosine-LR floor. 0 = anneal to zero (default). >0 keeps the model learning
                   # late so overfitting can actually push val loss UP (the classic U-curve).
NOREG = False      # if True, zero ALL regularization (dropout/weight-decay/noise/chdrop) so the
                   # model is free to memorize -> the real U-curve. Pass "noreg" on the CLI.
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


def top_n_channels(n):
    """Top-n electrodes by mean firing rate on the base-6 sessions (the deployment rule)."""
    fr = np.mean([E.load_electrode(s)[0].mean(1) for s in E.TRAIN], 0)
    return np.sort(np.argsort(fr)[-n:])


def run(chans, out_json, out_png, title):
    import torch
    import torch.nn as nn
    axes = np.sort(np.argsort(np.mean([E.load_electrode(s)[1].std(0)
                                       for s in E.TRAIN], 0))[-2:])
    data = prep(chans, axes)
    cfg = {**I20.CAUSAL, "n_out": 2}
    if NOREG:                                            # strip all regularization -> free to memorize
        cfg = {**cfg, "dropout": 0.0, "wd": 0.0, "noise": 0.0, "chdrop": 0.0}
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
    te_p = prep_eval(data["test"])       # test1: RECORDED per epoch for the plot, NEVER used to select

    torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(4)
    net = M.build_net(cfg, n_ch)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"], eta_min=ETA_MIN)
    mse = nn.MSELoss()
    Xt, Yt = torch.tensor(Xtr), torch.tensor(Ytn)
    idx = np.arange(len(Xt)); noise = cfg["noise"]; chd = cfg["chdrop"]

    hist = {"epoch": [], "train_loss": [], "val_loss": [], "eval_r2": [],
            "test_loss": [], "test_r2": [], "lr": []}
    best, best_ep = -np.inf, 0
    print(f"=== epoch-loss curve: {len(chans)}ch causal multiscale, seed {SEED}, "
          f"{cfg['epochs']} epochs ===", flush=True)
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

        def score_set(prepared):
            """mean (R2, loss-MSE-on-normalized-scale) over the sessions in a prepared split."""
            r2s, msel = [], []
            with torch.no_grad():
                for name, (Xe, Ye) in prepared.items():
                    p = net(torch.tensor(Xe)).numpy() * ys + ym
                    y = Ye.reshape(-1, 2); ph = p.reshape(-1, 2)
                    r2s.append(M.r2(y, ph).mean())
                    msel.append(float((((ph - ym) / ys - (y - ym) / ys) ** 2).mean()))
            return float(np.mean(r2s)), float(np.mean(msel))

        net.eval()
        er2, vl = score_set(ev_p)                        # EVAL -- used for selection
        ter2, tel = score_set(te_p)                      # TEST -- recorded only, NEVER selected on
        tl = float(np.mean(losses))
        hist["epoch"].append(ep + 1); hist["train_loss"].append(tl); hist["val_loss"].append(vl)
        hist["eval_r2"].append(er2); hist["test_loss"].append(tel); hist["test_r2"].append(ter2)
        hist["lr"].append(float(sched.get_last_lr()[0]))
        if er2 > best:                                   # selection is on EVAL only
            best, best_ep = er2, ep + 1
        print(f"  ep {ep+1:2d}  train_loss={tl:.4f}  val_loss={vl:.4f} eval_R2={er2:.3f}  |  "
              f"test_loss={tel:.4f} test_R2={ter2:.3f}", flush=True)

    hist["best_epoch"] = best_ep
    out_json = Path(out_json); out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(hist, indent=2), encoding="utf-8")
    print(f"\nBest epoch (by eval R2) = {best_ep}  [{time.time()-t0:.0f}s]")
    print(f"Wrote {out_json}")
    # auto-generate the figure so running THIS script alone trains + plots
    try:
        import experiments.tools.plot_epoch_curve as P
        P.render(out_json, out_png, title=title)
    except Exception as e:
        print(f"(plot skipped: {e})")


def main():
    # args (order-free): `py research/epoch_loss_curve.py [n_channels] [eta_min] [noreg]`
    #   n_channels : top-N electrodes (default 8)
    #   eta_min    : a float -> cosine-LR floor (default 0 = anneal to zero; 1e-3 = ~constant LR)
    #   noreg      : the word "noreg" -> zero all regularization (free to memorize -> real U-curve)
    global ETA_MIN, NOREG
    args = sys.argv[1:]
    n = 8
    for a in args:
        if a == "noreg":
            NOREG = True
        elif "." in a or "e" in a.lower():
            ETA_MIN = float(a)
        else:
            n = int(a)
    chans = top_n_channels(n)                             # top-N electrodes by firing rate
    tag = f"{n}ch" + (f"_lrmin{ETA_MIN:g}" if ETA_MIN > 0 else "") + ("_noreg" if NOREG else "")
    notes = (f", LR floored at {ETA_MIN:g}" if ETA_MIN > 0 else "") + (", NO regularization" if NOREG else "")
    run(chans,
        ROOT / "results" / "metrics" / f"epoch_loss_curve_{tag}.json",
        ROOT / "results" / "figures" / f"epoch_loss_curve_{tag}.png",
        f"{n}-channel decoder — training curves (seed 42, 24 sessions{notes})")


if __name__ == "__main__":
    main()
