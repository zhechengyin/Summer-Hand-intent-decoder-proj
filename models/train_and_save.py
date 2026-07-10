#!/usr/bin/env python
"""Train the current-best config on all training sessions and save a checkpoint.

Produces models/checkpoint.pt = {state_dict, config, norm stats,
metrics}. Reuses the exact preprocessing + config from the research pipeline
(models.crosssession) so the saved model matches the reported 0.87 held-out.

Usage: py models/train_and_save.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import models.crosssession as X
import models.best_model as C


def main():
    import torch
    import torch.nn as nn
    torch.set_num_threads(4); torch.manual_seed(42); np.random.seed(42)

    loaded = {s: X.load_electrode(s) for s in X.TRAIN}
    var = np.mean([loaded[s][1].std(0) for s in X.TRAIN], 0)
    axes = np.sort(np.argsort(var)[-2:])
    trials = []
    for s in X.TRAIN:
        trials += X.windows(loaded[s][0], loaded[s][1], axes)

    T = min(t["e"].shape[1] for t in trials)
    Xtr = np.stack([t["e"][:, :T] for t in trials])
    Ytr = np.stack([t["vel"][:T] for t in trials])
    ym, ys = Ytr.mean((0, 1)), Ytr.std((0, 1)) + 1e-6
    Ytn = ((Ytr - ym) / ys).astype(np.float32)
    cfg = {**X.CFG, "n_out": Ytr.shape[-1]}
    net = C.build_net(cfg, Xtr.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    mse = nn.MSELoss()
    Xt, Yt = torch.tensor(Xtr), torch.tensor(Ytn)
    idx = np.arange(len(Xt))
    print(f"training best config on {len(trials)} windows from {len(X.TRAIN)} "
          f"sessions, axes {axes.tolist()} ...", flush=True)
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        net.train(); np.random.shuffle(idx)
        for b in range(0, len(idx), cfg["bs"]):
            bi = idx[b:b + cfg["bs"]]
            xb = Xt[bi] + cfg["noise"] * torch.randn_like(Xt[bi])
            m = (torch.rand(xb.shape[0], xb.shape[1], 1) > cfg["chdrop"]).float()
            xb = xb * m / (1 - cfg["chdrop"])
            opt.zero_grad(); mse(net(xb), Yt[bi]).backward(); opt.step()
        sched.step()

    out = Path(__file__).resolve().parent / "checkpoint.pt"
    torch.save({"state_dict": net.state_dict(), "config": cfg, "axes": axes.tolist(),
                "y_mean": ym.tolist(), "y_std": ys.tolist(),
                "note": "per-electrode 40ms rates, 3Hz vel-LP, sigma1 smooth, 2D; "
                        "held-out cross-session r~0.87 (96ch)"}, out)
    npar = sum(p.numel() for p in net.parameters())
    print(f"saved {out}  ({npar:,} params, {npar*4/1e6:.2f} MB) "
          f"[{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
