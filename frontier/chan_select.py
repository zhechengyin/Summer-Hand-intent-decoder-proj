#!/usr/bin/env python
"""How should we pick the 8 channels? (hardware reads only 8 of 96 at a time)

Deployment reality: an 8-channel front-end. Question: does *how* we choose the 8
matter, and how close can a smart 8 get to the full 96-electrode ceiling (0.87)?

Strategies (each ends with a clean 8-channel TCN+GRU trained on just those 8,
matching deployment), held-out cross-session:
  random8   : 8 random electrodes (floor)
  firing8   : top-8 by mean train firing rate (naive static)
  learned8  : top-8 by a learned L1 stochastic-gate over all 96 (Balin'19 /
              Yamada'20 style) -- the decoder itself says which 8 it wants
Ceiling (96ch) comes from indy_nch.json / indy_crosssession (r=0.87).

Usage: py tools/indy_chan_select.py
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

import frontier.crosssession as X
import frontier.core as R

K = 8


def subset(trials, sel):
    return [{"e": t["e"][sel], "vel": t["vel"]} for t in trials]


def learned_select(train_trials, cfg, k=K, epochs=30, lam=3e-3, seed=0):
    """Train a 96-ch decoder with a per-channel L1 gate; return top-k gated chans."""
    import torch
    import torch.nn as nn
    torch.set_num_threads(4); torch.manual_seed(seed); np.random.seed(seed)
    n_ch = train_trials[0]["e"].shape[0]
    T = min(t["e"].shape[1] for t in train_trials)
    Xtr = np.stack([t["e"][:, :T] for t in train_trials])
    Ytr = np.stack([t["vel"][:T] for t in train_trials])
    ym, ys = Ytr.mean((0, 1)), Ytr.std((0, 1)) + 1e-6
    Ytn = ((Ytr - ym) / ys).astype(np.float32)

    class GateNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Parameter(torch.zeros(n_ch))      # sigmoid(0)=0.5
            self.body = R.build_net({**cfg, "n_out": Ytr.shape[-1]}, n_ch)

        def forward(self, x):
            return self.body(x * torch.sigmoid(self.gate)[None, :, None])

    net = GateNet()
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    mse = nn.MSELoss()
    Xt, Yt = torch.tensor(Xtr), torch.tensor(Ytn)
    idx = np.arange(len(Xt))
    for ep in range(epochs):
        net.train(); np.random.shuffle(idx)
        for b in range(0, len(idx), cfg["bs"]):
            xb = Xt[idx[b:b + cfg["bs"]]]
            xb = xb + cfg["noise"] * torch.randn_like(xb)
            opt.zero_grad()
            loss = mse(net(xb), Yt[idx[b:b + cfg["bs"]]]) \
                + lam * torch.sigmoid(net.gate).sum()
            loss.backward(); opt.step()
        sched.step()
    g = torch.sigmoid(net.gate).detach().numpy()
    sel = np.sort(np.argsort(g)[-k:])
    return sel, g


def main():
    print("=== channel-selection strategies (8-of-96, held-out cross-session) ===")
    print(f"TRAIN {[s[5:] for s in X.TRAIN]}  TEST {[s[5:] for s in X.TEST]}\n",
          flush=True)
    loaded = {s: X.load_electrode(s) for s in X.TRAIN + X.TEST}
    fr = np.mean([loaded[s][0].mean(1) for s in X.TRAIN], 0)
    var = np.mean([loaded[s][1].std(0) for s in X.TRAIN], 0)
    axes = np.sort(np.argsort(var)[-2:])

    tr96 = []
    for s in X.TRAIN:
        tr96 += X.windows(loaded[s][0], loaded[s][1], axes)
    te96 = {s: X.windows(loaded[s][0], loaded[s][1], axes) for s in X.TEST}

    rng = np.random.default_rng(0)
    sels = {
        "random8": np.sort(rng.choice(96, K, replace=False)),
        "firing8": np.sort(np.argsort(fr)[-K:]),
    }
    print("learning L1-gate channel importance on pooled train ...", flush=True)
    t0 = time.time()
    sels["learned8"], gate = learned_select(tr96, X.CFG)
    print(f"  gate done [{time.time()-t0:.0f}s]  learned8={sels['learned8'].tolist()}",
          flush=True)
    print(f"  firing8 ={sels['firing8'].tolist()}", flush=True)
    print(f"  overlap learned8 vs firing8: "
          f"{len(set(sels['learned8']) & set(sels['firing8']))}/8\n", flush=True)

    report = {}
    for nm, sel in sels.items():
        t0 = time.time()
        res = X.train_eval(subset(tr96, sel), {s: subset(v, sel)
                                               for s, v in te96.items()}, X.CFG)
        per = {s[5:]: float(r.mean()) for s, r in res.items()}
        mean = float(np.mean(list(per.values())))
        report[nm] = {"channels": sel.tolist(), "mean_r": mean, "per_session": per}
        print(f"  {nm:9s}: held-out mean r = {mean:.3f}  {per}  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    print("\n--- 8-channel selection (ceiling: 96ch = 0.87) ---")
    for nm in ("random8", "firing8", "learned8"):
        print(f"  {nm:9s}: {report[nm]['mean_r']:.3f}")
    out = ROOT / "results" / "metrics" / "indy_chan_select.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
