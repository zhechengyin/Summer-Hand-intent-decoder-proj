#!/usr/bin/env python
"""Evaluate the 8-channel STM32 decoder on the fixed train/eval/test split.

Reuses the per-electrode pipeline from models.tcn_gru.evaluate (load_electrode,
windows) but selects the top-8 firing electrodes on the base 6 sessions, trains
the shrunk model on 18 sessions, and reports R² (fp32 and int8). eval1 selects
the epoch; test1 is scored once.

Usage: py models/tcn_gru_8ch/evaluate.py
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

import models.tcn_gru.best_model as M
import models.tcn_gru.evaluate as E96          # reuse load_electrode / windows / EVAL / TEST
import models.tcn_gru_8ch.config as C


def quant_int8(w):
    import torch
    if w.dim() >= 2:                               # per-output-channel (TFLite-style)
        wf = w.reshape(w.shape[0], -1)
        scale = wf.abs().amax(1, keepdim=True) / 127.0
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        return (torch.round(wf / scale).clamp(-127, 127) * scale).reshape(w.shape)
    s = w.abs().max() / 127.0
    return w if s == 0 else torch.round(w / s).clamp(-127, 127) * s


def select_channels():
    """Top-8 firing electrodes on the base 6 training sessions (FIXED; LOG-053)."""
    fr = np.mean([E96.load_electrode(s)[0].mean(1) for s in C.BASE_TRAIN], 0)
    return np.sort(np.argsort(fr)[-C.N_CHANNELS:])


def movement_axes():
    var = np.mean([E96.load_electrode(s)[1].std(0) for s in C.BASE_TRAIN], 0)
    return np.sort(np.argsort(var)[-C.N_OUT:])


def wins(s, sel, axes):
    r, v = E96.load_electrode(s)
    return E96.windows(r[sel], v, axes)


def _pack(tri):
    T = min(t["e"].shape[1] for t in tri)
    X = np.stack([t["e"][:, :T] for t in tri]).astype(np.float32)
    Y = np.stack([t["vel"][:T] for t in tri]).astype(np.float32)
    return X, Y


def train(tr, ev_packed, cfg):
    import torch
    import torch.nn as nn
    torch.set_num_threads(4); torch.manual_seed(42); np.random.seed(42)
    Xtr, Ytr = _pack(tr)
    ym, ys = Ytr.mean((0, 1)), Ytr.std((0, 1)) + 1e-6
    Ytn = ((Ytr - ym) / ys).astype(np.float32)
    net = M.build_net(cfg, Xtr.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    mse = nn.MSELoss()
    Xt, Yt = torch.tensor(Xtr), torch.tensor(Ytn)
    idx = np.arange(len(Xt))

    def eval_r2():
        net.eval(); r2 = []
        for Xe, Ye in ev_packed.values():
            with torch.no_grad():
                pr = net(torch.tensor(Xe)).numpy() * ys + ym
            r2.append(M.r2(Ye.reshape(-1, Ye.shape[-1]), pr.reshape(-1, Ye.shape[-1])))
        return float(np.mean(r2))
    best, best_state = -1e9, None
    for ep in range(cfg["epochs"]):
        net.train(); np.random.shuffle(idx)
        for b in range(0, len(idx), cfg["bs"]):
            xb = Xt[idx[b:b + cfg["bs"]]]
            xb = xb + cfg["noise"] * torch.randn_like(xb)
            m = (torch.rand(xb.shape[0], xb.shape[1], 1) > cfg["chdrop"]).float()
            xb = xb * m / (1 - cfg["chdrop"])
            opt.zero_grad(); mse(net(xb), Yt[idx[b:b + cfg["bs"]]]).backward(); opt.step()
        sched.step()
        r = eval_r2()
        if r > best:
            best, best_state = r, {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net, ym, ys


def score(net, packed, ym, ys):
    import torch
    net.eval(); r2 = []
    for Xe, Ye in packed.values():
        with torch.no_grad():
            pr = net(torch.tensor(Xe)).numpy() * ys + ym
        r2.append(M.r2(Ye.reshape(-1, Ye.shape[-1]), pr.reshape(-1, Ye.shape[-1])))
    return float(np.mean(r2))


def main():
    import torch
    t0 = time.time()
    sel, axes = select_channels(), movement_axes()
    tr = [w for s in C.BASE_TRAIN + C.EXTRA_TRAIN for w in wins(s, sel, axes)]
    ev = {s: _pack(wins(s, sel, axes)) for s in E96.EVAL}
    te = {s: _pack(wins(s, sel, axes)) for s in E96.TEST}
    print(f"=== 8-ch STM32 decoder | channels {sel.tolist()} | axes {axes.tolist()} | "
          f"{len(C.BASE_TRAIN)+len(C.EXTRA_TRAIN)} sessions, {len(tr)} windows ===",
          flush=True)
    net, ym, ys = train(tr, ev, C.MODEL)
    fp32_e, fp32_t = score(net, ev, ym, ys), score(net, te, ym, ys)
    n = sum(p.numel() for p in net.parameters())
    with torch.no_grad():
        nq = 0
        for name, p in net.named_parameters():
            if "weight" in name and p.dim() >= 2:
                p.data = quant_int8(p.data); nq += p.numel()
    int8_e, int8_t = score(net, ev, ym, ys), score(net, te, ym, ys)
    print(f"  params {n:,} | fp32 {n*4/1024:.0f} kB -> int8 ~{(nq+(n-nq)*4)/1024:.0f} kB")
    print(f"  EVAL R2: fp32 {fp32_e:.3f} -> int8 {int8_e:.3f}")
    print(f"  TEST R2: fp32 {fp32_t:.3f} -> int8 {int8_t:.3f}  [{time.time()-t0:.0f}s]",
          flush=True)
    out = ROOT / "results" / "metrics" / "tcn_gru_8ch_eval.json"
    out.write_text(json.dumps({"channels": sel.tolist(), "axes": axes.tolist(),
                               "params": int(n), "int8_kb": (nq + (n - nq) * 4) / 1024,
                               "fp32_test_r2": fp32_t, "int8_test_r2": int8_t,
                               "fp32_eval_r2": fp32_e}, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
