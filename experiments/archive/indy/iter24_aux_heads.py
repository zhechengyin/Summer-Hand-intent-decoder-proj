#!/usr/bin/env python
"""Iteration 24: auxiliary-head regularization on the causal MULTISCALE model.

The model of record is now causal wide TCN+GRU + 2-scale input (raw + EWMA 0.2,
16 features), TEST R2 = 0.633 single / 0.646 3-seed (LOG-070/071). This tests the
two untried "second head" levers from NEXT_EXPERIMENTS on top of that model:

  * multi-task kinematics (B): predict velocity AND an auxiliary kinematic signal
    (position = causal cumulative integral of velocity; or speed = |v|) with a
    combined loss, to regularize the shared representation.
  * LFADS-lite (A): predict velocity AND reconstruct the (z-scored) raw firing
    rates from the GRU latent, forcing the latent to model population dynamics.

Only the velocity head is scored (R2 on test1). The aux head is a training-time
regularizer (lambda-weighted), dropped at inference -> deployable size unchanged.
Strictly causal, 8 ch, 24 sess, 3-seed. Same recipe/aug as the model of record.

Usage: py experiments/archive/indy/iter24_aux_heads.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.archive.indy.iter20_multiscale as I20
import models.tcn_gru.evaluate as E
import models.tcn_gru.best_model as M

SEEDS = (42, 1, 7)
ALPHA = 0.2                       # adopted 2-scale config: raw + EWMA(0.2)
LAMBDA = 0.3                      # aux-loss weight
WIN = I20.WIN
NV = 2                            # velocity output dims (scored)

# aux target builders: rates (8,T) z-scored, vel (T,2) -> aux (T, D) or None
AUX = {
    "none":  (0, None),
    "pos":   (2, lambda r, v: np.cumsum(v, axis=0).astype(np.float32)),          # causal integral
    "speed": (1, lambda r, v: np.linalg.norm(v, axis=1, keepdims=True).astype(np.float32)),
    "recon": (8, lambda r, v: r.T.astype(np.float32)),                            # LFADS-lite: reconstruct rates
}


def prep(aux_fn, axes):
    """2-scale multiscale windows with an optional per-window aux target."""
    def wins(s):
        r, v = E.load_electrode(s)
        rates = r[I20.CHANNELS]                                   # (8, T) raw rates
        feat = I20.ewma_feats(rates, [1.0, ALPHA])               # raw + EWMA -> (16, T)
        mu, sd = feat.mean(1, keepdims=True), feat.std(1, keepdims=True) + 1e-6
        fz = ((feat - mu) / sd).astype(np.float32)
        rmu, rsd = rates.mean(1, keepdims=True), rates.std(1, keepdims=True) + 1e-6
        rz = ((rates - rmu) / rsd).astype(np.float32)            # z-scored rates for recon aux
        out = []
        for k in range(fz.shape[1] // WIN):
            sl = slice(k * WIN, (k + 1) * WIN)
            vw = v[sl][:, axes]
            d = {"e": fz[:, sl], "vel": vw}
            if aux_fn is not None:
                d["aux"] = aux_fn(rz[:, sl], vw)
            out.append(d)
        return out
    tr = [x for s in I20.TRAIN for x in wins(s)]
    return {"train": tr, "eval": {s: wins(s) for s in E.EVAL},
            "test": {s: wins(s) for s in E.TEST}}


def _stack(trials, key, D):
    T = min(t["e"].shape[1] for t in trials)
    return np.stack([t[key][:T] for t in trials]).astype(np.float32)


def run_aux(data, aux_dim, seeds=SEEDS, lam=LAMBDA):
    """Custom train loop mirroring harness.run but with a lambda-weighted aux head.

    Only the velocity head (first NV cols) is scored. 3-seed ensemble -> test R2."""
    import torch
    import torch.nn as nn
    cfg = {**I20.CAUSAL, "n_out": NV + aux_dim}
    n_ch = data["train"][0]["e"].shape[0]

    Xtr = _stack(data["train"], "e", None)                       # (N, 16, T)
    Ytr = _stack(data["train"], "vel", NV)                       # (N, T, 2)
    vm, vs = Ytr.mean((0, 1)), Ytr.std((0, 1)) + 1e-6
    Yvn = ((Ytr - vm) / vs).astype(np.float32)
    if aux_dim:
        Atr = _stack(data["train"], "aux", aux_dim)              # (N, T, aux_dim)
        am, as_ = Atr.mean((0, 1)), Atr.std((0, 1)) + 1e-6
        Yan = ((Atr - am) / as_).astype(np.float32)

    def prep_eval(by):
        out = {}
        for name, tri in by.items():
            Tt = min(t["e"].shape[1] for t in tri)
            Xe = np.stack([t["e"][:, :Tt] for t in tri]).astype(np.float32)
            Ye = np.stack([t["vel"][:Tt] for t in tri]).astype(np.float32)
            out[name] = (Xe, Ye)
        return out
    ev_p, te_p = prep_eval(data["eval"]), prep_eval(data["test"])

    def predict_vel(net, Xe):
        net.eval()
        with torch.no_grad():
            return net(torch.tensor(Xe)).numpy()[..., :NV] * vs + vm

    ens_te = {k: [] for k in te_p}
    ens_ev = {k: [] for k in ev_p}
    n_params = None
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(4)
        net = M.build_net(cfg, n_ch)
        if n_params is None:
            n_params = sum(p.numel() for p in net.parameters())
        opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
        mse = nn.MSELoss()
        Xt, Yv = torch.tensor(Xtr), torch.tensor(Yvn)
        Ya = torch.tensor(Yan) if aux_dim else None
        idx = np.arange(len(Xt)); noise = cfg["noise"]; chd = cfg["chdrop"]
        best, best_state = -np.inf, None
        for ep in range(cfg["epochs"]):
            net.train(); np.random.shuffle(idx)
            for b in range(0, len(idx), cfg["bs"]):
                bi = idx[b:b + cfg["bs"]]
                xb = Xt[bi]
                if noise > 0:
                    xb = xb + noise * torch.randn_like(xb)
                if chd > 0:
                    m = (torch.rand(xb.shape[0], xb.shape[1], 1) > chd).float()
                    xb = xb * m / (1 - chd)
                pred = net(xb)
                loss = mse(pred[..., :NV], Yv[bi])
                if aux_dim:
                    loss = loss + lam * mse(pred[..., NV:], Ya[bi])
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            rs = []
            for name, (Xe, Ye) in ev_p.items():
                p = predict_vel(net, Xe)
                rs.append(M.corr(Ye.reshape(-1, NV), p.reshape(-1, NV)).mean())
            m_ = float(np.mean(rs))
            if m_ > best:
                best = m_
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(best_state)
        for name, (Xe, Ye) in ev_p.items():
            ens_ev[name].append(predict_vel(net, Xe))
        for name, (Xe, Ye) in te_p.items():
            ens_te[name].append(predict_vel(net, Xe))

    def agg(ens, part):
        r2 = []
        for name, preds in ens.items():
            yh = np.mean(preds, 0).reshape(-1, NV); y = part[name][1].reshape(-1, NV)
            r2.append(M.r2(y, yh))
        return float(np.mean(r2))
    return {"eval_r2": agg(ens_ev, ev_p), "test_r2": agg(ens_te, te_p),
            "n_params": int(n_params)}


def main():
    axes = np.sort(np.argsort(np.mean([E.load_electrode(s)[1].std(0)
                                       for s in E.TRAIN], 0))[-2:])
    configs = ["none", "pos", "recon"]        # baseline, multi-task-position, LFADS-lite
    print("=== Iteration 24: auxiliary-head regularization (multiscale, causal, 3-seed) ===")
    print(f"    ref: multiscale baseline 3-seed = 0.646 (LOG-070); lambda={LAMBDA}\n", flush=True)
    rows = {}
    for name in configs:
        aux_dim, aux_fn = AUX[name]
        t0 = time.time()
        res = run_aux(prep(aux_fn, axes), aux_dim)
        rows[name] = {"test_r2": res["test_r2"], "eval_r2": res["eval_r2"],
                      "aux_dim": aux_dim, "int8_kb": res["n_params"] / 1024}
        tag = "(baseline)" if name == "none" else ""
        print(f"  aux={name:6s} TEST R2={res['test_r2']:.3f}  EVAL R2={res['eval_r2']:.3f}  "
              f"(~{res['n_params']/1024:.0f} KB) {tag} [{time.time()-t0:.0f}s]", flush=True)

    base = rows["none"]["test_r2"]
    print(f"\n  deltas vs baseline ({base:.3f}):")
    for name in configs:
        if name != "none":
            print(f"    {name:6s}: {rows[name]['test_r2'] - base:+.3f}")
    out = ROOT / "results" / "metrics" / "iter24_aux_heads.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
