#!/usr/bin/env python
"""Training and evaluation harness for the strictly causal decoder.

``run`` operates on already prepared train/eval/test trials. Active pipelines
prepare data through ``src.intent_decoder.data`` so causal and normalization
assumptions remain explicit. Historical compatibility lives in
the research log rather than in an executable parallel harness.

Features include:
  * both Pearson r and R^2 (coefficient of determination) reporting,
  * optional architecture/loss hooks and seed ensembling.
"""
from __future__ import annotations

import numpy as np

from src.intent_decoder.model.tcn_gru import build_net as causal_build_net
from src.intent_decoder.model.tcn_gru import corr, r2


def _stack(trials):
    T = min(t["e"].shape[1] for t in trials)
    X = np.stack([t["e"][:, :T] for t in trials]).astype(np.float32)
    Y = np.stack([t["vel"][:T] for t in trials]).astype(np.float32)
    return X, Y, T


def run(data, cfg, build=None, loss_fn=None, post=None, seeds=(42,), verbose=False,
        ret_preds=False, ret_net=False):
    """Train (early-stop on eval mean r) and score. Ensembled over `seeds`.

    Returns dict: eval_r, eval_r2, test_r, test_r2, per-axis arrays, n_params.
    build(cfg, n_ch)->net (default strictly causal model); loss_fn(pred,targ)->scalar
    (default MSE); post(pred (n,T,D))->pred smooths outputs (default identity)."""
    import torch
    import torch.nn as nn
    build = build or causal_build_net
    post = post or (lambda p: p)
    n_ch = data["train"][0]["e"].shape[0]
    n_out = data["train"][0]["vel"].shape[-1]
    cfg = {**cfg, "n_out": n_out}
    Xtr, Ytr, T = _stack(data["train"])
    ym, ys = Ytr.mean((0, 1)), Ytr.std((0, 1)) + 1e-6
    Ytn = ((Ytr - ym) / ys).astype(np.float32)

    def prep_eval(by):
        out = {}
        for name, tri in by.items():
            Tt = min(t["e"].shape[1] for t in tri)
            Xe = np.stack([t["e"][:, :Tt] for t in tri]).astype(np.float32)
            Ye = np.stack([t["vel"][:Tt] for t in tri]).astype(np.float32)
            out[name] = (Xe, Ye)
        return out
    ev_p, te_p = prep_eval(data["eval"]), prep_eval(data["test"])

    def predict(net, Xe):
        net.eval()
        with torch.no_grad():
            return net(torch.tensor(Xe)).numpy() * ys + ym    # raw (post applied later)

    n_params = sum(p.numel() for p in build(cfg, n_ch).parameters())
    ens_ev = {k: [] for k in ev_p}
    ens_te = {k: [] for k in te_p}
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(4)
        net = build(cfg, n_ch)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
        mse = nn.MSELoss()
        lf = loss_fn or (lambda pr, ta: mse(pr, ta))
        Xt, Yt = torch.tensor(Xtr), torch.tensor(Ytn)
        idx = np.arange(len(Xt)); noise = cfg["noise"]; chd = cfg["chdrop"]
        best, best_state = -np.inf, None
        for ep in range(cfg["epochs"]):
            net.train(); np.random.shuffle(idx)
            for b in range(0, len(idx), cfg["bs"]):
                xb = Xt[idx[b:b + cfg["bs"]]]
                if noise > 0:
                    xb = xb + noise * torch.randn_like(xb)
                if chd > 0:
                    m = (torch.rand(xb.shape[0], xb.shape[1], 1) > chd).float()
                    xb = xb * m / (1 - chd)
                opt.zero_grad(); lf(net(xb), Yt[idx[b:b + cfg["bs"]]]).backward(); opt.step()
            sched.step()
            # early stop on eval mean r
            rs = []
            for name, (Xe, Ye) in ev_p.items():
                p = post(predict(net, Xe))
                rs.append(corr(Ye.reshape(-1, n_out), p.reshape(-1, n_out)).mean())
            m_ = float(np.mean(rs))
            if m_ > best:
                best = m_
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(best_state)
        for name, (Xe, Ye) in ev_p.items():
            ens_ev[name].append(predict(net, Xe))
        for name, (Xe, Ye) in te_p.items():
            ens_te[name].append(predict(net, Xe))
        if verbose:
            print(f"    seed {seed}: eval_r={best:.3f}", flush=True)

    raw_ev = {name: np.mean(preds, 0) for name, preds in ens_ev.items()}  # ensemble
    raw_te = {name: np.mean(preds, 0) for name, preds in ens_te.items()}

    def agg(raw, part):
        correlations, r2_values = [], []
        for name, p in raw.items():
            Ye = part[name][1]
            yh = post(p).reshape(-1, n_out); y = Ye.reshape(-1, n_out)
            correlations.append(corr(y, yh)); r2_values.append(r2(y, yh))
        return np.mean(correlations, 0), np.mean(r2_values, 0)
    er, er2 = agg(raw_ev, ev_p)
    tr, tr2 = agg(raw_te, te_p)
    out = {"eval_r": float(er.mean()), "eval_r2": float(er2.mean()),
           "test_r": float(tr.mean()), "test_r2": float(tr2.mean()),
           "test_r_ax": tr.tolist(), "test_r2_ax": tr2.tolist(),
           "n_params": int(n_params), "mb": n_params * 4 / 1e6, "kb": n_params * 4 / 1024}
    if ret_preds:                     # raw (pre-post) ensembled preds + targets
        out["eval_preds"] = {n: (ev_p[n][1], raw_ev[n]) for n in raw_ev}
        out["test_preds"] = {n: (te_p[n][1], raw_te[n]) for n in raw_te}
    if ret_net:                       # last-seed trained net + norm + prepped arrays
        out["net"] = net              # for quantization / export
        out["norm"] = (ym, ys)
        out["ev_p"], out["te_p"], out["n_out"] = ev_p, te_p, n_out
    return out
