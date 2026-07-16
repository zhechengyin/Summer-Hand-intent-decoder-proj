#!/usr/bin/env python
"""Iteration 36: META-LEARNED initialization (Reptile) -- make the pool earn its keep.

THE DIAGNOSIS: LOG-080 found that at 32ch, training from SCRATCH on half a session with NO pool
at all (0.576) matched fine-tuning the full 24-session pretrained model (0.584). The pool
contributes +0.008 -- essentially NOTHING. That is not a curiosity, it is a diagnosis: standard
pretraining optimizes "be a good POOL decoder", which is the WRONG objective for an
initialization you intend to ADAPT. Nobody ever asked this model to be adaptable, only accurate.

THE MOVE: Reptile (Nichol 2018; first-order MAML). Treat each pool session as a TASK. Inner loop:
k SGD steps on that session's data. Outer loop: move the initialization TOWARD the adapted weights
(theta <- theta + eps*(theta_adapted - theta)). This explicitly optimizes "how good am I AFTER a
short calibration" -- which is literally the deployment metric (LOG-084: calibration is the
insurance we run on the ~12% of sessions that drift).

THE TEST that matters (LOG-081): standard pretraining needs ~4 min of calibration to reach ~0.51;
60 s only reaches ~0.34. If the meta-init reaches ~0.5 at 60 s we have cut the calibration block
~4x -- the part that actually costs the user time.

Compares, on held-out sessions, at several calibration budgets:
  standard_init + finetune   (the LOG-080/081 baseline)
  reptile_init  + finetune   (same finetune, better starting point)
  scratch                    (no pool at all -- the bar the pool currently fails to clear)

Usage: py research/iter36_reptile.py
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.common.harness as H
import experiments.archive.indy.iter20_multiscale as I20
import experiments.archive.indy.iter27_fresh_session as I27
import experiments.archive.indy.iter28_calibration as I28
import experiments.archive.indy.iter31_channel_reselect as I31
import experiments.archive.indy.iter32_forward_split as I32
import models.tcn_gru.evaluate as E
import models.tcn_gru.best_model as M

SEED = 42
N_CH = 32
POOL = I32.POOL                      # Sep 15 - Dec 20 2016 (20 sessions) = the meta-training tasks
TARGETS = I27.FRESH + I32.FORWARD    # held-out: 4 backward + 4 forward
CAL_SIZES = [30, 120]                # windows -> 60 s, 240 s (the LOG-081 decision points)
FT_EPOCHS, FT_LR = 25, 3e-4
# Reptile hyperparameters
META_ITERS = 400                     # outer steps (each = one sampled session/task)
INNER_STEPS = 8                      # SGD steps per task
INNER_LR = 1e-3
META_EPS = 0.1                       # outer step size
OUT = ROOT / "results" / "metrics" / "iter36_reptile.json"


def reptile_train(task_data, cfg, n_ch, seed=SEED):
    """Meta-learn an initialization that ADAPTS fast. task_data: {session: (X, Y_norm)}."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(4)
    net = M.build_net({**cfg, "n_out": 2}, n_ch)
    mse = nn.MSELoss()
    names = list(task_data)
    t0 = time.time()
    for it in range(META_ITERS):
        s = names[np.random.randint(len(names))]              # sample a task
        X, Y = task_data[s]
        before = {k: v.detach().clone() for k, v in net.state_dict().items()}
        inner = torch.optim.SGD(net.parameters(), lr=INNER_LR, momentum=0.9)
        net.train()
        for _ in range(INNER_STEPS):                          # inner loop: adapt to this session
            bi = np.random.randint(0, len(X), size=min(cfg["bs"], len(X)))
            inner.zero_grad()
            mse(net(torch.tensor(X[bi])), torch.tensor(Y[bi])).backward()
            inner.step()
        # outer loop: move the init TOWARD the adapted weights
        after = net.state_dict()
        merged = {}
        for k in before:
            if before[k].dtype.is_floating_point:
                merged[k] = before[k] + META_EPS * (after[k] - before[k])
            else:
                merged[k] = after[k]
        net.load_state_dict(merged)
        if (it + 1) % 100 == 0:
            print(f"      reptile iter {it+1}/{META_ITERS} [{time.time()-t0:.0f}s]", flush=True)
    return net


def main():
    import torch
    t0 = time.time()
    counts = I27.load_counts_full(POOL + list(E.EVAL) + list(E.TEST) + TARGETS)
    axes = np.sort(np.argsort(np.mean([counts[s][1].std(0) for s in E.TRAIN], 0))[-2:])
    fr = np.mean([counts[s][0].mean(1) for s in POOL], 0)
    chans = np.sort(np.argsort(fr)[-N_CH:])
    print("=== Iteration 36: Reptile meta-learned initialization ===")
    print(f"    {N_CH} ch; {len(POOL)} meta-training sessions; {len(TARGETS)} held-out")
    print(f"    loaded {len(counts)} sessions [{time.time()-t0:.0f}s]\n", flush=True)

    # ---- standard pretrained init (the baseline) ----
    tr_by = I27.build(counts, chans, axes, POOL)
    data = {"train": [x for s in POOL for x in tr_by[s]],
            "eval": I27.build(counts, chans, axes, list(E.EVAL)),
            "test": I27.build(counts, chans, axes, list(E.TEST))}
    t1 = time.time()
    res = H.run(data, I20.CAUSAL, seeds=(SEED,), ret_net=True)
    std_state = {k: v.clone() for k, v in res["net"].state_dict().items()}
    ym, ys = res["norm"]
    print(f"  standard init: EVAL {res['eval_r2']:.3f} [{time.time()-t1:.0f}s]", flush=True)

    # ---- Reptile init (same data, same normalisation, different OBJECTIVE) ----
    task_data = {}
    for s in POOL:
        X, Y = I28.stack(tr_by[s])
        task_data[s] = (X, ((Y - ym) / ys).astype(np.float32))
    t1 = time.time()
    rep_net = reptile_train(task_data, I20.CAUSAL, 2 * N_CH)
    rep_state = {k: v.detach().clone() for k, v in rep_net.state_dict().items()}
    print(f"  reptile init done [{time.time()-t1:.0f}s]\n", flush=True)

    def predict(net, X, m, s_):
        net.eval()
        with torch.no_grad():
            return net(torch.tensor(X)).numpy() * s_ + m

    fresh_by = I27.build(counts, chans, axes, TARGETS)
    rows = {}
    for s in TARGETS:
        w = fresh_by[s]
        half = len(w) // 2
        Xt, Yt = I28.stack(w[half:]); yt = Yt.reshape(-1, 2)
        era = "backward" if s in I27.FRESH else "forward"
        rows[s] = {"era": era}
        # zero-shot from each init
        rows[s]["zero_shot_std"] = I28.score(yt, predict(
            _load(I20.CAUSAL, 2 * N_CH, std_state), Xt, ym, ys).reshape(-1, 2))
        rows[s]["zero_shot_rep"] = I28.score(yt, predict(
            _load(I20.CAUSAL, 2 * N_CH, rep_state), Xt, ym, ys).reshape(-1, 2))
        for N in CAL_SIZES:
            if N > half:
                continue
            Xc, Yc = I28.stack(w[:N])
            secs = int(N * I20.WIN * E.BIN)
            for tag, init in (("std", std_state), ("rep", rep_state)):
                net, m, s_ = I31.train(Xc, Yc, I20.CAUSAL, FT_EPOCHS, FT_LR,
                                       init=init, norm=(ym, ys))
                rows[s][f"ft_{tag}_{secs}s"] = I28.score(yt, predict(net, Xt, m, s_).reshape(-1, 2))
            net, m, s_ = I31.train(Xc, Yc, I20.CAUSAL, I20.CAUSAL["epochs"], I20.CAUSAL["lr"])
            rows[s][f"scratch_{secs}s"] = I28.score(yt, predict(net, Xt, m, s_).reshape(-1, 2))
        print(f"  {s:20s} [{era:8s}] " + "  ".join(
            f"{k}={v[0]:+.3f}" for k, v in rows[s].items() if k != "era"), flush=True)

    print("\n  === MEANS over held-out sessions ===")
    keys = [k for k in rows[TARGETS[0]] if k != "era"]
    summary = {}
    for k in keys:
        vals = [rows[s][k][0] for s in TARGETS if k in rows[s]]
        summary[k] = float(np.mean(vals))
        print(f"    {k:<20s} {summary[k]:+.3f}")
    for N in CAL_SIZES:
        secs = int(N * I20.WIN * E.BIN)
        if f"ft_rep_{secs}s" in summary and f"ft_std_{secs}s" in summary:
            d = summary[f"ft_rep_{secs}s"] - summary[f"ft_std_{secs}s"]
            print(f"    => reptile vs standard @{secs}s: {d:+.3f}")
    rows["_summary"] = summary
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")


def _load(cfg, n_ch, state):
    net = M.build_net({**cfg, "n_out": 2}, n_ch)
    net.load_state_dict(state)
    return net


if __name__ == "__main__":
    main()
