#!/usr/bin/env python
"""Iteration 31: per-session CHANNEL RE-SELECTION during calibration (uses the hardware's rescan).

Motivation: LOG-079 showed the fixed 8 channels (firing-selected on Oct-2016 base-6) COLLAPSE
on a drifted session -- plausibly because those specific electrodes went dead/changed. The
hardware can observe all 96 and re-route which N are peak-detected (user, 2026-07-14). And
LOG-080/082 established a labelled calibration block is REQUIRED anyway.

That combination dissolves the old objection (LOG-047: "arbitrary electrodes cannot be fed
into fixed input slots"): if we are adapting the model on calibration data regardless, we can
re-select the channels at the same time and let the model learn that new channel set.

Per fresh session, chronological 50/50: calibrate on the 1st half, score the 2nd.
  fixed_ft      : pool channels (base-6 firing) + finetune pool model     (= the LOG-080 result)
  fixed_scratch : pool channels + train from scratch on calib
  resel_scratch : channels RE-SELECTED by firing rate on the calibration half + scratch  <-- the test
  resel_ft      : re-selected channels + finetune pool model (input identities mismatched at
                  start -- included to show whether that mismatch hurts)

Key comparison: resel_scratch vs fixed_scratch (same training, only the channel set differs).
Also reports channel overlap with the pool set. Usage: py research/iter31_channel_reselect.py
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

import research.harness as H
import research.iter20_multiscale as I20
import research.iter27_fresh_session as I27
import research.iter28_calibration as I28
import models.tcn_gru.evaluate as E
import models.tcn_gru.best_model as M

SEED = 42
COUNTS = [8, 32]
FRESH = I27.FRESH
POOL = I27.POOL
FT_EPOCHS = 25
FT_LR = 3e-4
OUT = ROOT / "results" / "metrics" / "iter31_channel_reselect.json"


def build_one(counts, s, chans, axes):
    """Windows for one session with a given channel set (honest causal input)."""
    f = I27.feats(counts[s][0], chans)
    mu, sd = f.mean(1, keepdims=True), f.std(1, keepdims=True) + 1e-6
    fz = ((f - mu) / sd).astype(np.float32)
    v = counts[s][1]
    W = I20.WIN
    return [{"e": fz[:, k * W:(k + 1) * W], "vel": v[k * W:(k + 1) * W][:, axes]}
            for k in range(fz.shape[1] // W)]


def train(Xc, Yc, cfg, epochs, lr, init=None, norm=None):
    """Train (scratch or finetune-from-init) on calibration data. Returns (net, ym, ys)."""
    import torch
    import torch.nn as nn
    if norm is None:
        ym, ys = Yc.mean((0, 1)), Yc.std((0, 1)) + 1e-6
    else:
        ym, ys = norm
    net = M.build_net({**cfg, "n_out": 2}, Xc.shape[1])
    if init is not None:
        net.load_state_dict(init)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=cfg["wd"])
    mse = nn.MSELoss()
    Xt = torch.tensor(Xc); Yt = torch.tensor(((Yc - ym) / ys).astype(np.float32))
    for ep in range(epochs):
        net.train()
        idx = np.random.permutation(len(Xt))
        for b in range(0, len(idx), cfg["bs"]):
            bi = idx[b:b + cfg["bs"]]
            opt.zero_grad(); mse(net(Xt[bi]), Yt[bi]).backward(); opt.step()
    return net, ym, ys


def main():
    import torch
    t0 = time.time()
    counts = I27.load_counts_full(POOL + list(E.EVAL) + list(E.TEST) + FRESH)
    axes = np.sort(np.argsort(np.mean([counts[s][1].std(0) for s in E.TRAIN], 0))[-2:])
    print("=== Iteration 31: per-session channel RE-SELECTION during calibration ===")
    print(f"    loaded {len(counts)} sessions [{time.time()-t0:.0f}s]\n", flush=True)

    def predict(net, X, ym, ys):
        net.eval()
        with torch.no_grad():
            return net(torch.tensor(X)).numpy() * ys + ym

    rows = {}
    for n in COUNTS:
        pool_ch = I27.topN_channels(counts, n)
        tr_by = I27.build(counts, pool_ch, axes, POOL)
        data = {"train": [x for s in POOL for x in tr_by[s]],
                "eval": I27.build(counts, pool_ch, axes, list(E.EVAL)),
                "test": I27.build(counts, pool_ch, axes, list(E.TEST))}
        t1 = time.time()
        res = H.run(data, I20.CAUSAL, seeds=(SEED,), ret_net=True)
        pool_net, (pym, pys) = res["net"], res["norm"]
        pool_state = {k: v.clone() for k, v in pool_net.state_dict().items()}
        print(f"  [{n}ch] pool model: EVAL {res['eval_r2']:.3f} test1 {res['test_r2']:.3f} "
              f"[{time.time()-t1:.0f}s]", flush=True)

        rows[f"{n}ch"] = {}
        for s in FRESH:
            # --- fixed-channel windows ---
            wf = build_one(counts, s, pool_ch, axes)
            half = len(wf) // 2
            Xc_f, Yc_f = I28.stack(wf[:half]); Xt_f, Yt_f = I28.stack(wf[half:])
            yt = Yt_f.reshape(-1, 2)

            # --- re-select channels using ONLY the calibration half of this session ---
            T_half = (counts[s][0].shape[1] // I20.WIN) // 2 * I20.WIN
            fr_calib = counts[s][0][:, :T_half].mean(1)
            resel_ch = np.sort(np.argsort(fr_calib)[-n:])
            overlap = len(set(resel_ch.tolist()) & set(pool_ch.tolist()))
            wr = build_one(counts, s, resel_ch, axes)
            Xc_r, Yc_r = I28.stack(wr[:half]); Xt_r, _ = I28.stack(wr[half:])

            out = {}
            # fixed + finetune (LOG-080 condition)
            net, ym, ys = train(Xc_f, Yc_f, I20.CAUSAL, FT_EPOCHS, FT_LR,
                                init=pool_state, norm=(pym, pys))
            out["fixed_ft"] = I28.score(yt, predict(net, Xt_f, ym, ys).reshape(-1, 2))
            # fixed + scratch
            net, ym, ys = train(Xc_f, Yc_f, I20.CAUSAL, I20.CAUSAL["epochs"], I20.CAUSAL["lr"])
            out["fixed_scratch"] = I28.score(yt, predict(net, Xt_f, ym, ys).reshape(-1, 2))
            # re-selected + scratch  <-- the test
            net, ym, ys = train(Xc_r, Yc_r, I20.CAUSAL, I20.CAUSAL["epochs"], I20.CAUSAL["lr"])
            out["resel_scratch"] = I28.score(yt, predict(net, Xt_r, ym, ys).reshape(-1, 2))
            # re-selected + finetune (mismatched input identities at init)
            net, ym, ys = train(Xc_r, Yc_r, I20.CAUSAL, FT_EPOCHS, FT_LR,
                                init=pool_state, norm=(pym, pys))
            out["resel_ft"] = I28.score(yt, predict(net, Xt_r, ym, ys).reshape(-1, 2))

            rows[f"{n}ch"][s] = {"overlap_with_pool": overlap, "n": n,
                                 "resel_channels": resel_ch.tolist(),
                                 **{k: {"r2": v[0], "r": v[1]} for k, v in out.items()}}
            print(f"    {s}  (re-sel overlap {overlap}/{n} with pool set)")
            for k, v in out.items():
                print(f"       {k:<14s} R2={v[0]:+.3f}  r={v[1]:.3f}")
            print(flush=True)

        rows[f"{n}ch"]["mean"] = {}
        print(f"  [{n}ch] MEAN over fresh:")
        for c in ["fixed_ft", "fixed_scratch", "resel_scratch", "resel_ft"]:
            m = float(np.mean([rows[f"{n}ch"][s][c]["r2"] for s in FRESH]))
            rows[f"{n}ch"]["mean"][c] = m
            print(f"     {c:<14s} R2={m:+.3f}")
        d = rows[f"{n}ch"]["mean"]["resel_scratch"] - rows[f"{n}ch"]["mean"]["fixed_scratch"]
        ov = float(np.mean([rows[f"{n}ch"][s]["overlap_with_pool"] for s in FRESH]))
        print(f"     => re-selection vs fixed (same training): {d:+.3f}  "
              f"(mean overlap {ov:.1f}/{n})\n", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
