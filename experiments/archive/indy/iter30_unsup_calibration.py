#!/usr/bin/env python
"""Iteration 30: UNSUPERVISED calibration -- can we fix drift with NO labels?

Everything that works so far needs labelled calibration: affine (LOG-080/081) and
finetune both require known intended velocity. The user is unsure a labelled
calibration window exists at deployment, so this asks: how much can we recover using
ONLY the new session's neural data (no labels)?

Method: BatchNorm-statistics adaptation (a.k.a. AdaBN / test-time adaptation). Our net has
a BatchNorm1d right after the spatial 1x1 conv. Its running mean/var were estimated on the
24-session pool; on a drifted session those statistics are stale. Recomputing them from the
new session's inputs requires only FORWARD passes -- no labels, no gradients, no optimizer.
Dropout is kept OFF (only BN modules are put in train mode) so stat collection is clean.

Conditions (per fresh session; calibration = first N windows, scored on the fixed last half):
  zero_shot   : pool model                                   (no labels)
  bn_adapt    : + recomputed BN stats from calib INPUTS ONLY (NO LABELS)  <-- the question
  affine      : + per-axis gain/offset                       (needs labels)
  bn+affine   : bn_adapt then affine                         (needs labels)
  finetune    : pool weights adapted                         (needs labels; upper reference)

Usage: py experiments/archive/indy/iter30_unsup_calibration.py
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
import models.tcn_gru.evaluate as E
import models.tcn_gru.best_model as M

SEED = 42
COUNTS = [8, 32]
FRESH = I27.FRESH
POOL = I27.POOL
SIZES = [30, 120]                 # windows -> 60 s, 240 s
WIN_S = I20.WIN * E.BIN
FT_EPOCHS = 25
FT_LR = 3e-4
OUT = ROOT / "results" / "metrics" / "iter30_unsup_calibration.json"


def bn_adapt(net, X, bs=32):
    """Recompute BatchNorm running stats from unlabelled inputs. No labels/gradients."""
    import torch
    import torch.nn as nn
    m = copy.deepcopy(net)
    m.eval()                                        # dropout OFF
    bns = [mod for mod in m.modules() if isinstance(mod, nn.BatchNorm1d)]
    if not bns:
        return m
    for mod in bns:
        mod.reset_running_stats()
        mod.momentum = None                         # cumulative average over all batches
        mod.train()                                 # only BN collects stats
    with torch.no_grad():
        for b in range(0, len(X), bs):
            m(torch.tensor(X[b:b + bs]))
    m.eval()
    return m


def fit_affine(pred, truth):
    g = np.zeros(2); o = np.zeros(2)
    for a in range(2):
        A = np.vstack([pred[:, a], np.ones(len(pred))]).T
        g[a], o[a] = np.linalg.lstsq(A, truth[:, a], rcond=None)[0]
    return g, o


def main():
    import torch
    import torch.nn as nn
    t0 = time.time()
    counts = I27.load_counts_full(POOL + list(E.EVAL) + list(E.TEST) + FRESH)
    axes = np.sort(np.argsort(np.mean([counts[s][1].std(0) for s in E.TRAIN], 0))[-2:])
    print("=== Iteration 30: UNSUPERVISED calibration (BN-stat adaptation, no labels) ===")
    print(f"    loaded {len(counts)} sessions [{time.time()-t0:.0f}s]\n", flush=True)

    rows = {}
    for n in COUNTS:
        chans = I27.topN_channels(counts, n)
        tr_by = I27.build(counts, chans, axes, POOL)
        data = {"train": [x for s in POOL for x in tr_by[s]],
                "eval": I27.build(counts, chans, axes, list(E.EVAL)),
                "test": I27.build(counts, chans, axes, list(E.TEST))}
        t1 = time.time()
        res = H.run(data, I20.CAUSAL, seeds=(SEED,), ret_net=True)
        pool_net, (ym, ys) = res["net"], res["norm"]
        print(f"  [{n}ch] pool model: EVAL {res['eval_r2']:.3f} test1 {res['test_r2']:.3f} "
              f"[{time.time()-t1:.0f}s]", flush=True)

        def predict(net, X):
            net.eval()
            with torch.no_grad():
                return net(torch.tensor(X)).numpy() * ys + ym

        fresh_by = I27.build(counts, chans, axes, FRESH)
        rows[f"{n}ch"] = {}
        for s in FRESH:
            w = fresh_by[s]
            half = len(w) // 2
            calib_pool, test_w = w[:half], w[half:]
            Xt, Yt = I28.stack(test_w); yt = Yt.reshape(-1, 2)
            rows[f"{n}ch"][s] = {}
            rows[f"{n}ch"][s]["zero_shot"] = {
                "secs": 0, "scores": {"zero_shot": I28.score(yt, predict(pool_net, Xt).reshape(-1, 2))}}
            for N in SIZES:
                if N > len(calib_pool):
                    continue
                cw = calib_pool[:N]
                Xc, Yc = I28.stack(cw); yc = Yc.reshape(-1, 2)
                secs = N * WIN_S
                sc = {}
                # --- unsupervised: BN stats from calib INPUTS only ---
                bn = bn_adapt(pool_net, Xc)
                p_bn_t = predict(bn, Xt).reshape(-1, 2)
                sc["bn_adapt"] = I28.score(yt, p_bn_t)
                # --- supervised references ---
                p0_c = predict(pool_net, Xc).reshape(-1, 2)
                p0_t = predict(pool_net, Xt).reshape(-1, 2)
                g, o = fit_affine(p0_c, yc)
                sc["affine"] = I28.score(yt, p0_t * g + o)
                g2, o2 = fit_affine(predict(bn, Xc).reshape(-1, 2), yc)
                sc["bn+affine"] = I28.score(yt, p_bn_t * g2 + o2)
                ft = M.build_net({**I20.CAUSAL, "n_out": 2}, Xc.shape[1])
                ft.load_state_dict(pool_net.state_dict())
                opt = torch.optim.AdamW(ft.parameters(), lr=FT_LR, weight_decay=I20.CAUSAL["wd"])
                mse = nn.MSELoss()
                Xtc = torch.tensor(Xc); Ytc = torch.tensor(((Yc - ym) / ys).astype(np.float32))
                for ep in range(FT_EPOCHS):
                    ft.train()
                    idx = np.random.permutation(len(Xtc))
                    for b in range(0, len(idx), I20.CAUSAL["bs"]):
                        bi = idx[b:b + I20.CAUSAL["bs"]]
                        opt.zero_grad(); mse(ft(Xtc[bi]), Ytc[bi]).backward(); opt.step()
                sc["finetune"] = I28.score(yt, predict(ft, Xt).reshape(-1, 2))
                rows[f"{n}ch"][s][f"{int(secs)}s"] = {"secs": secs, "scores": sc}
            print(f"    {s}:")
            zs = rows[f"{n}ch"][s]["zero_shot"]["scores"]["zero_shot"]
            print(f"       zero_shot        R2={zs[0]:+.3f}/r={zs[1]:.3f}")
            for k, v in rows[f"{n}ch"][s].items():
                if k == "zero_shot":
                    continue
                for c, sv in v["scores"].items():
                    print(f"       {k:>5s} {c:<10s} R2={sv[0]:+.3f}/r={sv[1]:.3f}")
            print(flush=True)

        # means
        rows[f"{n}ch"]["mean"] = {}
        zs_m = float(np.mean([rows[f"{n}ch"][s]["zero_shot"]["scores"]["zero_shot"][0] for s in FRESH]))
        rows[f"{n}ch"]["mean"]["zero_shot"] = zs_m
        print(f"  [{n}ch] MEAN zero_shot R2={zs_m:+.3f}")
        for N in SIZES:
            key = f"{int(N*WIN_S)}s"
            got = [s for s in FRESH if key in rows[f"{n}ch"][s]]
            if len(got) < len(FRESH):
                print(f"     {key}: only {len(got)}/{len(FRESH)} sessions -- skipped")
                continue
            line = {}
            for c in ["bn_adapt", "affine", "bn+affine", "finetune"]:
                line[c] = float(np.mean([rows[f"{n}ch"][s][key]["scores"][c][0] for s in got]))
            rows[f"{n}ch"]["mean"][key] = line
            print(f"     {key}: " + "  ".join(f"{c}={v:+.3f}" for c, v in line.items()))
        print(flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
