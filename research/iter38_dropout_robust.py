#!/usr/bin/env python
"""Iteration 38: train FOR the failure mode -- high channel dropout as drift armour.

THE OBSERVED FAILURE: drift churns the channels. LOG-083 measured it -- only ~2.5 of the 8
pool-selected channels (and ~15/32) are still among the best months later. The model leans on
specific electrodes; when they change, it collapses (LOG-079/086: ~12% of sessions fail).

THE MOVE: match the augmentation to the perturbation. We train with chdrop=0.1 -- a default nobody
ever questioned, chosen long before drift was understood. If training FORCED the model to decode
while half its channels are missing, it would have to spread the code redundantly across the
population instead of relying on individual electrodes -- which is precisely what churn destroys.

THEORY: dropout as implicit ensembling / redundant coding (Srivastava 2014); and specifically,
augmenting with the deployment perturbation is the standard recipe for robustness. Channel churn
IS structured channel dropout, so train on it.

Sweep chdrop in {0.1 (current), 0.3, 0.5, 0.7}, 32ch, and measure ZERO-SHOT on 8 out-of-pool
sessions -- with the 3-4 known BAD (drifted) sessions reported separately, since that is where
robustness should show up. eval1 is reported too, to catch the obvious failure mode: heavy dropout
simply making a worse decoder.

Usage: py research/iter38_dropout_robust.py
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
import research.iter32_forward_split as I32
import models.tcn_gru.evaluate as E

SEED = 42
N_CH = 32
DROPS = [0.1, 0.3, 0.5, 0.7]
POOL = I32.POOL
TARGETS = I27.FRESH + I32.FORWARD
OUT = ROOT / "results" / "metrics" / "iter38_dropout_robust.json"


def main():
    import torch
    t0 = time.time()
    counts = I27.load_counts_full(POOL + list(E.EVAL) + list(E.TEST) + TARGETS)
    axes = np.sort(np.argsort(np.mean([counts[s][1].std(0) for s in E.TRAIN], 0))[-2:])
    fr = np.mean([counts[s][0].mean(1) for s in POOL], 0)
    chans = np.sort(np.argsort(fr)[-N_CH:])
    print("=== Iteration 38: channel-dropout sweep as drift armour ===")
    print(f"    {N_CH} ch; pool {len(POOL)} sessions; zero-shot on {len(TARGETS)} out-of-pool")
    print(f"    loaded {len(counts)} sessions [{time.time()-t0:.0f}s]\n", flush=True)

    tr_by = I27.build(counts, chans, axes, POOL)
    fresh_by = I27.build(counts, chans, axes, TARGETS)
    train_trials = [x for s in POOL for x in tr_by[s]]
    ev = I27.build(counts, chans, axes, list(E.EVAL))
    te = I27.build(counts, chans, axes, list(E.TEST))

    rows = {}
    for cd in DROPS:
        cfg = {**I20.CAUSAL, "chdrop": cd}
        t1 = time.time()
        res = H.run({"train": train_trials, "eval": ev, "test": te},
                    cfg, seeds=(SEED,), ret_net=True)
        net, (ym, ys) = res["net"], res["norm"]

        def predict(X):
            net.eval()
            with torch.no_grad():
                return net(torch.tensor(X)).numpy() * ys + ym

        per = {}
        for s in TARGETS:
            w = fresh_by[s]
            half = len(w) // 2
            Xt, Yt = I28.stack(w[half:])
            per[s] = I28.score(Yt.reshape(-1, 2), predict(Xt).reshape(-1, 2))
        rows[f"chdrop_{cd}"] = {"eval_r2": res["eval_r2"], "test1_r2": res["test_r2"],
                                "sessions": {s: {"r2": v[0], "r": v[1]} for s, v in per.items()},
                                "mean_zero_shot": float(np.mean([v[0] for v in per.values()]))}
        print(f"  chdrop={cd}: EVAL {res['eval_r2']:.3f} | test1 {res['test_r2']:.3f} | "
              f"zero-shot mean {rows[f'chdrop_{cd}']['mean_zero_shot']:+.3f} "
              f"[{time.time()-t1:.0f}s]", flush=True)
        for s in TARGETS:
            print(f"       {s:20s} {per[s][0]:+.3f}", flush=True)

    # the point of the experiment: does dropout armour the BAD sessions?
    base = rows[f"chdrop_{DROPS[0]}"]
    bad = [s for s in TARGETS if base["sessions"][s]["r2"] < 0.4]
    print(f"\n  === bad sessions at baseline chdrop={DROPS[0]}: {bad} ===")
    for cd in DROPS:
        r = rows[f"chdrop_{cd}"]
        mb = float(np.mean([r["sessions"][s]["r2"] for s in bad])) if bad else float("nan")
        mg = float(np.mean([r["sessions"][s]["r2"] for s in TARGETS if s not in bad]))
        print(f"    chdrop={cd}: BAD {mb:+.3f}  |  good {mg:+.3f}  |  all "
              f"{r['mean_zero_shot']:+.3f}  |  eval {r['eval_r2']:.3f}")
    rows["_bad"] = bad
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
