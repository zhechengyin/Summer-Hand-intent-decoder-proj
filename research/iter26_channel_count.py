#!/usr/bin/env python
"""Iteration 26: channel-count sweep in the CURRENT frame (causal + multiscale + 24 sess).

The old count sweep (LOG-042: 8/16/32/96 -> r 0.76/0.80/0.84/0.87) was bidirectional,
6-session, Pearson r. This re-runs it with OUR CURRENT BEST pipeline -- strictly-causal
wide TCN+GRU + multiscale input (raw + EWMA 0.2), 24 sessions -- and reports R^2, so we
know what more peak-detection channels actually buy TODAY.

Channels = top-N by mean firing rate on the base-6 sessions (same rule that picked the
deployed 8). Multiscale expands each to {raw, EWMA} -> N*2 features. N in {8,16,32}.
N=8 reproduces the model of record. Select on EVAL, 3-seed. Usage: py research/iter26_channel_count.py
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
import models.tcn_gru.evaluate as E

SEEDS = (42, 1, 7)
ALPHAS = [1.0, 0.2]                # model-of-record multiscale (raw + EWMA 0.2)
COUNTS = [32]                     # just 32 (8ch model of record ~0.63 is the known baseline)
WIN = I20.WIN


def topN_channels(n):
    fr = np.mean([E.load_electrode(s)[0].mean(1) for s in E.TRAIN], 0)   # base-6 firing
    return np.sort(np.argsort(fr)[-n:])


def prep(chans, axes):
    def wins(s):
        r, v = E.load_electrode(s)
        feat = I20.ewma_feats(r[chans], ALPHAS)                # (N*2, T)
        mu, sd = feat.mean(1, keepdims=True), feat.std(1, keepdims=True) + 1e-6
        fz = ((feat - mu) / sd).astype(np.float32)
        return [{"e": fz[:, k * WIN:(k + 1) * WIN], "vel": v[k * WIN:(k + 1) * WIN][:, axes]}
                for k in range(fz.shape[1] // WIN)]
    tr = [x for s in I20.TRAIN for x in wins(s)]
    return {"train": tr, "eval": {s: wins(s) for s in E.EVAL},
            "test": {s: wins(s) for s in E.TEST}}


def main():
    axes = np.sort(np.argsort(np.mean([E.load_electrode(s)[1].std(0)
                                       for s in E.TRAIN], 0))[-2:])
    print("=== Iteration 26: channel-count sweep, current causal+multiscale frame (3-seed) ===")
    print("    ref: 8ch model of record ~0.63 TEST R2; old bidir r-sweep 8/16/32=0.76/0.80/0.84\n",
          flush=True)
    rows = {}
    for n in COUNTS:
        chans = topN_channels(n)
        t0 = time.time()
        res = H.run(prep(chans, axes), I20.CAUSAL, seeds=SEEDS)
        rows[f"{n}ch"] = {"n_ch": n, "n_feat": n * len(ALPHAS),
                          "eval_r2": res["eval_r2"], "test_r2": res["test_r2"],
                          "int8_kb": res["n_params"] / 1024, "n_params": res["n_params"],
                          "channels": chans.tolist()}
        print(f"  {n:2d} ch ({n*len(ALPHAS):2d} feat)  EVAL R2={res['eval_r2']:.3f}  "
              f"TEST R2={res['test_r2']:.3f}  (~{res['n_params']/1024:.0f} KB int8, "
              f"{res['n_params']:,} p)  [{time.time()-t0:.0f}s]", flush=True)

    if "8ch" in rows:                             # gains vs the 8ch model of record (if run)
        base = rows["8ch"]
        print(f"\n  gains vs 8ch (EVAL {base['eval_r2']:.3f} / TEST {base['test_r2']:.3f}):")
        for n in COUNTS:
            if n != 8:
                r = rows[f"{n}ch"]
                print(f"    {n:2d}ch: {r['eval_r2']-base['eval_r2']:+.3f} EVAL / "
                      f"{r['test_r2']-base['test_r2']:+.3f} TEST")
    else:
        print("\n  (8ch not in this run; compare to the model of record ~0.63 TEST R2)")
    out = ROOT / "results" / "metrics" / "iter26_channel_count.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
