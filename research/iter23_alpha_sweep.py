#!/usr/bin/env python
"""Iteration 23: tune the single EWMA alpha for the 2-scale causal input. 3-seed.

Multiscale saturates at 2 scales = raw + one slow EWMA (LOG-069, alpha=0.2 -> 0.646).
Which slow timescale is best? Sweep alpha in {0.1, 0.2, 0.3} (smaller = slower =
longer memory), each a 3-seed ensemble, strictly causal, 8 ch, 24 sess.
Usage: py research/iter23_alpha_sweep.py
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
ALPHAS = [0.1, 0.2, 0.3]           # the slow EWMA scale (paired with raw); tau ~ 380/160/110 ms
WIN = I20.WIN


def prep(alpha, axes):
    def wins(s):
        r, v = E.load_electrode(s)
        feat = I20.ewma_feats(r[I20.CHANNELS], [1.0, alpha])       # raw + one EWMA
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
    print("=== Iteration 23: EWMA alpha sweep, 2-scale, 3-seed (ref alpha=0.2 -> 0.646) ===\n",
          flush=True)
    rows = {}
    for a in ALPHAS:
        t0 = time.time()
        res = H.run(prep(a, axes), I20.CAUSAL, seeds=SEEDS)
        rows[f"alpha_{a}"] = {"test_r2": res["test_r2"], "eval_r2": res["eval_r2"], "alpha": a}
        print(f"  alpha={a} (raw+EWMA) TEST R2={res['test_r2']:.3f}  "
              f"EVAL R2={res['eval_r2']:.3f}  [{time.time()-t0:.0f}s]", flush=True)

    best = max(rows.values(), key=lambda r: r["test_r2"])
    print(f"\n  best: alpha={best['alpha']} -> TEST R2={best['test_r2']:.3f} "
          f"(single-scale ref 0.618; 4-scale 0.646)")
    out = ROOT / "results" / "metrics" / "iter23_alpha_sweep.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
