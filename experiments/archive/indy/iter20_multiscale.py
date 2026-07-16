#!/usr/bin/env python
"""Iteration 20: multi-timescale causal INPUT features (8 ch, 24 sess). Ref = 0.606.

Instead of feeding one 40 ms rate per channel, feed each of the 8 channels at
several causal timescales -- raw + EWMA at alpha 0.5/0.25/0.1 (~fast..slow) -> 32
input features. Gives the causal model explicit long+short history at the input
(distinct from the dilation/receptive-field approach, which didn't help). Same
strictly-causal TCN+GRU. Usage: py experiments/archive/indy/iter20_multiscale.py
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

import experiments.common.harness as H
import experiments.archive.indy.iter7_final as I7
import models.tcn_gru.evaluate as E

CHANNELS = np.array([26, 51, 53, 66, 71, 73, 75, 94])         # deployed firing-8
ALPHAS = [1.0, 0.5, 0.25, 0.1]                                # raw + 3 EWMA scales
CAUSAL = {**E.CFG, "F": 64, "H": 64, "L": 1, "dils": [1, 2, 4, 8], "bidir": False}
TRAIN = list(E.TRAIN) + I7.EXTRA18                            # 24 sessions
WIN = int(round(2.0 / E.BIN))


def ewma_feats(rates, alphas):
    feats = []
    for a in alphas:
        if a >= 1.0:
            feats.append(rates); continue
        o = rates.copy()
        for t in range(1, rates.shape[1]):
            o[:, t] = a * rates[:, t] + (1 - a) * o[:, t - 1]
        feats.append(o)
    return np.concatenate(feats, 0)                           # (8*len(alphas), T)


def make_prep(multiscale, axes):
    def wins(s):
        r, v = E.load_electrode(s)
        feat = ewma_feats(r[CHANNELS], ALPHAS) if multiscale else r[CHANNELS]
        mu, sd = feat.mean(1, keepdims=True), feat.std(1, keepdims=True) + 1e-6
        fz = ((feat - mu) / sd).astype(np.float32)
        return [{"e": fz[:, k * WIN:(k + 1) * WIN], "vel": v[k * WIN:(k + 1) * WIN][:, axes]}
                for k in range(fz.shape[1] // WIN)]
    tr = [x for s in TRAIN for x in wins(s)]
    return {"train": tr, "eval": {s: wins(s) for s in E.EVAL},
            "test": {s: wins(s) for s in E.TEST}}


def main():
    axes = np.sort(np.argsort(np.mean([E.load_electrode(s)[1].std(0)
                                       for s in E.TRAIN], 0))[-2:])
    print("=== Iteration 20: multi-timescale causal input (ref single-scale 0.606) ===\n",
          flush=True)
    rows = {}
    for label, ms in [("single_8ch", False), ("multiscale_32ch", True)]:
        t0 = time.time()
        res = H.run(make_prep(ms, axes), CAUSAL)
        rows[label] = {"test_r2": res["test_r2"], "eval_r2": res["eval_r2"],
                       "int8_kb": res["n_params"] / 1024}
        print(f"  {label:16s} TEST R2={res['test_r2']:.3f}  EVAL R2={res['eval_r2']:.3f}  "
              f"(~{res['n_params']/1024:.0f} KB int8)  [{time.time()-t0:.0f}s]", flush=True)

    d = rows["multiscale_32ch"]["test_r2"] - rows["single_8ch"]["test_r2"]
    print(f"\n  multiscale vs single: {d:+.3f}")
    out = ROOT / "results" / "metrics" / "iter20_multiscale.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
