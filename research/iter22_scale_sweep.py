#!/usr/bin/env python
"""Iteration 22: how many timescales? scale-count sweep (3-seed, causal, 24 sess).

iter21 (3-seed): 1 scale (single) = 0.618, 4 scales = 0.646 (+0.028). Does adding
slower EWMA scales (more past history) keep helping? Test 2 and 6 scales here;
compare to the known 1 and 4. alpha smaller = slower = longer effective memory.
  ms2 : raw + alpha 0.2                          (16 features, ~160 ms memory)
  ms6 : raw + 0.5/0.25/0.1/0.05/0.025            (48 features, ~1.5 s memory)
Usage: py research/iter22_scale_sweep.py
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
CONFIGS = {
    "ms2_16ch": [1.0, 0.2],
    "ms6_48ch": [1.0, 0.5, 0.25, 0.1, 0.05, 0.025],
}
WIN = I20.WIN


def prep(alphas, axes):
    def wins(s):
        r, v = E.load_electrode(s)
        feat = I20.ewma_feats(r[I20.CHANNELS], alphas)
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
    print("=== Iteration 22: scale-count sweep, 3-seed (known: 1sc=0.618, 4sc=0.646) ===\n",
          flush=True)
    rows = {}
    for name, alphas in CONFIGS.items():
        t0 = time.time()
        res = H.run(prep(alphas, axes), I20.CAUSAL, seeds=SEEDS)
        rows[name] = {"test_r2": res["test_r2"], "eval_r2": res["eval_r2"],
                      "n_scales": len(alphas)}
        print(f"  {name:10s} ({len(alphas)} scales) TEST R2={res['test_r2']:.3f}  "
              f"EVAL R2={res['eval_r2']:.3f}  [{time.time()-t0:.0f}s]", flush=True)

    print("\n--- scale count vs R2 (3-seed) ---")
    print("  1 scale : 0.618 | 4 scales : 0.646 (from iter21)")
    for name, r in rows.items():
        print(f"  {r['n_scales']} scales : {r['test_r2']:.3f}  ({name})")
    out = ROOT / "results" / "metrics" / "iter22_scale_sweep.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
