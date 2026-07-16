#!/usr/bin/env python
"""Iteration 15: causal-with-bounded-lookahead R2-vs-latency tradeoff (8 ch, 24 sess).

Bidirectional is NOT deployable (needs the whole future). This sweeps the amount
of FUTURE context a causal TCN+GRU is allowed, so we can pick a latency:
  causal_0     : strictly causal, 0 ms latency (real-time)   -- expect ~0.606
  look2_80ms   : 2 future frames  = 80 ms latency
  look5_200ms  : 5 future frames  = 200 ms latency
  bidir_ceil   : full bidirectional (NON-deployable upper bound) ~0.677
Same wide config (F64/H64) as iter14's tcngru_causal for comparability.
Usage: py experiments/archive/indy/iter15_lookahead.py
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
import experiments.archive.indy.iter5_scale as I5
import experiments.archive.indy.iter7_final as I7
import experiments.common.architectures as A
import models.tcn_gru.best_model as M
import models.tcn_gru.evaluate as E

WIDE = {**E.CFG, "F": 64, "H": 64, "L": 1, "dils": [1, 2, 4, 8]}
BIN_MS = E.BIN * 1000                                          # 40 ms/frame

# (name, build_fn, cfg_overrides, latency_ms or None for non-deployable)
RUNS = [
    ("causal_0",    A.build_lookahead_tcngru(0), {"bidir": False}, 0),
    ("look2_80ms",  A.build_lookahead_tcngru(2), {"bidir": False}, 80),
    ("look5_200ms", A.build_lookahead_tcngru(5), {"bidir": False}, 200),
    ("bidir_ceil",  M.build_net,                 {"bidir": True},  None),
]


def main():
    data = I5.prep_more(I7.EXTRA18[:18], reselect=False)      # 24 sessions, fixed 8 ch
    print("=== Iteration 15: lookahead R2 vs latency (8 ch, 24 sess, wide) ===\n",
          flush=True)
    rows = {}
    for name, build_fn, ov, lat in RUNS:
        t0 = time.time()
        res = H.run(data, {**WIDE, **ov}, build=build_fn)
        rows[name] = {"test_r2": res["test_r2"], "eval_r2": res["eval_r2"],
                      "latency_ms": lat, "deployable": lat is not None}
        tag = f"{lat} ms" if lat is not None else "NON-deployable (full future)"
        print(f"  {name:12s} lat={tag:28s} TEST R2={res['test_r2']:.3f}  "
              f"EVAL R2={res['eval_r2']:.3f}  [{time.time()-t0:.0f}s]", flush=True)

    print("\n--- R2 vs latency (deployable = causal + bounded lookahead) ---")
    for name, r in rows.items():
        lat = f"{r['latency_ms']} ms" if r["deployable"] else "inf (bidir, N/A)"
        print(f"  {lat:16s}: R2={r['test_r2']:.3f}")
    out = ROOT / "results" / "metrics" / "iter15_lookahead.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
