#!/usr/bin/env python
"""Iteration 17: improve the STRICTLY-CAUSAL decoder (0 lookahead). Ref = 0.606.

Bidirectional and bounded-lookahead are out (40 ms/bin -> latency too high). All
models here are strictly causal (build_net bidir=False: causal TCN padding +
unidirectional GRU). Causal-specific levers:
  more_rf   : longer dilations = more PAST context (no future to lean on).
  two_layer : deeper GRU (more temporal state).
  bigger_F  : more width/capacity.
  + causal forward-EMA output smoothing (real-time Kalman-lite), swept on the ref.
8 ch, 24 sessions, fixed channels. Usage: py research/iter17_causal_improve.py
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
import research.iter5_scale as I5
import research.iter7_final as I7
import models.tcn_gru.best_model as M
import models.tcn_gru.evaluate as E

CB = {**E.CFG, "F": 64, "H": 64, "L": 1, "dils": [1, 2, 4, 8], "bidir": False}  # causal
VARIANTS = {
    "causal_ref": CB,
    "more_rf":    {**CB, "dils": [1, 2, 4, 8, 16, 32]},        # ~more past context
    "two_layer":  {**CB, "L": 2},
    "bigger_F96": {**CB, "F": 96, "H": 96},
}


def ema(alpha):                                                # causal forward EMA (real-time)
    def f(p):
        out = p.copy()
        for i in range(1, p.shape[1]):
            out[:, i] = alpha * p[:, i] + (1 - alpha) * out[:, i - 1]
        return out
    return f


def score_preds(preds, post):
    r2 = []
    for _, (Y, P) in preds.items():
        yh = post(P).reshape(-1, Y.shape[-1])
        r2.append(M.r2(Y.reshape(-1, Y.shape[-1]), yh))
    return float(np.mean(r2))


def main():
    data = I5.prep_more(I7.EXTRA18[:18], reselect=False)      # 24 sessions
    print("=== Iteration 17: improve strictly-causal decoder (ref 0.606) ===\n", flush=True)
    rows = {}
    for name, cfg in VARIANTS.items():
        t0 = time.time()
        rp = name == "causal_ref"
        res = H.run(data, cfg, ret_preds=rp)
        rows[name] = {"test_r2": res["test_r2"], "eval_r2": res["eval_r2"],
                      "int8_kb": res["n_params"] / 1024}
        print(f"  {name:11s} TEST R2={res['test_r2']:.3f}  EVAL R2={res['eval_r2']:.3f}  "
              f"(~{res['n_params']/1024:.0f} KB int8)  [{time.time()-t0:.0f}s]", flush=True)
        if rp:                                                # EMA post sweep on the ref
            for a in (0.4, 0.6, 0.8):
                te = score_preds(res["test_preds"], ema(a))
                ev = score_preds(res["eval_preds"], ema(a))
                rows[f"ref_ema{a}"] = {"test_r2": te, "eval_r2": ev}
                print(f"    + causal EMA(a={a}): TEST R2={te:.3f}  EVAL R2={ev:.3f}", flush=True)

    best = max(rows.values(), key=lambda r: r["test_r2"])
    print(f"\n--- causal improvements (ref 0.606); best TEST R2 = {best['test_r2']:.3f} ---")
    for name, r in rows.items():
        print(f"  {name:12s}: {r['test_r2']:.3f}")
    out = ROOT / "results" / "metrics" / "iter17_causal_improve.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
