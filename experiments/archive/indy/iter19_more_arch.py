#!/usr/bin/env python
"""Iteration 19: untested architectures, STRICTLY CAUSAL (8 ch, 24 sess). Ref = 0.606.

Rules out (or in) the architectures we hadn't tried:
  wiener        : linear ridge on 10 lagged 40 ms bins (classic BCI decoder floor).
  tcn_gru       : current model of record (causal) -- reference 0.606.
  tcn_lstm      : LSTM instead of GRU.
  lstm_only     : LSTM, no CNN front-end.
  plain_cnn_gru : NON-dilated causal conv stack + GRU (do the dilations matter?).
  transformer   : causal Transformer encoder (attention).
Report TEST R2 and ~int8 size. Usage: py experiments/archive/indy/iter19_more_arch.py
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
ARCHS = {
    "tcn_gru":       M.build_net,          # ref 0.606
    "tcn_lstm":      A.build_tcn_lstm,
    "lstm_only":     A.build_lstm_only,
    "plain_cnn_gru": A.build_plain_cnn_gru,
    "transformer":   A.build_transformer,
}


def wiener_r2(data, nlag=10, alpha=1e2):
    """Causal linear ridge on nlag+1 lagged bins of the 8 channels (BCI floor)."""
    def design(trials):
        X, Y = [], []
        for t in trials:
            e, v = t["e"], t["vel"]
            for i in range(nlag, e.shape[1]):
                X.append(e[:, i - nlag:i + 1].reshape(-1))
                Y.append(v[i])
        return np.asarray(X, np.float64), np.asarray(Y, np.float64)
    Xtr, Ytr = design(data["train"])
    d = Xtr.shape[1]
    W = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(d), Xtr.T @ Ytr)
    r2 = [M.r2(design(tr)[1], design(tr)[0] @ W) for tr in data["test"].values()]
    return float(np.mean(r2)), d


def main():
    data = I5.prep_more(I7.EXTRA18[:18], reselect=False)          # 24 sessions, 8 ch
    print("=== Iteration 19: more architectures, strictly causal (ref 0.606) ===\n",
          flush=True)
    rows = {}

    t0 = time.time()
    wr2, wd = wiener_r2(data)
    rows["wiener"] = {"test_r2": wr2, "int8_kb": wd * 2 / 1024, "causal": True}
    print(f"  {'wiener':13s} TEST R2={wr2:.3f}  (~{wd*2/1024:.1f} KB, linear ridge)  "
          f"[{time.time()-t0:.0f}s]", flush=True)

    for name, build in ARCHS.items():
        t0 = time.time()
        res = H.run(data, {**WIDE, "bidir": False}, build=build)
        rows[name] = {"test_r2": res["test_r2"], "eval_r2": res["eval_r2"],
                      "int8_kb": res["n_params"] / 1024, "causal": True}
        print(f"  {name:13s} TEST R2={res['test_r2']:.3f}  EVAL R2={res['eval_r2']:.3f}  "
              f"(~{res['n_params']/1024:.0f} KB int8)  [{time.time()-t0:.0f}s]", flush=True)

    print("\n--- strictly-causal architectures (ranked by EVAL; TCN+GRU ref = 0.606) ---")
    for name, r in sorted(rows.items(), key=lambda kv: -kv[1]["eval_r2"]):   # rank on EVAL, not TEST
        print(f"  {name:13s}: EVAL {r['eval_r2']:.3f}  TEST {r['test_r2']:.3f}  (~{r['int8_kb']:.0f} KB)")
    out = ROOT / "results" / "metrics" / "iter19_more_arch.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
