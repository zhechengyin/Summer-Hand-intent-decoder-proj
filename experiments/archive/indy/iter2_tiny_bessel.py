#!/usr/bin/env python
"""Iteration 2: STM32-sized 8-channel decoder + Bessel output filter.

New hardware constraints (user + Zhou/Sun/Basu 2023, arXiv:2312.15889):
  * 8 channels, spike detection (threshold crossings) -- already our input.
  * Must fit an STM32 -> target tens of kB, not our 0.77 MB model of record.
  * Metric is R^2 (avg of X,Y velocity).

Two levers from the paper:
  A) Shrink the TCN+GRU (F,H,L,dilations) toward the paper's 25-135 kB regime.
  B) Add a Bessel low-pass on the OUTPUT velocity (their +0.03-0.05 R^2 trick):
     forward = real-time; zero-phase = their block-bidirectional gold standard.
Filter cutoff is chosen on EVAL and reported on TEST (no test cherry-picking).

Baseline (this split, 8 ch, 0.75 MB): TEST R^2 = 0.548.
Usage: py experiments/archive/indy/iter2_tiny_bessel.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import bessel, sosfilt, sosfiltfilt

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.common.harness as H
import models.tcn_gru.best_model as M
import models.tcn_gru.evaluate as E

FS = 1.0 / E.BIN                      # 25 Hz output rate (40 ms bins)

# STM32-oriented shrink sweep (base of record: F64 H64 L2 dils[1,2,4,8,16] = 187k)
ARCHS = {
    "small":        dict(F=32, H=32, L=1, dils=[1, 2, 4, 8]),
    "tiny":         dict(F=24, H=24, L=1, dils=[1, 2, 4, 8]),
    "micro":        dict(F=16, H=16, L=1, dils=[1, 2, 4]),
    "nano":         dict(F=12, H=12, L=1, dils=[1, 2, 4]),
    "small_causal": dict(F=32, H=32, L=1, dils=[1, 2, 4, 8], bidir=False),  # real-time
}


def bessel_post(cut, mode, order):
    sos = bessel(order, cut / (FS / 2), btype="low", norm="phase", output="sos")
    def f(p):                                     # p: (n, T, D), filter along T
        if mode == "fwd":
            return sosfilt(sos, p, axis=1)
        return sosfiltfilt(sos, p, axis=1, padlen=6)
    return f


FILTERS = {"none": (lambda p: p)}
for c in (2.0, 3.0, 4.0, 5.0):
    FILTERS[f"fwd{c:.0f}"] = bessel_post(c, "fwd", 2)     # real-time, order 2
    FILTERS[f"zp{c:.0f}"] = bessel_post(c, "zp", 4)       # zero-phase, order 4


def score(preds, post):
    r2 = []
    for name, (Y, P) in preds.items():
        yh = post(P).reshape(-1, Y.shape[-1]); y = Y.reshape(-1, Y.shape[-1])
        r2.append(M.r2(y, yh))
    return float(np.mean(r2))


def main():
    data = H.prep(nch=8)
    print("=== Iteration 2: tiny 8-ch decoder + Bessel filter ===")
    print(f"(baseline 8ch 0.75MB TEST R2=0.548; output rate {FS:.0f} Hz)\n", flush=True)
    rows = {}
    for name, arch in ARCHS.items():
        cfg = {**E.CFG, **arch}
        t0 = time.time()
        res = H.run(data, cfg, ret_preds=True)
        # pick filter on EVAL, report on TEST
        eval_scores = {fn: score(res["eval_preds"], f) for fn, f in FILTERS.items()}
        best_f = max(eval_scores, key=eval_scores.get)
        test_none = score(res["test_preds"], FILTERS["none"])
        test_best = score(res["test_preds"], FILTERS[best_f])
        rows[name] = {"kb": res["kb"], "params": res["n_params"],
                      "test_r2_nofilt": test_none, "best_filter": best_f,
                      "test_r2_filt": test_best,
                      "eval_r2_nofilt": eval_scores["none"],
                      "eval_r2_filt": eval_scores[best_f]}
        print(f"  {name:13s} {res['kb']:6.1f} kB ({res['n_params']:>6,}p)  "
              f"TEST R2: {test_none:.3f} -> {test_best:.3f} "
              f"[{best_f}]  (+{test_best-test_none:.3f})  [{time.time()-t0:.0f}s]",
              flush=True)

    print("\n--- R2 vs size (TEST, filter picked on eval) ---")
    print(f"  {'model':13s} {'kB':>7s} {'no-filt':>8s} {'+bessel':>8s}")
    for name, r in rows.items():
        print(f"  {name:13s} {r['kb']:7.1f} {r['test_r2_nofilt']:8.3f} "
              f"{r['test_r2_filt']:8.3f}")
    print(f"\n  reference: base 0.75MB(187k) = 0.548 no-filter")
    out = ROOT / "results" / "metrics" / "iter2_tiny_bessel.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
