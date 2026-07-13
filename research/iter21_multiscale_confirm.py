#!/usr/bin/env python
"""Iteration 21: is the multiscale +0.006 real or noise? 3-seed A/B (causal, 24 sess).

iter20 gave single 0.606 vs multiscale 0.611 (+0.006 test, +0.015 eval) -- inside
the run-noise band. Re-run both as 3-seed ensembles (variance-reduced) to see if
the multiscale lead survives. Usage: py research/iter21_multiscale_confirm.py
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


def main():
    axes = np.sort(np.argsort(np.mean([E.load_electrode(s)[1].std(0)
                                       for s in E.TRAIN], 0))[-2:])
    print("=== Iteration 21: multiscale confirmation, 3-seed A/B (causal) ===\n", flush=True)
    rows = {}
    for label, ms in [("single_8ch", False), ("multiscale_32ch", True)]:
        t0 = time.time()
        res = H.run(I20.make_prep(ms, axes), I20.CAUSAL, seeds=SEEDS)
        rows[label] = {"test_r2": res["test_r2"], "eval_r2": res["eval_r2"]}
        print(f"  {label:16s} (3-seed) TEST R2={res['test_r2']:.3f}  "
              f"EVAL R2={res['eval_r2']:.3f}  [{time.time()-t0:.0f}s]", flush=True)

    d = rows["multiscale_32ch"]["test_r2"] - rows["single_8ch"]["test_r2"]
    verdict = "REAL (survives ensembling)" if d >= 0.008 else \
              "within noise (not adopting)" if abs(d) < 0.008 else "single wins"
    print(f"\n  multiscale - single (3-seed) = {d:+.3f} -> {verdict}")
    out = ROOT / "results" / "metrics" / "iter21_multiscale_confirm.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
