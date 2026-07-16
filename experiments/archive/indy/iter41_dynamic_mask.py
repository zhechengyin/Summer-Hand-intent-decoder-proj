#!/usr/bin/env python
"""Iteration 41: DYNAMIC mid-session channel selection -- does re-masking WITHIN a session help?

iter39/iter40 tested a per-session FIXED mask (choose 32 electrodes once, hold for the whole
session). This tests a mask that CHANGES within a session, using the hardware's rescan ability, to
track drift that happens during use. Same frozen 96-slot model (random-mask trained -- it handles
arbitrary subsets best, per iter39 salvage); only the MASK schedule differs. No retraining anywhere.

All strategies are CAUSAL (mask at time t uses only counts up to t) and scored on the SECOND half
of each out-of-pool session:
  pool_fixed      -- pool top-32, never changes (baseline; no adaptation)
  session_causal  -- top-32 by firing over the FIRST half (calibration), fixed for the scored half
  dynamic_30s     -- per scored window, top-32 by firing over the trailing 30 s (updates every 2 s)
  dynamic_10s     -- same, trailing 10 s (faster-adapting, noisier)
  session_whole   -- top-32 over the WHOLE session (mildly non-causal) = an upper reference

Question: does dynamic (mid-session) beat session_causal (pick once)? It should only help if there
is WITHIN-session non-stationarity; if sessions are locally stationary, dynamic ~= session_causal
(and a changing mask must at least not HURT). Reports stratified by healthy/drifted baseline.
Usage: py research/iter41_dynamic_mask.py
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
import experiments.archive.indy.iter27_fresh_session as I27
import experiments.archive.indy.iter32_forward_split as I32
import experiments.archive.indy.iter39_masked_identity as I39
import models.tcn_gru.evaluate as E
import models.tcn_gru.best_model as M

SEED = 42
N_ACTIVE = 32
WIN = I39.WIN
POOL = I32.POOL
TARGETS = I27.FRESH + I32.FORWARD
ROLL_30 = int(round(30.0 / E.BIN))                # 750 bins
ROLL_10 = int(round(10.0 / E.BIN))                # 250 bins
OUT = ROOT / "results" / "metrics" / "iter41_dynamic_mask.json"


def topmask(fr):
    m = np.zeros(96, np.float32); m[np.argsort(fr)[-N_ACTIVE:]] = 1.0
    return m


def score_dynamic(net, n192, vel, mask_of_window, ym, ys):
    """Score second half; mask_of_window(k)->96-mask can differ per window. CAUSAL by construction."""
    import torch
    net.eval(); Y, P = [], []
    nwin = n192.shape[1] // WIN
    with torch.no_grad():
        for k in range(nwin // 2, nwin):          # score the SECOND half
            sl = slice(k * WIN, (k + 1) * WIN)
            x = I39.MI.make_masked_input_torch(torch.tensor(n192[:, sl][None]),
                                               torch.tensor(mask_of_window(k)[None]))
            P.append(net(x).numpy()[0] * ys + ym); Y.append(vel[sl])
    Y = np.concatenate([y[:, I39.AXES].reshape(-1, 2) for y in Y])
    P = np.concatenate([p.reshape(-1, 2) for p in P])
    return float(M.r2(Y, P).mean())


def main():
    t0 = time.time()
    counts = I27.load_counts_full(POOL + list(E.EVAL) + list(E.TEST) + TARGETS)
    I39.AXES = np.sort(np.argsort(np.mean([counts[s][1].std(0) for s in E.TRAIN], 0))[-2:])
    fr_pool = np.mean([counts[s][0].mean(1) for s in POOL], 0)
    pool_mask = topmask(fr_pool)
    print("=== Iteration 41: dynamic mid-session channel selection ===")
    print(f"    loaded {len(counts)} sessions [{time.time()-t0:.0f}s]", flush=True)

    # frozen random-mask 96-slot model (same recipe/seed as iter40)
    tr_ns = []
    for s in POOL:
        wins, _ = I39.neural_windows(counts, s)
        tr_ns += wins
    ev_ns, _ = I39.neural_windows(counts, list(E.EVAL)[0])
    def rand_sampler(j, rng):
        m = np.zeros(96, np.float32); m[rng.choice(96, N_ACTIVE, replace=False)] = 1.0
        return m
    t1 = time.time()
    net, ym, ys = I39.train_masked(tr_ns, rand_sampler, I39.CFG, I39.CFG["epochs"],
                                   (ev_ns, pool_mask, None, None))
    print(f"  random-mask 96-slot model trained [{time.time()-t1:.0f}s]\n", flush=True)

    rows = {"sessions": {}}
    for s in TARGETS:
        n192 = I39.MI.build_neural_192(counts[s][0], ewma_alpha=I27.ALPHA)
        cnt = counts[s][0]                                     # (96, T) raw counts
        vel = counts[s][1]
        nwin = n192.shape[1] // WIN
        half = nwin // 2
        era = "backward" if s in I27.FRESH else "forward"

        calib_fr = cnt[:, :half * WIN].mean(1)                # first-half firing (calibration)
        sess_calib_mask = topmask(calib_fr)
        whole_mask = topmask(cnt.mean(1))                     # non-causal reference

        def roll_mask(k, R):
            b = k * WIN
            fr = cnt[:, max(0, b - R):b].mean(1) if b > 0 else calib_fr
            return topmask(fr)

        r = {"era": era,
             "pool_fixed": score_dynamic(net, n192, vel, lambda k: pool_mask, ym, ys),
             "session_causal": score_dynamic(net, n192, vel, lambda k: sess_calib_mask, ym, ys),
             "dynamic_30s": score_dynamic(net, n192, vel, lambda k: roll_mask(k, ROLL_30), ym, ys),
             "dynamic_10s": score_dynamic(net, n192, vel, lambda k: roll_mask(k, ROLL_10), ym, ys),
             "session_whole": score_dynamic(net, n192, vel, lambda k: whole_mask, ym, ys)}
        # how much does the dynamic mask actually CHANGE within the session?
        masks = [roll_mask(k, ROLL_30) for k in range(half, nwin)]
        churn = float(np.mean([1 - (masks[i] * masks[i - 1]).sum() / N_ACTIVE
                               for i in range(1, len(masks))])) if len(masks) > 1 else 0.0
        r["dyn_churn_per_2s"] = churn
        r["base_healthy"] = r["session_causal"] >= 0.4
        rows["sessions"][s] = r
        print(f"  {s:20s} [{era:8s}] pool {r['pool_fixed']:+.3f} | sess_c {r['session_causal']:+.3f} | "
              f"dyn30 {r['dynamic_30s']:+.3f} | dyn10 {r['dynamic_10s']:+.3f} | "
              f"whole {r['session_whole']:+.3f} | churn/2s {churn:.2f}", flush=True)

    S = rows["sessions"]; names = list(S)
    cfgs = ["pool_fixed", "session_causal", "dynamic_30s", "dynamic_10s", "session_whole"]
    healthy = [s for s in names if S[s]["base_healthy"]]
    failed = [s for s in names if not S[s]["base_healthy"]]
    def agg(sub):
        return {c: (round(float(np.mean([S[s][c] for s in sub])), 3) if sub else None) for c in cfgs}
    print("\n  === MEANS (pool | sess_causal | dyn30 | dyn10 | whole) ===")
    for lab, sub in (("ALL", names), (f"HEALTHY (n={len(healthy)})", healthy),
                     (f"FAILED/drift (n={len(failed)})", failed)):
        a = agg(sub); print(f"    {lab:22s} " + " | ".join(f"{a[c]:+.3f}" for c in cfgs))
    print(f"  mean dynamic-mask churn per 2 s: "
          f"{np.mean([S[s]['dyn_churn_per_2s'] for s in names]):.2f} of 32 channels")
    rows["_summary"] = {"all": agg(names), "healthy": agg(healthy), "failed": agg(failed)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
