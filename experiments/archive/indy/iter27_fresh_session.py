#!/usr/bin/env python
"""Iteration 27: THE UNBIASED HEADLINE -- frozen pipeline, scored once on FRESH sessions.

test1 is burned (~25 experiments have read it, LOG-073), so no number measured on it is
an unbiased headline. This freezes the entire pipeline and scores it ONCE on indy
sessions that have NEVER been used for anything -- not train, not eval, not test.

Combines the session's three findings:
  * HONEST CAUSAL INPUT (LOG-074): counts + causal EWMA(0.1), NO centered Gaussian, so
    the model is genuinely 0 ms lookahead (the cached Gaussian leaked ~160 ms of future).
  * 32 CHANNELS (LOG-078): the biggest lever (+0.12 on test1); does it hold on unseen data?
  * EVAL-ONLY SELECTION (LOG-073): epoch chosen on eval1; fresh sessions read once.

FROZEN config: strictly-causal wide TCN+GRU (F64/H64/L1, dils[1,2,4,8], bidir=False),
24-session pool, 40 ms bins, top-N firing channels selected on the base-6, 3-seed.
Nothing is tuned on the fresh sessions -- they are a pure read-out.

CAVEAT: the only never-used indy sessions are Apr-Jun 2016, while the training pool is
Sep 2016-Jan 2017 -> the fresh set is 3-6 months EARLIER than any training data. Distant
sessions drift and hurt (LOG-065), so this is a HARSH lower bound, not a like-for-like
test1 replacement. Reported alongside test1 for context. loco excluded (different monkey;
cross-subject transfer collapses, LOG-027).

Usage: py research/iter27_fresh_session.py
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
import research.iter25_causal_smoothing as I25
import models.tcn_gru.evaluate as E
import models.tcn_gru.best_model as M
import research.iter7_final as I7

SEEDS = (42, 1, 7)
ALPHA = 0.1                       # eval-valid causal EWMA (LOG-073)
COUNTS = [8, 32]
WIN = I20.WIN
POOL = list(E.TRAIN) + I7.EXTRA18                       # 24 training sessions
FRESH = ["indy_20160407_02", "indy_20160419_01",        # NEVER used for anything
         "indy_20160624_03", "indy_20160630_01"]
OUT = ROOT / "results" / "metrics" / "iter27_fresh_session.json"


def load_counts_full(names):
    """Unsmoothed counts for ALL 96 channels + primary-finger velocity.

    NOT research.iter25.load_counts -- that pre-slices to the deployed 8 channels, which
    would silently break top-32 selection. Also: some indy sessions track TWO fingers
    (finger_pos 6 cols) while the pool tracks one (3 cols). Verified that cols 0:3 of the
    6-col sessions match the pool's primary finger (std pattern [~0.6, ~7, ~4-7] vs pool
    [0.51, 7.36, 6.79]; cols 3:6 are a different, larger marker). So take vel[:, :3]
    everywhere to keep every session on the same physical signal."""
    saved = E.RATE_SIGMA
    E.RATE_SIGMA = 0.0                      # raw counts: no centered Gaussian (LOG-074)
    try:
        out = {}
        for s in names:
            r, v = E.load_source_electrode(s)
            out[s] = (r.astype(np.float32), v[:, :3])       # all 96 ch, primary finger
        return out
    finally:
        E.RATE_SIGMA = saved


def topN_channels(counts, n):
    """Top-N by mean firing on the base-6 (the same robust rule that picked the 8)."""
    fr = np.mean([counts[s][0].mean(1) for s in E.TRAIN], 0)
    return np.sort(np.argsort(fr)[-n:])


def feats(raw_counts, chans):
    """HONEST causal input: counts + causal EWMA, no Gaussian (LOG-074)."""
    c = raw_counts[chans]
    return np.concatenate([c, I25.ewma(c, ALPHA)], 0)


def build(counts, chans, axes, names):
    def wins(s):
        f = feats(counts[s][0], chans)
        mu, sd = f.mean(1, keepdims=True), f.std(1, keepdims=True) + 1e-6
        fz = ((f - mu) / sd).astype(np.float32)
        v = counts[s][1]
        return [{"e": fz[:, k * WIN:(k + 1) * WIN], "vel": v[k * WIN:(k + 1) * WIN][:, axes]}
                for k in range(fz.shape[1] // WIN)]
    return {s: wins(s) for s in names}


def main():
    t0 = time.time()
    need = POOL + list(E.EVAL) + list(E.TEST) + FRESH
    print("=== Iteration 27: UNBIASED headline on FRESH (never-used) sessions ===")
    print(f"    frozen: causal TCN+GRU + counts+causalEWMA({ALPHA}), 24 sess, 3-seed, select on eval1")
    print(f"    fresh (read ONCE): {FRESH}\n", flush=True)
    counts = load_counts_full(need)                     # 96 ch unsmoothed counts, no Gaussian
    axes = np.sort(np.argsort(np.mean([counts[s][1].std(0) for s in E.TRAIN], 0))[-2:])
    print(f"    loaded {len(counts)} sessions of raw counts [{time.time()-t0:.0f}s]\n", flush=True)

    rows = {}
    for n in COUNTS:
        chans = topN_channels(counts, n)
        t1 = time.time()
        tr_by = build(counts, chans, axes, POOL)
        data = {"train": [x for s in POOL for x in tr_by[s]],
                "eval": build(counts, chans, axes, list(E.EVAL)),
                "test": build(counts, chans, axes, list(E.TEST) + FRESH)}
        res = H.run(data, I20.CAUSAL, seeds=SEEDS, ret_preds=True)
        # per-session R2 from the ensembled preds
        per = {}
        for name, (Ye, P) in res["test_preds"].items():
            per[name] = float(M.r2(Ye.reshape(-1, 2), P.reshape(-1, 2)).mean())
        fresh_vals = [per[s] for s in FRESH]
        rows[f"{n}ch"] = {"n_ch": n, "eval_r2": res["eval_r2"],
                          "test1_r2": per[list(E.TEST)[0]],
                          "fresh_per_session": {s: per[s] for s in FRESH},
                          "fresh_mean_r2": float(np.mean(fresh_vals)),
                          "n_params": res["n_params"], "channels": chans.tolist()}
        r = rows[f"{n}ch"]
        print(f"  {n:2d} ch  EVAL {r['eval_r2']:.3f} | test1 {r['test1_r2']:.3f} (burned) | "
              f"FRESH mean {r['fresh_mean_r2']:.3f}   [{time.time()-t1:.0f}s]")
        for s in FRESH:
            print(f"        fresh {s}: R2={per[s]:.3f}")
        print(flush=True)

    print("  --- UNBIASED HEADLINE (fresh sessions, frozen pipeline, read once) ---")
    for n in COUNTS:
        r = rows[f"{n}ch"]
        print(f"    {n:2d}ch: FRESH mean R2 = {r['fresh_mean_r2']:.3f}  "
              f"(test1 {r['test1_r2']:.3f}, eval {r['eval_r2']:.3f})")
    if len(COUNTS) == 2:
        a, b = rows[f"{COUNTS[0]}ch"], rows[f"{COUNTS[1]}ch"]
        print(f"\n  32ch vs 8ch on FRESH: {b['fresh_mean_r2']-a['fresh_mean_r2']:+.3f} "
              f"(on burned test1 it was {b['test1_r2']-a['test1_r2']:+.3f})")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
