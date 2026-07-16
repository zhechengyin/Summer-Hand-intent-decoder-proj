#!/usr/bin/env python
"""Iteration 40: FAST masked-identity test -- one pool training, tested on the drifted sessions.

Same question as iter39 (RMTBD brief) but ~4x faster: instead of 4-fold leave-one-month-out (4
trainings/config), train ONCE on the 20-session pool (Sep-Dec 2016) and test the 4 configs on the
8 out-of-pool sessions we already know include the drifted ones (4 backward Apr-Jun 2016 + 4 forward
Jan 2017). test1 is NOT used -- it is interpolation (sits inside the pool months, LOG-084) and
cannot show drift rescue, which is the whole point. Single seed, labelled as such.

Reuses iter39's validated masking machinery (mask-correctness already proven there, LOG/iter39):
  fixed32          -- plain 64-feat, pool top-32 (baseline)
  slot_fixedmask   -- 96-slot masked rep, pool top-32 mask (isolates representation cost)
  slot_randommask  -- 96-slot, random 32/batch; tested on session top-32
  slot_sessionmask -- 96-slot, each train window masked to its session top-32; tested on session top-32
All channel selection is by firing rate (label-free). Usage: py research/iter40_masked_fast.py
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
import research.iter27_fresh_session as I27
import research.iter32_forward_split as I32
import research.iter39_masked_identity as I39
import models.tcn_gru.evaluate as E
import models.tcn_gru.best_model as M

SEED = 42
N_ACTIVE = 32
POOL = I32.POOL                                   # Sep 15 - Dec 20 2016 (20 sessions)
TARGETS = I27.FRESH + I32.FORWARD                 # 4 backward + 4 forward (out-of-pool)
OUT = ROOT / "results" / "metrics" / "iter40_masked_fast.json"


def main():
    t0 = time.time()
    counts = I27.load_counts_full(POOL + list(E.EVAL) + list(E.TEST) + TARGETS)
    I39.AXES = np.sort(np.argsort(np.mean([counts[s][1].std(0) for s in E.TRAIN], 0))[-2:])
    print("=== Iteration 40: FAST masked-identity (1 pool training, drifted-session test) ===")
    I39.run_mask_tests()
    fr_pool = np.mean([counts[s][0].mean(1) for s in POOL], 0)
    pool_chans = np.sort(np.argsort(fr_pool)[-N_ACTIVE:])
    pool_mask = np.zeros(96, np.float32); pool_mask[pool_chans] = 1.0
    print(f"    pool {len(POOL)} sess; targets {len(TARGETS)}; loaded {len(counts)} [{time.time()-t0:.0f}s]\n", flush=True)

    # --- training windows (192-neural) + per-session firing masks ---
    tr_ns, tr_masks_sess = [], []
    for s in POOL:
        wins, fr = I39.neural_windows(counts, s)
        sm = np.zeros(96, np.float32); sm[np.argsort(fr)[-N_ACTIVE:]] = 1.0
        for w in wins:
            tr_ns.append(w); tr_masks_sess.append(sm)
    ev_ns, _ = I39.neural_windows(counts, list(E.EVAL)[0])

    def rand_sampler(j, rng):
        m = np.zeros(96, np.float32); m[rng.choice(96, N_ACTIVE, replace=False)] = 1.0
        return m
    def sess_sampler(j, rng):
        return tr_masks_sess[j]

    CFG = I39.CFG
    npar = sum(p.numel() for p in M.build_net({**CFG, "n_out": 2}, I39.MI.IN_DIM).parameters())
    print(f"  96-slot model: {npar:,} params (~{npar/1024:.0f} KB int8)", flush=True)

    # --- config 1: fixed-32 baseline ---
    t1 = time.time()
    b_tr = [x for s in POOL for x in I39.fixed32_windows(counts, s, pool_chans)]
    b_ev = {list(E.EVAL)[0]: I39.fixed32_windows(counts, list(E.EVAL)[0], pool_chans)}
    b_te = {s: I39.fixed32_windows(counts, s, pool_chans) for s in TARGETS}
    b_res = H.run({"train": b_tr, "eval": b_ev, "test": b_te}, CFG, seeds=(SEED,), ret_preds=True)
    base_r2 = {s: float(M.r2(Ye.reshape(-1, 2), P.reshape(-1, 2)).mean())
               for s, (Ye, P) in b_res["test_preds"].items()}
    print(f"  fixed32 trained [{time.time()-t1:.0f}s]", flush=True)

    # --- configs 2-4: 96-slot masked ---
    nets = {}
    for name, sampler in (("slot_fixedmask", pool_mask), ("slot_randommask", rand_sampler),
                          ("slot_sessionmask", sess_sampler)):
        t1 = time.time()
        nets[name] = I39.train_masked(tr_ns, sampler, CFG, CFG["epochs"], (ev_ns, pool_mask, None, None))
        print(f"  {name} trained [{time.time()-t1:.0f}s]", flush=True)

    rows = {"_params": npar, "sessions": {}}
    for s in TARGETS:
        te_ns, fr_s = I39.neural_windows(counts, s)
        sess_mask = np.zeros(96, np.float32); sess_mask[np.argsort(fr_s)[-N_ACTIVE:]] = 1.0
        overlap = int((sess_mask * pool_mask).sum())
        era = "backward" if s in I27.FRESH else "forward"
        rec = {"era": era, "overlap": overlap, "fixed32": base_r2[s],
               "base_healthy": base_r2[s] >= 0.4}
        net, ym, ys = nets["slot_fixedmask"]
        rec["slot_fixedmask"] = I39.score_masked(net, te_ns, pool_mask, ym, ys)[0]
        for cn in ("slot_randommask", "slot_sessionmask"):
            net, ym, ys = nets[cn]
            rec[cn] = I39.score_masked(net, te_ns, sess_mask, ym, ys)[0]
        rows["sessions"][s] = rec
        print(f"    {s:20s} [{era:8s}] ov{overlap:2d}/32 fx32 {rec['fixed32']:+.3f} | "
              f"slotfix {rec['slot_fixedmask']:+.3f} | rand {rec['slot_randommask']:+.3f} | "
              f"sess {rec['slot_sessionmask']:+.3f}", flush=True)

    # aggregate / stratify
    S = rows["sessions"]; names = list(S)
    configs = ["fixed32", "slot_fixedmask", "slot_randommask", "slot_sessionmask"]
    healthy = [s for s in names if S[s]["base_healthy"]]
    failed = [s for s in names if not S[s]["base_healthy"]]

    def agg(sub):
        return {c: (round(float(np.mean([S[s][c] for s in sub])), 3) if sub else None) for c in configs}
    print("\n  === MEANS (fixed32 | slotfix | rand | sess) ===")
    for label, sub in (("ALL", names), (f"HEALTHY base>=0.4 (n={len(healthy)})", healthy),
                       (f"FAILED base<0.4 (n={len(failed)})", failed)):
        a = agg(sub)
        print(f"    {label:26s} " + " | ".join(f"{a[c]:+.3f}" for c in configs))
    rows["_summary"] = {"all": agg(names), "healthy": agg(healthy), "failed": agg(failed),
                        "n_healthy": len(healthy), "n_failed": len(failed)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
