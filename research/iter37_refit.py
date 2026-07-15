#!/usr/bin/env python
"""Iteration 37: ReFIT -- calibrate WITHOUT ground-truth velocity, using the task's own targets.

THE CONSTRAINT WE ARE ATTACKING: we established that fixing drift REQUIRES labels -- BatchNorm
adaptation gives only +0.02 (LOG-082) and manifold alignment/SOBI fails catastrophically (LOG-087,
r flips to -0.42). So the ~12% of drifted sessions need a labelled calibration block (LOG-081:
>=60 s, >=4 min for ~0.5). That is a real cost: the user must perform instructed movements.

THE LOOPHOLE: "labels" meant ground-truth KINEMATICS. But a deployed system always knows two other
things -- where the CURSOR is (it renders it) and where the TARGET is (it shows it). It just does
not know the user's intended velocity. ReFIT (Gilja et al. 2012, Nat Neurosci -- roughly doubled
closed-loop BCI performance) infers intention from geometry instead:

    intended direction = unit(target - cursor)          # the user is trying to reach the target
    intended velocity  = ||decoded speed|| * direction  # keep decoded magnitude, rotate direction

VERIFIED ENABLERS in the indy .mat: target_pos (2,T) on an 8x8 grid, cursor_pos (2,T), 342 trials
per session; cursor axis 0/1 correlate 1.000 with finger axes 1/2 -- i.e. cursor_pos IS exactly the
two axes we decode.

NO GROUND-TRUTH VELOCITY IS USED to build the pseudo-labels. Scoring is against true velocity.

Conditions per drifted session (calibrate on the 1st half, score the 2nd against TRUE velocity):
  zero_shot        : frozen pool model
  refit_ft_{60,240}s : fine-tune on ReFIT pseudo-labels        <-- NO true velocity
  true_ft_{60,240}s  : fine-tune on TRUE velocity              <-- the upper reference (LOG-080/081)
If refit approaches true_ft, the labelled calibration block is unnecessary -- the task supplies the
labels for free, continuously, during ordinary use.

CAVEAT: in this dataset the cursor is HAND-driven, so cursor_pos reflects true position (not
velocity). In a real closed-loop BCI the cursor is DECODER-driven and equally known, so the method
transfers; but the position information here is cleaner than closed-loop would be.

Usage: py research/iter37_refit.py
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

import h5py
import research.harness as H
import research.iter20_multiscale as I20
import research.iter27_fresh_session as I27
import research.iter28_calibration as I28
import research.iter31_channel_reselect as I31
import research.iter32_forward_split as I32
import models.tcn_gru.evaluate as E

SEED = 42
N_CH = 32
POOL = I32.POOL
TARGETS = I27.FRESH + I32.FORWARD
CAL_SIZES = [30, 120]                 # windows -> 60 s, 240 s
FT_EPOCHS, FT_LR = 25, 3e-4
AT_TARGET_DIST = 1.0                  # cursor within this distance => intended velocity ~ 0
OUT = ROOT / "results" / "metrics" / "iter37_refit.json"


def load_task(name):
    """Bin cursor_pos and target_pos onto the SAME 40 ms grid as the rates/velocity.

    SIGN: the dataset documents finger_pos as (z, -x, -y) -- the finger axes are NEGATED relative
    to cursor_pos. Measured: cursor[0] vs finger[1] = -0.9998, cursor[1] vs finger[2] = -0.9998.
    We decode finger axes [1,2], so we negate cursor/target into FINGER orientation; otherwise the
    ReFIT direction is inverted (caught by the assumption check: pseudo-labels scored -0.857 with
    true velocity instead of +0.857). Scale differs too (cursor ~+-70 vs finger ~+-8) but that is
    irrelevant: only the unit direction is used, and the speed comes from the decoder."""
    p = E.fetch(name)
    with h5py.File(p, "r") as f:
        t = np.array(f["t"]).squeeze()
        cur = np.array(f["cursor_pos"])        # (2, T)
        tgt = np.array(f["target_pos"])        # (2, T) 8x8 grid
    edges = np.arange(t[0], t[-1], E.BIN)
    centers = edges[:-1] + E.BIN / 2
    cb = -np.stack([np.interp(centers, t, cur[a]) for a in range(2)], 1)   # -> finger orientation
    tb = -np.stack([np.interp(centers, t, tgt[a]) for a in range(2)], 1)
    return cb, tb


def refit_labels(cursor, target, v_hat):
    """ReFIT intention: keep the decoded SPEED, rotate the DIRECTION toward the target.

    Uses ONLY cursor + target (both known to a deployed system) and the decoder's own output.
    Never touches ground-truth velocity."""
    d = target - cursor                                  # (n, 2) toward the target
    dist = np.linalg.norm(d, axis=1, keepdims=True)
    direction = d / np.maximum(dist, 1e-6)
    speed = np.linalg.norm(v_hat, axis=1, keepdims=True)  # decoder's own speed estimate
    v = speed * direction
    v[dist[:, 0] < AT_TARGET_DIST] = 0.0                  # at the target => intend to hold still
    return v.astype(np.float32)


def main():
    import torch
    t0 = time.time()
    counts = I27.load_counts_full(POOL + list(E.EVAL) + list(E.TEST) + TARGETS)
    axes = np.sort(np.argsort(np.mean([counts[s][1].std(0) for s in E.TRAIN], 0))[-2:])
    fr = np.mean([counts[s][0].mean(1) for s in POOL], 0)
    chans = np.sort(np.argsort(fr)[-N_CH:])
    print("=== Iteration 37: ReFIT -- calibrate with NO ground-truth velocity ===")
    print(f"    decoded axes = finger {axes.tolist()} (== cursor axes 0,1)")
    print(f"    loaded {len(counts)} sessions [{time.time()-t0:.0f}s]\n", flush=True)

    tr_by = I27.build(counts, chans, axes, POOL)
    data = {"train": [x for s in POOL for x in tr_by[s]],
            "eval": I27.build(counts, chans, axes, list(E.EVAL)),
            "test": I27.build(counts, chans, axes, list(E.TEST))}
    t1 = time.time()
    res = H.run(data, I20.CAUSAL, seeds=(SEED,), ret_net=True)
    pool_state = {k: v.clone() for k, v in res["net"].state_dict().items()}
    pool_net, (ym, ys) = res["net"], res["norm"]
    print(f"  pool model: EVAL {res['eval_r2']:.3f} [{time.time()-t1:.0f}s]\n", flush=True)

    def predict(net, X, m, s_):
        net.eval()
        with torch.no_grad():
            return net(torch.tensor(X)).numpy() * s_ + m

    fresh_by = I27.build(counts, chans, axes, TARGETS)
    rows = {}
    for s in TARGETS:
        w = fresh_by[s]
        half = len(w) // 2
        Xt, Yt = I28.stack(w[half:]); yt = Yt.reshape(-1, 2)
        era = "backward" if s in I27.FRESH else "forward"
        cb, tb = load_task(s)
        rows[s] = {"era": era}
        rows[s]["zero_shot"] = I28.score(yt, predict(pool_net, Xt, ym, ys).reshape(-1, 2))

        for N in CAL_SIZES:
            if N > half:
                continue
            secs = int(N * I20.WIN * E.BIN)
            Xc, Yc = I28.stack(w[:N])                       # Yc = TRUE vel (reference only)
            n_bins = N * I20.WIN
            cur_c, tgt_c = cb[:n_bins], tb[:n_bins]
            # --- ReFIT pseudo-labels: decoder's own speed + target-derived direction ---
            v_hat = predict(pool_net, Xc, ym, ys).reshape(-1, 2)
            v_ref = refit_labels(cur_c, tgt_c, v_hat).reshape(Yc.shape)
            # sanity: how well do the pseudo-labels match true velocity? (diagnostic only)
            agree = float(np.mean(I28.M.corr(Yc.reshape(-1, 2), v_ref.reshape(-1, 2))))
            net, m, s_ = I31.train(Xc, v_ref, I20.CAUSAL, FT_EPOCHS, FT_LR,
                                   init=pool_state, norm=(ym, ys))
            rows[s][f"refit_ft_{secs}s"] = I28.score(yt, predict(net, Xt, m, s_).reshape(-1, 2))
            rows[s][f"_pseudo_agree_{secs}s"] = (agree, 0.0)
            # --- reference: true labels ---
            net, m, s_ = I31.train(Xc, Yc, I20.CAUSAL, FT_EPOCHS, FT_LR,
                                   init=pool_state, norm=(ym, ys))
            rows[s][f"true_ft_{secs}s"] = I28.score(yt, predict(net, Xt, m, s_).reshape(-1, 2))

        print(f"  {s:20s} [{era:8s}] " + "  ".join(
            f"{k}={v[0]:+.3f}" for k, v in rows[s].items() if k != "era"), flush=True)

    print("\n  === MEANS over held-out sessions ===")
    keys = [k for k in rows[TARGETS[0]] if k != "era"]
    summary = {}
    for k in keys:
        summary[k] = float(np.mean([rows[s][k][0] for s in TARGETS if k in rows[s]]))
        print(f"    {k:<22s} {summary[k]:+.3f}")
    for N in CAL_SIZES:
        secs = int(N * I20.WIN * E.BIN)
        if f"refit_ft_{secs}s" in summary:
            zs, rf, tf = summary["zero_shot"], summary[f"refit_ft_{secs}s"], summary[f"true_ft_{secs}s"]
            print(f"    => @{secs}s: refit {rf-zs:+.3f} vs zero-shot | true-label {tf-zs:+.3f} | "
                  f"refit recovers {100*(rf-zs)/max(tf-zs,1e-9):.0f}% of the labelled gain")
    rows["_summary"] = summary
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
