#!/usr/bin/env python
"""Iteration 33: a LABEL-FREE drift detector -- predict which sessions need calibration.

LOG-084: forward in time, 32ch zero-shot is fine on MOST sessions (0.65-0.76) but ~1 in 4
drifts badly (indy_20170124_01: 0.201 -> 0.589 only after calibration). Calibration is
therefore insurance, not a routine cost -- IF we can tell in advance which sessions need it.

This asks: do cheap LABEL-FREE signals, computable from a short observation window at session
start, predict the zero-shot decode quality? If yes, the device can flag a bad session and
calibrate only then.

Setup: one pool model (train Sep 15 - Dec 20 2016, 20 sessions, honest causal input). Scored on
8 out-of-pool sessions: 4 BACKWARD (Apr-Jun 2016) + 4 FORWARD (Jan 2017). Per session, proxies
are computed on the FIRST half (the "observation window", no labels used) and zero-shot R2 is
measured on the SECOND half (what you'd get if you did NOT calibrate).

Label-free proxies:
  overlap_topN   : |top-N by firing on the window  ∩  pool's top-N| / N   (LOG-083: drifted
                   sessions retain only ~2.5/8 of the pool channels)
  firing_corr    : corr(per-channel mean firing over all 96, window vs pool)
  pred_std_ratio : std(model predictions on the window) / std(pool training velocity) -- a
                   confused model tends to collapse toward the mean (low output variance)

Reports each proxy's correlation with zero-shot R2 across the 8 sessions.
Usage: py experiments/archive/indy/iter33_drift_detector.py
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
import experiments.archive.indy.iter20_multiscale as I20
import experiments.archive.indy.iter27_fresh_session as I27
import experiments.archive.indy.iter28_calibration as I28
import experiments.archive.indy.iter32_forward_split as I32
import models.tcn_gru.evaluate as E
import models.tcn_gru.best_model as M

SEED = 42
COUNTS = [8, 32]
BACKWARD = I27.FRESH                 # Apr-Jun 2016
FORWARD = I32.FORWARD                # Jan 2017
POOL = I32.POOL                      # Sep 15 - Dec 20 2016 (20 sessions)
OUT = ROOT / "results" / "metrics" / "iter33_drift_detector.json"


def main():
    import torch
    t0 = time.time()
    targets = BACKWARD + FORWARD
    # test1 is loaded only because H.run requires a "test" dict; it is the interpolation
    # reference (Oct 24, inside the pool's date range -- LOG-084), not a generalization number.
    counts = I27.load_counts_full(POOL + list(E.EVAL) + list(E.TEST) + targets)
    axes = np.sort(np.argsort(np.mean([counts[s][1].std(0) for s in E.TRAIN], 0))[-2:])
    pool_fr = np.mean([counts[s][0].mean(1) for s in POOL], 0)      # 96-dim reference
    print("=== Iteration 33: label-free drift detector ===")
    print(f"    pool: {len(POOL)} sessions (Sep-Dec 2016); targets: 4 backward + 4 forward")
    print(f"    loaded {len(counts)} sessions [{time.time()-t0:.0f}s]\n", flush=True)

    rows = {}
    for n in COUNTS:
        chans = I27.topN_channels(counts, n)
        tr_by = I27.build(counts, chans, axes, POOL)
        data = {"train": [x for s in POOL for x in tr_by[s]],
                "eval": I27.build(counts, chans, axes, list(E.EVAL)),
                "test": I27.build(counts, chans, axes, list(E.TEST))}
        t1 = time.time()
        res = H.run(data, I20.CAUSAL, seeds=(SEED,), ret_net=True)
        net, (ym, ys) = res["net"], res["norm"]
        print(f"  [{n}ch] pool model: EVAL {res['eval_r2']:.3f} [{time.time()-t1:.0f}s]", flush=True)

        def predict(X):
            net.eval()
            with torch.no_grad():
                return net(torch.tensor(X)).numpy() * ys + ym

        recs = []
        by = I27.build(counts, chans, axes, targets)
        for s in targets:
            w = by[s]
            half = len(w) // 2
            Xw, Yw = I28.stack(w[:half])          # observation window (labels NOT used)
            Xt, Yt = I28.stack(w[half:])          # scored half
            zs = I28.score(Yt.reshape(-1, 2), predict(Xt).reshape(-1, 2))

            # --- label-free proxies from the window only ---
            T_half = (counts[s][0].shape[1] // I20.WIN) // 2 * I20.WIN
            fr_w = counts[s][0][:, :T_half].mean(1)
            top_w = set(np.argsort(fr_w)[-n:].tolist())
            overlap = len(top_w & set(chans.tolist())) / n
            firing_corr = float(np.corrcoef(fr_w, pool_fr)[0, 1])
            pw = predict(Xw).reshape(-1, 2)
            pred_std_ratio = float(np.mean(pw.std(0) / (ys + 1e-9)))

            era = "backward" if s in BACKWARD else "forward"
            recs.append({"session": s, "era": era, "zero_shot_r2": zs[0], "zero_shot_r": zs[1],
                         "overlap_topN": overlap, "firing_corr": firing_corr,
                         "pred_std_ratio": pred_std_ratio})
            print(f"    {s:20s} [{era:8s}] zsR2={zs[0]:+.3f}  overlap={overlap:.2f}  "
                  f"firing_corr={firing_corr:+.3f}  pred_std={pred_std_ratio:.3f}", flush=True)

        y = np.array([r["zero_shot_r2"] for r in recs])
        print(f"\n  [{n}ch] proxy vs zero-shot R2 (Pearson r over {len(recs)} sessions):")
        corrs = {}
        for p in ["overlap_topN", "firing_corr", "pred_std_ratio"]:
            x = np.array([r[p] for r in recs])
            c = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else float("nan")
            corrs[p] = c
            print(f"     {p:<16s} r = {c:+.3f}")
        rows[f"{n}ch"] = {"sessions": recs, "proxy_corr_with_zero_shot_r2": corrs,
                          "pool_eval_r2": res["eval_r2"]}
        print(flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    print("NOTE: only 8 sessions -> correlations are indicative, not conclusive.")


if __name__ == "__main__":
    main()
