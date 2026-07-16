#!/usr/bin/env python
"""Iteration 32: FORWARD-in-time generalization -- the real deployment scenario.

Why: the split dates expose a problem with every headline so far.
    train1-6 = Oct 5-14 2016 | eval1 = Oct 17 | test1 = Oct 24 | pool also has Sep, Oct 25-27,
    Dec 6-20, Jan 23-31 2017.
=> test1 (Oct 24) sits INSIDE the training pool's date range, surrounded by Oct 5-27 sessions.
   It is effectively INTERPOLATION, not extrapolation -- which is why it scores 0.63/0.755
   while never-seen sessions collapse (LOG-079).
And the LOG-079 "fresh" sessions (Apr-Jun 2016) are 3-6 months BEFORE all training data =
BACKWARD extrapolation, an unrealistically harsh test.

This does the honest, deployment-realistic thing: TRAIN ON THE PAST, DECODE THE FUTURE.
  train pool = the 20 sessions from Sep 15 - Dec 20 2016 (24-session pool minus Jan 2017)
  forward test = the 4 Jan 2017 sessions (~1 month after the newest training data)
  epoch selected on eval1 (Oct 17, inside the pool range)

Conditions per forward session (chronological 50/50: calibrate on 1st half, score 2nd):
  zero_shot | affine (2 scalars/axis) | finetune -- so we also learn whether calibration is
  still needed when the gap is only ~1 month. Honest causal input (counts + causal EWMA,
  no leaky Gaussian). Usage: py experiments/archive/indy/iter32_forward_split.py
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
import experiments.archive.indy.iter31_channel_reselect as I31
import models.tcn_gru.evaluate as E

SEED = 42
COUNTS = [8, 32]
FORWARD = ["indy_20170123_02", "indy_20170124_01",      # Jan 2017: the future
           "indy_20170127_03", "indy_20170131_02"]
POOL = [s for s in I27.POOL if s not in FORWARD]        # Sep 15 - Dec 20 2016 (20 sessions)
FT_EPOCHS = 25
FT_LR = 3e-4
OUT = ROOT / "results" / "metrics" / "iter32_forward_split.json"


def main():
    import torch
    t0 = time.time()
    counts = I27.load_counts_full(POOL + list(E.EVAL) + FORWARD)
    axes = np.sort(np.argsort(np.mean([counts[s][1].std(0) for s in E.TRAIN], 0))[-2:])
    print("=== Iteration 32: FORWARD-in-time (train Sep-Dec 2016 -> decode Jan 2017) ===")
    print(f"    train pool: {len(POOL)} sessions (Sep 15 - Dec 20 2016); eval1 = Oct 17")
    print(f"    forward test (~1 month ahead): {FORWARD}")
    print(f"    loaded {len(counts)} sessions [{time.time()-t0:.0f}s]\n", flush=True)

    def predict(net, X, ym, ys):
        net.eval()
        with torch.no_grad():
            return net(torch.tensor(X)).numpy() * ys + ym

    rows = {}
    for n in COUNTS:
        chans = I27.topN_channels(counts, n)
        tr_by = I27.build(counts, chans, axes, POOL)
        data = {"train": [x for s in POOL for x in tr_by[s]],
                "eval": I27.build(counts, chans, axes, list(E.EVAL)),
                "test": I27.build(counts, chans, axes, FORWARD)}   # scored per-session below
        t1 = time.time()
        res = H.run(data, I20.CAUSAL, seeds=(SEED,), ret_net=True, ret_preds=True)
        pool_net, (pym, pys) = res["net"], res["norm"]
        pool_state = {k: v.clone() for k, v in pool_net.state_dict().items()}
        print(f"  [{n}ch] pool model (20 sess): EVAL {res['eval_r2']:.3f}  "
              f"forward-mean(zero-shot, whole session) {res['test_r2']:.3f} "
              f"[{time.time()-t1:.0f}s]", flush=True)

        fwd_by = I27.build(counts, chans, axes, FORWARD)
        rows[f"{n}ch"] = {}
        for s in FORWARD:
            w = fwd_by[s]
            half = len(w) // 2
            Xc, Yc = I28.stack(w[:half]); Xt, Yt = I28.stack(w[half:])
            yt = Yt.reshape(-1, 2)
            out = {}
            p0_c = predict(pool_net, Xc, pym, pys).reshape(-1, 2)
            p0_t = predict(pool_net, Xt, pym, pys).reshape(-1, 2)
            out["zero_shot"] = I28.score(yt, p0_t)
            g, o = __import__("experiments.archive.indy.iter30_unsup_calibration", fromlist=["fit_affine"]).fit_affine(
                p0_c, Yc.reshape(-1, 2))
            out["affine"] = I28.score(yt, p0_t * g + o)
            net, ym, ys = I31.train(Xc, Yc, I20.CAUSAL, FT_EPOCHS, FT_LR,
                                    init=pool_state, norm=(pym, pys))
            out["finetune"] = I28.score(yt, predict(net, Xt, ym, ys).reshape(-1, 2))
            rows[f"{n}ch"][s] = {k: {"r2": v[0], "r": v[1]} for k, v in out.items()}
            print(f"    {s}: " + "  ".join(
                f"{k}: R2={v[0]:+.3f}/r={v[1]:.3f}" for k, v in out.items()), flush=True)

        rows[f"{n}ch"]["mean"] = {}
        print(f"  [{n}ch] MEAN over forward sessions:")
        for c in ["zero_shot", "affine", "finetune"]:
            m = float(np.mean([rows[f"{n}ch"][s][c]["r2"] for s in FORWARD]))
            mr = float(np.mean([rows[f"{n}ch"][s][c]["r"] for s in FORWARD]))
            rows[f"{n}ch"]["mean"][c] = {"r2": m, "r": mr}
            print(f"     {c:<10s} R2={m:+.3f}  r={mr:.3f}")
        print(flush=True)

    print("  Compare: BACKWARD fresh (Apr-Jun 2016, LOG-079/080) zero_shot 8ch +0.020 / "
          "32ch +0.253; finetune 8ch +0.389 / 32ch +0.584")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
