#!/usr/bin/env python
"""Iteration 34: validate the drift detector properly -- leave-one-MONTH-out cross-validation.

LOG-085 found a label-free drift detector (pred_std_ratio r=+0.92 with zero-shot R2) but on only
8 sessions, with thresholds fitted by eye. That is the weak point. This validates it on ~25
out-of-fold sessions for the cost of 4 trainings.

Design: hold out one MONTH at a time (Sep 2016 / Oct 2016 / Dec 2016 / Jan 2017), train on the
remaining sessions, and measure zero-shot R2 + the label-free proxies on the held-out month's
sessions. Every session is therefore evaluated by a model that never saw it OR its month --
which also prevents the LOG-084 interpolation trap (test1 was surrounded by same-month sessions).

eval1 (Oct 17) is held out of every fold and always used for epoch selection, so it never enters
training or evaluation. Channels are re-selected per fold on THAT FOLD'S TRAINING sessions (not
the fixed base-6) -- otherwise the Oct fold would pick channels using held-out data.

Proxies (label-free, from the session's FIRST half): overlap_topN, pred_std_ratio.
Zero-shot R2 is measured on the SECOND half. Usage: py experiments/archive/indy/iter34_detector_cv.py
"""
from __future__ import annotations

import json
import re
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
import experiments.archive.indy.iter7_final as I7
import models.tcn_gru.evaluate as E

SEED = 42
COUNTS = [32]                       # the recommended config (LOG-078/084)
OUT = ROOT / "results" / "metrics" / "iter34_detector_cv.json"

SPLIT = json.loads((ROOT / "models" / "tcn_gru" / "data_split.json").read_text())


def month_of(name):
    """Month key YYYYMM, resolving the renamed train*/eval1/test1 to their originals."""
    orig = SPLIT.get(name, name)
    m = re.search(r"(\d{4})(\d{2})\d{2}", orig)
    return f"{m.group(1)}{m.group(2)}"


def main():
    import torch
    t0 = time.time()
    alls = list(E.TRAIN) + list(E.EVAL) + list(E.TEST) + I7.EXTRA18
    evals = list(E.EVAL)                                  # eval1: always the selector
    folds = {}
    for s in alls:
        if s in evals:
            continue
        folds.setdefault(month_of(s), []).append(s)
    counts = I27.load_counts_full(alls)
    axes = np.sort(np.argsort(np.mean([counts[s][1].std(0) for s in E.TRAIN], 0))[-2:])
    print("=== Iteration 34: leave-one-MONTH-out validation of the drift detector ===")
    for k in sorted(folds):
        print(f"    fold {k}: {len(folds[k])} sessions")
    print(f"    eval1 held out of every fold as the epoch selector")
    print(f"    loaded {len(counts)} sessions [{time.time()-t0:.0f}s]\n", flush=True)

    rows = {}
    for n in COUNTS:
        recs = []
        for held, sess in sorted(folds.items()):
            train_s = [s for s in alls if s not in sess and s not in evals]
            # channels chosen on THIS FOLD'S training data only (no held-out leakage)
            fr = np.mean([counts[s][0].mean(1) for s in train_s], 0)
            chans = np.sort(np.argsort(fr)[-n:])
            pool_fr = fr
            tr_by = I27.build(counts, chans, axes, train_s)
            data = {"train": [x for s in train_s for x in tr_by[s]],
                    "eval": I27.build(counts, chans, axes, evals),
                    "test": I27.build(counts, chans, axes, sess)}
            t1 = time.time()
            res = H.run(data, I20.CAUSAL, seeds=(SEED,), ret_net=True)
            net, (ym, ys) = res["net"], res["norm"]
            print(f"  [{n}ch] fold {held}: train {len(train_s)} sess, EVAL {res['eval_r2']:.3f} "
                  f"[{time.time()-t1:.0f}s]", flush=True)

            def predict(X):
                net.eval()
                with torch.no_grad():
                    return net(torch.tensor(X)).numpy() * ys + ym

            by = I27.build(counts, chans, axes, sess)
            for s in sess:
                w = by[s]
                half = len(w) // 2
                Xw, _ = I28.stack(w[:half])
                Xt, Yt = I28.stack(w[half:])
                zs = I28.score(Yt.reshape(-1, 2), predict(Xt).reshape(-1, 2))
                T_half = (counts[s][0].shape[1] // I20.WIN) // 2 * I20.WIN
                fr_w = counts[s][0][:, :T_half].mean(1)
                overlap = len(set(np.argsort(fr_w)[-n:].tolist()) & set(chans.tolist())) / n
                pw = predict(Xw).reshape(-1, 2)
                psr = float(np.mean(pw.std(0) / (ys + 1e-9)))
                recs.append({"session": s, "fold": held, "zero_shot_r2": zs[0],
                             "zero_shot_r": zs[1], "overlap_topN": overlap,
                             "pred_std_ratio": psr})
                print(f"      {s:20s} zsR2={zs[0]:+.3f}  overlap={overlap:.2f}  "
                      f"pred_std={psr:.3f}", flush=True)

        y = np.array([r["zero_shot_r2"] for r in recs])
        print(f"\n  [{n}ch] {len(recs)} out-of-fold sessions; proxy vs zero-shot R2:")
        corrs = {}
        for p in ["overlap_topN", "pred_std_ratio"]:
            x = np.array([r[p] for r in recs])
            c = float(np.corrcoef(x, y)[0, 1])
            corrs[p] = c
            print(f"     {p:<16s} r = {c:+.3f}")
        print(f"     zero-shot R2: mean {y.mean():+.3f}  min {y.min():+.3f}  max {y.max():+.3f}")
        bad = [r for r in recs if r["zero_shot_r2"] < 0.4]
        print(f"     sessions below 0.4 zsR2: {len(bad)}/{len(recs)} "
              f"({100*len(bad)/len(recs):.0f}%) -- these are the ones needing calibration")
        rows[f"{n}ch"] = {"sessions": recs, "proxy_corr": corrs,
                          "mean_zero_shot_r2": float(y.mean())}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
