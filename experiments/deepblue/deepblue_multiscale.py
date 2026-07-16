#!/usr/bin/env python
"""Deep Blue finger SBP: our FULL architecture (causal multiscale EWMA input + causal
TCN+GRU), A/B vs the plain single-scale baseline (LOG-075 = 0.408 mean TEST R2).

LOG-075 fed raw current-bin SBP straight into the TCN+GRU. That omitted the one
input-side lever that worked on indy (LOG-068/074): expand each channel into
raw + causal EWMA history. This applies the ENTIRE model to Deep Blue -- multiscale
causal EWMA on all 96 SBP channels -> causal TCN+GRU -- and measures whether the
EWMA lever transfers to this dataset too.

EWMA is computed PER TRIAL (reset at trial boundaries) so no causal state leaks
across trials or across the train/eval/test split. Everything else is identical to
experiments/deepblue/deepblue_tcn_gru.py (MC files, chronological trial-disjoint 70/15/15 split,
TRAIN-only normalization, EVAL selection, 3-seed, strictly-causal TCN+GRU, 32 ms bins,
separate model per monkey). Usage: py experiments/deepblue/deepblue_multiscale.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.deepblue.deepblue_tcn_gru as db          # reuse validated loader/split/window/CFG
from src.intent_decoder.training import run

SEEDS = (42, 1, 7)
CONFIGS = {                                       # label: EWMA alphas (1.0 = raw)
    "raw96":         [1.0],                        # baseline (reproduces LOG-075 ~0.408)
    "raw+ewma0.1":   [1.0, 0.1],                   # 2-scale, eval-valid alpha (192 feat)
    "raw+3ewma":     [1.0, 0.5, 0.2, 0.1],         # 4-scale (384 feat)
}
OUT = ROOT / "results" / "metrics" / "deepblue_multiscale.json"


def ewma_causal(seg, alpha):
    """Causal EWMA down time axis of one trial segment (T, C); s[0]=x[0]."""
    o = seg.astype(np.float64).copy()
    for t in range(1, len(seg)):
        o[t] = alpha * seg[t] + (1 - alpha) * o[t - 1]
    return o.astype(np.float32)


def build_feats(x, trials, alphas):
    """Expand (T,96) SBP into (T, 96*len(alphas)) via per-trial causal EWMA scales."""
    if alphas == [1.0]:
        return x.astype(np.float32)
    blocks = [np.zeros_like(x) for _ in alphas]
    for tr in np.unique(trials):
        idx = np.flatnonzero(trials == tr)
        seg = x[idx]
        for j, a in enumerate(alphas):
            blocks[j][idx] = seg if a >= 1.0 else ewma_causal(seg, a)
    return np.concatenate(blocks, axis=1).astype(np.float32)


def prepare_ms(path, alphas):
    x, y, trials = db.load_mc(path)
    feats = build_feats(x, trials, alphas)                    # (T, 96*S)
    tr_ids, ev_ids, te_ids = db.chronological_split(trials)
    tr_mask = np.isin(trials, tr_ids)                         # TRAIN-only normalization
    mu = feats[tr_mask].mean(0, keepdims=True)
    sd = feats[tr_mask].std(0, keepdims=True) + 1e-6
    feats = ((feats - mu) / sd).astype(np.float32)
    train = db.window_trials(feats, y, trials, tr_ids)
    eval_ = db.window_trials(feats, y, trials, ev_ids)
    test = db.window_trials(feats, y, trials, te_ids)
    if not train or not eval_ or not test:
        raise ValueError("empty split")
    data = {"train": train, "eval": {"eval": eval_}, "test": {"test": test}}
    return data, {"n_feat": feats.shape[1], "windows": [len(train), len(eval_), len(test)]}


def main():
    print("=== Deep Blue: full architecture (multiscale causal EWMA + causal TCN+GRU), 3-seed ===")
    print("    A/B vs single-scale baseline (LOG-075 mean TEST R2 = 0.408); select on EVAL\n",
          flush=True)
    rows = {}
    for monkey, path in db.FILES.items():
        rows[monkey] = {}
        for label, alphas in CONFIGS.items():
            t0 = time.time()
            data, meta = prepare_ms(path, alphas)
            res = run(data, db.CFG, seeds=SEEDS)
            rows[monkey][label] = {"eval_r2": res["eval_r2"], "test_r2": res["test_r2"],
                                   "n_feat": meta["n_feat"], "n_params": res["n_params"]}
            print(f"  {monkey} {label:12s} n_feat={meta['n_feat']:3d}  "
                  f"EVAL R2={res['eval_r2']:.3f}  TEST R2={res['test_r2']:.3f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        print(flush=True)

    # per-config mean over monkeys, selected by EVAL
    print("  --- mean over monkeys (select on EVAL) ---")
    means = {}
    for label in CONFIGS:
        ev = np.mean([rows[m][label]["eval_r2"] for m in db.FILES])
        te = np.mean([rows[m][label]["test_r2"] for m in db.FILES])
        means[label] = {"eval_r2": float(ev), "test_r2": float(te)}
        print(f"    {label:12s} EVAL {ev:.3f}  TEST {te:.3f}")
    best = max(means, key=lambda k: means[k]["eval_r2"])
    base = means["raw96"]
    print(f"\n  best-by-EVAL: {best} -> EVAL {means[best]['eval_r2']:.3f}, TEST {means[best]['test_r2']:.3f}")
    print(f"  vs single-scale baseline raw96: "
          f"{means[best]['eval_r2']-base['eval_r2']:+.3f} EVAL / "
          f"{means[best]['test_r2']-base['test_r2']:+.3f} TEST", flush=True)

    rows["mean_over_monkeys"] = means
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
