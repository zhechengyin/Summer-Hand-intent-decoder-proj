#!/usr/bin/env python
"""Iteration 29: how LITTLE calibration data is enough? (the deployable question)

LOG-080 showed per-session calibration rescues the drift collapse (8ch 0.02->0.39,
32ch 0.25->0.58) -- but it used HALF a session (~7-12 min), which is not a realistic
calibration block. This sweeps calibration DURATION to find the knee: how many seconds
of labelled calibration buys most of the gain?

Per fresh session, chronological split: the LAST half is the fixed scored set (identical
across every calibration size, so sizes are directly comparable); the calibration data is
the FIRST N windows of the first half (realistic: you calibrate at session start).
Each window = 2.0 s (50 bins x 40 ms).

  affine   : per-axis gain/offset on the calibration data (2 scalars/axis, net untouched)
  finetune : pool weights fine-tuned on the calibration data

Sizes: 30 s / 1 / 2 / 4 / 8 min and the full first half. Usage: py research/iter29_calibration_size.py
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
import research.iter27_fresh_session as I27
import research.iter28_calibration as I28
import models.tcn_gru.evaluate as E
import models.tcn_gru.best_model as M

SEED = 42
COUNTS = [8, 32]
FRESH = I27.FRESH
POOL = I27.POOL
SIZES = [15, 30, 60, 120, 240, None]      # windows; None = all of the first half
WIN_S = I20.WIN * E.BIN                   # 2.0 s per window
FT_EPOCHS = 25
FT_LR = 3e-4
OUT = ROOT / "results" / "metrics" / "iter29_calibration_size.json"


def main():
    import torch
    import torch.nn as nn
    t0 = time.time()
    counts = I27.load_counts_full(POOL + list(E.EVAL) + list(E.TEST) + FRESH)
    axes = np.sort(np.argsort(np.mean([counts[s][1].std(0) for s in E.TRAIN], 0))[-2:])
    print("=== Iteration 29: calibration SIZE sweep (how few seconds suffice?) ===")
    print(f"    window = {WIN_S:.1f}s; scored set = fixed last half of each fresh session")
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
        pool_net, (ym, ys) = res["net"], res["norm"]
        print(f"  [{n}ch] pool model: EVAL {res['eval_r2']:.3f} test1 {res['test_r2']:.3f} "
              f"[{time.time()-t1:.0f}s]", flush=True)

        fresh_by = I27.build(counts, chans, axes, FRESH)
        rows[f"{n}ch"] = {}
        for s in FRESH:
            w = fresh_by[s]
            half = len(w) // 2
            calib_pool, test_w = w[:half], w[half:]
            Xt, Yt = I28.stack(test_w)
            yt = Yt.reshape(-1, 2)
            rows[f"{n}ch"][s] = {}

            def predict(net, X):
                net.eval()
                with torch.no_grad():
                    return net(torch.tensor(X)).numpy() * ys + ym

            p_test0 = predict(pool_net, Xt)
            zs = I28.score(yt, p_test0.reshape(-1, 2))
            rows[f"{n}ch"][s]["zero_shot"] = {"secs": 0, "affine": zs, "finetune": zs}

            for N in SIZES:
                # Key by the REQUESTED size, and SKIP (never silently cap) sizes a session
                # cannot supply -- capping produced per-session keys (408s/260s/248s/480s)
                # so a "480s" mean averaged only the 1 session that had that much data.
                if N is not None and N > len(calib_pool):
                    continue
                k = len(calib_pool) if N is None else N
                if k < 5:
                    continue
                cw = calib_pool[:k]
                Xc, Yc = I28.stack(cw)
                secs = k * WIN_S
                # affine
                pc = predict(pool_net, Xc).reshape(-1, 2); yc = Yc.reshape(-1, 2)
                g = np.zeros(2); o = np.zeros(2)
                for a in range(2):
                    A = np.vstack([pc[:, a], np.ones(len(pc))]).T
                    g[a], o[a] = np.linalg.lstsq(A, yc[:, a], rcond=None)[0]
                aff = I28.score(yt, p_test0.reshape(-1, 2) * g + o)
                # finetune
                ft = M.build_net({**I20.CAUSAL, "n_out": 2}, Xc.shape[1])
                ft.load_state_dict(pool_net.state_dict())
                opt = torch.optim.AdamW(ft.parameters(), lr=FT_LR,
                                        weight_decay=I20.CAUSAL["wd"])
                mse = nn.MSELoss()
                Xtc = torch.tensor(Xc)
                Ytc = torch.tensor(((Yc - ym) / ys).astype(np.float32))
                for ep in range(FT_EPOCHS):
                    ft.train()
                    idx = np.random.permutation(len(Xtc))
                    for b in range(0, len(idx), I20.CAUSAL["bs"]):
                        bi = idx[b:b + I20.CAUSAL["bs"]]
                        opt.zero_grad(); mse(ft(Xtc[bi]), Ytc[bi]).backward(); opt.step()
                fts = I28.score(yt, predict(ft, Xt).reshape(-1, 2))
                key = "all" if N is None else f"{int(secs)}s"
                rows[f"{n}ch"][s][key] = {"secs": secs, "n_win": k,
                                          "affine": aff, "finetune": fts}
            printable = {k: v for k, v in rows[f"{n}ch"][s].items()}
            print(f"    {s}:", flush=True)
            for k, v in printable.items():
                print(f"       {k:>6s} ({v['secs']:5.0f}s)  affine R2={v['affine'][0]:+.3f}  "
                      f"finetune R2={v['finetune'][0]:+.3f}/r={v['finetune'][1]:.3f}", flush=True)

        # mean across sessions per size
        keys = ["zero_shot"] + [f"{int(N*WIN_S)}s" for N in SIZES if N] + ["all"]
        print(f"\n  [{n}ch] MEAN over fresh sessions:")
        rows[f"{n}ch"]["mean"] = {}
        for k in keys:
            vals = [rows[f"{n}ch"][s][k] for s in FRESH if k in rows[f"{n}ch"][s]]
            if not vals:
                continue
            ma = float(np.mean([v["affine"][0] for v in vals]))
            mf = float(np.mean([v["finetune"][0] for v in vals]))
            sec = vals[0]["secs"]
            rows[f"{n}ch"]["mean"][k] = {"secs": sec, "n_sessions": len(vals),
                                         "affine_r2": ma, "finetune_r2": mf}
            # n_sessions matters: only rows with 4/4 are comparable across sizes.
            # 'all' is 4/4 but each session contributes a DIFFERENT duration.
            flag = "" if len(vals) == len(FRESH) else f"  <-- ONLY {len(vals)}/{len(FRESH)} sessions, NOT comparable"
            dur = "variable" if k == "all" else f"{sec:5.0f}s"
            print(f"     {k:>6s} ({dur})  affine {ma:+.3f}   finetune {mf:+.3f}"
                  f"  [{len(vals)}/{len(FRESH)} sess]{flag}")
        print(flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
