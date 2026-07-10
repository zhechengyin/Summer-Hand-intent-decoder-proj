#!/usr/bin/env python
"""How much does decoding lose with FEWER electrodes? (hardware = 8 channels)

Cross-session held-out (train 6 indy sessions, test 2 held-out), same as
indy_crosssession, but restrict to the top-N electrodes (selected by mean firing
rate on the TRAIN sessions -- a standard channel-selection proxy). Sweep N to see
the 8-channel cost vs the full 96.

Usage: py tools/indy_nch.py
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

import frontier.crosssession as X

NS = [8, 16, 32, 96]


def main():
    print("=== electrode-count sweep (held-out cross-session) ===")
    print(f"TRAIN {[s[5:] for s in X.TRAIN]}  TEST {[s[5:] for s in X.TEST]}\n",
          flush=True)
    loaded = {s: X.load_electrode(s) for s in X.TRAIN + X.TEST}
    # channel importance = mean firing rate over TRAIN
    fr = np.mean([loaded[s][0].mean(1) for s in X.TRAIN], 0)          # (96,)
    var = np.mean([loaded[s][1].std(0) for s in X.TRAIN], 0)
    axes = np.sort(np.argsort(var)[-2:])                             # 2D movement axes

    report = {}
    for N in NS:
        sel = np.sort(np.argsort(fr)[-N:])                          # top-N electrodes
        tr = []
        for s in X.TRAIN:
            tr += X.windows(loaded[s][0][sel], loaded[s][1], axes)
        te = {s: X.windows(loaded[s][0][sel], loaded[s][1], axes) for s in X.TEST}
        t0 = time.time()
        res = X.train_eval(tr, te, X.CFG)
        per = {s[5:]: float(r.mean()) for s, r in res.items()}
        mean = float(np.mean(list(per.values())))
        report[N] = {"mean_r": mean, "per_session": per}
        print(f"N={N:<3d} electrodes: held-out mean r = {mean:.3f}  {per}  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    print("\n--- electrode-count vs accuracy ---")
    for N in NS:
        print(f"  {N:>3d} ch: {report[N]['mean_r']:.3f}")
    out = ROOT / "results" / "metrics" / "indy_nch.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
