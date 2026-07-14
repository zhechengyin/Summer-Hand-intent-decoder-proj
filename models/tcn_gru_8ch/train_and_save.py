#!/usr/bin/env python
"""Train the 8-channel STM32 decoder and save a deployable checkpoint.

Writes checkpoint.pt with: fp32 state_dict, the selected 8 channel indices, the
movement axes, target normalization (ym, ys), the config, and the measured R².
The int8 form (~27 kB, lossless) is produced at export time from these weights.

Usage: py models/tcn_gru_8ch/train_and_save.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import models.tcn_gru_8ch.evaluate as EV
import models.tcn_gru_8ch.config as C
import models.tcn_gru.evaluate as E96


def main():
    import torch
    t0 = time.time()
    sel, axes = EV.select_channels(), EV.movement_axes()
    tr = [w for s in C.BASE_TRAIN + C.EXTRA_TRAIN for w in EV.wins(s, sel, axes)]
    ev = {s: EV._pack(EV.wins(s, sel, axes)) for s in E96.EVAL}
    te = {s: EV._pack(EV.wins(s, sel, axes)) for s in E96.TEST}
    net, ym, ys = EV.train(tr, ev, C.MODEL)
    test_r2 = EV.score(net, te, ym, ys)
    n = sum(p.numel() for p in net.parameters())
    ckpt = ROOT / "models" / "tcn_gru_8ch" / "checkpoint.pt"
    torch.save({"state_dict": net.state_dict(), "config": C.MODEL,
                "channels": sel.tolist(), "axes": axes.tolist(),
                "multiscale": C.MULTISCALE,          # raw+EWMA scales -> 16 input features
                "target_mean": ym.tolist(), "target_std": ys.tolist(),
                "n_channels": C.N_CHANNELS, "test_r2": test_r2,
                "params": int(n)}, ckpt)
    print(f"saved {ckpt}  ({n:,} params, {n*4/1024:.0f} kB fp32 / ~{n/1024:.0f} kB int8, "
          f"TEST R2={test_r2:.3f})  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
