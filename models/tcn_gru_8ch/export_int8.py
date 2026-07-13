#!/usr/bin/env python
"""Load the saved checkpoint, quantize weights to int8, and report the R² kept.

No retraining -- operates on the deployed weights. Confirms the int8 story for
whatever config is in the checkpoint. Usage: py models/tcn_gru_8ch/export_int8.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import models.tcn_gru.best_model as M
import models.tcn_gru_8ch.evaluate as EV
import models.tcn_gru.evaluate as E96


def main():
    import torch
    ck = torch.load(ROOT / "models" / "tcn_gru_8ch" / "checkpoint.pt", weights_only=False)
    sel = np.array(ck["channels"]); axes = np.array(ck["axes"])
    ym, ys = np.array(ck["target_mean"]), np.array(ck["target_std"])
    net = M.build_net(ck["config"], len(sel))
    net.load_state_dict(ck["state_dict"]); net.eval()
    te = {s: EV._pack(EV.wins(s, sel, axes)) for s in E96.TEST}

    def score():
        r2 = []
        for Xe, Ye in te.values():
            with torch.no_grad():
                pr = net(torch.tensor(Xe)).numpy() * ys + ym
            r2.append(M.r2(Ye.reshape(-1, Ye.shape[-1]), pr.reshape(-1, Ye.shape[-1])))
        return float(np.mean(r2))
    fp32 = score()
    n = sum(p.numel() for p in net.parameters()); nq = 0
    with torch.no_grad():
        for name, p in net.named_parameters():
            if "weight" in name and p.dim() >= 2:
                p.data = EV.quant_int8(p.data); nq += p.numel()
    int8 = score()
    print(f"checkpoint {ck['params']:,} params | fp32 {n*4/1024:.0f} kB -> "
          f"int8 ~{(nq+(n-nq)*4)/1024:.0f} kB")
    print(f"TEST R2: fp32 {fp32:.3f} -> int8 {int8:.3f}  ({int8-fp32:+.4f})", flush=True)


if __name__ == "__main__":
    main()
