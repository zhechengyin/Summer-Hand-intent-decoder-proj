#!/usr/bin/env python
"""Iteration 6: int8 quantization -- does the STM32-sized model survive int8?

Post-training per-output-channel symmetric int8 on all weight matrices (conv,
GRU, linear -- the TFLite convention), biases/BatchNorm left fp (they fold/are
int32 and negligible). Measures TEST R^2 fp32 vs int8 and the int8 weight size.
This estimates the STM32 accuracy/size story without a full TFLite export.

Uses the best data config from iter5 (18 sessions). Usage: py experiments/archive/indy/iter6_quant.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.common.harness as H
import experiments.archive.indy.iter5_scale as I5
import models.tcn_gru.best_model as M
import models.tcn_gru.evaluate as E

N_EXTRA = 12          # 18-session pool (set from iter5 best)
RESELECT = False      # channel selection source (set from iter5 best)


def quant_int8(w):
    import torch
    if w.dim() >= 2:                                   # per-output-channel (axis 0)
        wf = w.reshape(w.shape[0], -1)
        scale = (wf.abs().amax(1, keepdim=True) / 127.0)
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        return (torch.round(wf / scale).clamp(-127, 127) * scale).reshape(w.shape)
    s = w.abs().max() / 127.0
    return w if s == 0 else torch.round(w / s).clamp(-127, 127) * s


def score_r2(net, part, ys, ym, n_out):
    import torch
    net.eval()
    r2 = []
    for name, (Xe, Ye) in part.items():
        with torch.no_grad():
            pr = net(torch.tensor(Xe)).numpy() * ys + ym
        r2.append(M.r2(Ye.reshape(-1, n_out), pr.reshape(-1, n_out)))
    return float(np.mean(r2))


def main():
    import torch
    cfg = {**E.CFG, **I5.SMALL}
    data = I5.prep_more(I5.ALL_EXTRA[:N_EXTRA], reselect=RESELECT)
    print(f"=== Iteration 6: int8 quantization ({6+N_EXTRA} sess, reselect={RESELECT}) ===\n",
          flush=True)
    res = H.run(data, cfg, ret_net=True)
    net = res["net"]; ym, ys = res["norm"]
    ev_p, te_p, n_out = res["ev_p"], res["te_p"], res["n_out"]

    fp32_eval = score_r2(net, ev_p, ys, ym, n_out)
    fp32_test = score_r2(net, te_p, ys, ym, n_out)

    n_w = 0                                            # quantizable weight count
    with torch.no_grad():
        for name, p in net.named_parameters():
            if "weight" in name and p.dim() >= 2:
                p.data = quant_int8(p.data); n_w += p.numel()
    int8_eval = score_r2(net, ev_p, ys, ym, n_out)
    int8_test = score_r2(net, te_p, ys, ym, n_out)

    n_tot = res["n_params"]
    print(f"  params: {n_tot:,} (quantized weights {n_w:,})")
    print(f"  size:  fp32 {n_tot*4/1024:.0f} kB  ->  int8 ~{(n_w + (n_tot-n_w)*4)/1024:.0f} kB")
    print(f"  EVAL R2: fp32 {fp32_eval:.3f} -> int8 {int8_eval:.3f}  ({int8_eval-fp32_eval:+.3f})")
    print(f"  TEST R2: fp32 {fp32_test:.3f} -> int8 {int8_test:.3f}  ({int8_test-fp32_test:+.3f})",
          flush=True)
    out = ROOT / "results" / "metrics" / "iter6_quant.json"
    out.write_text(json.dumps({"sessions": 6 + N_EXTRA, "reselect": RESELECT,
                               "params": n_tot, "quant_weights": n_w,
                               "int8_kb": (n_w + (n_tot - n_w) * 4) / 1024,
                               "fp32_test_r2": fp32_test, "int8_test_r2": int8_test,
                               "fp32_eval_r2": fp32_eval, "int8_eval_r2": int8_eval},
                              indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
