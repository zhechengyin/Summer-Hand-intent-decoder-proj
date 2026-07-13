#!/usr/bin/env python
"""Iteration 14: lightweight-decoder architecture comparison (ARCHITECTURE_EXPERIMENT.md).

Fixed: 24-session data, eval1/test1, 8 firing channels, 40ms/2s/3Hz, seed/epochs/
optimizer/aug. Vary only the architecture. Report TEST R², ~int8 size, inference
latency (ms/forward, batch 1, 1 thread), and causal/lookahead label.
Usage: py research/iter14_architecture.py
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
import research.iter5_scale as I5
import research.iter7_final as I7
import research.architectures as A
import models.tcn_gru.best_model as M
import models.tcn_gru.evaluate as E

BASE = {**E.CFG, "F": 64, "H": 64, "L": 1, "dils": [1, 2, 4, 8]}   # comparable 'wide' size

# (name, build_fn, cfg_overrides, causal?)
MODELS = [
    ("tcngru_bidir",  M.build_net,        {"bidir": True},  False),   # model 1 (ref)
    ("tcngru_causal", M.build_net,        {"bidir": False}, True),    # model 5
    ("causal_tcn",    A.build_causal_tcn, {},               True),    # model 2
    ("dws_tcn",       A.build_dws_tcn,    {},               True),    # model 3
    ("gru_bidir",     A.build_gru_only,   {"bidir": True},  False),   # model 4 (bidir)
    ("gru_causal",    A.build_gru_only,   {"bidir": False}, True),    # model 4 (causal)
]


def latency_ms(build_fn, cfg, n_ch=8, T=50, n=200):
    import torch
    torch.set_num_threads(1)
    net = build_fn(cfg, n_ch); net.eval()
    x = torch.randn(1, n_ch, T)
    with torch.no_grad():
        for _ in range(15):
            net(x)
        t0 = time.perf_counter()
        for _ in range(n):
            net(x)
    return (time.perf_counter() - t0) / n * 1000


def main():
    data = I5.prep_more(I7.EXTRA18[:18], reselect=False)          # 24 sessions, fixed 8 ch
    print("=== Iteration 14: architecture comparison (8 ch, 24 sess, F64/H64) ===")
    print(f"  {'model':15s} {'R2':>6s} {'int8KB':>7s} {'lat_ms':>7s}  causal\n", flush=True)
    DONE = {"tcngru_bidir", "tcngru_causal", "causal_tcn"}   # completed in LOG-059 partial
    rows = {}
    for name, build_fn, ov, causal in MODELS:
        if name in DONE:
            continue
        cfg = {**BASE, **ov}
        t0 = time.time()
        res = H.run(data, cfg, build=build_fn)
        lat = latency_ms(build_fn, {**cfg, "n_out": 2})
        int8kb = res["n_params"] / 1024
        rows[name] = {"test_r2": res["test_r2"], "eval_r2": res["eval_r2"],
                      "params": res["n_params"], "int8_kb": int8kb,
                      "latency_ms": lat, "causal": causal}
        print(f"  {name:15s} {res['test_r2']:6.3f} {int8kb:7.0f} {lat:7.2f}  "
              f"{'yes' if causal else 'NO (2s ahead)'}  [{time.time()-t0:.0f}s]", flush=True)

    print("\n--- summary (Test R2 | int8 KB | latency ms | causal) ---")
    best_causal = max((r for r in rows.values() if r["causal"]), key=lambda r: r["test_r2"])
    for name, r in rows.items():
        tag = "  <- best causal" if r is best_causal else ""
        print(f"  {name:15s}: {r['test_r2']:.3f} | {r['int8_kb']:.0f} KB | "
              f"{r['latency_ms']:.2f} ms | {'causal' if r['causal'] else 'non-causal'}{tag}")
    out = ROOT / "results" / "metrics" / "iter14_architecture.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
