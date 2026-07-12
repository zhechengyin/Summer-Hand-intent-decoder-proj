#!/usr/bin/env python
"""Generate the 40 ms processed dataset from immutable source MAT files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import models.tcn_gru.evaluate as pipeline

METHOD_DIR = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "data" / "source_data" / "indy_loco"
OUTPUT_DIR = METHOD_DIR / "artifacts"
FILES = [f"train{i}" for i in range(1, 7)] + ["eval1", "test1"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pipeline.DATA = SOURCE_DIR
    pipeline.BIN = 0.040
    pipeline.VEL_LP = 3.0
    pipeline.RATE_SIGMA = 1.0
    manifest = {"method": "bin_40ms", "files": {}}
    for name in FILES:
        out = OUTPUT_DIR / f"{name}.npz"
        if out.exists() and not args.overwrite:
            print(f"skip {out.name} (use --overwrite)")
            continue
        rates, velocity = pipeline.load_source_electrode(name)
        np.savez_compressed(
            out, rates=rates.astype(np.float32), velocity=velocity,
            bin_s=np.float32(pipeline.BIN),
            velocity_lowpass_hz=np.float32(pipeline.VEL_LP),
            rate_smoothing_sigma_bins=np.float32(pipeline.RATE_SIGMA),
        )
        manifest["files"][name] = {
            "source": str((SOURCE_DIR / f"{name}.mat").relative_to(ROOT)),
            "output": str(out.relative_to(ROOT)),
            "rates_shape": list(rates.shape),
            "velocity_shape": list(velocity.shape),
        }
        print(f"wrote {out.name}: rates={rates.shape}, velocity={velocity.shape}")
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
