#!/usr/bin/env python
"""Generate unsmoothed 40 ms count artifacts from immutable Indy/Loco MAT files."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.intent_decoder.data.indy import load_counts_velocity, load_session_manifest, session_path

OUTPUT_DIR = ROOT / "data" / "processed" / "indy_loco" / "bin_40ms_causal_counts"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", nargs="+", help="Aliases or original Zenodo stems")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    registry = load_session_manifest()
    sessions = args.sessions or list(registry["aliases"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": "bin_40ms_causal_counts",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bin_s": 0.040,
        "velocity_lowpass_hz": 3.0,
        "velocity_filter": "causal_forward_butterworth",
        "velocity_difference": "backward",
        "input_smoothing": None,
        "files": {},
    }
    for session in sessions:
        output = OUTPUT_DIR / f"{session}.npz"
        if output.exists() and not args.overwrite:
            print(f"skip {output.name} (use --overwrite)")
            continue
        counts, velocity = load_counts_velocity(session)
        np.savez_compressed(
            output,
            counts=counts.astype(np.float32),
            velocity=velocity.astype(np.float32),
            bin_s=np.float32(0.040),
            velocity_lowpass_hz=np.float32(3.0),
            velocity_filter=np.asarray("causal_forward_butterworth"),
            velocity_difference=np.asarray("backward"),
        )
        source = session_path(session)
        manifest["files"][session] = {
            "source": str(source.relative_to(ROOT)),
            "output": str(output.relative_to(ROOT)),
            "counts_shape": list(counts.shape),
            "velocity_shape": list(velocity.shape),
        }
        print(f"wrote {output.name}: counts={counts.shape}, velocity={velocity.shape}")
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
