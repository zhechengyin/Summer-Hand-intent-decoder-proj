#!/usr/bin/env python3
"""Export the frozen Phase 2c NPZ checkpoint as deterministic float32 C data."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from scipy.signal import sosfilt_zi

EXPECTED_CHECKPOINT_SHA256 = (
    "87b84cc2c8baf9efdc1ccf37ad28f5f58ad13c4db2a8f8a273fe73fce9956101"
)
EXPECTED_FORMAT = "finger_movements_causal_cssd_lda_v2"
CHANNELS = 28
TREND_CHANNELS = 19

THIS_DIR = Path(__file__).resolve().parent
FIRMWARE_DIR = THIS_DIR.parent
MODEL_DIR = FIRMWARE_DIR.parent
DEFAULT_CHECKPOINT = (
    MODEL_DIR / "checkpoints" / "finger_movements_cssd_lda_phase2c_causal_400ms.npz"
)
DEFAULT_OUTPUT = FIRMWARE_DIR / "src"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def c_float(value: float) -> str:
    """Emit an exact C99 hexadecimal literal for one IEEE-754 float32."""
    scalar = np.float32(value)
    if not np.isfinite(scalar):
        raise ValueError("Firmware parameter is not finite")
    if scalar == 0.0:
        return "-0x0p+0f" if np.signbit(scalar) else "0x0p+0f"
    return float(scalar).hex() + "f"


def emit_1d(name: str, values: np.ndarray) -> str:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    body = ",\n    ".join(c_float(value) for value in flat)
    return f"const float {name}[{len(flat)}u] = {{\n    {body}\n}};\n"


def emit_2d(name: str, values: np.ndarray) -> str:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{name} is not two-dimensional")
    rows = []
    for row in matrix:
        rows.append("    {" + ", ".join(c_float(value) for value in row) + "}")
    return (
        f"const float {name}[{matrix.shape[0]}u][{matrix.shape[1]}u] = {{\n"
        + ",\n".join(rows)
        + "\n};\n"
    )


def folded_linear_state(
    checkpoint: np.lib.npyio.NpzFile, prefix: str
) -> tuple[np.ndarray, float]:
    mean = checkpoint[f"{prefix}_mean"].astype(np.float64)
    scale = checkpoint[f"{prefix}_scale"].astype(np.float64)
    coefficient = checkpoint[f"{prefix}_coefficient"].astype(np.float64)
    if np.any(scale <= 0.0):
        raise ValueError(f"{prefix} contains a non-positive feature scale")
    weights = coefficient / scale
    bias = float(checkpoint[f"{prefix}_intercept"]) - float(mean @ weights)
    return weights.astype(np.float32), float(np.float32(bias))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output.resolve()
    digest = sha256(checkpoint_path)
    if digest != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            "Refusing to export an unrecognized checkpoint: "
            f"expected {EXPECTED_CHECKPOINT_SHA256}, got {digest}"
        )

    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        if str(checkpoint["format_version"]) != EXPECTED_FORMAT:
            raise ValueError("Unsupported checkpoint format")
        channel_names = checkpoint["channel_names"].astype(str)
        trend_indices = checkpoint["trend_indices"].astype(np.uint8)
        bp_sos = checkpoint["bp_sos"].astype(np.float32)
        erd_sos = checkpoint["erd_sos"].astype(np.float32)
        bp_filters = checkpoint["bp_filters"].astype(np.float32)
        erd_filters = checkpoint["erd_filters"].astype(np.float32)
        if channel_names.shape != (CHANNELS,):
            raise ValueError("Checkpoint channel count changed")
        if trend_indices.shape != (TREND_CHANNELS,):
            raise ValueError("Checkpoint trend-channel count changed")
        if bp_sos.shape != (2, 6) or erd_sos.shape != (4, 6):
            raise ValueError("Checkpoint SOS layout changed")
        if bp_filters.shape != (2, CHANNELS) or erd_filters.shape != (2, CHANNELS):
            raise ValueError("Checkpoint CSSD layout changed")
        if not np.allclose(bp_sos[:, 3], 1.0) or not np.allclose(erd_sos[:, 3], 1.0):
            raise ValueError("C implementation requires normalized SOS a0=1")

        bp_zi = sosfilt_zi(checkpoint["bp_sos"].astype(np.float64)).astype(np.float32)
        erd_zi = sosfilt_zi(checkpoint["erd_sos"].astype(np.float64)).astype(np.float32)
        bp_weights, bp_bias = folded_linear_state(checkpoint, "bp_branch")
        erd_weights, erd_bias = folded_linear_state(checkpoint, "erd_branch")
        trend_weights, trend_bias = folded_linear_state(checkpoint, "trend_branch")
        fusion_weights, fusion_bias = folded_linear_state(checkpoint, "fusion")

    output_dir.mkdir(parents=True, exist_ok=True)
    header = """/* Generated by tools/export_checkpoint.py. Do not edit manually. */
#ifndef FM_CSSD_LDA_PARAMS_H
#define FM_CSSD_LDA_PARAMS_H

#include <stdint.h>
#include "fm_cssd_lda.h"

extern const float fm_bp_sos[FM_CSSD_LDA_BP_SOS_SECTIONS][6u];
extern const float fm_erd_sos[FM_CSSD_LDA_ERD_SOS_SECTIONS][6u];
extern const float fm_bp_initial_state[FM_CSSD_LDA_BP_SOS_SECTIONS][2u];
extern const float fm_erd_initial_state[FM_CSSD_LDA_ERD_SOS_SECTIONS][2u];
extern const float fm_bp_spatial_filters[2u][FM_CSSD_LDA_CHANNELS];
extern const float fm_erd_spatial_filters[2u][FM_CSSD_LDA_CHANNELS];
extern const uint8_t fm_trend_indices[19u];
extern const float fm_bp_lda_weights[8u];
extern const float fm_bp_lda_bias;
extern const float fm_erd_lda_weights[8u];
extern const float fm_erd_lda_bias;
extern const float fm_trend_lda_weights[38u];
extern const float fm_trend_lda_bias;
extern const float fm_fusion_weights[3u];
extern const float fm_fusion_bias;

#endif
"""

    source_parts = [
        "/* Generated by tools/export_checkpoint.py. Do not edit manually. */\n",
        '#include "fm_cssd_lda_params.h"\n\n',
        f'const char fm_cssd_lda_checkpoint_sha256[] = "{digest}";\n\n',
        "const char *const fm_cssd_lda_channel_names[FM_CSSD_LDA_CHANNELS] = {\n",
        "    " + ", ".join(f'"{name}"' for name in channel_names) + "\n};\n\n",
        emit_2d("fm_bp_sos", bp_sos),
        "\n",
        emit_2d("fm_erd_sos", erd_sos),
        "\n",
        emit_2d("fm_bp_initial_state", bp_zi),
        "\n",
        emit_2d("fm_erd_initial_state", erd_zi),
        "\n",
        emit_2d("fm_bp_spatial_filters", bp_filters),
        "\n",
        emit_2d("fm_erd_spatial_filters", erd_filters),
        "\n",
        "const uint8_t fm_trend_indices[19u] = {\n    ",
        ", ".join(f"{int(value)}u" for value in trend_indices),
        "\n};\n\n",
        emit_1d("fm_bp_lda_weights", bp_weights),
        f"\nconst float fm_bp_lda_bias = {c_float(bp_bias)};\n\n",
        emit_1d("fm_erd_lda_weights", erd_weights),
        f"\nconst float fm_erd_lda_bias = {c_float(erd_bias)};\n\n",
        emit_1d("fm_trend_lda_weights", trend_weights),
        f"\nconst float fm_trend_lda_bias = {c_float(trend_bias)};\n\n",
        emit_1d("fm_fusion_weights", fusion_weights),
        f"\nconst float fm_fusion_bias = {c_float(fusion_bias)};\n",
    ]
    (output_dir / "fm_cssd_lda_params.h").write_text(header, encoding="utf-8")
    (output_dir / "fm_cssd_lda_params.c").write_text(
        "".join(source_parts), encoding="utf-8"
    )
    print(f"checkpoint SHA-256={digest}")
    print(f"wrote {output_dir / 'fm_cssd_lda_params.h'}")
    print(f"wrote {output_dir / 'fm_cssd_lda_params.c'}")


if __name__ == "__main__":
    main()
