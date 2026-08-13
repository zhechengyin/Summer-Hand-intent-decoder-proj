#!/usr/bin/env python3
"""Compile the firmware C port and compare it with the frozen Python model."""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
FIRMWARE_DIR = THIS_DIR.parent
MODEL_DIR = FIRMWARE_DIR.parent
REPO_ROOT = MODEL_DIR.parents[2]
DEFAULT_CHECKPOINT = (
    MODEL_DIR / "checkpoints" / "finger_movements_cssd_lda_phase2c_causal_400ms.npz"
)
DEFAULT_DATA = REPO_ROOT / "data" / "processed" / "finger_movements" / "train.npz"

sys.path.insert(0, str(REPO_ROOT))
from models.finger_movements.cssd_lda.model import (  # noqa: E402
    FingerMovementsCausalCssdLda,
)


class Output(ctypes.Structure):
    _fields_ = [
        ("class_id", ctypes.c_int32),
        ("score", ctypes.c_float),
        ("probability_right", ctypes.c_float),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--cc", default="cc")
    return parser.parse_args()


def compile_library(compiler: str, output: Path) -> None:
    command = [
        compiler,
        "-std=c99",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-fPIC",
        "-shared",
        "-I",
        str(FIRMWARE_DIR / "include"),
        "-I",
        str(FIRMWARE_DIR / "src"),
        str(FIRMWARE_DIR / "src" / "fm_cssd_lda.c"),
        str(FIRMWARE_DIR / "src" / "fm_cssd_lda_params.c"),
        "-lm",
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    data_path = args.data.resolve()

    subprocess.run(
        [
            sys.executable,
            str(THIS_DIR / "export_checkpoint.py"),
            "--checkpoint",
            str(checkpoint),
        ],
        check=True,
    )
    with tempfile.TemporaryDirectory(prefix="fm_cssd_lda_") as temporary:
        library_path = Path(temporary) / "libfm_cssd_lda.so"
        compile_library(args.cc, library_path)
        library = ctypes.CDLL(str(library_path))

        library.fm_cssd_lda_state_size_bytes.restype = ctypes.c_size_t
        state_size = int(library.fm_cssd_lda_state_size_bytes())
        state_type = ctypes.c_uint8 * state_size
        float_pointer = ctypes.POINTER(ctypes.c_float)
        library.fm_cssd_lda_reset.argtypes = [ctypes.c_void_p, float_pointer]
        library.fm_cssd_lda_reset.restype = ctypes.c_int
        library.fm_cssd_lda_push_block.argtypes = [
            ctypes.c_void_p,
            float_pointer,
            ctypes.c_size_t,
            ctypes.POINTER(Output),
        ]
        library.fm_cssd_lda_push_block.restype = ctypes.c_int

        with np.load(data_path, allow_pickle=False) as data:
            x = data["x"].astype(np.float32)
            channel_names = data["channel_names"].astype(str)
        model = FingerMovementsCausalCssdLda.load(checkpoint)
        python_scores = model.decision_function(x, channel_names)
        python_probabilities = model.predict_proba(x, channel_names)[:, 1]
        python_predictions = model.predict(x, channel_names)

        c_scores = np.empty(len(x), dtype=np.float32)
        c_probabilities = np.empty(len(x), dtype=np.float32)
        c_predictions = np.empty(len(x), dtype=np.int64)
        chunked_mismatches = 0
        for case_index, case in enumerate(x):
            state = state_type()
            first_sample = np.ascontiguousarray(case[:, 0], dtype=np.float32)
            if (
                library.fm_cssd_lda_reset(
                    ctypes.byref(state), first_sample.ctypes.data_as(float_pointer)
                )
                != 0
            ):
                raise RuntimeError("C reset did not enter warm-up state")

            # C consumes sample-major frames; stored NPZ cases are channel-major.
            sample_major = np.ascontiguousarray(case.T, dtype=np.float32)
            output = Output()
            status = library.fm_cssd_lda_push_block(
                ctypes.byref(state),
                sample_major.ctypes.data_as(float_pointer),
                sample_major.shape[0],
                ctypes.byref(output),
            )
            if status != 1:
                raise RuntimeError(f"C case {case_index} did not produce a prediction")
            c_scores[case_index] = output.score
            c_probabilities[case_index] = output.probability_right
            c_predictions[case_index] = output.class_id

            chunked_state = state_type()
            if (
                library.fm_cssd_lda_reset(
                    ctypes.byref(chunked_state),
                    first_sample.ctypes.data_as(float_pointer),
                )
                != 0
            ):
                raise RuntimeError("Chunked C reset did not enter warm-up state")
            chunked_output = Output()
            for start in range(0, sample_major.shape[0], 5):
                chunk = np.ascontiguousarray(sample_major[start : start + 5])
                chunk_status = library.fm_cssd_lda_push_block(
                    ctypes.byref(chunked_state),
                    chunk.ctypes.data_as(float_pointer),
                    chunk.shape[0],
                    ctypes.byref(chunked_output),
                )
                expected_status = 1 if start == 45 else 0
                if chunk_status != expected_status:
                    raise RuntimeError(
                        f"Unexpected chunk status for case {case_index}, start {start}: "
                        f"expected {expected_status}, got {chunk_status}"
                    )
            if (
                chunked_output.class_id != output.class_id
                or chunked_output.score != output.score
                or chunked_output.probability_right != output.probability_right
            ):
                chunked_mismatches += 1

        prediction_mismatches = int(
            np.count_nonzero(c_predictions != python_predictions)
        )
        max_score_error = float(
            np.max(np.abs(c_scores.astype(np.float64) - python_scores))
        )
        max_probability_error = float(
            np.max(np.abs(c_probabilities.astype(np.float64) - python_probabilities))
        )
        if prediction_mismatches != 0:
            raise RuntimeError(
                f"C/Python prediction mismatches: {prediction_mismatches}"
            )
        if chunked_mismatches != 0:
            raise RuntimeError(
                f"Single-block/chunked C mismatches: {chunked_mismatches}"
            )
        if max_score_error > 5e-4 or max_probability_error > 1e-4:
            raise RuntimeError(
                "C/Python numerical error exceeds the frozen float32 tolerance: "
                f"score={max_score_error}, probability={max_probability_error}"
            )

        report = {
            "cases": int(len(x)),
            "prediction_mismatches": prediction_mismatches,
            "single_block_vs_ten_50ms_chunks_mismatches": chunked_mismatches,
            "maximum_score_absolute_error": max_score_error,
            "maximum_probability_absolute_error": max_probability_error,
            "persistent_state_bytes": state_size,
            "checkpoint": str(checkpoint),
            "data": str(data_path),
        }
        print(json.dumps(report, indent=2))
        print("firmware C validation=PASS")


if __name__ == "__main__":
    main()
