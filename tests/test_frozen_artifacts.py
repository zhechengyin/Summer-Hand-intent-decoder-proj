"""Small repository guardrails for the promoted firmware artifact."""

from __future__ import annotations

import hashlib
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CHECKPOINT_SHA256 = (
    "87b84cc2c8baf9efdc1ccf37ad28f5f58ad13c4db2a8f8a273fe73fce9956101"
)
CHECKPOINT = (
    REPOSITORY_ROOT
    / "models/finger_movements/cssd_lda/checkpoints"
    / "finger_movements_cssd_lda_phase2c_causal_400ms.npz"
)
FIRMWARE_PARAMETERS = (
    REPOSITORY_ROOT
    / "models/finger_movements/cssd_lda/firmware/src/fm_cssd_lda_params.c"
)


def test_frozen_phase2c_checkpoint_hash() -> None:
    """Prevent accidental replacement of the promoted all-TRAIN checkpoint."""
    assert CHECKPOINT.is_file()
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == (
        EXPECTED_CHECKPOINT_SHA256
    )


def test_firmware_parameters_identify_frozen_checkpoint() -> None:
    """Ensure committed C constants still identify their source checkpoint."""
    assert FIRMWARE_PARAMETERS.is_file()
    assert EXPECTED_CHECKPOINT_SHA256 in FIRMWARE_PARAMETERS.read_text(encoding="utf-8")
