"""Integrity tests for the six generated Cube.AI session bundles."""

from __future__ import annotations

import importlib.util
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VERIFIER_PATH = (
    REPOSITORY_ROOT / "indy_loco" / "deploy" / "verify_session_cubeai.py"
)
SPEC = importlib.util.spec_from_file_location("verify_session_cubeai", VERIFIER_PATH)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def test_all_session_cubeai_bundles() -> None:
    results = [VERIFIER.verify_session(session) for session in VERIFIER.SESSIONS]
    assert len(results) == 6
    assert {result["source_channels"] for result in results} == {96, 192}
    assert {result["encoder_bytes"] for result in results} == {248_156}
    assert {result["gru_head_bytes"] for result in results} == {100_360}
    assert max(result["chain_max_abs_error"] for result in results) <= 1.0e-5
