#!/usr/bin/env python3
"""Build and validate the twelve canonical session model packages."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARCHIVE = PROJECT / "history" / "legacy_pre_package_cleanup"
PHASE12 = ARCHIVE / "experiments" / "phase12_external_memory_representation_ablation"
SESSIONS: Final = (
    "indy_20160622_01",
    "indy_20160630_01",
    "indy_20170131_02",
    "loco_20170210_03",
    "loco_20170215_02",
    "loco_20170301_05",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build() -> None:
    packages: list[dict[str, Any]] = []
    for session in SESSIONS:
        replay = json.loads(
            (ARCHIVE / "models" / "midsize" / session / "deployment_replay.json").read_text()
        )
        metrics = json.loads(
            (
                PHASE12
                / "results"
                / "deployment_parity"
                / "by_session"
                / session
                / "metrics.json"
            ).read_text()
        )
        midsize_dir = ROOT / "midsize" / session
        large_dir = ROOT / "large" / session
        checkpoint = midsize_dir / "checkpoint.pt"
        checkpoint_hash = sha256(checkpoint)
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        common = {
            "schema_version": 1,
            "session": session,
            "subject": metrics["subject"],
            "selection": {
                "policy": "highest_phase7_test_r2_fold",
                "selected_fold": int(metrics["selected_fold"]),
                "phase7_chunked_test_r2_mean": float(metrics["selection_test_r2_mean"]),
                "selection_bias_warning": (
                    "The fold was chosen by test R2. This package is a GUI/deployment "
                    "demonstration candidate, not an unbiased generalization estimate."
                ),
            },
            "model": {
                "architecture": "MidsizeTCNGRU",
                "parameters": 86_978,
                "input_features": 192,
                "physical_channels": 96,
                "source_channels": int(checkpoint_data["source_channel_count"]),
                "selected_channel_indices": [
                    int(value) for value in checkpoint_data["selected_channel_indices"]
                ],
                "window_bins": 50,
                "bin_seconds": 0.04,
                "checkpoint": "checkpoint.pt",
                "checkpoint_sha256": checkpoint_hash,
            },
            "preprocessing": {
                "ewma_alpha": 0.1,
                "calibration_bins": 1_500,
                "calibration_seconds": 60.0,
                "continuous_rolling_window": True,
                "prediction_timestep": 49,
            },
        }
        midsize_manifest = {
            **common,
            "tier": "midsize",
            "package_status": "best-test-fold demonstration candidate",
            "held_out_replay": {
                "policy": "continuous rolling, bank ABSENT",
                **metrics["bank_absent"],
            },
        }
        write_json(midsize_dir / "manifest.json", midsize_manifest)

        memory = large_dir / "memory.memlib"
        with np.load(memory, allow_pickle=False) as archive:
            entry_count = int(archive["keys_int8"].shape[0])
            key_dims = int(archive["keys_int8"].shape[1])
            representation = str(archive["representation"])
            schema = str(archive["schema"])
        tuning = metrics["retrieval"]["tuning"]
        large_manifest = {
            **common,
            "tier": "large",
            "definition": "the same best-fold Midsize base plus GRU residual memory",
            "package_status": "PC external-memory demonstration candidate",
            "held_out_replay": {
                "policy": "continuous rolling, GRU bank READY",
                **metrics["bank_ready_gru"],
                "ready_minus_absent_r2_mean": metrics["ready_minus_absent"]["r2_mean"],
            },
            "memory": {
                "file": "memory.memlib",
                "sha256": sha256(memory),
                "schema": schema,
                "representation": representation,
                "representation_pca_dims": 32,
                "context_pca_dims": 32,
                "key_dims": key_dims,
                "key_storage": "int8",
                "residual_storage": "float16",
                "entries": entry_count,
                "neighbours": int(tuning["neighbours"]),
                "temperature": float(tuning["temperature"]),
                "blend": float(tuning["blend"]),
                "search_used_for_reported_score": "PC exact cKDTree",
                "firmware_bcimem_compatible": False,
            },
        }
        write_json(large_dir / "manifest.json", large_manifest)
        packages.extend(
            {
                "tier": tier,
                "session": session,
                "manifest": f"{tier}/{session}/manifest.json",
            }
            for tier in ("midsize", "large")
        )

    write_json(
        ROOT / "manifest.json",
        {
            "schema_version": 1,
            "authoritative": True,
            "package_count": 12,
            "sessions": list(SESSIONS),
            "tiers": {
                "midsize": "best-test-fold Midsize base, bank ABSENT",
                "large": "same base plus GRU residual memory, bank READY",
            },
            "packages": packages,
        },
    )


def validate() -> None:
    index = json.loads((ROOT / "manifest.json").read_text())
    assert index["authoritative"] is True
    assert index["package_count"] == 12
    assert tuple(index["sessions"]) == SESSIONS
    allowed = {
        "midsize": {"checkpoint.pt", "manifest.json"},
        "large": {"checkpoint.pt", "memory.memlib", "manifest.json"},
    }
    for session in SESSIONS:
        checkpoint_hashes = []
        for tier in ("midsize", "large"):
            directory = ROOT / tier / session
            actual = {path.name for path in directory.iterdir() if path.is_file()}
            assert actual == allowed[tier], (tier, session, actual)
            manifest = json.loads((directory / "manifest.json").read_text())
            assert manifest["session"] == session and manifest["tier"] == tier
            checkpoint = directory / manifest["model"]["checkpoint"]
            assert sha256(checkpoint) == manifest["model"]["checkpoint_sha256"]
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert payload["session"] == session
            assert int(payload["fold"]) == manifest["selection"]["selected_fold"]
            assert payload["selection_policy"] == "highest_phase7_test_r2_fold"
            checkpoint_hashes.append(sha256(checkpoint))
            if tier == "large":
                memory = directory / manifest["memory"]["file"]
                assert sha256(memory) == manifest["memory"]["sha256"]
                with np.load(memory, allow_pickle=False) as archive:
                    assert str(archive["schema"]) == "phase12_pc_memlib_v1"
                    assert str(archive["representation"]) == "deployment_parity_gru_hidden_49"
                    assert archive["keys_int8"].shape == (manifest["memory"]["entries"], 64)
                    assert archive["keys_int8"].dtype == np.int8
                    assert archive["residual_fp16"].shape == (manifest["memory"]["entries"], 2)
                    assert archive["residual_fp16"].dtype == np.float16
        assert checkpoint_hashes[0] == checkpoint_hashes[1]
    print("Package validation passed: 6 sessions x 2 tiers = 12 canonical packages")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"))
    args = parser.parse_args()
    build() if args.command == "build" else validate()


if __name__ == "__main__":
    main()
