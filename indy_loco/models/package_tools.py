#!/usr/bin/env python3
"""Build and validate the final Phase-13 pre-CubeAI model package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Final

import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ROUND3 = (
    PROJECT
    / "experiment"
    / "phase13_deployment_validation"
    / "results"
    / "rolling_retrain"
    / "final_30fold"
)
SOURCE_CHECKPOINTS = ROUND3 / "checkpoints"
FOLD_METRICS = ROUND3 / "phase13_round3_folds.csv"
SUMMARY_METRICS = ROUND3 / "phase13_round3_summary.csv"
SESSIONS: Final = (
    "indy_20160622_01",
    "indy_20160630_01",
    "indy_20170131_02",
    "loco_20170210_03",
    "loco_20170215_02",
    "loco_20170301_05",
)
TIERS: Final = ("midsize", "large")
FOLDS: Final = (1, 2, 3, 4, 5)
SELECTION_POLICY = "minimum_validation_loss_test_opened_once"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def fold_filename(fold: int, best_fold: int) -> str:
    suffix = "_best-test-fold" if fold == best_fold else ""
    return f"fold-{fold}{suffix}.pt"


def fold_rows() -> dict[str, list[dict[str, str]]]:
    rows = read_rows(FOLD_METRICS)
    grouped = {session: [] for session in SESSIONS}
    for row in rows:
        session = row["session"]
        if session not in grouped:
            raise ValueError(f"Unexpected session in fold metrics: {session}")
        grouped[session].append(row)
    for session, session_rows in grouped.items():
        folds = sorted(int(row["fold"]) for row in session_rows)
        if folds != list(FOLDS):
            raise ValueError(f"{session}: expected folds 1..5, found {folds}")
    return grouped


def summary_rows() -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    rows = read_rows(SUMMARY_METRICS)
    sessions = {row["session"]: row for row in rows if row["session"] in SESSIONS}
    overall = next(row for row in rows if row["session"] == "overall_fold_macro")
    if set(sessions) != set(SESSIONS):
        raise ValueError("Round-3 summary does not contain exactly six sessions")
    return sessions, overall


def checkpoint_record(
    session: str,
    row: dict[str, str],
    best_fold: int,
    destination: Path,
) -> dict[str, Any]:
    fold = int(row["fold"])
    source = SOURCE_CHECKPOINTS / f"{session}_fold{fold}.pt"
    if not source.is_file():
        raise FileNotFoundError(source)
    filename = fold_filename(fold, best_fold)
    target = destination / filename
    shutil.copy2(source, target)
    payload = torch.load(target, map_location="cpu", weights_only=False)
    if payload.get("session") != session or int(payload.get("fold", -1)) != fold:
        raise ValueError(f"Checkpoint identity mismatch: {target}")
    if payload.get("selection_policy") != SELECTION_POLICY:
        raise ValueError(f"Checkpoint was not selected by validation loss: {target}")
    if payload.get("test_evaluated_during_training") is not False:
        raise ValueError(f"Checkpoint opened test data during training: {target}")
    return {
        "fold": fold,
        "file": filename,
        "sha256": sha256(target),
        "best_test_fold": fold == best_fold,
        "best_epoch": int(row["best_epoch"]),
        "test_r2_mean": float(row["retrained_7min_rolling_r2"]),
        "test_bins": int(row["test_bins"]),
        "selection_policy": SELECTION_POLICY,
        "source": str(source.relative_to(PROJECT)),
    }


def build() -> None:
    grouped = fold_rows()
    summaries, overall = summary_rows()
    packages: list[dict[str, Any]] = []

    for session in SESSIONS:
        rows = sorted(grouped[session], key=lambda row: int(row["fold"]))
        best_row = max(rows, key=lambda row: float(row["retrained_7min_rolling_r2"]))
        best_fold = int(best_row["fold"])
        session_summary = summaries[session]

        for tier in TIERS:
            directory = ROOT / tier / session
            directory.mkdir(parents=True, exist_ok=True)
            unexpected = [path for path in directory.iterdir() if path.is_file()]
            if unexpected:
                names = ", ".join(sorted(path.name for path in unexpected))
                raise ValueError(f"Refusing to overwrite non-empty {directory}: {names}")

            checkpoints = [
                checkpoint_record(session, row, best_fold, directory) for row in rows
            ]
            first_payload = torch.load(
                directory / checkpoints[0]["file"],
                map_location="cpu",
                weights_only=False,
            )
            manifest: dict[str, Any] = {
                "schema_version": 2,
                "phase": "phase13_round3_final_pre_cubeai",
                "session": session,
                "subject": first_payload["subject"],
                "tier": tier,
                "package_status": "final_python_checkpoints_cubeai_pending",
                "paper_reporting": {
                    "primary_metric": "mean test R2 across five validation-selected folds",
                    "folds": 5,
                    "test_r2_mean": float(
                        session_summary["retrained_7min_rolling_r2_mean"]
                    ),
                    "test_r2_std": float(
                        session_summary["retrained_7min_rolling_r2_std"]
                    ),
                    "best_test_fold": best_fold,
                    "best_test_fold_r2": float(
                        best_row["retrained_7min_rolling_r2"]
                    ),
                    "best_fold_is_descriptive_only": True,
                    "selection_note": (
                        "Each fold checkpoint was selected by validation loss. The best-test-fold "
                        "marker is for inspection/deployment convenience and is not the paper estimate."
                    ),
                },
                "model": {
                    "architecture": "MidsizeTCNGRU",
                    "parameters": 86_978,
                    "input_features": 192,
                    "physical_channels": 96,
                    "source_channels": int(first_payload["source_channel_count"]),
                    "window_bins": 50,
                    "bin_seconds": 0.04,
                    "output_timestep": 49,
                    "shared_python_definition": "../model.py" if tier == "midsize" else "../../midsize/model.py",
                    "checkpoints": checkpoints,
                },
                "preprocessing": {
                    "ewma_alpha": 0.1,
                    "calibration_bins": 10_500,
                    "calibration_seconds": 420.0,
                    "calibration_minutes": 7.0,
                    "continuous_rolling_window": True,
                    "ewma_reset_at_reach": False,
                    "window_order": "oldest_to_newest",
                },
                "cubeai": {
                    "status": "not_run_in_this_phase",
                    "python_inputs_ready": True,
                    "input_files": [record["file"] for record in checkpoints],
                    "shared_model_definition": "midsize/model.py",
                    "generated_onnx_h5_c_aibundle_present": False,
                    "next_action": "Convert and validate every fold in the next CubeAI phase.",
                },
            }
            if tier == "large":
                manifest["definition"] = (
                    "The same neural checkpoint as Midsize plus fold-specific GRU-hidden[49] "
                    "external residual memory."
                )
                manifest["external_memory"] = {
                    "query_representation": "gru_hidden_49_plus_long_context",
                    "status": "fold_specific_memlib_rebuild_pending",
                    "compatible_memlib_files_present": False,
                    "reason": (
                        "Archived Phase-12 memlibs were built for older checkpoints and a "
                        "different preprocessing/evaluation protocol; reusing them would be invalid."
                    ),
                    "paper_test_r2_mean": None,
                    "next_action": (
                        "Build and evaluate one train-only residual bank per fold, then export "
                        "the validated firmware-compatible memory format."
                    ),
                }
            write_json(directory / "manifest.json", manifest)
            packages.append(
                {
                    "tier": tier,
                    "session": session,
                    "manifest": f"{tier}/{session}/manifest.json",
                    "checkpoint_count": 5,
                    "best_test_fold": best_fold,
                }
            )

    write_json(
        ROOT / "manifest.json",
        {
            "schema_version": 2,
            "authoritative": True,
            "phase": "phase13_round3_final_pre_cubeai",
            "status": "python_checkpoint_packaging_complete_cubeai_pending",
            "sessions": list(SESSIONS),
            "folds_per_session": 5,
            "tiers": {
                "midsize": {
                    "definition": "Phase-13 Round-3 deployment-aligned neural decoder",
                    "checkpoint_count": 30,
                    "paper_test_r2_mean": float(
                        overall["retrained_7min_rolling_r2_mean"]
                    ),
                    "paper_test_r2_std": float(
                        overall["retrained_7min_rolling_r2_std"]
                    ),
                    "paper_unit": "30 folds (six sessions x five folds)",
                },
                "large": {
                    "definition": "Same decoder plus fold-specific GRU residual memory",
                    "neural_checkpoint_count": 30,
                    "compatible_memlib_count": 0,
                    "paper_test_r2_mean": None,
                    "status": "memory_rebuild_and_evaluation_pending",
                },
            },
            "cubeai": {
                "status": "not_run",
                "generated_files_present": False,
                "planned_next_phase": True,
            },
            "packages": packages,
            "source_metrics": str(SUMMARY_METRICS.relative_to(PROJECT)),
        },
    )


def validate() -> None:
    index = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    if index["phase"] not in {
        "phase13_round3_final_pre_cubeai",
        "phase14_best_fold_cubeai_validated",
        "phase15_large_memory_pc_validated",
    }:
        raise ValueError("Unexpected active model phase")
    cubeai_complete = index["phase"] in {
        "phase14_best_fold_cubeai_validated",
        "phase15_large_memory_pc_validated",
    }
    expected_cubeai_status = (
        "six_best_session_folds_converted_and_host_validated"
        if cubeai_complete
        else "not_run"
    )
    if index["cubeai"]["status"] != expected_cubeai_status:
        raise ValueError("CubeAI status does not match the active model phase")
    grouped = fold_rows()
    summaries, overall = summary_rows()
    loaded_count = 0

    for session in SESSIONS:
        expected_best = int(
            max(
                grouped[session],
                key=lambda row: float(row["retrained_7min_rolling_r2"]),
            )["fold"]
        )
        tier_hashes: dict[str, list[str]] = {}
        for tier in TIERS:
            directory = ROOT / tier / session
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            records = manifest["model"]["checkpoints"]
            expected_files = {"manifest.json", *(record["file"] for record in records)}
            actual_files = {path.name for path in directory.iterdir() if path.is_file()}
            if actual_files != expected_files:
                raise ValueError(f"Unexpected files in {directory}: {actual_files ^ expected_files}")
            if len(records) != 5 or sorted(record["fold"] for record in records) != list(FOLDS):
                raise ValueError(f"{tier}/{session}: incomplete fold set")
            if sum(bool(record["best_test_fold"]) for record in records) != 1:
                raise ValueError(f"{tier}/{session}: expected one best-test-fold marker")
            if manifest["paper_reporting"]["best_test_fold"] != expected_best:
                raise ValueError(f"{tier}/{session}: wrong best-test fold")
            expected_mean = float(summaries[session]["retrained_7min_rolling_r2_mean"])
            if abs(manifest["paper_reporting"]["test_r2_mean"] - expected_mean) > 1e-12:
                raise ValueError(f"{tier}/{session}: session R2 mismatch")

            hashes = []
            for record in records:
                checkpoint = directory / record["file"]
                digest = sha256(checkpoint)
                if digest != record["sha256"]:
                    raise ValueError(f"Hash mismatch: {checkpoint}")
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if payload["session"] != session or int(payload["fold"]) != record["fold"]:
                    raise ValueError(f"Identity mismatch: {checkpoint}")
                if payload["selection_policy"] != SELECTION_POLICY:
                    raise ValueError(f"Selection policy mismatch: {checkpoint}")
                if payload["deployment_policy"]["calibration_bins"] != 10_500:
                    raise ValueError(f"Calibration mismatch: {checkpoint}")
                hashes.append(digest)
                loaded_count += 1
            tier_hashes[tier] = hashes
            if tier == "large" and manifest["external_memory"]["compatible_memlib_files_present"]:
                raise ValueError("Large package must not claim compatible memlibs before rebuild")
            if cubeai_complete:
                cubeai = manifest["cubeai"]
                if cubeai["status"] != "best_test_fold_converted_and_host_validated":
                    raise ValueError(f"{tier}/{session}: CubeAI best fold is not validated")
                if int(cubeai["converted_fold"]) != expected_best:
                    raise ValueError(f"{tier}/{session}: converted the wrong fold")
                expected_promoted = bool(
                    index["cubeai"][f"{tier}_firmware_and_gui_promoted"]
                )
                if bool(cubeai["firmware_or_gui_promoted"]) != expected_promoted:
                    raise ValueError(
                        f"{tier}/{session}: firmware/GUI promotion status differs "
                        "from the authoritative package index"
                    )
                midsize_package = ROOT / "midsize" / session / "cubeai" / f"fold-{expected_best}"
                conversion_manifest = json.loads(
                    (midsize_package / "manifest.json").read_text(encoding="utf-8")
                )
                if not conversion_manifest["held_out_accuracy"]["accepted"]:
                    raise ValueError(f"{tier}/{session}: CubeAI accuracy gate failed")
                if conversion_manifest["checkpoint"]["sha256"] != hashes[expected_best - 1]:
                    raise ValueError(f"{tier}/{session}: CubeAI checkpoint hash mismatch")
                if conversion_manifest["graph"]["hidden_49_view"] != "state_output[0][49][0:64]":
                    raise ValueError(f"{tier}/{session}: GRU hidden[49] output contract mismatch")
                for component in conversion_manifest["components"].values():
                    artifact_path = midsize_package / component["file"]
                    if sha256(artifact_path) != component["sha256"]:
                        raise ValueError(f"{tier}/{session}: CubeAI artifact hash mismatch")
                bundle_path = midsize_package / f"{session}_fold{expected_best}.aibundle"
                if sha256(bundle_path) != conversion_manifest["bundle"]["sha256"]:
                    raise ValueError(f"{tier}/{session}: CubeAI bundle hash mismatch")
                if tier == "large":
                    reference = json.loads(
                        (directory / "cubeai" / "manifest.json").read_text(encoding="utf-8")
                    )
                    if reference["external_memory_status"] != "compatible_gru_hidden_memlib_not_built":
                        raise ValueError(f"large/{session}: invalid external-memory status")
        if tier_hashes["midsize"] != tier_hashes["large"]:
            raise ValueError(f"{session}: Midsize/Large neural checkpoints differ")

    expected_overall = float(overall["retrained_7min_rolling_r2_mean"])
    if abs(index["tiers"]["midsize"]["paper_test_r2_mean"] - expected_overall) > 1e-12:
        raise ValueError("Overall paper R2 mismatch")
    if index["phase"] == "phase15_large_memory_pc_validated":
        summary_path = ROOT / index["tiers"]["large"]["pc_evaluation_summary"]
        memory_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        large = index["tiers"]["large"]
        checks = {
            "paper_test_r2_mean": "bank_ready_r2_mean",
            "paper_test_r2_std": "bank_ready_r2_std_across_folds",
            "paper_delta_vs_midsize": "ready_minus_absent_r2_mean",
        }
        for manifest_key, result_key in checks.items():
            if abs(float(large[manifest_key]) - float(memory_summary[result_key])) > 1e-12:
                raise ValueError(f"Large {manifest_key} differs from Phase-15 results")
        if int(large["pc_evaluation_memlib_count"]) != 30:
            raise ValueError("Large Phase-15 PC memlib count mismatch")
        if int(large["compatible_memlib_count"]) != 0:
            raise ValueError("PC evaluation memlibs must not be claimed as firmware compatible")
    if loaded_count != 60:
        raise ValueError(f"Expected 60 packaged checkpoint copies, loaded {loaded_count}")
    cubeai_message = (
        "six shared best-fold CubeAI packages host-validated"
        if cubeai_complete
        else "CubeAI not run"
    )
    memory_message = (
        "30 Large PC memlibs validated; firmware BCIMEM pending"
        if index["phase"] == "phase15_large_memory_pc_validated"
        else "Large memlib rebuild pending"
    )
    print(
        "Package validation passed: 6 sessions x 5 folds x 2 tiers = "
        f"60 checkpoint copies; {cubeai_message}; {memory_message}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"))
    args = parser.parse_args()
    build() if args.command == "build" else validate()


if __name__ == "__main__":
    main()
