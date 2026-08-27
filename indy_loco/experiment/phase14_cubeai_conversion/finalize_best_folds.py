#!/usr/bin/env python3
"""Verify and register the six Phase-14 best-fold CubeAI conversions."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
INDY_ROOT = HERE.parents[1]
MODELS = INDY_ROOT / "models"
RESULTS = HERE / "results"
SESSIONS = {
    "indy_20160622_01": 5,
    "indy_20160630_01": 4,
    "indy_20170131_02": 4,
    "loco_20170210_03": 5,
    "loco_20170215_02": 5,
    "loco_20170301_05": 1,
}
PAPER_MEAN = 0.7411375800768535
PAPER_STD = 0.06559769297745317


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def main() -> None:
    rows: list[dict[str, Any]] = []
    graph_abi: str | None = None
    for session, fold in SESSIONS.items():
        summary_path = RESULTS / f"{session}_fold{fold}" / "pilot_summary.json"
        package = MODELS / "midsize" / session / "cubeai" / f"fold-{fold}"
        package_manifest_path = package / "manifest.json"
        summary = read_json(summary_path)
        package_manifest = read_json(package_manifest_path)
        if summary["session"] != session or int(summary["fold"]) != fold:
            raise ValueError(f"identity mismatch: {summary_path}")
        if not summary["best_test_fold"]:
            raise ValueError(f"not marked best test fold: {summary_path}")
        if not summary["held_out_accuracy"]["accepted"]:
            raise ValueError(f"held-out gate failed: {summary_path}")
        if summary != package_manifest:
            raise ValueError(f"result/package manifest mismatch: {session}")
        checkpoint = MODELS / "midsize" / session / f"fold-{fold}_best-test-fold.pt"
        if summary["checkpoint"]["sha256"] != sha256(checkpoint):
            raise ValueError(f"checkpoint hash mismatch: {checkpoint}")
        for required in (
            "encoder.weights.bin",
            "gru_hidden_head.weights.bin",
            "encoder_int8.tflite",
            "gru_hidden_head.h5",
            f"{session}_fold{fold}.aibundle",
        ):
            if not (package / required).is_file():
                raise FileNotFoundError(package / required)
        current_abi = summary["graph"]["abi_id"]
        if graph_abi is None:
            graph_abi = current_abi
        elif graph_abi != current_abi:
            raise ValueError("all six conversions must share one graph ABI")

        accuracy = summary["held_out_accuracy"]
        row = {
            "session": session,
            "best_fold": fold,
            "float32_r2": accuracy["float32_pytorch"]["r2_mean"],
            "tflite_r2": accuracy["tflite_int8_encoder_torch_gru"]["r2_mean"],
            "cubeai_c_r2": accuracy["cubeai_c_int8_encoder_torch_gru"]["r2_mean"],
            "cubeai_r2_drop": accuracy["cubeai_r2_mean_drop"],
            "accepted": accuracy["accepted"],
            "package": str(package.relative_to(INDY_ROOT)),
        }
        rows.append(row)

        for tier in ("midsize", "large"):
            session_manifest_path = MODELS / tier / session / "manifest.json"
            session_manifest = read_json(session_manifest_path)
            session_manifest["phase"] = "phase14_best_fold_cubeai_validated"
            session_manifest["package_status"] = (
                "all_python_folds_present_best_fold_cubeai_validated_not_firmware_promoted"
            )
            shared_path = (
                f"cubeai/fold-{fold}"
                if tier == "midsize"
                else f"../../midsize/{session}/cubeai/fold-{fold}"
            )
            session_manifest["cubeai"] = {
                "status": "best_test_fold_converted_and_host_validated",
                "converted_fold": fold,
                "converted_checkpoint": f"fold-{fold}_best-test-fold.pt",
                "converted_checkpoint_count": 1,
                "other_fold_conversions_intentionally_not_run": 4,
                "package": shared_path,
                "shared_by_tiers": ["midsize", "large"],
                "graph_abi_id": graph_abi,
                "generated_c_aibundle_present": True,
                "firmware_or_gui_promoted": False,
                "held_out_cubeai_c_r2": row["cubeai_c_r2"],
                "held_out_cubeai_r2_drop": row["cubeai_r2_drop"],
                "paper_metric_is_not_this_selected_fold": True,
            }
            write_json(session_manifest_path, session_manifest)

        large_reference = {
            "schema_version": 1,
            "session": session,
            "fold": fold,
            "status": "shared_neural_cubeai_package_validated",
            "shared_package": f"../../../midsize/{session}/cubeai/fold-{fold}",
            "checkpoint_sha256": summary["checkpoint"]["sha256"],
            "graph_abi_id": graph_abi,
            "external_memory_status": "compatible_gru_hidden_memlib_not_built",
            "firmware_or_gui_promoted": False,
        }
        write_json(MODELS / "large" / session / "cubeai" / "manifest.json", large_reference)

    aggregate = {
        "schema_version": 1,
        "phase": "phase14_best_fold_cubeai_conversion",
        "status": "six_best_session_folds_converted_and_host_validated",
        "graph_abi_id": graph_abi,
        "converted_unique_neural_checkpoints": len(rows),
        "shared_by_tiers": ["midsize", "large"],
        "firmware_or_gui_promoted": False,
        "selected_fold_deployment_diagnostic": {
            "unit": "macro mean across six per-session best-test folds",
            "selection_warning": "descriptive deployment check only; test-selected folds are not the paper estimate",
            "float32_r2_mean": mean(rows, "float32_r2"),
            "tflite_r2_mean": mean(rows, "tflite_r2"),
            "cubeai_c_r2_mean": mean(rows, "cubeai_c_r2"),
            "cubeai_r2_mean_drop": mean(rows, "cubeai_r2_drop"),
            "maximum_abs_session_r2_drop": max(abs(float(row["cubeai_r2_drop"])) for row in rows),
            "all_accuracy_gates_passed": all(bool(row["accepted"]) for row in rows),
        },
        "official_paper_reporting": {
            "metric": "mean test R2 across all 30 validation-selected folds",
            "r2_mean": PAPER_MEAN,
            "r2_std": PAPER_STD,
            "unit": "six sessions x five folds",
        },
        "sessions": rows,
        "large_external_memory": {
            "status": "not_built",
            "query_source_available": "GRU state sequence output; hidden[49] is state row 49",
        },
    }
    write_json(RESULTS / "best_fold_conversion_summary.json", aggregate)
    with (RESULTS / "best_fold_conversion_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    index_path = MODELS / "manifest.json"
    index = read_json(index_path)
    index["phase"] = "phase14_best_fold_cubeai_validated"
    index["status"] = "six_best_fold_cubeai_packages_ready_not_firmware_promoted"
    index["cubeai"] = {
        "status": "six_best_session_folds_converted_and_host_validated",
        "generated_files_present": True,
        "unique_checkpoint_count": 6,
        "shared_by_tiers": ["midsize", "large"],
        "graph_abi_id": graph_abi,
        "firmware_or_gui_promoted": False,
        "summary": "../experiment/phase14_cubeai_conversion/results/best_fold_conversion_summary.json",
        "selected_fold_cubeai_c_r2_mean_descriptive_only": aggregate[
            "selected_fold_deployment_diagnostic"
        ]["cubeai_c_r2_mean"],
        "paper_r2_mean": PAPER_MEAN,
        "paper_r2_std": PAPER_STD,
    }
    write_json(index_path, index)
    print(json.dumps(aggregate["selected_fold_deployment_diagnostic"], indent=2))
    print(f"Official paper R2: {PAPER_MEAN:.4f} +/- {PAPER_STD:.4f}")


if __name__ == "__main__":
    main()
