#!/usr/bin/env python3
"""Phase 13: independently replay deployed Midsize and audit all 30 folds.

The deployment check executes the generated X-CUBE-AI host libraries, using
the same INT8 encoder weights, FP32 GRU/head weights, 60-second calibration,
causal EWMA, 50-bin rolling window, channel mapping, and target scaling as CM7.

The five-fold check reloads every saved Phase-7 checkpoint and independently
re-evaluates its untouched test split.  It reports means and standard
deviations rather than selecting the maximum test fold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
INDY_ROOT = HERE.parents[1]
TRAINING_ROOT = INDY_ROOT.parent
WORKSPACE_ROOT = TRAINING_ROOT.parent
FIRMWARE_ROOT = WORKSPACE_ROOT / "Custom-H747XIH6"
GUI_ROOT = WORKSPACE_ROOT / "BCI-STM32-Plot"
ARCHIVE = INDY_ROOT / "history" / "legacy_pre_package_cleanup"
PHASE7_RESULT = (
    INDY_ROOT / "history" / "results" / "indy" / "phase7_ann_vs_snn_fivefold"
)
RESULT_ROOT = HERE / "results"

SESSIONS = (
    "indy_20160622_01",
    "indy_20160630_01",
    "indy_20170131_02",
    "loco_20170210_03",
    "loco_20170215_02",
    "loco_20170301_05",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("all", "deployment", "fivefold"), default="all"
    )
    parser.add_argument("--session", action="append", choices=SESSIONS)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--stedgeai", type=Path)
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path(tempfile.gettempdir()) / "phase13_cubeai_host",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(tempfile.gettempdir()) / "phase13_fivefold_cache",
    )
    parser.add_argument("--rebuild-host", action="store_true")
    return parser.parse_args()


def import_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_command(command: list[str], *, cwd: Path | None = None) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}"
        )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def discover_stedgeai(explicit: Path | None) -> Path:
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(candidate)
    utilities = (
        Path.home()
        / "STM32Cube"
        / "Repository"
        / "Packs"
        / "STMicroelectronics"
        / "X-CUBE-AI"
    )
    candidates = sorted(
        list(utilities.glob("*/Utilities/macarm/stedgeai"))
        + list(utilities.glob("*/Utilities/linux/stedgeai"))
        + list(utilities.glob("*/Utilities/windows/stedgeai.exe"))
    )
    if not candidates:
        raise FileNotFoundError("X-CUBE-AI stedgeai was not found; pass --stedgeai")
    return candidates[-1]


def generated_workspace(root: Path, graph_name: str) -> Path:
    candidates = list(root.glob(f"**/inspector_{graph_name}/workspace"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one inspector_{graph_name}/workspace under {root}; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def generate_graph(
    *,
    stedgeai: Path,
    model: Path,
    model_type: str,
    graph_name: str,
    root: Path,
    rebuild: bool,
) -> tuple[Path, Path]:
    output = root / "output"
    workspace = root / "workspace"
    libraries = list(workspace.glob(f"**/lib/libai_{graph_name}.*"))
    generated_weights = output / f"{graph_name}_data.bin"
    if rebuild or not libraries or not generated_weights.is_file():
        if rebuild and root.exists():
            shutil.rmtree(root)
        output.mkdir(parents=True, exist_ok=True)
        workspace.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                str(stedgeai),
                "generate",
                "--target",
                "stm32",
                "--model",
                str(model),
                "--type",
                model_type,
                "--name",
                graph_name,
                "--no-inputs-allocation",
                "--no-outputs-allocation",
                "--binary",
                "--dll",
                "--output",
                str(output),
                "--workspace",
                str(workspace),
                "--quiet",
            ]
        )
    return generated_workspace(workspace, graph_name), generated_weights


def compile_runner(
    *,
    graph_name: str,
    source: Path,
    workspace: Path,
    output: Path,
    rebuild: bool,
) -> Path:
    library_candidates = list((workspace / "lib").glob(f"libai_{graph_name}.*"))
    if len(library_candidates) != 1:
        raise RuntimeError(f"could not resolve host library for {graph_name}")
    library = library_candidates[0].resolve()
    if rebuild or not output.is_file():
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            shutil.which("gcc") or "gcc",
            str(source),
            "-I",
            str(workspace / "generated"),
            "-I",
            str(workspace / "include"),
            "-L",
            str(workspace / "lib"),
            f"-lai_{graph_name}",
            "-o",
            str(output),
        ]
        if platform.system() != "Darwin":
            command.insert(-2, f"-Wl,-rpath,{workspace / 'lib'}")
        run_command(command)
        if platform.system() == "Darwin":
            linked = subprocess.run(
                ["otool", "-L", str(output)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            dependency = next(
                line.strip().split(" ", 1)[0]
                for line in linked.splitlines()[1:]
                if library.name in line
            )
            run_command(
                ["install_name_tool", "-change", dependency, str(library), str(output)]
            )
    return output


def prepare_gru_host(
    builder: Any, stedgeai: Path, work_root: Path, rebuild: bool
) -> Any:
    root = work_root / "gru"
    exposed = root / "indy_gru_head_hidden.h5"
    source = ARCHIVE / "deploy" / "model" / "indy_phase6_gru_head.h5"
    if rebuild or not exposed.is_file():
        exposed.parent.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                sys.executable,
                str(FIRMWARE_ROOT / "tools" / "export_gru_hidden_graph.py"),
                str(source),
                str(exposed),
            ]
        )
    workspace, _ = generate_graph(
        stedgeai=stedgeai,
        model=exposed,
        model_type="keras",
        graph_name="indy_gru_head",
        root=root / "generated",
        rebuild=rebuild,
    )
    runner = compile_runner(
        graph_name="indy_gru_head",
        source=FIRMWARE_ROOT / "tools" / "cubeai_gru_hidden_host_runner.c",
        workspace=workspace,
        output=root / "gru_hidden_runner",
        rebuild=rebuild,
    )
    return builder.HostGraph(runner=runner)


def prepare_encoder_host(
    builder: Any, session: str, stedgeai: Path, work_root: Path, rebuild: bool
) -> tuple[Any, bool]:
    archive = ARCHIVE / "models" / "midsize" / session / "cubeai_int8"
    workspace, generated_weights = generate_graph(
        stedgeai=stedgeai,
        model=archive / "encoder_int8.tflite",
        model_type="tflite",
        graph_name="indy_encoder",
        root=work_root / "encoder" / session,
        rebuild=rebuild,
    )
    expected_weights = archive / "encoder.weights.bin"
    weights_equal = generated_weights.read_bytes() == expected_weights.read_bytes()
    if not weights_equal:
        raise ValueError(f"{session}: regenerated encoder weight bytes changed")
    runner = compile_runner(
        graph_name="indy_encoder",
        source=FIRMWARE_ROOT / "tools" / "cubeai_encoder_host_runner.c",
        workspace=workspace,
        output=work_root / "encoder" / session / "encoder_runner",
        rebuild=rebuild,
    )
    return builder.HostGraph(runner=runner, weights=expected_weights), weights_equal


def bundle_body_matches_archive(session: str) -> bool:
    staged = GUI_ROOT / "data" / "ai_device_sessions" / "models" / f"{session}.aibundle"
    archived = (
        ARCHIVE / "models" / "midsize" / session / "cubeai_int8" / f"{session}.aibundle"
    )
    return staged.read_bytes()[256:] == archived.read_bytes()[256:]


def run_firmware_preprocess_host_test(work_root: Path) -> bool:
    output = work_root / "m7_decoder_preprocess_test"
    output.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            shutil.which("gcc") or "gcc",
            str(
                FIRMWARE_ROOT
                / "Custom-H747XIH6"
                / "Tests"
                / "host"
                / "test_m7_decoder_preprocess.c"
            ),
            str(
                FIRMWARE_ROOT
                / "Custom-H747XIH6"
                / "CM7"
                / "Core"
                / "Src"
                / "m7_decoder_preprocess.c"
            ),
            "-I",
            str(FIRMWARE_ROOT / "Custom-H747XIH6" / "CM7" / "Core" / "Inc"),
            "-lm",
            "-o",
            str(output),
        ]
    )
    completed = subprocess.run(
        [str(output)], check=True, capture_output=True, text=True
    )
    return "m7_decoder_preprocess: PASS" in completed.stdout


def checkpoint_matches_selected_fold(session: str, checkpoint: dict[str, Any]) -> bool:
    import torch

    fold = int(checkpoint["fold"])
    source = PHASE7_RESULT / "checkpoints" / f"{session}_fold{fold}.pt"
    if checkpoint.get("source_checkpoint_sha256") != sha256(source):
        return False
    original = torch.load(source, map_location="cpu", weights_only=False)
    if set(original["model_state"]) != set(checkpoint["model_state"]):
        return False
    return all(
        torch.equal(original["model_state"][key], checkpoint["model_state"][key])
        for key in original["model_state"]
    )


def rolling_inputs(normalized: np.ndarray, bins: np.ndarray) -> np.ndarray:
    offsets = np.arange(50, dtype=np.int64) - 49
    indices = bins[:, None] + offsets[None, :]
    return np.ascontiguousarray(
        normalized[:, indices].transpose(1, 0, 2), dtype=np.float32
    )


def fp32_inference(
    checkpoint_path: Path,
    normalized: np.ndarray,
    bins: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    if str(TRAINING_ROOT) not in sys.path:
        sys.path.insert(0, str(TRAINING_ROOT))
    from indy_loco.models.midsize.model import load_checkpoint

    model, checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    outputs = []
    with torch.inference_mode():
        for left in range(0, len(bins), batch_size):
            values = torch.from_numpy(
                rolling_inputs(normalized, bins[left : left + batch_size])
            )
            outputs.append(model(values)[:, -1].numpy().astype(np.float32))
    normalized_prediction = np.concatenate(outputs)
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
    target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)
    return normalized_prediction * target_std + target_mean, checkpoint


def deployment_replay(
    sessions: list[str],
    batch_size: int,
    stedgeai_path: Path | None,
    work_root: Path,
    rebuild: bool,
) -> dict[str, Any]:
    builder = import_file(
        "phase13_firmware_builder",
        FIRMWARE_ROOT / "tools" / "build_gru_hidden_bcimem.py",
    )
    stedgeai = discover_stedgeai(stedgeai_path)
    preprocess_host_test_passed = run_firmware_preprocess_host_test(work_root)
    gru = prepare_gru_host(builder, stedgeai, work_root, rebuild)
    gui_manifest = json.loads(
        (GUI_ROOT / "data" / "ai_device_sessions" / "manifest.json").read_text()
    )
    gui_by_session = {row["id"]: row for row in gui_manifest["sessions"]}
    rows: list[dict[str, Any]] = []
    for session in sessions:
        print(f"deployment replay: {session}", flush=True)
        checkpoint_path = INDY_ROOT / "models" / "midsize" / session / "checkpoint.pt"
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        dataset_path = GUI_ROOT / "data" / "ai_device_sessions" / f"{session}.npz"
        mask_path = (
            GUI_ROOT
            / "data"
            / "ai_device_sessions"
            / f"{session}_best_fold_test_bins.npz"
        )
        with np.load(dataset_path, allow_pickle=False) as archive:
            counts_all = np.asarray(archive["counts"])
            target_all = np.asarray(archive["velocity"], dtype=np.float32)
        with np.load(mask_path, allow_pickle=False) as archive:
            bins = np.asarray(archive["bin_indices"], dtype=np.int64)
            mask_fold = int(archive["selected_fold"])
        if mask_fold != int(checkpoint["fold"]):
            raise ValueError(f"{session}: mask/checkpoint fold mismatch")
        channels = np.asarray(checkpoint["selected_channel_indices"], dtype=np.int64)
        normalized, _, calibration = builder.firmware_preprocess(
            counts_all[channels], checkpoint["feature_std_floor"]
        )
        counts = counts_all[channels].astype(np.float32)
        training_features = np.concatenate(
            (counts, builder.causal_ewma(counts, builder.EWMA_ALPHA)), axis=0
        )
        training_normalized = (
            (
                training_features
                - np.asarray(checkpoint["feature_mean"], dtype=np.float32)
            )
            / np.asarray(checkpoint["feature_std"], dtype=np.float32)
        ).astype(np.float32)
        encoder, encoder_weights_verified = prepare_encoder_host(
            builder, session, stedgeai, work_root, rebuild
        )
        archive_root = ARCHIVE / "models" / "midsize" / session / "cubeai_int8"
        normalized_prediction, _ = builder.cubeai_inference(
            normalized=normalized,
            bins=bins,
            encoder=encoder,
            gru=gru,
            gru_weights=archive_root / "gru_head.weights.bin",
            scratch=work_root / "scratch" / session,
            batch_size=batch_size,
        )
        target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
        target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)
        cubeai_prediction = normalized_prediction * target_std + target_mean
        fp32_prediction, _ = fp32_inference(
            checkpoint_path, normalized, bins, batch_size
        )
        continuous_training_norm_prediction, _ = fp32_inference(
            checkpoint_path, training_normalized, bins, batch_size
        )
        target = target_all[bins]
        cubeai = builder.metrics(target, cubeai_prediction)
        fp32 = builder.metrics(target, fp32_prediction)
        continuous_training_norm = builder.metrics(
            target, continuous_training_norm_prediction
        )
        expected = float(gui_by_session[session]["memlib_test_absent_r2_mean"])
        delta = float(cubeai["r2_mean"] - expected)
        if abs(delta) > 2e-6:
            raise ValueError(
                f"{session}: fresh CubeAI R2 {cubeai['r2_mean']} != manifest {expected}"
            )
        row = {
            "session": session,
            "subject": session.split("_", 1)[0],
            "selected_fold": int(checkpoint["fold"]),
            "evaluation_bins": int(len(bins)),
            "phase7_selection_r2_mean": float(checkpoint["selection_test_r2_mean"]),
            "phase7_reference_same_bins_r2_mean": float(
                checkpoint["firmware_style_replay"]["phase7_reference_same_bins"][
                    "r2_mean"
                ]
            ),
            "fp32_continuous_training_norm_r2_mean": float(
                continuous_training_norm["r2_mean"]
            ),
            "fp32_deployment_r2_mean": float(fp32["r2_mean"]),
            "cubeai_deployment_r2_x": float(cubeai["r2_x"]),
            "cubeai_deployment_r2_y": float(cubeai["r2_y"]),
            "cubeai_deployment_r2_mean": float(cubeai["r2_mean"]),
            "cubeai_minus_fp32_r2_mean": float(cubeai["r2_mean"] - fp32["r2_mean"]),
            "manifest_difference": delta,
            "checkpoint_sha256": sha256(checkpoint_path),
            "checkpoint_matches_phase7_selected_fold": checkpoint_matches_selected_fold(
                session, checkpoint
            ),
            "encoder_generated_weights_match": encoder_weights_verified,
            "gui_bundle_body_matches_archived_bundle": bundle_body_matches_archive(
                session
            ),
            "calibration_mean_min": float(calibration["mean"].min()),
            "calibration_mean_max": float(calibration["mean"].max()),
        }
        rows.append(row)
        print(
            f"  FP32={fp32['r2_mean']:.6f}; CubeAI={cubeai['r2_mean']:.6f}; "
            f"manifest delta={delta:+.2e}",
            flush=True,
        )
    values = np.asarray([row["cubeai_deployment_r2_mean"] for row in rows])
    indy = np.asarray(
        [row["cubeai_deployment_r2_mean"] for row in rows if row["subject"] == "indy"]
    )
    loco = np.asarray(
        [row["cubeai_deployment_r2_mean"] for row in rows if row["subject"] == "loco"]
    )
    fp32_values = np.asarray([row["fp32_deployment_r2_mean"] for row in rows])
    same_bin_values = np.asarray(
        [row["phase7_reference_same_bins_r2_mean"] for row in rows]
    )
    continuous_training_norm_values = np.asarray(
        [row["fp32_continuous_training_norm_r2_mean"] for row in rows]
    )
    summary = {
        "numeric_path": "X-CUBE-AI 10.2 INT8 encoder -> FP32 GRU/head",
        "preprocessing": "CM7-equivalent 1500-bin calibration + causal EWMA + rolling 50-bin window",
        "sessions": rows,
        "aggregate": {
            "all_mean": float(values.mean()),
            "indy_mean": float(indy.mean()) if len(indy) else None,
            "loco_mean": float(loco.mean()) if len(loco) else None,
            "phase7_reference_same_bins_mean": float(same_bin_values.mean()),
            "fp32_continuous_training_norm_mean": float(
                continuous_training_norm_values.mean()
            ),
            "fp32_deployment_mean": float(fp32_values.mean()),
            "cubeai_minus_fp32_mean": float((values - fp32_values).mean()),
            "continuous_window_minus_phase7_same_bins_mean": float(
                (continuous_training_norm_values - same_bin_values).mean()
            ),
            "deployment_calibration_minus_continuous_training_norm_mean": float(
                (fp32_values - continuous_training_norm_values).mean()
            ),
            "deployment_preprocessing_minus_phase7_same_bins_mean": float(
                (fp32_values - same_bin_values).mean()
            ),
        },
        "all_integrity_checks_passed": all(
            row["checkpoint_matches_phase7_selected_fold"]
            and row["encoder_generated_weights_match"]
            and row["gui_bundle_body_matches_archived_bundle"]
            and abs(row["manifest_difference"]) <= 2e-6
            for row in rows
        ),
        "stedgeai": str(stedgeai),
        "firmware_preprocess_host_test_passed": preprocess_host_test_passed,
    }
    write_json(RESULT_ROOT / "deployment_midsize_cubeai.json", summary)
    write_csv(RESULT_ROOT / "deployment_midsize_cubeai.csv", rows)
    return summary


def fivefold_audit(
    sessions: list[str], threads: int, cache_root: Path
) -> dict[str, Any]:
    import torch

    torch.set_num_threads(threads)
    phase7 = import_file(
        "phase13_phase7",
        INDY_ROOT
        / "history"
        / "experiments"
        / "phase7"
        / "phase7_ann_vs_snn_fivefold.py",
    )
    phase7.CACHE_DIR = cache_root
    phase7.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    recorded = json.loads(
        (PHASE7_RESULT / "phase7_ann_vs_snn_fivefold_metrics.json").read_text()
    )
    recorded_by_key = {
        (row["session"], int(row["fold"])): row for row in recorded["results"]
    }
    rows: list[dict[str, Any]] = []
    for session in sessions:
        print(f"five-fold audit: {session}", flush=True)
        data = phase7.load_session(phase7.SESSION_BY_NAME[session])
        for fold in range(1, 6):
            arrays = phase7.prepare_fold(data, fold - 1)
            checkpoint_path = PHASE7_RESULT / "checkpoints" / f"{session}_fold{fold}.pt"
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            model = phase7.build_model()
            model.load_state_dict(checkpoint["model_state"], strict=True)
            evaluated = phase7.evaluate(
                model,
                arrays.test_x,
                arrays.test_y,
                arrays.test_mask,
                checkpoint["target_mean"],
                checkpoint["target_std"],
                torch.device("cpu"),
            )
            expected = recorded_by_key[(session, fold)]["test"]
            difference = float(evaluated["r2_mean"] - expected["r2_mean"])
            metadata_match = (
                int(checkpoint["fold"]) == fold
                and str(checkpoint["session"]) == session
                and np.array_equal(
                    checkpoint["selected_channel_indices"], arrays.channels
                )
                and np.allclose(
                    checkpoint["feature_mean"], arrays.feature_mean, rtol=0, atol=5e-7
                )
                and np.allclose(
                    checkpoint["feature_std"], arrays.feature_std, rtol=0, atol=5e-7
                )
                and np.allclose(
                    checkpoint["target_mean"], arrays.target_mean, rtol=0, atol=1e-7
                )
                and np.allclose(
                    checkpoint["target_std"], arrays.target_std, rtol=0, atol=1e-7
                )
            )
            if abs(difference) > 2e-6 or not metadata_match:
                raise ValueError(
                    f"{session} fold {fold}: checkpoint audit failed; "
                    f"R2 delta={difference}, metadata={metadata_match}"
                )
            rows.append(
                {
                    "session": session,
                    "subject": session.split("_", 1)[0],
                    "fold": fold,
                    "best_epoch": int(checkpoint["best_epoch"]),
                    "test_bins": int(evaluated["valid_bins"]),
                    "test_r2_x": float(evaluated["r2_x"]),
                    "test_r2_y": float(evaluated["r2_y"]),
                    "test_r2_mean": float(evaluated["r2_mean"]),
                    "recorded_difference": difference,
                    "metadata_match": metadata_match,
                    "checkpoint_sha256": sha256(checkpoint_path),
                }
            )
            print(
                f"  fold {fold}: R2={evaluated['r2_mean']:.6f}; "
                f"recorded delta={difference:+.2e}",
                flush=True,
            )
    session_summary = []
    for session in sessions:
        values = np.asarray(
            [row["test_r2_mean"] for row in rows if row["session"] == session]
        )
        session_summary.append(
            {
                "session": session,
                "folds": int(len(values)),
                "test_r2_mean": float(values.mean()),
                "test_r2_std_sample": float(values.std(ddof=1)),
                "test_r2_std_population": float(values.std(ddof=0)),
                "test_r2_min": float(values.min()),
                "test_r2_max": float(values.max()),
            }
        )
    all_values = np.asarray([row["test_r2_mean"] for row in rows])
    fold_macro = np.asarray(
        [
            np.mean([row["test_r2_mean"] for row in rows if row["fold"] == fold])
            for fold in range(1, 6)
        ]
    )
    summary = {
        "protocol": "five reach-level folds; train-only preprocessing; validation-selected checkpoint; test evaluated once",
        "fold_selection_for_report": "none; report mean and standard deviation",
        "cells": rows,
        "by_session": session_summary,
        "aggregate": {
            "checkpoint_cells": int(len(all_values)),
            "cell_macro_mean": float(all_values.mean()),
            "cell_macro_std_sample": float(all_values.std(ddof=1)),
            "cell_macro_std_population": float(all_values.std(ddof=0)),
            "six_session_mean_by_fold": [float(value) for value in fold_macro],
            "fold_macro_mean": float(fold_macro.mean()),
            "fold_macro_std_sample": float(fold_macro.std(ddof=1)),
            "fold_macro_std_population": float(fold_macro.std(ddof=0)),
        },
        "all_checkpoints_present": len(rows) == 5 * len(sessions),
        "all_recorded_scores_reproduced": all(
            abs(row["recorded_difference"]) <= 2e-6 for row in rows
        ),
        "all_metadata_checks_passed": all(row["metadata_match"] for row in rows),
    }
    write_json(RESULT_ROOT / "fivefold_checkpoint_audit.json", summary)
    write_csv(RESULT_ROOT / "fivefold_checkpoint_audit.csv", rows)
    write_csv(RESULT_ROOT / "fivefold_session_summary.csv", session_summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.threads <= 0:
        raise ValueError("--batch-size and --threads must be positive")
    sessions = list(dict.fromkeys(args.session or SESSIONS))
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = RESULT_ROOT / "phase13_report.json"
    report: dict[str, Any] = {}
    if report_path.is_file():
        prior = json.loads(report_path.read_text(encoding="utf-8"))
        if prior.get("sessions") == sessions:
            report = prior
    report.update(
        {
            "phase": "phase13_deployment_validation",
            "sessions": sessions,
        }
    )
    if args.mode in ("all", "deployment"):
        report["deployment"] = deployment_replay(
            sessions,
            args.batch_size,
            args.stedgeai,
            args.work_root.resolve(),
            args.rebuild_host,
        )
    if args.mode in ("all", "fivefold"):
        report["fivefold"] = fivefold_audit(
            sessions, args.threads, args.cache_root.resolve()
        )
    report["status"] = "complete"
    write_json(report_path, report)
    print(f"Phase 13 complete: {RESULT_ROOT}")


if __name__ == "__main__":
    main()
