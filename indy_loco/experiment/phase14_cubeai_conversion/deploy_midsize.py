#!/usr/bin/env python3
"""Build the six Phase-13 best-fold assets for the midsize firmware/GUI.

The encoder graph must be regenerated per session because CubeAI embeds INT8
quantization parameters in the generated C source.  The GRU remains the
midsize, velocity-only graph; its weights are verified byte-for-byte against
the already validated Phase-14 velocity+state graph before promotion.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


HERE = Path(__file__).resolve().parent
INDY_ROOT = HERE.parents[1]
WORKSPACE_ROOT = INDY_ROOT.parent.parent
DEFAULT_FIRMWARE_ROOT = WORKSPACE_ROOT / "Custom-H747XIH6" / "Custom-H747XIH6"
DEFAULT_GUI_ROOT = WORKSPACE_ROOT / "BCI-STM32-Plot"
DEFAULT_STEDGEAI = Path(
    "/Users/yinzhecheng/STM32Cube/Repository/Packs/STMicroelectronics/"
    "X-CUBE-AI/10.2.0/Utilities/macarm/stedgeai"
)
GRAPH_ABI_ID = "tcn64i8x6-gru64f32-p13-v3"

SESSIONS = (
    ("indy_20160622_01", 5, "indy_enc_i622"),
    ("indy_20160630_01", 4, "indy_enc_i630"),
    ("indy_20170131_02", 4, "indy_enc_i131"),
    ("loco_20170210_03", 5, "indy_enc_l210"),
    ("loco_20170215_02", 5, "indy_enc_l215"),
    ("loco_20170301_05", 1, "indy_enc_l301"),
)


def import_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PILOT = import_file("phase14_midsize_pilot", HERE / "run_pilot.py")
FP32 = PILOT.FP32


def run_generate(
    cli: Path, model: Path, model_type: str, name: str, root: Path
) -> Path:
    output = root / name / "output"
    workspace = root / name / "workspace"
    output.mkdir(parents=True)
    command = FP32.cubeai_base(
        cli, "generate", model, model_type, name, workspace, output
    ) + ["--binary"]
    subprocess.run(command, cwd=WORKSPACE_ROOT, check=True)
    return output


def copy_generated(source: Path, destination: Path, graph_name: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    expected = {
        f"{graph_name}.c",
        f"{graph_name}.h",
        f"{graph_name}_config.h",
        f"{graph_name}_data.c",
        f"{graph_name}_data.h",
        f"{graph_name}_data_params.c",
        f"{graph_name}_data_params.h",
    }
    available = {path.name for path in source.iterdir() if path.is_file()}
    missing = expected - available
    if missing:
        raise FileNotFoundError(f"{graph_name}: missing generated files {sorted(missing)}")
    for name in sorted(expected):
        shutil.copy2(source / name, destination / name)


def build_bundle(session: str, fold: int, destination: Path) -> dict[str, Any]:
    session_root = INDY_ROOT / "models" / "midsize" / session
    checkpoint_path = session_root / f"fold-{fold}_best-test-fold.pt"
    package = session_root / "cubeai" / f"fold-{fold}"
    _, checkpoint = PILOT.load_checkpoint(checkpoint_path)
    if checkpoint["session"] != session or int(checkpoint["fold"]) != fold:
        raise ValueError(f"{session}: checkpoint identity mismatch")
    deployment = checkpoint.get("deployment_policy", {})
    if int(deployment.get("calibration_bins", 0)) != 10_500:
        raise ValueError(f"{session}: checkpoint is not the seven-minute deployment")
    constants = {
        "feature_std_floor": np.asarray(checkpoint["feature_std_floor"], np.float32),
        "target_mean": np.asarray(checkpoint["target_mean"], np.float32),
        "target_std": np.asarray(checkpoint["target_std"], np.float32),
        "selected_channel_indices": np.asarray(
            checkpoint["selected_channel_indices"], np.int64
        ),
    }
    bundle_checkpoint = dict(checkpoint)
    bundle_checkpoint["model_id"] = session
    destination.parent.mkdir(parents=True, exist_ok=True)
    return FP32.build_bundle(
        destination,
        checkpoint_path,
        bundle_checkpoint,
        constants,
        package / "encoder.weights.bin",
        package / "gru_hidden_head.weights.bin",
        graph_abi_id=GRAPH_ABI_ID,
    )


def promote_model_metadata(report: dict[str, Any]) -> None:
    index_path = INDY_ROOT / "models" / "manifest.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["status"] = "six_best_fold_midsize_promoted_board_test_pending"
    midsize = index["tiers"]["midsize"]
    midsize["deployment_status"] = "six_best_folds_promoted_to_firmware_and_gui"
    midsize["deployment_graph_abi_id"] = GRAPH_ABI_ID
    midsize["calibration_bins"] = 10_500
    cubeai = index["cubeai"]
    cubeai.pop("firmware_or_gui_promoted", None)
    cubeai["midsize_firmware_and_gui_promoted"] = True
    cubeai["large_firmware_and_gui_promoted"] = False
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    by_session = {item["session"]: item for item in report["sessions"]}
    for session, fold, _ in SESSIONS:
        manifest_path = INDY_ROOT / "models" / "midsize" / session / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["package_status"] = (
            "all_python_folds_present_best_fold_midsize_promoted_board_test_pending"
        )
        manifest["cubeai"]["firmware_or_gui_promoted"] = True
        manifest["midsize_deployment"] = {
            "status": "firmware_and_gui_promoted_board_test_pending",
            "selected_fold": fold,
            "graph_abi_id": GRAPH_ABI_ID,
            "calibration_bins": 10_500,
            "bundle": by_session[session]["bundle"],
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firmware-root", type=Path, default=DEFAULT_FIRMWARE_ROOT)
    parser.add_argument("--gui-root", type=Path, default=DEFAULT_GUI_ROOT)
    parser.add_argument("--stedgeai", type=Path, default=DEFAULT_STEDGEAI)
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="Build and verify assets without copying them into firmware/GUI.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Refresh promotion metadata from an existing deployment manifest.",
    )
    args = parser.parse_args()
    firmware_generated = args.firmware_root.resolve() / "CM7" / "AI" / "Generated"
    gui_models = args.gui_root.resolve() / "data" / "ai_device_sessions" / "models"
    cli = args.stedgeai.expanduser().resolve()
    if not cli.is_file():
        raise FileNotFoundError(cli)

    stage = HERE / "results" / "midsize_deployment"
    stage.mkdir(parents=True, exist_ok=True)
    bundle_stage = stage / "bundles"
    bundle_stage.mkdir(parents=True, exist_ok=True)
    if args.metadata_only:
        report = json.loads(
            (stage / "deployment_manifest.json").read_text(encoding="utf-8")
        )
        promote_model_metadata(report)
        print("Midsize promotion metadata refreshed")
        return
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "phase13_midsize_best_folds_promoted",
        "graph_abi_id": GRAPH_ABI_ID,
        "calibration_bins": 10_500,
        "paper_r2_mean": 0.7411375800768535,
        "paper_r2_std": 0.06559769297745317,
        "sessions": [],
    }

    with tempfile.TemporaryDirectory(prefix="phase13-midsize-") as temporary:
        generate_root = Path(temporary)
        first_session, first_fold, _ = SESSIONS[0]
        first_package = (
            INDY_ROOT
            / "models"
            / "midsize"
            / first_session
            / "cubeai"
            / f"fold-{first_fold}"
        )
        hidden_model = tf.keras.models.load_model(
            first_package / "gru_hidden_head.h5", compile=False
        )
        velocity_model = tf.keras.Model(
            hidden_model.input, hidden_model.outputs[0], name="indy_gru_head"
        )
        velocity_path = generate_root / "gru_velocity_head.h5"
        velocity_model.save(velocity_path, include_optimizer=False)
        FP32.sanitize_h5_for_cubeai(velocity_path)
        gru_output = run_generate(
            cli, velocity_path, "keras", "indy_gru_head", generate_root
        )
        generated_gru_weights = gru_output / "indy_gru_head_data.bin"
        validated_gru_weights = first_package / "gru_hidden_head.weights.bin"
        if generated_gru_weights.read_bytes() != validated_gru_weights.read_bytes():
            raise ValueError("velocity-only and velocity+state GRU weight layouts differ")
        if not args.stage_only:
            copy_generated(gru_output, firmware_generated, "indy_gru_head")

        for session, fold, graph_name in SESSIONS:
            package = (
                INDY_ROOT / "models" / "midsize" / session / "cubeai" / f"fold-{fold}"
            )
            encoder_output = run_generate(
                cli,
                package / "encoder_int8.tflite",
                "tflite",
                graph_name,
                generate_root,
            )
            if (encoder_output / f"{graph_name}_data.bin").read_bytes() != (
                package / "encoder.weights.bin"
            ).read_bytes():
                raise ValueError(f"{session}: regenerated encoder weights differ")
            if not args.stage_only:
                copy_generated(encoder_output, firmware_generated, graph_name)

            bundle_path = bundle_stage / f"{session}.aibundle"
            bundle = build_bundle(session, fold, bundle_path)
            if not args.stage_only:
                gui_models.mkdir(parents=True, exist_ok=True)
                shutil.copy2(bundle_path, gui_models / bundle_path.name)
            report["sessions"].append(
                {
                    "session": session,
                    "selected_fold": fold,
                    "encoder_graph": graph_name,
                    "bundle": bundle,
                }
            )

    (stage / "deployment_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if not args.stage_only:
        promote_model_metadata(report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
