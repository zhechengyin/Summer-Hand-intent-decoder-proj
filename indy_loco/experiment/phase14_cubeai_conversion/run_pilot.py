#!/usr/bin/env python3
"""Convert and validate the first Phase-13 best-fold CubeAI pilot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import h5py
import numpy as np
import tensorflow as tf
import torch


HERE = Path(__file__).resolve().parent
INDY_ROOT = HERE.parents[1]
REPOSITORY_ROOT = INDY_ROOT.parent
MODELS_ROOT = INDY_ROOT / "models"
RESULT_ROOT = HERE / "results" / "indy_20160622_01_fold5"
FINAL_ROOT = (
    MODELS_ROOT
    / "midsize"
    / "indy_20160622_01"
    / "cubeai"
    / "fold-5"
)
CHECKPOINT = (
    MODELS_ROOT
    / "midsize"
    / "indy_20160622_01"
    / "fold-5_best-test-fold.pt"
)
ROUND3_SCRIPT = (
    INDY_ROOT
    / "experiment"
    / "phase13_deployment_validation"
    / "run_rolling_retrain.py"
)
ARCHIVED_FP32_BUILDER = (
    INDY_ROOT
    / "history"
    / "legacy_pre_package_cleanup"
    / "deploy"
    / "build_session_cubeai.py"
)
SESSION = "indy_20160622_01"
FOLD = 5
GRAPH_ABI_ID = "tcn64i8-gruseq64-xcai10-v1"
R2_DROP_LIMIT = 0.01
REPRESENTATIVE_WINDOWS = 512
VALIDATION_WINDOWS = 8

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from indy_loco.models.midsize.model import MidsizeTCNGRU, load_checkpoint  # noqa: E402


def import_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ROUND3 = import_file("phase14_round3", ROUND3_SCRIPT)
FP32 = import_file("phase14_archived_fp32_builder", ARCHIVED_FP32_BUILDER)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def artifact(path: Path) -> dict[str, Any]:
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def max_errors(expected: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    return FP32.max_errors(np.asarray(expected), np.asarray(actual))


def build_encoder(model: MidsizeTCNGRU) -> tf.keras.Model:
    features = tf.keras.Input(batch_shape=(1, 192, 50), name="features")
    values = tf.keras.layers.Permute((2, 1), name="to_time_major_features")(features)
    values = tf.keras.layers.Conv1D(64, 1, name="spatial_conv")(values)
    values = tf.keras.layers.LayerNormalization(
        axis=-1, epsilon=1.0e-5, name="spatial_layer_norm"
    )(values)
    values = tf.keras.layers.ReLU(name="spatial_relu")(values)
    for index, dilation in enumerate((1, 2, 4, 8)):
        residual = values
        padded = tf.keras.layers.ZeroPadding1D(
            padding=(2 * dilation, 0), name=f"tcn_{index}_causal_pad"
        )(values)
        taps = tf.keras.layers.Concatenate(axis=-1, name=f"tcn_{index}_taps")(
            (
                tf.keras.layers.Cropping1D(
                    cropping=(0, 2 * dilation), name=f"tcn_{index}_oldest"
                )(padded),
                tf.keras.layers.Cropping1D(
                    cropping=(dilation, dilation), name=f"tcn_{index}_middle"
                )(padded),
                tf.keras.layers.Cropping1D(
                    cropping=(2 * dilation, 0), name=f"tcn_{index}_newest"
                )(padded),
            )
        )
        values = tf.keras.layers.Conv1D(
            64, 1, padding="valid", name=f"tcn_{index}_conv"
        )(taps)
        values = tf.keras.layers.Add(name=f"tcn_{index}_residual")(
            (values, residual)
        )
        values = tf.keras.layers.ReLU(name=f"tcn_{index}_relu")(values)

    encoder = tf.keras.Model(features, values, name="indy_encoder_int8")
    spatial = model.spatial[0]
    encoder.get_layer("spatial_conv").set_weights(
        [
            spatial.weight.detach().numpy().transpose(2, 1, 0),
            spatial.bias.detach().numpy(),
        ]
    )
    normalization = model.spatial[1].normalization
    encoder.get_layer("spatial_layer_norm").set_weights(
        [
            normalization.weight.detach().numpy(),
            normalization.bias.detach().numpy(),
        ]
    )
    for index, convolution in enumerate(model.convolutions):
        kernel = convolution.weight.detach().numpy().transpose(2, 1, 0)
        tap_kernel = np.concatenate((kernel[0], kernel[1], kernel[2]), axis=0)[None]
        encoder.get_layer(f"tcn_{index}_conv").set_weights(
            [tap_kernel, convolution.bias.detach().numpy()]
        )
    return encoder


def build_gru_hidden_head(model: MidsizeTCNGRU) -> tf.keras.Model:
    encoded = tf.keras.Input(batch_shape=(1, 50, 64), name="encoded_sequence")
    states = tf.keras.layers.GRU(
        64,
        activation="tanh",
        recurrent_activation="sigmoid",
        reset_after=True,
        return_sequences=True,
        unroll=False,
        name="gru",
    )(encoded)
    velocity = tf.keras.layers.Dense(2, name="velocity_norm")(states)
    keras_model = tf.keras.Model(
        encoded,
        [velocity, states],
        name="indy_gru_hidden_head",
    )
    weight_ih = model.gru.weight_ih_l0.detach().numpy().T
    weight_hh = model.gru.weight_hh_l0.detach().numpy().T
    bias_ih = model.gru.bias_ih_l0.detach().numpy()
    bias_hh = model.gru.bias_hh_l0.detach().numpy()
    keras_model.get_layer("gru").set_weights(
        [
            FP32.reorder_gru_gates(weight_ih, axis=1),
            FP32.reorder_gru_gates(weight_hh, axis=1),
            np.stack(
                (
                    FP32.reorder_gru_gates(bias_ih, axis=0),
                    FP32.reorder_gru_gates(bias_hh, axis=0),
                )
            ),
        ]
    )
    keras_model.get_layer("velocity_norm").set_weights(
        [model.head.weight.detach().numpy().T, model.head.bias.detach().numpy()]
    )
    return keras_model


def torch_components(
    model: MidsizeTCNGRU, inputs: np.ndarray, batch_size: int = 128
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    encoded_parts: list[np.ndarray] = []
    velocity_parts: list[np.ndarray] = []
    hidden_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for left in range(0, len(inputs), batch_size):
            values = torch.from_numpy(inputs[left : left + batch_size])
            encoded = model.spatial(values)
            for convolution, padding in zip(
                model.convolutions, model.padding, strict=True
            ):
                encoded = model.activation(
                    convolution(encoded)[:, :, :-padding] + encoded
                )
            encoded = encoded.transpose(1, 2)
            states, final_hidden = model.gru(encoded)
            if not torch.allclose(final_hidden[0], states[:, -1], rtol=0, atol=0):
                raise ValueError("PyTorch GRU final hidden differs from timestep 49")
            velocity = model.head(states)
            encoded_parts.append(encoded.numpy().astype(np.float32))
            velocity_parts.append(velocity.numpy().astype(np.float32))
            hidden_parts.append(final_hidden[0].numpy().astype(np.float32))
    return (
        np.concatenate(encoded_parts),
        np.concatenate(velocity_parts),
        np.concatenate(hidden_parts),
    )


def quantize_encoder(
    encoder: tf.keras.Model,
    representative_inputs: np.ndarray,
    destination: Path,
) -> dict[str, Any]:
    def representative() -> Iterator[list[np.ndarray]]:
        for sample in representative_inputs:
            yield [sample[None]]

    converter = tf.lite.TFLiteConverter.from_keras_model(encoder)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    values = converter.convert()
    destination.write_bytes(values)
    interpreter = tf.lite.Interpreter(model_content=values)
    interpreter.allocate_tensors()
    details = interpreter.get_tensor_details()
    internal = [item for item in details if item["dtype"] not in (np.int32, np.int64)]
    return {
        **artifact(destination),
        "representative_windows": len(representative_inputs),
        "representative_source": "fold-5 training bins only",
        "external_input_dtype": str(interpreter.get_input_details()[0]["dtype"]),
        "external_output_dtype": str(interpreter.get_output_details()[0]["dtype"]),
        "int8_tensor_count": int(sum(item["dtype"] == np.int8 for item in internal)),
        "float_tensor_count": int(
            sum(item["dtype"] == np.float32 for item in internal)
        ),
    }


def tflite_predict(path: Path, inputs: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    output = []
    for sample in inputs:
        interpreter.set_tensor(input_detail["index"], sample[None])
        interpreter.invoke()
        output.append(interpreter.get_tensor(output_detail["index"])[0])
    return np.stack(output).astype(np.float32)


def head_from_encoded(
    model: MidsizeTCNGRU, encoded: np.ndarray, batch_size: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    velocities = []
    hidden = []
    with torch.inference_mode():
        for left in range(0, len(encoded), batch_size):
            states, final_hidden = model.gru(
                torch.from_numpy(encoded[left : left + batch_size])
            )
            velocities.append(model.head(states).numpy().astype(np.float32))
            hidden.append(final_hidden[0].numpy().astype(np.float32))
    return np.concatenate(velocities), np.concatenate(hidden)


def r2_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    residual = np.sum((target - prediction) ** 2, axis=0)
    total = np.sum((target - target.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - residual / np.maximum(total, 1e-12)
    return {
        "bins": int(len(target)),
        "r2_x": float(r2[0]),
        "r2_y": float(r2[1]),
        "r2_mean": float(r2.mean()),
        "mse": float(np.mean((target - prediction) ** 2)),
    }


def load_flat(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=np.float32).reshape(shape)


def output_csvs(directory: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in directory.rglob("*_val_c_outputs_*.csv"):
        index = int(path.stem.rsplit("_", 1)[1])
        result[index] = path
    return result


def run_cubeai(
    cli: Path,
    build_root: Path,
    encoder_path: Path,
    gru_path: Path,
    validation_count: int,
) -> tuple[dict[str, Any], Path, Path, Path, Path, Path, Path]:
    validation = build_root / "validation"
    cube = build_root / "cubeai"
    workspace = build_root / "workspace"
    logs = build_root / "logs"
    encoder_validate = cube / "encoder_validate"
    encoder_generate = cube / "encoder_generate"
    gru_validate = cube / "gru_validate"
    gru_generate = cube / "gru_generate"
    chain_validate = cube / "chain_validate"

    FP32.run(
        FP32.cubeai_base(
            cli,
            "validate",
            encoder_path,
            "tflite",
            "indy_encoder",
            workspace / "encoder_validate",
            encoder_validate,
        )
        + [
            "--mode",
            "host",
            "--valinput",
            str(validation / "model_inputs.csv"),
            "--valoutput",
            str(validation / "encoder_expected.csv"),
            "--save-csv",
        ],
        logs / "validate_encoder.log",
    )
    FP32.run(
        FP32.cubeai_base(
            cli,
            "generate",
            encoder_path,
            "tflite",
            "indy_encoder",
            workspace / "encoder_generate",
            encoder_generate,
        )
        + ["--binary", "--dll"],
        logs / "generate_encoder.log",
    )
    gru_validation_tail = [
        "--mode",
        "host",
        "--valinput",
        str(validation / "encoder_expected.csv"),
        "--valoutput",
        str(validation / "velocity_expected.csv"),
        str(validation / "hidden_expected.csv"),
        "--save-csv",
    ]
    FP32.run(
        FP32.cubeai_base(
            cli,
            "validate",
            gru_path,
            "keras",
            "indy_gru_hidden_head",
            workspace / "gru_validate",
            gru_validate,
        )
        + gru_validation_tail,
        logs / "validate_gru_hidden_head.log",
    )
    FP32.run(
        FP32.cubeai_base(
            cli,
            "generate",
            gru_path,
            "keras",
            "indy_gru_hidden_head",
            workspace / "gru_generate",
            gru_generate,
        )
        + ["--binary"],
        logs / "generate_gru_hidden_head.log",
    )
    encoder_c_path = FP32.find_single(
        encoder_validate, "*_val_c_outputs_1.csv"
    )
    FP32.run(
        FP32.cubeai_base(
            cli,
            "validate",
            gru_path,
            "keras",
            "indy_gru_hidden_head",
            workspace / "chain_validate",
            chain_validate,
        )
        + [
            "--mode",
            "host",
            "--valinput",
            str(encoder_c_path),
            "--valoutput",
            str(validation / "velocity_expected.csv"),
            str(validation / "hidden_expected.csv"),
            "--save-csv",
        ],
        logs / "validate_chain.log",
    )

    expected_encoder = load_flat(
        validation / "encoder_expected.csv", (validation_count, 50, 64)
    )
    encoder_c = load_flat(encoder_c_path, expected_encoder.shape)
    expected_velocity = load_flat(
        validation / "velocity_expected.csv", (validation_count, 50, 2)
    )
    expected_states = load_flat(
        validation / "hidden_expected.csv", (validation_count, 50, 64)
    )

    def compare_outputs(directory: Path) -> dict[str, Any]:
        files = output_csvs(directory)
        if set(files) != {1, 2}:
            raise ValueError(f"Expected two CubeAI outputs below {directory}: {files}")
        first = np.loadtxt(files[1], delimiter=",", dtype=np.float32)
        second = np.loadtxt(files[2], delimiter=",", dtype=np.float32)
        candidates = [(first, second), (second, first)]
        for velocity_flat, states_flat in candidates:
            if velocity_flat.size == expected_velocity.size and states_flat.size == expected_states.size:
                actual_states = states_flat.reshape(expected_states.shape)
                return {
                    "velocity": max_errors(
                        expected_velocity, velocity_flat.reshape(expected_velocity.shape)
                    ),
                    "gru_states": max_errors(expected_states, actual_states),
                    "hidden_49": max_errors(
                        expected_states[:, -1], actual_states[:, -1]
                    ),
                    "output_file_order": [
                        files[1].name if velocity_flat is first else files[2].name,
                        files[2].name if states_flat is second else files[1].name,
                    ],
                }
        raise ValueError(f"CubeAI output sizes do not match velocity/hidden outputs: {directory}")

    parity = {
        "encoder_cubeai_host_vs_tflite": max_errors(expected_encoder, encoder_c),
        "gru_hidden_head_cubeai_host_vs_python": compare_outputs(gru_validate),
        "chain_cubeai_host_vs_python": compare_outputs(chain_validate),
    }
    # The GRU graph must be numerically equivalent on its own.  The chained
    # result intentionally includes CubeAI's INT8 encoder backend rounding and
    # is judged by full held-out R2 below, not by bit parity with TFLite.
    for section in ("gru_hidden_head_cubeai_host_vs_python",):
        for output in ("velocity", "gru_states", "hidden_49"):
            if parity[section][output]["max_abs_error"] > 1e-5:
                raise ValueError(f"CubeAI {section}/{output} parity failed: {parity}")

    encoder_info = FP32.find_single(encoder_generate, "*_c_info.json")
    gru_info = FP32.find_single(gru_generate, "*_c_info.json")
    encoder_bin = FP32.find_single(encoder_generate, "*.bin")
    gru_bin = FP32.find_single(gru_generate, "*.bin")
    inspector = workspace / "encoder_generate" / "inspector_indy_encoder" / "workspace"
    generated_source = inspector / "generated"
    dynamic_library = FP32.find_single(inspector / "lib", "libai_indy_encoder.*")
    return (
        {
            "parity": parity,
            "encoder": FP32.network_metrics(encoder_info),
            "gru_hidden_head": FP32.network_metrics(gru_info),
        },
        encoder_bin,
        gru_bin,
        encoder_generate,
        gru_generate,
        generated_source,
        dynamic_library,
    )


def cubeai_host_encoder(
    build_root: Path,
    generated_source: Path,
    dynamic_library: Path,
    weights: Path,
    inputs: np.ndarray,
) -> np.ndarray:
    """Run the generated CubeAI encoder C graph over all held-out windows."""
    workspace = dynamic_library.parent.parent
    include_dir = workspace / "include"
    runner = build_root / "cubeai" / "indy_encoder_host_runner"
    compiler = shutil.which("clang") or shutil.which("cc")
    if compiler is None:
        raise FileNotFoundError("a host C compiler is required for CubeAI validation")
    library_name = dynamic_library.name
    library_stem = (
        library_name[3:].split(".", 1)[0]
        if library_name.startswith("lib")
        else library_name.split(".", 1)[0]
    )
    command = [
        compiler,
        "-std=c99",
        "-O2",
        str(FP32.DEPLOY_DIR / "cubeai_encoder_host_runner.c"),
        f"-I{generated_source}",
        f"-I{include_dir}",
        f"-L{dynamic_library.parent}",
        f"-l{library_stem}",
        "-o",
        str(runner),
    ]
    if sys.platform == "darwin":
        command.append(f"-Wl,-rpath,{dynamic_library.parent}")
    FP32.run(command, build_root / "logs" / "compile_host_runner.log")
    if sys.platform == "darwin":
        linked = subprocess.run(
            ["otool", "-L", str(runner)],
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        relative = next(
            (
                line.strip().split(" ", 1)[0]
                for line in linked.splitlines()[1:]
                if library_name in line
            ),
            None,
        )
        if relative and relative != str(dynamic_library):
            subprocess.run(
                [
                    "install_name_tool",
                    "-change",
                    relative,
                    str(dynamic_library),
                    str(runner),
                ],
                check=True,
            )
    input_path = build_root / "validation" / "held_out_inputs.bin"
    output_path = build_root / "validation" / "held_out_encoder_cubeai.bin"
    np.asarray(inputs, dtype="<f4").tofile(input_path)
    FP32.run(
        [
            str(runner),
            str(weights),
            str(input_path),
            str(output_path),
            str(len(inputs)),
        ],
        build_root / "logs" / "run_host_encoder_held_out.log",
    )
    output = np.fromfile(output_path, dtype="<f4")
    expected = len(inputs) * 50 * 64
    if output.size != expected:
        raise ValueError(
            f"CubeAI host encoder returned {output.size} floats, expected {expected}"
        )
    return output.reshape(len(inputs), 50, 64).astype(np.float32)


def parse_args() -> argparse.Namespace:
    default_cli = (
        Path.home()
        / "STM32Cube"
        / "Repository"
        / "Packs"
        / "STMicroelectronics"
        / "X-CUBE-AI"
        / "10.2.0"
        / "Utilities"
        / "macarm"
        / "stedgeai"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default=SESSION)
    parser.add_argument("--fold", type=int, default=FOLD)
    parser.add_argument("--stedgeai", type=Path, default=default_cli)
    parser.add_argument("--keep-build", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    global SESSION, FOLD, RESULT_ROOT, FINAL_ROOT, CHECKPOINT
    args = parse_args()
    SESSION = args.session
    FOLD = args.fold
    if not 1 <= FOLD <= 5:
        raise ValueError(f"fold must be in 1..5, got {FOLD}")
    RESULT_ROOT = HERE / "results" / f"{SESSION}_fold{FOLD}"
    FINAL_ROOT = MODELS_ROOT / "midsize" / SESSION / "cubeai" / f"fold-{FOLD}"
    CHECKPOINT = (
        MODELS_ROOT
        / "midsize"
        / SESSION
        / f"fold-{FOLD}_best-test-fold.pt"
    )
    cli = args.stedgeai.resolve()
    if not cli.is_file():
        raise FileNotFoundError(cli)
    if RESULT_ROOT.exists():
        if not args.overwrite:
            raise FileExistsError(f"Pilot result already exists: {RESULT_ROOT}; use --overwrite")
        shutil.rmtree(RESULT_ROOT)
    RESULT_ROOT.mkdir(parents=True)
    build_root = RESULT_ROOT / "build"
    model_dir = build_root / "model"
    validation_dir = build_root / "validation"
    model_dir.mkdir(parents=True)
    validation_dir.mkdir(parents=True)

    model, checkpoint = load_checkpoint(CHECKPOINT)
    if checkpoint["session"] != SESSION or int(checkpoint["fold"]) != FOLD:
        raise ValueError("Pilot checkpoint identity mismatch")
    session_spec = next(spec for spec in ROUND3.PHASE7.SESSIONS if spec.name == SESSION)
    session_data = ROUND3.PHASE7.load_session(session_spec)
    fold_data = ROUND3.prepare_fold(session_data, FOLD - 1)
    comparisons = {
        "selected_channel_indices": (fold_data.channels, np.asarray(checkpoint["selected_channel_indices"])),
        "feature_mean": (fold_data.calibration_mean, np.asarray(checkpoint["feature_mean"]).reshape(192)),
        "feature_std": (fold_data.calibration_effective_std, np.asarray(checkpoint["feature_std"]).reshape(192)),
        "target_mean": (fold_data.target_mean, np.asarray(checkpoint["target_mean"])),
        "target_std": (fold_data.target_std, np.asarray(checkpoint["target_std"])),
    }
    for name, (actual, expected) in comparisons.items():
        if not np.allclose(actual, expected, rtol=0, atol=1e-6):
            raise ValueError(f"Phase-13 data/checkpoint mismatch: {name}")

    train_selection = np.linspace(
        0,
        len(fold_data.train_bins) - 1,
        min(REPRESENTATIVE_WINDOWS, len(fold_data.train_bins)),
        dtype=np.int64,
    )
    representative_inputs = ROUND3.rolling_batch(
        fold_data.normalized_features, fold_data.train_bins[train_selection]
    )
    validation_selection = np.linspace(
        0,
        len(fold_data.test_bins) - 1,
        VALIDATION_WINDOWS,
        dtype=np.int64,
    )
    validation_inputs = ROUND3.rolling_batch(
        fold_data.normalized_features, fold_data.test_bins[validation_selection]
    )
    test_inputs = ROUND3.rolling_batch(
        fold_data.normalized_features, fold_data.test_bins
    )

    encoder = build_encoder(model)
    encoder_path = model_dir / "encoder_int8.tflite"
    quantization = quantize_encoder(encoder, representative_inputs, encoder_path)
    gru_model = build_gru_hidden_head(model)
    gru_path = model_dir / "gru_hidden_head.h5"
    gru_model.save(gru_path, include_optimizer=False)
    FP32.sanitize_h5_for_cubeai(gru_path)

    torch_encoded, torch_velocity, torch_hidden = torch_components(
        model, validation_inputs
    )
    quant_encoded = tflite_predict(encoder_path, validation_inputs)
    quant_velocity, quant_hidden = head_from_encoded(model, quant_encoded)
    keras_velocity = []
    keras_states = []
    for sample in quant_encoded:
        velocity, states = gru_model(sample[None], training=False)
        keras_velocity.append(velocity.numpy()[0])
        keras_states.append(states.numpy()[0])
    keras_velocity_array = np.stack(keras_velocity).astype(np.float32)
    keras_states_array = np.stack(keras_states).astype(np.float32)
    python_parity = {
        "causal_encoder_keras_vs_pytorch": max_errors(
            torch_encoded,
            np.concatenate(
                [encoder(sample[None], training=False).numpy() for sample in validation_inputs],
                axis=0,
            ),
        ),
        "tflite_encoder_vs_pytorch": max_errors(torch_encoded, quant_encoded),
        "keras_velocity_vs_torch_gru": max_errors(quant_velocity, keras_velocity_array),
        "keras_hidden49_vs_torch_gru": max_errors(
            quant_hidden, keras_states_array[:, -1]
        ),
        "torch_final_hidden_shape": list(torch_hidden.shape),
    }
    if python_parity["causal_encoder_keras_vs_pytorch"]["max_abs_error"] > 1e-5:
        raise ValueError(f"Keras encoder rewrite parity failed: {python_parity}")
    if python_parity["keras_velocity_vs_torch_gru"]["max_abs_error"] > 1e-5:
        raise ValueError(f"Keras GRU velocity parity failed: {python_parity}")
    if python_parity["keras_hidden49_vs_torch_gru"]["max_abs_error"] > 1e-5:
        raise ValueError(f"Keras GRU hidden parity failed: {python_parity}")

    np.savetxt(
        validation_dir / "model_inputs.csv",
        validation_inputs.reshape(VALIDATION_WINDOWS, -1),
        delimiter=",",
        fmt="%.9g",
    )
    np.savetxt(
        validation_dir / "encoder_expected.csv",
        quant_encoded.reshape(VALIDATION_WINDOWS, -1),
        delimiter=",",
        fmt="%.9g",
    )
    np.savetxt(
        validation_dir / "velocity_expected.csv",
        keras_velocity_array.reshape(VALIDATION_WINDOWS, -1),
        delimiter=",",
        fmt="%.9g",
    )
    np.savetxt(
        validation_dir / "hidden_expected.csv",
        keras_states_array.reshape(VALIDATION_WINDOWS, -1),
        delimiter=",",
        fmt="%.9g",
    )

    (
        cubeai,
        encoder_bin,
        gru_bin,
        encoder_generated,
        gru_generated,
        encoder_generated_source,
        encoder_dynamic_library,
    ) = run_cubeai(cli, build_root, encoder_path, gru_path, VALIDATION_WINDOWS)

    float_encoded, float_velocity, _ = torch_components(model, test_inputs)
    del float_encoded
    quant_test_encoded = tflite_predict(encoder_path, test_inputs)
    quant_test_velocity, _ = head_from_encoded(model, quant_test_encoded)
    cubeai_test_encoded = cubeai_host_encoder(
        build_root,
        encoder_generated_source,
        encoder_dynamic_library,
        encoder_bin,
        test_inputs,
    )
    cubeai_test_velocity, _ = head_from_encoded(model, cubeai_test_encoded)
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
    target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)
    float_prediction = float_velocity[:, -1] * target_std + target_mean
    quant_prediction = quant_test_velocity[:, -1] * target_std + target_mean
    cubeai_prediction = cubeai_test_velocity[:, -1] * target_std + target_mean
    target = fold_data.velocity[fold_data.test_bins]
    held_out = {
        "float32_pytorch": r2_metrics(target, float_prediction),
        "tflite_int8_encoder_torch_gru": r2_metrics(target, quant_prediction),
        "cubeai_c_int8_encoder_torch_gru": r2_metrics(target, cubeai_prediction),
    }
    held_out["tflite_r2_mean_drop"] = float(
        held_out["float32_pytorch"]["r2_mean"]
        - held_out["tflite_int8_encoder_torch_gru"]["r2_mean"]
    )
    held_out["cubeai_r2_mean_drop"] = float(
        held_out["float32_pytorch"]["r2_mean"]
        - held_out["cubeai_c_int8_encoder_torch_gru"]["r2_mean"]
    )
    held_out["accepted_max_drop"] = R2_DROP_LIMIT
    held_out["accepted"] = held_out["cubeai_r2_mean_drop"] <= R2_DROP_LIMIT
    if not held_out["accepted"]:
        raise ValueError(f"Pilot held-out R2 gate failed: {held_out}")

    package = RESULT_ROOT / "package"
    package.mkdir()
    shutil.copy2(encoder_path, package / encoder_path.name)
    shutil.copy2(gru_path, package / gru_path.name)
    shutil.copy2(encoder_bin, package / "encoder.weights.bin")
    shutil.copy2(gru_bin, package / "gru_hidden_head.weights.bin")
    shutil.copytree(encoder_generated, package / "encoder_generated")
    shutil.copytree(gru_generated, package / "gru_hidden_head_generated")
    constants = {
        "feature_std_floor": np.asarray(checkpoint["feature_std_floor"], dtype=np.float32),
        "target_mean": target_mean,
        "target_std": target_std,
        "selected_channel_indices": np.asarray(checkpoint["selected_channel_indices"], dtype=np.int64),
    }
    bundle_checkpoint = dict(checkpoint)
    bundle_checkpoint["model_id"] = f"{SESSION}_f{FOLD}"
    bundle_path = package / f"{SESSION}_fold{FOLD}.aibundle"
    bundle = FP32.build_bundle(
        bundle_path,
        CHECKPOINT,
        bundle_checkpoint,
        constants,
        package / "encoder.weights.bin",
        package / "gru_hidden_head.weights.bin",
        graph_abi_id=GRAPH_ABI_ID,
    )
    version = subprocess.run(
        [str(cli), "--version"], text=True, capture_output=True, check=True
    ).stdout.strip().splitlines()
    manifest = {
        "schema_version": 1,
        "phase": "phase14_cubeai_best_fold_conversion",
        "status": "best_fold_validated_not_promoted_to_firmware_or_gui",
        "session": SESSION,
        "fold": FOLD,
        "best_test_fold": True,
        "checkpoint": artifact(CHECKPOINT),
        "checkpoint_shared_by_tiers": ["midsize", "large"],
        "paper_reporting": {
            "required_metric": "30-fold mean and standard deviation",
            "midsize_r2_mean": 0.7411375800768535,
            "midsize_r2_std": 0.06559769297745317,
            "pilot_fold_r2_is_descriptive_only": True,
        },
        "graph": {
            "abi_id": GRAPH_ABI_ID,
            "encoder": "INT8 TCN with float32 input/output ABI",
            "gru_hidden_head": (
                "FP32 GRU with velocity and state-sequence outputs; hidden[49] is row 49"
            ),
            "state_output_shape": [1, 50, 64],
            "hidden_49_view": "state_output[0][49][0:64]",
            "state_output_bytes_float32": 12800,
        },
        "quantization": quantization,
        "python_parity": python_parity,
        "cubeai": cubeai,
        "held_out_accuracy": held_out,
        "components": {
            "encoder_weights": artifact(package / "encoder.weights.bin"),
            "gru_hidden_head_weights": artifact(package / "gru_hidden_head.weights.bin"),
            "encoder_model": artifact(package / encoder_path.name),
            "gru_hidden_head_model": artifact(package / gru_path.name),
        },
        "bundle": bundle,
        "cubeai_version": version,
        "next_gate": (
            "After review, batch-convert the highlighted best fold for the remaining five sessions; "
            "continue reporting the 30-fold average."
        ),
    }
    write_json(package / "manifest.json", manifest)
    write_json(RESULT_ROOT / "pilot_summary.json", manifest)
    if FINAL_ROOT.exists():
        shutil.rmtree(FINAL_ROOT)
    FINAL_ROOT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(package, FINAL_ROOT)
    if not args.keep_build:
        shutil.rmtree(build_root)
    print(json.dumps({
        "status": manifest["status"],
        "paper_r2": "0.7411 ± 0.0656",
        "pilot_float_r2": held_out["float32_pytorch"]["r2_mean"],
        "pilot_tflite_quantized_r2": held_out["tflite_int8_encoder_torch_gru"]["r2_mean"],
        "pilot_cubeai_c_quantized_r2": held_out["cubeai_c_int8_encoder_torch_gru"]["r2_mean"],
        "cubeai_r2_drop": held_out["cubeai_r2_mean_drop"],
        "final_package": str(FINAL_ROOT),
    }, indent=2))


if __name__ == "__main__":
    main()
