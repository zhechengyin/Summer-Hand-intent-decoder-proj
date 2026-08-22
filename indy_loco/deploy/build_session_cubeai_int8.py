#!/usr/bin/env python3
"""Build the six dynamic Encoder-INT8 / GRU-FP32 Cube.AI bundles.

The external ABI remains float32 ``1x192x50 -> 1x50x64`` so firmware
preprocessing and the validated FP32 GRU/head do not change.  The causal TCN
is exported as integer-friendly time taps plus 1x1 convolutions because
X-CUBE-AI 10.2 otherwise lowers dilated INT8 convolutions back to FP32.
Representative calibration and accuracy scoring use only the selected fold's
training and held-out test reaches respectively.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import torch


DEPLOY_DIR = Path(__file__).resolve().parent
INDY_ROOT = DEPLOY_DIR.parent
REPOSITORY_ROOT = INDY_ROOT.parent
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import build_session_cubeai as fp32  # noqa: E402
from indy_loco.experiments.active import (  # noqa: E402
    phase10_session_deployment_candidates as p10,
)
from indy_loco.history.experiments.phase7.phase7_ann_vs_snn_fivefold import (  # noqa: E402
    SESSION_BY_NAME,
    aggregate_40ms,
    binned_reach_bounds,
    eligible_reaches,
    load_session,
    make_fold_indices,
    split_fold,
)


OUTPUT_NAME = "cubeai_int8"
GRAPH_ABI_ID = "tcn64i8x6-gru64f32-xcai10-v2"
REPRESENTATIVE_WINDOWS = 512
ACCEPTED_R2_MEAN_DROP = 0.01
ENCODER_SOURCE_FILES = (
    "indy_encoder.c",
    "indy_encoder.h",
    "indy_encoder_config.h",
    "indy_encoder_data.c",
    "indy_encoder_data.h",
    "indy_encoder_data_params.c",
    "indy_encoder_data_params.h",
)
GRAPH_NAMES = {
    "indy_20160622_01": "indy_enc_i622",
    "indy_20160630_01": "indy_enc_i630",
    "indy_20170131_02": "indy_enc_i131",
    "loco_20170210_03": "indy_enc_l210",
    "loco_20170215_02": "indy_enc_l215",
    "loco_20170301_05": "indy_enc_l301",
}


def build_encoder(model: torch.nn.Module) -> tf.keras.Model:
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
        oldest = tf.keras.layers.Cropping1D(
            cropping=(0, 2 * dilation), name=f"tcn_{index}_oldest"
        )(padded)
        middle = tf.keras.layers.Cropping1D(
            cropping=(dilation, dilation), name=f"tcn_{index}_middle"
        )(padded)
        newest = tf.keras.layers.Cropping1D(
            cropping=(2 * dilation, 0), name=f"tcn_{index}_newest"
        )(padded)
        taps = tf.keras.layers.Concatenate(
            axis=-1, name=f"tcn_{index}_taps"
        )((oldest, middle, newest))
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
        tap_kernel = np.concatenate(
            (kernel[0], kernel[1], kernel[2]), axis=0
        )[None]
        encoder.get_layer(f"tcn_{index}_conv").set_weights(
            [tap_kernel, convolution.bias.detach().numpy()]
        )
    return encoder


def export_gru_head(model: torch.nn.Module, destination: Path) -> tf.keras.Model:
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
    output = tf.keras.layers.Dense(2, name="velocity_norm")(states)
    keras_model = tf.keras.Model(encoded, output, name="indy_gru_head")
    weight_ih = model.gru.weight_ih_l0.detach().numpy().T
    weight_hh = model.gru.weight_hh_l0.detach().numpy().T
    bias_ih = model.gru.bias_ih_l0.detach().numpy()
    bias_hh = model.gru.bias_hh_l0.detach().numpy()
    keras_model.get_layer("gru").set_weights(
        [
            fp32.reorder_gru_gates(weight_ih, axis=1),
            fp32.reorder_gru_gates(weight_hh, axis=1),
            np.stack(
                (
                    fp32.reorder_gru_gates(bias_ih, axis=0),
                    fp32.reorder_gru_gates(bias_hh, axis=0),
                )
            ),
        ]
    )
    keras_model.get_layer("velocity_norm").set_weights(
        [model.head.weight.detach().numpy().T, model.head.bias.detach().numpy()]
    )
    keras_model.save(destination, include_optimizer=False)
    fp32.sanitize_h5_for_cubeai(destination)
    return keras_model


def torch_encoder(model: torch.nn.Module, inputs: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        values = model.spatial(torch.from_numpy(inputs))
        for convolution, padding in zip(
            model.convolutions, model.padding, strict=True
        ):
            values = model.activation(
                convolution(values)[:, :, :-padding] + values
            )
        return values.transpose(1, 2).numpy().astype(np.float32)


def session_arrays(
    session: str, checkpoint: dict[str, Any], session_dir: Path
) -> dict[str, Any]:
    data = load_session(SESSION_BY_NAME[session])
    counts_all, velocity = aggregate_40ms(data)
    bounds = binned_reach_bounds(data)
    train_reaches, _, test_reaches = split_fold(
        make_fold_indices(eligible_reaches(data)),
        int(checkpoint["selected_fold"]) - 1,
    )
    channels = np.asarray(
        checkpoint["selected_channel_indices"], dtype=np.int64
    )
    counts = counts_all[channels].astype(np.float32)
    features = p10.continuous_features(counts)
    with np.load(
        session_dir / "deployment_golden_vectors.npz", allow_pickle=False
    ) as golden:
        calibration_mean = golden["calibration_mean"].astype(np.float32)
        effective_std = golden["calibration_effective_std"].astype(np.float32)
    normalized = (
        (features - calibration_mean[:, None]) / effective_std[:, None]
    ).astype(np.float32)

    representative_bins: list[int] = []
    for reach in train_reaches:
        start, stop = bounds[int(reach)]
        first = max(int(start) + p10.WINDOW_BINS - 1, p10.CALIBRATION_BINS - 1)
        representative_bins.extend(range(first, int(stop)))
    if len(representative_bins) < 64:
        raise ValueError(
            f"{session}: only {len(representative_bins)} training representative bins"
        )
    candidate_bins = np.asarray(representative_bins, dtype=np.int64)
    representative_count = min(REPRESENTATIVE_WINDOWS, len(candidate_bins))
    selection = np.linspace(
        0, len(candidate_bins) - 1, representative_count, dtype=np.int64
    )
    representative_inputs = p10.rolling_inputs(
        normalized, candidate_bins[selection]
    )

    reference_bins, reference_target, _ = p10.phase7_reference(
        checkpoint["_model"],
        counts,
        velocity,
        bounds,
        test_reaches,
        checkpoint,
        torch.device("cpu"),
    )
    post_calibration = reference_bins >= p10.CALIBRATION_BINS - 1
    test_bins = reference_bins[post_calibration]
    return {
        "representative_inputs": representative_inputs,
        "test_inputs": p10.rolling_inputs(normalized, test_bins),
        "test_target": reference_target[post_calibration].astype(np.float32),
        "test_bins": test_bins,
        "training_reach_count": int(len(train_reaches)),
        "test_reach_count": int(len(test_reaches)),
    }


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
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    model = converter.convert()
    destination.write_bytes(model)

    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()
    details = interpreter.get_tensor_details()
    operators = [item["op_name"] for item in interpreter._get_ops_details()]
    internal = [
        item for item in details if item["dtype"] not in (np.int32, np.int64)
    ]
    int8_tensors = sum(item["dtype"] == np.int8 for item in internal)
    float_tensors = sum(item["dtype"] == np.float32 for item in internal)
    return {
        "file": destination.name,
        "bytes": len(model),
        "sha256": hashlib.sha256(model).hexdigest(),
        "representative_windows": len(representative_inputs),
        "representative_source": "selected_best_fold_training_reaches_only",
        "external_input_dtype": str(interpreter.get_input_details()[0]["dtype"]),
        "external_output_dtype": str(interpreter.get_output_details()[0]["dtype"]),
        "int8_tensor_count": int(int8_tensors),
        "float_tensor_count": int(float_tensors),
        "operators": operators,
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


def head_predict(
    model: torch.nn.Module, encoded: np.ndarray, batch: int = 128
) -> np.ndarray:
    output = []
    with torch.inference_mode():
        for start in range(0, len(encoded), batch):
            states, _ = model.gru(torch.from_numpy(encoded[start : start + batch]))
            output.append(model.head(states).numpy())
    return np.concatenate(output).astype(np.float32)


def full_predict(
    model: torch.nn.Module, inputs: np.ndarray, batch: int = 128
) -> np.ndarray:
    output = []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch):
            output.append(model(torch.from_numpy(inputs[start : start + batch])).numpy())
    return np.concatenate(output).astype(np.float32)


def score_quantized(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    tflite_path: Path,
    arrays: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    inputs = arrays["test_inputs"]
    quant_encoded = tflite_predict(tflite_path, inputs)
    quant_normalized = head_predict(model, quant_encoded)[:, -1]
    float_normalized = full_predict(model, inputs)[:, -1]
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
    target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)
    quant_prediction = quant_normalized * target_std + target_mean
    float_prediction = float_normalized * target_std + target_mean
    float_metrics = p10.metric_values(arrays["test_target"], float_prediction)
    quant_metrics = p10.metric_values(arrays["test_target"], quant_prediction)
    drop = float_metrics["r2_mean"] - quant_metrics["r2_mean"]
    report = {
        "split": f"selected_best_fold_{checkpoint['selected_fold']}_held_out_test",
        "held_out_bins": int(len(inputs)),
        "test_reaches": arrays["test_reach_count"],
        "float32": float_metrics,
        "encoder_int8_gru_fp32": quant_metrics,
        "r2_mean_drop": float(drop),
        "prediction_max_abs_error": float(
            np.max(np.abs(float_prediction - quant_prediction))
        ),
        "prediction_mean_abs_error": float(
            np.mean(np.abs(float_prediction - quant_prediction))
        ),
        "accepted_max_r2_mean_drop": ACCEPTED_R2_MEAN_DROP,
        "accepted": bool(drop <= ACCEPTED_R2_MEAN_DROP),
    }
    if not report["accepted"]:
        raise ValueError(f"INT8 accuracy gate failed: {report}")
    return report, quant_normalized


def score_cubeai_host(
    checkpoint: dict[str, Any],
    arrays: dict[str, Any],
    float_metrics: dict[str, float],
    normalized_prediction: np.ndarray,
) -> dict[str, Any]:
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
    target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)
    prediction = normalized_prediction * target_std + target_mean
    metrics = p10.metric_values(arrays["test_target"], prediction)
    drop = float_metrics["r2_mean"] - metrics["r2_mean"]
    report = {
        "encoder_int8_gru_fp32": metrics,
        "r2_mean_drop": float(drop),
        "accepted_max_r2_mean_drop": ACCEPTED_R2_MEAN_DROP,
        "accepted": bool(drop <= ACCEPTED_R2_MEAN_DROP),
    }
    if not report["accepted"]:
        raise ValueError(f"Cube.AI host INT8 accuracy gate failed: {report}")
    return report


def write_validation_csv(
    model: torch.nn.Module,
    golden_inputs: np.ndarray,
    tflite_path: Path,
    validation_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    validation_dir.mkdir(parents=True, exist_ok=True)
    float_encoder = torch_encoder(model, golden_inputs)
    quant_encoder = tflite_predict(tflite_path, golden_inputs)
    quant_output = head_predict(model, quant_encoder)
    float_output = full_predict(model, golden_inputs)
    np.savetxt(
        validation_dir / "model_inputs.csv",
        golden_inputs.reshape(len(golden_inputs), -1),
        delimiter=",",
        fmt="%.9g",
    )
    np.savetxt(
        validation_dir / "encoder_quantized_expected.csv",
        quant_encoder.reshape(len(golden_inputs), -1),
        delimiter=",",
        fmt="%.9g",
    )
    np.savetxt(
        validation_dir / "model_quantized_expected.csv",
        quant_output.reshape(len(golden_inputs), -1),
        delimiter=",",
        fmt="%.9g",
    )
    causal_encoder = build_encoder(model)
    causal_values = np.concatenate(
        [
            causal_encoder(sample[None], training=False).numpy()
            for sample in golden_inputs
        ],
        axis=0,
    )
    parity = {
        "causal_tap_keras_vs_pytorch_encoder": fp32.max_errors(
            float_encoder, causal_values,
        ),
        "tflite_int8_encoder_vs_pytorch": fp32.max_errors(
            float_encoder, quant_encoder
        ),
        "tflite_int8_chain_vs_pytorch": fp32.max_errors(
            float_output, quant_output
        ),
    }
    if parity["causal_tap_keras_vs_pytorch_encoder"]["max_abs_error"] > 1e-5:
        raise ValueError(f"causal tap rewrite parity failed: {parity}")
    return quant_encoder, quant_output, parity


def operation_mix(report_path: Path) -> dict[str, Any]:
    text = report_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^\s*(\w+)\s+([\d,]+)\s+([\d.]+)%\s*$", re.MULTILINE
    )
    values = {
        name: {"operations": int(count.replace(",", "")), "percent": float(percent)}
        for name, count, percent in pattern.findall(text)
    }
    return values


def cubeai_convert(
    cli: Path,
    build_dir: Path,
    encoder_model: Path,
    gru_model: Path,
    expected_output: np.ndarray,
) -> tuple[dict[str, Any], Path, Path, Path, Path]:
    validation = build_dir / "validation"
    cube = build_dir / "cubeai"
    workspace = build_dir / "workspace"
    logs = build_dir / "logs"
    encoder_validate = cube / "encoder_validate"
    encoder_generate = cube / "encoder_generate"
    gru_generate = cube / "gru_generate"
    chain_validate = cube / "chain_validate"

    fp32.run(
        fp32.cubeai_base(
            cli, "validate", encoder_model, "tflite", "indy_encoder",
            workspace / "encoder_validate", encoder_validate,
        )
        + [
            "--mode", "host",
            "--valinput", str(validation / "model_inputs.csv"),
            "--valoutput", str(validation / "encoder_quantized_expected.csv"),
            "--save-csv",
        ],
        logs / "validate_encoder_int8.log",
    )
    fp32.run(
        fp32.cubeai_base(
            cli, "generate", encoder_model, "tflite", "indy_encoder",
            workspace / "encoder_generate", encoder_generate,
        ) + ["--binary", "--dll"],
        logs / "generate_encoder_int8.log",
    )
    fp32.run(
        fp32.cubeai_base(
            cli, "generate", gru_model, "keras", "indy_gru_head",
            workspace / "gru_generate", gru_generate,
        ) + ["--binary"],
        logs / "generate_gru_head.log",
    )
    encoder_c_output = fp32.find_single(
        encoder_validate, "*_val_c_outputs_1.csv"
    )
    fp32.run(
        fp32.cubeai_base(
            cli, "validate", gru_model, "keras", "indy_gru_head",
            workspace / "chain_validate", chain_validate,
        )
        + [
            "--mode", "host",
            "--valinput", str(encoder_c_output),
            "--valoutput", str(validation / "model_quantized_expected.csv"),
            "--save-csv",
        ],
        logs / "validate_chain_int8.log",
    )

    encoder_c = fp32.load_flat_csv(
        encoder_c_output, (len(expected_output), 50, 64)
    )
    encoder_expected = fp32.load_flat_csv(
        validation / "encoder_quantized_expected.csv",
        (len(expected_output), 50, 64),
    )
    chain_c = fp32.load_flat_csv(
        fp32.find_single(chain_validate, "*_val_c_outputs_1.csv"),
        expected_output.shape,
    )
    parity = {
        "encoder_cubeai_host_vs_tflite": fp32.max_errors(
            encoder_expected, encoder_c
        ),
        "chain_cubeai_host_vs_tflite_python": fp32.max_errors(
            expected_output, chain_c
        ),
    }
    encoder_info = fp32.find_single(encoder_generate, "*_c_info.json")
    gru_info = fp32.find_single(gru_generate, "*_c_info.json")
    encoder_metrics = fp32.network_metrics(encoder_info)
    gru_metrics = fp32.network_metrics(gru_info)
    encoder_binary = fp32.find_single(encoder_generate, "*.bin")
    gru_binary = fp32.find_single(gru_generate, "*.bin")
    if encoder_binary.stat().st_size != encoder_metrics["weights_bytes"]:
        raise ValueError("INT8 encoder binary size differs from Cube.AI report")
    report_path = fp32.find_single(encoder_generate, "*_generate_report.txt")
    inspector = workspace / "encoder_generate" / "inspector_indy_encoder" / "workspace"
    generated_source = inspector / "generated"
    dynamic_library = fp32.find_single(inspector / "lib", "libai_indy_encoder.*")
    return (
        {
            "parity": parity,
            "encoder": encoder_metrics,
            "gru_head": gru_metrics,
            "encoder_operation_mix": operation_mix(report_path),
        },
        encoder_binary,
        gru_binary,
        generated_source,
        dynamic_library,
    )


def cubeai_host_encoder(
    build_dir: Path,
    generated_source: Path,
    dynamic_library: Path,
    weights: Path,
    inputs: np.ndarray,
) -> np.ndarray:
    workspace = dynamic_library.parent.parent
    include_dir = workspace / "include"
    runner = build_dir / "cubeai" / "indy_encoder_host_runner"
    compiler = shutil.which("clang") or shutil.which("cc")
    if compiler is None:
        raise FileNotFoundError("a host C compiler is required for Cube.AI validation")
    library_name = dynamic_library.name
    if library_name.startswith("lib"):
        library_stem = library_name[3:].split(".", 1)[0]
    else:
        library_stem = library_name.split(".", 1)[0]
    command = [
        compiler,
        "-std=c99",
        "-O2",
        str(DEPLOY_DIR / "cubeai_encoder_host_runner.c"),
        f"-I{generated_source}",
        f"-I{include_dir}",
        f"-L{dynamic_library.parent}",
        f"-l{library_stem}",
        "-o",
        str(runner),
    ]
    if sys.platform == "darwin":
        command.append(f"-Wl,-rpath,{dynamic_library.parent}")
    fp32.run(command, build_dir / "logs" / "compile_host_runner.log")
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

    input_path = build_dir / "validation" / "held_out_inputs.bin"
    output_path = build_dir / "validation" / "held_out_encoder_cubeai.bin"
    np.asarray(inputs, dtype="<f4").tofile(input_path)
    fp32.run(
        [
            str(runner),
            str(weights),
            str(input_path),
            str(output_path),
            str(len(inputs)),
        ],
        build_dir / "logs" / "run_host_encoder_held_out.log",
    )
    output = np.fromfile(output_path, dtype="<f4")
    expected = len(inputs) * 50 * 64
    if output.size != expected:
        raise ValueError(
            f"Cube.AI host encoder returned {output.size} floats, expected {expected}"
        )
    return output.reshape(len(inputs), 50, 64).astype(np.float32)


def normalized_graph_hash(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"^\s*\* @date\s+.*$", "", text, flags=re.MULTILINE)
    text = re.sub(
        r'(#define AI_INDY_ENCODER_MODEL_SIGNATURE\s+)"[^"]+"',
        r'\1"<dynamic-weights>"',
        text,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_session(session: str, cli: Path, keep_build: bool) -> dict[str, Any]:
    session_dir = fp32.MIDSIZE_DIR / session
    output_dir = session_dir / OUTPUT_NAME
    build_dir = output_dir / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)

    model, checkpoint = fp32.load_candidate(session_dir)
    checkpoint["_model"] = model
    constants = fp32.validate_constants(session_dir, checkpoint)
    golden_inputs, _, golden_path = fp32.load_golden(session_dir, checkpoint)
    arrays = session_arrays(session, checkpoint, session_dir)
    del checkpoint["_model"]

    model_dir = build_dir / "model"
    model_dir.mkdir(parents=True)
    encoder = build_encoder(model)
    encoder_path = model_dir / "encoder_int8.tflite"
    quantization = quantize_encoder(
        encoder, arrays["representative_inputs"], encoder_path
    )
    gru_path = model_dir / "gru_head.h5"
    export_gru_head(model, gru_path)
    quant_encoder, quant_output, python_parity = write_validation_csv(
        model, golden_inputs, encoder_path, build_dir / "validation"
    )
    tflite_accuracy, _ = score_quantized(model, checkpoint, encoder_path, arrays)
    (
        cubeai,
        encoder_source,
        gru_source,
        generated_source,
        encoder_library,
    ) = cubeai_convert(
        cli,
        build_dir,
        encoder_path,
        gru_path,
        quant_output,
    )
    cubeai_held_out_encoded = cubeai_host_encoder(
        build_dir,
        generated_source,
        encoder_library,
        encoder_source,
        arrays["test_inputs"],
    )
    cubeai_held_out_prediction = head_predict(
        model, cubeai_held_out_encoded
    )[:, -1]
    cubeai_accuracy = score_cubeai_host(
        checkpoint,
        arrays,
        tflite_accuracy["float32"],
        cubeai_held_out_prediction,
    )
    accuracy = {
        "split": tflite_accuracy["split"],
        "held_out_bins": tflite_accuracy["held_out_bins"],
        "test_reaches": tflite_accuracy["test_reaches"],
        "float32": tflite_accuracy["float32"],
        "tflite_encoder_int8_gru_fp32": tflite_accuracy[
            "encoder_int8_gru_fp32"
        ],
        "tflite_r2_mean_drop": tflite_accuracy["r2_mean_drop"],
        "cubeai_encoder_int8_gru_fp32": cubeai_accuracy[
            "encoder_int8_gru_fp32"
        ],
        "cubeai_r2_mean_drop": cubeai_accuracy["r2_mean_drop"],
        "accepted_max_r2_mean_drop": ACCEPTED_R2_MEAN_DROP,
        "accepted": cubeai_accuracy["accepted"],
    }

    output_dir.mkdir(exist_ok=True)
    shutil.copyfile(encoder_path, output_dir / encoder_path.name)
    encoder_destination = output_dir / "encoder.weights.bin"
    gru_destination = output_dir / "gru_head.weights.bin"
    shutil.copyfile(encoder_source, encoder_destination)
    shutil.copyfile(gru_source, gru_destination)
    source_dir = output_dir / "generated"
    source_dir.mkdir(exist_ok=True)
    for name in ENCODER_SOURCE_FILES:
        shutil.copyfile(generated_source / name, source_dir / name)

    bundle_path = output_dir / f"{session}.aibundle"
    bundle = fp32.build_bundle(
        bundle_path,
        session_dir / "deployment_candidate.pt",
        checkpoint,
        constants,
        encoder_destination,
        gru_destination,
        graph_abi_id=GRAPH_ABI_ID,
    )
    report = {
        "schema_version": 1,
        "model_id": session,
        "source_channel_count": int(checkpoint["source_channel_count"]),
        "selected_fold": int(checkpoint["selected_fold"]),
        "status": "encoder_int8_gru_fp32_validated",
        "promotion_claimed": False,
        "graph_abi_id": GRAPH_ABI_ID,
        "encoder_graph_name": GRAPH_NAMES[session],
        "quantization": quantization,
        "held_out_accuracy": accuracy,
        "parity": python_parity | cubeai["parity"],
        "components": {
            "encoder": fp32.artifact_info(encoder_destination) | cubeai["encoder"],
            "gru_head": fp32.artifact_info(gru_destination) | cubeai["gru_head"],
        },
        "combined": {
            "weights_bytes": encoder_destination.stat().st_size
            + gru_destination.stat().st_size,
            "conservative_activations_bytes": cubeai["encoder"]["activations_bytes"]
            + cubeai["gru_head"]["activations_bytes"],
            "macc": cubeai["encoder"]["macc"] + cubeai["gru_head"]["macc"],
        },
        "encoder_operation_mix": cubeai["encoder_operation_mix"],
        "encoder_graph_source_hashes": {
            name: normalized_graph_hash(source_dir / name)
            for name in ENCODER_SOURCE_FILES
        },
        "training_representative_data": {
            "windows": int(len(arrays["representative_inputs"])),
            "maximum_windows": REPRESENTATIVE_WINDOWS,
            "training_reaches": arrays["training_reach_count"],
            "validation_or_test_reaches_used": False,
        },
        "deployment_golden_vectors": fp32.artifact_info(golden_path),
        "bundle": bundle,
        "python_environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "tensorflow": tf.__version__,
            "keras": tf.keras.__version__,
            "numpy": np.__version__,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if not keep_build:
        shutil.rmtree(build_dir)
    return report


def write_registry() -> dict[str, Any]:
    reports = []
    for session in fp32.SESSIONS:
        path = fp32.MIDSIZE_DIR / session / OUTPUT_NAME / "manifest.json"
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    encoder_sizes = {
        (
            report["components"]["encoder"]["weights_bytes"],
            report["components"]["encoder"]["activations_bytes"],
        )
        for report in reports
    }
    if len(encoder_sizes) != 1:
        raise ValueError(f"session encoder interfaces differ: {encoder_sizes}")
    if len({report["encoder_graph_name"] for report in reports}) != len(reports):
        raise ValueError("session encoder graph names must be unique")
    registry = {
        "schema_version": 1,
        "bundle_format": fp32.MODEL_BUNDLE_VERSION,
        "graph_abi_id": GRAPH_ABI_ID,
        "status": "encoder_int8_gru_fp32_all_sessions_validated",
        "promotion_claimed": False,
        "session_specific_encoder_graphs_required": True,
        "reason": "X-CUBE-AI embeds session-specific INT8 quantization metadata in C",
        "sessions": [
            {
                "model_id": report["model_id"],
                "encoder_graph_name": report["encoder_graph_name"],
                "manifest": f"{report['model_id']}/{OUTPUT_NAME}/manifest.json",
                "bundle": (
                    f"{report['model_id']}/{OUTPUT_NAME}/"
                    f"{report['model_id']}.aibundle"
                ),
                "bundle_bytes": report["bundle"]["total_bytes"],
                "bundle_crc32": report["bundle"]["crc32"],
                "bundle_sha256": report["bundle"]["sha256"],
                "held_out_r2_mean": report["held_out_accuracy"]
                ["cubeai_encoder_int8_gru_fp32"]["r2_mean"],
                "r2_mean_drop": report["held_out_accuracy"]
                ["cubeai_r2_mean_drop"],
            }
            for report in reports
        ],
    }
    (fp32.MIDSIZE_DIR / "cubeai_int8_bundles.json").write_text(
        json.dumps(registry, indent=2) + "\n", encoding="utf-8"
    )
    return registry


def repack_session(session: str) -> dict[str, Any]:
    session_dir = fp32.MIDSIZE_DIR / session
    output_dir = session_dir / OUTPUT_NAME
    checkpoint_path = session_dir / "deployment_candidate.pt"
    _, checkpoint = fp32.load_candidate(session_dir)
    constants = fp32.validate_constants(session_dir, checkpoint)
    manifest_path = output_dir / "manifest.json"
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    report["graph_abi_id"] = GRAPH_ABI_ID
    report["encoder_graph_name"] = GRAPH_NAMES[session]
    report["source_channel_count"] = int(checkpoint["source_channel_count"])
    report["bundle"] = fp32.build_bundle(
        output_dir / f"{session}.aibundle",
        checkpoint_path,
        checkpoint,
        constants,
        output_dir / "encoder.weights.bin",
        output_dir / "gru_head.weights.bin",
        graph_abi_id=GRAPH_ABI_ID,
    )
    manifest_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_cli = (
        Path.home()
        / "STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.2.0"
        / "Utilities/macarm/stedgeai"
    )
    parser.add_argument("--stedgeai", type=Path, default=default_cli)
    parser.add_argument("--session", choices=fp32.SESSIONS, action="append")
    parser.add_argument("--keep-build", action="store_true")
    parser.add_argument("--repack-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cli = args.stedgeai.resolve()
    if not cli.is_file():
        raise FileNotFoundError(f"Cube.AI CLI not found: {cli}")
    sessions = tuple(args.session or fp32.SESSIONS)
    for session in sessions:
        if args.repack_only:
            print(f"[{session}] repack validated INT8 bundle")
            repack_session(session)
            continue
        print(f"[{session}] training-only PTQ, held-out scoring, Cube.AI validation")
        report = build_session(session, cli, args.keep_build)
        print(
            f"[{session}] Cube.AI R2={report['held_out_accuracy']['cubeai_encoder_int8_gru_fp32']['r2_mean']:.6f}, "
            f"drop={report['held_out_accuracy']['cubeai_r2_mean_drop']:.6f}, "
            f"encoder={report['components']['encoder']['bytes']} bytes"
        )
    if set(sessions) == set(fp32.SESSIONS):
        registry = write_registry()
        print(f"validated {len(registry['sessions'])} session-graph INT8 bundles")


if __name__ == "__main__":
    main()
