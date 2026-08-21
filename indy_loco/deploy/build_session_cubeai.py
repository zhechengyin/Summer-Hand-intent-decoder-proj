#!/usr/bin/env python3
"""Build and validate six session-specific Cube.AI weight bundles.

The conversion intentionally reuses the verified Phase 6 split topology:

* ``indy_encoder``: ONNX LayerNorm + causal TCN
* ``indy_gru_head``: Keras reset-after GRU + linear head

Cube.AI itself produces the component weight binaries via ``--binary``.  This
script never reconstructs or guesses the private Cube.AI weight layout.
"""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import h5py
import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
import torch
import torch.nn as nn


DEPLOY_DIR = Path(__file__).resolve().parent
INDY_ROOT = DEPLOY_DIR.parent
REPOSITORY_ROOT = INDY_ROOT.parent
MIDSIZE_DIR = INDY_ROOT / "models" / "midsize"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from indy_loco.models.midsize.model import MidsizeTCNGRU  # noqa: E402


SESSIONS = (
    "indy_20160622_01",
    "indy_20160630_01",
    "indy_20170131_02",
    "loco_20170210_03",
    "loco_20170215_02",
    "loco_20170301_05",
)

EXPECTED_STATUS = "deployment_candidate_replay_complete"
MODEL_BUNDLE_VERSION = "bci-cubeai-bundle-v1"
GRAPH_ABI_ID = "tcn64-gru64-xcubeai10.2-f32-v1"
BUNDLE_MAGIC = b"BCIAIB1\0"
BUNDLE_HEADER_SIZE = 256
BUNDLE_ALIGNMENT = 32
BUNDLE_HEADER_CRC_OFFSET = 196


class Encoder(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.spatial = model.spatial
        self.convolutions = model.convolutions
        self.padding = model.padding
        self.activation = model.activation

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.spatial(values)
        for convolution, padding in zip(
            self.convolutions, self.padding, strict=True
        ):
            encoded = self.activation(
                convolution(encoded)[:, :, :-padding] + encoded
            )
        return encoded.transpose(1, 2)


def sha256_bytes(values: bytes) -> str:
    return hashlib.sha256(values).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def crc32_bytes(values: bytes) -> int:
    return binascii.crc32(values) & 0xFFFFFFFF


def crc32(path: Path) -> int:
    value = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value = binascii.crc32(block, value)
    return value & 0xFFFFFFFF


def align(value: int, alignment: int = BUNDLE_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def json_scalar(value: np.ndarray) -> Any:
    result = value.item()
    if isinstance(result, bytes):
        return result.decode("utf-8")
    return result


def load_candidate(session_dir: Path) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint_path = session_dir / "deployment_candidate.pt"
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("status") != EXPECTED_STATUS:
        raise ValueError(
            f"{checkpoint_path}: expected status {EXPECTED_STATUS!r}, "
            f"got {checkpoint.get('status')!r}"
        )
    if checkpoint.get("model_id") != session_dir.name:
        raise ValueError(f"{checkpoint_path}: model/session ID mismatch")
    config = checkpoint.get("model_config", {})
    expected_config = {
        "physical_channels": 96,
        "input_features": 192,
        "window_bins": 50,
        "tcn_width": 64,
        "gru_width": 64,
        "parameter_count": 86_978,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise ValueError(
                f"{checkpoint_path}: unexpected model_config[{key!r}]="
                f"{config.get(key)!r}"
            )
    if np.asarray(checkpoint.get("feature_std_floor")).shape != (192,):
        raise ValueError(f"{checkpoint_path}: missing feature_std_floor[192]")

    model = MidsizeTCNGRU()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    if sum(parameter.numel() for parameter in model.parameters()) != 86_978:
        raise ValueError(f"{checkpoint_path}: parameter count mismatch")
    return model, checkpoint


def validate_constants(session_dir: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    constants_path = session_dir / "deployment_constants.npz"
    with np.load(constants_path, allow_pickle=False) as source:
        constants = {key: source[key] for key in source.files}
    required_shapes = {
        "selected_channel_indices": (96,),
        "feature_std_floor": (192,),
        "target_mean": (2,),
        "target_std": (2,),
    }
    for key, shape in required_shapes.items():
        if key not in constants or constants[key].shape != shape:
            raise ValueError(f"{constants_path}: {key} must have shape {shape}")
    if json_scalar(constants["model_id"]) != checkpoint["model_id"]:
        raise ValueError(f"{constants_path}: model ID mismatch")
    comparisons = (
        ("feature_std_floor", np.float32),
        ("target_mean", np.float32),
        ("target_std", np.float32),
        ("selected_channel_indices", np.int64),
    )
    for key, dtype in comparisons:
        expected = np.asarray(checkpoint[
            "selected_channel_indices" if key == "selected_channel_indices" else key
        ], dtype=dtype)
        actual = np.asarray(constants[key], dtype=dtype)
        if not np.array_equal(actual, expected):
            raise ValueError(f"{constants_path}: {key} differs from checkpoint")
    return constants


def load_golden(
    session_dir: Path, checkpoint: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, Path]:
    golden_path = session_dir / "deployment_golden_vectors.npz"
    with np.load(golden_path, allow_pickle=False) as source:
        if json_scalar(source["model_id"]) != checkpoint["model_id"]:
            raise ValueError(f"{golden_path}: model ID mismatch")
        inputs = source["model_inputs"].astype(np.float32)
        expected = source["expected_normalized_sequence"].astype(np.float32)
    if inputs.shape != (8, 192, 50):
        raise ValueError(f"{golden_path}: unexpected inputs {inputs.shape}")
    if expected.shape != (8, 50, 2):
        raise ValueError(f"{golden_path}: unexpected outputs {expected.shape}")
    return inputs, expected, golden_path


def reorder_gru_gates(values: np.ndarray, axis: int) -> np.ndarray:
    reset, update, new = np.split(values, 3, axis=axis)
    return np.concatenate((update, reset, new), axis=axis)


def sanitize_h5_for_cubeai(path: Path) -> None:
    def sanitize(value: object) -> None:
        if isinstance(value, dict):
            for key in ("input_axes", "output_axes", "quantization_config"):
                value.pop(key, None)
            for child in value.values():
                sanitize(child)
        elif isinstance(value, list):
            for child in value:
                sanitize(child)

    with h5py.File(path, "r+") as handle:
        config = json.loads(handle.attrs["model_config"])
        sanitize(config)
        handle.attrs.modify("model_config", json.dumps(config).encode("utf-8"))


def export_models(
    model: nn.Module, model_dir: Path
) -> tuple[Encoder, tf.keras.Model, Path, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = model_dir / "encoder.onnx"
    gru_head_path = model_dir / "gru_head.h5"

    encoder = Encoder(model).eval()
    torch.onnx.export(
        encoder,
        (torch.zeros((1, 192, 50), dtype=torch.float32),),
        encoder_path,
        input_names=["features"],
        output_names=["encoded_sequence"],
        opset_version=16,
        dynamo=False,
        dynamic_axes=None,
        do_constant_folding=True,
    )
    graph = onnx.load(encoder_path)
    onnx.checker.check_model(graph)

    encoded = tf.keras.Input(
        batch_shape=(1, 50, 64), name="encoded_sequence"
    )
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
            reorder_gru_gates(weight_ih, axis=1),
            reorder_gru_gates(weight_hh, axis=1),
            np.stack(
                (
                    reorder_gru_gates(bias_ih, axis=0),
                    reorder_gru_gates(bias_hh, axis=0),
                )
            ),
        ]
    )
    keras_model.get_layer("velocity_norm").set_weights(
        [model.head.weight.detach().numpy().T, model.head.bias.detach().numpy()]
    )
    keras_model.save(gru_head_path, include_optimizer=False)
    sanitize_h5_for_cubeai(gru_head_path)
    return encoder, keras_model, encoder_path, gru_head_path


def max_errors(expected: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    difference = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    return {
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "timestep_49_max_abs_error": float(difference[:, 49].max()),
    }


def export_and_python_validate(
    model: nn.Module,
    inputs: np.ndarray,
    golden_expected: np.ndarray,
    build_dir: Path,
) -> tuple[dict[str, Any], Path, Path]:
    validation_dir = build_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    encoder, keras_model, encoder_path, gru_head_path = export_models(
        model, build_dir / "model"
    )

    with torch.inference_mode():
        pytorch_expected = model(torch.from_numpy(inputs)).numpy().astype(np.float32)
        encoded_torch = encoder(torch.from_numpy(inputs)).numpy().astype(np.float32)
    golden_parity = max_errors(golden_expected, pytorch_expected)
    if golden_parity["max_abs_error"] > 1.0e-6:
        raise ValueError(f"checkpoint/golden parity failed: {golden_parity}")

    encoder_session = ort.InferenceSession(
        str(encoder_path), providers=["CPUExecutionProvider"]
    )
    encoded_onnx = np.concatenate(
        [
            encoder_session.run(None, {"features": sample[None]})[0]
            for sample in inputs
        ],
        axis=0,
    ).astype(np.float32)
    split_outputs = np.concatenate(
        [
            keras_model(sample[None], training=False).numpy()
            for sample in encoded_torch
        ],
        axis=0,
    ).astype(np.float32)
    encoder_parity = max_errors(encoded_torch, encoded_onnx)
    split_parity = max_errors(pytorch_expected, split_outputs)
    if encoder_parity["max_abs_error"] > 1.0e-5:
        raise ValueError(f"encoder ONNX parity failed: {encoder_parity}")
    if split_parity["max_abs_error"] > 1.0e-5:
        raise ValueError(f"split Keras parity failed: {split_parity}")

    np.savetxt(
        validation_dir / "model_inputs.csv",
        inputs.reshape(len(inputs), -1),
        delimiter=",",
        fmt="%.9g",
    )
    np.savetxt(
        validation_dir / "encoder_expected.csv",
        encoded_torch.reshape(len(inputs), -1),
        delimiter=",",
        fmt="%.9g",
    )
    np.savetxt(
        validation_dir / "model_expected.csv",
        pytorch_expected.reshape(len(inputs), -1),
        delimiter=",",
        fmt="%.9g",
    )
    np.savez_compressed(
        validation_dir / "split_parity.npz",
        model_inputs=inputs,
        expected_normalized_sequence=pytorch_expected,
        encoder_pytorch=encoded_torch,
        encoder_onnx=encoded_onnx,
        split_keras_outputs=split_outputs,
    )
    return (
        {
            "checkpoint_vs_replay_golden": golden_parity,
            "encoder_onnx_vs_pytorch": encoder_parity,
            "split_keras_vs_pytorch": split_parity,
        },
        encoder_path,
        gru_head_path,
    )


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}); see {log_path}: "
            + subprocess.list2cmdline(command)
        )


def cubeai_base(
    cli: Path, operation: str, model: Path, model_type: str, name: str,
    workspace: Path, output: Path,
) -> list[str]:
    return [
        str(cli), operation,
        "--target", "stm32",
        "--model", str(model),
        "--type", model_type,
        "--name", name,
        "--optimization", "time",
        "--c-api", "legacy",
        "--workspace", str(workspace),
        "--output", str(output),
        "--verbosity", "1",
    ]


def find_single(root: Path, pattern: str) -> Path:
    matches = [path for path in root.rglob(pattern) if path.is_file()]
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern!r} below {root}, got {matches}")
    return matches[0]


def load_flat_csv(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", dtype=np.float32)
    return values.reshape(shape)


def network_metrics(c_info_path: Path) -> dict[str, int]:
    info = json.loads(c_info_path.read_text(encoding="utf-8"))
    memory = info["memory_footprint"]
    macc = sum(
        int(node.get("macc", 0))
        for graph in info.get("graphs", [])
        for node in graph.get("nodes", [])
    )
    return {
        "weights_bytes": int(memory["weights"]),
        "activations_bytes": int(memory["activations"]),
        "macc": macc,
    }


def cubeai_convert_and_validate(
    cli: Path,
    build_dir: Path,
    encoder_model: Path,
    gru_head_model: Path,
    expected_outputs: np.ndarray,
) -> tuple[dict[str, Any], Path, Path]:
    validation_dir = build_dir / "validation"
    cube_dir = build_dir / "cubeai"
    workspace = build_dir / "workspace"
    logs = build_dir / "logs"

    encoder_validate = cube_dir / "encoder_validate"
    encoder_generate = cube_dir / "encoder_generate"
    gru_validate = cube_dir / "gru_validate"
    gru_generate = cube_dir / "gru_generate"
    chain_validate = cube_dir / "chain_validate"

    run(
        cubeai_base(
            cli, "validate", encoder_model, "onnx", "indy_encoder",
            workspace / "encoder_validate", encoder_validate,
        ) + [
            "--mode", "host",
            "--valinput", str(validation_dir / "model_inputs.csv"),
            "--valoutput", str(validation_dir / "encoder_expected.csv"),
            "--save-csv",
        ],
        logs / "validate_encoder.log",
    )
    run(
        cubeai_base(
            cli, "generate", encoder_model, "onnx", "indy_encoder",
            workspace / "encoder_generate", encoder_generate,
        ) + ["--binary"],
        logs / "generate_encoder.log",
    )
    run(
        cubeai_base(
            cli, "validate", gru_head_model, "keras", "indy_gru_head",
            workspace / "gru_validate", gru_validate,
        ) + [
            "--mode", "host",
            "--valinput", str(validation_dir / "encoder_expected.csv"),
            "--valoutput", str(validation_dir / "model_expected.csv"),
            "--save-csv",
        ],
        logs / "validate_gru_head.log",
    )
    run(
        cubeai_base(
            cli, "generate", gru_head_model, "keras", "indy_gru_head",
            workspace / "gru_generate", gru_generate,
        ) + ["--binary"],
        logs / "generate_gru_head.log",
    )

    encoder_c_outputs_path = find_single(
        encoder_validate, "*_val_c_outputs_1.csv"
    )
    run(
        cubeai_base(
            cli, "validate", gru_head_model, "keras", "indy_gru_head",
            workspace / "chain_validate", chain_validate,
        ) + [
            "--mode", "host",
            "--valinput", str(encoder_c_outputs_path),
            "--valoutput", str(validation_dir / "model_expected.csv"),
            "--save-csv",
        ],
        logs / "validate_chain.log",
    )

    encoder_c = load_flat_csv(
        encoder_c_outputs_path, (len(expected_outputs), 50, 64)
    )
    encoder_reference = load_flat_csv(
        validation_dir / "encoder_expected.csv", (len(expected_outputs), 50, 64)
    )
    gru_c_path = find_single(gru_validate, "*_val_c_outputs_1.csv")
    chain_c_path = find_single(chain_validate, "*_val_c_outputs_1.csv")
    gru_c = load_flat_csv(gru_c_path, expected_outputs.shape)
    chain_c = load_flat_csv(chain_c_path, expected_outputs.shape)
    parity = {
        "encoder_host_c_vs_pytorch": max_errors(encoder_reference, encoder_c),
        "gru_head_host_c_vs_pytorch": max_errors(expected_outputs, gru_c),
        "generated_c_chain_vs_pytorch": max_errors(expected_outputs, chain_c),
    }
    if parity["encoder_host_c_vs_pytorch"]["max_abs_error"] > 1.0e-5:
        raise ValueError(f"Cube.AI encoder parity failed: {parity}")
    if parity["gru_head_host_c_vs_pytorch"]["max_abs_error"] > 1.0e-5:
        raise ValueError(f"Cube.AI GRU/head parity failed: {parity}")
    if parity["generated_c_chain_vs_pytorch"]["max_abs_error"] > 1.0e-5:
        raise ValueError(f"Cube.AI chain parity failed: {parity}")

    encoder_info = find_single(encoder_generate, "*_c_info.json")
    gru_info = find_single(gru_generate, "*_c_info.json")
    encoder_metrics = network_metrics(encoder_info)
    gru_metrics = network_metrics(gru_info)
    encoder_binary = find_single(encoder_generate, "*.bin")
    gru_binary = find_single(gru_generate, "*.bin")
    if encoder_binary.stat().st_size != encoder_metrics["weights_bytes"]:
        raise ValueError("encoder binary size differs from Cube.AI c_info")
    if gru_binary.stat().st_size != gru_metrics["weights_bytes"]:
        raise ValueError("GRU/head binary size differs from Cube.AI c_info")
    return (
        {
            "parity": parity,
            "encoder": encoder_metrics,
            "gru_head": gru_metrics,
        },
        encoder_binary,
        gru_binary,
    )


def encode_fixed_string(value: str, size: int) -> bytes:
    encoded = value.encode("utf-8")
    if len(encoded) >= size:
        raise ValueError(f"string too long for {size}-byte field: {value!r}")
    return encoded + bytes(size - len(encoded))


def build_params(constants: dict[str, Any]) -> bytes:
    floor = np.asarray(constants["feature_std_floor"], dtype="<f4")
    target_mean = np.asarray(constants["target_mean"], dtype="<f4")
    target_std = np.asarray(constants["target_std"], dtype="<f4")
    channels = np.asarray(constants["selected_channel_indices"], dtype="<u2")
    values = floor.tobytes() + target_mean.tobytes() + target_std.tobytes()
    values += channels.tobytes()
    if len(values) != 976:
        raise ValueError(f"unexpected parameter block size: {len(values)}")
    return values


def build_bundle(
    destination: Path,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    constants: dict[str, Any],
    encoder_binary: Path,
    gru_binary: Path,
) -> dict[str, Any]:
    encoder = encoder_binary.read_bytes()
    gru = gru_binary.read_bytes()
    params = build_params(constants)
    encoder_offset = BUNDLE_HEADER_SIZE
    gru_offset = align(encoder_offset + len(encoder))
    params_offset = align(gru_offset + len(gru))
    total_size = params_offset + len(params)

    checkpoint_digest = bytes.fromhex(sha256(checkpoint_path))
    body = bytearray(total_size - BUNDLE_HEADER_SIZE)
    body[encoder_offset - BUNDLE_HEADER_SIZE : encoder_offset - BUNDLE_HEADER_SIZE + len(encoder)] = encoder
    body[gru_offset - BUNDLE_HEADER_SIZE : gru_offset - BUNDLE_HEADER_SIZE + len(gru)] = gru
    body[params_offset - BUNDLE_HEADER_SIZE : params_offset - BUNDLE_HEADER_SIZE + len(params)] = params
    body_digest = hashlib.sha256(body).digest()

    header = bytearray(BUNDLE_HEADER_SIZE)
    struct.pack_into(
        "<8sHHII32sHHHHHH32sIIIIIIIII32s32sI32s",
        header,
        0,
        BUNDLE_MAGIC,
        1,
        BUNDLE_HEADER_SIZE,
        1,
        total_size,
        encode_fixed_string(checkpoint["model_id"], 32),
        int(checkpoint["source_channel_count"]),
        96,
        192,
        50,
        49,
        BUNDLE_ALIGNMENT,
        checkpoint_digest,
        encoder_offset,
        len(encoder),
        crc32_bytes(encoder),
        gru_offset,
        len(gru),
        crc32_bytes(gru),
        params_offset,
        len(params),
        crc32_bytes(params),
        body_digest,
        encode_fixed_string(MODEL_BUNDLE_VERSION, 32),
        0,
        encode_fixed_string(GRAPH_ABI_ID, 32),
    )
    header_crc = crc32_bytes(bytes(header))
    struct.pack_into("<I", header, BUNDLE_HEADER_CRC_OFFSET, header_crc)
    destination.write_bytes(bytes(header) + bytes(body))
    return {
        "format": MODEL_BUNDLE_VERSION,
        "format_version": 1,
        "graph_abi_id": GRAPH_ABI_ID,
        "magic_hex": BUNDLE_MAGIC.hex(),
        "header_bytes": BUNDLE_HEADER_SIZE,
        "alignment_bytes": BUNDLE_ALIGNMENT,
        "total_bytes": destination.stat().st_size,
        "crc32": f"{crc32(destination):08x}",
        "sha256": sha256(destination),
        "header_crc32": f"{header_crc:08x}",
        "body_sha256": body_digest.hex(),
        "encoder_offset": encoder_offset,
        "gru_head_offset": gru_offset,
        "parameters_offset": params_offset,
        "parameters_layout": {
            "endianness": "little",
            "feature_std_floor": {"dtype": "float32", "count": 192, "offset": 0},
            "target_mean": {"dtype": "float32", "count": 2, "offset": 768},
            "target_std": {"dtype": "float32", "count": 2, "offset": 776},
            "selected_channel_indices": {"dtype": "uint16", "count": 96, "offset": 784},
            "total_bytes": len(params),
        },
    }


def artifact_info(path: Path) -> dict[str, Any]:
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "crc32": f"{crc32(path):08x}",
        "sha256": sha256(path),
    }


def repack_session(session: str) -> dict[str, Any]:
    session_dir = MIDSIZE_DIR / session
    output_dir = session_dir / "cubeai"
    checkpoint_path = session_dir / "deployment_candidate.pt"
    _, checkpoint = load_candidate(session_dir)
    constants = validate_constants(session_dir, checkpoint)
    manifest_path = output_dir / "manifest.json"
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    report["bundle"] = build_bundle(
        output_dir / f"{session}.aibundle",
        checkpoint_path,
        checkpoint,
        constants,
        output_dir / "encoder.weights.bin",
        output_dir / "gru_head.weights.bin",
    )
    manifest_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_session(session: str, cli: Path, keep_build: bool) -> dict[str, Any]:
    session_dir = MIDSIZE_DIR / session
    checkpoint_path = session_dir / "deployment_candidate.pt"
    output_dir = session_dir / "cubeai"
    build_dir = output_dir / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)

    model, checkpoint = load_candidate(session_dir)
    constants = validate_constants(session_dir, checkpoint)
    inputs, expected, golden_path = load_golden(session_dir, checkpoint)
    python_report, encoder_model, gru_head_model = export_and_python_validate(
        model, inputs, expected, build_dir
    )
    cubeai_report, encoder_source, gru_source = cubeai_convert_and_validate(
        cli, build_dir, encoder_model, gru_head_model, expected
    )

    encoder_destination = output_dir / "encoder.weights.bin"
    gru_destination = output_dir / "gru_head.weights.bin"
    output_dir.mkdir(exist_ok=True)
    shutil.copyfile(encoder_source, encoder_destination)
    shutil.copyfile(gru_source, gru_destination)
    bundle_path = output_dir / f"{session}.aibundle"
    bundle_info = build_bundle(
        bundle_path,
        checkpoint_path,
        checkpoint,
        constants,
        encoder_destination,
        gru_destination,
    )

    version_result = subprocess.run(
        [str(cli), "--version"], text=True, capture_output=True, check=True
    )
    version = [line.strip() for line in version_result.stdout.splitlines() if line.strip()]
    report = {
        "schema_version": 1,
        "model_id": checkpoint["model_id"],
        "session": checkpoint["session"],
        "subject": checkpoint["subject"],
        "status": checkpoint["status"],
        "promotion_claimed": False,
        "architecture": checkpoint["model_config"],
        "source_channel_count": int(checkpoint["source_channel_count"]),
        "selected_channel_indices": np.asarray(
            constants["selected_channel_indices"], dtype=np.int64
        ).tolist(),
        "checkpoint": {
            "file": checkpoint_path.name,
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256(checkpoint_path),
        },
        "deployment_constants": artifact_info(
            session_dir / "deployment_constants.npz"
        ),
        "deployment_golden_vectors": artifact_info(golden_path),
        "models": {
            "encoder_onnx_sha256": sha256(encoder_model),
            "gru_head_h5_sha256": sha256(gru_head_model),
        },
        "components": {
            "encoder": artifact_info(encoder_destination)
            | cubeai_report["encoder"],
            "gru_head": artifact_info(gru_destination)
            | cubeai_report["gru_head"],
        },
        "combined": {
            "weights_bytes": (
                encoder_destination.stat().st_size + gru_destination.stat().st_size
            ),
            "conservative_activations_bytes": (
                cubeai_report["encoder"]["activations_bytes"]
                + cubeai_report["gru_head"]["activations_bytes"]
            ),
            "macc": (
                cubeai_report["encoder"]["macc"]
                + cubeai_report["gru_head"]["macc"]
            ),
        },
        "parity": python_report | cubeai_report["parity"],
        "bundle": bundle_info,
        "cubeai": {
            "version_output": version,
            "target": "stm32",
            "optimization": "time",
            "c_api": "legacy",
            "weight_binary_source": "official stedgeai generate --binary",
        },
        "python_environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "tensorflow": tf.__version__,
            "keras": tf.keras.__version__,
            "numpy": np.__version__,
            "h5py": h5py.__version__,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not keep_build:
        shutil.rmtree(build_dir)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_cli = (
        Path.home()
        / "STM32Cube"
        / "Repository"
        / "Packs"
        / "STMicroelectronics"
        / "X-CUBE-AI"
        / "10.2.0"
        / "Utilities"
        / ("windows" if os.name == "nt" else "linux")
        / ("stedgeai.exe" if os.name == "nt" else "stedgeai")
    )
    parser.add_argument("--stedgeai", type=Path, default=default_cli)
    parser.add_argument("--session", choices=SESSIONS, action="append")
    parser.add_argument("--keep-build", action="store_true")
    parser.add_argument(
        "--repack-only",
        action="store_true",
        help="rebuild headers/manifests from already validated Cube.AI binaries",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cli = args.stedgeai.resolve()
    if not cli.is_file():
        raise FileNotFoundError(f"Cube.AI CLI not found: {cli}")
    sessions = tuple(args.session or SESSIONS)
    reports = []
    for session in sessions:
        if args.repack_only:
            print(f"[{session}] repack validated Cube.AI binaries")
            report = repack_session(session)
        else:
            print(f"[{session}] export, Cube.AI conversion and parity validation")
            report = build_session(session, cli, args.keep_build)
        reports.append(report)
        chain = report["parity"]["generated_c_chain_vs_pytorch"]
        print(
            f"[{session}] {report['combined']['weights_bytes']} weight bytes, "
            f"chain max abs error {chain['max_abs_error']:.3e}"
        )
    completed_reports = []
    for session in SESSIONS:
        manifest_path = MIDSIZE_DIR / session / "cubeai" / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"all six manifests are required before writing the index: {manifest_path}"
            )
        completed_reports.append(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
    summary = {
        "schema_version": 1,
        "bundle_format": MODEL_BUNDLE_VERSION,
        "status": EXPECTED_STATUS,
        "promotion_claimed": False,
        "sessions": [
            {
                "model_id": report["model_id"],
                "manifest": f"{report['model_id']}/cubeai/manifest.json",
                "bundle": f"{report['model_id']}/cubeai/{report['model_id']}.aibundle",
                "bundle_bytes": report["bundle"]["total_bytes"],
                "bundle_crc32": report["bundle"]["crc32"],
                "bundle_sha256": report["bundle"]["sha256"],
            }
            for report in completed_reports
        ],
    }
    (MIDSIZE_DIR / "cubeai_bundles.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
