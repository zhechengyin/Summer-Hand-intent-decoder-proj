#!/usr/bin/env python3
"""Export a numerically exact two-network Cube.AI deployment bundle."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

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
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from indy_loco.models.midsize.model import load_checkpoint  # noqa: E402


CHECKPOINT_PATH = INDY_ROOT / "models" / "midsize" / "checkpoint.pt"
ENCODER_PATH = DEPLOY_DIR / "model" / "indy_phase6_encoder.onnx"
GRU_HEAD_PATH = DEPLOY_DIR / "model" / "indy_phase6_gru_head.h5"
VALIDATION_DIR = DEPLOY_DIR / "validation"


class Encoder(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.spatial = model.spatial
        self.convolutions = model.convolutions
        self.padding = model.padding
        self.activation = model.activation

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.spatial(values)
        for convolution, padding in zip(self.convolutions, self.padding, strict=True):
            encoded = self.activation(convolution(encoded)[:, :, :-padding] + encoded)
        return encoded.transpose(1, 2)


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


def export_encoder(model: nn.Module) -> Encoder:
    encoder = Encoder(model).eval()
    ENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        encoder,
        (torch.zeros((1, 192, 50), dtype=torch.float32),),
        ENCODER_PATH,
        input_names=["features"],
        output_names=["encoded_sequence"],
        opset_version=16,
        dynamo=False,
        dynamic_axes=None,
        do_constant_folding=True,
    )
    graph = onnx.load(ENCODER_PATH)
    onnx.checker.check_model(graph)
    return encoder


def export_gru_head(model: nn.Module) -> tf.keras.Model:
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
    keras_model = tf.keras.Model(encoded, output, name="indy_phase6_gru_head")

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
    keras_model.save(GRU_HEAD_PATH, include_optimizer=False)
    sanitize_h5_for_cubeai(GRU_HEAD_PATH)
    return keras_model


def main() -> None:
    model, _ = load_checkpoint(CHECKPOINT_PATH)
    model.eval()
    inputs = np.load(VALIDATION_DIR / "validation_inputs.npy").astype(np.float32)
    expected = np.load(VALIDATION_DIR / "pytorch_outputs.npy").astype(np.float32)

    encoder = export_encoder(model)
    keras_model = export_gru_head(model)
    with torch.inference_mode():
        encoded_torch = np.concatenate(
            [encoder(torch.from_numpy(sample[None])).numpy() for sample in inputs],
            axis=0,
        ).astype(np.float32)

    encoder_session = ort.InferenceSession(
        str(ENCODER_PATH), providers=["CPUExecutionProvider"]
    )
    encoded_onnx = np.concatenate(
        [
            encoder_session.run(None, {"features": sample[None]})[0]
            for sample in inputs
        ],
        axis=0,
    ).astype(np.float32)
    actual = np.concatenate(
        [keras_model(sample[None], training=False).numpy() for sample in encoded_torch],
        axis=0,
    ).astype(np.float32)

    encoder_difference = np.abs(encoded_torch - encoded_onnx)
    final_difference = np.abs(expected - actual)
    report = {
        "encoder_onnx_max_abs_error": float(encoder_difference.max()),
        "encoder_onnx_mean_abs_error": float(encoder_difference.mean()),
        "gru_head_keras_end_to_end_max_abs_error": float(final_difference.max()),
        "gru_head_keras_end_to_end_mean_abs_error": float(final_difference.mean()),
        "timestep_49_max_abs_error": float(final_difference[:, 49].max()),
        "encoder_onnx_operators": dict(
            sorted(Counter(node.op_type for node in onnx.load(ENCODER_PATH).graph.node).items())
        ),
    }
    if max(report[key] for key in (
        "encoder_onnx_max_abs_error",
        "gru_head_keras_end_to_end_max_abs_error",
    )) > 1.0e-5:
        raise ValueError(f"Split-model parity failed: {report}")

    np.save(VALIDATION_DIR / "encoder_pytorch_outputs.npy", encoded_torch)
    np.save(VALIDATION_DIR / "encoder_onnx_outputs.npy", encoded_onnx)
    np.savetxt(
        VALIDATION_DIR / "encoder_pytorch_outputs.csv",
        encoded_torch.reshape(len(encoded_torch), -1),
        delimiter=",",
        fmt="%.9g",
    )
    np.save(VALIDATION_DIR / "split_keras_outputs.npy", actual)
    np.savetxt(
        VALIDATION_DIR / "split_keras_outputs.csv",
        actual.reshape(len(actual), -1),
        delimiter=",",
        fmt="%.9g",
    )
    (VALIDATION_DIR / "split_model_parity.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Encoder: {ENCODER_PATH}")
    print(f"GRU head: {GRU_HEAD_PATH}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
