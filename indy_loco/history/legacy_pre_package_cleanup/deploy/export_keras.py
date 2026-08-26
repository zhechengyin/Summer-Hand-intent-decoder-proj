#!/usr/bin/env python3
"""Export the promoted PyTorch checkpoint as an equivalent Keras model.

The Keras GRU conversion path is used because X-CUBE-AI 10.2 accepts the
PyTorch-exported ONNX GRU but does not preserve its linear-before-reset
semantics in generated C code.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import h5py
import tensorflow as tf
import torch


DEPLOY_DIR = Path(__file__).resolve().parent
INDY_ROOT = DEPLOY_DIR.parent
REPOSITORY_ROOT = INDY_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from indy_loco.models.midsize.model import load_checkpoint  # noqa: E402


CHECKPOINT_PATH = INDY_ROOT / "models" / "midsize" / "checkpoint.pt"
KERAS_PATH = DEPLOY_DIR / "model" / "indy_phase6_fp32.h5"
VALIDATION_DIR = DEPLOY_DIR / "validation"


def reorder_gru_gates(values: np.ndarray, axis: int) -> np.ndarray:
    """Convert PyTorch [reset, update, new] to Keras [update, reset, new]."""
    reset, update, new = np.split(values, 3, axis=axis)
    return np.concatenate((update, reset, new), axis=axis)


def build_keras_model() -> tf.keras.Model:
    features = tf.keras.Input(batch_shape=(1, 192, 50), name="features")
    values = tf.keras.layers.Permute((2, 1), name="to_time_major_features")(features)
    values = tf.keras.layers.Conv1D(64, 1, name="spatial_conv")(values)
    values = tf.keras.layers.LayerNormalization(
        axis=-1, epsilon=1.0e-5, name="spatial_layer_norm"
    )(values)
    values = tf.keras.layers.ReLU(name="spatial_relu")(values)

    for index, dilation in enumerate((1, 2, 4, 8)):
        residual = values
        values = tf.keras.layers.ZeroPadding1D(
            padding=(2 * dilation, 0), name=f"tcn_{index}_causal_pad"
        )(values)
        values = tf.keras.layers.Conv1D(
            64,
            3,
            dilation_rate=dilation,
            padding="valid",
            name=f"tcn_{index}_conv",
        )(values)
        values = tf.keras.layers.Add(name=f"tcn_{index}_residual")(
            (values, residual)
        )
        values = tf.keras.layers.ReLU(name=f"tcn_{index}_relu")(values)

    values = tf.keras.layers.GRU(
        64,
        activation="tanh",
        recurrent_activation="sigmoid",
        reset_after=True,
        return_sequences=True,
        unroll=False,
        name="gru",
    )(values)
    outputs = tf.keras.layers.Dense(2, name="velocity_norm")(values)
    return tf.keras.Model(features, outputs, name="indy_phase6")


def copy_weights(keras_model: tf.keras.Model, torch_model: torch.nn.Module) -> None:
    spatial_conv = torch_model.spatial[0]
    keras_model.get_layer("spatial_conv").set_weights(
        [
            spatial_conv.weight.detach().numpy().transpose(2, 1, 0),
            spatial_conv.bias.detach().numpy(),
        ]
    )
    normalization = torch_model.spatial[1].normalization
    keras_model.get_layer("spatial_layer_norm").set_weights(
        [
            normalization.weight.detach().numpy(),
            normalization.bias.detach().numpy(),
        ]
    )
    for index, convolution in enumerate(torch_model.convolutions):
        keras_model.get_layer(f"tcn_{index}_conv").set_weights(
            [
                convolution.weight.detach().numpy().transpose(2, 1, 0),
                convolution.bias.detach().numpy(),
            ]
        )

    weight_ih = torch_model.gru.weight_ih_l0.detach().numpy().T
    weight_hh = torch_model.gru.weight_hh_l0.detach().numpy().T
    bias_ih = torch_model.gru.bias_ih_l0.detach().numpy()
    bias_hh = torch_model.gru.bias_hh_l0.detach().numpy()
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
        [
            torch_model.head.weight.detach().numpy().T,
            torch_model.head.bias.detach().numpy(),
        ]
    )


def sanitize_h5_for_cubeai(path: Path) -> None:
    """Remove Keras 3.15 initializer fields unknown to Cube.AI's Keras loader."""

    def sanitize(value: object) -> None:
        if isinstance(value, dict):
            value.pop("input_axes", None)
            value.pop("output_axes", None)
            value.pop("quantization_config", None)
            for child in value.values():
                sanitize(child)
        elif isinstance(value, list):
            for child in value:
                sanitize(child)

    with h5py.File(path, "r+") as handle:
        config = json.loads(handle.attrs["model_config"])
        sanitize(config)
        handle.attrs.modify("model_config", json.dumps(config).encode("utf-8"))


def main() -> None:
    torch_model, _ = load_checkpoint(CHECKPOINT_PATH)
    torch_model.eval()
    keras_model = build_keras_model()
    copy_weights(keras_model, torch_model)

    inputs = np.load(VALIDATION_DIR / "validation_inputs.npy").astype(np.float32)
    expected = np.load(VALIDATION_DIR / "pytorch_outputs.npy").astype(np.float32)
    actual = np.concatenate(
        [keras_model(sample[None], training=False).numpy() for sample in inputs], axis=0
    ).astype(np.float32)
    difference = np.abs(expected - actual)
    parity = {
        "samples": int(inputs.shape[0]),
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "timestep_49_max_abs_error": float(difference[:, 49].max()),
        "tensorflow": tf.__version__,
        "keras": tf.keras.__version__,
        "gru_reset_after": True,
        "gru_gate_mapping": "pytorch_rzn_to_keras_zrn",
    }
    if parity["max_abs_error"] > 1.0e-5:
        raise ValueError(f"PyTorch/Keras parity failed: {parity}")

    KERAS_PATH.parent.mkdir(parents=True, exist_ok=True)
    keras_model.save(KERAS_PATH, include_optimizer=False)
    sanitize_h5_for_cubeai(KERAS_PATH)
    np.save(VALIDATION_DIR / "keras_outputs.npy", actual)
    np.savetxt(
        VALIDATION_DIR / "keras_outputs.csv",
        actual.reshape(len(actual), -1),
        delimiter=",",
        fmt="%.9g",
    )
    (VALIDATION_DIR / "pytorch_keras_parity.json").write_text(
        json.dumps(parity, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Keras: {KERAS_PATH}")
    print(json.dumps(parity, indent=2))


if __name__ == "__main__":
    main()
