#!/usr/bin/env python3
"""Export the promoted Indy Phase 6 decoder and prepare validation vectors.

All outputs stay below ``indy_loco/deploy``.  The exported network has fixed
input/output shapes required by the Phase 9 firmware policy:

    input:  (1, 192, 50), float32
    output: (1, 50, 2), float32
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F


DEPLOY_DIR = Path(__file__).resolve().parent
INDY_ROOT = DEPLOY_DIR.parent
REPOSITORY_ROOT = INDY_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from indy_loco.models.midsize.model import load_checkpoint  # noqa: E402


CHECKPOINT_PATH = INDY_ROOT / "models" / "midsize" / "checkpoint.pt"
SOURCE_GOLDEN_PATH = (
    INDY_ROOT
    / "results"
    / "phase9_deployment_policy_replay"
    / "phase9_deployment_policy_replay_golden_vectors.npz"
)
ONNX_PATH = DEPLOY_DIR / "model" / "indy_phase6_fp32.onnx"
FUSED_REFERENCE_ONNX_PATH = (
    DEPLOY_DIR / "model" / "indy_phase6_reference_fused_gru.onnx"
)
VALIDATION_DIR = DEPLOY_DIR / "validation"
METADATA_DIR = DEPLOY_DIR / "metadata"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [json_value(item) for item in value]
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    return value


class CubeAIExportModel(nn.Module):
    """Inference-equivalent model with the fixed 50-step GRU unrolled.

    X-CUBE-AI 10.2 accepts the PyTorch-exported ONNX GRU node but its generated
    host C model does not reproduce PyTorch's ``linear_before_reset=1`` GRU
    outputs.  Expressing the exact PyTorch equations with primitive operators
    avoids that backend discrepancy while preserving the checkpoint weights.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.spatial = model.spatial
        self.convolutions = model.convolutions
        self.padding = model.padding
        self.activation = model.activation
        self.head = model.head
        self.weight_ih = nn.Parameter(
            model.gru.weight_ih_l0.detach().clone(), requires_grad=False
        )
        self.weight_hh = nn.Parameter(
            model.gru.weight_hh_l0.detach().clone(), requires_grad=False
        )
        self.bias_ih = nn.Parameter(
            model.gru.bias_ih_l0.detach().clone(), requires_grad=False
        )
        self.bias_hh = nn.Parameter(
            model.gru.bias_hh_l0.detach().clone(), requires_grad=False
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.spatial(values)
        for convolution, padding in zip(self.convolutions, self.padding, strict=True):
            encoded = self.activation(convolution(encoded)[:, :, :-padding] + encoded)
        sequence = encoded.transpose(1, 2)
        hidden = torch.zeros_like(sequence[:, 0, :])
        states = []
        width = 64
        for timestep in range(50):
            input_gates = F.linear(
                sequence[:, timestep, :], self.weight_ih, self.bias_ih
            )
            hidden_gates = F.linear(hidden, self.weight_hh, self.bias_hh)
            reset = torch.sigmoid(
                input_gates[:, :width] + hidden_gates[:, :width]
            )
            update = torch.sigmoid(
                input_gates[:, width : 2 * width]
                + hidden_gates[:, width : 2 * width]
            )
            candidate = torch.tanh(
                input_gates[:, 2 * width :] + reset * hidden_gates[:, 2 * width :]
            )
            hidden = (1.0 - update) * candidate + update * hidden
            states.append(hidden)
        return self.head(torch.stack(states, dim=1))


def torch_export(model: torch.nn.Module, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros((1, 192, 50), dtype=torch.float32)
    torch.onnx.export(
        model,
        (sample,),
        destination,
        input_names=["features"],
        output_names=["velocity_norm"],
        opset_version=16,
        dynamo=False,
        dynamic_axes=None,
        do_constant_folding=True,
    )


def export_onnx(model: torch.nn.Module) -> CubeAIExportModel:
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch_export(model, FUSED_REFERENCE_ONNX_PATH)
    export_model = CubeAIExportModel(model).eval()
    torch_export(export_model, ONNX_PATH)
    return export_model


def verify_onnx_structure() -> dict[str, Any]:
    graph = onnx.load(ONNX_PATH)
    onnx.checker.check_model(graph)
    inferred = onnx.shape_inference.infer_shapes(graph)

    input_shape = [dim.dim_value for dim in inferred.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [
        dim.dim_value for dim in inferred.graph.output[0].type.tensor_type.shape.dim
    ]
    if input_shape != [1, 192, 50]:
        raise ValueError(f"Unexpected ONNX input shape: {input_shape}")
    if output_shape != [1, 50, 2]:
        raise ValueError(f"Unexpected ONNX output shape: {output_shape}")

    operators = Counter(node.op_type for node in inferred.graph.node)
    return {
        "opset": int(inferred.opset_import[0].version),
        "input_name": inferred.graph.input[0].name,
        "input_shape": input_shape,
        "output_name": inferred.graph.output[0].name,
        "output_shape": output_shape,
        "operators": dict(sorted(operators.items())),
    }


def load_validation_inputs() -> tuple[np.ndarray, list[str]]:
    if not SOURCE_GOLDEN_PATH.exists():
        raise FileNotFoundError(f"Missing Phase 9 golden vectors: {SOURCE_GOLDEN_PATH}")

    with np.load(SOURCE_GOLDEN_PATH, allow_pickle=False) as golden:
        initial = golden["strategy_b_initial_normalized"].astype(np.float32)
        rolling = golden["strategy_b_first_five_normalized_windows"].astype(
            np.float32
        )

    rng = np.random.default_rng(20260819)
    random_input = rng.normal(0.0, 1.0, size=(1, 192, 50)).astype(np.float32)
    zeros = np.zeros((1, 192, 50), dtype=np.float32)
    inputs = np.concatenate((initial[None], rolling, random_input, zeros), axis=0)
    names = [
        "phase9_calibration_seed",
        "phase9_post_bin_1500",
        "phase9_post_bin_1501",
        "phase9_post_bin_1502",
        "phase9_post_bin_1503",
        "phase9_post_bin_1504",
        "deterministic_random",
        "zeros",
    ]
    if inputs.shape != (len(names), 192, 50):
        raise ValueError(f"Unexpected validation input shape: {inputs.shape}")
    return inputs, names


def predict_pytorch(model: torch.nn.Module, inputs: np.ndarray) -> np.ndarray:
    outputs = []
    with torch.inference_mode():
        for sample in inputs:
            tensor = torch.from_numpy(sample[None])
            outputs.append(model(tensor).cpu().numpy()[0])
    return np.stack(outputs).astype(np.float32)


def predict_onnx(inputs: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(
        str(ONNX_PATH), providers=["CPUExecutionProvider"]
    )
    outputs = []
    for sample in inputs:
        result = session.run(["velocity_norm"], {"features": sample[None]})[0]
        outputs.append(result[0])
    return np.stack(outputs).astype(np.float32)


def write_validation_artifacts(
    inputs: np.ndarray,
    names: list[str],
    pytorch_outputs: np.ndarray,
    onnx_outputs: np.ndarray,
) -> dict[str, Any]:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    np.save(VALIDATION_DIR / "validation_inputs.npy", inputs)
    np.save(VALIDATION_DIR / "pytorch_outputs.npy", pytorch_outputs)
    np.save(VALIDATION_DIR / "onnx_outputs.npy", onnx_outputs)

    np.savetxt(
        VALIDATION_DIR / "validation_inputs.csv",
        inputs.reshape(len(inputs), -1),
        delimiter=",",
        fmt="%.9g",
    )
    np.savetxt(
        VALIDATION_DIR / "pytorch_outputs.csv",
        pytorch_outputs.reshape(len(inputs), -1),
        delimiter=",",
        fmt="%.9g",
    )
    np.savetxt(
        VALIDATION_DIR / "onnx_outputs.csv",
        onnx_outputs.reshape(len(inputs), -1),
        delimiter=",",
        fmt="%.9g",
    )
    (VALIDATION_DIR / "validation_samples.json").write_text(
        json.dumps({"samples": names}, indent=2) + "\n", encoding="utf-8"
    )

    difference = np.abs(pytorch_outputs - onnx_outputs)
    per_sample = []
    for index, name in enumerate(names):
        per_sample.append(
            {
                "name": name,
                "max_abs_error": float(difference[index].max()),
                "mean_abs_error": float(difference[index].mean()),
            }
        )
    report = {
        "samples": len(names),
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "per_sample": per_sample,
    }
    (VALIDATION_DIR / "pytorch_onnx_parity.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if report["max_abs_error"] > 1e-5:
        raise ValueError(f"PyTorch/ONNX parity failed: {report}")
    return report


def write_metadata(checkpoint: dict[str, Any], onnx_info: dict[str, Any]) -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    feature_std_floor = np.asarray(checkpoint["feature_std_floor"], dtype=np.float32)
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
    target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)
    channels = np.asarray(checkpoint["channels"], dtype=np.int64)

    np.savez_compressed(
        METADATA_DIR / "indy_phase6_constants.npz",
        feature_std_floor=feature_std_floor,
        target_mean=target_mean,
        target_std=target_std,
        channels=channels,
    )
    metadata = {
        "checkpoint": {
            "source": str(CHECKPOINT_PATH),
            "sha256": sha256(CHECKPOINT_PATH),
            "parameter_count": int(checkpoint["parameter_count"]),
            "physical_channel_count": int(checkpoint["physical_channel_count"]),
            "input_feature_count": int(checkpoint["input_feature_count"]),
            "channel_selection": checkpoint["channel_selection"],
            "channels": channels.tolist(),
            "feature_std_floor": feature_std_floor.tolist(),
            "target_mean": target_mean.tolist(),
            "target_std": target_std.tolist(),
            "experiment_config": json_value(checkpoint.get("experiment_config", {})),
        },
        "deployment_policy": {
            "name": "phase9_B_rolling_calibration_seed",
            "bin_ms": 40,
            "calibration_bins": 1500,
            "window_bins": 50,
            "ewma_alpha": 0.1,
            "feature_order": "raw_count_0..95_then_ewma_0..95",
            "inference": "stateless_full_window_read_timestep_49",
        },
        "onnx": {
            **onnx_info,
            "path": str(ONNX_PATH),
            "sha256": sha256(ONNX_PATH),
            "export_strategy": "fixed_50_step_unrolled_pytorch_gru_equations",
            "fused_reference_path": str(FUSED_REFERENCE_ONNX_PATH),
            "fused_reference_sha256": sha256(FUSED_REFERENCE_ONNX_PATH),
            "reason": (
                "X-CUBE-AI 10.2 host C validation did not preserve the outputs "
                "of the fused ONNX linear_before_reset=1 GRU"
            ),
        },
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
        },
    }
    (METADATA_DIR / "indy_phase6_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    model, checkpoint = load_checkpoint(CHECKPOINT_PATH)
    model.eval()
    export_model = export_onnx(model)
    onnx_info = verify_onnx_structure()
    if "GRU" in onnx_info["operators"]:
        raise ValueError("Cube.AI deployment ONNX must not contain a fused GRU node")
    inputs, names = load_validation_inputs()
    pytorch_outputs = predict_pytorch(model, inputs)
    export_outputs = predict_pytorch(export_model, inputs)
    export_difference = np.abs(pytorch_outputs - export_outputs)
    if float(export_difference.max()) > 1e-5:
        raise ValueError(
            "Manual GRU equations do not reproduce the checkpoint: "
            f"max_abs_error={float(export_difference.max())}"
        )
    onnx_outputs = predict_onnx(inputs)
    parity = write_validation_artifacts(
        inputs, names, pytorch_outputs, onnx_outputs
    )
    write_metadata(checkpoint, onnx_info)
    print(f"ONNX: {ONNX_PATH}")
    print(f"Operators: {onnx_info['operators']}")
    print(f"PyTorch/manual-GRU max abs error: {float(export_difference.max()):.9g}")
    print(f"PyTorch/ONNX max abs error: {parity['max_abs_error']:.9g}")


if __name__ == "__main__":
    main()
