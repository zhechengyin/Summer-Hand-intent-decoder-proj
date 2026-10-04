"""Export the retained b2 W8A32 plan without folding INT8 weights into FP32.

Independent 50-bin windows; fixed first session/fold, not accuracy-selected.
Existing training, checkpoints and quantization results are read-only inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.nn import functional as F

from indy_loco.models.mingru_b.model import ModelB
from indy_loco.experiment.mingru_mixed_precision import quantization as qz

RESULTS = HERE.parent / "results"
PACKAGE = RESULTS / "b2_completion_v1/package"
QUANT = RESULTS / "b2_mixed_precision_v1"
OLD_EXPORT = REPO / "indy_loco/experiment/phase18_large_mingru/results/b_export_v1"
OUTPUT = RESULTS / "b2_cubeai_export_v1"
SESSION, FOLD = "indy_20160622_01", 1


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class Dequantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, scale, zero):
        return weight.float() * scale[:, None]

    @staticmethod
    def symbolic(g, weight, scale, zero):
        return g.op("DequantizeLinear", weight, scale, zero, axis_i=0)


class ExportLinear(nn.Module):
    def __init__(self, original, packed):
        super().__init__()
        self.register_buffer("qweight", packed["weight"])
        self.register_buffer("scale", packed["scale"])
        self.register_buffer("zero", torch.zeros_like(packed["scale"], dtype=torch.int8))
        self.register_buffer("bias", original.bias.detach().clone())

    def forward(self, values):
        return F.linear(values, Dequantize.apply(self.qweight, self.scale, self.zero), self.bias)


class Window(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, values):
        return self.model.forward_sequential(values)


def error(a, b):
    e = np.abs(a - b)
    return {"passed": bool(np.isfinite(b).all() and np.allclose(a, b, atol=2e-4, rtol=1e-4)),
            "max_abs": float(e.max()), "mean_abs": float(e.mean()), "atol": 2e-4, "rtol": 1e-4}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-split", type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((PACKAGE / "manifest.json").read_text())
    record = next(r for r in manifest["folds"] if (r["session"], r["fold"]) == (SESSION, FOLD))
    checkpoint = PACKAGE / record["file"]
    assert sha(checkpoint) == record["sha256"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    original = ModelB().eval()
    original.load_state_dict(payload["model_state"], strict=True)
    plan = next(p for p in qz.plans() if p["name"] == "w8a32_mingru_ffn")
    quantfile = QUANT / f"int8_weights/{SESSION}_fold{FOLD}.pt"
    packed = torch.load(quantfile, map_location="cpu", weights_only=False)
    assert packed["source_sha256"] == record["sha256"]
    reference = qz.apply_plan(original, plan, {n: 0. for n, m in original.named_modules() if isinstance(m, (nn.Linear, nn.Conv1d))})
    exported = ModelB().eval()
    exported.load_state_dict(payload["model_state"], strict=True)
    selected = []
    for name, module in list(exported.named_modules()):
        if isinstance(module, nn.Linear) and qz.group(name) in plan["groups"]:
            q, scale = qz.quantize_weight(module.weight)
            op = packed["operators"][name]
            assert torch.equal(q, op["weight"]) and torch.equal(scale, op["scale"])
            parent, _, leaf = name.rpartition(".")
            setattr(exported.get_submodule(parent), leaf, ExportLinear(module, op))
            selected.append(name)
    assert len(selected) == 9
    prior = json.loads((OLD_EXPORT / "export_verification.json").read_text())
    assert prior["preprocessing_evidence"] == record["preprocessing_evidence"]
    assert sha(OLD_EXPORT / "parity_inputs.npy") == prior["artifacts"]["parity_inputs.npy"]
    inputs = np.load(OLD_EXPORT / "parity_inputs.npy")
    count = args.samples_per_split
    assert 0 < count <= prior["samples_per_split"]
    source_count = prior["samples_per_split"]
    indices = np.r_[np.arange(count), source_count + np.arange(count)]
    inputs = np.ascontiguousarray(inputs[indices])
    wrapper = Window(exported).eval()
    with torch.inference_mode():
        parallel = reference(torch.from_numpy(inputs)).numpy()
        sequential = reference.forward_sequential(torch.from_numpy(inputs)).numpy()
        torch.onnx.export(wrapper, torch.from_numpy(inputs[:1]), OUTPUT / "b2_620kb_w8a32.onnx",
                          input_names=["normalized_window"], output_names=["normalized_velocity"],
                          opset_version=13, dynamo=False, do_constant_folding=True, external_data=False)
        probe = exported.blocks[0].mixer.projection
        torch.onnx.export(probe, torch.zeros(1, 128), OUTPUT / "real_layer_w8a32.onnx",
                          input_names=["input"], output_names=["output"], opset_version=13,
                          dynamo=False, do_constant_folding=True, external_data=False)
    graphfile = OUTPUT / "b2_620kb_w8a32.onnx"
    graph = onnx.load(graphfile)
    onnx.checker.check_model(graph, full_check=True)
    operators = Counter(n.op_type for n in graph.graph.node)
    assert operators["DequantizeLinear"] == 9
    int8weights = sum(onnx.numpy_helper.to_array(t).nbytes for t in graph.graph.initializer
                     if t.data_type == onnx.TensorProto.INT8 and len(t.dims) == 2)
    assert int8weights == 294912
    opts = ort.SessionOptions()
    opts.intra_op_num_threads, opts.inter_op_num_threads = 4, 1
    # ORT's default QDQ/MatMul fusion changes these W8A32 results by ~0.0075.
    # Validate the explicit graph's FP32 computation, without backend quantization.
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    runtime = ort.InferenceSession(str(graphfile), opts, providers=["CPUExecutionProvider"])
    predictions = np.concatenate([runtime.run(None, {"normalized_window": x[None]})[0] for x in inputs])
    comparisons = {"parallel_vs_sequential": error(parallel, sequential),
                   "parallel_vs_onnx": error(parallel, predictions),
                   "sequential_vs_onnx": error(sequential, predictions)}
    np.save(OUTPUT / "parity_inputs.npy", inputs)
    np.save(OUTPUT / "reference_outputs.npy", sequential)
    np.save(OUTPUT / "onnx_outputs.npy", predictions)
    host_ids = np.r_[np.arange(min(8, count)), count + np.arange(min(8, count))]
    np.save(OUTPUT / "host_validation_inputs.npy", inputs[host_ids])
    np.save(OUTPUT / "host_validation_outputs.npy", sequential[host_ids])
    # Data normalization belongs to the application and is deliberately outside the graph.
    save_json(OUTPUT / "preprocessing.json", {k: v.tolist() if hasattr(v, "tolist") else v for k, v in payload["scalers"].items()})
    report = {"status": "verified" if all(c["passed"] for c in comparisons.values()) else "failed",
              "session": SESSION, "fold": FOLD, "trial": "b2", "weight_policy": "ema",
              "checkpoint_sha256": sha(checkpoint), "quantpack_sha256": sha(quantfile),
              "selection": "predeclared first session/fold; not test-selected", "plan": plan,
              "input_shape": [1, 192, 50], "output_shape": [1, 1, 2],
              "samples": len(inputs), "samples_per_split": count, "test_used": False,
              "onnxruntime_optimization": "ORT_DISABLE_ALL; preserve explicit W8A32 arithmetic",
              "comparisons": comparisons, "memory": qz.ledger(original, plan),
              "graph_nodes": len(graph.graph.node), "operators": dict(operators),
              "int8_weight_bytes": int8weights, "quantized_operators": selected,
              "onnx_file_bytes": graphfile.stat().st_size,
              "onnx_initializer_bytes": sum(onnx.numpy_helper.to_array(t).nbytes for t in graph.graph.initializer),
              "artifacts": {p.name: sha(p) for p in OUTPUT.glob("*") if p.suffix in (".onnx", ".npy")}}
    save_json(OUTPUT / "export_verification.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "memory"}, indent=2), flush=True)
    if report["status"] != "verified":
        raise RuntimeError("ONNX parity failed")


if __name__ == "__main__":
    main()
