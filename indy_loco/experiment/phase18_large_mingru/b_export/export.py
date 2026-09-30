"""Export fixed model-B fold to static sequential FP32 ONNX and verify parity.

No optimizer, test-set selection, quantization, approximation or persistent state.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
REPO = PHASE.parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from indy_loco.experiment.phase18_large_mingru.ab_study import train as frozen
from indy_loco.models.mingru_b.load import ROOT as PACKAGE
from indy_loco.models.mingru_b.load import load_fold

SESSION = "indy_20160622_01"
FOLD = 1
OUTPUT = PHASE / "results/b_export_v1"


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


class SequentialWindow(nn.Module):
    """Static independent 50-bin windows, with only the final readout retained."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, normalized_window):
        return self.model.forward_sequential(normalized_window)


def evenly_spaced_bins(bins, count):
    bins = np.asarray(bins, dtype=np.int64)
    if count < 1 or len(bins) < count:
        raise ValueError("Not enough bins for the predeclared parity sample")
    return bins[np.linspace(0, len(bins) - 1, count, dtype=np.int64)]


def error_summary(reference, actual, atol=2e-4, rtol=1e-4):
    reference, actual = np.asarray(reference), np.asarray(actual)
    if reference.shape != actual.shape or not np.isfinite(actual).all():
        raise ValueError("Shape or finite-output parity failed")
    error = np.abs(actual - reference)
    return {
        "passed": bool(np.allclose(reference, actual, atol=atol, rtol=rtol)),
        "max_abs": float(error.max()),
        "mean_abs": float(error.mean()),
        "rmse": float(np.sqrt(np.mean((actual - reference) ** 2))),
        "atol": atol,
        "rtol": rtol,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-split", type=int, default=128)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frozen.contract.verify_protocol_lock()
    frozen.contract.configure_paths(
        frozen.contract.INDY / "data",
        frozen.contract.default_gui_root(),
        OUTPUT / ".cache",
    )
    data, evidence = frozen.prepare(SESSION, FOLD)
    model, scalers = load_fold(SESSION, FOLD, device="cpu")
    manifest = json.loads((PACKAGE / "manifest.json").read_text(encoding="utf-8"))
    record = next(
        row
        for row in manifest["folds"]
        if (row["session"], row["fold"]) == (SESSION, FOLD)
    )
    if record["preprocessing_evidence"] != evidence:
        raise ValueError(
            "Published checkpoint preprocessing does not match frozen fold"
        )
    for key in ("target_mean", "target_std"):
        if not np.array_equal(np.asarray(getattr(data, key)), scalers[key].numpy()):
            raise ValueError(f"Published scaler mismatch: {key}")
    wrapper = SequentialWindow(model).eval()
    windows, bins_by_split = [], {}
    for split, bins in (
        ("training", data.train_bins),
        ("validation", data.validation_bins),
    ):
        chosen = evenly_spaced_bins(bins, args.samples_per_split)
        bins_by_split[split] = chosen.tolist()
        windows.append(frozen.protocol.rolling_batch(data.normalized_features, chosen))
    inputs = np.ascontiguousarray(np.concatenate(windows), dtype=np.float32)
    np.save(OUTPUT / "parity_inputs.npy", inputs)
    predictions = {name: [] for name in ("parallel", "sequential", "onnx")}
    with torch.inference_mode():
        for start in range(0, len(inputs), 32):
            tensor = torch.from_numpy(inputs[start : start + 32])
            predictions["parallel"].append(model(tensor).numpy())
            predictions["sequential"].append(wrapper(tensor).numpy())
        # Explicitly establish no cross-call persistent state.
        first = wrapper(torch.from_numpy(inputs[:1])).clone()
        wrapper(torch.from_numpy(inputs[-1:]))
        again = wrapper(torch.from_numpy(inputs[:1]))
        if not torch.equal(first, again):
            raise ValueError("Independent-window reset check failed")
    graph_path = OUTPUT / "mingru_b_fp32_sequential.onnx"
    print("Exporting fixed batch1 x 192 x 50 FP32 graph", flush=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            torch.from_numpy(inputs[:1]),
            str(graph_path),
            input_names=["normalized_window"],
            output_names=["normalized_velocity"],
            opset_version=13,
            dynamo=False,
            do_constant_folding=True,
            external_data=False,
        )
    graph = onnx.load(graph_path)
    onnx.checker.check_model(graph, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    runtime = ort.InferenceSession(
        str(graph_path), options, providers=["CPUExecutionProvider"]
    )
    started = time.perf_counter()
    for window in inputs:
        predictions["onnx"].append(
            runtime.run(None, {"normalized_window": window[None]})[0]
        )
    elapsed = time.perf_counter() - started
    for key, values in predictions.items():
        predictions[key] = np.concatenate(values)
        np.save(OUTPUT / f"{key}_outputs.npy", predictions[key])
    result = {
        "status": "complete",
        "purpose": "FP32 export feasibility only; no test-set evaluation or model selection",
        "fixed_checkpoint": {
            "session": SESSION,
            "fold": FOLD,
            "weight_policy": "ema",
            "selection": "predeclared first session, first fold; not selected by accuracy",
            "path": str(PACKAGE / record["file"]),
            "sha256": record["sha256"],
            "parameters": sum(p.numel() for p in model.parameters()),
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "input_shape": [1, 192, 50],
        "output_shape": [1, 1, 2],
        "opset": 13,
        "precision": "float32",
        "independent_window_reset": True,
        "samples_per_split": args.samples_per_split,
        "bins_by_split": bins_by_split,
        "preprocessing_evidence": evidence,
        "operators": dict(
            sorted(Counter(node.op_type for node in graph.graph.node).items())
        ),
        "graph_nodes": len(graph.graph.node),
        "storage": {
            "onnx_file_bytes": graph_path.stat().st_size,
            "unique_fp32_parameter_bytes": sum(
                p.numel() * p.element_size() for p in model.parameters()
            ),
            "onnx_initializer_bytes": sum(
                onnx.numpy_helper.to_array(t).nbytes for t in graph.graph.initializer
            ),
            "onnx_initializers": len(graph.graph.initializer),
            "note": "Graph file includes metadata and unrolled arithmetic; initializer bytes count each shared weight once.",
        },
        "onnx_cpu_parity_wall_seconds": elapsed,
        "not_board_latency": True,
        "comparisons": {},
    }
    for split, selected in (
        ("training", slice(0, args.samples_per_split)),
        ("validation", slice(args.samples_per_split, None)),
    ):
        result["comparisons"][split] = {
            "parallel_vs_sequential": error_summary(
                predictions["parallel"][selected], predictions["sequential"][selected]
            ),
            "sequential_vs_onnx": error_summary(
                predictions["sequential"][selected], predictions["onnx"][selected]
            ),
            "parallel_vs_onnx": error_summary(
                predictions["parallel"][selected], predictions["onnx"][selected]
            ),
        }
    np.save(
        OUTPUT / "host_validation_inputs.npy",
        np.concatenate(
            (inputs[:8], inputs[args.samples_per_split : args.samples_per_split + 8])
        ),
    )
    np.save(
        OUTPUT / "host_validation_outputs.npy",
        np.concatenate(
            (
                predictions["sequential"][:8],
                predictions["sequential"][
                    args.samples_per_split : args.samples_per_split + 8
                ],
            )
        ),
    )
    result["artifacts"] = {
        p.name: sha256(p)
        for p in sorted(OUTPUT.glob("*"))
        if p.suffix in (".npy", ".onnx")
    }
    result["source_sha256"] = {
        str(p.relative_to(REPO)): sha256(p) for p in sorted(HERE.glob("*.py"))
    }
    write_json(OUTPUT / "export_verification.json", result)
    if not all(
        item["passed"]
        for split in result["comparisons"].values()
        for item in split.values()
    ):
        raise RuntimeError("Numerical parity failed; see export_verification.json")
    print(
        json.dumps(
            {
                "graph": str(graph_path),
                "nodes": result["graph_nodes"],
                "comparisons": result["comparisons"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
