"""Equivalent affine-prefix export, avoiding a 50-deep unrolled recurrent DAG."""
from __future__ import annotations

import json
from collections import Counter

import export as base

np, torch, onnx, ort = base.np, base.torch, base.onnx, base.ort
nn = torch.nn


class AffineMinGRU(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.projection = source.projection

    def forward(self, values, recurrence="sequential", last_only=False):
        candidate, gate = self.projection(values).chunk(2, -1)
        candidate = torch.where(candidate >= 0, candidate + .5, candidate.sigmoid())
        write = gate.sigmoid()
        a, b = 1. - write, write * candidate
        # Compose affine maps (a2,b2) o (a1,b1) = (a2*a1,b2+a2*b1).
        # Six prefix stages cover 50 timesteps, with zero initial state.
        for offset in (1, 2, 4, 8, 16, 32):
            # Explicit positive endpoints avoid Cube.AI 10.2 interpreting
            # ONNX end=-1 as inclusive (which otherwise grows 50 steps to 51).
            next_b = torch.cat((b[:, :offset], b[:, offset:50] + a[:, offset:50] * b[:, :50-offset]), 1)
            next_a = torch.cat((a[:, :offset], a[:, offset:50] * a[:, :50-offset]), 1)
            a, b = next_a, next_b
        return b[:, 49:50] if last_only else b


def main():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    manifest = json.loads((base.PACKAGE / "manifest.json").read_text())
    record = next(r for r in manifest["folds"] if (r["session"], r["fold"]) == (base.SESSION, base.FOLD))
    checkpoint = base.PACKAGE / record["file"]
    assert base.sha(checkpoint) == record["sha256"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = base.ModelB().eval()
    model.load_state_dict(payload["model_state"])
    packed = torch.load(base.QUANT / f"int8_weights/{base.SESSION}_fold{base.FOLD}.pt", map_location="cpu", weights_only=False)
    assert packed["source_sha256"] == record["sha256"]
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and base.qz.group(name) in ("mingru", "ffn"):
            parent, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(parent), leaf, base.ExportLinear(module, packed["operators"][name]))
    for block in model.blocks:
        block.mixer = AffineMinGRU(block.mixer)
    wrapper = base.Window(model).eval()
    inputs = np.load(base.OUTPUT / "parity_inputs.npy")
    reference = np.load(base.OUTPUT / "reference_outputs.npy")
    path = base.OUTPUT / "b2_620kb_affine_scan.onnx"
    with torch.inference_mode():
        pytorch = wrapper(torch.from_numpy(inputs)).numpy()
        torch.onnx.export(wrapper, torch.from_numpy(inputs[:1]), path, opset_version=13,
                          dynamo=False, do_constant_folding=True, external_data=False,
                          input_names=["normalized_window"], output_names=["normalized_velocity"])
    graph = onnx.load(path)
    # Express causal padding directly in Conv. Cube.AI 10.2 generates invalid
    # C for torch's separate Pad node with an omitted zero-value input.
    pads = [n for n in graph.graph.node if n.op_type == "Pad"]
    convolutions = [n for n in graph.graph.node if n.op_type == "Conv"]
    assert len(pads) == len(convolutions) == 1
    pad, conv = pads[0], convolutions[0]
    assert conv.input[0] == pad.output[0]
    conv.input[0] = pad.input[0]
    attrs = [a for a in conv.attribute if a.name != "pads"]
    del conv.attribute[:]
    conv.attribute.extend(attrs + [onnx.helper.make_attribute("pads", [4, 0])])
    needed = {o.name for o in graph.graph.output}
    keep = []
    for node in reversed(graph.graph.node):
        if needed.intersection(node.output):
            keep.append(node)
            needed.update(node.input)
    del graph.graph.node[:]
    graph.graph.node.extend(reversed(keep))
    onnx.save(graph, path)
    onnx.checker.check_model(graph, full_check=True)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads, opts.inter_op_num_threads = 4, 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    predictions = np.concatenate([session.run(None, {"normalized_window": x[None]})[0] for x in inputs])
    comparisons = {"sequential_vs_affine_pytorch": base.error(reference, pytorch),
                   "sequential_vs_affine_onnx": base.error(reference, predictions)}
    report = {"graph_sha256": base.sha(path), "nodes": len(graph.graph.node),
              "operators": dict(Counter(n.op_type for n in graph.graph.node)),
              "samples": len(inputs), "comparisons": comparisons,
              "precision": "same W8A32 weights; algebraically equivalent affine scan with FP32 reassociation",
              "independent_window": True, "persistent_state": False}
    base.save_json(base.OUTPUT / "compact_verification.json", report)
    print(json.dumps(report, indent=2))
    assert all(c["passed"] for c in comparisons.values())


if __name__ == "__main__":
    main()
