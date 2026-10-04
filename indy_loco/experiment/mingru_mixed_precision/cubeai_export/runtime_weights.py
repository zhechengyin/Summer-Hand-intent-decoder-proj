"""Retain INT8 storage via Cube.AI application-owned input buffers.

The nine weight buffers are fixed model parameters, not sensor inputs or state.
Use unit-scale INT8-to-FP32 conversion and explicit per-output scale after GEMM:
X @ (Q * s) == (X @ Q) * s mathematically; FP32 rounding is checked separately.
This avoids Cube.AI 10.2's incorrect per-channel runtime DQ behavior in a probe.
"""
from __future__ import annotations

import json
import re
import export as base

np, onnx, ort = base.np, base.onnx, base.ort
h, nh = onnx.helper, onnx.numpy_helper


def main():
    graph = onnx.load(base.OUTPUT / "b2_620kb_affine_scan.onnx")
    g = graph.graph
    initializers = {t.name: nh.to_array(t) for t in g.initializer}
    replacements, removed, weights = {}, set(), []
    for dq in list(g.node):
        if dq.op_type != "DequantizeLinear":
            continue
        qname, sname, _ = dq.input
        q = initializers[qname]
        assert q.ndim == 2 and q.dtype == np.int8
        trans = next(n for n in g.node if dq.output[0] in n.input)
        assert trans.op_type == "Transpose"
        assert list(next(a.ints for a in trans.attribute if a.name == "perm")) == [1, 0]
        mm = next(n for n in g.node if trans.output[0] in n.input)
        assert mm.op_type == "MatMul" and mm.input[1] == trans.output[0]
        name = re.sub(r"[^a-zA-Z0-9_]", "_", qname) + "_packed"
        w = np.ascontiguousarray(q.T[None])
        g.input.append(h.make_tensor_value_info(name, onnx.TensorProto.INT8, list(w.shape)))
        unscaled = mm.output[0] + "_unscaled"
        replacements[dq.name] = [h.make_node("DequantizeLinear", [name, "unit_scale", "unit_zero"],
                                                            list(trans.output), name=dq.name)]
        replacements[mm.name] = [h.make_node("MatMul", list(mm.input), [unscaled], name=mm.name),
                                 h.make_node("Mul", [unscaled, sname], list(mm.output), name=mm.name + "_per_channel_scale")]
        removed.add(trans.name)
        weights.append({"input_name": name, "shape": list(w.shape), "bytes": w.nbytes,
                        "source_initializer": qname, "data": w})
    assert len(weights) == 9
    nodes = []
    for n in g.node:
        if n.name not in removed:
            nodes.extend(replacements.get(n.name, [n]))
    del g.node[:]
    g.node.extend(nodes)
    used = {s for n in g.node for s in n.input}
    retained = [t for t in g.initializer if t.name in used]
    del g.initializer[:]
    g.initializer.extend(retained + [nh.from_array(np.array(1., np.float32), "unit_scale"),
                                    nh.from_array(np.array(0, np.int8), "unit_zero")])
    # Shape information comes from the modified graph, not the old rank-2 weights.
    del g.value_info[:]
    path = base.OUTPUT / "b2_620kb_runtime_weights.onnx"
    onnx.checker.check_model(graph, full_check=True)
    onnx.save(graph, path)
    inputs = np.load(base.OUTPUT / "parity_inputs.npy")
    expected = np.load(base.OUTPUT / "reference_outputs.npy")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads, opts.inter_op_num_threads = 4, 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    feed = {w["input_name"]: w["data"] for w in weights}
    actual = []
    for x in inputs:
        actual.append(session.run(None, {**feed, "normalized_window": x[None]})[0])
    actual = np.concatenate(actual)
    comparison = base.error(expected, actual)
    np.save(base.OUTPUT / "runtime_weights_onnx_outputs.npy", actual)
    # All 256 fixed training/validation windows will be checked against generated C.
    # NPZ remains compact by storing fixed weights once; CLI arrays use a smaller
    # 16-window set, while the adapter host check can reuse constant pointers.
    n = len(np.load(base.OUTPUT / "host_validation_inputs.npy"))
    input_files = [str(base.OUTPUT / "host_validation_inputs.npy")]
    header = ['#ifndef B2_PACKED_WEIGHTS_H', '#define B2_PACKED_WEIGHTS_H', '#include <stdint.h>',
              '#define B2_PACKED_WEIGHT_COUNT 9', '#define B2_PACKED_WEIGHT_BYTES 294912']
    source = ['/* Fixed b2 INT8 weights, transposed for Cube.AI runtime inputs. */', '#include "b2_packed_weights.h"']
    for i, w in enumerate(weights):
        arrayname = f"b2_packed_weight_{i}"
        macro = f"B2_WEIGHT_{i}_ATTR"
        header += [f"extern const int8_t {arrayname}[{w['bytes']}];"]
        source += [f"#ifndef {macro}", f"#define {macro}", "#endif",
                   f"{macro} const int8_t {arrayname}[{w['bytes']}] = {{"]
        flat = w["data"].reshape(-1)
        source += ["  " + ",".join(str(int(x)) for x in flat[j:j+32]) + "," for j in range(0, len(flat), 32)]
        source += ["};"]
        p = base.OUTPUT / f"host_weight_{i}.npy"
        np.save(p, np.repeat(w["data"], n, axis=0))
        input_files.append(str(p))
    header += ["extern const int8_t *const b2_packed_weights[B2_PACKED_WEIGHT_COUNT];", "#endif"]
    source += ["const int8_t *const b2_packed_weights[B2_PACKED_WEIGHT_COUNT] = {",
               ",".join(f"b2_packed_weight_{i}" for i in range(9)), "};"]
    adapter = base.OUTPUT / "adapter"
    adapter.mkdir(exist_ok=True)
    (adapter / "b2_packed_weights.h").write_text("\n".join(header) + "\n", encoding="utf-8")
    (adapter / "b2_packed_weights.c").write_text("\n".join(source) + "\n", encoding="utf-8")
    np.savez(base.OUTPUT / "runtime_packed_weights.npz", **feed)
    rows = [{k: v for k, v in w.items() if k != "data"} for w in weights]
    report = {"status": "verified" if comparison["passed"] else "failed", "graph_sha256": base.sha(path),
              "samples": len(inputs), "reference": "original b2 W8A32 sequential PyTorch", "comparison": comparison,
              "weights": rows, "packed_int8_bytes": sum(w["bytes"] for w in weights),
              "validation_input_files": input_files,
              "contract": "input 0 is normalized Bx192x50; inputs 1..9 are fixed INT8 weights, no sensor changes or persistent state",
              "arithmetic": "FP32 matrix multiply with integer-valued FP32 weights then per-channel scale, FP32 bias and activations",
              "limitations": "Requires generated-C verification; runtime expands a weight matrix temporarily. Not a native mixed INT8/FP32 GEMM kernel or measured MCU latency."}
    base.save_json(base.OUTPUT / "runtime_weights_verification.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("weights", "validation_input_files")}, indent=2))
    assert comparison["passed"]


if __name__ == "__main__":
    main()
