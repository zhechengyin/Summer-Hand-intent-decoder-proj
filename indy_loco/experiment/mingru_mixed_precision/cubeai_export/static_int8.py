"""Offline activation calibration for the fixed b2 deployment checkpoint."""
from pathlib import Path
import argparse
import hashlib
import json
from collections import Counter
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static,
)

ROOT = Path(__file__).resolve().parent.parent / 'results'
SOURCE = ROOT / 'b2_cubeai_export_v1'
OUTPUT = ROOT / 'b2_static_int8_v2'


class Reader(CalibrationDataReader):
    def __init__(self, windows):
        self.rows = iter(windows)

    def get_next(self):
        row = next(self.rows, None)
        return None if row is None else {'normalized_window': row[None]}


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=['compute', 'supported'], default='compute')
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    source = SOURCE / 'b2_direct_constant_20261004.onnx'
    model = onnx.load(source)
    constants = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == 'Constant':
            attr = next((a for a in node.attribute if a.name == 'value'), None)
            if attr is not None:
                constants[node.output[0]] = numpy_helper.to_array(attr.t)
    folded = 0
    for node in list(model.graph.node):
        if node.op_type != 'DequantizeLinear':
            continue
        weight, scale, zero = [constants[name] for name in node.input]
        assert scale.size == zero.size == 1
        values = (weight.astype(np.float32) - float(zero.item())) * float(scale.item())
        assert values.ndim == 3 and values.shape[0] == 1
        values = values[0]
        model.graph.initializer.append(numpy_helper.from_array(values, node.output[0]))
        model.graph.node.remove(node)
        folded += 1
    assert folded == 9
    used = {name for n in model.graph.node for name in n.input}
    for value in list(model.graph.initializer):
        if value.name not in used:
            model.graph.initializer.remove(value)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    floating = OUTPUT / 'constant_fp32.onnx'
    onnx.save(model, floating)
    windows = np.load(SOURCE / 'parity_inputs.npy')
    assert windows.shape == (256, 192, 50)
    provenance = json.loads((SOURCE / 'export_verification.json').read_text())
    input_hash = hashlib.sha256((SOURCE / 'parity_inputs.npy').read_bytes()).hexdigest()
    assert input_hash == provenance['artifacts']['parity_inputs.npy']
    assert provenance['samples_per_split'] == 128 and not provenance['test_used']
    # The source export stores 128 training windows, then 128 validation windows.
    calibration, validation = windows[:128], windows[128:]
    baseline_session, floating_session = session(source), session(floating)
    for window in validation:
        feed = {'normalized_window': window[None]}
        np.testing.assert_allclose(floating_session.run(None, feed)[0],
                                   baseline_session.run(None, feed)[0], atol=2e-6, rtol=1e-6)
    quantized = OUTPUT / f'static_w8a8_{args.scope}.onnx'
    operators = ['MatMul', 'Conv'] if args.scope == 'compute' else None
    quantize_static(str(floating), str(quantized), Reader(calibration),
                    quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8, per_channel=True,
                    calibrate_method=CalibrationMethod.MinMax,
                    op_types_to_quantize=operators,
                    extra_options={'ActivationSymmetric': True, 'WeightSymmetric': True})
    reference_session, quantized_session = session(source), session(quantized)
    reference = np.concatenate([reference_session.run(None, {'normalized_window': x[None]})[0].reshape(1, 2) for x in validation])
    predicted = np.concatenate([quantized_session.run(None, {'normalized_window': x[None]})[0].reshape(1, 2) for x in validation])
    np.save(OUTPUT / 'validation_reference.npy', reference)
    np.save(OUTPUT / f'validation_{args.scope}.npy', predicted)
    delta = predicted - reference
    qmodel = onnx.load(quantized)
    summary = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'checkpoint_sha256': provenance['checkpoint_sha256'],
               'calibration_and_validation_input_sha256': input_hash,
               'calibration': '128 predeclared training windows; minmax, symmetric INT8',
               'validation_windows': 128, 'held_out_test_used': False,
               'quantized_compute': operators or 'All operators supported by ORT static QDQ quantizer',
               'float_fallback': 'Normalization, nonlinearities, recurrence arithmetic; not a fully integer graph.',
               'validation_max_abs_normalized_velocity_error': float(np.max(np.abs(delta))),
               'validation_rmse_normalized_velocity_error': float(np.sqrt(np.mean(delta**2))),
               'finite': bool(np.isfinite(predicted).all()),
               'operators': dict(Counter(n.op_type for n in qmodel.graph.node)),
               'firmware_changed': False, 'board_flashed': False}
    (OUTPUT / f'quantization_{args.scope}_receipt.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
