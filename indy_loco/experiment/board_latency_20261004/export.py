"""Export one frozen screening checkpoint; calibrate exclusively on training bins."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
import torch
from torch import nn
from .model import Model
from .train import frozen

ROOT = Path(__file__).resolve().parent
EXPORT_HELPERS = ROOT.parent / 'mingru_mixed_precision/cubeai_export'
sys.path.insert(0, str(EXPORT_HELPERS))
from compact import AffineMinGRU


class Reader(CalibrationDataReader):
    def __init__(self, windows):
        self.rows = iter(windows)

    def get_next(self):
        row = next(self.rows, None)
        return None if row is None else {'normalized_window': row[None]}


class Window(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, value):
        return self.model.forward_sequential(value)


def runtime(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    assert saved['test_evaluated_during_training'] is False and 1 <= saved['fold'] <= 5
    if saved['fold'] != 1:
        winner_path=ROOT/'results/winner.json'
        winner=json.loads(winner_path.read_text())
        assert saved['architecture']==winner['architecture'] and winner['test_used_for_selection'] is False
        assert saved['signature']==hashlib.sha256(winner_path.read_bytes()).hexdigest()
    frozen.protocol.GUI_ROOT = Path.home() / 'Documents/STM32/BCI-STM32-Plot/data/ai_device_sessions'
    data, evidence = frozen.prepare(saved['session'], saved['fold'])
    assert evidence == saved['preprocessing_evidence']
    for split in ['train_bins', 'validation_bins', 'test_bins']:
        assert np.array_equal(getattr(data, split), saved['split_indices'][split])
    output = args.checkpoint.parent / (args.checkpoint.stem + '_export')
    output.mkdir(exist_ok=True)
    model = Model(saved['architecture']).eval()
    model.load_state_dict(saved['weight_policies']['ema']['model_state'])
    exported = copy.deepcopy(model)
    for block in exported.blocks:
        block.mixer = AffineMinGRU(block.mixer)
    # Fixed evenly spaced probes, chosen before inspecting any output error.
    probes = np.concatenate([getattr(data, split)[np.linspace(0, len(getattr(data, split))-1, 128, dtype=int)]
                             for split in ['train_bins', 'validation_bins']])
    windows = frozen.protocol.rolling_batch(data.normalized_features, probes)
    np.save(output/'parity_inputs.npy', windows)
    np.save(output/'probe_bins.npy', probes)
    with torch.inference_mode():
        reference = model.forward_sequential(torch.from_numpy(windows)).numpy()
        affine = Window(exported)(torch.from_numpy(windows)).numpy()
        np.testing.assert_allclose(affine, reference, atol=2e-4, rtol=1e-4)
        torch.onnx.export(Window(exported), torch.from_numpy(windows[:1]), output/'float.onnx',
                          input_names=['normalized_window'], output_names=['normalized_velocity'],
                          opset_version=13, dynamo=False, do_constant_folding=True, external_data=False)
    graph = onnx.load(output/'float.onnx')
    pads = [n for n in graph.graph.node if n.op_type == 'Pad']
    convs = [n for n in graph.graph.node if n.op_type == 'Conv']
    assert len(pads) == len(convs) == 1 and convs[0].input[0] == pads[0].output[0]
    convs[0].input[0] = pads[0].input[0]
    attrs = [a for a in convs[0].attribute if a.name != 'pads']
    del convs[0].attribute[:]
    convs[0].attribute.extend(attrs + [onnx.helper.make_attribute('pads', [4, 0])])
    needed, keep = {o.name for o in graph.graph.output}, []
    for node in reversed(graph.graph.node):
        if needed.intersection(node.output):
            keep.append(node)
            needed.update(node.input)
    del graph.graph.node[:]
    graph.graph.node.extend(reversed(keep))
    graph = onnx.shape_inference.infer_shapes(graph)
    onnx.checker.check_model(graph, full_check=True)
    onnx.save(graph, output/'float.onnx')
    float_runtime = runtime(output/'float.onnx')
    predicted = np.concatenate([float_runtime.run(None, {'normalized_window': x[None]})[0] for x in windows])
    np.testing.assert_allclose(predicted, reference, atol=2e-4, rtol=1e-4)
    quantize_static(str(output/'float.onnx'), str(output/'static_int8.onnx'), Reader(windows[:128]),
                    quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8, per_channel=True,
                    calibrate_method=CalibrationMethod.MinMax, op_types_to_quantize=['MatMul', 'Conv'],
                    extra_options={'ActivationSymmetric': True, 'WeightSymmetric': True})
    quantized = runtime(output/'static_int8.onnx')
    outputs = []
    for index, bin_index in enumerate(data.validation_bins):
        window = frozen.protocol.rolling_batch(data.normalized_features, np.asarray([bin_index]))
        outputs.append(quantized.run(None, {'normalized_window': window})[0].reshape(2))
    outputs = np.asarray(outputs)
    assert np.isfinite(outputs).all()
    np.save(output/'validation_normalized.npy', outputs)
    physical = outputs * data.target_std + data.target_mean
    report = dict(architecture=saved['architecture'], session=saved['session'], fold=saved['fold'],
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  quantization_sha256=hashlib.sha256((output/'static_int8.onnx').read_bytes()).hexdigest(),
                  calibration_bins=probes[:128].tolist(), calibration_source='training only',
                  validation_windows=len(outputs), test_evaluated=False,
                  validation=frozen.protocol.metrics(data.velocity[data.validation_bins], physical),
                  fp32_validation=saved['weight_policies']['ema']['validation'],
                  arithmetic='ORT explicit QDQ; generated MCU arithmetic still requires separate verification',
                  probe_export_max_abs=float(np.max(np.abs(predicted-reference))))
    (output/'receipt.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
