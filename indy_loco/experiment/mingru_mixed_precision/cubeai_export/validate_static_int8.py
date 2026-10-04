"""Compare the compiled Cube.AI INT8 graph with its quantized ONNX reference."""
from pathlib import Path
import ctypes as ct
import json
import os
import subprocess
import numpy as np
import onnx

ROOT = Path(__file__).resolve().parent.parent / 'results'
OUT = ROOT / 'b2_static_int8_v2'
WORK = ROOT / 'b2_cubeai_export_v1/static_int8_compute_v2_20261004/workspace/inspector_b2_mingru/workspace'
MINGW = Path.home() / 'STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.2.0/Utilities/windows/mingw64/bin'


def main():
    source = OUT / 'host_adapter.c'
    source.write_text('''#include "b2_mingru.h"
#include "b2_mingru_data.h"
static ai_handle network;
static unsigned char arena[AI_B2_MINGRU_DATA_ACTIVATIONS_SIZE] __attribute__((aligned(32)));
int initialize(void) {
    const ai_handle activations[] = {arena};
    ai_error error = ai_b2_mingru_create_and_init(&network, activations, 0);
    return error.type;
}
int infer(signed char *input, float *output) {
    ai_buffer *in = ai_b2_mingru_inputs_get(network, 0);
    ai_buffer *out = ai_b2_mingru_outputs_get(network, 0);
    in[0].data = input; out[0].data = output;
    return ai_b2_mingru_run(network, in, out);
}
''')
    dll = OUT / 'host_adapter.dll'
    command = [MINGW / 'gcc.exe', '-shared', '-O2', '-Wl,--export-all-symbols',
               f'-I{WORK / "generated"}', f'-I{WORK / "include"}', source,
               f'-L{WORK / "build"}', '-lai_b2_mingru', '-o', dll]
    result = subprocess.run(list(map(str, command)), capture_output=True, text=True)
    (OUT / 'host_compile.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stderr
    handles = [os.add_dll_directory(str(p)) for p in [WORK / 'lib', MINGW]]
    runtime = ct.CDLL(str(dll))
    assert runtime.initialize() == 0
    runtime.infer.argtypes = [np.ctypeslib.ndpointer(dtype=np.int8, flags='C_CONTIGUOUS'),
                             np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')]
    model = onnx.load(OUT / 'static_w8a8_compute.onnx')
    constants = {t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer}
    transpose = next(n for n in model.graph.node if n.op_type == 'Transpose' and n.input[0] == 'normalized_window')
    q = next(n for n in model.graph.node if n.op_type == 'QuantizeLinear' and n.input[0] == transpose.output[0])
    scale, zero = constants[q.input[1]], constants[q.input[2]]
    windows = np.load(ROOT / 'b2_cubeai_export_v1/parity_inputs.npy')[128:]
    outputs = np.zeros((len(windows), 2), np.float32)
    for index, window in enumerate(windows):
        quantized = np.clip(np.rint(window / scale) + zero, -128, 127).astype(np.int8)
        assert runtime.infer(quantized, outputs[index]) == 1
    reference = np.load(OUT / 'validation_compute.npy')
    delta = outputs - reference
    np.save(OUT / 'generated_c_validation.npy', outputs)
    report = {'samples': len(outputs), 'max_abs_error': float(np.max(np.abs(delta))),
              'rmse': float(np.sqrt(np.mean(delta**2))), 'finite': bool(np.isfinite(outputs).all()),
              'input_scale': float(scale.item()), 'input_zero_point': int(zero.item()),
              'board_tested': False}
    (OUT / 'generated_c_validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    for handle in handles:
        handle.close()


if __name__ == '__main__':
    main()
