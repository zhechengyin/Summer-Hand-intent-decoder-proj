"""Compile reference and fused Cube.AI graphs and compare full validation arithmetic."""
import argparse
import ctypes as ct
import json
import hashlib
import os
from pathlib import Path
import subprocess
import numpy as np
import onnx
import torch
from .train import frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--graphs', type=Path, required=True)
    parser.add_argument('--label', default='generated')
    args = parser.parse_args()
    work, graphs = args.work.resolve(), args.graphs.resolve()
    exported = args.checkpoint.parent/(args.checkpoint.stem+'_export')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    mingw = Path.home()/'STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.2.0/Utilities/windows/mingw64/bin'
    firmware = Path.home()/'Documents/STM32/Custom-H747XIH6/Custom-H747XIH6/CM7'
    dll_handles = [os.add_dll_directory(str(p)) for p in [mingw, work/'lib']]
    runtimes = []
    for kind in ['reference', 'fused']:
        graph = graphs/kind
        adapter = graph/'host_adapter.c'
        adapter.write_text('''#include "b2_mingru.h"
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
        dll = graph/'host_adapter.dll'
        command = [mingw/'gcc.exe','-shared','-std=c11','-O2','-DM7_MODEL_MEMORY_HOST','-Wl,--export-all-symbols',
                   f'-I{graph}', f'-I{work/"include"}', f'-I{firmware/"Core/Inc"}', adapter,
                   graph/'b2_mingru.c', graph/'b2_mingru_data.c', graph/'b2_mingru_data_params.c',
                   f'-L{work/"lib/static"}', '-lruntime','-lst_cmsis_nn','-lcmsis-nn','-lm','-o',dll]
        result = subprocess.run(list(map(str, command)), capture_output=True, text=True, timeout=120)
        (graph/'host_compile.log').write_text(result.stdout+result.stderr)
        assert result.returncode == 0, result.stderr[-4000:]
        runtime = ct.CDLL(str(dll))
        assert runtime.initialize() == 0
        runtime.infer.argtypes = [np.ctypeslib.ndpointer(dtype=np.int8,flags='C_CONTIGUOUS'),
                                 np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')]
        runtimes.append(runtime)
    model = onnx.load(exported/'static_int8.onnx')
    constants = {t.name:onnx.numpy_helper.to_array(t) for t in model.graph.initializer}
    transpose = next(n for n in model.graph.node if n.op_type=='Transpose' and n.input[0]=='normalized_window')
    quant = next(n for n in model.graph.node if n.op_type=='QuantizeLinear' and n.input[0]==transpose.output[0])
    scale, zero = constants[quant.input[1]], constants[quant.input[2]]
    assert scale.size == zero.size == 1
    def run(window):
        values = np.clip(np.rint(window/scale)+zero, -128,127).astype(np.int8)
        outputs = np.empty((2,2),np.float32)
        for index, runtime in enumerate(runtimes):
            assert runtime.infer(values,outputs[index]) == 1
        return outputs
    probes = np.asarray([run(x) for x in np.load(exported/'parity_inputs.npy')])
    np.testing.assert_allclose(probes[:,0],probes[:,1],atol=2e-4,rtol=1e-4)
    frozen.protocol.GUI_ROOT = Path.home()/'Documents/STM32/BCI-STM32-Plot/data/ai_device_sessions'
    data,evidence = frozen.prepare(saved['session'],saved['fold'])
    assert evidence == saved['preprocessing_evidence']
    results = np.asarray([run(frozen.protocol.rolling_batch(data.normalized_features,np.asarray([b]))[0])
                          for b in data.validation_bins])
    np.save(exported/f'{args.label}_validation.npy',results)
    delta = np.abs(results[:,0]-results[:,1])
    failed = ~np.isclose(results[:,0],results[:,1],atol=2e-4,rtol=1e-4)
    report = dict(samples=len(results),fixed_probes=len(probes),
                  reference_vs_fused_max_abs=float(delta.max()), failed_outputs=int(failed.sum()),
                  passed=not bool(failed.any()), bit_exact=bool(np.array_equal(results[:,0],results[:,1])),
                  ort_vs_generated_max_abs=float(np.max(np.abs(np.load(exported/'validation_normalized.npy')-results[:,0]))),
                  input_scale=float(scale.item()),input_zero_point=int(zero.item()),
                  validation=frozen.protocol.metrics(data.velocity[data.validation_bins],results[:,1]*data.target_std+data.target_mean),
                  graph_sha256={kind:hashlib.sha256((graphs/kind/'b2_mingru.c').read_bytes()).hexdigest()
                                for kind in ['reference','fused']},
                  test_evaluated=False,board_tested=False)
    (exported/f'{args.label}_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    assert report['passed']
    for handle in dll_handles:
        handle.close()


if __name__ == '__main__':
    main()
