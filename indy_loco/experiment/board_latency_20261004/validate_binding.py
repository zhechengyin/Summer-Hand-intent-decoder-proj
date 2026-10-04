"""Verify the deployable image, relocated weights, and independent-window binding."""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import subprocess
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',type=Path,required=True)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--export',type=Path,required=True)
    args=parser.parse_args()
    variant,work=args.variant.resolve(),args.work.resolve()
    core=variant/'source/CM7'
    graph=core/'AI/B2'
    report=json.loads((variant/'board_manifest.json').read_text())
    count=len(report['tensors'])
    mingw=Path.home()/'STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.2.0/Utilities/windows/mingw64/bin'
    dll=variant/'binding.dll'
    sources=[graph/'b2_mingru.c',graph/'b2_mingru_data.c']
    sources += [core/'Core/Src'/name for name in ['m7_model_memory.c','m7_b2_runtime.c','m7_b2_bundle.c']]
    command=[mingw/'gcc.exe','-shared','-std=c11','-O2','-DM7_MODEL_MEMORY_HOST','-Wl,--export-all-symbols',
             f'-I{graph}',f'-I{core/"Core/Inc"}',f'-I{work/"include"}',*sources,
             f'-L{work/"lib/static"}','-lruntime','-lst_cmsis_nn','-lcmsis-nn','-lm','-o',dll]
    result=subprocess.run(list(map(str,command)),capture_output=True,text=True,timeout=120)
    (variant/'binding_compile.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stderr[-3000:]
    handles=[os.add_dll_directory(str(p)) for p in [mingw,work/'lib']]
    runtime=ct.CDLL(str(dll))
    reference=ct.CDLL(str(variant/'fused/host_adapter.dll'))
    assert reference.initialize()==0
    class Bundle(ct.Structure):
        _fields_=[('offsets',ct.c_uint32*count),('floor',ct.c_float*192),('mean',ct.c_float*2),
                  ('std',ct.c_float*2),('channels',ct.c_uint16*96)]
    runtime.m7_b2_bundle_validate.argtypes=[ct.c_void_p,ct.c_uint32,ct.c_void_p,ct.POINTER(Bundle)]
    runtime.m7_b2_bundle_validate.restype=ct.c_uint8
    image=(variant/report['file']).read_bytes()
    source=ct.create_string_buffer(image[256:])
    checkpoint=ct.create_string_buffer(image[128:160])
    bundle=Bundle()
    assert runtime.m7_b2_bundle_validate(source,len(image)-256,checkpoint,ct.byref(bundle))==1
    assert runtime.m7_b2_bundle_validate(source,len(image)-257,checkpoint,ct.byref(Bundle()))==0
    runtime.m7_b2_runtime_init.argtypes=[ct.c_void_p,ct.c_size_t,ct.POINTER(ct.c_uint32)]
    runtime.m7_b2_runtime_init.restype=ct.c_int
    assert runtime.m7_b2_runtime_init(source,len(image)-256,bundle.offsets)==-1
    runtime.m7_model_memory_init()
    assert runtime.m7_b2_runtime_init(source,len(image)-256,bundle.offsets)==0
    ct.memset(source,0xa5,len(image)-256)
    float_array=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
    runtime.m7_b2_runtime_run.argtypes=[float_array,float_array]
    reference.infer.argtypes=[np.ctypeslib.ndpointer(dtype=np.int8,flags='C_CONTIGUOUS'),float_array]
    parity=json.loads((args.export/'generated_validation.json').read_text())
    windows=np.load(args.export/'parity_inputs.npy')
    outputs=np.empty((len(windows),2),np.float32)
    for index,window in enumerate(windows):
        expected=np.empty(2,np.float32)
        quantized=np.clip(np.rint(window/parity['input_scale']),-128,127).astype(np.int8)
        assert reference.infer(quantized,expected)==1
        assert runtime.m7_b2_runtime_run(window,outputs[index])==0
        np.testing.assert_array_equal(outputs[index],expected)
    repeated=np.empty(2,np.float32)
    assert runtime.m7_b2_runtime_run(windows[0],repeated)==0
    np.testing.assert_array_equal(repeated,outputs[0])
    runtime.m7_b2_runtime_destroy()
    assert runtime.m7_b2_runtime_run(windows[0],repeated)==-1
    ct.memmove(source,image[256:],len(image)-256)
    assert runtime.m7_b2_runtime_init(source,len(image)-256,bundle.offsets)==0
    assert runtime.m7_b2_runtime_run(windows[0],repeated)==0
    np.testing.assert_array_equal(repeated,outputs[0])
    runtime.m7_b2_runtime_destroy()
    result=dict(passed=True,bit_exact=True,windows=len(windows),real_image_verified=True,
                staging_overwritten=True,independent_window_reset=True,destroy_reload=True,
                image_sha256=report['sha256'])
    (variant/'binding_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    for handle in handles:
        handle.close()


if __name__=='__main__':
    main()
