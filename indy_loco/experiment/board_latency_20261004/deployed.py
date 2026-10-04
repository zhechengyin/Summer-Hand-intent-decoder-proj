"""Host execution of firmware preprocessing, INT8 input rounding and generated C."""
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from .train import OUTPUT

FIRMWARE = Path.home()/'Documents/STM32/Custom-H747XIH6'
MINGW = Path.home()/'STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.2.0/Utilities/windows/mingw64/bin'


def preprocessing(data):
    directory = OUTPUT/'host_preprocess'
    directory.mkdir(exist_ok=True)
    core = FIRMWARE/'Custom-H747XIH6/CM7/Core'
    original = (core/'Src/m7_decoder_preprocess.c').read_text()
    expression = 'M7_DECODER_EWMA_ALPHA * count\n\t\t\t\t\t+ (1.0f - M7_DECODER_EWMA_ALPHA) * context->ewma[ch]'
    assert original.count(expression) == 1
    # Cortex-M7 disassembly multiplies the old state, then VFMA adds alpha*count.
    source = original.replace(expression, 'fmaf(M7_DECODER_EWMA_ALPHA, count, (1.0f - M7_DECODER_EWMA_ALPHA) * context->ewma[ch])')
    source += '''
int build_features(const uint16_t *counts, unsigned bins, const float *floor,
                   float *features, float *mean, float *std) {
    M7DecoderPreprocess_t context;
    m7_decoder_preprocess_init(&context, floor);
    for (unsigned i=0; i<bins; ++i) {
        m7_decoder_preprocess_push_counts(&context, counts+96*i);
        if (i==10499) {
            for (unsigned f=0; f<192; ++f)
                for (unsigned j=0; j<50; ++j)
                    features[(10450+j)*192+f]=context.modelWindow[f][j];
        } else if (i>=10500) {
            for (unsigned f=0; f<192; ++f) features[i*192+f]=context.feature[f];
        }
    }
    memcpy(mean, context.mean, 192*sizeof(float));
    memcpy(std, context.effectiveStd, 192*sizeof(float));
    return context.calibrationN==10500;
}
void postprocess(const float *normalized, unsigned bins, const float *mean,
                 const float *std, float *out) {
    for (unsigned i=0; i<2*bins; ++i) out[i]=fmaf(normalized[i],std[i%2],mean[i%2]);
}
'''
    path=directory/'preprocess.c'
    path.write_text(source)
    dll=directory/'preprocess.dll'
    if not dll.exists() or (directory/'source.sha256').read_text() != hashlib.sha256(source.encode()).hexdigest():
        subprocess.run([str(MINGW/'gcc.exe'),'-shared','-O2','-ffp-contract=off','-Wl,--export-all-symbols',
                        f'-I{core/"Inc"}',str(path),'-lm','-o',str(dll)],check=True,capture_output=True)
        (directory/'source.sha256').write_text(hashlib.sha256(source.encode()).hexdigest())
    handle=os.add_dll_directory(str(MINGW))
    library=ct.CDLL(str(dll))
    f32=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
    library.build_features.argtypes=[np.ctypeslib.ndpointer(dtype=np.uint16,flags='C_CONTIGUOUS'),ct.c_uint,f32,f32,f32,f32]
    library.postprocess.argtypes=[f32,ct.c_uint,f32,f32,f32]
    counts=np.ascontiguousarray(data.counts.T,dtype=np.uint16)
    assert np.array_equal(counts.T,data.counts)
    features=np.zeros((len(counts),192),np.float32)
    mean,std=np.empty(192,np.float32),np.empty(192,np.float32)
    assert library.build_features(counts,len(counts),np.ascontiguousarray(data.feature_std_floor.reshape(-1)),features,mean,std)==1
    def postprocess(normalized):
        result=np.empty_like(normalized)
        library.postprocess(normalized,len(normalized),np.ascontiguousarray(data.target_mean.reshape(-1)),
                            np.ascontiguousarray(data.target_std.reshape(-1)),result)
        return result
    return np.ascontiguousarray(features.T),postprocess,dict(mean=mean.tolist(),std=std.tolist(),
         source_sha256=hashlib.sha256(original.encode()).hexdigest(),host_source_sha256=hashlib.sha256(source.encode()).hexdigest())


def graph_runtime(graphs, work):
    handles=[os.add_dll_directory(str(p)) for p in [MINGW,work/'lib']]
    library=ct.CDLL(str(graphs/'fused/host_adapter.dll'))
    assert library.initialize()==0
    library.infer.argtypes=[np.ctypeslib.ndpointer(dtype=np.int8,flags='C_CONTIGUOUS'),
                            np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')]
    return library,handles
