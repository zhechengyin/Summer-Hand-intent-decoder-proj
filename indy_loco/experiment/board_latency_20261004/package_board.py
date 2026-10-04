"""Stage a verified smaller-model firmware configuration and versioned Flash image."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import sys
import zlib
import numpy as np
import torch
from .train import frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--graphs', type=Path, required=True)
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--lut', action='store_true')
    args = parser.parse_args()
    root = Path.home()/'Documents/STM32/Custom-H747XIH6'
    sys.path.insert(0,str(root/'tools'))
    from build_model_flash_images import wrap, check_image
    saved = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    assert saved['session']=='indy_20160622_01' and saved['fold']==1
    exported = args.checkpoint.parent/(args.checkpoint.stem+'_export')
    parity = json.loads((exported/('lut_validation.json' if args.lut else 'generated_validation.json')).read_text())
    assert parity['passed'] and parity['samples']==3683
    architecture = saved['architecture']
    width, depth = (128,2) if architecture=='w128_d2' else (96,3)
    variant = args.graphs.resolve()
    core = variant/'source/CM7'
    backup = root/'output/optimized_b2_20261004/b2_selected_source/CM7'
    shutil.copytree(backup,core,dirs_exist_ok=True)
    graph = core/'AI/B2'
    for path in (variant/'fused').glob('*'):
        if (path.suffix == '.h' and path.name != 'm7_b2_profile.h') or path.name in ('b2_mingru.c','b2_mingru_data.c'):
            shutil.copyfile(path,graph/path.name)
    source = (graph/'b2_mingru.c').read_text()
    params = (variant/'reference/b2_mingru_data_params.c').read_text()
    header = (graph/'b2_mingru_data_params.h').read_text()
    sizes = list(map(int,re.findall(r'#define AI_B2_MINGRU_DATA_WEIGHT_\d+_SIZE\s+\((\d+)\)',header)))
    assert len(sizes)<=91
    names = {int(i):n for n,i in re.findall(r'(\w+)\.data = AI_PTR\(g_b2_mingru_weights_map\[(\d+)\]',source)}
    workspace = json.loads((variant/'fused/activation_lifetimes.json').read_text())['workspace_bytes']
    lut_bytes = json.loads((variant/'lut_generation.json').read_text())['lut_bytes'] if args.lut else 0
    if args.lut:
        assert parity['graph_sha256']['fused']==hashlib.sha256((variant/'fused/b2_mingru.c').read_bytes()).hexdigest()
    previous = int(re.search(r'#define AI_B2_MINGRU_DATA_ACTIVATIONS_SIZE\s+\((\d+)\)',header)[1])
    (graph/'b2_mingru_data_params.h').write_text(header.replace(str(previous),str(workspace)))
    (graph/'b2_mingru.c').write_text(source.replace(str(previous),str(workspace)))
    path = graph/'b2_mingru_data.c'
    text = path.read_text().replace('AI_HANDLE_PTR(g_b2_mingru_weights_table)','AI_HANDLE_NULL')
    path.write_text(re.sub(r'\bs_b2_mingru_\w+\b','NULL',text))
    sdram = {i for i,n in names.items() if (f'_blocks_{depth-1}_ffn_' in n and n.endswith('_weights_array'))
             or n.startswith('_head_head_0_') and n.endswith('_weights_array')
             or n in ('model_norm_weight_3D_array','model_norm_bias_3D_array','model_head_0_bias_3D_array')}
    for i in sorted(range(len(sizes)),reverse=True):
        if sum(sizes[j] for j in sdram)>=100000:
            break
        if sizes[i]>=16000:
            sdram.add(i)
    assert 100000<=sum(sizes[i] for i in sdram)<=200000
    capacities = {0:131072,2:65504,1:253952-workspace-lut_bytes,3:61440}
    used = dict.fromkeys(capacities,0)
    tiers = [4]*len(sizes)
    for i in sorted(set(range(len(sizes)))-sdram,key=lambda i:(-sizes[i],i)):
        size = (sizes[i]+31)&~31
        bank = next((bank for bank in capacities if used[bank]+size<=capacities[bank]),None)
        assert bank is not None
        tiers[i]=bank
        used[bank]+=size
    path = core/'Core/Src/m7_b2_runtime.c'
    text = path.read_text()
    text = re.sub(r'#if M7_B2_MEMORY_LAYOUT == 2.*?#else',
                  'static const uint8_t tiers[M7_B2_WEIGHT_SEGMENTS] = {'+','.join(map(str,tiers))+'};\n'
                  '#if M7_B2_MEMORY_LAYOUT == 2\n        tier = tiers[i];\n#else',text,flags=re.S)
    if args.lut:
        text = text.replace('#include "b2_mingru_data.h"','#include "b2_mingru_data.h"\n#include "b2_gelu_lut.h"')
        text = text.replace('M7_B2_MEMORY_LAYOUT ? M7_B2_ACTIVATION_BYTES : 0U',
                            'M7_B2_MEMORY_LAYOUT ? M7_B2_ACTIVATION_BYTES + B2_GELU_LUT_BYTES : 0U')
        text = text.replace('    memoryReport.dtcm_reserved_bytes = dtcm_used;',
                            '    memcpy(m7_model_memory_d1() + M7_B2_ACTIVATION_BYTES, b2_gelu_lut, B2_GELU_LUT_BYTES);\n'
                            '    memoryReport.d1_payload_bytes += B2_GELU_LUT_BYTES;\n'
                            '    memoryReport.dtcm_reserved_bytes = dtcm_used;')
    path.write_text(text)
    path = core/'Core/Inc/m7_b2_runtime.h'
    text = path.read_text()
    for key,value in [('WEIGHT_SEGMENTS',f'{len(sizes)}U'),('PARAMETER_BYTES',f'{sum(sizes)}U'),
                      ('INPUT_SCALE',f'{parity["input_scale"]:.17g}f')]:
        text = re.sub(r'(#define M7_B2_'+key+r')\s+\S+',r'\g<1> '+value,text)
    assert parity['input_zero_point']==0
    path.write_text(text)
    path = core/'Core/Inc/m7_model_memory.h'
    path.write_text(path.read_text().replace('204800U',f'{workspace}U'))
    for path in core.glob('*.ld'):
        text = path.read_text().replace('SIZEOF(.b2_activations_sdram) == 204800',
                                       f'SIZEOF(.b2_activations_sdram) == {workspace}')
        if args.lut:
            assert '  .ARM.extab' in text
            text = text.replace('  .ARM.extab','  .b2_lut_flash : { *(.b2_lut_flash) } >FLASH\n\n  .ARM.extab',1)
        path.write_text(text)
    path = core/'Core/Inc/m7_b2_build.h'
    version = 5 if args.lut else 4
    abi = f'mingru_{architecture}_'+('lut_v5' if args.lut else 'fused_v4')
    path.write_text(path.read_text().replace('mingru_b2_fused_cubeai1020_v3',abi).replace('BCIB2F03',f'BCIB2F0{version}').replace('VERSION 3U',f'VERSION {version}U'))
    checkpoint = hashlib.sha256(args.checkpoint.read_bytes()).digest()
    path = core/'Core/Src/m7_ai_decoder.c'
    text = re.sub(r'(verified_checkpoint_sha\[32\] = \{).*?(\};)',
                  lambda m:m[1]+','.join(f'0x{b:02x}' for b in checkpoint)+m[2],path.read_text(),flags=re.S)
    path.write_text(text)
    path = core/'Core/Src/m7_model_store.c'
    text = path.read_text().replace('#include "m7_model_store.h"','#include "m7_model_store.h"\n#include "m7_b2_build.h"')
    path.write_text(text.replace('"mingru_b2_fused_cubeai1020_v3"','M7_B2_IMAGE_ABI'))
    profile = (variant/'reference/m7_b2_profile.h').read_text().replace('#define M7_B2_PROFILE_ENABLE 1','#define M7_B2_PROFILE_ENABLE 0')
    (core/'Core/Inc/m7_b2_profile.h').write_text(profile)
    quant = hashlib.sha256((exported/'static_int8.onnx').read_bytes()).digest()
    files = sorted(p for p in core.rglob('*') if p.is_file() and p.name!='m7_b2_identity.h')
    hashes = {str(p.relative_to(core)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    runtime = hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).digest()
    identity = '#ifndef M7_B2_IDENTITY_H\n#define M7_B2_IDENTITY_H\n#include <stdint.h>\n'
    for name,digest in [('quantization',quant),('runtime',runtime)]:
        identity+=f'static const uint8_t m7_b2_{name}_sha[32] = {{'+','.join(f'0x{b:02x}' for b in digest)+'};\n'
    (core/'Core/Inc/m7_b2_identity.h').write_text(identity+'#endif\n')
    frozen.protocol.GUI_ROOT = root.parent/'BCI-STM32-Plot/data/ai_device_sessions'
    data,evidence = frozen.prepare(saved['session'],1)
    assert evidence==saved['preprocessing_evidence']
    payload=bytearray(1536)
    payload[:8]=f'BCIB2F0{version}'.encode()
    struct.pack_into('<6I',payload,8,version,len(sizes),64,1536,976,0)
    payload[32:64]=checkpoint
    for values,dtype in [(data.feature_std_floor,'<f4'),(data.target_mean,'<f4'),(data.target_std,'<f4'),(data.channels,'<u2')]:
        payload.extend(np.asarray(values,dtype=dtype).tobytes())
    assert len(payload)==2512
    definitions=dict(re.findall(r'const ai_u64 (\w+)\[\d+\] = \{(.*?)\};',params,re.S))
    symbols=re.findall(r'AI_HANDLE_PTR\((s_b2_mingru_\w+)\)',params)
    assert len(symbols)==len(sizes)
    tensors=[]
    for i,(symbol,size) in enumerate(zip(symbols,sizes)):
        values=re.findall(r'0x([0-9a-fA-F]+)U',definitions[symbol])
        contents=b''.join(struct.pack('<Q',int(v,16)) for v in values)[:size]
        assert len(contents)==size
        payload.extend(bytes(-len(payload)%32))
        offset=len(payload)
        struct.pack_into('<4I',payload,64+16*i,2,i,offset,size)
        payload.extend(contents)
        tensors.append(dict(index=i,name=names[i],bytes=size,offset=offset,bank=tiers[i],sha256=hashlib.sha256(contents).hexdigest()))
    struct.pack_into('<I',payload,28,len(payload))
    image=bytearray(wrap(2,bytes(payload),abi,saved['session'],checkpoint.hex()))
    struct.pack_into('<I',image,8,2)
    image[164:196],image[196:228]=quant,runtime
    image[28:32]=bytes(4)
    struct.pack_into('<I',image,28,zlib.crc32(image[:256]))
    check_image(image,2)
    name=f'{architecture}_'+('lut_' if args.lut else '')+'slot2.bin'
    (variant/name).write_bytes(image)
    report=dict(architecture=architecture,file=name,bytes=len(image),abi=abi,
                sha256=hashlib.sha256(image).hexdigest(),crc32=f'{zlib.crc32(image):08x}',
                checkpoint_sha256=checkpoint.hex(),quantization_sha256=quant.hex(),runtime_sha256=runtime.hex(),
                workspace_bytes=workspace,lut_bytes=lut_bytes,sdram_payload_bytes=sum(sizes[i] for i in sdram),
                tensors=tensors,runtime_files=hashes)
    (variant/'board_manifest.json').write_text(json.dumps(report,indent=2))
    if args.install:
        shutil.copytree(core,root/'Custom-H747XIH6/CM7',dirs_exist_ok=True)
        gui=root.parent/'BCI-STM32-Plot/data/ai_device_sessions'
        manifest=json.loads((gui/'persistent_manifest.json').read_text())
        profile=next(p for p in manifest['profiles'] if p['slot']==2)
        profile.update(image_file='persistent/'+name,image_bytes=len(image),image_sha256=report['sha256'],
                       image_crc32=report['crc32'],abi=abi,checkpoint_sha256=checkpoint.hex(),
                       quantization_sha256=quant.hex(),runtime_sha256=runtime.hex())
        # The same fold's evaluation mask is preserved; old model scores are not evidence for this checkpoint.
        for key in ('offline_test_r2_mean','previous_w8a32_offline_test_r2_mean','offline_test_reference'):
            profile.pop(key,None)
        profile['architecture']=architecture
        profile['precision']='INT8 matrix kernels and exact GELU lookup; FP32 normalization/recurrence' if args.lut else 'INT8 matrix kernels; FP32 normalization/recurrence'
        shutil.copyfile(variant/name,gui/'persistent'/name)
        (gui/'persistent_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('tensors','runtime_files')},indent=2))


if __name__=='__main__':
    main()
