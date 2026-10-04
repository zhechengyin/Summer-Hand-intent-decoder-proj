"""Evaluate a frozen winner with generated C and firmware-matched preprocessing."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .train import frozen, OUTPUT
from .deployed import preprocessing, graph_runtime, FIRMWARE
from .screen_exports import cube_root


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    args=parser.parse_args()
    winner_path=OUTPUT/'winner.json'
    winner=json.loads(winner_path.read_text())
    assert winner['qualified'] and winner['test_used_for_selection'] is False
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    assert saved['architecture']==winner['architecture']
    architecture,session,fold=saved['architecture'],saved['session'],saved['fold']
    exported=args.checkpoint.parent/(args.checkpoint.stem+'_export')
    parity=json.loads((exported/'lut_validation.json').read_text())
    assert parity['passed'] and parity['test_evaluated'] is False
    if session=='indy_20160622_01' and fold==1:
        tag=f'fallback_{architecture}'
        cube_tag=tag+'_20261004'
    else:
        tag=f'fallback_{architecture}_{session}_fold{fold}'
        cube_tag=tag
    graphs=FIRMWARE/'output/optimized_b2_20261004'/f'{tag}_lut'
    work=cube_root/cube_tag/'workspace/inspector_b2_mingru/workspace'
    assert hashlib.sha256((graphs/'fused/b2_mingru.c').read_bytes()).hexdigest()==parity['graph_sha256']['fused']
    frozen.protocol.GUI_ROOT=Path.home()/'Documents/STM32/BCI-STM32-Plot/data/ai_device_sessions'
    data,evidence=frozen.prepare(session,fold)
    assert evidence==saved['preprocessing_evidence']
    features,postprocess,preprocess_receipt=preprocessing(data)
    runtime,handles=graph_runtime(graphs,work)
    scale=np.float32(parity['input_scale'])
    assert parity['input_zero_point']==0
    active=features[:,10450:]
    reference_features=data.normalized_features[:,10450:]
    changed=np.clip(np.rint(active/scale),-128,127)-np.clip(np.rint(reference_features/scale),-128,127)
    changed_bins=np.pad(np.any(changed!=0,axis=0).astype(np.int64),(10450,0))
    cumulative=np.concatenate(([0],np.cumsum(changed_bins)))
    preprocess_receipt['quantization_boundaries']=dict(
        max_feature_difference=float(np.max(np.abs(active-reference_features))),
        changed_feature_codes=int(np.count_nonzero(changed)),
        max_code_difference=float(np.max(np.abs(changed))),
        validation_windows_with_changed_input=int(np.count_nonzero(cumulative[data.validation_bins+1]-cumulative[data.validation_bins-49])),
        explanation='Firmware EWMA FMA and online calibration rounding versus canonical training arrays; graph parity uses identical inputs separately')
    splits={}
    for name,bins in [('validation',data.validation_bins),('test',data.test_bins)]:
        normalized=np.empty((len(bins),2),np.float32)
        for i,b in enumerate(bins):
            window=np.ascontiguousarray(features[:,b-49:b+1])
            quantized=np.clip(np.rint(window/scale),-128,127).astype(np.int8)
            assert runtime.infer(quantized,normalized[i])==1
        predictions=postprocess(normalized)
        np.savez(exported/f'deployed_{name}.npz',bins=bins,normalized=normalized,predictions=predictions,
                 targets=data.velocity[bins])
        splits[name]=frozen.protocol.metrics(data.velocity[bins],predictions)
        if name=='validation':
            reference=np.load(exported/'lut_validation.npy')[:,1]
            splits[name]['preprocessing_output_max_abs']=float(np.max(np.abs(normalized-reference)))
    from indy_loco.models.midsize.model import load_checkpoint
    package=Path(__file__).resolve().parents[2]/'models/midsize'/session
    paths=list(package.glob(f'fold-{fold}*.pt'))
    assert len(paths)==1
    model,baseline=load_checkpoint(paths[0])
    torch.set_num_threads(2)
    baseline_score=frozen.protocol.evaluate_last(model,data.normalized_features,data.velocity,data.test_bins,
                                                data.target_mean,data.target_std,torch.device('cpu'),128)
    report=dict(session=session,fold=fold,architecture=architecture,splits=splits,
                midsize_test=baseline_score,paired_delta=splits['test']['r2_mean']-baseline_score['r2_mean'],
                winner_sha256=hashlib.sha256(winner_path.read_bytes()).hexdigest(),
                checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                graph_sha256=parity['graph_sha256']['fused'],preprocessing=preprocess_receipt,
                board_tested=False,test_used_for_selection=False)
    (exported/'deployed_evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='preprocessing'},indent=2),flush=True)


if __name__=='__main__':
    main()
