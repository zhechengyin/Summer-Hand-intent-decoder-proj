"""Train only the frozen validation/latency winner on its remaining 24 folds."""
import hashlib
import json
from pathlib import Path
import torch
from .train import frozen, recipe, OUTPUT
from .model import Model


def main():
    winner_path=OUTPUT/'winner.json'
    winner=json.loads(winner_path.read_text())
    architecture=winner['architecture']
    assert winner['test_used_for_selection'] is False and winner['qualified']
    signature=hashlib.sha256(winner_path.read_bytes()).hexdigest()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    frozen.protocol.GUI_ROOT=Path.home()/'Documents/STM32/BCI-STM32-Plot/data/ai_device_sessions'
    trial=recipe.TRIALS['b2']
    device=torch.device('cuda')
    for session in frozen.SESSIONS:
        for fold in (2,3,4,5):
            data,evidence=frozen.prepare(session,fold)
            path=OUTPUT/architecture/f'{session}_fold{fold}.pt'
            if path.exists():
                saved=torch.load(path,map_location='cpu',weights_only=False)
                assert saved['signature']==signature and saved['preprocessing_evidence']==evidence
            else:
                frozen.protocol.seed_everything(43,mps=False)
                model=Model(architecture).to(device)
                fitted=frozen.fit(model,'mingru_b',trial,43,data,device,path.with_suffix('.epochs.json'),
                                  f'{architecture}/{session}/fold{fold}')
                saved=dict(**fitted,architecture=architecture,session=session,fold=fold,seed=43,
                           signature=signature,hyperparameters=trial,training=recipe.FIXED,
                           preprocessing_evidence=evidence,test_evaluated_during_training=False,
                           parameters=sum(p.numel() for p in model.parameters()),
                           split_indices={k:getattr(data,k).copy() for k in ['train_bins','validation_bins','test_bins']})
                frozen.original.save_training_checkpoint(path,saved)
                del model
                torch.cuda.empty_cache()
            receipt=dict(architecture=architecture,session=session,fold=fold,seed=43,signature=signature,
                         weight_policy='ema',validation=saved['weight_policies']['ema']['validation'],
                         parameters=saved['parameters'],training_seconds=saved['training_seconds'],
                         checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),test_evaluated=False)
            path.with_suffix('.validation.json').write_text(json.dumps(receipt,indent=2))
            print(json.dumps(receipt),flush=True)
    print('Frozen winner: all 30 training checkpoints complete. Test evaluation is separate.',flush=True)


if __name__=='__main__':
    main()
