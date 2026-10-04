"""Bounded six-session fold-1 screen using the frozen b2 training function."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from indy_loco.experiment.phase18_large_mingru.ab_study import train as frozen
from indy_loco.experiment.phase18_large_mingru.b_tuning import plan as recipe
from .model import Model, ARCHITECTURES

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'results'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture', choices=list(ARCHITECTURES))
    parser.add_argument('--folds', nargs='+', type=int, default=[1])
    args = parser.parse_args()
    assert args.folds == [1], 'Promotion to remaining folds requires the frozen validation/latency winner'
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    frozen.protocol.GUI_ROOT = Path.home() / 'Documents/STM32/BCI-STM32-Plot/data/ai_device_sessions'
    trial = recipe.TRIALS['b2']
    assert trial['learning_rate'] == 0.0012 and recipe.FIXED['epochs'] == 60 and recipe.FIXED['patience'] == 15
    config = dict(architectures=ARCHITECTURES, trial=trial, training=recipe.FIXED, seed=43,
                  screen_folds=[1], sessions=list(frozen.SESSIONS), test_evaluated=False,
                  code_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               [ROOT/'model.py', ROOT/'train.py', Path(frozen.__file__)]})
    OUTPUT.mkdir(exist_ok=True)
    config_path = OUTPUT / 'screen_config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == json.loads(json.dumps(config)), 'Frozen screen configuration changed'
    else:
        config_path.write_text(json.dumps(config, indent=2))
    signature = hashlib.sha256(config_path.read_bytes()).hexdigest()
    device = torch.device('cuda')
    for session in frozen.SESSIONS:
        data, evidence = frozen.prepare(session, 1)
        for architecture in ([args.architecture] if args.architecture else ARCHITECTURES):
            destination = OUTPUT / architecture
            destination.mkdir(exist_ok=True)
            path = destination / f'{session}_fold1.pt'
            if path.exists():
                saved = torch.load(path, map_location='cpu', weights_only=False)
                assert saved['signature'] == signature and saved['preprocessing_evidence'] == evidence
            else:
                frozen.protocol.seed_everything(43, mps=False)
                model = Model(architecture).to(device)
                fitted = frozen.fit(model, 'mingru_b', trial, 43, data, device,
                                    path.with_suffix('.epochs.json'), f'{architecture}/{session}/fold1')
                saved = dict(**fitted, architecture=architecture, session=session, fold=1, seed=43,
                             signature=signature, hyperparameters=trial, training=recipe.FIXED,
                             preprocessing_evidence=evidence, test_evaluated_during_training=False,
                             parameters=sum(p.numel() for p in model.parameters()),
                             split_indices={k:getattr(data,k).copy() for k in ['train_bins','validation_bins','test_bins']})
                frozen.original.save_training_checkpoint(path, saved)
                del model
                torch.cuda.empty_cache()
            receipt = dict(architecture=architecture, session=session, fold=1, seed=43, signature=signature,
                           weight_policy='ema', validation=saved['weight_policies']['ema']['validation'],
                           parameters=saved['parameters'], training_seconds=saved['training_seconds'],
                           checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), test_evaluated=False)
            path.with_suffix('.validation.json').write_text(json.dumps(receipt, indent=2))
            print(json.dumps(receipt), flush=True)
    print('Prescribed fold-1 screening complete; no promotion or test evaluation performed.', flush=True)


if __name__ == '__main__':
    main()
