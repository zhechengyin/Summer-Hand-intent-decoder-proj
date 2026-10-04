"""Finite export/C-validation queue for the frozen winner remaining 24 checkpoints."""
import json
from pathlib import Path
import subprocess
import sys
import time
from .train import frozen, ROOT, OUTPUT
from .model import ARCHITECTURES

firmware=Path.home()/'Documents/STM32/Custom-H747XIH6'
repo=ROOT.parents[2]
convert=ROOT.parent/'mingru_mixed_precision/cubeai_export/convert.py'
cube_root=ROOT.parent/'mingru_mixed_precision/results/b2_cubeai_export_v1'
status_path=OUTPUT/'final_export_progress.json'
deadline=time.monotonic()+12*60*60


def status(**values):
    values['updated_at']=time.time()
    status_path.write_text(json.dumps(values,indent=2))
    print(json.dumps(values),flush=True)


def run(command,log):
    with log.open('w') as stream:
        result=subprocess.run([str(x) for x in command],cwd=repo,stdout=stream,stderr=subprocess.STDOUT,
                              timeout=1200,creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}); see {log}')


def main():
    winner=json.loads((OUTPUT/'winner.json').read_text())
    assert winner['qualified'] and winner['test_used_for_selection'] is False
    architecture=winner['architecture']
    pairs=[(session,fold) for session in frozen.SESSIONS for fold in (2,3,4,5)]
    for session,fold in pairs:
        receipt=OUTPUT/architecture/f'{session}_fold{fold}.validation.json'
        status(stage='waiting_for_checkpoint',session=session,architecture=architecture)
        while not receipt.exists():
            if time.monotonic()>deadline:
                raise TimeoutError('Bounded six-hour screen export deadline reached')
            time.sleep(10)
        checkpoint=receipt.with_suffix('').with_suffix('.pt')
        exported=checkpoint.parent/(checkpoint.stem+'_export')
        if (exported/'lut_validation.json').exists():
            prior=json.loads((exported/'lut_validation.json').read_text())
            assert prior['passed'] and prior['test_evaluated'] is False
            continue
        status(stage='exporting_train_calibrated_int8',session=session,architecture=architecture)
        if not (exported/'receipt.json').exists():
            run([sys.executable,'-X','utf8','-m','indy_loco.experiment.board_latency_20261004.export',checkpoint],
                checkpoint.with_suffix('.export.log'))
        tag=f'fallback_{architecture}_{session}_fold{fold}'
        destination=cube_root/tag
        if not (destination/'receipt.json').exists():
            run([sys.executable,'-X','utf8',convert,exported/'static_int8.onnx','--tag',tag,'--dll','--external-inputs'],
                checkpoint.with_suffix('.cube.log'))
        assert json.loads((destination/'receipt.json').read_text())['status']=='complete'
        work=destination/'workspace/inspector_b2_mingru/workspace'
        graphs=firmware/'output/optimized_b2_20261004'/tag
        reference=graphs/'reference'
        reference.mkdir(parents=True,exist_ok=True)
        import shutil
        for path in (work/'generated').glob('*'):
            shutil.copyfile(path,reference/path.name)
        tools=firmware/'tools'
        run([sys.executable,tools/'instrument_b2_profile.py','--graph',reference/'b2_mingru.c',
             '--output',graphs/'profile','--config',reference/'m7_b2_profile.h'],checkpoint.with_suffix('.instrument.log'))
        width,depth=ARCHITECTURES[architecture]
        run([sys.executable,tools/'fuse_b2_graph.py','--source',reference,'--output',graphs/'fused',
             '--width',width,'--depth',depth,'--tree','--square'],checkpoint.with_suffix('.fuse.log'))
        graph=graphs/'fused/b2_mingru.c'
        for script,options in [('compact_b2_workspace.py',['--prune']),('specialize_b2_graph.py',[]),
                               ('fuse_b2_normalization.py',['--width',width,'--depth',depth]),('compact_b2_workspace.py',[])]:
            run([sys.executable,tools/script,graph,*options],checkpoint.with_suffix('.'+script+'.log'))
        status(stage='checking_full_generated_validation',session=session,architecture=architecture)
        run([sys.executable,'-X','utf8','-m','indy_loco.experiment.board_latency_20261004.validate_generated',
             checkpoint,'--work',work,'--graphs',graphs],checkpoint.with_suffix('.generated.log'))
        destination=graphs.with_name(graphs.name+'_lut')
        run([sys.executable,tools/'fuse_quantized_gelu.py','--graphs',graphs,'--output',destination,
             '--work',work,'--width',width,'--depth',depth],checkpoint.with_suffix('.lut_generation.log'))
        run([sys.executable,'-X','utf8','-m','indy_loco.experiment.board_latency_20261004.validate_generated',
             checkpoint,'--work',work,'--graphs',destination,'--label','lut'],checkpoint.with_suffix('.lut_validation.log'))
    status(stage='complete',folds=len(pairs),test_evaluated=False)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        status(stage='failed',reason=str(error))
        raise
