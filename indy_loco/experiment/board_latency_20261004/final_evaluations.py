"""Evaluate exactly the frozen winner's 30 folds once their C parity gates pass."""
import json
import sys
import time
from .train import OUTPUT, frozen
from .screen_exports import run

winner=json.loads((OUTPUT/'winner.json').read_text())
assert winner['qualified'] and winner['test_used_for_selection'] is False
architecture=winner['architecture']
deadline=time.monotonic()+12*3600
pending=[(session,fold) for session in frozen.SESSIONS for fold in (1,2,3,4,5)]
status=OUTPUT/'final_evaluation_progress.json'
while pending:
    progressed=False
    for session,fold in list(pending):
        checkpoint=OUTPUT/architecture/f'{session}_fold{fold}.pt'
        exported=checkpoint.parent/(checkpoint.stem+'_export')
        receipt=exported/'deployed_evaluation.json'
        if not (exported/'lut_validation.json').exists():
            continue
        if receipt.exists():
            prior=json.loads(receipt.read_text())
            assert prior['architecture']==architecture and prior['test_used_for_selection'] is False
        else:
            status.write_text(json.dumps(dict(stage='evaluating_frozen_test',session=session,fold=fold)))
            run([sys.executable,'-X','utf8','-m','indy_loco.experiment.board_latency_20261004.evaluate_fold',checkpoint],
                checkpoint.with_suffix('.deployed_evaluation.log'))
        pending.remove((session,fold))
        progressed=True
    if not progressed:
        status.write_text(json.dumps(dict(stage='waiting_for_parity',remaining=len(pending))))
        if time.monotonic()>deadline:
            raise TimeoutError('Finite final evaluation deadline reached')
        time.sleep(10)
status.write_text(json.dumps(dict(stage='complete',folds=30,architecture=architecture)))
