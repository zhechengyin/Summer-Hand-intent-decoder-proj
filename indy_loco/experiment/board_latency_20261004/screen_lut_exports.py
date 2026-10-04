"""Bounded, validation-only LUT parity checks for the remaining screen checkpoints."""
import json
from pathlib import Path
import sys
import time
from .train import frozen, OUTPUT
from .model import ARCHITECTURES
from .screen_exports import firmware, cube_root, run

deadline=time.monotonic()+6*60*60
status_path=OUTPUT/'screen_lut_progress.json'
for session in frozen.SESSIONS:
    if session=='indy_20160622_01':
        continue
    for architecture,(width,depth) in ARCHITECTURES.items():
        checkpoint=OUTPUT/architecture/f'{session}_fold1.pt'
        exported=checkpoint.parent/(checkpoint.stem+'_export')
        source=exported/'generated_validation.json'
        status_path.write_text(json.dumps(dict(stage='waiting_for_reference',session=session,architecture=architecture)))
        while not source.exists():
            if time.monotonic()>deadline:
                raise TimeoutError('Bounded LUT validation deadline reached')
            time.sleep(10)
        assert json.loads(source.read_text())['passed']
        if (exported/'lut_validation.json').exists():
            assert json.loads((exported/'lut_validation.json').read_text())['passed']
            continue
        tag=f'fallback_{architecture}_{session}_fold1'
        graphs=firmware/'output/optimized_b2_20261004'/tag
        work=cube_root/tag/'workspace/inspector_b2_mingru/workspace'
        destination=graphs.with_name(graphs.name+'_lut')
        status_path.write_text(json.dumps(dict(stage='enumerating_lut',session=session,architecture=architecture)))
        run([sys.executable,firmware/'tools/fuse_quantized_gelu.py','--graphs',graphs,'--output',destination,
             '--work',work,'--width',width,'--depth',depth],checkpoint.with_suffix('.lut_generation.log'))
        status_path.write_text(json.dumps(dict(stage='full_validation',session=session,architecture=architecture)))
        run([sys.executable,'-X','utf8','-m','indy_loco.experiment.board_latency_20261004.validate_generated',
             checkpoint,'--work',work,'--graphs',destination,'--label','lut'],checkpoint.with_suffix('.lut_validation.log'))
status_path.write_text(json.dumps(dict(stage='complete',test_evaluated=False)))
