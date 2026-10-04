"""Matched fold-1 validation baseline for the bounded architecture screen."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from indy_loco.models.midsize.model import load_checkpoint
from .train import frozen, OUTPUT

torch.set_num_threads(2)
frozen.protocol.GUI_ROOT = Path.home() / 'Documents/STM32/BCI-STM32-Plot/data/ai_device_sessions'
package = Path(__file__).resolve().parents[2] / 'models/midsize'
rows = []
for session in frozen.SESSIONS:
    data, evidence = frozen.prepare(session, 1)
    paths = list((package/session).glob('fold-1*.pt'))
    assert len(paths) == 1
    model, saved = load_checkpoint(paths[0])
    for attribute, key in [('channels','selected_channel_indices'), ('calibration_mean','feature_mean'),
                           ('calibration_effective_std','feature_std'), ('feature_std_floor','feature_std_floor'),
                           ('target_mean','target_mean'), ('target_std','target_std')]:
        np.testing.assert_allclose(np.asarray(getattr(data, attribute)).reshape(-1),
                                   np.asarray(saved[key]).reshape(-1), atol=1e-6, rtol=1e-6)
    score = frozen.protocol.evaluate_last(model, data.normalized_features, data.velocity, data.validation_bins,
                                         data.target_mean, data.target_std, torch.device('cpu'), 128)
    row = dict(session=session, fold=1, validation=score, preprocessing_evidence=evidence,
               checkpoint_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest())
    rows.append(row)
    print(json.dumps({k:v for k,v in row.items() if k != 'preprocessing_evidence'}), flush=True)
report = dict(rows=rows, validation_mean_r2=float(np.mean([r['validation']['r2_mean'] for r in rows])),
              qualification_rule='candidate six-fold macro validation R2 must exceed matched Midsize; choose lowest measured board p95 among qualifiers',
              test_evaluated=False)
(OUTPUT/'midsize_validation.json').write_text(json.dumps(report, indent=2))
