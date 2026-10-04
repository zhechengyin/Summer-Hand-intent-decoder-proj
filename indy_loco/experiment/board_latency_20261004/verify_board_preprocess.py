"""Compare a pre-STOP final-window snapshot with host preprocessing (not yet captured)."""
import json
import struct
from pathlib import Path
import numpy as np
from .train import frozen
from .deployed import preprocessing,FIRMWARE

variant=FIRMWARE/'output/optimized_b2_20261004/fallback_w96_d3_lut'
offsets=json.loads((variant/'preprocess_offsets.json').read_text())
raw=(variant/'memory_readback/preprocess.bin').read_bytes()
assert len(raw)==offsets['size']
assert struct.unpack_from('<I',raw,8)[0]==10500, 'A pre-STOP capture is required; the standard replay runner clears preprocessing when it stops'
frozen.protocol.GUI_ROOT=Path.home()/'Documents/STM32/BCI-STM32-Plot/data/ai_device_sessions'
data,_=frozen.prepare('indy_20160622_01',1)
features,_,receipt=preprocessing(data)
actual={name:np.frombuffer(raw,dtype='<f4',count=9600 if name=='window' else 192,offset=offsets[name])
        for name in ['mean','std','window']}
expected=dict(mean=np.asarray(receipt['mean'],np.float32),std=np.asarray(receipt['std'],np.float32),
              window=np.ascontiguousarray(features[:,-50:]).reshape(-1))
report={name:dict(bit_exact=bool(np.array_equal(actual[name],expected[name])),
                  max_abs=float(np.max(np.abs(actual[name]-expected[name])))) for name in actual}
(variant/'preprocess_readback_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
assert all(r['bit_exact'] for r in report.values())
