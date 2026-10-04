"""Report all 30 frozen deployed-arithmetic tests against paired Midsize folds."""
import csv
import hashlib
import json
import numpy as np
from .train import OUTPUT, frozen

winner_path=OUTPUT/'winner.json'
winner=json.loads(winner_path.read_text())
signature=hashlib.sha256(winner_path.read_bytes()).hexdigest()
rows=[]
for session in frozen.SESSIONS:
    for fold in range(1,6):
        path=OUTPUT/winner['architecture']/f'{session}_fold{fold}_export/deployed_evaluation.json'
        receipt=json.loads(path.read_text())
        assert receipt['winner_sha256']==signature and receipt['test_used_for_selection'] is False
        candidate,baseline=receipt['splits']['test'],receipt['midsize_test']
        assert candidate['bins']==baseline['bins']
        row=dict(session=session,fold=fold,bins=candidate['bins'])
        for axis in ['x','y','mean']:
            key=f'r2_{axis}'
            row[f'candidate_{key}']=candidate[key]
            row[f'midsize_{key}']=baseline[key]
            row[f'delta_{key}']=candidate[key]-baseline[key]
        rows.append(row)
summary={}
for column in [k for k in rows[0] if 'r2_' in k]:
    values=np.array([r[column] for r in rows])
    summary[column]=dict(mean=float(values.mean()),std=float(values.std(ddof=1)),
                         minimum=float(values.min()),maximum=float(values.max()))
baseline_mean=summary['midsize_r2_mean']['mean']
assert abs(baseline_mean-0.74113758)<1e-6,baseline_mean
report=dict(architecture=winner['architecture'],folds=30,winner_sha256=signature,
            accuracy_pass=summary['candidate_r2_mean']['mean']>0.74113758,
            acceptance_threshold=0.74113758,summary=summary,
            winning_folds=sum(r['delta_r2_mean']>0 for r in rows),
            sessions={s:dict(candidate_mean=float(np.mean([r['candidate_r2_mean'] for r in rows if r['session']==s])),
                             midsize_mean=float(np.mean([r['midsize_r2_mean'] for r in rows if r['session']==s])),
                             paired_delta=float(np.mean([r['delta_r2_mean'] for r in rows if r['session']==s]))) for s in frozen.SESSIONS},
            rows=rows,test_used_for_selection=False,
            evidence='Generated C runtime with firmware preprocessing/FMA and INT8 ties-to-even rounding; physical board validation is a separate single-fold replay')
(OUTPUT/'final_30fold.json').write_text(json.dumps(report,indent=2))
with (OUTPUT/'final_30fold.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
