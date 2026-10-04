"""Freeze the validation-qualified winner using the prescribed repeated board timings."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .train import OUTPUT, frozen
from .model import ARCHITECTURES


def main():
    destination = OUTPUT/'winner.json'
    assert not destination.exists(), 'Winner is immutable once frozen'
    baseline = json.loads((OUTPUT/'midsize_validation.json').read_text())
    threshold = float(np.mean([r['validation']['r2_mean'] for r in baseline['rows']]))
    logs = Path.home()/'Documents/STM32/BCI-STM32-Plot/data/board_b2_logs'
    candidates = []
    for architecture in ARCHITECTURES:
        validation, receipts, runs, pooled = [], {}, [], []
        for session in frozen.SESSIONS:
            path = OUTPUT/architecture/f'{session}_fold1_export/lut_validation.json'
            record = json.loads(path.read_text())
            assert record['passed'] and record['test_evaluated'] is False
            validation.append(dict(session=session, **record['validation']))
            receipts[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        reference = None
        for run in (1, 2, 3):
            directory = logs/f'fallback-{architecture.replace("_", "-")}-lut-1024-r{run}-20261004'
            status = json.loads((directory/'status.json').read_text())
            assert status['stage'] == 'complete' and status['received_predictions'] == 1024
            timing, predictions = {}, {}
            for line in (directory/'packets.jsonl').read_text().splitlines():
                packet = json.loads(line)
                if packet.get('stage') != 'replaying':
                    continue
                t = packet.get('telemetry', {})
                if t.get('completed_predictions', 0):
                    timing.setdefault(t['prediction_sequence'], t)
                p = packet.get('prediction')
                if p:
                    predictions[p['prediction_sequence']] = p
            assert len(predictions) == 1024 and len(timing) >= 1014
            assert reference is None or predictions == reference
            reference = predictions
            final = list(timing.values())[-1]
            assert all(final[k] == 0 for k in ['sample_loss_events', 'lost_raw_samples_total', 'inference_overruns', 'skipped_inference_jobs'])
            times = np.array([t['last_inference_us']/1000 for t in timing.values()])
            pooled.extend(times.tolist())
            runs.append(dict(path=str(directory), samples=len(times), predictions=len(predictions),
                             missing_timing_snapshots=len(predictions)-len(times),
                             first_ms=float(times[0]) if 1 in timing else None,
                             median_ms=float(np.median(times)), p95_ms=float(np.percentile(times,95)),
                             p99_ms=float(np.percentile(times,99)), max_ms=float(times.max())))
        score = float(np.mean([v['r2_mean'] for v in validation]))
        candidates.append(dict(architecture=architecture, validation=validation, validation_mean=score,
                               qualified=score > threshold, receipts=receipts, runs=runs,
                               pooled_p95_ms=float(np.percentile(pooled,95)),
                               pooled_median_ms=float(np.median(pooled))))
    qualified = [c for c in candidates if c['qualified']]
    assert qualified, 'Neither fallback qualifies; retain qualifying b2, do not promote'
    selected = min(qualified, key=lambda c: c['pooled_p95_ms'])
    winner = dict(architecture=selected['architecture'], qualified=True, runtime='offline_gelu_lut_v5',
                  test_used_for_selection=False, baseline_validation_mean=threshold,
                  rule='six-session fold-1 C validation mean above matched Midsize, then lowest pooled 3x1024 board p95',
                  candidates=candidates,
                  screen_config_sha256=hashlib.sha256((OUTPUT/'screen_config.json').read_bytes()).hexdigest())
    destination.write_text(json.dumps(winner,indent=2))
    print(json.dumps(winner,indent=2))


if __name__ == '__main__':
    main()
