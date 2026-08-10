# Phase 2c past-only streaming causal results

For every prediction at point A, Phase 2c consumes only the historical interval
`[A-500 ms, A]`. Input arrives as five-sample/50 ms bins with causal IIR state
carried across the ten bins. After one startup warm-up, the intended firmware
update interval is 50 ms; it does not wait 500 ms after every prediction point.

TRAIN-only result across seeds 42/43/44 and five folds per seed:

- mean OOF balanced accuracy: 82.93%;
- seed standard deviation: 1.03 percentage points;
- worst-seed balanced accuracy: 81.67%;
- delta from frozen zero-phase reference: -3.79 percentage points;
- delta from the initial Phase 2c 500 ms horizon endpoint: exactly 0.00 points;
- official TEST: refused and not loaded.

Safeguards passed on all 64 held-out cases in the first fold:

- ten stateful 50 ms filtering calls exactly reproduced one full causal call;
- scores, probabilities, and predictions were identical;
- the first output appeared after ten startup bins and the next output appeared
  immediately after one additional 50 ms bin;
- appending ten extreme samples after A changed inference at A by exactly zero.

Files:

- `phase2c_metrics.json`: timing contract, data hash, model settings, checks,
  seed results, and aggregate metrics;
- `phase2c_fold_results.csv`: all 15 fold results;
- `phase2c_seed_results.csv`: three complete OOF results;
- `phase2c_predictions.csv`: 948 per-case OOF predictions;
- `phase2c_streaming_checks.csv`: numerical streaming and causality checks;
- `phase2c_streaming_causal.png`: seed comparison figure.

Dataset limitation: the official release contains isolated 500 ms epochs, not
continuous EEG around successive prediction points. It validates the endpoint
classifier and bin-wise implementation but cannot validate persistent filter
state across real continuous windows. No checkpoint is created in Phase 2c.
