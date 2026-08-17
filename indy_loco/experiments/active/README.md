# Active Indy Loco experiment

## Phase 5: 64-channel hyperparameter and detector-filter ablation

`phase5_64channel_detector_filtered_sweep.py` compares the same causal 64-channel
64/64 TCN+GRU under two policies:

- `baseline`: all 29 chronological training sessions;
- `detector_filtered`: 27 training sessions after excluding the two
  retrospective Phase-3c failures, `indy_20160630_01` and
  `indy_20161013_03`.

The experiment sweeps learning rate, weight decay and dropout at seed 43, then
checks the union of the two policy winners with seeds 42 and 44. Sampling stays
session-balanced, each policy fits channel selection and normalization from its
own allowed training sessions, December is inference-only validation, and
January is never loaded. Both policies draw the same number of windows per
epoch, so removing two sessions does not reduce the number of optimizer updates.

From the repository root, validate the protocol without loading arrays:

```bash
python indy_loco/experiments/active/phase5_64channel_detector_filtered_sweep.py \
  --validate-only
```

Run the complete CPU experiment:

```bash
python indy_loco/experiments/active/phase5_64channel_detector_filtered_sweep.py \
  --threads 4
```

If the process stops after one or more completed fits, resume it without
retraining completed configurations:

```bash
python indy_loco/experiments/active/phase5_64channel_detector_filtered_sweep.py \
  --threads 4 --resume
```

Outputs are written under
`../../results/phase5_64channel_detector_filtered_sweep/`. Experiment
checkpoints are not promoted automatically and the retained model checkpoints
under `../../models/indy_32ch/` are never modified.

This comparison is a retrospective exclusion ablation. It can test whether
removing the two known failures improves December validation performance, but
it cannot establish that the detector will identify future failures.
