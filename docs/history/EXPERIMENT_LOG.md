# Experiment Log

## 2026-07-29 — Repository reset

The completed Indy Loco research program was retired from the active project
and preserved under `history/indy/`. Its status, experiment record, data,
checkpoints, code, and results remain available there.

No result from the Indy archive is an active baseline for the new EEG or
firmware task. No new experiment has been registered yet.

Future entries must record:

- task and label definition;
- dataset version and split;
- input representation available to firmware;
- model architecture and parameter count;
- seed and training configuration;
- validation and locked-test metrics;
- measured latency, peak RAM, and Flash when available;
- the decision made from the result.

## 2026-07-30 — Phase 1b FingerMovements baseline comparison

Phase 1b compared Feature + Linear, Tiny MLP, Tiny EEGNet, and Tiny Multi-scale
CNN on the 316 official FingerMovements training cases. Seeds 42, 43, and 44
each used stratified five-fold cross-validation, for 60 complete fits and 1,200
epoch records. Normalization was fitted from each fold's training subset only.
The 100-case official test file was not loaded.

The fixed training policy was 20 epochs, AdamW, learning rate 0.001, weight
decay 0.0001, dropout 0.25, batch size 32, no augmentation, and no checkpoint
selection.

| Model | Mean OOF accuracy | Seed SD | Parameters |
| --- | ---: | ---: | ---: |
| Feature + Linear | 58.65% | 1.56 pp | 394 |
| Tiny MLP | 57.59% | 1.14 pp | 9,554 |
| Tiny EEGNet | 56.96% | 0.84 pp | 1,050 |
| Tiny Multi-scale CNN | 57.17% | 2.38 pp | 6,434 |

Feature + Linear was frozen as the Phase-1b baseline because it had both the
highest mean accuracy and the fewest trainable parameters. Tiny EEGNet was
retained as the neural firmware candidate because it had the best seed
stability and only 1,050 parameters. The result does not unlock the official
test set and does not establish cross-session or cross-day generalization.

The full interpretation and learning diagnostics are recorded in
`results/finger_movements/phase1b_baseline_comparison/README.md`.
