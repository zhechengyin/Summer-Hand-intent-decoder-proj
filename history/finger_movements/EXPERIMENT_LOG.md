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

The full interpretation and learning diagnostics are preserved in
`results/phase1b_baseline_comparison/README.md` within this archive.

## 2026-08-02 — Phase 1c representation comparison

Phase 1c compared the retained Feature + Linear and Tiny EEGNet pipelines with
a six-component regularized CSP + shrinkage LDA pipeline. Seeds 42, 43, and 44
used the same stratified five-fold partitions. Every learned preprocessing
operation was fitted from the current fold's training subset only. The
official test split was not loaded.

At 20 epochs, mean OOF accuracy was 58.65% for Feature + Linear, 56.96% for
Tiny EEGNet, and 54.11% for CSP + LDA. Per-case paired tests did not establish a
significant difference between Feature + Linear and Tiny EEGNet.

## 2026-08-03 — Phase 1c duration checks and final selection

The Tiny EEGNet duration check changed only training duration and extended each
fold fit to 60 epochs. Epoch 20 exactly reproduced the initial Phase 1c OOF
predictions for all three seeds. Among the registered milestones 20, 30, 40,
50, and 60, epoch 50 produced the highest mean OOF balanced accuracy at 59.18%.
Its seed standard deviation was 3.01 percentage points and its worst-seed
balanced accuracy was 56.03%.

Feature + Linear was then extended to 50 epochs with all other settings frozen.
Its epoch-20 predictions also reproduced exactly. At epoch 50 it achieved:

| Metric | Feature + Linear | Tiny EEGNet |
|---|---:|---:|
| Mean OOF accuracy | 60.02% | 59.18% |
| Mean OOF balanced accuracy | 60.05% | 59.18% |
| Seed SD, balanced accuracy | 0.36 pp | 3.01 pp |
| Worst-seed balanced accuracy | 59.84% | 56.03% |

Feature + Linear was selected because it had the better equal-duration mean,
substantially lower seed variability, stronger worst-seed behavior, and the
simpler inference pipeline. The per-seed paired comparisons were not
statistically significant, so the result is an engineering selection rather
than evidence of a large scientific performance difference.

Phase 1c froze Feature + Linear, 50 epochs, seed 42, AdamW, learning rate
0.001, weight decay 0.0001, dropout 0.25, and batch size 32. The official test
remained locked. No cross-validation checkpoint was promoted as a final model;
the final checkpoint must be trained once on all 316 official training cases.

## 2026-08-03 — Phase 1d data audit and classifier comparison

Phase 1d first audited the 316-case official training split without opening the
official test. Processed signals and labels matched the canonical TRAIN.ts
exactly. No non-finite values, zero-variance trials/channels, exact duplicate
trials, conflicting duplicate labels, or train/validation index/signal overlap
were found. A balanced 32-case subset reached 100% training accuracy. The
archive does not provide trial-level IDs for its three recording sessions, so
same-session mixing across random folds remains untestable.

The true-label L2 Logistic Regression control reached 64.37% mean OOF balanced
accuracy. Shuffled labels averaged 50.82% and ranged from 45.26% to 56.34%,
supporting the presence of real label-associated signal. The initially saved
empirical p-value compared a three-seed observed mean with individual-seed null
scores and is therefore not treated as a formal calibrated p-value.

Phase 1d then compared four classifiers on exactly the same 196 handcrafted
features and training-only preprocessing:

| Classifier | Mean OOF balanced accuracy | Seed SD | Worst seed |
|---|---:|---:|---:|
| L2 Logistic Regression | 64.37% | 1.50 pp | 62.68% |
| Linear SVM | 62.36% | 1.92 pp | 61.09% |
| Ridge Classifier | 61.94% | 2.05 pp | 59.83% |
| AdamW + dropout Linear | 60.05% | 0.36 pp | 59.84% |

All three seeds favored Logistic Regression over the AdamW baseline by 2.85 to
5.70 percentage points. Only one individual-seed paired comparison reached
p<0.05, so this is an engineering model-family decision rather than a strong
scientific significance claim.

The 196-feature L2 Logistic Regression pipeline replaced AdamW + dropout as the
active candidate. `C=1` is provisional: Phase 1e must use training-only nested
cross-validation to freeze regularization before final all-training-data
training or official-test evaluation. Phase 1d code, results, and the retired
AdamW implementation were archived after review.
