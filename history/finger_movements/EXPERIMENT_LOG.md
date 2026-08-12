# Experiment Log

> Validity notice added 2026-08-10: entries through 2026-08-07 used a retired
> UEA conversion later shown to contain deterministic adjacent-channel
> overlap. Those entries remain intact as historical provenance, but their
> scores and checkpoints are invalid for comparison with the corrected direct
> conversion of the official MATLAB release. Corrected evidence begins with
> the entries dated 2026-08-10 below.

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

## 2026-08-04 — Phase 1e Logistic regularization

Phase 1e kept the 196-feature representation and evaluated Logistic
regularization without opening the official test. Seeds 42, 43, and 44 used
five stratified outer folds, and every learned preprocessing operation used
only the relevant training subset.

The broad nested-CV sweep selected different `C` values across outer folds and
reached 62.16% mean outer OOF balanced accuracy, below the fixed `C=1`
reference at 64.37%. A reporting-only `selected_count` aggregation bug was
corrected; it did not change training or predictions.

A fixed-C upper refinement compared `C=1, 1.25, 1.5, 2, 2.5, 3, 4, 5` on
identical folds. `C=1.5` ranked first at 64.47%, only 0.10 percentage points
above `C=1`; seed-level changes were inconsistent and no paired comparison was
significant. Values at or above `C=2` did not improve mean performance. Phase
1e retained `C=1`.

## 2026-08-04 — Phase 1f representation × classifier comparison

Phase 1f crossed four training-fold-only representations with Logistic
Regression and automatic-shrinkage Fisher/LDA:

| Representation + classifier | Features | Mean OOF BA | Seed SD | Worst seed |
|---|---:|---:|---:|---:|
| Terminal low-pass + Logistic | 252 | 68.89% | 0.92 pp | 68.35% |
| Current + terminal + Logistic | 448 | 68.16% | 0.64 pp | 67.43% |
| Terminal low-pass + Fisher | 252 | 68.04% | 1.45 pp | 66.78% |
| Current + terminal + Fisher | 448 | 67.20% | 1.02 pp | 66.47% |
| Current 196 + Logistic | 196 | 64.37% | 1.50 pp | 62.68% |
| Current 196 + Fisher | 196 | 61.43% | 1.89 pp | 59.54% |
| Fourier + PCA + Fisher | 20 | 58.02% | 0.18 pp | 57.92% |
| Fourier + PCA + Logistic | 20 | 57.92% | 0.63 pp | 57.28% |

Terminal Logistic improved all three seeds over the 196-feature Logistic
baseline. The accuracy gains were 3.48, 7.28, and 2.85 percentage points for
seeds 42, 43, and 44. Only seed 43 reached p<0.05 in its individual paired
comparison, so the result supports an engineering selection rather than a
strong independent-session significance claim.

Adding all 196 former features to the terminal representation did not improve
the mean or worst-seed result. Fourier + 20-component PCA was substantially
worse. Phase 1f therefore froze the second-order causal 5 Hz low-pass, 252
terminal features, Logistic Regression, and `C=1`. The official test remained
locked, and no final all-training-data checkpoint was created.

## 2026-08-04 — Phase 1g terminal feature contribution

Phase 1g decomposed the frozen 252-dimensional representation into A (five
terminal samples, 140 features), B (50/100/200 ms terminal means, 84
features), and C (200 ms terminal slope, 28 features). All eight subsets were
evaluated using the same seeds 42/43/44, stratified five-fold partitions,
training-fold-only preprocessing, and Logistic Regression with `C=1`. The
official test was not loaded.

The complete ABC representation remained best at 68.89% mean OOF balanced
accuracy. AC reached 68.25%, BC reached 67.30%, and AB reached 65.93%. Exact
Shapley contributions were 6.86, 6.01, and 6.02 balanced-accuracy percentage
points for A, B, and C, respectively. B had the smallest full-model marginal
contribution: removing it reduced mean balanced accuracy by 0.63 percentage
points. Because project priority is maximum accuracy rather than feature
reduction, the full ABC representation remained frozen.

## 2026-08-05 — Phase 1h final all-training-data fit

Phase 1h fitted the frozen ABC terminal Logistic Regression pipeline once on
all 316 official training cases. This was final checkpoint training, not a new
model-selection experiment. Liblinear used `C=1`, `max_iter=100000`, and
`tol=1e-10`. Convex optimization converged after eight solver iterations, so
the final converged solution was saved without epoch or early-stopping
selection.

The apparent training mean log loss was 0.447477, balanced accuracy was
78.49%, and the confusion matrix was `[[123, 36], [32, 125]]`. These are fit
diagnostics and are not held-out generalization estimates. The checkpoint was
reloaded and reproduced every training decision score and prediction exactly.

Checkpoint:

```text
models/finger_movements/terminal_logistic/checkpoints/finger_movements_terminal_logistic_phase1h.npz
```

At the 2026-08-05 direction closeout, the unchanged checkpoint was moved to:

```text
history/finger_movements/models/terminal_logistic/checkpoints/finger_movements_terminal_logistic_phase1h.npz
```

SHA-256:

```text
f8fca725c3b638219bbd734257cd958779e595add2fe1118e1e78689bc120047
```

The official 100-case test split remained locked during fitting and was not
loaded by the training script.

## 2026-08-05 — Phase 1h official-test inference and direction closeout

After the model, preprocessing, decision threshold, and checkpoint hash were
frozen, the official 100-case test was opened once for pure inference. The
evaluation script contained no fitting operation and loaded channel
normalization, feature normalization, filter coefficients, weights, and bias
unchanged from the verified checkpoint.

| Metric | Result |
|---|---:|
| Accuracy | 62.00% |
| Balanced accuracy | 62.10% |
| Macro-F1 | 61.94% |
| Left recall | 67.35% |
| Right recall | 56.86% |
| Mean log loss | 0.653730 |

The confusion matrix was `[[33, 16], [22, 29]]`. Official-test balanced
accuracy was 6.78 percentage points below the 68.89% development OOF mean. The
checkpoint remained unchanged after evaluation.

Decision: close and archive the terminal-feature Logistic direction. It
remains a reproducible above-chance baseline, but 62% official-test accuracy is
not sufficient for final firmware promotion. Further small changes to
Logistic regularization or training duration are not justified. The next
experiment should change the EEG representation and must select all design
choices using only the 316 official training cases. The already-opened test may
be reported only as a post-hoc comparison; it is no longer a pristine model
selection gate.

## 2026-08-05 — Phase A2 paper-style CSSD + hierarchical LDA

Phase A2 implemented the three branches described by Wang et al. for BCI
Competition 2003 Data Set IV: low-frequency BP-CSSD, 10--33 Hz ERD-CSSD, and a
19-channel BP trend representation. Each branch was reduced to one Fisher/LDA
score and a final LDA combined the three scores. The implementation used
fourth-order zero-phase Butterworth filters and a `1e-6` CSSD covariance ridge.
Every CSSD projection, scaler, and LDA was fitted from the current training
fold only. Official TEST was refused.

Seeds 42, 43, and 44 with stratified five-fold cross-validation produced
59.81% mean OOF balanced accuracy, 1.87-point seed SD, and 57.90% worst-seed
balanced accuracy. This was 9.07 points below the archived 68.89% terminal
Logistic OOF baseline.

The TRAIN-only diagnosis found the following outer-fold train-to-validation
AUC changes:

- BP CSSD: 61.77% to 52.59%;
- ERD CSSD: 83.18% to 55.79%;
- BP trend: 79.54% to 65.83%;
- final fusion: 88.69% to 63.94%.

Inner-OOF fusion improved the complete three-branch model from 59.81% to
61.08%, but BP trend alone remained better at 62.25%. Cross-fold CSSD subspace
similarity was particularly weak for the BP-right and ERD-right directions,
and some fold pairs were nearly orthogonal. The result identified unstable,
overfitted spatial filters rather than missing fold coverage or final fusion
as the primary limitation.

Decision: archive Phase A2 as the immutable reference for Phase 2b. Its source
is preserved at `experiments/phasea2_cssd_lda.py` and its complete metrics,
predictions, diagnostic tables, and figures at `results/phasea2_cssd_lda/`
inside this archive. Phase 2b must remain self-contained and must not import
from these archived files.

## 2026-08-07 — Phase 2b isolated CSSD stabilization

Phase 2b first tested each proposed CSSD remedy separately against the exact
Phase A2 reference. The experiment compared empirical covariance with a
stronger fixed ridge, Ledoit-Wolf, and OAS; trial trace normalization; F2
component counts of three, two, or one per class; and hard or soft branch
voting. Seeds 42, 43, and 44 used paired five-fold OOF evaluation on all 316
official TRAIN cases. Every learned operation was fitted within the current
outer-training fold, and official TEST was refused.

The unchanged Phase A2 reference reproduced at 59.81% mean OOF balanced
accuracy. Reducing F2 from three to one component per class was the strongest
isolated change: 63.41% mean OOF balanced accuracy, 1.04-point seed SD, 62.04%
worst-seed accuracy, and a +3.59-point gain with all three seeds improving.
Two F2 components reached 61.40%, Ledoit-Wolf reached 60.87%, trial
normalization reached 60.55%, OAS reached 60.45%, and soft voting was
effectively neutral at 59.83%. Hard voting and the stronger fixed ridge were
harmful.

Decision: retain one F2 component per class as the only strong candidate, but
do not promote a model until the individually positive techniques have been
tested in controlled combinations.

## 2026-08-07 — Phase 2b full combination ablation

The follow-up crossed all compatible levels associated with the isolated
above-baseline techniques: empirical/Ledoit-Wolf/OAS covariance, trial trace
normalization off/on, three/two/one F2 components per class, and LDA/soft-vote
fusion. The complete `3 x 2 x 3 x 2` factorial design contained 36
combinations, 108 combination-seed groups, and 34,128 paired OOF predictions.
Every group covered the same 316 TRAIN cases exactly once. The seven matching
single-factor cells reproduced the isolated experiment numerically, and
official TEST was not loaded.

The best setting was empirical covariance, no trial normalization, one F2
component per class, and LDA fusion. It reproduced the isolated winner at
63.41% mean OOF balanced accuracy, 1.04-point seed SD, and 62.04% worst-seed
accuracy. Adding OAS reduced the mean to 62.88%; adding Ledoit-Wolf reduced it
to 62.77%; the best soft-vote combination reached 61.83%.

Matched effects across the factorial grid were +0.25 points for Ledoit-Wolf,
+0.34 for OAS, -0.50 for trial normalization, +0.61 for two F2 components
versus three, +1.13 for one F2 component versus three, and -0.97 for soft
voting. The important interactions were negative: trial normalization reduced
the one-component F2 benefit by 1.61 points, while Ledoit-Wolf reduced it by
0.61 points.

Decision: keep only the one-component F2 reduction. Retain empirical
covariance, no trial normalization, and LDA fusion. No combination exceeds the
archived terminal-feature Logistic representation's 68.89% OOF result, so no
CSSD model or checkpoint is promoted.

## 2026-08-10 — Official MATLAB data correction and evidence invalidation

The raw FingerMovements release was reacquired from the official BCI
Competition II Data Set IV source. Direct inspection showed the MATLAB signal
layout is `time x channels x trials`. The former UEA conversion contained a
deterministic sliding overlap between adjacent channel dimensions and was
retired.

The supported converter now produces official TRAIN as `(316, 28, 50)` with
class counts left 159/right 157 and canonical source indices 0--315. It applies
no filtering, normalization, feature extraction, or split reassignment. The
processed TRAIN SHA-256 is:

```text
a2025f277b5351839554e0ecf3398f1f4fd5151a4fc90f0e25c873734f5a91d1
```

Consequences:

- all Phase 1b--1h development metrics from the UEA conversion are invalid;
- the old Phase 1h checkpoint and 62.10% official-test result are retained only
  as provenance;
- the initial 59.81% Phase A2 and 63.41% Phase 2b results are invalid;
- official TEST was already exposed and cannot become pristine again;
- all corrected model choices must be based on official TRAIN-only validation.

The pre-correction isolated Phase 2b output was moved to
`results/phase2b_cssd_stabilization_retired_uea/` inside this archive so its
validity cannot be confused with corrected evidence.

## 2026-08-10 — Corrected Phase A2 baseline

The paper-style BP-CSSD, ERD-CSSD, BP-trend, and hierarchical LDA pipeline was
rerun on the corrected official MATLAB TRAIN data. Seeds 42, 43, and 44 each
used five stratified folds. Every CSSD filter, scaler, branch LDA, and fusion
LDA was fitted using only its outer-training fold. Official TEST was refused.

The corrected Phase A2 result was 85.03% mean OOF balanced accuracy, 1.27
percentage-point seed standard deviation, and 83.25% worst-seed balanced
accuracy. This replaces 59.81% as the valid Phase A2 reference. Results are
archived under `results/phasea2_cssd_lda_official_matlab/`.

## 2026-08-10 — Corrected Phase 2b combination ablation

All 36 combinations of empirical/Ledoit-Wolf/OAS covariance, trial trace
normalization off/on, F2 component counts three/two/one per class, and
LDA/soft-vote fusion were evaluated on the corrected official MATLAB TRAIN
data. The protocol produced 34,128 paired OOF predictions across 108
variant-seed groups. Every group contained all 316 cases exactly once and
official TEST was not loaded.

The best configuration was:

```text
covariance=empirical
trial trace normalization=on
F2 components per class=1
fusion=LDA
```

It achieved 86.72% mean OOF balanced accuracy, 0.68 percentage-point seed
standard deviation, and 86.09% worst-seed balanced accuracy. The individual
seed balanced accuracies were 86.40%, 87.67%, and 86.09% for seeds 42, 43, and
44. It improved all three seeds over the corrected 85.03% Phase A2 reference.

Decision: promote this configuration as the active offline research model.
This corrected result supersedes the old invalid-source conclusion that trial
normalization should be disabled. Complete evidence is archived under
`results/phase2b_combination_ablation/`.

## 2026-08-10 — Corrected terminal-Logistic control

The frozen Phase 1 terminal representation was re-evaluated, without changing
features or `C`, on the corrected official MATLAB TRAIN data. The exact
pipeline remained a second-order causal 5 Hz low-pass, 252 terminal ABC
features, fold-training-only channel and feature normalization, and L2
Logistic Regression with `C=1`.

Across seeds 42/43/44 and five folds per seed it reached 78.58% mean OOF
balanced accuracy, 1.04 percentage-point seed standard deviation, and 77.22%
worst-seed balanced accuracy. This is 6.45 points below corrected Phase A2 and
8.14 points below the Phase 2b winner. Official TEST was refused.

Decision: retain the result only as the corrected simple linear control; do
not promote or retrain a Logistic checkpoint. Code and evidence are archived
as `evaluate_archived_terminal_logistic_phase1.py` and
`results/archived_terminal_logistic_official_matlab/`.

## 2026-08-10 — Phase 2b closeout and active checkpoint

A self-contained active model was created under
`models/finger_movements/cssd_lda/`; it imports no archived experiment code.
The frozen winner was fitted once on all 316 corrected official TRAIN cases.
The checkpoint contains temporal filter coefficients, channel order, BP/ERD
CSSD filters, all branch/fusion scaler parameters, LDA coefficients, and
training metadata.

Checkpoint:

```text
models/finger_movements/cssd_lda/checkpoints/finger_movements_cssd_lda_phase2b.npz
```

SHA-256:

```text
1e95b1ab5eaf7277cadd658578ef343f67923fc2b197aec8e1231735163bbfa2
```

The apparent all-TRAIN accuracy and balanced accuracy were 90.82% and 90.84%,
respectively; they are fit diagnostics, not generalization estimates. After
saving, the checkpoint reproduced predictions exactly with zero decision-score
and probability error. Official TEST was refused and not loaded.

Deployment decision: this is the active offline research checkpoint, not a
firmware checkpoint. Its fourth-order zero-phase filters use future samples
inside each 500 ms trial. A future phase must replace them with causal
preprocessing and repeat TRAIN-only validation before real-time deployment.

## 2026-08-10 — Phase 2c initial causal horizon diagnostic

This initial Phase 2c diagnostic kept the frozen 86.72% zero-phase model and
checkpoint unchanged. It tested a self-contained strictly causal version of
the same empirical covariance, trial trace normalization, one-component BP/F2,
and hierarchical LDA family.

The bin length was frozen at 50 ms, equal to five samples at 100 Hz. Ten
horizon-specific models produced predictions at 50, 100, ..., 500 ms. Each
horizon's causal temporal signals, CSSD filters, branch scalers/LDAs, and
fusion LDA used only samples available by that horizon. All learned quantities
were fitted from outer-training trials only; whole cases remained the fold
unit. Seeds 42/43/44 used five stratified folds and official TEST was refused.

Forbidden operations included `sosfiltfilt`, `filtfilt`, centered windows,
whole-trial inference normalization, and any feature ending after the current
horizon. Filtering used left-to-right `sosfilt` initialized from the first
current sample. As an executable causality audit, all future samples after
each horizon were replaced by extreme synthetic values for held-out cases.
Current decision scores and probabilities changed by exactly zero at every
horizon.

| Prediction horizon | Mean OOF BA | Seed SD | Worst seed |
|---:|---:|---:|---:|
| 50 ms | 50.62% | 2.99 pp | 46.50% |
| 100 ms | 51.27% | 1.02 pp | 50.01% |
| 150 ms | 55.16% | 1.06 pp | 53.76% |
| 200 ms | 56.33% | 0.26 pp | 56.02% |
| 250 ms | 62.97% | 0.90 pp | 61.70% |
| 300 ms | 68.97% | 0.94 pp | 67.70% |
| 350 ms | 72.89% | 1.98 pp | 70.25% |
| 400 ms | 77.22% | 1.86 pp | 74.68% |
| 450 ms | 79.65% | 0.79 pp | 78.80% |
| 500 ms | 82.93% | 1.03 pp | 81.67% |

The 500 ms seed results were 84.20%, 82.92%, and 81.67%. Relative to the
paired zero-phase model, the causal deltas were -2.20, -4.75, and -4.42 points
for seeds 42, 43, and 44. Exact paired tests were not significant for seed 42
(`p=0.371`) and favored the zero-phase model for seeds 43 and 44 (`p=0.0315`
and `p=0.0336`). The mean causal cost was 3.79 points.

Decision: strict causality is feasible with a moderate rather than
catastrophic performance cost. Do not overwrite the offline baseline or its
checkpoint yet. The next decision is the required prediction horizon: 500 ms
maximizes observed accuracy, while 400/450 ms provide lower latency at 77.22%
and 79.65% mean BA. After that requirement is frozen, train and verify only the
chosen causal all-TRAIN checkpoint.

## 2026-08-10 — Phase 2c past-only rolling streaming correction

Phase 2c corrected the deployment interpretation of the 500 ms endpoint. The
prediction point is A and the model input is `[A-500 ms, A]`; no sample after A
is consumed. Thus 500 ms is historical context rather than a delay after A.
The design receives five samples per 50 ms update, carries causal SOS filter
state between bins, and maintains a 500 ms filtered ring buffer. Ten bins are
needed only for the first startup warm-up; the intended steady-state output
interval is then 50 ms.

The same seeds 42/43/44 and five fold partitions were used. All CSSD filters,
branch models, and fusion models were fitted from outer-training trials only.
Official TEST was refused. The endpoint OOF predictions and probabilities
exactly reproduced the initial Phase 2c 500 ms horizon endpoint:

| Metric | Phase 2c |
|---|---:|
| Mean OOF balanced accuracy | 82.93% |
| Seed SD | 1.03 pp |
| Worst-seed balanced accuracy | 81.67% |
| Delta versus zero-phase reference | -3.79 pp |
| Delta versus initial Phase 2c horizon endpoint | 0.00 pp |

Two implementation checks ran on all 64 validation cases in the first fold.
Processing the same history as ten stateful 50 ms calls exactly reproduced a
single full causal call: filtered samples, decision scores, probabilities, and
predictions all had zero error. The rolling implementation emitted first after
ten startup bins and emitted again after only one additional 50 ms bin.
Appending ten extreme synthetic samples after A also caused exactly zero
change to inference at A.

Decision: Phase 2c is the correct timing and streaming specification. It does
not create or replace a checkpoint yet. The official dataset contains isolated
500 ms epochs, so it cannot test persistent filter state across successive
overlapping windows or always-on behavior. Continuous recordings, including a
rest/no-intent condition, are required before claiming a complete real-time
decoder.

## 2026-08-10 — Phase 2c causal checkpoint promotion and sweep handoff

The completed horizon diagnostic and rolling-streaming evaluation were
archived without changing their metrics. The Phase 2b zero-phase model,
training entry point, checkpoint, and metadata were moved together to
`history/finger_movements/models/cssd_lda_offline_phase2b/` so the 86.72%
offline reference remains reproducible.

The active model was replaced by a self-contained strictly causal Phase 2c
implementation. It retains the selected empirical covariance, per-trial trace
normalization, one BP/F2 pattern per class, and hierarchical LDA structure,
but uses left-to-right SOS filtering and exposes persistent IIR plus 500 ms
ring-buffer state. It was fitted once on all 316 official TRAIN cases after
the TRAIN-only 82.93% OOF result had been frozen. Official TEST was refused.

Active checkpoint:

```text
models/finger_movements/cssd_lda/checkpoints/finger_movements_cssd_lda_phase2c_causal.npz
```

SHA-256:

```text
d92c23f7e6f8722d568d1b31963eab1328d5367ba32764b676d1ae0d73aaefd4
```

The apparent all-TRAIN balanced accuracy was 88.30%; this is a fitting
diagnostic, not a generalization estimate. Save/reload predictions were exact.
Ten stateful 50 ms updates reproduced batch endpoint inference on all 316
cases, with maximum score error `7.11e-15` and probability error `1.11e-15`.

The Phase 2c sweep runner performed a predeclared 4 x 4 comparison:

- past-context windows: 200, 300, 400, and 500 ms;
- streaming bins: 10, 20, 50, and 100 ms;
- seeds 42/43/44, five whole-case folds, TRAIN only.

Window size changes the classifier input and is refitted independently. Bin
size only changes causal chunking for an identical endpoint, so the script
requires exact filtering equivalence and does not use duplicated accuracy to
claim that one bin is statistically better.

The complete sweep produced:

| Window | Mean OOF BA | Seed SD | Worst seed |
|---:|---:|---:|---:|
| 200 ms | 79.62% | 1.23 pp | 78.46% |
| 300 ms | 79.43% | 0.26 pp | 79.11% |
| **400 ms** | **83.99%** | **0.54 pp** | **83.25%** |
| 500 ms | 82.93% | 1.03 pp | 81.67% |

Before freezing, the implementation was tightened so every candidate BP ring
is re-referenced to its own oldest sample. This prevents a shorter candidate
from depending on the removed part of the 500 ms epoch through its baseline.
The corrected sweep selects 400 ms, improving mean BA by 1.05 points,
worst-seed BA by 1.57 points, and seed SD by 0.49 points relative to 500 ms.
Its seed BAs were 83.25%, 84.20%, and 84.52%. All four bin sizes had exactly
zero causal-filter discrepancy.

Decision: freeze 400 ms history and retain the 50 ms update baseline. The
selected all-TRAIN checkpoint was fitted, reloaded, and verified at:

```text
models/finger_movements/cssd_lda/checkpoints/finger_movements_cssd_lda_phase2c_causal_400ms.npz
```

SHA-256:

```text
87b84cc2c8baf9efdc1ccf37ad28f5f58ad13c4db2a8f8a273fe73fce9956101
```

Apparent all-TRAIN BA was 89.89% (fit diagnostic only). Reload predictions
were exact; ten 50 ms chunks reproduced batch inference with maximum score
error `3.20e-14` and probability error `2.11e-15`. Official TEST was refused.
The prior 500 ms causal model/checkpoint and the completed sweep were archived.

## 2026-08-11 — Phase 2d frozen corrected official-TEST inference

Phase 2d began only after the Phase 2c 400 ms model, 50 ms update interval, all
preprocessing, and all-TRAIN checkpoint were frozen. The corrected 100-case
official TEST was then opened for pure inference. The script verified
checkpoint SHA-256
`87b84cc2c8baf9efdc1ccf37ad28f5f58ad13c4db2a8f8a273fe73fce9956101`
before loading TEST. It performed no fitting, normalization estimation,
recalibration, threshold selection, or model choice.

| Metric | Result |
|---|---:|
| Accuracy | 77.00% |
| Balanced accuracy | 77.05% |
| Macro-F1 | 77.00% |
| Mean log loss | 0.4584 |
| Left recall | 79.59% |
| Right recall | 74.51% |
| Accuracy Wilson 95% CI | 67.85%--84.16% |

The confusion matrix was `[[39, 10], [13, 38]]` with rows=true and
columns=predicted in left/right order. Batch inference and ten stateful 50 ms
chunks produced identical labels; maximum score/probability discrepancies were
`9.77e-15` and `1.78e-15`.

The TEST BA is 6.94 points below the frozen 83.99% OOF estimate. The errors are
not caused by a one-class collapse. Because official TEST was exposed during
an earlier project phase, this is recorded as a retrospective benchmark rather
than a pristine blind test. Decision: do not use this result to retune the
checkpoint; independent future data is required for a new selection gate. The
Phase 2d script and result were then archived; the checkpoint remains named
and frozen as Phase 2c because no training or model selection occurred here.

## 2026-08-12 — Phase 2e lightweight CSSD/LDA regularization comparison

Phase 2e compared five lightweight alternatives against the frozen Phase 2c
400 ms causal model on the exact same TRAIN-only seeds 42/43/44 and five
deterministic stratified folds per seed. Official TEST was refused. All CSSD
covariances, spatial projections, scalers, and classifiers were learned inside
the current training fold.

The variants were current empirical CSSD + SVD LDA, OAS-regularized CSSD,
analytical shrinkage LDA, their combination, and a lightweight ToeplitzLDA.
The Toeplitz branches used channel-major block-Toeplitz within-class covariance
with lag-diagonal averaging and fold-training-only OAS shrinkage. The final
three-score fusion remained a shrinkage LDA because those scalar scores have no
temporal block structure.

| Variant | Mean OOF BA | Seed SD | Worst seed | Fold SD | Seed BA deltas vs baseline |
|---|---:|---:|---:|---:|---|
| Current CSSD + LDA | 83.99% | 0.54 pp | 83.25% | 3.91 pp | reference |
| Regularized CSSD | 84.10% | 0.60 pp | 83.25% | 4.24 pp | +0.01 / +0.32 / 0.00 pp |
| Shrinkage LDA | 83.66% | 0.54 pp | 82.92% | 4.96 pp | +0.94 / -0.33 / -1.59 pp |
| Regularized CSSD + shrinkage LDA | 83.55% | 0.45 pp | 82.91% | 4.75 pp | +0.62 / -0.33 / -1.60 pp |
| ToeplitzLDA | 84.50% | 0.90 pp | 83.23% | 4.19 pp | +1.89 / +0.94 / -1.28 pp |
| Baseline + Toeplitz nested fusion | 84.09% | 0.98 pp | 83.24% | 4.57 pp | 0.00 / +1.27 / -0.96 pp |

ToeplitzLDA corrected 15/53, 15/50, and 9/49 baseline errors in seeds 42,
43, and 44, respectively, so it passed the predeclared requirement of
correcting at least 10% in every seed. The resulting fusion was evaluated with
four-fold inner-OOF stacking inside each outer fold. It failed to convert the
error complementarity into a stable gain: only seed 43 improved, seed 44
decreased, and both seed and fold variability exceeded the baseline.

All 5,688 recorded OOF predictions were identical between the full float64
reference and float32 inference path. The five single models all retained 323
deployment floats, approximately 1.28 KB parameters and 12.19 KB estimated
working RAM. Fusion increased parameter storage to 1.99 KB.

Decision: retain the frozen Phase 2c empirical CSSD + SVD-LDA checkpoint.
Regularized CSSD's +0.11-point mean gain was too small and inconsistent;
shrinkage LDA and the combination reduced mean BA; ToeplitzLDA's +0.52-point
mean gain failed seed consistency, worst-seed, and variability criteria; and
nested fusion did not stabilize it. The JSON mean-first ranking of ToeplitzLDA
is exploratory and is not a checkpoint-promotion decision.
