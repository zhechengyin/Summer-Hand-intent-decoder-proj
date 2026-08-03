# Phase 1b FingerMovements baseline report

Date: 2026-07-30

## Technical summary

Phase 1b completed all 60 planned fits: four model families, three seeds, and
five stratified folds per seed. The official 100-case test split remained
locked and was not loaded.

Feature + Linear is the current baseline winner. It achieved the highest mean
out-of-fold accuracy, 58.65%, while using only 394 trainable parameters. It
therefore dominates the three neural baselines on the two Phase-1b criteria of
mean accuracy and parameter count.

The result is not strong enough to declare a final model. Feature + Linear led
Tiny MLP by only 1.05 percentage points, Tiny EEGNet by 1.69 points, and Tiny
Multi-scale CNN by 1.48 points. Different models won different seeds, and the
three-seed uncertainty intervals overlap substantially.

Tiny EEGNet remains the most useful neural candidate: its mean accuracy was
56.96%, it had the smallest seed standard deviation at 0.84 percentage points,
and it used only 1,050 parameters. Tiny MLP and Tiny Multi-scale CNN do not
currently justify their larger parameter counts.

## Cross-seed comparison

Accuracy is calculated from one out-of-fold prediction for each of the 316
official training cases within each seed. Standard deviation is across seeds
42, 43, and 44.

| Model | Mean accuracy | Seed SD | Worst seed | Macro F1 | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| Feature + Linear | **58.65%** | 1.56 pp | 57.59% | **58.58%** | **394** |
| Tiny MLP | 57.59% | 1.14 pp | 56.65% | 57.57% | 9,554 |
| Tiny EEGNet | 56.96% | **0.84 pp** | 56.01% | 56.85% | 1,050 |
| Tiny Multi-scale CNN | 57.17% | 2.38 pp | 54.43% | 56.90% | 6,434 |

The majority-class baseline is 50.32%. All twelve model-seed results were
descriptively above that level, but no model reached 61% mean accuracy.

Model ranking was not consistent:

- seed 42: Tiny MLP led at 58.86%;
- seed 43: Feature + Linear led at 60.44%;
- seed 44: Tiny Multi-scale CNN led at 58.54%.

Feature + Linear wins the aggregate comparison, not every individual repeat.

## Training behavior and fold sensitivity

| Model | Mean final train accuracy | Mean final validation accuracy | Generalization gap | Validation fold range |
| --- | ---: | ---: | ---: | ---: |
| Feature + Linear | 68.93% | 58.62% | 10.31 pp | 52.38–67.19% |
| Tiny MLP | 66.61% | 57.57% | 9.04 pp | 49.21–70.31% |
| Tiny EEGNet | 63.84% | 56.94% | **6.90 pp** | 46.03–67.19% |
| Tiny Multi-scale CNN | 70.81% | 57.18% | **13.63 pp** | 42.86–67.74% |

Feature + Linear continued to improve its mean validation loss through epoch
20, although its mean validation accuracy peaked at epoch 15 and changed by
only 0.21 points afterward.

Tiny MLP reached its lowest mean validation loss at epoch 9. Tiny EEGNet reached
its lowest mean validation loss at epoch 14 and remained comparatively stable.
Tiny Multi-scale CNN had its best mean validation accuracy and loss at epoch 14,
then its train accuracy continued rising while validation loss worsened. This is
the clearest overfitting pattern in the experiment.

The individual fold ranges are much wider than the cross-seed mean differences.
Each validation fold contains only 62–64 cases, so a few cases change a fold
accuracy by several percentage points. The seed-level out-of-fold result is
therefore more reliable than selecting a model from one strong fold.

## Experiment integrity

The saved outputs contain:

- 1,200 complete epoch records;
- 60 unique model-seed-fold results;
- 12 unique model-seed out-of-fold results;
- four cross-seed summaries.

The experiment used:

- only the 316-case official training file;
- stratified five-fold cross-validation;
- seeds 42, 43, and 44;
- fold-training-only normalization;
- AdamW with learning rate 0.001 and weight decay 0.0001;
- dropout 0.25, batch size 32, and 20 fixed epochs;
- no augmentation, early stopping, checkpoint selection, or test loading.

The fixed-final-epoch rule avoids selecting an epoch on the same outer
validation fold used for reporting.

## Limitations

This experiment measures sensitivity to random fold assignment and
initialization within one small dataset. It does not prove cross-subject,
cross-session, or cross-day generalization because FingerMovements contains one
subject and does not expose session identifiers.

The three seed-level observations are too few for a confident claim that the
1–1.7 point model differences reflect a reproducible architecture advantage.
Approximate 95% intervals for the mean accuracy overlap for all four models.

Trainable parameter count is not complete firmware cost. In particular,
Feature + Linear requires time-domain statistics and FFT-derived band powers;
their operations, buffers, and numerical implementation are not represented by
the 394 classifier parameters. Tiny EEGNet consumes raw normalized input and
may therefore remain competitive after full latency and RAM measurement.

Random stratified folds may also be optimistic if samples from the same
unavailable recording session appear in both fold training and validation
subsets.

## Decision and recommended next step

1. Freeze Feature + Linear as the Phase-1b accuracy/size baseline to beat.
2. Retain Tiny EEGNet as the primary neural firmware candidate.
3. Do not promote Tiny MLP or Tiny Multi-scale CNN from this experiment.
4. Keep the official test set locked.
5. Before broad architecture sweeping, run a small, predeclared Phase 1c study
   of preprocessing and spatial features. Include the current feature-linear
   pipeline, Tiny EEGNet, and one firmware-compatible spatial
   projection/classifier inspired by CSP or Fisher discrimination.
6. If Phase 1c selects epochs or hyperparameters, use an inner validation split
   inside each outer fold. Do not select them from the outer validation score.

The next gate is not deployment yet. The immediate question is whether a
leakage-safe, low-cost spatial representation can raise cross-validated
accuracy meaningfully above the current 58.65% baseline.
