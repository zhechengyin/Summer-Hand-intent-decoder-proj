# Project Summary

Last updated: 2026-07-06

## Dataset And Task

- Dataset: OpenNeuro ds004022 multimodal EEG + fNIRS motor-imagery data.
- Task: 4-class same-limb hand-intent decoding: Reach, Grasp, Lift, Twist.
- Chance level: 0.25 because the task is balanced across 4 classes.
- Main evaluation protocols: subject-specific cross-validation and leave-one-run-out.
- Current state: real-data decoding is near chance. Best evidence so far is a
  small Riemannian benefit, not a solved decoder.
- Positive-control dataset: OpenNeuro ds004362 (PhysioNet eegmmidb) left/right
  hand MI (chance 0.50). Our Riemannian tangent front end hits 0.71 LORO there
  (LOG-011), confirming the pipeline extracts real MI signal when a decodable
  contrast exists -- ds004022 near-chance is task difficulty, not a bug. Probe:
  `tools/eegmmidb_probe.py`.

## Standing Modeling Policy

- Official N1 model paths must use fused EEG + fNIRS data.
- EEG-only or fNIRS-only runs are allowed only as clearly labeled diagnostic
  ablations. They are not official N1 results.
- Do not use LOSO anymore. Use subject-specific CV and leave-one-run-out only.
- Default model path: EEG bandpower + fNIRS hemodynamic features -> StandardScaler -> classifier.
- Advanced path 1: EEG FBCSP front end + aligned fNIRS features -> StandardScaler -> classifier.
- Advanced path 2: EEG Riemannian tangent-space front end + aligned fNIRS features -> StandardScaler -> classifier.
- Neural options: small GELU MLP, fused temporal CNN, bare fused SNN, or
  Riemannian-SNN only when requested. Neural models use AdamW. Classical models
  do not use Adam.
- Reports must stay honest. Do not inflate near-chance results.

## Current Result Ledger

Official fused EEG + fNIRS task results:

| Model | Subject-Specific Accuracy | Leave-One-Run-Out Accuracy | Current Read |
| --- | ---: | ---: | --- |
| Fused EEG bandpower + fNIRS hemodynamic + LDA | 0.2404 | 0.2393 | below/near chance |
| Fused EEG FBCSP + fNIRS + LDA | 0.2201 | 0.2357 | did not help |
| Fused EEG Riemannian tangent space + fNIRS + Logistic Regression | 0.2476 | 0.2715 | best LORO so far, modest |
| Fused GELU MLP | not clearly better than Riemannian | around 0.27 | no clear win |
| Fused raw-EEG temporal CNN + fNIRS dense branch | 0.2548 | 0.2596 | small, not decisive |
| Fused 10-sample moving-average EEG temporal CNN + fNIRS dense branch | 0.2559 | 0.2608 | tiny boost over raw CNN |
| Fused 50-sample moving-average EEG temporal CNN + fNIRS dense branch | 0.2631 | 0.2667 | best CNN variant, still near chance |
| Fused symbolic ERD/ERS timing features + fNIRS + LDA | 0.2548 | 0.2655 | modest, more class-balanced |
| Bare fused SNN + fNIRS | 0.2535 | 0.2630 | no clear win |
| Static Riemannian-SNN + fNIRS | 0.2512 | 0.2440 | worse than Riemannian LogReg |
| Windowed Riemannian-SNN + fNIRS | 0.2428 | 0.2465 | worse than Riemannian LogReg |
| Connectivity PLV+imcoh+wPLI (all ch) + fNIRS + LogReg | 0.2596 | **0.2965** | best LORO to date (LOG-009, replicated LOG-010) |
| Connectivity -> PLS-DA(8) -> LDA + fNIRS | 0.2715 | 0.2954 | model of record: ~same LORO, better subj-CV, 8 dims (LOG-010) |
| Connectivity -> shrinkage-LDA + fNIRS | 0.2668 | 0.2870 | simpler regularized alternative (LOG-010) |
| Multiview: Riemannian + connectivity + fNIRS + LogReg | 0.2656 | 0.2906 | early concat DILUTES; late fusion also does not beat connectivity |
| Connectivity wPLI (all ch) + fNIRS + LogReg | 0.2715 | 0.2762 | lower-dim connectivity corroboration |
| Riemannian tangent (Ledoit-Wolf) + fNIRS + LogReg | 0.2572 | 0.2763 | best fixed shrinkage family |
| Riemannian tangent + fNIRS -> PLS-DA(2) -> LDA | 0.2346 | 0.2799 | best pure-tangent LORO, run-stable |

Diagnostic EEG-only ablation, not an official N1 path:

| Model | Subject-Specific Accuracy | Leave-One-Run-Out Accuracy | Fused Comparator | Current Read |
| --- | ---: | ---: | --- | --- |
| EEG-only bandpower + LDA | 0.2429 | 0.2201 | fused 0.2404 / 0.2393 | fNIRS helps LORO here |
| EEG-only Riemannian tangent space + Logistic Regression | 0.2668 | 0.2715 | fused 0.2476 / 0.2715 | fNIRS hurts subject-CV, not LORO |
| EEG-only symbolic ERD/ERS + LDA | 0.2488 | 0.2655 | fused 0.2548 / 0.2655 | fNIRS helps subject-CV slightly |

Geometry and diagnostic probes:

| Probe | Result | Current Read |
| --- | --- | --- |
| K-means on EEG tangent space | best k 7, silhouette 0.2795, class NMI 0.0001, subject NMI 0.9929 | clusters are subject identity, not action class |
| K-means on EEG tangent space + fNIRS | best k 2, silhouette 0.9127, class NMI 0.0024, subject NMI 0.0024 | strong non-class structure, not four actions |
| Subject-ID from EEG tangent space -> NN | subject-CV 1.0000, LORO 0.9990 | Riemannian features identify people extremely well |
| Subject-ID from EEG tangent space + fNIRS -> NN | subject-CV 0.9950, LORO 0.9870 | fused features also identify people strongly |
| Cross-subject LOSO plain MLP (tangent+fNIRS) | pooled acc 0.2691 | cross-subject transfer ~= within-subject LORO |
| Cross-subject LOSO subject-adversarial MLP | pooled acc 0.2608 | gradient-reversal suppression did NOT help |

## Architecture Benchmark (classical vs deep)

Compact EEGNet-style CNN and an EEG Conformer transformer were benchmarked
against our best front ends under the identical CV (LOG-012). On BOTH datasets
the deep nets FAILED to beat the classical/geometric models, and the transformer
was weakest:

- ds004362 left/right MI (chance 0.50): Riemannian 0.707 > connectivity 0.573 >
  CNN 0.551 > Transformer 0.507 (LORO).
- ds004022 same-limb (chance 0.25): connectivity->PLS 0.295 > Riemannian 0.272 >
  CNN 0.258 > Transformer 0.232 (LORO).

Reason: transformer/CNN SOTA numbers (e.g. Conformer ~78% on BCI IV-2a) rely on
~10x more trials/subject + augmentation. At this data scale the compact
classical/geometric pipeline is the correct tool. Bench code:
`src/conformer.py`, `tools/architecture_bench.py`.

## Working Interpretation

- The task remains near chance. Current models are not reliable 4-class
  same-limb intent decoders.
- Functional connectivity (PLV / imaginary coherence / wPLI) is now the most
  useful EEG feature family: all-channel connectivity + fNIRS reaches the best
  LORO to date (0.2965, class-balanced). Riemannian tangent-space and symbolic
  ERD/ERS remain useful but weaker. Supervised PLS latent projection and
  Ledoit-Wolf shrinkage give small tangent LORO gains.
- The main bottleneck appears to be class separability and signal quality, not
  lack of model capacity. New evidence: subject-adversarial suppression did not
  help, and cross-subject transfer already matches within-subject LORO, so
  subject-nuisance variance is NOT the primary limiter -- class separability is.
- Riemannian features separate subjects extremely well. This may be useful for
  future subject-profile/adaptive BCI ideas, but it does not solve
  Reach/Grasp/Lift/Twist decoding.
- Leave-one-run-out remains useful because it tests whether a subject-specific
  model survives run-level drift. It is not the same as real-time deployment,
  but it is stricter than random within-run splits.
- fNIRS does not globally damage accuracy. It appears harmful for Riemannian
  subject-CV, helpful for bandpower LORO, mildly helpful for symbolic ERD/ERS
  subject-CV, and mostly irrelevant for the best Riemannian LORO score.

## Next Engineering Steps

0. NEW (LOG-009/010): functional connectivity is the best lead. Model of
   record = connectivity -> PLS-DA(8..16) -> LDA + fNIRS (robust, 8-dim, LORO
   ~0.295 with subj-CV lifted to ~0.272).
   - sweep bands (mu / low-beta / high-beta / broadband) and metric subsets
   - tune PLS components and try shrinkage-LDA head
   - DONE / dead ends: combining connectivity with the Riemannian tangent front
     end (early concat dilutes, late fusion does not beat connectivity alone);
     subject-adversarial suppression (negative). Do not revisit these.
1. Reproduce the fused baseline before making new claims.
2. Tune the Riemannian pipeline:
   - covariance shrinkage/regularization
   - EEG frequency window
   - classifier choice: Logistic Regression, SVM, LDA
3. Tune symbolic ERD/ERS:
   - moving-average window sweep around 5, 10, 20, 50 samples
   - ERD/ERS thresholds
   - window length and overlap
   - combine symbolic ERD/ERS features with Riemannian features
4. Test fusion controls:
   - fNIRS feature selection
   - late fusion / weighted fusion
   - regularized models that can suppress fNIRS when unhelpful
5. Consider imagery-vs-rest only as a sanity check. The actual target remains
   4-class Reach/Grasp/Lift/Twist.

## Pointers

- Detailed daily notes live in [DAILY_LOG.md](DAILY_LOG.md).
- Root index: [../PROJECT_MEMORY.md](../PROJECT_MEMORY.md).
