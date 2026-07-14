# Project Summary

Last updated: 2026-07-14

## CURRENT FOCUS: 8-channel STM32 monkey velocity decoder (LOG-042..073)

Hardware pivot (LOG-050): decode NHP finger velocity from **8 channels** (spike
detection) on an **STM32-class MCU**, in **real time**; metric is **R²**. Data:
indy sessions, fixed file split (train.../eval1/test1), per-electrode 40 ms
multiunit rates, 3 Hz vel-LP target. Reference: Zhou/Sun/Basu arXiv:2312.15889.

- **Deployable model of record: STRICTLY-CAUSAL wide TCN+GRU + MULTISCALE input**
  (`F64/H64/L1`, `bidir=False`; input = 8 channels × {raw, EWMA α} = 16
  features), firing top-8 on train1-6, 24 sessions → **TEST R² ≈ 0.63**
  (eval-valid), ~74 kB int8 (lossless), ~5.6 ms/pred.
  `models/tcn_gru_8ch/` (checkpoint is causal+multiscale, α=0.2). Multi-timescale
  input is the one model-side lever that beat the 0.606 ceiling (real on BOTH eval
  and test, LOG-068). ⚠️ **The earlier 0.646 headline was TEST-SELECTED** (α tuned
  on test1 — leakage; eval-valid α=0.1 → test 0.630; LOG-073). Alpha barely matters
  (within noise). ⚠️ **test1 is no longer untouched** (~25 experiments have read it);
  an unbiased final headline needs a freshly reserved session, pipeline frozen,
  scored once. Auxiliary heads (multi-task, LFADS-lite) don't help (LOG-072).
- ⚠️ **NOT ACTUALLY ZERO-LOOKAHEAD YET (LOG-074).** The cached input uses scipy
  `gaussian_filter1d(σ=1)`, which is CENTERED — it pulls ~30% of its weight from the
  future (~160 ms). So every "0 ms lookahead" number above silently used future input.
  Good news: it's ~free to fix — a strictly-causal input (**counts + causal EWMA** =
  0.636 EVAL / 0.630 TEST) MATCHES the leaky pipeline (0.638/0.626). The causal EWMA
  is the whole lever; the Gaussian (and its leak) adds ~nothing. **Adoption pending**:
  swap the centered Gaussian for a causal smoother in `models/tcn_gru/evaluate.py`,
  regenerate the 40 ms cache, retrain — then "0 ms lookahead" becomes true at ~0.63.
- **Bidirectional is NOT deployable** (needs the whole future). Bidir 0.677 and
  bounded-lookahead (0.619 @80ms, 0.623 @200ms) are OFFLINE references only —
  at 40 ms/bin that latency is too high for closed-loop (LOG-062/065).
- **int8 quantization is lossless**: no R² loss; fits STM32 F1/F4/H7 flash.
- **What moved R² (from 0.529 at 6 sessions):** (1) MORE TRAINING DATA is the main
  lever (6→9→12→18→24 sess = 0.529→…→0.655, LOG-052/053/055); (2) with 24 sessions
  a BIGGER model finally helps (wide 0.677 vs small 0.655, LOG-056) — it overfit at
  6 sessions. A single wide model matches a 3-seed ensemble (0.675).
- **What did NOT help:** the paper's Bessel output filter (redundant — we low-pass
  the velocity target at 3 Hz already, LOG-050); correlation loss / stronger aug /
  more reg (wash, LOG-051); Bessel/causal caveat below. Only seed-ensembling gave
  +0.022 (3× cost).
- **Pitfalls confirmed:** re-selecting the 8 channels on more data OVERFITS
  (0.628→0.502); learned/corr/low-freq/fft selection all lose or tie firing-rate
  top-8 on base-6 (LOG-043/046/053/063). Causal architecture/capacity/smoothing/
  distant-data tweaks don't beat causal 0.606 either (LOG-064/065) — the decoder is
  near its ceiling for this 8-channel input; more R² needs data/hardware, not model.

Older EEG/fNIRS and WAY-EEG-GAL work below is now legacy (see `legacy/`).

## Dataset And Task

- Dataset: OpenNeuro ds004022 multimodal EEG + fNIRS motor-imagery data.
- Task: 4-class same-limb hand-intent decoding: Reach, Grasp, Lift, Twist.
- Chance level: 0.25 because the task is balanced across 4 classes.
- Main evaluation protocols: subject-specific cross-validation and leave-one-run-out.
- Current state: real-data decoding is near chance. Best evidence so far is a
  small Riemannian benefit, not a solved decoder.
- Positive-control dataset: OpenNeuro ds004362 (PhysioNet eegmmidb) left/right
  hand MI (chance 0.50). Our Riemannian tangent front end hits 0.64 LORO there
  on 10 subjects (LOG-011/013), confirming the pipeline extracts real MI signal
  when a decodable contrast exists -- ds004022 near-chance is task difficulty,
  not a bug. Probe: `legacy/tools/eegmmidb_probe.py`.

## CURRENT BEST RESULT: EEG -> Finger/Hand Velocity (WAY-EEG-GAL)

- Dataset: WAY-EEG-GAL grasp-and-lift (32-ch EEG @ 500 Hz, figshare 988376),
  participants P1-P3. Task: continuous regression of 3D hand/finger marker
  velocity from EEG (not classification). Metric: Pearson r (pred vs true).
- BEST model: **BIGP seq2seq TCN+GRU** -- **3-subject mean r = 0.853**
  (P1 0.889, P2 0.813, P3 0.858; per-axis up to 0.926). Strong result, well above
  typical published EEG-kinematics numbers (0.3-0.5). See LOG-020/021/022/024.
- Best config of record: EEG low-pass 2 Hz -> 25 Hz, crop to movement window
  [1.5, 7.0] s, TCN dilations 1/2/4/8/16/32 + bidirectional GRU (hidden 64, 2
  layers) + F=64, data augmentation (Gaussian noise 0.1 + channel dropout 0.1),
  cosine LR, AdamW, 150 epochs, 3-fold over series. Marker 4 (best hand/finger
  sensor). Code: `legacy/tools/way_gal_kin_research.py --stage final_improved` (EEG, legacy).
- Model size: **201,155 trainable params** (~0.80 MB fp32) -- fits a <1 MB
  inference device. (Prior BIG: 188,803 params, mean r 0.843.)
- What drove it (+0.18 over the 0.664 baseline): (1) cropping to the movement
  window removes rest-period dilution; (2) larger temporal context + bigger GRU.
  lp=2 ~ lp=4 Hz; cross-subject pooling is a net wash. TCN+GRU beats lagged-
  linear (0.442) and sliding-window Riemannian tangent (0.359) -- tangent
  geometry is a classification tool, not for continuous regression.
- Cross-modality transfer (LOG-025): the SAME TCN+GRU (0.81 MB) decodes finger
  velocity from INTRACORTICAL primate spikes (Zenodo 3854034, indy session) at
  r 0.848 vs linear 0.731 -- architecture is modality-general (EEG voltage or
  binned spike rates, channels-x-time). Historical tool: `legacy/monkey_trials/velocity.py`.
- Axis/bin config (LOG-040/041): tested 3D velocity (held-out 3D mean r 0.731 --
  the two movement axes ~0.84-0.88, depth axis ~0.47 as the reach is ~planar) then
  REVERTED to 2D per user (movement axes only, held-out ~0.87). Bins switched
  50 ms -> 40 ms (40 ms ~= 50 ms, within noise). Current config: 2D velocity,
  40 ms bins.
- Cross-SESSION held-out generalisation (LOG-026/033): per-electrode features
  (96 consistent channels), train on 6 indy sessions, TEST on 2 sessions never in
  training. Tuned config (3 Hz velocity-target low-pass + sigma=1 firing-rate
  smoothing, LOG-030/032) -> held-out mean r **0.870** (0.878, 0.863), up from
  0.856. 0.77 MB TCN+GRU. Real-time inference ~6 ms/pred bidirectional (9x margin
  @50 ms) or ~3.7 ms causal; causal real-time costs only ~0.005 (LOG-031/032).
  Historical result; current evaluation tool: `models/tcn_gru/evaluate.py`. (Sorted units vary per session;
  per-electrode does not -- that is why pooling works. Does NOT transfer across
  subjects, LOG-027.) Activation set to ReLU (LOG-037/038; within-noise nominal
  best, not held-out re-validated). EEG pipeline keeps GELU default.
- HARDWARE CONSTRAINT -- 8 channels (LOG-042): device reads only 8 electrodes.
  Electrode-count sweep (top-N by firing rate): 8ch=0.760, 16ch=0.804, 32ch=0.841,
  96ch=0.869. 8 channels is viable (~0.76) but ~0.11 below full array; steep early
  slope means *which* 8 matters -> channel selection has leverage. NOTE: dataset
  has NO continuous broadband voltage (only spike times + 64-sample waveform
  snippets), so "raw voltage -> model" (Option 2) is not benchmarkable here; only
  peak-detection (Option 1) is. Historical tools: `legacy/monkey_trials/nch.py`,
  `legacy/monkey_trials/chan_select.py`.
- CHANNEL SELECTION -- which 8? (LOG-043): random8=0.690, firing8=0.760 (best),
  learned8 (L1 stochastic-gate)=0.706. Learned selection OVERFITS to the 6 train
  sessions (2/8 overlap with firing8, higher variance) and loses to the robust
  firing-rate heuristic. Decision: 8-ch device = top-8 by firing rate. Next:
  adaptive/switching gate for non-stationarity (must resist the same overfit).
- REPO CLEANUP (LOG-044): the current model is isolated in `models/` (readable
  architecture in `best_model.py`,
  config, held-out evaluation, training entry point, and checkpoint). Old monkey
  sweeps moved to `legacy/monkey_trials/`; earlier EEG/fNIRS work remains legacy.
- DATA SPLIT (LOG-045): replaced the 6-train/2-test cross-session script with
  fixed `.mat` partitions named `train1`..`train6`, `eval1`, and `test1`.
  Eight files make the closest whole-file approximation to 70/15/15 equal to
  75/12.5/12.5. Validation selects the epoch; test is final-only.
- GATE RE-RUN (LOG-046): on the fixed split, random8 eval/test = 0.724/0.684,
  firing8 = **0.776/0.746**, learned8 = 0.775/0.653. The learned gate nearly
  matches firing-rate selection on validation but collapses on untouched test;
  firing-rate top-8 remains the deployment choice.
- OLD-REPO BASELINE (LOG-047): Neural-ML's training-only joint velocity-
  correlation selector, reduced to top-8, scored eval/test 0.712/0.593. It is a
  fixed selector (not live) and badly overfits. Static firing8 remains best.
  A real switching system needs a multiplexing scheduler plus a channel-ID-aware
  decoder; feeding arbitrary electrodes into eight fixed input positions is invalid.
- DATA CONVENTION (LOG-048): raw MAT files now live under
  `data/source_data/indy_loco/`. Each preprocessing method owns a folder under
  `data/processed/` with its script, README, and gitignored artifacts. Generated
  40 ms and 50 ms NPZ datasets preserve one output per recording. The current
  evaluator consumes `bin_40ms/artifacts` with raw-generation fallback.
- MODEL CONVENTION (LOG-049): each model family now owns a self-contained folder
  under `models/`. The model of record is `models/tcn_gru/`, containing its
  architecture, config, split metadata, evaluator, trainer, checkpoint, and README.
- Cross-SUBJECT limit (LOG-027): indy-trained model on a held-out indy day =
  r 0.864, but on loco (the OTHER monkey) = r -0.048 (collapse). Does NOT
  transfer across subjects -- different brains/electrodes have no channel
  correspondence; needs per-person calibration/alignment. Generalisation ladder:
  within-session ~0.85 | across-days same-subject 0.856 | across-subjects ~0.
- Artifact check (LOG-023): motor-channels-only BIG TCN+GRU still reaches mean
  r 0.823 (P1 0.851, P2 0.763, P3 0.855) -- only 0.02 below all-channel 0.843,
  vs the linear decoder which lost ~0.13. So the deep model decodes largely
  GENUINE motor-cortical velocity, not mostly artifact; the conservative cortical
  estimate is r 0.823.

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

- ds004362 left/right MI (chance 0.50, 10 subjects): Riemannian 0.642 > CNN
  0.569 > connectivity 0.558 > Transformer 0.556 (LORO).
- ds004022 same-limb (chance 0.25): connectivity->PLS 0.295 > Riemannian 0.272 >
  CNN 0.258 > Transformer 0.232 (LORO).

Reason: transformer/CNN SOTA numbers (e.g. Conformer ~78% on BCI IV-2a) rely on
~10x more trials/subject + augmentation. At this data scale the compact
classical/geometric pipeline is the correct tool. Bench code:
`legacy/src/conformer.py`, `legacy/tools/architecture_bench.py`.

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
