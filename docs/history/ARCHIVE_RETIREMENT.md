# Retired code and experiment register

Retired on 2026-07-17. This document preserves the purpose, outcome, and reason
for removal of code that used to live under `experiments/archive/`, `legacy/`,
the archived compatibility helpers, and the two pre-causal model folders.

The detailed chronological evidence remains in `EXPERIMENT_LOG.md`. Historical
numbers below are not valid current benchmarks: many used centered smoothing,
central differences, whole-session normalization, repeated inspection of
`test1`, bidirectional models, or a held-out split that did not match deployment.

## Indy numbered experiments

| Retired experiment | Direction tested | Historical result or decision | Why code was removed |
| --- | --- | --- | --- |
| iter1 | Correlation loss, ensemble, causal output EMA | Single-model tricks did not establish a durable gain; ensembling helped at 3x cost. | Superseded training harness and burned split. |
| iter2 | STM32-size sweep and Bessel output filtering | About 100 kB cost little accuracy; Bessel added no gain because the target was already low-pass filtered. | Old preprocessing/model path was not end-to-end causal. |
| iter3 | Loss, augmentation, regularization, ensemble | Loss/augmentation/regularization were wash or worse; 3-seed ensemble was about +0.02. | Conclusion is recorded; implementation was compatibility-only. |
| iter4 | Add nearby Indy sessions | 6→9→12 sessions improved historical test R² 0.529→0.589→0.616. | Old split and preprocessing invalidate reuse of absolute scores. |
| iter5 | Scale to 18 sessions and reselect channels | 18-session fixed channels reached 0.628; pool-wide reselection fell to 0.502. | Superseded by month-level evaluation and prefix-only selection. |
| iter6 | Post-training int8 simulation | Historical int8 weight quantization showed essentially no score loss and about 27 kB weights. | Not a real export or STM32 timing artifact. |
| iter7 | 24 sessions and seed ensemble | Historical 24-session single/ensemble R² 0.655/0.675. | `test1` was repeatedly reused and preprocessing leaked future data. |
| iter8 | Capacity sweep | Wide F64/H64 model beat smaller variants historically; gains plateaued around 100–220 kB. | Architecture conclusion is captured in current config. |
| iter9 | Within-session training only | Mean R² 0.559, below pooled training because one session supplied too little data. | Superseded by pool-pretrain plus calibration studies. |
| iter10 | Spend a larger flash budget | Capacity plateaued near 220 kB; larger models did not help. | Broad capacity sweep is no longer a current decision. |
| iter11 | 10/20/40/80 ms and overlapping bins | 40 ms was the historical best trade-off; overlapping integration did not win. | Current bin rule is versioned in supported code/config. |
| iter12 | Channel scoring diagnostics | Best-channel sets drifted; firing was more stable than velocity correlation. | Diagnostic conclusion retained; code depended on old loader. |
| iter13 | Decode with alternate channel selectors | Alternate selectors could look better on a 24-session selection pool but did not beat the deployment base-six firing set. | Selection protocol was later recognized as unstable/leaky. |
| iter14 | TCN, GRU, depthwise TCN, bidirectional/causal comparison | Causal TCN+GRU was the strongest deployable family; bidirectional was an offline reference. | Stable causal implementation now lives in `src/intent_decoder/model/`. |
| iter15 | 0/80/200 ms lookahead | Historical R² 0.606/0.619/0.623 versus 0.677 bidirectional; delay was not worth the small gain. | Future-input branches are prohibited in supported code. |
| iter16 | Base-six channel selectors | Firing top-8 won (0.655); low-frequency and FFT scores did not improve it. | Decision retained; old score cannot be a current benchmark. |
| iter17 | More receptive field/depth/width and output EMA | None beat the then-causal reference; EMA added lag and hurt. | Broad architecture branch is closed. |
| iter18 | Add temporally distant sessions | 28 sessions slightly hurt versus 24 (0.600 vs 0.606), consistent with drift. | Replaced by explicit month-level folds. |
| iter19 | Wiener, LSTM, CNN-GRU, Transformer alternatives | TCN+GRU remained best; Wiener was far worse and Transformer underperformed. | Direction is closed pending genuinely new evidence. |
| iter20 | Raw counts plus several causal EWMA scales | Small historical gain motivated the current counts+EWMA feature family. | Reimplemented in `src/intent_decoder/features/causal.py`. |
| iter21 | Multi-seed confirmation of multiscale input | Two/multiscale features retained a modest lead. | Result was later affected by test reuse; only the feature direction survives. |
| iter22 | Number of EWMA timescales | Extra slow scales did not justify added complexity; some sweep rows were incomplete. | Current candidate uses raw plus one EWMA. |
| iter23 | EWMA alpha sweep | Alpha differences were small; earlier headline had selected on test. | Alpha must be retuned only inside corrected validation folds. |
| iter24 | Auxiliary kinematic/reconstruction heads | Auxiliary heads did not improve the velocity decoder. | Dead-end direction, inference model unchanged. |
| iter25 | Audit centered Gaussian leakage | Proved the supposed zero-lookahead cache used future samples; unsmoothed counts plus causal EWMA matched it. | Durable fix moved to supported loader/features and causality tests. |
| iter26 | 8/16/32 channel count | 32 channels was the largest historical robustness lever; final JSON write was incomplete. | Candidate is now explicitly 32-channel and must be rerun causally. |
| iter27 | Frozen evaluation on older fresh sessions | 8ch/32ch fresh mean R² about 0.054/0.305, exposing severe long-gap drift. | Backward 3–6 month split was too harsh and normalization was later corrected. |
| iter28 | Affine, fine-tune, and scratch calibration | Half-session calibration rescued 8ch 0.020→0.389 and 32ch 0.253→0.584. | Half-session protocol and whole-session statistics are retired. |
| iter29 | Calibration-duration sweep | 60 s was minimum useful; affine saturated early, fine-tuning continued improving through minutes. | Historical curve used a different scoring protocol; only 60 s remains the fixed observation rule. |
| iter30 | AdaBN label-free calibration | Marginal gain only; it did not close drift. | Current LayerNorm model has no BatchNorm running-state adaptation. |
| iter31 | Per-session channel reselection | Reselection helped, but changed channel identities required retraining rather than naive fine-tuning. | Replaced by prefix-only channel rules; old half-session implementation removed. |
| iter32 | Forward one-month split | Historical 32ch zero-shot/calibrated R² about 0.585/0.695; one failure dominated the mean. | Replaced by nested leave-one-month-out evaluation. |
| iter33 | Drift proxies | Prediction variance and top-channel overlap correlated with failures. | Threshold was chosen after seeing the same sessions. |
| iter34 | Leave-one-month-out detector CV | Suggested typical zero-shot performance near 0.75 and a bimodal failure tail. | Outer-fold threshold leakage required a new nested implementation. |
| iter35 | CORAL, PCA-Procrustes, and SOBI alignment | All label-free alignment variants lost to doing nothing; direction closed. | Historical implementation used half/whole-session statistics. |
| iter36 | Reptile meta-initialization | As configured it was worse than standard pretraining; standard pool prior helped most at 60 s. | Undertrained negative, not worth retaining as active code. |
| iter37 | ReFIT pseudo-label calibration | Helped bad sessions and hurt good ones; historical result motivated gating, but used half-session/240 s variants and cleaner hand-driven cursor data. | Must be redesigned under the fixed 60 s nested protocol before reuse. |
| iter38 | Heavy channel-dropout robustness | Failed on drifted sessions because dropout models ablation, while drift often substitutes channel meaning. | Dead end; keep default modest dropout. |
| iter39 | Full masked-identity month folds | Run was stopped after two folds; partial evidence supported identity-preserving masks. | Incomplete and used whole-session normalization/masks. |
| iter40 | Fast masked-identity test | Random masks preserved healthy accuracy and rescued one severe channel-death session. | Single-seed fast test was not sufficient for promotion. |
| iter41 | Dynamic within-session masks | Channel set was nearly stationary; frequent remasking slightly hurt. A whole-session upper reference was explicitly non-causal. | Non-causal reference and obsolete experiment removed; decision is “select once after observation.” |

The companion `masked_input.py` was also removed. Its identity-preserving idea is
documented above, but its original implementation normalized and selected masks
from a complete held-out session. No current pipeline imports it.

## Retired legacy research families

| Removed family | Direction and recorded outcome |
| --- | --- |
| Multimodal EEG+fNIRS (`legacy/main.py`, `legacy/src/`, configuration) | Four-class reach/grasp/lift/twist decoding on ds004022 remained near chance. Riemannian features gave only a small benefit; the project pivoted away from this task. |
| EEG positive controls and probes (`legacy/tools/*eeg*`, Riemannian tools) | EEGMMIDB left/right hand positive control reached about 0.64 LORO, showing the signal pipeline could decode a simpler MI contrast. These diagnostics are unrelated to the current intracortical target. |
| WAY-EEG-GAL tools (`legacy/tools/way_gal_*`) | Offline EEG-to-kinematics TCN+GRU historically reported mean Pearson r about 0.853. It used a different modality/task and bidirectional/offline processing, so it is not a deployable Indy result. |
| Early monkey sweeps (`legacy/monkey_trials/`) | Explored activation, binning, channel count/selection, speed bands and tuning. Durable conclusions—40 ms bins, firing-based selection, more channels help—were retested in the numbered Indy sequence. |
| Legacy architecture/probe utilities | Conformer, SNN, CSP, fusion, simulation and plotting helpers served the retired EEG/fNIRS pipelines and had no supported imports. |
| `results/n1_fused.joblib` | Binary artifact from the retired N1 fused EEG/fNIRS pipeline; removed because its provenance and runnable consumer were retired. |

## Retired model/checkpoint folders and compatibility code

| Removed path | Recorded state |
| --- | --- |
| `models/tcn_gru/` | Historical 96-channel code/checkpoint. It included future-capable/bidirectional and centered-processing paths and is not valid for the current causal protocol. |
| `models/tcn_gru_8ch/` | Historical 8-channel checkpoint, 75,714 parameters, recorded test R² 0.6325. Although the network was unidirectional, its input cache used centered Gaussian smoothing and `test1` was burned. |
| `configs/indy_8ch.yaml` | Manifest for the deleted historical 8-channel checkpoint; it had no supported artifact after retirement. |
| `experiments/common/` | Compatibility harness and architecture adapters used only by deleted numbered experiments; they imported historical model/data code. |
| `experiments/tools/epoch_loss_curve.py` and plot helper | Instrumented the deleted historical checkpoint and recorded test every epoch. It was not a current validation tool. |

## Surviving source of truth

- Current reusable code: `src/intent_decoder/`
- Current decision-changing experiment: `experiments/active/drift_detector_month_cv.py`
- Separate-input benchmark: `experiments/deepblue/`
- Current status and next steps: `docs/STATUS.md` and `docs/ROADMAP.md`
- Full historical detail: `docs/history/EXPERIMENT_LOG.md`
