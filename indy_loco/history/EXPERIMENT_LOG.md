# Indy Loco Experiment Log

This is the concise historical record of Indy velocity-decoding experiments.
Metrics refer to archived artifacts under `history/results/indy/` when paths
are read from the Indy project root.

## Phase summary

| Phase | Experiment | Main result | Decision |
|---|---|---|---|
| 0a | Data and month audit | Strong month identity in neural features: 72.97% month classification versus 16.95% permutation null | Treat cross-month shift as a primary risk |
| 0b | Sampling comparison | Session-balanced validation loss 0.5074 and R² 0.5342; won 3/3 seeds | Freeze session-balanced sampling |
| 1a–1e | Hyperparameter search and confirmation | Final values: LR 0.0009, WD 0.060, dropout 0.025; WD 0.060 slightly beat 0.025 over five seeds | Freeze optimization settings |
| 2 | January one-shot test | Pooled R² 0.5511; `20170124_01` R² -0.0524 | Test consumed; investigate drift |
| 3a | Label-free drift baseline | Firing-rate checks explained some failures but missed June 30 | Retain as interpretable first layer only |
| 3b | Leave-one-month-out decoder test | Macro R² 0.5597; pooled R² 0.5266; failures on June 30 and October 13 | Use as cross-month robustness benchmark |
| 3c | Hidden/output detector | Detected both known leave-month failures and passed the other 31 sessions across nine threshold settings | Retain as decoder-coupled second layer; not prospectively validated |
| 4a | Architecture sweep | 48/48 candidate used 45,266 parameters versus 78,786 and showed similar validation performance | Nominate 48/48 |
| 4b | Five-seed confirmation | 48/48 passed all predefined non-inferiority limits, with 42.5% fewer parameters | Promote 48/48 firmware candidate |
| 4c | Final 48/48 build | Epoch 10; validation pooled R² 0.5651; macro 0.5750; worst 0.3461 | Retain checkpoint |
| 5a | 64-channel width comparison | 64/64 reached validation R² 0.6625; 48/48 reached 0.6569 | Exploratory only; single seed, no promotion |
| 5 | 64-channel tuning and detector-filter ablation | Full-session winner: LR 0.0009, WD 0.025, dropout 0.10; pooled R² 0.6575 ± 0.0080 over seeds 42–44 | Freeze settings; keep all 29 sessions; detector remains a runtime gate |
| 6 | Channel, structure, and regularization sweep | 96 channels with 0.20 paired channel dropout reached pooled R² 0.7004 ± 0.0019 and macro R² 0.7023 ± 0.0015 | Promote seed-43 epoch-15 checkpoint as the strongest validation candidate |
| 7 | Six-session ANN-vs-SNN reach-level five-fold benchmark | Test R² 0.7056 ± 0.0722 over 30 folds; session means ranged from 0.6277 to 0.8066 | Benchmark complete; retain Phase 6 model as the source architecture and do not promote any session-specific fold checkpoint |
| 8 | Permitted neural-lookahead comparison | Indy test R² 0.7576 ± 0.0396 at 48 ms and 0.7554 ± 0.0397 at 100 ms over 15 folds per condition | Use 48 ms as the current high-accuracy/longer-latency operating point; do not describe it as causal |

## Phase 0 — Data and sampling

Thirty-seven sessions were converted to model-ready NPZ files and split chronologically: 29 train, 4 December validation, and 4 January test. Inputs were 40 ms spike-count bins. The model selected 32 channels from training prefixes, appended causal EWMAs, and predicted two velocity dimensions from the preceding 2 seconds.

The month audit showed that neural inputs changed substantially between months, while targets drifted less. The sampler comparison then tested window-, session-, and month-balanced exposure with seeds 42–44. Session balancing won all three seeds and became fixed for later work.

## Phase 1 — Optimization settings

Optuna and focused grids searched learning rate, weight decay, and dropout while preserving the sampler and split. The final settings were learning rate 0.0009, weight decay 0.060, dropout 0.025, batch size 32, and seed 43 for final checkpoint construction. Five-seed confirmation showed only a small advantage for weight decay 0.060, so the choice should not be interpreted as a universal optimum.

## Phase 2 — January evaluation

The frozen 64/64 decoder was evaluated once on four January sessions:

| Session | R² |
|---|---:|
| `indy_20170123_02` | 0.6947 |
| `indy_20170124_01` | -0.0524 |
| `indy_20170127_03` | 0.6989 |
| `indy_20170131_02` | 0.6780 |

Pooled R² was 0.5511 and session-macro R² was 0.5048. Because January results were then used to motivate detector design, January is historical holdout evidence rather than a pristine final test.

## Phase 3 — Cross-month robustness and detector

Strict leave-one-month-out training used only the remaining pre-January months for each fold. Results were:

| Held-out month | Macro R² | Worst-session R² |
|---|---:|---:|
| April | 0.4184 | 0.2321 |
| June | 0.3987 | -0.1365 |
| September | 0.7359 | 0.6873 |
| October | 0.5922 | -0.0541 |
| December | 0.5702 | 0.3144 |

The simple detector compared 60-second channel firing-rate patterns, global rate, unexpected silent channels, and low-dimensional activity statistics against training references. It detected the October failure but not June 30.

The second layer added frozen-decoder hidden-state and absolute-output distribution distances. Requiring both KLD scores to exceed their 0.99 training-reference quantiles caught June 30 and October 13 and passed the other 31 held-out sessions across nine threshold variants. The result is retrospective and decoder-specific; it was not validated on unseen future sessions.

## Phase 4 — Smaller firmware candidate

Phase 4a searched TCN width, GRU width, block count, kernel size, and GRU layers while freezing all data and optimization choices. Phase 4b compared 64/64 and 48/48 over five seeds and five held-out months (50 fits total).

| Architecture | Selection score | Macro R² | 10th-percentile R² | Worst R² | Parameters |
|---|---:|---:|---:|---:|---:|
| 64/64 | 0.4773 | 0.5500 | 0.2592 | -0.1600 | 78,786 |
| 48/48 | 0.4738 | 0.5441 | 0.2629 | -0.1746 | 45,266 |

The 48/48 model passed all predefined non-inferiority limits and reduced both parameters and multiply count by about 42%. A seed-43, 20-epoch final build selected epoch 10 by December validation loss and produced the retained `48x48checkpoint.pt`.

## Phase 5a — Exploratory 64-channel comparison

The authoritative run was repeated on CPU after an invalid unstable MPS attempt. Both models used 64 raw channels plus 64 causal EWMAs, seed 43, 30 epochs, and December validation selection; January was never loaded.

The 64/64 model achieved pooled validation R² 0.6625 at epoch 7; the 48/48 model achieved 0.6569 at epoch 4. This indicated value in additional channels, but one seed was insufficient to replace the confirmed 32-channel candidate. The experiment ended without promotion when the project changed datasets.

## Phase 5 — 64-channel tuning and detector-filter ablation

Phase 5 reopened the 64-channel 64/64 model and compared two policies under the
same session-balanced optimizer exposure: the canonical 29-session baseline
and a 27-session variant excluding the retrospective Phase-3c failures
`indy_20160630_01` and `indy_20161013_03`. January was never loaded.

The best full-session configuration was learning rate 0.0009, weight decay
0.025, and dropout 0.10. Across seeds 42–44 it achieved pooled December
validation R² `0.6575 ± 0.0080`, macro R² `0.6627 ± 0.0076`, mean worst-session
R² `0.5526`, and minimum worst-session R² `0.5186`.

The detector-filtered variant achieved pooled R² `0.6552 ± 0.0034` and macro
R² `0.6595`. It improved the worst-session floor but did not improve mean
performance. The decision was therefore to train on all 29 sessions and keep
the detector as a runtime compatibility gate rather than a retrospective data
deletion rule. Phase 6 inherits the winning baseline hyperparameters and tests
all 96 physical channels.

## Phase 6 — 96-channel regularized model

Phase 6 kept the 64/64 width, session-balanced sampling, chronological split,
causal preprocessing, learning rate 0.0009, weight decay 0.025, and model
dropout 0.10 fixed. Seed 43 screened channel count/ranking, kernel size, TCN
depth, and paired physical-channel dropout. The selected configurations and
fixed references were then confirmed with seeds 42 and 44. All 29 training
sessions updated weights; December was inference-only but selected
configurations and checkpoints; January was never loaded.

The winner used all 96 physical channels, kernel size 3, four TCN blocks, and
0.20 paired channel dropout. Dropping a physical channel removed its raw-count
and causal-EWMA streams together during training. The regularizer was inactive
during validation and inference.

| Configuration | Pooled validation R² | Macro R² | Worst-session mean | Worst-session minimum |
|---|---:|---:|---:|---:|
| 96 channels, paired dropout 0.20 | **0.7004 ± 0.0019** | **0.7023 ± 0.0015** | **0.6085** | **0.5950** |
| Stability-ranked 72 channels | 0.6735 ± 0.0076 | 0.6770 ± 0.0076 | 0.5961 | 0.5782 |
| Activity-ranked 64 channels | 0.6566 ± 0.0061 | 0.6617 ± 0.0058 | 0.5517 | 0.5176 |
| 96 channels without paired dropout | 0.6476 ± 0.0057 | 0.6504 ± 0.0052 | 0.5507 | 0.5381 |

The winning pooled R² values for seeds 42, 43, and 44 were 0.6978, 0.7022,
and 0.7012. The result therefore reflects a stable regularization effect rather
than one lucky initialization. Compared with the Phase 5 64-channel winner,
mean pooled R² improved by 0.0429 and the mean worst-session R² improved by
0.0559. All 96 channels without paired dropout were worse than the 64-channel
reference, showing that channel count alone did not produce the gain.

Seed 43 had the lowest validation loss and highest pooled/macro R² of the three
confirmed runs. Its epoch-15 state was promoted to
`models/indy_96ch/phase6_96ch_64x64_checkpoint.pt`. It has 86,978 parameters
and is the strongest validation-selected Indy candidate. This is not a new
January result, and the existing 32-channel firmware references remain intact.

## Phase 7 — Six-session ANN-vs-SNN benchmark

Completed on 2026-08-18. Phase 7 reused the Phase 6 architecture and frozen
hyperparameters but not its weights. A new seed-43 model was trained for each
of five reach-level folds in each of the six paper sessions, producing 30
independent fits. Four reach groups trained each fold; the fifth was divided
between checkpoint-selecting validation and post-selection test subsets.

Indy used all 96 physical channels. Loco selected 96 of its 192 source channels
inside each fold using training reaches only. Both subjects used 96 raw 40 ms
counts, 96 causal-EWMA streams, the 64/64 TCN+GRU, and 0.20 paired channel
dropout. Incomplete edge reaches and reaches over eight seconds were excluded.
Short reaches were causally right-padded and masked out of loss and metrics.

| Session | Test R² mean ± SD | Fold range |
|---|---:|---:|
| `indy_20160622_01` | **0.8066 ± 0.0153** | 0.7816–0.8216 |
| `indy_20160630_01` | 0.6828 ± 0.0115 | 0.6707–0.7004 |
| `indy_20170131_02` | 0.7443 ± 0.0445 | 0.6956–0.7998 |
| `loco_20170210_03` | 0.6749 ± 0.0289 | 0.6365–0.7067 |
| `loco_20170215_02` | 0.6277 ± 0.0838 | 0.5211–0.6981 |
| `loco_20170301_05` | 0.6974 ± 0.0590 | 0.6031–0.7500 |
| **All 30 folds** | **0.7056 ± 0.0722** | **0.5211–0.8216** |

The fold-macro result exceeded the paper table's ANN (0.6186), ANN3D (0.6467),
and SNN3D (0.6661) reference means by 0.0870, 0.0589, and 0.0395,
respectively. This is encouraging but not an exact architecture comparison:
Phase 7 used the project's 40 ms causal TCN+GRU and a 96-channel firmware input
limit, while the paper evaluated its own ANN/SNN pipelines. The Loco-only fold
mean was 0.6667, versus 0.7446 for Indy, and `loco_20170215_02` showed the
largest fold variance. The result therefore establishes a strong benchmark
mean, not uniform robustness across every session.

All fold checkpoints remain experiment evidence under
`results/indy/phase7_ann_vs_snn_fivefold/checkpoints/`. They are session/fold
specific and are not promoted as a general firmware checkpoint. The retained
Phase 6 checkpoint remains unchanged.

## Phase 8 — Permitted neural-lookahead comparison

Completed for Indy on 2026-08-18. Phase 8 held the Phase 6 64/64 TCN+GRU,
optimizer settings, temporal window, and paired channel dropout fixed, then
changed only the alignment between neural input and velocity target. The two
conditions deliberately allowed either 12 native samples (48 ms) or 25 native
samples (100 ms) of neural lookahead. Fresh seed-43 weights were trained for
each of five reach-level folds in each of three Indy sessions.

| Lookahead condition | Folds | Test R² mean ± SD | Fold range |
|---|---:|---:|---:|
| 48 ms | 15 | **0.7576 ± 0.0396** | 0.7071–0.8192 |
| 100 ms | 15 | 0.7554 ± 0.0397 | 0.6908–0.8135 |

The 48 ms condition was slightly stronger overall and requires less waiting,
so it is the preferred lookahead operating point. Session-specific effects
were mixed: 100 ms improved June 30 but reduced January 31. Phase 8 therefore
supports a controlled accuracy/latency tradeoff, not a claim that more future
information always improves every session. These models are deliberately
non-causal and are not interchangeable with the promoted Phase 6 general
checkpoint. The equivalent Loco runner is ready but has not yet produced
results.

## Phase 9 — Causal deployment-policy replay

Completed on 2026-08-19. No weights were trained and the promoted Phase 6
checkpoint remained unchanged. The replay preserved the exact 96 raw + 96
causal-EWMA feature order, 60-second session-local calibration, checkpoint
feature-std floor, and target de-normalization. December validation alone
compared the original non-overlapping block-reset inference policy against a
continuous stride-1 rolling past-window seeded by calibration bins 1450–1499.

| Validation interval | Block-reset pooled R² | Rolling pooled R² | Block-reset MSE | Rolling MSE |
|---|---:|---:|---:|---:|
| First 10 s | 0.7278 | **0.8059** | 8.2333 | **5.6674** |
| First 30 s | 0.7144 | **0.7690** | 10.1647 | **8.1631** |
| All post-calibration | 0.7021 | **0.7526** | 14.8720 | **12.3567** |

Rolling won all four validation sessions, improved pooled full-session R² by
`0.0505`, and reduced MSE by about `16.9%`. It was frozen before January was
loaded. Winner-only locked-test inference then reached pooled January R²
`0.7277` and MSE `12.3149`. The block replay matched corresponding full-block
outputs within `2.9e-6`, supporting implementation validity. The deployment
decision is to retain the final 50 calibration bins as the initial past window
and never reset that window every 50 bins. Phase 9 results remain active under
`results/phase9_deployment_policy_replay/`; no firmware was modified.

## Deployment-tier decision — Tiny, Mid-size, and Large

On 2026-08-18, the retained Indy results were organized into three deployment
tiers for system planning and presentation. Channel count is not used in the
tier names.

| Tier | Evidence assigned to the tier | Architecture and timing | Result used |
|---|---|---|---|
| **Tiny** | Retained `48x48checkpoint.pt` | Four-block width-48 causal TCN + one-layer width-48 unidirectional GRU; past-only input; 45,266 parameters | December pooled validation R² 0.5651 |
| **Mid-size** | Promoted Phase 6 checkpoint | Four-block width-64 causal TCN + one-layer width-64 unidirectional GRU; past-only input; 86,978 parameters | Seed-43 pooled validation R² 0.7022; three-seed mean 0.7004 ± 0.0019 |
| **Large** | Phase 8 48 ms lookahead operating point | Same 64/64 TCN+GRU topology, retrained with 48 ms future neural alignment; 86,978 parameters plus future-input buffering | 15-fold Indy test R² 0.7576 ± 0.0396 |

The decision is a deployment taxonomy, not a new controlled three-model
experiment. Tiny and Mid-size are chronological validation checkpoints; Large
is reach-level five-fold test evidence under a different, deliberately
non-causal protocol. The current project does not contain a retained Indy MLP,
so Tiny must not be labeled MLP. Large has the highest measured performance and
longest latency, but it is not yet a wider/deeper neural network and does not
by itself prove that off-chip memory is required. A structurally larger
off-chip model would need a separately approved experiment and checkpoint.

## Current handoff

The earlier Indy line was archived on 2026-07-29, then reopened for controlled
channel-count, paper-benchmark, and lookahead experiments. Phase 8 is complete
for Indy and pending for Loco. The three deployment tiers above summarize the
current operating points: compact causal Tiny, promoted causal Mid-size, and
48 ms lookahead Large. Phase 7 and Phase 8 fold checkpoints remain benchmark
evidence only; the Phase 6 seed-43 checkpoint remains the strongest general
validation candidate. The compact 48/48 checkpoint remains the smaller
firmware reference; neither retained checkpoint was overwritten.
