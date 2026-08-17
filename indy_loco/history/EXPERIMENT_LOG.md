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

## Current handoff

The earlier Indy line was archived on 2026-07-29, then reopened for controlled
channel-count experiments. Phase 6 is complete and its 96-channel seed-43
checkpoint is the strongest validation candidate. The 32-channel 48/48 model
remains the smaller firmware reference; neither historical checkpoint was
overwritten.
