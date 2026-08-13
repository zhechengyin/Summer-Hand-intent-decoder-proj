# Indy Loco Experiment Log

This is the concise record of the retired Indy velocity-decoding work. Metrics refer to archived artifacts under `results/indy/`.

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

## Closeout

The Indy work was archived on 2026-07-29. The 48/48 checkpoint is the preferred standalone artifact; the 64/64 checkpoint is retained for reproducing detector work. No result in this archive is an active project dependency.
