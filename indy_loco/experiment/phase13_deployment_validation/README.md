# Phase 13 deployment validation

This experiment answers two separate questions without mixing their metrics:

1. Does a fresh PC execution of the current firmware numeric path reproduce
   the GUI manifest's Midsize `bank ABSENT` R²?
2. Are all 30 Phase-7 checkpoints present and do they reproduce the paper-like
   five-fold test results when reported as mean and standard deviation?

The deployment replay uses the current canonical checkpoint, GUI dataset and
evaluation mask, X-CUBE-AI 10.2 INT8 encoder, X-CUBE-AI FP32 GRU/head, CM7
60-second calibration, causal EWMA, continuous rolling 50-bin windows, channel
mapping and target scaling. It also runs the same bins through FP32 PyTorch to
isolate the CubeAI/quantization contribution.

The five-fold audit does not retrain and does not choose the highest test fold.
It independently loads and evaluates all five validation-selected checkpoints
for each of the six sessions. If any checkpoint, preprocessing parameter or
recorded score fails verification, the command exits non-zero.

Run everything from the training repository root:

```bash
.venv-deploy/bin/python \
  indy_loco/experiment/phase13_deployment_validation/run.py --mode all
```

Run only one part:

```bash
.venv-deploy/bin/python \
  indy_loco/experiment/phase13_deployment_validation/run.py --mode deployment

.venv-deploy/bin/python \
  indy_loco/experiment/phase13_deployment_validation/run.py --mode fivefold
```

Durable JSON/CSV outputs are written to `results/` under this directory.
Generated X-CUBE-AI host workspaces and rebuilt Indy input caches are temporary
by default and are written under `/tmp`.

## Round 2: calibration-duration sweep

Round 2 changes only the length of the causal, unlabeled session-prefix
calibration. The primary comparison uses a common test mask after the longest
calibration so that longer calibration cannot look better merely by deleting
harder early bins:

```bash
.venv-deploy/bin/python \
  indy_loco/experiment/phase13_deployment_validation/run_calibration_sweep.py
```

After the FP32 knee is known, confirm selected durations through CubeAI by
passing values already present in `--minutes`, for example:

```bash
.venv-deploy/bin/python \
  indy_loco/experiment/phase13_deployment_validation/run_calibration_sweep.py \
  --cubeai-minutes 1 7 10
```

The completed six-session sweep selected **7 minutes** as the practical knee.
On the fixed common evaluation mask, exact CubeAI mean R² increased from
`0.5988` at 1 minute to `0.6927` at 7 minutes; extending calibration to 10
minutes reached `0.7060`. All six sessions improved from 1 to 7 minutes. The
8-minute maximum-duration sensitivity run also selected 7 minutes, so the
recommendation is not an artifact of the shorter common test mask left by the
10-minute sweep. See `results/calibration_sweep/TECHNICAL_REPORT.md` for the
metric definitions, per-session results, caveats, and deployment recommendation.

## Round 3: retrain for 7-minute calibration and rolling windows

Round 3 preserves the Phase-7 five-fold reach splits but trains on the exact
deployment input contract: continuous session-level EWMA, past-only rolling
50-bin windows, seven-minute unlabeled prefix calibration, and loss on the final
GRU timestep used by firmware. Validation loss selects each checkpoint and test
targets are opened only after the checkpoint is saved.

First validate the environment:

```bash
.venv-deploy/bin/python \
  indy_loco/experiment/phase13_deployment_validation/run_rolling_retrain.py \
  --validate-only
```

The recommended first run warm-starts the matching Phase-7 fold, updates all
weights, and gives the GRU/head four times the encoder/TCN learning rate:

```bash
.venv-deploy/bin/python \
  indy_loco/experiment/phase13_deployment_validation/run_rolling_retrain.py
```

Resume an interrupted run with the exact same arguments:

```bash
.venv-deploy/bin/python \
  indy_loco/experiment/phase13_deployment_validation/run_rolling_retrain.py \
  --resume
```

Useful controlled variants are:

```bash
# Only adjust GRU/head weights (faster isolation experiment).
.venv-deploy/bin/python \
  indy_loco/experiment/phase13_deployment_validation/run_rolling_retrain.py \
  --train-scope gru-head --output-name phase7_init_gru_head

# Fresh Phase-7-style training under the new deployment input contract.
.venv-deploy/bin/python \
  indy_loco/experiment/phase13_deployment_validation/run_rolling_retrain.py \
  --init scratch --output-name scratch_all

# One-session/one-fold smoke training before the complete 30-fit run.
.venv-deploy/bin/python \
  indy_loco/experiment/phase13_deployment_validation/run_rolling_retrain.py \
  --session indy_20160622_01 --fold 1 --epochs 2 \
  --output-name smoke_indy_fold1
```

Every fold reports four scores on identical post-calibration test bins:

1. original Phase-7 reach-local inference;
2. original weights with continuous windows and Phase-7 training normalization;
3. original weights with continuous windows and seven-minute calibration; and
4. retrained weights with continuous windows and seven-minute calibration.

All checkpoints, state files, CSVs, and JSON results remain under
`results/rolling_retrain/`; nothing is copied into the deployable model package
until a separate promotion decision is made.

The completed default run (`final_30fold`) produced a five-fold macro mean R² of
`0.7411`, compared with `0.6728` for the same Phase-7 weights under seven-minute
rolling deployment preprocessing and `0.7089` for matched Phase-7 reach-local
inference. Retraining improved 29 of 30 folds and all six session means. See
`results/rolling_retrain/final_30fold/ROUND3_TECHNICAL_REPORT.md` for the full
decomposition, paired statistics, weight-change audit, and limitations.

Rebuild the paired statistics and parameter-change audit from the saved
checkpoints with:

```bash
.venv-deploy/bin/python \
  indy_loco/experiment/phase13_deployment_validation/analyze_rolling_retrain.py
```
