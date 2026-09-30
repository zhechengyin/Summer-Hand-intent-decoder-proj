# Phase17: minGRU / Mamba-2 validation sweep

Authorized 2026-09-26. This is an extension of Phase17, not a new experiment
phase. The original Phase17 Python files, checkpoints and results are retained.
The larger-model/SDRAM experiment is deferred.

## Frozen design

- Same six sessions, five reach folds, preprocessing/scalers and 50-bin windows
  as Phase17. Every prepared fold must match the recorded array and split hashes
  of both completed original candidate runs. Fold seed stays 43.
- Same architectures: minGRU 88,066 parameters; Mamba-2 82,290 parameters.
  Only the existing paired channel dropout and post-stem dropout probabilities
  change; no new layers or dropout sites.
- Scratch FP32 AdamW, batch 128, clip 1.0, maximum 40 epochs, patience 10.
  Cosine LR schedule uses T_max=40; AdamW betas/eps and all-parameter weight
  decay retain PyTorch/Phase17 defaults. No architecture-specific optimizer
  exemptions. This is a revised training budget, not the old 20-epoch protocol.
- Within a fit, select the checkpoint with minimum validation normalized MSE.
  Across trials, maximize the equal-weight mean validation R2 of those selected
  checkpoints. Tie-break by lower mean validation loss, then trial ID.
- Train seeds 43/44/45 never alter data partitions or scalers. Seed 43 alone
  selects configurations; seeds 44/45 assess stability without reselection.

## Prespecified candidates (identical for both architectures)

These 12 configurations are a bounded coarse sweep: an original-hyperparameter
anchor, five individual probes and six joint candidates. They are not an
exhaustive Cartesian grid and do not establish globally optimal hyperparameters.
The anchor also receives the new 40-epoch maximum.

| Trial | Temporal/head LR | Stem LR multiplier | Weight decay | Paired channel dropout | Post-stem dropout |
|---|---:|---:|---:|---:|---:|
| t00 | 0.0003 | 0.25 | 0.025 | 0.2 | 0.1 |
| t01 | 0.0001 | 0.25 | 0.025 | 0.2 | 0.1 |
| t02 | 0.0010 | 0.25 | 0.025 | 0.2 | 0.1 |
| t03 | 0.0003 | 1.00 | 0.025 | 0.2 | 0.1 |
| t04 | 0.0003 | 0.25 | 0.001 | 0.2 | 0.1 |
| t05 | 0.0003 | 0.25 | 0.025 | 0.0 | 0.0 |
| t06 | 0.0001 | 1.00 | 0.010 | 0.1 | 0.0 |
| t07 | 0.0001 | 0.25 | 0.001 | 0.0 | 0.1 |
| t08 | 0.0003 | 1.00 | 0.050 | 0.3 | 0.2 |
| t09 | 0.0003 | 1.00 | 0.001 | 0.1 | 0.2 |
| t10 | 0.0010 | 1.00 | 0.010 | 0.1 | 0.1 |
| t11 | 0.0010 | 0.25 | 0.050 | 0.3 | 0.2 |

## Equal-budget stages

| Stage | Work | New training fits |
|---|---|---:|
| Screen | 12 configs x 6 sessions x fixed fold 1 x 2 models, seed 43 | 144 |
| Confirm | Top 2 per model, add folds 2-5 for all sessions | 96 |
| Seed check | Best config per model, all 30 folds at seeds 44 and 45 | 120 |
| Final test | Frozen finalists: 2 models x 3 seeds x 30 folds | 0 (180 evaluations) |
| Total | GPU training runs sequentially | 360 |

Screening uses a fixed fold, not the most favorable observed fold, and covers
both subjects. Selection across all 30 validation folds determines one
configuration per architecture and a provisional architecture winner, before
any new test prediction. Every selected checkpoint for all three seeds must
exist and pass integrity checks before test is opened. Do not pick a winning
training seed based on test, or promote/deploy based on this run automatically.

The runner saves session-level results, fold sample SD, paired changes versus
historical Midsize, parameter counts and FP32 sizes. These are weight sizes,
not peak runtime RAM or latency. Historical Midsize was warm-started.

## Execution and recovery

Run from the repository root using the existing CUDA virtual environment:

~~~powershell
.\.venv\Scripts\python.exe -X utf8 indy_loco\experiment\phase17_architecture_comparison\sweep\run_sweep.py --device cuda --preflight-only
powershell -NoProfile -ExecutionPolicy Bypass -File indy_loco\experiment\phase17_architecture_comparison\sweep\start_sweep.ps1
~~~

The launcher uses a hidden background process, records a unique stdout/stderr
pair and PID, and refuses another active Phase17 training process. The runner
also holds OS locks for the sweep and the original minGRU/Mamba-2/Transformer
result directories. The launcher resumes after preflight automatically.
No scheduled task is needed or re-enabled.

Everything is under ../results/sweep_v1/:

- config.json: immutable search, code/environment/input/reference hashes.
- preflight.json: 30 verified data folds, zero optimizer steps.
- progress.json: stage, active fit and PID.
- runs/<model>/<trial>/seed<seed>/epochs/: per-epoch validation logs.
- runs/.../checkpoints/: completed best-validation checkpoints and hash receipts.
- screen_selection.json, final_selection.json, test_gate.json: frozen
  promotion decisions and hashes of all 180 finalist checkpoints.
- metrics.json, summary.csv, sessions.csv, REPORT.md: final results.

To stop gracefully after the active fold, create this marker:

~~~powershell
New-Item -ItemType File -Path indy_loco\experiment\phase17_architecture_comparison\results\sweep_v1\STOP_AFTER_FOLD
~~~

To resume later, first remove that marker if you created it, then rerun the
launcher. Completed fold checkpoints are hash-verified and skipped; a fold
interrupted before its final checkpoint restarts from its fixed seed.
Optimizer/epoch-level resume is not supported. Never edit fingerprinted Python
files while the sweep is running. Integrity failures stop the run with an
explicit error; they are not bypassed with replacement outputs.

## Interpretation

The historical test has already been exposed. Global HPO on these overlapping
folds is not nested cross-validation: a bin held out in one fold can be used for
training in another fold. Although the runner never uses test metrics for
selection, final numbers are developmental results under the retained protocol,
not unbiased new-data evidence. Fresh sessions are needed for a clean final
generalization claim. Five folds are correlated; three training seeds do not
turn them into independent subjects.

Stage-one filtering can miss a configuration that would win on all five folds.
Larger architectures, their training details and MCU latency must be evaluated
separately after this sweep.
