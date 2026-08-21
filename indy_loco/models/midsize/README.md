# Midsize checkpoints

`model.py` contains the single 96-channel, 86,978-parameter TCN+GRU
architecture shared by every checkpoint in this directory.

The root `checkpoint.pt` remains the promoted Indy Phase 6 firmware model used
by `runtime.py`. It was not replaced while building the session checkpoints.

The six session subdirectories contain freshly trained Phase 7 fold-1
benchmark checkpoints:

- `indy_20160622_01/checkpoint.pt`
- `indy_20160630_01/checkpoint.pt`
- `indy_20170131_02/checkpoint.pt`
- `loco_20170210_03/checkpoint.pt`
- `loco_20170215_02/checkpoint.pt`
- `loco_20170301_05/checkpoint.pt`

The paper labels the 587-reach Loco file as `loco_20170131_02`. The published
file whose checksum and reach count match that benchmark row is
`loco_20170210_03`; the Phase 7 experiment therefore uses the corrected name.

## Protocol

- Phase 7 fold 1, seed 43, 20 epochs.
- A fresh model is trained for each session.
- Validation loss selects the frozen checkpoint; test reaches are evaluated
  only after checkpoint selection.
- Input is 96 raw 40 ms spike-count features plus 96 causal-EWMA features
  (`alpha=0.1`) over a 50-bin, 2-second window.
- Indy uses all 96 channels. Loco selects 96 of 192 channels using training
  reaches only; the selected indices and preprocessing statistics are stored
  inside each checkpoint.

These six files are marked `benchmark_only_not_promoted`. They are not the
current board firmware weights and must not be presented as Phase 9 deployment
replays. The attached paper uses different ANN/SNN/LSTM architectures; Phase 7
is paper-aligned only in its six sessions and reach-level five-fold structure.

The original archived Phase 7 weight files were absent, so these are honest
retraining outputs from the documented protocol, not byte-identical copies of
the archived run. `session_checkpoints.json` records their actual metrics and
SHA-256 hashes.

## Deployment candidates

`experiments/active/phase10_session_deployment_candidates.py` leaves every
benchmark `checkpoint.pt` unchanged and writes four deployment sidecars in
each session directory:

- `deployment_candidate.pt`: the same frozen weights plus the session model ID,
  96-channel mapping, target statistics, fitted feature-std floor, and exact
  firmware preprocessing contract.
- `deployment_constants.npz`: compact conversion input containing the channel
  mapping, `feature_std_floor[192]`, `target_mean[2]`, and `target_std[2]`.
- `deployment_golden_vectors.npz`: eight normalized 50-bin inputs and expected
  PyTorch outputs for conversion parity checks.
- `deployment_replay.json`: hashes, floor provenance, and Phase 7-versus-firmware
  replay metrics.

The feature-std floor is fitted without validation or test reaches. Fold-1
training reaches are ordered chronologically, concatenated, divided into
non-overlapping 1500-bin (60-second) blocks, and the non-silent per-feature
10th percentile is retained. A feature silent in every training block uses the
already-promoted Phase 6 floor at the same model-input position; the fallback
indices are explicit in `deployment_replay.json`.

The firmware replay uses continuous EWMA, first-60-second session calibration,
an oldest-to-newest rolling 50-bin window, and output timestep 49. It evaluates
only the original fold-1 held-out test-reach bins that remain after calibration.
Candidate status remains `deployment_candidate_replay_complete` and promotion
requires manual review; the script never silently declares a benchmark model a
production model.
