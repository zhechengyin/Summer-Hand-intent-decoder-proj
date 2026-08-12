# FingerMovements Experiment Archive

This directory preserves completed FingerMovements experiments. Nothing under
it is an active Python dependency.

## Data-validity boundary

The experiments through 2026-08-07 used a retired UEA conversion later found
to contain deterministic adjacent-channel overlap. Their scripts, metrics,
and the one-time official-test result are preserved for provenance, but those
numbers are invalid for comparison with the corrected official MATLAB data.

On 2026-08-10 the official MATLAB TRAIN data was converted directly and the
relevant models were rerun. Corrected-data evidence is explicitly identified
by its directory name or creation date.

## Contents

```text
EXPERIMENT_LOG.md   chronological protocol, invalidations, results, decisions
STATUS.md           archive snapshot and validity boundary
experiments/        completed Phase 1, Phase A2, and Phase 2b runners
models/             retired implementations and invalid-source checkpoints
results/            old provenance plus corrected-data completed evidence
```

Important corrected-data result directories:

```text
results/phasea2_cssd_lda_official_matlab/
results/phase2b_combination_ablation/
results/archived_terminal_logistic_official_matlab/
results/phase2c_horizon_diagnostic/
results/phase2c_streaming_causal/
results/phase2c_bin_window_sweep/
results/phase2d_official_test_400ms/
results/phase2e_lightweight_comparison/
results/phase2f_riemannian/
models/cssd_lda_offline_phase2b/
models/cssd_lda_causal_500ms_phase2c/
```

`results/phase2b_cssd_stabilization_retired_uea/` is explicitly retained as an
invalid-source result and must not be mixed with the corrected combination
ablation. Likewise, `results/phasea2_cssd_lda/` is the retired UEA result;
`results/phasea2_cssd_lda_official_matlab/` is the corrected rerun.

## Final archived outcome

On corrected official TRAIN data:

- terminal features + Logistic: 78.58% mean OOF balanced accuracy;
- Phase A2 CSSD + hierarchical LDA: 85.03%;
- Phase 2b zero-phase offline reference: 86.72%;
- selected Phase 2c causal 400 ms model: 83.99%;
- Phase 2d retrospective official-TEST BA of that frozen model: 77.05%.
- Phase 2e ToeplitzLDA exploratory mean OOF BA: 84.50%, not promoted.
- Phase 2f low-dimensional Riemannian mean OOF BA: 85.13%, not promoted.

The Phase 2b zero-phase winner is now archived under
`models/cssd_lda_offline_phase2b/`. Its causal Phase 2c successor is active
under `models/finger_movements/cssd_lda/`. All completed experimental runners
and comparison results remain here as immutable provenance. Phase 2f failed
its predeclared stability rule and did not create a checkpoint; the active
candidate remains the Phase 2c checkpoint.
