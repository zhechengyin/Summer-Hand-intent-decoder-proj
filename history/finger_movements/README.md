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
```

`results/phase2b_cssd_stabilization_retired_uea/` is explicitly retained as an
invalid-source result and must not be mixed with the corrected combination
ablation. Likewise, `results/phasea2_cssd_lda/` is the retired UEA result;
`results/phasea2_cssd_lda_official_matlab/` is the corrected rerun.

## Final archived outcome

On corrected official TRAIN data:

- terminal features + Logistic: 78.58% mean OOF balanced accuracy;
- Phase A2 CSSD + hierarchical LDA: 85.03%;
- Phase 2b winner: 86.72%.

The Phase 2b winner was promoted out of the archive to
`models/finger_movements/cssd_lda/`. All experimental runners and comparison
results remain here as immutable provenance.
