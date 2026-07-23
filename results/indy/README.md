# Indy Results

This directory contains the retained evidence for the Indy 32-channel decoder.
Each folder name matches one experiment phase. Metrics and figures from the same
experiment stay together.

| Folder | Experiment | Contents |
| --- | --- | --- |
| `phase0a_data_audit/` | Dataset integrity and month-drift audit | Full audit JSON, per-session CSV, month summary, month-pair distances, and two figures |
| `phase0b_sampler_selection/` | Window-, session-, and month-balanced sampler comparison | Three-seed aggregate metrics and comparison figure |
| `phase1e_seed_crosscheck/` | Final hyperparameter cross-seed confirmation | Retained five-seed metrics supporting the frozen configuration |
| `phase2_locked_january/` | One-shot locked January evaluation | Final inference-only metrics and evaluation figure |

## Phase mapping

- Phase 0a runner:
  `experiments/active/phase0a_data_audit.py`
- Phase 0b runner: deleted after its aggregate result was consolidated.
- Phase 1a--1e runners:
  `history/phase1/phase1a_optuna.py` through
  `history/phase1/phase1e_seed_crosscheck.py`
- Phase 2 runner:
  `history/phase2/phase2_locked_january.py`
- Phase 3: reserved for the label-free drift detector; no result folder exists
  until an experiment is actually run.

Phase 1a--1d intermediate databases, figures, metrics, and non-selected
checkpoints were intentionally deleted. The retained Phase 1e metrics and
`models/indy_32ch/checkpoint.pt` are the authoritative model-selection
artifacts.

The January split is consumed. Do not use Phase 2 results to select another
model, feature set, detector threshold, or hyperparameter configuration.
