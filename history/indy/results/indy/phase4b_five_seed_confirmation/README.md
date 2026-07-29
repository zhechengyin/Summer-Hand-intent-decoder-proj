# Phase 4b five-seed architecture confirmation

This folder contains the completed controlled comparison of:

- protected baseline: 64 TCN filters / 64 GRU hidden units;
- firmware candidate: 48 TCN filters / 48 GRU hidden units.

The fixed grid crossed seeds 42--46 with five complete pre-January held-month
folds: all 50 fold fits completed. Learning rate, weight decay, dropout,
sampling, data preprocessing, channel mapping and the seven-epoch training
point remained frozen.

Retained outputs:

- one metrics JSON with the predeclared non-inferiority decision;
- one 10-row architecture/seed table;
- one 50-row fold table;
- one 330-row held-session table;
- one summary figure;
- resumable fold progress and independent prepared arrays under `.cache/`.

January was never loaded and Phase 4b itself produced no checkpoint. All four
predeclared non-inferiority guardrails passed:

| Check | Candidate minus baseline | Limit | Result |
| --- | ---: | ---: | --- |
| Mean session-macro R² | -0.005919 | -0.010 | Pass |
| Mean session-q10 R² | +0.003694 | -0.020 | Pass |
| Mean worst-session R² | -0.014631 | -0.020 | Pass |
| Worst-month macro R² | -0.018616 | -0.020 | Pass |

The 48/48 architecture was nominated for firmware efficiency, not for higher
five-seed average accuracy. It has 45,266 parameters versus 78,786 and
approximately 42.7% fewer multiplies per 50-bin window.

The separately authorized fixed build has now completed:

| Build result | Value |
| --- | ---: |
| Checkpoint | `models/indy_32ch/48x48checkpoint.pt` |
| SHA-256 | `5c8b375787ff93f90006df5f0cfea07303660928c7b69a84d4d75e1a368319ef` |
| Seed / training budget | 43 / 20 epochs |
| Selected epoch | 10 |
| Train pooled R² | 0.763162 |
| December pooled R² | 0.565134 |
| December macro R² | 0.575004 |
| December worst-session R² | 0.346125 |

Only train data updated weights, December selected the checkpoint by minimum
pooled validation loss, and January remained unopened. Build evidence is in
`48x48checkpoint_build_metrics.json`. The protected integrated model remains
`models/indy_32ch/64x64checkpoint.pt`.

Runner:
`history/phase4/phase4b_five_seed_architecture_confirmation.py`.

The original command was:

```bash
python history/phase4/phase4b_five_seed_architecture_confirmation.py \
  --device cpu
```

The final builder is archived at
`history/phase4/train_48x48_checkpoint.py`. Both scripts are provenance only;
the completed metrics and retained checkpoints are authoritative.
