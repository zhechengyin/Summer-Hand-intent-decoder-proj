# Active Experiments

## Phase A2: paper-style CSSD + hierarchical LDA

`phasea2_cssd_lda.py` implements the three feature branches described by Wang
et al. for BCI Competition 2003 Data Set IV:

- BP: zero-phase 0--7 Hz, sample points 44--47, one left/right CSSD pair;
- ERD: zero-phase 10--33 Hz, points 19--50, three left/right CSSD pairs and
  eight-sample absolute pooling;
- BP trend: points 1--8 and 41--50 on the paper's retained 19 channels.

Each branch is reduced to one Fisher/LDA score. A final LDA combines the three
scores. This differs from the paper only at final fusion, where the paper used
a perceptron. Every CSSD projection, scaler, and LDA is fitted from the current
training fold only. The official test is refused.

Run from the repository root:

```bash
python experiments/active/phasea2_cssd_lda.py
```

Run one fold without writing results:

```bash
python experiments/active/phasea2_cssd_lda.py --validate-only
```

Run the TRAIN-only generalization diagnostics:

```bash
python experiments/active/phasea2_cssd_lda.py --diagnostics
```

The diagnostic mode leaves the standard Phase A2 predictions unchanged and
adds train-versus-validation layer metrics, all seven branch ablations,
inner-OOF fusion, cross-fold CSSD subspace stability, and per-case seed
stability. It never loads the official test.

The completed diagnostic run found that the CSSD branches are not stable
enough across folds. In particular, ERD branch AUC fell from 83.18% on outer
training folds to 55.79% on validation folds. Inner-OOF fusion improved the
three-branch result from 59.81% to 61.08% balanced accuracy, but the most
stable option was the non-CSSD BP-trend branch alone at 62.25%. This remains
below the archived 68.89% terminal-Logistic baseline.

There are no epochs: CSSD uses simultaneous covariance diagonalization and
LDA uses a closed-form linear discriminant solution. The zero-phase filters
make this an offline paper-reproduction baseline, not yet a causal firmware
candidate.
