# Active Experiments

Phase 2e is complete and remains here pending archival:

```text
phase2e_lightweight_regularization_comparison.py
```

It performs a paired TRAIN-only comparison on the exact Phase 2c seeds and
five stratified folds:

- current empirical CSSD + SVD LDA baseline;
- OAS-regularized CSSD;
- analytical shrinkage LDA;
- regularized CSSD + shrinkage LDA;
- block-Toeplitz shrinkage LDA.

Run the complete comparison:

```bash
python experiments/active/phase2e_lightweight_regularization_comparison.py
```

Run the one-fold implementation check without writing results:

```bash
python experiments/active/phase2e_lightweight_regularization_comparison.py --validate-only
```

Official TEST is refused. Every learned covariance, CSSD projection, scaler,
and classifier is fitted inside its current training fold. The runner records
mean and worst-seed balanced accuracy, seed consistency, fold variability,
OOF error complementarity, deployment parameters/RAM, and full float32
prediction equivalence.

Baseline/Toeplitz fusion is conditional. It runs only when ToeplitzLDA corrects
at least 10% of baseline errors in every seed; if enabled, its weights are
trained by four-fold inner OOF stacking rather than outer-validation labels.

The selected model remains the frozen Phase 2c 400 ms causal CSSD +
hierarchical LDA under `models/finger_movements/cssd_lda/`. Its completed
Phase 2d retrospective official-TEST runner and evidence are archived under:

```text
history/finger_movements/experiments/phase2d_evaluate_frozen_test.py
history/finger_movements/results/phase2d_official_test_400ms/
```

Result: ToeplitzLDA reached the highest mean BA at 84.50%, but improved only
seeds 42/43, decreased seed 44 by 1.28 points, slightly reduced worst-seed BA,
and increased fold variability. Nested fusion reached 84.09% and did not
stabilize the result. No Phase 2e variant is promoted; the frozen Phase 2c
checkpoint remains unchanged.
