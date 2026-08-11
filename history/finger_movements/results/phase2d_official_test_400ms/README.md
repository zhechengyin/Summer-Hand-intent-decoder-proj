# Phase 2d frozen official-TEST inference

This directory records Phase 2d pure inference of the frozen Phase 2c 400 ms
causal checkpoint on the 100 corrected official TEST cases. No fitting,
recalibration, threshold selection, or TEST-derived preprocessing was
performed.

| Metric | Result |
|---|---:|
| Accuracy | 77.00% |
| Balanced accuracy | 77.05% |
| Macro-F1 | 77.00% |
| Left recall | 79.59% |
| Right recall | 74.51% |
| Accuracy Wilson 95% CI | 67.85%--84.16% |

Confusion matrix, with rows=true and columns=predicted in `[left, right]`
order:

```text
[[39, 10],
 [13, 38]]
```

The TEST balanced accuracy is 6.94 percentage points below the frozen 83.99%
TRAIN-only OOF estimate. With only 100 TEST cases, the confidence interval is
wide. The result does not show a one-class collapse.

This is a retrospective benchmark, not a pristine blind test: official TEST
was already exposed earlier in the project. The result must not be used to
retune the frozen model.

Files:

- `phase2d_official_test_metrics.json`: hashes, inference contract, streaming
  equivalence, and aggregate metrics;
- `phase2d_official_test_predictions.csv`: all 100 predictions and scores.
