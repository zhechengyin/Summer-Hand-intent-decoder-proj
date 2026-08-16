# Phase 2d Frozen Official TEST Evaluation

The frozen 400 ms causal checkpoint was applied once to the corrected 100-case official TEST with no fitting, recalibration, or TEST-derived preprocessing.

| Metric | Result |
|---|---:|
| Accuracy | 77.00% |
| Balanced accuracy | 77.05% |
| Macro-F1 | 77.00% |
| Confusion matrix | `[[39, 10], [13, 38]]` |

Batch and stateful 50 ms-chunk inference were numerically equivalent. This is a retrospective benchmark, not a pristine blind result, because TEST had been exposed during the earlier invalid-source phase. It must not be used to retune the model.
