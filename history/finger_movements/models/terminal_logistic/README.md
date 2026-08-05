# Archived FingerMovements Terminal Low-pass + Logistic

This is the frozen model completed in Phase 1h. It is preserved for
reproducibility and comparison; active code must not import it.

## Input and target

- Input: `float32` EEG with shape `(cases, 28, 50)`.
- Sampling: 100 Hz; each case spans 500 ms and ends 130 ms before keypress.
- Labels: `left=0`, `right=1`.
- Output: one signed linear score; a non-negative score predicts right.

## Frozen pipeline

1. Normalize each channel using training-derived statistics.
2. Apply a second-order causal 5 Hz low-pass IIR initialized from each trial's
   first sample.
3. Extract 252 terminal ABC features: five final samples, three terminal-window
   means, and one final 200 ms slope per channel.
4. Standardize features using training-derived statistics.
5. Apply L2 Logistic Regression with `C=1`.

## Checkpoint and evidence

Checkpoint:

```text
checkpoints/finger_movements_terminal_logistic_phase1h.npz
```

SHA-256:

```text
f8fca725c3b638219bbd734257cd958779e595add2fe1118e1e78689bc120047
```

The checkpoint contains the filter coefficients, normalization arrays, 252
weights, bias, label mapping, and input contract. Reloaded inference exactly
reproduced all training scores and predictions.

- mean OOF balanced accuracy: 68.89%;
- apparent all-training balanced accuracy: 78.49%, not held out;
- one-time official-test accuracy: 62.00%;
- one-time official-test balanced accuracy: 62.10%;
- one-time official-test macro-F1: 61.94%.

The model is an archived baseline rather than the final firmware candidate.
