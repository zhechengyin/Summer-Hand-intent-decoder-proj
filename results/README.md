# Results

The active FingerMovements result is:

```text
results/finger_movements/phasea2_cssd_lda/
```

It contains fold metrics, seed-level OOF metrics, per-case OOF predictions,
three branch scores, and the original summary figure. The `phasea2_diagnostic`
files add train-versus-validation layer metrics, all seven branch ablations,
inner-OOF fusion, cross-fold CSSD subspace stability, per-case seed stability,
and a generalization-diagnostic figure. The official test is not loaded by
Phase A2.

The complete earlier FingerMovements direction, including its official-test
result, remains under `history/finger_movements/results/`. Completed Indy
results remain under `history/indy/results/`.
