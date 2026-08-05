# Active Experiments

## Phase 1g: terminal feature contribution

`phase1g_terminal_feature_ablation.py` measures how much each part of the
frozen 252-dimensional terminal representation contributes:

- A: five final low-pass samples per channel (140 features);
- B: final 50/100/200 ms low-pass means per channel (84 features);
- C: final 200 ms low-pass slope per channel (28 features).

It evaluates all eight A/B/C subsets using the frozen Logistic Regression
configuration (`C=1`), seeds 42/43/44, and five-fold out-of-fold validation.
It reports standalone results, leave-one-group-out degradation, and exact
three-group Shapley contributions.

Run from the repository root:

```bash
python experiments/active/phase1g_terminal_feature_ablation.py
```

For a one-fold implementation check that does not write results:

```bash
python experiments/active/phase1g_terminal_feature_ablation.py --validate-only
```

Phase 1b through Phase 1f are complete and archived under
[`history/finger_movements/`](../../history/finger_movements/README.md). The
frozen model remains unchanged under
[`models/finger_movements/terminal_logistic/`](../../models/finger_movements/terminal_logistic/README.md).
The official test split remains locked and is not loaded by Phase 1g.
