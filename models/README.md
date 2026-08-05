# Models

There is currently no promoted active model.

The completed FingerMovements terminal low-pass + Logistic Regression model
and its exact Phase 1h checkpoint are archived under:

```text
history/finger_movements/models/terminal_logistic/
```

It remains the reproducible baseline for future training-only comparisons,
but its 62.00% official-test accuracy was not strong enough for final firmware
promotion. New active implementations must be self-contained and must not
import code from `history/`.
