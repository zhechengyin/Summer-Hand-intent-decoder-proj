"""Prespecified, bounded and identical search budgets for both architectures."""

import math
import statistics

MODELS = ("mingru", "mamba2")
SEEDS = (43, 44, 45)
FOLDS = (1, 2, 3, 4, 5)
SCREEN_FOLDS = (1,)
TOP_K = 2
FIXED = {
    "epochs": 40,
    "patience": 10,
    "batch_size": 128,
    "gradient_clip": 1.0,
    "initialization": "scratch",
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR(T_max=40)",
    "precision": "FP32",
    "checkpoint_selection": "minimum_validation_normalized_MSE",
    "trial_selection": "maximum_fold_macro_validation_R2_at_selected_checkpoint",
    "split_seed": 43,
}

# Anchor + five interpretable probes + six prespecified joint configurations.
# No trial is chosen using the observed test scores. Architecture sizes are fixed.
_ROWS = (
    (3e-4, 0.25, 0.025, 0.2, 0.1),
    (1e-4, 0.25, 0.025, 0.2, 0.1),
    (1e-3, 0.25, 0.025, 0.2, 0.1),
    (3e-4, 1.00, 0.025, 0.2, 0.1),
    (3e-4, 0.25, 0.001, 0.2, 0.1),
    (3e-4, 0.25, 0.025, 0.0, 0.0),
    (1e-4, 1.00, 0.010, 0.1, 0.0),
    (1e-4, 0.25, 0.001, 0.0, 0.1),
    (3e-4, 1.00, 0.050, 0.3, 0.2),
    (3e-4, 1.00, 0.001, 0.1, 0.2),
    (1e-3, 1.00, 0.010, 0.1, 0.1),
    (1e-3, 0.25, 0.050, 0.3, 0.2),
)
TRIALS = {
    f"t{index:02d}": dict(
        zip(
            (
                "learning_rate",
                "stem_lr_scale",
                "weight_decay",
                "channel_dropout",
                "dropout",
            ),
            row,
            strict=True,
        )
    )
    for index, row in enumerate(_ROWS)
}


def expected_fits(session_count=6):
    screen = len(MODELS) * len(TRIALS) * session_count * len(SCREEN_FOLDS)
    confirm = len(MODELS) * TOP_K * session_count * (len(FOLDS) - len(SCREEN_FOLDS))
    seeds = len(MODELS) * (len(SEEDS) - 1) * session_count * len(FOLDS)
    return {
        "screen": screen,
        "confirm": confirm,
        "seed_check": seeds,
        "total": screen + confirm + seeds,
    }


def rank_trials(receipts, trials, sessions, folds, seed=43):
    """Reject incomplete/duplicate evidence; never consult test fields."""
    expected = {(s, f) for s in sessions for f in folds}
    ranked = []
    for trial in trials:
        selected = [r for r in receipts if r["trial"] == trial and r["seed"] == seed]
        identities = [(r["session"], r["fold"]) for r in selected]
        if len(identities) != len(set(identities)) or set(identities) != expected:
            raise ValueError(f"Incomplete or duplicate validation folds: {trial}")
        values = [float(r["validation"]["r2_mean"]) for r in selected]
        losses = [float(r["validation"]["normalized_loss"]) for r in selected]
        if not all(math.isfinite(x) for x in values + losses):
            raise ValueError("Nonfinite validation selection evidence")
        ranked.append(
            {
                "trial": trial,
                "folds": len(selected),
                "validation_r2_mean": statistics.mean(values),
                "validation_r2_sd": statistics.stdev(values)
                if len(values) > 1
                else 0.0,
                "validation_loss_mean": statistics.mean(losses),
                "worst_fold_validation_r2": min(values),
            }
        )
    return sorted(
        ranked,
        key=lambda r: (-r["validation_r2_mean"], r["validation_loss_mean"], r["trial"]),
    )


def require_test_gate(finalists, rows, sessions):
    """All finalists and seeds must exist before any test prediction is allowed."""
    expected = {
        (model, trial, seed, session, fold)
        for model, trial in finalists.items()
        for seed in SEEDS
        for session in sessions
        for fold in FOLDS
    }
    identities = [
        (r["model"], r["trial"], r["seed"], r["session"], r["fold"]) for r in rows
    ]
    if (
        set(finalists) != set(MODELS)
        or len(identities) != len(expected)
        or set(identities) != expected
    ):
        raise ValueError(
            "Test remains closed: missing/duplicate finalist folds or seeds"
        )
