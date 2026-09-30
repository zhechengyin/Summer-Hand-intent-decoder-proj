"""Prespecified A/B comparison: one screened LR and weight policy per architecture."""

import math
import statistics

MODELS = ("mingru_a", "mingru_b")
SEEDS = (43,)
FOLDS = (1, 2, 3, 4, 5)
SCREEN_FOLDS = (1,)
TOP_K = 1
WEIGHT_POLICIES = ("raw", "ema")
EMA_DECAY = 0.99
FIXED = {
    "epochs": 60,
    "patience": 15,
    "batch_size": 128,
    "gradient_clip": 1.0,
    "initialization": "scratch",
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR(T_max=60)",
    "precision": "FP32",
    "checkpoint_selection": "minimum_validation_normalized_MSE_separately_for_raw_and_ema",
    "early_stopping": "15_epochs_without_either_policy_improving_its_own_best",
    "ema_decay_per_optimizer_step": EMA_DECAY,
    "trial_selection": "maximum_fold_macro_validation_R2_at_selected_checkpoint",
    "split_seed": 43,
}

# Both architectures receive the same two learning rates and regularization.
TRIALS = {
    f"t{index:02d}": {
        "learning_rate": lr,
        "stem_lr_scale": 1.0,
        "weight_decay": 0.01,
        "head_weight_decay": 0.03,
        "channel_dropout": 0.1,
        "dropout": 0.1,
    }
    for index, lr in enumerate((1e-3, 6e-4))
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


def rank_trials(
    receipts, trials, sessions, folds, seed=43, weight_policies=WEIGHT_POLICIES
):
    """Rank complete LR/policy pairs using only selected-checkpoint validation."""
    expected = {(s, f) for s in sessions for f in folds}
    ranked = []
    for trial in trials:
        selected = [r for r in receipts if r["trial"] == trial and r["seed"] == seed]
        identities = [(r["session"], r["fold"]) for r in selected]
        if len(identities) != len(set(identities)) or set(identities) != expected:
            raise ValueError(f"Incomplete or duplicate validation folds: {trial}")
        for policy in weight_policies:
            if policy not in WEIGHT_POLICIES:
                raise ValueError("Unknown weight policy")
            scores = [r["policy_validation"][policy]["validation"] for r in selected]
            values = [float(r["r2_mean"]) for r in scores]
            losses = [float(r["normalized_loss"]) for r in scores]
            if not all(math.isfinite(x) for x in values + losses):
                raise ValueError("Nonfinite validation selection evidence")
            ranked.append(
                {
                    "trial": trial,
                    "weight_policy": policy,
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
        key=lambda r: (
            -r["validation_r2_mean"],
            r["validation_loss_mean"],
            r["trial"],
            WEIGHT_POLICIES.index(r["weight_policy"]),
        ),
    )


def require_test_gate(finalists, rows, sessions, weight_policies=None):
    """Both architectures need all 30 verified folds before either test opens."""
    expected = {
        (m, t, seed, s, f)
        for m, t in finalists.items()
        for seed in SEEDS
        for s in sessions
        for f in FOLDS
    }
    identities = [
        (r["model"], r["trial"], r["seed"], r["session"], r["fold"]) for r in rows
    ]
    if (
        set(finalists) != set(MODELS)
        or len(identities) != len(expected)
        or set(identities) != expected
    ):
        raise ValueError("Test remains closed: missing/duplicate finalist folds")
    if weight_policies is not None:
        if set(weight_policies) != set(MODELS) or any(
            v not in WEIGHT_POLICIES for v in weight_policies.values()
        ):
            raise ValueError("Invalid frozen weight policies")
        if any(r.get("weight_policy") != weight_policies[r["model"]] for r in rows):
            raise ValueError("Finalist weight policy differs from frozen selection")
