"""Prespecified EMA-only local tuning, with cached B as an immutable control."""

import copy
import math
import statistics

from indy_loco.experiment.phase18_large_mingru.ab_study import plan as previous

MODEL = "mingru_b"
BASELINE = "b0"
NEW_TRIALS = ("b1", "b2", "b3", "b4", "b5")
SEEDS = (43,)
FOLDS = previous.FOLDS
SCREEN_FOLDS = previous.SCREEN_FOLDS
TOP_K = 2
WEIGHT_POLICY = "ema"
EMA_DECAY = previous.EMA_DECAY
FIXED = copy.deepcopy(previous.FIXED)
TRIALS = {key: copy.deepcopy(previous.TRIALS["t00"]) for key in (BASELINE, *NEW_TRIALS)}
TRIALS["b1"]["learning_rate"] = 0.0008
TRIALS["b2"]["learning_rate"] = 0.0012
TRIALS["b3"]["weight_decay"] = 0.003
TRIALS["b4"]["weight_decay"] = 0.03
TRIALS["b5"]["head_weight_decay"] = 0.01


def expected_fits(session_count=6):
    screen = len(NEW_TRIALS) * session_count * len(SCREEN_FOLDS)
    confirm = TOP_K * session_count * (len(FOLDS) - len(SCREEN_FOLDS))
    return {"screen": screen, "confirm": confirm, "total": screen + confirm}


def rank_trials(rows, trials, sessions, folds, seed=43):
    """Rank complete, matched EMA validation receipts; never consult test data."""
    expected = {(session, fold) for session in sessions for fold in folds}
    ranking = []
    for trial in trials:
        selected = [
            row for row in rows if row["trial"] == trial and row["seed"] == seed
        ]
        identities = [(row["session"], row["fold"]) for row in selected]
        if len(identities) != len(expected) or set(identities) != expected:
            raise ValueError(f"Incomplete/duplicate validation folds: {trial}")
        if any(
            row.get("weight_policy") != WEIGHT_POLICY or row.get("model") != MODEL
            for row in selected
        ):
            raise ValueError(
                "Only the prescribed B architecture and EMA weights may enter ranking"
            )
        values = [float(row["validation"]["r2_mean"]) for row in selected]
        losses = [float(row["validation"]["normalized_loss"]) for row in selected]
        if not all(math.isfinite(value) for value in values + losses):
            raise ValueError("Nonfinite validation evidence")
        ranking.append(
            {
                "trial": trial,
                "weight_policy": WEIGHT_POLICY,
                "folds": len(values),
                "validation_r2_mean": statistics.mean(values),
                "validation_r2_sample_sd": statistics.stdev(values)
                if len(values) > 1
                else 0.0,
                "validation_loss_mean": statistics.mean(losses),
                "worst_fold_validation_r2": min(values),
            }
        )
    return sorted(
        ranking,
        key=lambda row: (
            -row["validation_r2_mean"],
            row["trial"] != BASELINE,
            row["validation_loss_mean"],
            row["trial"],
        ),
    )


def promote_candidates(rows, sessions):
    return [
        row["trial"]
        for row in rank_trials(rows, NEW_TRIALS, sessions, SCREEN_FOLDS)[:TOP_K]
    ]


def require_test_gate(finalists, rows, sessions):
    """Require all 30 EMA folds for B0 and both fully confirmed new candidates."""
    if (
        len(finalists) != TOP_K + 1
        or len(set(finalists)) != len(finalists)
        or BASELINE not in finalists
        or not set(finalists).issubset(TRIALS)
    ):
        raise ValueError("Test remains closed: expected B0 and two new finalists")
    expected = {
        (trial, seed, session, fold)
        for trial in finalists
        for seed in SEEDS
        for session in sessions
        for fold in FOLDS
    }
    identities = [
        (row["trial"], row["seed"], row["session"], row["fold"]) for row in rows
    ]
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError("Test remains closed: missing/duplicate finalist folds")
    if any(
        row.get("weight_policy") != WEIGHT_POLICY or row.get("model") != MODEL
        for row in rows
    ):
        raise ValueError("Finalist architecture or frozen EMA policy changed")
