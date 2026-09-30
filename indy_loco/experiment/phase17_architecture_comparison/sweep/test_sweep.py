"""Selection isolation, executable fit, resume integrity and pipeline gate checks."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from . import plan
from . import run_sweep as runner


class SweepTests(unittest.TestCase):
    def test_rank_never_uses_test_and_rejects_incomplete_evidence(self):
        rows = [
            {
                "trial": trial,
                "seed": 43,
                "session": session,
                "fold": 1,
                "validation": {"r2_mean": val, "normalized_loss": loss},
                "test": {"r2_mean": test},
            }
            for trial, val, loss, test in (("a", 0.6, 0.3, -0.9), ("b", 0.5, 0.4, 0.99))
            for session in ("indy", "loco")
        ]
        self.assertEqual(
            plan.rank_trials(rows, ["a", "b"], ["indy", "loco"], [1])[0]["trial"], "a"
        )
        with self.assertRaises(ValueError):
            plan.rank_trials(rows[:-1], ["a", "b"], ["indy", "loco"], [1])
        with self.assertRaises(ValueError):
            plan.rank_trials(rows + rows[:1], ["a", "b"], ["indy", "loco"], [1])

    def test_test_gate_requires_every_model_seed_and_fold(self):
        finalists = {"mingru": "t00", "mamba2": "t01"}
        rows = [
            {"model": m, "trial": t, "seed": seed, "session": s, "fold": f}
            for m, t in finalists.items()
            for seed in plan.SEEDS
            for s in runner.SESSIONS
            for f in plan.FOLDS
        ]
        plan.require_test_gate(finalists, rows, runner.SESSIONS)
        with self.assertRaises(ValueError):
            plan.require_test_gate(finalists, rows[:-1], runner.SESSIONS)
        with self.assertRaises(ValueError):
            plan.require_test_gate(finalists, rows + rows[:1], runner.SESSIONS)

    def test_dropout_and_learning_rates_reach_actual_model_optimizer(self):
        for name in plan.MODELS:
            trial = plan.TRIALS["t08"]
            model = runner.create_model(name, trial)
            groups = runner.optimizer_groups(model, name, trial)
            self.assertEqual(model.channel_dropout.probability, 0.3)
            self.assertEqual(model.dropout.p, 0.2)
            self.assertEqual(
                sum(p.numel() for p in model.parameters()),
                runner.models.EXPECTED_PARAMETERS[name],
            )
            for group in groups:
                self.assertEqual(group["lr"], 3e-4)
            self.assertEqual(
                {id(p) for g in groups for p in g["params"]},
                {id(p) for p in model.parameters()},
            )

    def test_real_fit_avoids_poisoned_test_targets_and_selects_by_validation_loss(self):
        torch.set_num_threads(2)
        rng = np.random.default_rng(1)
        data = SimpleNamespace(
            normalized_features=rng.normal(size=(192, 100)).astype(np.float32),
            velocity=rng.normal(size=(100, 2)).astype(np.float32),
            target_mean=np.zeros(2, dtype=np.float32),
            target_std=np.ones(2, dtype=np.float32),
            train_bins=np.arange(49, 53),
            validation_bins=np.arange(60, 63),
            test_bins=np.arange(80, 83),
        )
        data.velocity[data.test_bins] = np.nan
        with tempfile.TemporaryDirectory() as temporary:
            for name in plan.MODELS:
                trial = plan.TRIALS["t00"]
                scores = iter(
                    [
                        {"normalized_loss": 0.2, "r2_mean": 0.4},
                        {"normalized_loss": 0.3, "r2_mean": 0.8},
                    ]
                )

                def validation(model, features, velocity, bins, *args, scores=scores):
                    np.testing.assert_array_equal(bins, data.validation_bins)
                    return next(scores)

                with (
                    patch.dict(plan.FIXED, {"epochs": 2, "patience": 2}),
                    patch.object(
                        runner.protocol, "evaluate_last", side_effect=validation
                    ),
                ):
                    result = runner.fit(
                        runner.create_model(name, trial),
                        name,
                        trial,
                        43,
                        data,
                        torch.device("cpu"),
                        Path(temporary) / f"{name}.json",
                        name,
                    )
                self.assertEqual(result["best_epoch"], 1)
                self.assertEqual(result["validation"]["r2_mean"], 0.4)
                self.assertEqual(len(result["history"]), 2)
                self.assertTrue(
                    all(torch.isfinite(v).all() for v in result["model_state"].values())
                )

    def test_resume_rejects_wrong_seed_signature_or_preprocessing(self):
        saved = {
            "seed": 43,
            "signature": "fixed",
            "preprocessing_evidence": {"hash": "x"},
            "test_evaluated_during_training": False,
            "validation": {"r2_mean": 0.6, "normalized_loss": 0.2},
        }
        runner.check_saved(saved, {"seed": 43, "signature": "fixed"}, {"hash": "x"})
        for expected, evidence in (
            ({"seed": 44}, {"hash": "x"}),
            ({"signature": "new"}, {"hash": "x"}),
            ({"seed": 43}, {"hash": "y"}),
        ):
            with self.assertRaises(ValueError):
                runner.check_saved(saved, expected, evidence)

    def test_frozen_selection_cannot_be_replaced(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(runner, "OUTPUT", Path(temporary)),
        ):
            sweep = runner.Sweep({}, torch.device("cpu"))
            sweep.freeze("selection.json", {"winner": "mingru"})
            sweep.freeze("selection.json", {"winner": "mingru"})
            with self.assertRaises(ValueError):
                sweep.freeze("selection.json", {"winner": "mamba2"})

    def test_entire_schedule_runs_360_unique_fits_before_test(self):
        rows = []
        identities = set()
        sweep = runner.Sweep({}, torch.device("cpu"))

        def fake_fit(name, trial, seed, session, fold, stage):
            key = name, trial, seed, session, fold
            self.assertNotIn(key, identities)
            identities.add(key)
            rows.append(
                {
                    "model": name,
                    "trial": trial,
                    "seed": seed,
                    "session": session,
                    "fold": fold,
                    "checkpoint": "/".join(str(v) for v in key),
                    "checkpoint_sha256": "hash",
                    "validation": {
                        "r2_mean": 1 - int(trial[1:]) / 100,
                        "normalized_loss": 0.1,
                    },
                }
            )

        def collect(name, trials, folds, seeds=(43,)):
            return [
                r
                for r in rows
                if r["model"] == name
                and r["trial"] in trials
                and r["fold"] in folds
                and r["seed"] in seeds
            ]

        def final_test(selection, final_rows):
            self.assertEqual(len(rows), 360)
            self.assertEqual(len(final_rows), 180)
            self.assertEqual(selection["finalists"], {"mingru": "t00", "mamba2": "t00"})
            plan.require_test_gate(selection["finalists"], final_rows, runner.SESSIONS)

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(runner, "OUTPUT", Path(temporary)),
            patch.object(sweep, "ensure_fit", side_effect=fake_fit),
            patch.object(sweep, "collect", side_effect=collect),
            patch.object(sweep, "assert_unchanged"),
            patch.object(sweep, "test_finalists", side_effect=final_test) as evaluate,
        ):
            sweep.run()
            evaluate.assert_called_once()
        self.assertEqual(plan.expected_fits()["total"], 360)


if __name__ == "__main__":
    unittest.main()
