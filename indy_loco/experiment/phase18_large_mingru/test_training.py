"""Validation-only selection, executable fit, and saved-run integrity checks."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from . import plan
from . import train as runner


class TrainingTests(unittest.TestCase):
    def test_prespecified_budget_and_phase17_validation_anchor(self):
        self.assertEqual(plan.MODELS, ("mingru_large",))
        self.assertEqual(plan.SEEDS, (43,))
        self.assertEqual(tuple(plan.TRIALS), tuple(f"t{i:02d}" for i in range(6)))
        self.assertEqual(plan.FIXED["epochs"], 60)
        self.assertEqual(plan.FIXED["patience"], 15)
        self.assertEqual(plan.FIXED["split_seed"], 43)
        self.assertEqual(plan.expected_fits()["total"], 84)
        self.assertEqual(
            plan.TRIALS["t00"],
            {
                "learning_rate": 1e-3,
                "stem_lr_scale": 1.0,
                "weight_decay": 0.01,
                "channel_dropout": 0.1,
                "dropout": 0.1,
            },
        )

    def test_rank_uses_validation_and_rejects_missing_duplicate_nonfinite_rows(self):
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

        def rank(values):
            return plan.rank_trials(values, ["a", "b"], ["indy", "loco"], [1])

        self.assertEqual(rank(rows)[0]["trial"], "a")
        changed_test = copy.deepcopy(rows)
        for row in changed_test:
            row["test"]["r2_mean"] *= -1e6
        self.assertEqual(rank(rows), rank(changed_test))
        for invalid in (rows[:-1], rows + rows[:1]):
            with self.assertRaises(ValueError):
                rank(invalid)
        invalid = copy.deepcopy(rows)
        invalid[0]["validation"]["normalized_loss"] = float("nan")
        with self.assertRaises(ValueError):
            rank(invalid)

    def test_test_gate_requires_thirty_unique_selected_folds(self):
        finalists = {"mingru_large": "t00"}
        rows = [
            {"model": m, "trial": t, "seed": seed, "session": s, "fold": f}
            for m, t in finalists.items()
            for seed in plan.SEEDS
            for s in runner.SESSIONS
            for f in plan.FOLDS
        ]
        self.assertEqual(len(rows), 30)
        plan.require_test_gate(finalists, rows, runner.SESSIONS)
        for invalid in (rows[:-1], rows + rows[:1]):
            with self.assertRaises(ValueError):
                plan.require_test_gate(finalists, invalid, runner.SESSIONS)
        for field, wrong in (("seed", 44), ("trial", "t01"), ("model", "mingru")):
            invalid = copy.deepcopy(rows)
            invalid[0][field] = wrong
            with self.assertRaises(ValueError):
                plan.require_test_gate(finalists, invalid, runner.SESSIONS)

    def test_dropout_lr_groups_and_capacity_reach_real_model(self):
        for trial_id, trial in plan.TRIALS.items():
            with self.subTest(trial=trial_id):
                model = runner.create_model("mingru_large", trial)
                groups = runner.optimizer_groups(model, "mingru_large", trial)
                self.assertEqual(
                    model.channel_dropout.probability, trial["channel_dropout"]
                )
                self.assertEqual(model.dropout.p, trial["dropout"])
                self.assertEqual(sum(p.numel() for p in model.parameters()), 1_211_906)
                parameter_ids = [id(p) for group in groups for p in group["params"]]
                self.assertEqual(len(parameter_ids), len(set(parameter_ids)))
                self.assertEqual(
                    set(parameter_ids), {id(p) for p in model.parameters()}
                )
                self.assertIn("encoder_stem", {group["name"] for group in groups})
                for group in groups:
                    scale = (
                        trial["stem_lr_scale"]
                        if group["name"] == "encoder_stem"
                        else 1.0
                    )
                    self.assertAlmostEqual(group["lr"], trial["learning_rate"] * scale)

    def test_real_fit_ignores_poisoned_test_and_selects_minimum_validation_mse(self):
        old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old_threads)
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
        trial = plan.TRIALS["t00"]
        scores = iter(
            [
                {"normalized_loss": 0.2, "r2_mean": 0.4},
                {"normalized_loss": 0.3, "r2_mean": 0.8},
            ]
        )
        validation_states = []

        def validation(model, features, velocity, bins, *args):
            np.testing.assert_array_equal(bins, data.validation_bins)
            validation_states.append(
                {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
            )
            return next(scores)

        with tempfile.TemporaryDirectory() as temporary:
            epoch_path = Path(temporary) / "epochs.json"
            with (
                patch.dict(plan.FIXED, {"epochs": 2, "patience": 2}),
                patch.object(
                    runner.protocol, "evaluate_last", side_effect=validation
                ) as evaluate,
            ):
                result = runner.fit(
                    runner.create_model("mingru_large", trial),
                    "mingru_large",
                    trial,
                    43,
                    data,
                    torch.device("cpu"),
                    epoch_path,
                    "toy",
                )
            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual(result["best_epoch"], 1)
            self.assertEqual(result["validation"]["r2_mean"], 0.4)
            self.assertEqual(result["best_validation_loss"], 0.2)
            self.assertEqual(result["arithmetic_check"]["validation_windows"], 3)
            self.assertFalse(result["arithmetic_check"]["persistent_state"])
            self.assertEqual(len(result["history"]), 2)
            self.assertEqual(len(runner.read_json(epoch_path)), 2)
            for key, value in result["model_state"].items():
                self.assertTrue(torch.isfinite(value).all())
                torch.testing.assert_close(
                    value, validation_states[0][key], rtol=0, atol=0
                )
            self.assertTrue(
                any(
                    not torch.equal(value, validation_states[1][key])
                    for key, value in validation_states[0].items()
                )
            )

    def test_saved_identity_preprocessing_and_test_isolation_are_required(self):
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
        for declaration in (True, None):
            with self.assertRaises(ValueError):
                runner.check_saved(
                    {**saved, "test_evaluated_during_training": declaration},
                    {},
                    {"hash": "x"},
                )

    def test_resume_verifies_checkpoint_and_receipt_without_training(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(runner, "OUTPUT", Path(temporary)),
        ):
            sweep = runner.Sweep({}, torch.device("cpu"))
            session = runner.SESSIONS[0]
            args = ("mingru_large", "t00", 43, session, 1)
            evidence = {"hash": "frozen-preprocessing"}
            saved = {
                **runner.identity(*args, sweep.signature),
                "preprocessing_evidence": evidence,
                "test_evaluated_during_training": False,
                "validation": {"r2_mean": 0.6, "normalized_loss": 0.2},
                "best_epoch": 1,
                "training_seconds": 0.1,
                "arithmetic_check": {"persistent_state": False},
                "model_state": {"toy": torch.tensor([1.0])},
            }
            path = runner.checkpoint_path(*args)
            runner.original.save_training_checkpoint(path, saved)
            with (
                patch.object(runner, "prepare", return_value=(None, evidence)),
                patch.object(
                    runner, "fit", side_effect=AssertionError("Resume retrained")
                ),
            ):
                receipt = sweep.ensure_fit(*args, "screen")
                self.assertEqual(
                    receipt["checkpoint_sha256"], runner.protocol.sha256_file(path)
                )
                self.assertEqual(sweep.ensure_fit(*args, "screen"), receipt)
                receipt_path = path.with_suffix(".validation.json")
                modified = copy.deepcopy(receipt)
                modified["validation"]["r2_mean"] = 0.99
                runner.protocol.write_json_atomic(receipt_path, modified)
                with self.assertRaisesRegex(ValueError, "receipt"):
                    sweep.ensure_fit(*args, "screen")
                runner.protocol.write_json_atomic(receipt_path, receipt)
                with path.open("ab") as handle:
                    handle.write(b"tampered-checkpoint")
                with self.assertRaisesRegex(ValueError, "SHA|hash"):
                    sweep.ensure_fit(*args, "screen")

    def test_frozen_selection_cannot_be_replaced(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(runner, "OUTPUT", Path(temporary)),
        ):
            sweep = runner.Sweep({}, torch.device("cpu"))
            sweep.freeze("selection.json", {"trial": "t00"})
            sweep.freeze("selection.json", {"trial": "t00"})
            with self.assertRaises(ValueError):
                sweep.freeze("selection.json", {"trial": "t01"})

    def test_entire_schedule_finishes_84_unique_fits_before_30_tests(self):
        rows, identities = [], set()
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
                    "checkpoint": "/".join(str(value) for value in key),
                    "checkpoint_sha256": "hash",
                    "validation": {
                        "r2_mean": 1 - int(trial[1:]) / 100,
                        "normalized_loss": 0.1,
                    },
                }
            )

        def collect(name, trials, folds, seeds=(43,)):
            return [
                row
                for row in rows
                if (
                    row["model"] == name
                    and row["trial"] in trials
                    and row["fold"] in folds
                    and row["seed"] in seeds
                )
            ]

        def final_test(selection, final_rows):
            self.assertEqual(len(rows), 84)
            self.assertEqual(len(final_rows), 30)
            self.assertEqual(selection["finalists"], {"mingru_large": "t00"})
            self.assertFalse(selection["test_used_for_selection"])
            gate = runner.read_json(runner.OUTPUT / "test_gate.json")
            self.assertEqual(gate["verified_finalist_checkpoints"], 30)
            self.assertEqual(len(gate["checkpoints"]), 30)
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


if __name__ == "__main__":
    unittest.main()
