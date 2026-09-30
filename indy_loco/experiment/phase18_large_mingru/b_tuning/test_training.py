"""Fixed-B tuning selection, baseline isolation, and checkpoint contracts."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from ..ab_study import train as ab
from . import plan
from . import train as runner


def score(r2, loss=0.2):
    return {"r2_mean": r2, "normalized_loss": loss}


def receipt(trial, session, fold, r2=0.7, loss=0.2):
    return {
        "model": "mingru_b",
        "trial": trial,
        "session": session,
        "fold": fold,
        "seed": 43,
        "weight_policy": "ema",
        "validation": score(r2, loss),
    }


def fitted_state():
    policies = {
        policy: {
            "model_state": {"toy": torch.tensor([1.0])},
            "validation": score(0.6 if policy == "raw" else 0.7),
            "best_epoch": 1,
            "best_validation_loss": 0.2,
            "arithmetic_check": {"persistent_state": False},
        }
        for policy in ("raw", "ema")
    }
    return {
        **policies["raw"],
        "weight_policies": policies,
        "history": [{"epoch": 1}],
        "training_seconds": 0.1,
    }


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_prespecified_five_one_factor_changes_and_fixed_budget(self):
        self.assertEqual(plan.MODEL, "mingru_b")
        self.assertEqual(plan.BASELINE, "b0")
        self.assertEqual(plan.NEW_TRIALS, ("b1", "b2", "b3", "b4", "b5"))
        self.assertEqual(set(plan.TRIALS), {"b0", *plan.NEW_TRIALS})
        self.assertEqual(plan.TRIALS["b0"], ab.plan.TRIALS["t00"])
        self.assertEqual(plan.FIXED, ab.plan.FIXED)
        expected = {
            "b1": ("learning_rate", 0.0008),
            "b2": ("learning_rate", 0.0012),
            "b3": ("weight_decay", 0.003),
            "b4": ("weight_decay", 0.03),
            "b5": ("head_weight_decay", 0.01),
        }
        for trial, (key, value) in expected.items():
            self.assertEqual(plan.TRIALS[trial], {**plan.TRIALS["b0"], key: value})
        budget = plan.expected_fits()
        self.assertEqual(budget["screen"], 30)
        self.assertEqual(budget["confirm"], 48)
        self.assertEqual(budget["total"], 78)

    def test_validation_ranking_ignores_test_and_rejects_bad_evidence(self):
        sessions = ("indy", "loco")
        rows = [
            {**receipt(trial, session, 1, r2), "test": score(test)}
            for trial, r2, test in (("b0", 0.70, 0.99), ("b1", 0.72, -0.99))
            for session in sessions
        ]

        def rank(values):
            return plan.rank_trials(values, ("b0", "b1"), sessions, (1,))

        self.assertEqual(rank(rows)[0]["trial"], "b1")
        changed_test = copy.deepcopy(rows)
        for row in changed_test:
            row["test"] = score(1e20)
        self.assertEqual(rank(rows), rank(changed_test))
        for invalid in (rows[:-1], rows + rows[:1]):
            with self.assertRaises(ValueError):
                rank(invalid)
        for field, value in (
            ("r2_mean", float("nan")),
            ("normalized_loss", float("inf")),
        ):
            invalid = copy.deepcopy(rows)
            invalid[0]["validation"][field] = value
            with self.assertRaises(ValueError):
                rank(invalid)
        for key, value in (
            ("weight_policy", "raw"),
            ("seed", 44),
            ("model", "mingru_a"),
        ):
            invalid = copy.deepcopy(rows)
            invalid[0][key] = value
            with self.assertRaises(ValueError):
                rank(invalid)

    def test_equal_validation_retains_existing_baseline(self):
        rows = [receipt(trial, "indy", 1) for trial in ("b2", "b1", "b0")]
        self.assertEqual(
            plan.rank_trials(rows, ("b2", "b1", "b0"), ("indy",), (1,))[0]["trial"],
            "b0",
        )

    def test_test_gate_requires_baseline_plus_two_complete_new_candidates(self):
        finalists = ["b0", "b1", "b2"]
        rows = [
            receipt(trial, session, fold)
            for trial in finalists
            for session in runner.SESSIONS
            for fold in plan.FOLDS
        ]
        self.assertEqual(len(rows), 90)
        plan.require_test_gate(finalists, rows, runner.SESSIONS)
        for invalid in (rows[:-1], rows + rows[:1], rows[30:]):
            with self.assertRaises(ValueError):
                plan.require_test_gate(finalists, invalid, runner.SESSIONS)
        for key, value in (
            ("weight_policy", "raw"),
            ("seed", 44),
            ("model", "mingru_a"),
        ):
            invalid = copy.deepcopy(rows)
            invalid[0][key] = value
            with self.assertRaises(ValueError):
                plan.require_test_gate(finalists, invalid, runner.SESSIONS)
        for candidates in (["b1", "b2", "b3"], ["b0", "b1"], ["b0", "b1", "b1"]):
            with self.assertRaises(ValueError):
                plan.require_test_gate(candidates, rows, runner.SESSIONS)

    def test_model_and_optimizer_match_baseline_except_specified_hyperparameter(self):
        reference = None
        for trial in plan.TRIALS.values():
            ab.protocol.seed_everything(43, mps=False)
            model = ab.create_model(plan.MODEL, trial)
            self.assertEqual(sum(p.numel() for p in model.parameters()), 374402)
            if reference is None:
                reference = {
                    key: value.clone() for key, value in model.state_dict().items()
                }
            else:
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, reference[key], rtol=0, atol=0)
            groups = ab.optimizer_groups(model, plan.MODEL, trial)
            optimizer = torch.optim.AdamW(groups, weight_decay=trial["weight_decay"])
            ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(set(ids), {id(p) for p in model.parameters()})
            for group in optimizer.param_groups:
                self.assertEqual(group["lr"], trial["learning_rate"])
                self.assertEqual(
                    group["weight_decay"],
                    trial["head_weight_decay"]
                    if group["name"] == "output_head"
                    else trial["weight_decay"],
                )

    def test_frozen_selection_cannot_change_and_stop_precedes_data_loading(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(runner, "OUTPUT", Path(temporary)),
        ):
            sweep = runner.Sweep({}, torch.device("cpu"))
            selection = {"trial": "b1", "weight_policy": "ema"}
            sweep.freeze("selection.json", selection)
            sweep.freeze("selection.json", selection)
            with self.assertRaises(ValueError):
                sweep.freeze("selection.json", {**selection, "trial": "b2"})
            (runner.OUTPUT / "STOP_AFTER_FOLD").touch()
            with patch.object(runner, "prepare") as prepare:
                with self.assertRaises(runner.StopRequested):
                    sweep.ensure_fit("b1", 43, runner.SESSIONS[0], 1, "screen")
                prepare.assert_not_called()

    def test_resume_verifies_recipe_receipt_and_checkpoint_without_retraining(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(runner, "REPO", Path(temporary)),
            patch.object(runner, "OUTPUT", Path(temporary) / "run"),
        ):
            sweep = runner.Sweep({}, torch.device("cpu"))
            args = ("b1", 43, runner.SESSIONS[0], 1)
            evidence = {"hash": "fixed"}
            saved = {
                **runner.identity(*args, sweep.signature),
                **fitted_state(),
                "hyperparameters": plan.TRIALS["b1"],
                "training": plan.FIXED,
                "preprocessing_evidence": evidence,
                "test_evaluated_during_training": False,
            }
            path = runner.checkpoint_path(*args)
            runner.original.save_training_checkpoint(path, saved)
            with (
                patch.object(runner, "prepare", return_value=(None, evidence)),
                patch.object(
                    runner, "fit", side_effect=AssertionError("Unexpected retraining")
                ),
            ):
                row = sweep.ensure_fit(*args, "screen")
                self.assertEqual(row["weight_policy"], "ema")
                self.assertEqual(row["validation"]["r2_mean"], 0.7)
                self.assertEqual(sweep.ensure_fit(*args, "screen"), row)
                original_receipt = runner.read_json(
                    path.with_suffix(".validation.json")
                )
                changed = copy.deepcopy(original_receipt)
                changed["policy_validation"]["ema"]["validation"]["r2_mean"] = 0.99
                runner.protocol.write_json_atomic(
                    path.with_suffix(".validation.json"), changed
                )
                with self.assertRaisesRegex(ValueError, "receipt"):
                    sweep.ensure_fit(*args, "screen")
                runner.protocol.write_json_atomic(
                    path.with_suffix(".validation.json"), original_receipt
                )
                with path.open("ab") as stream:
                    stream.write(b"tampered")
                with self.assertRaisesRegex(ValueError, "SHA|hash"):
                    sweep.ensure_fit(*args, "screen")
            for field in ("hyperparameters", "training"):
                changed = {**saved, field: {}}
                with self.assertRaises(ValueError):
                    runner.check_saved(
                        changed, runner.identity(*args, sweep.signature), evidence
                    )

    def test_baseline_cannot_be_retrained_and_source_recipe_is_checked(self):
        sweep = runner.Sweep({}, torch.device("cpu"))
        with (
            patch.object(runner, "prepare") as prepare,
            patch.object(runner, "fit") as fit,
        ):
            with self.assertRaises(ValueError):
                sweep.ensure_fit("b0", 43, runner.SESSIONS[0], 1, "screen")
            prepare.assert_not_called()
            fit.assert_not_called()
        source = {"frozen": "source"}
        session = runner.SESSIONS[0]
        evidence = {"hash": "frozen"}
        saved = {
            **ab.identity(plan.MODEL, "t00", 43, session, 1, runner.digest(source)),
            **fitted_state(),
            "hyperparameters": plan.TRIALS["b0"],
            "training": plan.FIXED,
            "preprocessing_evidence": evidence,
            "test_evaluated_during_training": False,
        }
        with (
            patch.object(
                runner.original, "load_training_checkpoint", return_value=saved
            ),
            patch.object(runner, "prepare", return_value=(None, evidence)),
            patch.object(ab, "training_receipt", return_value={"verified": "receipt"}),
            patch.object(runner, "read_json", return_value={"verified": "receipt"}),
        ):
            runner.verify_baseline_fold(session, 1, source)
            invalid = copy.deepcopy(saved)
            invalid["hyperparameters"]["learning_rate"] = 0.1
            with patch.object(
                runner.original, "load_training_checkpoint", return_value=invalid
            ):
                with self.assertRaisesRegex(ValueError, "recipe"):
                    runner.verify_baseline_fold(session, 1, source)
            with patch.object(
                runner, "read_json", return_value={"tampered": "receipt"}
            ):
                with self.assertRaisesRegex(ValueError, "receipt"):
                    runner.verify_baseline_fold(session, 1, source)

    def test_new_fit_preserves_initialization_and_postconstruction_rng_reset(self):
        ab.protocol.seed_everything(43, mps=False)
        expected_model = ab.create_model(plan.MODEL, plan.TRIALS["b1"])
        expected_state = {
            key: value.clone() for key, value in expected_model.state_dict().items()
        }
        ab.protocol.seed_everything(43, mps=False)
        expected_draw = torch.rand(5)
        data = SimpleNamespace(
            **{
                key: np.zeros(2, dtype=np.float32)
                for key in (
                    "train_bins",
                    "validation_bins",
                    "test_bins",
                    "train_reaches",
                    "validation_reaches",
                    "test_reaches",
                    "target_mean",
                    "target_std",
                    "calibration_mean",
                    "calibration_effective_std",
                    "channels",
                )
            }
        )

        def inspect_fit(model, name, trial, seed, supplied_data, device, path, label):
            self.assertEqual(name, "mingru_b")
            self.assertEqual(trial, plan.TRIALS["b1"])
            self.assertEqual(seed, 43)
            self.assertIs(supplied_data, data)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, expected_state[key], rtol=0, atol=0)
            torch.testing.assert_close(torch.rand(5), expected_draw, rtol=0, atol=0)
            return fitted_state()

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(runner, "REPO", Path(temporary)),
            patch.object(runner, "OUTPUT", Path(temporary) / "run"),
            patch.object(runner, "prepare", return_value=(data, {"hash": "frozen"})),
            patch.object(runner, "fit", side_effect=inspect_fit) as fit,
        ):
            row = runner.Sweep({}, torch.device("cpu")).ensure_fit(
                "b1", 43, runner.SESSIONS[0], 1, "screen"
            )
            fit.assert_called_once()
            self.assertEqual(row["weight_policy"], "ema")

    def test_baseline_remapping_preserves_source_receipt_and_checkpoint(self):
        source_config = {"frozen": "source"}
        source_receipt = {
            "model": "mingru_b",
            "trial": "t00",
            "seed": 43,
            "session": runner.SESSIONS[0],
            "fold": 1,
            "signature": runner.digest(source_config),
            "checkpoint": "runs/source.pt",
            "checkpoint_sha256": "sourcehash",
            "policy_validation": {
                policy: {"validation": score(0.7)} for policy in ("raw", "ema")
            },
        }
        before = copy.deepcopy(source_receipt)
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(runner, "REPO", Path(temporary)),
            patch.object(runner, "BASELINE_OUTPUT", Path(temporary) / "baseline"),
            patch.object(runner, "read_json", return_value=source_config),
            patch.object(
                runner, "verify_baseline_fold", return_value=(None, source_receipt)
            ),
        ):
            sweep = runner.Sweep({"new": "configuration"}, torch.device("cpu"))
            row = sweep.baseline_row(runner.SESSIONS[0], 1)
        self.assertEqual(source_receipt, before)
        self.assertEqual(row["trial"], "b0")
        self.assertEqual(row["source_trial"], "t00")
        self.assertEqual(row["source_signature"], before["signature"])
        self.assertEqual(row["signature"], sweep.signature)
        self.assertEqual(row["checkpoint"], "baseline/runs/source.pt")
        self.assertEqual(row["checkpoint_sha256"], "sourcehash")
        self.assertEqual(row["weight_policy"], "ema")
        self.assertTrue(row["cached_control"])

    def test_prediction_verification_rejects_tampered_metrics_targets_and_policy(self):
        data = SimpleNamespace(
            test_bins=np.array([1, 2, 3]),
            velocity=np.array([[0, 0], [1, 2], [2, 1], [3, 4]], dtype=np.float32),
        )
        target = data.velocity[data.test_bins]
        arrays = {
            "target": target,
            "prediction": target + np.float32(0.1),
            "bins": data.test_bins,
            "checkpoint_sha256": np.asarray("checkpoint"),
            "weight_policy": np.asarray("ema"),
        }
        row = {"checkpoint_sha256": "checkpoint"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prediction.npz"

            def save(values):
                np.savez(path, **values)
                return {
                    "prediction_sha256": runner.protocol.sha256_file(path),
                    "test": runner.protocol.metrics(target, arrays["prediction"]),
                }

            result = save(arrays)
            runner.Sweep.verify_prediction(path, result, row, data)
            with self.assertRaisesRegex(ValueError, "hash"):
                runner.Sweep.verify_prediction(
                    path, {**result, "prediction_sha256": "changed"}, row, data
                )
            with self.assertRaisesRegex(ValueError, "scores"):
                runner.Sweep.verify_prediction(
                    path, {**result, "test": score(0.99)}, row, data
                )
            for key, changed in (
                ("bins", data.test_bins + 1),
                ("target", target + 1),
                ("checkpoint_sha256", np.asarray("other")),
                ("weight_policy", np.asarray("raw")),
            ):
                result = save({**arrays, key: changed})
                with self.assertRaisesRegex(ValueError, "identity|targets|policy"):
                    runner.Sweep.verify_prediction(path, result, row, data)

    def test_exactly_78_new_fits_before_test_and_baseline_can_remain_winner(self):
        for baseline_score in (0.60, 0.99):
            with self.subTest(baseline_score=baseline_score):
                rows = []
                identities = set()
                sweep = runner.Sweep({}, torch.device("cpu"))

                def fake_fit(
                    trial, seed, session, fold, stage, rows=rows, identities=identities
                ):
                    key = trial, seed, session, fold
                    self.assertNotIn(key, identities)
                    identities.add(key)
                    candidate = receipt(
                        trial, session, fold, 0.7 + 0.01 * int(trial[1])
                    )
                    candidate.update(
                        checkpoint="/".join(map(str, key)), checkpoint_sha256="hash"
                    )
                    rows.append(candidate)

                def collect(
                    trials, folds, seeds=(43,), rows=rows, baseline_score=baseline_score
                ):
                    selected = [
                        row
                        for row in rows
                        if row["trial"] in trials
                        and row["fold"] in folds
                        and row["seed"] in seeds
                    ]
                    if "b0" in trials:
                        selected += [
                            {
                                **receipt("b0", session, fold, baseline_score),
                                "checkpoint": f"baseline/{session}/{fold}",
                                "checkpoint_sha256": "baseline-hash",
                            }
                            for session in runner.SESSIONS
                            for fold in folds
                        ]
                    return selected

                def final_test(
                    selection, final_rows, rows=rows, baseline_score=baseline_score
                ):
                    self.assertEqual(len(rows), 78)
                    self.assertEqual(len(final_rows), 90)
                    self.assertEqual(selection["finalists"], ["b0", "b5", "b4"])
                    self.assertEqual(
                        selection["validation_selected_trial"],
                        "b0" if baseline_score == 0.99 else "b5",
                    )
                    self.assertEqual(selection["weight_policy"], "ema")
                    self.assertFalse(selection["test_used_for_selection"])
                    gate = runner.read_json(runner.OUTPUT / "test_gate.json")
                    self.assertEqual(gate["verified_new_training_fits"], 78)
                    self.assertEqual(gate["verified_finalist_checkpoints"], 90)
                    plan.require_test_gate(
                        selection["finalists"], final_rows, runner.SESSIONS
                    )

                with (
                    tempfile.TemporaryDirectory() as temporary,
                    patch.object(runner, "OUTPUT", Path(temporary)),
                    patch.object(sweep, "ensure_fit", side_effect=fake_fit),
                    patch.object(sweep, "collect", side_effect=collect),
                    patch.object(sweep, "assert_unchanged"),
                    patch.object(sweep, "write_validation_report"),
                    patch.object(
                        sweep, "test_finalists", side_effect=final_test
                    ) as final,
                ):
                    sweep.run()
                    final.assert_called_once()


if __name__ == "__main__":
    unittest.main()
