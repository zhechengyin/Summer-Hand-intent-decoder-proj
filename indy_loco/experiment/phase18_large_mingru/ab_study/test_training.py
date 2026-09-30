"""A/B policy selection, EMA updates, test isolation, and resume contracts."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from . import plan
from . import train as runner


def policy_score(r2, loss=0.2, epoch=1):
    return {
        "validation": {"r2_mean": r2, "normalized_loss": loss},
        "best_epoch": epoch,
        "best_validation_loss": loss,
        "arithmetic_check": {"persistent_state": False},
    }


class TrainingTests(unittest.TestCase):
    def test_fixed_budget_and_symmetric_training_plan(self):
        self.assertEqual(plan.MODELS, ("mingru_a", "mingru_b"))
        self.assertEqual(plan.SEEDS, (43,))
        self.assertEqual(tuple(plan.TRIALS), ("t00", "t01"))
        self.assertEqual(plan.WEIGHT_POLICIES, ("raw", "ema"))
        self.assertEqual(plan.EMA_DECAY, 0.99)
        self.assertEqual(plan.TOP_K, 1)
        self.assertEqual(plan.FIXED["epochs"], 60)
        self.assertEqual(plan.FIXED["patience"], 15)
        self.assertEqual(plan.FIXED["batch_size"], 128)
        self.assertEqual(plan.FIXED["split_seed"], 43)
        self.assertEqual(plan.expected_fits(), {
            "screen": 24, "confirm": 48, "seed_check": 0, "total": 72,
        })
        for trial, learning_rate in zip(plan.TRIALS.values(), (1e-3, 6e-4), strict=True):
            self.assertEqual(trial, {
                "learning_rate": learning_rate, "stem_lr_scale": 1.0,
                "weight_decay": 0.01, "head_weight_decay": 0.03,
                "channel_dropout": 0.1, "dropout": 0.1,
            })

    def test_joint_lr_policy_ranking_uses_only_complete_validation_evidence(self):
        rows = [
            {
                "trial": trial, "seed": 43, "session": session, "fold": 1,
                "policy_validation": {
                    "raw": policy_score(raw), "ema": policy_score(ema),
                },
                "test": {"r2_mean": test},
            }
            for trial, raw, ema, test in (
                ("t00", 0.61, 0.65, 0.99), ("t01", 0.62, 0.70, -0.99),
            )
            for session in ("indy", "loco")
        ]

        def rank(values, **kwargs):
            return plan.rank_trials(values, ("t00", "t01"), ("indy", "loco"), (1,), **kwargs)

        selected = rank(rows)[0]
        self.assertEqual((selected["trial"], selected["weight_policy"]), ("t01", "ema"))
        changed_test = copy.deepcopy(rows)
        for row in changed_test:
            row["test"]["r2_mean"] *= -1e9
        self.assertEqual(rank(rows), rank(changed_test))
        self.assertEqual({row["weight_policy"] for row in rank(rows, weight_policies=("raw",))}, {"raw"})
        for invalid in (rows[:-1], rows + rows[:1]):
            with self.assertRaises(ValueError):
                rank(invalid)
        nonfinite = copy.deepcopy(rows)
        nonfinite[0]["policy_validation"]["ema"]["validation"]["r2_mean"] = float("nan")
        with self.assertRaises(ValueError):
            rank(nonfinite)
        with self.assertRaises(ValueError):
            rank(rows, weight_policies=("unknown",))

    def test_policy_selection_preserves_original_receipt(self):
        row = {
            "trial": "t00", "checkpoint_sha256": "original-hash",
            **policy_score(0.6, 0.3, 2),
            "policy_validation": {
                "raw": policy_score(0.6, 0.3, 2),
                "ema": policy_score(0.7, 0.2, 4),
            },
        }
        original = copy.deepcopy(row)
        selected = runner.select_policy(row, "ema")
        self.assertEqual(row, original)
        self.assertEqual(selected["weight_policy"], "ema")
        self.assertEqual(selected["validation"]["r2_mean"], 0.7)
        self.assertEqual(selected["best_epoch"], 4)
        self.assertEqual(selected["checkpoint_sha256"], "original-hash")
        self.assertEqual(selected["policy_validation"], original["policy_validation"])
        with self.assertRaises((KeyError, ValueError)):
            runner.select_policy(row, "unknown")

    def test_test_gate_requires_both_architectures_and_frozen_policies(self):
        finalists = {"mingru_a": "t01", "mingru_b": "t00"}
        policies = {"mingru_a": "ema", "mingru_b": "raw"}
        rows = [
            {"model": model, "trial": trial, "seed": 43, "session": session,
             "fold": fold, "weight_policy": policies[model]}
            for model, trial in finalists.items()
            for session in runner.SESSIONS
            for fold in plan.FOLDS
        ]
        self.assertEqual(len(rows), 60)
        plan.require_test_gate(finalists, rows, runner.SESSIONS, policies)
        for invalid in (rows[:-1], rows[:30], rows + rows[:1]):
            with self.assertRaises(ValueError):
                plan.require_test_gate(finalists, invalid, runner.SESSIONS, policies)
        for field, value in (("weight_policy", "raw"), ("seed", 44), ("trial", "t00")):
            invalid = copy.deepcopy(rows)
            invalid[0][field] = value
            with self.assertRaises(ValueError):
                plan.require_test_gate(finalists, invalid, runner.SESSIONS, policies)

    def test_actual_optimizer_groups_cover_both_models_with_head_decay(self):
        for name in plan.MODELS:
            for trial in plan.TRIALS.values():
                with self.subTest(model=name, lr=trial["learning_rate"]):
                    model = runner.create_model(name, trial)
                    groups = runner.optimizer_groups(model, name, trial)
                    optimizer = torch.optim.AdamW(groups, weight_decay=trial["weight_decay"])
                    ids = [id(p) for group in groups for p in group["params"]]
                    self.assertEqual(len(ids), len(set(ids)))
                    self.assertEqual(set(ids), {id(p) for p in model.parameters()})
                    self.assertEqual([group["name"] for group in groups], ["temporal_head", "encoder_stem", "output_head"])
                    for group in optimizer.param_groups:
                        self.assertEqual(group["lr"], trial["learning_rate"])
                        self.assertEqual(group["weight_decay"], 0.03 if group["name"] == "output_head" else 0.01)
                    self.assertEqual({id(p) for p in groups[2]["params"]}, {id(p) for p in model.head.parameters()})

    def test_ema_update_is_separate_and_copies_integer_buffers(self):
        raw = nn.Linear(2, 2)
        raw.register_buffer("updates", torch.tensor(7, dtype=torch.int64))
        ema = copy.deepcopy(raw).requires_grad_(False)
        with torch.no_grad():
            for parameter in raw.parameters():
                parameter.fill_(3.0)
            for parameter in ema.parameters():
                parameter.fill_(1.0)
            ema.updates.zero_()
        before = {key: value.clone() for key, value in raw.state_dict().items()}
        runner.update_ema(ema, raw, decay=0.99)
        for parameter in ema.parameters():
            torch.testing.assert_close(parameter, torch.full_like(parameter, 1.02))
            self.assertIsNone(parameter.grad)
        self.assertEqual(int(ema.updates), 7)
        for key, value in raw.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        with torch.no_grad():
            next(ema.parameters()).zero_()
        torch.testing.assert_close(next(raw.parameters()), torch.full_like(raw.weight, 3.0))

    def test_fit_saves_separate_best_raw_ema_without_test_labels(self):
        old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old_threads)
        torch.set_num_threads(2)
        rng = np.random.default_rng(1)
        data = SimpleNamespace(
            normalized_features=rng.normal(size=(192, 100)).astype(np.float32),
            velocity=rng.normal(size=(100, 2)).astype(np.float32),
            target_mean=np.zeros(2, dtype=np.float32), target_std=np.ones(2, dtype=np.float32),
            train_bins=np.arange(49, 53), validation_bins=np.arange(60, 63), test_bins=np.arange(80, 83),
        )
        data.velocity[data.test_bins] = np.nan
        scores = iter([
            {"normalized_loss": 0.2, "r2_mean": 0.4},
            {"normalized_loss": 0.3, "r2_mean": 0.8},
            {"normalized_loss": 0.25, "r2_mean": 0.9},
            {"normalized_loss": 0.1, "r2_mean": 0.5},
            {"normalized_loss": 0.15, "r2_mean": 0.6},
            {"normalized_loss": 0.2, "r2_mean": 0.9},
            {"normalized_loss": 0.3, "r2_mean": 0.7},
            {"normalized_loss": 0.3, "r2_mean": 0.8},
        ])
        validation_states = []

        def validate(model, features, velocity, bins, *args):
            np.testing.assert_array_equal(bins, data.validation_bins)
            validation_states.append({key: value.detach().clone() for key, value in model.state_dict().items()})
            return next(scores)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "epochs.json"
            with patch.dict(plan.FIXED, {"epochs": 5, "patience": 1}), patch.object(runner.protocol, "evaluate_last", side_effect=validate) as evaluate:
                fitted = runner.fit(runner.create_model("mingru_a", plan.TRIALS["t00"]), "mingru_a", plan.TRIALS["t00"], 43, data, torch.device("cpu"), path, "toy")
            self.assertEqual(evaluate.call_count, 8)
            self.assertEqual(len(fitted["history"]), 4)
            self.assertEqual(len(runner.read_json(path)), 4)
            policies = fitted["weight_policies"]
            self.assertEqual(set(policies), {"raw", "ema"})
            for policy, epoch, state_index, r2 in (("raw", 3, 4, 0.6), ("ema", 2, 3, 0.5)):
                self.assertEqual(policies[policy]["best_epoch"], epoch)
                self.assertEqual(policies[policy]["validation"]["r2_mean"], r2)
                self.assertFalse(policies[policy]["arithmetic_check"]["persistent_state"])
                for key, value in policies[policy]["model_state"].items():
                    self.assertTrue(torch.isfinite(value).all())
                    torch.testing.assert_close(value, validation_states[state_index][key], rtol=0, atol=0)
            self.assertEqual(fitted["best_epoch"], 3)
            self.assertEqual(fitted["validation"]["r2_mean"], 0.6)

    def test_saved_identity_preprocessing_and_test_isolation_required(self):
        saved = {
            "seed": 43, "signature": "fixed", "preprocessing_evidence": {"hash": "x"},
            "test_evaluated_during_training": False, "validation": {"r2_mean": 0.6, "normalized_loss": 0.2},
            "history": [{"epoch": 1}],
            "weight_policies": {policy: policy_score(0.6) for policy in plan.WEIGHT_POLICIES},
        }
        runner.check_saved(saved, {"seed": 43, "signature": "fixed"}, {"hash": "x"})
        for expected, evidence in (({"seed": 44}, {"hash": "x"}), ({"signature": "other"}, {"hash": "x"}), ({"seed": 43}, {"hash": "changed"})):
            with self.assertRaises(ValueError):
                runner.check_saved(saved, expected, evidence)
        for value in (True, None):
            with self.assertRaises(ValueError):
                runner.check_saved({**saved, "test_evaluated_during_training": value}, {}, {"hash": "x"})

    def test_resume_verifies_checkpoint_receipt_and_policies_without_training(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(runner, "OUTPUT", Path(temporary)):
            sweep = runner.Sweep({}, torch.device("cpu"))
            args = ("mingru_a", "t00", 43, runner.SESSIONS[0], 1)
            evidence = {"hash": "frozen"}
            policies = {
                policy: {**policy_score(0.6), "model_state": {"toy": torch.tensor([1.0])}}
                for policy in plan.WEIGHT_POLICIES
            }
            saved = {
                **runner.identity(*args, sweep.signature), **policies["raw"],
                "preprocessing_evidence": evidence, "test_evaluated_during_training": False,
                "training_seconds": 0.1, "weight_policies": policies,
                "history": [{"epoch": 1}],
            }
            path = runner.checkpoint_path(*args)
            runner.original.save_training_checkpoint(path, saved)
            with patch.object(runner, "prepare", return_value=(None, evidence)), patch.object(runner, "fit", side_effect=AssertionError("Resume retrained")):
                receipt = sweep.ensure_fit(*args, "screen")
                self.assertEqual(receipt["checkpoint_sha256"], runner.protocol.sha256_file(path))
                self.assertEqual(set(receipt["policy_validation"]), {"raw", "ema"})
                self.assertEqual(sweep.ensure_fit(*args, "screen"), receipt)
                changed = copy.deepcopy(receipt)
                changed["policy_validation"]["ema"]["validation"]["r2_mean"] = 0.99
                runner.protocol.write_json_atomic(path.with_suffix(".validation.json"), changed)
                with self.assertRaisesRegex(ValueError, "receipt"):
                    sweep.ensure_fit(*args, "screen")
                runner.protocol.write_json_atomic(path.with_suffix(".validation.json"), receipt)
                with path.open("ab") as handle:
                    handle.write(b"tampered")
                with self.assertRaisesRegex(ValueError, "SHA|hash"):
                    sweep.ensure_fit(*args, "screen")

    def test_stop_and_frozen_selection_guards(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(runner, "OUTPUT", Path(temporary)):
            sweep = runner.Sweep({}, torch.device("cpu"))
            selection = {"trial": "t00", "weight_policy": "raw"}
            sweep.freeze("selection.json", selection)
            sweep.freeze("selection.json", selection)
            with self.assertRaises(ValueError):
                sweep.freeze("selection.json", {**selection, "weight_policy": "ema"})
            (runner.OUTPUT / "STOP_AFTER_FOLD").touch()
            with patch.object(runner, "prepare") as prepare, patch.object(runner, "fit") as fit:
                with self.assertRaises(runner.StopRequested):
                    sweep.ensure_fit("mingru_a", "t00", 43, runner.SESSIONS[0], 1, "screen")
                prepare.assert_not_called()
                fit.assert_not_called()

    def test_schedule_freezes_screen_policy_and_runs_72_fits_before_60_tests(self):
        rows, identities = [], set()
        sweep = runner.Sweep({}, torch.device("cpu"))

        def fake_fit(name, trial, seed, session, fold, stage):
            key = name, trial, seed, session, fold
            self.assertNotIn(key, identities)
            identities.add(key)
            if name == "mingru_a":
                raw, ema = ((0.60, 0.65) if trial == "t00" else (0.64, 0.70)) if fold == 1 else (0.99, 0.72)
            else:
                raw, ema = ((0.80, 0.78) if trial == "t00" else (0.75, 0.76)) if fold == 1 else (0.60, 0.99)
            rows.append({
                "model": name, "trial": trial, "seed": seed, "session": session, "fold": fold,
                "checkpoint": "/".join(str(value) for value in key), "checkpoint_sha256": "hash",
                **policy_score(raw),
                "policy_validation": {"raw": policy_score(raw), "ema": policy_score(ema)},
            })

        def collect(name, trials, folds, seeds=(43,), weight_policy=None):
            selected = [row for row in rows if row["model"] == name and row["trial"] in trials and row["fold"] in folds and row["seed"] in seeds]
            return [runner.select_policy(row, weight_policy) for row in selected] if weight_policy else selected

        def final_test(selection, final_rows):
            self.assertEqual(len(rows), 72)
            self.assertEqual(len(final_rows), 60)
            self.assertEqual(selection["finalists"], {"mingru_a": "t01", "mingru_b": "t00"})
            self.assertEqual(selection["weight_policies"], {"mingru_a": "ema", "mingru_b": "raw"})
            self.assertEqual(selection["validation_selected_architecture"], "mingru_a")
            self.assertFalse(selection["test_used_for_selection"])
            gate = runner.read_json(runner.OUTPUT / "test_gate.json")
            self.assertEqual(gate["verified_finalist_checkpoints"], 60)
            plan.require_test_gate(selection["finalists"], final_rows, runner.SESSIONS, selection["weight_policies"])

        with tempfile.TemporaryDirectory() as temporary, patch.object(runner, "OUTPUT", Path(temporary)), patch.object(sweep, "ensure_fit", side_effect=fake_fit), patch.object(sweep, "collect", side_effect=collect), patch.object(sweep, "assert_unchanged"), patch.object(sweep, "test_finalists", side_effect=final_test) as evaluate:
            sweep.run()
            evaluate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
