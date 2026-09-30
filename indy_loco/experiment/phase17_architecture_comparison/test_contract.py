"""Numerical/protocol checks only; no optimizer steps or real-data training."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch.nn import functional as F

from . import data_contract as contract
from . import models, run, summarize, training
from .capacity import report
from .training import verify_recipe


def recurrence(x, dt, a_log, b, c, skip):
    state = x.new_zeros(x.shape[0], x.shape[2], x.shape[3], b.shape[-1])
    outputs = []
    for index in range(x.shape[1]):
        decay = torch.exp(-a_log.exp()[None] * dt[:, index])
        write = torch.einsum("bh,bhp,bn->bhpn", dt[:, index], x[:, index], b[:, index])
        state = state * decay[:, :, None, None] + write
        outputs.append(
            torch.einsum("bhpn,bn->bhp", state, c[:, index])
            + skip[None, :, None] * x[:, index]
        )
    return torch.stack(outputs, dim=1)


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_protocol_lock_and_recipe(self):
        verify_recipe(contract.verify_protocol_lock())
        self.assertEqual(run.TRAINING.initialization, "scratch")

    def test_saved_report_combines_three_runs_and_rejects_mismatched_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("mingru", "mamba2", "transformer"):
                directory = root / name
                directory.mkdir()
                checkpoint = directory / "checkpoint.pt"
                predictions = directory / "predictions.npz"
                checkpoint.write_bytes(b"synthetic checkpoint fixture")
                predictions.write_bytes(b"synthetic predictions fixture")
                config = {
                    key: {}
                    for key in (
                        "training",
                        "inputs",
                        "references",
                        "environment",
                        "code_sha256",
                    )
                }
                config.update(
                    model=name,
                    sessions=list(contract.session_data.SESSION_BY_NAME),
                    folds=list(range(1, 6)),
                    device="cpu",
                    threads=4,
                    capacity=models.capacity_report(name),
                )
                rows = [
                    {
                        "session": session,
                        "fold": fold,
                        "checkpoint": checkpoint.name,
                        "checkpoint_sha256": summarize.sha256(checkpoint),
                        "predictions": predictions.name,
                        "predictions_sha256": summarize.sha256(predictions),
                        "preprocessing_evidence": {"same": True},
                        "test": {"r2_mean": 0.7},
                    }
                    for session in config["sessions"]
                    for fold in config["folds"]
                ]
                payload = {
                    "config": config,
                    "status": "complete",
                    "full_30fold": True,
                    "results": rows,
                    "summary": [
                        {
                            "group": "overall_fold_macro",
                            "folds": 30,
                            "r2_mean": 0.7,
                            "r2_std": 0.02,
                            "delta_vs_historical_mean": -0.04,
                        }
                    ],
                }
                (directory / "metrics.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            with contextlib.redirect_stdout(io.StringIO()):
                summarize.main(["--results-root", str(root)])
            combined = json.loads((root / "comparison/comparison.json").read_text())
            self.assertEqual(len(combined["models"]), 3)
            path = root / "mamba2/metrics.json"
            changed = json.loads(path.read_text())
            changed["config"]["inputs"] = {"different_source": True}
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(ValueError):
                summarize.main(["--results-root", str(root)])

    def test_instantiated_capacity_matches_proposal(self):
        expected = {row["parameters"] for row in report()["capacity"]}
        actual = {
            models.capacity_report(name)["parameters"] for name in models.MODEL_NAMES
        }
        self.assertEqual(actual, expected)

    def test_causal_reset_shape_and_gradients(self):
        for name in models.MODEL_NAMES:
            with self.subTest(name=name):
                run.model_preflight(name, torch.device("cpu"))
                model = models.build_model(name).eval()
                value = torch.randn(2, 192, 50)
                with torch.no_grad():
                    first = model(value)
                    model(torch.randn_like(value))
                    second = model(value)
                torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_mingru_scan_matches_zero_state_recurrence_and_gradients(self):
        torch.manual_seed(43)
        layer = models.MinGRU().double()
        value = torch.randn(2, 11, 64, dtype=torch.float64, requires_grad=True)
        actual = layer(value)
        candidate, gate = layer.projection(value).chunk(2, -1)
        candidate = torch.where(candidate >= 0, candidate + 0.5, candidate.sigmoid())
        gate = gate.sigmoid()
        state, outputs = torch.zeros_like(candidate[:, 0]), []
        for index in range(11):
            state = (1 - gate[:, index]) * state + gate[:, index] * candidate[:, index]
            outputs.append(state)
        expected = torch.stack(outputs, 1)
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
        parameters = (value, *layer.parameters())
        left = torch.autograd.grad(actual.square().sum(), parameters, retain_graph=True)
        right = torch.autograd.grad(expected.square().sum(), parameters)
        for a, b in zip(left, right, strict=True):
            torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-10)

    def test_ssd_matches_independent_recurrence_and_gradients(self):
        torch.manual_seed(12)
        values = (
            torch.randn(2, 9, 8, 16, dtype=torch.float64),
            torch.rand(2, 9, 8, dtype=torch.float64) * 0.1,
            torch.rand(8, dtype=torch.float64),
            torch.randn(2, 9, 64, dtype=torch.float64),
            torch.randn(2, 9, 64, dtype=torch.float64),
            torch.ones(8, dtype=torch.float64),
        )
        values = tuple(value.requires_grad_() for value in values)
        actual = models.ssd_dense(*values)
        expected = recurrence(*values)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
        left = torch.autograd.grad(actual.square().mean(), values, retain_graph=True)
        right = torch.autograd.grad(expected.square().mean(), values)
        for a, b in zip(left, right, strict=True):
            torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-10)

    def test_complete_mamba_mixer_gate_and_convolution_contract(self):
        torch.manual_seed(20)
        mixer = models.Mamba2Mixer().double()
        value = torch.randn(2, 8, 64, dtype=torch.float64)
        z, xbc, dt = mixer.in_proj(value).split((128, 256, 8), -1)
        # Independent lag-by-lag depthwise convolution with zero left context.
        padded = F.pad(xbc.transpose(1, 2), (3, 0))
        filtered = sum(
            padded[:, :, lag : lag + 8] * mixer.conv.weight[:, 0, lag][None, :, None]
            for lag in range(4)
        )
        filtered = F.silu(filtered + mixer.conv.bias[None, :, None]).transpose(1, 2)
        x, b, c = filtered.split((128, 64, 64), -1)
        mixed = recurrence(
            x.reshape(2, 8, 8, 16),
            F.softplus(dt + mixer.dt_bias),
            mixer.a_log,
            b,
            c,
            mixer.skip,
        ).flatten(2)
        gated = mixed * F.silu(z)
        expected = mixer.out_proj(
            gated
            / (gated.square().mean(-1, keepdim=True) + 1e-5).sqrt()
            * mixer.rms_weight
        )
        torch.testing.assert_close(mixer(value), expected, rtol=1e-9, atol=1e-10)

    def test_optimizer_groups_cover_every_parameter_once(self):
        for name in models.MODEL_NAMES:
            model = models.build_model(name)
            groups = models.parameter_groups(model, name)
            ids = [id(p) for group in groups for p in group["params"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(set(ids), {id(p) for p in model.parameters()})
            self.assertEqual([g["lr"] for g in groups], [3e-4, 7.5e-5])
            if name != "midsize_control":
                self.assertEqual(
                    {id(p) for p in groups[1]["params"]},
                    {id(p) for p in model.stem.parameters()},
                )

    def test_paired_dropout_and_window(self):
        layer = models.PairedChannelDropout().train()
        out = layer(torch.ones(3, 192, 50))
        torch.testing.assert_close(out[:, :96], out[:, 96:], rtol=0, atol=0)
        features = np.broadcast_to(np.arange(100, dtype=np.float32), (192, 100))
        windows = run.protocol.rolling_batch(features, np.array([49, 60]))
        np.testing.assert_array_equal(windows[:, 0, -1], [49, 60])
        self.assertEqual(windows.shape, (2, 192, 50))

    def test_training_hyperparameters_cannot_be_overridden(self):
        for flag in (
            "--epochs",
            "--batch-size",
            "--patience",
            "--learning-rate",
            "--init",
        ):
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                run.parse_args([flag, "1"], fixed_model="mingru")

    def test_checkpoints_reject_tampering_and_changed_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            payload = {
                "signature": "one",
                "preprocessing_evidence": {"x": 1},
                "session": "s",
                "fold": 1,
            }
            run.save_training_checkpoint(path, payload)
            saved = run.load_training_checkpoint(path)
            run.validate_saved(saved, "one", {"x": 1}, "s", 1)
            with self.assertRaises(ValueError):
                run.validate_saved(saved, "one", {"x": 2}, "s", 1)
            path.write_bytes(path.read_bytes() + b"tamper")
            with self.assertRaises(ValueError):
                run.load_training_checkpoint(path)

    def test_no_training_modes_never_call_fit(self):
        for flag in ("--validate-only", "--dry-run"):
            args = [flag, "--session", "indy_20160622_01", "--fold", "1"]
            prepared = SimpleNamespace(
                train_bins=[0], validation_bins=[1], test_bins=[2]
            )
            with (
                patch.object(
                    run,
                    "run_training",
                    side_effect=AssertionError("Training must not start"),
                ),
                patch.object(run, "model_preflight"),
                patch.object(run, "make_config", return_value={"capacity": {}}),
                patch.object(contract, "load_session"),
                patch.object(
                    contract, "prepare_verified_fold", return_value=(prepared, {})
                ),
                patch.object(contract, "load_reference"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                run.main(args, fixed_model="mingru")

    def test_roundoff_is_audited_but_material_drift_rejected(self):
        fields = {
            "calibration_mean": "feature_mean",
            "calibration_effective_std": "feature_std",
            "calibration_local_std": "calibration_local_std",
            "feature_std_floor": "feature_std_floor",
            "target_mean": "target_mean",
            "target_std": "target_std",
        }
        reference = {
            key: np.ones(2 if key.startswith("target") else 192, dtype=np.float32)
            for key in fields.values()
        }
        prepared = SimpleNamespace(counts=np.ones((96, 60), dtype=np.float32))
        for field, key in fields.items():
            setattr(prepared, field, np.nextafter(reference[key], np.float32(2)))
        with (
            patch.object(contract.protocol, "prepare_fold", return_value=prepared),
            patch.object(contract, "preprocessing_evidence", return_value={}),
        ):
            fixed, evidence = contract.prepare_verified_fold(None, 1, reference)
            np.testing.assert_array_equal(
                fixed.calibration_mean, reference["feature_mean"]
            )
            self.assertFalse(
                evidence["numeric_reproducibility"]["statistics"]["calibration_mean"][
                    "exact_before_pinning"
                ]
            )
            fixed.calibration_mean += 0.001
            with self.assertRaises(ValueError):
                contract.prepare_verified_fold(None, 1, reference)

    def test_aggregation_uses_all_folds_and_sample_sd(self):
        config = {"sessions": ["s"], "folds": [1, 2]}
        results = [
            {
                "session": "s",
                "subject": "indy",
                "fold": fold,
                "best_epoch": 2,
                "best_validation_loss": 1.0,
                "test": {"r2_mean": value},
                "training_seconds": 1.0,
                "test_prediction_seconds": 0.1,
                "history": [],
            }
            for fold, value in ((1, 0.6), (2, 0.8))
        ]
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(
                run, "historical_scores", return_value={("s", 1): 0.5, ("s", 2): 0.9}
            ),
        ):
            run.write_reports(Path(temporary), config, results)
            metrics = json.loads((Path(temporary) / "metrics.json").read_text())
            overall = next(
                row
                for row in metrics["summary"]
                if row["group"] == "overall_fold_macro"
            )
            self.assertAlmostEqual(overall["r2_mean"], 0.7)
            self.assertAlmostEqual(overall["r2_std"], np.std([0.6, 0.8], ddof=1))
            self.assertEqual(overall["wins_vs_historical"], 1)
            self.assertEqual(metrics["status"], "complete")
            self.assertFalse(metrics["full_30fold"])

    def test_validation_selection_patience_and_best_weight_restore_without_updates(
        self,
    ):
        class Dummy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(()))

            def forward(self, inputs):
                return self.weight.expand(inputs.shape[0], 50, 2)

        model = Dummy()
        prepared = SimpleNamespace(
            velocity=np.zeros((60, 2), dtype=np.float32),
            target_mean=np.zeros(2, dtype=np.float32),
            target_std=np.ones(2, dtype=np.float32),
            normalized_features=np.zeros((192, 60), dtype=np.float32),
            train_bins=np.array([49]),
            validation_bins=np.array([50]),
            test_bins=np.array([59]),
        )
        validation_calls = []

        def validate(*args):
            np.testing.assert_array_equal(args[3], prepared.validation_bins)
            validation_calls.append(1)
            # Distinct synthetic state identifies which epoch was restored.
            with torch.no_grad():
                model.weight.fill_(len(validation_calls))
            return {
                "normalized_loss": [1.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1][
                    len(validation_calls) - 1
                ],
                "r2_mean": 0.0,
            }

        groups = [{"params": [model.weight], "lr": 3e-4}, {"params": [], "lr": 7.5e-5}]
        optimizer = SimpleNamespace(
            param_groups=groups, zero_grad=model.zero_grad, step=Mock()
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(training, "parameter_groups", return_value=groups),
            patch.object(torch.optim, "AdamW", return_value=optimizer),
            patch.object(
                torch.optim.lr_scheduler, "CosineAnnealingLR", return_value=Mock()
            ) as scheduler,
            patch.object(run.protocol, "evaluate_last", side_effect=validate),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            fit = training.fit_model(
                model,
                "mingru",
                "synthetic",
                1,
                prepared,
                torch.device("cpu"),
                Path(temporary) / "epochs.json",
            )
        self.assertEqual(fit["best_epoch"], 2)
        self.assertEqual(len(fit["history"]), 8)
        self.assertEqual(float(model.weight.detach()), 2.0)
        self.assertEqual(scheduler.call_args.args[1], 20)

    def test_test_stage_waits_for_all_checkpoints_and_resume_skips_fitting(self):
        session = "indy_20160622_01"
        config = {"sessions": [session], "folds": [1, 2]}
        args = SimpleNamespace(model="mingru", indy_cache_root=Path("unused"))
        events = []

        def make_checkpoint(
            name,
            current_session,
            fold,
            prepared,
            reference,
            evidence,
            signature,
            output,
            device,
        ):
            events.append(f"fit{fold}")
            return {
                "session": current_session,
                "fold": fold,
                "signature": signature,
                "preprocessing_evidence": evidence,
            }

        def evaluate(checkpoint, prepared, path, output, device):
            fold = checkpoint["fold"]
            self.assertEqual(events[:2], ["fit1", "fit2"])
            events.append(f"test{fold}")
            prediction = output / f"dummy_prediction_{fold}.npz"
            prediction.write_bytes(b"test fixture")
            return {
                **checkpoint,
                "checkpoint": str(path.relative_to(output)),
                "checkpoint_sha256": run.protocol.sha256_file(path),
                "predictions": prediction.name,
                "predictions_sha256": run.protocol.sha256_file(prediction),
                "test": {"r2_mean": 0.0},
            }

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(contract, "load_session"),
            patch.object(contract, "load_reference"),
            patch.object(
                contract,
                "prepare_verified_fold",
                return_value=(None, {"input": "fixed"}),
            ),
            patch.object(run, "create_checkpoint", side_effect=make_checkpoint) as fit,
            patch.object(run, "evaluate_checkpoint", side_effect=evaluate) as evaluator,
            patch.object(run, "write_reports"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            run.run_training(
                args, config, "signature", Path(temporary), torch.device("cpu")
            )
            self.assertEqual(events, ["fit1", "fit2", "test1", "test2"])
            run.run_training(
                args, config, "signature", Path(temporary), torch.device("cpu")
            )
            self.assertEqual(fit.call_count, 2)
            self.assertEqual(evaluator.call_count, 2)


if __name__ == "__main__":
    unittest.main()
