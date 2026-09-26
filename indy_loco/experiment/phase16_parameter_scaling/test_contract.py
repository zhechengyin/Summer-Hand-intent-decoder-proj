"""No-training checks: frozen recipe, architecture, causality and provenance."""

import ast
import contextlib
import hashlib
import io
import json
import unittest
from dataclasses import FrozenInstanceError

import numpy as np
import torch

from indy_loco.experiment.phase16_parameter_scaling import protocol, run, session_data
from indy_loco.experiment.phase16_parameter_scaling.model import (
    BASELINE,
    Architecture,
    PairedChannelDropout,
    ScaledTCNGRU,
    parameter_report,
)
from indy_loco.experiment.phase16_parameter_scaling.stopping import ValidationPlateau
from indy_loco.experiment.phase16_parameter_scaling.transfer import (
    transfer_midsize_weights,
)
from indy_loco.models.midsize.model import MidsizeTCNGRU


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_frozen_recipe_and_loop_match_lock(self):
        run.check_protocol_lock()
        self.assertEqual(run.TRAINING.init, "midsize_transfer")
        self.assertEqual(run.LEARNING_RATE * run.ENCODER_LR_SCALE, 7.5e-5)
        with self.assertRaises(FrozenInstanceError):
            run.TRAINING.epochs = 1

    def test_copied_nodes_match_pinned_source_without_importing_history(self):
        lock = json.loads((run.HERE / "protocol_lock.json").read_text())
        for source, expected in lock["sources"].items():
            path = run.REPO / source
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
            tree = ast.parse(path.read_text())
            functions = {
                n.name: n
                for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            }
            key = "session_data_nodes" if "phase7" in source else "protocol_nodes"
            for name, digest in lock[key].items():
                if name in functions:
                    self.assertEqual(run.node_hash(functions[name]), digest, name)
        self.assertNotIn(
            "load_state_dict",
            ast.unparse(
                next(
                    n
                    for n in ast.parse((run.HERE / "run.py").read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "train_one"
                )
            ),
        )

    def test_original_architecture_matches_canonical_inference(self):
        torch.manual_seed(43)
        original = MidsizeTCNGRU().eval()
        torch.manual_seed(43)
        baseline = ScaledTCNGRU(BASELINE).eval()
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, baseline.state_dict()[key]), key)
        inputs = torch.randn(2, 192, 50)
        with torch.inference_mode():
            torch.testing.assert_close(
                original(inputs), baseline(inputs), rtol=0, atol=0
            )
        self.assertEqual(parameter_report(BASELINE)["total_parameters"], 86978)

    def test_causal_output_and_shape_for_multilayer_expansion(self):
        model = ScaledTCNGRU(Architecture()).eval()
        inputs = torch.randn(2, 192, 50)
        changed = inputs.clone()
        changed[:, :, 30:] += 100
        with torch.inference_mode():
            first, second = model(inputs), model(changed)
        self.assertEqual(first.shape, (2, 50, 2))
        torch.testing.assert_close(first[:, :30], second[:, :30], rtol=0, atol=0)
        self.assertEqual(model.gru.num_layers, 1)
        self.assertEqual(model.gru.dropout, 0)
        self.assertFalse(model.gru.bidirectional)
        self.assertEqual(model.dropout.p, 0.1)

    def test_channel_dropout_still_pairs_count_and_ewma(self):
        layer = PairedChannelDropout().train()
        values = layer(torch.ones(4, 192, 50))
        self.assertEqual(layer.probability, 0.2)
        self.assertTrue(torch.equal(values[:, :96], values[:, 96:]))
        self.assertTrue(torch.equal(values[:, :, :1].expand_as(values), values))

    def test_dilation_does_not_change_parameter_count(self):
        first = Architecture(64, 3, (1, 2, 4, 8), 64, 1)
        second = Architecture(64, 3, (1, 2, 4, 16), 64, 1)
        self.assertEqual(parameter_report(first), parameter_report(second))
        self.assertNotEqual(
            first.metadata()["encoder_receptive_field_bins"],
            second.metadata()["encoder_receptive_field_bins"],
        )

    def test_parameter_report_does_not_perturb_training_rng(self):
        torch.manual_seed(43)
        state = torch.get_rng_state().clone()
        report = parameter_report(Architecture())
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(report["total_parameters"], 131762)
        self.assertEqual(report["fp32_weight_bytes"], 527048)

    def test_hyperparameter_overrides_are_rejected(self):
        for option in (
            "--epochs",
            "--batch-size",
            "--learning-rate",
            "--patience",
            "--init",
            "--weight-decay",
        ):
            with (
                self.subTest(option=option),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                run.parse_args([option, "1"])

    def test_architecture_layers_and_dimensions_are_checked(self):
        with self.assertRaises(ValueError):
            run.architecture_for(run.parse_args(["--encoder-layers", "5"]))
        with self.assertRaises(ValueError):
            Architecture(encoder_dilations=(1, 2, 4, 0))
        self.assertEqual(
            run.architecture_for(run.parse_args(["--model", "baseline"])), BASELINE
        )

    def test_folds_partition_reaches_and_keep_validation_test_disjoint(self):
        eligible = np.arange(100)
        parts = session_data.make_fold_indices(eligible)
        for fold in range(5):
            train, validation, test = session_data.split_fold(parts, fold)
            self.assertEqual(len(train), 80)
            self.assertEqual(len(validation), 10)
            self.assertEqual(len(test), 10)
            self.assertEqual(len(set(train) | set(validation) | set(test)), 100)
            self.assertFalse(set(train) & set(validation))
            self.assertFalse(set(validation) & set(test))

    def test_rolling_windows_preserve_50_bins_and_final_target(self):
        features = np.broadcast_to(np.arange(100, dtype=np.float32), (192, 100))
        batch = protocol.rolling_batch(features, np.array([49, 60]))
        self.assertEqual(batch.shape, (2, 192, 50))
        np.testing.assert_array_equal(batch[0, 0], np.arange(50))
        np.testing.assert_array_equal(batch[1, 0], np.arange(11, 61))

    def test_transfer_baseline_exactly_reuses_source(self):
        source = MidsizeTCNGRU().eval()
        model = ScaledTCNGRU(BASELINE).eval()
        report = transfer_midsize_weights(model, source.state_dict())
        self.assertEqual(report["copied_parameters"], 86978)
        self.assertEqual(report["new_random_parameters"], 0)
        for key, tensor in source.state_dict().items():
            self.assertTrue(torch.equal(tensor, model.state_dict()[key]))

    def test_transfer_preserves_new_values_and_maps_each_gru_gate(self):
        source = MidsizeTCNGRU().state_dict()
        model = ScaledTCNGRU(Architecture())
        before = {k: v.clone() for k, v in model.state_dict().items()}
        report = transfer_midsize_weights(model, source)
        self.assertEqual(report["copied_parameters"], 86978)
        self.assertEqual(report["new_random_parameters"], 44784)
        self.assertFalse(report["function_preserving"])
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        after = model.state_dict()
        for name, target in after.items():
            expected = before[name].clone()
            if name.startswith("gru."):
                for gate in range(3):
                    old = slice(gate * 64, (gate + 1) * 64)
                    new = slice(gate * 80, gate * 80 + 64)
                    if target.ndim == 2:
                        expected[new, :64] = source[name][old]
                    else:
                        expected[new] = source[name][old]
            else:
                expected[tuple(slice(0, n) for n in source[name].shape)] = source[name]
            self.assertTrue(torch.equal(expected, target), name)

    def test_kernel_extension_aligns_causal_lags(self):
        source = MidsizeTCNGRU().state_dict()
        model = ScaledTCNGRU(Architecture(encoder_kernel_size=5))
        untouched = model.convolutions[0].weight[:, :, :2].detach().clone()
        transfer_midsize_weights(model, source)
        self.assertTrue(torch.equal(model.convolutions[0].weight[:, :, :2], untouched))
        self.assertTrue(
            torch.equal(
                model.convolutions[0].weight[:64, :64, 2:],
                source["convolutions.0.weight"],
            )
        )

    def test_invalid_source_rejected_before_destination_mutation(self):
        model = ScaledTCNGRU(Architecture())
        before = model.head.weight.detach().clone()
        source = MidsizeTCNGRU().state_dict()
        source["head.bias"][0] = float("nan")
        with self.assertRaises(ValueError):
            transfer_midsize_weights(model, source)
        self.assertTrue(torch.equal(before, model.head.weight))

    def test_fast_stop_plateau_waits_four_epochs(self):
        monitor = ValidationPlateau()
        self.assertEqual(
            [monitor.update(e, 1.0) for e in range(1, 5)], [False, False, False, True]
        )

    def test_fast_stop_accumulates_small_improvements(self):
        monitor = ValidationPlateau()
        decisions = [
            monitor.update(e, loss)
            for e, loss in enumerate([1.0, 0.998, 0.996, 0.994], 1)
        ]
        self.assertEqual(decisions, [False] * 4)
        self.assertEqual(monitor.bad_epochs, 0)
        self.assertFalse(monitor.update(5, 0.993))
        self.assertFalse(monitor.update(6, 0.992))
        self.assertTrue(monitor.update(7, 0.991))

    def test_fast_stop_caps_twenty_and_rejects_nonfinite(self):
        monitor = ValidationPlateau()
        for epoch in range(1, 21):
            self.assertEqual(monitor.update(epoch, 0.9**epoch), epoch == 20)
        with self.assertRaises(ValueError):
            ValidationPlateau().update(1, float("nan"))


if __name__ == "__main__":
    unittest.main()
