"""Checkpoint compatibility and training-only regularization tests."""

import copy
import unittest

import torch

from ..model import LargeMinGRU as FrozenLargeMinGRU
from ..model import parameter_groups as frozen_parameter_groups
from .model import LargeMinGRU, capacity_report, parameter_groups


class TuningModelTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        torch.manual_seed(1802)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    def test_initialization_and_checkpoint_keys_identical(self):
        torch.manual_seed(90)
        parent = FrozenLargeMinGRU()
        torch.manual_seed(90)
        tuned = LargeMinGRU(head_dropout=0.4)
        self.assertEqual(list(parent.state_dict()), list(tuned.state_dict()))
        self.assertEqual(
            list(dict(parent.named_parameters())), list(dict(tuned.named_parameters()))
        )
        for name, value in parent.state_dict().items():
            torch.testing.assert_close(value, tuned.state_dict()[name], rtol=0, atol=0)
        tuned.load_state_dict(parent.state_dict(), strict=True)
        parent.load_state_dict(tuned.state_dict(), strict=True)

    def test_zero_head_dropout_exact_outputs_gradients_and_rng(self):
        for training in (False, True):
            with self.subTest(training=training):
                parent = FrozenLargeMinGRU().double().train(training)
                tuned = LargeMinGRU(head_dropout=0.0).double().train(training)
                tuned.load_state_dict(parent.state_dict())
                inputs = torch.randn(2, 192, 8, dtype=torch.float64)
                left = inputs.clone().requires_grad_(True)
                right = inputs.clone().requires_grad_(True)
                torch.manual_seed(81)
                expected = parent(left)
                parent_rng = torch.get_rng_state().clone()
                torch.manual_seed(81)
                actual = tuned(right)
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)
                self.assertTrue(torch.equal(parent_rng, torch.get_rng_state()))
                expected.square().sum().backward()
                actual.square().sum().backward()
                torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
                for (name, value), (_, other) in zip(
                    parent.named_parameters(), tuned.named_parameters(), strict=True
                ):
                    torch.testing.assert_close(
                        value.grad, other.grad, rtol=0, atol=0, msg=name
                    )

    def test_nonzero_head_dropout_is_eval_identity(self):
        parent = FrozenLargeMinGRU().eval()
        tuned = LargeMinGRU(head_dropout=0.45).eval()
        tuned.load_state_dict(parent.state_dict())
        inputs = torch.randn(3, 192, 50)
        with torch.inference_mode():
            torch.testing.assert_close(parent(inputs), tuned(inputs), rtol=0, atol=0)

    def test_training_head_dropout_is_stochastic_and_differentiable(self):
        model = LargeMinGRU(dropout=0, channel_dropout=0, head_dropout=0.5).train()
        inputs = torch.randn(4, 192, 7)
        first, second = model(inputs), model(inputs)
        self.assertFalse(torch.equal(first, second))
        first.square().sum().backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
        self.assertEqual(model.head[1].probability, 0.5)
        self.assertEqual(model.head[3].probability, 0.5)

    def test_eval_parallel_sequential_and_reference_equivalence(self):
        model = LargeMinGRU(head_dropout=0.35).double().eval()
        sequential, reference = copy.deepcopy(model), copy.deepcopy(model)
        inputs = torch.randn(2, 192, 9, dtype=torch.float64)
        expected = model(inputs)
        for variant, output in (
            (sequential, sequential.forward_sequential(inputs)),
            (reference, reference.forward_reference(inputs)[:, -1:]),
        ):
            with self.subTest(path=variant is sequential):
                torch.testing.assert_close(expected, output, rtol=1e-9, atol=1e-11)
                output.square().sum().backward()
        expected.square().sum().backward()
        for variant in (sequential, reference):
            for (name, value), (_, other) in zip(
                model.named_parameters(), variant.named_parameters(), strict=True
            ):
                torch.testing.assert_close(
                    value.grad, other.grad, rtol=1e-8, atol=1e-11, msg=name
                )

    def test_capacity_remains_frozen(self):
        report = capacity_report()
        self.assertEqual(report["parameters"], 1_211_906)
        self.assertEqual(report["fp32_weight_bytes"], 4_847_624)
        self.assertEqual(report["dense_macs_per_prediction"], 13_649_920)
        self.assertEqual(
            report["parameter_groups"]["output_head"]["parameters"], 888_578
        )
        self.assertEqual(
            report["parameter_groups"]["encoder_stem"]["parameters"], 24_704
        )
        self.assertEqual(
            report["parameter_groups"]["temporal_head"]["parameters"], 298_624
        )

    def test_parameter_groups_cover_once_and_apply_head_decay(self):
        model = LargeMinGRU()
        groups = parameter_groups(
            model,
            learning_rate=6e-4,
            encoder_lr_scale=0.5,
            weight_decay=0.02,
            head_weight_decay=0.08,
        )
        self.assertEqual(
            [g["name"] for g in groups],
            ["temporal_head", "encoder_stem", "output_head"],
        )
        self.assertEqual([g["lr"] for g in groups], [6e-4, 3e-4, 6e-4])
        self.assertEqual([g["weight_decay"] for g in groups], [0.02, 0.02, 0.08])
        ids = [id(p) for group in groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        expected_head = {
            id(p) for n, p in model.named_parameters() if n.startswith("head.")
        }
        self.assertEqual({id(p) for p in groups[2]["params"]}, expected_head)
        self.assertEqual(
            [g["weight_decay"] for g in parameter_groups(model, weight_decay=0.03)],
            [0.03, 0.03, 0.03],
        )

    def test_default_group_split_preserves_adamw_update(self):
        parent = FrozenLargeMinGRU(dropout=0, channel_dropout=0).double()
        tuned = LargeMinGRU(dropout=0, channel_dropout=0, head_dropout=0).double()
        tuned.load_state_dict(parent.state_dict())
        original_optimizer = torch.optim.AdamW(
            frozen_parameter_groups(parent, 6e-4, 0.5), weight_decay=0.02
        )
        tuned_optimizer = torch.optim.AdamW(parameter_groups(tuned, 6e-4, 0.5, 0.02))
        inputs = torch.randn(2, 192, 3, dtype=torch.float64)
        parent(inputs).square().sum().backward()
        tuned(inputs).square().sum().backward()
        original_optimizer.step()
        tuned_optimizer.step()
        for (name, value), (_, other) in zip(
            parent.named_parameters(), tuned.named_parameters(), strict=True
        ):
            torch.testing.assert_close(value, other, rtol=0, atol=0, msg=name)

    def test_invalid_regularization_options(self):
        for probability in (-0.1, 1.0, float("nan"), float("inf")):
            with self.subTest(probability=probability):
                with self.assertRaisesRegex(ValueError, "head_dropout"):
                    LargeMinGRU(head_dropout=probability)
        model = LargeMinGRU()
        for kwargs in (
            {"learning_rate": 0},
            {"encoder_lr_scale": float("nan")},
            {"weight_decay": -0.1},
            {"head_weight_decay": float("inf")},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    parameter_groups(model, **kwargs)


if __name__ == "__main__":
    unittest.main()
