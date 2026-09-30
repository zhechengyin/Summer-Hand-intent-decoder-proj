"""Causality, window reset, arithmetic and initialization tests for A/B models."""

import copy
import unittest

import torch

from ..model import MinGRUBlock
from .model import (
    MODEL_NAMES,
    CausalTemporalFrontend,
    build_model,
    capacity_report,
    parameter_groups,
)


class ABModelTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        torch.manual_seed(1803)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    def test_same_seed_shared_parameters_are_identical(self):
        torch.manual_seed(43)
        model_a = build_model("mingru_a")
        torch.manual_seed(43)
        model_b = build_model("mingru_b")
        a_state, b_state = model_a.state_dict(), model_b.state_dict()
        self.assertEqual(
            set(b_state) - set(a_state),
            {
                "frontend.norm.weight",
                "frontend.norm.bias",
                "frontend.depthwise.weight",
                "frontend.depthwise.bias",
                "frontend.pointwise.weight",
                "frontend.pointwise.bias",
            },
        )
        for name, value in a_state.items():
            torch.testing.assert_close(value, b_state[name], rtol=0, atol=0, msg=name)
        for model in (model_a, model_b):
            self.assertTrue(all(type(block) is MinGRUBlock for block in model.blocks))

    def test_frontend_causal_prefix_and_local_support(self):
        frontend = CausalTemporalFrontend().double().eval()
        inputs = torch.randn(2, 12, 128, dtype=torch.float64)
        changed = inputs.clone()
        changed[:, 7:] = torch.randn_like(changed[:, 7:]) * 100
        torch.testing.assert_close(
            frontend(inputs)[:, :7], frontend(changed)[:, :7], rtol=0, atol=0
        )
        torch.testing.assert_close(
            frontend(inputs)[:, :7], frontend(inputs[:, :7]), rtol=1e-12, atol=1e-12
        )
        self.assertEqual(frontend(inputs[:, :1]).shape, (2, 1, 128))
        variable = inputs.clone().requires_grad_(True)
        frontend(variable)[:, 6].square().sum().backward()
        self.assertEqual(float(variable.grad[:, :2].abs().sum()), 0)
        self.assertEqual(float(variable.grad[:, 7:].abs().sum()), 0)
        self.assertGreater(float(variable.grad[:, 2:6].abs().sum()), 0)

    def test_pruned_outputs_and_gradients_match_full_reference(self):
        for name in MODEL_NAMES:
            for recurrence in ("parallel", "sequential"):
                with self.subTest(model=name, recurrence=recurrence):
                    actual_model = build_model(name, recurrence=recurrence).double()
                    reference_model = copy.deepcopy(actual_model)
                    values = torch.randn(2, 192, 7, dtype=torch.float64)
                    inputs = values.clone().requires_grad_(True)
                    reference_inputs = values.clone().requires_grad_(True)
                    torch.manual_seed(71)
                    actual = actual_model(inputs)
                    torch.manual_seed(71)
                    expected = reference_model.forward_reference(reference_inputs)[
                        :, -1:
                    ]
                    self.assertEqual(actual.shape, (2, 1, 2))
                    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-11)
                    actual.square().sum().backward()
                    expected.square().sum().backward()
                    torch.testing.assert_close(
                        inputs.grad, reference_inputs.grad, rtol=1e-8, atol=1e-11
                    )
                    for (key, value), (_, reference) in zip(
                        actual_model.named_parameters(),
                        reference_model.named_parameters(),
                        strict=True,
                    ):
                        torch.testing.assert_close(
                            value.grad, reference.grad, rtol=1e-8, atol=1e-11, msg=key
                        )

    def test_parallel_sequential_fp32_window_boundaries(self):
        for name in MODEL_NAMES:
            model = build_model(name).eval()
            for length in (1, 50):
                with self.subTest(model=name, length=length):
                    inputs = torch.randn(3, 192, length)
                    with torch.inference_mode():
                        torch.testing.assert_close(
                            model(inputs),
                            model.forward_sequential(inputs),
                            rtol=2e-4,
                            atol=2e-6,
                        )

    def test_full_model_prefix_causality(self):
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                model = build_model(name).double().eval()
                values = torch.randn(2, 192, 12, dtype=torch.float64)
                with torch.inference_mode():
                    full = model.forward_reference(values)
                    prefix = model.forward_reference(values[:, :, :7])
                torch.testing.assert_close(full[:, :7], prefix, rtol=1e-10, atol=1e-11)

    def test_history_used_and_no_state_leaks(self):
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                model = build_model(name).double().eval()
                inputs = torch.randn(2, 192, 7, dtype=torch.float64, requires_grad=True)
                model(inputs).square().sum().backward()
                self.assertGreater(float(inputs.grad[:, :, 0].abs().sum()), 1e-8)
                with torch.inference_mode():
                    first = model.forward_sequential(inputs[:1])
                    model.forward_sequential(torch.randn_like(inputs))
                    repeat = model.forward_sequential(inputs[:1])
                    batched = model.forward_sequential(inputs)
                torch.testing.assert_close(first, repeat, rtol=0, atol=0)
                torch.testing.assert_close(first, batched[:1], rtol=1e-10, atol=1e-11)
                self.assertEqual(list(model.named_buffers()), [])

    def test_layer_bulk_projection_and_last_only_shapes(self):
        model = build_model("mingru_b").eval()
        calls, handles = {}, []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):

                def record(_module, inputs, _outputs, name=name):
                    calls.setdefault(name, []).append(tuple(inputs[0].shape))

                handles.append(module.register_forward_hook(record))
        try:
            with torch.inference_mode():
                model.forward_sequential(torch.randn(2, 192, 50))
        finally:
            for handle in handles:
                handle.remove()
        self.assertTrue(all(len(shapes) == 1 for shapes in calls.values()))
        self.assertEqual(calls["frontend.depthwise"], [(2, 128, 54)])
        self.assertEqual(calls["frontend.pointwise"], [(2, 50, 128)])
        for block in range(3):
            self.assertEqual(calls[f"blocks.{block}.mixer.projection"], [(2, 50, 128)])
        self.assertEqual(calls["blocks.0.ffn.0"], [(2, 50, 128)])
        self.assertEqual(calls["blocks.1.ffn.0"], [(2, 50, 128)])
        self.assertEqual(calls["blocks.2.ffn.0"], [(2, 1, 128)])
        self.assertEqual(calls["head.0"], [(2, 1, 128)])
        self.assertEqual(calls["head.2"], [(2, 1, 256)])

    def test_capacity_and_groups(self):
        expected = {
            "mingru_a": (356_866, 1_427_464, 12_796_416),
            "mingru_b": (374_402, 1_497_608, 13_647_616),
        }
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                report = capacity_report(name)
                self.assertEqual(
                    (
                        report["parameters"],
                        report["fp32_weight_bytes"],
                        report["dense_macs_per_prediction"],
                    ),
                    expected[name],
                )
                single = capacity_report(name, 1)
                self.assertEqual(
                    single["dense_macs_per_prediction"],
                    single["dense_macs_full_sequence_reference"],
                )
                model = build_model(name)
                groups = parameter_groups(model, 6e-4, 0.5, 0.02, 0.04)
                self.assertEqual(
                    [g["name"] for g in groups],
                    ["temporal_head", "encoder_stem", "output_head"],
                )
                self.assertEqual([g["lr"] for g in groups], [6e-4, 3e-4, 6e-4])
                self.assertEqual(
                    [g["weight_decay"] for g in groups], [0.02, 0.02, 0.04]
                )
                ids = [id(p) for group in groups for p in group["params"]]
                self.assertEqual(len(ids), len(set(ids)))
                self.assertEqual(set(ids), {id(p) for p in model.parameters()})
                self.assertTrue(
                    {id(p) for p in model.frontend.parameters()}
                    <= {id(p) for p in groups[0]["params"]}
                )
                self.assertEqual(
                    {id(p) for p in model.head.parameters()},
                    {id(p) for p in groups[2]["params"]},
                )
                self.assertEqual(
                    [
                        g["weight_decay"]
                        for g in parameter_groups(model, weight_decay=0.03)
                    ],
                    [0.03, 0.03, 0.03],
                )

    def test_input_validation(self):
        for name in MODEL_NAMES:
            model = build_model(name)
            for shape in (
                (192, 50),
                (1, 191, 50),
                (1, 192, 0),
                (1, 192, 51),
                (0, 192, 3),
            ):
                with self.subTest(model=name, shape=shape):
                    with self.assertRaisesRegex(ValueError, "Expected floating batch"):
                        model(torch.zeros(shape))
            with self.assertRaises(ValueError):
                model(torch.zeros(1, 192, 5, dtype=torch.int64))
        with self.assertRaises(ValueError):
            build_model("other")
        for bins in (0, 51, 1.5):
            with self.assertRaises(ValueError):
                capacity_report("mingru_a", bins)
        with self.assertRaises(ValueError):
            parameter_groups(build_model("mingru_a"), head_weight_decay=-1)


if __name__ == "__main__":
    unittest.main()
