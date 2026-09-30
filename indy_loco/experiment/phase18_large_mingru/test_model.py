"""Dependency-pruning and recurrence-equivalence tests for the larger minGRU."""

import copy
import unittest

import torch

from .model import LargeMinGRU, MinGRU, capacity_report, parameter_groups


def check_pruned_forward_and_gradients_match_full_reference(recurrence):
    optimized = (
        LargeMinGRU(dropout=0.1, channel_dropout=0.1, recurrence=recurrence)
        .double()
        .train()
    )
    reference = copy.deepcopy(optimized)
    inputs = torch.randn(2, 192, 7, dtype=torch.float64, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_(True)
    torch.manual_seed(78)
    actual = optimized(inputs)
    torch.manual_seed(78)
    expected = reference.forward_reference(reference_inputs)[:, -1:]
    assert actual.shape == (2, 1, 2)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
    target = torch.randn_like(actual)
    (actual - target).square().sum().backward()
    (expected - target).square().sum().backward()
    torch.testing.assert_close(
        inputs.grad, reference_inputs.grad, rtol=1e-9, atol=1e-11
    )
    for (name, parameter), (other_name, other) in zip(
        optimized.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == other_name
        torch.testing.assert_close(
            parameter.grad, other.grad, rtol=1e-9, atol=1e-11, msg=name
        )


def check_parallel_and_sequential_forward_and_gradients_match():
    parallel = LargeMinGRU(dropout=0, channel_dropout=0).double().eval()
    sequential = copy.deepcopy(parallel)
    inputs = torch.randn(2, 192, 11, dtype=torch.float64, requires_grad=True)
    other_inputs = inputs.detach().clone().requires_grad_(True)
    actual = parallel(inputs)
    expected = sequential.forward_sequential(other_inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-11)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(inputs.grad, other_inputs.grad, rtol=1e-8, atol=1e-11)
    for (name, parameter), (_, other) in zip(
        parallel.named_parameters(), sequential.named_parameters(), strict=True
    ):
        torch.testing.assert_close(
            parameter.grad, other.grad, rtol=1e-8, atol=1e-11, msg=name
        )


def check_fp32_scan_and_recurrence_at_window_boundaries(length):
    model = LargeMinGRU().eval()
    inputs = torch.randn(3, 192, length)
    with torch.inference_mode():
        torch.testing.assert_close(
            model(inputs), model.forward_sequential(inputs), rtol=2e-4, atol=2e-6
        )


def check_no_state_leaks_between_windows_or_batch_members():
    model = LargeMinGRU().eval()
    inputs = torch.randn(2, 192, 50)
    with torch.inference_mode():
        first = model.forward_sequential(inputs[:1])
        model.forward_sequential(torch.randn_like(inputs))
        repeated = model.forward_sequential(inputs[:1])
        together = model.forward_sequential(inputs)
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    torch.testing.assert_close(first, together[:1], rtol=1e-5, atol=1e-6)
    assert not any("state" in name for name, _ in model.named_buffers())


def check_earlier_inputs_still_affect_final_prediction():
    model = LargeMinGRU(dropout=0, channel_dropout=0).double().eval()
    inputs = torch.randn(1, 192, 4, dtype=torch.float64, requires_grad=True)
    model(inputs).square().sum().backward()
    assert float(inputs.grad[:, :, 0].abs().sum()) > 1e-8


def check_bulk_linear_calls_reuse_weights_and_prune_only_final_ffn():
    model = LargeMinGRU().eval()
    calls = {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):

            def record(_module, arguments, _output, name=name):
                calls.setdefault(name, []).append(tuple(arguments[0].shape))

            handles.append(module.register_forward_hook(record))
    try:
        with torch.inference_mode():
            model.forward_sequential(torch.randn(2, 192, 50))
    finally:
        for handle in handles:
            handle.remove()
    assert all(len(shapes) == 1 for shapes in calls.values())
    for block in range(3):
        assert calls[f"blocks.{block}.mixer.projection"] == [(2, 50, 128)]
    for block in range(2):
        assert calls[f"blocks.{block}.ffn.0"] == [(2, 50, 128)]
    assert calls["blocks.2.ffn.0"] == [(2, 1, 128)]
    assert calls["head.0"] == [(2, 1, 128)]
    assert calls["head.2"] == [(2, 1, 768)]
    assert calls["head.4"] == [(2, 1, 1024)]


def check_capacity_and_optimizer_groups():
    report = capacity_report()
    assert report["parameters"] == 1_211_906
    assert report["fp32_weight_bytes"] == 4_847_624
    assert report["dense_macs_per_prediction"] == 13_649_920
    assert report["dense_macs_proposed_last_head_only"] == 16_861_184
    assert report["dense_macs_full_sequence_reference"] == 60_313_600
    assert report["dense_macs_saved_from_proposed"] == 3_211_264
    single = capacity_report(1)
    assert (
        single["dense_macs_per_prediction"]
        == single["dense_macs_full_sequence_reference"]
    )
    model = LargeMinGRU()
    groups = parameter_groups(model, learning_rate=6e-4, encoder_lr_scale=0.5)
    assert [group["name"] for group in groups] == ["temporal_head", "encoder_stem"]
    assert [group["lr"] for group in groups] == [6e-4, 3e-4]
    identities = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(identities) == len(set(identities)) == len(list(model.parameters()))


def check_paired_channel_dropout_preserved():
    model = LargeMinGRU(channel_dropout=0.25).train()
    values = model.channel_dropout(torch.ones(8, 192, 5))
    torch.testing.assert_close(values[:, :96], values[:, 96:], rtol=0, atol=0)
    torch.testing.assert_close(
        values[:, :, :1].expand_as(values), values, rtol=0, atol=0
    )
    assert set(values.unique().tolist()) == {0.0, float(torch.tensor(1 / 0.75))}


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        torch.manual_seed(17018)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    def test_pruned_forward_and_gradients(self):
        for recurrence in ("parallel", "sequential"):
            with self.subTest(recurrence=recurrence):
                check_pruned_forward_and_gradients_match_full_reference(recurrence)

    def test_parallel_and_sequential_forward_and_gradients(self):
        check_parallel_and_sequential_forward_and_gradients_match()

    def test_fp32_window_boundaries(self):
        for length in (1, 50):
            with self.subTest(length=length):
                check_fp32_scan_and_recurrence_at_window_boundaries(length)

    def test_independent_windows(self):
        check_no_state_leaks_between_windows_or_batch_members()

    def test_historical_inputs(self):
        check_earlier_inputs_still_affect_final_prediction()

    def test_bulk_linear_calls(self):
        check_bulk_linear_calls_reuse_weights_and_prune_only_final_ffn()

    def test_capacity(self):
        check_capacity_and_optimizer_groups()

    def test_paired_dropout(self):
        check_paired_channel_dropout_preserved()

    def test_invalid_shapes(self):
        shapes = [(192, 50), (1, 191, 50), (1, 192, 0), (1, 192, 51), (0, 192, 50)]
        model = LargeMinGRU()
        for shape in shapes:
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "Expected floating batch"):
                    model(torch.zeros(shape))

    def test_invalid_options(self):
        with self.assertRaisesRegex(ValueError, "Expected floating batch"):
            LargeMinGRU()(torch.zeros(1, 192, 50, dtype=torch.int64))
        for kwargs in (
            {"dropout": -0.1},
            {"channel_dropout": 1},
            {"recurrence": "persistent"},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    LargeMinGRU(**kwargs)
        with self.assertRaisesRegex(ValueError, "recurrence"):
            MinGRU()(torch.zeros(1, 2, 128), recurrence="persistent")


if __name__ == "__main__":
    unittest.main()
