"""Regression tests preventing future-data access in supported pipelines."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np

from src.intent_decoder.features.causal import causal_ewma, causal_velocity
from src.intent_decoder.model.tcn_gru import build_net, causal_config

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_CODE = (
    ROOT / "src",
    ROOT / "data" / "processing",
    ROOT / "experiments" / "active",
    ROOT / "experiments" / "deepblue",
)
FORBIDDEN_CALLS = {"filtfilt", "sosfiltfilt", "gaussian_filter1d", "gradient"}


class CausalityTests(unittest.TestCase):
    def test_ewma_prefix_does_not_change_when_future_changes(self):
        rng = np.random.default_rng(2)
        original = rng.normal(size=(4, 80)).astype(np.float32)
        changed = original.copy()
        changed[:, 50:] += 1000
        np.testing.assert_array_equal(
            causal_ewma(original, 0.1)[:, :50],
            causal_ewma(changed, 0.1)[:, :50],
        )

    def test_velocity_prefix_does_not_change_when_future_changes(self):
        rng = np.random.default_rng(3)
        original = rng.normal(size=(100, 3)).cumsum(0)
        changed = original.copy()
        changed[60:] -= 1000
        np.testing.assert_allclose(
            causal_velocity(original, 0.04, 3.0)[:60],
            causal_velocity(changed, 0.04, 3.0)[:60],
            atol=0,
            rtol=0,
        )

    def test_model_prefix_does_not_change_when_future_changes(self):
        import torch

        torch.manual_seed(4)
        config = {**causal_config(), "F": 8, "H": 8, "dils": [1, 2], "dropout": 0.0}
        model = build_net(config, 4).eval()
        original = torch.randn(2, 4, 40)
        changed = original.clone()
        changed[:, :, 25:] += 1000
        with torch.no_grad():
            first = model(original)[:, :25]
            second = model(changed)[:, :25]
        torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_model_training_graph_is_also_prefix_invariant(self):
        import torch

        torch.manual_seed(5)
        config = {**causal_config(), "F": 8, "H": 8, "dils": [1, 2], "dropout": 0.0}
        model = build_net(config, 4).train()
        original = torch.randn(2, 4, 40)
        changed = original.clone()
        changed[:, :, 25:] -= 1000
        first = model(original)[:, :25]
        second = model(changed)[:, :25]
        torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_bidirectional_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "bidirectional"):
            build_net({**causal_config(), "bidir": True}, 4)

    def test_supported_code_has_no_known_future_data_calls(self):
        violations = []
        for directory in SUPPORTED_CODE:
            for path in directory.rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                        if name in FORBIDDEN_CALLS:
                            violations.append(f"{path.relative_to(ROOT)}:{node.lineno} calls {name}")
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        modules = ([node.module] if isinstance(node, ast.ImportFrom)
                                   else [alias.name for alias in node.names])
                        if any(module and module.startswith("models.tcn_gru") for module in modules):
                            violations.append(
                                f"{path.relative_to(ROOT)}:{node.lineno} imports historical model code"
                            )
        self.assertEqual([], violations, "\n" + "\n".join(violations))


if __name__ == "__main__":
    unittest.main()
