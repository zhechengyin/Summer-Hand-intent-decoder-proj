"""Archived regression checks for future-data access in supported pipelines."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np

from indy_loco.data.processing.indy_loco.indy.causal_targets import (
    causal_sample_hold,
    causal_velocity,
)
from indy_loco.experiments.active.phase9_deployment_policy_replay import (
    apply_feature_stats,
    fit_feature_stats,
    load_session_manifest,
    multiscale_counts,
)
from indy_loco.models.tiny.model import TinyTCNGRU

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_CODE = (
    ROOT / "data" / "processing" / "indy_loco" / "indy" / "causal_targets.py",
    ROOT / "experiments" / "active" / "phase9_deployment_policy_replay.py",
    ROOT / "models",
)
FORBIDDEN_CALLS = {"filtfilt", "sosfiltfilt", "gaussian_filter1d", "gradient", "interp"}


class CausalityTests(unittest.TestCase):
    def test_chronological_split_is_complete_and_disjoint(self):
        manifest = load_session_manifest()
        expected = set(manifest["official_indy_sessions"])
        splits = manifest["chronological_split"]
        self.assertEqual(
            {"train": 29, "validation": 4, "test": 4},
            {name: len(sessions) for name, sessions in splits.items()},
        )
        flattened = [session for sessions in splits.values() for session in sessions]
        self.assertEqual(37, len(flattened))
        self.assertEqual(37, len(set(flattened)))
        self.assertEqual(expected, set(flattened))
        self.assertIn("indy_20161005_06", splits["train"])
        self.assertIn("indy_20161206_02", splits["validation"])
        self.assertIn("indy_20170131_02", splits["test"])

    def test_sample_hold_never_reads_the_next_kinematic_sample(self):
        timestamps = np.array([0.0, 0.004, 0.008, 0.012])
        values = np.array([[0.0], [1.0], [2.0], [10_000.0]])
        queries = np.array([0.003, 0.004, 0.007, 0.008])
        np.testing.assert_array_equal(
            causal_sample_hold(timestamps, values, queries).ravel(),
            np.array([0.0, 1.0, 1.0, 2.0]),
        )

    def test_ewma_prefix_does_not_change_when_future_changes(self):
        rng = np.random.default_rng(2)
        original = rng.normal(size=(4, 80)).astype(np.float32)
        changed = original.copy()
        changed[:, 50:] += 1000
        np.testing.assert_array_equal(
            multiscale_counts(original, (0.1,))[:, :50],
            multiscale_counts(changed, (0.1,))[:, :50],
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

    def test_feature_stats_ignore_scored_suffix(self):
        rng = np.random.default_rng(6)
        original = rng.normal(size=(8, 100)).astype(np.float32)
        changed = original.copy()
        changed[:, 60:] += 1000
        first = fit_feature_stats(original, observation_bins=60)
        second = fit_feature_stats(changed, observation_bins=60)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        np.testing.assert_array_equal(
            apply_feature_stats(original[:, :60], first),
            apply_feature_stats(changed[:, :60], second),
        )

    def test_model_prefix_does_not_change_when_future_changes(self):
        import torch

        torch.manual_seed(4)
        model = TinyTCNGRU().eval()
        original = torch.randn(2, 64, 40)
        changed = original.clone()
        changed[:, :, 25:] += 1000
        with torch.no_grad():
            first = model(original)[:, :25]
            second = model(changed)[:, :25]
        torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_model_training_graph_is_also_prefix_invariant(self):
        import torch

        torch.manual_seed(5)
        model = TinyTCNGRU().train()
        original = torch.randn(2, 64, 40)
        changed = original.clone()
        changed[:, :, 25:] -= 1000
        first = model(original)[:, :25]
        second = model(changed)[:, :25]
        torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_supported_code_has_no_known_future_data_calls(self):
        violations = []
        for supported_path in SUPPORTED_CODE:
            paths = (
                [supported_path]
                if supported_path.is_file()
                else supported_path.rglob("*.py")
            )
            for path in paths:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        name = getattr(node.func, "id", None) or getattr(
                            node.func, "attr", None
                        )
                        if name in FORBIDDEN_CALLS:
                            violations.append(
                                f"{path.relative_to(ROOT)}:{node.lineno} calls {name}"
                            )
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        modules = (
                            [node.module]
                            if isinstance(node, ast.ImportFrom)
                            else [alias.name for alias in node.names]
                        )
                        if any(
                            module and module.startswith("src.intent_decoder")
                            for module in modules
                        ):
                            violations.append(
                                f"{path.relative_to(ROOT)}:{node.lineno} imports removed src code"
                            )
        self.assertEqual([], violations, "\n" + "\n".join(violations))


if __name__ == "__main__":
    unittest.main()
