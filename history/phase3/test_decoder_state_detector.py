"""Archived regression checks for the completed second-layer detector."""
from __future__ import annotations

import unittest

import numpy as np

from models.indy_32ch.decoder_state_detector import (
    DecoderPrefixTrace,
    DecoderStateConfig,
    DecoderStateDetector,
    TwoLayerCompatibilityGate,
    extract_decoder_prefix_trace,
)
from models.indy_32ch.drift_detector import DetectorConfig, DriftDetector
from models.indy_32ch.model import build_net, causal_config


class DecoderStateDetectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DecoderStateConfig(
            observation_bins=60,
            chunk_bins=20,
            hidden_components=3,
            warning_quantile=0.90,
            severe_quantile=0.99,
        )
        rng = np.random.default_rng(7)
        self.traces = {}
        for month_index, month in enumerate(("201604", "201606", "201609")):
            for session_index in range(2):
                hidden = rng.normal(
                    loc=month_index * 0.10,
                    scale=1.0,
                    size=(60, 8),
                )
                output = rng.normal(
                    loc=0.0,
                    scale=1.0 + month_index * 0.05,
                    size=(60, 2),
                )
                self.traces[
                    f"indy_{month}{session_index + 1:02d}_0{session_index + 1}"
                ] = DecoderPrefixTrace(hidden, output)

    def test_forward_with_states_preserves_predictions(self) -> None:
        import torch

        config = {
            **causal_config(),
            "F": 8,
            "H": 7,
            "dils": [1, 2],
            "dropout": 0.0,
        }
        net = build_net(config, n_channels=4)
        net.eval()
        values = torch.randn(2, 4, 25)
        with torch.no_grad():
            normal = net(values)
            prediction, states = net.forward_with_states(values)
        self.assertTrue(torch.equal(normal, prediction))
        self.assertEqual(tuple(states.shape), (2, 25, 7))

    def test_detector_ignores_bins_after_observation_prefix(self) -> None:
        detector = DecoderStateDetector(self.config).fit(self.traces)
        candidate = next(iter(self.traces.values()))
        changed_suffix = DecoderPrefixTrace(
            np.concatenate(
                [candidate.hidden_states, np.full((20, 8), 1000.0)], axis=0
            ),
            np.concatenate(
                [candidate.predicted_velocity, np.full((20, 2), -1000.0)], axis=0
            ),
        )
        self.assertEqual(detector.score(candidate), detector.score(changed_suffix))

    def test_trace_extraction_ignores_future_count_bins(self) -> None:
        import torch

        config = {
            **causal_config(),
            "F": 8,
            "H": 7,
            "dils": [1, 2],
            "dropout": 0.0,
        }
        net = build_net(config, n_channels=64)
        rng = np.random.default_rng(23)
        prefix = rng.poisson(0.4, size=(32, 60)).astype(np.float32)
        changed = np.concatenate(
            [prefix, np.full((32, 30), 1000.0, dtype=np.float32)],
            axis=1,
        )
        arguments = (
            np.ones((64, 1), dtype=np.float32) * 0.1,
            np.zeros(2, dtype=np.float32),
            np.ones(2, dtype=np.float32),
            self.config,
            torch.device("cpu"),
        )
        first = extract_decoder_prefix_trace(net, prefix, *arguments)
        second = extract_decoder_prefix_trace(net, changed, *arguments)
        np.testing.assert_array_equal(first.hidden_states, second.hidden_states)
        np.testing.assert_array_equal(
            first.predicted_velocity,
            second.predicted_velocity,
        )

    def test_joint_hidden_and_output_fault_can_abstain(self) -> None:
        detector = DecoderStateDetector(self.config).fit(self.traces)
        rng = np.random.default_rng(11)
        fault = DecoderPrefixTrace(
            rng.normal(loc=20.0, scale=0.1, size=(60, 8)),
            rng.normal(loc=-20.0, scale=0.1, size=(60, 2)),
        )
        score = detector.score(fault)
        self.assertGreaterEqual(score.evidence_count, 2)
        self.assertEqual(score.decision, "abstain")

    def test_two_layer_gate_combines_without_changing_weights(self) -> None:
        rng = np.random.default_rng(19)
        counts = {
            name: rng.poisson(0.4, size=(32, 60)).astype(np.float32)
            for name in self.traces
        }
        gate = TwoLayerCompatibilityGate(
            DriftDetector(
                DetectorConfig(
                    observation_bins=60,
                    n_components=3,
                    warning_quantile=0.90,
                )
            ),
            DecoderStateDetector(self.config),
        ).fit(counts, self.traces)
        name = next(iter(self.traces))
        score = gate.score(counts[name], self.traces[name])
        self.assertIn(score.decision, {"pass", "warning", "abstain"})
        self.assertEqual(
            score.decision,
            "abstain"
            if (
                score.layer1.combined_decision == "abstain"
                or score.layer2.decision == "abstain"
            )
            else (
                "warning"
                if score.layer1.combined_decision == "warning"
                else "pass"
            ),
        )


if __name__ == "__main__":
    unittest.main()
