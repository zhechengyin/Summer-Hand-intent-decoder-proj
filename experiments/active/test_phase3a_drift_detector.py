"""Small regression checks for the active Phase-3a detector."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.indy_32ch.drift_detector import (  # noqa: E402
    DetectorConfig,
    DriftDetector,
    GaussianSummary,
    assert_pre_january,
    gaussian_kld,
)


class DriftDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(12)
        names = (
            "indy_20160401_01",
            "indy_20160402_01",
            "indy_20160601_01",
            "indy_20160602_01",
            "indy_20160901_01",
            "indy_20160902_01",
        )
        channel_rates = np.linspace(0.2, 2.0, 32)[:, None]
        self.sessions = {
            name: rng.poisson(
                channel_rates * (1.0 + 0.05 * index), size=(32, 1510)
            ).astype(np.float32)
            for index, name in enumerate(names)
        }
        self.config = DetectorConfig(
            observation_bins=1500,
            n_components=3,
            warning_quantile=0.90,
        )

    def test_january_is_a_hard_failure(self) -> None:
        with self.assertRaises(ValueError):
            assert_pre_january(["indy_20170124_01"])

    def test_identical_gaussians_have_zero_kld(self) -> None:
        summary = GaussianSummary(np.zeros(3), np.eye(3))
        self.assertAlmostEqual(gaussian_kld(summary, summary), 0.0, places=10)

    def test_samples_after_warmup_do_not_change_score(self) -> None:
        detector = DriftDetector(self.config).fit(self.sessions)
        candidate = self.sessions["indy_20160401_01"].copy()
        changed_suffix = candidate.copy()
        changed_suffix[:, 1500:] = 1000
        self.assertEqual(detector.score(candidate), detector.score(changed_suffix))

    def test_score_has_conservative_three_level_decision(self) -> None:
        detector = DriftDetector(self.config).fit(self.sessions)
        score = detector.score(self.sessions["indy_20160601_01"])
        self.assertIn(score.combined_decision, {"pass", "warning", "abstain"})
        if score.combined_decision == "abstain":
            self.assertGreaterEqual(score.combined_evidence_count, 2)


if __name__ == "__main__":
    unittest.main()
