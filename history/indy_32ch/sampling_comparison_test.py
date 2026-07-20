"""Tests for the fair Indy sampling comparison."""
from __future__ import annotations

import unittest

import numpy as np

from history.indy_32ch.train_sampling_comparison import (
    draw_epoch_indices,
    summarize_values,
)


class SamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sessions = [
            "indy_20160401_01",
            "indy_20160402_01",
            "indy_20160601_01",
        ]
        self.lengths = {
            "indy_20160401_01": 10,
            "indy_20160402_01": 20,
            "indy_20160601_01": 30,
        }

    def draw(self, strategy: str):
        return draw_epoch_indices(
            strategy,
            self.sessions,
            self.lengths,
            np.random.default_rng(42),
        )

    def test_every_strategy_draws_the_same_epoch_size(self):
        for strategy in ("window", "session", "month"):
            indices, _, _ = self.draw(strategy)
            self.assertEqual(60, len(indices))
            self.assertTrue(np.all(indices >= 0))
            self.assertTrue(np.all(indices < 60))

    def test_window_sampling_preserves_natural_session_lengths(self):
        indices, session_draws, month_draws = self.draw("window")
        np.testing.assert_array_equal(np.sort(indices), np.arange(60))
        self.assertEqual(self.lengths, session_draws)
        self.assertEqual({"2016-04": 30, "2016-06": 30}, month_draws)

    def test_session_sampling_equalizes_sessions(self):
        _, session_draws, _ = self.draw("session")
        self.assertEqual({session: 20 for session in self.sessions}, session_draws)

    def test_month_sampling_equalizes_month_then_session(self):
        _, session_draws, month_draws = self.draw("month")
        self.assertEqual({"2016-04": 30, "2016-06": 30}, month_draws)
        self.assertEqual(15, session_draws["indy_20160401_01"])
        self.assertEqual(15, session_draws["indy_20160402_01"])
        self.assertEqual(30, session_draws["indy_20160601_01"])

    def test_summary_uses_sample_standard_deviation(self):
        summary = summarize_values([1.0, 2.0, 3.0])
        self.assertEqual(3, summary["n"])
        self.assertEqual([1.0, 2.0, 3.0], summary["values"])
        self.assertAlmostEqual(2.0, summary["mean"])
        self.assertAlmostEqual(1.0, summary["std"])


if __name__ == "__main__":
    unittest.main()
