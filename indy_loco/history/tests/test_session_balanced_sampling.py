"""Archived checks for the frozen session-balanced Indy sampler."""

from __future__ import annotations

import unittest

import numpy as np

from indy_loco.models.indy_32ch.sampling import draw_session_balanced_indices


class SessionBalancedSamplingTests(unittest.TestCase):
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

    def test_draw_preserves_epoch_size_and_equalizes_sessions(self):
        indices, session_draws, month_draws = draw_session_balanced_indices(
            self.sessions,
            self.lengths,
            np.random.default_rng(42),
        )
        self.assertEqual(60, len(indices))
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < 60))
        self.assertEqual({session: 20 for session in self.sessions}, session_draws)
        self.assertEqual({"2016-04": 40, "2016-06": 20}, month_draws)

    def test_remainder_is_distributed_without_changing_total(self):
        lengths = {session: 11 for session in self.sessions}
        _, session_draws, _ = draw_session_balanced_indices(
            self.sessions,
            lengths,
            np.random.default_rng(7),
        )
        self.assertEqual(33, sum(session_draws.values()))
        self.assertLessEqual(
            max(session_draws.values()) - min(session_draws.values()), 1
        )

    def test_invalid_session_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            draw_session_balanced_indices(
                self.sessions,
                {**self.lengths, self.sessions[0]: 0},
                np.random.default_rng(42),
            )


if __name__ == "__main__":
    unittest.main()
