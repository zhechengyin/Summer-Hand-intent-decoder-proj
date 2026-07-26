"""Archived regression checks for Phase-3b isolation and frozen protocol."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).with_name("phase3b_leave_one_month_out.py")
SPEC = importlib.util.spec_from_file_location("phase3b", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {SCRIPT}")
phase3b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phase3b)


class Phase3bProtocolTests(unittest.TestCase):
    def test_development_pool_excludes_january(self) -> None:
        names, by_month = phase3b.development_sessions()
        self.assertEqual(len(names), 33)
        self.assertEqual(tuple(by_month), phase3b.EXPECTED_MONTHS)
        self.assertTrue(all(phase3b.session_month(name) < "2017-01" for name in names))

    def test_every_fold_removes_the_complete_month(self) -> None:
        names, by_month = phase3b.development_sessions()
        for month in phase3b.EXPECTED_MONTHS:
            training, held = phase3b.partition_fold(names, by_month, month)
            self.assertFalse(set(training) & set(held))
            self.assertEqual({phase3b.session_month(name) for name in held}, {month})
            self.assertNotIn(month, {phase3b.session_month(name) for name in training})
            self.assertEqual(len(training) + len(held), 33)

    def test_yaml_matches_frozen_seed_epoch_and_channels(self) -> None:
        channels, model_yaml, detector = phase3b.load_frozen_protocol()
        self.assertEqual(channels.shape, (32,))
        self.assertEqual(model_yaml["training"]["seed"], phase3b.FROZEN_SEED)
        self.assertEqual(
            model_yaml["training"]["checkpoint_epoch"], phase3b.CHECKPOINT_EPOCH
        )
        self.assertEqual(detector.observation_bins, 1500)

    def test_january_counts_are_rejected_before_path_access(self) -> None:
        channels, _, _ = phase3b.load_frozen_protocol()
        with self.assertRaises(ValueError):
            phase3b.load_counts_only("indy_20170124_01", channels)


if __name__ == "__main__":
    unittest.main()
