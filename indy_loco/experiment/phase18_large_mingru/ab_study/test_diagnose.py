"""Small CPU checks for training-only calibration and bounded lag diagnostics."""

import unittest

import numpy as np

from . import diagnose


class DiagnoseTests(unittest.TestCase):
    def test_training_affine_has_known_solution_and_no_validation_input(self):
        prediction = np.column_stack((np.arange(20.0), np.arange(20.0) ** 2))
        target = prediction * [1.2, 0.7] + [0.4, -0.2]
        gain, offset = diagnose.fit_training_affine(target, prediction)
        np.testing.assert_allclose(gain, [1.2, 0.7])
        np.testing.assert_allclose(offset, [0.4, -0.2])
        validation_prediction = np.array([[40.0, 2.0], [80.0, 5.0]])
        np.testing.assert_allclose(
            validation_prediction * gain + offset,
            validation_prediction * [1.2, 0.7] + [0.4, -0.2],
        )

    def test_lags_share_targets_and_never_cross_adjacent_reach_boundary(self):
        bins = np.r_[np.arange(0, 12), np.arange(12, 24), np.arange(40, 52)]
        bounds = np.array([[0, 12], [12, 24], [40, 52]])
        ids = diagnose.reach_ids(bins, bounds, [0, 1, 2])
        selected, pairs = diagnose.common_lag_pairs(bins, ids)
        np.testing.assert_array_equal(
            bins[selected], np.r_[np.arange(3, 9), np.arange(15, 21), np.arange(43, 49)]
        )
        for lag, predictions in pairs.items():
            np.testing.assert_array_equal(ids[selected], ids[predictions])
            np.testing.assert_array_equal(bins[selected] - bins[predictions], lag)
            self.assertEqual(len(selected), len(predictions))

    def test_negative_lag_detects_known_delayed_prediction(self):
        bins = np.arange(30)
        ids = np.zeros(30, dtype=int)
        target = np.column_stack((np.sin(bins / 3), np.cos(bins / 3)))
        prediction = np.roll(target, 2, axis=0)
        selected, pairs = diagnose.common_lag_pairs(bins, ids)
        scores = {
            lag: diagnose.axis_metrics(target[selected], prediction[index])[0]["r2"]
            for lag, index in pairs.items()
        }
        self.assertEqual(max(scores, key=scores.get), -2)
        self.assertAlmostEqual(scores[-2], 1.0)

    def test_acceleration_does_not_bridge_split_gaps_or_reaches(self):
        bins = np.array([0, 1, 5, 6, 7])
        ids = np.array([0, 0, 1, 1, 2])
        target = np.column_stack((np.arange(5.0), np.zeros(5)))
        _, acceleration = diagnose.motion_values(bins, ids, target)
        self.assertTrue(np.isnan(acceleration[[0, 2, 4]]).all())
        np.testing.assert_allclose(acceleration[[1, 3]], [25.0, 25.0])

    def test_regime_thresholds_are_unaffected_by_validation_labels(self):
        bins = np.arange(20)
        ids = np.zeros(20, dtype=int)
        train = np.column_stack((bins * 0.1, bins * 0.2))
        validation = train + 0.3
        first = diagnose.regime_diagnostics(
            bins, ids, train, bins, ids, validation, validation
        )
        second = diagnose.regime_diagnostics(
            bins, ids, train, bins, ids, validation * 100, validation
        )
        self.assertEqual(
            first["thresholds_from_training_only"],
            second["thresholds_from_training_only"],
        )

    def test_axis_bias_scale_and_r2_are_distinct(self):
        y = np.column_stack((np.arange(10.0), np.arange(10.0) * 2))
        p = 0.5 * y + 1.0
        rows = diagnose.axis_metrics(y, p)
        for axis, row in enumerate(rows):
            self.assertAlmostEqual(row["correlation_squared"], 1.0)
            self.assertAlmostEqual(row["prediction_target_std_ratio"], 0.5)
            self.assertAlmostEqual(row["bias"], float((p[:, axis] - y[:, axis]).mean()))
            self.assertLess(row["r2"], 1.0)


if __name__ == "__main__":
    unittest.main()
