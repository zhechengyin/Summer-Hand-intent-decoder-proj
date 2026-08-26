from __future__ import annotations

import numpy as np

from indy_loco.experiments.active.phase10_session_deployment_candidates import (
    CALIBRATION_BINS,
    EWMA_ALPHA,
    FEATURES,
    PHYSICAL_CHANNELS,
    WINDOW_BINS,
    continuous_features,
    fit_training_floor,
    rolling_inputs,
)


def test_continuous_features_are_raw_then_causal_ewma() -> None:
    counts = np.arange(PHYSICAL_CHANNELS * 4, dtype=np.float32).reshape(96, 4)
    features = continuous_features(counts)

    assert features.shape == (FEATURES, 4)
    np.testing.assert_array_equal(features[:PHYSICAL_CHANNELS], counts)
    np.testing.assert_array_equal(features[PHYSICAL_CHANNELS:, 0], counts[:, 0])
    np.testing.assert_allclose(
        features[PHYSICAL_CHANNELS:, 1],
        EWMA_ALPHA * counts[:, 1] + (1.0 - EWMA_ALPHA) * counts[:, 0],
        rtol=0,
        atol=1e-6,
    )


def test_rolling_inputs_are_oldest_to_newest_and_end_inclusive() -> None:
    timeline = np.broadcast_to(
        np.arange(CALIBRATION_BINS + 4, dtype=np.float32),
        (FEATURES, CALIBRATION_BINS + 4),
    ).copy()
    end_bins = np.asarray([CALIBRATION_BINS - 1, CALIBRATION_BINS + 2])

    windows = rolling_inputs(timeline, end_bins)

    assert windows.shape == (2, FEATURES, WINDOW_BINS)
    np.testing.assert_array_equal(
        windows[0, 0],
        np.arange(CALIBRATION_BINS - WINDOW_BINS, CALIBRATION_BINS),
    )
    assert windows[1, 0, -1] == CALIBRATION_BINS + 2


def test_floor_uses_fallback_only_for_training_silent_features() -> None:
    rng = np.random.default_rng(10)
    bins = CALIBRATION_BINS * 2
    counts = rng.poisson(0.5, size=(PHYSICAL_CHANNELS, bins)).astype(np.float32)
    counts[3] = 0
    bounds = np.asarray([[0, bins]], dtype=np.int64)
    fallback = np.linspace(0.01, 0.2, FEATURES, dtype=np.float32)

    floor, metadata = fit_training_floor(
        counts,
        bounds,
        np.asarray([0]),
        fallback,
    )

    assert floor.shape == (FEATURES,)
    assert np.all(np.isfinite(floor))
    assert np.all(floor > 0)
    assert 3 in metadata["silent_feature_fallback_indices"]
    assert PHYSICAL_CHANNELS + 3 in metadata["silent_feature_fallback_indices"]
    assert floor[3] == fallback[3]
    assert floor[PHYSICAL_CHANNELS + 3] == fallback[PHYSICAL_CHANNELS + 3]
