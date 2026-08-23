"""Synthetic sanity checks for all selectable feature extractors."""

import numpy as np

from channel_selection import select_channel_features
from data import downsample_ecog
from feature_extraction import (
    bandpower_psd_features,
    fft_spectrum_features,
    filterbank_features,
    mst_psd_features,
)
from models import MstAnnClassifier, TinyCnnClassifier, build_depth4_tree


def main():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 4, 3000))
    x_ds = downsample_ecog(x, 1000, 100)
    assert x_ds.shape == (4, 4, 300)

    mst, mst_axis = mst_psd_features(
        x_ds, fs=100, fmin=1, fmax=5, p=0.52, q=1, verbose=False
    )
    assert mst.shape == (4, 4, 5)
    assert mst_axis.tolist() == [1, 2, 3, 4, 5]

    band, band_axis = bandpower_psd_features(x_ds, fs=100)
    assert band.shape == (4, 4, 5)
    assert len(band_axis) == 5

    fft, fft_axis = fft_spectrum_features(x_ds, fs=100, fmin=1, fmax=35)
    assert fft.shape[:2] == (4, 4)
    assert fft.shape[2] == len(fft_axis)
    assert len(fft_axis) == 103

    fb, fb_axis = filterbank_features(x_ds, fs=100)
    assert fb.shape == (4, 4, 5)
    assert len(fb_axis) == 5

    flat = select_channel_features(band, [0, 2])
    assert flat.shape == (4, 10)

    ann_x = rng.normal(size=(40, 15))
    ann_y = np.asarray([-1, 1] * 20)
    ann = MstAnnClassifier(epochs=3, patience=2, random_state=0).fit(ann_x, ann_y)
    assert ann.fc1_weight_shape == (15, 64)
    assert ann.fc2_weight_shape == (64, 2)
    assert ann.predict(ann_x).shape == (40,)

    cnn_x = rng.normal(size=(40, 3 * 35))
    cnn = TinyCnnClassifier(
        input_channels=3,
        features_per_channel=35,
        epochs=3,
        patience=2,
        random_state=0,
        device="cpu",
    ).fit(cnn_x, ann_y)
    assert cnn.predict(cnn_x).shape == (40,)
    assert cnn.parameter_count == (8 * 3 * 3 + 8) + (8 * 2 + 2)

    tree = build_depth4_tree(random_state=0).fit(ann_x, ann_y)
    assert tree.get_depth() <= 4
    assert tree.predict(ann_x).shape == (40,)

    print("All synthetic sanity checks passed.")


if __name__ == "__main__":
    main()
