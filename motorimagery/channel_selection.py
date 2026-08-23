from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split


def select_channel_features(features: np.ndarray, channels) -> np.ndarray:
    """Convert [trials, channels, frequencies] to a 2-D feature matrix."""
    channels = np.asarray(channels, dtype=int)
    return features[:, channels, :].reshape(features.shape[0], -1)


def rank_channels_repeated_cv(features: np.ndarray, y: np.ndarray, estimator, n_splits: int = 10, n_repeats: int = 10, random_state: int = 2020) -> tuple[np.ndarray, np.ndarray]:
    """Rank channels from lowest to highest individual CV accuracy.

    Returns
    -------
    order_low_to_high : ndarray
        0-based channel indices, worst first.
    mean_accuracies : ndarray
        Mean repeated-CV accuracy for each channel.
    """
    n_channels = features.shape[1]
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    scores = np.empty(n_channels, dtype=np.float64)

    for ch in range(n_channels):
        x_ch = features[:, ch, :]
        fold_scores = []
        for train_idx, val_idx in cv.split(x_ch, y):
            model = clone(estimator)
            model.fit(x_ch[train_idx], y[train_idx])
            pred = model.predict(x_ch[val_idx])
            fold_scores.append(accuracy_score(y[val_idx], pred))
        scores[ch] = float(np.mean(fold_scores))
        print(f"Channel {ch + 1:02d}: repeated-CV accuracy = {scores[ch]:.4f}")

    order = np.argsort(scores)  # low -> high, matching paper Step 2
    return order, scores


def optimized_wrapper_selection(features: np.ndarray, y: np.ndarray, estimator, ranked_channels_low_to_high: np.ndarray, validation_fraction: float = 0.20, random_state: int = 2020, minimum_channels: int = 1) -> tuple[list[int], dict]:
    """
    The threshold m is the all-channel validation accuracy.
    A candidate deletion is accepted if its validation accuracy >= m.
    """
    idx = np.arange(len(y))
    tr_idx, va_idx = train_test_split(idx, test_size=validation_fraction, random_state=random_state, stratify=y)

    active = list(range(features.shape[1]))

    def validation_accuracy(channels):
        x_train = select_channel_features(features[tr_idx], channels)
        x_val = select_channel_features(features[va_idx], channels)
        model = clone(estimator)
        model.fit(x_train, y[tr_idx])
        return accuracy_score(y[va_idx], model.predict(x_val))

    baseline_m = validation_accuracy(active)
    print(f"\nWrapper threshold m (all-channel sub-validation accuracy): {baseline_m:.4f}")

    history = []
    pass_id = 0

    while True:
        pass_id += 1
        removed_this_pass = 0
        print(f"\nWrapper pass {pass_id}; active channels = {len(active)}")

        # Keep the original ranking order, but skip channels already removed.
        for ch in ranked_channels_low_to_high:
            ch = int(ch)
            if ch not in active:
                continue
            if len(active) <= minimum_channels:
                break

            candidate = [c for c in active if c != ch]
            acc = validation_accuracy(candidate)

            accepted = acc >= baseline_m
            history.append({
                "pass": pass_id,
                "channel_1based": ch + 1,
                "candidate_accuracy": float(acc),
                "threshold_m": float(baseline_m),
                "removed": bool(accepted),
                "remaining_if_removed": len(candidate),
            })

            if accepted:
                active = candidate
                removed_this_pass += 1
                print(
                    f"  remove ch {ch + 1:02d}: val={acc:.4f} >= m={baseline_m:.4f}; "
                    f"{len(active)} remain"
                )
            else:
                print(
                    f"  keep   ch {ch + 1:02d}: val={acc:.4f} <  m={baseline_m:.4f}"
                )

        if removed_this_pass == 0:
            break

    return active, {
        "baseline_m": float(baseline_m),
        "history": history,
        "train_indices": tr_idx.tolist(),
        "validation_indices": va_idx.tolist(),
    }
