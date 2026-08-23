"""
Feature extractors:
    mst         modified S-transform PSD (1--35 Hz)
    bandpower   Welch PSD integrated over broad frequency bands
    psd         alias for bandpower
    fft         one-sided FFT/Fourier power spectrum (1--35 Hz)
    filterbank  Butterworth band-pass filter bank + band power

Classifiers:
    svm         RBF-SVM
    ann         input -> FC(64) -> ReLU -> FC(2)
    cnn         tiny spectral Conv1d network
    tree        decision tree with maximum depth 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix

from channel_selection import (
    optimized_wrapper_selection,
    rank_channels_repeated_cv,
    select_channel_features,
)
from config import ReproductionConfig, PAPER_REPORTED_CHANNELS_0BASED
from data import (
    downsample_ecog,
    load_competition_test,
    load_competition_train,
    load_true_test_labels,
    validate_dataset_shape,
)
from feature_extraction import (
    bandpower_psd_features,
    fft_spectrum_features,
    filterbank_features,
    mst_psd_features,
)
from models import (
    MstAnnClassifier,
    TinyCnnClassifier,
    build_depth4_tree,
    build_rbf_svm,
    get_c_gamma,
    tune_rbf_svm,
)


def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def canonical_feature_method(name: str) -> str:
    return "bandpower" if name == "psd" else name


def extract_feature_tensor(x: np.ndarray, method: str, cfg: ReproductionConfig):
    """Return [trials, channels, features_per_channel] and feature-axis labels."""
    method = canonical_feature_method(method)

    if method == "mst":
        return mst_psd_features(
            x,
            fs=cfg.target_fs,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
            p=cfg.mst_p,
            q=cfg.mst_q,
            truncate=cfg.gaussian_truncate,
        )

    if method == "bandpower":
        return bandpower_psd_features(
            x,
            fs=cfg.target_fs,
            bands=cfg.spectral_bands,
            nperseg=cfg.welch_nperseg,
            log_power=cfg.log_power_features,
        )

    if method == "fft":
        return fft_spectrum_features(
            x,
            fs=cfg.target_fs,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
            log_power=cfg.log_power_features,
        )

    if method == "filterbank":
        return filterbank_features(
            x,
            fs=cfg.target_fs,
            bands=cfg.spectral_bands,
            order=cfg.filterbank_order,
            log_power=cfg.log_power_features,
        )

    raise ValueError(f"Unknown feature method: {method}")


def load_or_extract_features(x_train, x_test, cfg: ReproductionConfig, method: str, cache_path: Path):
    method = canonical_feature_method(method)

    if cache_path.exists():
        print(f"Loading cached {method} features: {cache_path}")
        cache = np.load(cache_path)
        return (
            cache["train_features"],
            cache["test_features"],
            cache["feature_axis"],
        )

    expected_down_samples = int(cfg.trial_seconds * cfg.target_fs)

    print("Downsampling train 1000 Hz -> 100 Hz...")
    tr_ds = downsample_ecog(x_train, cfg.original_fs, cfg.target_fs)
    print("Downsampling test 1000 Hz -> 100 Hz...")
    te_ds = downsample_ecog(x_test, cfg.original_fs, cfg.target_fs)

    validate_dataset_shape(tr_ds, cfg.n_channels, expected_down_samples)
    validate_dataset_shape(te_ds, cfg.n_channels, expected_down_samples)
    print(f"Downsampled train shape: {tr_ds.shape}")
    print(f"Downsampled test shape:  {te_ds.shape}")

    print(f"\nExtracting train {method} features...")
    train_features, feature_axis = extract_feature_tensor(tr_ds, method, cfg)

    print(f"\nExtracting test {method} features...")
    test_features, test_axis = extract_feature_tensor(te_ds, method, cfg)

    if not np.array_equal(feature_axis, test_axis):
        raise RuntimeError("Train and test feature axes are inconsistent.")

    np.savez_compressed(cache_path, train_features=train_features, test_features=test_features, feature_axis=feature_axis)
    print(f"Cached features: {cache_path}")
    return train_features, test_features, feature_axis


def feature_axis_for_json(values: np.ndarray):
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.number):
        return [float(v) for v in values]
    return [str(v) for v in values]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("run"))

    parser.add_argument(
        "--features",
        choices=("mst", "bandpower", "psd", "fft", "filterbank"),
        default="mst",
        help=(
            "Feature extractor. 'psd' is an alias for 'bandpower'. "
            "Default: mst."
        ),
    )
    parser.add_argument(
        "--linear-power",
        action="store_true",
        help=(
            "Use linear power for bandpower/FFT/filterbank. By default these "
            "three alternatives use 10*log10(power). MST remains paper-style raw power."
        ),
    )

    parser.add_argument(
        "--classifier",
        choices=("svm", "ann", "cnn", "tree"),
        default="svm",
        help="Final classifier after feature extraction and channel selection.",
    )

    channel_mode = parser.add_mutually_exclusive_group()
    channel_mode.add_argument(
        "--use-paper-channels",
        action="store_true",
        help="Use the 13 channels reported by Xu et al. instead of selecting channels.",
    )
    channel_mode.add_argument(
        "--use-all-channels",
        action="store_true",
        help="Skip channel selection and use all 64 channels.",
    )

    parser.add_argument(
        "--minimum-channels",
        type=int,
        default=32,
        help="Minimum number of channels retained by wrapper selection. Default: 32.",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Use C=1, gamma='scale' instead of training-only SVM grid search.",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Z-score features inside the SVM pipeline.",
    )

    parser.add_argument("--ann-epochs", type=int, default=300)
    parser.add_argument("--ann-learning-rate", type=float, default=1e-3)
    parser.add_argument("--ann-weight-decay", type=float, default=1e-4)
    parser.add_argument("--ann-batch-size", type=int, default=32)
    parser.add_argument("--ann-patience", type=int, default=30)
    parser.add_argument(
        "--ann-device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    cfg = ReproductionConfig()
    cfg.tune_svm = not args.no_tune
    cfg.standardize_features = args.standardize
    cfg.log_power_features = not args.linear_power
    cfg.minimum_channels = args.minimum_channels

    if not 1 <= cfg.minimum_channels <= cfg.n_channels:
        raise ValueError("--minimum-channels must be between 1 and 64.")

    method = canonical_feature_method(args.features)

    x_train, y_train = load_competition_train(args.train)
    x_test = load_competition_test(args.test)

    validate_dataset_shape(
        x_train,
        cfg.n_channels,
        int(cfg.trial_seconds * cfg.original_fs),
    )
    validate_dataset_shape(
        x_test,
        cfg.n_channels,
        int(cfg.trial_seconds * cfg.original_fs),
    )

    print(f"Raw train: {x_train.shape}; labels: {y_train.shape}")
    print(f"Raw test:  {x_test.shape}")
    print(f"Feature extractor: {method}")

    power_tag = "linear" if args.linear_power else "log"
    if method == "mst":
        power_tag = "paper"
    feature_cache = args.output / f"features_{method}_{power_tag}.npz"

    train_feat, test_feat, feature_axis = load_or_extract_features(x_train, x_test, cfg, method, feature_cache)

    print(f"\nFeature tensor train: {train_feat.shape}")
    print(f"Feature tensor test:  {test_feat.shape}")
    print(f"Features per channel: {train_feat.shape[2]}")
    print(
        f"All-channel flattened dimension = "
        f"{train_feat.shape[1]} x {train_feat.shape[2]} = "
        f"{train_feat.shape[1] * train_feat.shape[2]}"
    )

    # Channel ranking/wrapper remains SVM-based, matching the original pipeline.
    base_estimator = build_rbf_svm(c=1.0, gamma="scale", standardize=cfg.standardize_features)

    if args.use_all_channels:
        selected = list(range(cfg.n_channels))
        selection_meta = {
            "mode": "all_channels",
            "selected_channels_1based": [channel + 1 for channel in selected],
        }
        print("\nUsing all 64 channels.")

    elif args.use_paper_channels:
        selected = list(PAPER_REPORTED_CHANNELS_0BASED)
        selection_meta = {
            "mode": "paper_reported_channels",
            "selected_channels_1based": [c + 1 for c in selected],
        }
        print("\nUsing paper-reported 13 channels:")
        print([c + 1 for c in selected])
        if method != "mst":
            print(
                "NOTE: these 13 channels were selected in the Xu et al. MST pipeline; "
                f"they are not guaranteed to be optimal for {method}."
            )

    else:
        print("\nRanking each channel by 10 x 10-fold CV...")
        order, channel_scores = rank_channels_repeated_cv(
            train_feat,
            y_train,
            base_estimator,
            n_splits=cfg.ranking_folds,
            n_repeats=cfg.ranking_repeats,
            random_state=cfg.random_state,
        )
        np.save(args.output / f"{method}_channel_cv_scores.npy", channel_scores)
        np.save(args.output / f"{method}_channel_rank_low_to_high.npy", order)

        print("\nRunning optimized wrapper backward elimination...")
        selected, wrapper_meta = optimized_wrapper_selection(
            train_feat,
            y_train,
            base_estimator,
            order,
            validation_fraction=cfg.wrapper_validation_fraction,
            random_state=cfg.random_state,
            minimum_channels=cfg.minimum_channels,
        )
        selection_meta = {
            "mode": "optimized_wrapper",
            "minimum_channels": cfg.minimum_channels,
            "selected_channels_1based": [c + 1 for c in selected],
            "wrapper": wrapper_meta,
        }
        print(f"\nSelected {len(selected)} channels:")
        print([c + 1 for c in selected])

    xtr = select_channel_features(train_feat, selected)
    xte = select_channel_features(test_feat, selected)

    print(f"\nFinal training feature shape: {xtr.shape}")
    print(f"Final test feature shape: {xte.shape}")

    c = None
    gamma = None
    ann_metadata = None
    cnn_metadata = None
    tree_metadata = None

    if args.classifier == "ann":
        print("\nTraining ANN: input -> FC(64) -> ReLU -> FC(2)...")
        final_model = MstAnnClassifier(
            learning_rate=args.ann_learning_rate,
            weight_decay=args.ann_weight_decay,
            batch_size=args.ann_batch_size,
            epochs=args.ann_epochs,
            patience=args.ann_patience,
            random_state=cfg.random_state,
            device=args.ann_device,
        ).fit(xtr, y_train)
        training = final_model.training_result

        # PyTorch Linear includes both weights and biases.
        input_dim = int(xtr.shape[1])
        ann_params = input_dim * 64 + 64 + 64 * 2 + 2
        ann_metadata = {
            "fc1_weight_shape": list(final_model.fc1_weight_shape),
            "fc2_weight_shape": list(final_model.fc2_weight_shape),
            "parameter_count": ann_params,
            "int8_parameter_bytes_theoretical": ann_params,
            "best_epoch": training.best_epoch,
            "epochs_run": training.epochs_run,
            "best_validation_loss": training.best_validation_loss,
            "learning_rate": args.ann_learning_rate,
            "weight_decay": args.ann_weight_decay,
            "batch_size": args.ann_batch_size,
            "patience": args.ann_patience,
            "device": str(final_model.device),
        }
        print(f"ANN parameters: {ann_params:,}")
        final_model.save(args.output / f"ann_{method}.pt")

    elif args.classifier == "cnn":
        print("\nTraining tiny CNN: Conv1d(8) -> ReLU -> pool -> FC(2)...")
        final_model = TinyCnnClassifier(
            input_channels=len(selected),
            features_per_channel=train_feat.shape[2],
            learning_rate=args.ann_learning_rate,
            weight_decay=args.ann_weight_decay,
            batch_size=args.ann_batch_size,
            epochs=args.ann_epochs,
            patience=args.ann_patience,
            random_state=cfg.random_state,
            device=args.ann_device,
        ).fit(xtr, y_train)
        training = final_model.training_result
        cnn_metadata = {
            "architecture": "Conv1d(channels,8,kernel=3,padding=1)-ReLU-GlobalAvgPool-FC(8,2)",
            "input_shape_channels_features": [
                len(selected),
                int(train_feat.shape[2]),
            ],
            "parameter_count": final_model.parameter_count,
            "best_epoch": training.best_epoch,
            "epochs_run": training.epochs_run,
            "best_validation_loss": training.best_validation_loss,
            "device": str(final_model.device),
        }
        final_model.save(args.output / f"tiny_cnn_{method}.pt")

    elif args.classifier == "tree":
        print("\nTraining decision tree with max_depth=4...")
        final_model = build_depth4_tree(random_state=cfg.random_state)
        tree_metadata = {"max_depth": 4}

    elif cfg.tune_svm:
        print("\nTuning RBF-SVM on final selected-channel features...")
        variance = float(np.var(xtr))
        if variance <= 0 or not np.isfinite(variance):
            raise ValueError("Invalid feature variance for RBF gamma search.")
        base_gamma = 1.0 / (xtr.shape[1] * variance)
        gamma_grid = base_gamma * (2.0 ** np.arange(-10, 11))

        grid = tune_rbf_svm(
            xtr,
            y_train,
            c_grid=cfg.c_grid,
            gamma_grid=gamma_grid,
            cv_folds=cfg.svm_cv_folds,
            random_state=cfg.random_state,
            standardize=cfg.standardize_features,
        )
        final_model = grid.best_estimator_
        c, gamma = get_c_gamma(final_model)
        print(f"Selected-channel CV accuracy: {grid.best_score_:.4f}")
        print(f"Best final C = {c}")
        print(f"Best final gamma = {gamma}")

    else:
        final_model = build_rbf_svm(
            c=1.0,
            gamma="scale",
            standardize=cfg.standardize_features,
        )
        c, gamma = get_c_gamma(final_model)

    if args.classifier in ("svm", "tree"):
        final_model.fit(xtr, y_train)

    predictions = final_model.predict(xte).astype(int)
    np.savetxt(args.output / f"test_predictions_{method}_{args.classifier}.txt", predictions, fmt="%d")

    if args.classifier == "svm":
        joblib.dump(final_model, args.output / f"rbf_svm_{method}.joblib")
    elif args.classifier == "tree":
        joblib.dump(final_model, args.output / f"depth4_tree_{method}.joblib")

    feature_axis_kind = "frequency_hz" if method in ("mst", "fft") else "band"
    result = {
        "train_trials": int(len(y_train)),
        "test_trials": int(len(predictions)),
        "feature_extractor": method,
        "feature_axis_kind": feature_axis_kind,
        "feature_axis": feature_axis_for_json(feature_axis),
        "features_per_channel": int(train_feat.shape[2]),
        "log_power": None if method == "mst" else cfg.log_power_features,
        "mst_p": cfg.mst_p if method == "mst" else None,
        "mst_q": cfg.mst_q if method == "mst" else None,
        "selected_channels_1based": [c + 1 for c in selected],
        "n_selected_channels": len(selected),
        "final_feature_dimension": int(xtr.shape[1]),
        "classifier": args.classifier,
        "svm_C": c,
        "svm_gamma": None if gamma is None else str(gamma),
        "ann": ann_metadata,
        "cnn": cnn_metadata,
        "tree": tree_metadata,
        "standardized": cfg.standardize_features,
    }

    if args.labels is not None:
        y_test = load_true_test_labels(args.labels)
        if len(y_test) != len(predictions):
            raise ValueError(
                f"Test label count {len(y_test)} != predictions {len(predictions)}"
            )
        acc = accuracy_score(y_test, predictions)
        wrong = (np.flatnonzero(y_test != predictions) + 1).tolist()
        cm = confusion_matrix(y_test, predictions, labels=[-1, 1]).tolist()

        result["official_test_accuracy"] = float(acc)
        result["correct_trials"] = int(np.sum(y_test == predictions))
        result["incorrect_trial_numbers_1based"] = wrong
        result["confusion_matrix_labels_minus1_plus1"] = cm

        print("\n=== OFFICIAL TEST RESULT ===")
        print(f"Accuracy: {acc * 100:.2f}% ({np.sum(y_test == predictions)}/{len(y_test)})")
        print(f"Wrong trials (1-based): {wrong}")
        if method == "mst" and args.classifier == "svm":
            print("Xu et al. reported 98% for their MST + selected-channel RBF-SVM implementation.")
    else:
        print("\nNo test labels supplied; predictions were saved but accuracy was not computed.")

    save_json(args.output / f"selection_{method}.json", selection_meta)
    save_json(args.output / f"result_{method}_{args.classifier}.json", result)
    save_json(args.output / "config.json", cfg.__dict__)

    print(f"\nOutputs saved in: {args.output.resolve()}")


if __name__ == "__main__":
    main()
