#!/usr/bin/env python
"""Profile all 37 Indy sessions and quantify within- versus between-month drift.

The analysis is descriptive and never modifies raw or processed data. It checks
artifact integrity, profiles neural counts and velocity targets, flags sessions
that need review, and uses permutation tests to ask whether month labels explain
more variation than ordinary session-to-session variability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon, pdist, squareform
from scipy.stats import kruskal, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed" / "indy_loco" / "indy"
RAW_DIR = ROOT / "data" / "raw" / "indy_loco" / "indy"
MANIFEST_PATH = PROCESSED_DIR / "manifest.json"

SESSION_CSV = ROOT / "results" / "metrics" / "indy_session_quality.csv"
MONTH_CSV = ROOT / "results" / "metrics" / "indy_month_summary.csv"
PAIRWISE_CSV = ROOT / "results" / "metrics" / "indy_month_pairwise.csv"
SUMMARY_JSON = ROOT / "results" / "metrics" / "indy_month_drift_analysis.json"
OVERVIEW_FIGURE = ROOT / "results" / "figures" / "indy_month_drift_overview.png"
QUALITY_FIGURE = ROOT / "results" / "figures" / "indy_session_quality_overview.png"

BIN_S = 0.040
PREFIX_BINS = int(60 / BIN_S)
N_CHANNELS = 96
SELECTED_CHANNELS = 32
MONTH_ORDER = ["2016-04", "2016-06", "2016-09", "2016-10", "2016-12", "2017-01"]
MONTH_COLORS = {
    "2016-04": "#2368A2",
    "2016-06": "#7AA6C2",
    "2016-09": "#C58B18",
    "2016-10": "#E3BA5C",
    "2016-12": "#68735B",
    "2017-01": "#9AA68C",
}


def hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def month_from_session(session: str) -> str:
    return f"{session[5:9]}-{session[9:11]}"


def normalized_histogram(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts = np.histogram(values, bins=edges)[0].astype(np.float64)
    counts += 1e-12
    return counts / counts.sum()


def js_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(jensenshannon(first, second, base=2.0))


def normalized_entropy(probabilities: np.ndarray) -> float:
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(len(probabilities)))


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad < 1e-12:
        return np.zeros_like(values)
    return 0.67448975 * (values - median) / mad


def standardize_features(values: np.ndarray) -> np.ndarray:
    scaled = StandardScaler().fit_transform(values)
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)


def pseudo_f_statistic(features: np.ndarray, labels: np.ndarray) -> float:
    grand_mean = features.mean(axis=0)
    between = 0.0
    within = 0.0
    unique_labels = np.unique(labels)
    for label in unique_labels:
        group = features[labels == label]
        group_mean = group.mean(axis=0)
        between += len(group) * float(np.sum((group_mean - grand_mean) ** 2))
        within += float(np.sum((group - group_mean) ** 2))
    numerator = between / max(len(unique_labels) - 1, 1)
    denominator = within / max(len(features) - len(unique_labels), 1)
    return numerator / max(denominator, 1e-12)


def nearest_centroid_accuracy(features: np.ndarray, labels: np.ndarray) -> float:
    predictions = []
    unique_labels = np.unique(labels)
    for index in range(len(features)):
        distances = []
        for label in unique_labels:
            members = np.where(labels == label)[0]
            members = members[members != index]
            centroid = features[members].mean(axis=0)
            distances.append(float(np.linalg.norm(features[index] - centroid)))
        predictions.append(unique_labels[int(np.argmin(distances))])
    return float(np.mean(np.asarray(predictions) == labels))


def separation_diagnostics(
    features: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    permutations: int,
) -> dict:
    distance_matrix = squareform(pdist(features, metric="euclidean"))
    same_month = labels[:, None] == labels[None, :]
    upper = np.triu(np.ones_like(same_month, dtype=bool), k=1)
    within = distance_matrix[upper & same_month]
    between = distance_matrix[upper & ~same_month]

    observed_f = pseudo_f_statistic(features, labels)
    observed_accuracy = nearest_centroid_accuracy(features, labels)
    permuted_f = np.empty(permutations, dtype=np.float64)
    permuted_accuracy = np.empty(permutations, dtype=np.float64)
    for permutation in range(permutations):
        shuffled = rng.permutation(labels)
        permuted_f[permutation] = pseudo_f_statistic(features, shuffled)
        permuted_accuracy[permutation] = nearest_centroid_accuracy(features, shuffled)

    return {
        "within_distance_median": float(np.median(within)),
        "between_distance_median": float(np.median(between)),
        "between_within_ratio": float(np.median(between) / np.median(within)),
        "distance_mean_difference": float(between.mean() - within.mean()),
        "pseudo_f": float(observed_f),
        "pseudo_f_permutation_p": float(
            (1 + np.sum(permuted_f >= observed_f)) / (permutations + 1)
        ),
        "silhouette": float(silhouette_score(features, labels, metric="euclidean")),
        "leave_one_out_nearest_centroid_accuracy": float(observed_accuracy),
        "accuracy_null_mean": float(permuted_accuracy.mean()),
        "accuracy_permutation_p": float(
            (1 + np.sum(permuted_accuracy >= observed_accuracy)) / (permutations + 1)
        ),
        "within_pair_count": int(len(within)),
        "between_pair_count": int(len(between)),
    }


def plot_box_with_points(axis, frame: pd.DataFrame, column: str, ylabel: str, title: str) -> None:
    groups = [frame.loc[frame["month"] == month, column].to_numpy() for month in MONTH_ORDER]
    axis.boxplot(
        groups,
        positions=np.arange(len(MONTH_ORDER)),
        widths=0.55,
        patch_artist=True,
        boxprops={"facecolor": "#E9EEF3", "edgecolor": "#5B6470"},
        medianprops={"color": "#20262D", "linewidth": 1.5},
        whiskerprops={"color": "#5B6470"},
        capprops={"color": "#5B6470"},
        flierprops={"marker": ""},
    )
    for month_index, month in enumerate(MONTH_ORDER):
        values = groups[month_index]
        offsets = np.linspace(-0.14, 0.14, len(values)) if len(values) > 1 else np.array([0.0])
        axis.scatter(
            month_index + offsets,
            values,
            s=28,
            color=MONTH_COLORS[month],
            edgecolor="#20262D",
            linewidth=0.35,
            zorder=3,
        )
    axis.set_xticks(np.arange(len(MONTH_ORDER)), [month[5:] for month in MONTH_ORDER])
    axis.set_xlabel("Month (2016, except Jan 2017)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", color="#D8DDE3", linewidth=0.7, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)


def plot_pca(axis, coordinates: np.ndarray, frame: pd.DataFrame, title: str, variance: np.ndarray) -> None:
    for month in MONTH_ORDER:
        mask = frame["month"].to_numpy() == month
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=48,
            color=MONTH_COLORS[month],
            edgecolor="#20262D",
            linewidth=0.45,
            label=month,
        )
    axis.axhline(0, color="#D8DDE3", linewidth=0.7)
    axis.axvline(0, color="#D8DDE3", linewidth=0.7)
    axis.set_xlabel(f"PC1 ({variance[0] * 100:.1f}%)")
    axis.set_ylabel(f"PC2 ({variance[1] * 100:.1f}%)")
    axis.set_title(title)
    axis.spines[["top", "right"]].set_visible(False)


def plot_distance_heatmap(axis, matrix: np.ndarray, title: str, colorbar_label: str) -> None:
    image = axis.imshow(matrix, cmap="Blues", vmin=0)
    axis.set_xticks(np.arange(len(MONTH_ORDER)), [month[5:] for month in MONTH_ORDER], rotation=45)
    axis.set_yticks(np.arange(len(MONTH_ORDER)), [month[5:] for month in MONTH_ORDER])
    axis.set_title(title)
    for row in range(len(MONTH_ORDER)):
        for column in range(len(MONTH_ORDER)):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#FFFFFF" if matrix[row, column] > matrix.max() * 0.58 else "#20262D",
            )
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)


def save_figures(
    session_frame: pd.DataFrame,
    neural_pca: np.ndarray,
    neural_variance: np.ndarray,
    behavior_pca: np.ndarray,
    behavior_variance: np.ndarray,
    neural_month_matrix: np.ndarray,
    speed_month_matrix: np.ndarray,
) -> None:
    matplotlib_cache = ROOT / "results" / "large" / ".matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OVERVIEW_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(14.0, 15.0), dpi=180)
    plot_box_with_points(
        axes[0, 0], session_frame, "mean_firing_rate_hz", "Mean firing rate (Hz)",
        "Session-average neural activity",
    )
    plot_box_with_points(
        axes[0, 1], session_frame, "speed_rms_cm_s", "RMS speed (cm/s)",
        "Session target speed",
    )
    plot_pca(
        axes[1, 0], neural_pca, session_frame, "Neural channel-rate profiles", neural_variance
    )
    plot_pca(
        axes[1, 1], behavior_pca, session_frame, "Velocity target profiles", behavior_variance
    )
    plot_distance_heatmap(
        axes[2, 0], neural_month_matrix, "Monthly channel-composition distance", "Jensen-Shannon distance"
    )
    plot_distance_heatmap(
        axes[2, 1], speed_month_matrix, "Monthly speed-distribution distance", "Jensen-Shannon distance"
    )
    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("Indy dataset: session and month drift overview", y=0.998, fontsize=16)
    fig.text(
        0.5,
        0.005,
        "37 sessions · 96 M1 count channels · causal two-axis velocity · 40 ms bins",
        ha="center",
        color="#5B6470",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(OVERVIEW_FIGURE, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), dpi=180)
    plot_box_with_points(
        axes[0, 0], session_frame, "duration_min", "Duration (minutes)", "Session duration"
    )
    plot_box_with_points(
        axes[0, 1], session_frame, "prefix_silent_selected_channels", "Channels",
        "Selected channels silent during first 60 seconds",
    )
    plot_box_with_points(
        axes[1, 0], session_frame, "within_session_rate_spearman", "Spearman correlation",
        "First-half versus second-half channel rates",
    )
    plot_box_with_points(
        axes[1, 1], session_frame, "stationary_fraction", "Fraction of bins",
        "Near-stationary target bins (<1 cm/s)",
    )
    fig.suptitle("Indy dataset: session quality diagnostics", y=0.995, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(QUALITY_FIGURE, bbox_inches="tight")
    plt.close(fig)


def run_analysis(*, verify_raw_checksums: bool = True, permutations: int = 5000) -> dict:
    rng = np.random.default_rng(42)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    split_by_session = {
        session: split
        for split, sessions in manifest["splits"].items()
        for session in sessions
    }
    expected_sessions = list(manifest["raw_md5"])

    processed_files = sorted(PROCESSED_DIR.glob("*/*.npz"))
    processed_names = {path.stem for path in processed_files}
    raw_files = sorted(RAW_DIR.glob("indy_*.mat"))
    raw_names = {path.stem for path in raw_files}
    expected_names = set(expected_sessions)

    integrity = {
        "expected_session_count": len(expected_sessions),
        "processed_session_count": len(processed_files),
        "raw_session_count": len(raw_files),
        "missing_processed": sorted(expected_names - processed_names),
        "extra_processed": sorted(processed_names - expected_names),
        "missing_raw": sorted(expected_names - raw_names),
        "extra_raw": sorted(raw_names - expected_names),
        "artifact_sha256_failures": [],
        "embedded_source_md5_failures": [],
        "raw_md5_failures": [],
        "schema_failures": [],
        "shape_failures": [],
        "value_failures": [],
        "raw_checksums_verified": verify_raw_checksums,
    }

    arrays: dict[str, dict[str, np.ndarray]] = {}
    for path in processed_files:
        session = path.stem
        expected_sha256 = manifest["artifact_sha256"].get(session)
        if expected_sha256 and hash_file(path, "sha256") != expected_sha256:
            integrity["artifact_sha256_failures"].append(session)

        with np.load(path, allow_pickle=False) as artifact:
            keys = set(artifact.files)
            required = {
                "schema_version", "session", "source_md5", "bin_s", "counts", "velocity",
                "velocity_lowpass_hz", "velocity_filter", "velocity_difference", "kinematic_sampling",
            }
            if not required.issubset(keys):
                integrity["schema_failures"].append({"session": session, "missing_keys": sorted(required - keys)})
                continue
            schema = str(np.asarray(artifact["schema_version"]).item())
            embedded_session = str(np.asarray(artifact["session"]).item())
            source_md5 = str(np.asarray(artifact["source_md5"]).item())
            counts = artifact["counts"].copy()
            velocity = artifact["velocity"].copy()
            bin_s = float(np.asarray(artifact["bin_s"]).item())

        if (
            schema != "indy_counts_velocity_v2"
            or embedded_session != session
            or not np.isclose(bin_s, BIN_S, atol=1e-8, rtol=0)
        ):
            integrity["schema_failures"].append(
                {"session": session, "schema": schema, "embedded_session": embedded_session, "bin_s": bin_s}
            )
        if source_md5 != manifest["raw_md5"].get(session):
            integrity["embedded_source_md5_failures"].append(session)
        if counts.ndim != 2 or counts.shape[0] != N_CHANNELS or velocity.shape != (counts.shape[1], 2):
            integrity["shape_failures"].append(
                {"session": session, "counts_shape": list(counts.shape), "velocity_shape": list(velocity.shape)}
            )
        if counts.dtype != np.uint8 or np.any(counts < 0) or not np.isfinite(velocity).all():
            integrity["value_failures"].append(session)
        arrays[session] = {"counts": counts, "velocity": velocity}

    if verify_raw_checksums:
        for session in expected_sessions:
            path = RAW_DIR / f"{session}.mat"
            if path.exists() and hash_file(path, "md5") != manifest["raw_md5"][session]:
                integrity["raw_md5_failures"].append(session)

    training_sessions = manifest["splits"]["train"]
    prefix_rates = np.stack(
        [arrays[session]["counts"][:, :PREFIX_BINS].mean(axis=1) for session in training_sessions]
    )
    selected_channels = np.sort(np.argsort(prefix_rates.mean(axis=0))[-SELECTED_CHANNELS:])

    all_speeds = np.concatenate(
        [np.linalg.norm(payload["velocity"], axis=1) for payload in arrays.values()]
    )
    speed_cap = float(np.quantile(all_speeds, 0.999))
    speed_edges = np.linspace(0.0, speed_cap, 41)
    direction_edges = np.linspace(-np.pi, np.pi, 13)

    rows = []
    channel_rates = []
    channel_shares = []
    speed_histograms = []
    direction_histograms = []
    for session in expected_sessions:
        counts = arrays[session]["counts"].astype(np.float64)
        velocity = arrays[session]["velocity"].astype(np.float64)
        speed = np.linalg.norm(velocity, axis=1)
        rates = counts.mean(axis=1) / BIN_S
        shares = rates + 1e-12
        shares /= shares.sum()

        midpoint = counts.shape[1] // 2
        first_half_rates = counts[:, :midpoint].mean(axis=1) / BIN_S
        second_half_rates = counts[:, midpoint:].mean(axis=1) / BIN_S
        half_correlation = float(spearmanr(first_half_rates, second_half_rates).statistic)
        prefix_mean = counts[selected_channels, :PREFIX_BINS].mean(axis=1, keepdims=True)
        prefix_std = counts[selected_channels, :PREFIX_BINS].std(axis=1, keepdims=True)
        naive_z_max = float(np.max(np.abs((counts[selected_channels] - prefix_mean) / (prefix_std + 1e-6))))

        population_rate = counts.mean(axis=0) / BIN_S
        population_speed_correlation = float(np.corrcoef(population_rate, speed)[0, 1])
        moving = speed >= 1.0
        if moving.any():
            angles = np.arctan2(velocity[moving, 1], velocity[moving, 0])
            direction_hist = normalized_histogram(angles, direction_edges)
        else:
            direction_hist = np.full(len(direction_edges) - 1, 1 / (len(direction_edges) - 1))
        speed_hist = normalized_histogram(np.minimum(speed, speed_cap), speed_edges)

        row = {
            "session": session,
            "month": month_from_session(session),
            "split": split_by_session[session],
            "bins": counts.shape[1],
            "duration_min": counts.shape[1] * BIN_S / 60.0,
            "mean_firing_rate_hz": float(rates.mean()),
            "median_channel_rate_hz": float(np.median(rates)),
            "channel_rate_iqr_hz": float(np.quantile(rates, 0.75) - np.quantile(rates, 0.25)),
            "count_zero_fraction": float(np.mean(counts == 0)),
            "count_p99": float(np.quantile(counts, 0.99)),
            "count_max": int(counts.max()),
            "active_channels_ge_1hz": int(np.sum(rates >= 1.0)),
            "near_silent_channels_lt_0_1hz": int(np.sum(rates < 0.1)),
            "prefix_silent_all_channels": int(np.sum(counts[:, :PREFIX_BINS].std(axis=1) == 0)),
            "prefix_silent_selected_channels": int(np.sum(prefix_std[:, 0] == 0)),
            "naive_prefix_normalized_abs_max": naive_z_max,
            "within_session_rate_spearman": half_correlation,
            "population_rate_speed_correlation": population_speed_correlation,
            "speed_mean_cm_s": float(speed.mean()),
            "speed_median_cm_s": float(np.median(speed)),
            "speed_rms_cm_s": float(np.sqrt(np.mean(speed ** 2))),
            "speed_p95_cm_s": float(np.quantile(speed, 0.95)),
            "speed_p99_cm_s": float(np.quantile(speed, 0.99)),
            "stationary_fraction": float(np.mean(speed < 1.0)),
            "vx_mean_cm_s": float(velocity[:, 0].mean()),
            "vy_mean_cm_s": float(velocity[:, 1].mean()),
            "vx_std_cm_s": float(velocity[:, 0].std()),
            "vy_std_cm_s": float(velocity[:, 1].std()),
            "direction_entropy": normalized_entropy(direction_hist),
        }
        rows.append(row)
        channel_rates.append(rates)
        channel_shares.append(shares)
        speed_histograms.append(speed_hist)
        direction_histograms.append(direction_hist)

    session_frame = pd.DataFrame(rows)
    channel_rates_array = np.stack(channel_rates)
    channel_shares_array = np.stack(channel_shares)
    speed_histogram_array = np.stack(speed_histograms)
    direction_histogram_array = np.stack(direction_histograms)
    labels = session_frame["month"].to_numpy()

    neural_rate_features = standardize_features(np.log1p(channel_rates_array))
    centered_log_shares = np.log(channel_shares_array) - np.log(channel_shares_array).mean(axis=1, keepdims=True)
    neural_composition_features = standardize_features(centered_log_shares)
    neural_selected_features = standardize_features(np.log1p(channel_rates_array[:, selected_channels]))
    behavior_scalar_columns = [
        "speed_mean_cm_s", "speed_median_cm_s", "speed_rms_cm_s", "speed_p95_cm_s",
        "speed_p99_cm_s", "stationary_fraction", "vx_std_cm_s", "vy_std_cm_s",
        "direction_entropy",
    ]
    behavior_raw = np.column_stack(
        [
            session_frame[behavior_scalar_columns].to_numpy(),
            np.sqrt(speed_histogram_array),
            np.sqrt(direction_histogram_array),
        ]
    )
    behavior_features = standardize_features(behavior_raw)

    separation = {
        "neural_all_96_log_rates": separation_diagnostics(
            neural_rate_features, labels, rng, permutations
        ),
        "neural_channel_composition": separation_diagnostics(
            neural_composition_features, labels, rng, permutations
        ),
        "neural_selected_32_log_rates": separation_diagnostics(
            neural_selected_features, labels, rng, permutations
        ),
        "velocity_target_profile": separation_diagnostics(
            behavior_features, labels, rng, permutations
        ),
    }

    scalar_tests = {}
    scalar_test_columns = [
        "duration_min", "mean_firing_rate_hz", "count_zero_fraction",
        "active_channels_ge_1hz", "prefix_silent_selected_channels",
        "within_session_rate_spearman", "speed_rms_cm_s", "stationary_fraction",
        "population_rate_speed_correlation",
    ]
    for column in scalar_test_columns:
        groups = [session_frame.loc[session_frame["month"] == month, column].to_numpy() for month in MONTH_ORDER]
        statistic, p_value = kruskal(*groups)
        effect = max(0.0, float((statistic - len(MONTH_ORDER) + 1) / (len(session_frame) - len(MONTH_ORDER))))
        scalar_tests[column] = {
            "kruskal_h": float(statistic),
            "p_value": float(p_value),
            "epsilon_squared": effect,
        }

    session_dates = pd.to_datetime(
        session_frame["session"].str.extract(r"indy_(\d{8})_")[0],
        format="%Y%m%d",
    )
    elapsed_days = (session_dates - session_dates.min()).dt.total_seconds().to_numpy() / 86400.0
    temporal_correlations = {}
    for column in (
        "mean_firing_rate_hz",
        "count_zero_fraction",
        "active_channels_ge_1hz",
        "speed_rms_cm_s",
        "stationary_fraction",
    ):
        correlation = spearmanr(elapsed_days, session_frame[column].to_numpy())
        temporal_correlations[column] = {
            "spearman_rho": float(correlation.statistic),
            "p_value": float(correlation.pvalue),
        }

    split_summary = []
    for split in ("train", "validation", "test"):
        group = session_frame.loc[session_frame["split"] == split]
        split_summary.append(
            {
                "split": split,
                "sessions": int(len(group)),
                "mean_firing_rate_median_hz": float(group["mean_firing_rate_hz"].median()),
                "count_zero_fraction_median": float(group["count_zero_fraction"].median()),
                "active_channels_ge_1hz_median": float(group["active_channels_ge_1hz"].median()),
                "speed_rms_median_cm_s": float(group["speed_rms_cm_s"].median()),
                "stationary_fraction_median": float(group["stationary_fraction"].median()),
                "prefix_silent_selected_channels_median": float(
                    group["prefix_silent_selected_channels"].median()
                ),
            }
        )

    monthly_rows = []
    month_profiles = {}
    for month in MONTH_ORDER:
        mask = labels == month
        group = session_frame.loc[mask]
        mean_rates = channel_rates_array[mask].mean(axis=0)
        mean_shares = channel_shares_array[mask].mean(axis=0)
        mean_speed_hist = speed_histogram_array[mask].mean(axis=0)
        mean_direction_hist = direction_histogram_array[mask].mean(axis=0)
        top32 = np.sort(np.argsort(mean_rates)[-SELECTED_CHANNELS:])
        month_profiles[month] = {
            "rates": mean_rates,
            "shares": mean_shares / mean_shares.sum(),
            "speed_hist": mean_speed_hist / mean_speed_hist.sum(),
            "direction_hist": mean_direction_hist / mean_direction_hist.sum(),
            "top32": top32,
        }
        monthly_rows.append(
            {
                "month": month,
                "sessions": int(mask.sum()),
                "total_duration_min": float(group["duration_min"].sum()),
                "duration_median_min": float(group["duration_min"].median()),
                "mean_firing_rate_median_hz": float(group["mean_firing_rate_hz"].median()),
                "mean_firing_rate_iqr_hz": float(group["mean_firing_rate_hz"].quantile(0.75) - group["mean_firing_rate_hz"].quantile(0.25)),
                "count_zero_fraction_median": float(group["count_zero_fraction"].median()),
                "active_channels_ge_1hz_median": float(group["active_channels_ge_1hz"].median()),
                "prefix_silent_selected_channels_median": float(group["prefix_silent_selected_channels"].median()),
                "within_session_rate_spearman_median": float(group["within_session_rate_spearman"].median()),
                "speed_rms_median_cm_s": float(group["speed_rms_cm_s"].median()),
                "speed_rms_iqr_cm_s": float(group["speed_rms_cm_s"].quantile(0.75) - group["speed_rms_cm_s"].quantile(0.25)),
                "stationary_fraction_median": float(group["stationary_fraction"].median()),
                "population_rate_speed_correlation_median": float(group["population_rate_speed_correlation"].median()),
                "top32_overlap_with_train_selected": float(
                    len(set(top32) & set(selected_channels)) / len(set(top32) | set(selected_channels))
                ),
            }
        )
    month_frame = pd.DataFrame(monthly_rows)

    pairwise_rows = []
    neural_month_matrix = np.zeros((len(MONTH_ORDER), len(MONTH_ORDER)))
    speed_month_matrix = np.zeros_like(neural_month_matrix)
    for first, second in combinations(MONTH_ORDER, 2):
        first_profile = month_profiles[first]
        second_profile = month_profiles[second]
        first_index = MONTH_ORDER.index(first)
        second_index = MONTH_ORDER.index(second)
        channel_js = js_distance(first_profile["shares"], second_profile["shares"])
        speed_js = js_distance(first_profile["speed_hist"], second_profile["speed_hist"])
        direction_js = js_distance(first_profile["direction_hist"], second_profile["direction_hist"])
        neural_month_matrix[first_index, second_index] = channel_js
        neural_month_matrix[second_index, first_index] = channel_js
        speed_month_matrix[first_index, second_index] = speed_js
        speed_month_matrix[second_index, first_index] = speed_js
        first_top = set(first_profile["top32"])
        second_top = set(second_profile["top32"])
        pairwise_rows.append(
            {
                "month_a": first,
                "month_b": second,
                "channel_rate_spearman": float(
                    spearmanr(first_profile["rates"], second_profile["rates"]).statistic
                ),
                "channel_composition_js": channel_js,
                "top32_jaccard": float(len(first_top & second_top) / len(first_top | second_top)),
                "speed_distribution_js": speed_js,
                "direction_distribution_js": direction_js,
                "mean_rate_ratio_larger_to_smaller": float(
                    max(first_profile["rates"].mean(), second_profile["rates"].mean())
                    / min(first_profile["rates"].mean(), second_profile["rates"].mean())
                ),
            }
        )
    pairwise_frame = pd.DataFrame(pairwise_rows)

    outlier_columns = [
        "mean_firing_rate_hz", "count_zero_fraction",
        "active_channels_ge_1hz", "within_session_rate_spearman", "speed_rms_cm_s",
        "stationary_fraction", "population_rate_speed_correlation",
    ]
    global_outlier_z = np.column_stack(
        [robust_z(session_frame[column].to_numpy()) for column in outlier_columns]
    )
    within_month_outlier_z = np.zeros_like(global_outlier_z)
    for month in MONTH_ORDER:
        month_indices = np.where(labels == month)[0]
        within_month_outlier_z[month_indices] = np.column_stack(
            [
                robust_z(session_frame.loc[month_indices, column].to_numpy())
                for column in outlier_columns
            ]
        )
    session_frame["max_abs_global_robust_z"] = np.max(np.abs(global_outlier_z), axis=1)
    session_frame["max_abs_within_month_robust_z"] = np.max(
        np.abs(within_month_outlier_z), axis=1
    )
    session_frame["outlier_metrics"] = [
        ";".join(column for column, value in zip(outlier_columns, values) if abs(value) >= 3.5)
        for values in within_month_outlier_z
    ]
    session_frame["review_priority"] = "low"
    session_frame.loc[
        (session_frame["naive_prefix_normalized_abs_max"] >= 100)
        | (session_frame["max_abs_within_month_robust_z"] >= 3.5)
        | (session_frame["within_session_rate_spearman"] < 0.9),
        "review_priority",
    ] = "medium"
    session_frame.loc[
        session_frame["naive_prefix_normalized_abs_max"] >= 10_000,
        "review_priority",
    ] = "high"

    non_high_mask = session_frame["review_priority"].to_numpy() != "high"
    sensitivity_excluding_high_priority = {
        "excluded_sessions": session_frame.loc[
            ~non_high_mask, "session"
        ].tolist(),
        "neural_selected_32_log_rates": separation_diagnostics(
            neural_selected_features[non_high_mask],
            labels[non_high_mask],
            rng,
            permutations,
        ),
        "velocity_target_profile": separation_diagnostics(
            behavior_features[non_high_mask],
            labels[non_high_mask],
            rng,
            permutations,
        ),
    }

    neural_pca_model = PCA(n_components=2).fit(neural_rate_features)
    behavior_pca_model = PCA(n_components=2).fit(behavior_features)
    neural_pca = neural_pca_model.transform(neural_rate_features)
    behavior_pca = behavior_pca_model.transform(behavior_features)

    integrity_failure_count = sum(
        len(integrity[key])
        for key in (
            "missing_processed", "extra_processed", "missing_raw", "extra_raw",
            "artifact_sha256_failures", "embedded_source_md5_failures", "raw_md5_failures",
            "schema_failures", "shape_failures", "value_failures",
        )
    )
    integrity["failure_count"] = integrity_failure_count
    integrity["status"] = "pass" if integrity_failure_count == 0 else "fail"

    summary = {
        "analysis": "indy_month_drift_analysis_v1",
        "session_count": len(session_frame),
        "month_count": len(MONTH_ORDER),
        "months": MONTH_ORDER,
        "split_counts": manifest["split_counts"],
        "total_duration_hours": float(session_frame["duration_min"].sum() / 60.0),
        "speed_histogram_cap_cm_s_p99_9": speed_cap,
        "selected_channels_zero_based": selected_channels.tolist(),
        "integrity": integrity,
        "separation": separation,
        "sensitivity_excluding_high_priority": sensitivity_excluding_high_priority,
        "scalar_month_tests": scalar_tests,
        "temporal_correlations": temporal_correlations,
        "split_summary": split_summary,
        "session_review_counts": {
            priority: int((session_frame["review_priority"] == priority).sum())
            for priority in ("high", "medium", "low")
        },
        "high_priority_sessions": session_frame.loc[
            session_frame["review_priority"] == "high", "session"
        ].tolist(),
        "medium_priority_sessions": session_frame.loc[
            session_frame["review_priority"] == "medium", "session"
        ].tolist(),
        "largest_month_pairs": {
            "channel_composition": pairwise_frame.nlargest(3, "channel_composition_js")[
                ["month_a", "month_b", "channel_composition_js", "channel_rate_spearman", "top32_jaccard"]
            ].to_dict("records"),
            "speed_distribution": pairwise_frame.nlargest(3, "speed_distribution_js")[
                ["month_a", "month_b", "speed_distribution_js", "direction_distribution_js"]
            ].to_dict("records"),
        },
        "artifacts": {
            "session_csv": str(SESSION_CSV.relative_to(ROOT)),
            "month_csv": str(MONTH_CSV.relative_to(ROOT)),
            "pairwise_csv": str(PAIRWISE_CSV.relative_to(ROOT)),
            "overview_figure": str(OVERVIEW_FIGURE.relative_to(ROOT)),
            "quality_figure": str(QUALITY_FIGURE.relative_to(ROOT)),
        },
    }

    SESSION_CSV.parent.mkdir(parents=True, exist_ok=True)
    session_frame.to_csv(SESSION_CSV, index=False)
    month_frame.to_csv(MONTH_CSV, index=False)
    pairwise_frame.to_csv(PAIRWISE_CSV, index=False)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_figures(
        session_frame,
        neural_pca,
        neural_pca_model.explained_variance_ratio_,
        behavior_pca,
        behavior_pca_model.explained_variance_ratio_,
        neural_month_matrix,
        speed_month_matrix,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-raw-checksums",
        action="store_true",
        help="Skip recomputing MD5 for the 10 GB immutable raw MAT files.",
    )
    parser.add_argument("--permutations", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.permutations < 100:
        raise ValueError("--permutations must be at least 100")
    summary = run_analysis(
        verify_raw_checksums=not args.skip_raw_checksums,
        permutations=args.permutations,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
