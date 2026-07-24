"""Label-free 60-second compatibility gate for the frozen Indy decoder.

This module implements a conservative Phase-3a detector.  It is inspired by
MINDFUL's low-dimensional Gaussian KLD, but it is not a reproduction of that
paper: references are grouped by historical month and the PCA normalization is
fixed from reference sessions so that it can be deployed without target labels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class DetectorConfig:
    """All Phase-3a choices that must be fixed before January is inspected."""

    observation_bins: int = 1500
    bin_seconds: float = 0.04
    n_components: int = 5
    covariance_shrinkage: float = 0.10
    covariance_floor: float = 1e-4
    channel_scale_floor: float = 0.10
    warning_quantile: float = 0.99
    silent_rate_hz: float = 0.05
    expected_active_rate_hz: float = 0.50
    expected_active_fraction: float = 0.75


@dataclass(frozen=True)
class GaussianSummary:
    mean: np.ndarray
    covariance: np.ndarray


@dataclass(frozen=True)
class DetectorScore:
    """Reader-facing score for one 60-second prefix."""

    matched_simple_reference: str
    matched_mindful_reference: str
    pattern_distance: float
    robust_rate_distance: float
    absolute_log_rate_ratio: float
    unexpected_silent_channels: int
    global_mindful_kld: float
    multi_reference_mindful_kld: float
    simple_evidence_count: int
    combined_evidence_count: int
    simple_decision: str
    global_mindful_decision: str
    multi_mindful_decision: str
    combined_decision: str


def session_month(session_name: str) -> str:
    """Return YYYY-MM and reject names outside the canonical Indy convention."""
    parts = session_name.split("_")
    if len(parts) != 3 or len(parts[1]) != 8 or not parts[1].isdigit():
        raise ValueError(f"Invalid Indy session name: {session_name!r}")
    return f"{parts[1][:4]}-{parts[1][4:6]}"


def assert_pre_january(session_names: list[str] | tuple[str, ...]) -> None:
    """Make accidental use of the consumed January split a hard failure."""
    forbidden = [name for name in session_names if session_month(name) >= "2017-01"]
    if forbidden:
        raise ValueError(
            "Phase-3 detector development forbids January-or-later sessions: "
            + ", ".join(sorted(forbidden))
        )


def _validate_counts(counts: np.ndarray, config: DetectorConfig) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 32:
        raise ValueError(f"Expected counts with shape (32, bins), got {values.shape}")
    if values.shape[1] < config.observation_bins:
        raise ValueError(
            f"Need {config.observation_bins} bins, received {values.shape[1]}"
        )
    prefix = values[:, : config.observation_bins]
    if not np.isfinite(prefix).all() or np.any(prefix < 0):
        raise ValueError("Counts must be finite and non-negative")
    return prefix


def _regularized_gaussian(
    samples: np.ndarray, config: DetectorConfig
) -> GaussianSummary:
    values = np.asarray(samples, dtype=np.float64)
    covariance = np.cov(values, rowvar=False)
    if covariance.ndim == 0:
        covariance = np.asarray([[float(covariance)]])
    diagonal = np.diag(np.diag(covariance))
    covariance = (
        (1.0 - config.covariance_shrinkage) * covariance
        + config.covariance_shrinkage * diagonal
        + config.covariance_floor * np.eye(covariance.shape[0])
    )
    return GaussianSummary(values.mean(axis=0), covariance)


def gaussian_kld(reference: GaussianSummary, current: GaussianSummary) -> float:
    """KLD(reference || current), matching the direction used by MINDFUL."""
    dimension = reference.mean.size
    difference = current.mean - reference.mean
    sign_reference, logdet_reference = np.linalg.slogdet(reference.covariance)
    sign_current, logdet_current = np.linalg.slogdet(current.covariance)
    if sign_reference <= 0 or sign_current <= 0:
        raise ValueError("Regularized covariance must be positive definite")
    solved_covariance = np.linalg.solve(current.covariance, reference.covariance)
    solved_difference = np.linalg.solve(current.covariance, difference)
    value = 0.5 * (
        np.trace(solved_covariance)
        + difference @ solved_difference
        - dimension
        + logdet_current
        - logdet_reference
    )
    return float(max(value, 0.0))


def _higher_quantile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot calibrate a threshold from no values")
    try:
        return float(np.quantile(values, quantile, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility
        return float(np.quantile(values, quantile, interpolation="higher"))


class DriftDetector:
    """Fitted multi-reference detector with no access to velocity labels."""

    def __init__(self, config: DetectorConfig = DetectorConfig()) -> None:
        self.config = config
        self.reference_session_names: tuple[str, ...] = ()
        self.reference_months: tuple[str, ...] = ()
        self.reference_rate_profiles: dict[str, np.ndarray] = {}
        self.reference_active_masks: dict[str, np.ndarray] = {}
        self.channel_log_rate_scale: np.ndarray | None = None
        self.normalization_mean: np.ndarray | None = None
        self.normalization_std: np.ndarray | None = None
        self.pca_components: np.ndarray | None = None
        self.global_gaussian: GaussianSummary | None = None
        self.month_gaussians: dict[str, GaussianSummary] = {}
        self.thresholds: dict[str, float] = {}

    def _prefixes(
        self, sessions: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        names = tuple(sorted(sessions))
        assert_pre_january(names)
        if len(names) < 3:
            raise ValueError("At least three reference sessions are required")
        return {
            name: _validate_counts(sessions[name], self.config) for name in names
        }

    def _fit_rate_references(
        self, prefixes: Mapping[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
        rates = {
            name: counts.sum(axis=1)
            / (self.config.observation_bins * self.config.bin_seconds)
            for name, counts in prefixes.items()
        }
        profiles: dict[str, np.ndarray] = {}
        active_masks: dict[str, np.ndarray] = {}
        for month in sorted({session_month(name) for name in rates}):
            month_rates = np.stack(
                [rate for name, rate in rates.items() if session_month(name) == month]
            )
            profiles[month] = np.median(month_rates, axis=0)
            active_masks[month] = (
                (month_rates >= self.config.expected_active_rate_hz).mean(axis=0)
                >= self.config.expected_active_fraction
            )

        log_rates = np.stack([np.log1p(rate) for rate in rates.values()])
        median = np.median(log_rates, axis=0)
        mad_scale = 1.4826 * np.median(np.abs(log_rates - median), axis=0)
        scale = np.maximum(mad_scale, self.config.channel_scale_floor)
        return profiles, active_masks, scale

    def _fit_projection(
        self, prefixes: Mapping[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        # Every reference contributes exactly 1500 bins, so no long session or
        # month receives extra PCA weight.
        raw = np.concatenate([counts.T for counts in prefixes.values()], axis=0)
        mean = raw.mean(axis=0)
        std = np.maximum(raw.std(axis=0), 1e-6)
        normalized = (raw - mean) / std
        covariance = np.cov(normalized, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1][: self.config.n_components]
        components = eigenvectors[:, order].T
        latent = {
            name: ((counts.T - mean) / std) @ components.T
            for name, counts in prefixes.items()
        }
        return mean, std, components, latent

    def _fit_gaussian_references(
        self, latent: Mapping[str, np.ndarray]
    ) -> tuple[GaussianSummary, dict[str, GaussianSummary]]:
        global_reference = _regularized_gaussian(
            np.concatenate(list(latent.values()), axis=0), self.config
        )
        month_references = {
            month: _regularized_gaussian(
                np.concatenate(
                    [
                        values
                        for name, values in latent.items()
                        if session_month(name) == month
                    ],
                    axis=0,
                ),
                self.config,
            )
            for month in sorted({session_month(name) for name in latent})
        }
        return global_reference, month_references

    def _simple_metrics(
        self,
        counts: np.ndarray,
        profiles: Mapping[str, np.ndarray],
        active_masks: Mapping[str, np.ndarray],
        channel_scale: np.ndarray,
    ) -> tuple[str, float, float, float, int]:
        duration = self.config.observation_bins * self.config.bin_seconds
        rate = counts.sum(axis=1) / duration
        log_rate = np.log1p(rate)

        correlations: dict[str, float] = {}
        for month, profile in profiles.items():
            reference = np.log1p(profile)
            denominator = np.linalg.norm(log_rate - log_rate.mean()) * np.linalg.norm(
                reference - reference.mean()
            )
            correlations[month] = (
                float(
                    np.dot(log_rate - log_rate.mean(), reference - reference.mean())
                    / denominator
                )
                if denominator > 0
                else 0.0
            )
        matched = max(correlations, key=correlations.get)
        profile = profiles[matched]
        reference_log_rate = np.log1p(profile)
        pattern_distance = 1.0 - correlations[matched]
        robust_distance = float(
            np.median(np.abs(log_rate - reference_log_rate) / channel_scale)
        )
        rate_ratio = (np.median(rate) + 1e-6) / (np.median(profile) + 1e-6)
        absolute_log_ratio = float(abs(np.log(rate_ratio)))
        unexpected_silent = int(
            np.sum(
                (rate <= self.config.silent_rate_hz)
                & active_masks[matched]
            )
        )
        return (
            matched,
            float(pattern_distance),
            robust_distance,
            absolute_log_ratio,
            unexpected_silent,
        )

    def _project(self, counts: np.ndarray) -> np.ndarray:
        if (
            self.normalization_mean is None
            or self.normalization_std is None
            or self.pca_components is None
        ):
            raise RuntimeError("Detector has not been fitted")
        return (
            (counts.T - self.normalization_mean) / self.normalization_std
        ) @ self.pca_components.T

    def _raw_metrics(self, counts: np.ndarray) -> dict[str, float | int | str]:
        if (
            self.channel_log_rate_scale is None
            or self.global_gaussian is None
            or not self.reference_rate_profiles
            or not self.month_gaussians
        ):
            raise RuntimeError("Detector has not been fitted")
        simple = self._simple_metrics(
            counts,
            self.reference_rate_profiles,
            self.reference_active_masks,
            self.channel_log_rate_scale,
        )
        current_gaussian = _regularized_gaussian(self._project(counts), self.config)
        global_kld = gaussian_kld(self.global_gaussian, current_gaussian)
        monthly_kld = {
            month: gaussian_kld(reference, current_gaussian)
            for month, reference in self.month_gaussians.items()
        }
        matched_mindful = min(monthly_kld, key=monthly_kld.get)
        return {
            "matched_simple_reference": simple[0],
            "pattern_distance": simple[1],
            "robust_rate_distance": simple[2],
            "absolute_log_rate_ratio": simple[3],
            "unexpected_silent_channels": simple[4],
            "global_mindful_kld": global_kld,
            "matched_mindful_reference": matched_mindful,
            "multi_reference_mindful_kld": monthly_kld[matched_mindful],
        }

    def _fit_reference_state(self, prefixes: Mapping[str, np.ndarray]) -> None:
        (
            self.reference_rate_profiles,
            self.reference_active_masks,
            self.channel_log_rate_scale,
        ) = self._fit_rate_references(prefixes)
        (
            self.normalization_mean,
            self.normalization_std,
            self.pca_components,
            latent,
        ) = self._fit_projection(prefixes)
        self.global_gaussian, self.month_gaussians = self._fit_gaussian_references(
            latent
        )

    def _nested_month_calibration(
        self, prefixes: Mapping[str, np.ndarray]
    ) -> list[dict[str, float | int | str]]:
        """Score unseen inner months to match the outer validation problem."""
        months = sorted({session_month(name) for name in prefixes})
        if len(months) < 3:
            raise ValueError(
                "Nested month calibration requires at least three reference months"
            )
        rows: list[dict[str, float | int | str]] = []
        for held_month in months:
            inner_reference = {
                name: values
                for name, values in prefixes.items()
                if session_month(name) != held_month
            }
            inner_held = {
                name: values
                for name, values in prefixes.items()
                if session_month(name) == held_month
            }
            inner_detector = DriftDetector(self.config)
            inner_detector._fit_reference_state(inner_reference)
            rows.extend(
                inner_detector._raw_metrics(values)
                for values in inner_held.values()
            )
        return rows

    def fit(self, sessions: Mapping[str, np.ndarray]) -> "DriftDetector":
        """Fit references and empirical thresholds without using labels."""
        prefixes = self._prefixes(sessions)
        self.reference_session_names = tuple(sorted(prefixes))
        self.reference_months = tuple(
            sorted({session_month(name) for name in prefixes})
        )
        self._fit_reference_state(prefixes)

        calibration = self._nested_month_calibration(prefixes)
        names = (
            "pattern_distance",
            "robust_rate_distance",
            "absolute_log_rate_ratio",
            "unexpected_silent_channels",
            "global_mindful_kld",
            "multi_reference_mindful_kld",
        )
        self.thresholds = {
            name: _higher_quantile(
                [float(row[name]) for row in calibration],
                self.config.warning_quantile,
            )
            for name in names
        }
        return self

    @staticmethod
    def _decision(evidence_count: int) -> str:
        if evidence_count >= 2:
            return "abstain"
        if evidence_count == 1:
            return "warning"
        return "pass"

    def score(self, counts: np.ndarray) -> DetectorScore:
        """Score only the first 60 seconds; any later bins are ignored."""
        prefix = _validate_counts(counts, self.config)
        metrics = self._raw_metrics(prefix)
        pattern_abnormal = (
            float(metrics["pattern_distance"]) > self.thresholds["pattern_distance"]
            or float(metrics["robust_rate_distance"])
            > self.thresholds["robust_rate_distance"]
        )
        global_rate_abnormal = (
            float(metrics["absolute_log_rate_ratio"])
            > self.thresholds["absolute_log_rate_ratio"]
        )
        dropout_abnormal = (
            int(metrics["unexpected_silent_channels"])
            > self.thresholds["unexpected_silent_channels"]
        )
        global_mindful_abnormal = (
            float(metrics["global_mindful_kld"])
            > self.thresholds["global_mindful_kld"]
        )
        multi_mindful_abnormal = (
            float(metrics["multi_reference_mindful_kld"])
            > self.thresholds["multi_reference_mindful_kld"]
        )
        simple_count = int(pattern_abnormal) + int(global_rate_abnormal) + int(
            dropout_abnormal
        )
        combined_count = simple_count + int(multi_mindful_abnormal)
        return DetectorScore(
            matched_simple_reference=str(metrics["matched_simple_reference"]),
            matched_mindful_reference=str(metrics["matched_mindful_reference"]),
            pattern_distance=float(metrics["pattern_distance"]),
            robust_rate_distance=float(metrics["robust_rate_distance"]),
            absolute_log_rate_ratio=float(metrics["absolute_log_rate_ratio"]),
            unexpected_silent_channels=int(
                metrics["unexpected_silent_channels"]
            ),
            global_mindful_kld=float(metrics["global_mindful_kld"]),
            multi_reference_mindful_kld=float(
                metrics["multi_reference_mindful_kld"]
            ),
            simple_evidence_count=simple_count,
            combined_evidence_count=combined_count,
            simple_decision=self._decision(simple_count),
            global_mindful_decision="warning" if global_mindful_abnormal else "pass",
            multi_mindful_decision="warning" if multi_mindful_abnormal else "pass",
            combined_decision=self._decision(combined_count),
        )

    def metadata(self) -> dict:
        """JSON-safe fitted metadata; numerical reference arrays remain separate."""
        return {
            "method": "phase3a_multi_reference_rate_plus_low_dimensional_kld",
            "config": asdict(self.config),
            "reference_sessions": list(self.reference_session_names),
            "reference_months": list(self.reference_months),
            "thresholds": self.thresholds,
            "threshold_calibration": "nested_leave_one_complete_month_out",
            "decision_policy": {
                "warning": "one abnormal evidence family",
                "abstain": "at least two abnormal evidence families",
                "mindful_alone_can_abstain": False,
            },
        }

    def save(self, path, *, selected_channels: np.ndarray | None = None) -> None:
        """Save a non-pickle artifact suitable for later firmware export."""
        if self.global_gaussian is None or self.pca_components is None:
            raise RuntimeError("Detector has not been fitted")
        months = list(self.reference_months)
        metadata = self.metadata()
        arrays = {
            "metadata_json": np.asarray(
                __import__("json").dumps(metadata, sort_keys=True)
            ),
            "normalization_mean": self.normalization_mean.astype(np.float32),
            "normalization_std": self.normalization_std.astype(np.float32),
            "pca_components": self.pca_components.astype(np.float32),
            "channel_log_rate_scale": self.channel_log_rate_scale.astype(np.float32),
            "month_names": np.asarray(months),
            "month_rate_profiles": np.stack(
                [self.reference_rate_profiles[month] for month in months]
            ).astype(np.float32),
            "month_active_masks": np.stack(
                [self.reference_active_masks[month] for month in months]
            ).astype(np.uint8),
            "global_mean": self.global_gaussian.mean.astype(np.float32),
            "global_covariance": self.global_gaussian.covariance.astype(np.float32),
            "month_means": np.stack(
                [self.month_gaussians[month].mean for month in months]
            ).astype(np.float32),
            "month_covariances": np.stack(
                [self.month_gaussians[month].covariance for month in months]
            ).astype(np.float32),
        }
        if selected_channels is not None:
            channels = np.asarray(selected_channels, dtype=np.int16)
            if channels.shape != (32,):
                raise ValueError("selected_channels must contain exactly 32 indices")
            arrays["selected_channels_zero_based"] = channels
        np.savez_compressed(path, **arrays)
