from dataclasses import dataclass, field


@dataclass
class ReproductionConfig:
    # Acquisition / common preprocessing
    original_fs: int = 1000
    target_fs: int = 100
    n_channels: int = 64
    trial_seconds: float = 3.0

    # Shared spectral range
    fmin: int = 1
    fmax: int = 35

    # Xu et al. MST parameters
    mst_p: float = 0.52
    mst_q: float = 1.0
    gaussian_truncate: float = 4.0

    # Alternative spectral feature extractors
    spectral_bands: tuple = field(default_factory=lambda: (
        ("delta", 1.0, 4.0),
        ("theta", 4.0, 8.0),
        ("alpha", 8.0, 13.0),
        ("beta", 13.0, 30.0),
        ("low_gamma", 30.0, 35.0),
    ))
    welch_nperseg: int = 200
    filterbank_order: int = 4
    log_power_features: bool = True

    # Channel ranking / wrapper
    ranking_folds: int = 10
    ranking_repeats: int = 10
    wrapper_validation_fraction: float = 0.20
    minimum_channels: int = 32
    random_state: int = 2020

    # RBF-SVM tuning
    tune_svm: bool = True
    svm_cv_folds: int = 10
    c_grid: tuple = field(default_factory=lambda: tuple(2.0 ** k for k in range(-5, 16, 2)))
    gamma_grid: tuple = field(default_factory=lambda: tuple(2.0 ** k for k in range(-15, 4, 2)))
    standardize_features: bool = False


# Channels highlighted in Xu et al. (2020) Fig. 10B, 1-based.
PAPER_REPORTED_CHANNELS_1BASED = (
    9, 12, 13, 17, 18, 21, 22, 24, 38, 48, 56, 58, 63
)
PAPER_REPORTED_CHANNELS_0BASED = tuple(ch - 1 for ch in PAPER_REPORTED_CHANNELS_1BASED)
