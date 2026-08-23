"""
All functions accept ECoG shaped [trials, channels, time] and return
[trials, channels, features_per_channel].

Methods implemented:
- modified S-transform (MST): 35 time-averaged PSD values from 1--35 Hz
- bandpower / PSD: Welch PSD integrated over five physiological bands
- FFT / Fourier spectrum: one-sided Fourier power spectrum from 1--35 Hz
- filter bank: Butterworth band-pass filtering followed by band power
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt, welch


DEFAULT_BANDS = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("low_gamma", 30.0, 35.0),
)


def _validate(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"Expected [trials, channels, time], got {x.shape}.")
    if not np.isfinite(x).all():
        raise ValueError("Input contains non-finite values.")
    return x


def _log_power(values: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return values
    tiny = np.finfo(np.float64).tiny
    return 10.0 * np.log10(np.maximum(values, tiny))


def mst_frequency_coefficients(x: np.ndarray, fs: float, frequency_hz: float, p: float = 0.52, q: float = 1.0, truncate: float = 4.0) -> np.ndarray:
    """
    Return the modified S-transform coefficient versus time.

    This numerically discretizes Xu et al. (2020), equations 7--10, by
    Gaussian-smoothing the frequency-demodulated signal.
    """
    x = _validate(x)
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive.")

    time = np.arange(x.shape[-1], dtype=np.float64) / fs
    sigma_seconds = p / (abs(frequency_hz) ** q)
    sigma_samples = max(sigma_seconds * fs, 1e-12)
    demodulated = x * np.exp(-1j * 2.0 * np.pi * frequency_hz * time)
    real_part = gaussian_filter1d(
        demodulated.real,
        sigma=sigma_samples,
        axis=-1,
        mode="reflect",
        truncate=truncate,
    )
    imaginary_part = gaussian_filter1d(
        demodulated.imag,
        sigma=sigma_samples,
        axis=-1,
        mode="reflect",
        truncate=truncate,
    )
    return real_part + 1j * imaginary_part


def mst_psd_features(x: np.ndarray, fs: float = 100.0, fmin: int = 1, fmax: int = 35, p: float = 0.52, q: float = 1.0, truncate: float = 4.0, verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract mean MST power at every integer frequency from fmin to fmax.
    """
    x = _validate(x)
    frequencies = np.arange(fmin, fmax + 1, dtype=np.float64)
    output = np.empty((*x.shape[:2], len(frequencies)), dtype=np.float64)
    for index, frequency in enumerate(frequencies):
        coefficient = mst_frequency_coefficients(
            x=x,
            fs=fs,
            frequency_hz=float(frequency),
            p=p,
            q=q,
            truncate=truncate,
        )
        output[:, :, index] = np.mean(np.abs(coefficient) ** 2, axis=-1)
        completed = index + 1
        if verbose and (
            index == 0 or completed % 5 == 0 or completed == len(frequencies)
        ):
            print(
                f"MST: completed {completed}/{len(frequencies)} frequencies "
                f"({frequency:.0f} Hz)"
            )
    return output, frequencies


def bandpower_psd_features(x: np.ndarray, fs: float = 100.0, bands=DEFAULT_BANDS, nperseg: int = 200, log_power: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD -> integrated power in broad frequency bands.

    Example output with five bands:
        [trials, channels, 5]

    Band features are delta, theta, alpha, beta and low-gamma by default.
    """
    x = _validate(x)
    nperseg = min(int(nperseg), x.shape[-1])
    freqs, psd = welch(x, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2, detrend="constant", scaling="density", axis=-1)
    out = np.empty((x.shape[0], x.shape[1], len(bands)), dtype=np.float64)
    labels = []
    for band_idx, (name, low, high) in enumerate(bands):
        # Include the final upper endpoint for the last requested band.
        if band_idx == len(bands) - 1:
            mask = (freqs >= low) & (freqs <= high)
        else:
            mask = (freqs >= low) & (freqs < high)
        if np.count_nonzero(mask) < 2:
            raise ValueError(
                f"Too few Welch bins in band {name} ({low}-{high} Hz)."
            )
        out[:, :, band_idx] = np.trapezoid(
            psd[:, :, mask],
            freqs[mask],
            axis=-1,
        )
        labels.append(f"{name}:{low:g}-{high:g}Hz")

    return _log_power(out, log_power), np.asarray(labels)


def fft_spectrum_features(x: np.ndarray, fs: float = 100.0, fmin: float = 1.0, fmax: float = 35.0, log_power: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """One-sided FFT power spectrum between fmin and fmax.

    For a 3-s epoch at 100 Hz, FFT resolution is 1/3 Hz, therefore the
    1--35 Hz interval contains 103 spectral bins per channel.
    """
    x = _validate(x)
    n = x.shape[-1]

    # Remove DC and apply a Hann window to reduce spectral leakage.
    centered = x - x.mean(axis=-1, keepdims=True)
    window = np.hanning(n)
    windowed = centered * window

    spectrum = np.fft.rfft(windowed, axis=-1)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # Periodogram-like power scaling. Absolute scaling is less important than
    # preserving a consistent train/test representation.
    normalization = fs * np.sum(window ** 2)
    power = (np.abs(spectrum) ** 2) / normalization

    # Convert two-sided energy to one-sided power except DC/Nyquist.
    if power.shape[-1] > 2:
        power[..., 1:-1] *= 2.0

    mask = (freqs >= fmin) & (freqs <= fmax)
    selected_freqs = freqs[mask]
    selected_power = power[:, :, mask]
    return _log_power(selected_power, log_power), selected_freqs


def filterbank_features(x: np.ndarray, fs: float = 100.0, bands=DEFAULT_BANDS, order: int = 4, log_power: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Band-pass filter bank -> mean-square power in each filtered band.

    Each channel is passed through one Butterworth band-pass filter per band.
    The mean squared filtered signal is then used as the feature.
    """
    x = _validate(x)
    nyquist = fs / 2.0
    out = np.empty((x.shape[0], x.shape[1], len(bands)), dtype=np.float64)
    labels = []

    for band_idx, (name, low, high) in enumerate(bands):
        if not 0 < low < high < nyquist:
            raise ValueError(
                f"Band {name}={low}-{high} Hz must lie inside (0, {nyquist}) Hz."
            )
        sos = butter(N=order, Wn=[low, high], btype="bandpass", fs=fs, output="sos")
        filtered = sosfiltfilt(sos, x, axis=-1)
        out[:, :, band_idx] = np.mean(filtered ** 2, axis=-1)
        labels.append(f"{name}:{low:g}-{high:g}Hz")

    return _log_power(out, log_power), np.asarray(labels)
