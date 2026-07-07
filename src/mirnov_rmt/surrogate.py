"""Empirical surrogate thresholds for spectral coherence."""

from __future__ import annotations

import numpy as np

from mirnov_rmt.rmt import eigen_decomp
from mirnov_rmt.snapshots import fourier_snapshots
from mirnov_rmt.spectral import coherence_matrix, cross_spectral_matrix


def circular_shift_channels(Y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independently circularly shift each channel."""
    if Y.ndim != 2:
        raise ValueError("Y must have shape (d, T)")
    d, T = Y.shape
    shifted = np.empty_like(Y)
    shifts = rng.integers(0, T, size=d)
    for j, shift in enumerate(shifts):
        shifted[j] = np.roll(Y[j], int(shift))
    return shifted


def phase_randomize_channels(Y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Preserve channel FFT amplitudes while randomizing phases."""
    spectrum = np.fft.rfft(Y, axis=1)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=spectrum.shape)
    phases[:, 0] = 0.0
    if Y.shape[1] % 2 == 0:
        phases[:, -1] = 0.0
    randomized = np.abs(spectrum) * np.exp(1j * phases)
    return np.fft.irfft(randomized, n=Y.shape[1], axis=1)


def surrogate_threshold(
    Y: np.ndarray,
    L: int,
    H: int,
    r: int,
    B: int = 200,
    alpha: float = 0.05,
    method: str = "circular_shift",
    seed: int | None = None,
) -> tuple[float, np.ndarray]:
    """Return empirical lambda-max threshold from channel-wise surrogates."""
    if B <= 0:
        raise ValueError("B must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    rng = np.random.default_rng(seed)
    lambdas = np.empty(B, dtype=float)
    for b in range(B):
        if method == "circular_shift":
            Ys = circular_shift_channels(Y, rng)
        elif method == "phase_randomization":
            Ys = phase_randomize_channels(Y, rng)
        else:
            raise ValueError(f"unsupported surrogate method: {method}")
        X = fourier_snapshots(Ys, L=L, H=H, r=r)
        C = coherence_matrix(cross_spectral_matrix(X))
        lambdas[b] = float(eigen_decomp(C)[0][0])
    return float(np.quantile(lambdas, 1.0 - alpha)), lambdas
