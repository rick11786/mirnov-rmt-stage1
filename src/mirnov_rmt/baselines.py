"""Simple FFT and SVD baselines."""

from __future__ import annotations

import numpy as np

from mirnov_rmt.snapshots import fourier_snapshots


def fft_peak_stat(Y: np.ndarray, L: int, H: int, r: int, window: str = "hann") -> float:
    """Return average Fourier-bin power across channels and snapshots."""
    X = fourier_snapshots(Y, L=L, H=H, r=r, window=window)
    return float(np.mean(np.abs(X) ** 2))


def svd_stat(Y: np.ndarray) -> np.ndarray:
    """Return singular values after subtracting channel means."""
    centered = Y - np.mean(Y, axis=1, keepdims=True)
    return np.linalg.svd(centered, compute_uv=False)
