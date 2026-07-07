"""Welch-style local Fourier snapshots."""

from __future__ import annotations

import numpy as np


def _window_values(name: str, L: int) -> np.ndarray:
    if name == "hann":
        return np.hanning(L)
    if name in {"boxcar", "rect", "rectangular"}:
        return np.ones(L)
    raise ValueError(f"unsupported window: {name}")


def fourier_snapshots(
    Y: np.ndarray,
    L: int,
    H: int,
    r: int,
    window: str = "hann",
) -> np.ndarray:
    """Return complex Fourier snapshots with shape (d, K)."""
    if Y.ndim != 2:
        raise ValueError("Y must have shape (d, T)")
    d, T = Y.shape
    if L <= 0 or H <= 0:
        raise ValueError("L and H must be positive")
    if r < 0 or r >= L:
        raise ValueError("frequency bin r must satisfy 0 <= r < L")
    starts = list(range(0, T - L + 1, H))
    if not starts:
        raise ValueError("no complete Welch segments fit in Y")

    h = _window_values(window, L)
    norm = np.sqrt(np.sum(h**2))
    kernel = h * np.exp(-2j * np.pi * r * np.arange(L) / L) / norm

    X = np.empty((d, len(starts)), dtype=complex)
    for k, start in enumerate(starts):
        segment = Y[:, start : start + L]
        X[:, k] = segment @ kernel
    return X
