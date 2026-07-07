"""Synthetic Mirnov-like time-series generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mirnov_rmt.geometry import toroidal_angles


@dataclass(frozen=True)
class MirnovMetadata:
    """Metadata returned by the synthetic generator."""

    phi: np.ndarray
    noise_sigma: float
    signal_power: float
    snr_db: float | None


def _mode_amplitude(mode: dict[str, Any], T: int) -> np.ndarray:
    A = mode.get("A", 1.0)
    if np.isscalar(A):
        amp = np.full(T, float(A))
    else:
        amp = np.asarray(A, dtype=float)
        if amp.shape != (T,):
            raise ValueError("array-valued mode amplitude must have shape (T,)")

    envelope = mode.get("envelope")
    if envelope is None:
        return amp
    if isinstance(envelope, str):
        if envelope == "hann":
            return amp * np.hanning(T)
        raise ValueError(f"unknown envelope: {envelope}")
    env = np.asarray(envelope, dtype=float)
    if env.shape != (T,):
        raise ValueError("array-valued envelope must have shape (T,)")
    return amp * env


def generate_mirnov(
    d: int,
    T: int,
    modes: list[dict[str, Any]],
    snr_db: float | None = None,
    noise_sigma: float | None = None,
    fs: float = 1.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, MirnovMetadata]:
    """Generate real-valued synthetic Mirnov array data.

    The sampling frequency ``fs`` is recorded for API clarity; mode frequencies
    are specified in rad/sample.
    """
    del fs
    if d <= 0 or T <= 0:
        raise ValueError("d and T must be positive")
    if snr_db is not None and noise_sigma is not None:
        raise ValueError("provide only one of snr_db or noise_sigma")

    rng = np.random.default_rng(seed)
    phi = toroidal_angles(d)
    t = np.arange(T)
    signal = np.zeros((d, T), dtype=float)

    for mode in modes:
        amp = _mode_amplitude(mode, T)
        omega = float(mode["omega"])
        n = int(mode["n"])
        phase = float(mode.get("phase", rng.uniform(0.0, 2.0 * np.pi)))
        angle = omega * t[None, :] + n * phi[:, None] + phase
        signal += amp[None, :] * np.cos(angle)

    signal_power = float(np.mean(signal**2))
    if noise_sigma is None:
        if snr_db is None:
            noise_sigma = 1.0
        elif signal_power <= 0.0:
            noise_sigma = 1.0
        else:
            noise_sigma = float(np.sqrt(signal_power / (10.0 ** (snr_db / 10.0))))

    noise = rng.normal(0.0, float(noise_sigma), size=(d, T))
    metadata = MirnovMetadata(
        phi=phi,
        noise_sigma=float(noise_sigma),
        signal_power=signal_power,
        snr_db=snr_db,
    )
    return signal + noise, phi, metadata
