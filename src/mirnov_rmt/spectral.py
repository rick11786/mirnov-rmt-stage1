"""Spectral matrix helpers."""

from __future__ import annotations

import numpy as np


def cross_spectral_matrix(X: np.ndarray) -> np.ndarray:
    """Return S = X X* / K for snapshots X with shape (d, K)."""
    if X.ndim != 2:
        raise ValueError("X must have shape (d, K)")
    K = X.shape[1]
    if K <= 0:
        raise ValueError("X must contain at least one snapshot")
    return X @ X.conj().T / K


def coherence_matrix(S: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize a cross-spectral matrix to unit diagonal coherence form."""
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("S must be square")
    diag = np.real(np.diag(S))
    scale = 1.0 / np.sqrt(np.maximum(diag, eps))
    C = scale[:, None] * S * scale[None, :]
    return 0.5 * (C + C.conj().T)
