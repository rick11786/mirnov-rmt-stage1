"""Toroidal template fitting from outlier eigenvectors."""

from __future__ import annotations

import numpy as np


def toroidal_templates(phi: np.ndarray, n_candidates: list[int] | np.ndarray) -> dict[int, np.ndarray]:
    """Return normalized toroidal templates exp(i n phi)/sqrt(d)."""
    phi = np.asarray(phi)
    d = phi.size
    if d <= 0:
        raise ValueError("phi must not be empty")
    return {int(n): np.exp(1j * int(n) * phi) / np.sqrt(d) for n in n_candidates}


def estimate_n(
    eigvec: np.ndarray,
    phi: np.ndarray,
    n_candidates: list[int] | np.ndarray,
) -> tuple[int, dict[int, float]]:
    """Estimate toroidal mode number by squared template alignment."""
    v = np.asarray(eigvec, dtype=complex)
    norm = np.linalg.norm(v)
    if norm == 0.0:
        raise ValueError("eigvec must be nonzero")
    v = v / norm
    scores = {
        n: float(abs(np.vdot(v, template)) ** 2)
        for n, template in toroidal_templates(phi, n_candidates).items()
    }
    n_hat = max(scores, key=scores.get)
    return int(n_hat), scores
