"""Random matrix threshold helpers."""

from __future__ import annotations

import numpy as np


def mp_edges(d: int, K: int) -> tuple[float, float]:
    """Return Marchenko-Pastur lower and upper edges for q=d/K."""
    if d <= 0 or K <= 0:
        raise ValueError("d and K must be positive")
    q = d / K
    root = np.sqrt(q)
    return (1.0 - root) ** 2, (1.0 + root) ** 2


def eigen_decomp(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return eigenvalues descending and matching eigenvectors."""
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]
