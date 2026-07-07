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


def mp_detect(
    C: np.ndarray,
    K: int,
    delta: float = 0.0,
    threshold: float | None = None,
) -> dict[str, object]:
    """Detect eigenvalue outliers above an MP or supplied threshold."""
    eigvals, eigvecs = eigen_decomp(C)
    tau = mp_edges(C.shape[0], K)[1] + delta if threshold is None else threshold
    estimated_rank = int(np.sum(eigvals > tau))
    return {
        "detected": estimated_rank > 0,
        "estimated_rank": estimated_rank,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "threshold": float(tau),
    }
