"""Convenience wrappers for one-frequency spectral coherence analysis."""

from __future__ import annotations

import numpy as np

from mirnov_rmt.rmt import mp_detect
from mirnov_rmt.snapshots import fourier_snapshots
from mirnov_rmt.spectral import coherence_matrix, cross_spectral_matrix


def analyze_frequency(Y: np.ndarray, L: int, H: int, r: int) -> dict[str, object]:
    """Build X, S, C and run MP detection for one Fourier bin."""
    X = fourier_snapshots(Y, L=L, H=H, r=r)
    S = cross_spectral_matrix(X)
    C = coherence_matrix(S)
    detection = mp_detect(C, K=X.shape[1])
    return {"X": X, "S": S, "C": C, **detection}
