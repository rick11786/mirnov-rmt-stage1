"""Sensor geometry helpers."""

from __future__ import annotations

import numpy as np


def toroidal_angles(d: int) -> np.ndarray:
    """Return equally spaced toroidal probe angles."""
    if d <= 0:
        raise ValueError("d must be positive")
    return 2.0 * np.pi * np.arange(d) / d
