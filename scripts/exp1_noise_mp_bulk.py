"""Experiment 1: noise-only MP bulk."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mirnov_rmt.rmt import eigen_decomp, mp_edges
from mirnov_rmt.snapshots import fourier_snapshots
from mirnov_rmt.spectral import coherence_matrix, cross_spectral_matrix
from mirnov_rmt.synthetic import generate_mirnov


ROOT = Path(__file__).resolve().parents[1]


def mp_density(x: np.ndarray, q: float) -> np.ndarray:
    """Marchenko-Pastur density for unit noise covariance."""
    lam_minus = (1.0 - np.sqrt(q)) ** 2
    lam_plus = (1.0 + np.sqrt(q)) ** 2
    density = np.zeros_like(x, dtype=float)
    mask = (x >= lam_minus) & (x <= lam_plus) & (x > 0.0)
    density[mask] = np.sqrt((lam_plus - x[mask]) * (x[mask] - lam_minus)) / (2.0 * np.pi * q * x[mask])
    return density


def run(seed: int = 101) -> dict[str, float]:
    d, L, H, K, r = 48, 256, 256, 64, 30
    T = (K - 1) * H + L
    Y, _, _ = generate_mirnov(d=d, T=T, modes=[], noise_sigma=1.0, seed=seed)
    X = fourier_snapshots(Y, L=L, H=H, r=r)
    C = coherence_matrix(cross_spectral_matrix(X))
    eigvals = eigen_decomp(C)[0]
    lam_minus, lam_plus = mp_edges(d, X.shape[1])

    x_grid = np.linspace(max(1e-6, lam_minus * 0.4), lam_plus * 1.12, 500)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(eigvals, bins=32, density=True, color="#87aeca", alpha=0.78, edgecolor="white", label="sample eigenvalues")
    ax.plot(x_grid, mp_density(x_grid, d / X.shape[1]), color="#1f2937", linewidth=2.0, label="MP density")
    ax.axvline(lam_minus, color="#222222", linestyle="--", label=rf"$\lambda_-$={lam_minus:.2f}")
    ax.axvline(lam_plus, color="#c03a2b", linestyle="--", label=rf"$\lambda_+$={lam_plus:.2f}")
    ax.scatter(eigvals, np.full_like(eigvals, -0.015), marker="|", color="#334155", s=70, clip_on=False, label="eigenvalue rug")
    ax.set_title("Noise-only coherence eigenvalues vs MP support")
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Density")
    ax.set_ylim(bottom=-0.05)
    ax.legend()
    out = ROOT / "figures" / "fig1_noise_mp_bulk.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)

    summary = {
        "d": d,
        "K": X.shape[1],
        "q": d / X.shape[1],
        "lambda_minus": lam_minus,
        "lambda_plus": lam_plus,
        "lambda_max": float(eigvals[0]),
        "lambda_min": float(eigvals[-1]),
        "mp_outlier_count": int(np.sum(eigvals > lam_plus)),
    }
    pd.DataFrame([summary]).to_csv(ROOT / "results" / "exp1_noise_mp_bulk.csv", index=False)
    return summary


if __name__ == "__main__":
    print(run())
