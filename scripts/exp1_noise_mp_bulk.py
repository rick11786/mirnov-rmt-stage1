"""Experiment 1: noise-only MP bulk."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

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

    rng = np.random.default_rng(seed + 1000)
    empirical_eigvals = []
    B_density = 200
    for _ in range(B_density):
        Yb, _, _ = generate_mirnov(
            d=d,
            T=T,
            modes=[],
            noise_sigma=1.0,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        Xb = fourier_snapshots(Yb, L=L, H=H, r=r)
        Cb = coherence_matrix(cross_spectral_matrix(Xb))
        empirical_eigvals.append(eigen_decomp(Cb)[0])
    empirical_eigvals = np.concatenate(empirical_eigvals)

    x_grid = np.linspace(1e-6, 4.0, 600)
    x_mp = np.linspace(lam_minus, lam_plus, 500)
    empirical_density = gaussian_kde(empirical_eigvals)(x_grid)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(eigvals, bins=np.linspace(0.0, 4.0, 34), density=True, color="#cad7e3", alpha=0.75, edgecolor="white", label="one realization")
    ax.plot(x_grid, empirical_density, color="#2563a6", linewidth=2.1, label=f"simulation density (B={B_density})")
    ax.plot(x_mp, mp_density(x_mp, d / X.shape[1]), color="#c2410c", linewidth=2.0, linestyle="--", label="MP density")
    ax.annotate(
        rf"$\lambda_-$={lam_minus:.2f}",
        xy=(lam_minus, 0.0),
        xytext=(0.31, 0.94),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "-", "color": "#475569", "lw": 1.0},
        color="#475569",
        ha="left",
    )
    ax.annotate(
        rf"$\lambda_+$={lam_plus:.2f}",
        xy=(lam_plus, 0.0),
        xytext=(0.78, 0.80),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "-", "color": "#475569", "lw": 1.0},
        color="#475569",
        ha="left",
    )
    ax.set_title("Noise-only coherence eigenvalues vs MP support")
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Density")
    ax.set_xlim(0.2, 4.0)
    ax.set_ylim(bottom=0.0)
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
        "density_trials": B_density,
    }
    pd.DataFrame([summary]).to_csv(ROOT / "results" / "exp1_noise_mp_bulk.csv", index=False)
    return summary


if __name__ == "__main__":
    print(run())
