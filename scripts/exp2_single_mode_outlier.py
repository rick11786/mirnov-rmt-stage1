"""Experiment 2: single coherent mode outlier and n recovery."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory

from mirnov_rmt.mode_fit import estimate_n
from mirnov_rmt.pipeline import analyze_frequency
from mirnov_rmt.rmt import mp_edges
from mirnov_rmt.synthetic import generate_mirnov


ROOT = Path(__file__).resolve().parents[1]


def axis_marker(ax, x: float, label: str, color: str = "#475569", marker: str = "v") -> None:
    """Place a compact marker and label on the x-axis."""
    transform = blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot([x], [0.0], marker=marker, markersize=5, color=color, transform=transform, clip_on=False)
    ax.text(x, -0.075, label, transform=transform, ha="center", va="top", fontsize=8.5, color=color, clip_on=False)


def mp_density(x: np.ndarray, q: float) -> np.ndarray:
    """Marchenko-Pastur density for unit noise covariance."""
    lam_minus = (1.0 - np.sqrt(q)) ** 2
    lam_plus = (1.0 + np.sqrt(q)) ** 2
    density = np.zeros_like(x, dtype=float)
    mask = (x >= lam_minus) & (x <= lam_plus) & (x > 0.0)
    density[mask] = np.sqrt((lam_plus - x[mask]) * (x[mask] - lam_minus)) / (2.0 * np.pi * q * x[mask])
    return density


def run(seed: int = 202) -> dict[str, float]:
    d, L, H, K, r = 48, 256, 256, 64, 30
    T = (K - 1) * H + L
    n_true, snr_db = 2, -30.0
    omega = 2.0 * np.pi * r / L
    Y, phi, _ = generate_mirnov(
        d=d,
        T=T,
        modes=[{"A": 1.0, "omega": omega, "n": n_true, "phase": 0.4}],
        snr_db=snr_db,
        seed=seed,
    )
    result = analyze_frequency(Y, L=L, H=H, r=r)
    eigvals = result["eigvals"]
    eigvecs = result["eigvecs"]
    threshold = float(result["threshold"])
    n_candidates = np.arange(-10, 11)
    n_hat, scores = estimate_n(eigvecs[:, 0], phi, n_candidates)

    outlier_mask = eigvals > threshold
    bulk_like = eigvals[~outlier_mask]
    outliers = eigvals[outlier_mask]
    x_max = max(float(eigvals[0]) * 1.08, threshold * 1.2)
    x_mp = np.linspace(mp_edges(d, K)[0], threshold, 500)

    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    ax.hist(bulk_like, bins=np.linspace(0.0, x_max, 36), density=True, color="#87aeca", alpha=0.78, edgecolor="white", label="bulk eigenvalues")
    ax.plot(x_mp, mp_density(x_mp, d / K), color="#c2410c", linewidth=2.0, linestyle="--", label="MP density")
    axis_marker(ax, threshold, f"MP {threshold:.2f}")
    for idx, val in enumerate(outliers, start=1):
        axis_marker(ax, val, f"outlier {idx}: {val:.2f}", color="#b42318", marker="D")
    ax.set_title("Single weak mode: MP bulk and detected outlier")
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Density")
    ax.set_ylim(bottom=0.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig2_single_outlier_eigs.png", dpi=160)
    plt.close(fig)

    score_values = np.array([scores[int(n)] for n in n_candidates])
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.bar(n_candidates, score_values, color="#629b65")
    ax.axvline(n_true, color="#c03a2b", linestyle="--", label=f"true n={n_true}")
    ax.set_title(f"Leading eigenvector toroidal template scores; n_hat={n_hat}")
    ax.set_xlabel("Candidate n")
    ax.set_ylabel(r"$|\hat v^* v_n|^2$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig2_n_scores.png", dpi=160)
    plt.close(fig)

    summary = {
        "d": d,
        "K": K,
        "snr_db": snr_db,
        "n_true": n_true,
        "n_hat": n_hat,
        "lambda_max": float(eigvals[0]),
        "threshold": threshold,
        "outlier_count": int(np.sum(outlier_mask)),
        "detected": bool(result["detected"]),
        "alignment_true_n": float(scores[n_true]),
    }
    pd.DataFrame([summary]).to_csv(ROOT / "results" / "exp2_single_mode_outlier.csv", index=False)
    pd.DataFrame({"n": n_candidates, "score": score_values}).to_csv(
        ROOT / "results" / "exp2_n_scores.csv", index=False
    )
    return summary


if __name__ == "__main__":
    print(run())
