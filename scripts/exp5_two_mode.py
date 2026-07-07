"""Experiment 5: two-mode frequency separation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory

from mirnov_rmt.baselines import svd_stat
from mirnov_rmt.mode_fit import estimate_n
from mirnov_rmt.rmt import eigen_decomp, mp_edges
from mirnov_rmt.snapshots import fourier_snapshots
from mirnov_rmt.spectral import coherence_matrix, cross_spectral_matrix
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


def _coherence_at(Y: np.ndarray, L: int, H: int, r: int) -> tuple[np.ndarray, np.ndarray]:
    X = fourier_snapshots(Y, L=L, H=H, r=r)
    C = coherence_matrix(cross_spectral_matrix(X))
    return eigen_decomp(C)


def run(seed: int = 505) -> dict[str, float]:
    d, L, H, K, r1 = 48, 256, 256, 64, 30
    T = (K - 1) * H + L
    n1, n2 = 2, 1
    omega1 = 2.0 * np.pi * r1 / L
    delta_specs = [
        ("2pi_over_L", 2.0 * np.pi / L),
        ("pi_over_L", np.pi / L),
        ("pi_over_2L", np.pi / (2.0 * L)),
        ("pi_over_4L", np.pi / (4.0 * L)),
    ]
    rng = np.random.default_rng(seed)
    rows = []
    band_rows = []
    example_band_eigvals = None
    example_threshold = None
    for label, delta in delta_specs:
        omega2 = omega1 + delta
        Y, phi, _ = generate_mirnov(
            d=d,
            T=T,
            modes=[
                {"A": 1.0, "omega": omega1, "n": n1, "phase": 0.1},
                {"A": 0.75, "omega": omega2, "n": n2, "phase": 1.2},
            ],
            noise_sigma=0.55,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        svd_ratio = float(svd_stat(Y)[1] / svd_stat(Y)[0])
        fft_powers = []
        for rr in range(r1 - 4, r1 + 6):
            X = fourier_snapshots(Y, L=L, H=H, r=rr)
            fft_powers.append({"bin": rr, "power": float(np.mean(np.abs(X) ** 2))})
            eigvals, eigvecs = _coherence_at(Y, L, H, rr)
            threshold = mp_edges(d, K)[1]
            rank = int(np.sum(eigvals > threshold))
            n_hat_1, scores_1 = estimate_n(eigvecs[:, 0], phi, np.arange(-10, 11))
            rows.append(
                {
                    "delta_label": label,
                    "delta_omega": delta,
                    "bin": rr,
                    "fft_power": fft_powers[-1]["power"],
                    "lambda_max": float(eigvals[0]),
                    "mp_threshold": threshold,
                    "estimated_rank": rank,
                    "leading_n_hat": n_hat_1,
                    "leading_score_n1": float(scores_1[n1]),
                    "leading_score_n2": float(scores_1[n2]),
                    "svd_sigma2_over_sigma1": svd_ratio,
                }
            )

        # Band-averaged matrix over bins around the two close modes.
        C_band = None
        for rr in range(r1 - 1, r1 + 3):
            X = fourier_snapshots(Y, L=L, H=H, r=rr)
            S = cross_spectral_matrix(X)
            C = coherence_matrix(S)
            C_band = C if C_band is None else C_band + C
        C_band = C_band / 4.0
        eigvals_band, eigvecs_band = eigen_decomp(C_band)
        threshold = mp_edges(d, K)[1]
        if label == "pi_over_L":
            example_band_eigvals = eigvals_band.copy()
            example_threshold = threshold
        n_hat_a, scores_a = estimate_n(eigvecs_band[:, 0], phi, np.arange(-10, 11))
        n_hat_b, scores_b = estimate_n(eigvecs_band[:, 1], phi, np.arange(-10, 11))
        band_rows.append(
            {
                "delta_label": label,
                "delta_omega": delta,
                "band_rank": int(np.sum(eigvals_band > threshold)),
                "band_lambda1": float(eigvals_band[0]),
                "band_lambda2": float(eigvals_band[1]),
                "band_n_hat_1": n_hat_a,
                "band_n_hat_2": n_hat_b,
                "band_score1_n1": float(scores_a[n1]),
                "band_score1_n2": float(scores_a[n2]),
                "band_score2_n1": float(scores_b[n1]),
                "band_score2_n2": float(scores_b[n2]),
            }
        )

    df = pd.DataFrame(rows)
    band_df = pd.DataFrame(band_rows)
    df.to_csv(ROOT / "results" / "exp5_two_mode_bins.csv", index=False)
    band_df.to_csv(ROOT / "results" / "exp5_two_mode_band.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), sharex=True)
    axes = axes.ravel()
    for ax, (label, _) in zip(axes, delta_specs):
        sub = df[df["delta_label"] == label]
        ax.plot(sub["bin"], sub["fft_power"], marker="o", label="FFT avg power")
        ax2 = ax.twinx()
        ax2.plot(sub["bin"], sub["estimated_rank"], marker="s", color="#c03a2b", label="MP rank")
        ax.set_title(label)
        ax.set_xlabel("Fourier bin")
        ax.set_ylabel("Power")
        ax2.set_ylabel("Rank")
        ax2.set_ylim(-0.2, max(2.5, sub["estimated_rank"].max() + 0.5))
    fig.suptitle("Two-mode frequency separation: FFT power and coherence rank")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig5_two_mode_frequency_separation.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    x = np.arange(len(band_df))
    ax.plot(x, band_df["band_lambda1"], marker="o", label="band lambda1")
    ax.plot(x, band_df["band_lambda2"], marker="s", label="band lambda2")
    ax.axhline(mp_edges(d, K)[1], color="#c03a2b", linestyle="--", label="MP edge")
    ax.set_xticks(x, labels=band_df["delta_label"], rotation=20)
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Band-averaged coherence outliers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig5_band_averaged_outliers.png", dpi=160)
    plt.close(fig)

    if example_band_eigvals is not None and example_threshold is not None:
        outlier_mask = example_band_eigvals > example_threshold
        bulk_like = example_band_eigvals[~outlier_mask]
        outliers = example_band_eigvals[outlier_mask]
        x_max = max(float(example_band_eigvals[0]) * 1.08, example_threshold * 1.2)
        x_mp = np.linspace(mp_edges(d, K)[0], example_threshold, 500)

        fig, ax = plt.subplots(figsize=(7.4, 4.7))
        ax.hist(
            bulk_like,
            bins=np.linspace(0.0, x_max, 38),
            density=True,
            color="#b9cadb",
            alpha=0.78,
            edgecolor="white",
            label="bulk eigenvalues",
        )
        ax.plot(x_mp, mp_density(x_mp, d / K), color="#c2410c", linewidth=2.0, linestyle="--", label="MP density")
        axis_marker(ax, example_threshold, f"MP {example_threshold:.2f}")
        for idx, val in enumerate(outliers, start=1):
            axis_marker(ax, val, f"outlier {idx}: {val:.2f}", color="#b42318", marker="D")
        ax.set_title("Two-mode example: two band-averaged coherence outliers")
        ax.set_xlabel("Eigenvalue")
        ax.set_ylabel("Density")
        ax.set_ylim(bottom=0.0)
        ax.legend()
        fig.tight_layout()
        fig.savefig(ROOT / "figures" / "fig5_two_spike_band_spectrum.png", dpi=160)
        plt.close(fig)

    return {
        "cases": len(delta_specs),
        "max_band_rank": int(band_df["band_rank"].max()),
        "mean_band_rank": float(band_df["band_rank"].mean()),
    }


if __name__ == "__main__":
    print(run())
