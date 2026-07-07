"""Experiment 5: two-mode frequency separation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mirnov_rmt.baselines import svd_stat
from mirnov_rmt.mode_fit import estimate_n
from mirnov_rmt.rmt import eigen_decomp, mp_edges
from mirnov_rmt.snapshots import fourier_snapshots
from mirnov_rmt.spectral import coherence_matrix, cross_spectral_matrix
from mirnov_rmt.synthetic import generate_mirnov


ROOT = Path(__file__).resolve().parents[1]


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

    return {
        "cases": len(delta_specs),
        "max_band_rank": int(band_df["band_rank"].max()),
        "mean_band_rank": float(band_df["band_rank"].mean()),
    }


if __name__ == "__main__":
    print(run())
