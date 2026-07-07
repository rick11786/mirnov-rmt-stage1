"""Experiment 2: single coherent mode outlier and n recovery."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mirnov_rmt.mode_fit import estimate_n
from mirnov_rmt.pipeline import analyze_frequency
from mirnov_rmt.rmt import mp_edges
from mirnov_rmt.synthetic import generate_mirnov


ROOT = Path(__file__).resolve().parents[1]


def run(seed: int = 202) -> dict[str, float]:
    d, L, H, K, r = 48, 256, 256, 64, 30
    T = (K - 1) * H + L
    n_true, snr_db = 2, 0.0
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

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(np.arange(1, len(eigvals) + 1), eigvals, marker="o", markersize=3, linewidth=1)
    ax.axhline(threshold, color="#c03a2b", linestyle="--", label=f"MP edge={threshold:.2f}")
    ax.set_title("Single-mode coherence eigenvalue spectrum")
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue")
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
