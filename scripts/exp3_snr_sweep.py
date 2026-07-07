"""Experiment 3: SNR sweep and BBP-type transition."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mirnov_rmt.mode_fit import estimate_n, toroidal_templates
from mirnov_rmt.pipeline import analyze_frequency
from mirnov_rmt.rmt import mp_edges
from mirnov_rmt.surrogate import surrogate_threshold
from mirnov_rmt.synthetic import generate_mirnov


ROOT = Path(__file__).resolve().parents[1]


def _trial(d: int, L: int, H: int, K: int, r: int, snr_db: float | None, seed: int) -> dict[str, float]:
    T = (K - 1) * H + L
    n_true = 2
    omega = 2.0 * np.pi * r / L
    modes = [] if snr_db is None else [{"A": 1.0, "omega": omega, "n": n_true, "phase": 0.3}]
    Y, phi, _ = generate_mirnov(
        d=d,
        T=T,
        modes=modes,
        snr_db=snr_db,
        noise_sigma=1.0 if snr_db is None else None,
        seed=seed,
    )
    result = analyze_frequency(Y, L=L, H=H, r=r)
    eigvals = result["eigvals"]
    eigvec = result["eigvecs"][:, 0]
    n_hat, scores = estimate_n(eigvec, phi, np.arange(-10, 11))
    true_template = toroidal_templates(phi, [n_true])[n_true]
    alignment = float(abs(np.vdot(eigvec / np.linalg.norm(eigvec), true_template)) ** 2)
    return {
        "lambda_max": float(eigvals[0]),
        "mp_threshold": float(result["threshold"]),
        "mp_detected": bool(result["detected"]),
        "alignment": alignment,
        "n_hat": n_hat,
        "n_correct": n_hat == n_true,
        "score_true_n": float(scores[n_true]),
    }


def run(seed: int = 303, M: int = 60, surrogate_B: int = 40) -> dict[str, float]:
    d, L, H, K, r = 48, 256, 256, 64, 30
    snr_grid = np.array([-80, -70, -60, -50, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20], dtype=float)
    rng = np.random.default_rng(seed)

    # Calibrate one empirical null threshold and false-positive rate.
    T = (K - 1) * H + L
    Y_null, _, _ = generate_mirnov(d=d, T=T, modes=[], noise_sigma=1.0, seed=seed + 1)
    tau_sur, null_sur_lambdas = surrogate_threshold(
        Y_null, L=L, H=H, r=r, B=surrogate_B, alpha=0.05, seed=seed + 2
    )
    null_trials = [_trial(d, L, H, K, r, None, int(rng.integers(0, 2**31 - 1))) for _ in range(M)]
    mp_false_positive = float(np.mean([row["mp_detected"] for row in null_trials]))
    sur_false_positive = float(np.mean([row["lambda_max"] > tau_sur for row in null_trials]))

    rows = []
    raw_rows = []
    for snr_db in snr_grid:
        trials = [_trial(d, L, H, K, r, float(snr_db), int(rng.integers(0, 2**31 - 1))) for _ in range(M)]
        for trial in trials:
            raw_rows.append({"snr_db": snr_db, **trial})
        rows.append(
            {
                "snr_db": snr_db,
                "M": M,
                "mp_detection_probability": float(np.mean([t["mp_detected"] for t in trials])),
                "sur_detection_probability": float(np.mean([t["lambda_max"] > tau_sur for t in trials])),
                "mean_lambda_max": float(np.mean([t["lambda_max"] for t in trials])),
                "mean_lambda_minus_mp": float(np.mean([t["lambda_max"] - t["mp_threshold"] for t in trials])),
                "mean_alignment": float(np.mean([t["alignment"] for t in trials])),
                "n_accuracy": float(np.mean([t["n_correct"] for t in trials])),
                "mp_false_positive": mp_false_positive,
                "sur_false_positive": sur_false_positive,
                "surrogate_threshold": tau_sur,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "exp3_snr_sweep.csv", index=False)
    pd.DataFrame(raw_rows).to_csv(ROOT / "results" / "exp3_snr_sweep_raw.csv", index=False)
    pd.DataFrame({"lambda_max": null_sur_lambdas}).to_csv(
        ROOT / "results" / "exp3_surrogate_null_lambdas.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(df["snr_db"], df["mp_detection_probability"], marker="o", label="MP edge")
    ax.plot(df["snr_db"], df["sur_detection_probability"], marker="s", label="surrogate")
    ax.axhline(mp_false_positive, color="#5f6673", linestyle=":", label="MP noise-only FPR")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Detection probability")
    ax.set_title("SNR sweep detection")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig3_snr_detection.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(df["snr_db"], df["mean_alignment"], marker="o", label="alignment")
    ax.plot(df["snr_db"], df["n_accuracy"], marker="s", label="n accuracy")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Mean score / probability")
    ax.set_title("Eigenvector alignment and n recovery vs SNR")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig3_alignment_n_accuracy.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(df["snr_db"], df["mean_lambda_minus_mp"], marker="o")
    ax.axhline(0.0, color="#c03a2b", linestyle="--")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(r"Mean $\lambda_{max}-\lambda_+$")
    ax.set_title("Outlier margin vs SNR")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig3_lambda_margin.png", dpi=160)
    plt.close(fig)

    return {
        "mp_false_positive": mp_false_positive,
        "sur_false_positive": sur_false_positive,
        "tau_sur": tau_sur,
        "M": M,
    }


if __name__ == "__main__":
    print(run())
