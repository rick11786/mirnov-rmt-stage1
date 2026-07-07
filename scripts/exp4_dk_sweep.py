"""Experiment 4: d/K aspect-ratio sweep."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mirnov_rmt.mode_fit import estimate_n
from mirnov_rmt.pipeline import analyze_frequency
from mirnov_rmt.synthetic import generate_mirnov


ROOT = Path(__file__).resolve().parents[1]


def run(seed: int = 404, M: int = 40) -> dict[str, float]:
    d_list = [16, 32, 48, 64]
    K_list = [8, 16, 32, 64, 128]
    L, H, r, snr_db, n_true = 256, 256, 30, 0.0, 2
    omega = 2.0 * np.pi * r / L
    rng = np.random.default_rng(seed)
    rows = []
    for d in d_list:
        for K in K_list:
            detections = []
            n_correct = []
            ranks = []
            for _ in range(M):
                T = (K - 1) * H + L
                Y, phi, _ = generate_mirnov(
                    d=d,
                    T=T,
                    modes=[{"A": 1.0, "omega": omega, "n": n_true, "phase": 0.2}],
                    snr_db=snr_db,
                    seed=int(rng.integers(0, 2**31 - 1)),
                )
                result = analyze_frequency(Y, L=L, H=H, r=r)
                n_hat, _ = estimate_n(result["eigvecs"][:, 0], phi, np.arange(-min(10, d // 2), min(10, d // 2) + 1))
                detections.append(bool(result["detected"]))
                n_correct.append(n_hat == n_true)
                ranks.append(int(result["estimated_rank"]))
            rows.append(
                {
                    "d": d,
                    "K": K,
                    "q": d / K,
                    "M": M,
                    "snr_db": snr_db,
                    "mp_detection_probability": float(np.mean(detections)),
                    "n_accuracy": float(np.mean(n_correct)),
                    "mean_estimated_rank": float(np.mean(ranks)),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "exp4_dk_sweep.csv", index=False)

    for metric, filename, title in [
        ("mp_detection_probability", "fig4_dk_detection_heatmap.png", "MP detection probability"),
        ("n_accuracy", "fig4_dk_n_accuracy_heatmap.png", "n recovery accuracy"),
    ]:
        pivot = df.pivot(index="d", columns="K", values=metric)
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        image = ax.imshow(pivot.values, vmin=0, vmax=1, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
        ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
        ax.set_xlabel("K snapshots")
        ax.set_ylabel("d channels")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, label=metric)
        fig.tight_layout()
        fig.savefig(ROOT / "figures" / filename, dpi=160)
        plt.close(fig)

    return {
        "M": M,
        "mean_detection": float(df["mp_detection_probability"].mean()),
        "mean_n_accuracy": float(df["n_accuracy"].mean()),
    }


if __name__ == "__main__":
    print(run())
