"""Run all stage-one experiments and rebuild the local HTML report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import exp1_noise_mp_bulk
import exp2_single_mode_outlier
import exp3_snr_sweep
import exp4_dk_sweep
import exp5_two_mode
from mirnov_rmt.plotting import write_report


ROOT = Path(__file__).resolve().parents[1]


def _csv_table(path: Path, max_rows: int = 12) -> str:
    if not path.exists():
        return f"<p>Missing <code>{path.name}</code>.</p>"
    df = pd.read_csv(path)
    return df.head(max_rows).to_html(index=False, float_format=lambda x: f"{x:.4g}")


def _method_table() -> str:
    rows = [
        ("FFT peak", "frequency energy", "noise Monte Carlo / surrogate", "peak frequency", "no spatial coherence"),
        ("SVD/BOD", "time-domain energy components", "singular threshold", "rank / spatial PCs", "PCs not necessarily modes"),
        ("MP edge", "spectral coherence outlier", "analytic MP edge", "rank / eigenvector / n", "idealized null"),
        ("MP + surrogate", "spectral coherence outlier", "empirical null", "rank / eigenvector / n", "more compute"),
    ]
    df = pd.DataFrame(rows, columns=["Method", "Detection target", "Threshold", "Output", "Limitation"])
    return df.to_html(index=False)


def rebuild_report(summaries: dict[str, dict[str, object]]) -> None:
    sections = [
        (
            "Summary",
            "<p>Stage one has been executed on purely synthetic Mirnov-like data. "
            "Generated PNG/CSV files are intentionally ignored by git and are meant "
            "for local inspection.</p>"
            f"<pre>{summaries}</pre>",
            [],
        ),
        (
            "Experiment 1: Noise-only MP Bulk",
            "<p>Noise-only spectral coherence eigenvalues are compared with the "
            "Marchenko-Pastur support.</p>" + _csv_table(ROOT / "results" / "exp1_noise_mp_bulk.csv"),
            ["figures/fig1_noise_mp_bulk.png"],
        ),
        (
            "Experiment 2: Single-mode Outlier",
            "<p>A coherent rotating mode is injected at the target Fourier bin. "
            "The leading eigenvector is matched against toroidal templates.</p>"
            + _csv_table(ROOT / "results" / "exp2_single_mode_outlier.csv"),
            ["figures/fig2_single_outlier_eigs.png", "figures/fig2_n_scores.png"],
        ),
        (
            "Experiment 3: SNR Sweep",
            "<p>Detection probability, outlier margin, eigenvector alignment and "
            "n recovery are evaluated over SNR.</p>" + _csv_table(ROOT / "results" / "exp3_snr_sweep.csv"),
            [
                "figures/fig3_snr_detection.png",
                "figures/fig3_alignment_n_accuracy.png",
                "figures/fig3_lambda_margin.png",
            ],
        ),
        (
            "Experiment 4: d/K Sweep",
            "<p>The aspect ratio q=d/K changes the MP edge and therefore the "
            "detectability of weak coherent modes.</p>" + _csv_table(ROOT / "results" / "exp4_dk_sweep.csv", max_rows=25),
            ["figures/fig4_dk_detection_heatmap.png", "figures/fig4_dk_n_accuracy_heatmap.png"],
        ),
        (
            "Experiment 5: Two-mode Frequency Separation",
            "<p>Two close rotating modes are compared using FFT average power, "
            "per-bin coherence rank and a band-averaged coherence matrix.</p>"
            + _csv_table(ROOT / "results" / "exp5_two_mode_band.csv"),
            ["figures/fig5_two_mode_frequency_separation.png", "figures/fig5_band_averaged_outliers.png"],
        ),
        ("Method Comparison", _method_table(), []),
    ]
    write_report(ROOT / "report.html", "Synthetic Mirnov RMT Stage 1 Report", sections)


def main() -> None:
    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    summaries = {
        "exp1": exp1_noise_mp_bulk.run(),
        "exp2": exp2_single_mode_outlier.run(),
        "exp3": exp3_snr_sweep.run(),
        "exp4": exp4_dk_sweep.run(),
        "exp5": exp5_two_mode.run(),
    }
    rebuild_report(summaries)
    print("Completed stage-one experiments.")
    print(f"Report: {ROOT / 'report.html'}")


if __name__ == "__main__":
    main()
