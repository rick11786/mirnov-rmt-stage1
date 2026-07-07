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


def _glossary() -> str:
    rows = [
        ("d", "Number of Mirnov channels / sensors in the synthetic toroidal array."),
        ("K", "Number of Welch Fourier snapshots used to estimate the spectral matrix."),
        ("q=d/K", "High-dimensional aspect ratio controlling the MP bulk width."),
        ("n_true", "The toroidal mode number used to generate the synthetic rotating mode."),
        ("n_hat", "The toroidal mode number estimated from the leading outlier eigenvector."),
        ("lambda_max", "Largest eigenvalue of the spectral coherence matrix at the target frequency."),
        ("MP edge", "Analytic upper edge lambda_+=(1+sqrt(d/K))^2 under the white-noise null."),
        ("outlier", "An eigenvalue above the chosen threshold; statistically, a coherent spatial direction."),
        ("score(n)", "Squared alignment |v_hat^* v_n|^2 between an eigenvector and toroidal template n."),
        ("alignment_true_n", "score(n_true); close to 1 means the eigenvector recovers the injected mode shape."),
    ]
    return pd.DataFrame(rows, columns=["Term", "Meaning"]).to_html(index=False)


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
            "Glossary",
            "<p>The plots and CSV files use the following notation.</p>" + _glossary(),
            [],
        ),
        (
            "Experiment 1: Noise-only MP Bulk",
            "<p>Noise-only spectral coherence eigenvalues are compared with the "
            "Marchenko-Pastur support. The blue solid curve is a Monte Carlo "
            "noise-only density estimate from repeated simulations; the orange "
            "dashed curve is the analytic MP density, plotted only on its support. "
            "The histogram shows one representative realization.</p>"
            + _csv_table(ROOT / "results" / "exp1_noise_mp_bulk.csv"),
            ["figures/fig1_noise_mp_bulk.png"],
        ),
        (
            "Experiment 2: Single-mode Outlier",
            "<p>A weak coherent rotating mode is injected at the target Fourier bin "
            "using <code>SNR=-30 dB</code>, so the MP bulk and the spike remain visible "
            "on the same axis. Red vertical markers label all eigenvalues above the "
            "MP edge. The leading outlier eigenvector is then matched against "
            "toroidal templates; the score plot shows <code>score(n)=|v_hat^*v_n|^2</code>.</p>"
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
