# Synthetic Mirnov RMT Stage 1

Small isolated project for synthetic Mirnov array experiments with spectral
coherence outlier detection.

This project is intentionally separate from the main MCF/topoquest code and
uses no real DIII-D data.

## Working location

The runnable project lives in WSL:

```text
/home/wenyin/local_side_projects/mirnov_rmt_stage1
```

A Windows-readable mirror/report location is:

```text
C:\Users\28105\Documents\mirnov_rmt_stage1
```

## Planned outputs

- `figures/*.png`: experiment figures
- `results/*.csv`: numerical summaries
- `report.html`: browser-readable local report

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## First checks

```bash
pytest
```
