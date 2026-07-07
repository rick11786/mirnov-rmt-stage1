# Synthetic Mirnov RMT Stage 1

English | [中文](#中文说明)

## Overview

This is a small, isolated research-code project for synthetic Mirnov array
experiments with spectral coherence outlier detection.

The project starts from synthetic rotating Mirnov-like time series, builds
frequency-local Fourier snapshots, constructs cross-spectral and spectral
coherence matrices, and tests whether coherent spatial modes appear as
random-matrix outlier eigenvalues.

It deliberately does **not** use real DIII-D data and does **not** attempt to
reproduce a full SSI pipeline.

## Research Questions

1. Under a noise-only null, do coherence eigenvalues roughly follow the
   Marchenko-Pastur bulk?
2. With one coherent rotating mode, does the largest eigenvalue leave the bulk?
3. As SNR decreases, does the outlier disappear into the bulk?
4. How does the aspect ratio `d/K` affect detection difficulty?
5. Can the leading outlier eigenvector recover a candidate toroidal mode number?

## Repository Policy

This repository is meant to keep code, lightweight tests, and documentation.

Generated figures and numerical outputs are **not committed by default**:

- `figures/*.png` is ignored
- `results/*.csv` is ignored
- only `figures/.gitkeep` and `results/.gitkeep` are tracked

This keeps the GitHub repository small. Final figures can be reviewed locally
through `report.html` or selectively committed later if needed for a paper,
release, or reproducibility snapshot.

## Local Locations

Runnable WSL project:

```text
/home/wenyin/local_side_projects/mirnov_rmt_stage1
```

Windows-readable viewing folder:

```text
C:\Users\28105\Documents\mirnov_rmt_stage1
```

Recommended Windows browser path for live WSL outputs:

```text
\\wsl$\Ubuntu\home\wenyin\local_side_projects\mirnov_rmt_stage1\report.html
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## First Checks

```bash
pytest
```

## Planned Outputs

- `report.html`: browser-readable local report
- `figures/*.png`: experiment figures, ignored by git
- `results/*.csv`: numerical summaries, ignored by git

---

# 中文说明

## 项目概述

这是一个独立的小型研究代码项目，用于做合成 Mirnov 阵列和 spectral
coherence 随机矩阵 outlier 检验。

项目从合成的 rotating Mirnov-like 时间序列出发，生成频率局部的 Fourier
snapshots，构造 cross-spectral matrix 和 spectral coherence matrix，然后检验
跨阵列相干模态是否表现为随机矩阵谱中的 outlier eigenvalue。

本项目刻意不使用真实 DIII-D 数据，也不复现完整 SSI 流程。

## 研究问题

1. 在纯噪声 null 下，coherence 特征值 bulk 是否大致符合 Marchenko-Pastur？
2. 加入一个 coherent rotating mode 后，最大特征值是否脱离 bulk？
3. 降低 SNR 时，outlier 是否被 bulk 吞掉？
4. `d/K` 的高维比例如何影响检测难度？
5. 最大 outlier eigenvector 是否能恢复候选 toroidal mode number？

## 仓库策略

这个 GitHub 仓库主要保存代码、轻量测试和文档。

默认不提交生成结果：

- `figures/*.png` 被忽略
- `results/*.csv` 被忽略
- 只跟踪 `figures/.gitkeep` 和 `results/.gitkeep`

这样可以避免仓库因为实验图片和大量 CSV 变得过大。最终图片可以通过本地
`report.html` 查看；如果后续需要论文、release 或可复现实验快照，再选择性提交。

## 本地位置

可运行的 WSL 项目：

```text
/home/wenyin/local_side_projects/mirnov_rmt_stage1
```

Windows 下可查看的文件夹：

```text
C:\Users\28105\Documents\mirnov_rmt_stage1
```

推荐在 Windows 浏览器中直接打开 WSL 里的实时报告：

```text
\\wsl$\Ubuntu\home\wenyin\local_side_projects\mirnov_rmt_stage1\report.html
```

## 环境配置

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 基础检查

```bash
pytest
```

## 计划产物

- `report.html`：本地浏览器可读报告
- `figures/*.png`：实验图片，默认不进 git
- `results/*.csv`：数值结果，默认不进 git
