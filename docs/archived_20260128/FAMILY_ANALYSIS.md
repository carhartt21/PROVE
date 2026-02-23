# Strategy Family Analysis

This document describes the strategy family analysis scripts and their outputs.

## Overview

The PROVE project evaluates multiple data augmentation strategies for semantic segmentation. These strategies are grouped into **families** based on their underlying approach:

| Family | Description | Example Strategies |
|--------|-------------|-------------------|
| **2D Rendering** | Rule-based weather effects | gen_automold, gen_imgaug_weather, gen_Weather_Effect_Generator |
| **CNN/GAN** | GAN-based image translation | gen_CUT, gen_cycleGAN, gen_stargan_v2, gen_SUSTechGAN |
| **Style Transfer** | Neural style transfer methods | gen_NST, gen_LANIT, gen_TSIT |
| **Diffusion** | Standard diffusion models | gen_Img2Img, gen_IP2P, gen_UniControl |
| **Multimodal Diffusion** | Large multimodal diffusion | gen_flux1_kontext, gen_step1x_new, gen_Qwen_Image_Edit |
| **Standard Augmentation** | Traditional augmentation | std_autoaugment, std_randaugment, std_std_photometric_distort |
| **Standard Mixing** | Sample mixing methods | std_cutmix, std_mixup |
| **Baseline** | No augmentation | baseline |

> **Note**: StyleID and EDICT have been excluded due to 0/4 training dataset coverage (only have ACDC/BDD100k images).

## Scripts

### 1. `analyze_strategy_families.py`

Main family analysis script for single (non-combined) strategies.

**Usage:**
```bash
mamba run -n prove python analyze_strategy_families.py
```

**Output Directory:** `result_figures/family_analysis/`

**Generated Files:**
- `family_summary.csv` - Summary statistics per family
- `family_analysis_report.txt` - Detailed text report
- `family_publication_summary.png` - Publication-ready overview figure
- `inter_family_comparison.png` - Comparison between families
- `intra_*.png` - Comparison within each family

### 2. `analyze_family_domains.py`

Analyzes family performance across weather domains (requires per-domain test results).

**Usage:**
```bash
mamba run -n prove python analyze_family_domains.py
```

**Output Directory:** `result_figures/family_analysis/`

**Generated Files:**
- `family_domain_heatmap.png` - Family × Domain performance heatmap
- `family_robustness_analysis.png` - Adverse vs normal weather analysis
- `family_domain_summary.csv` - Domain-level statistics
- `intra_*_domains.png` - Per-family domain analysis

### 3. `analyze_combination_ablation.py`

Dedicated analysis for combination strategies (moved to `WEIGHTS_COMBINATIONS`).

**Usage:**
```bash
mamba run -n prove python analyze_combination_ablation.py
```

**Output Directory:** `result_figures/combination_ablation/`

**Generated Files:**
- `combination_overview.png` - Performance overview
- `combination_ablation_summary.csv` - Summary statistics
- `combination_results.csv` - Raw results
- `combination_ablation_report.txt` - Detailed report
- `synergy_analysis.png` - Synergy effects (if component data available)
- `component_interaction.png` - Component interaction heatmap

## Data Directories

| Directory | Content |
|-----------|---------|
| `${AWARE_DATA_ROOT}/WEIGHTS/` | Single strategy results (25 strategies) |
| `${AWARE_DATA_ROOT}/WEIGHTS_COMBINATIONS/` | Combined strategy results (13 combinations) |
| `${AWARE_DATA_ROOT}/WEIGHTS_EXTENDED/` | Extended training results |
| `${AWARE_DATA_ROOT}/WEIGHTS_RATIO_ABLATION/` | Ratio ablation results |

## Key Findings

### Family Performance Ranking

| Rank | Family | Mean mIoU | Best Strategy | Improvement |
|------|--------|-----------|---------------|-------------|
| 1 | Style Transfer | 55.55 | gen_LANIT | +0.59 |
| 2 | Diffusion | 55.34 | gen_UniControl | +0.37 |
| 3 | Standard Augmentation | 55.29 | std_randaugment | +0.33 |
| 4 | CNN/GAN | 55.24 | gen_CUT | +0.28 |
| 5 | Standard Mixing | 55.21 | std_mixup | +0.25 |
| 6 | Baseline | 54.96 | - | - |
| 7 | Multimodal Diffusion | 53.64 | gen_step1x_new | +0.46 |
| 8 | 2D Rendering | 53.61 | gen_automold | +0.54 |

### Combination Ablation

| Combination Type | Best Combination | mIoU |
|------------------|------------------|------|
| Standard + Standard | std_randaugment+std_mixup | 56.06 |
| Generative + Standard | gen_CUT+std_mixup | 55.27 |

## Requirements

- Python environment: `mamba activate prove`
- Input: `downstream_results.csv` in PROVE repository root
- Output: `result_figures/family_analysis/` and `result_figures/combination_ablation/`

## Related Documentation

- [UNIFIED_TRAINING.md](docs/UNIFIED_TRAINING.md) - Training procedures
- [UNIFIED_TESTING.md](docs/UNIFIED_TESTING.md) - Testing procedures
- [RESULT_VISUALIZATION.md](docs/RESULT_VISUALIZATION.md) - Visualization options
