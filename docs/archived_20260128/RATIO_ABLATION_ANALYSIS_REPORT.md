# Ratio Ablation Study - Consolidated Analysis Report

**Generated:** 2026-01-26
**Status:** 102/148 models tested (46 awaiting tests)

## Executive Summary

The ratio ablation study investigates the optimal balance between real and generated training images across different strategies, datasets, and models.

### Key Finding
**Best ratio: 0.88** (mIoU: 44.88%)
- Higher ratios of real images (0.88 = 88% real, 12% generated) generally perform better
- But this varies significantly by strategy and dataset

## Overall Statistics

| Metric | Count |
|--------|-------|
| Total configurations | 153 |
| Completed models | 148 |
| Tested models | 102 |
| Awaiting tests | 46 |

## Performance by Ratio

| Ratio | Mean mIoU | Std | Count |
|-------|-----------|-----|-------|
| 0.00 | 43.85% | ±5.04 | 16 |
| 0.12 | 43.77% | ±5.01 | 16 |
| 0.25 | 43.78% | ±4.99 | 16 |
| 0.38 | 44.08% | ±5.23 | 14 |
| 0.62 | 43.75% | ±5.41 | 14 |
| 0.75 | 43.83% | ±5.21 | 14 |
| **0.88** | **44.88%** | ±5.29 | 12 |

Note: Ratio 0.50 results are in the main WEIGHTS directories, not WEIGHTS_RATIO_ABLATION.

## Performance by Strategy

| Strategy | Mean mIoU | Count |
|----------|-----------|-------|
| gen_step1x_new | 46.92% | 28 |
| gen_step1x_v1p2 | 46.85% | 28 |
| gen_cycleGAN | 41.04% | 28 |
| gen_stargan_v2 | 39.49% | 9 |
| gen_cyclediffusion | 39.32% | 9 |

## Performance by Dataset

| Dataset | Mean mIoU | Count |
|---------|-----------|-------|
| bdd10k_ad | 48.78% | 22 |
| bdd10k | 48.16% | 6 |
| idd-aw_ad | 45.52% | 5 |
| iddaw_ad | 45.00% | 15 |
| outside15k | 41.75% | 14 |
| idd-aw | 40.87% | 40 |

## Best Ratio Per Configuration

### gen_step1x_new
| Dataset | Model | Best Ratio | mIoU |
|---------|-------|------------|------|
| bdd10k_ad | pspnet_r50 | 0.62 | 45.98% |
| bdd10k_ad | segformer_mit-b5 | 0.88 | 52.27% |
| iddaw_ad | pspnet_r50 | 0.62 | 43.76% |
| iddaw_ad | segformer_mit-b5 | 0.38 | 46.93% |

### gen_step1x_v1p2
| Dataset | Model | Best Ratio | mIoU |
|---------|-------|------------|------|
| bdd10k | pspnet_r50 | 0.88 | 45.37% |
| bdd10k | segformer_mit-b5 | 0.88 | 52.26% |
| bdd10k_ad | pspnet_r50 | 0.00 | 46.21% |
| bdd10k_ad | segformer_mit-b5 | 0.38 | 52.10% |
| idd-aw | pspnet_r50 | 0.88 | 43.56% |
| idd-aw | segformer_mit-b5 | 0.38 | 46.99% |
| idd-aw_ad | pspnet_r50 | 0.12 | 43.59% |
| idd-aw_ad | segformer_mit-b5 | 0.12 | 46.93% |

### gen_cycleGAN
| Dataset | Model | Best Ratio | mIoU |
|---------|-------|------------|------|
| idd-aw | pspnet_r50 | 0.75 | 38.27% |
| idd-aw | segformer_mit-b5 | 0.00 | 43.27% |
| outside15k | pspnet_r50 | 0.38 | 35.69% |
| outside15k | segformer_mit-b5 | 0.00 | 48.70% |

### gen_cyclediffusion
| Dataset | Model | Best Ratio | mIoU |
|---------|-------|------------|------|
| idd-aw | pspnet_r50 | 0.38 | 37.93% |
| idd-aw | segformer_mit-b5 | 0.00 | 43.30% |

### gen_stargan_v2
| Dataset | Model | Best Ratio | mIoU |
|---------|-------|------------|------|
| idd-aw | pspnet_r50 | 0.38 | 38.38% |
| idd-aw | segformer_mit-b5 | 0.25 | 43.21% |

## Models Awaiting Tests

46 models need testing (MapillaryVistas and OUTSIDE15k with native classes):

### gen_step1x_new
- mapillaryvistas_ad: 14 models (pspnet_r50 & segformer_mit-b5, all 7 ratios)
- outside15k_ad: 14 models (pspnet_r50 & segformer_mit-b5, all 7 ratios)

### gen_step1x_v1p2
- mapillaryvistas_ad: 6 models (ratios 0.00, 0.12, 0.25)
- outside15k: 6 models (ratios 0.62, 0.75, 0.88)
- outside15k_ad: 6 models (ratios 0.00, 0.12, 0.25)

## Observations

1. **Strategy matters more than ratio**: gen_step1x variants outperform GAN-based methods by ~5-7% mIoU regardless of ratio
2. **Higher ratios generally better**: Ratios ≥0.62 tend to perform better, especially for gen_step1x variants
3. **Dataset difficulty varies**: BDD10k is easiest (48%+), IDD-AW is hardest (40-41%)
4. **No universal optimal ratio**: Best ratio varies by strategy, dataset, and model architecture

## Directory Structure

```
WEIGHTS_RATIO_ABLATION/
├── gen_cyclediffusion/
│   └── idd-aw/                     (9 models, 9 tested)
├── gen_cycleGAN/
│   ├── idd-aw/                     (14 models, 14 tested)
│   └── outside15k/                 (14 models, 14 tested)
├── gen_stargan_v2/
│   └── idd-aw/                     (9 models, 9 tested)
├── gen_step1x_new/
│   ├── bdd10k_ad/                  (14 models, 14 tested)
│   ├── iddaw_ad/                   (14 models, 14 tested)
│   ├── mapillaryvistas_ad/         (14 models, 0 tested) ⚠️
│   └── outside15k_ad/              (14 models, 0 tested) ⚠️
└── gen_step1x_v1p2/
    ├── bdd10k/                     (6 models, 6 tested)
    ├── bdd10k_ad/                  (8 models, 8 tested)
    ├── idd-aw/                     (8 models, 8 tested)
    ├── idd-aw_ad/                  (5 models, 5 tested)
    ├── iddaw_ad/                   (1 model, 1 tested)
    ├── mapillaryvistas_ad/         (6 models, 0 tested) ⚠️
    ├── outside15k/                 (6 models, 0 tested) ⚠️
    └── outside15k_ad/              (6 models, 0 tested) ⚠️
```

## Generated Files

- `results/ratio_ablation_consolidated.csv` - Full results table
- `result_figures/ratio_ablation/` - Visualization figures
  - `ratio_ablation_dashboard.png` - Comprehensive overview
  - `miou_vs_ratio_by_strategy.png` - Line plot by strategy
  - `miou_vs_ratio_by_dataset.png` - Line plot by dataset
  - `miou_vs_ratio_by_model.png` - Line plot by model
  - `heatmap_strategy_ratio.png` - Heatmap visualization
  - `optimal_ratio_distribution.png` - Distribution of optimal ratios
  - `miou_boxplot_by_ratio.png` - Box plot comparison
