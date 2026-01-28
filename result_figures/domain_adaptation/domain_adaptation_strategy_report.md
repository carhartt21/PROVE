# Domain Adaptation Strategy Comparison

*Generated: 2026-01-28 00:43*

## Overview

This analysis evaluates which training strategies produce models that generalize
best to the ACDC adverse weather benchmark.

**Total Results**: 64 configurations
**Strategies Evaluated**: 16
**Datasets**: idd-aw, bdd10k
**Models**: segformer_mit-b5, pspnet_r50

---

## Strategy Rankings (by Mean mIoU)

| Rank | Strategy | Mean mIoU | Δ Baseline | Family | N |
|------|----------|-----------|------------|--------|---|
| 1 | gen_stargan_v2 | 26.94% | +1.96% | Generative | 4 |
| 2 | photometric_distort | 26.87% | +1.88% | Photometric | 4 |
| 3 | gen_cycleGAN | 26.70% | +1.72% | Generative | 4 |
| 4 | std_autoaugment | 26.64% | +1.66% | Standard Aug | 4 |
| 5 | gen_step1x_v1p2 | 26.61% | +1.62% | Generative | 4 |
| 6 | gen_cyclediffusion | 26.59% | +1.60% | Generative | 4 |
| 7 | gen_UniControl | 26.57% | +1.58% | Generative | 4 |
| 8 | gen_albumentations_weather | 26.52% | +1.53% | Generative | 4 |
| 9 | gen_flux_kontext | 26.33% | +1.35% | Generative | 4 |
| 10 | std_mixup | 26.29% | +1.31% | Standard Aug | 4 |
| 11 | gen_TSIT | 26.27% | +1.28% | Generative | 4 |
| 12 | std_cutmix | 26.26% | +1.28% | Standard Aug | 4 |
| 13 | gen_step1x_new | 26.18% | +1.20% | Generative | 4 |
| 14 | std_randaugment | 26.04% | +1.06% | Standard Aug | 4 |
| 15 | gen_automold | 26.01% | +1.03% | Generative | 4 |
| 16 | baseline | 24.98% | — | Baseline | 4 |

---

## Per-Domain Performance

### By Strategy (Mean mIoU per Domain)

| Strategy |clear_day | foggy | night | rainy | snowy |
|----------|-----------|-----------|-----------|-----------|-----------|
| gen_stargan_v2 | 32.1% | 28.1% | 12.4% | 26.4% | 26.6% |
| photometric_distort | 32.5% | 28.1% | 12.6% | 26.5% | 26.2% |
| gen_cycleGAN | 31.8% | 28.3% | 13.0% | 26.6% | 26.7% |
| std_autoaugment | 32.9% | 27.8% | 11.2% | 26.6% | 25.6% |
| gen_step1x_v1p2 | 32.5% | 27.8% | 12.3% | 26.1% | 25.3% |
| gen_cyclediffusion | 32.0% | 28.3% | 12.5% | 26.2% | 25.8% |
| gen_UniControl | 32.2% | 28.0% | 12.1% | 26.1% | 25.9% |
| gen_albumentations_weather | 32.5% | 27.8% | 12.1% | 25.9% | 26.3% |
| gen_flux_kontext | 31.9% | 27.6% | 12.6% | 25.9% | 25.7% |
| std_mixup | 31.5% | 27.5% | 13.0% | 25.8% | 26.1% |

---

## Key Findings

1. **Best Strategy**: `gen_stargan_v2` with 26.94% mean mIoU
2. **Strategies Beating Baseline**: 15/15
3. **Best Strategy Family**: Photometric (26.87% mean)
4. **Most Challenging Domain**: night (12.3% mean)