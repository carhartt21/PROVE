# Stage 2 Strategy Leaderboard

**Training:** All domains (not filtered to clear_day)
**Last Updated:** 2026-01-18 01:15
**Baseline mIoU:** 43.74%
**Total Test Results:** 150
**Strategies Evaluated:** 19

## Overall Rankings

| Rank | Strategy | Type | mIoU | Gain | Normal | Adverse | Gap | Models |
|------|----------|------|------|------|--------|---------|-----|--------|
| 1 | gen_Weather_Effect_Generator | Generative | 50.78 | +7.03 | 50.94 | 51.02 | -0.09 | 3 |
| 2 | gen_VisualCloze | Generative | 45.68 | +1.93 | 46.16 | 46.71 | -0.55 | 7 |
| 3 | gen_CUT | Generative | 44.47 | +0.73 | 45.17 | 44.18 | 0.99 | 9 |
| 4 | gen_Img2Img | Generative | 44.24 | +0.49 | 45.08 | 42.93 | 2.15 | 6 |
| 5 | gen_CNetSeg | Generative | 44.08 | +0.34 | 44.89 | 43.39 | 1.50 | 10 |
| 6 | gen_IP2P | Generative | 44.02 | +0.28 | 44.82 | 43.27 | 1.55 | 8 |
| 7 | gen_albumentations_weather | Generative | 43.89 | +0.14 | 44.86 | 43.16 | 1.70 | 10 |
| 8 | gen_flux_kontext | Generative | 43.89 | +0.14 | 44.70 | 43.33 | 1.37 | 10 |
| 9 | baseline | Baseline | 43.74 | - | 44.42 | 43.93 | 0.48 | 8 |
| 10 | gen_automold | Generative | 43.71 | -0.04 | 44.59 | 43.14 | 1.45 | 10 |
| 11 | gen_augmenters | Generative | 43.69 | -0.05 | 44.48 | 43.16 | 1.32 | 9 |
| 12 | gen_LANIT | Generative | 43.59 | -0.16 | 44.52 | 41.49 | 3.03 | 5 |
| 13 | gen_cycleGAN | Generative | 43.56 | -0.19 | 44.51 | 42.46 | 2.04 | 9 |
| 14 | gen_Attribute_Hallucination | Generative | 43.54 | -0.20 | 44.37 | 42.82 | 1.55 | 8 |
| 15 | gen_SUSTechGAN | Generative | 43.13 | -0.61 | 43.91 | 42.66 | 1.25 | 9 |
| 16 | gen_UniControl | Generative | 42.87 | -0.88 | 43.70 | 41.96 | 1.74 | 7 |
| 17 | gen_Qwen_Image_Edit | Generative | 42.56 | -1.18 | 43.41 | 42.22 | 1.19 | 9 |
| 18 | gen_step1x_v1p2 | Generative | 42.49 | -1.26 | 43.13 | 41.55 | 1.58 | 5 |
| 19 | gen_TSIT | Generative | 42.48 | -1.26 | 43.53 | 41.66 | 1.87 | 8 |

## Per-Dataset Breakdown

### bdd10k

| Strategy | mIoU | Models |
|----------|------|--------|
| gen_Weather_Effect_Generator | 51.87 | 1 |
| gen_UniControl | 45.25 | 1 |
| gen_cycleGAN | 44.30 | 2 |
| gen_CNetSeg | 44.19 | 2 |
| gen_augmenters | 43.92 | 2 |
| gen_automold | 43.81 | 2 |
| gen_TSIT | 43.50 | 2 |
| gen_flux_kontext | 43.33 | 2 |
| gen_SUSTechGAN | 43.26 | 2 |
| gen_Qwen_Image_Edit | 43.23 | 2 |
| gen_albumentations_weather | 43.09 | 2 |
| gen_VisualCloze | 42.97 | 2 |

### idd-aw

| Strategy | mIoU | Models |
|----------|------|--------|
| gen_CUT | 44.92 | 3 |
| baseline | 44.49 | 3 |
| gen_automold | 43.69 | 2 |
| gen_augmenters | 43.68 | 2 |
| gen_Attribute_Hallucination | 43.67 | 2 |
| gen_SUSTechGAN | 43.65 | 2 |
| gen_Qwen_Image_Edit | 43.62 | 2 |
| gen_VisualCloze | 43.60 | 2 |
| gen_flux_kontext | 43.59 | 2 |
| gen_albumentations_weather | 43.57 | 2 |
| gen_CNetSeg | 43.57 | 2 |
| gen_IP2P | 43.56 | 2 |
| gen_cycleGAN | 43.53 | 2 |
| gen_UniControl | 43.51 | 2 |
| gen_TSIT | 43.47 | 2 |

### mapillaryvistas

| Strategy | mIoU | Models |
|----------|------|--------|
| gen_Weather_Effect_Generator | 50.23 | 2 |
| gen_LANIT | 50.19 | 2 |
| gen_augmenters | 50.02 | 2 |
| gen_UniControl | 49.97 | 2 |
| gen_TSIT | 49.78 | 2 |
| gen_cycleGAN | 49.76 | 2 |
| baseline | 49.72 | 2 |
| gen_CUT | 49.31 | 3 |
| gen_flux_kontext | 49.29 | 3 |
| gen_Img2Img | 49.05 | 3 |
| gen_IP2P | 48.99 | 3 |
| gen_albumentations_weather | 48.97 | 3 |
| gen_Qwen_Image_Edit | 48.92 | 3 |
| gen_VisualCloze | 48.87 | 3 |
| gen_SUSTechGAN | 48.86 | 3 |
| gen_step1x_v1p2 | 48.85 | 3 |
| gen_CNetSeg | 48.76 | 3 |
| gen_automold | 48.65 | 3 |
| gen_Attribute_Hallucination | 48.13 | 3 |

### outside15k

| Strategy | mIoU | Models |
|----------|------|--------|
| gen_CNetSeg | 39.67 | 3 |
| gen_albumentations_weather | 39.55 | 3 |
| gen_Img2Img | 39.43 | 3 |
| gen_IP2P | 39.36 | 3 |
| gen_augmenters | 39.32 | 3 |
| gen_CUT | 39.19 | 3 |
| gen_LANIT | 39.18 | 3 |
| gen_flux_kontext | 39.05 | 3 |
| baseline | 39.02 | 3 |
| gen_cycleGAN | 38.94 | 3 |
| gen_Attribute_Hallucination | 38.86 | 3 |
| gen_automold | 38.70 | 3 |
| gen_UniControl | 33.92 | 2 |
| gen_SUSTechGAN | 33.90 | 2 |
| gen_TSIT | 33.18 | 2 |
| gen_step1x_v1p2 | 32.94 | 2 |
| gen_Qwen_Image_Edit | 31.30 | 2 |
