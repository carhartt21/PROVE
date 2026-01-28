# Stage 2 Strategy Leaderboard

**Training:** All domains (not filtered to clear_day)
**Last Updated:** 2026-01-27 15:25
**Baseline mIoU:** 41.35%
**Total Test Results:** 324
**Strategies Evaluated:** 27

## Overall Rankings

| Rank | Strategy | Type | mIoU | Gain | Normal | Adverse | Gap | Models |
|------|----------|------|------|------|--------|---------|-----|--------|
| 1 | gen_stargan_v2 | Generative | 41.73 | +0.38 | 42.37 | 41.16 | 1.21 | 12 |
| 2 | gen_UniControl | Generative | 41.70 | +0.35 | 42.37 | 40.95 | 1.41 | 12 |
| 3 | gen_CNetSeg | Generative | 41.69 | +0.34 | 42.26 | 40.97 | 1.29 | 12 |
| 4 | gen_VisualCloze | Generative | 41.67 | +0.32 | 42.26 | 41.04 | 1.21 | 12 |
| 5 | gen_cycleGAN | Generative | 41.64 | +0.29 | 42.26 | 40.81 | 1.45 | 12 |
| 6 | gen_Attribute_Hallucination | Generative | 41.61 | +0.26 | 42.25 | 40.87 | 1.37 | 12 |
| 7 | gen_cyclediffusion | Generative | 41.59 | +0.24 | 42.33 | 40.86 | 1.47 | 12 |
| 8 | gen_CUT | Generative | 41.58 | +0.23 | 42.15 | 41.03 | 1.13 | 12 |
| 9 | gen_augmenters | Generative | 41.58 | +0.23 | 42.09 | 41.01 | 1.08 | 12 |
| 10 | gen_SUSTechGAN | Generative | 41.57 | +0.22 | 42.18 | 40.81 | 1.36 | 12 |
| 11 | gen_LANIT | Generative | 41.55 | +0.19 | 42.20 | 40.77 | 1.43 | 12 |
| 12 | gen_TSIT | Generative | 41.53 | +0.18 | 42.14 | 40.73 | 1.40 | 12 |
| 13 | gen_step1x_v1p2 | Generative | 41.50 | +0.15 | 42.08 | 40.74 | 1.35 | 12 |
| 14 | gen_albumentations_weather | Generative | 41.48 | +0.13 | 42.16 | 40.76 | 1.41 | 12 |
| 15 | gen_Img2Img | Generative | 41.48 | +0.13 | 42.08 | 40.81 | 1.26 | 12 |
| 16 | gen_flux_kontext | Generative | 41.42 | +0.07 | 42.09 | 40.91 | 1.18 | 12 |
| 17 | std_autoaugment | Standard Aug | 41.38 | +0.03 | 41.97 | 40.87 | 1.10 | 12 |
| 18 | gen_automold | Generative | 41.38 | +0.03 | 42.04 | 40.64 | 1.40 | 12 |
| 19 | gen_IP2P | Generative | 41.36 | +0.01 | 41.97 | 40.85 | 1.12 | 12 |
| 20 | baseline | Baseline | 41.35 | - | 41.85 | 41.18 | 0.66 | 12 |
| 21 | photometric_distort | Augmentation | 41.32 | -0.03 | 41.92 | 40.75 | 1.17 | 12 |
| 22 | gen_Qwen_Image_Edit | Generative | 41.26 | -0.09 | 41.79 | 40.76 | 1.03 | 12 |
| 23 | gen_step1x_new | Generative | 41.24 | -0.11 | 41.82 | 40.41 | 1.41 | 12 |
| 24 | std_mixup | Standard Aug | 41.16 | -0.20 | 41.85 | 39.64 | 2.21 | 12 |
| 25 | std_randaugment | Standard Aug | 41.14 | -0.21 | 41.72 | 40.47 | 1.25 | 12 |
| 26 | std_cutmix | Standard Aug | 40.81 | -0.54 | 41.62 | 39.96 | 1.66 | 12 |
| 27 | gen_Weather_Effect_Generator | Generative | 40.71 | -0.64 | 41.36 | 39.94 | 1.42 | 12 |

## Per-Dataset Breakdown

### bdd10k

| Strategy | mIoU | Models |
|----------|------|--------|
| gen_cycleGAN | 46.94 | 3 |
| std_autoaugment | 46.90 | 3 |
| gen_Attribute_Hallucination | 46.77 | 3 |
| gen_CNetSeg | 46.72 | 3 |
| gen_UniControl | 46.72 | 3 |
| gen_Qwen_Image_Edit | 46.70 | 3 |
| gen_LANIT | 46.56 | 3 |
| gen_augmenters | 46.56 | 3 |
| gen_Weather_Effect_Generator | 46.54 | 3 |
| gen_automold | 46.54 | 3 |
| photometric_distort | 46.49 | 3 |
| gen_cyclediffusion | 46.48 | 3 |
| gen_CUT | 46.43 | 3 |
| gen_TSIT | 46.39 | 3 |
| gen_step1x_v1p2 | 46.31 | 3 |
| std_randaugment | 46.29 | 3 |
| gen_stargan_v2 | 46.13 | 3 |
| gen_IP2P | 46.12 | 3 |
| gen_VisualCloze | 46.09 | 3 |
| gen_albumentations_weather | 46.08 | 3 |
| std_mixup | 46.04 | 3 |
| gen_Img2Img | 46.04 | 3 |
| gen_SUSTechGAN | 45.94 | 3 |
| baseline | 45.78 | 3 |
| gen_flux_kontext | 45.60 | 3 |
| gen_step1x_new | 45.50 | 3 |
| std_cutmix | 45.18 | 3 |

### idd-aw

| Strategy | mIoU | Models |
|----------|------|--------|
| gen_CUT | 44.92 | 3 |
| gen_flux_kontext | 44.84 | 3 |
| gen_Attribute_Hallucination | 44.78 | 3 |
| gen_automold | 44.74 | 3 |
| gen_augmenters | 44.73 | 3 |
| gen_SUSTechGAN | 44.72 | 3 |
| gen_VisualCloze | 44.72 | 3 |
| gen_step1x_v1p2 | 44.71 | 3 |
| gen_IP2P | 44.69 | 3 |
| gen_albumentations_weather | 44.68 | 3 |
| gen_cycleGAN | 44.67 | 3 |
| gen_stargan_v2 | 44.65 | 3 |
| gen_LANIT | 44.64 | 3 |
| gen_TSIT | 44.64 | 3 |
| gen_Qwen_Image_Edit | 44.64 | 3 |
| gen_UniControl | 44.64 | 3 |
| gen_CNetSeg | 44.63 | 3 |
| std_autoaugment | 44.62 | 3 |
| gen_Img2Img | 44.60 | 3 |
| std_randaugment | 44.56 | 3 |
| baseline | 44.49 | 3 |
| photometric_distort | 44.48 | 3 |
| gen_cyclediffusion | 44.45 | 3 |
| std_cutmix | 44.43 | 3 |
| gen_step1x_new | 44.28 | 3 |
| std_mixup | 44.22 | 3 |
| gen_Weather_Effect_Generator | 44.19 | 3 |

### mapillaryvistas

| Strategy | mIoU | Models |
|----------|------|--------|
| gen_VisualCloze | 36.24 | 3 |
| gen_flux_kontext | 36.20 | 3 |
| baseline | 36.12 | 3 |
| gen_stargan_v2 | 36.11 | 3 |
| gen_TSIT | 36.11 | 3 |
| gen_SUSTechGAN | 36.06 | 3 |
| gen_Attribute_Hallucination | 36.04 | 3 |
| gen_UniControl | 36.03 | 3 |
| gen_step1x_v1p2 | 36.01 | 3 |
| gen_cycleGAN | 36.00 | 3 |
| gen_Qwen_Image_Edit | 35.89 | 3 |
| gen_Img2Img | 35.86 | 3 |
| gen_LANIT | 35.79 | 3 |
| gen_CUT | 35.79 | 3 |
| gen_step1x_new | 35.76 | 3 |
| gen_CNetSeg | 35.73 | 3 |
| gen_augmenters | 35.72 | 3 |
| photometric_distort | 35.70 | 3 |
| gen_cyclediffusion | 35.68 | 3 |
| gen_albumentations_weather | 35.61 | 3 |
| gen_automold | 35.54 | 3 |
| gen_IP2P | 35.29 | 3 |
| std_autoaugment | 34.88 | 3 |
| std_cutmix | 34.85 | 3 |
| std_mixup | 34.82 | 3 |
| std_randaugment | 33.98 | 3 |
| gen_Weather_Effect_Generator | 33.75 | 3 |

### outside15k

| Strategy | mIoU | Models |
|----------|------|--------|
| gen_stargan_v2 | 40.01 | 3 |
| gen_cyclediffusion | 39.75 | 3 |
| std_randaugment | 39.74 | 3 |
| gen_CNetSeg | 39.67 | 3 |
| gen_VisualCloze | 39.62 | 3 |
| gen_SUSTechGAN | 39.56 | 3 |
| gen_albumentations_weather | 39.55 | 3 |
| std_mixup | 39.54 | 3 |
| gen_Img2Img | 39.43 | 3 |
| gen_UniControl | 39.41 | 3 |
| gen_step1x_new | 39.41 | 3 |
| gen_IP2P | 39.36 | 3 |
| gen_augmenters | 39.32 | 3 |
| gen_CUT | 39.19 | 3 |
| gen_LANIT | 39.18 | 3 |
| std_autoaugment | 39.14 | 3 |
| gen_flux_kontext | 39.05 | 3 |
| baseline | 39.02 | 3 |
| gen_TSIT | 38.98 | 3 |
| gen_step1x_v1p2 | 38.97 | 3 |
| gen_cycleGAN | 38.94 | 3 |
| gen_Attribute_Hallucination | 38.86 | 3 |
| std_cutmix | 38.80 | 3 |
| gen_automold | 38.70 | 3 |
| photometric_distort | 38.60 | 3 |
| gen_Weather_Effect_Generator | 38.36 | 3 |
| gen_Qwen_Image_Edit | 37.83 | 3 |
