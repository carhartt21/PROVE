# Stage 2 Strategy Leaderboard

**Training:** All domains (not filtered to clear_day)
**Last Updated:** 2026-01-21 12:15
**Baseline mIoU:** 44.48%
**Total Test Results:** 318
**Strategies Evaluated:** 27

## Overall Rankings

| Rank | Strategy | Type | mIoU | Gain | Normal | Adverse | Gap | Models |
|------|----------|------|------|------|--------|---------|-----|--------|
| 1 | std_cutmix | Standard Aug | 45.94 | +1.45 | 46.65 | 45.79 | 0.86 | 10 |
| 2 | gen_UniControl | Generative | 45.00 | +0.51 | 45.73 | 44.74 | 0.99 | 12 |
| 3 | gen_CUT | Generative | 44.96 | +0.48 | 45.67 | 44.66 | 1.02 | 12 |
| 4 | gen_CNetSeg | Generative | 44.95 | +0.46 | 45.63 | 44.55 | 1.08 | 12 |
| 5 | gen_stargan_v2 | Generative | 44.93 | +0.45 | 45.66 | 44.59 | 1.07 | 12 |
| 6 | gen_cycleGAN | Generative | 44.93 | +0.45 | 45.63 | 44.38 | 1.25 | 12 |
| 7 | gen_augmenters | Generative | 44.92 | +0.44 | 45.59 | 44.79 | 0.80 | 12 |
| 8 | gen_VisualCloze | Generative | 44.82 | +0.34 | 45.53 | 44.65 | 0.88 | 12 |
| 9 | gen_albumentations_weather | Generative | 44.82 | +0.34 | 45.64 | 44.41 | 1.22 | 12 |
| 10 | gen_LANIT | Generative | 44.82 | +0.34 | 45.55 | 44.28 | 1.27 | 12 |
| 11 | gen_IP2P | Generative | 44.79 | +0.30 | 45.55 | 44.51 | 1.04 | 12 |
| 12 | gen_Img2Img | Generative | 44.78 | +0.29 | 45.52 | 44.49 | 1.04 | 12 |
| 13 | std_randaugment | Standard Aug | 44.77 | +0.29 | 45.44 | 44.48 | 0.96 | 12 |
| 14 | gen_SUSTechGAN | Generative | 44.77 | +0.29 | 45.49 | 44.39 | 1.10 | 12 |
| 15 | std_autoaugment | Standard Aug | 44.75 | +0.27 | 45.43 | 44.65 | 0.78 | 12 |
| 16 | gen_TSIT | Generative | 44.72 | +0.24 | 45.49 | 44.35 | 1.14 | 12 |
| 17 | gen_step1x_v1p2 | Generative | 44.71 | +0.23 | 45.36 | 44.38 | 0.98 | 12 |
| 18 | gen_cyclediffusion | Generative | 44.70 | +0.21 | 45.57 | 44.17 | 1.40 | 11 |
| 19 | gen_flux_kontext | Generative | 44.69 | +0.21 | 45.47 | 44.45 | 1.02 | 12 |
| 20 | gen_automold | Generative | 44.66 | +0.17 | 45.43 | 44.35 | 1.08 | 12 |
| 21 | gen_Attribute_Hallucination | Generative | 44.63 | +0.15 | 45.40 | 44.34 | 1.06 | 12 |
| 22 | gen_Qwen_Image_Edit | Generative | 44.52 | +0.04 | 45.20 | 44.40 | 0.80 | 12 |
| 23 | photometric_distort | Augmentation | 44.50 | +0.01 | 45.18 | 44.24 | 0.94 | 12 |
| 24 | baseline | Baseline | 44.48 | - | 45.08 | 44.89 | 0.20 | 12 |
| 25 | gen_step1x_new | Generative | 44.45 | -0.03 | 45.08 | 44.07 | 1.00 | 12 |
| 26 | gen_Weather_Effect_Generator | Generative | 44.34 | -0.15 | 45.12 | 43.72 | 1.40 | 11 |
| 27 | std_mixup | Standard Aug | 44.24 | -0.25 | 45.19 | 42.63 | 2.57 | 10 |

## Per-Dataset Breakdown

### bdd10k

| Strategy | mIoU | Models |
|----------|------|--------|
| std_cutmix | 47.11 | 2 |
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
| std_mixup | 52.96 | 1 |
| gen_Weather_Effect_Generator | 50.23 | 2 |
| gen_cyclediffusion | 49.81 | 2 |
| gen_CUT | 49.31 | 3 |
| gen_flux_kontext | 49.29 | 3 |
| gen_UniControl | 49.22 | 3 |
| gen_cycleGAN | 49.17 | 3 |
| gen_augmenters | 49.07 | 3 |
| gen_Img2Img | 49.05 | 3 |
| gen_IP2P | 48.99 | 3 |
| std_cutmix | 48.99 | 3 |
| gen_albumentations_weather | 48.97 | 3 |
| gen_stargan_v2 | 48.93 | 3 |
| gen_Qwen_Image_Edit | 48.92 | 3 |
| gen_LANIT | 48.89 | 3 |
| gen_VisualCloze | 48.87 | 3 |
| gen_TSIT | 48.87 | 3 |
| gen_SUSTechGAN | 48.86 | 3 |
| gen_step1x_v1p2 | 48.85 | 3 |
| gen_CNetSeg | 48.76 | 3 |
| gen_automold | 48.65 | 3 |
| baseline | 48.65 | 3 |
| gen_step1x_new | 48.62 | 3 |
| std_randaugment | 48.50 | 3 |
| photometric_distort | 48.41 | 3 |
| std_autoaugment | 48.35 | 3 |
| gen_Attribute_Hallucination | 48.13 | 3 |

### outside15k

| Strategy | mIoU | Models |
|----------|------|--------|
| std_cutmix | 42.46 | 2 |
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
| gen_automold | 38.70 | 3 |
| photometric_distort | 38.60 | 3 |
| gen_Weather_Effect_Generator | 38.36 | 3 |
| gen_Qwen_Image_Edit | 37.83 | 3 |
