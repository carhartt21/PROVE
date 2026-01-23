# Stage 2 Strategy Leaderboard

**Training:** All domains (not filtered to clear_day)
**Last Updated:** 2026-01-22 16:40
**Baseline mIoU:** 43.10%
**Total Test Results:** 243
**Strategies Evaluated:** 27

## Overall Rankings

| Rank | Strategy | Type | mIoU | Gain | Normal | Adverse | Gap | Models |
|------|----------|------|------|------|--------|---------|-----|--------|
| 1 | gen_CNetSeg | Generative | 43.68 | +0.58 | 44.56 | 42.71 | 1.85 | 9 |
| 2 | gen_stargan_v2 | Generative | 43.60 | +0.50 | 44.53 | 42.94 | 1.58 | 9 |
| 3 | gen_UniControl | Generative | 43.59 | +0.49 | 44.58 | 42.77 | 1.81 | 9 |
| 4 | gen_cyclediffusion | Generative | 43.56 | +0.47 | 44.58 | 42.70 | 1.88 | 9 |
| 5 | std_autoaugment | Standard Aug | 43.55 | +0.46 | 44.38 | 42.88 | 1.50 | 9 |
| 6 | gen_augmenters | Generative | 43.54 | +0.44 | 44.39 | 42.76 | 1.63 | 9 |
| 7 | std_randaugment | Standard Aug | 43.53 | +0.43 | 44.39 | 42.67 | 1.72 | 9 |
| 8 | gen_cycleGAN | Generative | 43.52 | +0.42 | 44.40 | 42.50 | 1.90 | 9 |
| 9 | gen_CUT | Generative | 43.51 | +0.42 | 44.37 | 42.80 | 1.57 | 9 |
| 10 | gen_VisualCloze | Generative | 43.48 | +0.38 | 44.41 | 42.73 | 1.68 | 9 |
| 11 | gen_Attribute_Hallucination | Generative | 43.47 | +0.38 | 44.44 | 42.50 | 1.94 | 9 |
| 12 | gen_LANIT | Generative | 43.46 | +0.37 | 44.42 | 42.41 | 2.01 | 9 |
| 13 | gen_albumentations_weather | Generative | 43.44 | +0.34 | 44.44 | 42.52 | 1.92 | 9 |
| 14 | gen_SUSTechGAN | Generative | 43.41 | +0.31 | 44.33 | 42.48 | 1.85 | 9 |
| 15 | gen_IP2P | Generative | 43.39 | +0.29 | 44.29 | 42.65 | 1.64 | 9 |
| 16 | gen_Img2Img | Generative | 43.35 | +0.26 | 44.29 | 42.53 | 1.77 | 9 |
| 17 | gen_TSIT | Generative | 43.34 | +0.24 | 44.29 | 42.32 | 1.97 | 9 |
| 18 | gen_step1x_v1p2 | Generative | 43.33 | +0.24 | 44.20 | 42.39 | 1.81 | 9 |
| 19 | gen_automold | Generative | 43.33 | +0.23 | 44.33 | 42.42 | 1.91 | 9 |
| 20 | std_mixup | Standard Aug | 43.27 | +0.17 | 44.29 | 41.46 | 2.83 | 9 |
| 21 | photometric_distort | Augmentation | 43.19 | +0.10 | 44.09 | 42.46 | 1.64 | 9 |
| 22 | gen_flux_kontext | Generative | 43.16 | +0.07 | 44.15 | 42.43 | 1.71 | 9 |
| 23 | baseline | Baseline | 43.10 | - | 43.84 | 42.97 | 0.87 | 9 |
| 24 | gen_step1x_new | Generative | 43.06 | -0.03 | 43.87 | 42.02 | 1.85 | 9 |
| 25 | gen_Qwen_Image_Edit | Generative | 43.06 | -0.04 | 43.90 | 42.37 | 1.53 | 9 |
| 26 | gen_Weather_Effect_Generator | Generative | 43.03 | -0.07 | 43.95 | 42.00 | 1.95 | 9 |
| 27 | std_cutmix | Standard Aug | 42.80 | -0.29 | 43.91 | 41.79 | 2.12 | 9 |

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
