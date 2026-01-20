# Stage 2 Strategy Leaderboard

**Training:** All domains (not filtered to clear_day)
**Last Updated:** 2026-01-20 14:55
**Baseline mIoU:** 43.74%
**Total Test Results:** 275
**Strategies Evaluated:** 24

## Overall Rankings

| Rank | Strategy | Type | mIoU | Gain | Normal | Adverse | Gap | Models |
|------|----------|------|------|------|--------|---------|-----|--------|
| 1 | gen_IP2P | Generative | 45.08 | +1.34 | 45.74 | 44.72 | 1.03 | 10 |
| 2 | gen_Img2Img | Generative | 45.07 | +1.33 | 45.72 | 44.62 | 1.11 | 10 |
| 3 | gen_UniControl | Generative | 45.00 | +1.25 | 45.73 | 44.74 | 0.99 | 12 |
| 4 | gen_LANIT | Generative | 44.98 | +1.23 | 45.64 | 44.34 | 1.30 | 11 |
| 5 | gen_CNetSeg | Generative | 44.95 | +1.20 | 45.63 | 44.55 | 1.08 | 12 |
| 6 | gen_cycleGAN | Generative | 44.93 | +1.19 | 45.63 | 44.38 | 1.25 | 12 |
| 7 | gen_augmenters | Generative | 44.92 | +1.18 | 45.59 | 44.79 | 0.80 | 12 |
| 8 | gen_stargan_v2 | Generative | 44.86 | +1.11 | 45.66 | 44.33 | 1.33 | 11 |
| 9 | gen_VisualCloze | Generative | 44.82 | +1.08 | 45.53 | 44.65 | 0.88 | 12 |
| 10 | gen_albumentations_weather | Generative | 44.82 | +1.08 | 45.64 | 44.41 | 1.22 | 12 |
| 11 | std_randaugment | Standard Aug | 44.77 | +1.03 | 45.44 | 44.48 | 0.96 | 12 |
| 12 | gen_SUSTechGAN | Generative | 44.77 | +1.03 | 45.49 | 44.39 | 1.10 | 12 |
| 13 | std_autoaugment | Standard Aug | 44.75 | +1.01 | 45.43 | 44.65 | 0.78 | 12 |
| 14 | gen_TSIT | Generative | 44.72 | +0.97 | 45.49 | 44.35 | 1.14 | 12 |
| 15 | gen_step1x_v1p2 | Generative | 44.71 | +0.97 | 45.36 | 44.38 | 0.98 | 12 |
| 16 | gen_automold | Generative | 44.66 | +0.91 | 45.43 | 44.35 | 1.08 | 12 |
| 17 | gen_CUT | Generative | 44.47 | +0.73 | 45.17 | 44.18 | 0.99 | 9 |
| 18 | gen_flux_kontext | Generative | 44.39 | +0.64 | 45.18 | 44.30 | 0.88 | 16 |
| 19 | gen_Weather_Effect_Generator | Generative | 44.34 | +0.59 | 45.12 | 43.72 | 1.40 | 11 |
| 20 | gen_Qwen_Image_Edit | Generative | 44.32 | +0.57 | 45.05 | 44.19 | 0.86 | 15 |
| 21 | gen_step1x_new | Generative | 44.31 | +0.57 | 44.98 | 43.81 | 1.17 | 11 |
| 22 | photometric_distort | Augmentation | 44.17 | +0.43 | 45.00 | 43.62 | 1.38 | 10 |
| 23 | gen_Attribute_Hallucination | Generative | 43.92 | +0.18 | 44.67 | 43.68 | 0.99 | 9 |
| 24 | baseline | Baseline | 43.74 | - | 44.42 | 43.93 | 0.48 | 8 |

## Per-Dataset Breakdown

### bdd10k

| Strategy | mIoU | Models |
|----------|------|--------|
| gen_IP2P | 51.71 | 1 |
| gen_Img2Img | 51.50 | 1 |
| gen_LANIT | 48.30 | 2 |
| gen_cycleGAN | 46.94 | 3 |
| std_autoaugment | 46.90 | 3 |
| gen_CNetSeg | 46.72 | 3 |
| gen_UniControl | 46.72 | 3 |
| gen_augmenters | 46.56 | 3 |
| gen_Weather_Effect_Generator | 46.54 | 3 |
| gen_automold | 46.54 | 3 |
| photometric_distort | 46.49 | 3 |
| gen_TSIT | 46.39 | 3 |
| gen_step1x_v1p2 | 46.31 | 3 |
| std_randaugment | 46.29 | 3 |
| gen_stargan_v2 | 46.13 | 3 |
| gen_VisualCloze | 46.09 | 3 |
| gen_albumentations_weather | 46.08 | 3 |
| gen_SUSTechGAN | 45.94 | 3 |
| gen_step1x_new | 45.50 | 3 |
| gen_Qwen_Image_Edit | 45.31 | 5 |
| gen_flux_kontext | 44.69 | 5 |

### idd-aw

| Strategy | mIoU | Models |
|----------|------|--------|
| gen_CUT | 44.92 | 3 |
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
| gen_UniControl | 44.64 | 3 |
| gen_CNetSeg | 44.63 | 3 |
| std_autoaugment | 44.62 | 3 |
| gen_Img2Img | 44.60 | 3 |
| std_randaugment | 44.56 | 3 |
| baseline | 44.49 | 3 |
| photometric_distort | 44.48 | 3 |
| gen_Qwen_Image_Edit | 44.48 | 4 |
| gen_flux_kontext | 44.34 | 5 |
| gen_step1x_new | 44.28 | 3 |
| gen_Weather_Effect_Generator | 44.19 | 3 |

### mapillaryvistas

| Strategy | mIoU | Models |
|----------|------|--------|
| photometric_distort | 52.97 | 1 |
| gen_stargan_v2 | 50.52 | 2 |
| gen_Weather_Effect_Generator | 50.23 | 2 |
| gen_step1x_new | 49.92 | 2 |
| baseline | 49.72 | 2 |
| gen_CUT | 49.31 | 3 |
| gen_flux_kontext | 49.29 | 3 |
| gen_UniControl | 49.22 | 3 |
| gen_cycleGAN | 49.17 | 3 |
| gen_augmenters | 49.07 | 3 |
| gen_Img2Img | 49.05 | 3 |
| gen_IP2P | 48.99 | 3 |
| gen_albumentations_weather | 48.97 | 3 |
| gen_Qwen_Image_Edit | 48.92 | 3 |
| gen_LANIT | 48.89 | 3 |
| gen_VisualCloze | 48.87 | 3 |
| gen_TSIT | 48.87 | 3 |
| gen_SUSTechGAN | 48.86 | 3 |
| gen_step1x_v1p2 | 48.85 | 3 |
| gen_CNetSeg | 48.76 | 3 |
| gen_automold | 48.65 | 3 |
| std_randaugment | 48.50 | 3 |
| std_autoaugment | 48.35 | 3 |
| gen_Attribute_Hallucination | 48.13 | 3 |

### outside15k

| Strategy | mIoU | Models |
|----------|------|--------|
| gen_stargan_v2 | 40.01 | 3 |
| std_randaugment | 39.74 | 3 |
| gen_CNetSeg | 39.67 | 3 |
| gen_VisualCloze | 39.62 | 3 |
| gen_SUSTechGAN | 39.56 | 3 |
| gen_albumentations_weather | 39.55 | 3 |
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
