# Cityscapes-Gen Strategy Leaderboard (by mIoU)

**Cityscapes-Gen**: Cityscapes Generative Augmentation + ACDC cross-domain testing

**Metric**: mIoU (Mean Intersection over Union)

**Last Updated**: 2026-02-10 09:52
**Baseline mIoU**: 50.85%
**Total Results**: 267 test results from 25 strategies

---

## Overall Strategy Ranking

Sorted by mIoU. Gain = improvement over baseline. Domain Gap = Normal mIoU - Adverse mIoU (lower is better).

| Strategy | Type | mIoU | Std | Gain | Normal mIoU | Adverse mIoU | Domain Gap | Num Tests |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_augmenters | Generative | 52.09 | 9.23 | 1.24 | - | 46.0 | - | 6 |
| gen_cycleGAN | Generative | 51.27 | 9.4 | 0.43 | - | 45.43 | - | 6 |
| gen_VisualCloze | Generative | 50.97 | 10.76 | 0.12 | - | 45.03 | - | 12 |
| gen_flux_kontext | Generative | 50.95 | 10.37 | 0.1 | - | 45.0 | - | 12 |
| baseline | Baseline | 50.85 | 10.7 | 0.0 | - | 44.79 | - | 12 |
| std_mixup | Standard Aug | 50.83 | 10.26 | -0.02 | - | 44.67 | - | 12 |
| std_cutmix | Standard Aug | 50.8 | 10.47 | -0.05 | - | 44.75 | - | 12 |
| gen_TSIT | Generative | 50.74 | 9.91 | -0.11 | - | 44.59 | - | 6 |
| std_randaugment | Standard Aug | 50.69 | 10.73 | -0.16 | - | 44.79 | - | 12 |
| gen_automold | Generative | 50.61 | 10.22 | -0.24 | - | 44.8 | - | 12 |
| gen_step1x_v1p2 | Generative | 50.58 | 10.72 | -0.26 | - | 44.53 | - | 12 |
| std_autoaugment | Standard Aug | 50.51 | 10.58 | -0.34 | - | 44.59 | - | 12 |
| gen_albumentations_weather | Generative | 50.38 | 10.55 | -0.47 | - | 44.44 | - | 12 |
| gen_step1x_new | Generative | 50.22 | 10.36 | -0.63 | - | 44.43 | - | 12 |
| gen_SUSTechGAN | Generative | 49.13 | 9.3 | -1.72 | - | 44.57 | - | 11 |
| gen_cyclediffusion | Generative | 49.05 | 9.2 | -1.8 | - | 44.9 | - | 11 |
| gen_Img2Img | Generative | 49.05 | 9.06 | -1.79 | - | 45.54 | - | 12 |
| gen_Qwen_Image_Edit | Generative | 48.69 | 9.35 | -2.16 | - | 44.97 | - | 12 |
| gen_IP2P | Generative | 48.35 | 9.96 | -2.49 | - | 44.11 | - | 12 |
| gen_Weather_Effect_Generator | Generative | 47.89 | 10.49 | -2.96 | - | 43.02 | - | 9 |
| gen_CNetSeg | Generative | 47.09 | 10.54 | -3.76 | - | 42.3 | - | 9 |
| gen_CUT | Generative | 46.99 | 7.9 | -3.85 | - | 44.99 | - | 12 |
| gen_stargan_v2 | Generative | 46.97 | 10.75 | -3.88 | - | 42.1 | - | 9 |
| gen_Attribute_Hallucination | Generative | 46.89 | 7.83 | -3.96 | - | 45.13 | - | 10 |
| gen_UniControl | Generative | 46.58 | 8.06 | -4.27 | - | 44.93 | - | 10 |
---

## Per-Dataset Breakdown

mIoU performance on each dataset.

| Strategy | Type | cityscapes | cityscapes_gain | acdc | acdc_gain |
| --- | --- | --- | --- | --- | --- |
| gen_augmenters | Generative | 55.06 | +1.03 | 46.16 | +1.67 |
| gen_cycleGAN | Generative | 54.26 | +0.23 | 45.31 | +0.82 |
| gen_flux_kontext | Generative | 53.98 | -0.05 | 44.89 | +0.40 |
| gen_VisualCloze | Generative | 54.12 | +0.09 | 44.68 | +0.19 |
| std_mixup | Standard Aug | 53.89 | -0.14 | 44.71 | +0.22 |
| baseline | Baseline | 54.03 | +0.00 | 44.49 | +0.00 |
| std_cutmix | Standard Aug | 53.94 | -0.09 | 44.53 | +0.04 |
| gen_TSIT | Generative | 53.94 | -0.09 | 44.35 | -0.14 |
| std_randaugment | Standard Aug | 53.86 | -0.17 | 44.34 | -0.15 |
| gen_automold | Generative | 53.67 | -0.36 | 44.49 | +0.00 |
| gen_step1x_v1p2 | Generative | 53.80 | -0.23 | 44.16 | -0.33 |
| std_autoaugment | Standard Aug | 53.61 | -0.42 | 44.32 | -0.17 |
| gen_albumentations_weather | Generative | 53.53 | -0.50 | 44.08 | -0.41 |
| gen_step1x_new | Generative | 53.30 | -0.72 | 44.05 | -0.44 |
| gen_Img2Img | Generative | 51.52 | -2.51 | 45.36 | +0.87 |
| gen_SUSTechGAN | Generative | 51.80 | -2.23 | 44.47 | -0.02 |
| gen_cyclediffusion | Generative | 51.62 | -2.41 | 44.55 | +0.06 |
| gen_Qwen_Image_Edit | Generative | 51.25 | -2.77 | 44.85 | +0.36 |
| gen_IP2P | Generative | 51.14 | -2.88 | 43.47 | -1.02 |
| gen_Weather_Effect_Generator | Generative | 51.05 | -2.97 | 42.62 | -1.87 |
| gen_CUT | Generative | 48.50 | -5.52 | 45.11 | +0.62 |
| gen_Attribute_Hallucination | Generative | 48.51 | -5.52 | 44.88 | +0.39 |
| gen_UniControl | Generative | 48.18 | -5.85 | 44.58 | +0.09 |
| gen_CNetSeg | Generative | 50.34 | -3.69 | 41.66 | -2.83 |
| gen_stargan_v2 | Generative | 50.18 | -3.84 | 41.61 | -2.88 |
---

## Per-Domain Breakdown

mIoU performance on each weather domain. Normal = clear_day. Adverse = foggy, night, rainy, snowy.

| Strategy | Type | clear_day | foggy | night | rainy | snowy | Normal Avg | Adverse Avg | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Baseline | - | 59.30 | 28.41 | 46.33 | 45.11 | - | 44.79 | - |
| gen_Attribute_Hallucination | Generative | - | 59.38 | 29.24 | 46.15 | 45.74 | - | 45.13 | - |
| gen_CNetSeg | Generative | - | 55.89 | 25.27 | 44.92 | 43.12 | - | 42.30 | - |
| gen_CUT | Generative | - | 59.01 | 29.32 | 46.85 | 44.80 | - | 44.99 | - |
| gen_IP2P | Generative | - | 58.58 | 28.03 | 45.60 | 44.22 | - | 44.11 | - |
| gen_Img2Img | Generative | - | 58.80 | 28.93 | 47.75 | 46.67 | - | 45.54 | - |
| gen_Qwen_Image_Edit | Generative | - | 58.61 | 28.79 | 47.15 | 45.32 | - | 44.97 | - |
| gen_SUSTechGAN | Generative | - | 57.83 | 28.89 | 46.03 | 45.55 | - | 44.57 | - |
| gen_TSIT | Generative | - | 59.97 | 27.19 | 45.03 | 46.17 | - | 44.59 | - |
| gen_UniControl | Generative | - | 59.35 | 28.20 | 46.41 | 45.74 | - | 44.93 | - |
| gen_VisualCloze | Generative | - | 59.58 | 28.80 | 46.31 | 45.42 | - | 45.03 | - |
| gen_Weather_Effect_Generator | Generative | - | 56.81 | 26.24 | 45.57 | 43.48 | - | 43.02 | - |
| gen_albumentations_weather | Generative | - | 58.61 | 28.47 | 46.00 | 44.68 | - | 44.44 | - |
| gen_augmenters | Generative | - | 59.59 | 29.90 | 47.79 | 46.72 | - | 46.00 | - |
| gen_automold | Generative | - | 58.06 | 28.00 | 47.47 | 45.65 | - | 44.80 | - |
| gen_cycleGAN | Generative | - | 60.08 | 28.78 | 47.25 | 45.62 | - | 45.43 | - |
| gen_cyclediffusion | Generative | - | 59.19 | 29.07 | 46.17 | 45.16 | - | 44.90 | - |
| gen_flux_kontext | Generative | - | 59.00 | 29.11 | 46.76 | 45.14 | - | 45.00 | - |
| gen_stargan_v2 | Generative | - | 56.25 | 25.82 | 43.71 | 42.60 | - | 42.10 | - |
| gen_step1x_new | Generative | - | 57.38 | 27.34 | 47.44 | 45.54 | - | 44.43 | - |
| gen_step1x_v1p2 | Generative | - | 58.86 | 28.32 | 46.29 | 44.67 | - | 44.53 | - |
| std_autoaugment | Standard Aug | - | 58.64 | 28.79 | 45.66 | 45.26 | - | 44.59 | - |
| std_cutmix | Standard Aug | - | 59.44 | 28.69 | 46.23 | 44.62 | - | 44.75 | - |
| std_mixup | Standard Aug | - | 58.53 | 28.59 | 46.64 | 44.93 | - | 44.67 | - |
| std_randaugment | Standard Aug | - | 58.95 | 28.68 | 46.42 | 45.13 | - | 44.79 | - |
