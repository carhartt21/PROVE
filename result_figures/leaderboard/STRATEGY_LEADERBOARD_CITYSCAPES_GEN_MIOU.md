# Cityscapes-Gen Strategy Leaderboard (by mIoU)

**Cityscapes-Gen**: Cityscapes Generative Augmentation + ACDC cross-domain testing

**Metric**: mIoU (Mean Intersection over Union)

**Last Updated**: 2026-02-10 14:41
**Baseline mIoU**: 50.85%
**Total Results**: 292 test results from 25 strategies

---

## Overall Strategy Ranking

Sorted by mIoU. Gain = improvement over baseline. Domain Gap = Normal mIoU - Adverse mIoU (lower is better).

| Strategy | Type | mIoU | Std | Gain | Normal mIoU | Adverse mIoU | Domain Gap | Num Tests |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_VisualCloze | Generative | 50.97 | 10.76 | 0.12 | 54.12 | 44.68 | 9.44 | 12 |
| gen_flux_kontext | Generative | 50.95 | 10.37 | 0.1 | 53.98 | 44.89 | 9.09 | 12 |
| baseline | Baseline | 50.85 | 10.7 | 0.0 | 54.03 | 44.49 | 9.54 | 12 |
| std_mixup | Standard Aug | 50.83 | 10.26 | -0.02 | 53.89 | 44.71 | 9.17 | 12 |
| std_cutmix | Standard Aug | 50.8 | 10.47 | -0.05 | 53.94 | 44.53 | 9.41 | 12 |
| std_randaugment | Standard Aug | 50.69 | 10.73 | -0.16 | 53.86 | 44.34 | 9.52 | 12 |
| gen_automold | Generative | 50.61 | 10.22 | -0.24 | 53.67 | 44.49 | 9.18 | 12 |
| gen_step1x_v1p2 | Generative | 50.58 | 10.72 | -0.26 | 53.8 | 44.16 | 9.64 | 12 |
| gen_albumentations_weather | Generative | 50.38 | 10.55 | -0.47 | 53.53 | 44.08 | 9.45 | 12 |
| gen_step1x_new | Generative | 50.22 | 10.36 | -0.63 | 53.3 | 44.05 | 9.25 | 12 |
| std_autoaugment | Standard Aug | 49.2 | 10.65 | -1.64 | 52.33 | 42.95 | 9.38 | 15 |
| gen_SUSTechGAN | Generative | 49.13 | 9.3 | -1.72 | 51.8 | 44.47 | 7.33 | 11 |
| gen_cyclediffusion | Generative | 49.05 | 9.2 | -1.8 | 51.62 | 44.55 | 7.07 | 11 |
| gen_Img2Img | Generative | 49.05 | 9.06 | -1.79 | 51.52 | 45.36 | 6.15 | 12 |
| gen_cycleGAN | Generative | 48.77 | 10.24 | -2.08 | 51.9 | 42.5 | 9.4 | 9 |
| gen_Qwen_Image_Edit | Generative | 48.69 | 9.35 | -2.16 | 51.25 | 44.85 | 6.4 | 12 |
| gen_Weather_Effect_Generator | Generative | 48.36 | 9.31 | -2.49 | 50.92 | 44.52 | 6.39 | 12 |
| gen_IP2P | Generative | 48.35 | 9.96 | -2.49 | 51.14 | 43.47 | 7.67 | 12 |
| gen_augmenters | Generative | 48.33 | 10.46 | -2.52 | 51.46 | 43.12 | 8.34 | 9 |
| gen_TSIT | Generative | 48.27 | 9.48 | -2.58 | 50.81 | 44.46 | 6.35 | 12 |
| gen_CNetSeg | Generative | 48.01 | 9.5 | -2.84 | 50.57 | 44.17 | 6.4 | 12 |
| gen_Attribute_Hallucination | Generative | 47.92 | 8.06 | -2.93 | 49.94 | 44.88 | 5.07 | 11 |
| gen_stargan_v2 | Generative | 47.78 | 9.63 | -3.07 | 50.32 | 43.96 | 6.36 | 12 |
| gen_CUT | Generative | 46.99 | 7.9 | -3.85 | 48.5 | 45.11 | 3.4 | 12 |
| gen_UniControl | Generative | 46.58 | 8.06 | -4.27 | 48.18 | 44.58 | 3.59 | 10 |
---

## Per-Dataset Breakdown

mIoU performance on each dataset.

| Strategy | Type | cityscapes | cityscapes_gain | acdc | acdc_gain |
| --- | --- | --- | --- | --- | --- |
| gen_flux_kontext | Generative | 53.98 | -0.05 | 44.89 | +0.40 |
| gen_VisualCloze | Generative | 54.12 | +0.09 | 44.68 | +0.19 |
| std_mixup | Standard Aug | 53.89 | -0.14 | 44.71 | +0.22 |
| baseline | Baseline | 54.03 | +0.00 | 44.49 | +0.00 |
| std_cutmix | Standard Aug | 53.94 | -0.09 | 44.53 | +0.04 |
| std_randaugment | Standard Aug | 53.86 | -0.17 | 44.34 | -0.15 |
| gen_automold | Generative | 53.67 | -0.36 | 44.49 | +0.00 |
| gen_step1x_v1p2 | Generative | 53.80 | -0.23 | 44.16 | -0.33 |
| gen_albumentations_weather | Generative | 53.53 | -0.50 | 44.08 | -0.41 |
| gen_step1x_new | Generative | 53.30 | -0.72 | 44.05 | -0.44 |
| gen_Img2Img | Generative | 51.52 | -2.51 | 45.36 | +0.87 |
| gen_SUSTechGAN | Generative | 51.80 | -2.23 | 44.47 | -0.02 |
| gen_cyclediffusion | Generative | 51.62 | -2.41 | 44.55 | +0.06 |
| gen_Qwen_Image_Edit | Generative | 51.25 | -2.77 | 44.85 | +0.36 |
| gen_Weather_Effect_Generator | Generative | 50.92 | -3.11 | 44.52 | +0.03 |
| std_autoaugment | Standard Aug | 52.33 | -1.70 | 42.95 | -1.54 |
| gen_TSIT | Generative | 50.81 | -3.21 | 44.46 | -0.03 |
| gen_Attribute_Hallucination | Generative | 49.94 | -4.08 | 44.88 | +0.39 |
| gen_CNetSeg | Generative | 50.57 | -3.46 | 44.17 | -0.32 |
| gen_IP2P | Generative | 51.14 | -2.88 | 43.47 | -1.02 |
| gen_augmenters | Generative | 51.46 | -2.57 | 43.12 | -1.37 |
| gen_cycleGAN | Generative | 51.90 | -2.12 | 42.50 | -1.99 |
| gen_stargan_v2 | Generative | 50.32 | -3.70 | 43.96 | -0.53 |
| gen_CUT | Generative | 48.50 | -5.52 | 45.11 | +0.62 |
| gen_UniControl | Generative | 48.18 | -5.85 | 44.58 | +0.09 |
---

## Per-Domain Breakdown

mIoU performance on each weather domain. Normal = clear_day. Adverse = foggy, night, rainy, snowy.

| Strategy | Type | clear_day | foggy | night | rainy | snowy | Normal Avg | Adverse Avg | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Baseline | - | 59.30 | 28.41 | 46.33 | 45.11 | - | 44.79 | - |
| gen_Attribute_Hallucination | Generative | - | 59.38 | 29.24 | 46.15 | 45.74 | - | 45.13 | - |
| gen_CNetSeg | Generative | - | 58.01 | 27.69 | 47.48 | 45.27 | - | 44.61 | - |
| gen_CUT | Generative | - | 59.01 | 29.32 | 46.85 | 44.80 | - | 44.99 | - |
| gen_IP2P | Generative | - | 58.58 | 28.03 | 45.60 | 44.22 | - | 44.11 | - |
| gen_Img2Img | Generative | - | 58.80 | 28.93 | 47.75 | 46.67 | - | 45.54 | - |
| gen_Qwen_Image_Edit | Generative | - | 58.61 | 28.79 | 47.15 | 45.32 | - | 44.97 | - |
| gen_SUSTechGAN | Generative | - | 57.83 | 28.89 | 46.03 | 45.55 | - | 44.57 | - |
| gen_TSIT | Generative | - | 59.44 | 28.18 | 45.85 | 45.41 | - | 44.72 | - |
| gen_UniControl | Generative | - | 59.35 | 28.20 | 46.41 | 45.74 | - | 44.93 | - |
| gen_VisualCloze | Generative | - | 59.58 | 28.80 | 46.31 | 45.42 | - | 45.03 | - |
| gen_Weather_Effect_Generator | Generative | - | 57.79 | 28.16 | 47.71 | 45.14 | - | 44.70 | - |
| gen_albumentations_weather | Generative | - | 58.61 | 28.47 | 46.00 | 44.68 | - | 44.44 | - |
| gen_augmenters | Generative | - | 56.87 | 27.66 | 45.37 | 43.07 | - | 43.24 | - |
| gen_automold | Generative | - | 58.06 | 28.00 | 47.47 | 45.65 | - | 44.80 | - |
| gen_cycleGAN | Generative | - | 57.39 | 26.70 | 44.82 | 42.81 | - | 42.93 | - |
| gen_cyclediffusion | Generative | - | 59.19 | 29.07 | 46.17 | 45.16 | - | 44.90 | - |
| gen_flux_kontext | Generative | - | 59.00 | 29.11 | 46.76 | 45.14 | - | 45.00 | - |
| gen_stargan_v2 | Generative | - | 58.30 | 28.34 | 46.10 | 44.63 | - | 44.34 | - |
| gen_step1x_new | Generative | - | 57.38 | 27.34 | 47.44 | 45.54 | - | 44.43 | - |
| gen_step1x_v1p2 | Generative | - | 58.86 | 28.32 | 46.29 | 44.67 | - | 44.53 | - |
| std_autoaugment | Standard Aug | - | 57.65 | 27.43 | 44.75 | 44.29 | - | 43.53 | - |
| std_cutmix | Standard Aug | - | 59.44 | 28.69 | 46.23 | 44.62 | - | 44.75 | - |
| std_mixup | Standard Aug | - | 58.53 | 28.59 | 46.64 | 44.93 | - | 44.67 | - |
| std_randaugment | Standard Aug | - | 58.95 | 28.68 | 46.42 | 45.13 | - | 44.79 | - |
---

## Per-Model Breakdown

mIoU performance on each model architecture. Gain columns show improvement over baseline per model.

| Strategy | Type | deeplabv3plus_r50 | deeplabv3plus_r50_gain | mask2former_swin-b | mask2former_swin-b_gain | pspnet_r50 | pspnet_r50_gain | segformer_mit-b3 | segformer_mit-b3_gain | segnext_mscan-b | segnext_mscan-b_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_VisualCloze | Generative | - | - | 57.67 | +0.60 | 43.20 | -0.23 | 51.67 | +0.14 | 51.34 | -0.03 |
| gen_flux_kontext | Generative | - | - | 57.44 | +0.37 | 43.52 | +0.09 | 51.61 | +0.08 | 51.23 | -0.14 |
| baseline | Baseline | - | - | 57.07 | +0.00 | 43.43 | +0.00 | 51.53 | +0.00 | 51.37 | +0.00 |
| std_mixup | Standard Aug | - | - | 56.93 | -0.14 | 44.08 | +0.65 | 52.21 | +0.69 | 50.09 | -1.28 |
| std_cutmix | Standard Aug | - | - | 57.22 | +0.15 | 44.08 | +0.65 | 50.75 | -0.78 | 51.16 | -0.21 |
| std_randaugment | Standard Aug | - | - | 57.64 | +0.57 | 43.72 | +0.29 | 50.51 | -1.02 | 50.88 | -0.49 |
| gen_automold | Generative | - | - | 55.53 | -1.54 | 43.37 | -0.06 | 51.89 | +0.36 | 51.65 | +0.29 |
| gen_step1x_v1p2 | Generative | - | - | 57.21 | +0.14 | 43.78 | +0.35 | 50.60 | -0.93 | 50.74 | -0.62 |
| gen_albumentations_weather | Generative | - | - | 56.09 | -0.98 | 42.99 | -0.44 | 51.35 | -0.17 | 51.09 | -0.27 |
| gen_step1x_new | Generative | - | - | 56.22 | -0.85 | 43.26 | -0.17 | 50.39 | -1.14 | 51.00 | -0.36 |
| gen_cyclediffusion | Generative | - | - | 52.28 | -4.79 | 43.86 | +0.44 | 51.09 | -0.43 | 50.03 | -1.33 |
| gen_SUSTechGAN | Generative | - | - | 50.90 | -6.17 | 43.75 | +0.32 | 51.62 | +0.09 | 50.84 | -0.52 |
| std_autoaugment | Standard Aug | 43.97 | - | 57.86 | +0.79 | 43.66 | +0.23 | 50.41 | -1.11 | 50.12 | -1.25 |
| gen_cycleGAN | Generative | - | - | - | - | 43.75 | +0.33 | 52.40 | +0.88 | 50.14 | -1.22 |
| gen_IP2P | Generative | - | - | 51.51 | -5.56 | 42.60 | -0.83 | 50.68 | -0.84 | 49.67 | -1.70 |
| gen_Img2Img | Generative | - | - | 51.53 | -5.54 | 37.74 | -5.69 | 51.97 | +0.44 | 52.04 | +0.67 |
| gen_Attribute_Hallucination | Generative | - | - | 51.88 | -5.19 | 44.27 | +0.84 | 51.18 | -0.34 | 44.52 | -6.84 |
| gen_Qwen_Image_Edit | Generative | - | - | 50.92 | -6.15 | 37.07 | -6.35 | 51.88 | +0.35 | 51.77 | +0.40 |
| gen_TSIT | Generative | - | - | 52.10 | -4.97 | 37.03 | -6.39 | 50.44 | -1.08 | 51.04 | -0.33 |
| gen_Weather_Effect_Generator | Generative | - | - | 50.23 | -6.84 | 36.85 | -6.58 | 52.17 | +0.64 | 50.97 | -0.39 |
| gen_CNetSeg | Generative | - | - | 51.70 | -5.37 | 36.88 | -6.55 | 49.97 | -1.56 | 51.00 | -0.36 |
| gen_augmenters | Generative | - | - | - | - | 37.05 | -6.38 | 52.72 | +1.19 | 51.46 | +0.10 |
| gen_stargan_v2 | Generative | - | - | 51.02 | -6.05 | 35.34 | -8.09 | 51.36 | -0.17 | 50.33 | -1.03 |
| gen_CUT | Generative | - | - | 52.71 | -4.36 | 37.15 | -6.27 | 50.77 | -0.75 | 45.45 | -5.92 |
| gen_UniControl | Generative | - | - | 51.12 | -5.95 | 36.13 | -7.30 | 51.57 | +0.05 | 45.00 | -6.37 |
