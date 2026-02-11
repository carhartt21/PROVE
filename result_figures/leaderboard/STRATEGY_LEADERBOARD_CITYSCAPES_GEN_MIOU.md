# Cityscapes-Gen Strategy Leaderboard (by mIoU)

**Cityscapes-Gen**: Cityscapes Generative Augmentation + ACDC cross-domain testing

**Metric**: mIoU (Mean Intersection over Union)

**Last Updated**: 2026-02-11 11:56
**Baseline mIoU**: 49.43%
**Total Results**: 372 test results from 25 strategies

---

## Overall Strategy Ranking

Sorted by mIoU. Gain = improvement over baseline. Domain Gap = Normal mIoU - Adverse mIoU (lower is better).

| Strategy | Type | mIoU | Std | Gain | Normal mIoU | Adverse mIoU | Domain Gap | Num Tests |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_Attribute_Hallucination | Generative | 51.05 | 10.39 | 1.62 | 54.13 | 44.88 | 9.26 | 12 |
| gen_Img2Img | Generative | 49.7 | 10.88 | 0.27 | 52.87 | 43.35 | 9.52 | 15 |
| gen_augmenters | Generative | 49.58 | 11.38 | 0.16 | 52.82 | 43.11 | 9.72 | 15 |
| gen_Qwen_Image_Edit | Generative | 49.52 | 10.64 | 0.09 | 52.67 | 43.22 | 9.45 | 15 |
| baseline | Baseline | 49.43 | 10.87 | 0.0 | 52.65 | 42.99 | 9.66 | 15 |
| gen_CUT | Generative | 49.38 | 11.07 | -0.04 | 52.53 | 43.08 | 9.45 | 15 |
| gen_VisualCloze | Generative | 49.33 | 11.22 | -0.1 | 52.58 | 42.81 | 9.77 | 15 |
| gen_flux_kontext | Generative | 49.31 | 10.95 | -0.11 | 52.49 | 42.96 | 9.52 | 15 |
| gen_automold | Generative | 49.24 | 10.47 | -0.19 | 52.36 | 42.99 | 9.37 | 15 |
| std_cutmix | Standard Aug | 49.24 | 10.88 | -0.19 | 52.45 | 42.8 | 9.66 | 15 |
| std_autoaugment | Standard Aug | 49.2 | 10.65 | -0.22 | 52.33 | 42.95 | 9.38 | 15 |
| gen_step1x_v1p2 | Generative | 49.19 | 10.92 | -0.24 | 52.45 | 42.66 | 9.79 | 15 |
| gen_cycleGAN | Generative | 49.19 | 11.28 | -0.23 | 52.43 | 42.71 | 9.72 | 15 |
| std_mixup | Standard Aug | 49.19 | 10.81 | -0.24 | 52.36 | 42.84 | 9.52 | 15 |
| gen_Weather_Effect_Generator | Generative | 49.12 | 10.51 | -0.3 | 52.26 | 42.84 | 9.42 | 15 |
| gen_cyclediffusion | Generative | 49.1 | 11.15 | -0.32 | 52.34 | 42.64 | 9.7 | 15 |
| gen_UniControl | Generative | 49.08 | 10.77 | -0.35 | 52.22 | 42.8 | 9.42 | 15 |
| std_randaugment | Standard Aug | 49.07 | 11.2 | -0.35 | 52.36 | 42.5 | 9.86 | 15 |
| gen_SUSTechGAN | Generative | 49.06 | 10.72 | -0.37 | 52.24 | 42.7 | 9.54 | 15 |
| gen_CNetSeg | Generative | 48.99 | 10.71 | -0.43 | 52.19 | 42.6 | 9.59 | 15 |
| gen_TSIT | Generative | 48.97 | 11.18 | -0.46 | 52.2 | 42.49 | 9.72 | 15 |
| gen_albumentations_weather | Generative | 48.79 | 10.99 | -0.64 | 52.03 | 42.3 | 9.73 | 15 |
| gen_stargan_v2 | Generative | 48.76 | 11.12 | -0.67 | 52.04 | 42.2 | 9.84 | 15 |
| gen_step1x_new | Generative | 48.56 | 10.99 | -0.86 | 51.79 | 42.11 | 9.68 | 15 |
| gen_IP2P | Generative | 48.46 | 11.53 | -0.97 | 51.84 | 41.69 | 10.15 | 15 |
---

## Per-Dataset Breakdown

mIoU performance on each dataset.

| Strategy | Type | cityscapes | cityscapes_gain | acdc | acdc_gain |
| --- | --- | --- | --- | --- | --- |
| gen_Attribute_Hallucination | Generative | 54.13 | +1.49 | 44.88 | +1.89 |
| gen_Img2Img | Generative | 52.87 | +0.22 | 43.35 | +0.36 |
| gen_augmenters | Generative | 52.82 | +0.18 | 43.11 | +0.12 |
| gen_Qwen_Image_Edit | Generative | 52.67 | +0.02 | 43.22 | +0.23 |
| baseline | Baseline | 52.65 | +0.00 | 42.99 | +0.00 |
| gen_CUT | Generative | 52.53 | -0.11 | 43.08 | +0.09 |
| gen_flux_kontext | Generative | 52.49 | -0.16 | 42.96 | -0.02 |
| gen_VisualCloze | Generative | 52.58 | -0.06 | 42.81 | -0.17 |
| gen_automold | Generative | 52.36 | -0.29 | 42.99 | +0.00 |
| std_autoaugment | Standard Aug | 52.33 | -0.32 | 42.95 | -0.04 |
| std_cutmix | Standard Aug | 52.45 | -0.19 | 42.80 | -0.19 |
| std_mixup | Standard Aug | 52.36 | -0.28 | 42.84 | -0.14 |
| gen_cycleGAN | Generative | 52.43 | -0.21 | 42.71 | -0.27 |
| gen_step1x_v1p2 | Generative | 52.45 | -0.20 | 42.66 | -0.33 |
| gen_Weather_Effect_Generator | Generative | 52.26 | -0.38 | 42.84 | -0.15 |
| gen_UniControl | Generative | 52.22 | -0.43 | 42.80 | -0.19 |
| gen_cyclediffusion | Generative | 52.34 | -0.31 | 42.64 | -0.35 |
| gen_SUSTechGAN | Generative | 52.24 | -0.41 | 42.70 | -0.29 |
| std_randaugment | Standard Aug | 52.36 | -0.29 | 42.50 | -0.49 |
| gen_CNetSeg | Generative | 52.19 | -0.46 | 42.60 | -0.38 |
| gen_TSIT | Generative | 52.20 | -0.44 | 42.49 | -0.50 |
| gen_albumentations_weather | Generative | 52.03 | -0.61 | 42.30 | -0.69 |
| gen_stargan_v2 | Generative | 52.04 | -0.61 | 42.20 | -0.78 |
| gen_step1x_new | Generative | 51.79 | -0.86 | 42.11 | -0.88 |
| gen_IP2P | Generative | 51.84 | -0.80 | 41.69 | -1.29 |
---

## Per-Domain Breakdown

mIoU performance on each weather domain. Normal = clear_day. Adverse = foggy, night, rainy, snowy.

| Strategy | Type | clear_day | foggy | night | rainy | snowy | Normal Avg | Adverse Avg | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Baseline | - | 57.86 | 27.15 | 45.32 | 44.03 | - | 43.59 | - |
| gen_Attribute_Hallucination | Generative | - | 59.38 | 29.24 | 46.15 | 45.74 | - | 45.13 | - |
| gen_CNetSeg | Generative | - | 56.97 | 26.16 | 46.27 | 44.28 | - | 43.42 | - |
| gen_CUT | Generative | - | 57.72 | 27.50 | 45.35 | 43.53 | - | 43.53 | - |
| gen_IP2P | Generative | - | 57.22 | 26.40 | 44.49 | 42.88 | - | 42.75 | - |
| gen_Img2Img | Generative | - | 57.55 | 26.81 | 46.29 | 45.34 | - | 44.00 | - |
| gen_Qwen_Image_Edit | Generative | - | 57.49 | 27.43 | 45.87 | 44.10 | - | 43.72 | - |
| gen_SUSTechGAN | Generative | - | 56.70 | 27.28 | 44.85 | 43.93 | - | 43.19 | - |
| gen_TSIT | Generative | - | 58.01 | 26.20 | 44.67 | 44.06 | - | 43.24 | - |
| gen_UniControl | Generative | - | 57.90 | 26.67 | 45.21 | 44.43 | - | 43.55 | - |
| gen_VisualCloze | Generative | - | 57.96 | 27.32 | 45.06 | 44.04 | - | 43.60 | - |
| gen_Weather_Effect_Generator | Generative | - | 56.67 | 26.68 | 46.23 | 44.05 | - | 43.41 | - |
| gen_albumentations_weather | Generative | - | 57.12 | 26.82 | 44.62 | 43.49 | - | 43.01 | - |
| gen_augmenters | Generative | - | 57.81 | 27.50 | 45.71 | 43.97 | - | 43.75 | - |
| gen_automold | Generative | - | 57.18 | 26.70 | 46.24 | 44.48 | - | 43.65 | - |
| gen_cycleGAN | Generative | - | 57.95 | 27.14 | 45.28 | 43.95 | - | 43.58 | - |
| gen_cyclediffusion | Generative | - | 57.86 | 27.35 | 44.74 | 43.81 | - | 43.44 | - |
| gen_flux_kontext | Generative | - | 57.64 | 27.49 | 45.39 | 43.54 | - | 43.52 | - |
| gen_stargan_v2 | Generative | - | 56.67 | 26.71 | 44.72 | 43.62 | - | 42.93 | - |
| gen_step1x_new | Generative | - | 56.11 | 25.82 | 45.85 | 44.01 | - | 42.95 | - |
| gen_step1x_v1p2 | Generative | - | 57.42 | 26.98 | 45.37 | 43.56 | - | 43.33 | - |
| std_autoaugment | Standard Aug | - | 57.65 | 27.43 | 44.75 | 44.29 | - | 43.53 | - |
| std_cutmix | Standard Aug | - | 57.82 | 27.32 | 45.01 | 43.06 | - | 43.30 | - |
| std_mixup | Standard Aug | - | 57.34 | 27.01 | 45.45 | 43.68 | - | 43.37 | - |
| std_randaugment | Standard Aug | - | 57.78 | 27.12 | 45.12 | 43.76 | - | 43.45 | - |
---

## Per-Model Breakdown

mIoU performance on each model architecture. Gain columns show improvement over baseline per model.

| Strategy | Type | deeplabv3plus_r50 | deeplabv3plus_r50_gain | mask2former_swin-b | mask2former_swin-b_gain | pspnet_r50 | pspnet_r50_gain | segformer_mit-b3 | segformer_mit-b3_gain | segnext_mscan-b | segnext_mscan-b_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_Attribute_Hallucination | Generative | - | - | 57.75 | +0.68 | 44.27 | +0.84 | 51.18 | -0.34 | 50.99 | -0.38 |
| gen_Img2Img | Generative | 42.82 | -0.92 | 57.34 | +0.27 | 44.31 | +0.88 | 51.97 | +0.44 | 52.04 | +0.67 |
| gen_augmenters | Generative | 42.04 | -1.70 | 58.08 | +1.01 | 43.61 | +0.18 | 52.72 | +1.19 | 51.46 | +0.10 |
| gen_Qwen_Image_Edit | Generative | 43.35 | -0.38 | 56.58 | -0.49 | 44.01 | +0.58 | 51.88 | +0.35 | 51.77 | +0.40 |
| baseline | Baseline | 43.74 | +0.00 | 57.07 | +0.00 | 43.43 | +0.00 | 51.53 | +0.00 | 51.37 | +0.00 |
| gen_CUT | Generative | 42.06 | -1.67 | 58.48 | +1.41 | 44.03 | +0.60 | 50.77 | -0.75 | 51.57 | +0.20 |
| gen_VisualCloze | Generative | 42.75 | -0.99 | 57.67 | +0.60 | 43.20 | -0.23 | 51.67 | +0.14 | 51.34 | -0.03 |
| gen_flux_kontext | Generative | 42.77 | -0.97 | 57.44 | +0.37 | 43.52 | +0.09 | 51.61 | +0.08 | 51.23 | -0.14 |
| gen_automold | Generative | 43.74 | +0.00 | 55.53 | -1.54 | 43.37 | -0.06 | 51.89 | +0.36 | 51.65 | +0.29 |
| std_cutmix | Standard Aug | 42.97 | -0.76 | 57.22 | +0.15 | 44.08 | +0.65 | 50.75 | -0.78 | 51.16 | -0.21 |
| std_autoaugment | Standard Aug | 43.97 | +0.23 | 57.86 | +0.79 | 43.66 | +0.23 | 50.41 | -1.11 | 50.12 | -1.25 |
| std_mixup | Standard Aug | 42.64 | -1.10 | 56.93 | -0.14 | 44.08 | +0.65 | 52.21 | +0.69 | 50.09 | -1.28 |
| gen_cycleGAN | Generative | 41.94 | -1.79 | 57.71 | +0.64 | 43.75 | +0.33 | 52.40 | +0.88 | 50.14 | -1.22 |
| gen_step1x_v1p2 | Generative | 43.59 | -0.14 | 57.21 | +0.14 | 43.78 | +0.35 | 50.60 | -0.93 | 50.74 | -0.62 |
| gen_Weather_Effect_Generator | Generative | 43.20 | -0.53 | 55.43 | -1.64 | 43.84 | +0.41 | 52.17 | +0.64 | 50.97 | -0.39 |
| gen_cyclediffusion | Generative | 42.43 | -1.31 | 58.10 | +1.03 | 43.86 | +0.44 | 51.09 | -0.43 | 50.03 | -1.33 |
| gen_UniControl | Generative | 42.93 | -0.81 | 56.70 | -0.37 | 43.07 | -0.35 | 51.57 | +0.05 | 51.10 | -0.26 |
| std_randaugment | Standard Aug | 42.62 | -1.12 | 57.64 | +0.57 | 43.72 | +0.29 | 50.51 | -1.02 | 50.88 | -0.49 |
| gen_SUSTechGAN | Generative | 42.89 | -0.85 | 56.20 | -0.87 | 43.75 | +0.32 | 51.62 | +0.09 | 50.84 | -0.52 |
| gen_CNetSeg | Generative | 43.47 | -0.27 | 56.70 | -0.37 | 43.83 | +0.40 | 49.97 | -1.56 | 51.00 | -0.36 |
| gen_TSIT | Generative | 41.81 | -1.93 | 57.74 | +0.67 | 43.80 | +0.37 | 50.44 | -1.08 | 51.04 | -0.33 |
| gen_albumentations_weather | Generative | 42.42 | -1.31 | 56.09 | -0.98 | 42.99 | -0.44 | 51.35 | -0.17 | 51.09 | -0.27 |
| gen_stargan_v2 | Generative | 42.62 | -1.11 | 56.48 | -0.59 | 43.00 | -0.42 | 51.36 | -0.17 | 50.33 | -1.03 |
| gen_step1x_new | Generative | 41.94 | -1.80 | 56.22 | -0.85 | 43.26 | -0.17 | 50.39 | -1.14 | 51.00 | -0.36 |
| gen_IP2P | Generative | 42.10 | -1.63 | 57.24 | +0.17 | 42.60 | -0.83 | 50.68 | -0.84 | 49.67 | -1.70 |
