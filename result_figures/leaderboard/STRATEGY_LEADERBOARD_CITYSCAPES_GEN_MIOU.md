# Cityscapes-Gen Strategy Leaderboard (by mIoU)

**Cityscapes-Gen**: Cityscapes Generative Augmentation + ACDC cross-domain testing

**Metric**: mIoU (Mean Intersection over Union)

**Last Updated**: 2026-02-13 15:51
**Baseline mIoU**: 52.65%
**Total Results**: 250 test results from 25 strategies

---

## Overall Strategy Ranking

Sorted by mIoU. Gain = improvement over baseline. Domain Gap = Normal mIoU - Adverse mIoU (lower is better).

| Strategy | Type | mIoU | Std | Gain | Normal mIoU | Adverse mIoU | Domain Gap | Num Tests |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_Img2Img | Generative | 52.87 | 11.44 | 0.22 | 62.39 | 43.35 | 19.05 | 10 |
| gen_augmenters | Generative | 52.82 | 11.92 | 0.18 | 62.54 | 43.11 | 19.43 | 10 |
| gen_Qwen_Image_Edit | Generative | 52.67 | 11.23 | 0.02 | 62.12 | 43.22 | 18.91 | 10 |
| gen_Attribute_Hallucination | Generative | 52.67 | 11.43 | 0.03 | 62.22 | 43.12 | 19.1 | 10 |
| baseline | Baseline | 52.65 | 11.49 | 0.0 | 62.3 | 42.99 | 19.32 | 10 |
| gen_VisualCloze | Generative | 52.58 | 11.8 | -0.06 | 62.35 | 42.81 | 19.53 | 10 |
| gen_CUT | Generative | 52.53 | 11.62 | -0.11 | 61.99 | 43.08 | 18.91 | 10 |
| gen_flux_kontext | Generative | 52.49 | 11.48 | -0.16 | 62.01 | 42.96 | 19.05 | 10 |
| gen_step1x_v1p2 | Generative | 52.45 | 11.57 | -0.2 | 62.24 | 42.66 | 19.58 | 10 |
| std_cutmix | Standard Aug | 52.45 | 11.49 | -0.19 | 62.11 | 42.8 | 19.32 | 10 |
| gen_cycleGAN | Generative | 52.43 | 11.8 | -0.21 | 62.15 | 42.71 | 19.44 | 10 |
| std_mixup | Standard Aug | 52.36 | 11.4 | -0.28 | 61.88 | 42.84 | 19.04 | 10 |
| gen_automold | Generative | 52.36 | 11.01 | -0.29 | 61.72 | 42.99 | 18.73 | 10 |
| std_randaugment | Standard Aug | 52.36 | 11.76 | -0.29 | 62.21 | 42.5 | 19.71 | 10 |
| gen_cyclediffusion | Generative | 52.34 | 11.73 | -0.31 | 62.04 | 42.64 | 19.41 | 10 |
| std_autoaugment | Standard Aug | 52.33 | 11.23 | -0.32 | 61.71 | 42.95 | 18.76 | 10 |
| gen_Weather_Effect_Generator | Generative | 52.26 | 11.05 | -0.38 | 61.68 | 42.84 | 18.84 | 10 |
| gen_SUSTechGAN | Generative | 52.24 | 11.27 | -0.41 | 61.78 | 42.7 | 19.08 | 10 |
| gen_UniControl | Generative | 52.22 | 11.3 | -0.43 | 61.63 | 42.8 | 18.83 | 10 |
| gen_TSIT | Generative | 52.2 | 11.76 | -0.44 | 61.92 | 42.49 | 19.43 | 10 |
| gen_CNetSeg | Generative | 52.19 | 11.26 | -0.46 | 61.77 | 42.6 | 19.17 | 10 |
| gen_stargan_v2 | Generative | 52.04 | 11.64 | -0.61 | 61.87 | 42.2 | 19.67 | 10 |
| gen_albumentations_weather | Generative | 52.03 | 11.56 | -0.61 | 61.76 | 42.3 | 19.46 | 10 |
| gen_IP2P | Generative | 51.84 | 12.12 | -0.8 | 61.99 | 41.69 | 20.3 | 10 |
| gen_step1x_new | Generative | 51.79 | 11.5 | -0.86 | 61.47 | 42.11 | 19.36 | 10 |
---

## Per-Dataset Breakdown

mIoU performance on each dataset.

| Strategy | Type | cityscapes | cityscapes_gain | acdc | acdc_gain |
| --- | --- | --- | --- | --- | --- |
| gen_Img2Img | Generative | 62.39 | +0.09 | 43.35 | +0.36 |
| gen_augmenters | Generative | 62.54 | +0.23 | 43.11 | +0.12 |
| gen_Qwen_Image_Edit | Generative | 62.12 | -0.18 | 43.22 | +0.23 |
| gen_Attribute_Hallucination | Generative | 62.22 | -0.08 | 43.12 | +0.14 |
| baseline | Baseline | 62.30 | +0.00 | 42.99 | +0.00 |
| gen_VisualCloze | Generative | 62.35 | +0.04 | 42.81 | -0.17 |
| gen_CUT | Generative | 61.99 | -0.32 | 43.08 | +0.09 |
| gen_flux_kontext | Generative | 62.01 | -0.29 | 42.96 | -0.02 |
| std_cutmix | Standard Aug | 62.11 | -0.19 | 42.80 | -0.19 |
| gen_step1x_v1p2 | Generative | 62.24 | -0.07 | 42.66 | -0.33 |
| gen_cycleGAN | Generative | 62.15 | -0.15 | 42.71 | -0.27 |
| std_mixup | Standard Aug | 61.88 | -0.42 | 42.84 | -0.14 |
| gen_automold | Generative | 61.72 | -0.58 | 42.99 | +0.00 |
| std_randaugment | Standard Aug | 62.21 | -0.09 | 42.50 | -0.49 |
| gen_cyclediffusion | Generative | 62.04 | -0.26 | 42.64 | -0.35 |
| std_autoaugment | Standard Aug | 61.71 | -0.59 | 42.95 | -0.04 |
| gen_Weather_Effect_Generator | Generative | 61.68 | -0.62 | 42.84 | -0.15 |
| gen_SUSTechGAN | Generative | 61.78 | -0.53 | 42.70 | -0.29 |
| gen_UniControl | Generative | 61.63 | -0.67 | 42.80 | -0.19 |
| gen_TSIT | Generative | 61.92 | -0.39 | 42.49 | -0.50 |
| gen_CNetSeg | Generative | 61.77 | -0.53 | 42.60 | -0.38 |
| gen_stargan_v2 | Generative | 61.87 | -0.43 | 42.20 | -0.78 |
| gen_albumentations_weather | Generative | 61.76 | -0.54 | 42.30 | -0.69 |
| gen_IP2P | Generative | 61.99 | -0.31 | 41.69 | -1.29 |
| gen_step1x_new | Generative | 61.47 | -0.84 | 42.11 | -0.88 |
---

## Per-Domain Breakdown

mIoU performance on each weather domain. Normal = clear_day. Adverse = foggy, night, rainy, snowy.

| Strategy | Type | clear_day | foggy | night | rainy | snowy | Normal Avg | Adverse Avg | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Baseline | - | 57.86 | 27.15 | 45.32 | 44.03 | - | 43.59 | - |
| gen_Attribute_Hallucination | Generative | - | 57.77 | 27.61 | 45.13 | 44.32 | - | 43.71 | - |
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
| gen_Img2Img | Generative | 46.58 | -0.53 | 60.25 | +0.20 | 47.60 | +0.64 | 54.89 | +0.40 | 55.03 | +0.42 |
| gen_augmenters | Generative | 46.04 | -1.08 | 61.06 | +1.01 | 46.89 | -0.07 | 55.55 | +1.05 | 54.57 | -0.03 |
| gen_Qwen_Image_Edit | Generative | 46.70 | -0.42 | 59.42 | -0.63 | 47.47 | +0.51 | 54.87 | +0.38 | 54.89 | +0.28 |
| gen_Attribute_Hallucination | Generative | 46.83 | -0.29 | 60.68 | +0.63 | 47.48 | +0.52 | 54.14 | -0.35 | 54.22 | -0.39 |
| baseline | Baseline | 47.12 | +0.00 | 60.05 | +0.00 | 46.96 | +0.00 | 54.49 | +0.00 | 54.61 | +0.00 |
| gen_VisualCloze | Generative | 46.44 | -0.68 | 60.70 | +0.65 | 46.73 | -0.24 | 54.57 | +0.08 | 54.47 | -0.14 |
| gen_CUT | Generative | 45.61 | -1.51 | 61.36 | +1.31 | 47.46 | +0.50 | 53.60 | -0.89 | 54.63 | +0.02 |
| gen_flux_kontext | Generative | 46.51 | -0.60 | 60.27 | +0.22 | 46.98 | +0.01 | 54.43 | -0.06 | 54.25 | -0.35 |
| std_cutmix | Standard Aug | 46.52 | -0.59 | 60.18 | +0.13 | 47.46 | +0.49 | 53.93 | -0.56 | 54.18 | -0.43 |
| gen_step1x_v1p2 | Generative | 47.06 | -0.06 | 60.33 | +0.28 | 47.23 | +0.26 | 53.66 | -0.83 | 53.96 | -0.64 |
| gen_cycleGAN | Generative | 45.85 | -1.26 | 60.60 | +0.55 | 47.19 | +0.23 | 54.99 | +0.50 | 53.52 | -1.08 |
| std_mixup | Standard Aug | 46.27 | -0.85 | 59.94 | -0.11 | 47.33 | +0.36 | 54.98 | +0.49 | 53.30 | -1.30 |
| gen_automold | Generative | 47.11 | -0.00 | 58.15 | -1.90 | 46.95 | -0.02 | 54.78 | +0.29 | 54.80 | +0.19 |
| std_randaugment | Standard Aug | 46.35 | -0.77 | 60.34 | +0.29 | 47.26 | +0.29 | 53.64 | -0.85 | 54.20 | -0.41 |
| gen_cyclediffusion | Generative | 46.16 | -0.96 | 61.01 | +0.96 | 47.15 | +0.19 | 54.04 | -0.45 | 53.34 | -1.26 |
| std_autoaugment | Standard Aug | 47.22 | +0.11 | 60.57 | +0.52 | 47.04 | +0.08 | 53.29 | -1.20 | 53.53 | -1.08 |
| gen_Weather_Effect_Generator | Generative | 46.75 | -0.37 | 58.03 | -2.02 | 47.33 | +0.37 | 55.00 | +0.51 | 54.21 | -0.39 |
| gen_SUSTechGAN | Generative | 46.51 | -0.60 | 58.84 | -1.21 | 47.18 | +0.22 | 54.59 | +0.10 | 54.07 | -0.53 |
| gen_UniControl | Generative | 46.56 | -0.56 | 59.50 | -0.55 | 46.55 | -0.42 | 54.32 | -0.17 | 54.16 | -0.45 |
| gen_TSIT | Generative | 45.41 | -1.71 | 60.56 | +0.51 | 47.18 | +0.21 | 53.71 | -0.78 | 54.17 | -0.44 |
| gen_CNetSeg | Generative | 47.03 | -0.08 | 59.20 | -0.85 | 47.30 | +0.33 | 53.08 | -1.41 | 54.33 | -0.28 |
| gen_stargan_v2 | Generative | 46.35 | -0.76 | 59.21 | -0.84 | 46.84 | -0.13 | 54.30 | -0.19 | 53.49 | -1.12 |
| gen_albumentations_weather | Generative | 46.05 | -1.07 | 59.00 | -1.04 | 46.60 | -0.37 | 54.39 | -0.10 | 54.13 | -0.48 |
| gen_IP2P | Generative | 45.86 | -1.25 | 60.10 | +0.05 | 46.29 | -0.68 | 53.84 | -0.65 | 53.12 | -1.49 |
| gen_step1x_new | Generative | 45.73 | -1.38 | 58.81 | -1.24 | 46.87 | -0.10 | 53.42 | -1.07 | 54.11 | -0.49 |
