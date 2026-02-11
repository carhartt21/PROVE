# Stage 1 Strategy Leaderboard (by mIoU)

**Stage 1**: Clear Day Training — cross-domain robustness evaluation

**Metric**: mIoU (Mean Intersection over Union)

**Last Updated**: 2026-02-11 22:03
**Baseline mIoU**: 37.61%
**Total Results**: 379 test results from 26 strategies

---

## Overall Strategy Ranking

Sorted by mIoU. Gain = improvement over baseline. Domain Gap = Normal mIoU - Adverse mIoU (lower is better).

| Strategy | Type | mIoU | Std | Gain | Normal mIoU | Adverse mIoU | Domain Gap | Num Tests |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_UniControl | Generative | 40.12 | 6.39 | 2.51 | 40.62 | 34.9 | 5.72 | 14 |
| std_randaugment | Standard Aug | 40.04 | 5.59 | 2.43 | 40.71 | 34.93 | 5.78 | 16 |
| gen_Img2Img | Generative | 39.99 | 6.24 | 2.37 | 40.74 | 34.58 | 6.16 | 14 |
| std_autoaugment | Standard Aug | 39.99 | 5.85 | 2.37 | 40.45 | 34.88 | 5.56 | 16 |
| gen_Attribute_Hallucination | Generative | 39.94 | 6.11 | 2.33 | 40.49 | 34.51 | 5.97 | 14 |
| std_cutmix | Standard Aug | 39.91 | 5.76 | 2.3 | 40.64 | 34.98 | 5.66 | 16 |
| gen_Qwen_Image_Edit | Generative | 39.86 | 5.97 | 2.24 | 40.51 | 34.51 | 6.01 | 14 |
| gen_CNetSeg | Generative | 39.86 | 6.08 | 2.25 | 40.53 | 34.55 | 5.99 | 14 |
| gen_stargan_v2 | Generative | 39.86 | 6.15 | 2.25 | 40.55 | 34.54 | 6.0 | 14 |
| gen_flux_kontext | Generative | 39.82 | 5.34 | 2.2 | 40.47 | 34.78 | 5.68 | 16 |
| std_mixup | Standard Aug | 39.81 | 5.83 | 2.2 | 40.39 | 34.61 | 5.78 | 16 |
| gen_IP2P | Generative | 39.79 | 5.88 | 2.17 | 40.5 | 34.6 | 5.9 | 14 |
| gen_augmenters | Generative | 39.79 | 5.9 | 2.18 | 40.39 | 34.46 | 5.92 | 14 |
| gen_cyclediffusion | Generative | 39.74 | 5.86 | 2.12 | 40.56 | 34.63 | 5.93 | 14 |
| gen_VisualCloze | Generative | 39.67 | 5.79 | 2.05 | 40.5 | 34.57 | 5.93 | 14 |
| gen_CUT | Generative | 39.62 | 5.79 | 2.0 | 40.33 | 34.38 | 5.95 | 14 |
| gen_Weather_Effect_Generator | Generative | 39.59 | 5.7 | 1.98 | 40.47 | 34.35 | 6.12 | 14 |
| gen_step1x_new | Generative | 39.44 | 5.57 | 1.82 | 40.11 | 34.31 | 5.81 | 16 |
| gen_cycleGAN | Generative | 39.31 | 4.89 | 1.69 | 39.96 | 34.43 | 5.53 | 16 |
| gen_SUSTechGAN | Generative | 39.31 | 5.51 | 1.7 | 39.94 | 34.06 | 5.88 | 14 |
| gen_TSIT | Generative | 39.24 | 6.91 | 1.63 | 40.02 | 33.94 | 6.08 | 14 |
| gen_automold | Generative | 39.23 | 5.76 | 1.61 | 39.85 | 33.85 | 6.0 | 13 |
| gen_LANIT | Generative | 39.03 | 5.7 | 1.42 | 39.86 | 33.76 | 6.11 | 14 |
| gen_albumentations_weather | Generative | 38.87 | 5.25 | 1.26 | 39.42 | 33.42 | 5.99 | 13 |
| gen_step1x_v1p2 | Generative | 38.78 | 4.9 | 1.16 | 39.22 | 33.65 | 5.57 | 15 |
| baseline | Baseline | 37.61 | 6.55 | 0.0 | 38.17 | 32.69 | 5.48 | 16 |
---

## Per-Dataset Breakdown

mIoU performance on each dataset.

| Strategy | Type | bdd10k | bdd10k_gain | iddaw | iddaw_gain | mapillaryvistas | mapillaryvistas_gain | outside15k | outside15k_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| std_randaugment | Standard Aug | 46.16 | +3.88 | 37.74 | +1.75 | 34.73 | +1.68 | 41.55 | +2.41 |
| std_autoaugment | Standard Aug | 46.38 | +4.10 | 36.94 | +0.95 | 34.75 | +1.70 | 41.89 | +2.75 |
| std_cutmix | Standard Aug | 46.28 | +3.99 | 37.06 | +1.07 | 34.67 | +1.63 | 41.65 | +2.51 |
| gen_flux_kontext | Generative | 44.53 | +2.25 | 37.80 | +1.82 | 34.91 | +1.87 | 42.01 | +2.87 |
| std_mixup | Standard Aug | 46.37 | +4.09 | 37.74 | +1.75 | 33.46 | +0.41 | 41.67 | +2.53 |
| gen_UniControl | Generative | 47.22 | +4.94 | 38.10 | +2.11 | 33.01 | -0.03 | 40.45 | +1.31 |
| gen_Img2Img | Generative | 46.85 | +4.57 | 38.13 | +2.14 | 32.93 | -0.11 | 40.37 | +1.23 |
| gen_Attribute_Hallucination | Generative | 46.70 | +4.42 | 37.74 | +1.75 | 32.91 | -0.14 | 40.90 | +1.76 |
| gen_Qwen_Image_Edit | Generative | 46.36 | +4.08 | 37.82 | +1.83 | 32.88 | -0.17 | 40.89 | +1.75 |
| gen_stargan_v2 | Generative | 46.55 | +4.27 | 37.68 | +1.69 | 32.71 | -0.33 | 41.00 | +1.85 |
| gen_CNetSeg | Generative | 46.59 | +4.31 | 37.70 | +1.71 | 32.87 | -0.17 | 40.77 | +1.63 |
| gen_step1x_new | Generative | 44.70 | +2.42 | 37.87 | +1.88 | 33.51 | +0.46 | 41.68 | +2.54 |
| gen_augmenters | Generative | 46.13 | +3.85 | 37.75 | +1.76 | 32.87 | -0.17 | 40.98 | +1.84 |
| gen_IP2P | Generative | 46.28 | +4.00 | 37.81 | +1.82 | 32.88 | -0.17 | 40.67 | +1.53 |
| gen_cyclediffusion | Generative | 46.06 | +3.78 | 38.04 | +2.05 | 32.68 | -0.37 | 40.63 | +1.49 |
| gen_automold | Generative | 46.07 | +3.79 | 37.81 | +1.83 | 32.90 | -0.14 | 40.59 | +1.45 |
| gen_cycleGAN | Generative | 43.27 | +0.99 | 37.85 | +1.86 | 34.79 | +1.74 | 41.32 | +2.18 |
| gen_VisualCloze | Generative | 45.74 | +3.46 | 38.03 | +2.04 | 32.94 | -0.11 | 40.48 | +1.34 |
| gen_CUT | Generative | 45.98 | +3.70 | 37.74 | +1.75 | 32.68 | -0.37 | 40.57 | +1.43 |
| gen_Weather_Effect_Generator | Generative | 45.82 | +3.54 | 37.59 | +1.60 | 33.04 | -0.01 | 40.52 | +1.38 |
| gen_step1x_v1p2 | Generative | 43.03 | +0.75 | 37.77 | +1.78 | 33.52 | +0.47 | 41.85 | +2.70 |
| gen_SUSTechGAN | Generative | 44.84 | +2.56 | 37.84 | +1.85 | 32.77 | -0.27 | 40.44 | +1.30 |
| gen_albumentations_weather | Generative | 45.01 | +2.73 | 37.71 | +1.72 | 32.76 | -0.28 | 40.39 | +1.25 |
| gen_TSIT | Generative | 46.56 | +4.28 | 37.71 | +1.72 | 30.09 | -2.96 | 40.68 | +1.54 |
| gen_LANIT | Generative | 44.56 | +2.28 | 36.89 | +0.90 | 32.88 | -0.17 | 40.66 | +1.52 |
| baseline | Baseline | 42.28 | +0.00 | 35.99 | +0.00 | 33.05 | -0.00 | 39.14 | -0.00 |
---

## Per-Domain Breakdown

mIoU performance on each weather domain. Normal = clear_day, cloudy. Adverse = foggy, night, rainy, snowy.

| Strategy | Type | clear_day | cloudy | dawn_dusk | foggy | night | rainy | snowy | Normal Avg | Adverse Avg | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Baseline | 39.42 | 36.91 | 32.16 | 36.81 | 26.87 | 34.02 | 33.05 | 38.17 | 32.69 | 5.48 |
| gen_Attribute_Hallucination | Generative | 41.97 | 39.00 | 33.60 | 37.92 | 28.23 | 35.88 | 36.01 | 40.49 | 34.51 | 5.97 |
| gen_CNetSeg | Generative | 41.66 | 39.41 | 33.45 | 37.95 | 28.07 | 35.94 | 36.23 | 40.53 | 34.55 | 5.99 |
| gen_CUT | Generative | 41.61 | 39.05 | 33.39 | 38.05 | 28.60 | 35.39 | 35.49 | 40.33 | 34.38 | 5.95 |
| gen_IP2P | Generative | 41.74 | 39.25 | 33.42 | 38.12 | 28.67 | 35.79 | 35.82 | 40.50 | 34.60 | 5.90 |
| gen_Img2Img | Generative | 41.88 | 39.61 | 33.64 | 37.88 | 28.19 | 36.49 | 35.79 | 40.74 | 34.58 | 6.16 |
| gen_LANIT | Generative | 41.09 | 38.64 | 32.76 | 37.69 | 27.52 | 34.75 | 35.07 | 39.86 | 33.76 | 6.11 |
| gen_Qwen_Image_Edit | Generative | 41.77 | 39.26 | 33.73 | 38.11 | 28.63 | 35.84 | 35.44 | 40.51 | 34.51 | 6.01 |
| gen_SUSTechGAN | Generative | 41.28 | 38.60 | 33.09 | 37.50 | 27.86 | 35.24 | 35.63 | 39.94 | 34.06 | 5.88 |
| gen_TSIT | Generative | 41.17 | 38.87 | 32.72 | 37.24 | 28.27 | 35.10 | 35.15 | 40.02 | 33.94 | 6.08 |
| gen_UniControl | Generative | 41.93 | 39.31 | 33.31 | 38.59 | 28.49 | 36.54 | 35.98 | 40.62 | 34.90 | 5.72 |
| gen_VisualCloze | Generative | 41.70 | 39.29 | 33.57 | 38.70 | 28.57 | 35.58 | 35.43 | 40.50 | 34.57 | 5.93 |
| gen_Weather_Effect_Generator | Generative | 41.59 | 39.36 | 33.86 | 37.86 | 28.37 | 35.55 | 35.62 | 40.47 | 34.35 | 6.12 |
| gen_albumentations_weather | Generative | 40.92 | 37.91 | 33.18 | 37.29 | 28.08 | 34.59 | 33.74 | 39.42 | 33.42 | 5.99 |
| gen_augmenters | Generative | 41.72 | 39.06 | 33.75 | 37.71 | 28.35 | 35.73 | 36.07 | 40.39 | 34.46 | 5.92 |
| gen_automold | Generative | 41.25 | 38.46 | 32.64 | 38.03 | 28.51 | 34.67 | 34.19 | 39.85 | 33.85 | 6.00 |
| gen_cycleGAN | Generative | 41.22 | 38.71 | 33.93 | 38.76 | 28.84 | 35.33 | 34.80 | 39.96 | 34.43 | 5.53 |
| gen_cyclediffusion | Generative | 41.67 | 39.44 | 33.59 | 38.51 | 28.73 | 35.51 | 35.75 | 40.56 | 34.63 | 5.93 |
| gen_flux_kontext | Generative | 41.73 | 39.20 | 34.57 | 38.43 | 29.26 | 36.08 | 35.37 | 40.47 | 34.78 | 5.68 |
| gen_stargan_v2 | Generative | 41.76 | 39.33 | 33.41 | 38.51 | 27.94 | 35.83 | 35.89 | 40.55 | 34.54 | 6.00 |
| gen_step1x_new | Generative | 41.32 | 38.91 | 33.41 | 38.39 | 28.43 | 35.38 | 35.02 | 40.11 | 34.31 | 5.81 |
| gen_step1x_v1p2 | Generative | 40.68 | 37.77 | 33.39 | 37.66 | 28.35 | 34.63 | 33.97 | 39.22 | 33.65 | 5.57 |
| std_autoaugment | Standard Aug | 41.81 | 39.08 | 33.83 | 38.60 | 28.90 | 36.28 | 35.76 | 40.45 | 34.88 | 5.56 |
| std_cutmix | Standard Aug | 41.67 | 39.60 | 33.87 | 38.19 | 29.19 | 36.49 | 36.05 | 40.64 | 34.98 | 5.66 |
| std_mixup | Standard Aug | 41.64 | 39.14 | 33.64 | 38.07 | 28.54 | 36.15 | 35.70 | 40.39 | 34.61 | 5.78 |
| std_randaugment | Standard Aug | 41.93 | 39.48 | 34.00 | 38.57 | 28.90 | 36.21 | 36.03 | 40.71 | 34.93 | 5.78 |
---

## Per-Model Breakdown

mIoU performance on each model architecture. Gain columns show improvement over baseline per model.

| Strategy | Type | mask2former_swin-b | mask2former_swin-b_gain | pspnet_r50 | pspnet_r50_gain | segformer_mit-b3 | segformer_mit-b3_gain | segnext_mscan-b | segnext_mscan-b_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_UniControl | Generative | 46.91 | +2.17 | 34.74 | +2.66 | 41.11 | +4.90 | 41.12 | +3.69 |
| gen_Img2Img | Generative | 46.60 | +1.87 | 34.77 | +2.69 | 40.67 | +4.46 | 41.22 | +3.79 |
| gen_Attribute_Hallucination | Generative | 45.23 | +0.50 | 34.88 | +2.79 | 41.23 | +5.02 | 41.08 | +3.65 |
| gen_stargan_v2 | Generative | 45.35 | +0.61 | 34.82 | +2.74 | 40.81 | +4.60 | 41.21 | +3.78 |
| gen_CNetSeg | Generative | 45.00 | +0.27 | 34.77 | +2.69 | 41.16 | +4.95 | 41.09 | +3.66 |
| gen_Qwen_Image_Edit | Generative | 44.96 | +0.22 | 34.83 | +2.75 | 41.10 | +4.89 | 41.09 | +3.66 |
| gen_VisualCloze | Generative | 46.15 | +1.41 | 34.96 | +2.88 | 40.69 | +4.48 | 40.11 | +2.68 |
| gen_cyclediffusion | Generative | 45.07 | +0.33 | 34.85 | +2.77 | 40.61 | +4.40 | 41.08 | +3.65 |
| gen_augmenters | Generative | 44.67 | -0.06 | 34.80 | +2.72 | 40.87 | +4.67 | 41.26 | +3.83 |
| gen_IP2P | Generative | 44.68 | -0.06 | 34.90 | +2.82 | 41.00 | +4.79 | 41.01 | +3.58 |
| gen_Weather_Effect_Generator | Generative | 44.32 | -0.41 | 34.75 | +2.67 | 40.76 | +4.55 | 40.92 | +3.49 |
| gen_CUT | Generative | 43.86 | -0.87 | 34.84 | +2.76 | 40.92 | +4.71 | 40.96 | +3.53 |
| std_randaugment | Standard Aug | 43.18 | -1.56 | 34.75 | +2.67 | 40.88 | +4.67 | 41.37 | +3.94 |
| gen_SUSTechGAN | Generative | 44.93 | +0.19 | 34.77 | +2.69 | 40.79 | +4.58 | 39.56 | +2.12 |
| gen_TSIT | Generative | 45.36 | +0.62 | 34.86 | +2.78 | 38.79 | +2.58 | 41.02 | +3.59 |
| std_autoaugment | Standard Aug | 43.93 | -0.81 | 34.86 | +2.78 | 40.92 | +4.71 | 40.25 | +2.82 |
| gen_automold | Generative | 45.07 | +0.33 | 34.84 | +2.76 | 41.14 | +4.93 | 38.62 | +1.19 |
| std_cutmix | Standard Aug | 43.73 | -1.00 | 34.81 | +2.72 | 39.97 | +3.76 | 41.14 | +3.71 |
| gen_flux_kontext | Generative | 43.84 | -0.89 | 34.88 | +2.80 | 39.41 | +3.20 | 41.12 | +3.69 |
| std_mixup | Standard Aug | 42.56 | -2.18 | 34.81 | +2.73 | 40.80 | +4.59 | 41.07 | +3.64 |
| gen_LANIT | Generative | 45.17 | +0.43 | 34.82 | +2.73 | 39.14 | +2.93 | 40.07 | +2.64 |
| gen_step1x_new | Generative | 42.78 | -1.96 | 34.81 | +2.73 | 39.03 | +2.82 | 41.14 | +3.71 |
| gen_albumentations_weather | Generative | 43.28 | -1.45 | 35.03 | +2.95 | 40.63 | +4.42 | 38.70 | +1.27 |
| gen_cycleGAN | Generative | 43.46 | -1.27 | 34.84 | +2.76 | 39.23 | +3.03 | 39.70 | +2.27 |
| gen_step1x_v1p2 | Generative | 42.11 | -2.63 | 34.96 | +2.88 | 39.31 | +3.10 | 38.71 | +1.28 |
| baseline | Baseline | 44.74 | +0.00 | 32.08 | +0.00 | 36.21 | +0.00 | 37.43 | +0.00 |
