# Stage 1 Strategy Leaderboard (by mIoU)

**Stage 1**: Clear Day Training — cross-domain robustness evaluation

**Metric**: mIoU (Mean Intersection over Union)

**Last Updated**: 2026-02-10 14:41
**Baseline mIoU**: 33.63%
**Total Results**: 364 test results from 26 strategies

---

## Overall Strategy Ranking

Sorted by mIoU. Gain = improvement over baseline. Domain Gap = Normal mIoU - Adverse mIoU (lower is better).

| Strategy | Type | mIoU | Std | Gain | Normal mIoU | Adverse mIoU | Domain Gap | Num Tests |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_Img2Img | Generative | 39.99 | 6.24 | 6.35 | 40.74 | 34.58 | 6.16 | 14 |
| gen_UniControl | Generative | 39.96 | 6.08 | 6.33 | 40.58 | 34.59 | 5.99 | 14 |
| gen_Attribute_Hallucination | Generative | 39.94 | 6.11 | 6.31 | 40.49 | 34.51 | 5.97 | 14 |
| gen_Qwen_Image_Edit | Generative | 39.86 | 5.97 | 6.22 | 40.51 | 34.51 | 6.01 | 14 |
| gen_stargan_v2 | Generative | 39.86 | 6.15 | 6.23 | 40.55 | 34.54 | 6.0 | 14 |
| gen_CNetSeg | Generative | 39.86 | 6.08 | 6.23 | 40.53 | 34.55 | 5.99 | 14 |
| gen_IP2P | Generative | 39.79 | 5.88 | 6.15 | 40.5 | 34.6 | 5.9 | 14 |
| gen_augmenters | Generative | 39.79 | 5.9 | 6.16 | 40.39 | 34.46 | 5.92 | 14 |
| std_randaugment | Standard Aug | 39.77 | 5.92 | 6.14 | 40.44 | 34.33 | 6.11 | 14 |
| std_mixup | Standard Aug | 39.73 | 5.94 | 6.09 | 40.43 | 34.49 | 5.94 | 14 |
| gen_VisualCloze | Generative | 39.67 | 5.79 | 6.03 | 40.5 | 34.57 | 5.93 | 14 |
| gen_cyclediffusion | Generative | 39.64 | 5.84 | 6.01 | 40.45 | 34.61 | 5.84 | 14 |
| gen_CUT | Generative | 39.62 | 5.79 | 5.98 | 40.33 | 34.38 | 5.95 | 14 |
| gen_Weather_Effect_Generator | Generative | 39.59 | 5.7 | 5.96 | 40.47 | 34.35 | 6.12 | 14 |
| std_autoaugment | Standard Aug | 39.57 | 6.09 | 5.93 | 40.15 | 34.35 | 5.79 | 14 |
| std_cutmix | Standard Aug | 39.54 | 6.0 | 5.91 | 40.31 | 34.27 | 6.04 | 14 |
| gen_flux_kontext | Generative | 39.35 | 5.49 | 5.71 | 40.17 | 34.16 | 6.01 | 14 |
| gen_SUSTechGAN | Generative | 39.31 | 5.51 | 5.68 | 39.94 | 34.06 | 5.88 | 14 |
| gen_step1x_new | Generative | 39.31 | 5.62 | 5.68 | 40.14 | 34.12 | 6.02 | 14 |
| gen_TSIT | Generative | 39.24 | 6.91 | 5.61 | 40.02 | 33.94 | 6.08 | 14 |
| gen_automold | Generative | 39.23 | 5.76 | 5.59 | 39.85 | 33.85 | 6.0 | 13 |
| gen_LANIT | Generative | 39.03 | 5.7 | 5.4 | 39.86 | 33.76 | 6.11 | 14 |
| gen_cycleGAN | Generative | 38.93 | 5.1 | 5.3 | 39.76 | 33.79 | 5.97 | 14 |
| gen_albumentations_weather | Generative | 38.87 | 5.25 | 5.24 | 39.42 | 33.42 | 5.99 | 13 |
| gen_step1x_v1p2 | Generative | 38.49 | 4.79 | 4.85 | 39.12 | 33.19 | 5.93 | 13 |
| baseline | Baseline | 33.63 | 9.4 | 0.0 | 34.24 | 28.89 | 5.35 | 17 |
---

## Per-Dataset Breakdown

mIoU performance on each dataset.

| Strategy | Type | bdd10k | bdd10k_gain | idd-aw | idd-aw_gain | mapillaryvistas | mapillaryvistas_gain | outside15k | outside15k_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_Attribute_Hallucination | Generative | 46.70 | +4.42 | - | - | 32.91 | +6.26 | 40.90 | +8.04 |
| gen_stargan_v2 | Generative | 46.55 | +4.27 | - | - | 32.71 | +6.07 | 41.00 | +8.14 |
| gen_CNetSeg | Generative | 46.59 | +4.31 | - | - | 32.87 | +6.22 | 40.77 | +7.92 |
| gen_Img2Img | Generative | 46.85 | +4.57 | - | - | 32.93 | +6.28 | 40.37 | +7.52 |
| gen_UniControl | Generative | 46.68 | +4.40 | - | - | 33.01 | +6.37 | 40.45 | +7.60 |
| gen_Qwen_Image_Edit | Generative | 46.36 | +4.08 | - | - | 32.88 | +6.23 | 40.89 | +8.03 |
| gen_augmenters | Generative | 46.13 | +3.85 | - | - | 32.87 | +6.22 | 40.98 | +8.12 |
| std_autoaugment | Standard Aug | 46.38 | +4.10 | - | - | 32.82 | +6.17 | 40.75 | +7.89 |
| std_randaugment | Standard Aug | 46.16 | +3.88 | - | - | 32.83 | +6.18 | 40.89 | +8.04 |
| gen_IP2P | Generative | 46.28 | +4.00 | - | - | 32.88 | +6.23 | 40.67 | +7.82 |
| std_cutmix | Standard Aug | 46.28 | +3.99 | - | - | 32.90 | +6.25 | 40.51 | +7.66 |
| std_mixup | Standard Aug | 46.37 | +4.09 | - | - | 32.83 | +6.18 | 40.43 | +7.57 |
| gen_automold | Generative | 46.07 | +3.79 | - | - | 32.90 | +6.25 | 40.59 | +7.74 |
| gen_Weather_Effect_Generator | Generative | 45.82 | +3.54 | - | - | 33.04 | +6.39 | 40.52 | +7.67 |
| gen_cyclediffusion | Generative | 46.06 | +3.78 | - | - | 32.68 | +6.03 | 40.63 | +7.78 |
| gen_CUT | Generative | 45.98 | +3.70 | - | - | 32.68 | +6.03 | 40.57 | +7.71 |
| gen_VisualCloze | Generative | 45.74 | +3.46 | - | - | 32.94 | +6.29 | 40.48 | +7.62 |
| gen_flux_kontext | Generative | 44.53 | +2.25 | - | - | 32.98 | +6.33 | 40.86 | +8.00 |
| gen_albumentations_weather | Generative | 45.01 | +2.73 | - | - | 32.76 | +6.11 | 40.39 | +7.54 |
| gen_LANIT | Generative | 44.56 | +2.28 | - | - | 32.88 | +6.23 | 40.66 | +7.81 |
| gen_step1x_new | Generative | 44.70 | +2.42 | - | - | 32.94 | +6.29 | 40.44 | +7.59 |
| gen_SUSTechGAN | Generative | 44.84 | +2.56 | - | - | 32.77 | +6.12 | 40.44 | +7.58 |
| gen_TSIT | Generative | 46.56 | +4.28 | - | - | 30.09 | +3.44 | 40.68 | +7.83 |
| gen_cycleGAN | Generative | 43.27 | +0.99 | - | - | 32.90 | +6.25 | 40.61 | +7.76 |
| gen_step1x_v1p2 | Generative | 43.03 | +0.75 | - | - | 32.85 | +6.20 | 40.53 | +7.68 |
| baseline | Baseline | 42.28 | +0.00 | - | - | 26.65 | +0.00 | 32.85 | +0.00 |
---

## Per-Domain Breakdown

mIoU performance on each weather domain. Normal = clear_day, cloudy. Adverse = foggy, night, rainy, snowy.

| Strategy | Type | clear_day | cloudy | dawn_dusk | foggy | night | rainy | snowy | Normal Avg | Adverse Avg | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Baseline | 35.42 | 33.05 | 28.25 | 32.55 | 23.80 | 30.02 | 29.17 | 34.24 | 28.89 | 5.35 |
| gen_Attribute_Hallucination | Generative | 41.97 | 39.00 | 33.60 | 37.92 | 28.23 | 35.88 | 36.01 | 40.49 | 34.51 | 5.97 |
| gen_CNetSeg | Generative | 41.66 | 39.41 | 33.45 | 37.95 | 28.07 | 35.94 | 36.23 | 40.53 | 34.55 | 5.99 |
| gen_CUT | Generative | 41.61 | 39.05 | 33.39 | 38.05 | 28.60 | 35.39 | 35.49 | 40.33 | 34.38 | 5.95 |
| gen_IP2P | Generative | 41.74 | 39.25 | 33.42 | 38.12 | 28.67 | 35.79 | 35.82 | 40.50 | 34.60 | 5.90 |
| gen_Img2Img | Generative | 41.88 | 39.61 | 33.64 | 37.88 | 28.19 | 36.49 | 35.79 | 40.74 | 34.58 | 6.16 |
| gen_LANIT | Generative | 41.09 | 38.64 | 32.76 | 37.69 | 27.52 | 34.75 | 35.07 | 39.86 | 33.76 | 6.11 |
| gen_Qwen_Image_Edit | Generative | 41.77 | 39.26 | 33.73 | 38.11 | 28.63 | 35.84 | 35.44 | 40.51 | 34.51 | 6.01 |
| gen_SUSTechGAN | Generative | 41.28 | 38.60 | 33.09 | 37.50 | 27.86 | 35.24 | 35.63 | 39.94 | 34.06 | 5.88 |
| gen_TSIT | Generative | 41.17 | 38.87 | 32.72 | 37.24 | 28.27 | 35.10 | 35.15 | 40.02 | 33.94 | 6.08 |
| gen_UniControl | Generative | 41.92 | 39.24 | 33.56 | 38.35 | 28.16 | 36.04 | 35.82 | 40.58 | 34.59 | 5.99 |
| gen_VisualCloze | Generative | 41.70 | 39.29 | 33.57 | 38.70 | 28.57 | 35.58 | 35.43 | 40.50 | 34.57 | 5.93 |
| gen_Weather_Effect_Generator | Generative | 41.59 | 39.36 | 33.86 | 37.86 | 28.37 | 35.55 | 35.62 | 40.47 | 34.35 | 6.12 |
| gen_albumentations_weather | Generative | 40.92 | 37.91 | 33.18 | 37.29 | 28.08 | 34.59 | 33.74 | 39.42 | 33.42 | 5.99 |
| gen_augmenters | Generative | 41.72 | 39.06 | 33.75 | 37.71 | 28.35 | 35.73 | 36.07 | 40.39 | 34.46 | 5.92 |
| gen_automold | Generative | 41.25 | 38.46 | 32.64 | 38.03 | 28.51 | 34.67 | 34.19 | 39.85 | 33.85 | 6.00 |
| gen_cycleGAN | Generative | 40.92 | 38.61 | 33.25 | 38.08 | 27.95 | 34.62 | 34.53 | 39.76 | 33.79 | 5.97 |
| gen_cyclediffusion | Generative | 41.53 | 39.37 | 33.55 | 38.53 | 28.61 | 35.58 | 35.73 | 40.45 | 34.61 | 5.84 |
| gen_flux_kontext | Generative | 41.34 | 39.00 | 33.74 | 37.93 | 28.22 | 35.44 | 35.05 | 40.17 | 34.16 | 6.01 |
| gen_stargan_v2 | Generative | 41.76 | 39.33 | 33.41 | 38.51 | 27.94 | 35.83 | 35.89 | 40.55 | 34.54 | 6.00 |
| gen_step1x_new | Generative | 41.28 | 39.00 | 33.14 | 38.40 | 27.89 | 35.09 | 35.10 | 40.14 | 34.12 | 6.02 |
| gen_step1x_v1p2 | Generative | 40.48 | 37.75 | 33.01 | 37.35 | 27.78 | 34.08 | 33.54 | 39.12 | 33.19 | 5.93 |
| std_autoaugment | Standard Aug | 41.44 | 38.86 | 33.24 | 37.97 | 28.24 | 35.80 | 35.41 | 40.15 | 34.35 | 5.79 |
| std_cutmix | Standard Aug | 41.45 | 39.17 | 33.30 | 37.53 | 28.18 | 35.80 | 35.59 | 40.31 | 34.27 | 6.04 |
| std_mixup | Standard Aug | 41.65 | 39.21 | 33.42 | 38.11 | 28.07 | 35.89 | 35.89 | 40.43 | 34.49 | 5.94 |
| std_randaugment | Standard Aug | 41.75 | 39.12 | 33.25 | 37.87 | 27.90 | 35.76 | 35.77 | 40.44 | 34.33 | 6.11 |
---

## Per-Model Breakdown

mIoU performance on each model architecture. Gain columns show improvement over baseline per model.

| Strategy | Type | hrnet_hr48 | hrnet_hr48_gain | mask2former_swin-b | mask2former_swin-b_gain | pspnet_r50 | pspnet_r50_gain | segformer_mit-b3 | segformer_mit-b3_gain | segnext_mscan-b | segnext_mscan-b_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_Img2Img | Generative | - | - | 46.60 | +0.02 | 34.77 | +2.69 | 40.67 | +4.46 | 41.22 | +3.79 |
| gen_UniControl | Generative | - | - | 45.82 | -0.76 | 34.74 | +2.66 | 41.11 | +4.90 | 41.12 | +3.69 |
| gen_Attribute_Hallucination | Generative | - | - | 45.23 | -1.35 | 34.88 | +2.79 | 41.23 | +5.02 | 41.08 | +3.65 |
| gen_stargan_v2 | Generative | - | - | 45.35 | -1.23 | 34.82 | +2.74 | 40.81 | +4.60 | 41.21 | +3.78 |
| gen_CNetSeg | Generative | - | - | 45.00 | -1.57 | 34.77 | +2.69 | 41.16 | +4.95 | 41.09 | +3.66 |
| gen_Qwen_Image_Edit | Generative | - | - | 44.96 | -1.62 | 34.83 | +2.75 | 41.10 | +4.89 | 41.09 | +3.66 |
| gen_VisualCloze | Generative | - | - | 46.15 | -0.43 | 34.96 | +2.88 | 40.69 | +4.48 | 40.11 | +2.68 |
| gen_augmenters | Generative | - | - | 44.67 | -1.90 | 34.80 | +2.72 | 40.87 | +4.67 | 41.26 | +3.83 |
| gen_IP2P | Generative | - | - | 44.68 | -1.90 | 34.90 | +2.82 | 41.00 | +4.79 | 41.01 | +3.58 |
| std_mixup | Standard Aug | - | - | 44.73 | -1.85 | 34.81 | +2.73 | 40.80 | +4.59 | 41.07 | +3.64 |
| std_randaugment | Standard Aug | - | - | 44.39 | -2.19 | 34.75 | +2.67 | 40.88 | +4.67 | 41.37 | +3.94 |
| std_autoaugment | Standard Aug | - | - | 44.92 | -1.66 | 34.86 | +2.78 | 40.92 | +4.71 | 40.25 | +2.82 |
| gen_cyclediffusion | Generative | - | - | 44.39 | -2.18 | 34.85 | +2.77 | 40.61 | +4.40 | 41.08 | +3.65 |
| std_cutmix | Standard Aug | - | - | 44.94 | -1.64 | 34.81 | +2.72 | 39.97 | +3.76 | 41.14 | +3.71 |
| gen_Weather_Effect_Generator | Generative | - | - | 44.32 | -2.26 | 34.75 | +2.67 | 40.76 | +4.55 | 40.92 | +3.49 |
| gen_CUT | Generative | - | - | 43.86 | -2.72 | 34.84 | +2.76 | 40.92 | +4.71 | 40.96 | +3.53 |
| gen_step1x_new | Generative | - | - | 45.24 | -1.34 | 34.81 | +2.73 | 39.03 | +2.82 | 41.14 | +3.71 |
| gen_SUSTechGAN | Generative | - | - | 44.93 | -1.65 | 34.77 | +2.69 | 40.79 | +4.58 | 39.56 | +2.12 |
| gen_TSIT | Generative | - | - | 45.36 | -1.22 | 34.86 | +2.78 | 38.79 | +2.58 | 41.02 | +3.59 |
| gen_flux_kontext | Generative | - | - | 44.60 | -1.98 | 34.88 | +2.80 | 39.41 | +3.20 | 41.12 | +3.69 |
| gen_automold | Generative | - | - | 45.07 | -1.51 | 34.84 | +2.76 | 41.14 | +4.93 | 38.62 | +1.19 |
| gen_LANIT | Generative | - | - | 45.17 | -1.41 | 34.82 | +2.73 | 39.14 | +2.93 | 40.07 | +2.64 |
| gen_cycleGAN | Generative | - | - | 44.98 | -1.60 | 34.84 | +2.76 | 39.23 | +3.03 | 39.70 | +2.27 |
| gen_albumentations_weather | Generative | - | - | 43.28 | -3.30 | 35.03 | +2.95 | 40.63 | +4.42 | 38.70 | +1.27 |
| gen_step1x_v1p2 | Generative | - | - | 43.55 | -3.03 | 34.96 | +2.88 | 39.31 | +3.10 | 38.71 | +1.28 |
| baseline | Baseline | 18.58 | +0.00 | 46.58 | +0.00 | 32.08 | +0.00 | 36.21 | +0.00 | 37.43 | +0.00 |
