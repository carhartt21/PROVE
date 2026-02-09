# Stage 1 Strategy Leaderboard (by mIoU)

**Stage 1**: All models trained with `clear_day` domain filter only.

**Metric**: mIoU (Mean Intersection over Union)

**Total Results**: 332 test results from 26 strategies

---

## Overall Strategy Ranking

Sorted by mIoU. Gain = improvement over baseline. Domain Gap = Normal mIoU - Adverse mIoU (lower is better).

| Strategy | Type | mIoU | Std | Gain | Normal mIoU | Adverse mIoU | Domain Gap | Num Tests |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_cyclediffusion | Generative | 41.77 | 4.76 | 8.14 | 42.99 | 37.52 | 5.47 | 7 |
| gen_albumentations_weather | Generative | 41.46 | 4.54 | 7.83 | 42.14 | 36.86 | 5.28 | 6 |
| gen_UniControl | Generative | 41.43 | 5.85 | 7.8 | 42.01 | 36.86 | 5.15 | 7 |
| gen_Img2Img | Generative | 40.6 | 4.85 | 6.97 | 41.22 | 36.08 | 5.14 | 7 |
| gen_Attribute_Hallucination | Generative | 39.94 | 6.11 | 6.31 | 40.49 | 34.51 | 5.97 | 14 |
| gen_Qwen_Image_Edit | Generative | 39.86 | 5.97 | 6.22 | 40.51 | 34.51 | 6.01 | 14 |
| gen_stargan_v2 | Generative | 39.86 | 6.15 | 6.23 | 40.55 | 34.54 | 6.0 | 14 |
| gen_CNetSeg | Generative | 39.86 | 6.08 | 6.23 | 40.53 | 34.55 | 5.99 | 14 |
| gen_IP2P | Generative | 39.79 | 5.88 | 6.15 | 40.5 | 34.6 | 5.9 | 14 |
| gen_augmenters | Generative | 39.79 | 5.9 | 6.16 | 40.39 | 34.46 | 5.92 | 14 |
| std_randaugment | Standard Aug | 39.77 | 5.92 | 6.14 | 40.44 | 34.33 | 6.11 | 14 |
| std_mixup | Standard Aug | 39.73 | 5.94 | 6.09 | 40.43 | 34.49 | 5.94 | 14 |
| gen_CUT | Generative | 39.62 | 5.79 | 5.98 | 40.33 | 34.38 | 5.95 | 14 |
| gen_Weather_Effect_Generator | Generative | 39.59 | 5.7 | 5.96 | 40.47 | 34.35 | 6.12 | 14 |
| std_autoaugment | Standard Aug | 39.57 | 6.09 | 5.93 | 40.15 | 34.35 | 5.79 | 14 |
| std_cutmix | Standard Aug | 39.54 | 6.0 | 5.91 | 40.31 | 34.27 | 6.04 | 14 |
| gen_VisualCloze | Generative | 39.32 | 5.89 | 5.69 | 40.01 | 34.08 | 5.93 | 13 |
| gen_SUSTechGAN | Generative | 39.31 | 5.51 | 5.68 | 39.94 | 34.06 | 5.88 | 14 |
| gen_TSIT | Generative | 39.24 | 6.91 | 5.61 | 40.02 | 33.94 | 6.08 | 14 |
| gen_automold | Generative | 39.23 | 5.76 | 5.59 | 39.85 | 33.85 | 6.0 | 13 |
| gen_cycleGAN | Generative | 38.93 | 5.1 | 5.3 | 39.76 | 33.79 | 5.97 | 14 |
| gen_flux_kontext | Generative | 38.7 | 5.12 | 5.06 | 39.4 | 33.25 | 6.15 | 13 |
| gen_step1x_new | Generative | 38.64 | 5.23 | 5.0 | 39.38 | 33.2 | 6.18 | 13 |
| gen_step1x_v1p2 | Generative | 38.49 | 4.79 | 4.85 | 39.12 | 33.19 | 5.93 | 13 |
| gen_LANIT | Generative | 38.42 | 5.43 | 4.78 | 39.13 | 33.02 | 6.11 | 13 |
| baseline | Baseline | 33.63 | 9.4 | - | 34.24 | 28.89 | 5.35 | 17 |
---

## Per-Dataset Breakdown

mIoU performance on each dataset.

| Strategy | Type | bdd10k | bdd10k_gain | idd-aw | idd-aw_gain | mapillaryvistas | mapillaryvistas_gain | outside15k | outside15k_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_UniControl | Generative | 48.81 | +6.53 | - | - | 35.04 | +8.39 | 43.08 | +10.22 |
| gen_Img2Img | Generative | 46.28 | +3.99 | - | - | 34.79 | +8.14 | 43.01 | +10.15 |
| gen_albumentations_weather | Generative | 46.34 | +4.05 | - | - | 34.74 | +8.09 | 42.60 | +9.74 |
| gen_cyclediffusion | Generative | 45.41 | +3.13 | - | - | 34.38 | +7.73 | 42.78 | +9.93 |
| gen_Attribute_Hallucination | Generative | 46.70 | +4.42 | - | - | 32.91 | +6.26 | 40.90 | +8.04 |
| gen_stargan_v2 | Generative | 46.55 | +4.27 | - | - | 32.71 | +6.07 | 41.00 | +8.14 |
| gen_CNetSeg | Generative | 46.59 | +4.31 | - | - | 32.87 | +6.22 | 40.77 | +7.92 |
| gen_Qwen_Image_Edit | Generative | 46.36 | +4.08 | - | - | 32.88 | +6.23 | 40.89 | +8.03 |
| gen_augmenters | Generative | 46.13 | +3.85 | - | - | 32.87 | +6.22 | 40.98 | +8.12 |
| std_autoaugment | Standard Aug | 46.38 | +4.10 | - | - | 32.82 | +6.17 | 40.75 | +7.89 |
| std_randaugment | Standard Aug | 46.16 | +3.88 | - | - | 32.83 | +6.18 | 40.89 | +8.04 |
| gen_IP2P | Generative | 46.28 | +4.00 | - | - | 32.88 | +6.23 | 40.67 | +7.82 |
| gen_VisualCloze | Generative | 46.36 | +4.08 | - | - | 32.94 | +6.29 | 40.48 | +7.62 |
| std_cutmix | Standard Aug | 46.28 | +3.99 | - | - | 32.90 | +6.25 | 40.51 | +7.66 |
| std_mixup | Standard Aug | 46.37 | +4.09 | - | - | 32.83 | +6.18 | 40.43 | +7.57 |
| gen_automold | Generative | 46.07 | +3.79 | - | - | 32.90 | +6.25 | 40.59 | +7.74 |
| gen_Weather_Effect_Generator | Generative | 45.82 | +3.54 | - | - | 33.04 | +6.39 | 40.52 | +7.67 |
| gen_CUT | Generative | 45.98 | +3.70 | - | - | 32.68 | +6.03 | 40.57 | +7.71 |
| gen_SUSTechGAN | Generative | 44.84 | +2.56 | - | - | 32.77 | +6.12 | 40.44 | +7.58 |
| gen_TSIT | Generative | 46.56 | +4.28 | - | - | 30.09 | +3.44 | 40.68 | +7.83 |
| gen_flux_kontext | Generative | 43.45 | +1.17 | - | - | 32.98 | +6.33 | 40.86 | +8.00 |
| gen_LANIT | Generative | 43.75 | +1.46 | - | - | 32.88 | +6.23 | 40.66 | +7.81 |
| gen_step1x_new | Generative | 43.56 | +1.28 | - | - | 32.94 | +6.29 | 40.44 | +7.59 |
| gen_cycleGAN | Generative | 43.27 | +0.99 | - | - | 32.90 | +6.25 | 40.61 | +7.76 |
| gen_step1x_v1p2 | Generative | 43.03 | +0.75 | - | - | 32.85 | +6.20 | 40.53 | +7.68 |
| baseline | Baseline | 42.28 | +0.00 | - | - | 26.65 | - | 32.85 | - |
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
| gen_Img2Img | Generative | 42.45 | 39.99 | 36.72 | 39.98 | 30.48 | 37.07 | 36.78 | 41.22 | 36.08 | 5.14 |
| gen_LANIT | Generative | 40.49 | 37.77 | 32.26 | 36.94 | 27.64 | 33.98 | 33.51 | 39.13 | 33.02 | 6.11 |
| gen_Qwen_Image_Edit | Generative | 41.77 | 39.26 | 33.73 | 38.11 | 28.63 | 35.84 | 35.44 | 40.51 | 34.51 | 6.01 |
| gen_SUSTechGAN | Generative | 41.28 | 38.60 | 33.09 | 37.50 | 27.86 | 35.24 | 35.63 | 39.94 | 34.06 | 5.88 |
| gen_TSIT | Generative | 41.17 | 38.87 | 32.72 | 37.24 | 28.27 | 35.10 | 35.15 | 40.02 | 33.94 | 6.08 |
| gen_UniControl | Generative | 43.16 | 40.86 | 36.52 | 41.40 | 30.41 | 37.90 | 37.74 | 42.01 | 36.86 | 5.15 |
| gen_VisualCloze | Generative | 41.34 | 38.69 | 33.26 | 38.07 | 28.88 | 35.06 | 34.31 | 40.01 | 34.08 | 5.93 |
| gen_Weather_Effect_Generator | Generative | 41.59 | 39.36 | 33.86 | 37.86 | 28.37 | 35.55 | 35.62 | 40.47 | 34.35 | 6.12 |
| gen_albumentations_weather | Generative | 43.38 | 40.90 | 36.33 | 42.03 | 29.92 | 37.70 | 37.80 | 42.14 | 36.86 | 5.28 |
| gen_augmenters | Generative | 41.72 | 39.06 | 33.75 | 37.71 | 28.35 | 35.73 | 36.07 | 40.39 | 34.46 | 5.92 |
| gen_automold | Generative | 41.25 | 38.46 | 32.64 | 38.03 | 28.51 | 34.67 | 34.19 | 39.85 | 33.85 | 6.00 |
| gen_cycleGAN | Generative | 40.92 | 38.61 | 33.25 | 38.08 | 27.95 | 34.62 | 34.53 | 39.76 | 33.79 | 5.97 |
| gen_cyclediffusion | Generative | 43.57 | 42.41 | 36.33 | 42.75 | 29.73 | 38.29 | 39.30 | 42.99 | 37.52 | 5.47 |
| gen_flux_kontext | Generative | 40.76 | 38.03 | 33.19 | 36.84 | 28.33 | 34.47 | 33.34 | 39.40 | 33.25 | 6.15 |
| gen_stargan_v2 | Generative | 41.76 | 39.33 | 33.41 | 38.51 | 27.94 | 35.83 | 35.89 | 40.55 | 34.54 | 6.00 |
| gen_step1x_new | Generative | 40.64 | 38.12 | 32.44 | 37.32 | 27.77 | 34.11 | 33.58 | 39.38 | 33.20 | 6.18 |
| gen_step1x_v1p2 | Generative | 40.48 | 37.75 | 33.01 | 37.35 | 27.78 | 34.08 | 33.54 | 39.12 | 33.19 | 5.93 |
| std_autoaugment | Standard Aug | 41.44 | 38.86 | 33.24 | 37.97 | 28.24 | 35.80 | 35.41 | 40.15 | 34.35 | 5.79 |
| std_cutmix | Standard Aug | 41.45 | 39.17 | 33.30 | 37.53 | 28.18 | 35.80 | 35.59 | 40.31 | 34.27 | 6.04 |
| std_mixup | Standard Aug | 41.65 | 39.21 | 33.42 | 38.11 | 28.07 | 35.89 | 35.89 | 40.43 | 34.49 | 5.94 |
| std_randaugment | Standard Aug | 41.75 | 39.12 | 33.25 | 37.87 | 27.90 | 35.76 | 35.77 | 40.44 | 34.33 | 6.11 |
