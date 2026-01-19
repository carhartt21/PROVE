# Stage 1 Strategy Leaderboard

**Stage 1**: All models trained with `clear_day` domain filter only.

**Total Results**: 391 test results from 27 strategies

---

## Overall Strategy Ranking

Sorted by mIoU. Gain = improvement over baseline. Domain Gap = Normal mIoU - Adverse mIoU (lower is better).

| Strategy | Type | mIoU | Std | Gain | Normal mIoU | Adverse mIoU | Domain Gap | Num Tests |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_flux_kontext | Generative | 46.31 | 6.11 | 4.66 | 46.33 | 42.71 | 3.62 | 24 |
| gen_step1x_new | Generative | 45.78 | 9.16 | 4.14 | 46.28 | 40.51 | 5.78 | 12 |
| gen_step1x_v1p2 | Generative | 45.42 | 9.05 | 3.78 | 45.89 | 40.19 | 5.7 | 12 |
| gen_Qwen_Image_Edit | Generative | 45.27 | 8.27 | 3.63 | 45.91 | 40.41 | 5.51 | 12 |
| gen_cycleGAN | Generative | 43.52 | 6.19 | 1.88 | 43.45 | 36.95 | 6.5 | 52 |
| gen_Attribute_Hallucination | Generative | 43.17 | 6.67 | 1.53 | 43.94 | 37.7 | 6.25 | 12 |
| std_autoaugment | Standard Aug | 42.93 | 6.82 | 1.29 | 43.96 | 37.79 | 6.17 | 10 |
| std_mixup | Standard Aug | 42.87 | 6.54 | 1.23 | 43.55 | 37.39 | 6.15 | 12 |
| gen_automold | Generative | 42.84 | 6.99 | 1.2 | 43.58 | 37.29 | 6.29 | 12 |
| gen_stargan_v2 | Generative | 42.84 | 6.81 | 1.2 | 43.27 | 35.35 | 7.93 | 25 |
| gen_CNetSeg | Generative | 42.78 | 7.05 | 1.14 | 43.58 | 37.62 | 5.95 | 12 |
| gen_albumentations_weather | Generative | 42.77 | 7.1 | 1.12 | 43.5 | 37.45 | 6.06 | 12 |
| gen_IP2P | Generative | 42.72 | 7.04 | 1.08 | 43.39 | 37.11 | 6.28 | 12 |
| gen_SUSTechGAN | Generative | 42.7 | 6.78 | 1.06 | 43.24 | 37.37 | 5.86 | 12 |
| gen_CUT | Generative | 42.66 | 6.83 | 1.02 | 43.34 | 37.36 | 5.98 | 12 |
| gen_Img2Img | Generative | 42.64 | 6.7 | 1.0 | 43.49 | 37.28 | 6.21 | 12 |
| std_cutmix | Standard Aug | 42.63 | 7.11 | 0.99 | 43.31 | 37.14 | 6.17 | 12 |
| gen_TSIT | Generative | 42.56 | 7.29 | 0.92 | 43.19 | 36.81 | 6.38 | 12 |
| gen_VisualCloze | Generative | 42.54 | 6.53 | 0.9 | 43.25 | 37.12 | 6.13 | 12 |
| gen_Weather_Effect_Generator | Generative | 42.51 | 7.12 | 0.87 | 43.19 | 36.54 | 6.65 | 11 |
| gen_UniControl | Generative | 42.49 | 6.92 | 0.85 | 43.16 | 37.57 | 5.6 | 12 |
| std_randaugment | Standard Aug | 42.36 | 7.0 | 0.72 | 43.11 | 36.92 | 6.19 | 11 |
| gen_LANIT | Generative | 42.32 | 7.06 | 0.68 | 43.02 | 36.52 | 6.5 | 12 |
| gen_augmenters | Generative | 42.27 | 6.73 | 0.63 | 43.15 | 36.91 | 6.24 | 12 |
| photometric_distort | Augmentation | 41.83 | 7.67 | 0.19 | 42.65 | 36.72 | 5.93 | 11 |
| baseline | Baseline | 41.64 | 6.98 | - | 42.22 | 36.08 | 6.14 | 12 |
| gen_cyclediffusion | Generative | 40.83 | 5.68 | -0.81 | 41.4 | 31.41 | 9.98 | 19 |
---

## Per-Dataset Breakdown

mIoU performance on each dataset.

| Strategy | Type | bdd10k | bdd10k_gain | idd-aw | idd-aw_gain | mapillaryvistas | mapillaryvistas_gain | outside15k | outside15k_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_step1x_new | Generative | 46.32 | +2.23 | 39.30 | +0.73 | 48.62 | +1.62 | 48.87 | +11.97 |
| gen_step1x_v1p2 | Generative | 46.07 | +1.98 | 39.69 | +1.12 | 47.98 | +0.98 | 47.94 | +11.04 |
| gen_Qwen_Image_Edit | Generative | 45.90 | +1.81 | 39.81 | +1.23 | 51.97 | +4.97 | 43.39 | +6.49 |
| gen_cycleGAN | Generative | 46.03 | +1.94 | 40.28 | +1.71 | 49.48 | +2.48 | 41.06 | +4.16 |
| gen_stargan_v2 | Generative | 46.15 | +2.06 | 39.50 | +0.93 | 47.07 | +0.07 | 43.02 | +6.12 |
| gen_cyclediffusion | Generative | 46.28 | +2.19 | 39.45 | +0.88 | 51.52 | +4.52 | 37.36 | +0.46 |
| gen_flux_kontext | Generative | 46.14 | +2.05 | 39.30 | +0.72 | 49.36 | +2.37 | 38.19 | +1.29 |
| gen_Attribute_Hallucination | Generative | 46.49 | +2.40 | 40.00 | +1.42 | 48.18 | +1.19 | 38.02 | +1.12 |
| gen_Weather_Effect_Generator | Generative | 46.19 | +2.11 | 39.52 | +0.95 | 49.09 | +2.09 | 37.44 | +0.54 |
| std_mixup | Standard Aug | 46.04 | +1.95 | 39.92 | +1.35 | 47.89 | +0.90 | 37.62 | +0.72 |
| gen_automold | Generative | 46.37 | +2.28 | 39.64 | +1.07 | 48.17 | +1.18 | 37.17 | +0.27 |
| gen_CNetSeg | Generative | 45.98 | +1.89 | 39.66 | +1.09 | 48.39 | +1.39 | 37.11 | +0.21 |
| gen_albumentations_weather | Generative | 46.29 | +2.20 | 39.80 | +1.23 | 48.11 | +1.11 | 36.86 | -0.04 |
| gen_IP2P | Generative | 46.33 | +2.25 | 39.64 | +1.06 | 48.16 | +1.16 | 36.75 | -0.15 |
| gen_SUSTechGAN | Generative | 46.25 | +2.16 | 39.55 | +0.97 | 47.71 | +0.71 | 37.28 | +0.38 |
| gen_CUT | Generative | 45.91 | +1.82 | 39.52 | +0.94 | 48.07 | +1.08 | 37.15 | +0.25 |
| gen_Img2Img | Generative | 46.32 | +2.23 | 39.90 | +1.32 | 47.02 | +0.02 | 37.31 | +0.41 |
| std_cutmix | Standard Aug | 46.51 | +2.42 | 39.91 | +1.33 | 47.75 | +0.75 | 36.36 | -0.54 |
| gen_TSIT | Generative | 45.81 | +1.72 | 39.38 | +0.81 | 48.29 | +1.30 | 36.74 | -0.16 |
| gen_VisualCloze | Generative | 45.88 | +1.80 | 39.55 | +0.97 | 47.09 | +0.09 | 37.65 | +0.75 |
| gen_UniControl | Generative | 46.03 | +1.94 | 39.29 | +0.71 | 47.56 | +0.57 | 37.09 | +0.20 |
| gen_LANIT | Generative | 45.54 | +1.45 | 39.39 | +0.81 | 47.25 | +0.25 | 37.10 | +0.20 |
| gen_augmenters | Generative | 45.88 | +1.79 | 39.95 | +1.38 | 46.15 | -0.85 | 37.12 | +0.22 |
| baseline | Baseline | 44.09 | - | 38.58 | - | 47.00 | - | 36.90 | - |
| std_randaugment | Standard Aug | 46.20 | +2.11 | 40.00 | +1.43 | 48.16 | +1.16 | 31.44 | -5.46 |
| photometric_distort | Augmentation | 45.75 | +1.66 | 39.92 | +1.35 | 48.26 | +1.26 | 29.15 | -7.75 |
| std_autoaugment | Standard Aug | 46.33 | +2.24 | 39.66 | +1.09 | 47.46 | +0.46 | 28.97 | -7.93 |
---

## Per-Domain Breakdown

mIoU performance on each weather domain. Normal = clear_day, cloudy. Adverse = foggy, night, rainy, snowy.

| Strategy | Type | clear_day | cloudy | dawn_dusk | foggy | night | rainy | snowy | Normal Avg | Adverse Avg | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Baseline | 44.33 | 40.11 | 33.78 | 42.18 | 29.73 | 36.44 | 35.96 | 42.22 | 36.08 | 6.14 |
| gen_Attribute_Hallucination | Generative | 45.67 | 42.22 | 35.66 | 44.10 | 30.96 | 38.11 | 37.62 | 43.94 | 37.70 | 6.25 |
| gen_CNetSeg | Generative | 45.48 | 41.68 | 35.15 | 44.42 | 31.29 | 37.73 | 37.05 | 43.58 | 37.62 | 5.95 |
| gen_CUT | Generative | 45.27 | 41.42 | 34.79 | 44.38 | 30.68 | 37.23 | 37.16 | 43.34 | 37.36 | 5.98 |
| gen_IP2P | Generative | 45.28 | 41.50 | 35.18 | 42.80 | 30.22 | 37.79 | 37.62 | 43.39 | 37.11 | 6.28 |
| gen_Img2Img | Generative | 45.24 | 41.74 | 34.86 | 43.26 | 31.25 | 37.43 | 37.19 | 43.49 | 37.28 | 6.21 |
| gen_LANIT | Generative | 44.97 | 41.07 | 34.59 | 42.31 | 30.05 | 37.02 | 36.72 | 43.02 | 36.52 | 6.50 |
| gen_Qwen_Image_Edit | Generative | 47.68 | 44.14 | 38.37 | 46.95 | 33.19 | 39.60 | 41.89 | 45.91 | 40.41 | 5.51 |
| gen_SUSTechGAN | Generative | 45.19 | 41.28 | 35.07 | 43.55 | 31.04 | 37.55 | 37.36 | 43.24 | 37.37 | 5.86 |
| gen_TSIT | Generative | 45.30 | 41.08 | 34.89 | 42.78 | 30.28 | 37.22 | 36.95 | 43.19 | 36.81 | 6.38 |
| gen_UniControl | Generative | 45.14 | 41.19 | 34.78 | 44.40 | 30.77 | 37.69 | 37.41 | 43.16 | 37.57 | 5.60 |
| gen_VisualCloze | Generative | 45.19 | 41.32 | 34.90 | 43.15 | 30.34 | 37.68 | 37.31 | 43.25 | 37.12 | 6.13 |
| gen_Weather_Effect_Generator | Generative | 45.29 | 41.09 | 34.26 | 42.69 | 29.43 | 37.03 | 37.02 | 43.19 | 36.54 | 6.65 |
| gen_albumentations_weather | Generative | 45.37 | 41.63 | 35.46 | 44.33 | 31.09 | 37.52 | 36.84 | 43.50 | 37.45 | 6.06 |
| gen_augmenters | Generative | 44.92 | 41.38 | 34.70 | 43.42 | 30.43 | 37.01 | 36.76 | 43.15 | 36.91 | 6.24 |
| gen_automold | Generative | 45.32 | 41.84 | 34.85 | 43.22 | 30.79 | 37.92 | 37.24 | 43.58 | 37.29 | 6.29 |
| gen_cycleGAN | Generative | 46.43 | 40.47 | 35.89 | 42.11 | 34.28 | 36.99 | 34.44 | 43.45 | 36.95 | 6.50 |
| gen_cyclediffusion | Generative | 44.79 | 38.00 | 32.43 | 35.45 | 27.27 | 32.36 | 30.58 | 41.40 | 31.41 | 9.98 |
| gen_flux_kontext | Generative | 48.09 | 44.56 | 41.16 | 48.08 | 39.32 | 41.94 | 41.48 | 46.33 | 42.71 | 3.62 |
| gen_stargan_v2 | Generative | 46.11 | 40.44 | 36.51 | 39.79 | 31.84 | 35.66 | 34.10 | 43.27 | 35.35 | 7.93 |
| gen_step1x_new | Generative | 47.99 | 44.58 | 37.29 | 46.72 | 33.29 | 40.43 | 41.58 | 46.28 | 40.51 | 5.78 |
| gen_step1x_v1p2 | Generative | 47.58 | 44.20 | 37.80 | 46.23 | 33.08 | 40.29 | 41.15 | 45.89 | 40.19 | 5.70 |
| photometric_distort | Augmentation | 44.36 | 40.94 | 34.14 | 43.46 | 30.33 | 36.74 | 36.34 | 42.65 | 36.72 | 5.93 |
| std_autoaugment | Standard Aug | 45.59 | 42.33 | 36.69 | 44.44 | 30.26 | 37.87 | 38.59 | 43.96 | 37.79 | 6.17 |
| std_cutmix | Standard Aug | 45.19 | 41.43 | 35.35 | 43.19 | 30.68 | 37.28 | 37.41 | 43.31 | 37.14 | 6.17 |
| std_mixup | Standard Aug | 45.47 | 41.63 | 35.30 | 43.06 | 31.13 | 38.00 | 37.38 | 43.55 | 37.39 | 6.15 |
| std_randaugment | Standard Aug | 44.95 | 41.27 | 35.18 | 43.26 | 29.85 | 37.34 | 37.23 | 43.11 | 36.92 | 6.19 |
