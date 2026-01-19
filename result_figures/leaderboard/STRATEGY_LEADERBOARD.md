# Stage 1 Strategy Leaderboard

**Stage 1**: All models trained with `clear_day` domain filter only.

**Total Results**: 346 test results from 28 strategies

---

## Overall Strategy Ranking

Sorted by mIoU. Gain = improvement over baseline. Domain Gap = Normal mIoU - Adverse mIoU (lower is better).

| Strategy | Type | mIoU | Std | Gain | Normal mIoU | Adverse mIoU | Domain Gap | Num Tests |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| std_randaugment | Standard Aug | 46.74 | 12.49 | 5.49 | 47.27 | 38.7 | 8.57 | 17 |
| photometric_distort | Augmentation | 46.35 | 12.73 | 5.1 | 47.03 | 38.69 | 8.34 | 17 |
| gen_step1x_new | Generative | 45.78 | 9.16 | 4.53 | 46.28 | 40.51 | 5.78 | 12 |
| gen_step1x_v1p2 | Generative | 45.42 | 9.05 | 4.17 | 45.89 | 40.19 | 5.7 | 12 |
| gen_Qwen_Image_Edit | Generative | 45.27 | 8.27 | 4.02 | 45.91 | 40.41 | 5.51 | 12 |
| std_autoaugment | Standard Aug | 44.41 | 8.29 | 3.16 | 44.91 | 37.75 | 7.16 | 15 |
| gen_stargan_v2 | Generative | 44.21 | 8.95 | 2.96 | 44.94 | 39.04 | 5.9 | 12 |
| gen_Attribute_Hallucination | Generative | 43.17 | 6.38 | 1.93 | 43.91 | 37.41 | 6.5 | 13 |
| gen_cycleGAN | Generative | 42.99 | 6.53 | 1.74 | 43.62 | 37.4 | 6.22 | 12 |
| gen_flux_kontext | Generative | 42.92 | 6.63 | 1.67 | 43.65 | 37.39 | 6.25 | 12 |
| gen_automold | Generative | 42.84 | 6.99 | 1.59 | 43.58 | 37.29 | 6.29 | 12 |
| gen_CNetSeg | Generative | 42.83 | 6.75 | 1.58 | 43.59 | 37.35 | 6.25 | 13 |
| gen_IP2P | Generative | 42.77 | 6.74 | 1.52 | 43.4 | 36.88 | 6.52 | 13 |
| gen_albumentations_weather | Generative | 42.77 | 7.1 | 1.52 | 43.5 | 37.45 | 6.06 | 12 |
| gen_SUSTechGAN | Generative | 42.7 | 6.78 | 1.45 | 43.24 | 37.37 | 5.86 | 12 |
| gen_CUT | Generative | 42.66 | 6.83 | 1.41 | 43.34 | 37.36 | 5.98 | 12 |
| gen_Img2Img | Generative | 42.64 | 6.7 | 1.39 | 43.49 | 37.28 | 6.21 | 12 |
| gen_TSIT | Generative | 42.56 | 7.29 | 1.31 | 43.19 | 36.81 | 6.38 | 12 |
| gen_VisualCloze | Generative | 42.54 | 6.53 | 1.29 | 43.25 | 37.12 | 6.13 | 12 |
| gen_Weather_Effect_Generator | Generative | 42.51 | 7.12 | 1.26 | 43.19 | 36.54 | 6.65 | 11 |
| gen_UniControl | Generative | 42.49 | 6.92 | 1.24 | 43.16 | 37.57 | 5.6 | 12 |
| gen_LANIT | Generative | 42.32 | 7.06 | 1.07 | 43.02 | 36.52 | 6.5 | 12 |
| gen_augmenters | Generative | 42.27 | 6.73 | 1.02 | 43.15 | 36.91 | 6.24 | 12 |
| std_mixup | Standard Aug | 42.21 | 6.07 | 0.96 | 42.79 | 35.2 | 7.6 | 15 |
| gen_cyclediffusion | Generative | 42.19 | 7.26 | 0.94 | 43.06 | 35.95 | 7.1 | 10 |
| std_cutmix | Standard Aug | 42.15 | 6.45 | 0.9 | 42.73 | 35.2 | 7.53 | 15 |
| baseline | Baseline | 41.25 | 6.83 | - | 41.77 | 34.98 | 6.79 | 13 |
| gen_EDICT | Generative | 41.03 | 5.77 | -0.21 | 44.22 | 40.11 | 4.12 | 2 |
---

## Per-Dataset Breakdown

mIoU performance on each dataset.

| Strategy | Type | bdd10k | bdd10k_gain | idd-aw | idd-aw_gain | mapillaryvistas | mapillaryvistas_gain | outside15k | outside15k_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| std_randaugment | Standard Aug | 46.20 | +2.11 | 48.10 | +10.03 | 48.16 | +1.16 | 42.23 | +5.33 |
| gen_step1x_new | Generative | 46.32 | +2.23 | 39.30 | +1.23 | 48.62 | +1.62 | 48.87 | +11.97 |
| photometric_distort | Augmentation | 45.75 | +1.66 | 47.95 | +9.88 | 48.26 | +1.26 | 40.75 | +3.85 |
| std_autoaugment | Standard Aug | 46.33 | +2.24 | 39.84 | +1.77 | 47.46 | +0.46 | 48.56 | +11.66 |
| gen_step1x_v1p2 | Generative | 46.07 | +1.98 | 39.69 | +1.62 | 47.98 | +0.98 | 47.94 | +11.04 |
| gen_Qwen_Image_Edit | Generative | 45.90 | +1.81 | 39.81 | +1.74 | 51.97 | +4.97 | 43.39 | +6.49 |
| gen_stargan_v2 | Generative | 46.15 | +2.06 | 39.54 | +1.47 | 48.12 | +1.13 | 43.02 | +6.12 |
| gen_cyclediffusion | Generative | 46.28 | +2.19 | 39.84 | +1.77 | 51.52 | +4.52 | 37.36 | +0.46 |
| gen_Attribute_Hallucination | Generative | 46.49 | +2.40 | 40.80 | +2.73 | 48.18 | +1.19 | 38.02 | +1.12 |
| gen_Weather_Effect_Generator | Generative | 46.19 | +2.11 | 39.52 | +1.46 | 49.09 | +2.09 | 37.44 | +0.54 |
| gen_CNetSeg | Generative | 45.98 | +1.89 | 40.58 | +2.51 | 48.39 | +1.39 | 37.11 | +0.21 |
| gen_cycleGAN | Generative | 46.03 | +1.94 | 40.03 | +1.96 | 48.03 | +1.04 | 37.86 | +0.96 |
| gen_IP2P | Generative | 46.33 | +2.25 | 40.57 | +2.50 | 48.16 | +1.16 | 36.75 | -0.15 |
| gen_flux_kontext | Generative | 46.14 | +2.05 | 39.30 | +1.23 | 48.06 | +1.06 | 38.19 | +1.29 |
| gen_automold | Generative | 46.37 | +2.28 | 39.64 | +1.57 | 48.17 | +1.18 | 37.17 | +0.27 |
| std_mixup | Standard Aug | 46.04 | +1.95 | 39.74 | +1.67 | 47.89 | +0.90 | 37.62 | +0.72 |
| gen_albumentations_weather | Generative | 46.29 | +2.20 | 39.80 | +1.73 | 48.11 | +1.11 | 36.86 | -0.04 |
| gen_SUSTechGAN | Generative | 46.25 | +2.16 | 39.55 | +1.48 | 47.71 | +0.71 | 37.28 | +0.38 |
| std_cutmix | Standard Aug | 46.51 | +2.42 | 40.07 | +2.00 | 47.75 | +0.75 | 36.36 | -0.54 |
| gen_CUT | Generative | 45.91 | +1.82 | 39.52 | +1.45 | 48.07 | +1.08 | 37.15 | +0.25 |
| gen_Img2Img | Generative | 46.32 | +2.23 | 39.90 | +1.83 | 47.02 | +0.02 | 37.31 | +0.41 |
| gen_TSIT | Generative | 45.81 | +1.72 | 39.38 | +1.31 | 48.29 | +1.30 | 36.74 | -0.16 |
| gen_VisualCloze | Generative | 45.88 | +1.80 | 39.55 | +1.48 | 47.09 | +0.09 | 37.65 | +0.75 |
| gen_UniControl | Generative | 46.03 | +1.94 | 39.29 | +1.22 | 47.56 | +0.57 | 37.09 | +0.20 |
| gen_LANIT | Generative | 45.54 | +1.45 | 39.39 | +1.32 | 47.25 | +0.25 | 37.10 | +0.20 |
| gen_augmenters | Generative | 45.88 | +1.79 | 39.95 | +1.88 | 46.15 | -0.85 | 37.12 | +0.22 |
| baseline | Baseline | 44.09 | - | 38.07 | - | 47.00 | - | 36.90 | - |
| gen_EDICT | Generative | 41.03 | -3.05 | - | - | - | - | - | - |
---

## Per-Domain Breakdown

mIoU performance on each weather domain. Normal = clear_day, cloudy. Adverse = foggy, night, rainy, snowy.

| Strategy | Type | clear_day | cloudy | dawn_dusk | foggy | night | rainy | snowy | Normal Avg | Adverse Avg | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Baseline | 44.16 | 39.37 | 33.18 | 40.69 | 28.99 | 35.41 | 34.81 | 41.77 | 34.98 | 6.79 |
| gen_Attribute_Hallucination | Generative | 45.76 | 42.06 | 35.99 | 43.54 | 31.39 | 37.68 | 37.02 | 43.91 | 37.41 | 6.50 |
| gen_CNetSeg | Generative | 45.59 | 41.59 | 35.46 | 43.89 | 31.63 | 37.34 | 36.53 | 43.59 | 37.35 | 6.25 |
| gen_CUT | Generative | 45.27 | 41.42 | 34.79 | 44.38 | 30.68 | 37.23 | 37.16 | 43.34 | 37.36 | 5.98 |
| gen_EDICT | Generative | 43.12 | 45.33 | 35.27 | 55.22 | 19.94 | 38.93 | 46.33 | 44.22 | 40.11 | 4.12 |
| gen_IP2P | Generative | 45.41 | 41.39 | 35.52 | 42.30 | 30.72 | 37.48 | 37.02 | 43.40 | 36.88 | 6.52 |
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
| gen_cycleGAN | Generative | 45.70 | 41.54 | 35.60 | 43.58 | 30.88 | 37.71 | 37.42 | 43.62 | 37.40 | 6.22 |
| gen_cyclediffusion | Generative | 45.17 | 40.94 | 33.37 | 41.98 | 28.48 | 37.09 | 36.28 | 43.06 | 35.95 | 7.10 |
| gen_flux_kontext | Generative | 45.52 | 41.77 | 35.11 | 43.60 | 30.78 | 37.76 | 37.44 | 43.65 | 37.39 | 6.25 |
| gen_stargan_v2 | Generative | 46.67 | 43.21 | 37.33 | 45.30 | 32.30 | 39.17 | 39.40 | 44.94 | 39.04 | 5.90 |
| gen_step1x_new | Generative | 47.99 | 44.58 | 37.29 | 46.72 | 33.29 | 40.43 | 41.58 | 46.28 | 40.51 | 5.78 |
| gen_step1x_v1p2 | Generative | 47.58 | 44.20 | 37.80 | 46.23 | 33.08 | 40.29 | 41.15 | 45.89 | 40.19 | 5.70 |
| photometric_distort | Augmentation | 49.40 | 44.66 | 37.52 | 44.31 | 34.47 | 38.81 | 37.15 | 47.03 | 38.69 | 8.34 |
| std_autoaugment | Standard Aug | 47.31 | 42.51 | 35.92 | 43.46 | 31.55 | 37.89 | 38.10 | 44.91 | 37.75 | 7.16 |
| std_cutmix | Standard Aug | 45.21 | 40.24 | 34.82 | 40.49 | 29.96 | 35.47 | 34.88 | 42.73 | 35.20 | 7.53 |
| std_mixup | Standard Aug | 45.38 | 40.21 | 34.58 | 40.02 | 30.02 | 35.94 | 34.80 | 42.79 | 35.20 | 7.60 |
| std_randaugment | Standard Aug | 49.91 | 44.64 | 38.24 | 44.08 | 33.83 | 39.21 | 37.69 | 47.27 | 38.70 | 8.57 |
