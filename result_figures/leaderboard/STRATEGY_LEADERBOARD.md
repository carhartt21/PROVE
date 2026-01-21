# Stage 1 Strategy Leaderboard

**Stage 1**: All models trained with `clear_day` domain filter only.

**Total Results**: 324 test results from 27 strategies

---

## Overall Strategy Ranking

Sorted by mIoU. Gain = improvement over baseline. Domain Gap = Normal mIoU - Adverse mIoU (lower is better).

| Strategy | Type | mIoU | Std | Gain | Normal mIoU | Adverse mIoU | Domain Gap | Num Tests |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_Qwen_Image_Edit | Generative | 43.61 | 8.51 | 1.97 | 44.39 | 38.56 | 5.83 | 12 |
| gen_Attribute_Hallucination | Generative | 43.17 | 6.67 | 1.53 | 43.94 | 37.7 | 6.25 | 12 |
| std_autoaugment | Standard Aug | 43.02 | 6.69 | 1.38 | 43.58 | 38.34 | 5.24 | 12 |
| gen_cycleGAN | Generative | 42.99 | 6.53 | 1.35 | 43.62 | 37.4 | 6.22 | 12 |
| gen_step1x_new | Generative | 42.92 | 7.08 | 1.28 | 43.63 | 37.45 | 6.18 | 12 |
| gen_flux_kontext | Generative | 42.92 | 6.63 | 1.28 | 43.65 | 37.39 | 6.25 | 12 |
| gen_stargan_v2 | Generative | 42.89 | 6.65 | 1.25 | 43.72 | 37.6 | 6.11 | 12 |
| gen_cyclediffusion | Generative | 42.88 | 6.77 | 1.24 | 43.61 | 37.44 | 6.17 | 12 |
| gen_automold | Generative | 42.84 | 6.99 | 1.2 | 43.58 | 37.29 | 6.29 | 12 |
| gen_CNetSeg | Generative | 42.78 | 7.05 | 1.14 | 43.58 | 37.62 | 5.95 | 12 |
| gen_albumentations_weather | Generative | 42.77 | 7.1 | 1.12 | 43.5 | 37.45 | 6.06 | 12 |
| gen_Weather_Effect_Generator | Generative | 42.73 | 6.83 | 1.09 | 43.4 | 37.07 | 6.33 | 12 |
| gen_IP2P | Generative | 42.72 | 7.04 | 1.08 | 43.39 | 37.11 | 6.28 | 12 |
| gen_SUSTechGAN | Generative | 42.7 | 6.78 | 1.06 | 43.24 | 37.37 | 5.86 | 12 |
| gen_CUT | Generative | 42.66 | 6.83 | 1.02 | 43.34 | 37.36 | 5.98 | 12 |
| gen_Img2Img | Generative | 42.64 | 6.7 | 1.0 | 43.49 | 37.28 | 6.21 | 12 |
| gen_TSIT | Generative | 42.56 | 7.29 | 0.92 | 43.19 | 36.81 | 6.38 | 12 |
| gen_VisualCloze | Generative | 42.54 | 6.53 | 0.9 | 43.25 | 37.12 | 6.13 | 12 |
| gen_step1x_v1p2 | Generative | 42.53 | 7.25 | 0.89 | 43.26 | 37.13 | 6.13 | 12 |
| gen_UniControl | Generative | 42.49 | 6.92 | 0.85 | 43.16 | 37.57 | 5.6 | 12 |
| std_mixup | Standard Aug | 42.41 | 6.9 | 0.77 | 43.19 | 36.87 | 6.32 | 12 |
| photometric_distort | Augmentation | 42.37 | 7.55 | 0.73 | 43.09 | 37.09 | 6.01 | 12 |
| gen_LANIT | Generative | 42.32 | 7.06 | 0.68 | 43.02 | 36.52 | 6.5 | 12 |
| gen_augmenters | Generative | 42.27 | 6.73 | 0.63 | 43.15 | 36.91 | 6.24 | 12 |
| std_randaugment | Standard Aug | 42.14 | 6.71 | 0.5 | 42.79 | 37.27 | 5.52 | 12 |
| std_cutmix | Standard Aug | 41.91 | 6.97 | 0.27 | 42.86 | 36.89 | 5.97 | 12 |
| baseline | Baseline | 41.64 | 6.98 | - | 42.22 | 36.08 | 6.14 | 12 |
---

## Per-Dataset Breakdown

mIoU performance on each dataset.

| Strategy | Type | bdd10k | bdd10k_gain | idd-aw | idd-aw_gain | mapillaryvistas | mapillaryvistas_gain | outside15k | outside15k_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_Qwen_Image_Edit | Generative | 45.90 | +1.81 | 39.81 | +1.23 | 51.97 | +4.97 | 36.78 | -0.12 |
| gen_Attribute_Hallucination | Generative | 46.49 | +2.40 | 40.00 | +1.42 | 48.18 | +1.19 | 38.02 | +1.12 |
| std_autoaugment | Standard Aug | 46.09 | +2.00 | 39.80 | +1.22 | 48.59 | +1.59 | 37.60 | +0.70 |
| gen_cycleGAN | Generative | 46.03 | +1.94 | 40.03 | +1.45 | 48.03 | +1.04 | 37.86 | +0.96 |
| gen_step1x_new | Generative | 46.32 | +2.23 | 39.30 | +0.73 | 48.62 | +1.62 | 37.45 | +0.55 |
| gen_flux_kontext | Generative | 46.14 | +2.05 | 39.29 | +0.72 | 48.06 | +1.06 | 38.19 | +1.29 |
| gen_stargan_v2 | Generative | 46.15 | +2.06 | 39.54 | +0.97 | 48.12 | +1.13 | 37.74 | +0.84 |
| gen_cyclediffusion | Generative | 46.28 | +2.19 | 39.84 | +1.26 | 48.06 | +1.06 | 37.36 | +0.46 |
| gen_automold | Generative | 46.37 | +2.28 | 39.64 | +1.07 | 48.17 | +1.18 | 37.17 | +0.27 |
| gen_CNetSeg | Generative | 45.98 | +1.89 | 39.66 | +1.09 | 48.39 | +1.39 | 37.11 | +0.21 |
| gen_albumentations_weather | Generative | 46.29 | +2.20 | 39.80 | +1.23 | 48.11 | +1.11 | 36.86 | -0.04 |
| gen_Weather_Effect_Generator | Generative | 46.19 | +2.11 | 39.52 | +0.95 | 47.77 | +0.78 | 37.44 | +0.54 |
| gen_IP2P | Generative | 46.33 | +2.25 | 39.64 | +1.06 | 48.16 | +1.16 | 36.75 | -0.15 |
| gen_SUSTechGAN | Generative | 46.25 | +2.16 | 39.55 | +0.97 | 47.71 | +0.71 | 37.28 | +0.38 |
| gen_CUT | Generative | 45.91 | +1.82 | 39.52 | +0.94 | 48.07 | +1.08 | 37.15 | +0.25 |
| gen_Img2Img | Generative | 46.32 | +2.23 | 39.90 | +1.32 | 47.02 | +0.02 | 37.31 | +0.41 |
| gen_TSIT | Generative | 45.81 | +1.72 | 39.38 | +0.81 | 48.29 | +1.30 | 36.74 | -0.16 |
| gen_VisualCloze | Generative | 45.88 | +1.80 | 39.55 | +0.97 | 47.09 | +0.09 | 37.65 | +0.75 |
| gen_step1x_v1p2 | Generative | 46.07 | +1.98 | 39.69 | +1.12 | 47.98 | +0.98 | 36.36 | -0.53 |
| gen_UniControl | Generative | 46.03 | +1.94 | 39.29 | +0.71 | 47.56 | +0.57 | 37.09 | +0.20 |
| std_mixup | Standard Aug | 45.56 | +1.47 | 38.63 | +0.06 | 47.80 | +0.80 | 37.67 | +0.77 |
| photometric_distort | Augmentation | 45.75 | +1.66 | 39.92 | +1.35 | 48.26 | +1.26 | 35.55 | -1.35 |
| gen_LANIT | Generative | 45.54 | +1.45 | 39.39 | +0.81 | 47.25 | +0.25 | 37.10 | +0.20 |
| gen_augmenters | Generative | 45.88 | +1.79 | 39.95 | +1.38 | 46.15 | -0.85 | 37.12 | +0.22 |
| std_randaugment | Standard Aug | 45.25 | +1.17 | 39.55 | +0.97 | 46.37 | -0.63 | 37.38 | +0.48 |
| std_cutmix | Standard Aug | 44.50 | +0.41 | 39.33 | +0.75 | 46.90 | -0.10 | 36.91 | +0.01 |
| baseline | Baseline | 44.09 | - | 38.58 | - | 47.00 | - | 36.90 | - |
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
| gen_Qwen_Image_Edit | Generative | 46.19 | 42.59 | 36.72 | 44.77 | 31.73 | 38.26 | 39.47 | 44.39 | 38.56 | 5.83 |
| gen_SUSTechGAN | Generative | 45.19 | 41.28 | 35.07 | 43.55 | 31.04 | 37.55 | 37.36 | 43.24 | 37.37 | 5.86 |
| gen_TSIT | Generative | 45.30 | 41.08 | 34.89 | 42.78 | 30.28 | 37.22 | 36.95 | 43.19 | 36.81 | 6.38 |
| gen_UniControl | Generative | 45.14 | 41.19 | 34.78 | 44.40 | 30.77 | 37.69 | 37.41 | 43.16 | 37.57 | 5.60 |
| gen_VisualCloze | Generative | 45.19 | 41.32 | 34.90 | 43.15 | 30.34 | 37.68 | 37.31 | 43.25 | 37.12 | 6.13 |
| gen_Weather_Effect_Generator | Generative | 45.38 | 41.42 | 35.09 | 43.10 | 30.55 | 37.38 | 37.25 | 43.40 | 37.07 | 6.33 |
| gen_albumentations_weather | Generative | 45.37 | 41.63 | 35.46 | 44.33 | 31.09 | 37.52 | 36.84 | 43.50 | 37.45 | 6.06 |
| gen_augmenters | Generative | 44.92 | 41.38 | 34.70 | 43.42 | 30.43 | 37.01 | 36.76 | 43.15 | 36.91 | 6.24 |
| gen_automold | Generative | 45.32 | 41.84 | 34.85 | 43.22 | 30.79 | 37.92 | 37.24 | 43.58 | 37.29 | 6.29 |
| gen_cycleGAN | Generative | 45.70 | 41.54 | 35.60 | 43.58 | 30.88 | 37.71 | 37.42 | 43.62 | 37.40 | 6.22 |
| gen_cyclediffusion | Generative | 45.56 | 41.66 | 35.11 | 43.87 | 30.47 | 38.23 | 37.18 | 43.61 | 37.44 | 6.17 |
| gen_flux_kontext | Generative | 45.52 | 41.77 | 35.11 | 43.60 | 30.78 | 37.76 | 37.44 | 43.65 | 37.39 | 6.25 |
| gen_stargan_v2 | Generative | 45.55 | 41.88 | 35.83 | 44.03 | 31.05 | 38.12 | 37.21 | 43.72 | 37.60 | 6.11 |
| gen_step1x_new | Generative | 45.46 | 41.80 | 35.03 | 43.62 | 30.84 | 37.80 | 37.56 | 43.63 | 37.45 | 6.18 |
| gen_step1x_v1p2 | Generative | 45.02 | 41.51 | 35.32 | 43.46 | 30.79 | 37.49 | 36.80 | 43.26 | 37.13 | 6.13 |
| photometric_distort | Augmentation | 44.93 | 41.26 | 34.35 | 43.71 | 31.11 | 37.29 | 36.25 | 43.09 | 37.09 | 6.01 |
| std_autoaugment | Standard Aug | 45.48 | 41.69 | 36.15 | 45.46 | 32.64 | 37.79 | 37.47 | 43.58 | 38.34 | 5.24 |
| std_cutmix | Standard Aug | 44.61 | 41.11 | 34.01 | 44.12 | 30.12 | 36.60 | 36.74 | 42.86 | 36.89 | 5.97 |
| std_mixup | Standard Aug | 45.27 | 41.12 | 34.60 | 43.29 | 29.87 | 37.48 | 36.84 | 43.19 | 36.87 | 6.32 |
| std_randaugment | Standard Aug | 44.50 | 41.07 | 35.48 | 43.11 | 31.71 | 37.21 | 37.05 | 42.79 | 37.27 | 5.52 |
