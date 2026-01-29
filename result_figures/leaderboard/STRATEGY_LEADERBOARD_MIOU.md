# Stage 1 Strategy Leaderboard (by mIoU)

**Stage 1**: All models trained with `clear_day` domain filter only.

**Metric**: mIoU (Mean Intersection over Union)

**Total Results**: 309 test results from 28 strategies

---

## Overall Strategy Ranking

Sorted by mIoU. Gain = improvement over baseline. Domain Gap = Normal mIoU - Adverse mIoU (lower is better).

| Strategy | Type | mIoU | Std | Gain | Normal mIoU | Adverse mIoU | Domain Gap | Num Tests |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_augmenters | Generative | 42.93 | 3.42 | 2.01 | 44.5 | 35.72 | 8.79 | 6 |
| gen_TSIT | Generative | 42.9 | 3.53 | 1.98 | 44.35 | 36.2 | 8.15 | 6 |
| gen_Weather_Effect_Generator | Generative | 42.77 | 3.19 | 1.86 | 44.73 | 35.6 | 9.13 | 6 |
| gen_Qwen_Image_Edit | Generative | 42.21 | 3.45 | 1.29 | 43.44 | 35.83 | 7.61 | 9 |
| std_minimal | Standard Aug | 41.28 | 4.39 | 0.36 | 42.19 | 35.34 | 6.85 | 12 |
| gen_stargan_v2 | Generative | 41.25 | 3.68 | 0.34 | 42.45 | 34.93 | 7.51 | 9 |
| gen_cycleGAN | Generative | 40.96 | 4.35 | 0.04 | 41.81 | 35.07 | 6.74 | 12 |
| gen_step1x_v1p2 | Generative | 40.96 | 4.42 | 0.04 | 42.02 | 35.3 | 6.72 | 12 |
| gen_Attribute_Hallucination | Generative | 40.93 | 4.39 | 0.01 | 41.87 | 35.13 | 6.74 | 12 |
| std_cutmix | Standard Aug | 40.93 | 4.33 | 0.01 | 41.83 | 35.04 | 6.79 | 12 |
| gen_SUSTechGAN | Generative | 40.92 | 4.31 | 0.0 | 41.76 | 35.1 | 6.66 | 12 |
| baseline | Baseline | 40.92 | 4.38 | - | 41.8 | 35.25 | 6.55 | 12 |
| std_photometric_distort | Augmentation | 40.91 | 4.4 | -0.01 | 41.83 | 35.45 | 6.38 | 12 |
| std_mixup | Standard Aug | 40.9 | 4.13 | -0.02 | 41.71 | 35.25 | 6.47 | 12 |
| gen_flux_kontext | Generative | 40.85 | 4.37 | -0.07 | 41.73 | 35.37 | 6.36 | 12 |
| gen_CUT | Generative | 40.83 | 4.32 | -0.09 | 41.7 | 35.16 | 6.55 | 12 |
| gen_step1x_new | Generative | 40.82 | 4.22 | -0.1 | 41.77 | 35.07 | 6.69 | 12 |
| gen_UniControl | Generative | 40.82 | 4.26 | -0.09 | 41.72 | 35.02 | 6.71 | 12 |
| gen_IP2P | Generative | 40.81 | 4.15 | -0.11 | 41.77 | 35.2 | 6.57 | 12 |
| gen_LANIT | Generative | 40.81 | 4.27 | -0.11 | 41.57 | 35.17 | 6.4 | 12 |
| std_autoaugment | Standard Aug | 40.81 | 4.36 | -0.11 | 41.68 | 34.78 | 6.9 | 12 |
| gen_VisualCloze | Generative | 40.8 | 4.35 | -0.12 | 41.76 | 34.81 | 6.95 | 12 |
| gen_cyclediffusion | Generative | 40.79 | 4.18 | -0.13 | 41.79 | 35.15 | 6.63 | 12 |
| gen_automold | Generative | 40.79 | 4.26 | -0.13 | 41.77 | 34.9 | 6.87 | 12 |
| gen_albumentations_weather | Generative | 40.79 | 4.26 | -0.13 | 41.76 | 34.89 | 6.87 | 12 |
| gen_Img2Img | Generative | 40.78 | 4.22 | -0.14 | 41.8 | 35.24 | 6.56 | 12 |
| std_randaugment | Standard Aug | 40.68 | 4.21 | -0.23 | 41.48 | 34.94 | 6.54 | 12 |
| gen_CNetSeg | Generative | 40.53 | 4.33 | -0.38 | 41.71 | 34.75 | 6.96 | 9 |
---

## Per-Dataset Breakdown

mIoU performance on each dataset.

| Strategy | Type | bdd10k | bdd10k_gain | idd-aw | idd-aw_gain | mapillaryvistas | mapillaryvistas_gain | outside15k | outside15k_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_augmenters | Generative | 45.68 | +0.07 | 40.18 | +0.12 | - | - | - | - |
| gen_TSIT | Generative | 45.71 | +0.11 | 40.08 | +0.03 | - | - | - | - |
| gen_Weather_Effect_Generator | Generative | 45.28 | -0.33 | 40.27 | +0.21 | - | - | - | - |
| gen_Qwen_Image_Edit | Generative | 45.25 | -0.36 | 40.15 | +0.09 | 37.62 | +2.14 | 43.03 | +0.50 |
| std_minimal | Standard Aug | 46.15 | +0.54 | 40.89 | +0.83 | 35.71 | +0.24 | 42.38 | -0.15 |
| gen_cycleGAN | Generative | 45.77 | +0.16 | 40.21 | +0.15 | 35.57 | +0.09 | 42.31 | -0.22 |
| gen_step1x_v1p2 | Generative | 45.73 | +0.12 | 40.40 | +0.34 | 35.40 | -0.08 | 42.30 | -0.23 |
| gen_Attribute_Hallucination | Generative | 45.61 | -0.00 | 40.26 | +0.20 | 35.44 | -0.04 | 42.41 | -0.12 |
| std_cutmix | Standard Aug | 45.44 | -0.17 | 40.32 | +0.26 | 35.56 | +0.08 | 42.38 | -0.15 |
| gen_SUSTechGAN | Generative | 45.64 | +0.03 | 40.18 | +0.12 | 35.73 | +0.26 | 42.14 | -0.39 |
| baseline | Baseline | 45.61 | - | 40.06 | -0.00 | 35.48 | - | 42.53 | - |
| std_photometric_distort | Augmentation | 45.62 | +0.01 | 40.53 | +0.48 | 35.36 | -0.11 | 42.13 | -0.40 |
| std_mixup | Standard Aug | 45.35 | -0.26 | 40.31 | +0.25 | 35.77 | +0.29 | 42.19 | -0.34 |
| gen_flux_kontext | Generative | 45.51 | -0.10 | 40.22 | +0.16 | 35.48 | +0.00 | 42.19 | -0.34 |
| gen_CUT | Generative | 45.46 | -0.14 | 40.14 | +0.08 | 35.45 | -0.03 | 42.27 | -0.26 |
| gen_UniControl | Generative | 45.45 | -0.15 | 40.26 | +0.21 | 35.52 | +0.05 | 42.05 | -0.48 |
| gen_step1x_new | Generative | 45.22 | -0.38 | 40.34 | +0.28 | 35.47 | -0.01 | 42.25 | -0.28 |
| std_autoaugment | Standard Aug | 45.49 | -0.11 | 40.02 | -0.04 | 35.47 | -0.00 | 42.26 | -0.28 |
| gen_IP2P | Generative | 45.22 | -0.38 | 40.20 | +0.14 | 35.62 | +0.15 | 42.18 | -0.35 |
| gen_LANIT | Generative | 45.19 | -0.41 | 40.14 | +0.08 | 35.47 | -0.00 | 42.42 | -0.11 |
| gen_VisualCloze | Generative | 45.41 | -0.20 | 40.08 | +0.03 | 35.38 | -0.10 | 42.34 | -0.19 |
| gen_automold | Generative | 45.44 | -0.16 | 40.25 | +0.19 | 35.56 | +0.09 | 41.92 | -0.62 |
| gen_albumentations_weather | Generative | 45.26 | -0.34 | 40.13 | +0.07 | 35.58 | +0.10 | 42.19 | -0.34 |
| gen_cyclediffusion | Generative | 45.17 | -0.44 | 40.20 | +0.14 | 35.45 | -0.02 | 42.32 | -0.21 |
| gen_Img2Img | Generative | 45.39 | -0.22 | 40.25 | +0.19 | 35.48 | +0.00 | 41.98 | -0.55 |
| std_randaugment | Standard Aug | 45.22 | -0.39 | 39.92 | -0.14 | 35.43 | -0.04 | 42.17 | -0.37 |
| gen_stargan_v2 | Generative | 45.28 | -0.33 | 40.15 | +0.09 | 34.65 | -0.83 | 40.17 | -2.36 |
| gen_CNetSeg | Generative | 45.16 | -0.44 | 40.19 | +0.14 | 34.47 | -1.00 | 39.79 | -2.74 |
---

## Per-Domain Breakdown

mIoU performance on each weather domain. Normal = clear_day, cloudy. Adverse = foggy, night, rainy, snowy.

| Strategy | Type | clear_day | cloudy | dawn_dusk | foggy | night | rainy | snowy | Normal Avg | Adverse Avg | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Baseline | 43.46 | 40.14 | 34.01 | 41.32 | 28.61 | 35.84 | 35.21 | 41.80 | 35.25 | 6.55 |
| gen_Attribute_Hallucination | Generative | 43.44 | 40.29 | 34.75 | 40.75 | 28.25 | 36.06 | 35.45 | 41.87 | 35.13 | 6.74 |
| gen_CNetSeg | Generative | 43.11 | 40.31 | 35.07 | 40.53 | 27.72 | 34.85 | 35.89 | 41.71 | 34.75 | 6.96 |
| gen_CUT | Generative | 43.37 | 40.04 | 34.09 | 41.15 | 28.52 | 35.77 | 35.20 | 41.70 | 35.16 | 6.55 |
| gen_IP2P | Generative | 43.38 | 40.17 | 34.37 | 41.66 | 28.58 | 35.63 | 34.93 | 41.77 | 35.20 | 6.57 |
| gen_Img2Img | Generative | 43.33 | 40.26 | 34.36 | 41.43 | 28.52 | 35.57 | 35.43 | 41.80 | 35.24 | 6.56 |
| gen_LANIT | Generative | 43.31 | 39.83 | 34.09 | 41.13 | 28.53 | 35.85 | 35.16 | 41.57 | 35.17 | 6.40 |
| gen_Qwen_Image_Edit | Generative | 45.05 | 41.84 | 34.55 | 41.98 | 28.44 | 36.17 | 36.73 | 43.44 | 35.83 | 7.61 |
| gen_SUSTechGAN | Generative | 43.51 | 40.01 | 34.24 | 41.56 | 28.25 | 35.74 | 34.84 | 41.76 | 35.10 | 6.66 |
| gen_TSIT | Generative | 45.92 | 42.78 | 36.01 | 43.09 | 26.62 | 36.12 | 38.96 | 44.35 | 36.20 | 8.15 |
| gen_UniControl | Generative | 43.41 | 40.03 | 34.56 | 41.05 | 27.92 | 35.99 | 35.10 | 41.72 | 35.02 | 6.71 |
| gen_VisualCloze | Generative | 43.39 | 40.13 | 34.19 | 40.13 | 28.43 | 35.44 | 35.25 | 41.76 | 34.81 | 6.95 |
| gen_Weather_Effect_Generator | Generative | 46.12 | 43.33 | 36.31 | 42.25 | 26.11 | 35.42 | 38.62 | 44.73 | 35.60 | 9.13 |
| gen_albumentations_weather | Generative | 43.48 | 40.04 | 34.17 | 40.90 | 28.15 | 35.28 | 35.25 | 41.76 | 34.89 | 6.87 |
| gen_augmenters | Generative | 46.12 | 42.88 | 35.99 | 42.00 | 26.75 | 35.81 | 38.31 | 44.50 | 35.72 | 8.79 |
| gen_automold | Generative | 43.33 | 40.21 | 34.05 | 40.53 | 28.12 | 35.66 | 35.27 | 41.77 | 34.90 | 6.87 |
| gen_cycleGAN | Generative | 43.52 | 40.10 | 34.18 | 40.62 | 28.40 | 35.72 | 35.54 | 41.81 | 35.07 | 6.74 |
| gen_cyclediffusion | Generative | 43.36 | 40.21 | 34.34 | 40.80 | 28.60 | 35.63 | 35.57 | 41.79 | 35.15 | 6.63 |
| gen_flux_kontext | Generative | 43.30 | 40.16 | 34.44 | 41.48 | 28.52 | 36.03 | 35.46 | 41.73 | 35.37 | 6.36 |
| gen_stargan_v2 | Generative | 44.15 | 40.75 | 33.48 | 41.80 | 26.65 | 35.47 | 35.82 | 42.45 | 34.93 | 7.51 |
| gen_step1x_new | Generative | 43.30 | 40.23 | 34.25 | 41.09 | 28.10 | 35.81 | 35.28 | 41.77 | 35.07 | 6.69 |
| gen_step1x_v1p2 | Generative | 43.40 | 40.64 | 34.31 | 40.94 | 28.65 | 36.03 | 35.58 | 42.02 | 35.30 | 6.72 |
| std_autoaugment | Standard Aug | 43.33 | 40.04 | 34.44 | 39.76 | 28.23 | 35.69 | 35.44 | 41.68 | 34.78 | 6.90 |
| std_cutmix | Standard Aug | 43.47 | 40.19 | 33.97 | 40.45 | 28.45 | 35.69 | 35.56 | 41.83 | 35.04 | 6.79 |
| std_minimal | Standard Aug | 43.90 | 40.49 | 34.50 | 40.86 | 28.43 | 36.31 | 35.76 | 42.19 | 35.34 | 6.85 |
| std_mixup | Standard Aug | 43.36 | 40.06 | 34.19 | 41.26 | 28.48 | 35.97 | 35.26 | 41.71 | 35.25 | 6.47 |
| std_photometric_distort | Augmentation | 43.22 | 40.43 | 34.96 | 41.03 | 29.22 | 36.06 | 35.48 | 41.83 | 35.45 | 6.38 |
| std_randaugment | Standard Aug | 43.21 | 39.75 | 34.57 | 40.94 | 28.14 | 35.74 | 34.95 | 41.48 | 34.94 | 6.54 |
