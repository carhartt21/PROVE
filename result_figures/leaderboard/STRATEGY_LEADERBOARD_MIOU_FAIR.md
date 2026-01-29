# Stage 1 Strategy Leaderboard (Fair Comparison) (by mIoU)

**Stage 1**: All models trained with `clear_day` domain filter only.

**Metric**: mIoU (Mean Intersection over Union)

**Fair Comparison Mode**: Only includes dataset+model configurations where ALL strategies have test results.
This ensures equal coverage and prevents incomplete results from skewing rankings.

**Total Results**: 168 test results from 28 strategies

---

## Overall Strategy Ranking

Sorted by mIoU. Gain = improvement over baseline. Domain Gap = Normal mIoU - Adverse mIoU (lower is better).

| Strategy | Type | mIoU | Std | Gain | Normal mIoU | Adverse mIoU | Domain Gap | Num Tests |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| std_minimal | Standard Aug | 43.52 | 3.15 | 0.69 | 45.43 | 36.29 | 9.14 | 6 |
| std_photometric_distort | Augmentation | 43.08 | 3.26 | 0.24 | 44.73 | 36.66 | 8.07 | 6 |
| gen_step1x_v1p2 | Generative | 43.06 | 3.47 | 0.23 | 45.03 | 36.03 | 9.01 | 6 |
| gen_cycleGAN | Generative | 42.99 | 3.39 | 0.15 | 44.62 | 35.64 | 8.98 | 6 |
| gen_Attribute_Hallucination | Generative | 42.93 | 3.37 | 0.1 | 44.81 | 35.74 | 9.07 | 6 |
| gen_augmenters | Generative | 42.93 | 3.42 | 0.1 | 44.5 | 35.72 | 8.79 | 6 |
| gen_SUSTechGAN | Generative | 42.91 | 3.4 | 0.07 | 44.58 | 35.77 | 8.82 | 6 |
| gen_TSIT | Generative | 42.9 | 3.53 | 0.07 | 44.35 | 36.2 | 8.15 | 6 |
| std_cutmix | Standard Aug | 42.88 | 3.29 | 0.05 | 44.67 | 35.58 | 9.09 | 6 |
| gen_flux_kontext | Generative | 42.87 | 3.36 | 0.03 | 44.6 | 36.12 | 8.48 | 6 |
| gen_UniControl | Generative | 42.86 | 3.22 | 0.03 | 44.59 | 35.9 | 8.69 | 6 |
| gen_automold | Generative | 42.85 | 3.29 | 0.01 | 44.64 | 35.34 | 9.3 | 6 |
| std_mixup | Standard Aug | 42.83 | 3.09 | -0.0 | 44.43 | 35.91 | 8.52 | 6 |
| baseline | Baseline | 42.83 | 3.45 | - | 44.46 | 35.98 | 8.48 | 6 |
| gen_Img2Img | Generative | 42.82 | 3.17 | -0.01 | 44.78 | 35.95 | 8.83 | 6 |
| gen_CUT | Generative | 42.8 | 3.37 | -0.03 | 44.51 | 35.98 | 8.53 | 6 |
| gen_step1x_new | Generative | 42.78 | 3.07 | -0.05 | 44.63 | 35.68 | 8.95 | 6 |
| gen_Weather_Effect_Generator | Generative | 42.77 | 3.19 | -0.06 | 44.73 | 35.6 | 9.13 | 6 |
| std_autoaugment | Standard Aug | 42.76 | 3.5 | -0.08 | 44.44 | 35.26 | 9.18 | 6 |
| gen_VisualCloze | Generative | 42.75 | 3.31 | -0.09 | 44.51 | 35.43 | 9.09 | 6 |
| gen_stargan_v2 | Generative | 42.71 | 3.14 | -0.12 | 44.41 | 35.73 | 8.68 | 6 |
| gen_IP2P | Generative | 42.71 | 3.19 | -0.12 | 44.56 | 36.17 | 8.39 | 6 |
| gen_Qwen_Image_Edit | Generative | 42.7 | 3.16 | -0.13 | 44.69 | 35.66 | 9.03 | 6 |
| gen_albumentations_weather | Generative | 42.7 | 3.25 | -0.14 | 44.63 | 35.16 | 9.47 | 6 |
| gen_cyclediffusion | Generative | 42.69 | 3.08 | -0.15 | 44.47 | 35.64 | 8.82 | 6 |
| gen_CNetSeg | Generative | 42.68 | 3.12 | -0.15 | 44.31 | 35.73 | 8.59 | 6 |
| gen_LANIT | Generative | 42.67 | 3.14 | -0.17 | 44.34 | 35.83 | 8.51 | 6 |
| std_randaugment | Standard Aug | 42.57 | 3.27 | -0.26 | 44.18 | 35.42 | 8.76 | 6 |
---

## Per-Dataset Breakdown

mIoU performance on each dataset.

| Strategy | Type | bdd10k | bdd10k_gain | idd-aw | idd-aw_gain | mapillaryvistas | mapillaryvistas_gain | outside15k | outside15k_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| std_minimal | Standard Aug | 46.15 | +0.54 | 40.89 | +0.83 | - | - | - | - |
| std_photometric_distort | Augmentation | 45.62 | +0.01 | 40.53 | +0.48 | - | - | - | - |
| gen_step1x_v1p2 | Generative | 45.73 | +0.12 | 40.40 | +0.34 | - | - | - | - |
| gen_cycleGAN | Generative | 45.77 | +0.16 | 40.21 | +0.15 | - | - | - | - |
| gen_Attribute_Hallucination | Generative | 45.61 | -0.00 | 40.26 | +0.20 | - | - | - | - |
| gen_augmenters | Generative | 45.68 | +0.07 | 40.18 | +0.12 | - | - | - | - |
| gen_SUSTechGAN | Generative | 45.64 | +0.03 | 40.18 | +0.12 | - | - | - | - |
| gen_TSIT | Generative | 45.71 | +0.11 | 40.08 | +0.03 | - | - | - | - |
| std_cutmix | Standard Aug | 45.44 | -0.17 | 40.32 | +0.26 | - | - | - | - |
| gen_flux_kontext | Generative | 45.51 | -0.10 | 40.22 | +0.16 | - | - | - | - |
| gen_UniControl | Generative | 45.45 | -0.15 | 40.26 | +0.21 | - | - | - | - |
| gen_automold | Generative | 45.44 | -0.16 | 40.25 | +0.19 | - | - | - | - |
| baseline | Baseline | 45.61 | - | 40.06 | -0.00 | - | - | - | - |
| std_mixup | Standard Aug | 45.35 | -0.26 | 40.31 | +0.25 | - | - | - | - |
| gen_Img2Img | Generative | 45.39 | -0.22 | 40.25 | +0.19 | - | - | - | - |
| gen_CUT | Generative | 45.46 | -0.14 | 40.14 | +0.08 | - | - | - | - |
| gen_step1x_new | Generative | 45.22 | -0.38 | 40.34 | +0.28 | - | - | - | - |
| gen_Weather_Effect_Generator | Generative | 45.28 | -0.33 | 40.27 | +0.21 | - | - | - | - |
| std_autoaugment | Standard Aug | 45.49 | -0.11 | 40.02 | -0.04 | - | - | - | - |
| gen_VisualCloze | Generative | 45.41 | -0.20 | 40.08 | +0.03 | - | - | - | - |
| gen_stargan_v2 | Generative | 45.28 | -0.33 | 40.15 | +0.09 | - | - | - | - |
| gen_IP2P | Generative | 45.22 | -0.38 | 40.20 | +0.14 | - | - | - | - |
| gen_Qwen_Image_Edit | Generative | 45.25 | -0.36 | 40.15 | +0.09 | - | - | - | - |
| gen_albumentations_weather | Generative | 45.26 | -0.34 | 40.13 | +0.07 | - | - | - | - |
| gen_cyclediffusion | Generative | 45.17 | -0.44 | 40.20 | +0.14 | - | - | - | - |
| gen_CNetSeg | Generative | 45.16 | -0.44 | 40.19 | +0.14 | - | - | - | - |
| gen_LANIT | Generative | 45.19 | -0.41 | 40.14 | +0.08 | - | - | - | - |
| std_randaugment | Standard Aug | 45.22 | -0.39 | 39.92 | -0.14 | - | - | - | - |
---

## Per-Domain Breakdown

mIoU performance on each weather domain. Normal = clear_day, cloudy. Adverse = foggy, night, rainy, snowy.

| Strategy | Type | clear_day | cloudy | dawn_dusk | foggy | night | rainy | snowy | Normal Avg | Adverse Avg | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Baseline | 46.00 | 42.93 | 36.10 | 42.43 | 26.77 | 36.08 | 38.65 | 44.46 | 35.98 | 8.48 |
| gen_Attribute_Hallucination | Generative | 46.07 | 43.54 | 36.36 | 41.75 | 26.62 | 35.97 | 38.60 | 44.81 | 35.74 | 9.07 |
| gen_CNetSeg | Generative | 45.83 | 42.80 | 36.27 | 41.63 | 26.89 | 35.46 | 38.94 | 44.31 | 35.73 | 8.59 |
| gen_CUT | Generative | 45.96 | 43.05 | 36.01 | 42.93 | 26.67 | 35.59 | 38.72 | 44.51 | 35.98 | 8.53 |
| gen_IP2P | Generative | 45.92 | 43.20 | 36.10 | 43.69 | 27.03 | 35.48 | 38.47 | 44.56 | 36.17 | 8.39 |
| gen_Img2Img | Generative | 46.08 | 43.47 | 36.66 | 43.17 | 26.55 | 35.41 | 38.68 | 44.78 | 35.95 | 8.83 |
| gen_LANIT | Generative | 45.93 | 42.74 | 36.02 | 42.76 | 26.59 | 35.49 | 38.47 | 44.34 | 35.83 | 8.51 |
| gen_Qwen_Image_Edit | Generative | 45.89 | 43.48 | 35.90 | 42.51 | 26.57 | 35.34 | 38.21 | 44.69 | 35.66 | 9.03 |
| gen_SUSTechGAN | Generative | 46.21 | 42.96 | 36.01 | 43.02 | 26.67 | 35.53 | 37.85 | 44.58 | 35.77 | 8.82 |
| gen_TSIT | Generative | 45.92 | 42.78 | 36.01 | 43.09 | 26.62 | 36.12 | 38.96 | 44.35 | 36.20 | 8.15 |
| gen_UniControl | Generative | 46.11 | 43.07 | 36.56 | 42.75 | 26.57 | 35.72 | 38.55 | 44.59 | 35.90 | 8.69 |
| gen_VisualCloze | Generative | 45.98 | 43.05 | 36.70 | 41.28 | 26.41 | 35.03 | 38.99 | 44.51 | 35.43 | 9.09 |
| gen_Weather_Effect_Generator | Generative | 46.12 | 43.33 | 36.31 | 42.25 | 26.11 | 35.42 | 38.62 | 44.73 | 35.60 | 9.13 |
| gen_albumentations_weather | Generative | 46.16 | 43.09 | 36.05 | 41.51 | 26.15 | 34.63 | 38.33 | 44.63 | 35.16 | 9.47 |
| gen_augmenters | Generative | 46.12 | 42.88 | 35.99 | 42.00 | 26.75 | 35.81 | 38.31 | 44.50 | 35.72 | 8.79 |
| gen_automold | Generative | 46.11 | 43.17 | 36.44 | 41.33 | 26.30 | 35.49 | 38.24 | 44.64 | 35.34 | 9.30 |
| gen_cycleGAN | Generative | 46.24 | 43.00 | 36.16 | 41.62 | 26.38 | 35.38 | 39.16 | 44.62 | 35.64 | 8.98 |
| gen_cyclediffusion | Generative | 45.87 | 43.07 | 36.59 | 41.76 | 27.00 | 35.27 | 38.55 | 44.47 | 35.64 | 8.82 |
| gen_flux_kontext | Generative | 45.90 | 43.31 | 36.36 | 43.02 | 26.66 | 35.78 | 39.01 | 44.60 | 36.12 | 8.48 |
| gen_stargan_v2 | Generative | 45.92 | 42.90 | 36.06 | 42.49 | 26.25 | 35.55 | 38.64 | 44.41 | 35.73 | 8.68 |
| gen_step1x_new | Generative | 45.89 | 43.37 | 36.23 | 42.19 | 26.13 | 35.79 | 38.59 | 44.63 | 35.68 | 8.95 |
| gen_step1x_v1p2 | Generative | 46.19 | 43.88 | 36.21 | 42.39 | 27.06 | 35.89 | 38.76 | 45.03 | 36.03 | 9.01 |
| std_autoaugment | Standard Aug | 46.02 | 42.85 | 36.17 | 40.38 | 26.65 | 35.20 | 38.80 | 44.44 | 35.26 | 9.18 |
| std_cutmix | Standard Aug | 46.13 | 43.21 | 35.63 | 41.06 | 26.77 | 35.65 | 38.84 | 44.67 | 35.58 | 9.09 |
| std_minimal | Standard Aug | 46.93 | 43.94 | 36.67 | 42.60 | 26.47 | 36.44 | 39.65 | 45.43 | 36.29 | 9.14 |
| std_mixup | Standard Aug | 45.92 | 42.95 | 36.55 | 42.62 | 26.68 | 35.72 | 38.62 | 44.43 | 35.91 | 8.52 |
| std_photometric_distort | Augmentation | 45.98 | 43.48 | 37.46 | 43.48 | 27.52 | 36.70 | 38.96 | 44.73 | 36.66 | 8.07 |
| std_randaugment | Standard Aug | 45.79 | 42.56 | 36.42 | 42.17 | 26.07 | 35.31 | 38.13 | 44.18 | 35.42 | 8.76 |
