# Stage 1 Strategy Leaderboard (Fair Comparison) (by mIoU)

**Stage 1**: All models trained with `clear_day` domain filter only.

**Metric**: mIoU (Mean Intersection over Union)

**Fair Comparison Mode**: Only includes dataset+model configurations where ALL strategies have test results.
This ensures equal coverage and prevents incomplete results from skewing rankings.

**Total Results**: 156 test results from 26 strategies

---

## Overall Strategy Ranking

Sorted by mIoU. Gain = improvement over baseline. Domain Gap = Normal mIoU - Adverse mIoU (lower is better).

| Strategy | Type | mIoU | Std | Gain | Normal mIoU | Adverse mIoU | Domain Gap | Num Tests |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_Attribute_Hallucination | Generative | 42.56 | 5.79 | 2.9 | 43.08 | 37.54 | 5.54 | 6 |
| gen_UniControl | Generative | 42.51 | 5.6 | 2.84 | 43.2 | 37.78 | 5.42 | 6 |
| gen_VisualCloze | Generative | 42.46 | 5.85 | 2.8 | 43.3 | 38.16 | 5.13 | 6 |
| gen_automold | Generative | 42.45 | 5.4 | 2.79 | 43.23 | 37.5 | 5.74 | 6 |
| gen_CNetSeg | Generative | 42.44 | 5.57 | 2.78 | 43.19 | 37.55 | 5.64 | 6 |
| gen_Qwen_Image_Edit | Generative | 42.38 | 5.36 | 2.72 | 43.13 | 37.57 | 5.56 | 6 |
| gen_stargan_v2 | Generative | 42.32 | 5.81 | 2.66 | 43.26 | 37.33 | 5.94 | 6 |
| std_autoaugment | Standard Aug | 42.25 | 5.31 | 2.59 | 42.62 | 37.67 | 4.95 | 6 |
| gen_IP2P | Generative | 42.23 | 5.22 | 2.56 | 43.11 | 37.23 | 5.88 | 6 |
| gen_SUSTechGAN | Generative | 42.17 | 5.3 | 2.5 | 42.94 | 37.27 | 5.67 | 6 |
| gen_augmenters | Generative | 42.14 | 5.17 | 2.48 | 42.96 | 37.18 | 5.78 | 6 |
| std_mixup | Standard Aug | 42.11 | 5.26 | 2.45 | 42.99 | 37.15 | 5.84 | 6 |
| std_randaugment | Standard Aug | 42.05 | 5.15 | 2.38 | 42.76 | 37.23 | 5.53 | 6 |
| gen_Weather_Effect_Generator | Generative | 41.94 | 5.0 | 2.28 | 42.94 | 37.15 | 5.79 | 6 |
| gen_CUT | Generative | 41.9 | 4.87 | 2.24 | 42.74 | 37.11 | 5.63 | 6 |
| gen_cyclediffusion | Generative | 41.86 | 5.21 | 2.2 | 42.92 | 37.49 | 5.43 | 6 |
| std_cutmix | Standard Aug | 41.63 | 5.84 | 1.96 | 42.57 | 36.89 | 5.68 | 6 |
| gen_Img2Img | Generative | 41.58 | 4.49 | 1.91 | 42.3 | 36.87 | 5.43 | 6 |
| gen_albumentations_weather | Generative | 41.46 | 4.54 | 1.8 | 42.14 | 36.86 | 5.28 | 6 |
| gen_cycleGAN | Generative | 41.15 | 4.89 | 1.48 | 42.1 | 36.39 | 5.71 | 6 |
| gen_LANIT | Generative | 41.15 | 5.05 | 1.48 | 42.13 | 36.51 | 5.62 | 6 |
| gen_flux_kontext | Generative | 41.14 | 4.54 | 1.48 | 42.03 | 36.22 | 5.81 | 6 |
| gen_step1x_new | Generative | 41.1 | 4.96 | 1.43 | 42.1 | 36.42 | 5.68 | 6 |
| gen_TSIT | Generative | 40.98 | 8.13 | 1.31 | 41.82 | 36.19 | 5.63 | 6 |
| gen_step1x_v1p2 | Generative | 40.72 | 4.08 | 1.06 | 41.44 | 36.16 | 5.28 | 6 |
| baseline | Baseline | 39.67 | 8.62 | - | 40.4 | 34.76 | 5.64 | 6 |
---

## Per-Dataset Breakdown

mIoU performance on each dataset.

| Strategy | Type | bdd10k | bdd10k_gain | idd-aw | idd-aw_gain | mapillaryvistas | mapillaryvistas_gain | outside15k | outside15k_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_Attribute_Hallucination | Generative | 49.12 | +0.21 | - | - | 34.90 | +7.20 | 43.14 | +6.27 |
| gen_UniControl | Generative | 48.81 | -0.10 | - | - | 35.10 | +7.40 | 43.08 | +6.21 |
| gen_CNetSeg | Generative | 48.72 | -0.20 | - | - | 35.02 | +7.33 | 43.18 | +6.31 |
| gen_Qwen_Image_Edit | Generative | 48.27 | -0.65 | - | - | 35.05 | +7.36 | 43.46 | +6.59 |
| gen_stargan_v2 | Generative | 48.61 | -0.30 | - | - | 34.56 | +6.86 | 43.44 | +6.57 |
| gen_automold | Generative | 48.46 | -0.45 | - | - | 34.99 | +7.30 | 43.12 | +6.25 |
| gen_VisualCloze | Generative | 48.88 | -0.03 | - | - | 34.81 | +7.11 | 42.68 | +5.81 |
| gen_IP2P | Generative | 47.84 | -1.08 | - | - | 34.81 | +7.12 | 43.53 | +6.66 |
| gen_augmenters | Generative | 47.64 | -1.27 | - | - | 34.87 | +7.18 | 43.52 | +6.65 |
| std_autoaugment | Standard Aug | 48.12 | -0.79 | - | - | 34.88 | +7.18 | 43.00 | +6.13 |
| std_cutmix | Standard Aug | 47.97 | -0.94 | - | - | 34.89 | +7.19 | 42.96 | +6.09 |
| std_mixup | Standard Aug | 47.93 | -0.98 | - | - | 34.92 | +7.22 | 42.84 | +5.97 |
| gen_SUSTechGAN | Generative | 47.99 | -0.92 | - | - | 34.75 | +7.05 | 42.91 | +6.04 |
| std_randaugment | Standard Aug | 47.63 | -1.28 | - | - | 34.73 | +7.03 | 43.22 | +6.35 |
| gen_Weather_Effect_Generator | Generative | 47.46 | -1.45 | - | - | 35.20 | +7.50 | 42.86 | +5.99 |
| gen_CUT | Generative | 47.17 | -1.74 | - | - | 34.89 | +7.20 | 43.12 | +6.25 |
| gen_cyclediffusion | Generative | 47.50 | -1.41 | - | - | 34.38 | +6.69 | 42.78 | +5.91 |
| gen_Img2Img | Generative | 46.28 | -2.64 | - | - | 34.85 | +7.15 | 43.01 | +6.14 |
| gen_albumentations_weather | Generative | 46.34 | -2.58 | - | - | 34.74 | +7.04 | 42.60 | +5.73 |
| gen_flux_kontext | Generative | 44.49 | -4.42 | - | - | 35.23 | +7.54 | 43.21 | +6.34 |
| gen_cycleGAN | Generative | 44.76 | -4.15 | - | - | 34.98 | +7.28 | 43.10 | +6.23 |
| gen_LANIT | Generative | 44.97 | -3.94 | - | - | 35.08 | +7.39 | 42.70 | +5.83 |
| gen_step1x_new | Generative | 44.80 | -4.12 | - | - | 35.02 | +7.33 | 42.49 | +5.62 |
| gen_step1x_v1p2 | Generative | 43.81 | -5.10 | - | - | 34.75 | +7.05 | 42.92 | +6.05 |
| gen_TSIT | Generative | 48.44 | -0.48 | - | - | 26.81 | -0.89 | 42.96 | +6.10 |
| baseline | Baseline | 48.91 | - | - | - | 27.70 | - | 36.87 | - |
---

## Per-Domain Breakdown

mIoU performance on each weather domain. Normal = clear_day, cloudy. Adverse = foggy, night, rainy, snowy.

| Strategy | Type | clear_day | cloudy | dawn_dusk | foggy | night | rainy | snowy | Normal Avg | Adverse Avg | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Baseline | 41.35 | 39.46 | 33.55 | 38.48 | 27.85 | 36.60 | 36.10 | 40.40 | 34.76 | 5.64 |
| gen_Attribute_Hallucination | Generative | 44.57 | 41.60 | 36.99 | 41.79 | 30.56 | 38.76 | 39.04 | 43.08 | 37.54 | 5.54 |
| gen_CNetSeg | Generative | 44.21 | 42.17 | 36.26 | 42.05 | 30.35 | 38.50 | 39.28 | 43.19 | 37.55 | 5.64 |
| gen_CUT | Generative | 43.97 | 41.51 | 36.18 | 42.04 | 30.72 | 37.57 | 38.10 | 42.74 | 37.11 | 5.63 |
| gen_IP2P | Generative | 44.17 | 42.04 | 36.26 | 41.23 | 30.43 | 38.31 | 38.96 | 43.11 | 37.23 | 5.88 |
| gen_Img2Img | Generative | 43.62 | 40.98 | 36.28 | 41.25 | 30.56 | 37.76 | 37.91 | 42.30 | 36.87 | 5.43 |
| gen_LANIT | Generative | 43.15 | 41.12 | 35.76 | 41.36 | 29.81 | 37.58 | 37.30 | 42.13 | 36.51 | 5.62 |
| gen_Qwen_Image_Edit | Generative | 44.20 | 42.06 | 36.52 | 42.00 | 31.11 | 38.70 | 38.48 | 43.13 | 37.57 | 5.56 |
| gen_SUSTechGAN | Generative | 44.09 | 41.79 | 36.32 | 41.51 | 30.00 | 38.32 | 39.25 | 42.94 | 37.27 | 5.67 |
| gen_TSIT | Generative | 42.81 | 40.83 | 34.65 | 39.81 | 30.21 | 36.97 | 37.78 | 41.82 | 36.19 | 5.63 |
| gen_UniControl | Generative | 44.47 | 41.94 | 36.02 | 42.96 | 30.58 | 38.76 | 38.84 | 43.20 | 37.78 | 5.42 |
| gen_VisualCloze | Generative | 44.48 | 42.11 | 36.12 | 43.20 | 31.65 | 38.82 | 38.99 | 43.30 | 38.16 | 5.13 |
| gen_Weather_Effect_Generator | Generative | 43.87 | 42.00 | 36.65 | 41.20 | 30.33 | 38.55 | 38.52 | 42.94 | 37.15 | 5.79 |
| gen_albumentations_weather | Generative | 43.38 | 40.90 | 36.33 | 42.03 | 29.92 | 37.70 | 37.80 | 42.14 | 36.86 | 5.28 |
| gen_augmenters | Generative | 44.21 | 41.70 | 36.93 | 41.61 | 30.34 | 38.00 | 38.74 | 42.96 | 37.18 | 5.78 |
| gen_automold | Generative | 44.31 | 42.15 | 35.81 | 41.91 | 30.58 | 38.58 | 38.91 | 43.23 | 37.50 | 5.74 |
| gen_cycleGAN | Generative | 43.11 | 41.08 | 36.00 | 41.20 | 29.87 | 37.04 | 37.44 | 42.10 | 36.39 | 5.71 |
| gen_cyclediffusion | Generative | 43.75 | 42.09 | 36.23 | 42.83 | 30.77 | 38.06 | 38.29 | 42.92 | 37.49 | 5.43 |
| gen_flux_kontext | Generative | 43.19 | 40.86 | 36.21 | 40.51 | 30.02 | 37.53 | 36.80 | 42.03 | 36.22 | 5.81 |
| gen_stargan_v2 | Generative | 44.27 | 42.26 | 35.80 | 41.92 | 30.35 | 38.60 | 38.44 | 43.26 | 37.33 | 5.94 |
| gen_step1x_new | Generative | 42.97 | 41.23 | 35.69 | 41.64 | 29.74 | 37.12 | 37.18 | 42.10 | 36.42 | 5.68 |
| gen_step1x_v1p2 | Generative | 42.59 | 40.30 | 35.73 | 41.23 | 29.71 | 36.78 | 36.93 | 41.44 | 36.16 | 5.28 |
| std_autoaugment | Standard Aug | 44.01 | 41.24 | 36.46 | 41.65 | 30.92 | 39.39 | 38.73 | 42.62 | 37.67 | 4.95 |
| std_cutmix | Standard Aug | 43.55 | 41.59 | 35.30 | 41.07 | 29.96 | 37.88 | 38.65 | 42.57 | 36.89 | 5.68 |
| std_mixup | Standard Aug | 44.06 | 41.91 | 36.17 | 41.21 | 29.83 | 38.83 | 38.73 | 42.99 | 37.15 | 5.84 |
| std_randaugment | Standard Aug | 43.96 | 41.56 | 35.97 | 42.00 | 30.13 | 38.16 | 38.64 | 42.76 | 37.23 | 5.53 |
