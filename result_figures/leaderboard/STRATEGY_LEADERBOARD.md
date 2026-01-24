# Stage 1 Strategy Leaderboard

**Stage 1**: All models trained with `clear_day` domain filter only.

**Total Results**: 324 test results from 27 strategies

---

## Overall Strategy Ranking

Sorted by mIoU. Gain = improvement over baseline. Domain Gap = Normal mIoU - Adverse mIoU (lower is better).

| Strategy | Type | mIoU | Std | Gain | Normal mIoU | Adverse mIoU | Domain Gap | Num Tests |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_Attribute_Hallucination | Generative | 39.83 | 6.75 | 1.36 | 40.68 | 33.87 | 6.8 | 12 |
| gen_cycleGAN | Generative | 39.6 | 6.71 | 1.13 | 40.32 | 33.66 | 6.66 | 12 |
| gen_Img2Img | Generative | 39.58 | 6.81 | 1.11 | 40.53 | 33.65 | 6.88 | 12 |
| gen_stargan_v2 | Generative | 39.55 | 6.68 | 1.08 | 40.47 | 33.68 | 6.79 | 12 |
| gen_flux_kontext | Generative | 39.54 | 6.78 | 1.07 | 40.34 | 33.69 | 6.65 | 12 |
| gen_cyclediffusion | Generative | 39.52 | 6.88 | 1.05 | 40.35 | 33.57 | 6.79 | 12 |
| gen_CNetSeg | Generative | 39.47 | 6.88 | 1.0 | 40.36 | 33.71 | 6.65 | 12 |
| gen_IP2P | Generative | 39.47 | 6.91 | 1.0 | 40.22 | 33.3 | 6.92 | 12 |
| gen_augmenters | Generative | 39.46 | 6.88 | 0.99 | 40.42 | 33.75 | 6.66 | 12 |
| gen_Weather_Effect_Generator | Generative | 39.43 | 6.9 | 0.96 | 40.14 | 33.33 | 6.81 | 12 |
| gen_SUSTechGAN | Generative | 39.43 | 6.84 | 0.96 | 40.04 | 33.6 | 6.44 | 12 |
| gen_automold | Generative | 39.43 | 7.06 | 0.96 | 40.31 | 33.45 | 6.86 | 12 |
| gen_step1x_new | Generative | 39.41 | 7.01 | 0.94 | 40.21 | 33.5 | 6.71 | 12 |
| std_autoaugment | Standard Aug | 39.41 | 6.9 | 0.94 | 40.05 | 34.21 | 5.84 | 12 |
| gen_VisualCloze | Generative | 39.4 | 6.69 | 0.93 | 40.22 | 33.55 | 6.68 | 12 |
| gen_albumentations_weather | Generative | 39.31 | 7.19 | 0.84 | 40.15 | 33.47 | 6.68 | 12 |
| gen_Qwen_Image_Edit | Generative | 39.27 | 7.16 | 0.8 | 40.17 | 33.47 | 6.69 | 12 |
| gen_LANIT | Generative | 39.21 | 6.95 | 0.74 | 40.0 | 32.96 | 7.03 | 12 |
| gen_UniControl | Generative | 39.18 | 7.0 | 0.71 | 40.01 | 33.67 | 6.34 | 12 |
| gen_CUT | Generative | 39.17 | 6.98 | 0.7 | 40.0 | 33.4 | 6.59 | 12 |
| gen_step1x_v1p2 | Generative | 39.16 | 7.19 | 0.69 | 39.97 | 33.13 | 6.84 | 12 |
| gen_TSIT | Generative | 39.11 | 7.18 | 0.63 | 39.87 | 33.03 | 6.84 | 12 |
| std_mixup | Standard Aug | 38.99 | 6.91 | 0.52 | 39.8 | 32.91 | 6.89 | 12 |
| photometric_distort | Augmentation | 38.96 | 7.36 | 0.48 | 39.83 | 33.13 | 6.7 | 12 |
| std_randaugment | Standard Aug | 38.92 | 7.14 | 0.45 | 39.66 | 33.6 | 6.06 | 12 |
| std_cutmix | Standard Aug | 38.64 | 7.07 | 0.17 | 39.6 | 33.04 | 6.57 | 12 |
| baseline | Baseline | 38.47 | 6.78 | - | 39.18 | 32.36 | 6.81 | 12 |
---

## Per-Dataset Breakdown

mIoU performance on each dataset.

| Strategy | Type | bdd10k | bdd10k_gain | idd-aw | idd-aw_gain | mapillaryvistas | mapillaryvistas_gain | outside15k | outside15k_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_Attribute_Hallucination | Generative | 46.49 | +2.40 | 40.00 | +1.42 | 34.82 | +0.50 | 38.02 | +1.12 |
| gen_cycleGAN | Generative | 46.03 | +1.94 | 40.03 | +1.45 | 34.49 | +0.18 | 37.86 | +0.96 |
| gen_Img2Img | Generative | 46.32 | +2.23 | 39.90 | +1.32 | 34.78 | +0.46 | 37.31 | +0.41 |
| gen_stargan_v2 | Generative | 46.15 | +2.06 | 39.54 | +0.97 | 34.78 | +0.46 | 37.74 | +0.84 |
| gen_flux_kontext | Generative | 46.14 | +2.05 | 39.29 | +0.72 | 34.53 | +0.21 | 38.19 | +1.29 |
| gen_cyclediffusion | Generative | 46.28 | +2.19 | 39.84 | +1.26 | 34.60 | +0.29 | 37.36 | +0.46 |
| gen_IP2P | Generative | 46.33 | +2.25 | 39.64 | +1.06 | 35.14 | +0.82 | 36.75 | -0.15 |
| gen_CNetSeg | Generative | 45.98 | +1.89 | 39.66 | +1.09 | 35.11 | +0.79 | 37.11 | +0.21 |
| gen_augmenters | Generative | 45.88 | +1.79 | 39.95 | +1.38 | 34.88 | +0.56 | 37.12 | +0.22 |
| gen_SUSTechGAN | Generative | 46.25 | +2.16 | 39.55 | +0.97 | 34.65 | +0.33 | 37.28 | +0.38 |
| gen_Weather_Effect_Generator | Generative | 46.19 | +2.11 | 39.52 | +0.95 | 34.57 | +0.25 | 37.44 | +0.54 |
| gen_automold | Generative | 46.37 | +2.28 | 39.64 | +1.07 | 34.52 | +0.20 | 37.17 | +0.27 |
| std_autoaugment | Standard Aug | 46.09 | +2.00 | 39.80 | +1.22 | 34.16 | -0.16 | 37.60 | +0.70 |
| gen_step1x_new | Generative | 46.32 | +2.23 | 39.30 | +0.73 | 34.56 | +0.24 | 37.45 | +0.55 |
| gen_VisualCloze | Generative | 45.88 | +1.80 | 39.55 | +0.97 | 34.52 | +0.20 | 37.65 | +0.75 |
| gen_albumentations_weather | Generative | 46.29 | +2.20 | 39.80 | +1.23 | 34.27 | -0.05 | 36.86 | -0.04 |
| gen_Qwen_Image_Edit | Generative | 45.90 | +1.81 | 39.81 | +1.23 | 34.59 | +0.28 | 36.78 | -0.12 |
| gen_LANIT | Generative | 45.54 | +1.45 | 39.39 | +0.81 | 34.82 | +0.50 | 37.10 | +0.20 |
| gen_UniControl | Generative | 46.03 | +1.94 | 39.29 | +0.71 | 34.33 | +0.01 | 37.09 | +0.20 |
| gen_CUT | Generative | 45.91 | +1.82 | 39.52 | +0.94 | 34.11 | -0.21 | 37.15 | +0.25 |
| gen_step1x_v1p2 | Generative | 46.07 | +1.98 | 39.69 | +1.12 | 34.51 | +0.19 | 36.36 | -0.53 |
| gen_TSIT | Generative | 45.81 | +1.72 | 39.38 | +0.81 | 34.49 | +0.17 | 36.74 | -0.16 |
| std_mixup | Standard Aug | 45.56 | +1.47 | 38.63 | +0.06 | 34.12 | -0.20 | 37.67 | +0.77 |
| photometric_distort | Augmentation | 45.75 | +1.66 | 39.92 | +1.35 | 34.59 | +0.27 | 35.55 | -1.35 |
| std_randaugment | Standard Aug | 45.25 | +1.17 | 39.55 | +0.97 | 33.51 | -0.81 | 37.38 | +0.48 |
| std_cutmix | Standard Aug | 44.50 | +0.41 | 39.33 | +0.75 | 33.84 | -0.48 | 36.91 | +0.01 |
| baseline | Baseline | 44.09 | - | 38.58 | - | 34.32 | - | 36.90 | - |
---

## Per-Domain Breakdown

mIoU performance on each weather domain. Normal = clear_day, cloudy. Adverse = foggy, night, rainy, snowy.

| Strategy | Type | clear_day | cloudy | dawn_dusk | foggy | night | rainy | snowy | Normal Avg | Adverse Avg | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Baseline | 41.05 | 37.30 | 31.78 | 37.47 | 26.20 | 33.52 | 32.27 | 39.18 | 32.36 | 6.81 |
| gen_Attribute_Hallucination | Generative | 42.20 | 39.16 | 33.99 | 39.36 | 27.41 | 34.76 | 33.96 | 40.68 | 33.87 | 6.80 |
| gen_CNetSeg | Generative | 42.04 | 38.68 | 32.80 | 39.56 | 27.17 | 34.50 | 33.61 | 40.36 | 33.71 | 6.65 |
| gen_CUT | Generative | 41.61 | 38.38 | 32.12 | 39.07 | 27.16 | 33.85 | 33.53 | 40.00 | 33.40 | 6.59 |
| gen_IP2P | Generative | 41.95 | 38.49 | 33.02 | 38.16 | 26.73 | 34.27 | 34.05 | 40.22 | 33.30 | 6.92 |
| gen_Img2Img | Generative | 42.11 | 38.95 | 32.36 | 38.82 | 27.38 | 34.54 | 33.85 | 40.53 | 33.65 | 6.88 |
| gen_LANIT | Generative | 41.75 | 38.25 | 32.68 | 37.74 | 26.74 | 33.96 | 33.41 | 40.00 | 32.96 | 7.03 |
| gen_Qwen_Image_Edit | Generative | 41.71 | 38.63 | 32.74 | 38.69 | 27.27 | 34.12 | 33.81 | 40.17 | 33.47 | 6.69 |
| gen_SUSTechGAN | Generative | 41.80 | 38.28 | 32.90 | 38.62 | 27.50 | 34.32 | 33.97 | 40.04 | 33.60 | 6.44 |
| gen_TSIT | Generative | 41.67 | 38.08 | 32.44 | 38.29 | 26.73 | 33.85 | 33.24 | 39.87 | 33.03 | 6.84 |
| gen_UniControl | Generative | 41.74 | 38.27 | 32.45 | 39.25 | 27.26 | 34.35 | 33.80 | 40.01 | 33.67 | 6.34 |
| gen_VisualCloze | Generative | 41.96 | 38.48 | 33.05 | 38.71 | 27.34 | 34.46 | 33.69 | 40.22 | 33.55 | 6.68 |
| gen_Weather_Effect_Generator | Generative | 41.98 | 38.31 | 32.79 | 38.34 | 26.95 | 34.55 | 33.49 | 40.14 | 33.33 | 6.81 |
| gen_albumentations_weather | Generative | 41.74 | 38.56 | 33.00 | 38.97 | 27.26 | 34.30 | 33.36 | 40.15 | 33.47 | 6.68 |
| gen_augmenters | Generative | 41.98 | 38.86 | 32.98 | 39.84 | 27.40 | 34.24 | 33.54 | 40.42 | 33.75 | 6.66 |
| gen_automold | Generative | 41.77 | 38.85 | 33.01 | 38.58 | 26.91 | 34.59 | 33.73 | 40.31 | 33.45 | 6.86 |
| gen_cycleGAN | Generative | 42.07 | 38.57 | 33.34 | 39.12 | 27.41 | 34.40 | 33.73 | 40.32 | 33.66 | 6.66 |
| gen_cyclediffusion | Generative | 42.07 | 38.64 | 32.89 | 38.99 | 27.07 | 34.82 | 33.39 | 40.35 | 33.57 | 6.79 |
| gen_flux_kontext | Generative | 42.05 | 38.63 | 32.81 | 39.32 | 27.17 | 34.49 | 33.79 | 40.34 | 33.69 | 6.65 |
| gen_stargan_v2 | Generative | 42.11 | 38.84 | 33.33 | 39.09 | 27.23 | 34.74 | 33.65 | 40.47 | 33.68 | 6.79 |
| gen_step1x_new | Generative | 41.80 | 38.61 | 32.49 | 38.74 | 26.95 | 34.51 | 33.79 | 40.21 | 33.50 | 6.71 |
| gen_step1x_v1p2 | Generative | 41.59 | 38.36 | 32.62 | 38.31 | 27.15 | 33.86 | 33.20 | 39.97 | 33.13 | 6.84 |
| photometric_distort | Augmentation | 41.34 | 38.32 | 32.52 | 38.50 | 27.17 | 33.90 | 32.94 | 39.83 | 33.13 | 6.70 |
| std_autoaugment | Standard Aug | 41.78 | 38.33 | 33.70 | 40.18 | 28.59 | 34.60 | 33.46 | 40.05 | 34.21 | 5.84 |
| std_cutmix | Standard Aug | 41.23 | 37.98 | 32.07 | 39.38 | 26.32 | 33.43 | 33.02 | 39.60 | 33.04 | 6.57 |
| std_mixup | Standard Aug | 41.64 | 37.96 | 32.36 | 38.28 | 26.24 | 34.03 | 33.11 | 39.80 | 32.91 | 6.89 |
| std_randaugment | Standard Aug | 41.24 | 38.08 | 32.86 | 38.89 | 28.16 | 34.03 | 33.32 | 39.66 | 33.60 | 6.06 |
