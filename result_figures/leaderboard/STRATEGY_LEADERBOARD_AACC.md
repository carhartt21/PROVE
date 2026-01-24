# Stage 1 Strategy Leaderboard (by aAcc)

**Stage 1**: All models trained with `clear_day` domain filter only.

**Metric**: aAcc (Pixel Accuracy (Overall Accuracy))

**Total Results**: 324 test results from 27 strategies

---

## Overall Strategy Ranking

Sorted by aAcc. Gain = improvement over baseline. Domain Gap = Normal aAcc - Adverse aAcc (lower is better).

| Strategy | Type | aAcc | Std | Gain | Normal aAcc | Adverse aAcc | Domain Gap | Num Tests |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_Attribute_Hallucination | Generative | 85.25 | 6.31 | 0.61 | 87.11 | 80.32 | 6.79 | 12 |
| gen_cycleGAN | Generative | 85.15 | 6.35 | 0.51 | 87.0 | 80.3 | 6.7 | 12 |
| std_autoaugment | Standard Aug | 85.14 | 6.27 | 0.51 | 86.82 | 80.62 | 6.19 | 12 |
| gen_stargan_v2 | Generative | 85.13 | 6.35 | 0.5 | 86.95 | 80.31 | 6.64 | 12 |
| gen_Img2Img | Generative | 85.11 | 6.53 | 0.48 | 87.01 | 80.19 | 6.83 | 12 |
| std_randaugment | Standard Aug | 85.08 | 6.24 | 0.45 | 86.86 | 80.58 | 6.28 | 12 |
| gen_flux_kontext | Generative | 85.08 | 6.37 | 0.45 | 86.97 | 79.93 | 7.04 | 12 |
| gen_CNetSeg | Generative | 85.06 | 6.53 | 0.43 | 86.92 | 80.09 | 6.83 | 12 |
| gen_augmenters | Generative | 85.01 | 6.76 | 0.37 | 86.91 | 80.02 | 6.9 | 12 |
| gen_SUSTechGAN | Generative | 84.98 | 6.5 | 0.35 | 86.81 | 80.18 | 6.63 | 12 |
| gen_VisualCloze | Generative | 84.97 | 6.51 | 0.33 | 86.76 | 80.03 | 6.74 | 12 |
| gen_CUT | Generative | 84.96 | 6.58 | 0.33 | 86.77 | 80.17 | 6.6 | 12 |
| gen_Qwen_Image_Edit | Generative | 84.93 | 6.83 | 0.3 | 86.77 | 80.01 | 6.75 | 12 |
| gen_automold | Generative | 84.9 | 6.75 | 0.27 | 86.8 | 79.85 | 6.96 | 12 |
| gen_step1x_new | Generative | 84.89 | 6.66 | 0.26 | 86.79 | 79.94 | 6.85 | 12 |
| gen_albumentations_weather | Generative | 84.88 | 6.77 | 0.24 | 86.76 | 79.86 | 6.9 | 12 |
| gen_Weather_Effect_Generator | Generative | 84.87 | 6.67 | 0.24 | 86.76 | 79.67 | 7.09 | 12 |
| gen_UniControl | Generative | 84.87 | 6.67 | 0.24 | 86.72 | 79.85 | 6.87 | 12 |
| photometric_distort | Augmentation | 84.86 | 6.86 | 0.23 | 86.85 | 79.97 | 6.87 | 12 |
| gen_cyclediffusion | Generative | 84.83 | 6.84 | 0.2 | 86.62 | 79.76 | 6.86 | 12 |
| gen_IP2P | Generative | 84.81 | 6.84 | 0.18 | 86.64 | 79.71 | 6.93 | 12 |
| std_mixup | Standard Aug | 84.77 | 6.28 | 0.14 | 86.69 | 79.4 | 7.28 | 12 |
| gen_LANIT | Generative | 84.76 | 6.75 | 0.13 | 86.6 | 79.58 | 7.03 | 12 |
| gen_TSIT | Generative | 84.73 | 6.82 | 0.1 | 86.7 | 79.41 | 7.29 | 12 |
| gen_step1x_v1p2 | Generative | 84.69 | 7.16 | 0.06 | 86.59 | 79.67 | 6.92 | 12 |
| std_cutmix | Standard Aug | 84.64 | 6.74 | 0.01 | 86.57 | 79.42 | 7.15 | 12 |
| baseline | Baseline | 84.63 | 6.86 | - | 86.36 | 79.61 | 6.75 | 12 |
---

## Per-Dataset Breakdown

aAcc performance on each dataset.

| Strategy | Type | bdd10k | bdd10k_gain | idd-aw | idd-aw_gain | mapillaryvistas | mapillaryvistas_gain | outside15k | outside15k_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_Attribute_Hallucination | Generative | 90.74 | +0.37 | 84.47 | +1.04 | 89.25 | +0.08 | 76.52 | +0.96 |
| gen_cycleGAN | Generative | 90.70 | +0.34 | 84.39 | +0.96 | 89.11 | -0.06 | 76.38 | +0.82 |
| std_autoaugment | Standard Aug | 90.75 | +0.39 | 83.99 | +0.56 | 89.16 | -0.00 | 76.64 | +1.08 |
| gen_stargan_v2 | Generative | 90.67 | +0.31 | 84.14 | +0.70 | 89.26 | +0.10 | 76.45 | +0.89 |
| gen_Img2Img | Generative | 90.72 | +0.35 | 84.21 | +0.77 | 89.35 | +0.18 | 76.15 | +0.59 |
| std_randaugment | Standard Aug | 90.53 | +0.16 | 84.04 | +0.60 | 89.12 | -0.05 | 76.63 | +1.07 |
| gen_flux_kontext | Generative | 90.71 | +0.35 | 83.64 | +0.20 | 89.29 | +0.13 | 76.68 | +1.12 |
| gen_CNetSeg | Generative | 90.63 | +0.26 | 83.99 | +0.56 | 89.39 | +0.22 | 76.23 | +0.67 |
| gen_augmenters | Generative | 90.65 | +0.28 | 84.33 | +0.89 | 89.34 | +0.18 | 75.70 | +0.14 |
| gen_SUSTechGAN | Generative | 90.71 | +0.34 | 83.81 | +0.38 | 89.26 | +0.10 | 76.15 | +0.59 |
| gen_VisualCloze | Generative | 90.73 | +0.36 | 83.75 | +0.31 | 89.14 | -0.03 | 76.25 | +0.69 |
| gen_CUT | Generative | 90.71 | +0.35 | 83.94 | +0.51 | 89.17 | +0.01 | 76.01 | +0.44 |
| gen_Qwen_Image_Edit | Generative | 90.71 | +0.34 | 84.10 | +0.67 | 89.31 | +0.15 | 75.60 | +0.04 |
| gen_automold | Generative | 90.72 | +0.35 | 83.86 | +0.43 | 89.27 | +0.11 | 75.77 | +0.21 |
| gen_step1x_new | Generative | 90.78 | +0.41 | 83.51 | +0.08 | 89.20 | +0.03 | 76.07 | +0.51 |
| gen_albumentations_weather | Generative | 90.71 | +0.35 | 83.90 | +0.47 | 89.24 | +0.08 | 75.65 | +0.09 |
| gen_Weather_Effect_Generator | Generative | 90.70 | +0.33 | 83.60 | +0.17 | 89.18 | +0.02 | 76.01 | +0.45 |
| gen_UniControl | Generative | 90.83 | +0.46 | 83.43 | -0.00 | 89.19 | +0.03 | 76.02 | +0.46 |
| photometric_distort | Augmentation | 90.57 | +0.20 | 84.43 | +0.99 | 89.17 | +0.00 | 75.27 | -0.29 |
| gen_cyclediffusion | Generative | 90.79 | +0.42 | 83.41 | -0.02 | 89.31 | +0.15 | 75.80 | +0.24 |
| gen_IP2P | Generative | 90.71 | +0.35 | 83.58 | +0.15 | 89.29 | +0.13 | 75.67 | +0.11 |
| std_mixup | Standard Aug | 90.28 | -0.09 | 83.13 | -0.31 | 89.08 | -0.08 | 76.59 | +1.03 |
| gen_LANIT | Generative | 90.51 | +0.15 | 83.34 | -0.10 | 89.31 | +0.15 | 75.87 | +0.30 |
| gen_TSIT | Generative | 90.58 | +0.21 | 83.60 | +0.16 | 89.17 | +0.00 | 75.56 | +0.00 |
| gen_step1x_v1p2 | Generative | 90.69 | +0.32 | 83.94 | +0.50 | 89.21 | +0.04 | 74.91 | -0.65 |
| std_cutmix | Standard Aug | 90.59 | +0.22 | 83.45 | +0.02 | 88.80 | -0.36 | 75.71 | +0.15 |
| baseline | Baseline | 90.37 | - | 83.43 | -0.00 | 89.16 | -0.00 | 75.56 | - |
---

## Per-Domain Breakdown

aAcc performance on each weather domain. Normal = clear_day, cloudy. Adverse = foggy, night, rainy, snowy.

| Strategy | Type | clear_day | cloudy | dawn_dusk | foggy | night | rainy | snowy | Normal Avg | Adverse Avg | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Baseline | 85.74 | 86.97 | 80.44 | 85.32 | 75.74 | 79.65 | 77.72 | 86.36 | 79.61 | 6.75 |
| gen_Attribute_Hallucination | Generative | 86.14 | 88.09 | 82.21 | 85.97 | 76.90 | 80.23 | 78.19 | 87.11 | 80.32 | 6.79 |
| gen_CNetSeg | Generative | 86.12 | 87.72 | 81.55 | 85.95 | 76.63 | 80.02 | 77.77 | 86.92 | 80.09 | 6.83 |
| gen_CUT | Generative | 85.95 | 87.59 | 80.46 | 86.22 | 76.44 | 79.82 | 78.21 | 86.77 | 80.17 | 6.60 |
| gen_IP2P | Generative | 85.90 | 87.38 | 80.88 | 85.15 | 76.10 | 79.73 | 77.87 | 86.64 | 79.71 | 6.93 |
| gen_Img2Img | Generative | 86.12 | 87.90 | 80.88 | 85.82 | 77.06 | 80.07 | 77.79 | 87.01 | 80.19 | 6.83 |
| gen_LANIT | Generative | 85.93 | 87.27 | 80.50 | 85.01 | 75.97 | 79.89 | 77.43 | 86.60 | 79.58 | 7.03 |
| gen_Qwen_Image_Edit | Generative | 85.93 | 87.60 | 81.01 | 86.01 | 76.24 | 79.72 | 78.09 | 86.77 | 80.01 | 6.75 |
| gen_SUSTechGAN | Generative | 85.91 | 87.71 | 81.51 | 85.98 | 76.40 | 79.83 | 78.52 | 86.81 | 80.18 | 6.63 |
| gen_TSIT | Generative | 85.87 | 87.53 | 80.84 | 85.54 | 75.17 | 79.56 | 77.37 | 86.70 | 79.41 | 7.29 |
| gen_UniControl | Generative | 85.98 | 87.46 | 80.69 | 85.81 | 76.36 | 79.91 | 77.32 | 86.72 | 79.85 | 6.87 |
| gen_VisualCloze | Generative | 86.01 | 87.51 | 81.24 | 85.73 | 76.50 | 80.03 | 77.85 | 86.76 | 80.03 | 6.74 |
| gen_Weather_Effect_Generator | Generative | 85.97 | 87.55 | 81.45 | 85.63 | 75.75 | 79.98 | 77.33 | 86.76 | 79.67 | 7.09 |
| gen_albumentations_weather | Generative | 85.92 | 87.60 | 81.03 | 85.73 | 76.55 | 79.67 | 77.49 | 86.76 | 79.86 | 6.90 |
| gen_augmenters | Generative | 86.02 | 87.81 | 81.04 | 86.17 | 76.39 | 79.92 | 77.59 | 86.91 | 80.02 | 6.90 |
| gen_automold | Generative | 85.93 | 87.67 | 81.14 | 85.72 | 76.07 | 80.14 | 77.45 | 86.80 | 79.85 | 6.96 |
| gen_cycleGAN | Generative | 86.12 | 87.88 | 81.60 | 86.09 | 77.20 | 80.05 | 77.87 | 87.00 | 80.30 | 6.70 |
| gen_cyclediffusion | Generative | 86.07 | 87.17 | 80.20 | 85.08 | 76.09 | 80.06 | 77.82 | 86.62 | 79.76 | 6.86 |
| gen_flux_kontext | Generative | 86.11 | 87.83 | 81.71 | 85.74 | 76.23 | 80.09 | 77.67 | 86.97 | 79.93 | 7.04 |
| gen_stargan_v2 | Generative | 86.11 | 87.78 | 81.29 | 86.25 | 76.51 | 80.35 | 78.12 | 86.95 | 80.31 | 6.64 |
| gen_step1x_new | Generative | 85.92 | 87.67 | 80.51 | 85.82 | 76.05 | 79.82 | 78.08 | 86.79 | 79.94 | 6.85 |
| gen_step1x_v1p2 | Generative | 85.77 | 87.40 | 80.53 | 85.61 | 76.41 | 79.29 | 77.35 | 86.59 | 79.67 | 6.92 |
| photometric_distort | Augmentation | 85.80 | 87.90 | 80.59 | 85.66 | 76.90 | 79.60 | 77.74 | 86.85 | 79.97 | 6.87 |
| std_autoaugment | Standard Aug | 86.07 | 87.56 | 83.31 | 85.69 | 79.13 | 80.10 | 77.56 | 86.82 | 80.62 | 6.19 |
| std_cutmix | Standard Aug | 85.76 | 87.38 | 80.48 | 85.49 | 75.44 | 79.43 | 77.30 | 86.57 | 79.42 | 7.15 |
| std_mixup | Standard Aug | 86.00 | 87.37 | 80.71 | 85.05 | 75.06 | 80.11 | 77.38 | 86.69 | 79.40 | 7.28 |
| std_randaugment | Standard Aug | 85.95 | 87.77 | 82.78 | 85.67 | 78.96 | 79.95 | 77.75 | 86.86 | 80.58 | 6.28 |
