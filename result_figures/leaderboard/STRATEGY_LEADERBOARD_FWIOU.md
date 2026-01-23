# Stage 1 Strategy Leaderboard (by fwIoU)

**Stage 1**: All models trained with `clear_day` domain filter only.

**Metric**: fwIoU (Frequency-Weighted IoU)

**Total Results**: 324 test results from 27 strategies

---

## Overall Strategy Ranking

Sorted by fwIoU. Gain = improvement over baseline. Domain Gap = Normal fwIoU - Adverse fwIoU (lower is better).

| Strategy | Type | fwIoU | Std | Gain | Normal fwIoU | Adverse fwIoU | Domain Gap | Num Tests |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_Attribute_Hallucination | Generative | 76.55 | 8.15 | 0.81 | 79.52 | 70.74 | 8.78 | 12 |
| std_autoaugment | Standard Aug | 76.39 | 8.14 | 0.65 | 79.07 | 70.93 | 8.14 | 12 |
| gen_cycleGAN | Generative | 76.39 | 8.25 | 0.65 | 79.36 | 70.52 | 8.83 | 12 |
| gen_Img2Img | Generative | 76.37 | 8.42 | 0.63 | 79.39 | 70.45 | 8.93 | 12 |
| gen_stargan_v2 | Generative | 76.35 | 8.28 | 0.61 | 79.27 | 70.62 | 8.65 | 12 |
| gen_flux_kontext | Generative | 76.35 | 8.27 | 0.61 | 79.37 | 70.28 | 9.09 | 12 |
| std_randaugment | Standard Aug | 76.29 | 8.07 | 0.55 | 79.05 | 71.02 | 8.03 | 12 |
| gen_augmenters | Generative | 76.29 | 8.61 | 0.55 | 79.33 | 70.4 | 8.93 | 12 |
| gen_VisualCloze | Generative | 76.2 | 8.42 | 0.46 | 79.06 | 70.31 | 8.75 | 12 |
| gen_CNetSeg | Generative | 76.18 | 8.61 | 0.44 | 79.16 | 70.28 | 8.88 | 12 |
| gen_CUT | Generative | 76.18 | 8.48 | 0.44 | 79.09 | 70.39 | 8.7 | 12 |
| gen_SUSTechGAN | Generative | 76.17 | 8.46 | 0.43 | 79.12 | 70.38 | 8.74 | 12 |
| gen_Qwen_Image_Edit | Generative | 76.16 | 8.8 | 0.42 | 79.11 | 70.29 | 8.83 | 12 |
| gen_automold | Generative | 76.1 | 8.66 | 0.36 | 79.1 | 70.1 | 9.0 | 12 |
| gen_UniControl | Generative | 76.09 | 8.66 | 0.35 | 79.04 | 70.09 | 8.94 | 12 |
| gen_albumentations_weather | Generative | 76.08 | 8.71 | 0.34 | 79.09 | 70.07 | 9.03 | 12 |
| photometric_distort | Augmentation | 76.05 | 8.84 | 0.31 | 79.13 | 70.31 | 8.82 | 12 |
| gen_Weather_Effect_Generator | Generative | 76.01 | 8.67 | 0.27 | 79.03 | 69.85 | 9.17 | 12 |
| gen_IP2P | Generative | 76.01 | 8.79 | 0.27 | 78.9 | 69.99 | 8.9 | 12 |
| gen_step1x_new | Generative | 76.0 | 8.65 | 0.26 | 79.0 | 70.12 | 8.88 | 12 |
| gen_cyclediffusion | Generative | 75.98 | 8.89 | 0.24 | 78.83 | 70.07 | 8.76 | 12 |
| gen_step1x_v1p2 | Generative | 75.92 | 8.99 | 0.18 | 78.91 | 70.01 | 8.9 | 12 |
| gen_LANIT | Generative | 75.89 | 8.73 | 0.15 | 78.8 | 69.71 | 9.1 | 12 |
| std_mixup | Standard Aug | 75.86 | 8.3 | 0.12 | 78.94 | 69.43 | 9.51 | 12 |
| gen_TSIT | Generative | 75.82 | 8.85 | 0.09 | 78.94 | 69.55 | 9.39 | 12 |
| baseline | Baseline | 75.74 | 8.76 | - | 78.53 | 69.78 | 8.76 | 12 |
| std_cutmix | Standard Aug | 75.66 | 8.63 | -0.08 | 78.74 | 69.35 | 9.39 | 12 |
---

## Per-Dataset Breakdown

fwIoU performance on each dataset.

| Strategy | Type | bdd10k | bdd10k_gain | idd-aw | idd-aw_gain | mapillaryvistas | mapillaryvistas_gain | outside15k | outside15k_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_Attribute_Hallucination | Generative | 83.72 | +0.61 | 74.68 | +1.43 | 82.33 | +0.07 | 65.48 | +1.14 |
| gen_cycleGAN | Generative | 83.68 | +0.56 | 74.55 | +1.31 | 82.15 | -0.11 | 65.20 | +0.87 |
| std_autoaugment | Standard Aug | 83.79 | +0.67 | 73.78 | +0.54 | 82.22 | -0.03 | 65.75 | +1.41 |
| gen_Img2Img | Generative | 83.70 | +0.58 | 74.25 | +1.00 | 82.44 | +0.18 | 65.09 | +0.76 |
| gen_flux_kontext | Generative | 83.70 | +0.58 | 73.68 | +0.43 | 82.39 | +0.13 | 65.65 | +1.31 |
| gen_stargan_v2 | Generative | 83.64 | +0.53 | 74.16 | +0.92 | 82.33 | +0.07 | 65.26 | +0.92 |
| std_randaugment | Standard Aug | 83.48 | +0.37 | 73.89 | +0.64 | 82.11 | -0.15 | 65.67 | +1.34 |
| gen_augmenters | Generative | 83.58 | +0.46 | 74.50 | +1.26 | 82.46 | +0.20 | 64.60 | +0.26 |
| gen_VisualCloze | Generative | 83.75 | +0.63 | 73.76 | +0.51 | 82.14 | -0.12 | 65.16 | +0.82 |
| gen_CUT | Generative | 83.69 | +0.57 | 74.00 | +0.75 | 82.17 | -0.09 | 64.85 | +0.51 |
| gen_CNetSeg | Generative | 83.51 | +0.40 | 73.89 | +0.64 | 82.49 | +0.23 | 64.82 | +0.48 |
| gen_SUSTechGAN | Generative | 83.68 | +0.56 | 73.77 | +0.52 | 82.31 | +0.05 | 64.92 | +0.58 |
| gen_Qwen_Image_Edit | Generative | 83.71 | +0.60 | 74.17 | +0.92 | 82.41 | +0.15 | 64.34 | +0.00 |
| gen_automold | Generative | 83.70 | +0.58 | 73.70 | +0.45 | 82.35 | +0.09 | 64.66 | +0.32 |
| gen_UniControl | Generative | 83.87 | +0.75 | 73.53 | +0.28 | 82.20 | -0.06 | 64.74 | +0.41 |
| gen_albumentations_weather | Generative | 83.67 | +0.55 | 73.91 | +0.67 | 82.29 | +0.03 | 64.43 | +0.09 |
| photometric_distort | Augmentation | 83.47 | +0.35 | 74.67 | +1.43 | 82.22 | -0.04 | 63.85 | -0.49 |
| gen_Weather_Effect_Generator | Generative | 83.67 | +0.55 | 73.34 | +0.09 | 82.23 | -0.03 | 64.82 | +0.48 |
| gen_IP2P | Generative | 83.67 | +0.56 | 73.36 | +0.12 | 82.42 | +0.16 | 64.58 | +0.24 |
| gen_step1x_new | Generative | 83.77 | +0.65 | 73.12 | -0.13 | 82.22 | -0.04 | 64.91 | +0.57 |
| gen_cyclediffusion | Generative | 83.82 | +0.71 | 72.94 | -0.30 | 82.42 | +0.16 | 64.72 | +0.39 |
| gen_step1x_v1p2 | Generative | 83.64 | +0.52 | 73.96 | +0.71 | 82.24 | -0.02 | 63.85 | -0.49 |
| gen_LANIT | Generative | 83.41 | +0.29 | 72.97 | -0.28 | 82.40 | +0.14 | 64.79 | +0.46 |
| std_mixup | Standard Aug | 83.10 | -0.01 | 72.93 | -0.32 | 82.11 | -0.15 | 65.28 | +0.94 |
| gen_TSIT | Generative | 83.49 | +0.38 | 73.39 | +0.14 | 82.20 | -0.06 | 64.22 | -0.12 |
| baseline | Baseline | 83.12 | - | 73.25 | - | 82.26 | - | 64.34 | - |
| std_cutmix | Standard Aug | 83.39 | +0.28 | 73.36 | +0.12 | 81.47 | -0.78 | 64.42 | +0.08 |
---

## Per-Domain Breakdown

fwIoU performance on each weather domain. Normal = clear_day, cloudy. Adverse = foggy, night, rainy, snowy.

| Strategy | Type | clear_day | cloudy | dawn_dusk | foggy | night | rainy | snowy | Normal Avg | Adverse Avg | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Baseline | 77.06 | 80.01 | 72.04 | 78.07 | 64.58 | 69.33 | 67.14 | 78.53 | 69.78 | 8.76 |
| gen_Attribute_Hallucination | Generative | 77.60 | 81.45 | 73.99 | 78.93 | 66.04 | 70.07 | 67.91 | 79.52 | 70.74 | 8.78 |
| gen_CNetSeg | Generative | 77.52 | 80.81 | 73.00 | 78.61 | 65.29 | 69.89 | 67.33 | 79.16 | 70.28 | 8.88 |
| gen_CUT | Generative | 77.38 | 80.80 | 71.96 | 78.97 | 65.09 | 69.68 | 67.84 | 79.09 | 70.39 | 8.70 |
| gen_IP2P | Generative | 77.31 | 80.48 | 72.54 | 77.87 | 64.90 | 69.66 | 67.54 | 78.90 | 69.99 | 8.90 |
| gen_Img2Img | Generative | 77.61 | 81.16 | 72.59 | 78.64 | 65.78 | 70.02 | 67.38 | 79.39 | 70.45 | 8.93 |
| gen_LANIT | Generative | 77.37 | 80.24 | 72.04 | 77.57 | 64.43 | 69.80 | 67.02 | 78.80 | 69.71 | 9.10 |
| gen_Qwen_Image_Edit | Generative | 77.36 | 80.87 | 72.62 | 78.75 | 65.02 | 69.62 | 67.75 | 79.11 | 70.29 | 8.83 |
| gen_SUSTechGAN | Generative | 77.29 | 80.95 | 73.00 | 78.75 | 64.87 | 69.69 | 68.21 | 79.12 | 70.38 | 8.74 |
| gen_TSIT | Generative | 77.25 | 80.62 | 72.28 | 78.36 | 63.58 | 69.39 | 66.86 | 78.94 | 69.55 | 9.39 |
| gen_UniControl | Generative | 77.39 | 80.68 | 72.42 | 78.64 | 65.01 | 69.77 | 66.95 | 79.04 | 70.09 | 8.94 |
| gen_VisualCloze | Generative | 77.46 | 80.66 | 72.78 | 78.51 | 65.38 | 69.85 | 67.51 | 79.06 | 70.31 | 8.75 |
| gen_Weather_Effect_Generator | Generative | 77.40 | 80.65 | 72.88 | 78.30 | 64.38 | 69.89 | 66.83 | 79.03 | 69.85 | 9.17 |
| gen_albumentations_weather | Generative | 77.33 | 80.86 | 72.46 | 78.66 | 65.06 | 69.55 | 66.99 | 79.09 | 70.07 | 9.03 |
| gen_augmenters | Generative | 77.49 | 81.16 | 72.73 | 79.08 | 65.40 | 69.91 | 67.20 | 79.33 | 70.40 | 8.93 |
| gen_automold | Generative | 77.37 | 80.82 | 72.58 | 78.41 | 64.73 | 70.15 | 67.09 | 79.10 | 70.10 | 9.00 |
| gen_cycleGAN | Generative | 77.60 | 81.12 | 73.10 | 78.92 | 65.86 | 69.91 | 67.41 | 79.36 | 70.52 | 8.83 |
| gen_cyclediffusion | Generative | 77.58 | 80.08 | 71.70 | 77.79 | 64.79 | 70.08 | 67.61 | 78.83 | 70.07 | 8.76 |
| gen_flux_kontext | Generative | 77.56 | 81.18 | 73.10 | 78.64 | 65.11 | 70.01 | 67.34 | 79.37 | 70.28 | 9.09 |
| gen_stargan_v2 | Generative | 77.52 | 81.02 | 72.86 | 79.06 | 65.42 | 70.27 | 67.72 | 79.27 | 70.62 | 8.65 |
| gen_step1x_new | Generative | 77.26 | 80.73 | 72.23 | 78.46 | 64.70 | 69.67 | 67.66 | 79.00 | 70.12 | 8.88 |
| gen_step1x_v1p2 | Generative | 77.18 | 80.64 | 72.38 | 78.60 | 65.22 | 69.31 | 66.93 | 78.91 | 70.01 | 8.90 |
| photometric_distort | Augmentation | 77.16 | 81.11 | 72.25 | 78.70 | 65.79 | 69.54 | 67.22 | 79.13 | 70.31 | 8.82 |
| std_autoaugment | Standard Aug | 77.54 | 80.61 | 74.91 | 78.49 | 67.90 | 70.15 | 67.20 | 79.07 | 70.93 | 8.14 |
| std_cutmix | Standard Aug | 77.04 | 80.44 | 71.90 | 77.90 | 63.93 | 69.03 | 66.53 | 78.74 | 69.35 | 9.39 |
| std_mixup | Standard Aug | 77.39 | 80.49 | 72.15 | 77.44 | 63.74 | 69.76 | 66.79 | 78.94 | 69.43 | 9.51 |
| std_randaugment | Standard Aug | 77.35 | 80.76 | 74.24 | 78.46 | 68.19 | 69.88 | 67.56 | 79.05 | 71.02 | 8.03 |
