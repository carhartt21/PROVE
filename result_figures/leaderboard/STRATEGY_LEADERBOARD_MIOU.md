# Stage 1 Strategy Leaderboard (by mIoU)

**Stage 1**: All models trained with `clear_day` domain filter only.

**Metric**: mIoU (Mean Intersection over Union)

**Total Results**: 79 test results from 13 strategies

---

## Overall Strategy Ranking

Sorted by mIoU. Gain = improvement over baseline. Domain Gap = Normal mIoU - Adverse mIoU (lower is better).

| Strategy | Type | mIoU | Std | Gain | Normal mIoU | Adverse mIoU | Domain Gap | Num Tests |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_step1x_v1p2 | Generative | 45.73 | 2.22 | 3.64 | 49.52 | 44.08 | 5.44 | 3 |
| std_photometric_distort | Augmentation | 43.08 | 3.26 | 0.98 | 44.73 | 36.66 | 8.07 | 6 |
| gen_cycleGAN | Generative | 42.99 | 3.39 | 0.9 | 44.62 | 35.64 | 8.98 | 6 |
| std_cutmix | Standard Aug | 42.88 | 3.29 | 0.79 | 44.67 | 35.58 | 9.09 | 6 |
| gen_flux_kontext | Generative | 42.87 | 3.36 | 0.77 | 44.6 | 36.12 | 8.48 | 6 |
| gen_automold | Generative | 42.85 | 3.29 | 0.75 | 44.64 | 35.34 | 9.3 | 6 |
| std_mixup | Standard Aug | 42.83 | 3.09 | 0.74 | 44.43 | 35.91 | 8.52 | 6 |
| gen_step1x_new | Generative | 42.78 | 3.07 | 0.69 | 44.63 | 35.68 | 8.95 | 6 |
| std_autoaugment | Standard Aug | 42.76 | 3.5 | 0.66 | 44.44 | 35.26 | 9.18 | 6 |
| gen_albumentations_weather | Generative | 42.7 | 3.25 | 0.6 | 44.63 | 35.16 | 9.47 | 6 |
| gen_LANIT | Generative | 42.67 | 3.14 | 0.58 | 44.34 | 35.83 | 8.51 | 6 |
| baseline | Baseline | 42.09 | 3.71 | - | 43.48 | 35.67 | 7.82 | 7 |
| std_minimal | Standard Aug | 40.92 | 4.78 | -1.17 | 42.27 | 35.22 | 7.05 | 9 |
---

## Per-Dataset Breakdown

mIoU performance on each dataset.

| Strategy | Type | bdd10k | bdd10k_gain | idd-aw | idd-aw_gain | mapillaryvistas | mapillaryvistas_gain | outside15k | outside15k_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_step1x_v1p2 | Generative | 45.73 | +0.12 | - | - | - | - | - | - |
| std_photometric_distort | Augmentation | 45.62 | +0.01 | 40.53 | +0.48 | - | - | - | - |
| gen_cycleGAN | Generative | 45.77 | +0.16 | 40.21 | +0.15 | - | - | - | - |
| std_cutmix | Standard Aug | 45.44 | -0.17 | 40.32 | +0.26 | - | - | - | - |
| gen_flux_kontext | Generative | 45.51 | -0.10 | 40.22 | +0.16 | - | - | - | - |
| gen_automold | Generative | 45.44 | -0.16 | 40.25 | +0.19 | - | - | - | - |
| std_mixup | Standard Aug | 45.35 | -0.26 | 40.31 | +0.25 | - | - | - | - |
| gen_step1x_new | Generative | 45.22 | -0.38 | 40.34 | +0.28 | - | - | - | - |
| std_autoaugment | Standard Aug | 45.49 | -0.11 | 40.02 | -0.04 | - | - | - | - |
| gen_albumentations_weather | Generative | 45.26 | -0.34 | 40.13 | +0.07 | - | - | - | - |
| gen_LANIT | Generative | 45.19 | -0.41 | 40.14 | +0.08 | - | - | - | - |
| baseline | Baseline | 45.61 | - | 40.06 | - | 37.65 | - | - | - |
| std_minimal | Standard Aug | 46.15 | +0.54 | 40.89 | +0.83 | 35.71 | -1.93 | - | - |
---

## Per-Domain Breakdown

mIoU performance on each weather domain. Normal = clear_day, cloudy. Adverse = foggy, night, rainy, snowy.

| Strategy | Type | clear_day | cloudy | dawn_dusk | foggy | night | rainy | snowy | Normal Avg | Adverse Avg | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Baseline | 44.93 | 42.04 | 36.26 | 41.30 | 27.62 | 35.84 | 37.91 | 43.48 | 35.67 | 7.82 |
| gen_LANIT | Generative | 45.93 | 42.74 | 36.02 | 42.76 | 26.59 | 35.49 | 38.47 | 44.34 | 35.83 | 8.51 |
| gen_albumentations_weather | Generative | 46.16 | 43.09 | 36.05 | 41.51 | 26.15 | 34.63 | 38.33 | 44.63 | 35.16 | 9.47 |
| gen_automold | Generative | 46.11 | 43.17 | 36.44 | 41.33 | 26.30 | 35.49 | 38.24 | 44.64 | 35.34 | 9.30 |
| gen_cycleGAN | Generative | 46.24 | 43.00 | 36.16 | 41.62 | 26.38 | 35.38 | 39.16 | 44.62 | 35.64 | 8.98 |
| gen_flux_kontext | Generative | 45.90 | 43.31 | 36.36 | 43.02 | 26.66 | 35.78 | 39.01 | 44.60 | 36.12 | 8.48 |
| gen_step1x_new | Generative | 45.89 | 43.37 | 36.23 | 42.19 | 26.13 | 35.79 | 38.59 | 44.63 | 35.68 | 8.95 |
| gen_step1x_v1p2 | Generative | 47.52 | 51.52 | 39.09 | 54.86 | 24.69 | 44.26 | 52.53 | 49.52 | 44.08 | 5.44 |
| std_autoaugment | Standard Aug | 46.02 | 42.85 | 36.17 | 40.38 | 26.65 | 35.20 | 38.80 | 44.44 | 35.26 | 9.18 |
| std_cutmix | Standard Aug | 46.13 | 43.21 | 35.63 | 41.06 | 26.77 | 35.65 | 38.84 | 44.67 | 35.58 | 9.09 |
| std_minimal | Standard Aug | 43.45 | 41.08 | 37.29 | 40.19 | 28.15 | 35.53 | 37.00 | 42.27 | 35.22 | 7.05 |
| std_mixup | Standard Aug | 45.92 | 42.95 | 36.55 | 42.62 | 26.68 | 35.72 | 38.62 | 44.43 | 35.91 | 8.52 |
| std_photometric_distort | Augmentation | 45.98 | 43.48 | 37.46 | 43.48 | 27.52 | 36.70 | 38.96 | 44.73 | 36.66 | 8.07 |
