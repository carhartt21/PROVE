# Stage 2 Strategy Leaderboard (by mIoU)

**Stage 2**: All Domains Training — domain-inclusive evaluation

**Metric**: mIoU (Mean Intersection over Union)

**Last Updated**: 2026-02-10 09:52
**Baseline mIoU**: 40.80%
**Total Results**: 137 test results from 20 strategies

---

## Overall Strategy Ranking

Sorted by mIoU. Gain = improvement over baseline. Domain Gap = Normal mIoU - Adverse mIoU (lower is better).

| Strategy | Type | mIoU | Std | Gain | Normal mIoU | Adverse mIoU | Domain Gap | Num Tests |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| std_randaugment | Standard Aug | 42.01 | 5.46 | 1.21 | 42.33 | 40.1 | 2.22 | 4 |
| gen_IP2P | Generative | 41.98 | 5.51 | 1.18 | 42.49 | 40.15 | 2.34 | 4 |
| gen_VisualCloze | Generative | 41.9 | 5.32 | 1.11 | 42.37 | 39.58 | 2.79 | 4 |
| gen_albumentations_weather | Generative | 41.87 | 5.27 | 1.07 | 42.1 | 40.14 | 1.96 | 4 |
| gen_cyclediffusion | Generative | 41.85 | 5.24 | 1.05 | 42.46 | 39.99 | 2.47 | 4 |
| gen_flux_kontext | Generative | 41.84 | 5.35 | 1.04 | 42.3 | 39.55 | 2.75 | 4 |
| gen_Img2Img | Generative | 41.78 | 5.31 | 0.98 | 42.32 | 39.89 | 2.43 | 4 |
| gen_Attribute_Hallucination | Generative | 41.77 | 5.33 | 0.98 | 42.33 | 39.95 | 2.38 | 4 |
| gen_step1x_new | Generative | 41.77 | 5.13 | 0.97 | 42.4 | 40.11 | 2.29 | 4 |
| gen_UniControl | Generative | 41.77 | 5.16 | 0.97 | 42.19 | 39.94 | 2.24 | 4 |
| gen_automold | Generative | 41.77 | 5.27 | 0.97 | 42.38 | 39.67 | 2.71 | 4 |
| gen_LANIT | Generative | 41.76 | 5.33 | 0.96 | 42.23 | 39.89 | 2.33 | 4 |
| gen_CUT | Generative | 41.75 | 5.19 | 0.95 | 42.22 | 40.1 | 2.11 | 4 |
| gen_cycleGAN | Generative | 41.74 | 5.29 | 0.94 | 42.23 | 39.85 | 2.37 | 4 |
| gen_step1x_v1p2 | Generative | 41.69 | 5.13 | 0.89 | 42.2 | 39.9 | 2.3 | 4 |
| gen_SUSTechGAN | Generative | 41.67 | 5.11 | 0.87 | 42.33 | 39.67 | 2.66 | 4 |
| baseline | Baseline | 40.8 | 5.62 | 0.0 | 41.3 | 39.24 | 2.06 | 37 |
| std_autoaugment | Standard Aug | 40.53 | 5.68 | -0.27 | 40.97 | 38.88 | 2.09 | 14 |
| std_cutmix | Standard Aug | 40.0 | 5.78 | -0.8 | 40.39 | 38.13 | 2.26 | 12 |
| std_mixup | Standard Aug | 39.86 | 5.98 | -0.94 | 40.02 | 38.6 | 1.43 | 10 |
---

## Per-Dataset Breakdown

mIoU performance on each dataset.

| Strategy | Type | bdd10k | bdd10k_gain | idd-aw | idd-aw_gain | mapillaryvistas | mapillaryvistas_gain | outside15k | outside15k_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| std_randaugment | Standard Aug | 47.70 | +3.16 | - | - | 35.09 | -0.13 | 44.68 | +2.52 |
| gen_IP2P | Generative | 47.71 | +3.17 | - | - | 35.00 | -0.23 | 44.69 | +2.52 |
| gen_flux_kontext | Generative | 47.04 | +2.50 | - | - | 34.97 | -0.26 | 44.94 | +2.78 |
| gen_VisualCloze | Generative | 47.18 | +2.64 | - | - | 35.00 | -0.23 | 44.72 | +2.55 |
| gen_albumentations_weather | Generative | 47.14 | +2.59 | - | - | 35.07 | -0.16 | 44.67 | +2.50 |
| gen_Img2Img | Generative | 46.98 | +2.44 | - | - | 34.98 | -0.25 | 44.81 | +2.64 |
| gen_cyclediffusion | Generative | 46.92 | +2.38 | - | - | 35.02 | -0.21 | 44.77 | +2.61 |
| gen_step1x_new | Generative | 46.40 | +1.86 | - | - | 35.09 | -0.14 | 45.11 | +2.94 |
| gen_automold | Generative | 47.13 | +2.59 | - | - | 34.99 | -0.24 | 44.41 | +2.24 |
| gen_cycleGAN | Generative | 46.92 | +2.38 | - | - | 34.89 | -0.34 | 44.68 | +2.51 |
| gen_Attribute_Hallucination | Generative | 46.75 | +2.21 | - | - | 34.77 | -0.46 | 44.96 | +2.80 |
| gen_CUT | Generative | 46.81 | +2.27 | - | - | 35.02 | -0.21 | 44.65 | +2.48 |
| gen_UniControl | Generative | 46.43 | +1.89 | - | - | 34.96 | -0.26 | 45.01 | +2.84 |
| gen_LANIT | Generative | 46.62 | +2.08 | - | - | 34.72 | -0.51 | 45.06 | +2.90 |
| gen_step1x_v1p2 | Generative | 46.41 | +1.87 | - | - | 34.97 | -0.26 | 44.88 | +2.72 |
| gen_SUSTechGAN | Generative | 46.36 | +1.81 | - | - | 34.91 | -0.32 | 44.76 | +2.60 |
| std_mixup | Standard Aug | 45.75 | +1.21 | - | - | 33.43 | -1.80 | 44.52 | +2.36 |
| baseline | Baseline | 44.54 | +0.00 | 40.82 | +0.00 | 35.23 | +0.00 | 42.17 | +0.00 |
| std_autoaugment | Standard Aug | 45.94 | +1.39 | - | - | 33.22 | -2.01 | 42.18 | +0.01 |
| std_cutmix | Standard Aug | 45.85 | +1.31 | - | - | 33.42 | -1.81 | 41.89 | -0.28 |
---

## Per-Domain Breakdown

mIoU performance on each weather domain. Normal = clear_day. Adverse = cloudy, dawn_dusk, foggy, night, rainy, snowy.

| Strategy | Type | clear_day | cloudy | dawn_dusk | foggy | night | rainy | snowy | Normal Avg | Adverse Avg | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Baseline | 41.30 | 41.21 | 36.17 | 46.61 | 32.27 | 38.90 | 40.29 | 41.30 | 39.24 | 2.06 |
| gen_Attribute_Hallucination | Generative | 42.33 | 41.37 | 36.55 | 46.29 | 33.52 | 39.84 | 42.14 | 42.33 | 39.95 | 2.38 |
| gen_CUT | Generative | 42.22 | 41.45 | 37.40 | 46.86 | 33.38 | 39.80 | 41.72 | 42.22 | 40.10 | 2.11 |
| gen_IP2P | Generative | 42.49 | 41.42 | 36.65 | 48.08 | 32.62 | 39.95 | 42.16 | 42.49 | 40.15 | 2.34 |
| gen_Img2Img | Generative | 42.32 | 41.23 | 36.48 | 46.94 | 33.57 | 39.63 | 41.51 | 42.32 | 39.89 | 2.43 |
| gen_LANIT | Generative | 42.23 | 41.36 | 37.42 | 46.62 | 32.60 | 39.72 | 41.64 | 42.23 | 39.89 | 2.33 |
| gen_SUSTechGAN | Generative | 42.33 | 40.99 | 36.23 | 46.65 | 32.56 | 39.60 | 42.01 | 42.33 | 39.67 | 2.66 |
| gen_UniControl | Generative | 42.19 | 41.76 | 36.40 | 46.57 | 32.88 | 39.80 | 42.25 | 42.19 | 39.94 | 2.24 |
| gen_VisualCloze | Generative | 42.37 | 41.16 | 36.64 | 44.96 | 32.78 | 40.22 | 41.74 | 42.37 | 39.58 | 2.79 |
| gen_albumentations_weather | Generative | 42.10 | 41.64 | 36.40 | 47.53 | 33.63 | 39.92 | 41.73 | 42.10 | 40.14 | 1.96 |
| gen_automold | Generative | 42.38 | 41.41 | 36.63 | 46.19 | 32.60 | 39.26 | 41.92 | 42.38 | 39.67 | 2.71 |
| gen_cycleGAN | Generative | 42.23 | 41.22 | 36.91 | 46.68 | 32.71 | 40.14 | 41.47 | 42.23 | 39.85 | 2.37 |
| gen_cyclediffusion | Generative | 42.46 | 41.35 | 37.45 | 46.21 | 33.30 | 40.04 | 41.57 | 42.46 | 39.99 | 2.47 |
| gen_flux_kontext | Generative | 42.30 | 41.38 | 36.76 | 45.10 | 32.38 | 40.10 | 41.57 | 42.30 | 39.55 | 2.75 |
| gen_step1x_new | Generative | 42.40 | 41.34 | 37.56 | 47.33 | 32.84 | 39.66 | 41.94 | 42.40 | 40.11 | 2.29 |
| gen_step1x_v1p2 | Generative | 42.20 | 41.51 | 37.41 | 47.21 | 32.90 | 39.58 | 40.78 | 42.20 | 39.90 | 2.30 |
| std_autoaugment | Standard Aug | 40.97 | 40.74 | 36.34 | 44.97 | 32.18 | 38.90 | 40.16 | 40.97 | 38.88 | 2.09 |
| std_cutmix | Standard Aug | 40.39 | 39.72 | 35.95 | 43.83 | 31.68 | 38.42 | 39.19 | 40.39 | 38.13 | 2.26 |
| std_mixup | Standard Aug | 40.02 | 40.32 | 36.73 | 44.15 | 31.52 | 38.55 | 40.31 | 40.02 | 38.60 | 1.43 |
| std_randaugment | Standard Aug | 42.33 | 41.34 | 37.27 | 46.95 | 32.78 | 40.61 | 41.66 | 42.33 | 40.10 | 2.22 |
