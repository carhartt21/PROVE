# Stage 2 Strategy Leaderboard (by mIoU)

**Stage 2**: All Domains Training — domain-inclusive evaluation

**Metric**: mIoU (Mean Intersection over Union)

**Last Updated**: 2026-02-16 15:24
**Baseline mIoU**: 40.85%
**Total Results**: 429 test results from 26 strategies

---

## Overall Strategy Ranking

Sorted by mIoU. Gain = improvement over baseline. Domain Gap = Normal mIoU - Adverse mIoU (lower is better).

| Strategy | Type | mIoU | Std | Gain | Normal mIoU | Adverse mIoU | Domain Gap | Num Tests |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_LANIT | Generative | 41.25 | 4.33 | 0.4 | 41.58 | 39.66 | 1.92 | 6 |
| gen_CNetSeg | Generative | 41.2 | 5.72 | 0.34 | 41.61 | 39.5 | 2.11 | 16 |
| gen_SUSTechGAN | Generative | 41.2 | 5.69 | 0.35 | 41.77 | 39.4 | 2.37 | 16 |
| gen_stargan_v2 | Generative | 41.12 | 5.49 | 0.27 | 41.59 | 39.23 | 2.36 | 16 |
| gen_albumentations_weather | Generative | 41.11 | 5.53 | 0.26 | 41.45 | 39.41 | 2.03 | 16 |
| gen_flux_kontext | Generative | 41.11 | 5.54 | 0.25 | 41.5 | 39.09 | 2.41 | 16 |
| gen_step1x_new | Generative | 41.11 | 5.49 | 0.25 | 41.57 | 39.64 | 1.93 | 16 |
| std_randaugment | Standard Aug | 41.07 | 5.58 | 0.22 | 41.41 | 39.54 | 1.87 | 16 |
| gen_Weather_Effect_Generator | Generative | 41.05 | 5.57 | 0.19 | 41.39 | 39.43 | 1.96 | 16 |
| gen_UniControl | Generative | 41.04 | 5.48 | 0.18 | 41.45 | 39.63 | 1.82 | 16 |
| gen_Qwen_Image_Edit | Generative | 40.97 | 5.37 | 0.12 | 41.44 | 39.29 | 2.14 | 16 |
| gen_VisualCloze | Generative | 40.9 | 5.53 | 0.04 | 41.22 | 39.19 | 2.03 | 16 |
| gen_cyclediffusion | Generative | 40.86 | 5.54 | 0.01 | 41.36 | 39.32 | 2.04 | 16 |
| std_autoaugment | Standard Aug | 40.85 | 5.48 | -0.01 | 41.2 | 39.28 | 1.92 | 16 |
| baseline | Baseline | 40.85 | 5.47 | 0.0 | 41.34 | 39.32 | 2.02 | 39 |
| gen_TSIT | Generative | 40.82 | 5.29 | -0.03 | 41.24 | 39.39 | 1.85 | 16 |
| gen_step1x_v1p2 | Generative | 40.81 | 5.32 | -0.05 | 41.27 | 39.24 | 2.03 | 16 |
| std_mixup | Standard Aug | 40.8 | 5.57 | -0.05 | 41.15 | 39.38 | 1.78 | 16 |
| gen_Img2Img | Generative | 40.77 | 5.41 | -0.09 | 41.25 | 39.24 | 2.01 | 16 |
| gen_automold | Generative | 40.71 | 5.32 | -0.14 | 41.13 | 39.01 | 2.12 | 16 |
| gen_CUT | Generative | 40.68 | 5.26 | -0.17 | 41.06 | 39.26 | 1.8 | 16 |
| gen_IP2P | Generative | 40.62 | 6.36 | -0.23 | 41.04 | 39.04 | 2.0 | 16 |
| std_cutmix | Standard Aug | 40.44 | 6.24 | -0.42 | 40.76 | 38.77 | 1.99 | 16 |
| gen_augmenters | Generative | 40.44 | 6.07 | -0.41 | 40.8 | 38.68 | 2.13 | 16 |
| gen_cycleGAN | Generative | 40.34 | 6.14 | -0.51 | 40.71 | 38.84 | 1.88 | 16 |
| gen_Attribute_Hallucination | Generative | 39.91 | 5.68 | -0.94 | 40.31 | 38.31 | 2.0 | 16 |
---

## Per-Dataset Breakdown

mIoU performance on each dataset.

| Strategy | Type | bdd10k | bdd10k_gain | iddaw | iddaw_gain | mapillaryvistas | mapillaryvistas_gain | outside15k | outside15k_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_LANIT | Generative | 46.62 | +2.08 | 40.64 | +1.37 | 36.65 | +0.78 | 43.48 | +1.25 |
| gen_SUSTechGAN | Generative | 46.59 | +2.04 | 39.83 | +0.55 | 35.08 | -0.79 | 43.30 | +1.08 |
| gen_CNetSeg | Generative | 46.58 | +2.04 | 39.85 | +0.57 | 35.03 | -0.84 | 43.33 | +1.11 |
| gen_stargan_v2 | Generative | 46.24 | +1.70 | 39.84 | +0.57 | 35.18 | -0.69 | 43.21 | +0.99 |
| gen_albumentations_weather | Generative | 46.41 | +1.87 | 39.75 | +0.47 | 35.21 | -0.66 | 43.08 | +0.86 |
| gen_flux_kontext | Generative | 46.32 | +1.77 | 39.68 | +0.40 | 35.21 | -0.66 | 43.23 | +1.00 |
| gen_step1x_new | Generative | 46.37 | +1.83 | 39.75 | +0.48 | 35.16 | -0.72 | 43.14 | +0.92 |
| std_randaugment | Standard Aug | 46.76 | +2.21 | 39.80 | +0.53 | 35.17 | -0.70 | 42.56 | +0.34 |
| gen_Weather_Effect_Generator | Generative | 46.31 | +1.77 | 39.82 | +0.54 | 35.10 | -0.77 | 42.96 | +0.73 |
| gen_UniControl | Generative | 46.46 | +1.92 | 39.89 | +0.61 | 35.17 | -0.70 | 42.63 | +0.40 |
| gen_Qwen_Image_Edit | Generative | 46.11 | +1.57 | 39.81 | +0.54 | 35.24 | -0.63 | 42.71 | +0.49 |
| gen_VisualCloze | Generative | 46.86 | +2.32 | 39.73 | +0.45 | 35.24 | -0.63 | 41.76 | -0.47 |
| gen_cyclediffusion | Generative | 46.70 | +2.16 | 39.94 | +0.67 | 35.12 | -0.75 | 41.69 | -0.53 |
| std_autoaugment | Standard Aug | 46.42 | +1.88 | 39.69 | +0.41 | 35.25 | -0.62 | 42.03 | -0.19 |
| gen_TSIT | Generative | 46.05 | +1.51 | 39.85 | +0.58 | 35.18 | -0.69 | 42.21 | -0.01 |
| std_mixup | Standard Aug | 46.95 | +2.41 | 39.88 | +0.61 | 35.35 | -0.53 | 41.04 | -1.19 |
| gen_step1x_v1p2 | Generative | 46.19 | +1.65 | 39.83 | +0.55 | 35.04 | -0.83 | 42.16 | -0.07 |
| gen_Img2Img | Generative | 46.53 | +1.99 | 39.75 | +0.48 | 35.23 | -0.65 | 41.56 | -0.66 |
| gen_automold | Generative | 46.31 | +1.77 | 39.90 | +0.63 | 35.23 | -0.64 | 41.39 | -0.84 |
| gen_CUT | Generative | 46.04 | +1.50 | 39.82 | +0.55 | 35.24 | -0.63 | 41.62 | -0.60 |
| gen_IP2P | Generative | 47.36 | +2.82 | 36.89 | -2.38 | 35.18 | -0.69 | 43.07 | +0.85 |
| baseline | Baseline | 44.54 | +0.00 | 39.27 | +0.00 | 35.87 | +0.00 | 42.22 | +0.00 |
| gen_augmenters | Generative | 46.44 | +1.90 | 37.04 | -2.24 | 35.23 | -0.64 | 43.06 | +0.84 |
| std_cutmix | Standard Aug | 47.30 | +2.76 | 37.02 | -2.25 | 35.19 | -0.68 | 42.23 | +0.01 |
| gen_cycleGAN | Generative | 46.86 | +2.32 | 36.92 | -2.36 | 35.13 | -0.75 | 42.48 | +0.26 |
| gen_Attribute_Hallucination | Generative | 46.00 | +1.46 | 37.03 | -2.24 | 35.20 | -0.67 | 41.40 | -0.82 |
---

## Per-Domain Breakdown

mIoU performance on each weather domain. Normal = clear_day. Adverse = cloudy, dawn_dusk, foggy, night, rainy, snowy.

| Strategy | Type | clear_day | cloudy | dawn_dusk | foggy | night | rainy | snowy | Normal Avg | Adverse Avg | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Baseline | 41.34 | 41.19 | 36.38 | 46.56 | 32.56 | 38.94 | 40.28 | 41.34 | 39.32 | 2.02 |
| gen_Attribute_Hallucination | Generative | 40.31 | 40.03 | 35.87 | 44.22 | 32.10 | 38.17 | 39.48 | 40.31 | 38.31 | 2.00 |
| gen_CNetSeg | Generative | 41.61 | 41.35 | 37.48 | 44.92 | 33.10 | 39.51 | 40.64 | 41.61 | 39.50 | 2.11 |
| gen_CUT | Generative | 41.06 | 40.59 | 36.97 | 45.14 | 33.15 | 39.12 | 40.61 | 41.06 | 39.26 | 1.80 |
| gen_IP2P | Generative | 41.04 | 40.72 | 36.43 | 45.36 | 32.51 | 38.56 | 40.64 | 41.04 | 39.04 | 2.00 |
| gen_Img2Img | Generative | 41.25 | 40.82 | 36.95 | 44.81 | 33.20 | 38.94 | 40.71 | 41.25 | 39.24 | 2.01 |
| gen_LANIT | Generative | 41.58 | 40.49 | 38.19 | 45.64 | 33.83 | 39.48 | 40.34 | 41.58 | 39.66 | 1.92 |
| gen_Qwen_Image_Edit | Generative | 41.44 | 41.00 | 37.33 | 45.23 | 32.95 | 38.99 | 40.25 | 41.44 | 39.29 | 2.14 |
| gen_SUSTechGAN | Generative | 41.77 | 41.05 | 36.81 | 45.42 | 33.10 | 39.29 | 40.73 | 41.77 | 39.40 | 2.37 |
| gen_TSIT | Generative | 41.24 | 40.98 | 36.88 | 45.81 | 33.16 | 39.02 | 40.52 | 41.24 | 39.39 | 1.85 |
| gen_UniControl | Generative | 41.45 | 40.97 | 37.11 | 45.77 | 33.29 | 39.29 | 41.33 | 41.45 | 39.63 | 1.82 |
| gen_VisualCloze | Generative | 41.22 | 40.89 | 37.03 | 44.72 | 32.65 | 39.43 | 40.41 | 41.22 | 39.19 | 2.03 |
| gen_Weather_Effect_Generator | Generative | 41.39 | 41.09 | 37.32 | 45.44 | 32.86 | 39.50 | 40.37 | 41.39 | 39.43 | 1.96 |
| gen_albumentations_weather | Generative | 41.45 | 41.11 | 37.04 | 45.15 | 33.24 | 39.56 | 40.40 | 41.45 | 39.41 | 2.03 |
| gen_augmenters | Generative | 40.80 | 40.23 | 36.17 | 44.64 | 32.25 | 38.90 | 39.87 | 40.80 | 38.68 | 2.13 |
| gen_automold | Generative | 41.13 | 40.74 | 36.67 | 44.69 | 32.46 | 39.07 | 40.44 | 41.13 | 39.01 | 2.12 |
| gen_cycleGAN | Generative | 40.71 | 40.61 | 36.33 | 44.87 | 32.09 | 38.77 | 40.35 | 40.71 | 38.84 | 1.88 |
| gen_cyclediffusion | Generative | 41.36 | 40.88 | 36.84 | 45.34 | 33.00 | 39.38 | 40.47 | 41.36 | 39.32 | 2.04 |
| gen_flux_kontext | Generative | 41.50 | 41.07 | 36.57 | 44.27 | 32.54 | 39.43 | 40.65 | 41.50 | 39.09 | 2.41 |
| gen_stargan_v2 | Generative | 41.59 | 41.16 | 36.75 | 45.25 | 33.06 | 39.41 | 39.76 | 41.59 | 39.23 | 2.36 |
| gen_step1x_new | Generative | 41.57 | 41.27 | 37.43 | 45.81 | 33.02 | 39.35 | 40.93 | 41.57 | 39.64 | 1.93 |
| gen_step1x_v1p2 | Generative | 41.27 | 40.83 | 36.78 | 45.72 | 32.78 | 39.04 | 40.29 | 41.27 | 39.24 | 2.03 |
| std_autoaugment | Standard Aug | 41.20 | 40.71 | 37.14 | 45.00 | 32.99 | 39.21 | 40.65 | 41.20 | 39.28 | 1.92 |
| std_cutmix | Standard Aug | 40.76 | 40.27 | 36.58 | 44.25 | 32.49 | 38.90 | 40.15 | 40.76 | 38.77 | 1.99 |
| std_mixup | Standard Aug | 41.15 | 41.06 | 36.95 | 45.45 | 33.07 | 39.15 | 40.57 | 41.15 | 39.38 | 1.78 |
| std_randaugment | Standard Aug | 41.41 | 41.12 | 37.35 | 45.25 | 33.12 | 39.31 | 41.10 | 41.41 | 39.54 | 1.87 |
---

## Per-Model Breakdown

mIoU performance on each model architecture. Gain columns show improvement over baseline per model.

| Strategy | Type | deeplabv3plus_r50 | deeplabv3plus_r50_gain | hrnet_hr48 | hrnet_hr48_gain | mask2former_swin-b | mask2former_swin-b_gain | pspnet_r50 | pspnet_r50_gain | segformer_mit-b3 | segformer_mit-b3_gain | segformer_mit-b5 | segformer_mit-b5_gain | segnext_mscan-b | segnext_mscan-b_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Baseline | 37.57 | +0.00 | 38.26 | +0.00 | 44.30 | +0.00 | 36.22 | +0.00 | 43.60 | +0.00 | 48.14 | +0.00 | 43.86 | +0.00 |
| gen_SUSTechGAN | Generative | - | - | - | - | 45.13 | +0.83 | 35.77 | -0.45 | 41.67 | -1.93 | - | - | 42.24 | -1.62 |
| gen_CNetSeg | Generative | - | - | - | - | 44.94 | +0.64 | 35.73 | -0.49 | 41.82 | -1.78 | - | - | 42.30 | -1.55 |
| gen_stargan_v2 | Generative | - | - | - | - | 44.45 | +0.15 | 35.77 | -0.45 | 42.06 | -1.54 | - | - | 42.20 | -1.66 |
| gen_albumentations_weather | Generative | - | - | - | - | 44.54 | +0.24 | 35.82 | -0.40 | 41.87 | -1.73 | - | - | 42.22 | -1.63 |
| gen_flux_kontext | Generative | - | - | - | - | 44.38 | +0.08 | 35.70 | -0.52 | 41.84 | -1.76 | - | - | 42.50 | -1.36 |
| gen_step1x_new | Generative | - | - | - | - | 44.53 | +0.23 | 35.92 | -0.30 | 41.77 | -1.83 | - | - | 42.20 | -1.66 |
| std_randaugment | Standard Aug | - | - | - | - | 44.27 | -0.03 | 35.66 | -0.56 | 42.01 | -1.60 | - | - | 42.35 | -1.51 |
| gen_Weather_Effect_Generator | Generative | - | - | - | - | 44.54 | +0.25 | 35.63 | -0.58 | 41.64 | -1.96 | - | - | 42.36 | -1.49 |
| gen_UniControl | Generative | - | - | - | - | 44.28 | -0.02 | 35.73 | -0.49 | 41.77 | -1.84 | - | - | 42.37 | -1.49 |
| gen_LANIT | Generative | - | - | - | - | 40.23 | -4.06 | - | - | 41.76 | -1.84 | - | - | - | - |
| gen_Qwen_Image_Edit | Generative | - | - | - | - | 44.03 | -0.27 | 35.83 | -0.39 | 41.74 | -1.87 | - | - | 42.28 | -1.58 |
| gen_VisualCloze | Generative | - | - | - | - | 43.48 | -0.82 | 35.78 | -0.44 | 41.90 | -1.70 | - | - | 42.42 | -1.44 |
| gen_cyclediffusion | Generative | - | - | - | - | 43.49 | -0.81 | 35.67 | -0.55 | 41.85 | -1.76 | - | - | 42.45 | -1.41 |
| std_autoaugment | Standard Aug | - | - | - | - | 43.43 | -0.87 | 35.56 | -0.66 | 41.79 | -1.81 | - | - | 42.60 | -1.26 |
| gen_TSIT | Generative | - | - | - | - | 43.69 | -0.61 | 35.70 | -0.52 | 41.68 | -1.92 | - | - | 42.21 | -1.64 |
| gen_step1x_v1p2 | Generative | - | - | - | - | 43.47 | -0.83 | 35.89 | -0.32 | 41.69 | -1.91 | - | - | 42.17 | -1.69 |
| std_mixup | Standard Aug | - | - | - | - | 43.27 | -1.03 | 35.81 | -0.40 | 41.85 | -1.75 | - | - | 42.28 | -1.58 |
| gen_Img2Img | Generative | - | - | - | - | 43.34 | -0.96 | 35.72 | -0.50 | 41.78 | -1.82 | - | - | 42.23 | -1.63 |
| gen_automold | Generative | - | - | - | - | 42.83 | -1.47 | 35.70 | -0.52 | 41.77 | -1.83 | - | - | 42.54 | -1.31 |
| gen_CUT | Generative | - | - | - | - | 42.79 | -1.51 | 35.74 | -0.48 | 41.75 | -1.86 | - | - | 42.45 | -1.41 |
| gen_IP2P | Generative | - | - | - | - | 42.32 | -1.98 | 35.82 | -0.40 | 41.98 | -1.62 | - | - | 42.38 | -1.48 |
| gen_augmenters | Generative | - | - | - | - | 41.77 | -2.52 | 35.62 | -0.59 | 42.01 | -1.59 | - | - | 42.36 | -1.50 |
| std_cutmix | Standard Aug | - | - | - | - | 41.74 | -2.56 | 35.63 | -0.59 | 41.88 | -1.72 | - | - | 42.49 | -1.36 |
| gen_cycleGAN | Generative | - | - | - | - | 41.82 | -2.48 | 35.63 | -0.59 | 41.74 | -1.86 | - | - | 42.18 | -1.67 |
| gen_Attribute_Hallucination | Generative | - | - | - | - | 39.78 | -4.51 | 35.74 | -0.48 | 41.77 | -1.83 | - | - | 42.33 | -1.53 |
