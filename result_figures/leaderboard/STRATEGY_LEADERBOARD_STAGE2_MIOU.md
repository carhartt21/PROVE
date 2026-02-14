# Stage 2 Strategy Leaderboard (by mIoU)

**Stage 2**: All Domains Training — domain-inclusive evaluation

**Metric**: mIoU (Mean Intersection over Union)

**Last Updated**: 2026-02-13 23:12
**Baseline mIoU**: 40.85%
**Total Results**: 344 test results from 26 strategies

---

## Overall Strategy Ranking

Sorted by mIoU. Gain = improvement over baseline. Domain Gap = Normal mIoU - Adverse mIoU (lower is better).

| Strategy | Type | mIoU | Std | Gain | Normal mIoU | Adverse mIoU | Domain Gap | Num Tests |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_LANIT | Generative | 41.76 | 5.33 | 0.91 | 42.23 | 39.89 | 2.33 | 4 |
| gen_step1x_new | Generative | 41.11 | 5.49 | 0.25 | 41.57 | 39.64 | 1.93 | 16 |
| gen_flux_kontext | Generative | 41.11 | 5.54 | 0.25 | 41.5 | 39.09 | 2.41 | 16 |
| gen_UniControl | Generative | 41.04 | 5.48 | 0.18 | 41.45 | 39.63 | 1.82 | 16 |
| gen_Qwen_Image_Edit | Generative | 40.97 | 5.37 | 0.12 | 41.44 | 39.29 | 2.14 | 16 |
| gen_albumentations_weather | Generative | 40.86 | 6.1 | 0.01 | 41.04 | 38.71 | 2.32 | 11 |
| baseline | Baseline | 40.85 | 5.47 | 0.0 | 41.34 | 39.32 | 2.02 | 39 |
| std_autoaugment | Standard Aug | 40.85 | 5.48 | -0.01 | 41.2 | 39.28 | 1.92 | 16 |
| gen_Img2Img | Generative | 40.77 | 5.41 | -0.09 | 41.25 | 39.24 | 2.01 | 16 |
| gen_CUT | Generative | 40.68 | 5.26 | -0.17 | 41.06 | 39.26 | 1.8 | 16 |
| gen_augmenters | Generative | 40.44 | 6.07 | -0.41 | 40.8 | 38.68 | 2.13 | 16 |
| std_cutmix | Standard Aug | 40.44 | 6.24 | -0.42 | 40.76 | 38.77 | 1.99 | 16 |
| std_randaugment | Standard Aug | 40.4 | 5.52 | -0.45 | 40.63 | 38.67 | 1.96 | 14 |
| gen_cycleGAN | Generative | 40.34 | 6.14 | -0.51 | 40.71 | 38.84 | 1.88 | 16 |
| gen_IP2P | Generative | 40.12 | 5.87 | -0.73 | 40.47 | 38.1 | 2.37 | 10 |
| gen_SUSTechGAN | Generative | 40.06 | 5.83 | -0.8 | 40.48 | 37.67 | 2.81 | 10 |
| std_mixup | Standard Aug | 39.95 | 5.22 | -0.91 | 40.2 | 38.34 | 1.86 | 14 |
| gen_VisualCloze | Generative | 39.59 | 5.48 | -1.26 | 39.9 | 37.46 | 2.44 | 10 |
| gen_step1x_v1p2 | Generative | 39.57 | 5.42 | -1.28 | 39.94 | 37.62 | 2.32 | 10 |
| gen_cyclediffusion | Generative | 39.48 | 5.49 | -1.37 | 39.87 | 37.51 | 2.36 | 10 |
| gen_automold | Generative | 39.42 | 5.42 | -1.44 | 39.76 | 37.45 | 2.31 | 10 |
| gen_Attribute_Hallucination | Generative | 39.38 | 5.45 | -1.48 | 39.74 | 37.37 | 2.37 | 10 |
| gen_stargan_v2 | Generative | 39.2 | 6.05 | -1.66 | 39.41 | 36.35 | 3.06 | 8 |
| gen_CNetSeg | Generative | 39.18 | 6.23 | -1.67 | 39.25 | 36.85 | 2.4 | 8 |
| gen_Weather_Effect_Generator | Generative | 39.03 | 6.05 | -1.82 | 39.17 | 36.79 | 2.38 | 8 |
| gen_TSIT | Generative | 38.69 | 5.55 | -2.16 | 38.84 | 36.51 | 2.33 | 8 |
---

## Per-Dataset Breakdown

mIoU performance on each dataset.

| Strategy | Type | bdd10k | bdd10k_gain | iddaw | iddaw_gain | mapillaryvistas | mapillaryvistas_gain | outside15k | outside15k_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_LANIT | Generative | 46.62 | +2.08 | 40.64 | +1.37 | 34.72 | -1.15 | 45.06 | +2.84 |
| gen_albumentations_weather | Generative | 47.83 | +3.29 | 40.61 | +1.34 | 35.21 | -0.66 | 43.08 | +0.86 |
| gen_IP2P | Generative | 47.71 | +3.17 | 40.53 | +1.26 | 35.18 | -0.69 | 43.07 | +0.85 |
| gen_SUSTechGAN | Generative | 46.36 | +1.81 | 40.65 | +1.38 | 35.08 | -0.79 | 43.30 | +1.08 |
| gen_VisualCloze | Generative | 47.18 | +2.64 | 40.71 | +1.44 | 35.24 | -0.63 | 41.76 | -0.47 |
| gen_flux_kontext | Generative | 46.32 | +1.77 | 39.68 | +0.40 | 35.21 | -0.66 | 43.23 | +1.00 |
| gen_step1x_new | Generative | 46.37 | +1.83 | 39.75 | +0.48 | 35.16 | -0.72 | 43.14 | +0.92 |
| gen_cyclediffusion | Generative | 46.92 | +2.38 | 40.67 | +1.40 | 35.12 | -0.75 | 41.69 | -0.53 |
| gen_automold | Generative | 47.13 | +2.59 | 40.56 | +1.28 | 35.23 | -0.64 | 41.39 | -0.84 |
| gen_UniControl | Generative | 46.46 | +1.92 | 39.89 | +0.61 | 35.17 | -0.70 | 42.63 | +0.40 |
| gen_step1x_v1p2 | Generative | 46.41 | +1.87 | 40.49 | +1.22 | 35.04 | -0.83 | 42.16 | -0.07 |
| gen_Attribute_Hallucination | Generative | 46.75 | +2.21 | 40.62 | +1.35 | 35.20 | -0.67 | 41.40 | -0.82 |
| gen_Qwen_Image_Edit | Generative | 46.11 | +1.57 | 39.81 | +0.54 | 35.24 | -0.63 | 42.71 | +0.49 |
| std_autoaugment | Standard Aug | 46.42 | +1.88 | 39.69 | +0.41 | 35.25 | -0.62 | 42.03 | -0.19 |
| gen_Img2Img | Generative | 46.53 | +1.99 | 39.75 | +0.48 | 35.23 | -0.65 | 41.56 | -0.66 |
| gen_CUT | Generative | 46.04 | +1.50 | 39.82 | +0.55 | 35.24 | -0.63 | 41.62 | -0.60 |
| std_randaugment | Standard Aug | 46.03 | +1.49 | 38.86 | -0.41 | 35.17 | -0.70 | 42.56 | +0.34 |
| baseline | Baseline | 44.54 | +0.00 | 39.27 | +0.00 | 35.87 | +0.00 | 42.22 | +0.00 |
| gen_augmenters | Generative | 46.44 | +1.90 | 37.04 | -2.24 | 35.23 | -0.64 | 43.06 | +0.84 |
| std_cutmix | Standard Aug | 47.30 | +2.76 | 37.02 | -2.25 | 35.19 | -0.68 | 42.23 | +0.01 |
| gen_cycleGAN | Generative | 46.86 | +2.32 | 36.92 | -2.36 | 35.13 | -0.75 | 42.48 | +0.26 |
| std_mixup | Standard Aug | 45.75 | +1.21 | 38.83 | -0.45 | 35.35 | -0.53 | 41.04 | -1.19 |
| gen_stargan_v2 | Generative | - | - | - | - | 35.18 | -0.69 | 43.21 | +0.99 |
| gen_CNetSeg | Generative | - | - | - | - | 35.03 | -0.84 | 43.33 | +1.11 |
| gen_Weather_Effect_Generator | Generative | - | - | - | - | 35.10 | -0.77 | 42.96 | +0.73 |
| gen_TSIT | Generative | - | - | - | - | 35.18 | -0.69 | 42.21 | -0.01 |
---

## Per-Domain Breakdown

mIoU performance on each weather domain. Normal = clear_day. Adverse = cloudy, dawn_dusk, foggy, night, rainy, snowy.

| Strategy | Type | clear_day | cloudy | dawn_dusk | foggy | night | rainy | snowy | Normal Avg | Adverse Avg | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Baseline | 41.34 | 41.19 | 36.38 | 46.56 | 32.56 | 38.94 | 40.28 | 41.34 | 39.32 | 2.02 |
| gen_Attribute_Hallucination | Generative | 39.74 | 38.64 | 35.50 | 43.34 | 32.16 | 37.20 | 37.39 | 39.74 | 37.37 | 2.37 |
| gen_CNetSeg | Generative | 39.25 | 37.57 | 36.00 | 43.04 | 31.84 | 37.08 | 35.58 | 39.25 | 36.85 | 2.40 |
| gen_CUT | Generative | 41.06 | 40.59 | 36.97 | 45.14 | 33.15 | 39.12 | 40.61 | 41.06 | 39.26 | 1.80 |
| gen_IP2P | Generative | 40.47 | 39.34 | 36.04 | 45.06 | 32.45 | 37.60 | 38.11 | 40.47 | 38.10 | 2.37 |
| gen_Img2Img | Generative | 41.25 | 40.82 | 36.95 | 44.81 | 33.20 | 38.94 | 40.71 | 41.25 | 39.24 | 2.01 |
| gen_LANIT | Generative | 42.23 | 41.36 | 37.42 | 46.62 | 32.60 | 39.72 | 41.64 | 42.23 | 39.89 | 2.33 |
| gen_Qwen_Image_Edit | Generative | 41.44 | 41.00 | 37.33 | 45.23 | 32.95 | 38.99 | 40.25 | 41.44 | 39.29 | 2.14 |
| gen_SUSTechGAN | Generative | 40.48 | 38.83 | 35.66 | 43.79 | 32.42 | 37.74 | 37.59 | 40.48 | 37.67 | 2.81 |
| gen_TSIT | Generative | 38.84 | 37.69 | 35.11 | 42.93 | 32.01 | 36.21 | 35.11 | 38.84 | 36.51 | 2.33 |
| gen_UniControl | Generative | 41.45 | 40.97 | 37.11 | 45.77 | 33.29 | 39.29 | 41.33 | 41.45 | 39.63 | 1.82 |
| gen_VisualCloze | Generative | 39.90 | 38.60 | 36.16 | 43.02 | 31.91 | 37.52 | 37.58 | 39.90 | 37.46 | 2.44 |
| gen_Weather_Effect_Generator | Generative | 39.17 | 37.67 | 36.16 | 42.83 | 32.28 | 36.70 | 35.10 | 39.17 | 36.79 | 2.38 |
| gen_albumentations_weather | Generative | 41.04 | 40.02 | 36.12 | 44.91 | 32.74 | 39.09 | 39.40 | 41.04 | 38.71 | 2.32 |
| gen_augmenters | Generative | 40.80 | 40.23 | 36.17 | 44.64 | 32.25 | 38.90 | 39.87 | 40.80 | 38.68 | 2.13 |
| gen_automold | Generative | 39.76 | 38.40 | 35.71 | 43.59 | 32.01 | 37.33 | 37.64 | 39.76 | 37.45 | 2.31 |
| gen_cycleGAN | Generative | 40.71 | 40.61 | 36.33 | 44.87 | 32.09 | 38.77 | 40.35 | 40.71 | 38.84 | 1.88 |
| gen_cyclediffusion | Generative | 39.87 | 38.62 | 35.71 | 43.84 | 32.23 | 37.64 | 37.03 | 39.87 | 37.51 | 2.36 |
| gen_flux_kontext | Generative | 41.50 | 41.07 | 36.57 | 44.27 | 32.54 | 39.43 | 40.65 | 41.50 | 39.09 | 2.41 |
| gen_stargan_v2 | Generative | 39.41 | 37.51 | 34.46 | 42.55 | 31.73 | 36.90 | 34.94 | 39.41 | 36.35 | 3.06 |
| gen_step1x_new | Generative | 41.57 | 41.27 | 37.43 | 45.81 | 33.02 | 39.35 | 40.93 | 41.57 | 39.64 | 1.93 |
| gen_step1x_v1p2 | Generative | 39.94 | 38.65 | 36.00 | 44.13 | 32.32 | 37.40 | 37.21 | 39.94 | 37.62 | 2.32 |
| std_autoaugment | Standard Aug | 41.20 | 40.71 | 37.14 | 45.00 | 32.99 | 39.21 | 40.65 | 41.20 | 39.28 | 1.92 |
| std_cutmix | Standard Aug | 40.76 | 40.27 | 36.58 | 44.25 | 32.49 | 38.90 | 40.15 | 40.76 | 38.77 | 1.99 |
| std_mixup | Standard Aug | 40.20 | 39.86 | 36.04 | 44.48 | 32.25 | 38.23 | 39.21 | 40.20 | 38.34 | 1.86 |
| std_randaugment | Standard Aug | 40.63 | 40.06 | 36.59 | 44.49 | 32.45 | 38.70 | 39.75 | 40.63 | 38.67 | 1.96 |
---

## Per-Model Breakdown

mIoU performance on each model architecture. Gain columns show improvement over baseline per model.

| Strategy | Type | deeplabv3plus_r50 | deeplabv3plus_r50_gain | hrnet_hr48 | hrnet_hr48_gain | mask2former_swin-b | mask2former_swin-b_gain | pspnet_r50 | pspnet_r50_gain | segformer_mit-b3 | segformer_mit-b3_gain | segformer_mit-b5 | segformer_mit-b5_gain | segnext_mscan-b | segnext_mscan-b_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_LANIT | Generative | - | - | - | - | - | - | - | - | 41.76 | -1.84 | - | - | - | - |
| baseline | Baseline | 37.57 | +0.00 | 38.26 | +0.00 | 44.30 | +0.00 | 36.22 | +0.00 | 43.60 | +0.00 | 48.14 | +0.00 | 43.86 | +0.00 |
| gen_flux_kontext | Generative | - | - | - | - | 44.38 | +0.08 | 35.70 | -0.52 | 41.84 | -1.76 | - | - | 42.50 | -1.36 |
| gen_step1x_new | Generative | - | - | - | - | 44.53 | +0.23 | 35.92 | -0.30 | 41.77 | -1.83 | - | - | 42.20 | -1.66 |
| gen_UniControl | Generative | - | - | - | - | 44.28 | -0.02 | 35.73 | -0.49 | 41.77 | -1.84 | - | - | 42.37 | -1.49 |
| gen_Qwen_Image_Edit | Generative | - | - | - | - | 44.03 | -0.27 | 35.83 | -0.39 | 41.74 | -1.87 | - | - | 42.28 | -1.58 |
| std_autoaugment | Standard Aug | - | - | - | - | 43.43 | -0.87 | 35.56 | -0.66 | 41.79 | -1.81 | - | - | 42.60 | -1.26 |
| gen_Img2Img | Generative | - | - | - | - | 43.34 | -0.96 | 35.72 | -0.50 | 41.78 | -1.82 | - | - | 42.23 | -1.63 |
| std_randaugment | Standard Aug | - | - | - | - | 42.76 | -1.54 | 35.66 | -0.56 | 42.01 | -1.60 | - | - | 42.35 | -1.51 |
| gen_CUT | Generative | - | - | - | - | 42.79 | -1.51 | 35.74 | -0.48 | 41.75 | -1.86 | - | - | 42.45 | -1.41 |
| gen_augmenters | Generative | - | - | - | - | 41.77 | -2.52 | 35.62 | -0.59 | 42.01 | -1.59 | - | - | 42.36 | -1.50 |
| std_cutmix | Standard Aug | - | - | - | - | 41.74 | -2.56 | 35.63 | -0.59 | 41.88 | -1.72 | - | - | 42.49 | -1.36 |
| gen_albumentations_weather | Generative | - | - | - | - | 43.88 | -0.42 | 33.02 | -3.20 | 41.87 | -1.73 | - | - | 42.72 | -1.13 |
| gen_cycleGAN | Generative | - | - | - | - | 41.82 | -2.48 | 35.63 | -0.59 | 41.74 | -1.86 | - | - | 42.18 | -1.67 |
| std_mixup | Standard Aug | - | - | - | - | 39.74 | -4.56 | 35.81 | -0.40 | 41.85 | -1.75 | - | - | 42.28 | -1.58 |
| gen_IP2P | Generative | - | - | - | - | 43.39 | -0.91 | 33.07 | -3.15 | 41.98 | -1.62 | - | - | 40.20 | -3.65 |
| gen_SUSTechGAN | Generative | - | - | - | - | 43.64 | -0.66 | 33.04 | -3.18 | 41.67 | -1.93 | - | - | 40.25 | -3.60 |
| gen_stargan_v2 | Generative | - | - | - | - | 43.70 | -0.59 | 32.95 | -3.26 | 40.01 | -3.60 | - | - | 40.12 | -3.73 |
| gen_CNetSeg | Generative | - | - | - | - | 43.77 | -0.53 | 32.83 | -3.39 | 39.89 | -3.71 | - | - | 40.22 | -3.63 |
| gen_step1x_v1p2 | Generative | - | - | - | - | 41.40 | -2.89 | 33.16 | -3.06 | 41.69 | -1.91 | - | - | 39.91 | -3.95 |
| gen_Weather_Effect_Generator | Generative | - | - | - | - | 43.60 | -0.70 | 32.68 | -3.54 | 39.82 | -3.78 | - | - | 40.02 | -3.83 |
| gen_VisualCloze | Generative | - | - | - | - | 40.96 | -3.34 | 32.96 | -3.26 | 41.90 | -1.70 | - | - | 40.22 | -3.64 |
| gen_cyclediffusion | Generative | - | - | - | - | 40.61 | -3.69 | 32.74 | -3.48 | 41.85 | -1.76 | - | - | 40.37 | -3.48 |
| gen_automold | Generative | - | - | - | - | 40.32 | -3.98 | 32.76 | -3.46 | 41.77 | -1.83 | - | - | 40.47 | -3.39 |
| gen_Attribute_Hallucination | Generative | - | - | - | - | 40.12 | -4.17 | 32.89 | -3.33 | 41.77 | -1.83 | - | - | 40.32 | -3.54 |
| gen_TSIT | Generative | - | - | - | - | 42.17 | -2.12 | 32.77 | -3.45 | 39.85 | -3.76 | - | - | 39.98 | -3.88 |
