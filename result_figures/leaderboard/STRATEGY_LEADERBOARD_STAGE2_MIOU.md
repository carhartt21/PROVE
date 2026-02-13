# Stage 2 Strategy Leaderboard (by mIoU)

**Stage 2**: All Domains Training — domain-inclusive evaluation

**Metric**: mIoU (Mean Intersection over Union)

**Last Updated**: 2026-02-13 15:51
**Baseline mIoU**: 40.85%
**Total Results**: 295 test results from 24 strategies

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
| std_autoaugment | Standard Aug | 40.85 | 5.48 | -0.01 | 41.2 | 39.28 | 1.92 | 16 |
| baseline | Baseline | 40.85 | 5.47 | 0.0 | 41.34 | 39.32 | 2.02 | 39 |
| gen_Img2Img | Generative | 40.77 | 5.41 | -0.09 | 41.25 | 39.24 | 2.01 | 16 |
| gen_CUT | Generative | 40.68 | 5.26 | -0.17 | 41.06 | 39.26 | 1.8 | 16 |
| std_cutmix | Standard Aug | 40.44 | 6.24 | -0.42 | 40.76 | 38.77 | 1.99 | 16 |
| gen_augmenters | Generative | 40.44 | 6.07 | -0.41 | 40.8 | 38.68 | 2.13 | 16 |
| gen_cycleGAN | Generative | 40.34 | 6.14 | -0.51 | 40.71 | 38.84 | 1.88 | 16 |
| gen_albumentations_weather | Generative | 40.09 | 5.85 | -0.76 | 40.29 | 37.94 | 2.35 | 10 |
| std_mixup | Standard Aug | 39.95 | 5.22 | -0.91 | 40.2 | 38.34 | 1.86 | 14 |
| gen_VisualCloze | Generative | 39.83 | 6.08 | -1.03 | 39.96 | 37.94 | 2.02 | 8 |
| std_randaugment | Standard Aug | 39.7 | 5.46 | -1.15 | 39.89 | 37.85 | 2.04 | 12 |
| gen_step1x_v1p2 | Generative | 39.59 | 6.01 | -1.27 | 39.8 | 38.06 | 1.74 | 8 |
| gen_automold | Generative | 39.37 | 5.74 | -1.49 | 39.71 | 37.34 | 2.37 | 9 |
| gen_IP2P | Generative | 39.09 | 6.26 | -1.76 | 39.16 | 37.86 | 1.3 | 7 |
| gen_cyclediffusion | Generative | 38.68 | 6.67 | -2.17 | 38.9 | 37.15 | 1.75 | 6 |
| gen_Attribute_Hallucination | Generative | 38.67 | 6.64 | -2.18 | 38.85 | 37.26 | 1.59 | 6 |
| gen_SUSTechGAN | Generative | 38.55 | 6.54 | -2.3 | 38.83 | 36.95 | 1.88 | 6 |
| gen_stargan_v2 | Generative | 33.32 | 3.34 | -7.53 | 32.91 | 32.2 | 0.71 | 3 |
| gen_CNetSeg | Generative | 33.05 | 3.27 | -7.8 | 32.49 | 32.15 | 0.34 | 3 |
---

## Per-Dataset Breakdown

mIoU performance on each dataset.

| Strategy | Type | bdd10k | bdd10k_gain | iddaw | iddaw_gain | mapillaryvistas | mapillaryvistas_gain | outside15k | outside15k_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_IP2P | Generative | 47.71 | +3.17 | 40.53 | +1.26 | 35.18 | -0.69 | 44.69 | +2.47 |
| gen_VisualCloze | Generative | 47.18 | +2.64 | 40.71 | +1.44 | 35.24 | -0.63 | 44.87 | +2.65 |
| gen_LANIT | Generative | 46.62 | +2.08 | 40.64 | +1.37 | 34.72 | -1.15 | 45.06 | +2.84 |
| gen_step1x_v1p2 | Generative | 46.41 | +1.87 | 40.49 | +1.22 | 35.04 | -0.83 | 44.81 | +2.59 |
| gen_albumentations_weather | Generative | 47.14 | +2.59 | 40.61 | +1.34 | 35.21 | -0.66 | 43.08 | +0.86 |
| gen_cyclediffusion | Generative | 46.92 | +2.38 | 40.67 | +1.40 | 33.25 | -2.62 | 44.77 | +2.55 |
| gen_Attribute_Hallucination | Generative | 46.75 | +2.21 | 40.62 | +1.35 | 33.23 | -2.64 | 44.96 | +2.74 |
| gen_SUSTechGAN | Generative | 46.36 | +1.81 | 40.65 | +1.38 | 33.17 | -2.70 | 44.76 | +2.54 |
| gen_automold | Generative | 47.13 | +2.59 | 40.56 | +1.28 | 35.23 | -0.64 | 41.89 | -0.33 |
| gen_flux_kontext | Generative | 46.32 | +1.77 | 39.68 | +0.40 | 35.21 | -0.66 | 43.23 | +1.00 |
| gen_step1x_new | Generative | 46.37 | +1.83 | 39.75 | +0.48 | 35.16 | -0.72 | 43.14 | +0.92 |
| gen_UniControl | Generative | 46.46 | +1.92 | 39.89 | +0.61 | 35.17 | -0.70 | 42.63 | +0.40 |
| gen_Qwen_Image_Edit | Generative | 46.11 | +1.57 | 39.81 | +0.54 | 35.24 | -0.63 | 42.71 | +0.49 |
| std_autoaugment | Standard Aug | 46.42 | +1.88 | 39.69 | +0.41 | 35.25 | -0.62 | 42.03 | -0.19 |
| gen_Img2Img | Generative | 46.53 | +1.99 | 39.75 | +0.48 | 35.23 | -0.65 | 41.56 | -0.66 |
| gen_CUT | Generative | 46.04 | +1.50 | 39.82 | +0.55 | 35.24 | -0.63 | 41.62 | -0.60 |
| baseline | Baseline | 44.54 | +0.00 | 39.27 | +0.00 | 35.87 | +0.00 | 42.22 | +0.00 |
| gen_augmenters | Generative | 46.44 | +1.90 | 37.04 | -2.24 | 35.23 | -0.64 | 43.06 | +0.84 |
| std_cutmix | Standard Aug | 47.30 | +2.76 | 37.02 | -2.25 | 35.19 | -0.68 | 42.23 | +0.01 |
| gen_cycleGAN | Generative | 46.86 | +2.32 | 36.92 | -2.36 | 35.13 | -0.75 | 42.48 | +0.26 |
| std_mixup | Standard Aug | 45.75 | +1.21 | 38.83 | -0.45 | 35.35 | -0.53 | 41.04 | -1.19 |
| std_randaugment | Standard Aug | 44.89 | +0.35 | 37.86 | -1.41 | 35.17 | -0.70 | 42.56 | +0.34 |
| gen_stargan_v2 | Generative | - | - | - | - | 33.32 | -2.55 | - | - |
| gen_CNetSeg | Generative | - | - | - | - | 33.05 | -2.82 | - | - |
---

## Per-Domain Breakdown

mIoU performance on each weather domain. Normal = clear_day. Adverse = cloudy, dawn_dusk, foggy, night, rainy, snowy.

| Strategy | Type | clear_day | cloudy | dawn_dusk | foggy | night | rainy | snowy | Normal Avg | Adverse Avg | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Baseline | 41.34 | 41.19 | 36.38 | 46.56 | 32.56 | 38.94 | 40.28 | 41.34 | 39.32 | 2.02 |
| gen_Attribute_Hallucination | Generative | 38.85 | 38.35 | 36.63 | 41.93 | 31.53 | 36.87 | 38.27 | 38.85 | 37.26 | 1.59 |
| gen_CNetSeg | Generative | 32.49 | 32.94 | 36.55 | 33.67 | 27.57 | 31.16 | 31.02 | 32.49 | 32.15 | 0.34 |
| gen_CUT | Generative | 41.06 | 40.59 | 36.97 | 45.14 | 33.15 | 39.12 | 40.61 | 41.06 | 39.26 | 1.80 |
| gen_IP2P | Generative | 39.16 | 38.62 | 37.89 | 42.85 | 32.03 | 37.09 | 38.69 | 39.16 | 37.86 | 1.30 |
| gen_Img2Img | Generative | 41.25 | 40.82 | 36.95 | 44.81 | 33.20 | 38.94 | 40.71 | 41.25 | 39.24 | 2.01 |
| gen_LANIT | Generative | 42.23 | 41.36 | 37.42 | 46.62 | 32.60 | 39.72 | 41.64 | 42.23 | 39.89 | 2.33 |
| gen_Qwen_Image_Edit | Generative | 41.44 | 41.00 | 37.33 | 45.23 | 32.95 | 38.99 | 40.25 | 41.44 | 39.29 | 2.14 |
| gen_SUSTechGAN | Generative | 38.83 | 37.95 | 36.24 | 42.10 | 30.80 | 36.60 | 38.00 | 38.83 | 36.95 | 1.88 |
| gen_UniControl | Generative | 41.45 | 40.97 | 37.11 | 45.77 | 33.29 | 39.29 | 41.33 | 41.45 | 39.63 | 1.82 |
| gen_VisualCloze | Generative | 39.96 | 38.87 | 37.47 | 42.43 | 32.36 | 37.82 | 38.68 | 39.96 | 37.94 | 2.02 |
| gen_albumentations_weather | Generative | 40.29 | 39.00 | 35.74 | 44.28 | 32.74 | 38.05 | 37.81 | 40.29 | 37.94 | 2.35 |
| gen_augmenters | Generative | 40.80 | 40.23 | 36.17 | 44.64 | 32.25 | 38.90 | 39.87 | 40.80 | 38.68 | 2.13 |
| gen_automold | Generative | 39.71 | 38.35 | 35.98 | 43.08 | 31.92 | 37.00 | 37.72 | 39.71 | 37.34 | 2.37 |
| gen_cycleGAN | Generative | 40.71 | 40.61 | 36.33 | 44.87 | 32.09 | 38.77 | 40.35 | 40.71 | 38.84 | 1.88 |
| gen_cyclediffusion | Generative | 38.90 | 38.34 | 36.83 | 41.94 | 31.17 | 36.91 | 37.70 | 38.90 | 37.15 | 1.75 |
| gen_flux_kontext | Generative | 41.50 | 41.07 | 36.57 | 44.27 | 32.54 | 39.43 | 40.65 | 41.50 | 39.09 | 2.41 |
| gen_stargan_v2 | Generative | 32.91 | 32.87 | 35.67 | 34.27 | 28.09 | 31.36 | 30.93 | 32.91 | 32.20 | 0.71 |
| gen_step1x_new | Generative | 41.57 | 41.27 | 37.43 | 45.81 | 33.02 | 39.35 | 40.93 | 41.57 | 39.64 | 1.93 |
| gen_step1x_v1p2 | Generative | 39.80 | 38.97 | 37.54 | 44.13 | 32.60 | 37.39 | 37.72 | 39.80 | 38.06 | 1.74 |
| std_autoaugment | Standard Aug | 41.20 | 40.71 | 37.14 | 45.00 | 32.99 | 39.21 | 40.65 | 41.20 | 39.28 | 1.92 |
| std_cutmix | Standard Aug | 40.76 | 40.27 | 36.58 | 44.25 | 32.49 | 38.90 | 40.15 | 40.76 | 38.77 | 1.99 |
| std_mixup | Standard Aug | 40.20 | 39.86 | 36.04 | 44.48 | 32.25 | 38.23 | 39.21 | 40.20 | 38.34 | 1.86 |
| std_randaugment | Standard Aug | 39.89 | 39.14 | 36.17 | 43.94 | 31.95 | 37.89 | 38.01 | 39.89 | 37.85 | 2.04 |
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
| gen_CUT | Generative | - | - | - | - | 42.79 | -1.51 | 35.74 | -0.48 | 41.75 | -1.86 | - | - | 42.45 | -1.41 |
| gen_augmenters | Generative | - | - | - | - | 41.77 | -2.52 | 35.62 | -0.59 | 42.01 | -1.59 | - | - | 42.36 | -1.50 |
| std_cutmix | Standard Aug | - | - | - | - | 41.74 | -2.56 | 35.63 | -0.59 | 41.88 | -1.72 | - | - | 42.49 | -1.36 |
| gen_cycleGAN | Generative | - | - | - | - | 41.82 | -2.48 | 35.63 | -0.59 | 41.74 | -1.86 | - | - | 42.18 | -1.67 |
| std_randaugment | Standard Aug | - | - | - | - | 42.76 | -1.54 | 35.66 | -0.56 | 42.01 | -1.60 | - | - | 40.11 | -3.74 |
| std_mixup | Standard Aug | - | - | - | - | 39.74 | -4.56 | 35.81 | -0.40 | 41.85 | -1.75 | - | - | 42.28 | -1.58 |
| gen_albumentations_weather | Generative | - | - | - | - | 43.88 | -0.42 | 33.02 | -3.20 | 41.87 | -1.73 | - | - | 39.83 | -4.03 |
| gen_automold | Generative | - | - | - | - | 40.77 | -3.53 | 32.76 | -3.46 | 41.77 | -1.83 | - | - | 40.47 | -3.39 |
| gen_VisualCloze | Generative | - | - | - | - | 41.13 | -3.16 | 29.42 | -6.80 | 41.90 | -1.70 | - | - | 40.22 | -3.64 |
| gen_step1x_v1p2 | Generative | - | - | - | - | 40.90 | -3.40 | 29.23 | -6.99 | 41.69 | -1.91 | - | - | 39.91 | -3.95 |
| gen_IP2P | Generative | - | - | - | - | 40.88 | -3.42 | 29.38 | -6.84 | 41.98 | -1.62 | - | - | 35.46 | -8.40 |
| gen_Attribute_Hallucination | Generative | - | - | - | - | - | - | 29.32 | -6.90 | 41.77 | -1.83 | - | - | 35.60 | -8.26 |
| gen_cyclediffusion | Generative | - | - | - | - | - | - | 29.17 | -7.05 | 41.85 | -1.76 | - | - | 35.56 | -8.30 |
| gen_SUSTechGAN | Generative | - | - | - | - | - | - | 29.24 | -6.97 | 41.67 | -1.93 | - | - | 35.37 | -8.49 |
| gen_stargan_v2 | Generative | - | - | - | - | - | - | 29.47 | -6.75 | 35.02 | -8.58 | - | - | 35.48 | -8.38 |
| gen_CNetSeg | Generative | - | - | - | - | - | - | 29.29 | -6.93 | 34.55 | -9.05 | - | - | 35.30 | -8.56 |
