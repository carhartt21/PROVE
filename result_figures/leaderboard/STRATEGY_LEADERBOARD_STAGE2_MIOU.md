# Stage 2 Strategy Leaderboard (by mIoU)

**Stage 2**: All Domains Training — domain-inclusive evaluation

**Metric**: mIoU (Mean Intersection over Union)

**Last Updated**: 2026-02-12 21:13
**Baseline mIoU**: 40.80%
**Total Results**: 240 test results from 22 strategies

---

## Overall Strategy Ranking

Sorted by mIoU. Gain = improvement over baseline. Domain Gap = Normal mIoU - Adverse mIoU (lower is better).

| Strategy | Type | mIoU | Std | Gain | Normal mIoU | Adverse mIoU | Domain Gap | Num Tests |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_IP2P | Generative | 41.98 | 5.51 | 1.18 | 42.49 | 40.15 | 2.34 | 4 |
| gen_VisualCloze | Generative | 41.9 | 5.32 | 1.11 | 42.37 | 39.58 | 2.79 | 4 |
| gen_albumentations_weather | Generative | 41.87 | 5.27 | 1.07 | 42.1 | 40.14 | 1.96 | 4 |
| gen_cyclediffusion | Generative | 41.85 | 5.24 | 1.05 | 42.46 | 39.99 | 2.47 | 4 |
| gen_automold | Generative | 41.77 | 5.27 | 0.97 | 42.38 | 39.67 | 2.71 | 4 |
| gen_Attribute_Hallucination | Generative | 41.77 | 5.33 | 0.98 | 42.33 | 39.95 | 2.38 | 4 |
| gen_LANIT | Generative | 41.76 | 5.33 | 0.96 | 42.23 | 39.89 | 2.33 | 4 |
| gen_step1x_v1p2 | Generative | 41.69 | 5.13 | 0.89 | 42.2 | 39.9 | 2.3 | 4 |
| gen_SUSTechGAN | Generative | 41.67 | 5.11 | 0.87 | 42.33 | 39.67 | 2.66 | 4 |
| gen_step1x_new | Generative | 41.11 | 5.49 | 0.31 | 41.57 | 39.64 | 1.93 | 16 |
| gen_UniControl | Generative | 41.04 | 5.48 | 0.24 | 41.45 | 39.63 | 1.82 | 16 |
| gen_Qwen_Image_Edit | Generative | 40.97 | 5.37 | 0.17 | 41.44 | 39.29 | 2.14 | 16 |
| baseline | Baseline | 40.8 | 5.62 | 0.0 | 41.3 | 39.24 | 2.06 | 37 |
| gen_Img2Img | Generative | 40.77 | 5.41 | -0.03 | 41.25 | 39.24 | 2.01 | 16 |
| std_autoaugment | Standard Aug | 40.76 | 5.88 | -0.04 | 41.14 | 39.15 | 1.99 | 14 |
| gen_CUT | Generative | 40.68 | 5.26 | -0.12 | 41.06 | 39.26 | 1.8 | 16 |
| gen_flux_kontext | Generative | 40.62 | 5.97 | -0.18 | 41.0 | 38.43 | 2.57 | 13 |
| gen_augmenters | Generative | 40.44 | 6.07 | -0.36 | 40.8 | 38.68 | 2.13 | 16 |
| gen_cycleGAN | Generative | 39.85 | 5.78 | -0.95 | 40.23 | 38.07 | 2.17 | 12 |
| std_mixup | Standard Aug | 39.58 | 5.74 | -1.22 | 39.85 | 37.99 | 1.86 | 11 |
| std_cutmix | Standard Aug | 39.35 | 6.02 | -1.45 | 39.71 | 37.62 | 2.08 | 13 |
| std_randaugment | Standard Aug | 38.83 | 5.99 | -1.97 | 39.16 | 36.85 | 2.31 | 8 |
---

## Per-Dataset Breakdown

mIoU performance on each dataset.

| Strategy | Type | bdd10k | bdd10k_gain | iddaw | iddaw_gain | mapillaryvistas | mapillaryvistas_gain | outside15k | outside15k_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_IP2P | Generative | 47.71 | +3.17 | 40.53 | +1.26 | 35.00 | -0.23 | 44.69 | +2.52 |
| gen_VisualCloze | Generative | 47.18 | +2.64 | 40.71 | +1.44 | 35.00 | -0.23 | 44.72 | +2.55 |
| gen_albumentations_weather | Generative | 47.14 | +2.59 | 40.61 | +1.34 | 35.07 | -0.16 | 44.67 | +2.50 |
| gen_cyclediffusion | Generative | 46.92 | +2.38 | 40.67 | +1.40 | 35.02 | -0.21 | 44.77 | +2.61 |
| gen_Attribute_Hallucination | Generative | 46.75 | +2.21 | 40.62 | +1.35 | 34.77 | -0.46 | 44.96 | +2.80 |
| gen_automold | Generative | 47.13 | +2.59 | 40.56 | +1.28 | 34.99 | -0.24 | 44.41 | +2.24 |
| gen_LANIT | Generative | 46.62 | +2.08 | 40.64 | +1.37 | 34.72 | -0.51 | 45.06 | +2.90 |
| gen_step1x_v1p2 | Generative | 46.41 | +1.87 | 40.49 | +1.22 | 34.97 | -0.26 | 44.88 | +2.72 |
| gen_SUSTechGAN | Generative | 46.36 | +1.81 | 40.65 | +1.38 | 34.91 | -0.32 | 44.76 | +2.60 |
| gen_step1x_new | Generative | 46.37 | +1.83 | 39.75 | +0.48 | 35.16 | -0.07 | 43.14 | +0.97 |
| gen_UniControl | Generative | 46.46 | +1.92 | 39.89 | +0.61 | 35.17 | -0.06 | 42.63 | +0.46 |
| gen_Qwen_Image_Edit | Generative | 46.11 | +1.57 | 39.81 | +0.54 | 35.24 | +0.01 | 42.71 | +0.54 |
| gen_Img2Img | Generative | 46.53 | +1.99 | 39.75 | +0.48 | 35.23 | -0.00 | 41.56 | -0.60 |
| gen_CUT | Generative | 46.04 | +1.50 | 39.82 | +0.55 | 35.24 | +0.01 | 41.62 | -0.55 |
| gen_augmenters | Generative | 46.44 | +1.90 | 37.04 | -2.24 | 35.23 | +0.00 | 43.06 | +0.89 |
| std_autoaugment | Standard Aug | 46.42 | +1.88 | 39.69 | +0.41 | 33.22 | -2.01 | 42.18 | +0.01 |
| baseline | Baseline | 44.54 | +0.00 | 39.27 | +0.00 | 35.23 | +0.00 | 42.17 | +0.00 |
| gen_flux_kontext | Generative | 46.32 | +1.77 | 38.77 | -0.50 | 33.32 | -1.91 | 42.18 | +0.01 |
| gen_cycleGAN | Generative | 45.68 | +1.14 | 38.74 | -0.54 | 33.11 | -2.12 | 41.88 | -0.29 |
| std_mixup | Standard Aug | 45.75 | +1.21 | 38.83 | -0.45 | 33.43 | -1.80 | 40.69 | -1.47 |
| std_cutmix | Standard Aug | 45.85 | +1.31 | 37.02 | -2.25 | 33.42 | -1.81 | 41.89 | -0.28 |
| std_randaugment | Standard Aug | 44.89 | +0.35 | 37.86 | -1.41 | 32.23 | -3.00 | 40.36 | -1.81 |
---

## Per-Domain Breakdown

mIoU performance on each weather domain. Normal = clear_day. Adverse = cloudy, dawn_dusk, foggy, night, rainy, snowy.

| Strategy | Type | clear_day | cloudy | dawn_dusk | foggy | night | rainy | snowy | Normal Avg | Adverse Avg | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Baseline | 41.30 | 41.21 | 36.17 | 46.61 | 32.27 | 38.90 | 40.29 | 41.30 | 39.24 | 2.06 |
| gen_Attribute_Hallucination | Generative | 42.33 | 41.37 | 36.55 | 46.29 | 33.52 | 39.84 | 42.14 | 42.33 | 39.95 | 2.38 |
| gen_CUT | Generative | 41.06 | 40.59 | 36.97 | 45.14 | 33.15 | 39.12 | 40.61 | 41.06 | 39.26 | 1.80 |
| gen_IP2P | Generative | 42.49 | 41.42 | 36.65 | 48.08 | 32.62 | 39.95 | 42.16 | 42.49 | 40.15 | 2.34 |
| gen_Img2Img | Generative | 41.25 | 40.82 | 36.95 | 44.81 | 33.20 | 38.94 | 40.71 | 41.25 | 39.24 | 2.01 |
| gen_LANIT | Generative | 42.23 | 41.36 | 37.42 | 46.62 | 32.60 | 39.72 | 41.64 | 42.23 | 39.89 | 2.33 |
| gen_Qwen_Image_Edit | Generative | 41.44 | 41.00 | 37.33 | 45.23 | 32.95 | 38.99 | 40.25 | 41.44 | 39.29 | 2.14 |
| gen_SUSTechGAN | Generative | 42.33 | 40.99 | 36.23 | 46.65 | 32.56 | 39.60 | 42.01 | 42.33 | 39.67 | 2.66 |
| gen_UniControl | Generative | 41.45 | 40.97 | 37.11 | 45.77 | 33.29 | 39.29 | 41.33 | 41.45 | 39.63 | 1.82 |
| gen_VisualCloze | Generative | 42.37 | 41.16 | 36.64 | 44.96 | 32.78 | 40.22 | 41.74 | 42.37 | 39.58 | 2.79 |
| gen_albumentations_weather | Generative | 42.10 | 41.64 | 36.40 | 47.53 | 33.63 | 39.92 | 41.73 | 42.10 | 40.14 | 1.96 |
| gen_augmenters | Generative | 40.80 | 40.23 | 36.17 | 44.64 | 32.25 | 38.90 | 39.87 | 40.80 | 38.68 | 2.13 |
| gen_automold | Generative | 42.38 | 41.41 | 36.63 | 46.19 | 32.60 | 39.26 | 41.92 | 42.38 | 39.67 | 2.71 |
| gen_cycleGAN | Generative | 40.23 | 40.04 | 35.43 | 44.30 | 31.18 | 38.18 | 39.27 | 40.23 | 38.07 | 2.17 |
| gen_cyclediffusion | Generative | 42.46 | 41.35 | 37.45 | 46.21 | 33.30 | 40.04 | 41.57 | 42.46 | 39.99 | 2.47 |
| gen_flux_kontext | Generative | 41.00 | 40.82 | 35.62 | 43.67 | 31.00 | 39.05 | 40.43 | 41.00 | 38.43 | 2.57 |
| gen_step1x_new | Generative | 41.57 | 41.27 | 37.43 | 45.81 | 33.02 | 39.35 | 40.93 | 41.57 | 39.64 | 1.93 |
| gen_step1x_v1p2 | Generative | 42.20 | 41.51 | 37.41 | 47.21 | 32.90 | 39.58 | 40.78 | 42.20 | 39.90 | 2.30 |
| std_autoaugment | Standard Aug | 41.14 | 40.84 | 36.62 | 45.03 | 32.30 | 39.29 | 40.82 | 41.14 | 39.15 | 1.99 |
| std_cutmix | Standard Aug | 39.71 | 39.19 | 35.44 | 43.26 | 31.43 | 37.78 | 38.63 | 39.71 | 37.62 | 2.08 |
| std_mixup | Standard Aug | 39.85 | 39.76 | 35.92 | 43.73 | 31.01 | 38.17 | 39.35 | 39.85 | 37.99 | 1.86 |
| std_randaugment | Standard Aug | 39.16 | 38.89 | 34.64 | 42.65 | 29.85 | 37.24 | 37.80 | 39.16 | 36.85 | 2.31 |
---

## Per-Model Breakdown

mIoU performance on each model architecture. Gain columns show improvement over baseline per model.

| Strategy | Type | deeplabv3plus_r50 | deeplabv3plus_r50_gain | hrnet_hr48 | hrnet_hr48_gain | mask2former_swin-b | mask2former_swin-b_gain | pspnet_r50 | pspnet_r50_gain | segformer_mit-b3 | segformer_mit-b3_gain | segformer_mit-b5 | segformer_mit-b5_gain | segnext_mscan-b | segnext_mscan-b_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Baseline | 37.57 | +0.00 | 38.26 | +0.00 | 46.76 | +0.00 | 36.22 | +0.00 | 43.60 | +0.00 | 48.14 | +0.00 | 43.86 | +0.00 |
| gen_flux_kontext | Generative | - | - | - | - | 47.90 | +1.14 | 35.70 | -0.52 | 41.84 | -1.76 | - | - | 42.50 | -1.36 |
| gen_IP2P | Generative | - | - | - | - | - | - | - | - | 41.98 | -1.62 | - | - | - | - |
| gen_VisualCloze | Generative | - | - | - | - | - | - | - | - | 41.90 | -1.70 | - | - | - | - |
| gen_albumentations_weather | Generative | - | - | - | - | - | - | - | - | 41.87 | -1.73 | - | - | - | - |
| gen_cyclediffusion | Generative | - | - | - | - | - | - | - | - | 41.85 | -1.76 | - | - | - | - |
| gen_automold | Generative | - | - | - | - | - | - | - | - | 41.77 | -1.83 | - | - | - | - |
| gen_Attribute_Hallucination | Generative | - | - | - | - | - | - | - | - | 41.77 | -1.83 | - | - | - | - |
| gen_LANIT | Generative | - | - | - | - | - | - | - | - | 41.76 | -1.84 | - | - | - | - |
| gen_step1x_v1p2 | Generative | - | - | - | - | - | - | - | - | 41.69 | -1.91 | - | - | - | - |
| gen_SUSTechGAN | Generative | - | - | - | - | - | - | - | - | 41.67 | -1.93 | - | - | - | - |
| std_autoaugment | Standard Aug | - | - | - | - | 45.40 | -1.35 | 35.56 | -0.66 | 41.79 | -1.81 | - | - | 42.60 | -1.26 |
| gen_step1x_new | Generative | - | - | - | - | 44.53 | -2.23 | 35.92 | -0.30 | 41.77 | -1.83 | - | - | 42.20 | -1.66 |
| gen_UniControl | Generative | - | - | - | - | 44.28 | -2.47 | 35.73 | -0.49 | 41.77 | -1.84 | - | - | 42.37 | -1.49 |
| gen_Qwen_Image_Edit | Generative | - | - | - | - | 44.03 | -2.73 | 35.83 | -0.39 | 41.74 | -1.87 | - | - | 42.28 | -1.58 |
| gen_Img2Img | Generative | - | - | - | - | 43.34 | -3.42 | 35.72 | -0.50 | 41.78 | -1.82 | - | - | 42.23 | -1.63 |
| gen_CUT | Generative | - | - | - | - | 42.79 | -3.96 | 35.74 | -0.48 | 41.75 | -1.86 | - | - | 42.45 | -1.41 |
| gen_augmenters | Generative | - | - | - | - | 41.77 | -4.98 | 35.62 | -0.59 | 42.01 | -1.59 | - | - | 42.36 | -1.50 |
| gen_cycleGAN | Generative | - | - | - | - | - | - | 35.63 | -0.59 | 41.74 | -1.86 | - | - | 42.18 | -1.67 |
| std_mixup | Standard Aug | - | - | - | - | - | - | 35.81 | -0.40 | 41.85 | -1.75 | - | - | 41.58 | -2.28 |
| std_randaugment | Standard Aug | - | - | - | - | - | - | 35.66 | -0.56 | 42.01 | -1.60 | - | - | - | - |
| std_cutmix | Standard Aug | - | - | - | - | 31.53 | -15.22 | 35.63 | -0.59 | 41.88 | -1.72 | - | - | 42.49 | -1.36 |
