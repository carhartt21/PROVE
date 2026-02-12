# Stage 2 Strategy Leaderboard (by mIoU)

**Stage 2**: All Domains Training — domain-inclusive evaluation

**Metric**: mIoU (Mean Intersection over Union)

**Last Updated**: 2026-02-12 13:41
**Baseline mIoU**: 40.80%
**Total Results**: 195 test results from 22 strategies

---

## Overall Strategy Ranking

Sorted by mIoU. Gain = improvement over baseline. Domain Gap = Normal mIoU - Adverse mIoU (lower is better).

| Strategy | Type | mIoU | Std | Gain | Normal mIoU | Adverse mIoU | Domain Gap | Num Tests |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_augmenters | Generative | 42.05 | 0.0 | 1.26 | 42.37 | 40.8 | 1.57 | 1 |
| gen_IP2P | Generative | 41.98 | 5.51 | 1.18 | 42.49 | 40.15 | 2.34 | 4 |
| gen_VisualCloze | Generative | 41.9 | 5.32 | 1.11 | 42.37 | 39.58 | 2.79 | 4 |
| gen_albumentations_weather | Generative | 41.87 | 5.27 | 1.07 | 42.1 | 40.14 | 1.96 | 4 |
| gen_cyclediffusion | Generative | 41.85 | 5.24 | 1.05 | 42.46 | 39.99 | 2.47 | 4 |
| gen_Attribute_Hallucination | Generative | 41.77 | 5.33 | 0.98 | 42.33 | 39.95 | 2.38 | 4 |
| gen_automold | Generative | 41.77 | 5.27 | 0.97 | 42.38 | 39.67 | 2.71 | 4 |
| gen_LANIT | Generative | 41.76 | 5.33 | 0.96 | 42.23 | 39.89 | 2.33 | 4 |
| gen_CUT | Generative | 41.75 | 5.19 | 0.95 | 42.22 | 40.1 | 2.11 | 4 |
| gen_step1x_v1p2 | Generative | 41.69 | 5.13 | 0.89 | 42.2 | 39.9 | 2.3 | 4 |
| gen_SUSTechGAN | Generative | 41.67 | 5.11 | 0.87 | 42.33 | 39.67 | 2.66 | 4 |
| gen_step1x_new | Generative | 41.11 | 5.49 | 0.31 | 41.57 | 39.64 | 1.93 | 16 |
| gen_UniControl | Generative | 41.03 | 5.03 | 0.23 | 41.34 | 39.71 | 1.63 | 9 |
| baseline | Baseline | 40.8 | 5.62 | 0.0 | 41.3 | 39.24 | 2.06 | 37 |
| gen_Img2Img | Generative | 40.77 | 5.41 | -0.03 | 41.25 | 39.24 | 2.01 | 16 |
| gen_Qwen_Image_Edit | Generative | 40.7 | 5.44 | -0.1 | 41.15 | 39.11 | 2.04 | 15 |
| std_autoaugment | Standard Aug | 40.53 | 5.68 | -0.27 | 40.97 | 38.88 | 2.09 | 14 |
| std_cutmix | Standard Aug | 40.0 | 5.78 | -0.8 | 40.39 | 38.13 | 2.26 | 12 |
| std_mixup | Standard Aug | 39.58 | 5.74 | -1.22 | 39.85 | 37.99 | 1.86 | 11 |
| std_randaugment | Standard Aug | 38.83 | 5.99 | -1.97 | 39.16 | 36.85 | 2.31 | 8 |
| gen_flux_kontext | Generative | 38.77 | 5.92 | -2.03 | 39.16 | 36.47 | 2.69 | 8 |
| gen_cycleGAN | Generative | 38.69 | 5.88 | -2.11 | 39.08 | 36.77 | 2.31 | 8 |
---

## Per-Dataset Breakdown

mIoU performance on each dataset.

| Strategy | Type | bdd10k | bdd10k_gain | iddaw | iddaw_gain | mapillaryvistas | mapillaryvistas_gain | outside15k | outside15k_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_augmenters | Generative | 42.05 | -2.49 | - | - | - | - | - | - |
| gen_IP2P | Generative | 47.71 | +3.17 | 40.53 | +1.26 | 35.00 | -0.23 | 44.69 | +2.52 |
| gen_VisualCloze | Generative | 47.18 | +2.64 | 40.71 | +1.44 | 35.00 | -0.23 | 44.72 | +2.55 |
| gen_albumentations_weather | Generative | 47.14 | +2.59 | 40.61 | +1.34 | 35.07 | -0.16 | 44.67 | +2.50 |
| gen_cyclediffusion | Generative | 46.92 | +2.38 | 40.67 | +1.40 | 35.02 | -0.21 | 44.77 | +2.61 |
| gen_Attribute_Hallucination | Generative | 46.75 | +2.21 | 40.62 | +1.35 | 34.77 | -0.46 | 44.96 | +2.80 |
| gen_automold | Generative | 47.13 | +2.59 | 40.56 | +1.28 | 34.99 | -0.24 | 44.41 | +2.24 |
| gen_LANIT | Generative | 46.62 | +2.08 | 40.64 | +1.37 | 34.72 | -0.51 | 45.06 | +2.90 |
| gen_CUT | Generative | 46.81 | +2.27 | 40.50 | +1.23 | 35.02 | -0.21 | 44.65 | +2.48 |
| gen_step1x_v1p2 | Generative | 46.41 | +1.87 | 40.49 | +1.22 | 34.97 | -0.26 | 44.88 | +2.72 |
| gen_SUSTechGAN | Generative | 46.36 | +1.81 | 40.65 | +1.38 | 34.91 | -0.32 | 44.76 | +2.60 |
| gen_UniControl | Generative | 45.67 | +1.12 | 38.87 | -0.41 | 35.35 | +0.12 | 45.01 | +2.84 |
| gen_step1x_new | Generative | 46.37 | +1.83 | 39.75 | +0.48 | 35.16 | -0.07 | 43.14 | +0.97 |
| gen_Qwen_Image_Edit | Generative | 46.11 | +1.57 | 39.81 | +0.54 | 35.24 | +0.01 | 41.93 | -0.24 |
| gen_Img2Img | Generative | 46.53 | +1.99 | 39.75 | +0.48 | 35.23 | -0.00 | 41.56 | -0.60 |
| baseline | Baseline | 44.54 | +0.00 | 39.27 | +0.00 | 35.23 | +0.00 | 42.17 | +0.00 |
| std_autoaugment | Standard Aug | 45.94 | +1.39 | 39.37 | +0.09 | 33.22 | -2.01 | 42.18 | +0.01 |
| std_cutmix | Standard Aug | 45.85 | +1.31 | 38.85 | -0.42 | 33.42 | -1.81 | 41.89 | -0.28 |
| std_mixup | Standard Aug | 45.75 | +1.21 | 38.83 | -0.45 | 33.43 | -1.80 | 40.69 | -1.47 |
| std_randaugment | Standard Aug | 44.89 | +0.35 | 37.86 | -1.41 | 32.23 | -3.00 | 40.36 | -1.81 |
| gen_flux_kontext | Generative | 44.57 | +0.03 | 37.70 | -1.57 | 32.11 | -3.12 | 40.70 | -1.46 |
| gen_cycleGAN | Generative | 44.57 | +0.02 | 37.68 | -1.60 | 32.13 | -3.10 | 40.38 | -1.79 |
---

## Per-Domain Breakdown

mIoU performance on each weather domain. Normal = clear_day. Adverse = cloudy, dawn_dusk, foggy, night, rainy, snowy.

| Strategy | Type | clear_day | cloudy | dawn_dusk | foggy | night | rainy | snowy | Normal Avg | Adverse Avg | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Baseline | 41.30 | 41.21 | 36.17 | 46.61 | 32.27 | 38.90 | 40.29 | 41.30 | 39.24 | 2.06 |
| gen_Attribute_Hallucination | Generative | 42.33 | 41.37 | 36.55 | 46.29 | 33.52 | 39.84 | 42.14 | 42.33 | 39.95 | 2.38 |
| gen_CUT | Generative | 42.22 | 41.45 | 37.40 | 46.86 | 33.38 | 39.80 | 41.72 | 42.22 | 40.10 | 2.11 |
| gen_IP2P | Generative | 42.49 | 41.42 | 36.65 | 48.08 | 32.62 | 39.95 | 42.16 | 42.49 | 40.15 | 2.34 |
| gen_Img2Img | Generative | 41.25 | 40.82 | 36.95 | 44.81 | 33.20 | 38.94 | 40.71 | 41.25 | 39.24 | 2.01 |
| gen_LANIT | Generative | 42.23 | 41.36 | 37.42 | 46.62 | 32.60 | 39.72 | 41.64 | 42.23 | 39.89 | 2.33 |
| gen_Qwen_Image_Edit | Generative | 41.15 | 40.83 | 37.32 | 44.79 | 32.74 | 38.72 | 40.28 | 41.15 | 39.11 | 2.04 |
| gen_SUSTechGAN | Generative | 42.33 | 40.99 | 36.23 | 46.65 | 32.56 | 39.60 | 42.01 | 42.33 | 39.67 | 2.66 |
| gen_UniControl | Generative | 41.34 | 41.62 | 36.93 | 45.79 | 32.22 | 39.61 | 42.07 | 41.34 | 39.71 | 1.63 |
| gen_VisualCloze | Generative | 42.37 | 41.16 | 36.64 | 44.96 | 32.78 | 40.22 | 41.74 | 42.37 | 39.58 | 2.79 |
| gen_albumentations_weather | Generative | 42.10 | 41.64 | 36.40 | 47.53 | 33.63 | 39.92 | 41.73 | 42.10 | 40.14 | 1.96 |
| gen_augmenters | Generative | 42.37 | 45.71 | 37.54 | 45.40 | 28.06 | 41.30 | 46.82 | 42.37 | 40.80 | 1.57 |
| gen_automold | Generative | 42.38 | 41.41 | 36.63 | 46.19 | 32.60 | 39.26 | 41.92 | 42.38 | 39.67 | 2.71 |
| gen_cycleGAN | Generative | 39.08 | 38.85 | 34.57 | 42.61 | 30.02 | 36.90 | 37.67 | 39.08 | 36.77 | 2.31 |
| gen_cyclediffusion | Generative | 42.46 | 41.35 | 37.45 | 46.21 | 33.30 | 40.04 | 41.57 | 42.46 | 39.99 | 2.47 |
| gen_flux_kontext | Generative | 39.16 | 38.82 | 34.42 | 41.38 | 29.63 | 36.84 | 37.71 | 39.16 | 36.47 | 2.69 |
| gen_step1x_new | Generative | 41.57 | 41.27 | 37.43 | 45.81 | 33.02 | 39.35 | 40.93 | 41.57 | 39.64 | 1.93 |
| gen_step1x_v1p2 | Generative | 42.20 | 41.51 | 37.41 | 47.21 | 32.90 | 39.58 | 40.78 | 42.20 | 39.90 | 2.30 |
| std_autoaugment | Standard Aug | 40.97 | 40.74 | 36.34 | 44.97 | 32.18 | 38.90 | 40.16 | 40.97 | 38.88 | 2.09 |
| std_cutmix | Standard Aug | 40.39 | 39.72 | 35.95 | 43.83 | 31.68 | 38.42 | 39.19 | 40.39 | 38.13 | 2.26 |
| std_mixup | Standard Aug | 39.85 | 39.76 | 35.92 | 43.73 | 31.01 | 38.17 | 39.35 | 39.85 | 37.99 | 1.86 |
| std_randaugment | Standard Aug | 39.16 | 38.89 | 34.64 | 42.65 | 29.85 | 37.24 | 37.80 | 39.16 | 36.85 | 2.31 |
---

## Per-Model Breakdown

mIoU performance on each model architecture. Gain columns show improvement over baseline per model.

| Strategy | Type | deeplabv3plus_r50 | deeplabv3plus_r50_gain | hrnet_hr48 | hrnet_hr48_gain | mask2former_swin-b | mask2former_swin-b_gain | pspnet_r50 | pspnet_r50_gain | segformer_mit-b3 | segformer_mit-b3_gain | segformer_mit-b5 | segformer_mit-b5_gain | segnext_mscan-b | segnext_mscan-b_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Baseline | 37.57 | +0.00 | 38.26 | +0.00 | 46.76 | +0.00 | 36.22 | +0.00 | 43.60 | +0.00 | 48.14 | +0.00 | 43.86 | +0.00 |
| gen_augmenters | Generative | - | - | - | - | - | - | 42.05 | +5.84 | - | - | - | - | - | - |
| gen_IP2P | Generative | - | - | - | - | - | - | - | - | 41.98 | -1.62 | - | - | - | - |
| gen_VisualCloze | Generative | - | - | - | - | - | - | - | - | 41.90 | -1.70 | - | - | - | - |
| gen_albumentations_weather | Generative | - | - | - | - | - | - | - | - | 41.87 | -1.73 | - | - | - | - |
| gen_cyclediffusion | Generative | - | - | - | - | - | - | - | - | 41.85 | -1.76 | - | - | - | - |
| gen_automold | Generative | - | - | - | - | - | - | - | - | 41.77 | -1.83 | - | - | - | - |
| gen_Attribute_Hallucination | Generative | - | - | - | - | - | - | - | - | 41.77 | -1.83 | - | - | - | - |
| gen_LANIT | Generative | - | - | - | - | - | - | - | - | 41.76 | -1.84 | - | - | - | - |
| gen_CUT | Generative | - | - | - | - | - | - | - | - | 41.75 | -1.86 | - | - | - | - |
| gen_step1x_v1p2 | Generative | - | - | - | - | - | - | - | - | 41.69 | -1.91 | - | - | - | - |
| gen_SUSTechGAN | Generative | - | - | - | - | - | - | - | - | 41.67 | -1.93 | - | - | - | - |
| gen_step1x_new | Generative | - | - | - | - | 44.53 | -2.23 | 35.92 | -0.30 | 41.77 | -1.83 | - | - | 42.20 | -1.66 |
| std_autoaugment | Standard Aug | - | - | - | - | 43.80 | -2.96 | 35.56 | -0.66 | 41.79 | -1.81 | - | - | 42.60 | -1.26 |
| gen_Qwen_Image_Edit | Generative | - | - | - | - | 43.69 | -3.07 | 35.83 | -0.39 | 41.74 | -1.87 | - | - | 42.28 | -1.58 |
| gen_Img2Img | Generative | - | - | - | - | 43.34 | -3.42 | 35.72 | -0.50 | 41.78 | -1.82 | - | - | 42.23 | -1.63 |
| gen_UniControl | Generative | - | - | - | - | - | - | 38.58 | +2.36 | 41.77 | -1.84 | - | - | 41.69 | -2.16 |
| std_cutmix | Standard Aug | - | - | - | - | - | - | 35.63 | -0.59 | 41.88 | -1.72 | - | - | 42.49 | -1.36 |
| std_mixup | Standard Aug | - | - | - | - | - | - | 35.81 | -0.40 | 41.85 | -1.75 | - | - | 41.58 | -2.28 |
| std_randaugment | Standard Aug | - | - | - | - | - | - | 35.66 | -0.56 | 42.01 | -1.60 | - | - | - | - |
| gen_flux_kontext | Generative | - | - | - | - | - | - | 35.70 | -0.52 | 41.84 | -1.76 | - | - | - | - |
| gen_cycleGAN | Generative | - | - | - | - | - | - | 35.63 | -0.59 | 41.74 | -1.86 | - | - | - | - |
