# Stage 2 Detailed Per-Dataset and Per-Domain Analysis

## Per-Dataset mIoU by Strategy

| Strategy | Type | bdd10k | Δbdd10k | idd-aw | Δidd-aw | mapillaryvistas | Δmapillaryvistas | outside15k | Δoutside15k | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | Baseline | 45.8 | +0.0 | 44.5 | +0.0 | 36.1 | +0.0 | 39.0 | +0.0 | +0.00 |
| gen_Attribute_Hallucination | Generative | 46.8 | +1.0 | 44.8 | +0.3 | 36.0 | -0.1 | 38.9 | -0.2 | +0.26 |
| gen_CNetSeg | Generative | 46.7 | +0.9 | 44.6 | +0.1 | 35.7 | -0.4 | 39.7 | +0.7 | +0.34 |
| gen_CUT | Generative | 46.4 | +0.7 | 44.9 | +0.4 | 35.8 | -0.3 | 39.2 | +0.2 | +0.23 |
| gen_IP2P | Generative | 46.1 | +0.3 | 44.7 | +0.2 | 35.3 | -0.8 | 39.4 | +0.3 | +0.01 |
| gen_Img2Img | Generative | 46.0 | +0.3 | 44.6 | +0.1 | 35.9 | -0.3 | 39.4 | +0.4 | +0.13 |
| gen_LANIT | Generative | 46.6 | +0.8 | 44.6 | +0.2 | 35.8 | -0.3 | 39.2 | +0.2 | +0.19 |
| gen_Qwen_Image_Edit | Generative | 46.7 | +0.9 | 44.6 | +0.1 | 35.9 | -0.2 | 37.8 | -1.2 | -0.09 |
| gen_SUSTechGAN | Generative | 45.9 | +0.2 | 44.7 | +0.2 | 36.1 | -0.1 | 39.6 | +0.5 | +0.22 |
| gen_TSIT | Generative | 46.4 | +0.6 | 44.6 | +0.1 | 36.1 | -0.0 | 39.0 | -0.0 | +0.18 |
| gen_UniControl | Generative | 46.7 | +0.9 | 44.6 | +0.1 | 36.0 | -0.1 | 39.4 | +0.4 | +0.35 |
| gen_VisualCloze | Generative | 46.1 | +0.3 | 44.7 | +0.2 | 36.2 | +0.1 | 39.6 | +0.6 | +0.32 |
| gen_Weather_Effect_Generator | Generative | 46.5 | +0.8 | 44.2 | -0.3 | 33.7 | -2.4 | 38.4 | -0.7 | -0.64 |
| gen_albumentations_weather | Generative | 46.1 | +0.3 | 44.7 | +0.2 | 35.6 | -0.5 | 39.6 | +0.5 | +0.13 |
| gen_augmenters | Generative | 46.6 | +0.8 | 44.7 | +0.2 | 35.7 | -0.4 | 39.3 | +0.3 | +0.23 |
| gen_automold | Generative | 46.5 | +0.8 | 44.7 | +0.3 | 35.5 | -0.6 | 38.7 | -0.3 | +0.03 |
| gen_cycleGAN | Generative | 46.9 | +1.2 | 44.7 | +0.2 | 36.0 | -0.1 | 38.9 | -0.1 | +0.29 |
| gen_cyclediffusion | Generative | 46.5 | +0.7 | 44.5 | -0.0 | 35.7 | -0.4 | 39.7 | +0.7 | +0.24 |
| gen_flux_kontext | Generative | 45.6 | -0.2 | 44.8 | +0.4 | 36.2 | +0.1 | 39.0 | +0.0 | +0.07 |
| gen_stargan_v2 | Generative | 46.1 | +0.4 | 44.7 | +0.2 | 36.1 | -0.0 | 40.0 | +1.0 | +0.38 |
| gen_step1x_new | Generative | 45.5 | -0.3 | 44.3 | -0.2 | 35.8 | -0.4 | 39.4 | +0.4 | -0.11 |
| gen_step1x_v1p2 | Generative | 46.3 | +0.5 | 44.7 | +0.2 | 36.0 | -0.1 | 39.0 | -0.0 | +0.15 |
| photometric_distort | Augmentation | 46.5 | +0.7 | 44.5 | -0.0 | 35.7 | -0.4 | 38.6 | -0.4 | -0.03 |
| std_autoaugment | Standard Aug | 46.9 | +1.1 | 44.6 | +0.1 | 34.9 | -1.2 | 39.1 | +0.1 | +0.03 |
| std_cutmix | Standard Aug | 45.2 | -0.6 | 44.4 | -0.1 | 34.8 | -1.3 | 38.8 | -0.2 | -0.54 |
| std_mixup | Standard Aug | 46.0 | +0.3 | 44.2 | -0.3 | 34.8 | -1.3 | 39.5 | +0.5 | -0.20 |
| std_randaugment | Standard Aug | 46.3 | +0.5 | 44.6 | +0.1 | 34.0 | -2.1 | 39.7 | +0.7 | -0.21 |

## Per-Domain mIoU by Strategy

| Strategy | Type | clear_day | cloudy | dawn_dusk | foggy | night | rainy | snowy | Normal Avg | Adverse Avg | Gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | Baseline | 41.85 | 42.05 | 37.51 | 54.07 | 33.37 | 39.45 | 40.66 | 41.85 | 41.18 | 0.66 |
| gen_Attribute_Hallucination | Generative | 42.25 | 42.02 | 37.35 | 51.98 | 33.57 | 39.28 | 41.04 | 42.25 | 40.87 | 1.37 |
| gen_CNetSeg | Generative | 42.26 | 41.97 | 37.71 | 51.69 | 33.71 | 39.67 | 41.06 | 42.26 | 40.97 | 1.29 |
| gen_CUT | Generative | 42.15 | 42.10 | 38.14 | 52.10 | 33.28 | 39.64 | 40.91 | 42.15 | 41.03 | 1.13 |
| gen_IP2P | Generative | 41.97 | 41.92 | 37.55 | 52.29 | 33.39 | 39.09 | 40.86 | 41.97 | 40.85 | 1.12 |
| gen_Img2Img | Generative | 42.08 | 41.94 | 37.80 | 51.90 | 33.12 | 39.15 | 40.98 | 42.08 | 40.81 | 1.26 |
| gen_LANIT | Generative | 42.20 | 42.14 | 37.65 | 52.24 | 32.75 | 39.06 | 40.77 | 42.20 | 40.77 | 1.43 |
| gen_Qwen_Image_Edit | Generative | 41.79 | 42.12 | 37.80 | 51.60 | 33.38 | 39.37 | 40.31 | 41.79 | 40.76 | 1.03 |
| gen_SUSTechGAN | Generative | 42.18 | 41.92 | 37.72 | 51.60 | 33.31 | 39.23 | 41.10 | 42.18 | 40.81 | 1.36 |
| gen_TSIT | Generative | 42.14 | 42.11 | 37.50 | 51.92 | 33.22 | 38.93 | 40.72 | 42.14 | 40.73 | 1.40 |
| gen_UniControl | Generative | 42.37 | 42.16 | 37.57 | 52.28 | 33.01 | 39.44 | 41.25 | 42.37 | 40.95 | 1.41 |
| gen_VisualCloze | Generative | 42.26 | 42.08 | 37.62 | 52.38 | 33.77 | 39.38 | 41.03 | 42.26 | 41.04 | 1.21 |
| gen_Weather_Effect_Generator | Generative | 41.36 | 41.16 | 36.93 | 50.90 | 32.35 | 38.68 | 39.61 | 41.36 | 39.94 | 1.42 |
| gen_albumentations_weather | Generative | 42.16 | 41.75 | 37.52 | 51.65 | 33.60 | 39.25 | 40.77 | 42.16 | 40.76 | 1.41 |
| gen_augmenters | Generative | 42.09 | 41.95 | 37.44 | 52.42 | 33.57 | 39.55 | 41.13 | 42.09 | 41.01 | 1.08 |
| gen_automold | Generative | 42.04 | 42.02 | 37.25 | 51.89 | 33.10 | 38.71 | 40.86 | 42.04 | 40.64 | 1.40 |
| gen_cycleGAN | Generative | 42.26 | 41.71 | 37.84 | 51.51 | 33.61 | 39.35 | 40.83 | 42.26 | 40.81 | 1.45 |
| gen_cyclediffusion | Generative | 42.33 | 42.15 | 38.12 | 51.91 | 33.39 | 39.49 | 40.10 | 42.33 | 40.86 | 1.47 |
| gen_flux_kontext | Generative | 42.09 | 42.06 | 37.80 | 52.29 | 33.46 | 39.10 | 40.75 | 42.09 | 40.91 | 1.18 |
| gen_stargan_v2 | Generative | 42.37 | 42.12 | 38.00 | 52.78 | 33.48 | 39.40 | 41.15 | 42.37 | 41.16 | 1.21 |
| gen_step1x_new | Generative | 41.82 | 41.55 | 37.37 | 51.39 | 32.74 | 38.98 | 40.45 | 41.82 | 40.41 | 1.41 |
| gen_step1x_v1p2 | Generative | 42.08 | 41.91 | 37.77 | 51.26 | 33.26 | 39.38 | 40.84 | 42.08 | 40.74 | 1.35 |
| photometric_distort | Augmentation | 41.92 | 41.80 | 38.06 | 51.62 | 33.81 | 38.88 | 40.36 | 41.92 | 40.75 | 1.17 |
| std_autoaugment | Standard Aug | 41.97 | 41.55 | 38.22 | 51.44 | 33.36 | 39.55 | 41.07 | 41.97 | 40.87 | 1.10 |
| std_cutmix | Standard Aug | 41.62 | 41.12 | 36.98 | 50.36 | 33.20 | 38.58 | 39.51 | 41.62 | 39.96 | 1.66 |
| std_mixup | Standard Aug | 41.85 | 41.52 | 36.25 | 48.88 | 32.15 | 38.60 | 40.43 | 41.85 | 39.64 | 2.21 |
| std_randaugment | Standard Aug | 41.72 | 41.52 | 37.43 | 51.18 | 33.05 | 39.15 | 40.47 | 41.72 | 40.47 | 1.25 |