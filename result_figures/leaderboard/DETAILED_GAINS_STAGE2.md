# Stage 2 Detailed Per-Dataset and Per-Domain Analysis

## Per-Dataset mIoU by Strategy

| Strategy | Type | bdd10k | Δbdd10k | idd-aw | Δidd-aw | outside15k | Δoutside15k | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | Baseline | 45.8 | +0.0 | 44.5 | +0.0 | 39.0 | +0.0 | +0.00 |
| gen_Attribute_Hallucination | Generative | 46.8 | +1.0 | 44.8 | +0.3 | 38.9 | -0.2 | +0.38 |
| gen_CNetSeg | Generative | 46.7 | +0.9 | 44.6 | +0.1 | 39.7 | +0.7 | +0.58 |
| gen_CUT | Generative | 46.4 | +0.7 | 44.9 | +0.4 | 39.2 | +0.2 | +0.42 |
| gen_IP2P | Generative | 46.1 | +0.3 | 44.7 | +0.2 | 39.4 | +0.3 | +0.29 |
| gen_Img2Img | Generative | 46.0 | +0.3 | 44.6 | +0.1 | 39.4 | +0.4 | +0.26 |
| gen_LANIT | Generative | 46.6 | +0.8 | 44.6 | +0.2 | 39.2 | +0.2 | +0.37 |
| gen_Qwen_Image_Edit | Generative | 46.7 | +0.9 | 44.6 | +0.1 | 37.8 | -1.2 | -0.04 |
| gen_SUSTechGAN | Generative | 45.9 | +0.2 | 44.7 | +0.2 | 39.6 | +0.5 | +0.31 |
| gen_TSIT | Generative | 46.4 | +0.6 | 44.6 | +0.1 | 39.0 | -0.0 | +0.24 |
| gen_UniControl | Generative | 46.7 | +0.9 | 44.6 | +0.1 | 39.4 | +0.4 | +0.49 |
| gen_VisualCloze | Generative | 46.1 | +0.3 | 44.7 | +0.2 | 39.6 | +0.6 | +0.38 |
| gen_Weather_Effect_Generator | Generative | 46.5 | +0.8 | 44.2 | -0.3 | 38.4 | -0.7 | -0.07 |
| gen_albumentations_weather | Generative | 46.1 | +0.3 | 44.7 | +0.2 | 39.6 | +0.5 | +0.34 |
| gen_augmenters | Generative | 46.6 | +0.8 | 44.7 | +0.2 | 39.3 | +0.3 | +0.44 |
| gen_automold | Generative | 46.5 | +0.8 | 44.7 | +0.3 | 38.7 | -0.3 | +0.23 |
| gen_cycleGAN | Generative | 46.9 | +1.2 | 44.7 | +0.2 | 38.9 | -0.1 | +0.42 |
| gen_cyclediffusion | Generative | 46.5 | +0.7 | 44.5 | -0.0 | 39.7 | +0.7 | +0.47 |
| gen_flux_kontext | Generative | 45.6 | -0.2 | 44.8 | +0.4 | 39.0 | +0.0 | +0.07 |
| gen_stargan_v2 | Generative | 46.1 | +0.4 | 44.7 | +0.2 | 40.0 | +1.0 | +0.50 |
| gen_step1x_new | Generative | 45.5 | -0.3 | 44.3 | -0.2 | 39.4 | +0.4 | -0.03 |
| gen_step1x_v1p2 | Generative | 46.3 | +0.5 | 44.7 | +0.2 | 39.0 | -0.0 | +0.24 |
| photometric_distort | Augmentation | 46.5 | +0.7 | 44.5 | -0.0 | 38.6 | -0.4 | +0.10 |
| std_autoaugment | Standard Aug | 46.9 | +1.1 | 44.6 | +0.1 | 39.1 | +0.1 | +0.46 |
| std_cutmix | Standard Aug | 45.2 | -0.6 | 44.4 | -0.1 | 38.8 | -0.2 | -0.29 |
| std_mixup | Standard Aug | 46.0 | +0.3 | 44.2 | -0.3 | 39.5 | +0.5 | +0.17 |
| std_randaugment | Standard Aug | 46.3 | +0.5 | 44.6 | +0.1 | 39.7 | +0.7 | +0.43 |

## Per-Domain mIoU by Strategy

| Strategy | Type | clear_day | cloudy | dawn_dusk | foggy | night | rainy | snowy | Normal Avg | Adverse Avg | Gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | Baseline | 43.84 | 44.17 | 37.26 | 57.79 | 33.84 | 41.62 | 43.12 | 43.84 | 42.97 | 0.87 |
| gen_Attribute_Hallucination | Generative | 44.44 | 44.24 | 36.92 | 55.08 | 33.82 | 41.19 | 43.73 | 44.44 | 42.50 | 1.94 |
| gen_CNetSeg | Generative | 44.56 | 44.19 | 37.15 | 55.00 | 34.46 | 41.92 | 43.56 | 44.56 | 42.71 | 1.85 |
| gen_CUT | Generative | 44.37 | 44.36 | 37.91 | 55.50 | 33.58 | 41.83 | 43.60 | 44.37 | 42.80 | 1.57 |
| gen_IP2P | Generative | 44.29 | 44.31 | 37.51 | 55.74 | 33.80 | 41.18 | 43.35 | 44.29 | 42.65 | 1.64 |
| gen_Img2Img | Generative | 44.29 | 44.14 | 37.47 | 55.34 | 33.44 | 41.17 | 43.60 | 44.29 | 42.53 | 1.77 |
| gen_LANIT | Generative | 44.42 | 44.44 | 37.33 | 55.39 | 32.92 | 41.16 | 43.25 | 44.42 | 42.41 | 2.01 |
| gen_Qwen_Image_Edit | Generative | 43.90 | 44.33 | 37.41 | 54.74 | 33.71 | 41.42 | 42.60 | 43.90 | 42.37 | 1.53 |
| gen_SUSTechGAN | Generative | 44.33 | 44.09 | 37.39 | 54.95 | 33.68 | 41.14 | 43.60 | 44.33 | 42.48 | 1.85 |
| gen_TSIT | Generative | 44.29 | 44.30 | 36.98 | 55.17 | 33.40 | 40.85 | 43.19 | 44.29 | 42.32 | 1.97 |
| gen_UniControl | Generative | 44.58 | 44.42 | 37.41 | 56.04 | 33.33 | 41.53 | 43.86 | 44.58 | 42.77 | 1.81 |
| gen_VisualCloze | Generative | 44.41 | 44.16 | 37.14 | 55.82 | 34.20 | 41.53 | 43.51 | 44.41 | 42.73 | 1.68 |
| gen_Weather_Effect_Generator | Generative | 43.95 | 43.83 | 37.27 | 54.54 | 32.88 | 40.99 | 42.50 | 43.95 | 42.00 | 1.95 |
| gen_albumentations_weather | Generative | 44.44 | 43.89 | 37.41 | 54.99 | 34.05 | 41.46 | 43.33 | 44.44 | 42.52 | 1.92 |
| gen_augmenters | Generative | 44.39 | 44.19 | 37.10 | 55.84 | 34.01 | 41.67 | 43.74 | 44.39 | 42.76 | 1.63 |
| gen_automold | Generative | 44.33 | 44.39 | 36.86 | 55.48 | 33.49 | 40.84 | 43.46 | 44.33 | 42.42 | 1.91 |
| gen_cycleGAN | Generative | 44.40 | 43.76 | 37.32 | 54.93 | 34.06 | 41.45 | 43.45 | 44.40 | 42.50 | 1.90 |
| gen_cyclediffusion | Generative | 44.58 | 44.51 | 37.71 | 55.62 | 33.75 | 41.64 | 42.95 | 44.58 | 42.70 | 1.88 |
| gen_flux_kontext | Generative | 44.15 | 44.25 | 37.26 | 55.25 | 33.75 | 41.05 | 43.04 | 44.15 | 42.43 | 1.71 |
| gen_stargan_v2 | Generative | 44.53 | 44.39 | 37.75 | 56.50 | 33.91 | 41.52 | 43.59 | 44.53 | 42.94 | 1.58 |
| gen_step1x_new | Generative | 43.87 | 43.73 | 36.64 | 54.85 | 32.82 | 40.94 | 43.11 | 43.87 | 42.02 | 1.85 |
| gen_step1x_v1p2 | Generative | 44.20 | 44.12 | 37.20 | 54.75 | 33.48 | 41.50 | 43.26 | 44.20 | 42.39 | 1.81 |
| photometric_distort | Augmentation | 44.09 | 44.07 | 37.80 | 54.78 | 34.32 | 40.88 | 42.91 | 44.09 | 42.46 | 1.64 |
| std_autoaugment | Standard Aug | 44.38 | 43.96 | 38.48 | 55.07 | 34.01 | 41.83 | 43.95 | 44.38 | 42.88 | 1.50 |
| std_cutmix | Standard Aug | 43.91 | 43.37 | 36.73 | 53.93 | 33.72 | 40.65 | 42.32 | 43.91 | 41.79 | 2.12 |
| std_mixup | Standard Aug | 44.29 | 44.02 | 36.16 | 52.11 | 32.49 | 40.75 | 43.21 | 44.29 | 41.46 | 2.83 |
| std_randaugment | Standard Aug | 44.39 | 44.20 | 37.64 | 55.11 | 33.77 | 41.50 | 43.81 | 44.39 | 42.67 | 1.72 |