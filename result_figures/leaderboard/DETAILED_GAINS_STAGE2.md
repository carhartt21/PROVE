# Stage 2 Detailed Per-Dataset and Per-Domain Analysis

## Per-Dataset mIoU by Strategy

| Strategy | Type | bdd10k | Δbdd10k | idd-aw | Δidd-aw | mapillaryvistas | Δmapillaryvistas | outside15k | Δoutside15k | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | Baseline | 45.8 | +0.0 | 44.5 | +0.0 | 48.6 | +0.0 | 39.0 | +0.0 | +0.00 |
| gen_Attribute_Hallucination | Generative | 46.8 | +1.0 | 44.8 | +0.3 | 48.1 | -0.5 | 38.9 | -0.2 | +0.15 |
| gen_CNetSeg | Generative | 46.7 | +0.9 | 44.6 | +0.1 | 48.8 | +0.1 | 39.7 | +0.7 | +0.46 |
| gen_CUT | Generative | 46.4 | +0.7 | 44.9 | +0.4 | 49.3 | +0.7 | 39.2 | +0.2 | +0.48 |
| gen_IP2P | Generative | 46.1 | +0.3 | 44.7 | +0.2 | 49.0 | +0.3 | 39.4 | +0.3 | +0.30 |
| gen_Img2Img | Generative | 46.0 | +0.3 | 44.6 | +0.1 | 49.0 | +0.4 | 39.4 | +0.4 | +0.29 |
| gen_LANIT | Generative | 46.6 | +0.8 | 44.6 | +0.2 | 48.9 | +0.2 | 39.2 | +0.2 | +0.34 |
| gen_Qwen_Image_Edit | Generative | 46.7 | +0.9 | 44.6 | +0.1 | 48.9 | +0.3 | 37.8 | -1.2 | +0.04 |
| gen_SUSTechGAN | Generative | 45.9 | +0.2 | 44.7 | +0.2 | 48.9 | +0.2 | 39.6 | +0.5 | +0.29 |
| gen_TSIT | Generative | 46.4 | +0.6 | 44.6 | +0.1 | 48.9 | +0.2 | 39.0 | -0.0 | +0.24 |
| gen_UniControl | Generative | 46.7 | +0.9 | 44.6 | +0.1 | 49.2 | +0.6 | 39.4 | +0.4 | +0.51 |
| gen_VisualCloze | Generative | 46.1 | +0.3 | 44.7 | +0.2 | 48.9 | +0.2 | 39.6 | +0.6 | +0.34 |
| gen_Weather_Effect_Generator | Generative | 46.5 | +0.8 | 44.2 | -0.3 | 50.2 | +1.6 | 38.4 | -0.7 | +0.34 |
| gen_albumentations_weather | Generative | 46.1 | +0.3 | 44.7 | +0.2 | 49.0 | +0.3 | 39.6 | +0.5 | +0.34 |
| gen_augmenters | Generative | 46.6 | +0.8 | 44.7 | +0.2 | 49.1 | +0.4 | 39.3 | +0.3 | +0.44 |
| gen_automold | Generative | 46.5 | +0.8 | 44.7 | +0.3 | 48.7 | +0.0 | 38.7 | -0.3 | +0.17 |
| gen_cycleGAN | Generative | 46.9 | +1.2 | 44.7 | +0.2 | 49.2 | +0.5 | 38.9 | -0.1 | +0.45 |
| gen_cyclediffusion | Generative | 46.5 | +0.7 | 44.5 | -0.0 | 49.8 | +1.2 | 39.7 | +0.7 | +0.64 |
| gen_flux_kontext | Generative | 45.6 | -0.2 | 44.8 | +0.4 | 49.3 | +0.6 | 39.0 | +0.0 | +0.21 |
| gen_stargan_v2 | Generative | 46.1 | +0.4 | 44.7 | +0.2 | 48.9 | +0.3 | 40.0 | +1.0 | +0.45 |
| gen_step1x_new | Generative | 45.5 | -0.3 | 44.3 | -0.2 | 48.6 | -0.0 | 39.4 | +0.4 | -0.03 |
| gen_step1x_v1p2 | Generative | 46.3 | +0.5 | 44.7 | +0.2 | 48.9 | +0.2 | 39.0 | -0.0 | +0.23 |
| photometric_distort | Augmentation | 46.5 | +0.7 | 44.5 | -0.0 | 48.4 | -0.2 | 38.6 | -0.4 | +0.01 |
| std_autoaugment | Standard Aug | 46.9 | +1.1 | 44.6 | +0.1 | 48.3 | -0.3 | 39.1 | +0.1 | +0.27 |
| std_cutmix | Standard Aug | 47.1 | +1.3 | 44.4 | -0.1 | 49.0 | +0.3 | 42.5 | +3.4 | +1.26 |
| std_mixup | Standard Aug | 46.0 | +0.3 | 44.2 | -0.3 | 53.0 | +4.3 | 39.5 | +0.5 | +1.21 |
| std_randaugment | Standard Aug | 46.3 | +0.5 | 44.6 | +0.1 | 48.5 | -0.2 | 39.7 | +0.7 | +0.29 |

## Per-Domain mIoU by Strategy

| Strategy | Type | clear_day | cloudy | dawn_dusk | foggy | night | rainy | snowy | Normal Avg | Adverse Avg | Gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | Baseline | 45.08 | 45.08 | 40.12 | 59.68 | 37.23 | 42.89 | 44.31 | 45.08 | 44.89 | 0.20 |
| gen_Attribute_Hallucination | Generative | 45.40 | 45.04 | 39.88 | 57.22 | 36.97 | 42.53 | 44.43 | 45.40 | 44.34 | 1.06 |
| gen_CNetSeg | Generative | 45.63 | 45.19 | 40.01 | 56.77 | 37.62 | 43.18 | 44.53 | 45.63 | 44.55 | 1.08 |
| gen_CUT | Generative | 45.67 | 45.38 | 40.50 | 56.98 | 37.20 | 43.07 | 44.80 | 45.67 | 44.66 | 1.02 |
| gen_IP2P | Generative | 45.55 | 45.23 | 40.00 | 57.80 | 37.04 | 42.49 | 44.49 | 45.55 | 44.51 | 1.04 |
| gen_Img2Img | Generative | 45.52 | 45.20 | 40.37 | 57.15 | 36.82 | 42.74 | 44.65 | 45.52 | 44.49 | 1.04 |
| gen_LANIT | Generative | 45.55 | 45.45 | 40.37 | 56.82 | 36.42 | 42.52 | 44.08 | 45.55 | 44.28 | 1.27 |
| gen_Qwen_Image_Edit | Generative | 45.20 | 45.28 | 40.30 | 57.15 | 37.05 | 42.65 | 43.98 | 45.20 | 44.40 | 0.80 |
| gen_SUSTechGAN | Generative | 45.49 | 45.21 | 40.41 | 56.72 | 37.22 | 42.28 | 44.52 | 45.49 | 44.39 | 1.10 |
| gen_TSIT | Generative | 45.49 | 45.22 | 40.03 | 57.16 | 37.09 | 42.37 | 44.21 | 45.49 | 44.35 | 1.14 |
| gen_UniControl | Generative | 45.73 | 45.53 | 40.61 | 57.74 | 36.78 | 42.69 | 45.07 | 45.73 | 44.74 | 0.99 |
| gen_VisualCloze | Generative | 45.53 | 45.16 | 40.02 | 57.76 | 37.47 | 42.95 | 44.51 | 45.53 | 44.65 | 0.88 |
| gen_Weather_Effect_Generator | Generative | 45.12 | 44.77 | 39.47 | 56.31 | 35.76 | 42.23 | 43.80 | 45.12 | 43.72 | 1.40 |
| gen_albumentations_weather | Generative | 45.64 | 44.92 | 40.31 | 56.92 | 37.46 | 42.57 | 44.30 | 45.64 | 44.41 | 1.22 |
| gen_augmenters | Generative | 45.59 | 45.21 | 40.17 | 58.19 | 37.34 | 43.03 | 44.79 | 45.59 | 44.79 | 0.80 |
| gen_automold | Generative | 45.43 | 45.36 | 39.85 | 57.23 | 37.05 | 42.16 | 44.47 | 45.43 | 44.35 | 1.08 |
| gen_cycleGAN | Generative | 45.63 | 44.90 | 39.93 | 56.70 | 37.46 | 42.62 | 44.70 | 45.63 | 44.38 | 1.25 |
| gen_cyclediffusion | Generative | 45.57 | 45.27 | 39.94 | 56.66 | 36.59 | 42.60 | 43.97 | 45.57 | 44.17 | 1.40 |
| gen_flux_kontext | Generative | 45.47 | 45.36 | 40.48 | 57.19 | 37.34 | 42.24 | 44.08 | 45.47 | 44.45 | 1.02 |
| gen_stargan_v2 | Generative | 45.66 | 45.41 | 40.34 | 57.73 | 36.95 | 42.74 | 44.38 | 45.66 | 44.59 | 1.07 |
| gen_step1x_new | Generative | 45.08 | 44.87 | 40.07 | 56.56 | 36.40 | 42.36 | 44.18 | 45.08 | 44.07 | 1.00 |
| gen_step1x_v1p2 | Generative | 45.36 | 45.18 | 40.16 | 56.54 | 36.97 | 42.87 | 44.56 | 45.36 | 44.38 | 0.98 |
| photometric_distort | Augmentation | 45.18 | 45.01 | 40.56 | 56.34 | 37.49 | 42.28 | 43.78 | 45.18 | 44.24 | 0.94 |
| std_autoaugment | Standard Aug | 45.43 | 44.90 | 40.98 | 56.78 | 37.45 | 42.88 | 44.88 | 45.43 | 44.65 | 0.78 |
| std_cutmix | Standard Aug | 46.65 | 46.03 | 42.06 | 58.05 | 39.65 | 43.77 | 45.19 | 46.65 | 45.79 | 0.86 |
| std_mixup | Standard Aug | 45.19 | 44.70 | 37.71 | 53.48 | 34.30 | 41.68 | 43.88 | 45.19 | 42.63 | 2.57 |
| std_randaugment | Standard Aug | 45.44 | 45.10 | 40.29 | 56.79 | 36.95 | 42.76 | 44.99 | 45.44 | 44.48 | 0.96 |