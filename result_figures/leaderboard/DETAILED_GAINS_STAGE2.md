# Stage 2 Detailed Per-Dataset and Per-Domain Analysis

## Per-Dataset mIoU by Strategy

| Strategy | Type | bdd10k | Δbdd10k | idd-aw | Δidd-aw | mapillaryvistas | Δmapillaryvistas | outside15k | Δoutside15k | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | Baseline | - | - | 44.5 | +0.0 | 49.7 | +0.0 | 39.0 | +0.0 | +0.00 |
| gen_Attribute_Hallucination | Generative | - | - | 44.8 | +0.3 | 48.1 | -1.6 | 38.9 | -0.2 | -0.48 |
| gen_CNetSeg | Generative | 46.7 | +0.0 | 44.6 | +0.1 | 48.8 | -1.0 | 39.7 | +0.7 | -0.04 |
| gen_CUT | Generative | - | - | 44.9 | +0.4 | 49.3 | -0.4 | 39.2 | +0.2 | +0.07 |
| gen_IP2P | Generative | 51.7 | +0.0 | 44.7 | +0.2 | 49.0 | -0.7 | 39.4 | +0.3 | -0.05 |
| gen_Img2Img | Generative | 51.5 | +0.0 | 44.6 | +0.1 | 49.0 | -0.7 | 39.4 | +0.4 | -0.04 |
| gen_LANIT | Generative | 48.3 | +0.0 | 44.6 | +0.2 | 48.9 | -0.8 | 39.2 | +0.2 | -0.13 |
| gen_Qwen_Image_Edit | Generative | 45.3 | +0.0 | 44.5 | -0.0 | 48.9 | -0.8 | 37.8 | -1.2 | -0.50 |
| gen_SUSTechGAN | Generative | 45.9 | +0.0 | 44.7 | +0.2 | 48.9 | -0.9 | 39.6 | +0.5 | -0.02 |
| gen_TSIT | Generative | 46.4 | +0.0 | 44.6 | +0.1 | 48.9 | -0.8 | 39.0 | -0.0 | -0.18 |
| gen_UniControl | Generative | 46.7 | +0.0 | 44.6 | +0.1 | 49.2 | -0.5 | 39.4 | +0.4 | +0.01 |
| gen_VisualCloze | Generative | 46.1 | +0.0 | 44.7 | +0.2 | 48.9 | -0.8 | 39.6 | +0.6 | -0.00 |
| gen_Weather_Effect_Generator | Generative | 46.5 | +0.0 | 44.2 | -0.3 | 50.2 | +0.5 | 38.4 | -0.7 | -0.11 |
| gen_albumentations_weather | Generative | 46.1 | +0.0 | 44.7 | +0.2 | 49.0 | -0.7 | 39.6 | +0.5 | -0.00 |
| gen_augmenters | Generative | 46.6 | +0.0 | 44.7 | +0.2 | 49.1 | -0.6 | 39.3 | +0.3 | -0.02 |
| gen_automold | Generative | 46.5 | +0.0 | 44.7 | +0.3 | 48.7 | -1.1 | 38.7 | -0.3 | -0.28 |
| gen_cycleGAN | Generative | 46.9 | +0.0 | 44.7 | +0.2 | 49.2 | -0.5 | 38.9 | -0.1 | -0.11 |
| gen_flux_kontext | Generative | 44.7 | +0.0 | 44.3 | -0.1 | 49.3 | -0.4 | 39.0 | +0.0 | -0.14 |
| gen_stargan_v2 | Generative | 46.1 | +0.0 | 44.7 | +0.2 | 50.5 | +0.8 | 40.0 | +1.0 | +0.49 |
| gen_step1x_new | Generative | 45.5 | +0.0 | 44.3 | -0.2 | 49.9 | +0.2 | 39.4 | +0.4 | +0.10 |
| gen_step1x_v1p2 | Generative | 46.3 | +0.0 | 44.7 | +0.2 | 48.9 | -0.9 | 39.0 | -0.0 | -0.17 |
| photometric_distort | Augmentation | 46.5 | +0.0 | 44.5 | -0.0 | 53.0 | +3.3 | 38.6 | -0.4 | +0.71 |
| std_autoaugment | Standard Aug | 46.9 | +0.0 | 44.6 | +0.1 | 48.3 | -1.4 | 39.1 | +0.1 | -0.28 |
| std_randaugment | Standard Aug | 46.3 | +0.0 | 44.6 | +0.1 | 48.5 | -1.2 | 39.7 | +0.7 | -0.11 |

## Per-Domain mIoU by Strategy

| Strategy | Type | clear_day | cloudy | dawn_dusk | foggy | night | rainy | snowy | Normal Avg | Adverse Avg | Gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | Baseline | 44.42 | 43.43 | 39.24 | 58.93 | 39.43 | 41.63 | 40.94 | 44.42 | 43.93 | 0.48 |
| gen_Attribute_Hallucination | Generative | 44.67 | 43.36 | 39.75 | 56.51 | 39.69 | 41.40 | 41.37 | 44.67 | 43.68 | 0.99 |
| gen_CNetSeg | Generative | 45.63 | 45.19 | 40.01 | 56.77 | 37.62 | 43.18 | 44.53 | 45.63 | 44.55 | 1.08 |
| gen_CUT | Generative | 45.17 | 43.87 | 40.44 | 56.58 | 40.03 | 42.21 | 41.93 | 45.17 | 44.18 | 0.99 |
| gen_IP2P | Generative | 45.74 | 44.72 | 40.20 | 57.48 | 39.48 | 42.64 | 43.78 | 45.74 | 44.72 | 1.03 |
| gen_Img2Img | Generative | 45.72 | 44.68 | 40.70 | 56.76 | 38.86 | 42.86 | 43.82 | 45.72 | 44.62 | 1.11 |
| gen_LANIT | Generative | 45.64 | 45.21 | 40.52 | 56.17 | 37.84 | 42.57 | 43.71 | 45.64 | 44.34 | 1.30 |
| gen_Qwen_Image_Edit | Generative | 45.05 | 45.63 | 40.06 | 57.03 | 35.46 | 42.52 | 44.42 | 45.05 | 44.19 | 0.86 |
| gen_SUSTechGAN | Generative | 45.49 | 45.21 | 40.41 | 56.72 | 37.22 | 42.28 | 44.52 | 45.49 | 44.39 | 1.10 |
| gen_TSIT | Generative | 45.49 | 45.22 | 40.03 | 57.16 | 37.09 | 42.37 | 44.21 | 45.49 | 44.35 | 1.14 |
| gen_UniControl | Generative | 45.73 | 45.53 | 40.61 | 57.74 | 36.78 | 42.69 | 45.07 | 45.73 | 44.74 | 0.99 |
| gen_VisualCloze | Generative | 45.53 | 45.16 | 40.02 | 57.76 | 37.47 | 42.95 | 44.51 | 45.53 | 44.65 | 0.88 |
| gen_Weather_Effect_Generator | Generative | 45.12 | 44.77 | 39.47 | 56.31 | 35.76 | 42.23 | 43.80 | 45.12 | 43.72 | 1.40 |
| gen_albumentations_weather | Generative | 45.64 | 44.92 | 40.31 | 56.92 | 37.46 | 42.57 | 44.30 | 45.64 | 44.41 | 1.22 |
| gen_augmenters | Generative | 45.59 | 45.21 | 40.17 | 58.19 | 37.34 | 43.03 | 44.79 | 45.59 | 44.79 | 0.80 |
| gen_automold | Generative | 45.43 | 45.36 | 39.85 | 57.23 | 37.05 | 42.16 | 44.47 | 45.43 | 44.35 | 1.08 |
| gen_cycleGAN | Generative | 45.63 | 44.90 | 39.93 | 56.70 | 37.46 | 42.62 | 44.70 | 45.63 | 44.38 | 1.25 |
| gen_flux_kontext | Generative | 45.18 | 45.58 | 40.65 | 56.64 | 36.36 | 42.22 | 44.34 | 45.18 | 44.30 | 0.88 |
| gen_stargan_v2 | Generative | 45.66 | 45.34 | 39.91 | 57.66 | 36.36 | 42.47 | 44.22 | 45.66 | 44.33 | 1.33 |
| gen_step1x_new | Generative | 44.98 | 44.75 | 39.34 | 56.69 | 35.66 | 42.19 | 44.21 | 44.98 | 43.81 | 1.17 |
| gen_step1x_v1p2 | Generative | 45.36 | 45.18 | 40.16 | 56.54 | 36.97 | 42.87 | 44.56 | 45.36 | 44.38 | 0.98 |
| photometric_distort | Augmentation | 45.00 | 44.71 | 39.33 | 56.13 | 36.14 | 41.77 | 43.63 | 45.00 | 43.62 | 1.38 |
| std_autoaugment | Standard Aug | 45.43 | 44.90 | 40.98 | 56.78 | 37.45 | 42.88 | 44.88 | 45.43 | 44.65 | 0.78 |
| std_randaugment | Standard Aug | 45.44 | 45.10 | 40.29 | 56.79 | 36.95 | 42.76 | 44.99 | 45.44 | 44.48 | 0.96 |