# Stage 2 Detailed Per-Dataset and Per-Domain Analysis

## Per-Dataset mIoU by Strategy

| Strategy | Type | bdd10k | Δbdd10k | idd-aw | Δidd-aw | mapillaryvistas | Δmapillaryvistas | outside15k | Δoutside15k | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | Baseline | - | - | 44.5 | +0.0 | 49.7 | +0.0 | 39.0 | +0.0 | +0.00 |
| gen_Attribute_Hallucination | Generative | - | - | 43.7 | -0.8 | 48.1 | -1.6 | 38.9 | -0.2 | -0.85 |
| gen_CNetSeg | Generative | 44.2 | +0.0 | 43.6 | -0.9 | 48.8 | -1.0 | 39.7 | +0.7 | -0.30 |
| gen_CUT | Generative | - | - | 44.9 | +0.4 | 49.3 | -0.4 | 39.2 | +0.2 | +0.07 |
| gen_IP2P | Generative | - | - | 43.6 | -0.9 | 49.0 | -0.7 | 39.4 | +0.3 | -0.44 |
| gen_Img2Img | Generative | - | - | - | - | 49.0 | -0.7 | 39.4 | +0.4 | -0.13 |
| gen_LANIT | Generative | - | - | - | - | 50.2 | +0.5 | 39.2 | +0.2 | +0.32 |
| gen_Qwen_Image_Edit | Generative | 43.2 | +0.0 | 43.6 | -0.9 | 48.9 | -0.8 | 31.3 | -7.7 | -2.35 |
| gen_SUSTechGAN | Generative | 43.3 | +0.0 | 43.7 | -0.8 | 48.9 | -0.9 | 33.9 | -5.1 | -1.70 |
| gen_TSIT | Generative | 43.5 | +0.0 | 43.5 | -1.0 | 49.8 | +0.1 | 33.2 | -5.8 | -1.70 |
| gen_UniControl | Generative | 45.2 | +0.0 | 43.5 | -1.0 | 50.0 | +0.3 | 33.9 | -5.1 | -1.45 |
| gen_VisualCloze | Generative | 43.0 | +0.0 | 43.6 | -0.9 | 48.9 | -0.8 | - | - | -0.58 |
| gen_Weather_Effect_Generator | Generative | 51.9 | +0.0 | - | - | 50.2 | +0.5 | - | - | +0.26 |
| gen_albumentations_weather | Generative | 43.1 | +0.0 | 43.6 | -0.9 | 49.0 | -0.7 | 39.6 | +0.5 | -0.28 |
| gen_augmenters | Generative | 43.9 | +0.0 | 43.7 | -0.8 | 50.0 | +0.3 | 39.3 | +0.3 | -0.05 |
| gen_automold | Generative | 43.8 | +0.0 | 43.7 | -0.8 | 48.7 | -1.1 | 38.7 | -0.3 | -0.54 |
| gen_cycleGAN | Generative | 44.3 | +0.0 | 43.5 | -1.0 | 49.8 | +0.0 | 38.9 | -0.1 | -0.25 |
| gen_flux_kontext | Generative | 43.3 | +0.0 | 43.6 | -0.9 | 49.3 | -0.4 | 39.0 | +0.0 | -0.32 |
| gen_step1x_v1p2 | Generative | - | - | - | - | 48.9 | -0.9 | 32.9 | -6.1 | -3.47 |

## Per-Domain mIoU by Strategy

| Strategy | Type | clear_day | cloudy | dawn_dusk | foggy | night | rainy | snowy | Normal Avg | Adverse Avg | Gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | Baseline | 44.42 | 43.43 | 39.24 | 58.93 | 39.43 | 41.63 | 40.94 | 44.42 | 43.93 | 0.48 |
| gen_Attribute_Hallucination | Generative | 44.37 | 42.69 | 38.69 | 56.04 | 38.42 | 40.66 | 40.41 | 44.37 | 42.82 | 1.55 |
| gen_CNetSeg | Generative | 44.89 | 44.08 | 38.63 | 56.43 | 36.74 | 41.94 | 42.51 | 44.89 | 43.39 | 1.50 |
| gen_CUT | Generative | 45.17 | 43.87 | 40.44 | 56.58 | 40.03 | 42.21 | 41.93 | 45.17 | 44.18 | 0.99 |
| gen_IP2P | Generative | 44.82 | 43.00 | 38.43 | 57.16 | 39.11 | 40.91 | 40.97 | 44.82 | 43.27 | 1.55 |
| gen_Img2Img | Generative | 45.08 | 42.55 | 37.80 | 57.31 | 38.08 | 41.04 | 40.77 | 45.08 | 42.93 | 2.15 |
| gen_LANIT | Generative | 44.52 | 41.64 | 35.81 | 56.56 | 37.04 | 38.76 | 39.16 | 44.52 | 41.49 | 3.03 |
| gen_Qwen_Image_Edit | Generative | 43.41 | 43.36 | 38.02 | 54.73 | 34.92 | 40.47 | 41.81 | 43.41 | 42.22 | 1.19 |
| gen_SUSTechGAN | Generative | 43.91 | 43.46 | 38.85 | 54.69 | 36.00 | 40.47 | 42.51 | 43.91 | 42.66 | 1.25 |
| gen_TSIT | Generative | 43.53 | 42.93 | 37.44 | 54.16 | 34.21 | 39.80 | 41.43 | 43.53 | 41.66 | 1.87 |
| gen_UniControl | Generative | 43.70 | 42.58 | 37.83 | 53.56 | 35.75 | 40.30 | 41.75 | 43.70 | 41.96 | 1.74 |
| gen_VisualCloze | Generative | 46.16 | 46.90 | 43.56 | 59.91 | 39.17 | 44.37 | 46.35 | 46.16 | 46.71 | -0.55 |
| gen_Weather_Effect_Generator | Generative | 50.94 | 50.30 | 47.86 | 61.93 | 43.81 | 49.64 | 52.60 | 50.94 | 51.02 | -0.09 |
| gen_albumentations_weather | Generative | 44.86 | 43.53 | 38.88 | 56.65 | 36.64 | 41.04 | 42.20 | 44.86 | 43.16 | 1.70 |
| gen_augmenters | Generative | 44.48 | 43.82 | 38.22 | 57.53 | 35.68 | 41.15 | 42.54 | 44.48 | 43.16 | 1.32 |
| gen_automold | Generative | 44.59 | 44.17 | 38.75 | 56.60 | 36.29 | 40.66 | 42.36 | 44.59 | 43.14 | 1.45 |
| gen_cycleGAN | Generative | 44.51 | 43.26 | 37.32 | 55.77 | 35.51 | 40.72 | 42.18 | 44.51 | 42.46 | 2.04 |
| gen_flux_kontext | Generative | 44.70 | 44.10 | 39.52 | 56.65 | 36.77 | 40.95 | 42.02 | 44.70 | 43.33 | 1.37 |
| gen_step1x_v1p2 | Generative | 43.13 | 41.36 | 36.80 | 53.48 | 37.44 | 39.66 | 40.57 | 43.13 | 41.55 | 1.58 |