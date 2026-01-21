# Detailed Per-Dataset and Per-Domain Analysis

## Per-Dataset mIoU by Strategy

| Strategy | Type | bdd10k | Δbdd10k | idd-aw | Δidd-aw | mapillaryvistas | Δmapillaryvistas | outside15k | Δoutside15k | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gen_Qwen_Image_Edit | Generative | 45.9 | +1.8 | 39.8 | +1.2 | 52.0 | +5.0 | 36.8 | -0.1 | +1.97 |
| gen_Attribute_Hallucination | Generative | 46.5 | +2.4 | 40.0 | +1.4 | 48.2 | +1.2 | 38.0 | +1.1 | +1.53 |
| std_autoaugment | Standard Aug | 46.1 | +2.0 | 39.8 | +1.2 | 48.6 | +1.6 | 37.6 | +0.7 | +1.38 |
| gen_cycleGAN | Generative | 46.0 | +1.9 | 40.0 | +1.5 | 48.0 | +1.0 | 37.9 | +1.0 | +1.35 |
| gen_step1x_new | Generative | 46.3 | +2.2 | 39.3 | +0.7 | 48.6 | +1.6 | 37.4 | +0.6 | +1.28 |
| gen_flux_kontext | Generative | 46.1 | +2.1 | 39.3 | +0.7 | 48.1 | +1.1 | 38.2 | +1.3 | +1.28 |
| gen_stargan_v2 | Generative | 46.1 | +2.1 | 39.5 | +1.0 | 48.1 | +1.1 | 37.7 | +0.8 | +1.25 |
| gen_cyclediffusion | Generative | 46.3 | +2.2 | 39.8 | +1.3 | 48.1 | +1.1 | 37.4 | +0.5 | +1.24 |
| gen_automold | Generative | 46.4 | +2.3 | 39.6 | +1.1 | 48.2 | +1.2 | 37.2 | +0.3 | +1.20 |
| gen_CNetSeg | Generative | 46.0 | +1.9 | 39.7 | +1.1 | 48.4 | +1.4 | 37.1 | +0.2 | +1.14 |
| gen_albumentations_weather | Generative | 46.3 | +2.2 | 39.8 | +1.2 | 48.1 | +1.1 | 36.9 | -0.0 | +1.12 |
| gen_Weather_Effect_Generator | Generative | 46.2 | +2.1 | 39.5 | +0.9 | 47.8 | +0.8 | 37.4 | +0.5 | +1.09 |
| gen_IP2P | Generative | 46.3 | +2.2 | 39.6 | +1.1 | 48.2 | +1.2 | 36.7 | -0.1 | +1.08 |
| gen_SUSTechGAN | Generative | 46.2 | +2.2 | 39.5 | +1.0 | 47.7 | +0.7 | 37.3 | +0.4 | +1.06 |
| gen_CUT | Generative | 45.9 | +1.8 | 39.5 | +0.9 | 48.1 | +1.1 | 37.2 | +0.3 | +1.02 |
| gen_Img2Img | Generative | 46.3 | +2.2 | 39.9 | +1.3 | 47.0 | +0.0 | 37.3 | +0.4 | +1.00 |
| gen_TSIT | Generative | 45.8 | +1.7 | 39.4 | +0.8 | 48.3 | +1.3 | 36.7 | -0.2 | +0.92 |
| gen_VisualCloze | Generative | 45.9 | +1.8 | 39.5 | +1.0 | 47.1 | +0.1 | 37.6 | +0.8 | +0.90 |
| gen_step1x_v1p2 | Generative | 46.1 | +2.0 | 39.7 | +1.1 | 48.0 | +1.0 | 36.4 | -0.5 | +0.89 |
| gen_UniControl | Generative | 46.0 | +1.9 | 39.3 | +0.7 | 47.6 | +0.6 | 37.1 | +0.2 | +0.85 |
| std_mixup | Standard Aug | 45.6 | +1.5 | 38.6 | +0.1 | 47.8 | +0.8 | 37.7 | +0.8 | +0.77 |
| photometric_distort | Augmentation | 45.8 | +1.7 | 39.9 | +1.3 | 48.3 | +1.3 | 35.6 | -1.3 | +0.73 |
| gen_LANIT | Generative | 45.5 | +1.4 | 39.4 | +0.8 | 47.3 | +0.3 | 37.1 | +0.2 | +0.68 |
| gen_augmenters | Generative | 45.9 | +1.8 | 40.0 | +1.4 | 46.1 | -0.9 | 37.1 | +0.2 | +0.63 |
| std_randaugment | Standard Aug | 45.3 | +1.2 | 39.5 | +1.0 | 46.4 | -0.6 | 37.4 | +0.5 | +0.50 |
| std_cutmix | Standard Aug | 44.5 | +0.4 | 39.3 | +0.8 | 46.9 | -0.1 | 36.9 | +0.0 | +0.27 |
| baseline | Baseline | 44.1 | +0.0 | 38.6 | +0.0 | 47.0 | +0.0 | 36.9 | +0.0 | +0.00 |

## Per-Domain mIoU by Strategy

| Strategy                     | Type         |   clear_day |   cloudy |   dawn_dusk |   foggy |   night |   rainy |   snowy |   Normal Avg |   Adverse Avg |   Gap |
|:-----------------------------|:-------------|------------:|---------:|------------:|--------:|--------:|--------:|--------:|-------------:|--------------:|------:|
| baseline                     | Baseline     |       44.33 |    40.11 |       33.78 |   42.18 |   29.73 |   36.44 |   35.96 |        42.22 |         36.08 |  6.14 |
| gen_Attribute_Hallucination  | Generative   |       45.67 |    42.22 |       35.66 |   44.1  |   30.96 |   38.11 |   37.62 |        43.94 |         37.7  |  6.25 |
| gen_CNetSeg                  | Generative   |       45.48 |    41.68 |       35.15 |   44.42 |   31.29 |   37.73 |   37.05 |        43.58 |         37.62 |  5.95 |
| gen_CUT                      | Generative   |       45.27 |    41.42 |       34.79 |   44.38 |   30.68 |   37.23 |   37.16 |        43.34 |         37.36 |  5.98 |
| gen_IP2P                     | Generative   |       45.28 |    41.5  |       35.18 |   42.8  |   30.22 |   37.79 |   37.62 |        43.39 |         37.11 |  6.28 |
| gen_Img2Img                  | Generative   |       45.24 |    41.74 |       34.86 |   43.26 |   31.25 |   37.43 |   37.19 |        43.49 |         37.28 |  6.21 |
| gen_LANIT                    | Generative   |       44.97 |    41.07 |       34.59 |   42.31 |   30.05 |   37.02 |   36.72 |        43.02 |         36.52 |  6.5  |
| gen_Qwen_Image_Edit          | Generative   |       46.19 |    42.59 |       36.72 |   44.77 |   31.73 |   38.26 |   39.47 |        44.39 |         38.56 |  5.83 |
| gen_SUSTechGAN               | Generative   |       45.19 |    41.28 |       35.07 |   43.55 |   31.04 |   37.55 |   37.36 |        43.24 |         37.37 |  5.86 |
| gen_TSIT                     | Generative   |       45.3  |    41.08 |       34.89 |   42.78 |   30.28 |   37.22 |   36.95 |        43.19 |         36.81 |  6.38 |
| gen_UniControl               | Generative   |       45.14 |    41.19 |       34.78 |   44.4  |   30.77 |   37.69 |   37.41 |        43.16 |         37.57 |  5.6  |
| gen_VisualCloze              | Generative   |       45.19 |    41.32 |       34.9  |   43.15 |   30.34 |   37.68 |   37.31 |        43.25 |         37.12 |  6.13 |
| gen_Weather_Effect_Generator | Generative   |       45.38 |    41.42 |       35.09 |   43.1  |   30.55 |   37.38 |   37.25 |        43.4  |         37.07 |  6.33 |
| gen_albumentations_weather   | Generative   |       45.37 |    41.63 |       35.46 |   44.33 |   31.09 |   37.52 |   36.84 |        43.5  |         37.45 |  6.06 |
| gen_augmenters               | Generative   |       44.92 |    41.38 |       34.7  |   43.42 |   30.43 |   37.01 |   36.76 |        43.15 |         36.91 |  6.24 |
| gen_automold                 | Generative   |       45.32 |    41.84 |       34.85 |   43.22 |   30.79 |   37.92 |   37.24 |        43.58 |         37.29 |  6.29 |
| gen_cycleGAN                 | Generative   |       45.7  |    41.54 |       35.6  |   43.58 |   30.88 |   37.71 |   37.42 |        43.62 |         37.4  |  6.22 |
| gen_cyclediffusion           | Generative   |       45.56 |    41.66 |       35.11 |   43.87 |   30.47 |   38.23 |   37.18 |        43.61 |         37.44 |  6.17 |
| gen_flux_kontext             | Generative   |       45.52 |    41.77 |       35.11 |   43.6  |   30.78 |   37.76 |   37.44 |        43.65 |         37.39 |  6.25 |
| gen_stargan_v2               | Generative   |       45.55 |    41.88 |       35.83 |   44.03 |   31.05 |   38.12 |   37.21 |        43.72 |         37.6  |  6.11 |
| gen_step1x_new               | Generative   |       45.46 |    41.8  |       35.03 |   43.62 |   30.84 |   37.8  |   37.56 |        43.63 |         37.45 |  6.18 |
| gen_step1x_v1p2              | Generative   |       45.02 |    41.51 |       35.32 |   43.46 |   30.79 |   37.49 |   36.8  |        43.26 |         37.13 |  6.13 |
| photometric_distort          | Augmentation |       44.93 |    41.26 |       34.35 |   43.71 |   31.11 |   37.29 |   36.25 |        43.09 |         37.09 |  6.01 |
| std_autoaugment              | Standard Aug |       45.48 |    41.69 |       36.15 |   45.46 |   32.64 |   37.79 |   37.47 |        43.58 |         38.34 |  5.24 |
| std_cutmix                   | Standard Aug |       44.61 |    41.11 |       34.01 |   44.12 |   30.12 |   36.6  |   36.74 |        42.86 |         36.89 |  5.97 |
| std_mixup                    | Standard Aug |       45.27 |    41.12 |       34.6  |   43.29 |   29.87 |   37.48 |   36.84 |        43.19 |         36.87 |  6.32 |
| std_randaugment              | Standard Aug |       44.5  |    41.07 |       35.48 |   43.11 |   31.71 |   37.21 |   37.05 |        42.79 |         37.27 |  5.52 |