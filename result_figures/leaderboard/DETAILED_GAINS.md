# Detailed Per-Dataset and Per-Domain Analysis

## Per-Dataset mIoU by Strategy

| Strategy | Type | bdd10k | Δbdd10k | idd-aw | Δidd-aw | mapillaryvistas | Δmapillaryvistas | outside15k | Δoutside15k | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gen_flux_kontext | Generative | 46.1 | +2.1 | 39.3 | +0.7 | 49.4 | +2.4 | 38.2 | +1.3 | +1.61 |
| gen_step1x_new | Generative | 46.3 | +2.2 | 39.3 | +0.7 | 48.6 | +1.6 | 48.9 | +12.0 | +4.14 |
| gen_step1x_v1p2 | Generative | 46.1 | +2.0 | 39.7 | +1.1 | 48.0 | +1.0 | 47.9 | +11.0 | +3.78 |
| gen_Qwen_Image_Edit | Generative | 45.9 | +1.8 | 39.8 | +1.2 | 52.0 | +5.0 | 43.4 | +6.5 | +3.63 |
| gen_cycleGAN | Generative | 46.0 | +1.9 | 40.3 | +1.7 | 49.5 | +2.5 | 41.1 | +4.2 | +2.57 |
| gen_Attribute_Hallucination | Generative | 46.5 | +2.4 | 40.0 | +1.4 | 48.2 | +1.2 | 38.0 | +1.1 | +1.53 |
| std_autoaugment | Standard Aug | 46.3 | +2.2 | 39.7 | +1.1 | 47.5 | +0.5 | 29.0 | -7.9 | -1.03 |
| std_mixup | Standard Aug | 46.0 | +1.9 | 39.9 | +1.3 | 47.9 | +0.9 | 37.6 | +0.7 | +1.23 |
| gen_automold | Generative | 46.4 | +2.3 | 39.6 | +1.1 | 48.2 | +1.2 | 37.2 | +0.3 | +1.20 |
| gen_stargan_v2 | Generative | 46.1 | +2.1 | 39.5 | +0.9 | 47.1 | +0.1 | 43.0 | +6.1 | +2.29 |
| gen_CNetSeg | Generative | 46.0 | +1.9 | 39.7 | +1.1 | 48.4 | +1.4 | 37.1 | +0.2 | +1.14 |
| gen_albumentations_weather | Generative | 46.3 | +2.2 | 39.8 | +1.2 | 48.1 | +1.1 | 36.9 | -0.0 | +1.12 |
| gen_IP2P | Generative | 46.3 | +2.2 | 39.6 | +1.1 | 48.2 | +1.2 | 36.7 | -0.1 | +1.08 |
| gen_SUSTechGAN | Generative | 46.2 | +2.2 | 39.5 | +1.0 | 47.7 | +0.7 | 37.3 | +0.4 | +1.06 |
| gen_CUT | Generative | 45.9 | +1.8 | 39.5 | +0.9 | 48.1 | +1.1 | 37.2 | +0.3 | +1.02 |
| gen_Img2Img | Generative | 46.3 | +2.2 | 39.9 | +1.3 | 47.0 | +0.0 | 37.3 | +0.4 | +1.00 |
| std_cutmix | Standard Aug | 46.5 | +2.4 | 39.9 | +1.3 | 47.7 | +0.7 | 36.4 | -0.5 | +0.99 |
| gen_TSIT | Generative | 45.8 | +1.7 | 39.4 | +0.8 | 48.3 | +1.3 | 36.7 | -0.2 | +0.92 |
| gen_VisualCloze | Generative | 45.9 | +1.8 | 39.5 | +1.0 | 47.1 | +0.1 | 37.6 | +0.8 | +0.90 |
| gen_Weather_Effect_Generator | Generative | 46.2 | +2.1 | 39.5 | +0.9 | 49.1 | +2.1 | 37.4 | +0.5 | +1.42 |
| gen_UniControl | Generative | 46.0 | +1.9 | 39.3 | +0.7 | 47.6 | +0.6 | 37.1 | +0.2 | +0.85 |
| std_randaugment | Standard Aug | 46.2 | +2.1 | 40.0 | +1.4 | 48.2 | +1.2 | 31.4 | -5.5 | -0.19 |
| gen_LANIT | Generative | 45.5 | +1.4 | 39.4 | +0.8 | 47.3 | +0.3 | 37.1 | +0.2 | +0.68 |
| gen_augmenters | Generative | 45.9 | +1.8 | 40.0 | +1.4 | 46.1 | -0.9 | 37.1 | +0.2 | +0.63 |
| photometric_distort | Augmentation | 45.8 | +1.7 | 39.9 | +1.3 | 48.3 | +1.3 | 29.1 | -7.8 | -0.87 |
| baseline | Baseline | 44.1 | +0.0 | 38.6 | +0.0 | 47.0 | +0.0 | 36.9 | +0.0 | +0.00 |
| gen_cyclediffusion | Generative | 46.3 | +2.2 | 39.5 | +0.9 | 51.5 | +4.5 | 37.4 | +0.5 | +2.01 |

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
| gen_Qwen_Image_Edit          | Generative   |       47.68 |    44.14 |       38.37 |   46.95 |   33.19 |   39.6  |   41.89 |        45.91 |         40.41 |  5.51 |
| gen_SUSTechGAN               | Generative   |       45.19 |    41.28 |       35.07 |   43.55 |   31.04 |   37.55 |   37.36 |        43.24 |         37.37 |  5.86 |
| gen_TSIT                     | Generative   |       45.3  |    41.08 |       34.89 |   42.78 |   30.28 |   37.22 |   36.95 |        43.19 |         36.81 |  6.38 |
| gen_UniControl               | Generative   |       45.14 |    41.19 |       34.78 |   44.4  |   30.77 |   37.69 |   37.41 |        43.16 |         37.57 |  5.6  |
| gen_VisualCloze              | Generative   |       45.19 |    41.32 |       34.9  |   43.15 |   30.34 |   37.68 |   37.31 |        43.25 |         37.12 |  6.13 |
| gen_Weather_Effect_Generator | Generative   |       45.29 |    41.09 |       34.26 |   42.69 |   29.43 |   37.03 |   37.02 |        43.19 |         36.54 |  6.65 |
| gen_albumentations_weather   | Generative   |       45.37 |    41.63 |       35.46 |   44.33 |   31.09 |   37.52 |   36.84 |        43.5  |         37.45 |  6.06 |
| gen_augmenters               | Generative   |       44.92 |    41.38 |       34.7  |   43.42 |   30.43 |   37.01 |   36.76 |        43.15 |         36.91 |  6.24 |
| gen_automold                 | Generative   |       45.32 |    41.84 |       34.85 |   43.22 |   30.79 |   37.92 |   37.24 |        43.58 |         37.29 |  6.29 |
| gen_cycleGAN                 | Generative   |       46.43 |    40.47 |       35.89 |   42.11 |   34.28 |   36.99 |   34.44 |        43.45 |         36.95 |  6.5  |
| gen_cyclediffusion           | Generative   |       44.79 |    38    |       32.43 |   35.45 |   27.27 |   32.36 |   30.58 |        41.4  |         31.41 |  9.98 |
| gen_flux_kontext             | Generative   |       48.09 |    44.56 |       41.16 |   48.08 |   39.32 |   41.94 |   41.48 |        46.33 |         42.71 |  3.62 |
| gen_stargan_v2               | Generative   |       46.11 |    40.44 |       36.51 |   39.79 |   31.84 |   35.66 |   34.1  |        43.27 |         35.35 |  7.93 |
| gen_step1x_new               | Generative   |       47.99 |    44.58 |       37.29 |   46.72 |   33.29 |   40.43 |   41.58 |        46.28 |         40.51 |  5.78 |
| gen_step1x_v1p2              | Generative   |       47.58 |    44.2  |       37.8  |   46.23 |   33.08 |   40.29 |   41.15 |        45.89 |         40.19 |  5.7  |
| photometric_distort          | Augmentation |       44.36 |    40.94 |       34.14 |   43.46 |   30.33 |   36.74 |   36.34 |        42.65 |         36.72 |  5.93 |
| std_autoaugment              | Standard Aug |       45.59 |    42.33 |       36.69 |   44.44 |   30.26 |   37.87 |   38.59 |        43.96 |         37.79 |  6.17 |
| std_cutmix                   | Standard Aug |       45.19 |    41.43 |       35.35 |   43.19 |   30.68 |   37.28 |   37.41 |        43.31 |         37.14 |  6.17 |
| std_mixup                    | Standard Aug |       45.47 |    41.63 |       35.3  |   43.06 |   31.13 |   38    |   37.38 |        43.55 |         37.39 |  6.15 |
| std_randaugment              | Standard Aug |       44.95 |    41.27 |       35.18 |   43.26 |   29.85 |   37.34 |   37.23 |        43.11 |         36.92 |  6.19 |