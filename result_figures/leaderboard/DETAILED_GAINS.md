# Detailed Per-Dataset and Per-Domain Analysis

## Per-Dataset mIoU by Strategy

| Strategy | Type | bdd10k | Δbdd10k | idd-aw | Δidd-aw | mapillaryvistas | Δmapillaryvistas | outside15k | Δoutside15k | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| std_randaugment | Standard Aug | 46.2 | +2.1 | 48.1 | +10.0 | 48.2 | +1.2 | 42.2 | +5.3 | +4.66 |
| photometric_distort | Augmentation | 45.8 | +1.7 | 47.9 | +9.9 | 48.3 | +1.3 | 40.8 | +3.9 | +4.16 |
| gen_step1x_new | Generative | 46.3 | +2.2 | 39.3 | +1.2 | 48.6 | +1.6 | 48.9 | +12.0 | +4.26 |
| gen_step1x_v1p2 | Generative | 46.1 | +2.0 | 39.7 | +1.6 | 48.0 | +1.0 | 47.9 | +11.0 | +3.91 |
| gen_Qwen_Image_Edit | Generative | 45.9 | +1.8 | 39.8 | +1.7 | 52.0 | +5.0 | 43.4 | +6.5 | +3.75 |
| std_autoaugment | Standard Aug | 46.3 | +2.2 | 39.8 | +1.8 | 47.5 | +0.5 | 48.6 | +11.7 | +4.03 |
| gen_stargan_v2 | Generative | 46.1 | +2.1 | 39.5 | +1.5 | 48.1 | +1.1 | 43.0 | +6.1 | +2.69 |
| gen_Attribute_Hallucination | Generative | 46.5 | +2.4 | 40.8 | +2.7 | 48.2 | +1.2 | 38.0 | +1.1 | +1.86 |
| gen_cycleGAN | Generative | 46.0 | +1.9 | 40.0 | +2.0 | 48.0 | +1.0 | 37.9 | +1.0 | +1.47 |
| gen_flux_kontext | Generative | 46.1 | +2.1 | 39.3 | +1.2 | 48.1 | +1.1 | 38.2 | +1.3 | +1.41 |
| gen_automold | Generative | 46.4 | +2.3 | 39.6 | +1.6 | 48.2 | +1.2 | 37.2 | +0.3 | +1.33 |
| gen_CNetSeg | Generative | 46.0 | +1.9 | 40.6 | +2.5 | 48.4 | +1.4 | 37.1 | +0.2 | +1.50 |
| gen_IP2P | Generative | 46.3 | +2.2 | 40.6 | +2.5 | 48.2 | +1.2 | 36.7 | -0.1 | +1.44 |
| gen_albumentations_weather | Generative | 46.3 | +2.2 | 39.8 | +1.7 | 48.1 | +1.1 | 36.9 | -0.0 | +1.25 |
| gen_SUSTechGAN | Generative | 46.2 | +2.2 | 39.5 | +1.5 | 47.7 | +0.7 | 37.3 | +0.4 | +1.18 |
| gen_CUT | Generative | 45.9 | +1.8 | 39.5 | +1.4 | 48.1 | +1.1 | 37.2 | +0.3 | +1.15 |
| gen_Img2Img | Generative | 46.3 | +2.2 | 39.9 | +1.8 | 47.0 | +0.0 | 37.3 | +0.4 | +1.12 |
| gen_TSIT | Generative | 45.8 | +1.7 | 39.4 | +1.3 | 48.3 | +1.3 | 36.7 | -0.2 | +1.04 |
| gen_VisualCloze | Generative | 45.9 | +1.8 | 39.5 | +1.5 | 47.1 | +0.1 | 37.6 | +0.8 | +1.03 |
| gen_Weather_Effect_Generator | Generative | 46.2 | +2.1 | 39.5 | +1.5 | 49.1 | +2.1 | 37.4 | +0.5 | +1.55 |
| gen_UniControl | Generative | 46.0 | +1.9 | 39.3 | +1.2 | 47.6 | +0.6 | 37.1 | +0.2 | +0.98 |
| gen_LANIT | Generative | 45.5 | +1.4 | 39.4 | +1.3 | 47.3 | +0.3 | 37.1 | +0.2 | +0.80 |
| gen_augmenters | Generative | 45.9 | +1.8 | 40.0 | +1.9 | 46.1 | -0.9 | 37.1 | +0.2 | +0.76 |
| std_mixup | Standard Aug | 46.0 | +1.9 | 39.7 | +1.7 | 47.9 | +0.9 | 37.6 | +0.7 | +1.31 |
| gen_cyclediffusion | Generative | 46.3 | +2.2 | 39.8 | +1.8 | 51.5 | +4.5 | 37.4 | +0.5 | +2.23 |
| std_cutmix | Standard Aug | 46.5 | +2.4 | 40.1 | +2.0 | 47.7 | +0.7 | 36.4 | -0.5 | +1.16 |
| baseline | Baseline | 44.1 | +0.0 | 38.1 | +0.0 | 47.0 | +0.0 | 36.9 | +0.0 | +0.00 |
| gen_EDICT | Generative | 41.0 | -3.1 | - | - | - | - | - | - | -3.05 |

## Per-Domain mIoU by Strategy

| Strategy                     | Type         |   clear_day |   cloudy |   dawn_dusk |   foggy |   night |   rainy |   snowy |   Normal Avg |   Adverse Avg |   Gap |
|:-----------------------------|:-------------|------------:|---------:|------------:|--------:|--------:|--------:|--------:|-------------:|--------------:|------:|
| baseline                     | Baseline     |       44.16 |    39.37 |       33.18 |   40.69 |   28.99 |   35.41 |   34.81 |        41.77 |         34.98 |  6.79 |
| gen_Attribute_Hallucination  | Generative   |       45.76 |    42.06 |       35.99 |   43.54 |   31.39 |   37.68 |   37.02 |        43.91 |         37.41 |  6.5  |
| gen_CNetSeg                  | Generative   |       45.59 |    41.59 |       35.46 |   43.89 |   31.63 |   37.34 |   36.53 |        43.59 |         37.35 |  6.25 |
| gen_CUT                      | Generative   |       45.27 |    41.42 |       34.79 |   44.38 |   30.68 |   37.23 |   37.16 |        43.34 |         37.36 |  5.98 |
| gen_EDICT                    | Generative   |       43.12 |    45.33 |       35.27 |   55.22 |   19.94 |   38.93 |   46.33 |        44.22 |         40.11 |  4.12 |
| gen_IP2P                     | Generative   |       45.41 |    41.39 |       35.52 |   42.3  |   30.72 |   37.48 |   37.02 |        43.4  |         36.88 |  6.52 |
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
| gen_cycleGAN                 | Generative   |       45.7  |    41.54 |       35.6  |   43.58 |   30.88 |   37.71 |   37.42 |        43.62 |         37.4  |  6.22 |
| gen_cyclediffusion           | Generative   |       45.17 |    40.94 |       33.37 |   41.98 |   28.48 |   37.09 |   36.28 |        43.06 |         35.95 |  7.1  |
| gen_flux_kontext             | Generative   |       45.52 |    41.77 |       35.11 |   43.6  |   30.78 |   37.76 |   37.44 |        43.65 |         37.39 |  6.25 |
| gen_stargan_v2               | Generative   |       46.67 |    43.21 |       37.33 |   45.3  |   32.3  |   39.17 |   39.4  |        44.94 |         39.04 |  5.9  |
| gen_step1x_new               | Generative   |       47.99 |    44.58 |       37.29 |   46.72 |   33.29 |   40.43 |   41.58 |        46.28 |         40.51 |  5.78 |
| gen_step1x_v1p2              | Generative   |       47.58 |    44.2  |       37.8  |   46.23 |   33.08 |   40.29 |   41.15 |        45.89 |         40.19 |  5.7  |
| photometric_distort          | Augmentation |       49.4  |    44.66 |       37.52 |   44.31 |   34.47 |   38.81 |   37.15 |        47.03 |         38.69 |  8.34 |
| std_autoaugment              | Standard Aug |       47.31 |    42.51 |       35.92 |   43.46 |   31.55 |   37.89 |   38.1  |        44.91 |         37.75 |  7.16 |
| std_cutmix                   | Standard Aug |       45.21 |    40.24 |       34.82 |   40.49 |   29.96 |   35.47 |   34.88 |        42.73 |         35.2  |  7.53 |
| std_mixup                    | Standard Aug |       45.38 |    40.21 |       34.58 |   40.02 |   30.02 |   35.94 |   34.8  |        42.79 |         35.2  |  7.6  |
| std_randaugment              | Standard Aug |       49.91 |    44.64 |       38.24 |   44.08 |   33.83 |   39.21 |   37.69 |        47.27 |         38.7  |  8.57 |