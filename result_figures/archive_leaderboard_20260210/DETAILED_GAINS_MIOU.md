# Detailed Per-Dataset and Per-Domain Analysis (mIoU)

## Per-Dataset mIoU by Strategy

| Strategy | Type | bdd10k | Δbdd10k | idd-aw | Δidd-aw | mapillaryvistas | Δmapillaryvistas | outside15k | Δoutside15k | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gen_Attribute_Hallucination | Generative | 49.1 | +0.2 | - | - | 34.9 | +7.2 | 43.1 | +6.3 | +4.56 |
| gen_UniControl | Generative | 48.8 | -0.1 | - | - | 35.1 | +7.4 | 43.1 | +6.2 | +4.50 |
| gen_VisualCloze | Generative | 48.9 | -0.0 | - | - | 34.8 | +7.1 | 42.7 | +5.8 | +4.30 |
| gen_automold | Generative | 48.5 | -0.4 | - | - | 35.0 | +7.3 | 43.1 | +6.3 | +4.37 |
| gen_CNetSeg | Generative | 48.7 | -0.2 | - | - | 35.0 | +7.3 | 43.2 | +6.3 | +4.48 |
| gen_Qwen_Image_Edit | Generative | 48.3 | -0.6 | - | - | 35.1 | +7.4 | 43.5 | +6.6 | +4.44 |
| gen_stargan_v2 | Generative | 48.6 | -0.3 | - | - | 34.6 | +6.9 | 43.4 | +6.6 | +4.38 |
| std_autoaugment | Standard Aug | 48.1 | -0.8 | - | - | 34.9 | +7.2 | 43.0 | +6.1 | +4.18 |
| gen_IP2P | Generative | 47.8 | -1.1 | - | - | 34.8 | +7.1 | 43.5 | +6.7 | +4.23 |
| gen_SUSTechGAN | Generative | 48.0 | -0.9 | - | - | 34.7 | +7.1 | 42.9 | +6.0 | +4.06 |
| gen_augmenters | Generative | 47.6 | -1.3 | - | - | 34.9 | +7.2 | 43.5 | +6.7 | +4.19 |
| std_mixup | Standard Aug | 47.9 | -1.0 | - | - | 34.9 | +7.2 | 42.8 | +6.0 | +4.07 |
| std_randaugment | Standard Aug | 47.6 | -1.3 | - | - | 34.7 | +7.0 | 43.2 | +6.3 | +4.03 |
| gen_Weather_Effect_Generator | Generative | 47.5 | -1.5 | - | - | 35.2 | +7.5 | 42.9 | +6.0 | +4.01 |
| gen_CUT | Generative | 47.2 | -1.7 | - | - | 34.9 | +7.2 | 43.1 | +6.2 | +3.90 |
| gen_cyclediffusion | Generative | 47.5 | -1.4 | - | - | 34.4 | +6.7 | 42.8 | +5.9 | +3.73 |
| std_cutmix | Standard Aug | 48.0 | -0.9 | - | - | 34.9 | +7.2 | 43.0 | +6.1 | +4.11 |
| gen_Img2Img | Generative | 46.3 | -2.6 | - | - | 34.8 | +7.2 | 43.0 | +6.1 | +3.55 |
| gen_albumentations_weather | Generative | 46.3 | -2.6 | - | - | 34.7 | +7.0 | 42.6 | +5.7 | +3.40 |
| gen_cycleGAN | Generative | 44.8 | -4.2 | - | - | 35.0 | +7.3 | 43.1 | +6.2 | +3.12 |
| gen_LANIT | Generative | 45.0 | -3.9 | - | - | 35.1 | +7.4 | 42.7 | +5.8 | +3.09 |
| gen_flux_kontext | Generative | 44.5 | -4.4 | - | - | 35.2 | +7.5 | 43.2 | +6.3 | +3.15 |
| gen_step1x_new | Generative | 44.8 | -4.1 | - | - | 35.0 | +7.3 | 42.5 | +5.6 | +2.94 |
| gen_TSIT | Generative | 48.4 | -0.5 | - | - | 26.8 | -0.9 | 43.0 | +6.1 | +1.58 |
| gen_step1x_v1p2 | Generative | 43.8 | -5.1 | - | - | 34.7 | +7.1 | 42.9 | +6.1 | +2.67 |
| baseline | Baseline | 48.9 | +0.0 | - | - | 27.7 | +0.0 | 36.9 | +0.0 | +0.00 |

## Per-Domain mIoU by Strategy

| Strategy                     | Type         |   clear_day |   cloudy |   dawn_dusk |   foggy |   night |   rainy |   snowy |   Normal Avg |   Adverse Avg |   Gap |
|:-----------------------------|:-------------|------------:|---------:|------------:|--------:|--------:|--------:|--------:|-------------:|--------------:|------:|
| baseline                     | Baseline     |       41.35 |    39.46 |       33.55 |   38.48 |   27.85 |   36.6  |   36.1  |        40.4  |         34.76 |  5.64 |
| gen_Attribute_Hallucination  | Generative   |       44.57 |    41.6  |       36.99 |   41.79 |   30.56 |   38.76 |   39.04 |        43.08 |         37.54 |  5.54 |
| gen_CNetSeg                  | Generative   |       44.21 |    42.17 |       36.26 |   42.05 |   30.35 |   38.5  |   39.28 |        43.19 |         37.55 |  5.64 |
| gen_CUT                      | Generative   |       43.97 |    41.51 |       36.18 |   42.04 |   30.72 |   37.57 |   38.1  |        42.74 |         37.11 |  5.63 |
| gen_IP2P                     | Generative   |       44.17 |    42.04 |       36.26 |   41.23 |   30.43 |   38.31 |   38.96 |        43.11 |         37.23 |  5.88 |
| gen_Img2Img                  | Generative   |       43.62 |    40.98 |       36.28 |   41.25 |   30.56 |   37.76 |   37.91 |        42.3  |         36.87 |  5.43 |
| gen_LANIT                    | Generative   |       43.15 |    41.12 |       35.76 |   41.36 |   29.81 |   37.58 |   37.3  |        42.13 |         36.51 |  5.62 |
| gen_Qwen_Image_Edit          | Generative   |       44.2  |    42.06 |       36.52 |   42    |   31.11 |   38.7  |   38.48 |        43.13 |         37.57 |  5.56 |
| gen_SUSTechGAN               | Generative   |       44.09 |    41.79 |       36.32 |   41.51 |   30    |   38.32 |   39.25 |        42.94 |         37.27 |  5.67 |
| gen_TSIT                     | Generative   |       42.81 |    40.83 |       34.65 |   39.81 |   30.21 |   36.97 |   37.78 |        41.82 |         36.19 |  5.63 |
| gen_UniControl               | Generative   |       44.47 |    41.94 |       36.02 |   42.96 |   30.58 |   38.76 |   38.84 |        43.2  |         37.78 |  5.42 |
| gen_VisualCloze              | Generative   |       44.48 |    42.11 |       36.12 |   43.2  |   31.65 |   38.82 |   38.99 |        43.3  |         38.16 |  5.13 |
| gen_Weather_Effect_Generator | Generative   |       43.87 |    42    |       36.65 |   41.2  |   30.33 |   38.55 |   38.52 |        42.94 |         37.15 |  5.79 |
| gen_albumentations_weather   | Generative   |       43.38 |    40.9  |       36.33 |   42.03 |   29.92 |   37.7  |   37.8  |        42.14 |         36.86 |  5.28 |
| gen_augmenters               | Generative   |       44.21 |    41.7  |       36.93 |   41.61 |   30.34 |   38    |   38.74 |        42.96 |         37.18 |  5.78 |
| gen_automold                 | Generative   |       44.31 |    42.15 |       35.81 |   41.91 |   30.58 |   38.58 |   38.91 |        43.23 |         37.5  |  5.74 |
| gen_cycleGAN                 | Generative   |       43.11 |    41.08 |       36    |   41.2  |   29.87 |   37.04 |   37.44 |        42.1  |         36.39 |  5.71 |
| gen_cyclediffusion           | Generative   |       43.75 |    42.09 |       36.23 |   42.83 |   30.77 |   38.06 |   38.29 |        42.92 |         37.49 |  5.43 |
| gen_flux_kontext             | Generative   |       43.19 |    40.86 |       36.21 |   40.51 |   30.02 |   37.53 |   36.8  |        42.03 |         36.22 |  5.81 |
| gen_stargan_v2               | Generative   |       44.27 |    42.26 |       35.8  |   41.92 |   30.35 |   38.6  |   38.44 |        43.26 |         37.33 |  5.94 |
| gen_step1x_new               | Generative   |       42.97 |    41.23 |       35.69 |   41.64 |   29.74 |   37.12 |   37.18 |        42.1  |         36.42 |  5.68 |
| gen_step1x_v1p2              | Generative   |       42.59 |    40.3  |       35.73 |   41.23 |   29.71 |   36.78 |   36.93 |        41.44 |         36.16 |  5.28 |
| std_autoaugment              | Standard Aug |       44.01 |    41.24 |       36.46 |   41.65 |   30.92 |   39.39 |   38.73 |        42.62 |         37.67 |  4.95 |
| std_cutmix                   | Standard Aug |       43.55 |    41.59 |       35.3  |   41.07 |   29.96 |   37.88 |   38.65 |        42.57 |         36.89 |  5.68 |
| std_mixup                    | Standard Aug |       44.06 |    41.91 |       36.17 |   41.21 |   29.83 |   38.83 |   38.73 |        42.99 |         37.15 |  5.84 |
| std_randaugment              | Standard Aug |       43.96 |    41.56 |       35.97 |   42    |   30.13 |   38.16 |   38.64 |        42.76 |         37.23 |  5.53 |