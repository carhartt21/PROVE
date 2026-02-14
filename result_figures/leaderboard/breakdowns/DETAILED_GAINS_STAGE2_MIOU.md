# Stage 2 Detailed Per-Dataset and Per-Domain Analysis (mIoU)

## Per-Dataset mIoU by Strategy

| Strategy | Type | bdd10k | Δbdd10k | iddaw | Δiddaw | mapillaryvistas | Δmapillaryvistas | outside15k | Δoutside15k | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gen_LANIT | Generative | 46.6 | +2.1 | 40.6 | +1.4 | 34.7 | -1.1 | 45.1 | +2.8 | +1.28 |
| gen_step1x_new | Generative | 46.4 | +1.8 | 39.8 | +0.5 | 35.2 | -0.7 | 43.1 | +0.9 | +0.63 |
| gen_flux_kontext | Generative | 46.3 | +1.8 | 39.7 | +0.4 | 35.2 | -0.7 | 43.2 | +1.0 | +0.63 |
| gen_UniControl | Generative | 46.5 | +1.9 | 39.9 | +0.6 | 35.2 | -0.7 | 42.6 | +0.4 | +0.56 |
| gen_Qwen_Image_Edit | Generative | 46.1 | +1.6 | 39.8 | +0.5 | 35.2 | -0.6 | 42.7 | +0.5 | +0.49 |
| gen_albumentations_weather | Generative | 47.8 | +3.3 | 40.6 | +1.3 | 35.2 | -0.7 | 43.1 | +0.9 | +1.21 |
| baseline | Baseline | 44.5 | +0.0 | 39.3 | +0.0 | 35.9 | +0.0 | 42.2 | +0.0 | +0.00 |
| std_autoaugment | Standard Aug | 46.4 | +1.9 | 39.7 | +0.4 | 35.2 | -0.6 | 42.0 | -0.2 | +0.37 |
| gen_Img2Img | Generative | 46.5 | +2.0 | 39.8 | +0.5 | 35.2 | -0.6 | 41.6 | -0.7 | +0.29 |
| gen_CUT | Generative | 46.0 | +1.5 | 39.8 | +0.5 | 35.2 | -0.6 | 41.6 | -0.6 | +0.20 |
| gen_augmenters | Generative | 46.4 | +1.9 | 37.0 | -2.2 | 35.2 | -0.6 | 43.1 | +0.8 | -0.04 |
| std_cutmix | Standard Aug | 47.3 | +2.8 | 37.0 | -2.3 | 35.2 | -0.7 | 42.2 | +0.0 | -0.04 |
| std_randaugment | Standard Aug | 46.0 | +1.5 | 38.9 | -0.4 | 35.2 | -0.7 | 42.6 | +0.3 | +0.18 |
| gen_cycleGAN | Generative | 46.9 | +2.3 | 36.9 | -2.4 | 35.1 | -0.7 | 42.5 | +0.3 | -0.13 |
| gen_IP2P | Generative | 47.7 | +3.2 | 40.5 | +1.3 | 35.2 | -0.7 | 43.1 | +0.8 | +1.15 |
| gen_SUSTechGAN | Generative | 46.4 | +1.8 | 40.7 | +1.4 | 35.1 | -0.8 | 43.3 | +1.1 | +0.87 |
| std_mixup | Standard Aug | 45.8 | +1.2 | 38.8 | -0.4 | 35.3 | -0.5 | 41.0 | -1.2 | -0.24 |
| gen_VisualCloze | Generative | 47.2 | +2.6 | 40.7 | +1.4 | 35.2 | -0.6 | 41.8 | -0.5 | +0.75 |
| gen_step1x_v1p2 | Generative | 46.4 | +1.9 | 40.5 | +1.2 | 35.0 | -0.8 | 42.2 | -0.1 | +0.55 |
| gen_cyclediffusion | Generative | 46.9 | +2.4 | 40.7 | +1.4 | 35.1 | -0.8 | 41.7 | -0.5 | +0.62 |
| gen_automold | Generative | 47.1 | +2.6 | 40.6 | +1.3 | 35.2 | -0.6 | 41.4 | -0.8 | +0.60 |
| gen_Attribute_Hallucination | Generative | 46.7 | +2.2 | 40.6 | +1.3 | 35.2 | -0.7 | 41.4 | -0.8 | +0.51 |
| gen_stargan_v2 | Generative | - | - | - | - | 35.2 | -0.7 | 43.2 | +1.0 | +0.15 |
| gen_CNetSeg | Generative | - | - | - | - | 35.0 | -0.8 | 43.3 | +1.1 | +0.13 |
| gen_Weather_Effect_Generator | Generative | - | - | - | - | 35.1 | -0.8 | 43.0 | +0.7 | -0.02 |
| gen_TSIT | Generative | - | - | - | - | 35.2 | -0.7 | 42.2 | -0.0 | -0.35 |

## Per-Domain mIoU by Strategy

| Strategy                     | Type         |   clear_day |   cloudy |   dawn_dusk |   foggy |   night |   rainy |   snowy |   Normal Avg |   Adverse Avg |   Gap |
|:-----------------------------|:-------------|------------:|---------:|------------:|--------:|--------:|--------:|--------:|-------------:|--------------:|------:|
| baseline                     | Baseline     |       41.34 |    41.19 |       36.38 |   46.56 |   32.56 |   38.94 |   40.28 |        41.34 |         39.32 |  2.02 |
| gen_Attribute_Hallucination  | Generative   |       39.74 |    38.64 |       35.5  |   43.34 |   32.16 |   37.2  |   37.39 |        39.74 |         37.37 |  2.37 |
| gen_CNetSeg                  | Generative   |       39.25 |    37.57 |       36    |   43.04 |   31.84 |   37.08 |   35.58 |        39.25 |         36.85 |  2.4  |
| gen_CUT                      | Generative   |       41.06 |    40.59 |       36.97 |   45.14 |   33.15 |   39.12 |   40.61 |        41.06 |         39.26 |  1.8  |
| gen_IP2P                     | Generative   |       40.47 |    39.34 |       36.04 |   45.06 |   32.45 |   37.6  |   38.11 |        40.47 |         38.1  |  2.37 |
| gen_Img2Img                  | Generative   |       41.25 |    40.82 |       36.95 |   44.81 |   33.2  |   38.94 |   40.71 |        41.25 |         39.24 |  2.01 |
| gen_LANIT                    | Generative   |       42.23 |    41.36 |       37.42 |   46.62 |   32.6  |   39.72 |   41.64 |        42.23 |         39.89 |  2.33 |
| gen_Qwen_Image_Edit          | Generative   |       41.44 |    41    |       37.33 |   45.23 |   32.95 |   38.99 |   40.25 |        41.44 |         39.29 |  2.14 |
| gen_SUSTechGAN               | Generative   |       40.48 |    38.83 |       35.66 |   43.79 |   32.42 |   37.74 |   37.59 |        40.48 |         37.67 |  2.81 |
| gen_TSIT                     | Generative   |       38.84 |    37.69 |       35.11 |   42.93 |   32.01 |   36.21 |   35.11 |        38.84 |         36.51 |  2.33 |
| gen_UniControl               | Generative   |       41.45 |    40.97 |       37.11 |   45.77 |   33.29 |   39.29 |   41.33 |        41.45 |         39.63 |  1.82 |
| gen_VisualCloze              | Generative   |       39.9  |    38.6  |       36.16 |   43.02 |   31.91 |   37.52 |   37.58 |        39.9  |         37.46 |  2.44 |
| gen_Weather_Effect_Generator | Generative   |       39.17 |    37.67 |       36.16 |   42.83 |   32.28 |   36.7  |   35.1  |        39.17 |         36.79 |  2.38 |
| gen_albumentations_weather   | Generative   |       41.04 |    40.02 |       36.12 |   44.91 |   32.74 |   39.09 |   39.4  |        41.04 |         38.71 |  2.32 |
| gen_augmenters               | Generative   |       40.8  |    40.23 |       36.17 |   44.64 |   32.25 |   38.9  |   39.87 |        40.8  |         38.68 |  2.13 |
| gen_automold                 | Generative   |       39.76 |    38.4  |       35.71 |   43.59 |   32.01 |   37.33 |   37.64 |        39.76 |         37.45 |  2.31 |
| gen_cycleGAN                 | Generative   |       40.71 |    40.61 |       36.33 |   44.87 |   32.09 |   38.77 |   40.35 |        40.71 |         38.84 |  1.88 |
| gen_cyclediffusion           | Generative   |       39.87 |    38.62 |       35.71 |   43.84 |   32.23 |   37.64 |   37.03 |        39.87 |         37.51 |  2.36 |
| gen_flux_kontext             | Generative   |       41.5  |    41.07 |       36.57 |   44.27 |   32.54 |   39.43 |   40.65 |        41.5  |         39.09 |  2.41 |
| gen_stargan_v2               | Generative   |       39.41 |    37.51 |       34.46 |   42.55 |   31.73 |   36.9  |   34.94 |        39.41 |         36.35 |  3.06 |
| gen_step1x_new               | Generative   |       41.57 |    41.27 |       37.43 |   45.81 |   33.02 |   39.35 |   40.93 |        41.57 |         39.64 |  1.93 |
| gen_step1x_v1p2              | Generative   |       39.94 |    38.65 |       36    |   44.13 |   32.32 |   37.4  |   37.21 |        39.94 |         37.62 |  2.32 |
| std_autoaugment              | Standard Aug |       41.2  |    40.71 |       37.14 |   45    |   32.99 |   39.21 |   40.65 |        41.2  |         39.28 |  1.92 |
| std_cutmix                   | Standard Aug |       40.76 |    40.27 |       36.58 |   44.25 |   32.49 |   38.9  |   40.15 |        40.76 |         38.77 |  1.99 |
| std_mixup                    | Standard Aug |       40.2  |    39.86 |       36.04 |   44.48 |   32.25 |   38.23 |   39.21 |        40.2  |         38.34 |  1.86 |
| std_randaugment              | Standard Aug |       40.63 |    40.06 |       36.59 |   44.49 |   32.45 |   38.7  |   39.75 |        40.63 |         38.67 |  1.96 |