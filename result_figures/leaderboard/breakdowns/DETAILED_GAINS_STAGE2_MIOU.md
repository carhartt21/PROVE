# Stage 2 Detailed Per-Dataset and Per-Domain Analysis (mIoU)

## Per-Dataset mIoU by Strategy

| Strategy | Type | bdd10k | Δbdd10k | iddaw | Δiddaw | mapillaryvistas | Δmapillaryvistas | outside15k | Δoutside15k | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gen_IP2P | Generative | 47.7 | +3.2 | 40.5 | +1.3 | 35.0 | -0.2 | 44.7 | +2.5 | +1.68 |
| gen_VisualCloze | Generative | 47.2 | +2.6 | 40.7 | +1.4 | 35.0 | -0.2 | 44.7 | +2.6 | +1.60 |
| gen_albumentations_weather | Generative | 47.1 | +2.6 | 40.6 | +1.3 | 35.1 | -0.2 | 44.7 | +2.5 | +1.57 |
| gen_cyclediffusion | Generative | 46.9 | +2.4 | 40.7 | +1.4 | 35.0 | -0.2 | 44.8 | +2.6 | +1.54 |
| gen_automold | Generative | 47.1 | +2.6 | 40.6 | +1.3 | 35.0 | -0.2 | 44.4 | +2.2 | +1.47 |
| gen_Attribute_Hallucination | Generative | 46.7 | +2.2 | 40.6 | +1.3 | 34.8 | -0.5 | 45.0 | +2.8 | +1.47 |
| gen_LANIT | Generative | 46.6 | +2.1 | 40.6 | +1.4 | 34.7 | -0.5 | 45.1 | +2.9 | +1.46 |
| gen_step1x_v1p2 | Generative | 46.4 | +1.9 | 40.5 | +1.2 | 35.0 | -0.3 | 44.9 | +2.7 | +1.39 |
| gen_SUSTechGAN | Generative | 46.4 | +1.8 | 40.7 | +1.4 | 34.9 | -0.3 | 44.8 | +2.6 | +1.37 |
| gen_step1x_new | Generative | 46.4 | +1.8 | 39.8 | +0.5 | 35.2 | -0.1 | 43.1 | +1.0 | +0.80 |
| gen_UniControl | Generative | 46.5 | +1.9 | 39.9 | +0.6 | 35.2 | -0.1 | 42.6 | +0.5 | +0.73 |
| gen_Qwen_Image_Edit | Generative | 46.1 | +1.6 | 39.8 | +0.5 | 35.2 | +0.0 | 42.7 | +0.5 | +0.67 |
| baseline | Baseline | 44.5 | +0.0 | 39.3 | +0.0 | 35.2 | +0.0 | 42.2 | +0.0 | +0.00 |
| gen_Img2Img | Generative | 46.5 | +2.0 | 39.8 | +0.5 | 35.2 | -0.0 | 41.6 | -0.6 | +0.46 |
| std_autoaugment | Standard Aug | 46.4 | +1.9 | 39.7 | +0.4 | 33.2 | -2.0 | 42.2 | +0.0 | +0.07 |
| gen_CUT | Generative | 46.0 | +1.5 | 39.8 | +0.5 | 35.2 | +0.0 | 41.6 | -0.5 | +0.38 |
| gen_flux_kontext | Generative | 46.3 | +1.8 | 38.8 | -0.5 | 33.3 | -1.9 | 42.2 | +0.0 | -0.16 |
| gen_augmenters | Generative | 46.4 | +1.9 | 37.0 | -2.2 | 35.2 | +0.0 | 43.1 | +0.9 | +0.14 |
| gen_cycleGAN | Generative | 45.7 | +1.1 | 38.7 | -0.5 | 33.1 | -2.1 | 41.9 | -0.3 | -0.45 |
| std_mixup | Standard Aug | 45.8 | +1.2 | 38.8 | -0.4 | 33.4 | -1.8 | 40.7 | -1.5 | -0.63 |
| std_cutmix | Standard Aug | 45.9 | +1.3 | 37.0 | -2.3 | 33.4 | -1.8 | 41.9 | -0.3 | -0.76 |
| std_randaugment | Standard Aug | 44.9 | +0.3 | 37.9 | -1.4 | 32.2 | -3.0 | 40.4 | -1.8 | -1.47 |

## Per-Domain mIoU by Strategy

| Strategy                    | Type         |   clear_day |   cloudy |   dawn_dusk |   foggy |   night |   rainy |   snowy |   Normal Avg |   Adverse Avg |   Gap |
|:----------------------------|:-------------|------------:|---------:|------------:|--------:|--------:|--------:|--------:|-------------:|--------------:|------:|
| baseline                    | Baseline     |       41.3  |    41.21 |       36.17 |   46.61 |   32.27 |   38.9  |   40.29 |        41.3  |         39.24 |  2.06 |
| gen_Attribute_Hallucination | Generative   |       42.33 |    41.37 |       36.55 |   46.29 |   33.52 |   39.84 |   42.14 |        42.33 |         39.95 |  2.38 |
| gen_CUT                     | Generative   |       41.06 |    40.59 |       36.97 |   45.14 |   33.15 |   39.12 |   40.61 |        41.06 |         39.26 |  1.8  |
| gen_IP2P                    | Generative   |       42.49 |    41.42 |       36.65 |   48.08 |   32.62 |   39.95 |   42.16 |        42.49 |         40.15 |  2.34 |
| gen_Img2Img                 | Generative   |       41.25 |    40.82 |       36.95 |   44.81 |   33.2  |   38.94 |   40.71 |        41.25 |         39.24 |  2.01 |
| gen_LANIT                   | Generative   |       42.23 |    41.36 |       37.42 |   46.62 |   32.6  |   39.72 |   41.64 |        42.23 |         39.89 |  2.33 |
| gen_Qwen_Image_Edit         | Generative   |       41.44 |    41    |       37.33 |   45.23 |   32.95 |   38.99 |   40.25 |        41.44 |         39.29 |  2.14 |
| gen_SUSTechGAN              | Generative   |       42.33 |    40.99 |       36.23 |   46.65 |   32.56 |   39.6  |   42.01 |        42.33 |         39.67 |  2.66 |
| gen_UniControl              | Generative   |       41.45 |    40.97 |       37.11 |   45.77 |   33.29 |   39.29 |   41.33 |        41.45 |         39.63 |  1.82 |
| gen_VisualCloze             | Generative   |       42.37 |    41.16 |       36.64 |   44.96 |   32.78 |   40.22 |   41.74 |        42.37 |         39.58 |  2.79 |
| gen_albumentations_weather  | Generative   |       42.1  |    41.64 |       36.4  |   47.53 |   33.63 |   39.92 |   41.73 |        42.1  |         40.14 |  1.96 |
| gen_augmenters              | Generative   |       40.8  |    40.23 |       36.17 |   44.64 |   32.25 |   38.9  |   39.87 |        40.8  |         38.68 |  2.13 |
| gen_automold                | Generative   |       42.38 |    41.41 |       36.63 |   46.19 |   32.6  |   39.26 |   41.92 |        42.38 |         39.67 |  2.71 |
| gen_cycleGAN                | Generative   |       40.23 |    40.04 |       35.43 |   44.3  |   31.18 |   38.18 |   39.27 |        40.23 |         38.07 |  2.17 |
| gen_cyclediffusion          | Generative   |       42.46 |    41.35 |       37.45 |   46.21 |   33.3  |   40.04 |   41.57 |        42.46 |         39.99 |  2.47 |
| gen_flux_kontext            | Generative   |       41    |    40.82 |       35.62 |   43.67 |   31    |   39.05 |   40.43 |        41    |         38.43 |  2.57 |
| gen_step1x_new              | Generative   |       41.57 |    41.27 |       37.43 |   45.81 |   33.02 |   39.35 |   40.93 |        41.57 |         39.64 |  1.93 |
| gen_step1x_v1p2             | Generative   |       42.2  |    41.51 |       37.41 |   47.21 |   32.9  |   39.58 |   40.78 |        42.2  |         39.9  |  2.3  |
| std_autoaugment             | Standard Aug |       41.14 |    40.84 |       36.62 |   45.03 |   32.3  |   39.29 |   40.82 |        41.14 |         39.15 |  1.99 |
| std_cutmix                  | Standard Aug |       39.71 |    39.19 |       35.44 |   43.26 |   31.43 |   37.78 |   38.63 |        39.71 |         37.62 |  2.08 |
| std_mixup                   | Standard Aug |       39.85 |    39.76 |       35.92 |   43.73 |   31.01 |   38.17 |   39.35 |        39.85 |         37.99 |  1.86 |
| std_randaugment             | Standard Aug |       39.16 |    38.89 |       34.64 |   42.65 |   29.85 |   37.24 |   37.8  |        39.16 |         36.85 |  2.31 |