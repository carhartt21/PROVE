# Stage 2 Detailed Per-Dataset and Per-Domain Analysis (mIoU)

## Per-Dataset mIoU by Strategy

| Strategy | Type | bdd10k | Δbdd10k | iddaw | Δiddaw | mapillaryvistas | Δmapillaryvistas | outside15k | Δoutside15k | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gen_LANIT | Generative | 46.6 | +2.1 | 40.6 | +1.4 | 34.7 | -1.1 | 45.1 | +2.8 | +1.28 |
| gen_step1x_new | Generative | 46.4 | +1.8 | 39.8 | +0.5 | 35.2 | -0.7 | 43.1 | +0.9 | +0.63 |
| gen_flux_kontext | Generative | 46.3 | +1.8 | 39.7 | +0.4 | 35.2 | -0.7 | 43.2 | +1.0 | +0.63 |
| gen_UniControl | Generative | 46.5 | +1.9 | 39.9 | +0.6 | 35.2 | -0.7 | 42.6 | +0.4 | +0.56 |
| gen_Qwen_Image_Edit | Generative | 46.1 | +1.6 | 39.8 | +0.5 | 35.2 | -0.6 | 42.7 | +0.5 | +0.49 |
| std_autoaugment | Standard Aug | 46.4 | +1.9 | 39.7 | +0.4 | 35.2 | -0.6 | 42.0 | -0.2 | +0.37 |
| baseline | Baseline | 44.5 | +0.0 | 39.3 | +0.0 | 35.9 | +0.0 | 42.2 | +0.0 | +0.00 |
| gen_Img2Img | Generative | 46.5 | +2.0 | 39.8 | +0.5 | 35.2 | -0.6 | 41.6 | -0.7 | +0.29 |
| gen_CUT | Generative | 46.0 | +1.5 | 39.8 | +0.5 | 35.2 | -0.6 | 41.6 | -0.6 | +0.20 |
| std_cutmix | Standard Aug | 47.3 | +2.8 | 37.0 | -2.3 | 35.2 | -0.7 | 42.2 | +0.0 | -0.04 |
| gen_augmenters | Generative | 46.4 | +1.9 | 37.0 | -2.2 | 35.2 | -0.6 | 43.1 | +0.8 | -0.04 |
| gen_cycleGAN | Generative | 46.9 | +2.3 | 36.9 | -2.4 | 35.1 | -0.7 | 42.5 | +0.3 | -0.13 |
| gen_albumentations_weather | Generative | 47.1 | +2.6 | 40.6 | +1.3 | 35.2 | -0.7 | 43.1 | +0.9 | +1.03 |
| std_mixup | Standard Aug | 45.8 | +1.2 | 38.8 | -0.4 | 35.3 | -0.5 | 41.0 | -1.2 | -0.24 |
| gen_VisualCloze | Generative | 47.2 | +2.6 | 40.7 | +1.4 | 35.2 | -0.6 | 44.9 | +2.6 | +1.52 |
| std_randaugment | Standard Aug | 44.9 | +0.3 | 37.9 | -1.4 | 35.2 | -0.7 | 42.6 | +0.3 | -0.36 |
| gen_step1x_v1p2 | Generative | 46.4 | +1.9 | 40.5 | +1.2 | 35.0 | -0.8 | 44.8 | +2.6 | +1.21 |
| gen_automold | Generative | 47.1 | +2.6 | 40.6 | +1.3 | 35.2 | -0.6 | 41.9 | -0.3 | +0.73 |
| gen_IP2P | Generative | 47.7 | +3.2 | 40.5 | +1.3 | 35.2 | -0.7 | 44.7 | +2.5 | +1.55 |
| gen_cyclediffusion | Generative | 46.9 | +2.4 | 40.7 | +1.4 | 33.2 | -2.6 | 44.8 | +2.6 | +0.93 |
| gen_Attribute_Hallucination | Generative | 46.7 | +2.2 | 40.6 | +1.3 | 33.2 | -2.6 | 45.0 | +2.7 | +0.91 |
| gen_SUSTechGAN | Generative | 46.4 | +1.8 | 40.7 | +1.4 | 33.2 | -2.7 | 44.8 | +2.5 | +0.76 |
| gen_stargan_v2 | Generative | - | - | - | - | 33.3 | -2.5 | - | - | -2.55 |
| gen_CNetSeg | Generative | - | - | - | - | 33.0 | -2.8 | - | - | -2.82 |

## Per-Domain mIoU by Strategy

| Strategy                    | Type         |   clear_day |   cloudy |   dawn_dusk |   foggy |   night |   rainy |   snowy |   Normal Avg |   Adverse Avg |   Gap |
|:----------------------------|:-------------|------------:|---------:|------------:|--------:|--------:|--------:|--------:|-------------:|--------------:|------:|
| baseline                    | Baseline     |       41.34 |    41.19 |       36.38 |   46.56 |   32.56 |   38.94 |   40.28 |        41.34 |         39.32 |  2.02 |
| gen_Attribute_Hallucination | Generative   |       38.85 |    38.35 |       36.63 |   41.93 |   31.53 |   36.87 |   38.27 |        38.85 |         37.26 |  1.59 |
| gen_CNetSeg                 | Generative   |       32.49 |    32.94 |       36.55 |   33.67 |   27.57 |   31.16 |   31.02 |        32.49 |         32.15 |  0.34 |
| gen_CUT                     | Generative   |       41.06 |    40.59 |       36.97 |   45.14 |   33.15 |   39.12 |   40.61 |        41.06 |         39.26 |  1.8  |
| gen_IP2P                    | Generative   |       39.16 |    38.62 |       37.89 |   42.85 |   32.03 |   37.09 |   38.69 |        39.16 |         37.86 |  1.3  |
| gen_Img2Img                 | Generative   |       41.25 |    40.82 |       36.95 |   44.81 |   33.2  |   38.94 |   40.71 |        41.25 |         39.24 |  2.01 |
| gen_LANIT                   | Generative   |       42.23 |    41.36 |       37.42 |   46.62 |   32.6  |   39.72 |   41.64 |        42.23 |         39.89 |  2.33 |
| gen_Qwen_Image_Edit         | Generative   |       41.44 |    41    |       37.33 |   45.23 |   32.95 |   38.99 |   40.25 |        41.44 |         39.29 |  2.14 |
| gen_SUSTechGAN              | Generative   |       38.83 |    37.95 |       36.24 |   42.1  |   30.8  |   36.6  |   38    |        38.83 |         36.95 |  1.88 |
| gen_UniControl              | Generative   |       41.45 |    40.97 |       37.11 |   45.77 |   33.29 |   39.29 |   41.33 |        41.45 |         39.63 |  1.82 |
| gen_VisualCloze             | Generative   |       39.96 |    38.87 |       37.47 |   42.43 |   32.36 |   37.82 |   38.68 |        39.96 |         37.94 |  2.02 |
| gen_albumentations_weather  | Generative   |       40.29 |    39    |       35.74 |   44.28 |   32.74 |   38.05 |   37.81 |        40.29 |         37.94 |  2.35 |
| gen_augmenters              | Generative   |       40.8  |    40.23 |       36.17 |   44.64 |   32.25 |   38.9  |   39.87 |        40.8  |         38.68 |  2.13 |
| gen_automold                | Generative   |       39.71 |    38.35 |       35.98 |   43.08 |   31.92 |   37    |   37.72 |        39.71 |         37.34 |  2.37 |
| gen_cycleGAN                | Generative   |       40.71 |    40.61 |       36.33 |   44.87 |   32.09 |   38.77 |   40.35 |        40.71 |         38.84 |  1.88 |
| gen_cyclediffusion          | Generative   |       38.9  |    38.34 |       36.83 |   41.94 |   31.17 |   36.91 |   37.7  |        38.9  |         37.15 |  1.75 |
| gen_flux_kontext            | Generative   |       41.5  |    41.07 |       36.57 |   44.27 |   32.54 |   39.43 |   40.65 |        41.5  |         39.09 |  2.41 |
| gen_stargan_v2              | Generative   |       32.91 |    32.87 |       35.67 |   34.27 |   28.09 |   31.36 |   30.93 |        32.91 |         32.2  |  0.71 |
| gen_step1x_new              | Generative   |       41.57 |    41.27 |       37.43 |   45.81 |   33.02 |   39.35 |   40.93 |        41.57 |         39.64 |  1.93 |
| gen_step1x_v1p2             | Generative   |       39.8  |    38.97 |       37.54 |   44.13 |   32.6  |   37.39 |   37.72 |        39.8  |         38.06 |  1.74 |
| std_autoaugment             | Standard Aug |       41.2  |    40.71 |       37.14 |   45    |   32.99 |   39.21 |   40.65 |        41.2  |         39.28 |  1.92 |
| std_cutmix                  | Standard Aug |       40.76 |    40.27 |       36.58 |   44.25 |   32.49 |   38.9  |   40.15 |        40.76 |         38.77 |  1.99 |
| std_mixup                   | Standard Aug |       40.2  |    39.86 |       36.04 |   44.48 |   32.25 |   38.23 |   39.21 |        40.2  |         38.34 |  1.86 |
| std_randaugment             | Standard Aug |       39.89 |    39.14 |       36.17 |   43.94 |   31.95 |   37.89 |   38.01 |        39.89 |         37.85 |  2.04 |