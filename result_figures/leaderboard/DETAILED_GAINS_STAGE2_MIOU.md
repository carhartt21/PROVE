# Stage 2 Detailed Per-Dataset and Per-Domain Analysis (mIoU)

## Per-Dataset mIoU by Strategy

| Strategy | Type | bdd10k | Δbdd10k | idd-aw | Δidd-aw | mapillaryvistas | Δmapillaryvistas | outside15k | Δoutside15k | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| std_randaugment | Standard Aug | 47.7 | +3.2 | - | - | 35.1 | -0.1 | 44.7 | +2.5 | +1.85 |
| gen_IP2P | Generative | 47.7 | +3.2 | - | - | 35.0 | -0.2 | 44.7 | +2.5 | +1.82 |
| gen_VisualCloze | Generative | 47.2 | +2.6 | - | - | 35.0 | -0.2 | 44.7 | +2.6 | +1.66 |
| gen_albumentations_weather | Generative | 47.1 | +2.6 | - | - | 35.1 | -0.2 | 44.7 | +2.5 | +1.65 |
| gen_cyclediffusion | Generative | 46.9 | +2.4 | - | - | 35.0 | -0.2 | 44.8 | +2.6 | +1.59 |
| gen_flux_kontext | Generative | 47.0 | +2.5 | - | - | 35.0 | -0.3 | 44.9 | +2.8 | +1.67 |
| gen_Img2Img | Generative | 47.0 | +2.4 | - | - | 35.0 | -0.2 | 44.8 | +2.6 | +1.61 |
| gen_Attribute_Hallucination | Generative | 46.7 | +2.2 | - | - | 34.8 | -0.5 | 45.0 | +2.8 | +1.51 |
| gen_step1x_new | Generative | 46.4 | +1.9 | - | - | 35.1 | -0.1 | 45.1 | +2.9 | +1.55 |
| gen_UniControl | Generative | 46.4 | +1.9 | - | - | 35.0 | -0.3 | 45.0 | +2.8 | +1.49 |
| gen_automold | Generative | 47.1 | +2.6 | - | - | 35.0 | -0.2 | 44.4 | +2.2 | +1.53 |
| gen_LANIT | Generative | 46.6 | +2.1 | - | - | 34.7 | -0.5 | 45.1 | +2.9 | +1.49 |
| gen_CUT | Generative | 46.8 | +2.3 | - | - | 35.0 | -0.2 | 44.7 | +2.5 | +1.51 |
| gen_cycleGAN | Generative | 46.9 | +2.4 | - | - | 34.9 | -0.3 | 44.7 | +2.5 | +1.52 |
| gen_step1x_v1p2 | Generative | 46.4 | +1.9 | - | - | 35.0 | -0.3 | 44.9 | +2.7 | +1.44 |
| gen_SUSTechGAN | Generative | 46.4 | +1.8 | - | - | 34.9 | -0.3 | 44.8 | +2.6 | +1.36 |
| baseline | Baseline | 44.5 | +0.0 | 40.8 | +0.0 | 35.2 | +0.0 | 42.2 | +0.0 | +0.00 |
| std_autoaugment | Standard Aug | 45.9 | +1.4 | - | - | 33.2 | -2.0 | 42.2 | +0.0 | -0.20 |
| std_cutmix | Standard Aug | 45.9 | +1.3 | - | - | 33.4 | -1.8 | 41.9 | -0.3 | -0.26 |
| std_mixup | Standard Aug | 45.8 | +1.2 | - | - | 33.4 | -1.8 | 44.5 | +2.4 | +0.59 |

## Per-Domain mIoU by Strategy

| Strategy                    | Type         |   clear_day |   cloudy |   dawn_dusk |   foggy |   night |   rainy |   snowy |   Normal Avg |   Adverse Avg |   Gap |
|:----------------------------|:-------------|------------:|---------:|------------:|--------:|--------:|--------:|--------:|-------------:|--------------:|------:|
| baseline                    | Baseline     |       41.3  |    41.21 |       36.17 |   46.61 |   32.27 |   38.9  |   40.29 |        41.3  |         39.24 |  2.06 |
| gen_Attribute_Hallucination | Generative   |       42.33 |    41.37 |       36.55 |   46.29 |   33.52 |   39.84 |   42.14 |        42.33 |         39.95 |  2.38 |
| gen_CUT                     | Generative   |       42.22 |    41.45 |       37.4  |   46.86 |   33.38 |   39.8  |   41.72 |        42.22 |         40.1  |  2.11 |
| gen_IP2P                    | Generative   |       42.49 |    41.42 |       36.65 |   48.08 |   32.62 |   39.95 |   42.16 |        42.49 |         40.15 |  2.34 |
| gen_Img2Img                 | Generative   |       42.32 |    41.23 |       36.48 |   46.94 |   33.57 |   39.63 |   41.51 |        42.32 |         39.89 |  2.43 |
| gen_LANIT                   | Generative   |       42.23 |    41.36 |       37.42 |   46.62 |   32.6  |   39.72 |   41.64 |        42.23 |         39.89 |  2.33 |
| gen_SUSTechGAN              | Generative   |       42.33 |    40.99 |       36.23 |   46.65 |   32.56 |   39.6  |   42.01 |        42.33 |         39.67 |  2.66 |
| gen_UniControl              | Generative   |       42.19 |    41.76 |       36.4  |   46.57 |   32.88 |   39.8  |   42.25 |        42.19 |         39.94 |  2.24 |
| gen_VisualCloze             | Generative   |       42.37 |    41.16 |       36.64 |   44.96 |   32.78 |   40.22 |   41.74 |        42.37 |         39.58 |  2.79 |
| gen_albumentations_weather  | Generative   |       42.1  |    41.64 |       36.4  |   47.53 |   33.63 |   39.92 |   41.73 |        42.1  |         40.14 |  1.96 |
| gen_automold                | Generative   |       42.38 |    41.41 |       36.63 |   46.19 |   32.6  |   39.26 |   41.92 |        42.38 |         39.67 |  2.71 |
| gen_cycleGAN                | Generative   |       42.23 |    41.22 |       36.91 |   46.68 |   32.71 |   40.14 |   41.47 |        42.23 |         39.85 |  2.37 |
| gen_cyclediffusion          | Generative   |       42.46 |    41.35 |       37.45 |   46.21 |   33.3  |   40.04 |   41.57 |        42.46 |         39.99 |  2.47 |
| gen_flux_kontext            | Generative   |       42.3  |    41.38 |       36.76 |   45.1  |   32.38 |   40.1  |   41.57 |        42.3  |         39.55 |  2.75 |
| gen_step1x_new              | Generative   |       42.4  |    41.34 |       37.56 |   47.33 |   32.84 |   39.66 |   41.94 |        42.4  |         40.11 |  2.29 |
| gen_step1x_v1p2             | Generative   |       42.2  |    41.51 |       37.41 |   47.21 |   32.9  |   39.58 |   40.78 |        42.2  |         39.9  |  2.3  |
| std_autoaugment             | Standard Aug |       40.97 |    40.74 |       36.34 |   44.97 |   32.18 |   38.9  |   40.16 |        40.97 |         38.88 |  2.09 |
| std_cutmix                  | Standard Aug |       40.39 |    39.72 |       35.95 |   43.83 |   31.68 |   38.42 |   39.19 |        40.39 |         38.13 |  2.26 |
| std_mixup                   | Standard Aug |       40.02 |    40.32 |       36.73 |   44.15 |   31.52 |   38.55 |   40.31 |        40.02 |         38.6  |  1.43 |
| std_randaugment             | Standard Aug |       42.33 |    41.34 |       37.27 |   46.95 |   32.78 |   40.61 |   41.66 |        42.33 |         40.1  |  2.22 |