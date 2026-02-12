# Stage 1 Detailed Per-Dataset and Per-Domain Analysis (mIoU)

## Per-Dataset mIoU by Strategy

| Strategy | Type | bdd10k | Δbdd10k | iddaw | Δiddaw | mapillaryvistas | Δmapillaryvistas | outside15k | Δoutside15k | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gen_UniControl | Generative | 47.2 | +4.9 | 38.1 | +2.1 | 34.9 | +1.8 | 41.3 | +2.2 | +2.76 |
| gen_cyclediffusion | Generative | 46.1 | +3.8 | 38.0 | +2.0 | 34.6 | +1.5 | 41.8 | +2.7 | +2.51 |
| gen_Img2Img | Generative | 46.9 | +4.6 | 38.1 | +2.1 | 33.6 | +0.5 | 41.6 | +2.4 | +2.42 |
| std_randaugment | Standard Aug | 46.2 | +3.9 | 37.7 | +1.8 | 34.7 | +1.7 | 41.5 | +2.4 | +2.43 |
| std_autoaugment | Standard Aug | 46.4 | +4.1 | 36.9 | +1.0 | 34.7 | +1.7 | 41.9 | +2.7 | +2.37 |
| std_cutmix | Standard Aug | 46.3 | +4.0 | 37.1 | +1.1 | 34.7 | +1.6 | 41.6 | +2.5 | +2.30 |
| gen_Attribute_Hallucination | Generative | 46.7 | +4.4 | 37.7 | +1.8 | 33.5 | +0.5 | 41.6 | +2.5 | +2.28 |
| gen_IP2P | Generative | 46.3 | +4.0 | 37.8 | +1.8 | 33.5 | +0.5 | 41.8 | +2.7 | +2.24 |
| gen_Qwen_Image_Edit | Generative | 46.4 | +4.1 | 37.8 | +1.8 | 32.9 | -0.2 | 40.9 | +1.7 | +1.87 |
| gen_CNetSeg | Generative | 46.6 | +4.3 | 37.7 | +1.7 | 32.9 | -0.2 | 40.8 | +1.6 | +1.87 |
| gen_stargan_v2 | Generative | 46.5 | +4.3 | 37.7 | +1.7 | 32.7 | -0.3 | 41.0 | +1.9 | +1.87 |
| gen_SUSTechGAN | Generative | 44.8 | +2.6 | 37.8 | +1.8 | 34.7 | +1.6 | 41.9 | +2.8 | +2.21 |
| gen_flux_kontext | Generative | 44.5 | +2.3 | 37.8 | +1.8 | 34.9 | +1.9 | 42.0 | +2.9 | +2.20 |
| std_mixup | Standard Aug | 46.4 | +4.1 | 37.7 | +1.7 | 33.5 | +0.4 | 41.7 | +2.5 | +2.20 |
| gen_augmenters | Generative | 46.1 | +3.8 | 37.7 | +1.8 | 32.9 | -0.2 | 41.0 | +1.8 | +1.82 |
| gen_VisualCloze | Generative | 45.7 | +3.5 | 38.0 | +2.0 | 33.5 | +0.5 | 41.8 | +2.6 | +2.14 |
| gen_automold | Generative | 46.1 | +3.8 | 37.8 | +1.8 | 34.8 | +1.8 | 41.5 | +2.4 | +2.44 |
| gen_CUT | Generative | 46.0 | +3.7 | 37.7 | +1.7 | 33.3 | +0.3 | 41.5 | +2.4 | +2.03 |
| gen_Weather_Effect_Generator | Generative | 45.8 | +3.5 | 37.6 | +1.6 | 33.0 | -0.0 | 40.5 | +1.4 | +1.63 |
| gen_step1x_new | Generative | 44.7 | +2.4 | 37.9 | +1.9 | 33.5 | +0.5 | 41.7 | +2.5 | +1.82 |
| gen_albumentations_weather | Generative | 45.0 | +2.7 | 37.7 | +1.7 | 34.7 | +1.7 | 41.6 | +2.5 | +2.16 |
| gen_cycleGAN | Generative | 43.3 | +1.0 | 37.9 | +1.9 | 34.8 | +1.7 | 41.3 | +2.2 | +1.69 |
| gen_TSIT | Generative | 46.6 | +4.3 | 37.7 | +1.7 | 30.1 | -3.0 | 40.7 | +1.5 | +1.15 |
| gen_LANIT | Generative | 44.6 | +2.3 | 36.9 | +0.9 | 32.9 | -0.2 | 40.7 | +1.5 | +1.13 |
| gen_step1x_v1p2 | Generative | 43.0 | +0.8 | 37.8 | +1.8 | 33.5 | +0.5 | 41.8 | +2.7 | +1.43 |
| baseline | Baseline | 42.3 | +0.0 | 36.0 | +0.0 | 33.0 | -0.0 | 39.1 | -0.0 | -0.00 |

## Per-Domain mIoU by Strategy

| Strategy                     | Type         |   clear_day |   cloudy |   dawn_dusk |   foggy |   night |   rainy |   snowy |   Normal Avg |   Adverse Avg |   Gap |
|:-----------------------------|:-------------|------------:|---------:|------------:|--------:|--------:|--------:|--------:|-------------:|--------------:|------:|
| baseline                     | Baseline     |       39.42 |    36.91 |       32.16 |   36.81 |   26.87 |   34.02 |   33.05 |        38.17 |         32.69 |  5.48 |
| gen_Attribute_Hallucination  | Generative   |       41.79 |    39.07 |       34.15 |   38.23 |   28.69 |   36.17 |   35.81 |        40.43 |         34.72 |  5.71 |
| gen_CNetSeg                  | Generative   |       41.66 |    39.41 |       33.45 |   37.95 |   28.07 |   35.94 |   36.23 |        40.53 |         34.55 |  5.99 |
| gen_CUT                      | Generative   |       41.48 |    38.96 |       33.89 |   38.27 |   29.11 |   35.72 |   35.54 |        40.22 |         34.66 |  5.56 |
| gen_IP2P                     | Generative   |       41.68 |    39.24 |       33.81 |   38.2  |   29.07 |   36.14 |   35.73 |        40.46 |         34.79 |  5.67 |
| gen_Img2Img                  | Generative   |       41.85 |    39.39 |       34.01 |   37.89 |   28.68 |   36.58 |   35.74 |        40.62 |         34.72 |  5.9  |
| gen_LANIT                    | Generative   |       41.09 |    38.64 |       32.76 |   37.69 |   27.52 |   34.75 |   35.07 |        39.86 |         33.76 |  6.11 |
| gen_Qwen_Image_Edit          | Generative   |       41.77 |    39.26 |       33.73 |   38.11 |   28.63 |   35.84 |   35.44 |        40.51 |         34.51 |  6.01 |
| gen_SUSTechGAN               | Generative   |       41.67 |    38.87 |       34.05 |   38.37 |   28.9  |   36.01 |   36.26 |        40.27 |         34.89 |  5.38 |
| gen_TSIT                     | Generative   |       41.17 |    38.87 |       32.72 |   37.24 |   28.27 |   35.1  |   35.15 |        40.02 |         33.94 |  6.08 |
| gen_UniControl               | Generative   |       42.09 |    39.58 |       34.23 |   39.24 |   29.64 |   37.05 |   36.27 |        40.84 |         35.55 |  5.29 |
| gen_VisualCloze              | Generative   |       41.66 |    39.23 |       33.89 |   38.69 |   28.87 |   35.97 |   35.47 |        40.45 |         34.75 |  5.7  |
| gen_Weather_Effect_Generator | Generative   |       41.59 |    39.36 |       33.86 |   37.86 |   28.37 |   35.55 |   35.62 |        40.47 |         34.35 |  6.12 |
| gen_albumentations_weather   | Generative   |       41.31 |    38.42 |       34.1  |   38.06 |   29.13 |   35.49 |   34.44 |        39.87 |         34.28 |  5.59 |
| gen_augmenters               | Generative   |       41.72 |    39.06 |       33.75 |   37.71 |   28.35 |   35.73 |   36.07 |        40.39 |         34.46 |  5.92 |
| gen_automold                 | Generative   |       41.56 |    38.86 |       33.68 |   38.8  |   29.52 |   35.44 |   34.69 |        40.21 |         34.61 |  5.6  |
| gen_cycleGAN                 | Generative   |       41.22 |    38.71 |       33.93 |   38.76 |   28.84 |   35.33 |   34.8  |        39.96 |         34.43 |  5.53 |
| gen_cyclediffusion           | Generative   |       42    |    39.61 |       34.16 |   38.83 |   29.71 |   36.12 |   35.89 |        40.8  |         35.14 |  5.67 |
| gen_flux_kontext             | Generative   |       41.73 |    39.2  |       34.57 |   38.43 |   29.26 |   36.08 |   35.37 |        40.47 |         34.78 |  5.68 |
| gen_stargan_v2               | Generative   |       41.76 |    39.33 |       33.41 |   38.51 |   27.94 |   35.83 |   35.89 |        40.55 |         34.54 |  6    |
| gen_step1x_new               | Generative   |       41.32 |    38.91 |       33.41 |   38.39 |   28.43 |   35.38 |   35.02 |        40.11 |         34.31 |  5.81 |
| gen_step1x_v1p2              | Generative   |       40.68 |    37.77 |       33.39 |   37.66 |   28.35 |   34.63 |   33.97 |        39.22 |         33.65 |  5.57 |
| std_autoaugment              | Standard Aug |       41.81 |    39.08 |       33.83 |   38.6  |   28.9  |   36.28 |   35.76 |        40.45 |         34.88 |  5.56 |
| std_cutmix                   | Standard Aug |       41.67 |    39.6  |       33.87 |   38.19 |   29.19 |   36.49 |   36.05 |        40.64 |         34.98 |  5.66 |
| std_mixup                    | Standard Aug |       41.64 |    39.14 |       33.64 |   38.07 |   28.54 |   36.15 |   35.7  |        40.39 |         34.61 |  5.78 |
| std_randaugment              | Standard Aug |       41.93 |    39.48 |       34    |   38.57 |   28.9  |   36.21 |   36.03 |        40.71 |         34.93 |  5.78 |