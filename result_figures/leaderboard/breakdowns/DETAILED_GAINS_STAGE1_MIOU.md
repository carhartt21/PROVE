# Stage 1 Detailed Per-Dataset and Per-Domain Analysis (mIoU)

## Per-Dataset mIoU by Strategy

| Strategy | Type | bdd10k | Δbdd10k | iddaw | Δiddaw | mapillaryvistas | Δmapillaryvistas | outside15k | Δoutside15k | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gen_automold | Generative | 46.2 | +3.9 | 37.8 | +1.8 | 34.8 | +1.8 | 41.5 | +2.4 | +2.48 |
| gen_UniControl | Generative | 47.2 | +4.9 | 38.1 | +2.1 | 34.9 | +1.8 | 41.3 | +2.2 | +2.76 |
| gen_albumentations_weather | Generative | 45.9 | +3.6 | 37.7 | +1.7 | 34.7 | +1.7 | 41.6 | +2.5 | +2.38 |
| gen_augmenters | Generative | 46.1 | +3.8 | 37.7 | +1.8 | 34.7 | +1.7 | 42.2 | +3.0 | +2.57 |
| gen_Qwen_Image_Edit | Generative | 46.4 | +4.1 | 37.8 | +1.8 | 34.8 | +1.7 | 41.5 | +2.4 | +2.51 |
| gen_cyclediffusion | Generative | 46.1 | +3.8 | 38.0 | +2.0 | 34.6 | +1.5 | 41.8 | +2.7 | +2.51 |
| gen_stargan_v2 | Generative | 46.5 | +4.3 | 37.7 | +1.7 | 34.7 | +1.6 | 41.5 | +2.3 | +2.48 |
| std_randaugment | Standard Aug | 46.2 | +3.9 | 37.7 | +1.8 | 34.7 | +1.7 | 41.5 | +2.4 | +2.43 |
| gen_Img2Img | Generative | 46.9 | +4.6 | 38.1 | +2.1 | 33.6 | +0.5 | 41.6 | +2.4 | +2.42 |
| std_autoaugment | Standard Aug | 46.4 | +4.1 | 36.9 | +1.0 | 34.7 | +1.7 | 41.9 | +2.7 | +2.37 |
| gen_CNetSeg | Generative | 46.6 | +4.3 | 37.7 | +1.7 | 33.5 | +0.5 | 42.0 | +2.8 | +2.33 |
| std_cutmix | Standard Aug | 46.3 | +4.0 | 37.1 | +1.1 | 34.7 | +1.6 | 41.6 | +2.5 | +2.30 |
| gen_Attribute_Hallucination | Generative | 46.7 | +4.4 | 37.7 | +1.8 | 33.5 | +0.5 | 41.6 | +2.5 | +2.28 |
| gen_IP2P | Generative | 46.3 | +4.0 | 37.8 | +1.8 | 33.5 | +0.5 | 41.8 | +2.7 | +2.24 |
| gen_SUSTechGAN | Generative | 44.8 | +2.6 | 37.8 | +1.8 | 34.7 | +1.6 | 41.9 | +2.8 | +2.21 |
| gen_flux_kontext | Generative | 44.5 | +2.3 | 37.8 | +1.8 | 34.9 | +1.9 | 42.0 | +2.9 | +2.20 |
| std_mixup | Standard Aug | 46.4 | +4.1 | 37.7 | +1.7 | 33.5 | +0.4 | 41.7 | +2.5 | +2.20 |
| gen_VisualCloze | Generative | 45.7 | +3.5 | 38.0 | +2.0 | 33.5 | +0.5 | 41.8 | +2.6 | +2.14 |
| gen_Weather_Effect_Generator | Generative | 45.8 | +3.5 | 37.6 | +1.6 | 33.5 | +0.5 | 41.7 | +2.5 | +2.04 |
| gen_CUT | Generative | 46.0 | +3.7 | 37.7 | +1.7 | 33.3 | +0.3 | 41.5 | +2.4 | +2.03 |
| gen_TSIT | Generative | 46.6 | +4.3 | 37.7 | +1.7 | 32.6 | -0.5 | 41.6 | +2.5 | +2.00 |
| gen_step1x_v1p2 | Generative | 44.1 | +1.8 | 37.8 | +1.8 | 33.5 | +0.5 | 41.8 | +2.7 | +1.70 |
| gen_step1x_new | Generative | 44.7 | +2.4 | 37.9 | +1.9 | 33.5 | +0.5 | 41.7 | +2.5 | +1.82 |
| gen_cycleGAN | Generative | 43.3 | +1.0 | 37.9 | +1.9 | 34.8 | +1.7 | 41.3 | +2.2 | +1.69 |
| gen_LANIT | Generative | 44.6 | +2.3 | 36.9 | +0.9 | 32.9 | -0.2 | 40.7 | +1.5 | +1.13 |
| baseline | Baseline | 42.3 | +0.0 | 36.0 | +0.0 | 33.0 | -0.0 | 39.1 | -0.0 | -0.00 |

## Per-Domain mIoU by Strategy

| Strategy                     | Type         |   clear_day |   cloudy |   dawn_dusk |   foggy |   night |   rainy |   snowy |   Normal Avg |   Adverse Avg |   Gap |
|:-----------------------------|:-------------|------------:|---------:|------------:|--------:|--------:|--------:|--------:|-------------:|--------------:|------:|
| baseline                     | Baseline     |       39.42 |    36.91 |       32.16 |   36.81 |   26.87 |   34.02 |   33.05 |        38.17 |         32.69 |  5.48 |
| gen_Attribute_Hallucination  | Generative   |       41.79 |    39.07 |       34.15 |   38.23 |   28.69 |   36.17 |   35.81 |        40.43 |         34.72 |  5.71 |
| gen_CNetSeg                  | Generative   |       41.65 |    39.44 |       33.82 |   38.26 |   28.74 |   36.22 |   36.07 |        40.54 |         34.82 |  5.72 |
| gen_CUT                      | Generative   |       41.48 |    38.96 |       33.89 |   38.27 |   29.11 |   35.72 |   35.54 |        40.22 |         34.66 |  5.56 |
| gen_IP2P                     | Generative   |       41.68 |    39.24 |       33.81 |   38.2  |   29.07 |   36.14 |   35.73 |        40.46 |         34.79 |  5.67 |
| gen_Img2Img                  | Generative   |       41.85 |    39.39 |       34.01 |   37.89 |   28.68 |   36.58 |   35.74 |        40.62 |         34.72 |  5.9  |
| gen_LANIT                    | Generative   |       41.09 |    38.64 |       32.76 |   37.69 |   27.52 |   34.75 |   35.07 |        39.86 |         33.76 |  6.11 |
| gen_Qwen_Image_Edit          | Generative   |       41.93 |    39.49 |       34.61 |   38.71 |   29.47 |   36.33 |   35.69 |        40.71 |         35.05 |  5.66 |
| gen_SUSTechGAN               | Generative   |       41.67 |    38.87 |       34.05 |   38.37 |   28.9  |   36.01 |   36.26 |        40.27 |         34.89 |  5.38 |
| gen_TSIT                     | Generative   |       41.46 |    39.18 |       33.56 |   38    |   29.09 |   35.8  |   35.38 |        40.32 |         34.57 |  5.75 |
| gen_UniControl               | Generative   |       42.09 |    39.58 |       34.23 |   39.24 |   29.64 |   37.05 |   36.27 |        40.84 |         35.55 |  5.29 |
| gen_VisualCloze              | Generative   |       41.66 |    39.23 |       33.89 |   38.69 |   28.87 |   35.97 |   35.47 |        40.45 |         34.75 |  5.7  |
| gen_Weather_Effect_Generator | Generative   |       41.53 |    39.28 |       33.94 |   38.02 |   28.93 |   35.74 |   35.57 |        40.4  |         34.57 |  5.84 |
| gen_albumentations_weather   | Generative   |       42.11 |    39.78 |       34.91 |   39.37 |   28.69 |   36.83 |   36.88 |        40.94 |         35.44 |  5.5  |
| gen_augmenters               | Generative   |       42.02 |    39.27 |       34.42 |   38.41 |   29.29 |   36.32 |   36.28 |        40.65 |         35.07 |  5.57 |
| gen_automold                 | Generative   |       42.42 |    39.97 |       34.37 |   40.12 |   29.12 |   36.46 |   36.44 |        41.19 |         35.53 |  5.66 |
| gen_cycleGAN                 | Generative   |       41.22 |    38.71 |       33.93 |   38.76 |   28.84 |   35.33 |   34.8  |        39.96 |         34.43 |  5.53 |
| gen_cyclediffusion           | Generative   |       42    |    39.61 |       34.16 |   38.83 |   29.71 |   36.12 |   35.89 |        40.8  |         35.14 |  5.67 |
| gen_flux_kontext             | Generative   |       41.73 |    39.2  |       34.57 |   38.43 |   29.26 |   36.08 |   35.37 |        40.47 |         34.78 |  5.68 |
| gen_stargan_v2               | Generative   |       41.93 |    39.43 |       33.92 |   39.17 |   28.76 |   36.31 |   35.83 |        40.68 |         35.02 |  5.66 |
| gen_step1x_new               | Generative   |       41.32 |    38.91 |       33.41 |   38.39 |   28.43 |   35.38 |   35.02 |        40.11 |         34.31 |  5.81 |
| gen_step1x_v1p2              | Generative   |       41.46 |    39.07 |       33.95 |   38.97 |   28.11 |   35.89 |   36.19 |        40.26 |         34.79 |  5.47 |
| std_autoaugment              | Standard Aug |       41.81 |    39.08 |       33.83 |   38.6  |   28.9  |   36.28 |   35.76 |        40.45 |         34.88 |  5.56 |
| std_cutmix                   | Standard Aug |       41.67 |    39.6  |       33.87 |   38.19 |   29.19 |   36.49 |   36.05 |        40.64 |         34.98 |  5.66 |
| std_mixup                    | Standard Aug |       41.64 |    39.14 |       33.64 |   38.07 |   28.54 |   36.15 |   35.7  |        40.39 |         34.61 |  5.78 |
| std_randaugment              | Standard Aug |       41.93 |    39.48 |       34    |   38.57 |   28.9  |   36.21 |   36.03 |        40.71 |         34.93 |  5.78 |