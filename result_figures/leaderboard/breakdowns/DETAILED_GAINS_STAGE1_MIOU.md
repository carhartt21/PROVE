# Stage 1 Detailed Per-Dataset and Per-Domain Analysis (mIoU)

## Per-Dataset mIoU by Strategy

| Strategy | Type | bdd10k | Δbdd10k | iddaw | Δiddaw | mapillaryvistas | Δmapillaryvistas | outside15k | Δoutside15k | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gen_UniControl | Generative | 47.2 | +4.9 | 38.1 | +5.2 | 33.0 | +6.4 | 40.5 | +7.6 | +6.02 |
| gen_Img2Img | Generative | 46.9 | +4.6 | 38.1 | +5.2 | 32.9 | +6.3 | 40.4 | +7.5 | +5.89 |
| gen_Attribute_Hallucination | Generative | 46.7 | +4.4 | 37.7 | +4.8 | 32.9 | +6.3 | 40.9 | +8.0 | +5.88 |
| gen_Qwen_Image_Edit | Generative | 46.4 | +4.1 | 37.8 | +4.9 | 32.9 | +6.2 | 40.9 | +8.0 | +5.81 |
| gen_stargan_v2 | Generative | 46.5 | +4.3 | 37.7 | +4.7 | 32.7 | +6.1 | 41.0 | +8.1 | +5.81 |
| gen_CNetSeg | Generative | 46.6 | +4.3 | 37.7 | +4.8 | 32.9 | +6.2 | 40.8 | +7.9 | +5.81 |
| gen_IP2P | Generative | 46.3 | +4.0 | 37.8 | +4.9 | 32.9 | +6.2 | 40.7 | +7.8 | +5.73 |
| gen_augmenters | Generative | 46.1 | +3.8 | 37.7 | +4.8 | 32.9 | +6.2 | 41.0 | +8.1 | +5.75 |
| std_randaugment | Standard Aug | 46.2 | +3.9 | 37.7 | +4.8 | 32.8 | +6.2 | 40.9 | +8.0 | +5.73 |
| gen_cyclediffusion | Generative | 46.1 | +3.8 | 38.0 | +5.1 | 32.7 | +6.0 | 40.6 | +7.8 | +5.67 |
| std_mixup | Standard Aug | 46.4 | +4.1 | 37.7 | +4.8 | 32.8 | +6.2 | 40.4 | +7.6 | +5.66 |
| gen_VisualCloze | Generative | 45.7 | +3.5 | 38.0 | +5.1 | 32.9 | +6.3 | 40.5 | +7.6 | +5.62 |
| gen_CUT | Generative | 46.0 | +3.7 | 37.7 | +4.8 | 32.7 | +6.0 | 40.6 | +7.7 | +5.56 |
| gen_Weather_Effect_Generator | Generative | 45.8 | +3.5 | 37.6 | +4.7 | 33.0 | +6.4 | 40.5 | +7.7 | +5.56 |
| std_autoaugment | Standard Aug | 46.4 | +4.1 | 36.9 | +4.0 | 32.8 | +6.2 | 40.7 | +7.9 | +5.54 |
| std_cutmix | Standard Aug | 46.3 | +4.0 | 37.1 | +4.1 | 32.9 | +6.3 | 40.5 | +7.7 | +5.51 |
| gen_flux_kontext | Generative | 44.5 | +2.3 | 37.8 | +4.9 | 33.0 | +6.3 | 40.9 | +8.0 | +5.37 |
| gen_SUSTechGAN | Generative | 44.8 | +2.6 | 37.8 | +4.9 | 32.8 | +6.1 | 40.4 | +7.6 | +5.29 |
| gen_step1x_new | Generative | 44.7 | +2.4 | 37.9 | +4.9 | 32.9 | +6.3 | 40.4 | +7.6 | +5.31 |
| gen_TSIT | Generative | 46.6 | +4.3 | 37.7 | +4.8 | 30.1 | +3.4 | 40.7 | +7.8 | +5.08 |
| gen_automold | Generative | 46.1 | +3.8 | 37.8 | +4.9 | 32.9 | +6.3 | 40.6 | +7.7 | +5.66 |
| gen_LANIT | Generative | 44.6 | +2.3 | 36.9 | +4.0 | 32.9 | +6.2 | 40.7 | +7.8 | +5.07 |
| gen_cycleGAN | Generative | 43.3 | +1.0 | 37.9 | +4.9 | 32.9 | +6.3 | 40.6 | +7.8 | +4.98 |
| gen_albumentations_weather | Generative | 45.0 | +2.7 | 37.7 | +4.8 | 32.8 | +6.1 | 40.4 | +7.5 | +5.29 |
| gen_step1x_v1p2 | Generative | 43.0 | +0.8 | 37.8 | +4.8 | 32.8 | +6.2 | 40.5 | +7.7 | +4.87 |
| baseline | Baseline | 42.3 | +0.0 | 32.9 | +0.0 | 26.6 | +0.0 | 32.9 | +0.0 | +0.00 |

## Per-Domain mIoU by Strategy

| Strategy                     | Type         |   clear_day |   cloudy |   dawn_dusk |   foggy |   night |   rainy |   snowy |   Normal Avg |   Adverse Avg |   Gap |
|:-----------------------------|:-------------|------------:|---------:|------------:|--------:|--------:|--------:|--------:|-------------:|--------------:|------:|
| baseline                     | Baseline     |       35.42 |    33.05 |       28.25 |   32.55 |   23.8  |   30.02 |   29.17 |        34.24 |         28.89 |  5.35 |
| gen_Attribute_Hallucination  | Generative   |       41.97 |    39    |       33.6  |   37.92 |   28.23 |   35.88 |   36.01 |        40.49 |         34.51 |  5.97 |
| gen_CNetSeg                  | Generative   |       41.66 |    39.41 |       33.45 |   37.95 |   28.07 |   35.94 |   36.23 |        40.53 |         34.55 |  5.99 |
| gen_CUT                      | Generative   |       41.61 |    39.05 |       33.39 |   38.05 |   28.6  |   35.39 |   35.49 |        40.33 |         34.38 |  5.95 |
| gen_IP2P                     | Generative   |       41.74 |    39.25 |       33.42 |   38.12 |   28.67 |   35.79 |   35.82 |        40.5  |         34.6  |  5.9  |
| gen_Img2Img                  | Generative   |       41.88 |    39.61 |       33.64 |   37.88 |   28.19 |   36.49 |   35.79 |        40.74 |         34.58 |  6.16 |
| gen_LANIT                    | Generative   |       41.09 |    38.64 |       32.76 |   37.69 |   27.52 |   34.75 |   35.07 |        39.86 |         33.76 |  6.11 |
| gen_Qwen_Image_Edit          | Generative   |       41.77 |    39.26 |       33.73 |   38.11 |   28.63 |   35.84 |   35.44 |        40.51 |         34.51 |  6.01 |
| gen_SUSTechGAN               | Generative   |       41.28 |    38.6  |       33.09 |   37.5  |   27.86 |   35.24 |   35.63 |        39.94 |         34.06 |  5.88 |
| gen_TSIT                     | Generative   |       41.17 |    38.87 |       32.72 |   37.24 |   28.27 |   35.1  |   35.15 |        40.02 |         33.94 |  6.08 |
| gen_UniControl               | Generative   |       41.93 |    39.31 |       33.31 |   38.59 |   28.49 |   36.54 |   35.98 |        40.62 |         34.9  |  5.72 |
| gen_VisualCloze              | Generative   |       41.7  |    39.29 |       33.57 |   38.7  |   28.57 |   35.58 |   35.43 |        40.5  |         34.57 |  5.93 |
| gen_Weather_Effect_Generator | Generative   |       41.59 |    39.36 |       33.86 |   37.86 |   28.37 |   35.55 |   35.62 |        40.47 |         34.35 |  6.12 |
| gen_albumentations_weather   | Generative   |       40.92 |    37.91 |       33.18 |   37.29 |   28.08 |   34.59 |   33.74 |        39.42 |         33.42 |  5.99 |
| gen_augmenters               | Generative   |       41.72 |    39.06 |       33.75 |   37.71 |   28.35 |   35.73 |   36.07 |        40.39 |         34.46 |  5.92 |
| gen_automold                 | Generative   |       41.25 |    38.46 |       32.64 |   38.03 |   28.51 |   34.67 |   34.19 |        39.85 |         33.85 |  6    |
| gen_cycleGAN                 | Generative   |       40.92 |    38.61 |       33.25 |   38.08 |   27.95 |   34.62 |   34.53 |        39.76 |         33.79 |  5.97 |
| gen_cyclediffusion           | Generative   |       41.67 |    39.44 |       33.59 |   38.51 |   28.73 |   35.51 |   35.75 |        40.56 |         34.63 |  5.93 |
| gen_flux_kontext             | Generative   |       41.34 |    39    |       33.74 |   37.93 |   28.22 |   35.44 |   35.05 |        40.17 |         34.16 |  6.01 |
| gen_stargan_v2               | Generative   |       41.76 |    39.33 |       33.41 |   38.51 |   27.94 |   35.83 |   35.89 |        40.55 |         34.54 |  6    |
| gen_step1x_new               | Generative   |       41.28 |    39    |       33.14 |   38.4  |   27.89 |   35.09 |   35.1  |        40.14 |         34.12 |  6.02 |
| gen_step1x_v1p2              | Generative   |       40.48 |    37.75 |       33.01 |   37.35 |   27.78 |   34.08 |   33.54 |        39.12 |         33.19 |  5.93 |
| std_autoaugment              | Standard Aug |       41.44 |    38.86 |       33.24 |   37.97 |   28.24 |   35.8  |   35.41 |        40.15 |         34.35 |  5.79 |
| std_cutmix                   | Standard Aug |       41.45 |    39.17 |       33.3  |   37.53 |   28.18 |   35.8  |   35.59 |        40.31 |         34.27 |  6.04 |
| std_mixup                    | Standard Aug |       41.65 |    39.21 |       33.42 |   38.11 |   28.07 |   35.89 |   35.89 |        40.43 |         34.49 |  5.94 |
| std_randaugment              | Standard Aug |       41.75 |    39.12 |       33.25 |   37.87 |   27.9  |   35.76 |   35.77 |        40.44 |         34.33 |  6.11 |