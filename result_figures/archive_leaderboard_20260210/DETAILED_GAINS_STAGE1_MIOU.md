# Stage 1 Detailed Per-Dataset and Per-Domain Analysis (mIoU)

## Per-Dataset mIoU by Strategy

| Strategy | Type | bdd10k | Δbdd10k | idd-aw | Δidd-aw | mapillaryvistas | Δmapillaryvistas | outside15k | Δoutside15k | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gen_Img2Img | Generative | 46.9 | +4.6 | - | - | 32.9 | +6.3 | 40.4 | +7.5 | +6.12 |
| gen_UniControl | Generative | 46.7 | +4.4 | - | - | 33.0 | +6.4 | 40.5 | +7.6 | +6.12 |
| gen_Attribute_Hallucination | Generative | 46.7 | +4.4 | - | - | 32.9 | +6.3 | 40.9 | +8.0 | +6.24 |
| gen_Qwen_Image_Edit | Generative | 46.4 | +4.1 | - | - | 32.9 | +6.2 | 40.9 | +8.0 | +6.11 |
| gen_stargan_v2 | Generative | 46.5 | +4.3 | - | - | 32.7 | +6.1 | 41.0 | +8.1 | +6.16 |
| gen_CNetSeg | Generative | 46.6 | +4.3 | - | - | 32.9 | +6.2 | 40.8 | +7.9 | +6.15 |
| gen_IP2P | Generative | 46.3 | +4.0 | - | - | 32.9 | +6.2 | 40.7 | +7.8 | +6.01 |
| gen_augmenters | Generative | 46.1 | +3.8 | - | - | 32.9 | +6.2 | 41.0 | +8.1 | +6.07 |
| std_randaugment | Standard Aug | 46.2 | +3.9 | - | - | 32.8 | +6.2 | 40.9 | +8.0 | +6.03 |
| std_mixup | Standard Aug | 46.4 | +4.1 | - | - | 32.8 | +6.2 | 40.4 | +7.6 | +5.95 |
| gen_cyclediffusion | Generative | 46.1 | +3.8 | - | - | 32.7 | +6.0 | 40.6 | +7.8 | +5.86 |
| gen_CUT | Generative | 46.0 | +3.7 | - | - | 32.7 | +6.0 | 40.6 | +7.7 | +5.82 |
| gen_Weather_Effect_Generator | Generative | 45.8 | +3.5 | - | - | 33.0 | +6.4 | 40.5 | +7.7 | +5.86 |
| std_autoaugment | Standard Aug | 46.4 | +4.1 | - | - | 32.8 | +6.2 | 40.7 | +7.9 | +6.05 |
| std_cutmix | Standard Aug | 46.3 | +4.0 | - | - | 32.9 | +6.3 | 40.5 | +7.7 | +5.97 |
| gen_VisualCloze | Generative | 46.4 | +4.1 | - | - | 32.9 | +6.3 | 40.5 | +7.6 | +6.00 |
| gen_SUSTechGAN | Generative | 44.8 | +2.6 | - | - | 32.8 | +6.1 | 40.4 | +7.6 | +5.42 |
| gen_TSIT | Generative | 46.6 | +4.3 | - | - | 30.1 | +3.4 | 40.7 | +7.8 | +5.18 |
| gen_automold | Generative | 46.1 | +3.8 | - | - | 32.9 | +6.3 | 40.6 | +7.7 | +5.92 |
| gen_cycleGAN | Generative | 43.3 | +1.0 | - | - | 32.9 | +6.3 | 40.6 | +7.8 | +5.00 |
| gen_albumentations_weather | Generative | 45.0 | +2.7 | - | - | 32.8 | +6.1 | 40.4 | +7.5 | +5.46 |
| gen_flux_kontext | Generative | 43.5 | +1.2 | - | - | 33.0 | +6.3 | 40.9 | +8.0 | +5.17 |
| gen_step1x_new | Generative | 43.6 | +1.3 | - | - | 32.9 | +6.3 | 40.4 | +7.6 | +5.05 |
| gen_step1x_v1p2 | Generative | 43.0 | +0.8 | - | - | 32.8 | +6.2 | 40.5 | +7.7 | +4.88 |
| gen_LANIT | Generative | 43.7 | +1.5 | - | - | 32.9 | +6.2 | 40.7 | +7.8 | +5.17 |
| baseline | Baseline | 42.3 | +0.0 | - | - | 26.6 | +0.0 | 32.9 | +0.0 | +0.00 |

## Per-Domain mIoU by Strategy

| Strategy                     | Type         |   clear_day |   cloudy |   dawn_dusk |   foggy |   night |   rainy |   snowy |   Normal Avg |   Adverse Avg |   Gap |
|:-----------------------------|:-------------|------------:|---------:|------------:|--------:|--------:|--------:|--------:|-------------:|--------------:|------:|
| baseline                     | Baseline     |       35.42 |    33.05 |       28.25 |   32.55 |   23.8  |   30.02 |   29.17 |        34.24 |         28.89 |  5.35 |
| gen_Attribute_Hallucination  | Generative   |       41.97 |    39    |       33.6  |   37.92 |   28.23 |   35.88 |   36.01 |        40.49 |         34.51 |  5.97 |
| gen_CNetSeg                  | Generative   |       41.66 |    39.41 |       33.45 |   37.95 |   28.07 |   35.94 |   36.23 |        40.53 |         34.55 |  5.99 |
| gen_CUT                      | Generative   |       41.61 |    39.05 |       33.39 |   38.05 |   28.6  |   35.39 |   35.49 |        40.33 |         34.38 |  5.95 |
| gen_IP2P                     | Generative   |       41.74 |    39.25 |       33.42 |   38.12 |   28.67 |   35.79 |   35.82 |        40.5  |         34.6  |  5.9  |
| gen_Img2Img                  | Generative   |       41.88 |    39.61 |       33.64 |   37.88 |   28.19 |   36.49 |   35.79 |        40.74 |         34.58 |  6.16 |
| gen_LANIT                    | Generative   |       40.49 |    37.77 |       32.26 |   36.94 |   27.64 |   33.98 |   33.51 |        39.13 |         33.02 |  6.11 |
| gen_Qwen_Image_Edit          | Generative   |       41.77 |    39.26 |       33.73 |   38.11 |   28.63 |   35.84 |   35.44 |        40.51 |         34.51 |  6.01 |
| gen_SUSTechGAN               | Generative   |       41.28 |    38.6  |       33.09 |   37.5  |   27.86 |   35.24 |   35.63 |        39.94 |         34.06 |  5.88 |
| gen_TSIT                     | Generative   |       41.17 |    38.87 |       32.72 |   37.24 |   28.27 |   35.1  |   35.15 |        40.02 |         33.94 |  6.08 |
| gen_UniControl               | Generative   |       41.92 |    39.24 |       33.56 |   38.35 |   28.16 |   36.04 |   35.82 |        40.58 |         34.59 |  5.99 |
| gen_VisualCloze              | Generative   |       41.37 |    38.69 |       33.28 |   38.09 |   28.9  |   35.04 |   34.34 |        40.03 |         34.09 |  5.94 |
| gen_Weather_Effect_Generator | Generative   |       41.59 |    39.36 |       33.86 |   37.86 |   28.37 |   35.55 |   35.62 |        40.47 |         34.35 |  6.12 |
| gen_albumentations_weather   | Generative   |       40.92 |    37.91 |       33.18 |   37.29 |   28.08 |   34.59 |   33.74 |        39.42 |         33.42 |  5.99 |
| gen_augmenters               | Generative   |       41.72 |    39.06 |       33.75 |   37.71 |   28.35 |   35.73 |   36.07 |        40.39 |         34.46 |  5.92 |
| gen_automold                 | Generative   |       41.25 |    38.46 |       32.64 |   38.03 |   28.51 |   34.67 |   34.19 |        39.85 |         33.85 |  6    |
| gen_cycleGAN                 | Generative   |       40.92 |    38.61 |       33.25 |   38.08 |   27.95 |   34.62 |   34.53 |        39.76 |         33.79 |  5.97 |
| gen_cyclediffusion           | Generative   |       41.53 |    39.37 |       33.55 |   38.53 |   28.61 |   35.58 |   35.73 |        40.45 |         34.61 |  5.84 |
| gen_flux_kontext             | Generative   |       40.76 |    38.03 |       33.19 |   36.84 |   28.33 |   34.47 |   33.34 |        39.4  |         33.25 |  6.15 |
| gen_stargan_v2               | Generative   |       41.76 |    39.33 |       33.41 |   38.51 |   27.94 |   35.83 |   35.89 |        40.55 |         34.54 |  6    |
| gen_step1x_new               | Generative   |       40.64 |    38.12 |       32.44 |   37.32 |   27.77 |   34.11 |   33.58 |        39.38 |         33.2  |  6.18 |
| gen_step1x_v1p2              | Generative   |       40.48 |    37.75 |       33.01 |   37.35 |   27.78 |   34.08 |   33.54 |        39.12 |         33.19 |  5.93 |
| std_autoaugment              | Standard Aug |       41.44 |    38.86 |       33.24 |   37.97 |   28.24 |   35.8  |   35.41 |        40.15 |         34.35 |  5.79 |
| std_cutmix                   | Standard Aug |       41.45 |    39.17 |       33.3  |   37.53 |   28.18 |   35.8  |   35.59 |        40.31 |         34.27 |  6.04 |
| std_mixup                    | Standard Aug |       41.65 |    39.21 |       33.42 |   38.11 |   28.07 |   35.89 |   35.89 |        40.43 |         34.49 |  5.94 |
| std_randaugment              | Standard Aug |       41.75 |    39.12 |       33.25 |   37.87 |   27.9  |   35.76 |   35.77 |        40.44 |         34.33 |  6.11 |