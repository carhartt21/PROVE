# Stage 1 Detailed Per-Dataset and Per-Domain Analysis (mIoU)

## Per-Dataset mIoU by Strategy

| Strategy | Type | bdd10k | Δbdd10k | iddaw | Δiddaw | mapillaryvistas | Δmapillaryvistas | outside15k | Δoutside15k | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gen_UniControl | Generative | 47.2 | +4.9 | 38.1 | +2.1 | 33.0 | -0.0 | 40.5 | +1.3 | +2.08 |
| gen_Img2Img | Generative | 46.9 | +4.6 | 38.1 | +2.1 | 32.9 | -0.1 | 40.4 | +1.2 | +1.96 |
| std_autoaugment | Standard Aug | 46.4 | +4.1 | 36.9 | +1.0 | 34.7 | +1.7 | 41.9 | +2.7 | +2.37 |
| gen_Attribute_Hallucination | Generative | 46.7 | +4.4 | 37.7 | +1.8 | 32.9 | -0.1 | 40.9 | +1.8 | +1.95 |
| std_cutmix | Standard Aug | 46.3 | +4.0 | 37.1 | +1.1 | 34.7 | +1.6 | 41.6 | +2.5 | +2.30 |
| gen_Qwen_Image_Edit | Generative | 46.4 | +4.1 | 37.8 | +1.8 | 32.9 | -0.2 | 40.9 | +1.7 | +1.87 |
| gen_stargan_v2 | Generative | 46.5 | +4.3 | 37.7 | +1.7 | 32.7 | -0.3 | 41.0 | +1.9 | +1.87 |
| gen_CNetSeg | Generative | 46.6 | +4.3 | 37.7 | +1.7 | 32.9 | -0.2 | 40.8 | +1.6 | +1.87 |
| std_mixup | Standard Aug | 46.4 | +4.1 | 37.7 | +1.7 | 33.5 | +0.4 | 41.7 | +2.5 | +2.20 |
| gen_IP2P | Generative | 46.3 | +4.0 | 37.8 | +1.8 | 32.9 | -0.2 | 40.7 | +1.5 | +1.79 |
| gen_augmenters | Generative | 46.1 | +3.8 | 37.7 | +1.8 | 32.9 | -0.2 | 41.0 | +1.8 | +1.82 |
| std_randaugment | Standard Aug | 46.2 | +3.9 | 37.7 | +1.8 | 32.8 | -0.2 | 40.9 | +1.7 | +1.79 |
| gen_flux_kontext | Generative | 44.5 | +2.3 | 37.8 | +1.8 | 33.0 | -0.1 | 42.0 | +2.9 | +1.72 |
| gen_cyclediffusion | Generative | 46.1 | +3.8 | 38.0 | +2.0 | 32.7 | -0.4 | 40.6 | +1.5 | +1.74 |
| gen_VisualCloze | Generative | 45.7 | +3.5 | 38.0 | +2.0 | 32.9 | -0.1 | 40.5 | +1.3 | +1.68 |
| gen_CUT | Generative | 46.0 | +3.7 | 37.7 | +1.7 | 32.7 | -0.4 | 40.6 | +1.4 | +1.63 |
| gen_Weather_Effect_Generator | Generative | 45.8 | +3.5 | 37.6 | +1.6 | 33.0 | -0.0 | 40.5 | +1.4 | +1.63 |
| gen_SUSTechGAN | Generative | 44.8 | +2.6 | 37.8 | +1.8 | 32.8 | -0.3 | 40.4 | +1.3 | +1.36 |
| gen_step1x_new | Generative | 44.7 | +2.4 | 37.9 | +1.9 | 32.9 | -0.1 | 40.4 | +1.3 | +1.37 |
| gen_TSIT | Generative | 46.6 | +4.3 | 37.7 | +1.7 | 30.1 | -3.0 | 40.7 | +1.5 | +1.15 |
| gen_automold | Generative | 46.1 | +3.8 | 37.8 | +1.8 | 32.9 | -0.1 | 40.6 | +1.4 | +1.73 |
| gen_cycleGAN | Generative | 43.3 | +1.0 | 37.9 | +1.9 | 32.9 | -0.1 | 41.3 | +2.2 | +1.22 |
| gen_LANIT | Generative | 44.6 | +2.3 | 36.9 | +0.9 | 32.9 | -0.2 | 40.7 | +1.5 | +1.13 |
| gen_albumentations_weather | Generative | 45.0 | +2.7 | 37.7 | +1.7 | 32.8 | -0.3 | 40.4 | +1.2 | +1.35 |
| gen_step1x_v1p2 | Generative | 43.0 | +0.8 | 37.8 | +1.8 | 32.8 | -0.2 | 40.5 | +1.4 | +0.93 |
| baseline | Baseline | 42.3 | +0.0 | 36.0 | +0.0 | 33.0 | -0.0 | 39.1 | -0.0 | -0.00 |

## Per-Domain mIoU by Strategy

| Strategy                     | Type         |   clear_day |   cloudy |   dawn_dusk |   foggy |   night |   rainy |   snowy |   Normal Avg |   Adverse Avg |   Gap |
|:-----------------------------|:-------------|------------:|---------:|------------:|--------:|--------:|--------:|--------:|-------------:|--------------:|------:|
| baseline                     | Baseline     |       39.42 |    36.91 |       32.16 |   36.81 |   26.87 |   34.02 |   33.05 |        38.17 |         32.69 |  5.48 |
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
| gen_cycleGAN                 | Generative   |       41.22 |    38.68 |       33.29 |   38.66 |   28.19 |   35.09 |   34.67 |        39.95 |         34.15 |  5.8  |
| gen_cyclediffusion           | Generative   |       41.67 |    39.44 |       33.59 |   38.51 |   28.73 |   35.51 |   35.75 |        40.56 |         34.63 |  5.93 |
| gen_flux_kontext             | Generative   |       41.76 |    39.18 |       33.83 |   38.43 |   28.56 |   35.87 |   35.2  |        40.47 |         34.51 |  5.96 |
| gen_stargan_v2               | Generative   |       41.76 |    39.33 |       33.41 |   38.51 |   27.94 |   35.83 |   35.89 |        40.55 |         34.54 |  6    |
| gen_step1x_new               | Generative   |       41.28 |    39    |       33.14 |   38.4  |   27.89 |   35.09 |   35.1  |        40.14 |         34.12 |  6.02 |
| gen_step1x_v1p2              | Generative   |       40.48 |    37.75 |       33.01 |   37.35 |   27.78 |   34.08 |   33.54 |        39.12 |         33.19 |  5.93 |
| std_autoaugment              | Standard Aug |       41.81 |    39.08 |       33.83 |   38.6  |   28.9  |   36.28 |   35.76 |        40.45 |         34.88 |  5.56 |
| std_cutmix                   | Standard Aug |       41.67 |    39.6  |       33.87 |   38.19 |   29.19 |   36.49 |   36.05 |        40.64 |         34.98 |  5.66 |
| std_mixup                    | Standard Aug |       41.64 |    39.14 |       33.64 |   38.07 |   28.54 |   36.15 |   35.7  |        40.39 |         34.61 |  5.78 |
| std_randaugment              | Standard Aug |       41.75 |    39.12 |       33.25 |   37.87 |   27.9  |   35.76 |   35.77 |        40.44 |         34.33 |  6.11 |