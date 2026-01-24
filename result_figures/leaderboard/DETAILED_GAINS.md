# Detailed Per-Dataset and Per-Domain Analysis

## Per-Dataset mIoU by Strategy

| Strategy | Type | bdd10k | Δbdd10k | idd-aw | Δidd-aw | mapillaryvistas | Δmapillaryvistas | outside15k | Δoutside15k | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gen_Attribute_Hallucination | Generative | 46.5 | +2.4 | 40.0 | +1.4 | 34.8 | +0.5 | 38.0 | +1.1 | +1.36 |
| gen_cycleGAN | Generative | 46.0 | +1.9 | 40.0 | +1.5 | 34.5 | +0.2 | 37.9 | +1.0 | +1.13 |
| gen_Img2Img | Generative | 46.3 | +2.2 | 39.9 | +1.3 | 34.8 | +0.5 | 37.3 | +0.4 | +1.11 |
| gen_stargan_v2 | Generative | 46.1 | +2.1 | 39.5 | +1.0 | 34.8 | +0.5 | 37.7 | +0.8 | +1.08 |
| gen_flux_kontext | Generative | 46.1 | +2.1 | 39.3 | +0.7 | 34.5 | +0.2 | 38.2 | +1.3 | +1.07 |
| gen_cyclediffusion | Generative | 46.3 | +2.2 | 39.8 | +1.3 | 34.6 | +0.3 | 37.4 | +0.5 | +1.05 |
| gen_CNetSeg | Generative | 46.0 | +1.9 | 39.7 | +1.1 | 35.1 | +0.8 | 37.1 | +0.2 | +1.00 |
| gen_IP2P | Generative | 46.3 | +2.2 | 39.6 | +1.1 | 35.1 | +0.8 | 36.7 | -0.1 | +1.00 |
| gen_augmenters | Generative | 45.9 | +1.8 | 40.0 | +1.4 | 34.9 | +0.6 | 37.1 | +0.2 | +0.99 |
| gen_Weather_Effect_Generator | Generative | 46.2 | +2.1 | 39.5 | +0.9 | 34.6 | +0.2 | 37.4 | +0.5 | +0.96 |
| gen_SUSTechGAN | Generative | 46.2 | +2.2 | 39.5 | +1.0 | 34.7 | +0.3 | 37.3 | +0.4 | +0.96 |
| gen_automold | Generative | 46.4 | +2.3 | 39.6 | +1.1 | 34.5 | +0.2 | 37.2 | +0.3 | +0.96 |
| gen_step1x_new | Generative | 46.3 | +2.2 | 39.3 | +0.7 | 34.6 | +0.2 | 37.4 | +0.6 | +0.94 |
| std_autoaugment | Standard Aug | 46.1 | +2.0 | 39.8 | +1.2 | 34.2 | -0.2 | 37.6 | +0.7 | +0.94 |
| gen_VisualCloze | Generative | 45.9 | +1.8 | 39.5 | +1.0 | 34.5 | +0.2 | 37.6 | +0.8 | +0.93 |
| gen_albumentations_weather | Generative | 46.3 | +2.2 | 39.8 | +1.2 | 34.3 | -0.0 | 36.9 | -0.0 | +0.84 |
| gen_Qwen_Image_Edit | Generative | 45.9 | +1.8 | 39.8 | +1.2 | 34.6 | +0.3 | 36.8 | -0.1 | +0.80 |
| gen_LANIT | Generative | 45.5 | +1.4 | 39.4 | +0.8 | 34.8 | +0.5 | 37.1 | +0.2 | +0.74 |
| gen_UniControl | Generative | 46.0 | +1.9 | 39.3 | +0.7 | 34.3 | +0.0 | 37.1 | +0.2 | +0.71 |
| gen_CUT | Generative | 45.9 | +1.8 | 39.5 | +0.9 | 34.1 | -0.2 | 37.2 | +0.3 | +0.70 |
| gen_step1x_v1p2 | Generative | 46.1 | +2.0 | 39.7 | +1.1 | 34.5 | +0.2 | 36.4 | -0.5 | +0.69 |
| gen_TSIT | Generative | 45.8 | +1.7 | 39.4 | +0.8 | 34.5 | +0.2 | 36.7 | -0.2 | +0.63 |
| std_mixup | Standard Aug | 45.6 | +1.5 | 38.6 | +0.1 | 34.1 | -0.2 | 37.7 | +0.8 | +0.52 |
| photometric_distort | Augmentation | 45.8 | +1.7 | 39.9 | +1.3 | 34.6 | +0.3 | 35.6 | -1.3 | +0.48 |
| std_randaugment | Standard Aug | 45.3 | +1.2 | 39.5 | +1.0 | 33.5 | -0.8 | 37.4 | +0.5 | +0.45 |
| std_cutmix | Standard Aug | 44.5 | +0.4 | 39.3 | +0.8 | 33.8 | -0.5 | 36.9 | +0.0 | +0.17 |
| baseline | Baseline | 44.1 | +0.0 | 38.6 | +0.0 | 34.3 | +0.0 | 36.9 | +0.0 | +0.00 |

## Per-Domain mIoU by Strategy

| Strategy                     | Type         |   clear_day |   cloudy |   dawn_dusk |   foggy |   night |   rainy |   snowy |   Normal Avg |   Adverse Avg |   Gap |
|:-----------------------------|:-------------|------------:|---------:|------------:|--------:|--------:|--------:|--------:|-------------:|--------------:|------:|
| baseline                     | Baseline     |       41.05 |    37.3  |       31.78 |   37.47 |   26.2  |   33.52 |   32.27 |        39.18 |         32.36 |  6.81 |
| gen_Attribute_Hallucination  | Generative   |       42.2  |    39.16 |       33.99 |   39.36 |   27.41 |   34.76 |   33.96 |        40.68 |         33.87 |  6.8  |
| gen_CNetSeg                  | Generative   |       42.04 |    38.68 |       32.8  |   39.56 |   27.17 |   34.5  |   33.61 |        40.36 |         33.71 |  6.65 |
| gen_CUT                      | Generative   |       41.61 |    38.38 |       32.12 |   39.07 |   27.16 |   33.85 |   33.53 |        40    |         33.4  |  6.59 |
| gen_IP2P                     | Generative   |       41.95 |    38.49 |       33.02 |   38.16 |   26.73 |   34.27 |   34.05 |        40.22 |         33.3  |  6.92 |
| gen_Img2Img                  | Generative   |       42.11 |    38.95 |       32.36 |   38.82 |   27.38 |   34.54 |   33.85 |        40.53 |         33.65 |  6.88 |
| gen_LANIT                    | Generative   |       41.75 |    38.25 |       32.68 |   37.74 |   26.74 |   33.96 |   33.41 |        40    |         32.96 |  7.03 |
| gen_Qwen_Image_Edit          | Generative   |       41.71 |    38.63 |       32.74 |   38.69 |   27.27 |   34.12 |   33.81 |        40.17 |         33.47 |  6.69 |
| gen_SUSTechGAN               | Generative   |       41.8  |    38.28 |       32.9  |   38.62 |   27.5  |   34.32 |   33.97 |        40.04 |         33.6  |  6.44 |
| gen_TSIT                     | Generative   |       41.67 |    38.08 |       32.44 |   38.29 |   26.73 |   33.85 |   33.24 |        39.87 |         33.03 |  6.84 |
| gen_UniControl               | Generative   |       41.74 |    38.27 |       32.45 |   39.25 |   27.26 |   34.35 |   33.8  |        40.01 |         33.67 |  6.34 |
| gen_VisualCloze              | Generative   |       41.96 |    38.48 |       33.05 |   38.71 |   27.34 |   34.46 |   33.69 |        40.22 |         33.55 |  6.68 |
| gen_Weather_Effect_Generator | Generative   |       41.98 |    38.31 |       32.79 |   38.34 |   26.95 |   34.55 |   33.49 |        40.14 |         33.33 |  6.81 |
| gen_albumentations_weather   | Generative   |       41.74 |    38.56 |       33    |   38.97 |   27.26 |   34.3  |   33.36 |        40.15 |         33.47 |  6.68 |
| gen_augmenters               | Generative   |       41.98 |    38.86 |       32.98 |   39.84 |   27.4  |   34.24 |   33.54 |        40.42 |         33.75 |  6.66 |
| gen_automold                 | Generative   |       41.77 |    38.85 |       33.01 |   38.58 |   26.91 |   34.59 |   33.73 |        40.31 |         33.45 |  6.86 |
| gen_cycleGAN                 | Generative   |       42.07 |    38.57 |       33.34 |   39.12 |   27.41 |   34.4  |   33.73 |        40.32 |         33.66 |  6.66 |
| gen_cyclediffusion           | Generative   |       42.07 |    38.64 |       32.89 |   38.99 |   27.07 |   34.82 |   33.39 |        40.35 |         33.57 |  6.79 |
| gen_flux_kontext             | Generative   |       42.05 |    38.63 |       32.81 |   39.32 |   27.17 |   34.49 |   33.79 |        40.34 |         33.69 |  6.65 |
| gen_stargan_v2               | Generative   |       42.11 |    38.84 |       33.33 |   39.09 |   27.23 |   34.74 |   33.65 |        40.47 |         33.68 |  6.79 |
| gen_step1x_new               | Generative   |       41.8  |    38.61 |       32.49 |   38.74 |   26.95 |   34.51 |   33.79 |        40.21 |         33.5  |  6.71 |
| gen_step1x_v1p2              | Generative   |       41.59 |    38.36 |       32.62 |   38.31 |   27.15 |   33.86 |   33.2  |        39.97 |         33.13 |  6.84 |
| photometric_distort          | Augmentation |       41.34 |    38.32 |       32.52 |   38.5  |   27.17 |   33.9  |   32.94 |        39.83 |         33.13 |  6.7  |
| std_autoaugment              | Standard Aug |       41.78 |    38.33 |       33.7  |   40.18 |   28.59 |   34.6  |   33.46 |        40.05 |         34.21 |  5.84 |
| std_cutmix                   | Standard Aug |       41.23 |    37.98 |       32.07 |   39.38 |   26.32 |   33.43 |   33.02 |        39.6  |         33.04 |  6.57 |
| std_mixup                    | Standard Aug |       41.64 |    37.96 |       32.36 |   38.28 |   26.24 |   34.03 |   33.11 |        39.8  |         32.91 |  6.89 |
| std_randaugment              | Standard Aug |       41.24 |    38.08 |       32.86 |   38.89 |   28.16 |   34.03 |   33.32 |        39.66 |         33.6  |  6.06 |