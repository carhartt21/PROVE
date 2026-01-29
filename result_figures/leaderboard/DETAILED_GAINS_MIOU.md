# Detailed Per-Dataset and Per-Domain Analysis (mIoU)

## Per-Dataset mIoU by Strategy

| Strategy | Type | bdd10k | Δbdd10k | idd-aw | Δidd-aw | mapillaryvistas | Δmapillaryvistas | outside15k | Δoutside15k | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| std_minimal | Standard Aug | 46.1 | +0.5 | 40.9 | +0.8 | - | - | - | - | +0.69 |
| std_photometric_distort | Augmentation | 45.6 | +0.0 | 40.5 | +0.5 | - | - | - | - | +0.24 |
| gen_step1x_v1p2 | Generative | 45.7 | +0.1 | 40.4 | +0.3 | - | - | - | - | +0.23 |
| gen_cycleGAN | Generative | 45.8 | +0.2 | 40.2 | +0.1 | - | - | - | - | +0.15 |
| gen_Attribute_Hallucination | Generative | 45.6 | -0.0 | 40.3 | +0.2 | - | - | - | - | +0.10 |
| gen_augmenters | Generative | 45.7 | +0.1 | 40.2 | +0.1 | - | - | - | - | +0.10 |
| gen_SUSTechGAN | Generative | 45.6 | +0.0 | 40.2 | +0.1 | - | - | - | - | +0.07 |
| gen_TSIT | Generative | 45.7 | +0.1 | 40.1 | +0.0 | - | - | - | - | +0.07 |
| std_cutmix | Standard Aug | 45.4 | -0.2 | 40.3 | +0.3 | - | - | - | - | +0.05 |
| gen_flux_kontext | Generative | 45.5 | -0.1 | 40.2 | +0.2 | - | - | - | - | +0.03 |
| gen_UniControl | Generative | 45.5 | -0.2 | 40.3 | +0.2 | - | - | - | - | +0.03 |
| gen_automold | Generative | 45.4 | -0.2 | 40.2 | +0.2 | - | - | - | - | +0.01 |
| std_mixup | Standard Aug | 45.3 | -0.3 | 40.3 | +0.3 | - | - | - | - | -0.00 |
| baseline | Baseline | 45.6 | +0.0 | 40.1 | -0.0 | - | - | - | - | -0.00 |
| gen_Img2Img | Generative | 45.4 | -0.2 | 40.3 | +0.2 | - | - | - | - | -0.01 |
| gen_CUT | Generative | 45.5 | -0.1 | 40.1 | +0.1 | - | - | - | - | -0.03 |
| gen_step1x_new | Generative | 45.2 | -0.4 | 40.3 | +0.3 | - | - | - | - | -0.05 |
| gen_Weather_Effect_Generator | Generative | 45.3 | -0.3 | 40.3 | +0.2 | - | - | - | - | -0.06 |
| std_autoaugment | Standard Aug | 45.5 | -0.1 | 40.0 | -0.0 | - | - | - | - | -0.08 |
| gen_VisualCloze | Generative | 45.4 | -0.2 | 40.1 | +0.0 | - | - | - | - | -0.09 |
| gen_stargan_v2 | Generative | 45.3 | -0.3 | 40.1 | +0.1 | - | - | - | - | -0.12 |
| gen_IP2P | Generative | 45.2 | -0.4 | 40.2 | +0.1 | - | - | - | - | -0.12 |
| gen_Qwen_Image_Edit | Generative | 45.3 | -0.4 | 40.1 | +0.1 | - | - | - | - | -0.13 |
| gen_albumentations_weather | Generative | 45.3 | -0.3 | 40.1 | +0.1 | - | - | - | - | -0.14 |
| gen_cyclediffusion | Generative | 45.2 | -0.4 | 40.2 | +0.1 | - | - | - | - | -0.15 |
| gen_CNetSeg | Generative | 45.2 | -0.4 | 40.2 | +0.1 | - | - | - | - | -0.15 |
| gen_LANIT | Generative | 45.2 | -0.4 | 40.1 | +0.1 | - | - | - | - | -0.17 |
| std_randaugment | Standard Aug | 45.2 | -0.4 | 39.9 | -0.1 | - | - | - | - | -0.26 |

## Per-Domain mIoU by Strategy

| Strategy                     | Type         |   clear_day |   cloudy |   dawn_dusk |   foggy |   night |   rainy |   snowy |   Normal Avg |   Adverse Avg |   Gap |
|:-----------------------------|:-------------|------------:|---------:|------------:|--------:|--------:|--------:|--------:|-------------:|--------------:|------:|
| baseline                     | Baseline     |       46    |    42.93 |       36.1  |   42.43 |   26.77 |   36.08 |   38.65 |        44.46 |         35.98 |  8.48 |
| gen_Attribute_Hallucination  | Generative   |       46.07 |    43.54 |       36.36 |   41.75 |   26.62 |   35.97 |   38.6  |        44.81 |         35.74 |  9.07 |
| gen_CNetSeg                  | Generative   |       45.83 |    42.8  |       36.27 |   41.63 |   26.89 |   35.46 |   38.94 |        44.31 |         35.73 |  8.59 |
| gen_CUT                      | Generative   |       45.96 |    43.05 |       36.01 |   42.93 |   26.67 |   35.59 |   38.72 |        44.51 |         35.98 |  8.53 |
| gen_IP2P                     | Generative   |       45.92 |    43.2  |       36.1  |   43.69 |   27.03 |   35.48 |   38.47 |        44.56 |         36.17 |  8.39 |
| gen_Img2Img                  | Generative   |       46.08 |    43.47 |       36.66 |   43.17 |   26.55 |   35.41 |   38.68 |        44.78 |         35.95 |  8.83 |
| gen_LANIT                    | Generative   |       45.93 |    42.74 |       36.02 |   42.76 |   26.59 |   35.49 |   38.47 |        44.34 |         35.83 |  8.51 |
| gen_Qwen_Image_Edit          | Generative   |       45.89 |    43.48 |       35.9  |   42.51 |   26.57 |   35.34 |   38.21 |        44.69 |         35.66 |  9.03 |
| gen_SUSTechGAN               | Generative   |       46.21 |    42.96 |       36.01 |   43.02 |   26.67 |   35.53 |   37.85 |        44.58 |         35.77 |  8.82 |
| gen_TSIT                     | Generative   |       45.92 |    42.78 |       36.01 |   43.09 |   26.62 |   36.12 |   38.96 |        44.35 |         36.2  |  8.15 |
| gen_UniControl               | Generative   |       46.11 |    43.07 |       36.56 |   42.75 |   26.57 |   35.72 |   38.55 |        44.59 |         35.9  |  8.69 |
| gen_VisualCloze              | Generative   |       45.98 |    43.05 |       36.7  |   41.28 |   26.41 |   35.03 |   38.99 |        44.51 |         35.43 |  9.09 |
| gen_Weather_Effect_Generator | Generative   |       46.12 |    43.33 |       36.31 |   42.25 |   26.11 |   35.42 |   38.62 |        44.73 |         35.6  |  9.13 |
| gen_albumentations_weather   | Generative   |       46.16 |    43.09 |       36.05 |   41.51 |   26.15 |   34.63 |   38.33 |        44.63 |         35.16 |  9.47 |
| gen_augmenters               | Generative   |       46.12 |    42.88 |       35.99 |   42    |   26.75 |   35.81 |   38.31 |        44.5  |         35.72 |  8.79 |
| gen_automold                 | Generative   |       46.11 |    43.17 |       36.44 |   41.33 |   26.3  |   35.49 |   38.24 |        44.64 |         35.34 |  9.3  |
| gen_cycleGAN                 | Generative   |       46.24 |    43    |       36.16 |   41.62 |   26.38 |   35.38 |   39.16 |        44.62 |         35.64 |  8.98 |
| gen_cyclediffusion           | Generative   |       45.87 |    43.07 |       36.59 |   41.76 |   27    |   35.27 |   38.55 |        44.47 |         35.64 |  8.82 |
| gen_flux_kontext             | Generative   |       45.9  |    43.31 |       36.36 |   43.02 |   26.66 |   35.78 |   39.01 |        44.6  |         36.12 |  8.48 |
| gen_stargan_v2               | Generative   |       45.92 |    42.9  |       36.06 |   42.49 |   26.25 |   35.55 |   38.64 |        44.41 |         35.73 |  8.68 |
| gen_step1x_new               | Generative   |       45.89 |    43.37 |       36.23 |   42.19 |   26.13 |   35.79 |   38.59 |        44.63 |         35.68 |  8.95 |
| gen_step1x_v1p2              | Generative   |       46.19 |    43.88 |       36.21 |   42.39 |   27.06 |   35.89 |   38.76 |        45.03 |         36.03 |  9.01 |
| std_autoaugment              | Standard Aug |       46.02 |    42.85 |       36.17 |   40.38 |   26.65 |   35.2  |   38.8  |        44.44 |         35.26 |  9.18 |
| std_cutmix                   | Standard Aug |       46.13 |    43.21 |       35.63 |   41.06 |   26.77 |   35.65 |   38.84 |        44.67 |         35.58 |  9.09 |
| std_minimal                  | Standard Aug |       46.93 |    43.94 |       36.67 |   42.6  |   26.47 |   36.44 |   39.65 |        45.43 |         36.29 |  9.14 |
| std_mixup                    | Standard Aug |       45.92 |    42.95 |       36.55 |   42.62 |   26.68 |   35.72 |   38.62 |        44.43 |         35.91 |  8.52 |
| std_photometric_distort      | Augmentation |       45.98 |    43.48 |       37.46 |   43.48 |   27.52 |   36.7  |   38.96 |        44.73 |         36.66 |  8.07 |
| std_randaugment              | Standard Aug |       45.79 |    42.56 |       36.42 |   42.17 |   26.07 |   35.31 |   38.13 |        44.18 |         35.42 |  8.76 |