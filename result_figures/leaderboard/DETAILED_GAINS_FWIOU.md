# Detailed Per-Dataset and Per-Domain Analysis (fwIoU)

## Per-Dataset fwIoU by Strategy

| Strategy | Type | bdd10k | Δbdd10k | idd-aw | Δidd-aw | mapillaryvistas | Δmapillaryvistas | outside15k | Δoutside15k | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gen_Attribute_Hallucination | Generative | 83.7 | +0.6 | 74.7 | +1.4 | 82.3 | +0.1 | 65.5 | +1.1 | +0.81 |
| std_autoaugment | Standard Aug | 83.8 | +0.7 | 73.8 | +0.5 | 82.2 | -0.0 | 65.8 | +1.4 | +0.65 |
| gen_cycleGAN | Generative | 83.7 | +0.6 | 74.6 | +1.3 | 82.1 | -0.1 | 65.2 | +0.9 | +0.65 |
| gen_Img2Img | Generative | 83.7 | +0.6 | 74.3 | +1.0 | 82.4 | +0.2 | 65.1 | +0.8 | +0.63 |
| gen_stargan_v2 | Generative | 83.6 | +0.5 | 74.2 | +0.9 | 82.3 | +0.1 | 65.3 | +0.9 | +0.61 |
| gen_flux_kontext | Generative | 83.7 | +0.6 | 73.7 | +0.4 | 82.4 | +0.1 | 65.6 | +1.3 | +0.61 |
| std_randaugment | Standard Aug | 83.5 | +0.4 | 73.9 | +0.6 | 82.1 | -0.1 | 65.7 | +1.3 | +0.55 |
| gen_augmenters | Generative | 83.6 | +0.5 | 74.5 | +1.3 | 82.5 | +0.2 | 64.6 | +0.3 | +0.55 |
| gen_VisualCloze | Generative | 83.7 | +0.6 | 73.8 | +0.5 | 82.1 | -0.1 | 65.2 | +0.8 | +0.46 |
| gen_CNetSeg | Generative | 83.5 | +0.4 | 73.9 | +0.6 | 82.5 | +0.2 | 64.8 | +0.5 | +0.44 |
| gen_CUT | Generative | 83.7 | +0.6 | 74.0 | +0.8 | 82.2 | -0.1 | 64.8 | +0.5 | +0.44 |
| gen_SUSTechGAN | Generative | 83.7 | +0.6 | 73.8 | +0.5 | 82.3 | +0.1 | 64.9 | +0.6 | +0.43 |
| gen_Qwen_Image_Edit | Generative | 83.7 | +0.6 | 74.2 | +0.9 | 82.4 | +0.2 | 64.3 | +0.0 | +0.42 |
| gen_automold | Generative | 83.7 | +0.6 | 73.7 | +0.5 | 82.3 | +0.1 | 64.7 | +0.3 | +0.36 |
| gen_UniControl | Generative | 83.9 | +0.8 | 73.5 | +0.3 | 82.2 | -0.1 | 64.7 | +0.4 | +0.35 |
| gen_albumentations_weather | Generative | 83.7 | +0.6 | 73.9 | +0.7 | 82.3 | +0.0 | 64.4 | +0.1 | +0.34 |
| photometric_distort | Augmentation | 83.5 | +0.4 | 74.7 | +1.4 | 82.2 | -0.0 | 63.8 | -0.5 | +0.31 |
| gen_Weather_Effect_Generator | Generative | 83.7 | +0.6 | 73.3 | +0.1 | 82.2 | -0.0 | 64.8 | +0.5 | +0.27 |
| gen_IP2P | Generative | 83.7 | +0.6 | 73.4 | +0.1 | 82.4 | +0.2 | 64.6 | +0.2 | +0.27 |
| gen_step1x_new | Generative | 83.8 | +0.7 | 73.1 | -0.1 | 82.2 | -0.0 | 64.9 | +0.6 | +0.26 |
| gen_cyclediffusion | Generative | 83.8 | +0.7 | 72.9 | -0.3 | 82.4 | +0.2 | 64.7 | +0.4 | +0.24 |
| gen_step1x_v1p2 | Generative | 83.6 | +0.5 | 74.0 | +0.7 | 82.2 | -0.0 | 63.9 | -0.5 | +0.18 |
| gen_LANIT | Generative | 83.4 | +0.3 | 73.0 | -0.3 | 82.4 | +0.1 | 64.8 | +0.5 | +0.15 |
| std_mixup | Standard Aug | 83.1 | -0.0 | 72.9 | -0.3 | 82.1 | -0.1 | 65.3 | +0.9 | +0.12 |
| gen_TSIT | Generative | 83.5 | +0.4 | 73.4 | +0.1 | 82.2 | -0.1 | 64.2 | -0.1 | +0.09 |
| baseline | Baseline | 83.1 | +0.0 | 73.2 | +0.0 | 82.3 | +0.0 | 64.3 | +0.0 | +0.00 |
| std_cutmix | Standard Aug | 83.4 | +0.3 | 73.4 | +0.1 | 81.5 | -0.8 | 64.4 | +0.1 | -0.08 |

## Per-Domain fwIoU by Strategy

| Strategy                     | Type         |   clear_day |   cloudy |   dawn_dusk |   foggy |   night |   rainy |   snowy |   Normal Avg |   Adverse Avg |   Gap |
|:-----------------------------|:-------------|------------:|---------:|------------:|--------:|--------:|--------:|--------:|-------------:|--------------:|------:|
| baseline                     | Baseline     |       77.06 |    80.01 |       72.04 |   78.07 |   64.58 |   69.33 |   67.14 |        78.53 |         69.78 |  8.76 |
| gen_Attribute_Hallucination  | Generative   |       77.6  |    81.45 |       73.99 |   78.93 |   66.04 |   70.07 |   67.91 |        79.52 |         70.74 |  8.78 |
| gen_CNetSeg                  | Generative   |       77.52 |    80.81 |       73    |   78.61 |   65.29 |   69.89 |   67.33 |        79.16 |         70.28 |  8.88 |
| gen_CUT                      | Generative   |       77.38 |    80.8  |       71.96 |   78.97 |   65.09 |   69.68 |   67.84 |        79.09 |         70.39 |  8.7  |
| gen_IP2P                     | Generative   |       77.31 |    80.48 |       72.54 |   77.87 |   64.9  |   69.66 |   67.54 |        78.9  |         69.99 |  8.9  |
| gen_Img2Img                  | Generative   |       77.61 |    81.16 |       72.59 |   78.64 |   65.78 |   70.02 |   67.38 |        79.39 |         70.45 |  8.93 |
| gen_LANIT                    | Generative   |       77.37 |    80.24 |       72.04 |   77.57 |   64.43 |   69.8  |   67.02 |        78.8  |         69.71 |  9.1  |
| gen_Qwen_Image_Edit          | Generative   |       77.36 |    80.87 |       72.62 |   78.75 |   65.02 |   69.62 |   67.75 |        79.11 |         70.29 |  8.83 |
| gen_SUSTechGAN               | Generative   |       77.29 |    80.95 |       73    |   78.75 |   64.87 |   69.69 |   68.21 |        79.12 |         70.38 |  8.74 |
| gen_TSIT                     | Generative   |       77.25 |    80.62 |       72.28 |   78.36 |   63.58 |   69.39 |   66.86 |        78.94 |         69.55 |  9.39 |
| gen_UniControl               | Generative   |       77.39 |    80.68 |       72.42 |   78.64 |   65.01 |   69.77 |   66.95 |        79.04 |         70.09 |  8.94 |
| gen_VisualCloze              | Generative   |       77.46 |    80.66 |       72.78 |   78.51 |   65.38 |   69.85 |   67.51 |        79.06 |         70.31 |  8.75 |
| gen_Weather_Effect_Generator | Generative   |       77.4  |    80.65 |       72.88 |   78.3  |   64.38 |   69.89 |   66.83 |        79.03 |         69.85 |  9.17 |
| gen_albumentations_weather   | Generative   |       77.33 |    80.86 |       72.46 |   78.66 |   65.06 |   69.55 |   66.99 |        79.09 |         70.07 |  9.03 |
| gen_augmenters               | Generative   |       77.49 |    81.16 |       72.73 |   79.08 |   65.4  |   69.91 |   67.2  |        79.33 |         70.4  |  8.93 |
| gen_automold                 | Generative   |       77.37 |    80.82 |       72.58 |   78.41 |   64.73 |   70.15 |   67.09 |        79.1  |         70.1  |  9    |
| gen_cycleGAN                 | Generative   |       77.6  |    81.12 |       73.1  |   78.92 |   65.86 |   69.91 |   67.41 |        79.36 |         70.52 |  8.83 |
| gen_cyclediffusion           | Generative   |       77.58 |    80.08 |       71.7  |   77.79 |   64.79 |   70.08 |   67.61 |        78.83 |         70.07 |  8.76 |
| gen_flux_kontext             | Generative   |       77.56 |    81.18 |       73.1  |   78.64 |   65.11 |   70.01 |   67.34 |        79.37 |         70.28 |  9.09 |
| gen_stargan_v2               | Generative   |       77.52 |    81.02 |       72.86 |   79.06 |   65.42 |   70.27 |   67.72 |        79.27 |         70.62 |  8.65 |
| gen_step1x_new               | Generative   |       77.26 |    80.73 |       72.23 |   78.46 |   64.7  |   69.67 |   67.66 |        79    |         70.12 |  8.88 |
| gen_step1x_v1p2              | Generative   |       77.18 |    80.64 |       72.38 |   78.6  |   65.22 |   69.31 |   66.93 |        78.91 |         70.01 |  8.9  |
| photometric_distort          | Augmentation |       77.16 |    81.11 |       72.25 |   78.7  |   65.79 |   69.54 |   67.22 |        79.13 |         70.31 |  8.82 |
| std_autoaugment              | Standard Aug |       77.54 |    80.61 |       74.91 |   78.49 |   67.9  |   70.15 |   67.2  |        79.07 |         70.93 |  8.14 |
| std_cutmix                   | Standard Aug |       77.04 |    80.44 |       71.9  |   77.9  |   63.93 |   69.03 |   66.53 |        78.74 |         69.35 |  9.39 |
| std_mixup                    | Standard Aug |       77.39 |    80.49 |       72.15 |   77.44 |   63.74 |   69.76 |   66.79 |        78.94 |         69.43 |  9.51 |
| std_randaugment              | Standard Aug |       77.35 |    80.76 |       74.24 |   78.46 |   68.19 |   69.88 |   67.56 |        79.05 |         71.02 |  8.03 |