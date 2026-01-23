# Detailed Per-Dataset and Per-Domain Analysis (aAcc)

## Per-Dataset aAcc by Strategy

| Strategy | Type | bdd10k | Δbdd10k | idd-aw | Δidd-aw | mapillaryvistas | Δmapillaryvistas | outside15k | Δoutside15k | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gen_Attribute_Hallucination | Generative | 90.7 | +0.4 | 84.5 | +1.0 | 89.2 | +0.1 | 76.5 | +1.0 | +0.61 |
| gen_cycleGAN | Generative | 90.7 | +0.3 | 84.4 | +1.0 | 89.1 | -0.1 | 76.4 | +0.8 | +0.51 |
| std_autoaugment | Standard Aug | 90.8 | +0.4 | 84.0 | +0.6 | 89.2 | -0.0 | 76.6 | +1.1 | +0.51 |
| gen_stargan_v2 | Generative | 90.7 | +0.3 | 84.1 | +0.7 | 89.3 | +0.1 | 76.5 | +0.9 | +0.50 |
| gen_Img2Img | Generative | 90.7 | +0.4 | 84.2 | +0.8 | 89.3 | +0.2 | 76.2 | +0.6 | +0.48 |
| std_randaugment | Standard Aug | 90.5 | +0.2 | 84.0 | +0.6 | 89.1 | -0.0 | 76.6 | +1.1 | +0.45 |
| gen_flux_kontext | Generative | 90.7 | +0.3 | 83.6 | +0.2 | 89.3 | +0.1 | 76.7 | +1.1 | +0.45 |
| gen_CNetSeg | Generative | 90.6 | +0.3 | 84.0 | +0.6 | 89.4 | +0.2 | 76.2 | +0.7 | +0.43 |
| gen_augmenters | Generative | 90.6 | +0.3 | 84.3 | +0.9 | 89.3 | +0.2 | 75.7 | +0.1 | +0.37 |
| gen_SUSTechGAN | Generative | 90.7 | +0.3 | 83.8 | +0.4 | 89.3 | +0.1 | 76.2 | +0.6 | +0.35 |
| gen_VisualCloze | Generative | 90.7 | +0.4 | 83.7 | +0.3 | 89.1 | -0.0 | 76.2 | +0.7 | +0.33 |
| gen_CUT | Generative | 90.7 | +0.3 | 83.9 | +0.5 | 89.2 | +0.0 | 76.0 | +0.4 | +0.33 |
| gen_Qwen_Image_Edit | Generative | 90.7 | +0.3 | 84.1 | +0.7 | 89.3 | +0.1 | 75.6 | +0.0 | +0.30 |
| gen_automold | Generative | 90.7 | +0.4 | 83.9 | +0.4 | 89.3 | +0.1 | 75.8 | +0.2 | +0.27 |
| gen_step1x_new | Generative | 90.8 | +0.4 | 83.5 | +0.1 | 89.2 | +0.0 | 76.1 | +0.5 | +0.26 |
| gen_albumentations_weather | Generative | 90.7 | +0.3 | 83.9 | +0.5 | 89.2 | +0.1 | 75.6 | +0.1 | +0.24 |
| gen_Weather_Effect_Generator | Generative | 90.7 | +0.3 | 83.6 | +0.2 | 89.2 | +0.0 | 76.0 | +0.4 | +0.24 |
| gen_UniControl | Generative | 90.8 | +0.5 | 83.4 | -0.0 | 89.2 | +0.0 | 76.0 | +0.5 | +0.24 |
| photometric_distort | Augmentation | 90.6 | +0.2 | 84.4 | +1.0 | 89.2 | +0.0 | 75.3 | -0.3 | +0.23 |
| gen_cyclediffusion | Generative | 90.8 | +0.4 | 83.4 | -0.0 | 89.3 | +0.1 | 75.8 | +0.2 | +0.20 |
| gen_IP2P | Generative | 90.7 | +0.3 | 83.6 | +0.1 | 89.3 | +0.1 | 75.7 | +0.1 | +0.18 |
| std_mixup | Standard Aug | 90.3 | -0.1 | 83.1 | -0.3 | 89.1 | -0.1 | 76.6 | +1.0 | +0.14 |
| gen_LANIT | Generative | 90.5 | +0.1 | 83.3 | -0.1 | 89.3 | +0.1 | 75.9 | +0.3 | +0.13 |
| gen_TSIT | Generative | 90.6 | +0.2 | 83.6 | +0.2 | 89.2 | +0.0 | 75.6 | +0.0 | +0.10 |
| gen_step1x_v1p2 | Generative | 90.7 | +0.3 | 83.9 | +0.5 | 89.2 | +0.0 | 74.9 | -0.7 | +0.06 |
| std_cutmix | Standard Aug | 90.6 | +0.2 | 83.5 | +0.0 | 88.8 | -0.4 | 75.7 | +0.1 | +0.01 |
| baseline | Baseline | 90.4 | +0.0 | 83.4 | -0.0 | 89.2 | -0.0 | 75.6 | +0.0 | -0.00 |

## Per-Domain aAcc by Strategy

| Strategy                     | Type         |   clear_day |   cloudy |   dawn_dusk |   foggy |   night |   rainy |   snowy |   Normal Avg |   Adverse Avg |   Gap |
|:-----------------------------|:-------------|------------:|---------:|------------:|--------:|--------:|--------:|--------:|-------------:|--------------:|------:|
| baseline                     | Baseline     |       85.74 |    86.97 |       80.44 |   85.32 |   75.74 |   79.65 |   77.72 |        86.36 |         79.61 |  6.75 |
| gen_Attribute_Hallucination  | Generative   |       86.14 |    88.09 |       82.21 |   85.97 |   76.9  |   80.23 |   78.19 |        87.11 |         80.32 |  6.79 |
| gen_CNetSeg                  | Generative   |       86.12 |    87.72 |       81.55 |   85.95 |   76.63 |   80.02 |   77.77 |        86.92 |         80.09 |  6.83 |
| gen_CUT                      | Generative   |       85.95 |    87.59 |       80.46 |   86.22 |   76.44 |   79.82 |   78.21 |        86.77 |         80.17 |  6.6  |
| gen_IP2P                     | Generative   |       85.9  |    87.38 |       80.88 |   85.15 |   76.1  |   79.73 |   77.87 |        86.64 |         79.71 |  6.93 |
| gen_Img2Img                  | Generative   |       86.12 |    87.9  |       80.88 |   85.82 |   77.06 |   80.07 |   77.79 |        87.01 |         80.19 |  6.83 |
| gen_LANIT                    | Generative   |       85.93 |    87.27 |       80.5  |   85.01 |   75.97 |   79.89 |   77.43 |        86.6  |         79.58 |  7.03 |
| gen_Qwen_Image_Edit          | Generative   |       85.93 |    87.6  |       81.01 |   86.01 |   76.24 |   79.72 |   78.09 |        86.77 |         80.01 |  6.75 |
| gen_SUSTechGAN               | Generative   |       85.91 |    87.71 |       81.51 |   85.98 |   76.4  |   79.83 |   78.52 |        86.81 |         80.18 |  6.63 |
| gen_TSIT                     | Generative   |       85.87 |    87.53 |       80.84 |   85.54 |   75.17 |   79.56 |   77.37 |        86.7  |         79.41 |  7.29 |
| gen_UniControl               | Generative   |       85.98 |    87.46 |       80.69 |   85.81 |   76.36 |   79.91 |   77.32 |        86.72 |         79.85 |  6.87 |
| gen_VisualCloze              | Generative   |       86.01 |    87.51 |       81.24 |   85.73 |   76.5  |   80.03 |   77.85 |        86.76 |         80.03 |  6.74 |
| gen_Weather_Effect_Generator | Generative   |       85.97 |    87.55 |       81.45 |   85.63 |   75.75 |   79.98 |   77.33 |        86.76 |         79.67 |  7.09 |
| gen_albumentations_weather   | Generative   |       85.92 |    87.6  |       81.03 |   85.73 |   76.55 |   79.67 |   77.49 |        86.76 |         79.86 |  6.9  |
| gen_augmenters               | Generative   |       86.02 |    87.81 |       81.04 |   86.17 |   76.39 |   79.92 |   77.59 |        86.91 |         80.02 |  6.9  |
| gen_automold                 | Generative   |       85.93 |    87.67 |       81.14 |   85.72 |   76.07 |   80.14 |   77.45 |        86.8  |         79.85 |  6.96 |
| gen_cycleGAN                 | Generative   |       86.12 |    87.88 |       81.6  |   86.09 |   77.2  |   80.05 |   77.87 |        87    |         80.3  |  6.7  |
| gen_cyclediffusion           | Generative   |       86.07 |    87.17 |       80.2  |   85.08 |   76.09 |   80.06 |   77.82 |        86.62 |         79.76 |  6.86 |
| gen_flux_kontext             | Generative   |       86.11 |    87.83 |       81.71 |   85.74 |   76.23 |   80.09 |   77.67 |        86.97 |         79.93 |  7.04 |
| gen_stargan_v2               | Generative   |       86.11 |    87.78 |       81.29 |   86.25 |   76.51 |   80.35 |   78.12 |        86.95 |         80.31 |  6.64 |
| gen_step1x_new               | Generative   |       85.92 |    87.67 |       80.51 |   85.82 |   76.05 |   79.82 |   78.08 |        86.79 |         79.94 |  6.85 |
| gen_step1x_v1p2              | Generative   |       85.77 |    87.4  |       80.53 |   85.61 |   76.41 |   79.29 |   77.35 |        86.59 |         79.67 |  6.92 |
| photometric_distort          | Augmentation |       85.8  |    87.9  |       80.59 |   85.66 |   76.9  |   79.6  |   77.74 |        86.85 |         79.97 |  6.87 |
| std_autoaugment              | Standard Aug |       86.07 |    87.56 |       83.31 |   85.69 |   79.13 |   80.1  |   77.56 |        86.82 |         80.62 |  6.19 |
| std_cutmix                   | Standard Aug |       85.76 |    87.38 |       80.48 |   85.49 |   75.44 |   79.43 |   77.3  |        86.57 |         79.42 |  7.15 |
| std_mixup                    | Standard Aug |       86    |    87.37 |       80.71 |   85.05 |   75.06 |   80.11 |   77.38 |        86.69 |         79.4  |  7.28 |
| std_randaugment              | Standard Aug |       85.95 |    87.77 |       82.78 |   85.67 |   78.96 |   79.95 |   77.75 |        86.86 |         80.58 |  6.28 |