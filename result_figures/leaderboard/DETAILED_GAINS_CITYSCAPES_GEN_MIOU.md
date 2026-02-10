# Cityscapes-Gen Detailed Per-Dataset and Per-Domain Analysis (mIoU)

## Per-Dataset mIoU by Strategy

| Strategy | Type | cityscapes | Δcityscapes | acdc | Δacdc | Avg |
|---|---|---:|---:|---:|---:|---:|
| gen_VisualCloze | Generative | 54.1 | +0.1 | 44.7 | +0.2 | +0.14 |
| gen_flux_kontext | Generative | 54.0 | -0.0 | 44.9 | +0.4 | +0.18 |
| baseline | Baseline | 54.0 | +0.0 | 44.5 | +0.0 | +0.00 |
| std_mixup | Standard Aug | 53.9 | -0.1 | 44.7 | +0.2 | +0.04 |
| std_cutmix | Standard Aug | 53.9 | -0.1 | 44.5 | +0.0 | -0.03 |
| std_randaugment | Standard Aug | 53.9 | -0.2 | 44.3 | -0.2 | -0.16 |
| gen_automold | Generative | 53.7 | -0.4 | 44.5 | +0.0 | -0.18 |
| gen_step1x_v1p2 | Generative | 53.8 | -0.2 | 44.2 | -0.3 | -0.28 |
| gen_albumentations_weather | Generative | 53.5 | -0.5 | 44.1 | -0.4 | -0.45 |
| gen_step1x_new | Generative | 53.3 | -0.7 | 44.0 | -0.4 | -0.58 |
| std_autoaugment | Standard Aug | 52.3 | -1.7 | 42.9 | -1.5 | -1.62 |
| gen_SUSTechGAN | Generative | 51.8 | -2.2 | 44.5 | -0.0 | -1.13 |
| gen_cyclediffusion | Generative | 51.6 | -2.4 | 44.6 | +0.1 | -1.17 |
| gen_Img2Img | Generative | 51.5 | -2.5 | 45.4 | +0.9 | -0.82 |
| gen_cycleGAN | Generative | 51.9 | -2.1 | 42.5 | -2.0 | -2.06 |
| gen_Qwen_Image_Edit | Generative | 51.3 | -2.8 | 44.9 | +0.4 | -1.21 |
| gen_Weather_Effect_Generator | Generative | 50.9 | -3.1 | 44.5 | +0.0 | -1.54 |
| gen_IP2P | Generative | 51.1 | -2.9 | 43.5 | -1.0 | -1.95 |
| gen_augmenters | Generative | 51.5 | -2.6 | 43.1 | -1.4 | -1.97 |
| gen_TSIT | Generative | 50.8 | -3.2 | 44.5 | -0.0 | -1.62 |
| gen_CNetSeg | Generative | 50.6 | -3.5 | 44.2 | -0.3 | -1.89 |
| gen_Attribute_Hallucination | Generative | 49.9 | -4.1 | 44.9 | +0.4 | -1.85 |
| gen_stargan_v2 | Generative | 50.3 | -3.7 | 44.0 | -0.5 | -2.11 |
| gen_CUT | Generative | 48.5 | -5.5 | 45.1 | +0.6 | -2.45 |
| gen_UniControl | Generative | 48.2 | -5.9 | 44.6 | +0.1 | -2.88 |

## Per-Domain mIoU by Strategy

| Strategy                     | Type         | clear_day   |   foggy |   night |   rainy |   snowy | Normal Avg   |   Adverse Avg | Gap   |
|:-----------------------------|:-------------|:------------|--------:|--------:|--------:|--------:|:-------------|--------------:|:------|
| baseline                     | Baseline     | -           |   59.3  |   28.41 |   46.33 |   45.11 | -            |         44.79 | -     |
| gen_Attribute_Hallucination  | Generative   | -           |   59.38 |   29.24 |   46.15 |   45.74 | -            |         45.13 | -     |
| gen_CNetSeg                  | Generative   | -           |   58.01 |   27.69 |   47.48 |   45.27 | -            |         44.61 | -     |
| gen_CUT                      | Generative   | -           |   59.01 |   29.32 |   46.85 |   44.8  | -            |         44.99 | -     |
| gen_IP2P                     | Generative   | -           |   58.58 |   28.03 |   45.6  |   44.22 | -            |         44.11 | -     |
| gen_Img2Img                  | Generative   | -           |   58.8  |   28.93 |   47.75 |   46.67 | -            |         45.54 | -     |
| gen_Qwen_Image_Edit          | Generative   | -           |   58.61 |   28.79 |   47.15 |   45.32 | -            |         44.97 | -     |
| gen_SUSTechGAN               | Generative   | -           |   57.83 |   28.89 |   46.03 |   45.55 | -            |         44.57 | -     |
| gen_TSIT                     | Generative   | -           |   59.44 |   28.18 |   45.85 |   45.41 | -            |         44.72 | -     |
| gen_UniControl               | Generative   | -           |   59.35 |   28.2  |   46.41 |   45.74 | -            |         44.93 | -     |
| gen_VisualCloze              | Generative   | -           |   59.58 |   28.8  |   46.31 |   45.42 | -            |         45.03 | -     |
| gen_Weather_Effect_Generator | Generative   | -           |   57.79 |   28.16 |   47.71 |   45.14 | -            |         44.7  | -     |
| gen_albumentations_weather   | Generative   | -           |   58.61 |   28.47 |   46    |   44.68 | -            |         44.44 | -     |
| gen_augmenters               | Generative   | -           |   56.87 |   27.66 |   45.37 |   43.07 | -            |         43.24 | -     |
| gen_automold                 | Generative   | -           |   58.06 |   28    |   47.47 |   45.65 | -            |         44.8  | -     |
| gen_cycleGAN                 | Generative   | -           |   57.39 |   26.7  |   44.82 |   42.81 | -            |         42.93 | -     |
| gen_cyclediffusion           | Generative   | -           |   59.19 |   29.07 |   46.17 |   45.16 | -            |         44.9  | -     |
| gen_flux_kontext             | Generative   | -           |   59    |   29.11 |   46.76 |   45.14 | -            |         45    | -     |
| gen_stargan_v2               | Generative   | -           |   58.3  |   28.34 |   46.1  |   44.63 | -            |         44.34 | -     |
| gen_step1x_new               | Generative   | -           |   57.38 |   27.34 |   47.44 |   45.54 | -            |         44.43 | -     |
| gen_step1x_v1p2              | Generative   | -           |   58.86 |   28.32 |   46.29 |   44.67 | -            |         44.53 | -     |
| std_autoaugment              | Standard Aug | -           |   57.65 |   27.43 |   44.75 |   44.29 | -            |         43.53 | -     |
| std_cutmix                   | Standard Aug | -           |   59.44 |   28.69 |   46.23 |   44.62 | -            |         44.75 | -     |
| std_mixup                    | Standard Aug | -           |   58.53 |   28.59 |   46.64 |   44.93 | -            |         44.67 | -     |
| std_randaugment              | Standard Aug | -           |   58.95 |   28.68 |   46.42 |   45.13 | -            |         44.79 | -     |