# Cityscapes-Gen Detailed Per-Dataset and Per-Domain Analysis (mIoU)

## Per-Dataset mIoU by Strategy

| Strategy | Type | cityscapes | Δcityscapes | acdc | Δacdc | Avg |
|---|---|---:|---:|---:|---:|---:|
| gen_Attribute_Hallucination | Generative | 63.4 | +1.1 | 44.9 | +1.9 | +1.49 |
| gen_Img2Img | Generative | 62.4 | +0.1 | 43.3 | +0.4 | +0.22 |
| gen_augmenters | Generative | 62.5 | +0.2 | 43.1 | +0.1 | +0.18 |
| gen_Qwen_Image_Edit | Generative | 62.1 | -0.2 | 43.2 | +0.2 | +0.02 |
| baseline | Baseline | 62.3 | +0.0 | 43.0 | +0.0 | +0.00 |
| gen_VisualCloze | Generative | 62.3 | +0.0 | 42.8 | -0.2 | -0.06 |
| gen_CUT | Generative | 62.0 | -0.3 | 43.1 | +0.1 | -0.11 |
| gen_flux_kontext | Generative | 62.0 | -0.3 | 43.0 | -0.0 | -0.16 |
| gen_step1x_v1p2 | Generative | 62.2 | -0.1 | 42.7 | -0.3 | -0.20 |
| std_cutmix | Standard Aug | 62.1 | -0.2 | 42.8 | -0.2 | -0.19 |
| gen_cycleGAN | Generative | 62.1 | -0.2 | 42.7 | -0.3 | -0.21 |
| std_mixup | Standard Aug | 61.9 | -0.4 | 42.8 | -0.1 | -0.28 |
| gen_automold | Generative | 61.7 | -0.6 | 43.0 | +0.0 | -0.29 |
| std_randaugment | Standard Aug | 62.2 | -0.1 | 42.5 | -0.5 | -0.29 |
| gen_cyclediffusion | Generative | 62.0 | -0.3 | 42.6 | -0.4 | -0.31 |
| std_autoaugment | Standard Aug | 61.7 | -0.6 | 42.9 | -0.0 | -0.32 |
| gen_Weather_Effect_Generator | Generative | 61.7 | -0.6 | 42.8 | -0.1 | -0.38 |
| gen_SUSTechGAN | Generative | 61.8 | -0.5 | 42.7 | -0.3 | -0.41 |
| gen_UniControl | Generative | 61.6 | -0.7 | 42.8 | -0.2 | -0.43 |
| gen_TSIT | Generative | 61.9 | -0.4 | 42.5 | -0.5 | -0.44 |
| gen_CNetSeg | Generative | 61.8 | -0.5 | 42.6 | -0.4 | -0.46 |
| gen_stargan_v2 | Generative | 61.9 | -0.4 | 42.2 | -0.8 | -0.61 |
| gen_albumentations_weather | Generative | 61.8 | -0.5 | 42.3 | -0.7 | -0.61 |
| gen_IP2P | Generative | 62.0 | -0.3 | 41.7 | -1.3 | -0.80 |
| gen_step1x_new | Generative | 61.5 | -0.8 | 42.1 | -0.9 | -0.86 |

## Per-Domain mIoU by Strategy

| Strategy                     | Type         | clear_day   |   foggy |   night |   rainy |   snowy | Normal Avg   |   Adverse Avg | Gap   |
|:-----------------------------|:-------------|:------------|--------:|--------:|--------:|--------:|:-------------|--------------:|:------|
| baseline                     | Baseline     | -           |   57.86 |   27.15 |   45.32 |   44.03 | -            |         43.59 | -     |
| gen_Attribute_Hallucination  | Generative   | -           |   59.38 |   29.24 |   46.15 |   45.74 | -            |         45.13 | -     |
| gen_CNetSeg                  | Generative   | -           |   56.97 |   26.16 |   46.27 |   44.28 | -            |         43.42 | -     |
| gen_CUT                      | Generative   | -           |   57.72 |   27.5  |   45.35 |   43.53 | -            |         43.53 | -     |
| gen_IP2P                     | Generative   | -           |   57.22 |   26.4  |   44.49 |   42.88 | -            |         42.75 | -     |
| gen_Img2Img                  | Generative   | -           |   57.55 |   26.81 |   46.29 |   45.34 | -            |         44    | -     |
| gen_Qwen_Image_Edit          | Generative   | -           |   57.49 |   27.43 |   45.87 |   44.1  | -            |         43.72 | -     |
| gen_SUSTechGAN               | Generative   | -           |   56.7  |   27.28 |   44.85 |   43.93 | -            |         43.19 | -     |
| gen_TSIT                     | Generative   | -           |   58.01 |   26.2  |   44.67 |   44.06 | -            |         43.24 | -     |
| gen_UniControl               | Generative   | -           |   57.9  |   26.67 |   45.21 |   44.43 | -            |         43.55 | -     |
| gen_VisualCloze              | Generative   | -           |   57.96 |   27.32 |   45.06 |   44.04 | -            |         43.6  | -     |
| gen_Weather_Effect_Generator | Generative   | -           |   56.67 |   26.68 |   46.23 |   44.05 | -            |         43.41 | -     |
| gen_albumentations_weather   | Generative   | -           |   57.12 |   26.82 |   44.62 |   43.49 | -            |         43.01 | -     |
| gen_augmenters               | Generative   | -           |   57.81 |   27.5  |   45.71 |   43.97 | -            |         43.75 | -     |
| gen_automold                 | Generative   | -           |   57.18 |   26.7  |   46.24 |   44.48 | -            |         43.65 | -     |
| gen_cycleGAN                 | Generative   | -           |   57.95 |   27.14 |   45.28 |   43.95 | -            |         43.58 | -     |
| gen_cyclediffusion           | Generative   | -           |   57.86 |   27.35 |   44.74 |   43.81 | -            |         43.44 | -     |
| gen_flux_kontext             | Generative   | -           |   57.64 |   27.49 |   45.39 |   43.54 | -            |         43.52 | -     |
| gen_stargan_v2               | Generative   | -           |   56.67 |   26.71 |   44.72 |   43.62 | -            |         42.93 | -     |
| gen_step1x_new               | Generative   | -           |   56.11 |   25.82 |   45.85 |   44.01 | -            |         42.95 | -     |
| gen_step1x_v1p2              | Generative   | -           |   57.42 |   26.98 |   45.37 |   43.56 | -            |         43.33 | -     |
| std_autoaugment              | Standard Aug | -           |   57.65 |   27.43 |   44.75 |   44.29 | -            |         43.53 | -     |
| std_cutmix                   | Standard Aug | -           |   57.82 |   27.32 |   45.01 |   43.06 | -            |         43.3  | -     |
| std_mixup                    | Standard Aug | -           |   57.34 |   27.01 |   45.45 |   43.68 | -            |         43.37 | -     |
| std_randaugment              | Standard Aug | -           |   57.78 |   27.12 |   45.12 |   43.76 | -            |         43.45 | -     |