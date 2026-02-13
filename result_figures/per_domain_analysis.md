# Per-Domain Analysis: Normal vs Adverse Weather

**Results with per-domain breakdown: 420 / 420**

## Weather Domain Categories
- **Normal**: clear_day, cloudy
- **Adverse**: foggy, night, rainy, snowy
- **Transition**: dawn_dusk

## Strategy Weather Performance

| Strategy | Normal mIoU | Adverse mIoU | Domain Gap | Count |
|----------|-------------|--------------|------------|-------|
| gen_UniControl | 40.84 | 35.55 | +5.29 | 16 |
| gen_automold | 41.19 | 35.53 | +5.66 | 17 |
| gen_albumentations_weather | 40.94 | 35.44 | +5.50 | 17 |
| gen_cyclediffusion | 40.80 | 35.14 | +5.67 | 16 |
| gen_augmenters | 40.65 | 35.07 | +5.57 | 16 |
| gen_Qwen_Image_Edit | 40.71 | 35.05 | +5.66 | 16 |
| gen_stargan_v2 | 40.68 | 35.02 | +5.66 | 16 |
| std_cutmix | 40.64 | 34.98 | +5.66 | 16 |
| std_randaugment | 40.71 | 34.93 | +5.78 | 16 |
| gen_SUSTechGAN | 40.27 | 34.89 | +5.38 | 16 |
| std_autoaugment | 40.45 | 34.88 | +5.56 | 16 |
| gen_CNetSeg | 40.54 | 34.82 | +5.72 | 16 |
| gen_step1x_v1p2 | 40.26 | 34.79 | +5.47 | 17 |
| gen_IP2P | 40.46 | 34.79 | +5.67 | 16 |
| gen_flux_kontext | 40.47 | 34.78 | +5.68 | 16 |
| gen_VisualCloze | 40.45 | 34.75 | +5.70 | 16 |
| gen_Attribute_Hallucination | 40.43 | 34.72 | +5.71 | 16 |
| gen_Img2Img | 40.62 | 34.72 | +5.90 | 16 |
| gen_CUT | 40.22 | 34.66 | +5.56 | 16 |
| std_mixup | 40.39 | 34.61 | +5.78 | 16 |
| gen_TSIT | 40.32 | 34.57 | +5.75 | 16 |
| gen_Weather_Effect_Generator | 40.40 | 34.57 | +5.84 | 16 |
| gen_cycleGAN | 39.96 | 34.43 | +5.53 | 16 |
| gen_step1x_new | 40.11 | 34.31 | +5.81 | 16 |
| gen_LANIT | 39.86 | 33.76 | +6.11 | 14 |
| baseline | 35.10 | 29.96 | +5.14 | 19 |
