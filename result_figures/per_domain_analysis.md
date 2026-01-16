# Per-Domain Analysis: Normal vs Adverse Weather

**Results with per-domain breakdown: 181 / 181**

## Weather Domain Categories
- **Normal**: clear_day, cloudy
- **Adverse**: foggy, night, rainy, snowy
- **Transition**: dawn_dusk

## Strategy Weather Performance

| Strategy | Normal mIoU | Adverse mIoU | Domain Gap | Count |
|----------|-------------|--------------|------------|-------|
| gen_EDICT | 47.23 | 43.74 | +3.49 | 5 |
| std_randaugment | 35.73 | 27.69 | +8.04 | 6 |
| gen_cyclediffusion | 31.19 | 27.26 | +3.92 | 5 |
| photometric_distort | 34.95 | 27.11 | +7.85 | 6 |
| std_cutmix | 31.87 | 24.91 | +6.96 | 7 |
| gen_Attribute_Hallucination | 26.25 | 24.02 | +2.24 | 11 |
| baseline | 26.87 | 23.64 | +3.23 | 11 |
| gen_VisualCloze | 25.07 | 22.63 | +2.44 | 6 |
| std_mixup | 25.30 | 22.62 | +2.68 | 4 |
| gen_TSIT | 24.81 | 22.55 | +2.26 | 6 |
| gen_UniControl | 25.16 | 22.45 | +2.71 | 6 |
| gen_Img2Img | 24.14 | 22.13 | +2.02 | 7 |
| gen_stargan_v2 | 24.04 | 21.63 | +2.41 | 6 |
| gen_CUT | 22.21 | 20.65 | +1.56 | 11 |
| gen_step1x_v1p2 | 23.10 | 20.52 | +2.58 | 6 |
| gen_automold | 22.48 | 18.96 | +3.52 | 9 |
| gen_augmenters | 20.96 | 18.60 | +2.36 | 9 |
| gen_albumentations_weather | 20.22 | 17.77 | +2.45 | 9 |
| gen_SUSTechGAN | 19.56 | 17.47 | +2.09 | 5 |
| gen_CNetSeg | 18.87 | 17.11 | +1.77 | 8 |
| gen_cycleGAN | 18.61 | 16.59 | +2.02 | 7 |
| gen_IP2P | 16.98 | 15.41 | +1.57 | 8 |
| gen_LANIT | 15.42 | 13.92 | +1.49 | 5 |
| gen_step1x_new | 12.43 | 10.76 | +1.68 | 3 |
| gen_Weather_Effect_Generator | 10.03 | 9.02 | +1.01 | 5 |
| std_autoaugment | 10.40 | 8.26 | +2.13 | 3 |
| gen_Qwen_Image_Edit | 7.66 | 6.34 | +1.32 | 4 |
| gen_flux_kontext | 3.41 | 3.49 | -0.08 | 3 |
