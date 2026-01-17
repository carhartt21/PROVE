# PRELIMINARY STRATEGY LEADERBOARD

**Total Valid Results: 346**

## Overall Strategy Performance (mIoU %)

| Strategy | Mean mIoU | Std | Count |
|----------|-----------|-----|-------|
| std_randaugment | 46.74 | 12.49 | 17 |
| photometric_distort | 46.35 | 12.73 | 17 |
| gen_step1x_new | 45.78 | 9.16 | 12 |
| gen_step1x_v1p2 | 45.42 | 9.05 | 12 |
| gen_Qwen_Image_Edit | 45.27 | 8.27 | 12 |
| std_autoaugment | 44.41 | 8.29 | 15 |
| gen_stargan_v2 | 44.21 | 8.95 | 12 |
| gen_Attribute_Hallucination | 43.17 | 6.38 | 13 |
| gen_cycleGAN | 42.99 | 6.53 | 12 |
| gen_flux_kontext | 42.92 | 6.63 | 12 |
| gen_automold | 42.84 | 6.99 | 12 |
| gen_CNetSeg | 42.83 | 6.75 | 13 |
| gen_IP2P | 42.77 | 6.74 | 13 |
| gen_albumentations_weather | 42.77 | 7.10 | 12 |
| gen_SUSTechGAN | 42.70 | 6.78 | 12 |
| gen_CUT | 42.66 | 6.83 | 12 |
| gen_Img2Img | 42.64 | 6.70 | 12 |
| gen_TSIT | 42.56 | 7.29 | 12 |
| gen_VisualCloze | 42.54 | 6.53 | 12 |
| gen_Weather_Effect_Generator | 42.51 | 7.12 | 11 |
| gen_UniControl | 42.49 | 6.92 | 12 |
| gen_LANIT | 42.32 | 7.06 | 12 |
| gen_augmenters | 42.27 | 6.73 | 12 |
| std_mixup | 42.21 | 6.07 | 15 |
| gen_cyclediffusion | 42.19 | 7.26 | 10 |
| std_cutmix | 42.15 | 6.45 | 15 |
| baseline | 41.25 | 6.83 | 13 |
| gen_EDICT | 41.03 | 5.77 | 2 |

## Per-Dataset Breakdown

### bdd10k (83 results)

| Rank | Strategy | mIoU |
|------|----------|------|
| 1 | std_cutmix | 46.51% |
| 2 | gen_Attribute_Hallucination | 46.49% |
| 3 | gen_automold | 46.37% |
| 4 | gen_IP2P | 46.33% |
| 5 | std_autoaugment | 46.33% |
| 6 | gen_Img2Img | 46.32% |
| 7 | gen_step1x_new | 46.32% |
| 8 | gen_albumentations_weather | 46.29% |
| 9 | gen_cyclediffusion | 46.28% |
| 10 | gen_SUSTechGAN | 46.25% |
| 11 | std_randaugment | 46.20% |
| 12 | gen_Weather_Effect_Generator | 46.19% |
| 13 | gen_stargan_v2 | 46.15% |
| 14 | gen_flux_kontext | 46.14% |
| 15 | gen_step1x_v1p2 | 46.07% |
| 16 | std_mixup | 46.04% |
| 17 | gen_cycleGAN | 46.03% |
| 18 | gen_UniControl | 46.03% |
| 19 | gen_CNetSeg | 45.98% |
| 20 | gen_CUT | 45.91% |
| 21 | gen_Qwen_Image_Edit | 45.90% |
| 22 | gen_VisualCloze | 45.88% |
| 23 | gen_augmenters | 45.88% |
| 24 | gen_TSIT | 45.81% |
| 25 | photometric_distort | 45.75% |
| 26 | gen_LANIT | 45.54% |
| 27 | baseline | 44.09% |
| 28 | gen_EDICT | 41.03% |

### idd-aw (104 results)

| Rank | Strategy | mIoU |
|------|----------|------|
| 1 | std_randaugment | 48.10% |
| 2 | photometric_distort | 47.95% |
| 3 | gen_Attribute_Hallucination | 40.80% |
| 4 | gen_CNetSeg | 40.58% |
| 5 | gen_IP2P | 40.57% |
| 6 | std_cutmix | 40.07% |
| 7 | gen_cycleGAN | 40.03% |
| 8 | gen_augmenters | 39.95% |
| 9 | gen_Img2Img | 39.90% |
| 10 | std_autoaugment | 39.84% |
| 11 | gen_cyclediffusion | 39.84% |
| 12 | gen_Qwen_Image_Edit | 39.81% |
| 13 | gen_albumentations_weather | 39.80% |
| 14 | std_mixup | 39.74% |
| 15 | gen_step1x_v1p2 | 39.69% |
| 16 | gen_automold | 39.64% |
| 17 | gen_SUSTechGAN | 39.55% |
| 18 | gen_VisualCloze | 39.55% |
| 19 | gen_stargan_v2 | 39.54% |
| 20 | gen_Weather_Effect_Generator | 39.52% |
| 21 | gen_CUT | 39.52% |
| 22 | gen_LANIT | 39.39% |
| 23 | gen_TSIT | 39.38% |
| 24 | gen_step1x_new | 39.30% |
| 25 | gen_flux_kontext | 39.30% |
| 26 | gen_UniControl | 39.29% |
| 27 | baseline | 38.07% |

### mapillaryvistas (78 results)

| Rank | Strategy | mIoU |
|------|----------|------|
| 1 | gen_Qwen_Image_Edit | 51.97% |
| 2 | gen_cyclediffusion | 51.52% |
| 3 | gen_Weather_Effect_Generator | 49.09% |
| 4 | gen_step1x_new | 48.62% |
| 5 | gen_CNetSeg | 48.39% |
| 6 | gen_TSIT | 48.29% |
| 7 | photometric_distort | 48.26% |
| 8 | gen_Attribute_Hallucination | 48.18% |
| 9 | gen_automold | 48.17% |
| 10 | std_randaugment | 48.16% |
| 11 | gen_IP2P | 48.16% |
| 12 | gen_stargan_v2 | 48.12% |
| 13 | gen_albumentations_weather | 48.11% |
| 14 | gen_CUT | 48.07% |
| 15 | gen_flux_kontext | 48.06% |
| 16 | gen_cycleGAN | 48.03% |
| 17 | gen_step1x_v1p2 | 47.98% |
| 18 | std_mixup | 47.89% |
| 19 | std_cutmix | 47.75% |
| 20 | gen_SUSTechGAN | 47.71% |
| 21 | gen_UniControl | 47.56% |
| 22 | std_autoaugment | 47.46% |
| 23 | gen_LANIT | 47.25% |
| 24 | gen_VisualCloze | 47.09% |
| 25 | gen_Img2Img | 47.02% |
| 26 | baseline | 47.00% |
| 27 | gen_augmenters | 46.15% |

### outside15k (81 results)

| Rank | Strategy | mIoU |
|------|----------|------|
| 1 | gen_step1x_new | 48.87% |
| 2 | std_autoaugment | 48.56% |
| 3 | gen_step1x_v1p2 | 47.94% |
| 4 | gen_Qwen_Image_Edit | 43.39% |
| 5 | gen_stargan_v2 | 43.02% |
| 6 | std_randaugment | 42.23% |
| 7 | photometric_distort | 40.75% |
| 8 | gen_flux_kontext | 38.19% |
| 9 | gen_Attribute_Hallucination | 38.02% |
| 10 | gen_cycleGAN | 37.86% |
| 11 | gen_VisualCloze | 37.65% |
| 12 | std_mixup | 37.62% |
| 13 | gen_Weather_Effect_Generator | 37.44% |
| 14 | gen_cyclediffusion | 37.36% |
| 15 | gen_Img2Img | 37.31% |
| 16 | gen_SUSTechGAN | 37.28% |
| 17 | gen_automold | 37.17% |
| 18 | gen_CUT | 37.15% |
| 19 | gen_augmenters | 37.12% |
| 20 | gen_CNetSeg | 37.11% |
| 21 | gen_LANIT | 37.10% |
| 22 | gen_UniControl | 37.09% |
| 23 | baseline | 36.90% |
| 24 | gen_albumentations_weather | 36.86% |
| 25 | gen_IP2P | 36.75% |
| 26 | gen_TSIT | 36.74% |
| 27 | std_cutmix | 36.36% |

