# Training Tracker - Stage 2 (All Domains)

**Last Updated:** 2026-01-16 14:54

## Progress Summary

| Category | Total | Complete | Partial | Running | Pending | Failed |
|----------|-------|----------|---------|---------|---------|--------|
| **Generative (gen_*)** | 83 | 3 | 43 | 0 | 38 | 0 |
| **Standard (std_*)** | 28 | 3 | 5 | 0 | 19 | 1 |
| **TOTAL** | 111 | 6 | 48 | 0 | 57 | 1 |

### Generative Image Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Notes |
|----------|--------|--------|-----------------|------------|-------|
| gen_Attribute_Hallucination | ✅ | 2/3 | ⏳ | ⏳ |  |
| gen_augmenters | 2/3 | 2/3 | 2/3 | 2/3 |  |
| gen_automold | 2/3 | 2/3 | 2/3 | 2/3 |  |
| gen_CNetSeg | 2/3 | 2/3 | 2/3 | 2/3 |  |
| gen_CUT | ✅ | ✅ | ⏳ | ⏳ |  |
| gen_cyclediffusion | ⏳ | ⏳ | ⏳ | ⏳ |  |
| gen_cycleGAN | 2/3 | 2/3 | 2/3 | 2/3 |  |
| gen_flux_kontext | ⏳ | ⏳ | ⏳ | ⏳ |  |
| gen_Img2Img | 2/3 | ⏳ | ⏳ | ⏳ |  |
| gen_IP2P | 1/3 | 2/3 | ⏳ | ⏳ |  |
| gen_LANIT | 1/3 | ⏳ | ⏳ | ⏳ |  |
| gen_Qwen_Image_Edit | ⏳ | 1/3 | ⏳ | ⏳ | No BDD10k data |
| gen_stargan_v2 | ⏳ | ⏳ | ⏳ | ⏳ |  |
| gen_step1x_new | ⏳ | ⏳ | ⏳ | ⏳ |  |
| gen_step1x_v1p2 | ⏳ | ⏳ | ⏳ | ⏳ |  |
| gen_SUSTechGAN | 2/3 | 2/3 | 2/3 | 2/3 |  |
| gen_TSIT | 2/3 | 2/3 | 1/3 | 1/3 |  |
| gen_UniControl | 1/3 | 2/3 | 2/3 | 2/3 |  |
| gen_VisualCloze | 2/3 | 2/3 | 2/3 | 2/3 |  |
| gen_Weather_Effect_Generator | 1/3 | ⏳ | ⏳ | ⏳ |  |
| gen_albumentations_weather | 2/3 | 2/3 | 2/3 | 2/3 |  |
### Standard Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Notes |
|----------|--------|--------|-----------------|------------|-------|
| baseline | 2/3 | ✅ | 2/3 | 2/3 |  |
| photometric_distort | ⏳ | ❌ | ⏳ | ⏳ |  |
| std_minimal | 1/3 | 1/3 | ✅ | ✅ |  |
| std_autoaugment | ⏳ | ⏳ | ⏳ | ⏳ |  |
| std_cutmix | ⏳ | ⏳ | ⏳ | ⏳ |  |
| std_mixup | ⏳ | ⏳ | ⏳ | ⏳ |  |
| std_randaugment | ⏳ | ⏳ | ⏳ | ⏳ |  |
