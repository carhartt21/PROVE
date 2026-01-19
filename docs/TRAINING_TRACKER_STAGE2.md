# Training Tracker - Stage 2 (All Domains)

**Last Updated:** 2026-01-19 09:25

## Progress Summary

| Category | Total | Complete | Partial | Running | Pending | Failed |
|----------|-------|----------|---------|---------|---------|--------|
| **Generative (gen_*)** | 83 | 76 | 4 | 0 | 4 | 0 |
| **Standard (std_*)** | 24 | 15 | 1 | 0 | 8 | 0 |
| **TOTAL** | 107 | 91 | 5 | 0 | 12 | 0 |

### Generative Image Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Notes |
|----------|--------|--------|-----------------|------------|-------|
| gen_Attribute_Hallucination | ✅ | ✅ | ✅ | ✅ |  |
| gen_augmenters | ✅ | ✅ | ✅ | ✅ |  |
| gen_automold | ✅ | ✅ | ✅ | ✅ |  |
| gen_CNetSeg | 2/3 | 2/3 | ✅ | ✅ |  |
| gen_CUT | ✅ | ✅ | ✅ | ✅ |  |
| gen_cyclediffusion | ⏳ | ⏳ | ⏳ | ⏳ |  |
| gen_cycleGAN | ✅ | ✅ | ✅ | ✅ |  |
| gen_flux_kontext | 1/3 | 1/3 | ✅ | ✅ |  |
| gen_Img2Img | ✅ | ✅ | ✅ | ✅ |  |
| gen_IP2P | ✅ | ✅ | ✅ | ✅ |  |
| gen_LANIT | ✅ | ✅ | ✅ | ✅ |  |
| gen_Qwen_Image_Edit | ✅ | ✅ | ✅ | ✅ | No BDD10k data |
| gen_stargan_v2 | ✅ | ✅ | ✅ | ✅ |  |
| gen_step1x_new | ✅ | ✅ | ✅ | ✅ |  |
| gen_step1x_v1p2 | ✅ | ✅ | ✅ | ✅ |  |
| gen_SUSTechGAN | ✅ | ✅ | ✅ | ✅ |  |
| gen_TSIT | ✅ | ✅ | ✅ | ✅ |  |
| gen_UniControl | ✅ | ✅ | ✅ | ✅ |  |
| gen_VisualCloze | ✅ | ✅ | ✅ | ✅ |  |
| gen_Weather_Effect_Generator | ✅ | ✅ | ✅ | ✅ |  |
| gen_albumentations_weather | ✅ | ✅ | ✅ | ✅ |  |
### Standard Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Notes |
|----------|--------|--------|-----------------|------------|-------|
| baseline | 2/3 | ✅ | ✅ | ✅ |  |
| photometric_distort | ✅ | ✅ | ✅ | ✅ |  |
| std_autoaugment | ✅ | ✅ | ✅ | ✅ |  |
| std_cutmix | ⏳ | ⏳ | ⏳ | ⏳ |  |
| std_mixup | ⏳ | ⏳ | ⏳ | ⏳ |  |
| std_randaugment | ✅ | ✅ | ✅ | ✅ |  |
