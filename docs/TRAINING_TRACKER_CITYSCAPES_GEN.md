# Training Tracker - Cityscapes-Gen

**Last Updated:** 2026-02-09 21:41

## Progress Summary

| Category | Configs | Complete (4/4) | Partial | Running | Pending | Failed |
|----------|---------|----------------|---------|---------|---------|--------|
| **Generative (gen_*)** | 20 | 8 | 8 | 3 | 2 | 0 |
| **Standard (std_*)** | 7 | 5 | 0 | 0 | 2 | 0 |
| **TOTAL** | 27 | 13 | 8 | 3 | 4 | 0 |

### Individual Model Trainings

| Category | Total Models | ✅ Complete | 🔄 Running | ⏳ Pending | ❌ Failed |
|----------|-------------|-------------|------------|-----------|----------|
| **Generative (gen_*)** | 80 | 46 | 5 | 33 | 0 |
| **Standard (std_*)** | 28 | 20 | 0 | 8 | 0 |
| **TOTAL** | 108 | 66 | 5 | 41 | 0 |

### Generative Image Augmentation Strategies

| Strategy | Cityscapes | Notes |
|----------|----------|-------|
| gen_Attribute_Hallucination | 🔄 |  |
| gen_augmenters | 1/4 |  |
| gen_automold | ✅ |  |
| gen_CNetSeg | 1/4 |  |
| gen_CUT | 1/4 |  |
| gen_cyclediffusion | ✅ |  |
| gen_cycleGAN | ⏳ |  |
| gen_flux_kontext | ✅ |  |
| gen_Img2Img | 1/4 |  |
| gen_IP2P | 🔄 |  |
| gen_LANIT | ⏳ |  |
| gen_Qwen_Image_Edit | 1/4 | No BDD10k data |
| gen_stargan_v2 | 1/4 |  |
| gen_step1x_new | ✅ |  |
| gen_step1x_v1p2 | ✅ |  |
| gen_SUSTechGAN | ✅ |  |
| gen_TSIT | 1/4 |  |
| gen_UniControl | 🔄 |  |
| gen_VisualCloze | ✅ |  |
| gen_Weather_Effect_Generator | 1/4 |  |
| gen_albumentations_weather | ✅ |  |
### Standard Augmentation Strategies

| Strategy | Cityscapes | Notes |
|----------|----------|-------|
| baseline | ✅ |  |
| std_minimal | ⏳ |  |
| std_photometric_distort | ⏳ |  |
| std_autoaugment | ✅ |  |
| std_cutmix | ✅ |  |
| std_mixup | ✅ |  |
| std_randaugment | ✅ |  |
