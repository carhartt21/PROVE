# Training Tracker - Stage 1 (Clear Day)

**Last Updated:** 2026-01-19 16:30

## Progress Summary

| Category | Total | Complete | Partial | Running | Pending | Failed |
|----------|-------|----------|---------|---------|---------|--------|
| **Generative (gen_*)** | 84 | 84 | 0 | 0 | 0 | 0 |
| **Standard (std_*)** | 48 | 0 | 0 | 48 | 0 | 0 |
| **TOTAL** | 132 | 84 | 0 | 48 | 0 | 0 |

> **Note:** All 48 std_* models are being retrained due to a bug fix (StandardAugmentationHook).
> Old models backed up at `/scratch/aaa_exchange/AWARE/WEIGHTS_STD_OLD/`
> Jobs: 9660252-9660299. Monitor: `bjobs -w | grep tr_std`

### Generative Image Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Notes |
|----------|--------|--------|-----------------|------------|-------|
| gen_Attribute_Hallucination | ✅ | ✅ | ✅ | ✅ |  |
| gen_augmenters | ✅ | ✅ | ✅ | ✅ |  |
| gen_automold | ✅ | ✅ | ✅ | ✅ |  |
| gen_CNetSeg | ✅ | ✅ | ✅ | ✅ |  |
| gen_CUT | ✅ | ✅ | ✅ | ✅ |  |
| gen_cyclediffusion | ✅ | ✅ | ✅ | ✅ |  |
| gen_cycleGAN | ✅ | ✅ | ✅ | ✅ |  |
| gen_flux_kontext | ✅ | ✅ | ✅ | ✅ |  |
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
| baseline | ✅ | ✅ | ✅ | ✅ |  |
| photometric_distort | ✅ | ✅ | ✅ | ✅ |  |
| std_autoaugment | 🔄 | 🔄 | 🔄 | 🔄 | Retraining (Jobs 9660276-9660287) |
| std_cutmix | 🔄 | 🔄 | 🔄 | 🔄 | Retraining (Jobs 9660252-9660263) |
| std_mixup | 🔄 | 🔄 | 🔄 | 🔄 | Retraining (Jobs 9660264-9660275) |
| std_randaugment | 🔄 | 🔄 | 🔄 | 🔄 | Retraining (Jobs 9660288-9660299) |
