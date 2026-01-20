# Training Tracker - Stage 1 (Clear Day)

**Last Updated:** 2026-01-20 15:15

> ⚠️ **Class Configuration Issue (2026-01-20):** 7 models were found trained with incorrect class counts (19 instead of native 66/24 for MapillaryVistas/OUTSIDE15k). Retraining jobs submitted: 9669526-9669538. See TODO.md for details.

## Progress Summary

| Category | Total | Complete | Partial | Running | Pending | Failed |
|----------|-------|----------|---------|---------|---------|--------|
| **Generative (gen_*)** | 83 | 84 | 0 | 0 | 0 | 0 |
| **Standard (std_*)** | 24 | 24 | 0 | 0 | 0 | 0 |
| **TOTAL** | 107 | 108 | 0 | 0 | 0 | 0 |

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
| std_autoaugment | ✅ | ✅ | ✅ | ✅ |  |
| std_cutmix | ✅ | ✅ | ✅ | ✅ |  |
| std_mixup | ✅ | ✅ | ✅ | ✅ |  |
| std_randaugment | ✅ | ✅ | ✅ | ✅ |  |
