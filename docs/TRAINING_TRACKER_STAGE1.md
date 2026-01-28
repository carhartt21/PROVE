# Training Tracker - Stage 1 (Clear Day)

**Last Updated:** 2026-01-24 13:01

> ⚠️ **CRITICAL: MapillaryVistas BGR/RGB Bug (2026-01-21)**
> All 81 MapillaryVistas models (Stage 1) have been INVALIDATED due to BGR/RGB channel mismatch in label loading.
> - **Bug:** `mmcv.imfrombytes()` returns BGR, but `MapillaryRGBToClassId` expected RGB
> - **Effect:** Models learned wrong class mappings (e.g., Sky trained as Phone Booth)
> - **Fix:** Commit d7b2b99 - Swapped channel indices in `custom_transforms.py`
> - **Retraining:** 81 jobs submitted (9739253-9739333), ~4-8 hours each
> - **Backup:** Old models at `/scratch/aaa_exchange/AWARE/WEIGHTS_BACKUP_BUGGY_MAPILLARY/stage1/`

## Progress Summary

| Category | Total | Complete | Partial | Running | Pending | Failed |
|----------|-------|----------|---------|---------|---------|--------|
| **Generative (gen_*)** | 83 | 84 | 0 | 0 | 0 | 0 |
| **Standard (std_*)** | 24 | 24 | 0 | 0 | 0 | 0 |
| **TOTAL** | 107 | 108 | 0 | 0 | 0 | 0 |

> **Note:** MapillaryVistas column shows 🔄 for all strategies (retraining in progress).

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
