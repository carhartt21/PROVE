# Training Tracker - Stage 1 (Clear Day)

**Last Updated:** 2026-01-21 18:10

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
| **Generative (gen_*)** | 83 | 62 | 0 | 16 | 5 | 0 |
| **Standard (std_*)** | 24 | 18 | 0 | 5 | 1 | 0 |
| **TOTAL** | 107 | 80 | 0 | 21 | 6 | 0 |

> **Note:** MapillaryVistas column shows 🔄 for all strategies (retraining in progress).

### Generative Image Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Notes |
|----------|--------|--------|-----------------|------------|-------|
| gen_Attribute_Hallucination | ✅ | ✅ | 🔄 | ✅ | MapillaryVistas retraining |
| gen_augmenters | ✅ | ✅ | 🔄 | ✅ | MapillaryVistas retraining |
| gen_automold | ✅ | ✅ | 🔄 | ✅ | MapillaryVistas retraining |
| gen_CNetSeg | ✅ | ✅ | 🔄 | ✅ | MapillaryVistas retraining |
| gen_CUT | ✅ | ✅ | 🔄 | ✅ | MapillaryVistas retraining |
| gen_cyclediffusion | ✅ | ✅ | 🔄 | ✅ | MapillaryVistas retraining |
| gen_cycleGAN | ✅ | ✅ | 🔄 | ✅ | MapillaryVistas retraining |
| gen_flux_kontext | ✅ | ✅ | 🔄 | ✅ | MapillaryVistas retraining |
| gen_Img2Img | ✅ | ✅ | 🔄 | ✅ | MapillaryVistas retraining |
| gen_IP2P | ✅ | ✅ | 🔄 | ✅ | MapillaryVistas retraining |
| gen_LANIT | ✅ | ✅ | 🔄 | ✅ | MapillaryVistas retraining |
| gen_Qwen_Image_Edit | ✅ | ✅ | 🔄 | ✅ | MapillaryVistas retraining |
| gen_stargan_v2 | ✅ | ✅ | 🔄 | ✅ | MapillaryVistas retraining |
| gen_step1x_new | ✅ | ✅ | 🔄 | ✅ | MapillaryVistas retraining |
| gen_step1x_v1p2 | ✅ | ✅ | 🔄 | ✅ | MapillaryVistas retraining |
| gen_SUSTechGAN | ✅ | ✅ | 🔄 | ✅ | MapillaryVistas retraining |
| gen_TSIT | ✅ | ✅ | 🔄 | ✅ | MapillaryVistas retraining |
| gen_UniControl | ✅ | ✅ | 🔄 | ✅ | MapillaryVistas retraining |
| gen_VisualCloze | ✅ | ✅ | 🔄 | ✅ | MapillaryVistas retraining |
| gen_Weather_Effect_Generator | ✅ | ✅ | 🔄 | ✅ | MapillaryVistas retraining |
| gen_albumentations_weather | ✅ | ✅ | 🔄 | ✅ | MapillaryVistas retraining |
### Standard Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Notes |
|----------|--------|--------|-----------------|------------|-------|
| baseline | ✅ | ✅ | 🔄 | ✅ | MapillaryVistas retraining |
| photometric_distort | ✅ | ✅ | 🔄 | ✅ | MapillaryVistas retraining |
| std_autoaugment | ✅ | ✅ | 🔄 | ✅ | MapillaryVistas retraining |
| std_cutmix | ✅ | ✅ | 🔄 | ✅ | MapillaryVistas retraining |
| std_mixup | ✅ | ✅ | 🔄 | ✅ | MapillaryVistas retraining |
| std_randaugment | ✅ | ✅ | 🔄 | ✅ | MapillaryVistas retraining |
