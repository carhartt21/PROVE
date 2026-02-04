# Training Tracker - Stage 1 (Clear Day)

**Last Updated:** 2026-02-04 16:08

---

## ⚠️ CRITICAL: gen_* Results Invalid (2026-01-28)

> **MixedDataLoader was NEVER connected!** Generated images were never loaded during training.
> 
> - All `gen_*` models trained identically to baseline (only PhotoMetricDistortion differed)
> - Ratio parameter had NO EFFECT on training
> - **Bug Status:** ✅ FIXED | **Retraining:** ⏳ Required
> 
> See [BUG_REPORT](BUG_REPORT_CROSS_DATASET_CONTAMINATION.md) for details.

---

> ⚠️ **MapillaryVistas BGR/RGB Bug (2026-01-21) - RESOLVED**
> All 81 MapillaryVistas models were retrained after BGR/RGB channel fix.

## Progress Summary (⚠️ gen_* results INVALID)

| Category | Configs | Complete (4/4) | Partial | Running | Pending | Failed |
|----------|---------|----------------|---------|---------|---------|--------|
| **Generative (gen_*)** | 83 | 0 | 8 | 9 | 5 | 62 |
| **Standard (std_*)** | 28 | 4 | 4 | 6 | 8 | 6 |
| **TOTAL** | 111 | 4 | 12 | 15 | 13 | 68 |

### Individual Model Trainings

| Category | Total Models | ✅ Complete | 🔄 Running | ⏳ Pending | ❌ Failed |
|----------|-------------|-------------|------------|-----------|----------|
| **Generative (gen_*)** | 332 | 70 | 9 | 132 | 125 |
| **Standard (std_*)** | 112 | 46 | 9 | 47 | 10 |
| **TOTAL** | 444 | 116 | 18 | 179 | 135 |

### Generative Image Augmentation Strategies (❌ INVALID)

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Notes |
|----------|--------|--------|-----------------|------------|-------|
| gen_Attribute_Hallucination | ❌ | ❌ | ❌ | ❌ | Needs retraining |
| gen_augmenters | ❌ | ❌ | ❌ | ❌ | Needs retraining |
| gen_automold | ❌ | ❌ | ❌ | ❌ | Needs retraining |
| gen_CNetSeg | ❌ | ❌ | ❌ | ❌ | Needs retraining |
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
| baseline | 🔄 | 🔄 | 3/4 | 3/4 |  |
| std_minimal | ⏳ | ⏳ | ⏳ | ⏳ |  |
| std_photometric_distort | ⏳ | ⏳ | ⏳ | ⏳ |  |
| std_autoaugment | ✅ | ✅ | 3/4 | 3/4 |  |
| std_cutmix | ✅ | ✅ | 🔄 | 🔄 |  |
| std_mixup | 🔄 | 🔄 | ❌ | ❌ |  |
| std_randaugment | ❌ | ❌ | ❌ | ❌ |  |
