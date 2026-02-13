# Training Tracker - Stage 2 (All Domains)

**Last Updated:** 2026-02-13 15:47

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
| **Generative (gen_*)** | 80 | 38 | 18 | 11 | 11 | 2 |
| **Standard (std_*)** | 20 | 9 | 2 | 1 | 8 | 0 |
| **TOTAL** | 100 | 47 | 20 | 12 | 19 | 2 |

### Individual Model Trainings

| Category | Total Models | ✅ Complete | 🔄 Running | ⏳ Pending | ❌ Failed |
|----------|-------------|-------------|------------|-----------|----------|
| **Generative (gen_*)** | 320 | 192 | 18 | 102 | 8 |
| **Standard (std_*)** | 80 | 43 | 1 | 36 | 0 |
| **TOTAL** | 400 | 235 | 19 | 138 | 8 |

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
|----------|------ | ------ | --------------- | ----------|-------|
| baseline | ✅ | 🔄 | ✅ | ✅ |  |
| std_minimal | ⏳ | ⏳ | ⏳ | ⏳ |  |
| std_photometric_distort | ⏳ | ⏳ | ⏳ | ⏳ |  |
| std_autoaugment | ✅ | ✅ | ✅ | ✅ |  |
| std_randaugment | 2/4 | 2/4 | ✅ | ✅ |  |
