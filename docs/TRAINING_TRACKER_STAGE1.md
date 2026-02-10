# Training Tracker - Stage 1 (Clear Day)

**Last Updated:** 2026-02-10 09:52

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
| **Generative (gen_*)** | 83 | 33 | 42 | 9 | 0 | 0 |
| **Standard (std_*)** | 28 | 8 | 10 | 2 | 8 | 0 |
| **TOTAL** | 111 | 41 | 52 | 11 | 8 | 0 |

### Individual Model Trainings

| Category | Total Models | ✅ Complete | 🔄 Running | ⏳ Pending | ❌ Failed |
|----------|-------------|-------------|------------|-----------|----------|
| **Generative (gen_*)** | 332 | 285 | 9 | 42 | 0 |
| **Standard (std_*)** | 112 | 68 | 2 | 42 | 0 |
| **TOTAL** | 444 | 353 | 11 | 84 | 0 |

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
| baseline | 🔄 | 🔄 | 3/4 | 3/4 |  |
| std_minimal | ⏳ | ⏳ | ⏳ | ⏳ |  |
| std_photometric_distort | ⏳ | ⏳ | ⏳ | ⏳ |  |
| std_autoaugment | ✅ | ✅ | 3/4 | 3/4 |  |
| std_cutmix | ✅ | ✅ | 3/4 | 3/4 |  |
| std_mixup | ✅ | ✅ | 3/4 | 3/4 |  |
| std_randaugment | ✅ | ✅ | 3/4 | 3/4 |  |
