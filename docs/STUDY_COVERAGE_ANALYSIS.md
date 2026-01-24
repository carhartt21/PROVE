# Study Coverage Analysis

**Last Updated:** 2026-01-23 (17:00)

## Summary

| Study | Path | Checkpoints | Strategies | Owner | Status |
|-------|------|-------------|------------|-------|--------|
| **Stage 1** | `WEIGHTS/` | 405 | 27 | mima2416 | ✅ **COMPLETE** |
| **Stage 2** | `WEIGHTS_STAGE_2/` | 291 | 27 | mima2416 | 🔄 MV 59% |
| **Ratio Ablation** | `WEIGHTS_RATIO_ABLATION/` | 187 | 6 | mima2416 | ✅ Valid (non-MV) |
| **Extended Training** | `WEIGHTS_EXTENDED/` | ~959 | 9 | chge7185 | ✅ Valid (non-MV) |
| **Combinations** | `WEIGHTS_COMBINATIONS/` | 53 | 27 | chge7185 | ✅ Valid (IDD-AW) |
| **Domain Adaptation** | Testing-only | N/A | 6 | mima2416 | ⏳ Ready |

### MapillaryVistas BGR/RGB Bug Status

The BGR/RGB bug in `custom_transforms.py` affected all MapillaryVistas training. Buggy checkpoints have been backed up:

| Study | MV Status | Backup Location |
|-------|-----------|-----------------|
| **Stage 1** | ✅ **COMPLETE** | All 81/81 retrained + tested |
| **Stage 2** | 🔄 59% (48/81) | Retraining in progress |
| **Ratio Ablation** | 📦 Backed up | `WEIGHTS_BACKUP_BUGGY_MAPILLARY/ratio_ablation/` |
| **Extended Training** | 📦 Backed up | `WEIGHTS_BACKUP_BUGGY_MAPILLARY/extended_training/` |
| **Combinations** | 📦 Backed up | `WEIGHTS_BACKUP_BUGGY_MAPILLARY/combinations/` |

**Note:** Backed up checkpoints are INVALID and cannot be used.

---

## Stage 1: Clear Day Training

**Path:** `/scratch/aaa_exchange/AWARE/WEIGHTS/`
**Status:** ✅ **COMPLETE** (Training 405/405 + Testing 405/405)

### Baseline Analysis Results

Publication-ready analysis available at `result_figures/baseline_consolidated/stage1_baseline_output/`:

| Metric | Value |
|--------|-------|
| Overall mIoU | 33.3% |
| Domain Gap | 10.1% |
| Most Robust Model | SegFormer (8.7% gap) |
| Hardest Domain | Night (-14.9% from Clear Day) |
| Largest Dataset Gap | IDD-AW (17.6%) |
| Smallest Dataset Gap | Mapillary (2.6%) |

### Coverage Matrix

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Total |
|----------|:------:|:------:|:---------------:|:----------:|------:|
| baseline | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_Attribute_Hallucination | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_CNetSeg | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_CUT | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_IP2P | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_Img2Img | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_LANIT | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_Qwen_Image_Edit | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_SUSTechGAN | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_TSIT | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_UniControl | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_VisualCloze | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_Weather_Effect_Generator | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_albumentations_weather | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_augmenters | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_automold | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_cycleGAN | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_cyclediffusion | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_flux_kontext | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_stargan_v2 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_step1x_new | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_step1x_v1p2 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| photometric_distort | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| std_autoaugment | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| std_cutmix | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| std_mixup | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| std_randaugment | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |

**Total:** 405 checkpoints (27 strategies × 4 datasets × 3 models) ✅

---

## Stage 2: All Domains Training

**Path:** `/scratch/aaa_exchange/AWARE/WEIGHTS_STAGE_2/`
**Status:** 🔄 Non-MV Complete (243/243) | MapillaryVistas Retraining (59%)

### Leaderboard (Top 10 + Notable)
| Rank | Strategy | mIoU | Gain vs Baseline |
|------|----------|------|------------------|
| 1 | gen_CNetSeg | 43.68% | +0.58 |
| 2 | gen_stargan_v2 | 43.60% | +0.50 |
| 3 | gen_UniControl | 43.59% | +0.49 |
| 4 | gen_cyclediffusion | 43.56% | +0.47 |
| 5 | std_autoaugment | 43.55% | +0.46 |
| ... | ... | ... | ... |
| 23 | baseline | 43.10% | - |
| 27 | std_cutmix | 42.80% | -0.29 |

### Coverage Matrix

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Total |
|----------|:------:|:------:|:---------------:|:----------:|------:|
| baseline | ✅ 3/3 | ✅ 3/3 | 🔄 | ✅ 3/3 | 9+ |
| gen_Attribute_Hallucination | ✅ 3/3 | ✅ 3/3 | 🔄 | ✅ 3/3 | 9+ |
| gen_CNetSeg | ✅ 3/3 | ✅ 3/3 | 🔄 | ✅ 3/3 | 9+ |
| gen_CUT | ✅ 3/3 | ✅ 3/3 | 🔄 | ✅ 3/3 | 9+ |
| gen_IP2P | ✅ 3/3 | ✅ 3/3 | 🔄 | ✅ 3/3 | 9+ |
| gen_Img2Img | ✅ 3/3 | ✅ 3/3 | 🔄 | ✅ 3/3 | 9+ |
| gen_LANIT | ✅ 3/3 | ✅ 3/3 | 🔄 | ✅ 3/3 | 9+ |
| gen_Qwen_Image_Edit | ✅ 3/3 | ✅ 3/3 | 🔄 | ✅ 3/3 | 9+ |
| gen_SUSTechGAN | ✅ 3/3 | ✅ 3/3 | 🔄 | ✅ 3/3 | 9+ |
| gen_TSIT | ✅ 3/3 | ✅ 3/3 | 🔄 | ✅ 3/3 | 9+ |
| gen_UniControl | ✅ 3/3 | ✅ 3/3 | 🔄 | ✅ 3/3 | 9+ |
| gen_VisualCloze | ✅ 3/3 | ✅ 3/3 | 🔄 | ✅ 3/3 | 9+ |
| gen_Weather_Effect_Generator | ✅ 3/3 | ✅ 3/3 | 🔄 | ✅ 3/3 | 9+ |
| gen_albumentations_weather | ✅ 3/3 | ✅ 3/3 | 🔄 | ✅ 3/3 | 9+ |
| gen_augmenters | ✅ 3/3 | ✅ 3/3 | 🔄 | ✅ 3/3 | 9+ |
| gen_automold | ✅ 3/3 | ✅ 3/3 | 🔄 | ✅ 3/3 | 9+ |
| gen_cycleGAN | ✅ 3/3 | ✅ 3/3 | 🔄 | ✅ 3/3 | 9+ |
| gen_cyclediffusion | ✅ 3/3 | ✅ 3/3 | 🔄 | ✅ 3/3 | 9+ |
| gen_flux_kontext | ✅ 3/3 | ✅ 3/3 | 🔄 | ✅ 3/3 | 9+ |
| gen_stargan_v2 | ✅ 3/3 | ✅ 3/3 | 🔄 | ✅ 3/3 | 9+ |
| gen_step1x_new | ✅ 3/3 | ✅ 3/3 | 🔄 | ✅ 3/3 | 9+ |
| gen_step1x_v1p2 | ✅ 3/3 | ✅ 3/3 | 🔄 | ✅ 3/3 | 9+ |
| photometric_distort | ✅ 3/3 | ✅ 3/3 | 🔄 | ✅ 3/3 | 9+ |
| std_autoaugment | ✅ 3/3 | ✅ 3/3 | 🔄 | ✅ 3/3 | 9+ |
| std_cutmix | ✅ 3/3 | ✅ 3/3 | 🔄 | ✅ 3/3 | 9+ |
| std_mixup | ✅ 3/3 | ✅ 3/3 | 🔄 | ✅ 3/3 | 9+ |
| std_randaugment | ✅ 3/3 | ✅ 3/3 | 🔄 | ✅ 3/3 | 9+ |

**Non-MV Complete:** 243/243 (27 strategies × 3 datasets × 3 models) ✅
**MapillaryVistas:** 🔄 48/81 complete (59%), 33 remaining

### Notes
- MapillaryVistas training in progress (48/81 done, 59%)
- Tests will run as training completes using `./scripts/run_stage2_mapillary_tests.sh`

---

## Ratio Ablation Study

**Path:** `/scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION/`
**Owner:** mima2416
**Status:** ✅ Valid (BDD10k, IDD-AW, OUTSIDE15k) | 📦 MV backed up (buggy)

### Purpose
Test the impact of different real/generated image mixing ratios on model performance.

### Ratios Tested
0.00, 0.125, 0.25, 0.375, 0.50*, 0.625, 0.75, 0.875, 1.00

*Note: Ratio 0.50 is the standard training in `WEIGHTS/`

### Coverage Summary

| Strategy | Checkpoints | Datasets | Notes |
|----------|-------------|----------|-------|
| gen_step1x_new | 56 | BDD10k, IDD-AW, MV*, OUTSIDE15k | Most complete |
| gen_step1x_v1p2 | 46 | BDD10k, IDD-AW, MV*, OUTSIDE15k | |
| gen_TSIT | 39 | BDD10k, IDD-AW, MV*, OUTSIDE15k | |
| gen_cycleGAN | 28 | IDD-AW, MV*, OUTSIDE15k | |
| gen_stargan_v2 | 9 | IDD-AW, MV* | Limited |
| gen_cyclediffusion | 9 | IDD-AW | Limited |

**Total Valid Checkpoints:** 187 (after MV backup)

### Notes
- MapillaryVistas checkpoints backed up to `WEIGHTS_BACKUP_BUGGY_MAPILLARY/ratio_ablation/`
- Analysis script: `analysis_scripts/analyze_ratio_ablation.py`
- Visualization: `analysis_scripts/visualize_ratio_ablation.py`

---

## Extended Training Study

**Path:** `/scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED/`
**Owner:** chge7185
**Status:** ✅ Valid (BDD10k, IDD-AW, OUTSIDE15k) | 📦 MV backed up (logs only)

### Purpose
Evaluate training convergence and performance at extended iteration milestones.

### Iterations Tested
40k, 60k, 80k, 100k, 120k, 140k, 160k, 320k

### Coverage Summary

| Strategy | Checkpoints | Datasets | Notes |
|----------|-------------|----------|-------|
| gen_cyclediffusion | 192 | 4 datasets | Most complete |
| gen_step1x_new | 120 | 4 datasets | |
| gen_albumentations_weather | 96 | 3 datasets | |
| gen_automold | 95 | 5 datasets | Partial |
| gen_cycleGAN | 96 | 3 datasets | |
| gen_flux_kontext | 96 | 3 datasets | |
| gen_TSIT | 96 | 3 datasets | |
| gen_UniControl | 96 | 3 datasets | |
| std_randaugment | 72 | 4 datasets | |

**Total Checkpoints:** ~959

### Key Findings (from analysis)
- **160k iterations** captures ~75% of gains at 50% compute cost
- Performance plateaus vary by strategy

### Notes
- MapillaryVistas directories backed up (logs/configs only, no checkpoints saved)
- Analysis: `analysis_scripts/analyze_extended_training.py`
- Report: [docs/EXTENDED_TRAINING_ANALYSIS.md](EXTENDED_TRAINING_ANALYSIS.md)

---

## Combination Strategies Study

**Path:** `/scratch/aaa_exchange/AWARE/WEIGHTS_COMBINATIONS/`
**Owner:** chge7185
**Status:** ✅ Valid (IDD-AW only) | 📦 MV backed up (54 checkpoints)

### Purpose
Test combining generative augmentation with standard augmentation techniques.

### Coverage Summary

| Category | Combinations | Checkpoints |
|----------|-------------|-------------|
| gen_flux_kontext + std_* | 5 | 10 |
| gen_Qwen_Image_Edit + std_* | 5 | 10 |
| gen_step1x_new + std_* | 5 | 9 |
| gen_stargan_v2 + photometric | 1 | 2 |
| gen_Attribute_Hallucination + photometric | 1 | 2 |
| std_* + std_* | 10 | 20 |
| **Total** | **27** | **53** |

### Notes
- All valid checkpoints are IDD-AW only
- Models: pspnet_r50, segformer_mit-b5
- MapillaryVistas backed up to `WEIGHTS_BACKUP_BUGGY_MAPILLARY/combinations/`
- Missing: BDD10k, OUTSIDE15k coverage

---

## Domain Adaptation Ablation

**Path:** Testing-only (uses `WEIGHTS/` checkpoints)
**Status:** ⏳ Ready to start (scripts ready)

### Purpose
Evaluate **cross-dataset domain generalization** using Stage 1 models.

### Study Design
| Source (Training) | Target (Testing) | Domains |
|-------------------|------------------|---------|
| BDD10k models | ACDC | foggy, night, rainy, snowy |
| IDD-AW models | ACDC | foggy, night, rainy, snowy |
| MapillaryVistas models | ACDC | foggy, night, rainy, snowy |
| All datasets | Cityscapes | clear_day |

### Strategies to Test
- Top 5 generative: gen_Qwen_Image_Edit, gen_Attribute_Hallucination, gen_cycleGAN, gen_flux_kontext, gen_step1x_new
- Baseline models

### Research Questions
1. Which training dataset provides best domain generalization?
2. Do generative augmentations improve cross-dataset transfer?
3. Which adverse weather conditions are hardest to transfer to?

### Notes
- **No training required** - uses existing Stage 1 checkpoints
- Script: `./scripts/run_domain_adaptation_tests.py`
- Doc: [docs/DOMAIN_ADAPTATION_ABLATION.md](DOMAIN_ADAPTATION_ABLATION.md)

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ 3/3 | All 3 models complete (deeplabv3plus_r50, pspnet_r50, segformer_mit-b5) |
| 🔶 2/3 | 2 of 3 models complete |
| 🔶 1/3 | 1 of 3 models complete |
| 🔄 | Retraining in progress |
| ⏳ | Pending/queued |
| ⏸️ | Never trained (lower priority, generated data available) |
| ❌ | Not available / no data |
| 📦 | Backed up (buggy checkpoints, INVALID, awaiting retrain) |
| ⚠️ INVALID | Trained with bug, cannot use results |
