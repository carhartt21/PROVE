# Retraining Progress Tracker

**Last Updated:** 2026-01-10 19:52

This document tracks the progress of retraining models with the corrected native class labels.

## Overview

### Native Class Configuration
| Dataset | Classes | Notes |
|---------|---------|-------|
| ACDC | 19 | trainId format |
| Cityscapes | 19 | trainId format |
| BDD10k | 19 | trainId format |
| IDD-AW | 19 | trainId format |
| MapillaryVistas | 66 | Native classes |
| OUTSIDE15k | 24 | Native classes |

### Training Stages

| Stage | Focus | Description | Domain Filter | Status |
|-------|-------|-------------|---------------|--------|
| **Stage 1** | Initial Training | All strategies on clear_day | `clear_day` | 🔄 In Progress |
| Stage 2 | Top Performers | Top 10 methods on all domains | All | ⏳ Pending |
| Stage 3 | Extended Training | Top methods with 160k iterations | All | ⏳ Pending |

### Ablation Studies

| Study | Focus | Description | Status |
|-------|-------|-------------|--------|
| **Ratio Ablation** | Real/Gen ratio | Compare 0.125, 0.25, 0.375. 0.5, 0.625, 0.75, 0,875, 1.0 ratios | ⏳ Pending |
| **Domain Adaptation** | Multi-domain transfer | Test on ACDC (train+val) and Cityscapes (val)  | ⏳ Pending |
| **Combination** | Strategy combinations | Test gen + std and std+std augmentation combos | ⏳ Pending |
---

### Stage Progression Criteria

**Stage 1 → Stage 2:**
- Stage 1 complete with ≥90% configurations trained
- Select top 10 strategies by average mIoU on clear_day

**Stage 2 → Stage 3:**
- Stage 2 complete for selected strategies
- Select top 5 strategies for extended training

**Ablation Studies:**
- Begin after Stage 1 completion
- Run in parallel with Stage 2/3

---

## Stage 1: Clear Day Baseline Training

**Objective:** Train all strategies with `clear_day` domain filter to establish baseline performance.

### Models
- deeplabv3plus_r50 ← **Currently training (102 jobs)**
- pspnet_r50 ← Pending
- segformer_mit-b5 ← Pending

*Note: All three models are trained in Stage 1. Later stages may focus on top-performing models only.*

### Training Configuration
- **Iterations:** 80,000
- **Real/Gen Ratio:** 0.5 (for gen_* strategies), 1.0 (for std_* strategies)
- **Domain Filter:** `clear_day`
- **Job Duration:** 24 hours per job

### Training Progress by Model

| Model | Status | Jobs |
|-------|--------|------|
| deeplabv3plus_r50 | 🔄 In Progress | 102 submitted |
| pspnet_r50 | ⏳ Pending | After DeepLabV3+ completes |
| segformer_mit-b5 | ⏳ Pending | After DeepLabV3+ completes |

---

## Strategy Status Matrix

### Legend
- ✅ Complete with valid weights
- 🔄 Currently training
- ⏳ Pending
- ❌ Failed / Needs attention
- ➖ Not applicable

### Generative Image Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Notes |
|----------|--------|--------|-----------------|------------|-------|
| gen_Attribute_Hallucination | ✅ | ⏳ | 🔄 | ✅ |  |
| gen_augmenters | ✅ | ⏳ | ⏳ | ✅ |  |
| gen_automold | ✅ | ⏳ | ⏳ | ✅ |  |
| gen_CNetSeg | ✅ | ⏳ | 🔄 | ✅ |  |
| gen_CUT | ✅ | ❌ | 🔄 | ✅ |  |
| gen_cyclediffusion | ✅ | ⏳ | ⏳ | ⏳ |  |
| gen_cycleGAN | ✅ | ⏳ | ⏳ | ✅ |  |
| gen_flux_kontext | ➖ | ➖ | ⏳ | 🔄 |  |
| gen_Img2Img | ✅ | ❌ | 🔄 | ✅ |  |
| gen_IP2P | ✅ | ❌ | 🔄 | ✅ |  |
| gen_LANIT | ✅ | ❌ | 🔄 | ✅ |  |
| gen_Qwen_Image_Edit | ➖ | ❌ | 🔄 | ✅ | No BDD10k data |
| gen_stargan_v2 | 🔄 | ⏳ | ⏳ | 🔄 |  |
| gen_step1x_new | ➖ | ⏳ | ⏳ | 🔄 |  |
| gen_step1x_v1p2 | 🔄 | ⏳ | ⏳ | 🔄 |  |
| gen_SUSTechGAN | ✅ | ⏳ | ⏳ | ✅ |  |
| gen_TSIT | ✅ | ⏳ | ⏳ | ✅ |  |
| gen_UniControl | ✅ | ⏳ | ⏳ | ✅ |  |
| gen_VisualCloze | ✅ | ⏳ | ⏳ | ✅ |  |
| gen_Weather_Effect_Generator | ✅ | ⏳ | ⏳ | ✅ |  |
| gen_albumentations_weather | ✅ | ⏳ | ⏳ | ✅ |  |

### Standard Augmentation Strategies

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Notes |
|----------|--------|--------|-----------------|------------|-------|
| baseline | ✅ | ❌ | 🔄 | ✅ |  |
| photometric_distort | ✅ | ✅ | ✅ | ✅ |  |
| std_autoaugment | ✅ | ⏳ | ✅ | ✅ |  |
| std_cutmix | ✅ | ⏳ | ✅ | ✅ |  |
| std_mixup | 🔄 | ⏳ | 🔄 | ✅ |  |
| std_randaugment | ✅ | ✅ | ✅ | ✅ |  |

### Excluded Methods
The following methods are excluded due to insufficient training data coverage:
- **EDICT** - 0/4 training datasets (only ACDC+BDD100k)
- **StyleID** - 0/4 training datasets (only ACDC+BDD100k)
- **flux2** - 0/4 training datasets (only ACDC+BDD100k)
- **AOD-Net** - No manifest / permission denied
- **NST** - Generated images missing (manifest exists but `/scratch/aaa_exchange/AWARE/GENERATED_IMAGES/NST/` does not exist)
- **AOD-Net** - No manifest / permission denied

### Skipped Configurations
The following strategy/dataset combinations are skipped due to incomplete coverage:
- **flux1_kontext/BDD10k** - Only MapillaryVistas and OUTSIDE15k available
- **flux1_kontext/IDD-AW** - Only MapillaryVistas and OUTSIDE15k available
- **step1x_new/BDD10k** - Incomplete coverage (1,212 images via symlinks)

---

## Job Management

### Current Jobs

```bash
# Check running jobs
bjobs -a | grep retrain

# Check job details
bjobs -l <JOB_ID>

# View job output
tail -f logs/retrain/retrain_<strategy>_<JOBID>.out
```

### Submit Jobs

```bash
# Submit all retraining jobs
python scripts/retrain_affected_models.py --submit-all

# Submit specific strategy
python scripts/retrain_affected_models.py --submit-strategy gen_NST

# Generate scripts without submitting
python scripts/retrain_affected_models.py --generate-scripts
```

---

## Progress Summary

### Stage 1 Progress

| Category | Total | Complete | Running | Pending | Failed |
|----------|-------|----------|---------|---------|--------|
| **Generative (gen_*)** | 83 | 32 | 13 | 30 | 5 |
| **Standard (std_*)** | 24 | 17 | 3 | 3 | 1 |
| **TOTAL** | 107 | 49 | 16 | 33 | 6 |

*Note: Stage 1 trains 3 models per strategy×dataset = 324 total configurations.*
*Total = 28 strategies × 4 datasets × 3 models - 12 skipped configs = 324 configs*

---

## Update Script

Run this command to update the tracker status:

```bash
# Update tracker with current status
python scripts/update_retraining_tracker.py

# Verbose mode (shows all configurations)
python scripts/update_retraining_tracker.py --verbose

# Check status only (no update)
python scripts/update_retraining_tracker.py --no-update
```

---

## Changelog

### 2026-01-09
- Initial tracker creation
- Set up Stage 1 focus on clear_day training
- Defined 28 active strategies (22 gen + 6 std)
- Excluded EDICT, StyleID, flux2, AOD-Net due to insufficient data
