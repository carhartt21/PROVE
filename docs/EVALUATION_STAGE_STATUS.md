# Evaluation Stage Status

**Last Updated:** 2026-01-19 (17:30)

## Overview

| Stage | Training | Testing | Description |
|-------|----------|---------|-------------|
| **Stage 1** | 403 models | 272 valid tests | Clear-day domain training |
| **Stage 2** | 288 models | 267 tested (93%) | All-domains training |

**Note:** Stage 1 test count reduced from 395 to 272 after cleanup of ratio ablation test results that were incorrectly mixed in WEIGHTS directory.

---

## Stage 1: Clear-Day Domain Training

**Status: ✅ Complete (testing cleanup done)**

### Recent Cleanup (2026-01-19)
- Moved ratio ablation test results from WEIGHTS to WEIGHTS_RATIO_ABLATION
- Removed duplicate test directories (kept _fixed versions)
- Cleaned up multiple test runs (kept latest timestamp only)
- Regenerated `downstream_results.csv` with 272 valid results

### Description
- **Training Domain Filter:** `clear_day` only
- **Weights Directory:** `/scratch/aaa_exchange/AWARE/WEIGHTS/`
- **Purpose:** Train models on clear weather conditions only, evaluate cross-domain robustness

### Coverage
- **Training:** 403/405 models complete (99.5%)
- **Testing:** 272 valid test results (after cleanup)
- **Strategies:** 23 unique strategies
- **Datasets:** 4 (BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k)
- **Models:** 3 (DeepLabV3+, PSPNet, SegFormer)

### Strategies Included
| Category | Strategies |
|----------|------------|
| **Generative (21)** | gen_Attribute_Hallucination, gen_augmenters, gen_automold, gen_CNetSeg, gen_CUT, gen_cyclediffusion, gen_cycleGAN, gen_flux_kontext, gen_Img2Img, gen_IP2P, gen_LANIT, gen_Qwen_Image_Edit, gen_stargan_v2, gen_step1x_new, gen_step1x_v1p2, gen_SUSTechGAN, gen_TSIT, gen_UniControl, gen_VisualCloze, gen_Weather_Effect_Generator, gen_albumentations_weather |
| **Standard (2)** | baseline, photometric_distort |
| **Standard (RETRAINING)** | std_autoaugment, std_cutmix, std_mixup, std_randaugment (48 jobs running)

### Active Retraining
The std_* strategies are being retrained due to a bug fix in the StandardAugmentationHook.
- **Jobs:** 9660252-9660299 (21 RUN, 27 PEND)
- **Monitor:** `bjobs -w | grep tr_std`

### Key Files
- Training Tracker: [TRAINING_TRACKER_STAGE1.md](TRAINING_TRACKER_STAGE1.md)
- Training Coverage: [TRAINING_COVERAGE_STAGE1.md](TRAINING_COVERAGE_STAGE1.md)
- Leaderboard: `result_figures/leaderboard/STRATEGY_LEADERBOARD.md`

---

## Stage 2: All-Domains Training

**Status: 🔄 In Progress (97.6% training, 93% testing)**

### Description
- **Training Domain Filter:** None (all domains)
- **Weights Directory:** `/scratch/aaa_exchange/AWARE/WEIGHTS_STAGE_2/`
- **Purpose:** Train models on all weather conditions, evaluate domain-inclusive performance

### Coverage
- **Training:** 281/288 models complete (97.6%)
- **Testing:** 267/288 models tested (93%)
- **Strategies:** 24 (21 generative + 3 standard)
- **Datasets:** 4 (BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k)
- **Models:** 3 (DeepLabV3+, PSPNet, SegFormer)

### Strategies Included
| Category | Strategies |
|----------|------------|
| **Generative (21)** | gen_Attribute_Hallucination, gen_augmenters, gen_automold, gen_CNetSeg, gen_CUT, gen_cycleGAN, gen_flux_kontext, gen_Img2Img, gen_IP2P, gen_LANIT, gen_Qwen_Image_Edit, gen_stargan_v2, gen_step1x_new, gen_step1x_v1p2, gen_SUSTechGAN, gen_TSIT, gen_UniControl, gen_VisualCloze, gen_Weather_Effect_Generator, gen_albumentations_weather |
| **Standard (3)** | baseline, photometric_distort, std_autoaugment, std_randaugment |

### Excluded from Stage 2
- `std_cutmix` - Not part of Stage 2 training plan
- `std_mixup` - Not part of Stage 2 training plan
- `gen_cyclediffusion` - Not part of Stage 2 training plan

### Pending Work
| Task | Count | Status |
|------|-------|--------|
| Training Jobs | 7 | 🏃 Running |
| Testing Jobs | 125 | ⏳ Pending |

**Pending Testing (Stage 1):**
- gen_cyclediffusion/mapillaryvistas/deeplabv3plus_r50 (Job 9660496)
- gen_cyclediffusion/mapillaryvistas/pspnet_r50 (Job 9660497)

**Training Jobs Running:**
- baseline/BDD10k/SegFormer (resume from iter_70000)
- gen_CNetSeg/BDD10k/SegFormer
- gen_CNetSeg/IDD-AW/SegFormer
- gen_flux_kontext/BDD10k/DeepLabV3+
- gen_flux_kontext/BDD10k/PSPNet
- gen_flux_kontext/IDD-AW/DeepLabV3+
- gen_flux_kontext/IDD-AW/PSPNet

### Key Files
- Training Tracker: [TRAINING_TRACKER_STAGE2.md](TRAINING_TRACKER_STAGE2.md)
- Training Coverage: [TRAINING_COVERAGE_STAGE2.md](TRAINING_COVERAGE_STAGE2.md)

---

## Comparison

| Aspect | Stage 1 | Stage 2 |
|--------|---------|---------|
| Training Domain | Clear-day only | All domains |
| Total Strategies | 27 | 24 |
| Expected Models | 405 | 288 |
| Training Complete | 99.5% | 97.6% |
| Testing Complete | 98.0% | 93.0% |
| Purpose | Cross-domain robustness | Domain-inclusive performance |

---

## Scripts

### Training Submission
```bash
# Stage 1
./scripts/submit_training.sh --dataset BDD10k --model deeplabv3plus_r50 --strategy baseline --domain-filter clear_day

# Stage 2
./scripts/submit_training.sh --dataset BDD10k --model deeplabv3plus_r50 --strategy baseline
```

### Testing Submission
```bash
# Auto-submit missing tests (Stage 1)
python scripts/auto_submit_tests.py --dry-run

# Auto-submit missing tests (Stage 2)
python scripts/submit_missing_stage2_tests_auto.py --dry-run
```

### Update Trackers
```bash
# Stage 1
python scripts/update_training_tracker.py --stage 1 --coverage-report

# Stage 2  
python scripts/update_training_tracker.py --stage 2 --coverage-report
```

---

## Next Steps

1. ✅ Stage 1 training and testing nearly complete
2. 🔄 Stage 2 training: 7 jobs running, will complete within 24 hours
3. 🔄 Stage 2 testing: 125 jobs submitted, will complete within 12 hours
4. 📊 After completion: Generate analysis reports and leaderboards

---

## Ablation Studies

### 1. Ratio Ablation Study

**Status: ✅ Complete**

**Purpose:** Investigate the optimal real-to-generated image ratio for training.

**Location:** `/scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION/`

**Ratios Tested:** 0.00, 0.12, 0.25, 0.38, 0.50 (standard), 0.62, 0.75, 0.88

| Strategy | Datasets | Models | Checkpoints |
|----------|----------|--------|-------------|
| gen_TSIT | BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k | PSPNet, SegFormer | 465 |
| gen_step1x_new | BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k | PSPNet, SegFormer | 448 |
| gen_step1x_v1p2 | BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k | PSPNet, SegFormer | 416 |
| gen_cycleGAN | BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k | DeepLabV3+, PSPNet, SegFormer | 336 |
| gen_stargan_v2 | BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k | DeepLabV3+, PSPNet, SegFormer | 118 |
| gen_flux_kontext | BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k | DeepLabV3+, PSPNet, SegFormer | 112 |
| gen_cyclediffusion | BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k | DeepLabV3+, PSPNet, SegFormer | 81 |
| **Total** | **4 datasets** | **2-3 models** | **1,976** |

**Key Finding:** Optimal ratio is typically around 0.50 (50% real, 50% generated).

---

### 2. Extended Training Study

**Status: ✅ Complete**

**Purpose:** Evaluate the effect of training for longer (up to 320k iterations vs standard 80k).

**Location:** `/scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED/`

**Iterations Tested:** 90k, 100k, 110k, ... 320k (in 10k increments)

| Strategy | Datasets | Models | Checkpoints |
|----------|----------|--------|-------------|
| gen_cyclediffusion | BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k | PSPNet, SegFormer | 192 |
| gen_step1x_new | BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k | PSPNet, SegFormer | 120 |
| gen_TSIT | BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k | PSPNet, SegFormer | 96 |
| gen_UniControl | BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k | PSPNet, SegFormer | 96 |
| gen_albumentations_weather | BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k | PSPNet, SegFormer | 96 |
| gen_automold | BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k | PSPNet, SegFormer | 95 |
| gen_cycleGAN | BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k | PSPNet, SegFormer | 96 |
| gen_flux_kontext | BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k | PSPNet, SegFormer | 96 |
| std_randaugment | BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k | PSPNet, SegFormer | 72 |
| **Total** | **4 datasets** | **2 models** | **959** |

**Key Finding:** Extended training provides marginal improvements (~1-2% mIoU) at 2-4x computational cost.

---

### 3. Strategy Combinations Study

**Status: 🔶 Partial (by chge7185)**

**Purpose:** Test combining generative augmentation with standard augmentation methods.

**Location:** `/scratch/aaa_exchange/AWARE/WEIGHTS_COMBINATIONS_chge7185/`

| Combination | Datasets | Models | Checkpoints |
|-------------|----------|--------|-------------|
| gen_step1x_new + photometric_distort | MapillaryVistas | PSPNet, SegFormer | 33 |
| gen_Attribute_Hallucination + photometric_distort | MapillaryVistas | PSPNet, SegFormer | 32 |
| gen_flux_kontext + photometric_distort | MapillaryVistas | PSPNet, SegFormer | 32 |
| gen_Qwen_Image_Edit + photometric_distort | MapillaryVistas | PSPNet, SegFormer | 32 |
| gen_stargan_v2 + photometric_distort | MapillaryVistas | PSPNet, SegFormer | 32 |
| std_autoaugment + photometric_distort | MapillaryVistas | PSPNet, SegFormer | 32 |
| std_cutmix + photometric_distort | MapillaryVistas | PSPNet, SegFormer | 32 |
| std_mixup + photometric_distort | MapillaryVistas | PSPNet, SegFormer | 32 |
| std_randaugment + photometric_distort | MapillaryVistas | PSPNet, SegFormer | 32 |
| gen_step1x_new + std_mixup | MapillaryVistas | PSPNet, SegFormer | 1 |
| gen_step1x_new + std_randaugment | MapillaryVistas | PSPNet, SegFormer | 3 |
| **Total** | **1 dataset** | **2 models** | **293** |

**Note:** Limited to MapillaryVistas dataset only. Some combinations still in progress.

---

### 4. Domain Adaptation Ablation

**Status: ⏳ Planned**

**Purpose:** Evaluate training on specific weather domains vs all domains.

**Location:** `/scratch/aaa_exchange/AWARE/WEIGHTS/domain_adaptation_ablation/`

**Note:** Directory structure created but training not yet started.

