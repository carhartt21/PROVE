# Evaluation Stage Status

**Last Updated:** 2026-02-01 (09:30)

---

## ⚠️ CRITICAL WARNING: gen_* Results Invalid

> **MixedDataLoader Bug (Jan 28, 2026):** Generated images were **NEVER LOADED** during training.
> All `gen_*` strategy results below are **INVALID** - only pipeline augmentation was used.
> 
> **Bug Status:** ✅ FIXED | **Retraining:** ⏳ Required
> 
> See [BUG_REPORT](BUG_REPORT_CROSS_DATASET_CONTAMINATION.md) for details.

---

## ✅ NEW: Cityscapes Pipeline Verification Stage

For pipeline verification, a new `cityscapes` stage is available:

```bash
python scripts/batch_training_submission.py --stage cityscapes --dry-run
```

**Details:**
- **Output:** `WEIGHTS_CITYSCAPES/baseline/cityscapes/{model}/`
- **Iterations:** 160,000
- **Purpose:** Verify pipeline correctness using standard Cityscapes benchmark

---

## Overview

| Stage | Training | Testing | Status |
|-------|----------|---------|--------|
| **Stage 1** | ✅ 324/324 (100%) | ✅ 330/330 (100%) | ⚠️ gen_* results **INVALID** |
| **Stage 2** | ✅ 325/325 (100%) | ✅ 344/344 (100%) | ⚠️ gen_* results **INVALID** |

**📊 Comprehensive Ablation Analysis Report:** [ABLATION_STUDIES_ANALYSIS.md](ABLATION_STUDIES_ANALYSIS.md)

⚠️ **Training/Testing complete, but gen_* results require retraining with fixed MixedDataLoader!**


## Stage 1: Clear-Day Domain Training

**Status: ✅ COMPLETE (Training + Testing)**

### Description
- **Training Domain Filter:** `clear_day` only
- **Weights Directory:** `${AWARE_DATA_ROOT}/WEIGHTS/`
- **Purpose:** Train models on clear weather conditions, evaluate cross-domain robustness

### Coverage
| Metric | Count | Percentage |
|--------|-------|------------|
| Training Complete | 405/405 | ✅ 100% |
| Testing Complete | 405/405 | ✅ 100% |
| MapillaryVistas Training | 81/81 | ✅ 100% |
| MapillaryVistas Testing | 81/81 | ✅ 100% |

**Stage 1 is now fully complete including MapillaryVistas!**

### Strategies (27)
| Category | Count | Strategies |
|----------|-------|------------|
| Generative | 21 | gen_Attribute_Hallucination, gen_augmenters, gen_automold, gen_CNetSeg, gen_CUT, gen_cyclediffusion, gen_cycleGAN, gen_flux_kontext, gen_Img2Img, gen_IP2P, gen_LANIT, gen_Qwen_Image_Edit, gen_stargan_v2, gen_step1x_new, gen_step1x_v1p2, gen_SUSTechGAN, gen_TSIT, gen_UniControl, gen_VisualCloze, gen_Weather_Effect_Generator, gen_albumentations_weather |
| Standard | 6 | baseline, std_photometric_distort, std_autoaugment, std_cutmix, std_mixup, std_randaugment |

### Leaderboard (Top 15) - Updated 2026-01-23 with MapillaryVistas
| Rank | Strategy | mIoU | Gain |
|------|----------|------|------|
| 1 | gen_Attribute_Hallucination | 39.83% | +1.36 |
| 2 | gen_cycleGAN | 39.60% | +1.13 |
| 3 | gen_Img2Img | 39.58% | +1.11 |
| 4 | gen_stargan_v2 | 39.55% | +1.08 |
| 5 | gen_flux_kontext | 39.54% | +1.07 |
| 6 | gen_cyclediffusion | 39.52% | +1.05 |
| 7 | gen_CNetSeg | 39.47% | +1.00 |
| 8 | gen_IP2P | 39.47% | +1.00 |
| 9 | gen_augmenters | 39.46% | +0.99 |
| 10 | gen_Weather_Effect_Generator | 39.43% | +0.96 |
| 11 | gen_SUSTechGAN | 39.43% | +0.96 |
| 12 | gen_automold | 39.43% | +0.96 |
| 13 | gen_step1x_new | 39.41% | +0.94 |
| 14 | std_autoaugment | 39.41% | +0.94 |
| 15 | gen_VisualCloze | 39.40% | +0.93 |
| - | baseline | 38.47% | - |

**Key Insight:** All 26 strategies beat baseline in Stage 1!

### Key Files
- Training Tracker: [TRAINING_TRACKER_STAGE1.md](TRAINING_TRACKER_STAGE1.md)
- Leaderboard: \`result_figures/leaderboard/STRATEGY_LEADERBOARD.md\`

---

## Stage 2: All-Domains Training

**Status: ✅ Non-MV Complete | 🔄 MapillaryVistas Retraining (59%)**

### Description
- **Training Domain Filter:** None (all domains)
- **Weights Directory:** `${AWARE_DATA_ROOT}/WEIGHTS_STAGE_2/`
- **Purpose:** Train models on all weather conditions, evaluate domain-inclusive performance

### Coverage
| Metric | Count | Percentage |
|--------|-------|------------|
| Training Complete (non-MV) | 243/243 | ✅ 100% |
| Testing Complete (non-MV) | 243/243 | ✅ 100% |
| MapillaryVistas Training | 48/81 | 🔄 59% |
| MapillaryVistas Testing | 0/48 | ⏳ Waiting |

**Note:** MapillaryVistas tests will run as training completes. Script ready: `./scripts/run_stage2_mapillary_tests.sh`
Afterwards, update leaderboard.


### Leaderboard (Top 10)
| Rank | Strategy | mIoU | Gain |
|------|----------|------|------|
| 1 | gen_CNetSeg | 43.68% | +0.58 |
| 2 | gen_stargan_v2 | 43.60% | +0.50 |
| 3 | gen_UniControl | 43.59% | +0.49 |
| 4 | gen_cyclediffusion | 43.56% | +0.47 |
| 5 | std_autoaugment | 43.55% | +0.46 |
| 6 | gen_augmenters | 43.54% | +0.44 |
| 7 | std_randaugment | 43.53% | +0.43 |
| 8 | gen_cycleGAN | 43.52% | +0.42 |
| 9 | gen_CUT | 43.51% | +0.42 |
| 10 | gen_VisualCloze | 43.48% | +0.38 |
| - | baseline | 43.10% | - |
| 27 | std_cutmix | 42.80% | -0.29 |

### Strategies Coverage (All 27)
| Strategy | Training | Testing | Notes |
|----------|:--------:|:-------:|-------|
| baseline | ✅ 12/12 | ✅ 12/12 | |
| gen_Attribute_Hallucination | ✅ 12/12 | ✅ 12/12 | |
| gen_CNetSeg | ✅ 12/12 | ✅ 12/12 | |
| gen_CUT | ✅ 12/12 | ✅ 12/12 | |
| gen_cycleGAN | ✅ 12/12 | ✅ 12/12 | |
| gen_cyclediffusion | ✅ 12/12 | 🔄 Testing | MapillaryVistas test pending |
| gen_flux_kontext | ✅ 12/12 | ✅ 12/12 | |
| gen_Img2Img | ✅ 12/12 | ✅ 12/12 | |
| gen_IP2P | ✅ 13/12 | ✅ 12/12 | +1 backup folder |
| gen_LANIT | ✅ 12/12 | ✅ 12/12 | |
| gen_Qwen_Image_Edit | ✅ 12/12 | ✅ 12/12 | |
| gen_stargan_v2 | ✅ 12/12 | ✅ 12/12 | |
| gen_step1x_new | ✅ 12/12 | 🔄 Testing | MapillaryVistas test running |
| gen_step1x_v1p2 | ✅ 12/12 | ✅ 12/12 | |
| gen_SUSTechGAN | ✅ 12/12 | ✅ 12/12 | |
| gen_TSIT | ✅ 12/12 | ✅ 12/12 | |
| gen_UniControl | ✅ 12/12 | ✅ 12/12 | |
| gen_VisualCloze | ✅ 12/12 | ✅ 12/12 | |
| gen_Weather_Effect_Generator | ✅ 12/12 | 🔄 Testing | MapillaryVistas test running |
| gen_albumentations_weather | ✅ 12/12 | ✅ 12/12 | |
| gen_augmenters | ✅ 12/12 | ✅ 12/12 | |
| gen_automold | ✅ 12/12 | ✅ 12/12 | |
| std_photometric_distort | ✅ 12/12 | ✅ 12/12 | |
| std_autoaugment | ✅ 12/12 | ✅ 12/12 | |
| **std_cutmix** | 🔄 10/12 | ⏳ 10/12 | **2 jobs resuming** |
| std_mixup | ✅ 12/12 | 🔄 Testing | MapillaryVistas test running |
| std_randaugment | ✅ 12/12 | ✅ 12/12 | |

### Key Files
- Training Tracker: [TRAINING_TRACKER_STAGE2.md](TRAINING_TRACKER_STAGE2.md)

---

## Comparison

| Aspect | Stage 1 | Stage 2 |
|--------|---------|---------|
| Training Domain | Clear-day only | All domains |
| Total Strategies | 27 | 27 |
| Training Complete | ✅ 405/405 (100%) | 🔄 291/324 (90%) |
| Testing Complete | ✅ 405/405 (100%) | ✅ 243/243 (100% non-MV) |
| MapillaryVistas Status | ✅ Complete | 🔄 59% Training |
| Baseline mIoU | 41.64% | 43.10% |
| Best Strategy | TBD (need to regenerate with MV) | gen_CNetSeg (43.68%) |

**Note:** Stage 1 leaderboard should be regenerated now that MapillaryVistas results are available.

---

## 📊 Stage 1 Baseline Analysis

**Publication-ready analysis generated:** 2026-01-23

| Output | Description |
|--------|-------------|
| **Location** | `result_figures/baseline_consolidated/stage1_baseline_output/` |
| **Script** | `result_figures/baseline_consolidated/generate_stage1_baseline.py` |

### Tables
| Table | Content |
|-------|---------|
| Table 1 | Overall Baseline Performance (12 configs + average) |
| Table 2 | Model Architecture Robustness |
| Table 3 | Per-Domain Degradation |
| Table 4 | Dataset Challenge Levels |

### Figures
| Figure | Content |
|--------|---------|
| Figure 1 | Cross-Domain Robustness (grouped bar chart) |
| Figure 2 | Dataset × Domain Performance (heatmap) |
| Figure 3 | Domain Gap by Dataset (horizontal bars) |
| Figure 4 | Performance Distribution (box plots) |

### Key Findings
| Metric | Value | Notes |
|--------|-------|-------|
| Overall mIoU | 33.3% | Average across 12 configs |
| Domain Gap | 10.1% | Clear Day - Adverse Avg |
| Most Robust Model | SegFormer | Gap 8.7% |
| Hardest Domain | Night | -14.9% from Clear Day |
| Largest Dataset Gap | IDD-AW | 17.6% domain gap |
| Smallest Dataset Gap | Mapillary | 2.6% domain gap |

---

## Scripts

### Auto-Submit Tests
\`\`\`bash
# Stage 1
python scripts/auto_submit_tests.py --dry-run
python scripts/auto_submit_tests.py

# Stage 2
python scripts/auto_submit_tests_stage2.py --dry-run
python scripts/auto_submit_tests_stage2.py
\`\`\`

### Training Submission
\`\`\`bash
# Stage 1
./scripts/submit_training.sh --dataset BDD10k --model deeplabv3plus_r50 --strategy baseline --domain-filter clear_day

# Stage 2 (no domain filter)
./scripts/submit_training.sh --dataset BDD10k --model deeplabv3plus_r50 --strategy baseline

# Stage 2 pending strategies (one-time)
./scripts/submit_stage2_pending.sh
\`\`\`

### Update Trackers
\`\`\`bash
python scripts/update_training_tracker.py --stage 1
python scripts/update_training_tracker.py --stage 2
python scripts/update_testing_tracker.py              # Stage 1 (default)
python scripts/update_testing_tracker.py --stage 2    # Stage 2
\`\`\`

### Generate Leaderboards
\`\`\`bash
python analysis_scripts/generate_stage1_leaderboard.py
python analysis_scripts/generate_stage2_leaderboard.py
\`\`\`

---

## Ablation Studies

**📊 Full Analysis:** [ABLATION_STUDIES_ANALYSIS.md](ABLATION_STUDIES_ANALYSIS.md)

### 1. Domain Adaptation Study
**Status:** 🔄 Running (64 tests complete)

- **Location:** Uses Stage 1 checkpoints (no additional training)
- **Tests:** 64/~100 complete
- **Key Finding:** BDD10k→ACDC best (23.7%), gen_TSIT leads (+3.9% vs baseline)
- **CSV:** `result_figures/domain_adaptation_analysis.csv`

### 2. Ratio Ablation Study
**Status:** � ACTIVE TRAINING

- **Location:** `${AWARE_DATA_ROOT}/WEIGHTS_RATIO_ABLATION/`
- **Directory Structure:** Reorganized to `stage1/` and `stage2/` subdirectories
- **Ratios:** 0.00, 0.12, 0.25, 0.38, 0.62, 0.75, 0.88
- **Models:** pspnet_r50, segformer_mit-b5
- **Datasets:** BDD10k, IDD-AW

| Stage | Strategy Type | Trained | In Queue | Owner |
|-------|--------------|---------|----------|-------|
| Stage 1 | Existing | 32 | 52 (10 RUN, 42 PEND) | ${USER} |
| Stage 1 | Top-5 New | ? | 117 | chge7185 |
| Stage 2 | Existing | 56 | 0 | Complete |
| Stage 2 | Top-5 New | 0 | Pending | - |

**Training Locks:** ✅ Enabled (prevents duplicate training)
**Archived Logs:** `_archived_logs_20260126/` - 117 dirs from buggy runs (validation logs only, no checkpoints, NOT USABLE)

**Key Finding (preliminary):** Higher ratios (0.62-0.88) slightly outperform lower ratios
- **CSV:** `result_figures/ratio_ablation_analysis.csv`

### 3. Extended Training Study
**Status:** ✅ Complete

- **Location:** `${AWARE_DATA_ROOT}/WEIGHTS_EXTENDED/`
- **Iterations:** 40k to 160k (20k increments) + 320k
- **Checkpoints:** 959
- **Key Finding:** 160k iterations = 75% of gains at 50% compute cost
- **CSV:** `result_figures/extended_training_analysis.csv`

### 4. Strategy Combinations Study
**Status:** ✅ Complete (ALL TESTED)

- **Location:** `${AWARE_DATA_ROOT}/WEIGHTS_COMBINATIONS/`
- **Checkpoints:** 53 (all IDD-AW, all tested)
- **Key Finding:** std_photometric_distort combos dominate (45.1% avg vs ~40% others)
- **CSV:** `result_figures/combinations_analysis.csv`

---

## Next Steps

1. **Monitor Ratio Ablation Training** (~12-24 hours)
   - ${USER}: 52 jobs (Stage 1 existing strategies)
   - chge7185: 117 jobs (Stage 1 top-5 new strategies)
   ```bash
   bjobs -u ${USER} | grep -c "RUN\|PEND"
   bjobs -u chge7185 | wc -l
   ```

2. **After Training Completes**
   - Run tests on completed models
   - Generate ratio ablation analysis
   ```bash
   python analysis_scripts/analyze_ratio_ablation.py --verbose
   python analysis_scripts/visualize_ratio_ablation.py
   ```

3. **Submit Stage 2 Ratio Ablation** (when queue opens)
   ```bash
   python scripts/submit_ratio_ablation_training.py --stage 2 --dry-run
   ```

4. **Publication Preparation**
   - Finalize ablation study figures
   - Run statistical significance tests
   - Document in paper
