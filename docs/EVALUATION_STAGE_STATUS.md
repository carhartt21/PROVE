# Evaluation Stage Status

**Last Updated:** 2026-01-23 (15:30)

## Overview

| Stage | Training | Testing | Status |
|-------|----------|---------|--------|
| **Stage 1** | ✅ 405/405 (100%) | ✅ 405/405 (100%) | ✅ **COMPLETE** |
| **Stage 2** | 🔄 291/324 (90%) | ✅ 243/243 (100% non-MV) | 🔄 MV Retraining |

## 🔧 Critical Bug Fix: BGR→RGB in MapillaryVistas Labels

**Issue Discovered:** MapillaryVistas RGB label decoding used BGR channel order (cv2.imread default).

**Impact:** ALL MapillaryVistas models were INVALID (training and testing).

**Fixes Applied:**
1. **Training Bug (d7b2b99):** Fixed `custom_transforms.py` - Changed BGR→RGB channel order in `MapillaryRGBToClassId`
2. **Testing Bug (9313a5e):** Fixed `fine_grained_test.py` - Changed BGR→RGB in test-time label loading

**Retraining Status:**
| Stage | Complete | Running | Pending | Total |
|-------|----------|---------|---------|-------|
| Stage 1 | ✅ 81/81 | 0 | 0 | 81 |
| Stage 2 | 48/81 | 6 | ~27 | 81 |

**Testing Status:**
| Stage | Complete | Pending | Total |
|-------|----------|---------|-------|
| Stage 1 | ✅ 81/81 | 0 | 81 |
| Stage 2 | 0/48 | 48 | 48 |

## 📊 Performance Issue: MapillaryVistas 16x Slowdown

**Observation:** MapillaryVistas tests take ~3.15s/image vs BDD10k ~0.2s/image.

### Investigation Results (Jan 21)

| Component | Time/Image | Status |
|-----------|------------|--------|
| Model Inference | ~30ms | ✅ Identical for both datasets |
| Per-class IoU | ~20ms | ✅ Negligible overhead |
| RGB Label Decode | ~5ms | ✅ Negligible overhead |
| **Unknown Overhead** | ~3000ms | ❓ Not identified |

**Key Finding:** Model inference is NOT the bottleneck. Both 19-class and 66-class models run at identical speed (~30ms/image). The ~3000ms mystery overhead is elsewhere in the test pipeline.

**Practical Impact:** Each MapillaryVistas test takes ~4-5 hours (4949 images × 3.15s).

---

## Stage 1: Clear-Day Domain Training

**Status: ✅ COMPLETE (Training + Testing)**

### Description
- **Training Domain Filter:** `clear_day` only
- **Weights Directory:** `/scratch/aaa_exchange/AWARE/WEIGHTS/`
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
| Standard | 6 | baseline, photometric_distort, std_autoaugment, std_cutmix, std_mixup, std_randaugment |

### Leaderboard (Top 15)
| Rank | Strategy | mIoU | Gain |
|------|----------|------|------|
| 1 | gen_Qwen_Image_Edit | 43.61% | +1.97 |
| 2 | gen_Attribute_Hallucination | 43.17% | +1.53 |
| 3 | gen_cycleGAN | 42.99% | +1.35 |
| 4 | gen_flux_kontext | 42.92% | +1.28 |
| 5 | gen_step1x_new | 42.92% | +1.28 |
| 6 | gen_stargan_v2 | 42.89% | +1.25 |
| 7 | gen_cyclediffusion | 42.88% | +1.24 |
| 8 | gen_automold | 42.84% | +1.20 |
| 9 | gen_CNetSeg | 42.78% | +1.14 |
| 10 | gen_albumentations_weather | 42.77% | +1.12 |
| 11 | gen_Weather_Effect_Generator | 42.73% | +1.09 |
| 12 | gen_IP2P | 42.72% | +1.08 |
| 13 | gen_SUSTechGAN | 42.70% | +1.06 |
| 14 | std_autoaugment | 42.67% | +1.03 |
| 15 | gen_CUT | 42.66% | +1.02 |
| - | baseline | 41.64% | - |

### Key Files
- Training Tracker: [TRAINING_TRACKER_STAGE1.md](TRAINING_TRACKER_STAGE1.md)
- Leaderboard: \`result_figures/leaderboard/STRATEGY_LEADERBOARD.md\`

---

## Stage 2: All-Domains Training

**Status: ✅ Non-MV Complete | 🔄 MapillaryVistas Retraining (59%)**

### Description
- **Training Domain Filter:** None (all domains)
- **Weights Directory:** `/scratch/aaa_exchange/AWARE/WEIGHTS_STAGE_2/`
- **Purpose:** Train models on all weather conditions, evaluate domain-inclusive performance

### Coverage
| Metric | Count | Percentage |
|--------|-------|------------|
| Training Complete (non-MV) | 243/243 | ✅ 100% |
| Testing Complete (non-MV) | 243/243 | ✅ 100% |
| MapillaryVistas Training | 48/81 | 🔄 59% |
| MapillaryVistas Testing | 0/48 | ⏳ Waiting |

**Note:** MapillaryVistas tests will run as training completes. Script ready: `./scripts/run_stage2_mapillary_tests.sh`

### ✅ std_cutmix Artifact RESOLVED

**Issue:** std_cutmix previously appeared #1 with +1.45 gain due to incomplete testing.

**Resolution:** Both missing tests completed (2026-01-22):
- `bdd10k/pspnet_r50`: mIoU = 41.31%
- `outside15k/deeplabv3plus_r50`: mIoU = 31.48%

**Result:** std_cutmix now ranks **#27 (last)** at -0.29 below baseline.

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
| photometric_distort | ✅ 12/12 | ✅ 12/12 | |
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

### 1. Ratio Ablation Study
**Status:** ✅ Complete

- **Location:** \`/scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION/\`
- **Ratios:** 0.00, 0.12, 0.25, 0.38, 0.50, 0.62, 0.75, 0.88
- **Checkpoints:** 1,976
- **Finding:** Optimal ratio ~0.50

### 2. Extended Training Study
**Status:** ✅ Complete

- **Location:** \`/scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED/\`
- **Iterations:** 40k to 160k (in 20k increments) + 320k
- **Checkpoints:** 959
- **Finding:** Extended training provides marginal improvements (~1-2% mIoU)

### 3. Strategy Combinations Study
**Status:** 🔶 Partial (by chge7185)

- **Location:** \`/scratch/aaa_exchange/AWARE/WEIGHTS_COMBINATIONS_chge7185/\`
- **Checkpoints:** 293

### 4. Domain Adaptation Ablation
**Status:** ⏳ Ready to start

- **Location:** \`/scratch/aaa_exchange/AWARE/WEIGHTS/domain_adaptation_ablation/\`
- **Configs Ready:** 84
- **Script:** \`./scripts/submit_domain_adaptation_ablation.sh --all-strategies\`

---

## Next Steps

1. **Monitor MapillaryVistas Retest Jobs**
   - 162 test jobs (81 Stage 1 + 81 Stage 2)
   - ~4-5 hours per job
   - Job IDs: 9681356-9681938

2. **Monitor std_cutmix Training Resume**
   - Job 9675473: OUTSIDE15k/deeplabv3plus_r50 (40000→80000)
   - After completion, submit test job

3. **After MapillaryVistas Retests Complete**
   - Regenerate leaderboards with correct MapillaryVistas results
   - Update `downstream_results.csv`
   - Verify leaderboard rankings

4. **Publication Preparation**
   - Finalize figures with corrected MapillaryVistas data
   - Run statistical significance tests
