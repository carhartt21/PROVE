# Evaluation Stage Status

**Last Updated:** 2026-01-21 (12:35)

## Overview

| Stage | Training | Testing | Status |
|-------|----------|---------|--------|
| **Stage 1** | 107/107 (100%) | 346/346 (100%) | ✅ Complete |
| **Stage 2** | 323/325 (99.4%) | 333/~340 (98%) | 🔄 Training (2 resume), 🔄 Testing (6 running) |

---

## Stage 1: Clear-Day Domain Training

**Status: ✅ Training Complete | 🔄 Testing In Progress**

### Description
- **Training Domain Filter:** \`clear_day\` only
- **Weights Directory:** \`/scratch/aaa_exchange/AWARE/WEIGHTS/\`
- **Purpose:** Train models on clear weather conditions, evaluate cross-domain robustness

### Coverage
| Metric | Count | Percentage |
|--------|-------|------------|
| Training Complete | 107/107 | 100% |
| Testing Complete | 346/346 | 100% |
| Testing Pending | 0 | - |

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

**Status: 🔄 Training Resume (2 jobs) | 🔄 Testing In Progress (6 jobs)**

### Description
- **Training Domain Filter:** None (all domains)
- **Weights Directory:** `/scratch/aaa_exchange/AWARE/WEIGHTS_STAGE_2/`
- **Purpose:** Train models on all weather conditions, evaluate domain-inclusive performance

### Coverage
| Metric | Count | Percentage |
|--------|-------|------------|
| Training Complete | 323/325 | 99.4% |
| Training Running | 2 | std_cutmix resume |
| Testing Complete | 333/~340 | 98% |
| Testing Running | 6 | MapillaryVistas tests |

### 🔍 Critical Finding: std_cutmix Artifact

**Issue:** std_cutmix appeared #1 in Stage 2 leaderboard with +1.45 gain over baseline.

**Investigation Findings:**
- std_cutmix only has **10/12 configurations** tested
- Missing configs are **lower-performing** ones:
  - `bdd10k/pspnet_r50` (baseline: 44.17 mIoU)
  - `outside15k/deeplabv3plus_r50` (baseline: 30.18 mIoU)

**Calculation:**
| Metric | Value |
|--------|-------|
| std_cutmix avg (10 configs) | 45.94 mIoU |
| baseline avg (12 configs) | 44.48 mIoU |
| **Estimated std_cutmix avg (12 configs)** | **~44.48 mIoU = 0.00 gain** |

**Root Cause:** Training for 2 std_cutmix configs was incomplete - stopped early.

**Fix Status:** Resume training submitted (jobs 9675468, 9675473)

### Active Training Jobs (2) - Resuming
| Strategy | Dataset | Model | Progress | Job ID |
|----------|---------|-------|----------|--------|
| std_cutmix | BDD10k | pspnet_r50 | 50000→80000 | 9675468 |
| std_cutmix | OUTSIDE15k | deeplabv3plus_r50 | 40000→80000 | 9675473 |

### Active Test Jobs (6) - Running
| Strategy | Dataset | Model | Job ID |
|----------|---------|-------|--------|
| gen_cycleGAN | MapillaryVistas | pspnet_r50 | 9672242 |
| gen_stargan_v2 | MapillaryVistas | deeplabv3plus_r50 | 9672788 |
| gen_Weather_Effect_Generator | MapillaryVistas | pspnet_r50 | 9672789 |
| gen_step1x_new | MapillaryVistas | pspnet_r50 | 9672790 |
| std_mixup | MapillaryVistas | deeplabv3plus_r50 | 9673188 |
| std_mixup | MapillaryVistas | pspnet_r50 | 9673189 |

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
| Training Complete | 107/107 (100%) | 96/132 (73%) |
| Testing Complete | 346/346 (100%) | 289/292 (99%) |
| Baseline mIoU | 41.64% | 44.48% |
| Best Strategy | gen_Qwen_Image_Edit (43.61%) | gen_UniControl (45.00%) |

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

1. **Monitor Active Jobs**
   - 36 Stage 2 training jobs (~8-12 hours each)
   - 26 Stage 1 test jobs (~30 min each)
   - 7 Stage 2 test jobs (~30 min each)

2. **After Training Completes**
   - Run \`python scripts/auto_submit_tests_stage2.py\` for new tests
   - Update trackers

3. **After Testing Completes**
   - Regenerate \`downstream_results.csv\`
   - Update Stage 2 leaderboard
   - Finalize publication figures
