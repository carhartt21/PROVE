# PROVE Project TODO List

**Last Updated:** 2026-01-21 (14:10)

## Current Job Status Summary

### Stage 1 (Clear Day Domain) - WEIGHTS directory
| Category | Running | Pending | Complete | Total |
|----------|--------:|--------:|---------:|------:|
| Training | 0 | 0 | 107 | 107 |
| Testing | 81 | ~70 | 0 | 81 |

✅ **Stage 1 training 100% complete**
🔄 **Stage 1 MapillaryVistas retest running (BGR→RGB fix)**

### Stage 2 (All Domains) - WEIGHTS_STAGE_2 directory
| Category | Running | Pending | Complete | Total |
|----------|--------:|--------:|---------:|------:|
| Training | 1 | 0 | 324 | 325 |
| Testing | 81 | ~70 | 0 | 81 |

**Stage 2 Status (as of 2026-01-21 14:10):**
- **Training:** 324/325 complete (99.7%) - 1 std_cutmix job still running (OUTSIDE15k)
- **Testing:** MapillaryVistas retest submitted (162 jobs total, 81 per stage)
- **Key Finding:** BGR→RGB bug fix in fine_grained_test.py - all MapillaryVistas results invalid

---

## � Critical Bug Fix: BGR→RGB in MapillaryVistas Labels (Jan 21)

**Issue:** MapillaryVistas label decoding used BGR channel order (cv2.imread default) instead of RGB.

**Impact:** All MapillaryVistas test results were INVALID - colors decoded incorrectly.

**Fix Applied:** Commit 9313a5e - Changed `r = gt_seg_map[:, :, 0]` to `r = gt_seg_map[:, :, 2]` in fine_grained_test.py.

**Retest Status:** 
- ✅ Submitted 162 retest jobs (81 Stage 1 + 81 Stage 2)
- 🔄 11 running, ~152 pending
- Expected completion: ~4-5 hours per job

---

## 🔄 Active Jobs (Jan 21, 2026 - 14:10)

### Stage 2 Training (1 job) - Running
Resume training for incomplete std_cutmix model:
| Strategy | Dataset | Model | Progress | Job ID |
|----------|---------|-------|----------|--------|
| std_cutmix | OUTSIDE15k | deeplabv3plus_r50 | 40000→80000 | 9675473 |

### MapillaryVistas Retest (162 jobs) - Running
All MapillaryVistas tests resubmitted after BGR→RGB fix:
| Stage | Running | Pending | Job ID Range |
|-------|---------|---------|--------------|
| Stage 1 | ~6 | ~75 | 9681356-9681666 |
| Stage 2 | ~5 | ~76 | 9681687-9681938 |

---

## 🔍 Critical Finding: std_cutmix Artifact (Jan 21)

**Issue:** std_cutmix appeared #1 in Stage 2 leaderboard with +1.45 gain over baseline.

**Investigation:** Compared std_cutmix vs baseline per-configuration and found:
- std_cutmix only has **10/12 configurations** tested
- Missing configs are **lower-performing** ones:
  - `bdd10k/pspnet_r50` (baseline: 44.17 mIoU)
  - `outside15k/deeplabv3plus_r50` (baseline: 30.18 mIoU)

**Calculation:**
- std_cutmix current avg (10 configs): 45.94 mIoU
- baseline avg (12 configs): 44.48 mIoU
- **Estimated std_cutmix avg with all 12 configs: ~44.48 mIoU = 0.00 gain**

**Root Cause:** Training for 2 std_cutmix configs was incomplete:
- `bdd10k/pspnet_r50`: stopped at iter_50000
- `outside15k/deeplabv3plus_r50`: stopped at iter_40000

**Fix:** Submitted resume training jobs (9675468, 9675473). After completion:
1. Submit tests for the 2 completed models
2. Regenerate Stage 2 leaderboard
3. Verify std_cutmix equals baseline (~0.00 gain)

---

## Stage 2 Leaderboard Analysis (Jan 21, 2026)

### Key Finding: No Strategy Beats Baseline in Stage 2

After investigating the Stage 2 leaderboard, we found:
1. **std_cutmix's +1.45 lead is an artifact** of missing low-performing configs
2. **gen_step1x_new** genuinely underperforms (#25, -0.49 vs baseline)
3. **gen_Weather_Effect_Generator** genuinely underperforms (#24, -0.34 vs baseline)

### Anomalous Model Counts Investigation

Expected: 12 models per strategy (4 datasets × 3 models)

#### Strategies with Incomplete Training
| Strategy | Dataset | Model | Status | Progress |
|----------|---------|-------|--------|----------|
| std_cutmix | BDD10k | pspnet_r50 | 🔄 Resuming | 50000→80000 |
| std_cutmix | OUTSIDE15k | deeplabv3plus_r50 | 🔄 Resuming | 40000→80000 |

#### Extra Entry (Harmless)
| Strategy | Count | Issue |
|----------|-------|-------|
| gen_IP2P | 13 | Backup folder with iter_80000.pth |

### Stage 1 vs Stage 2 Performance Comparison

| Strategy | Stage 1 Rank | Stage 2 Rank | Change | Notes |
|----------|-------------|--------------|--------|-------|
| gen_step1x_new | #5 (+1.28) | #25 (-0.49) | ↓20 | Artifacts hurt small objects |
| gen_Qwen_Image_Edit | #1 (+1.97) | TBD | TBD | Still testing |
| std_cutmix | TBD | #1* (+1.45) | - | *Artifact - actually ~0.00 |

See [STAGE_COMPARISON_ANALYSIS.md](docs/STAGE_COMPARISON_ANALYSIS.md) for full analysis.

---

## Stage 1 Leaderboard (Top 15)

| Rank | Strategy | mIoU | Gain vs Baseline |
|------|----------|------|------------------|
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
| ... | baseline | 41.64% | - |

**Note:** gen_cyclediffusion (#7) is missing from Stage 2 - training running.

---

## Stage 2 Coverage Analysis

### All Strategies (27 total)
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

---

## Pending Tasks

### High Priority
1. **Monitor submitted jobs**
   - 36 Stage 2 training jobs (gen_cyclediffusion, std_cutmix, std_mixup)
   - 26 Stage 1 test jobs
   - 7 Stage 2 test jobs
   
2. **After Stage 2 training completes:**
   - Run \`python scripts/auto_submit_tests_stage2.py\` to submit tests
   - Update training tracker: \`python scripts/update_training_tracker.py --stage 2\`

3. **After all tests complete:**
   - Regenerate leaderboards
   - Update downstream_results.csv

### Medium Priority
4. **Extended Training Testing**
   - 504 test jobs submitted for extended training analysis
   - Analyze results with \`analyze_extended_training.py\`

5. **Domain Adaptation Ablation**
   - 84 configurations ready
   - Script: \`./scripts/submit_domain_adaptation_ablation.sh --all-strategies\`

### Low Priority
6. **Publication preparation**
   - Finalize figures for IEEE paper
   - Run statistical significance tests

---

## Key Scripts

### Job Submission
\`\`\`bash
# Stage 1 test submission
python scripts/auto_submit_tests.py --dry-run
python scripts/auto_submit_tests.py

# Stage 2 test submission  
python scripts/auto_submit_tests_stage2.py --dry-run
python scripts/auto_submit_tests_stage2.py

# Stage 2 pending training (one-time)
./scripts/submit_stage2_pending.sh --dry-run
./scripts/submit_stage2_pending.sh
\`\`\`

### Monitoring
\`\`\`bash
# Check all jobs
bjobs -u mima2416 -w

# Check specific types
bjobs -w | grep fg_    # Stage 1 tests
bjobs -w | grep fg2_   # Stage 2 tests
bjobs -w | grep tr_    # Training jobs

# Update trackers
python scripts/update_training_tracker.py --stage 1
python scripts/update_training_tracker.py --stage 2
python scripts/update_testing_tracker.py
\`\`\`

### Analysis
\`\`\`bash
# Regenerate leaderboards
python analysis_scripts/generate_stage1_leaderboard.py
python analysis_scripts/generate_stage2_leaderboard.py

# Analyze test results
python test_result_analyzer.py --root /scratch/aaa_exchange/AWARE/WEIGHTS --comprehensive
\`\`\`

---

## Recently Completed (Jan 20, 2026)

### Stage 2 Gap Analysis
- ✅ Analyzed top 15 strategies coverage in Stage 2
- ✅ Identified gen_cyclediffusion, std_cutmix, std_mixup as missing
- ✅ Submitted 36 training jobs for missing strategies

### New Scripts Created
- ✅ \`scripts/auto_submit_tests_stage2.py\` - Auto-submit Stage 2 test jobs
- ✅ \`scripts/submit_stage2_pending.sh\` - Submit pending Stage 2 training

### Bug Fixes
- ✅ Fixed \`conda activate\` → \`mamba activate\` in auto_submit_tests.py
- ✅ Fixed batch-size parameter (was 1, now 8-10)

---

## Directory Structure

\`\`\`
WEIGHTS/                     # Stage 1 (clear_day training)
├── baseline/
│   ├── bdd10k/
│   ├── idd-aw/
│   ├── mapillaryvistas/
│   └── outside15k/
├── gen_*/                   # Generative strategies
└── std_*/                   # Standard augmentation

WEIGHTS_STAGE_2/             # Stage 2 (all domains training)
├── baseline/
├── gen_*/
└── std_*/

WEIGHTS_RATIO_ABLATION/      # Ratio ablation study
WEIGHTS_EXTENDED/            # Extended training study
\`\`\`
