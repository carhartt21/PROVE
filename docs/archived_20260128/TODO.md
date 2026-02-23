# PROVE Project TODO List

**Last Updated:** 2026-01-20 (21:45)

## Current Job Status Summary

### Stage 1 (Clear Day Domain) - WEIGHTS directory
| Category | Running | Pending | Complete | Total |
|----------|--------:|--------:|---------:|------:|
| Training | 0 | 0 | 107 | 107 |
| Testing | ~26 | 0 | ~320 | 346 |

✅ **Stage 1 training 100% complete**
🔄 **Stage 1 testing 93% complete** (26 tests just submitted)

### Stage 2 (All Domains) - WEIGHTS_STAGE_2 directory
| Category | Running | Pending | Complete | Total |
|----------|--------:|--------:|---------:|------:|
| Training | 0 | 36 | 96 | 132 |
| Testing | 0 | 7 | ~265 | ~272 |

**Stage 2 Status (as of 2026-01-20 21:45):**
- **Training:** 96/132 complete (73%) - 36 jobs just submitted
- **Testing:** ~265 tests complete, 7 tests just submitted
- **Pending Strategies:** gen_cyclediffusion, std_cutmix, std_mixup (all 12 configs each)

---

## 🔄 Active Jobs (Jan 20, 2026)

### Stage 1 Testing (26 jobs)
Just submitted via \`auto_submit_tests.py\`:
- gen_Attribute_Hallucination, gen_CNetSeg, gen_flux_kontext
- gen_Qwen_Image_Edit, gen_stargan_v2, gen_step1x_new
- gen_step1x_v1p2, gen_Weather_Effect_Generator
- std_autoaugment, std_mixup, std_randaugment

**Job IDs:** 9670274-9670299

### Stage 2 Training (36 jobs)
Submitted via \`submit_stage2_pending.sh\`:
| Strategy | Jobs | IDs |
|----------|------|-----|
| gen_cyclediffusion | 12 | 9670343-9670354 |
| std_cutmix | 12 | 9670355-9670366 |
| std_mixup | 12 | 9670367-9670378 |

### Stage 2 Testing (7 jobs)
Submitted via \`auto_submit_tests_stage2.py\`:
| Strategy | Dataset | Model | Job ID |
|----------|---------|-------|--------|
| gen_IP2P | BDD10k | pspnet | 9670325 |
| gen_stargan_v2 | MapillaryVistas | deeplabv3plus | 9670326 |
| gen_step1x_new | MapillaryVistas | pspnet | 9670327 |
| gen_Weather_Effect_Generator | MapillaryVistas | pspnet | 9670328 |
| baseline | MapillaryVistas | deeplabv3plus | 9670329 |
| std_std_photometric_distort | MapillaryVistas | deeplabv3plus | 9670330 |
| std_std_photometric_distort | MapillaryVistas | pspnet | 9670331 |

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

**Note:** gen_cyclediffusion (#7) is missing from Stage 2 - training just submitted.

---

## Stage 2 Coverage Analysis

### Top 15 Strategies Coverage
| Strategy | Training | Testing | Notes |
|----------|:--------:|:-------:|-------|
| gen_Qwen_Image_Edit | ✅ 12/12 | ✅ 12/12 | |
| gen_Attribute_Hallucination | ✅ 12/12 | ✅ 12/12 | |
| gen_cycleGAN | ✅ 12/12 | ✅ 12/12 | |
| gen_flux_kontext | ✅ 12/12 | ✅ 12/12 | |
| gen_step1x_new | ✅ 12/12 | 🔄 11/12 | mapillary/pspnet test submitted |
| gen_stargan_v2 | ✅ 12/12 | 🔄 11/12 | mapillary/deeplabv3plus test submitted |
| **gen_cyclediffusion** | ⏳ 0/12 | ⏳ 0/12 | **Training just submitted** |
| gen_automold | ✅ 12/12 | ✅ 12/12 | |
| gen_CNetSeg | ✅ 12/12 | ✅ 12/12 | |
| gen_albumentations_weather | ✅ 12/12 | ✅ 12/12 | |
| gen_Weather_Effect_Generator | ✅ 12/12 | 🔄 11/12 | mapillary/pspnet test submitted |
| gen_IP2P | ✅ 12/12 | ✅ 12/12 | |
| gen_SUSTechGAN | ✅ 12/12 | ✅ 12/12 | |
| std_autoaugment | ✅ 12/12 | ✅ 12/12 | |
| gen_CUT | ✅ 12/12 | ✅ 12/12 | |

### Standard Augmentation Strategies (Stage 2)
| Strategy | Training | Testing |
|----------|:--------:|:-------:|
| baseline | ✅ 12/12 | 🔄 11/12 |
| std_std_photometric_distort | ✅ 12/12 | 🔄 10/12 |
| std_autoaugment | ✅ 12/12 | ✅ 12/12 |
| std_randaugment | ✅ 12/12 | ✅ 12/12 |
| **std_cutmix** | ⏳ 0/12 | ⏳ 0/12 | **Training just submitted** |
| **std_mixup** | ⏳ 0/12 | ⏳ 0/12 | **Training just submitted** |

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
bjobs -u ${USER} -w

# Check specific types
bjobs -w | grep fg_    # Stage 1 tests
bjobs -w | grep fg2_   # Stage 2 tests
bjobs -w | grep tr_    # Training jobs

# Update trackers
python scripts/update_training_tracker.py --stage 1
python scripts/update_training_tracker.py --stage 2
python scripts/update_testing_tracker.py              # Stage 1 (default)
python scripts/update_testing_tracker.py --stage 2    # Stage 2
\`\`\`

### Analysis
\`\`\`bash
# Regenerate leaderboards
python analysis_scripts/generate_stage1_leaderboard.py
python analysis_scripts/generate_stage2_leaderboard.py

# Analyze test results
python test_result_analyzer.py --root ${AWARE_DATA_ROOT}/WEIGHTS --comprehensive
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
