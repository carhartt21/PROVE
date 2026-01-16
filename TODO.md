# TODO - Upcoming Tasks

*Last updated: 2026-01-16 (15:00)*

## Current Job Status Summary

### Stage 1 (Clear Day Domain) - WEIGHTS directory
| Category | Running | Pending | Done | Total |
|----------|--------:|--------:|-----:|------:|
| Training | 0 | 1 | 110 | 111 |
| Testing | 2 | 5 | 328 | 335 |

### Stage 2 (All Domains - Adverse Weather) - WEIGHTS_STAGE_2 directory
| Category | Running | Pending | Done | Partial | Total |
|----------|--------:|--------:|-----:|--------:|------:|
| Training | 0 | 57 | 6 | 48 | 111 |
| Testing | - | - | - | - | - |

**Note:** Stage 2 uses all 3 models (DeepLabV3+, PSPNet, SegFormer).
Partial indicates configurations where 1/3 or 2/3 models are complete.

### Ablation Studies
| Study | Owner | Running | Pending | Total |
|-------|-------|--------:|--------:|------:|
| Ratio Ablation | mima2416 | 13 | 54 | 112 |
| Extended Training | chge7185 | 6 | 454 | 460 |

---

## Directory Structure Changes (Jan 16, 2026)

### ⚠️ IMPORTANT: WEIGHTS Directory Restructuring

The WEIGHTS directory structure has been reorganized:

**Before:**
```
WEIGHTS/
├── baseline/
│   ├── bdd10k_cd/          # _cd = clear_day (Stage 1)
│   ├── bdd10k_ad/          # _ad = all_domains (Stage 2)
│   └── ...
```

**After:**
```
WEIGHTS/                     # Stage 1 (clear_day only)
├── baseline/
│   ├── bdd10k/             # No suffix
│   ├── idd-aw/
│   └── ...

WEIGHTS_STAGE_2/             # Stage 2 (all domains)
├── baseline/
│   ├── bdd10k/             # No suffix
│   ├── idd-aw/
│   └── ...
```

**Key Changes:**
- `_cd` and `_ad` suffixes removed from dataset directories
- Stage 1 models → `WEIGHTS/`
- Stage 2 models → `WEIGHTS_STAGE_2/`
- Scripts updated: `unified_training_config.py`, `update_training_tracker.py`, `auto_submit_tests.py`, `test_result_analyzer.py`

---

## ⚠️ CRITICAL: Wrong num_classes Models

**Issue Discovered:** Some models were trained with wrong number of classes:

| Dataset | Expected | Strategies Affected |
|---------|----------|---------------------|
| MapillaryVistas | 66 classes | std_minimal, gen_cyclediffusion (trained with 19) |
| OUTSIDE15K | 24 classes | std_minimal, std_cutmix, std_mixup, gen_cyclediffusion, gen_flux_kontext (trained with 19) |

**Total:** 21 models need retraining (7 configs × 3 models each)

**Status:** Pending - need to create retraining jobs with `--use-native-classes` flag

---

## Active Tasks

### Stage 1 Training Completion (99.7%)

| Job ID | Configuration | Status |
|--------|---------------|--------|
| 9602498 | gen_step1x_new / IDD-AW / PSPNet | 🔄 RUNNING |

- ✅ gen_IP2P / IDD-AW / DeepLabV3+ completed (Job 9602408)
- Last training needed to reach 336/336 (100%)

### Stage 1 Testing Coverage (93.4%)

**Status:** 312/334 complete, 22 jobs pending

| Category | Pending Jobs |
|----------|-------------|
| Generative strategies | 14 jobs (gen_CNetSeg, gen_IP2P, gen_Weather_Effect, etc.) |
| Standard strategies | 8 jobs (photometric_distort, std_minimal, std_randaugment) |

**Note:** Some tests depend on training completion (gen_IP2P, gen_step1x_new tests).

### Stage 2 Training (5.4%)

**Status:** 6/111 strategy-dataset combinations complete

- **Models:** PSPNet, SegFormer only (DeepLabV3+ excluded)
- **Datasets:** BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k (adverse conditions)
- **Training not yet prioritized** - focus is on completing Stage 1 first

### Ratio Ablation Study (mima2416)

**Status:** 🔄 13 jobs running, 54 pending

- Strategies: gen_step1x_new, gen_step1x_v1p2
- Models: PSPNet, SegFormer  
- Datasets: BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k
- Ratios: 0.0, 0.125, 0.25, 0.375, 0.625, 0.75, 0.875

### Extended Training Ablation (chge7185)

**Status:** 🔄 6 jobs running, 454 pending

- **Owner:** User chge7185
- **Duration:** 320k iterations (4× standard 80k) + finer iteration checkpoints (110k, 120k, ..., 320k)
- **Strategies:** gen_automold, gen_flux_kontext, gen_step1x_new, std_randaugment, etc.
- **Models:** PSPNet, SegFormer
- **Datasets:** BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k

**Previous Analysis (Jan 15):**
- Report: [docs/EXTENDED_TRAINING_ANALYSIS.md](docs/EXTENDED_TRAINING_ANALYSIS.md)
- Key finding: 160k captures 75% of gains at 50% compute cost
- Diminishing returns: 90k→160k (+0.75 mIoU), 160k→240k (+0.39), 240k→320k (+0.10)

---

## Pending Tasks

### High Priority

1. **Monitor Stage 1 Training/Testing Completion**
   - Run \`python scripts/update_training_tracker.py\` periodically
   - Run \`python scripts/update_testing_tracker.py\` after tests complete

2. **Ratio Ablation Analysis** (when jobs complete)
   - Generate ratio ablation figures
   - Analyze optimal mixing ratios per strategy

3. **Extended Training Analysis Follow-up** (when chge7185 jobs complete)
   - Analyze finer iteration granularity (110k, 120k, ..., 320k)
   - Update convergence curves

### Medium Priority

4. **Stage 2 Training Submission**
   - After Stage 1 is 100% complete
   - Submit PSPNet and SegFormer models for all strategies
   - Script: \`./scripts/generate_stage2_training_jobs.py\`

5. **Generate Final Stage 1 Leaderboard**
   - After all 334 tests complete
   - Update strategy rankings

### Low Priority

6. **Domain Adaptation Ablation Study** (optional)
   - Cross-dataset generalization evaluation
   - Script ready: \`./scripts/submit_domain_adaptation_ablation.sh --all\`

7. **Augmentation Combination Training** (optional)
   - Combine top std and gen strategies
   - Script ready: \`./scripts/submit_combination_training.sh\`

---

## Recently Completed (Jan 16, 2026)

### Directory Restructuring (Afternoon)
- ✅ **Created WEIGHTS_STAGE_2 directory** for Stage 2 (all_domains) training
- ✅ **Moved all _ad directories** from WEIGHTS to WEIGHTS_STAGE_2 (62 directories)
- ✅ **Removed _cd and _ad suffixes** from all dataset directories
- ✅ **Updated scripts** for new directory structure:
  - `unified_training_config.py` - Uses WEIGHTS for Stage 1, WEIGHTS_STAGE_2 for Stage 2
  - `update_training_tracker.py` - Checks both directories
  - `auto_submit_tests.py` - Removed _cd suffix lookups
  - `test_result_analyzer.py` - Updated baseline calculation
- ✅ **Created separate tracker files**:
  - `docs/TRAINING_TRACKER_STAGE1.md`
  - `docs/TRAINING_TRACKER_STAGE2.md`

### Bug Fixes (Morning)
- ✅ **Fixed path naming issue** in `unified_training_config.py`
  - Changed `dataset.lower().replace('-', '')` to `dataset.lower()` 
  - Keeps hyphen in "idd-aw" for consistent folder naming

### Data Migration (Earlier)
- ✅ **Merged iddaw_cd folders** into idd-aw_cd
  - 31 models moved/replaced across 11 strategies
  - All iddaw_cd folders removed

- ✅ **Moved test results** to correct location
  - 28 test results moved from `results/` to `WEIGHTS/.../test_results_detailed/`
  - Testing coverage jumped from 288 → 312 complete

### Code Cleanup
- ✅ **Updated tracker scripts**
  - Removed iddaw fallback logic from `update_training_tracker.py`
  - Updated Stage 2 to track all 3 models (DeepLabV3+, PSPNet, SegFormer)

### Training
- ✅ **gen_IP2P / IDD-AW / DeepLabV3+** - Job 9602408 (DONE)

---

## Monitoring Commands

\`\`\`bash
# Check all running/pending jobs
bjobs -w

# Check specific job types
bjobs -w | grep ratio_  # Ratio ablation jobs
bjobs -w | grep fg_     # Test jobs
bjobs -u chge7185       # Extended training jobs

# Update trackers
cd scripts && python update_training_tracker.py          # Stage 1
cd scripts && python update_training_tracker.py --stage 2  # Stage 2
cd scripts && python update_testing_tracker.py

# Submit missing tests (auto-detection)
cd scripts && python auto_submit_tests.py --dry-run   # Preview
cd scripts && python auto_submit_tests.py --limit 20  # Submit up to 20

# View job history
bhist -n 20
\`\`\`
