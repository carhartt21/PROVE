# PROVE Project TODO List

**Last Updated:** 2026-01-22 (17:00)

## Current Job Status Summary

### Stage 1 (Clear Day Domain) - WEIGHTS directory
| Category | Running | Pending | Complete | Total |
|----------|--------:|--------:|---------:|------:|
| Training | 4 | 77 | 80 | 161 |
| Testing | 0 | 0 | ~80 | ~80 |

⚠️ **Stage 1 MapillaryVistas RETRAINING (162 jobs total, 81 per stage)**
- All MapillaryVistas models invalidated due to BGR/RGB bug in training

### Stage 2 (All Domains) - WEIGHTS_STAGE_2 directory
| Category | Running | Pending | Complete | Total |
|----------|--------:|--------:|---------:|------:|
| Training | 0 | 0 | 243 | 243 |
| Testing | 0 | 0 | 243 | 243 |

**Stage 2 Status (as of 2026-01-22 17:00):**
- **Training:** ✅ 243/243 complete (excluding MapillaryVistas retraining)
- **Testing:** ✅ 243/243 complete
- **std_cutmix artifact resolved:** Now ranks #27 at -0.29 below baseline
- **Top performer:** gen_CNetSeg (+0.58 over baseline)

---

## 🚨 CRITICAL: MapillaryRGBToClassId TRAINING Bug (Jan 21)

### The Bug

**Root Cause:** `mmcv.imfrombytes()` returns BGR by default, but `MapillaryRGBToClassId` transform was treating input as RGB.

**Code Location:** `custom_transforms.py` line ~117

**Wrong (before fix):**
```python
# Treated BGR input as RGB (WRONG!)
r = seg_map[:, :, 0]  # Actually B channel
g = seg_map[:, :, 1]  # G channel (OK)
b = seg_map[:, :, 2]  # Actually R channel
```

**Fixed:**
```python
# Correct BGR channel indexing
r = seg_map[:, :, 2]  # R is channel 2 in BGR
g = seg_map[:, :, 1]  # G is channel 1
b = seg_map[:, :, 0]  # B is channel 0
```

### Impact

**Training Impact:** ALL 162 MapillaryVistas models learned WRONG class mappings:
- Sky RGB (70,130,180) → was decoded as class 42 (Phone Booth) instead of class 27 (Sky)
- Vegetation RGB (107,142,35) → was decoded as class 25 (Mountain) instead of class 30 (Vegetation)
- Car RGB (0,0,142) → was decoded as class 54 (Car Mount) instead of class 55 (Car)

**Evidence:** Training logs showed `nan` for Sky and Vegetation IoU from iteration 0

### Fix Applied

**Commit:** d7b2b99 - "fix(training): Fix BGR/RGB channel order in MapillaryRGBToClassId"

**Files Modified:**
- `custom_transforms.py`: Fixed channel indexing for BGR input

### Retraining Status

| Stage | Jobs Submitted | Running | Pending | Job IDs |
|-------|---------------|---------|---------|---------|
| Stage 1 | 81 | ~4 | ~77 | 9739253-9739333 |
| Stage 2 | 81 | ~4 | ~77 | 9739334-9739414 |
| **Total** | **162** | **~8** | **~154** | 9739253-9739414 |

**Backup Location:** `/scratch/aaa_exchange/AWARE/WEIGHTS_BACKUP_BUGGY_MAPILLARY/`

**Monitor Progress:**
```bash
bjobs -u mima2416 -w | grep "rt_map" | wc -l  # Total jobs
bjobs -u mima2416 -w | grep "rt_map" | grep " RUN "  # Running jobs
```

---

## ⚠️ Testing Pipeline SEPARATE Bug (Already Fixed)

**Note:** There was also a BGR/RGB bug in `fine_grained_test.py` for test-time label loading.
That was fixed in commit 9313a5e (Jan 21).

**However**, the TRAINING bug in `custom_transforms.py` means all MapillaryVistas models 
learned wrong classes, so even correct test evaluation would show garbage results.

The training bug fix (d7b2b99) is the critical one that requires full retraining.

**Strategies Removed (permanently):**
- `std_minimal` - Not useful, removed from all datasets

**Strategies Retraining (21 jobs, IDs 9611966-9611986):**
- MapillaryVistas: gen_cyclediffusion (×3), gen_TSIT (×3)
- OUTSIDE15k: std_cutmix (×3), std_mixup (×3), gen_cyclediffusion (×3), gen_flux_kontext (×3), gen_TSIT (×3)

**Native Classes Default:** `unified_training.py` now uses native classes by default.
Use `--no-native-classes` to force Cityscapes 19 classes.

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
- Nested `_cd` directories cleaned up (gen_Qwen_Image_Edit, gen_UniControl)

---

## Active Tasks

### 🔄 Stage 1 Retraining (21 jobs running)

**Job IDs:** 9611966-9611986 (moved to top of queue)

| Strategy | MapillaryVistas | OUTSIDE15k |
|----------|-----------------|------------|
| gen_cyclediffusion | 🔄 3 jobs | 🔄 3 jobs |
| gen_flux_kontext | ✅ | 🔄 3 jobs |
| gen_TSIT | 🔄 3 jobs | 🔄 3 jobs |
| std_cutmix | ✅ | 🔄 3 jobs |
| std_mixup | ✅ | 🔄 3 jobs |

- Once complete, Stage 1 will be 107/107 (100%)

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
5. **Domain Adaptation Ablation** ✅ Ready
   - Scripts created: \`run_domain_adaptation_tests.py\`, \`submit_domain_adaptation_ablation.sh\`
   - All 27 strategies available via \`--all-strategies\` flag
   - Test matrix: 2 source datasets × 3 models × 27 strategies = 162 configurations
   - Usage: \`python scripts/run_domain_adaptation_tests.py --all --all-strategies --dry-run\`

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

