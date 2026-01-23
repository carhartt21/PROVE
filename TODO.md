# PROVE Project TODO List

**Last Updated:** 2026-01-23 (15:30)

## Current Job Status Summary

### Stage 1 (Clear Day Domain) - WEIGHTS directory
| Category | Running | Pending | Complete | Total |
|----------|--------:|--------:|---------:|------:|
| Training | 0 | 0 | 405 | 405 |
| Testing (non-MV) | 0 | 0 | ~324 | ~324 |
| **MV Testing** | **0** | **0** | **81** | **81** |

✅ **Stage 1 MapillaryVistas FULLY COMPLETE (Training + Testing)**
- All 81 MapillaryVistas Stage 1 models trained ✅
- All 81 MapillaryVistas Stage 1 tests completed ✅
- Results available in `test_results_detailed/*/results.json`

### Stage 2 (All Domains) - WEIGHTS_STAGE_2 directory
| Category | Running | Pending | Complete | Total |
|----------|--------:|--------:|---------:|------:|
| Training (non-MV) | 0 | 0 | 243 | 243 |
| Testing (non-MV) | 0 | 0 | 243 | 243 |
| **MV Training** | **6** | **27** | **48** | **81** |
| **MV Testing** | **0** | **48** | **0** | **48** |

**Stage 2 Status (as of 2026-01-23 15:30):**
- **Non-MV Training:** ✅ 243/243 complete
- **Non-MV Testing:** ✅ 243/243 complete
- **MapillaryVistas Training:** 🔄 48/81 complete (59%), 33 remaining
- **Top performer:** gen_CNetSeg (+0.58 over baseline at 43.68% mIoU)

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

### ✅ Stage 1 MapillaryVistas FULLY COMPLETE

All 81 MapillaryVistas Stage 1 models have completed:
- ✅ Training after BGR/RGB bug fix
- ✅ Fine-grained testing with per-domain/per-class metrics

**Results Location:** `/scratch/aaa_exchange/AWARE/WEIGHTS/*/mapillaryvistas/*/test_results_detailed/*/results.json`

### 🎯 IMMEDIATE: Generate Stage 1 Leaderboard with MapillaryVistas

Now that all Stage 1 tests are complete (including MV), generate the final leaderboard:

```bash
python analysis_scripts/generate_stage1_leaderboard.py
```

### 🔄 Stage 2 MapillaryVistas Retraining (59% complete)

**Status:** 48/81 complete, 6 running, ~27 pending

| Progress | Count |
|----------|-------|
| Complete | 48 |
| Running | 6 |
| Pending | ~27 |

**Monitor:** `bjobs -u mima2416 | grep rt_map`

**Test Script Ready:** `./scripts/run_stage2_mapillary_tests.sh`
- Run after training completes for each model
- Current script finds 48 models ready for testing

### Stage 2 Non-MapillaryVistas (100% Complete)

- **Training:** ✅ 243/243 complete
- **Testing:** ✅ 243/243 complete
- **Top performer:** gen_CNetSeg (+0.58 over baseline at 43.68% mIoU)

### Ratio Ablation Study

**Location:** `WEIGHTS_RATIO_ABLATION/`

- Strategies: gen_step1x_new, gen_step1x_v1p2
- Models: PSPNet, SegFormer  
- Datasets: BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k
- Ratios: 0.0, 0.125, 0.25, 0.375, 0.625, 0.75, 0.875

### Extended Training Ablation (chge7185)

**Location:** `WEIGHTS_EXTENDED/`

- **Owner:** User chge7185
- **Duration:** 320k iterations (4× standard 80k)
- **Strategies:** gen_automold, gen_flux_kontext, gen_step1x_new, std_randaugment, etc.
- **Models:** PSPNet, SegFormer
- **Datasets:** BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k

**Previous Analysis (Jan 15):**
- Report: [docs/EXTENDED_TRAINING_ANALYSIS.md](docs/EXTENDED_TRAINING_ANALYSIS.md)
- Key finding: 160k captures 75% of gains at 50% compute cost

---

## Pending Tasks

### High Priority

1. **🎯 Run Stage 1 MapillaryVistas Tests**
   - Script: `./scripts/run_stage1_mapillary_tests.sh --gpu 0`
   - 81 tests, ~13.5 hours

2. **Monitor Stage 2 MapillaryVistas Retraining**
   - Monitor: `bjobs -u mima2416 | grep rt_map`
   - 43 jobs remaining

3. **Generate Final Stage 1 Leaderboard** (after MV tests complete)
   - Script: `python analysis_scripts/generate_stage1_leaderboard.py`

### Medium Priority

4. **Run Stage 2 MapillaryVistas Tests** (after retraining completes)
   - Use auto_submit_tests_stage2.py or create similar local script

5. **Domain Adaptation Ablation** ✅ Ready
   - Scripts created: `run_domain_adaptation_tests.py`, `submit_domain_adaptation_ablation.sh`
   - All 27 strategies available via `--all-strategies` flag
   - Test matrix: 2 source datasets × 3 models × 27 strategies = 162 configurations
   - Usage: `python scripts/run_domain_adaptation_tests.py --all --all-strategies --dry-run`

6. **Ratio Ablation Analysis** (when jobs complete)
   - Generate ratio ablation figures
   - Analyze optimal mixing ratios per strategy

### Low Priority

7. **Extended Training Analysis Follow-up** (when chge7185 jobs complete)
   - Analyze finer iteration granularity
   - Update convergence curves

8. **Augmentation Combination Training** (optional)
   - Combine top std and gen strategies
   - Script ready: `./scripts/submit_combination_training.sh`

---

## Recently Completed

### Jan 23, 2026
- ✅ **Created Stage 1 MapillaryVistas test script** (`run_stage1_mapillary_tests.sh`)
  - Supports `--gpu`, `--limit`, `--batch-size`, `--dry-run` options
  - Finds all 81 MV configs needing tests

### Jan 22, 2026
- ✅ **Stage 2 non-MV testing complete** (243/243)
- ✅ **std_cutmix artifact resolved** - now ranks #27 at -0.29
- ✅ **Domain adaptation scripts enhanced** with all 27 strategies
- ✅ **Leaderboard scripts auto-refresh by default**

### Jan 21, 2026
- ✅ **Fixed BGR/RGB bug** in `custom_transforms.py` (commit d7b2b99)
- ✅ **Submitted MapillaryVistas retraining** (162 jobs total)
- ✅ **Stage 1 MV retraining complete** (81/81)

### Jan 16, 2026

### Directory Restructuring
- ✅ **Created WEIGHTS_STAGE_2 directory** for Stage 2 (all_domains) training
- ✅ **Moved all _ad directories** from WEIGHTS to WEIGHTS_STAGE_2 (62 directories)
- ✅ **Removed _cd and _ad suffixes** from all dataset directories
- ✅ **Updated scripts** for new directory structure
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

