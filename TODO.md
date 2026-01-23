# PROVE Project TODO List

**Last Updated:** 2026-01-23 (17:00)

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
**Status:** ✅ Sufficient coverage (187 checkpoints)

| Strategy | Stage 1 Rank | Checkpoints | Datasets |
|----------|-------------|-------------|----------|
| gen_cycleGAN | #2 (+1.13) | 28 | IDD-AW, OUTSIDE15k |
| gen_stargan_v2 | #4 (+1.08) | 9 | IDD-AW |
| gen_cyclediffusion | #6 (+1.05) | 9 | IDD-AW |
| gen_step1x_new | #13 (+0.94) | 56 | BDD10k, IDD-AW, MV*, OUTSIDE15k |
| gen_step1x_v1p2 | #21 (+0.69) | 46 | BDD10k, IDD-AW, MV*, OUTSIDE15k |
| gen_TSIT | #22 (+0.63) | 39 | BDD10k, IDD-AW, OUTSIDE15k |

*MV = MapillaryVistas (backed up - buggy)

**Ratios:** 0.00, 0.125, 0.25, 0.375, 0.625, 0.75, 0.875 (0.50 = standard training)

**Next Action:** Run analysis on existing data (`analysis_scripts/analyze_ratio_ablation.py`)

### Extended Training Study

**Location:** `WEIGHTS_EXTENDED/`
**Owner:** User chge7185
**Status:** ✅ Sufficient coverage (~959 checkpoints)

| Strategy | Stage 1 Rank | Checkpoints | Datasets |
|----------|-------------|-------------|----------|
| gen_cycleGAN | #2 (+1.13) | 96 | BDD10k, IDD-AW, OUTSIDE15k |
| gen_flux_kontext | #5 (+1.07) | 96 | BDD10k, MV*, OUTSIDE15k |
| gen_cyclediffusion | #6 (+1.05) | 192 | All 4 datasets |
| gen_step1x_new | #13 (+0.94) | 120 | All 4 datasets |
| std_randaugment | #25 (+0.45) | 72 | All 4 datasets |
| gen_albumentations_weather | #16 (+0.84) | 96 | BDD10k, IDD-AW, OUTSIDE15k |
| gen_TSIT | #22 (+0.63) | 96 | BDD10k, IDD-AW, OUTSIDE15k |
| gen_UniControl | #19 (+0.71) | 96 | BDD10k, IDD-AW, OUTSIDE15k |
| gen_automold | #12 (+0.96) | 95 | BDD10k, IDD-AW |

**Iterations:** 40k, 60k, 80k, 100k, 120k, 140k, 160k, 320k

**Key Finding (Analysis Complete):**
- **160k iterations** captures ~75% of gains at 50% compute cost
- Report: [docs/EXTENDED_TRAINING_ANALYSIS.md](docs/EXTENDED_TRAINING_ANALYSIS.md)

---

## Pending Tasks

### High Priority

1. **🎯 Domain Adaptation Ablation** (NO TRAINING REQUIRED!)
   - **Status:** Ready to start
   - **Cost:** Testing only - uses existing Stage 1 checkpoints
   - **Strategies:** Top 5 (gen_Attribute_Hallucination, gen_cycleGAN, gen_stargan_v2, gen_flux_kontext, gen_cyclediffusion) + baseline
   - **Model:** SegFormer (most robust from baseline analysis: 8.7% domain gap)
   - **Test matrix:** 3 source datasets × 5 target domains (Cityscapes + ACDC) × 6 strategies
   - **Script:** `python scripts/run_domain_adaptation_tests.py --all --dry-run`
   - **Value:** High publication value with zero training cost

2. **Monitor Stage 2 MapillaryVistas Retraining**
   - Current: 48/81 complete (59%)
   - Monitor: `bjobs -u mima2416 | grep rt_map`

3. **Run Stage 2 MapillaryVistas Tests** (as training completes)
   - Script: `./scripts/run_stage2_mapillary_tests.sh`

### Medium Priority

4. **Ratio Ablation Analysis** (NO ADDITIONAL TRAINING)
   - **Current coverage:** ✅ Sufficient
   - **Top performers covered:** gen_cycleGAN, gen_stargan_v2, gen_cyclediffusion (all top-10)
   - **Checkpoints:** 187 (6 strategies × multiple ratios × datasets)
   - **Action:** Run analysis on existing data
   - **Script:** `analysis_scripts/analyze_ratio_ablation.py`

5. **Extended Training Analysis** (NO ADDITIONAL TRAINING)
   - **Current coverage:** ✅ Sufficient
   - **Top performers covered:** gen_cycleGAN, gen_cyclediffusion, gen_flux_kontext
   - **Checkpoints:** ~959 (9 strategies × 8 iterations × datasets)
   - **Key finding:** 160k iterations = 75% of gains at 50% compute
   - **Action:** Analysis complete, document in paper

### Low Priority

6. **Combination Strategies Analysis** (NO ADDITIONAL TRAINING)
   - **Current coverage:** 53 checkpoints (IDD-AW only)
   - **Status:** Sufficient for initial exploratory analysis
   - **Action:** Analyze existing IDD-AW results
   - **Future:** If results warrant, expand to BDD10k (most validated)

---

## Ablation Study Summary (Based on Leaderboard Analysis)

### Top Strategies (Both Stage 1 & Stage 2 Top 10)
1. gen_Attribute_Hallucination (+1.36 Stage 1, +0.90 Stage 2)
2. gen_cycleGAN (+1.13 Stage 1, +0.13 Stage 2)
3. gen_stargan_v2 (+1.08 Stage 1, +0.21 Stage 2)
4. gen_cyclediffusion (+1.05 Stage 1, +0.17 Stage 2)
5. gen_CNetSeg (+1.00 Stage 1, +0.29 Stage 2)
6. gen_augmenters (+0.99 Stage 1, +0.15 Stage 2)
7. std_autoaugment (+0.94 Stage 1, +0.16 Stage 2)

### Current Ablation Coverage vs Top Strategies

| Study | Top Strategies Covered | Status |
|-------|----------------------|--------|
| **Ratio Ablation** | gen_cycleGAN, gen_stargan_v2, gen_cyclediffusion | ✅ Sufficient |
| **Extended Training** | gen_cycleGAN, gen_cyclediffusion, gen_flux_kontext | ✅ Sufficient |
| **Combinations** | Multiple gen_* + std_* | ✅ Sufficient (IDD-AW) |
| **Domain Adaptation** | All 6 top strategies | ⏳ Ready (no training) |

### Recommendation: ZERO NEW TRAINING NEEDED
- All ablation studies have sufficient coverage of top-performing strategies
- Focus on **testing** and **analysis** of existing checkpoints
- **Domain Adaptation is highest priority** (zero training cost, high value)

---

## Recently Completed

7. **Extended Training Analysis Follow-up** (when chge7185 jobs complete)
   - Analyze finer iteration granularity
   - Update convergence curves

8. **Augmentation Combination Training** (optional)
   - Combine top std and gen strategies
   - Script ready: `./scripts/submit_combination_training.sh`

---

## Recently Completed

### Jan 23, 2026
- ✅ **Stage 1 Baseline Analysis (Publication)** - 4 figures + 4 tables
  - Script: `result_figures/baseline_consolidated/generate_stage1_baseline.py`
  - Output: `result_figures/baseline_consolidated/stage1_baseline_output/`
  - Key findings: Overall 33.3% mIoU, 10.1% domain gap
  - SegFormer most robust (8.7% gap), Night hardest domain (-14.9%)

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

