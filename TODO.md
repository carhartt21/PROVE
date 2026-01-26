# PROVE Project TODO List

**Last Updated:** 2026-01-26 (14:30)

## Current Job Status Summary

### Stage 1 (Clear Day Domain) - WEIGHTS directory
| Category | Complete | Notes |
|----------|:--------:|-------|
| Training | 324 | All models trained |
| Testing | 330 | All tests complete |

✅ **Stage 1 FULLY COMPLETE (Training + Testing)**

### Stage 2 (All Domains) - WEIGHTS_STAGE_2 directory
| Category | Complete | Notes |
|----------|:--------:|-------|
| Training | 325 | All models trained |
| Testing | 344 | All tests complete |

✅ **Stage 2 FULLY COMPLETE (Training + Testing)**
- **Top performer:** gen_stargan_v2 (41.73% mIoU)

### ~~Batch Size Ablation Study~~ (CANCELLED - 2026-01-26)

**Status:** ❌ Cancelled after preliminary analysis

**Preliminary Results:**
| BS | Time/Sample | Throughput | Speedup |
|----|-------------|------------|---------|
| 2 | 0.0408s | 24.5 img/s | 1.0x (baseline) |
| 4 | 0.0389s | 25.7 img/s | 1.05x |
| 8 | 0.0386s | 25.9 img/s | 1.06x |
| 16 | 0.0365s | 27.4 img/s | 1.12x |

**Conclusion:** Speedup is minimal (~5-12%), not worth changing existing experiments for comparability. **Continuing with batch_size=2**.

### 🔄 Ratio Ablation Study - WEIGHTS_RATIO_ABLATION

**Last Updated:** 2026-01-26 14:28

**Directory Structure (reorganized):**
```
WEIGHTS_RATIO_ABLATION/
├── stage1/  # domain_filter=clear_day (51 models)
│   ├── gen_cycleGAN/{idd-aw,outside15k}/
│   ├── gen_cyclediffusion/idd-aw/
│   └── gen_stargan_v2/idd-aw/
└── stage2/  # no domain_filter (102 models)
    ├── gen_step1x_new/{bdd10k,idd-aw,mapillaryvistas,outside15k}/
    └── gen_step1x_v1p2/{bdd10k,idd-aw,mapillaryvistas,outside15k}/
```

**Current Queue:** 52 pending (Stage 1 existing strategies)

| Stage | Strategy Type | Trained | Missing | Jobs |
|-------|--------------|---------|---------|------|
| **Stage 1** | Existing (gen_cycleGAN, etc.) | 32 | 52 | 🔄 52 PEND (151089-151140) |
| **Stage 1** | Top-5 (new strategies) | 23 | 117 | ⏳ Not submitted |
| **Stage 2** | Existing (gen_step1x_*) | 56 | 0 | ✅ Complete |
| **Stage 2** | Top-5 (new strategies) | 0 | 140 | ⏳ Not submitted |

**Existing Strategies (prior runs):**
| Strategy | Stage | Models | Tested |
|----------|-------|--------|--------|
| gen_cycleGAN | 1 | 14 → 28 (🔄 +14 training) | 14 |
| gen_cyclediffusion | 1 | 9 → 14 (🔄 +5 training) | 9 |
| gen_stargan_v2 | 1 | 9 → 14 (🔄 +5 training) | 9 |
| gen_step1x_new | 2 | 56 | 56 ✅ |
| gen_step1x_v1p2 | 2 | 46 | 0 |

**Configuration:** 7 ratios × 2 models × 2 datasets = 28 jobs per strategy
- **Ratios:** 0.00, 0.12, 0.25, 0.38, 0.62, 0.75, 0.88
- **Models:** pspnet_r50, segformer_mit-b5
- **Datasets:** BDD10k, IDD-AW

**Training Locks:** Enabled (prevents parallel training of same config)

**Monitor Progress:**
```bash
bjobs -u mima2416 | wc -l  # Check queue
find /scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION/stage1 -name "iter_80000.pth" | wc -l  # Stage 1
find /scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION/stage2 -name "iter_80000.pth" | wc -l  # Stage 2
python scripts/submit_ratio_ablation_training.py --stage 1 --existing-strategies --preflight  # Status
```

### Extended Training Study - WEIGHTS_EXTENDED

| Category | Complete | Notes |
|----------|:--------:|-------|
| Training | 0 models | Owner: chge7185 |
| Testing | 774 results | Initial checkpoints tested |

---

## 🎯 Next Steps

### Immediate Actions

1. **Monitor Stage 1 Ratio Ablation** (~12-24 hours)
   ```bash
   bjobs -u mima2416 | grep -c "RUN\|PEND"
   python scripts/submit_ratio_ablation_training.py --stage 1 --existing-strategies --preflight
   ```

2. **Run Tests on Completed Ratio Ablation Models** (after training completes)
   ```bash
   python scripts/submit_ablation_tests.py --stage 1 --dry-run
   python scripts/submit_ablation_tests.py --stage 2 --dry-run
   ```

3. **Submit Stage 2 New Strategies** (when queue opens)
   ```bash
   python scripts/submit_ratio_ablation_training.py --stage 2 --dry-run
   python scripts/submit_ratio_ablation_training.py --stage 2
   ```

### After Ratio Ablation Completes

4. **Generate Ratio Ablation Analysis**
   ```bash
   python analysis_scripts/analyze_ratio_ablation.py
   python analysis_scripts/visualize_ratio_ablation.py
   ```

5. **Update Final Leaderboards**
   ```bash
   python analysis_scripts/generate_stage1_leaderboard.py
   python analysis_scripts/generate_stage2_leaderboard.py
   ```

---

### ⏸️ Paused Tasks

**Initial Checkpoint Tests (10k-80k iterations)**
- **Status:** ~774 complete
- **Purpose:** Complete learning curves for extended training analysis

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

### Retraining Status - ✅ COMPLETE (Jan 24)

All MapillaryVistas retraining is complete:

| Stage | Status | Models |
|-------|--------|--------|
| Stage 1 | ✅ Complete | 81/81 |
| Stage 2 | ✅ Complete | 81/81 (by user chge7185) |
| **Total** | ✅ **COMPLETE** | 162/162 |

**Backup Location:** `/scratch/aaa_exchange/AWARE/WEIGHTS_BACKUP_BUGGY_MAPILLARY/`

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

### ✅ Stage 1 FULLY COMPLETE

All Stage 1 training and testing is complete:
- ✅ 405 models trained (81 per dataset × 5 datasets - but ACDC is test-only)
- ✅ All fine-grained testing completed with per-domain/per-class metrics
- ✅ MapillaryVistas 81/81 models retrained after BGR/RGB bug fix

**Results Location:** `/scratch/aaa_exchange/AWARE/WEIGHTS/*/*/test_results_detailed/*/results.json`

### ✅ Stage 2 FULLY COMPLETE

All Stage 2 training and testing is complete:
- ✅ 325 models trained (BDD: 82, IDD-AW: 81, MV: 81, OUTSIDE15k: 81)
- ✅ 324 test results available (all strategies × all datasets)
- ✅ MapillaryVistas 81/81 models retrained after BGR/RGB bug fix (completed by user chge7185)

**Results Location:** `/scratch/aaa_exchange/AWARE/WEIGHTS_STAGE_2/*/*/test_results_detailed/*/results.json`

### 🎯 IMMEDIATE: Generate Final Leaderboards

Both stages are fully complete - generate comprehensive leaderboards:

```bash
# Stage 1 Leaderboard
python analysis_scripts/generate_stage1_leaderboard.py

# Stage 2 Leaderboard  
python analysis_scripts/generate_stage2_leaderboard.py
```

### 🎯 NEXT: Stage Comparison Analysis

Compare Stage 1 vs Stage 2 performance across all strategies and datasets.

### ✅ Stage 2 Non-MapillaryVistas (100% Complete)

- **Training:** ✅ 243/243 complete
- **Testing:** ✅ 243/243 complete
- **Top performer:** gen_CNetSeg (+0.58 over baseline at 43.68% mIoU)

### Ratio Ablation Study

**Location:** `WEIGHTS_RATIO_ABLATION/`
**Status:** 🔄 Training submitted (280 jobs - 2026-01-25 00:44)

#### Current Training Progress
| Stage | Strategy | Jobs | Status |
|-------|----------|------|--------|
| **Stage 1** | gen_Attribute_Hallucination | 28 | 🔄 Submitted |
| **Stage 1** | gen_cycleGAN | 28 | 🔄 Submitted |
| **Stage 1** | gen_Img2Img | 28 | 🔄 Submitted |
| **Stage 1** | gen_stargan_v2 | 28 | 🔄 Submitted |
| **Stage 1** | gen_flux_kontext | 28 | 🔄 Submitted |
| **Stage 2** | gen_stargan_v2 | 28 | 🔄 Submitted |
| **Stage 2** | gen_UniControl | 28 | 🔄 Submitted |
| **Stage 2** | gen_CNetSeg | 28 | 🔄 Submitted |
| **Stage 2** | gen_VisualCloze | 28 | 🔄 Submitted |
| **Stage 2** | gen_cycleGAN | 28 | 🔄 Submitted |

**Configuration:**
- **Ratios:** 0.00, 0.12, 0.25, 0.38, 0.62, 0.75, 0.88 (7 values, excluding 0.50 from main training)
- **Models:** pspnet_r50, segformer_mit-b5 (matching existing ablation pattern)
- **Datasets:** BDD10k, IDD-AW

**Existing Trained Models (prior runs):**
| Strategy | Models | Notes |
|----------|--------|-------|
| gen_cycleGAN | 28 | Stage 1 idd-aw only |
| gen_cyclediffusion | 9 | Stage 1 partial |
| gen_stargan_v2 | 9 | Stage 1 partial |
| gen_step1x_new | 56 | Stage 2 complete (all 4 datasets) |
| gen_step1x_v1p2 | 46 | Partial |

**⚠️ Bug Fix (2026-01-25):** Moved corrupted gen_TSIT ratio ablation to backup:
`/scratch/aaa_exchange/AWARE/WEIGHTS_BACKUP_BUGGY_TSIT_RATIO_ABLATION/`
- Bug: `CityscapesLabelIdToTrainId` incorrectly applied to BDD10k trainIds
- Impact: 13/19 classes corrupted (only 6 learned), ~0.2% mIoU on tests
- Will retrain gen_TSIT with correct config after current batch completes

**Monitor Progress:**
```bash
bjobs -u mima2416 | wc -l  # Total pending jobs
find /scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION -name "iter_80000.pth" | wc -l  # Trained models
```

**Scripts:**
- `scripts/submit_ratio_ablation_training.py` - Training submission script (updated 2026-01-25)

### Extended Training Study

**Location:** `WEIGHTS_EXTENDED/`
**Owner:** chge7185 + mima2416
**Status:** 🔄 Testing in progress (714/1169 results = 61%)

| Phase | Iterations | Tests Complete | Tests Total | Coverage |
|-------|------------|----------------|-------------|----------|
| **Initial** | 10k-80k | 205 | 392 | 52% |
| **Extended** | 90k-320k | 509 | 936 | 54% |
| **Total** | 10k-320k | **714** | **1328** | **54%** |

**Analysis Findings (Jan 24, 22:20):**
- Learning curve: 37.7% (10k) → 45.0% (80k) → 49.8% (320k)
- **77.4%** of configs improve beyond 80k baseline
- **Mean improvement: +1.41 mIoU** with extended training
- Most configs converge at 310k-320k iterations
- Best strategy: gen_cyclediffusion (53.8% mIoU)

**Currently Pending Jobs: 224** (192 initial + 32 DA)
**Additional Tests Available:** 455 (90k-320k missing)

**Iterations:** 10k, 20k, 30k, 40k, 50k, 60k, 70k, 80k, 90k-320k (every 10k)

**Key Finding (Analysis Complete):**
- **160k iterations** captures ~75% of gains at 50% compute cost
- Report: [docs/EXTENDED_TRAINING_ANALYSIS.md](docs/EXTENDED_TRAINING_ANALYSIS.md)

---

## Pending Tasks

### High Priority

1. **🔄 Ablation Study Testing** (RUNNING - 180 jobs pending)
   - **Ratio Ablation:** 141 jobs for gen_TSIT, gen_step1x_new, gen_step1x_v1p2
   - **Extended Training:** 39 jobs for 320k iteration
   - **Monitor:** `bjobs -u mima2416 | grep -E "abl_|ext_" | wc -l`
   - **Est. Completion:** ~4 hours from submission (2026-01-24 14:35)

2. **🎯 Domain Adaptation Ablation** (READY - no training needed)
   - **Status:** ⏳ Ready to start
   - **Partial Results:** 64 tests complete
   - **Key findings so far:**
     - BDD10k→ACDC: 23.7% mIoU (best source dataset)
     - IDD-AW→ACDC: 13.8% mIoU 
     - gen_TSIT leads (+3.9% vs baseline at 21.4%)
     - SegFormer best model (24.0% on ACDC)
   - **Report:** [docs/ABLATION_STUDIES_ANALYSIS.md](docs/ABLATION_STUDIES_ANALYSIS.md)

### ✅ COMPLETED (Jan 24):

3. **Stage 2 MapillaryVistas Retraining** - ✅ Complete (81/81 by user chge7185)

4. **Stage 2 MapillaryVistas Testing** - ✅ Complete (81/81)

5. **Stage 1 & Stage 2 Leaderboards** - ✅ Generated

6. **Stage Comparison Analysis** - ✅ 6 figures generated in `result_figures/stage_comparison/`

### Medium Priority

7. **Run Ablation Analysis Scripts** (after testing completes)
   - `analysis_scripts/analyze_ratio_ablation.py` - Ratio study analysis
   - `analysis_scripts/analyze_extended_training.py` - Extended training curves
   - `analysis_scripts/visualize_ratio_ablation.py` - Generate figures

8. **Top Strategy Ablation Gap** (OPTIONAL)
   - gen_Attribute_Hallucination: No ratio/extended coverage
   - gen_Img2Img: No ratio/extended coverage
   - **Decision:** Only needed if paper requires deeper analysis of top performers

### Low Priority

9. **Combination Strategies Analysis** (ALL TESTED ✅)
   - **Checkpoints:** 53 (all IDD-AW)
   - **Key finding:** photometric_distort combos dominate (45.1% avg vs ~40% others)
   - **Best combo:** std_mixup + photometric_distort (45.2%)
   - **Action:** Analyze existing IDD-AW results

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
| **Extended Training** | gen_cycleGAN, gen_cyclediffusion, gen_flux_kontext | 🔶 69% tested |
| **Combinations** | Multiple gen_* + std_* | ✅ Sufficient (IDD-AW) |
| **Domain Adaptation** | All 6 top strategies | 🔄 MV re-testing (32 jobs) |

### Recommendation: ZERO NEW TRAINING NEEDED
- All ablation studies have sufficient coverage of top-performing strategies
- Focus on **testing** and **analysis** of existing checkpoints
- **Initial checkpoints (10k-80k)** - 192 tests remaining to complete learning curves
- **Domain Adaptation MapillaryVistas** - 32 re-tests running after BGR/RGB bug fix

---

## Recently Completed

### Jan 25, 2026 (01:30)
- ✅ **Stage 1 Ratio Ablation Training Submitted** - 140 jobs for top 5 strategies
  - Strategies: gen_Attribute_Hallucination, gen_cycleGAN, gen_Img2Img, gen_stargan_v2, gen_flux_kontext
  - Configuration: 7 ratios × 2 models × 2 datasets = 28 jobs per strategy
  - Status: 2 running, 118 pending
- ✅ **Fixed ratio ablation submission script** - `scripts/submit_ratio_ablation_training.py`
  - Fixed: `--output-dir` → `--work-dir` argument
  - Fixed: LSF syntax for BatchGPU queue
  - Added: `--preflight` option for checking existing weights
- ✅ **Moved buggy gen_TSIT ratio ablation** to backup directory
  - Bug: `CityscapesLabelIdToTrainId` incorrectly applied to BDD10k trainIds
  - Location: `WEIGHTS_BACKUP_BUGGY_TSIT_RATIO_ABLATION/`

### Jan 24, 2026 (Evening Session - 22:10)
- ✅ **Initial checkpoint tests (10k-80k)** - 269/392 complete (69%) for extended training curves
- ✅ **Domain Adaptation MapillaryVistas cleanup** - Moved 28 buggy results to backup
- ✅ **MapillaryVistas DA re-tests submitted** - 32 jobs using fixed models
- ✅ **Fixed submit_initial_checkpoint_tests.py** - Now correctly detects timestamp-based result dirs
- ✅ **Updated STUDY_COVERAGE_ANALYSIS.md** - Full ablation status documented

### Jan 24, 2026 (Earlier)
- ✅ **Stage 1 & Stage 2 FULLY COMPLETE** - All 648 tests (324 + 324) finished
- ✅ **Stage comparison analysis** - 6 figures generated in `result_figures/stage_comparison/`
- ✅ **Stage 1 & Stage 2 Leaderboards** - Generated with full results
- ✅ **Killed 80 duplicate jobs** - Work already completed by user chge7185
- ✅ **MapillaryVistas Stage 2 retraining** - 81/81 complete (by user chge7185)
- ✅ **Submitted ablation tests** - 180 jobs (141 ratio + 39 extended)
- ✅ **Created ablation test scripts**:
  - `scripts/submit_ablation_tests.py` - Ratio ablation tests
  - `scripts/submit_extended_tests.py` - Extended training tests

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
- ✅ **Created WEIGHTS_STAGE_2 directory** for Stage 2 (all_domains) training
- ✅ **Moved all _ad directories** from WEIGHTS to WEIGHTS_STAGE_2 (62 directories)
- ✅ **Removed _cd and _ad suffixes** from all dataset directories
- ✅ **Updated scripts** for new directory structure

