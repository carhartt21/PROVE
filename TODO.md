# PROVE Project TODO

**Last Updated:** 2026-02-08 (20:40)

---

## 🆕 Stage 1 Gap Analysis & Fixes (2026-02-08 16:30)

### Tracker Analysis Results

| Metric | Stage 1 Training | Stage 1 Testing | Stage 2 Training | Stage 2 Testing |
|--------|-----------------|-----------------|------------------|-----------------|
| Complete | 325/444 (73.2%) | 243/255 (95.3%) | 49/444 (11.0%) | 58 total |
| Running | 11 | 0 | 2 | 0 |
| Pending | 99 | 32 | 393 | 99 |
| Failed | 13 | 12 missing | 4 | 0 |

### 🐛 Bug Fix: Testing Tracker IDD-AW Directory Naming
- **Issue:** `update_testing_tracker.py` uses `idd-aw` as directory name, but actual directories use `iddaw` (no hyphen)
- **Impact:** ALL IDD-AW test results (26 strategies) reported as "pending" when they actually exist
- Training tracker already normalizes (`dataset.replace('-', '')`) but testing tracker didn't
- **Fix:** ✅ Added `dataset.replace('-', '')` normalization to testing tracker (3 locations)
- **Result:** IDD-AW now shows **26 complete** (was 0); total tests: 340 complete (was 243)

### 📊 Failed Training Investigation (26 stale trainings found)

**Category 1: Wrong model variant — 6 entries (IGNORE)**
All are `segformer_mit-b5_ratio0p50` (should be B3, not B5). Stale leftovers from incorrect config.
- gen_LANIT, gen_VisualCloze, gen_albumentations_weather, gen_automold, gen_flux_kontext, gen_step1x_v1p2

**Category 2: Resumable (>50% complete) — 3 entries (RESUME)**
| Strategy | Dataset | Model | Progress | Action |
|----------|---------|-------|----------|--------|
| gen_LANIT | bdd10k | segnext_mscan-b | 50000/80000 (62%) | ⏳ Resume |
| gen_flux_kontext | bdd10k | segnext_mscan-b | 65000/80000 (81%) | ⏳ Resume |
| gen_step1x_new | bdd10k | segnext_mscan-b | 60000/80000 (75%) | ⏳ Resume |

**Category 3: Early failures (<50% complete) — 17 entries**
Affected strategies: gen_Img2Img (6), gen_albumentations_weather (4), gen_cyclediffusion (3),
gen_VisualCloze (1), gen_UniControl (1), gen_automold (1), gen_step1x_v1p2 (1)
Common pattern: SegNeXt and PSPNet models stopped at iter_2000-5000 of 15000-80000.
Likely cause: OOM or training crash. Need restart.

### 📋 Action Plan
1. ✅ Fix testing tracker IDD-AW bug — IDD-AW: 0→26 complete, total: 243→340
2. ✅ IDD-AW test gap was a tracker bug — tests actually exist (26/28 complete)
3. ✅ 13 "missing" BDD10k tests are models with incomplete training (not untested)
4. ✅ Investigated 26 stale trainings: 6 wrong model, 3 resumable, 17 early failures
5. ✅ Submitted test jobs for 3 partially-complete models using best_val checkpoints:
   - gen_LANIT/bdd10k/segnext: best_val_mIoU_iter_45000 → Job 2098551
   - gen_flux_kontext/bdd10k/segnext: best_val_mIoU_iter_60000 → Job 2098668
   - gen_step1x_new/bdd10k/segnext: best_val_mIoU_iter_60000 → Job 2098749
   - These models have old 80k configs but trained 50-65k iters (well past current 15k standard)
6. ✅ Submitted 28 gen_ strategy restart jobs (`batch_training_submission.py --resume`)
7. ✅ Deleted 27 wrong-model `segformer_mit-b5` directories (~90GB reclaimed)
8. ✅ Investigated Cityscapes-gen retests: **quoting bug** caused 10/12 original retests to fail silently
9. ✅ Submitted 28 properly-formatted Cityscapes test jobs (2119269-2119296) using bash script files

### 🐛 Bug Fix: max_iters Config Regex
- **Issue:** `auto_submit_tests.py` and `update_training_tracker.py` used `r'max_iters\s*=\s*(\d+)'`
  which only matches keyword format (`max_iters=15000`) but NOT dict format (`'max_iters': 80000`)
- **Impact:** Older configs using dict literal syntax were silently skipped
- **Fix:** Changed regex to `r"'?max_iters'?\s*[=:]\s*(\d+)"` to handle both formats

### Current Cluster Status (2026-02-08 20:40)
| Category | RUN | PEND |
|----------|----:|-----:|
| Stage 1 training | 2 | 0 |
| Cityscapes-gen training | 5 | 65 |
| Cityscapes-gen retests (new) | 0 | 28 |
| gen_ strategy restarts | 0 | 28 |
| **Total** | **7** | **121** |

---

## 🆕 Cityscapes-Gen Evaluation Stage (2026-02-08)

### ✅ Bug Fixes Applied (Training)
Two critical bugs were found and fixed that caused ALL gen_ cityscapes-gen jobs to fail with `ValueError: No generated images found for dataset 'Cityscapes'`:

1. **`generated_images_dataset.py`** (case sensitivity bug):
   - `get_dataset_entries('Cityscapes')` did case-sensitive substring match on `original_path` containing `CITYSCAPES` → always returned 0
   - `get_available_datasets()` had hardcoded dataset list missing `'Cityscapes'`
   - **Fix:** Now uses `dataset` CSV column with case-insensitive match

2. **`tools/generate_manifests.py`** (nested directory bug):
   - Attribute_Hallucination's Cityscapes at `Cityscapes/generated/{domain}/{city}/` was missed
   - **Fix:** Added intermediary directory traversal (`generated/`, `test_latest/`, etc.)

### ✅ Bug Fixes Applied (Testing) — commit `0cc2d69`
Five compounding bugs caused ALL Cityscapes test results to be empty (`mIoU=N/A`). ACDC cross-domain tests were unaffected.

1. **`fine_grained_test.py`** — Hardcoded weather domains:
   - `domains = ['clear_day', 'cloudy', ...]` was hardcoded; Cityscapes uses cities (`frankfurt`, `lindau`, `munster`)
   - **Fix:** `domains = DATASET_DOMAINS.get(folder_name, [...])` + added `'Cityscapes': ['frankfurt', 'lindau', 'munster']`

2. **`fine_grained_test.py`** — Label filename mismatch:
   - Images named `*_leftImg8bit.png`, but code only matched `_rgb_anon` / `_leftImg8bit` for ACDC/Cityscapes style
   - **Fix:** Added explicit Cityscapes matching: `_leftImg8bit` → `_gtFine_labelIds`

3. **`fine_grained_test.py`** — Wrong label_type:
   - `cityscapes_trainid` assumed labels contain train IDs; FINAL_SPLITS has `_gtFine_labelIds` (original label IDs needing remapping)
   - **Fix:** Changed to `cityscapes_labelid`

4. **`scripts/batch_training_submission.py`** — Wrong DATA_ROOT for tests:
   - Used native Cityscapes path (`/scratch/.../CITYSCAPES`) instead of unified FINAL_SPLITS
   - **Fix:** Always use `FINAL_SPLITS` + `test` split

**Impact:** 12 completed segformer trainings had empty Cityscapes test results. 12 re-test jobs submitted (jobs 1996609-1996617, 2022051-2022053).

### 🐛 Bug Fix: Retest Submission Quoting (2026-02-08 20:40)
- **Issue:** Original 12 retest jobs used `bash -c '...'` with single-quoted echo statements inside, breaking shell quoting
- **Root cause:** `echo 'Re-testing Cityscapes: ...'` inside `bash -c '...'` terminated the outer single quote
- **Impact:** 10/12 retests ran for **0 seconds** (Python command never executed); 1 ran 23s but produced no output
- **Only 1 of 12 retests worked** (gen_VisualCloze, job 2022053, 30s runtime → mIoU=63.27)
- The other 2 valid results (gen_SUSTechGAN=63.48, gen_step1x_v1p2=62.85) came from **auto-tests** after training completed with fixed code
- **Fix:** Re-submitted 28 test jobs (2119269-2119296) using standalone bash script files (`test_job.sh`) instead of inline `bash -c`

### ✅ Manifest Fixes
- Regenerated Attribute_Hallucination CSV: now captures 14,875 Cityscapes entries
- stargan_v2 reorganized: 205,248 images at 100% match (was 17,850)
- All 25 method CSVs verified with Cityscapes coverage

### 🔄 Jobs Status (2026-02-08 20:40)
**Training: 35 cityscapes-gen jobs** (of original 91):

| Model | Total | Completed | Running | Still Pending |
|-------|-------|-----------|---------|---------------|
| segformer_mit-b3 | 24 | 22 | 0 | 2 (training) |
| pspnet_r50 | 24 | 5 | 1 | 18 |
| segnext_mscan-b | 24 | 2 | 1 | 21 |
| mask2former_swin-b | 19 | 2 | 3 | 14 |
| **Total** | **91** | **31** | **5** | **55** |

**Testing: 28 re-test jobs submitted** (2119269-2119296):
- Cover all 31 completed trainings minus 3 with valid results
- Properly formatted using bash script files (no quoting issues)
- Status: PEND (waiting for GPU slots)

**Valid results so far (3/35):**
- gen_SUSTechGAN: mIoU=63.48 | gen_VisualCloze: mIoU=63.27 | gen_step1x_v1p2: mIoU=62.85

### Cityscapes Images per Method
| Method | Cityscapes Count | Total |
|--------|-----------------|-------|
| automold | 38,675 | 134,375 |
| CNetSeg, CUT, cycleGAN, cyclediffusion, Img2Img, IP2P, stargan_v2, SUSTechGAN, UniControl, VisualCloze | 17,850 each | varies |
| Qwen-Image-Edit | 17,816 | 100,650 |
| TSIT | 17,850 | 209,250 |
| step1x_v1p2 | 17,738 | 122,813 |
| Attribute_Hallucination, augmenters | 14,875 each | varies |
| step1x_new | 11,053 | 97,761 |
| albumentations_weather | 8,925 | 104,625 |
| Weather_Effect_Generator | 8,185 | 90,364 |
| flux_kontext | 5,185 | 94,142 |

### 📊 Weights Directory
```
/scratch/aaa_exchange/AWARE/WEIGHTS_CITYSCAPES_GEN/
├── baseline/cityscapes/segformer_mit-b3/                   ✅ trained, re-testing
├── std_autoaugment/cityscapes/segformer_mit-b3/            ✅ trained, re-testing
├── std_cutmix/cityscapes/segformer_mit-b3/                 ✅ trained, re-testing
├── std_mixup/cityscapes/segformer_mit-b3/                  ✅ trained, re-testing
├── std_randaugment/cityscapes/segformer_mit-b3/            ✅ trained, re-testing
├── gen_albumentations_weather/.../segformer_ratio0p50/     ✅ trained, re-testing
├── gen_automold/.../segformer_ratio0p50/                   ✅ trained, re-testing
├── gen_flux_kontext/.../segformer_ratio0p50/               ✅ trained, re-testing
├── gen_step1x_new/.../segformer_ratio0p50/                 ✅ trained, re-testing
├── gen_SUSTechGAN/.../segformer_ratio0p50/                 ✅ trained, re-testing
├── gen_step1x_v1p2/.../segformer_ratio0p50/               ✅ trained, re-testing
├── gen_VisualCloze/.../segformer_ratio0p50/                ✅ trained, re-testing
├── gen_{7 others}/cityscapes/segformer_ratio0p50/         🔄 training (7 running)
├── gen_*/cityscapes/{pspnet,segnext,mask2former}_ratio0p50/ ⏳ pending (72 jobs)
```

### 📋 Next Steps for Cityscapes-Gen Stage
1. **Verify re-test results** - Confirm 12 re-test jobs produce valid mIoU for Cityscapes
2. **Extend tracker scripts** - Add `--stage cityscapes-gen` to `update_training_tracker.py` and `update_testing_tracker.py`
   - New WEIGHTS_ROOT: `/scratch/aaa_exchange/AWARE/WEIGHTS_CITYSCAPES_GEN`
   - Dataset: `['cityscapes']` (single dataset, not the usual 4)
   - Models: same 4 models (pspnet, segformer, segnext, mask2former) with `_ratio0p50` for gen strategies
   - Max iters: 20,000 (from `batch_training_submission.py`)
   - Output files: `TRAINING_TRACKER_CITYSCAPES_GEN.md`, `TESTING_TRACKER_CITYSCAPES_GEN.md`, `TRAINING_COVERAGE_CITYSCAPES_GEN.md`, `TESTING_COVERAGE_CITYSCAPES_GEN.md`
   - Strategy lists: same gen (21) + std (baseline + 4 augmentation)
2. **Monitor 91 running/queued cityscapes-gen jobs** - Check for early failures
3. **Re-test future completed trainings** - The 79 remaining jobs have old buggy test scripts; run `bash /tmp/retest_cityscapes.sh` (or similar) after each batch completes
4. **Monitor 7 running segformer jobs** - ETA ~1-3 hours for remaining
5. **After training completes** - ACDC cross-domain tests are auto-included and working
6. **Analyze results** - Compare gen_ vs baseline/std on Cityscapes val + ACDC

### ✅ Completed Analysis (2026-02-08)
- Full code review of `update_training_tracker.py` (1363 lines) and `update_testing_tracker.py` (1299 lines)
- Documented structure, functions, stage handling, output files, constants, directory traversal, and markdown format
- Identified all changes needed for cityscapes-gen stage extension

---

## 🆕 Training Progress Update (2026-02-04)

### 📊 Training Status Summary

| Metric | Stage 1 | Stage 2 |
|--------|---------|---------|
| Configs Complete | 4/111 (3.6%) | 3/111 (2.7%) |
| **Models Complete** | **116/444 (26.1%)** | **45/444 (10.1%)** |
| Models Running | 18 | 1 |
| Models Pending | 179 | 399 |
| Models Failed | 135 | 3 |

### 🔄 Baseline Jobs (2026-02-04)
All missing baseline Mask2Former jobs submitted and moved to top of queue:

| Job ID | Stage | Dataset | Model | Status |
|--------|-------|---------|-------|--------|
| 1167137 | 1 | BDD10k | Mask2Former | RUN |
| 1187813 | 1 | IDD-AW | Mask2Former | PEND (top) |
| 1187814 | 1 | MapillaryVistas | Mask2Former | PEND (top) |
| 1187815 | 1 | OUTSIDE15k | Mask2Former | PEND (top) |
| 1187816 | 2 | BDD10k | Mask2Former | PEND (top) |
| 1187817 | 2 | IDD-AW | Mask2Former | PEND (top) |
| 1187818 | 2 | MapillaryVistas | Mask2Former | PEND (top) |
| 1187819 | 2 | OUTSIDE15k | Mask2Former | PEND (top) |

### � Planned Next Steps
1. **Monitor cityscapes-gen jobs** - Check for early failures, verify checkpoints
2. **Cross-domain testing** - After training completes, run ACDC cross-domain tests (auto-included in job scripts)
3. **Analyze results** - Compare gen_ vs baseline/std strategies on Cityscapes val + ACDC
4. **Stage 1/2 completion** - Continue monitoring remaining Stage 1/2 training jobs

### 📊 Current Queue Status (2026-02-08 20:40)

| Category | Running | Pending | Total |
|----------|---------|---------|-------|
| Stage 1 training | 2 | 0 | 2 |
| Cityscapes-gen training | 5 | 55 | 60 |
| Cityscapes-gen testing | 0 | 28 | 28 |
| Stage 1 gen_ restarts | 0 | 28 | 28 |
| Stage 1 test jobs | 0 | 10 | 10 |
| **Total** | **7** | **121** | **128** |

---

## 🛠️ Script Improvements (2026-02-04)

### ✅ Training Tracker Improvements
- **Reading target iterations from config**: Now reads `max_iters` from `training_config.py` instead of hardcoded 80000
- **Individual model counts**: Shows per-model completion status (not just configuration-level)
- **Default iterations**: 15,000 for new training regime (was 80,000)

**Updated files:**
- `scripts/update_training_tracker.py`
- `scripts/generate_baseline_overview.py`

---

## 🆕 Mask2Former Swin-B Integration (2026-02-03)

### ✅ Mask2Former Added to Training Pipeline
- **Model**: Mask2Former with Swin-B backbone (22k pretrained)
- **Configuration**: batch_size=8, max_iters=10,000, lr=0.0004 (4x scaled)
- **GPU Mode**: `exclusive_process` for memory-intensive training
- **Pretrained Weights**: `/scratch/aaa_exchange/AWARE/pretrained/swin/swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth`

### 🐛 Fixed: Validation Shape Mismatch Bug (2026-02-03)
- **Issue**: Mask2Former Stage 1 jobs failing with IndexError: shape [1024,1024] vs [512,512]
- **Root Cause**: Config generated before commit `8a4948c` that fixed BDD10k/IDD-AW resize to 512x512
- **Fix**: Cleaned old configs and resubmitted jobs with correct resize scale
- **Commit**: `8a4948c` - Extend validation resize fix to BDD10k and IDD-AW

### 🔄 Active Stage 1 Baseline Jobs (6 jobs)
| Job ID | Job Name | Status |
|--------|----------|--------|
| 1093758 | s1_baseline_bdd10k_mask2former | PEND (top) |
| 1093759 | s1_baseline_iddaw_mask2former | PEND (top) |
| 1093760 | s1_baseline_mapillaryvistas_mask2former | PEND (top) |
| 1093761 | s1_baseline_outside15k_mask2former | PEND (top) |
| 1093764 | s1_baseline_bdd10k_segformer | PEND (top) |
| 1093765 | s1_baseline_iddaw_pspnet | PEND (top) |

### 📈 Stage 1 Baseline Status (2026-02-04)

| Dataset | PSPNet | SegFormer | SegNeXt | Mask2Former |
|---------|:------:|:---------:|:-------:|:-----------:|
| BDD10k | ✅ 40.7% | ✅ 46.3% | ✅ 44.9% | 🔄 |
| IDD-AW | ✅ 26.2% | ✅ 34.0% | ✅ 35.1% | 🔄 |
| MapillaryVistas | ✅ 29.0% | ✅ 27.7% | ✅ 34.6% | 🔄 |
| OUTSIDE15k | ✅ 39.5% | ✅ 38.9% | ✅ 40.4% | 🔄 |

**Legend:** ✅ Complete | 🔄 Running/Submitted | ⏳ Pending | ❌ Failed

**Best mIoU per Dataset (Stage 1 baseline):**
- **BDD10k**: SegFormer 46.25%
- **IDD-AW**: SegNeXt 35.09%
- **MapillaryVistas**: SegNeXt 34.64%
- **OUTSIDE15k**: SegNeXt 40.36%

### 📈 Cluster Status
- **Running Jobs**: 1
- **Pending Jobs**: 346
- **Total Jobs**: 347

### 🔄 Active Stage 2 Jobs
Stage 2 Mask2Former baseline jobs pending (BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k)

---

## 🆕 Resume Training Feature (2026-02-02)

Added `--resume` flag to batch_training_submission.py to resume interrupted training:

```bash
# Dry run to see what would be resumed
python scripts/batch_training_submission.py --stage 1 --resume --dry-run

# Resume specific strategies
python scripts/batch_training_submission.py --stage 1 --resume --strategies gen_step1x_new -y
```

**How it works:**
- Finds the latest `iter_*.pth` checkpoint in the weights directory
- Passes `--resume-from` to unified_training.py to continue training
- Skips jobs that are already complete (have final checkpoint)
- Jobs without checkpoints start fresh

---

## 🔧 URGENT: Recent Actions (2026-02-02)

### ✅ Fixed MapillaryVistas/OUTSIDE15k Validation Bug
- **Issue**: Shape mismatch during validation - mask [1024, 1024] vs prediction [512, 512]
- **Root Cause**: Validation pipeline resized 512x512 images to (2048, 1024), causing mismatch
- **Fix**: Changed validation resize to (512, 512) for MapillaryVistas and OUTSIDE15k
- **Commit**: `81053c8` - [unified_training_config.py](unified_training_config.py#L2467-L2470)
- **Jobs Resubmitted**: 11 MapillaryVistas + 1 OUTSIDE15k (all gen_* strategies)

### ✅ Priority 1-3 Coverage Jobs Submitted (26 jobs)

**mIoU Analysis Results (93 total results):**
- **Stage 1 Gaps**: IDD-AW (only 3 baseline), OUTSIDE15k (only 3 baseline), MapillaryVistas (being addressed)
- **Stage 2 Gap**: ALL datasets only have baseline (NO gen_* or std_*)

**Jobs Submitted (2026-02-02 08:26-08:30):**

| Priority | Dataset | Strategies | Jobs | Status |
|----------|---------|------------|------|--------|
| **1** | IDD-AW | gen_* (step1x_new, flux_kontext, albumentations_weather) | 9 | 🔄 2 RUN, 7 PEND |
| **2** | OUTSIDE15k | gen_* (same 3 strategies) | 9 | ⏳ PEND |
| **3a** | IDD-AW | std_* (autoaugment, cutmix, mixup, randaugment) | 4 | ⏳ PEND |
| **3b** | OUTSIDE15k | std_* (same 4 strategies) | 4 | ⏳ PEND |

**Models Used**: segnext_mscan-b, segformer_mit-b3, pspnet_r50

**Impact:**
- IDD-AW: 3 configs → 16 configs (+433% coverage)
- OUTSIDE15k: 3 configs → 16 configs (+433% coverage)

**Current Cluster Status** (08:30):
- Total jobs: 38
- Running: 13 (cluster fully utilized)
- Pending: 24

### 🔜 Remaining Priorities (Not Yet Submitted)

**Priority 4**: MapillaryVistas std_* strategies (~6 jobs)
- Wait for current 11 gen_* jobs to complete

**Priority 5**: Stage 2 gen_* experiments (~12 jobs)
- Test top 3 gen strategies (step1x_new, flux_kontext, albumentations_weather)
- Datasets: BDD10k, IDD-AW
- Models: segnext_mscan-b, segformer_mit-b3

---

## �🔧 Current Status

### 🎉 PIPELINE FIX APPLIED & VERIFIED!

The critical pipeline bug has been fixed and verified through Cityscapes replication:

| Model | Before Fix | After Fix (Cityscapes) | Expected |
|-------|-----------|------------------------|----------|
| SegFormer MIT-B3 | ~45% | **79.98%** ✅ | ~79-80% |
| SegNeXt MSCAN-B | ~45% | **81.22%** ✅ | ~77-79% |
| DeepLabV3+ R50 | ~38% | 58.02% | ~77% (needs 769x769) |

**Fix summary:** Added `RandomResize(0.5-2.0x)` before `RandomCrop` - see [PIPELINE_COMPARISON_ANALYSIS.md](PIPELINE_COMPARISON_ANALYSIS.md)

### 🔄 NEW TRAINING DEFAULTS (2026-02-01)

**Updated default training parameters based on convergence analysis:**

| Parameter | Old Default | New Default | Rationale |
|-----------|-------------|-------------|-----------|
| `max_iters` (seg) | 80,000 | **15,000** | ~98% of final mIoU, 80% faster |
| `max_iters` (det) | 40,000 | **10,000** | ~98% of final mAP, 75% faster |
| `checkpoint_interval` | 5,000 | **2,000** | 8 checkpoints for detailed tracking |
| `eval_interval` | 5,000 | **2,000** | Aligned with checkpoints |

**Convergence analysis from Cityscapes replication (BS=2, 160k iters):**
- 8k iters: ~87.5% of final mIoU
- 16k iters: ~92.5% of final mIoU
- 32k iters: ~95% of final mIoU
- 64k iters: ~97.5% of final mIoU
- 120k iters: ~98% of final mIoU ← **15k iters at BS=16 equivalent**

### 🏃 Active Jobs: Stage 1 Full Resubmission (15k iters)

All Stage 1 jobs have been resubmitted with optimized parameters:

| Status | Count |
|--------|-------|
| Running | 14 |
| Pending | 556 |
| Skipped (existing) | 33 |
| **Total** | **570** |

**Command used:**
```bash
python scripts/batch_training_submission.py --stage 1 --max-iters 15000 \
    --checkpoint-interval 2000 --eval-interval 2000 -y
```

### 🏃 Active Jobs: PROVE Cityscapes Replication (BS16, 20k iters)

Testing PROVE default parameters (batch size 16) on Cityscapes with 20k iterations (~320k samples):

| Job ID | Model | Status | Max Iters | Ckpt/Eval Interval | Output Directory |
|--------|-------|--------|-----------|-------------------|------------------|
| 1006852 | SegFormer MIT-B3 | ⏳ PEND | 20k | 2k | `WEIGHTS_CITYSCAPES/baseline/cityscapes/segformer_mit-b3` |
| 1006853 | SegNeXt MSCAN-B | ⏳ PEND | 20k | 2k | `WEIGHTS_CITYSCAPES/baseline/cityscapes/segnext_mscan-b` |

### ✅ NEW Features Added (2026-02-01)

1. **CLI arguments for checkpoint/eval intervals:**
   - `--checkpoint-interval N`: Save checkpoint every N iterations
   - `--eval-interval N`: Run validation every N iterations

2. **Auto LR scaling:** When `--batch-size` is specified, `lr_scale_factor` auto-adjusts:
   - `lr = base_lr × (batch_size / 2)` (linear scaling rule)

3. **Validation pipeline fix:** Uses full resolution instead of 512x512

### ✅ Cross-Domain Testing Script (2026-02-01)

Test Cityscapes replication models on ACDC (foggy, night, rainy, snowy):

```bash
python scripts/test_cityscapes_replication_on_acdc.py --dry-run      # Preview
python scripts/test_cityscapes_replication_on_acdc.py --submit-jobs  # Submit
```
| 1004640 | OUTSIDE15k | SegFormer B3 | ⏳ PEND | - |
| 1004641 | OUTSIDE15k | SegNeXt MSCAN-B | ⏳ PEND | - |
| 1004642 | OUTSIDE15k | HRNet HR48 | ⏳ PEND | - |

**⚠️ BDD10k Permission Issue:**
- Directories `bdd10k/segformer_mit-b3` and `bdd10k/hrnet_hr48` owned by mima2416 with 755 permissions
- Created alternative directories with `_fixed` suffix, need to resubmit jobs

**Available Models (5):**
- DeepLabV3+ R50, PSPNet R50 (CNN)
- SegFormer MIT-B3, SegNeXt MSCAN-B (Transformer)
- HRNet HR48 (High-Resolution Net)

**Cityscapes Replication (Verified):**
| Model | mIoU | Expected | Status |
|-------|------|----------|--------|
| SegFormer MIT-B3 | **79.98%** | ~79% | ✅ Complete |
| SegNeXt MSCAN-B | **81.22%** | ~77% | ✅ Complete |
| HRNet HR48 | 65.67% | ~78% (needs 512x1024) | ✅ Complete |
| DeepLabV3+ R50 | 58.02% | ~77% (needs 769x769) | ✅ Complete |
| PSPNet R50 | 57.64% | ~76% (needs 769x769) | ✅ Complete |
| OCRNet HR48 | 49.25% | ~79% (needs 512x1024) | ⚠️ Config issues |

**Cityscapes Proper Crop Size Jobs (Complete):**
| Job ID | Model | Crop Size | Final mIoU | Expected |
|--------|-------|-----------|------------|----------|
| 1004205 | DeepLabV3+ R50 | 769x769 | **66.57%** | 79.6% |
| 1004206 | PSPNet R50 | 769x769 | **72.50%** | 78.5% |
| 1004207 | HRNet HR48 | 512x1024 | 67.65% (ongoing) | 80.6% |
| 1004208 | OCRNet HR48 | 512x1024 | 56.00% (issues) | 81.3% |

**Note:** PSPNet achieved 72.5% with 769x769 crop (vs 57.6% with 512x512), confirming crop size matters for CNNs.

### Current Training Configuration (2026-01-31)
| Setting | Value |
|---------|-------|
| **Pipeline** | ✅ RandomResize(0.5-2.0x) → RandomCrop(512,512) → RandomFlip → PhotoMetricDistortion |
| **Batch Size** | 16 |
| **Max Iterations** | 80,000 |
| **Warmup Iterations** | 1,000 |
| **Loss Function** | CrossEntropyLoss |
| **Checkpoint Interval** | 5,000 |
| **Eval Interval** | 5,000 |
| **Early Stop Patience** | 5 validations (25k iters) |
| **LR Scale Factor** | 8.0 (batch_size=16 / base=2) |

---

## 🚨 CRITICAL: Pipeline Bug FIXED (2026-01-31)

### Issue (RESOLVED)
Training pipeline was **missing multi-scale augmentation** - all strategies performed within ~0.91% of each other (44.78-45.69% mIoU).

### Root Cause (FIXED)
```python
# BEFORE (WRONG):
dict(type='Resize', scale=(512, 512), keep_ratio=False),  # Fixed size
# THEN for non-baseline (no effect since same size):
dict(type='RandomCrop', crop_size=(512, 512), ...)

# AFTER (CORRECT - now applied):
dict(type='RandomResize', scale=(2048, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
dict(type='RandomFlip', prob=0.5),
dict(type='PhotoMetricDistortion'),
```

### Verification (Complete)
Cityscapes replication with correct pipeline achieved expected results:
- SegFormer: **79.33%** mIoU (expected ~79%)
- SegNeXt: **79.81%** mIoU (expected ~77%)

### Files Modified
- [unified_training_config.py](unified_training_config.py) - All `_build_*_training_pipeline` methods fixed
- [PIPELINE_COMPARISON_ANALYSIS.md](PIPELINE_COMPARISON_ANALYSIS.md) - Detailed analysis

---

## ✅ Recently Completed

### 2026-01-31
- [x] ✅ **CRITICAL FIX:** Added RandomResize(0.5-2.0x) to all training pipelines
- [x] ✅ Verified fix via Cityscapes replication (79%+ mIoU for transformers)
- [x] ✅ **Submitted 12 Stage 1 baseline jobs** (SegFormer, SegNeXt, HRNet × 4 datasets)
- [x] ✅ Cityscapes replication: SegFormer achieved **79.98%**, SegNeXt achieved **81.22%**
- [x] ✅ Added SegNeXt MSCAN-B to available models
- [x] ✅ Updated batch submission scripts with SegNeXt
- [x] ✅ Created PIPELINE_COMPARISON_ANALYSIS.md
- [x] ✅ Collected Cityscapes replication results (all 6 models completed)

### 2026-01-30
- [x] Diagnosed pipeline bug: missing RandomResize causing all strategies to converge
- [x] Identified crop size requirements for CNN models

---

## 📋 Next Steps

### Immediate (Once current jobs complete)
1. **Monitor training progress** - Jobs 1004334-1004336
2. **Evaluate fixed pipeline** - Compare new baseline results with old ~45% results
3. **Re-run all strategies** if baseline shows improvement

### Analysis Tasks
- [ ] Compare before/after mIoU on BDD10k
- [ ] Re-run augmentation strategies (cutmix, mixup, etc.) with fixed pipeline
- [ ] Re-run generative strategies (CycleGAN, IP2P, etc.) with fixed pipeline

### Documentation
- [ ] Update analysis scripts with new results
- [ ] Create comparison table: old pipeline vs new pipeline

---

## ⚠️ Lovasz Loss Findings (2026-01-30)

### Investigation Summary
Analyzed chge7185's Stage 2 training (baseline_bdd10k_deeplabv3plus with Lovasz loss)

### Validation mIoU Progression
```
Iter    mIoU    Best?
5k      35.43   
10k     35.86   ↑ best
15k     34.68   
20k     36.35   ↑ best
25k     35.18   
30k     34.09   
35k     37.56   ↑ best
40k     37.55   
45k     35.81   
50k     33.49   
55k     39.15   ↑ best
60k     35.61   
65k     33.94   
70k     39.69   ↑ best (current best)
75k     39.63   
```

### Key Observations
1. **Oscillating mIoU (33-40%)** - No clear convergence trend
2. **Early stopping not triggered** - keeps finding new bests intermittently
3. **Per-class issues:** 8 classes at 0% IoU (wall, rider, bus, train, motorcycle, bicycle)
4. **Historical CE baseline:** 44-48% mIoU (more stable)

### Possible Causes
- `classes='present'` in Lovasz loss (uneven gradients for rare classes)
- High LR with batch size 16
- Multi-domain training complexity (Stage 2)

### Recommendation
Consider comparing with CrossEntropy loss baseline to determine which is more stable.

### Optimizer & Scheduler
| Model Type | Optimizer | Base LR | Scaled LR (BS=16) |
|------------|-----------|---------|-------------------|
| DeepLabV3+/PSPNet (CNN) | SGD (momentum=0.9) | 0.01 | 0.08 |
| SegFormer (Transformer) | AdamW | 0.00006 | 0.00048 |

---

## ✅ Recently Completed

### 2026-01-31
- [x] ✅ **Cityscapes Replication Setup** - All 6 models training
  - Fixed paths from mima2416 to chge7185
  - Changed from 4-GPU to single-GPU execution
  - Downloaded pretrained weights locally (cluster nodes have no internet)
  - Created `prepare_cityscapes.py` to generate labelTrainIds files (5000 files converted)
  - Fixed duplicate job submissions
  - All jobs now running on makalu94/makalu95

### 2026-01-30
- [x] ✅ Refactored training loss CLI to single `--aux-loss` across training, batch submission, locks, and docs
- [x] ✅ Validated `--aux-loss` with config-only checks (focal, lovasz, boundary all working)
- [x] ✅ Updated README.md, INSTRUCTIONS.md, TODO.md with aux-loss documentation

### 2026-01-29
- [x] **Cleared all old weights** (996 GB removed)
  - WEIGHTS/: 955 GB cleared
  - WEIGHTS_STAGE_2/: 41 GB cleared
- [x] **Killed all running/pending jobs** (54 jobs terminated)
- [x] **Updated training configuration:**
  - batch_size: 8 → 16
  - max_iters: 10,000 → 80,000
  - warmup_iters: 500 → 1,000
  - lr_scale_factor: 4.0 → 8.0
- [x] **Added best checkpoint saving logic:**
  - save_best='val/mIoU' with rule='greater'
  - max_keep_ckpts=-1 (keep all)
- [x] **Aligned checkpoint/eval intervals:**
  - Both at 10,000 iterations for proper best checkpoint selection
- [x] **Submitted Stage 1 baseline jobs with Lovasz loss:**
  - 12 jobs: 4 datasets × 3 models
  - Job IDs: 895295-895306

---

## 📋 Evaluation Completion Checklist

### Phase 1: Clear Old Data ✅
- [x] Delete old WEIGHTS/ (955 GB cleared)
- [x] Delete old WEIGHTS_STAGE_2/ (41 GB cleared)
- [x] Kill all old jobs (54 jobs terminated)

### Phase 2: Stage 1 Training (Clear Day Only) 🔄 IN PROGRESS

**New Configuration: batch_size=16, max_iters=80,000, warmup=1,000, Lovasz loss**

| Strategy | Progress | Status |
|----------|----------|--------|
| baseline | 0/12 (0%) | 🔄 12 jobs pending |
| std_* | 0/42 | ⏳ Pending |
| gen_* | 0/57+ | ⏳ Pending |

**Submitted Jobs (2026-01-29):**
- baseline_bdd10k_* (895295-895297)
- baseline_iddaw_* (895298-895300)
- baseline_mapillaryvistas_* (895301-895303)
- baseline_outside15k_* (895304-895306)

### Phase 3: Stage 2 Training (All Domains) 🔄 IN PROGRESS (chge7185)

| Model | Dataset | Progress | mIoU | Status |
|-------|---------|----------|------|--------|
| DeepLabV3+ | BDD10k | ~84% (67k/80k) | 39.69% | 🔄 Running |
| SegFormer | BDD10k | ~59% (47k/80k) | ~37% | 🔄 Running |
| PSPNet | BDD10k | ~37% (30k/80k) | ~36% | 🔄 Running |

**⚠️ Note:** Stage 2 mIoU is lower than Stage 1 because it trains on ALL weather conditions, including challenging night/rain domains (night domain: ~22.8% mIoU).

---

## 🎯 Proposed Next Steps

### ✅ NEW: Cityscapes Training Stage Added (2026-02-01)

Added support for direct Cityscapes training via `--stage cityscapes` in `batch_training_submission.py`:

```bash
# Dry run (preview)
python scripts/batch_training_submission.py --stage cityscapes --dry-run

# Submit all 5 models
python scripts/batch_training_submission.py --stage cityscapes -y
```

**Configuration:**
- **Dataset**: Cityscapes (2975 train / 500 val)
- **Data root**: `/scratch/aaa_exchange/AWARE/CITYSCAPES`
- **Iterations**: 160,000
- **Output**: `WEIGHTS_CITYSCAPES/baseline/cityscapes/{model}/`
- **Models**: All 5 (DeepLabV3+, PSPNet, SegFormer, SegNeXt, HRNet)

**Purpose:** Pipeline verification using standard benchmark results.

### 🚨 HIGH PRIORITY: Pipeline Bug Verification

1. **Wait for Cityscapes replication jobs (6 jobs pending)**
   - Work dirs: `/scratch/aaa_exchange/AWARE/CITYSCAPES_REPLICATION/`
   - Expected mIoU: 76-82% if pipeline is correct
   - Monitor: `bjobs -w | grep "cs_"` and `bpeek <job_id>`

2. **Analyze results when complete**
   - Compare achieved mIoU vs published benchmarks
   - If matching → **Pipeline bug confirmed**
   - If not matching → Investigate further (data, training config, etc.)

3. **Fix PROVE pipeline if bug confirmed**
   - Update [unified_training_config.py](unified_training_config.py)
   - Add RandomResize(ratio_range=(0.5, 2.0)) before RandomCrop
   - Re-run all training with fixed pipeline

### ⚠️ Decision Pending: Loss Function
Based on Lovasz instability findings, decide between:
- **Option A:** Add CrossEntropy comparison jobs (run both)
- **Option B:** Kill Lovasz jobs, switch to CrossEntropy only
- **Option C:** Let Lovasz jobs finish, analyze final results

### After Pipeline Fix Verified
3. **Submit std_* strategies** (after pipeline fix)
   - `python scripts/batch_training_submission.py --stage 1 --strategy-type std --aux-loss lovasz`

4. **Submit gen_* strategies**
   - `python scripts/batch_training_submission.py --stage 1 --strategy-type gen --aux-loss lovasz`

5. **Run testing on completed models**
   - `python scripts/auto_submit_tests.py --stage 1 --dry-run`

---

## 🎯 Strategy Lists

### STD_STRATEGIES (7)
| Strategy | Augmentation | Description |
|----------|--------------|-------------|
| baseline | None | Pure baseline, no augmentation |
| std_minimal | RandomCrop + RandomFlip | Geometric augmentation only |
| std_photometric_distort | PhotoMetricDistortion | Color augmentation only |
| std_autoaugment | AutoAugment | Policy-based augmentation |
| std_cutmix | CutMix | Cut-paste augmentation |
| std_mixup | MixUp | Linear interpolation |
| std_randaugment | RandAugment | Random augmentation policy |

### GEN_STRATEGIES (19)
| Rank | Strategy | CQS | FID | mIoU |
|------|----------|-----|-----|------|
| 1 | gen_cycleGAN | -0.78 | 92.65 | 46.76 |
| 2 | gen_flux_kontext | -0.66 | 80.30 | 40.01 |
| 3 | gen_step1x_new | -0.47 | 86.64 | 35.92 |
| 4 | gen_LANIT | -0.29 | 106.24 | 44.48 |
| 5 | gen_albumentations_weather | 0.07 | 123.94 | 43.99 |
| 6 | gen_automold | 0.16 | 121.12 | 33.50 |
| 7 | gen_step1x_v1p2 | 0.18 | 91.63 | 24.61 |
| 8 | gen_VisualCloze | 0.26 | 99.34 | 24.02 |
| 9 | gen_SUSTechGAN | 0.36 | 147.49 | 49.13 |
| 10 | gen_cyclediffusion | 0.39 | 138.77 | 33.18 |
| 11 | gen_IP2P | 0.41 | 114.22 | 28.52 |
| 12 | gen_Attribute_Hallucination | 0.64 | 117.95 | 21.60 |
| 13 | gen_UniControl | 0.73 | 114.90 | 22.51 |
| 14 | gen_CUT | 0.78 | 119.38 | 18.11 |
| 15 | gen_Img2Img | 0.84 | 120.25 | 15.04 |
| 16 | gen_Qwen_Image_Edit | 0.85 | 111.41 | 17.18 |
| 17 | gen_CNetSeg | 1.43 | 120.77 | 13.00 |
| 18 | gen_stargan_v2 | 1.52 | 100.28 | 4.84 |
| 19 | gen_Weather_Effect_Generator | - | - | - |

*Excluded: gen_EDICT, gen_StyleID (no generated images available)*

---

## 🚀 Batch Training Submission

\`\`\`bash
# ============================================
# ALWAYS DRY-RUN FIRST
# ============================================
python scripts/batch_training_submission.py --stage 1 --dry-run

# ============================================
# Submit by Strategy Type
# ============================================
# STD strategies only (84 jobs: 7 × 4 × 3)
python scripts/batch_training_submission.py --stage 1 --strategy-type std

# GEN strategies only (228 jobs: 19 × 4 × 3)
python scripts/batch_training_submission.py --stage 1 --strategy-type gen

# All strategies (312 jobs: 26 × 4 × 3)
python scripts/batch_training_submission.py --stage 1 --strategy-type all

# ============================================
# Submit with Limit
# ============================================
python scripts/batch_training_submission.py --stage 1 --limit 50

# ============================================
# Submit Specific Strategies
# ============================================
python scripts/batch_training_submission.py --stage 1 \\
    --strategies baseline std_minimal gen_cycleGAN

# ============================================
# Submit for Specific Dataset/Model
# ============================================
python scripts/batch_training_submission.py --stage 1 \\
    --datasets BDD10k --models deeplabv3plus_r50
\`\`\`

---

## 📁 File Organization

\`\`\`
/scratch/aaa_exchange/AWARE/
├── WEIGHTS/                    # Stage 1 outputs (DELETE & RETRAIN)
├── WEIGHTS_STAGE_2/            # Stage 2 outputs (DELETE & RETRAIN)  
├── WEIGHTS_RATIO_ABLATION/     # Ratio ablation study
├── WEIGHTS_EXTENDED/           # Extended training study
├── WEIGHTS_COMBINATIONS/       # Combination strategies
├── GENERATED_IMAGES/           # Synthetic images (KEEP)
└── training_locks/             # Lock files for multi-user safety
\`\`\`

---

## ⏱️ Time Estimates

| Phase | Jobs | Time/Job | Parallel | Total |
|-------|------|----------|----------|-------|
| Stage 1 Training | ~100 | ~24-48 hours | 6-10 | ~1-2 weeks |
| Stage 2 Training | ~60 | ~24-48 hours | 6-10 | ~3-5 days |
| Testing | ~160 | ~15 min | 20 | ~2 hours |
| **Total** | | | | **~2 weeks** |

**Note:** Training time increased due to batch_size=16, max_iters=80,000 (vs previous 10,000)

---

## 📝 Quick Commands

\`\`\`bash
# Job Management (LSF)
bjobs -w                          # List all jobs
bjobs -u mima2416 | grep RUN      # Running jobs only
bkill <job_id>                    # Kill specific job
bkill 0                           # Kill ALL jobs

# Testing
python fine_grained_test.py --config path/config.py \\
    --checkpoint path/iter_80000.pth --dataset BDD10k

# Auto-submit tests
python scripts/auto_submit_tests.py --dry-run

# Update trackers
python scripts/update_training_tracker.py --stage 1
python scripts/update_testing_tracker.py --stage 1
\`\`\`

---

## ⚠️ Important Notes

1. **Always dry-run before submitting batch jobs**
2. **Pre-flight checks automatically skip:**
   - Jobs with existing checkpoints
   - Jobs locked by another user
   - gen_* strategies without generated images
3. **File permissions set to 775/664 for multi-user access**
4. **Testing runs automatically after training completes**
