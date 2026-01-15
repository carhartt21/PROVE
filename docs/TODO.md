# TODO - Upcoming Tasks

*Last updated: 2026-01-15 (11:35)*

## Active Training/Testing Jobs

### IDD-AW Retraining Pipeline v4 (Jan 15, 2026)

**Status:** 🔄 In Progress (28 new jobs submitted + 1 running from v3)

**Issue Fixed:** Original rt3_ jobs failed due to train_script.py having hardcoded `iddaw_cd` paths after directory consolidation. Fixed by using `unified_training.py` directly.

**New Jobs (rt4_*):** 28 jobs with `-n 10` (multi-CPU)
- Job IDs: 9564069-9564096
- Each job: trains model → automatically runs fine_grained_test.py

**Still Running (rt3_*):** 1 job
- rt3_IP2P_iddaw_dlv3 (job 9561538) - uses gen_IP2P strategy

**Strategies (10 total):**
- baseline, gen_Attribute_Hallucination, gen_ControlNet_seg2image, gen_CUT, gen_InstructPix2Pix
- std_autoaugment, std_cutmix, std_mixup, std_photometric_distort, std_randaugment

**Models:** DeepLabV3+, PSPNet, SegFormer (29 total jobs)
| std_randaugment | 9561553 | 9561554 | 9561555 |

### Ratio Ablation Study (Jan 15, 2026)

**Status:** 🔄 Running (112 jobs with auto-test)

**UPDATED:** Jobs now include automatic testing after training completion.

| Rank | Strategy | Job Status |
|------|----------|------------|
| 🥇 | gen_cyclediffusion | ⏳ Needs verification |
| 🥈 | gen_step1x_new | ✅ 56 jobs running |
| 🥉 | gen_step1x_v1p2 | ✅ 56 jobs running |

**Note:** gen_stargan_v2 and gen_TSIT jobs are NOT needed (strategies 4 & 5 not required for analysis).

**Models:** PSPNet, SegFormer (DeepLabV3+ intentionally excluded)
**Datasets:** BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k
**Ratios:** 0.0, 0.125, 0.25, 0.375, 0.625, 0.75, 0.875

### std_minimal Strategy (Jan 14-15, 2026)

**Status:** ✅ Training Complete, ⏳ Tests Pending

Training completed for all 12 models (4 datasets × 3 architectures).
Test jobs need to be submitted.

---

## Pending Tasks

### High Priority

1. **✅ WEIGHTS Directory Consolidation (COMPLETED Jan 15)**
   
   **Summary:**
   - 44 directories renamed (safe renames)
   - 14 merge conflicts resolved (newer checkpoints kept)
   - 0 `iddaw_*` directories remaining
   - All IDD-AW dirs now use `idd-aw_*` naming convention
   
   **Current State:**
   - 28 `idd-aw_cd` directories (Stage 1)
   - 16 `idd-aw_ad` directories (Stage 2)
   - ~14 `*_old_backup` directories (can be deleted to reclaim space)
   
   **Scripts Created:**
   - `scripts/safe_rename_weights.sh`
   - `scripts/merge_weights_conflicts.sh`
   - `scripts/resolve_weights_conflicts.sh`
   
   **See:** [WEIGHTS_CONSOLIDATION_PLAN.md](WEIGHTS_CONSOLIDATION_PLAN.md) for full details.

2. **✅ Ratio Ablation Strategies - Submissions Complete**
   - gen_step1x_new: 56 jobs ✅
   - gen_step1x_v1p2: 56 jobs ✅
   - gen_stargan_v2 and gen_TSIT: NOT needed (strategies 4 & 5 not required)
   - gen_cyclediffusion: Needs verification

3. **Submit std_minimal Test Jobs**
   - Training completed, tests needed for: BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k
   - 12 test jobs (4 datasets × 3 architectures)

4. **Periodic Coverage Updates**
   - Run `python scripts/update_retraining_tracker.py` periodically
   - Updates: [TRAINING_COVERAGE.md](TRAINING_COVERAGE.md), [TESTING_COVERAGE.md](TESTING_COVERAGE.md)
   - Recommended frequency: After job batches complete

### Medium Priority

5. **Ratio Ablation Analysis**
   - Once training completes, run comprehensive analysis
   - Update ratio ablation figures

6. **Generate Final Leaderboard**
   - After all tests complete, regenerate strategy leaderboard

### Low Priority

7. **Documentation Cleanup**
   - Archive old scripts in archived_scripts_20260115/
   - Update README with recent changes

8. **Code Quality**
   - Review and clean up one-time fix scripts
   - Add unit tests for critical transforms

9. **✅ Mapillary Class Mapping Bug (FIXED Jan 15)**
   - **Issue:** Test reports showed wrong class names (e.g., "road" was showing IoU for class 0, which is actually "Bird")
   - **Root Cause:** `MAPILLARY_CLASSES` and `OUTSIDE15K_CLASSES` were not defined, so `get_dataset_config()` fell back to `CITYSCAPES_CLASSES` (19 classes)
   - **Fix:** Added proper class name lists in `fine_grained_test.py`:
     - `MAPILLARY_CLASSES`: 66 class names ordered by ID (Bird, Ground Animal, ..., Road@13, ..., Unlabeled)
     - `OUTSIDE15K_CLASSES`: 24 class names
   - **Files Changed:** [fine_grained_test.py](../fine_grained_test.py#L100-L130)

---

## Recently Completed

### Jan 15, 2026
- ✅ Fixed `submit_ratio_ablation.sh` - was calling non-existent `train_unified.sh`
- ✅ Identified rtfix_ retraining bug - missing `--domain-filter clear_day`
- ✅ Updated scripts to include automatic testing after training
- ✅ Created `scripts/retrain_iddaw_with_test.sh` (train+test in single job)
- ✅ Updated `submit_ratio_ablation.sh` to include auto-test
- ✅ Resubmitted IDD-AW jobs (30 train+test jobs)
- ✅ Resubmitted ratio ablation jobs (112 train+test jobs)
- ✅ Cancelled old jobs that lacked auto-test
- ✅ **WEIGHTS Directory Consolidation:**
  - Analyzed all `iddaw_*` directories - found they're valid Stage 2 checkpoints
  - Renamed 44 directories (safe renames)
  - Resolved 14 merge conflicts (kept newer checkpoints)
  - All IDD-AW dirs now use `idd-aw_*` naming convention
  - Created consolidation scripts: `safe_rename_weights.sh`, `merge_weights_conflicts.sh`, `resolve_weights_conflicts.sh`
  - Documented in [WEIGHTS_CONSOLIDATION_PLAN.md](WEIGHTS_CONSOLIDATION_PLAN.md)
- ✅ **IDD-AW rt3_ Jobs Fixed:**
  - 29/30 rt3_ jobs failed due to train_script.py hardcoded paths
  - Resubmitted 28 rt4_ jobs with `-n 10` (multi-CPU)
  - Job IDs: 9564069-9564096
  - 1 original rt3_ job (9561538) still running successfully
- ✅ **Mapillary Class Mapping Bug Fixed:**
  - Test reports were showing wrong class names (e.g., "road" for class 0 = Bird, "car" for class 13 = Road)
  - Added `MAPILLARY_CLASSES` (66 names) and `OUTSIDE15K_CLASSES` (24 names) to `fine_grained_test.py`
  - IoU values were always correct, only labels in reports were wrong
  - Verified fix: Road@13=92.29%, Building@17=91.82%, Car@55=1.03%
  - See Bug Reference section below for details

### Jan 14, 2026
- ✅ Identified IDD-AW training data corruption bug
- ✅ Cancelled 11 buggy IDD-AW test jobs
- ✅ Created retraining script: `scripts/retrain_iddaw_corrupted.sh` (⚠️ had bug)
- ✅ Updated `scripts/submit_ratio_ablation.sh` with correct top 5 strategies
- ✅ Killed incorrectly submitted DeepLabV3+ ratio ablation jobs
- ✅ Killed 168 jobs from incorrect top 5 strategies

### Jan 13, 2026
- ✅ Started ratio ablation training for SegFormer models
- ✅ Submitted extended training jobs

---

## Monitoring Commands

```bash
# Check all running/pending jobs
bjobs -w

# Check IDD-AW train+test jobs (v3)
bjobs -w | grep rt3_

# Check ratio ablation jobs
bjobs -w | grep ratio_

# Count jobs by type
bjobs -w | grep -c rt3_
bjobs -w | grep -c ratio_

# View recent job history
bhist -n 20

# Check specific job details
bjobs -l <JOB_ID>

# Monitor job output in real-time
tail -f logs/rt3_*_*.out
tail -f logs/ratio_*_*.log
```

---

## Bug Reference

### Mapillary Class Naming Bug (Fixed Jan 15)
- **Issue:** Test reports showed wrong class names for 66-class Mapillary tests
- **Root Cause:** `DATASET_LABEL_CONFIG['MapillaryVistas']['classes']` was `None`, falling back to 19-class `CITYSCAPES_CLASSES`
- **Effect:** Reports showed "road" for class 0 (actually "Bird"), "car" for class 13 (actually "Road")
- **Scope:** All previous Mapillary 66-class test reports have incorrect class labels (but correct IoU values)
- **Fix:** Added `MAPILLARY_CLASSES` (66 names) and `OUTSIDE15K_CLASSES` (24 names) in `fine_grained_test.py`
- **Files Changed:** [fine_grained_test.py](../fine_grained_test.py#L100-L130)
- **Note:** No recomputation needed - IoU values were always correct, only report labels were wrong

