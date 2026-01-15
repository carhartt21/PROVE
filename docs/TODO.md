# TODO - Upcoming Tasks

*Last updated: 2026-01-15 (16:30)*

## Active Training/Testing Jobs

### Stage 1 Training Coverage Status (Jan 15, 2026)

**Status:** 🔄 88.1% Complete, 11.0% Pending, 0.9% Running

| Status | Count | Percentage |
|--------|------:|----------:|
| ✅ Complete | 296 | 88.1% |
| 🔄 Running | 3 | 0.9% |
| ⏳ Pending | 37 | 11.0% |
| ⚠️ Missing | 0 | 0.0% |

**All configurations are now covered** (either complete, running, or pending).

### Stage 1 Testing

- Finish when trainings are complete

### IDD-AW Retraining Pipeline v4 (Jan 15, 2026)

**Status:** 🔄 In Progress (28 new jobs submitted + 1 running from v3)

**Issue Fixed:** Original rt3_ jobs failed due to train_script.py having hardcoded `iddaw_cd` paths after directory consolidation. Fixed by using `unified_training.py` directly.

**New Jobs (rt4_*):** 28 jobs with `-n 10` (multi-CPU)
- Job IDs: 9564069-9564096
- Each job: trains model → automatically runs fine_grained_test.py

**Still Running (rt3_*):** 1 job
- rt3_IP2P_iddaw_dlv3 (job 9561538) - uses gen_IP2P strategy

**Strategies (10 total):**
- baseline, gen_Attribute_Hallucination, gen_CNetSeg, gen_CUT, gen_IP2P
- std_autoaugment, std_cutmix, std_mixup, photometric_distort, std_randaugment

**Models:** DeepLabV3+, PSPNet, SegFormer (29 total jobs)

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

**Status:** 🔄 Training + Testing In Progress

**Training:**
- ✅ 8/12 models complete (BDD10k, IDD-AW × DeepLabV3+/PSPNet)
- 🔄 4 jobs pending (MapillaryVistas, OUTSIDE15k × DeepLabV3+ only, plus SegFormer variants)
  - Job 9566336: std_minimal_mapillary_cd
  - Job 9566337: std_minimal_outside15k_cd
  - Job 9566639: std_minimal_iddaw_segf  
  - Job 9566640: std_minimal_bdd10k_segf

**Testing:**
- 🔄 8 test jobs resubmitted (Jan 15, 16:16) - fixed missing --config/--output-dir
  - Job IDs: 9571762-9571769 (1 running, 7 pending)
  - Datasets: BDD10k_ad, IDD-AW_ad, MapillaryVistas_ad, OUTSIDE15k_ad

### Additional Missing Config Jobs (Jan 15, 2026)

**Status:** 🔄 Submitted (moved to top of queue)

| Job ID | Config | Status |
|--------|--------|--------|
| 9566641 | gen_step1x_new / IDD-AW / PSPNet | ⏳ PEND |
| 9566642 | gen_Weather_Effect_Generator / BDD10k / SegFormer | ⏳ PEND |

### Extended Training (320k) - User chge7185 (Jan 15, 2026)

**Status:** 🔄 Running (34 jobs total - 6 running, 28 pending, 22 completed)

**Owner:** User chge7185
**Duration:** 320k iterations (4× standard 80k)

**Strategies (6 total):**
| Strategy | Description |
|----------|-------------|
| gen_automold | Photorealistic weather augmentation |
| gen_cyclediffusion | Cycle-consistent diffusion |
| gen_flux_kontext | Flux context-aware generation |
| gen_step1x_new | Step1X new version |
| gen_step1x_v1p2 | Step1X version 1.2 |
| std_randaugment | Standard random augmentation |

**📊 Analysis Complete (Jan 15, 2026):**
- Full diminishing returns analysis completed
- **Report:** [docs/EXTENDED_TRAINING_ANALYSIS.md](EXTENDED_TRAINING_ANALYSIS.md)
- **Figures:** `result_figures/extended_training_analysis.png`, `result_figures/extended_training_by_strategy.png`

**Key Findings:**
| Training Phase | Avg mIoU Gain | % of Initial |
|----------------|---------------|--------------|
| 90k → 160k | +0.75 | 100% |
| 160k → 240k | +0.39 | 52% |
| 240k → 320k | +0.10 | 13% |

**Recommendation:** 160k training captures 75% of gains at 50% compute cost.

**Models:** PSPNet, SegFormer (DeepLabV3+ excluded)
**Datasets:** BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k
**Job ID Range:** 9529824, 9558712-9558746

---

## Pending Tasks

### High Priority

1. **Periodic Coverage Updates**
   - Run `python scripts/update_training_tracker.py` periodically
   - Updates: [TRAINING_TRACKER.md](TRAINING_TRACKER.md), [TESTING_COVERAGE.md](TESTING_COVERAGE.md)
   - Recommended frequency: After job batches complete

2. **gen_cyclediffusion Verification**
   - Needs verification before ratio ablation submission
   - Check if standard training completed correctly

3. **Domain Adaptation Ablation Study** ⭐ NEW
   - Cross-dataset domain generalization evaluation
   - See [DOMAIN_ADAPTATION_ABLATION.md](DOMAIN_ADAPTATION_ABLATION.md) for full details
   - **Scope:** 12 jobs total (6 full + 6 clear_day baseline)
   - **Available checkpoints:** 8/12 currently (4 missing)
   - **Missing checkpoint training:** Job 9565922 submitted (MapillaryVistas/segformer_ad) - moved to TOP
   - **Source datasets:** BDD10k, IDD-AW, MapillaryVistas
   - **Target domains:** Cityscapes (clear_day) + ACDC (foggy, night, rainy, snowy)
   - **Models:** PSPNet, SegFormer (DeepLabV3+ excluded)
   - **Status:** ⏳ Ready to submit (8 jobs available, 4 pending training)
   - **Script:** `./scripts/submit_domain_adaptation_ablation.sh --all`

4. **Augmentation Combination Training**
   - Combine the Top-3 std and gen strategies using Segformer and PSPNet
   - Decide which datasets to use (maybe not all to limit training time)

### Medium Priority

3. **Ratio Ablation Analysis**
   - Once training completes, run comprehensive analysis
   - Update ratio ablation figures

4. **Generate Final Leaderboard**
   - After all tests complete, regenerate strategy leaderboard

5. **Extended Training Analysis (Follow-up)**
   - Analyze remaining configs when training completes
   - Compare different strategies' convergence rates
   - Investigate why gen_TSIT saturates early (peak @ 190k)
   - Consider 160k as new default training length (cost-effective)

### Low Priority

5. **Documentation Cleanup**
   - Archive old scripts in archived_scripts_20260115/
   - Update README with recent changes

6. **Code Quality**
   - Review and clean up one-time fix scripts
   - Add unit tests for critical transforms

7. **Clean up WEIGHTS backup directories**
   - ~14 `*_old_backup` directories can be deleted to reclaim space

---

## Recently Completed

### Jan 15, 2026
- ✅ **Training Coverage Now 100% (no missing configs)**
  - Updated `scripts/update_training_tracker.py` to recognize all job patterns
  - Added rt3_/rt4_ job prefix support with abbreviation mapping
  - Added truncated dataset name support (iddaw, mapillary)
  - Coverage report: 296 complete + 37 pending + 3 running = 336 total
- ✅ **Submitted 4 Missing Configurations:**
  - std_minimal/IDD-AW/SegFormer (Job 9566639)
  - std_minimal/BDD10k/SegFormer (Job 9566640)
  - gen_step1x_new/IDD-AW/PSPNet (Job 9566641)
  - gen_Weather_Effect_Generator/BDD10k/SegFormer (Job 9566642)
- ✅ Cancelled duplicate std_minimal jobs (9566334, 9566335)
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
- ✅ **Ratio Ablation Submissions Complete:**
  - gen_step1x_new: 56 jobs submitted
  - gen_step1x_v1p2: 56 jobs submitted
  - gen_stargan_v2 and gen_TSIT: NOT needed (strategies 4 & 5 not required)
- ✅ **std_minimal Test Jobs Resubmitted:**
  - Original jobs failed (missing --config/--output-dir args)
  - Fixed script: `scripts/submit_std_minimal_tests.sh`
  - 8 new test jobs submitted (Job IDs: 9571762-9571769)
  - Datasets: BDD10k_ad, IDD-AW_ad, MapillaryVistas_ad, OUTSIDE15k_ad
- ✅ **README.md Updated:**
  - Added comprehensive `fine_grained_test.py` documentation
  - Added important training modes for `unified_training.py`
  - Added `--max-iters`, `--use-native-classes` to options table
- ✅ **Scripts Archived:**
  - Moved 9 one-time scripts to `archived_scripts_20260115/`
  - submit_iddaw_tests_*.sh, submit_missing_*.sh, submit_all_remaining_tests.sh
- ✅ **Created Reusable Submission Templates:**
  - `scripts/submit_training.sh` - Training job template with all options
  - `scripts/submit_testing.sh` - Testing job template with auto-detection
  - Both support --dry-run for previewing commands
- ✅ **Extended Training Diminishing Returns Analysis:**
  - Analyzed 22 configurations at 320k iterations
  - Created report: [docs/EXTENDED_TRAINING_ANALYSIS.md](EXTENDED_TRAINING_ANALYSIS.md)
  - Generated visualizations: `result_figures/extended_training_analysis.png`, `result_figures/extended_training_by_strategy.png`
  - Key finding: 160k captures 75% of gains at 50% compute cost
  - Updated [EXTENDED_TRAINING.md](EXTENDED_TRAINING.md) with analysis results

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

