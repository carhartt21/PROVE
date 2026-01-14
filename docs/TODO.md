# TODO - Upcoming Tasks

*Last updated: 2026-01-14*

## Active Training/Testing Jobs

### IDD-AW Retraining Pipeline (Jan 14, 2026)

**Status:** 🔄 In Progress

11 IDD-AW models are being retrained due to a data corruption bug where `CityscapesLabelIdToTrainId` transform was incorrectly applied to IDD-AW training data (which already uses trainID format).

| Job ID | Job Name | Type | Status |
|--------|----------|------|--------|
| 9555793 | rtfix_baseline_iddaw_dlv3 | Training | PEND |
| 9555794 | rtfix_baseline_iddaw_segf | Training | PEND |
| 9555795 | rtfix_AttrHall_iddaw_dlv3 | Training | PEND |
| 9555796 | rtfix_AttrHall_iddaw_psp | Training | PEND |
| 9555797 | rtfix_CNetSeg_iddaw_dlv3 | Training | PEND |
| 9555798 | rtfix_CNetSeg_iddaw_psp | Training | PEND |
| 9555799 | rtfix_CUT_iddaw_dlv3 | Training | PEND |
| 9555800 | rtfix_CUT_iddaw_psp | Training | PEND |
| 9555801 | rtfix_CUT_iddaw_segf | Training | PEND |
| 9555802 | rtfix_IP2P_iddaw_dlv3 | Training | PEND |
| 9555803 | rtfix_IP2P_iddaw_psp | Training | PEND |

**Test jobs** (9555847-9555857) have dependencies and will start automatically after training completes.

### std_minimal Strategy Retraining (Jan 14, 2026)

**Status:** 🔄 Running

9 models being trained for the std_minimal strategy:
- BDD10k: DeepLabV3+, PSPNet, SegFormer (running)
- IDD-AW: DeepLabV3+, PSPNet, SegFormer (running)
- MapillaryVistas: DeepLabV3+, PSPNet, SegFormer (running)
- OUTSIDE15k: DeepLabV3+, PSPNet, SegFormer (running)

### Ratio Ablation Study (Jan 13-14, 2026)

**Status:** 🔄 Running (117 jobs)

**Correct Top 5 gen_* Strategies (from TESTING_TRACKER.md):**
| Rank | Strategy | Avg mIoU | Job Status |
|------|----------|----------|------------|
| 🥇 | gen_cyclediffusion | 55.8 | ⏳ Needs verification |
| 🥈 | gen_step1x_new | 52.8 | ✅ 56 jobs submitted |
| 🥉 | gen_step1x_v1p2 | 52.5 | ✅ 56 jobs submitted |
| 4 | gen_stargan_v2 | 52.4 | ❌ Not submitted |
| 5 | gen_TSIT | 52.0 | ❌ Not submitted |

**Models:** PSPNet, SegFormer (DeepLabV3+ intentionally excluded)
**Datasets:** BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k
**Ratios:** 0.0, 0.125, 0.25, 0.375, 0.625, 0.75, 0.875

**Currently running from Jan 13:**
- gen_cycleGAN SegFormer (3 datasets)
- gen_flux_kontext SegFormer (2 datasets)

---

## Pending Tasks

### High Priority

1. **Submit Remaining Ratio Ablation Strategies**
   - gen_stargan_v2 (56 jobs needed)
   - gen_TSIT (56 jobs needed)
   - Verify gen_cyclediffusion status
   - Command: `bash scripts/submit_ratio_ablation.sh --strategy <strategy_name>`

2. **Monitor IDD-AW Retraining**
   - Check job completion: `bjobs -w | grep rtfix`
   - Expected completion: ~12-24 hours
   - After completion, verify test results show mIoU > 30%

2. **Update TESTING_COVERAGE.md**
   - Once IDD-AW tests complete, update the coverage document
   - Remove the 11 buggy configurations from the list

3. **MapillaryVistas Missing Tests**
   - 8 test jobs were submitted (9554870-9554877)
   - Monitor for completion and verify results

### Medium Priority

4. **std_minimal Strategy Tests**
   - Once training completes, submit test jobs for all datasets
   - Need tests for: BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k

5. **Ratio Ablation Analysis**
   - Once SegFormer training completes, run comprehensive analysis
   - Update ratio ablation figures

6. **Generate Final Leaderboard**
   - After all tests complete, regenerate strategy leaderboard
   - Update preliminary results document

### Low Priority

7. **Documentation Cleanup**
   - Archive old scripts in archived_scripts_20260114/
   - Update README with recent changes

8. **Code Quality**
   - Review and clean up one-time fix scripts
   - Add unit tests for critical transforms

---

## Recently Completed

### Jan 14, 2026
- ✅ Identified IDD-AW training data corruption bug
- ✅ Cancelled 11 buggy IDD-AW test jobs
- ✅ Created retraining script: `scripts/retrain_iddaw_corrupted.sh`
- ✅ Submitted 11 training + 11 test jobs with dependencies
- ✅ Deleted old corrupted checkpoints
- ✅ Fixed model name issues in submit_missing_training.sh
- ✅ Updated `scripts/submit_ratio_ablation.sh` with correct top 5 strategies
- ✅ Submitted ratio ablation jobs for gen_step1x_new and gen_step1x_v1p2 (56 jobs each)
- ✅ Killed incorrectly submitted DeepLabV3+ ratio ablation jobs (DeepLabV3+ intentionally excluded)
- ✅ Killed 168 jobs from incorrect top 5 strategies (gen_albumentations_weather, gen_UniControl, gen_automold)

### Jan 13, 2026
- ✅ Started ratio ablation training for SegFormer models
- ✅ Submitted extended training jobs

---

## Monitoring Commands

```bash
# Check all running/pending jobs
bjobs -w

# Check IDD-AW retraining status
bjobs -w | grep rtfix

# Check std_minimal training status
bjobs -w | grep rt_std

# Check ratio ablation jobs
bjobs -w | grep ratio_

# Check test job status
bjobs -w | grep test_

# View recent job history
bhist -n 20

# Check specific job details
bjobs -l <JOB_ID>
```

---

## Bug Reference

### IDD-AW Label Corruption (Fixed)
- **Issue:** `CityscapesLabelIdToTrainId` transform was incorrectly applied to IDD-AW data
- **Root Cause:** Training config generated before fix (commit e268acf on Jan 11 12:12)
- **Effect:** Labels 8-18 were corrupted, models had 0% IoU on half the classes
- **Fix:** Updated `unified_training_config.py` to exclude transform for IDD-AW
- **Recovery:** Retraining all affected models with fixed config

See: [docs/IDDAW_LABEL_FIX.md](IDDAW_LABEL_FIX.md) for more details.
