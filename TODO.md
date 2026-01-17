# PROVE Project TODO List

**Last Updated:** 2026-01-23

## In Progress

### Extended Training Testing
- [x] Modified `submit_test_extended_training.sh` to use `fine_grained_test.py`
- [x] Added `find_dataset_dir()` function to handle directory naming variations
- [x] Submitted 504 test jobs (9 strategies × 4 datasets × 2 models × 7 iterations)
- [ ] Wait for all extended training test jobs to complete
- [ ] Analyze complete results with `analyze_extended_training.py`
- [ ] Generate final IEEE figures with full dataset

### Domain Adaptation Ablation (READY)
- [x] Updated `submit_domain_adaptation_ablation.sh` for correct weights paths
- [x] Fixed `conda activate` → `mamba activate`
- [x] Script now uses WEIGHTS_STAGE_2 for full dataset, WEIGHTS for clear_day
- [x] Added support for Top 15 augmentation strategies
- [x] Updated documentation with strategy tables
- [ ] Submit baseline jobs (8 available): `./scripts/submit_domain_adaptation_ablation.sh --all-full`
- [ ] Submit strategy jobs (76 available): `./scripts/submit_domain_adaptation_ablation.sh --all-strategies`
- [ ] Analyze results with `analyze_domain_adaptation_ablation.py`

### Analysis Scripts
- [x] Updated `analyze_extended_training.py` with Pattern 5 for new output format
- [x] Created `generate_ieee_figures_extended_training.py` for publication figures
- [x] Updated documentation in `docs/EXTENDED_TRAINING.md`
- [x] Updated `docs/TESTING_TRACKER.md` with extended training section
- [x] Updated `docs/DOMAIN_ADAPTATION_ABLATION.md` with correct paths

## Pending

### Publication
- [ ] Finalize extended training analysis figures
- [ ] Run statistical significance tests on extended training results
- [ ] Prepare tables for publication

### Job Monitoring
- [ ] Monitor extended training test jobs: `bjobs | grep test_ext`
- [ ] Check for failed jobs and resubmit if needed
- [ ] Document any strategies that fail consistently

## Completed

### Domain Adaptation Infrastructure (2026-01-23)
- [x] Script updated to use WEIGHTS_STAGE_2 for full dataset models
- [x] Script updated to use WEIGHTS for clear_day models
- [x] Documentation updated with checkpoint availability table
- [x] 8 jobs available: 5 full dataset + 3 clear_day

### Extended Training Infrastructure (2026-01-23)
- [x] All 9 strategies have complete checkpoints (40k-160k iterations)
- [x] Test command uses `fine_grained_test.py` for per-domain metrics
- [x] Directory naming variations handled (`bdd10k_ad`, `iddaw_ad`, etc.)
- [x] Test output stored in `test_results_iter_{N}/{timestamp}/results.json`
- [x] Analysis scripts support new output format
- [x] IEEE figure generation working with partial results (66 results)

### Documentation Updates (2026-01-23)
- [x] EXTENDED_TRAINING.md - Added testing and analysis sections
- [x] TESTING_TRACKER.md - Added extended training section
- [x] DOMAIN_ADAPTATION_ABLATION.md - Updated checkpoint paths

## Notes

### Domain Adaptation Checkpoint Availability

**Baseline Models:** 8 / 12 available
**Strategy Models:** 76 / 90 available  
**Total:** 84 configurations ready for evaluation

| Dataset | Model | Full (WEIGHTS_STAGE_2) | Clear_day (WEIGHTS) |
|---------|-------|:----------------------:|:-------------------:|
| BDD10k | pspnet_r50 | ✅ | ✅ |
| BDD10k | segformer_mit-b5 | ❌ | ✅ |
| IDD-AW | pspnet_r50 | ✅ | ❌ |
| IDD-AW | segformer_mit-b5 | ✅ | ❌ |
| MapillaryVistas | pspnet_r50 | ✅ | ✅ |
| MapillaryVistas | segformer_mit-b5 | ❌ | ✅ |

**Top 15 Strategies:** gen_cyclediffusion, gen_flux_kontext, gen_step1x_new, gen_step1x_v1p2, gen_stargan_v2, gen_cycleGAN, gen_automold, gen_albumentations_weather, gen_TSIT, gen_UniControl, std_randaugment, std_autoaugment, std_cutmix, std_mixup, photometric_distort

### Extended Training Test Configuration
- **Strategies:** gen_albumentations_weather, gen_automold, gen_cyclediffusion, gen_cycleGAN, gen_flux_kontext, gen_step1x_new, gen_TSIT, gen_UniControl, std_randaugment
- **Datasets:** BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k
- **Models:** segformer_mit-b5, pspnet_r50
- **Iterations:** 40000, 60000, 80000, 100000, 120000, 140000, 160000
- **Total Jobs:** 504

### Key Scripts
- `scripts/submit_test_extended_training.sh` - Submit extended training test jobs
- `scripts/submit_domain_adaptation_ablation.sh` - Submit domain adaptation jobs
- `analysis_scripts/analyze_extended_training.py` - Analyze extended training results
- `analysis_scripts/generate_ieee_figures_extended_training.py` - Generate IEEE figures
- `fine_grained_test.py` - Per-domain, per-class testing

### Output Directories
- Extended training tests: `{WEIGHTS_EXTENDED}/{strategy}/{dataset}/{model}_ratio0p50/test_results_iter_{N}/`
- Extended training figures: `result_figures/extended_training/{ieee/,preview/,data/}`
- Domain adaptation results: `{WEIGHTS}/domain_adaptation_ablation/`
# TODO - Upcoming Tasks

*Last updated: 2026-01-16 (16:20)*

## Current Job Status Summary

### Stage 1 (Clear Day Domain) - WEIGHTS directory
| Category | Running | Pending | Done | Total |
|----------|--------:|--------:|-----:|------:|
| Training | 21 | 0 | 101 | 107* |
| Testing | 2 | 5 | 328 | 335 |

*Note: Reduced from 111 to 107 strategies after removing std_minimal (not useful)

### Stage 2 (All Domains - Adverse Weather) - WEIGHTS_STAGE_2 directory
| Category | Running | Pending | Done | Partial | Total |
|----------|--------:|--------:|-----:|--------:|------:|
| Training | 0 | 77 | 4 | 26 | 107 |
| Testing | - | - | - | - | - |

**Note:** Stage 2 uses all 3 models (DeepLabV3+, PSPNet, SegFormer).
Totals reduced to 107 after removing std_minimal.
Partial indicates configurations where 1/3 or 2/3 models are complete.

### Ablation Studies
| Study | Owner | Running | Pending | Total |
|-------|-------|--------:|--------:|------:|
| Ratio Ablation | mima2416 | 13 | 54 | 112 |
| Extended Training | chge7185 | 6 | 454 | 460 |

---

## ✅ RESOLVED: Wrong num_classes Models (Jan 16)

**Issue:** Some models were trained with wrong number of classes.
**Resolution:** Removed incorrect checkpoints and submitted 21 retraining jobs.

| Dataset | Expected Classes | Actions Taken |
|---------|------------------|---------------|
| MapillaryVistas | 66 | Removed wrong models, retraining 6 jobs |
| OUTSIDE15k | 24 | Removed wrong models, retraining 15 jobs |

**Strategies Removed (permanently):**
- `std_minimal` - Not useful, removed from all datasets

**Strategies Retraining (21 jobs, IDs 9611966-9611986):**
- MapillaryVistas: gen_cyclediffusion (×3), gen_TSIT (×3)
- OUTSIDE15k: std_cutmix (×3), std_mixup (×3), gen_cyclediffusion (×3), gen_flux_kontext (×3), gen_TSIT (×3)

**Native Classes Default:** `unified_training.py` now uses native classes by default.
Use `--no-native-classes` to force Cityscapes 19 classes.

---

## 🔄 Stage 2 Retraining (MapillaryVistas/OUTSIDE15k)

**Issue:** All Stage 2 checkpoints for MapillaryVistas/OUTSIDE15k were trained with 19 classes.

**Actions Taken:**
- Deleted all Stage 2 MapillaryVistas and OUTSIDE15k checkpoints
- Removed `std_minimal` from Stage 2
- Submitted full retraining set for MapillaryVistas + OUTSIDE15k across all Stage 2 strategies (except `gen_EDICT`)

**Jobs Submitted:** 114 total (19 strategies × 2 datasets × 3 models)

**Status:** PENDING (BatchGPU queue)

**Validation Note (Stage 2 tests):**
- BDD10k classwise IoU looks normal; only rare classes (train/rider/motorcycle/bicycle) hit 0–0.01 IoU in some runs
- IDD-AW Stage 2 tests not available yet (no results.json to scan)

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
