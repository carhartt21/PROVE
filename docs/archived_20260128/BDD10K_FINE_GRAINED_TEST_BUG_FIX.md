# BDD10k Fine-Grained Test Bug Fix

## Summary

A critical bug was discovered in `fine_grained_test.py` that caused incorrect evaluation metrics for datasets that use Cityscapes trainID format (BDD10k, BDD100k, Cityscapes).

## Bug Description

### Root Cause
The old version of `fine_grained_test.py` (before commit `f6576fe`, Jan 10, 2026) **always** applied `CityscapesLabelIdToTrainId` conversion to ALL datasets, regardless of their actual label format.

### Impact on BDD10k
BDD10k labels are already in **trainID format** (0-18, 255). When incorrectly processed through the labelID-to-trainID conversion:

| Original trainID | Label Name | Converted To | Result |
|-----------------|------------|--------------|--------|
| 0 | road | 255 | IGNORED |
| 1 | sidewalk | 255 | IGNORED |
| 7 | traffic sign | 0 | Mislabeled as road |
| 8 | vegetation | 1 | Mislabeled as sidewalk |
| 13 | car | 255 | IGNORED |
| ... | ... | ... | ... |

### Symptoms in Test Results
The corrupted evaluation showed:
- Only 6 classes with `area_label > 0` (classes 0-5 or those that happened to map to valid trainIDs)
- Classes 6-18 (traffic light, traffic sign, vegetation, car, etc.) had `area_label = 0`
- mIoU values around 0.8-1.5% instead of expected 30-40%
- The model appeared to predict classes correctly (`area_pred > 0`) but no matching labels were found

### Example of Incorrect Results
```
Per-class for clear_day domain:
  road: area_label=1,108,509      # Actually trainID 7 (traffic sign)
  sidewalk: area_label=35,719,551 # Actually trainID 8 (vegetation)
  vegetation: area_label=0        # Lost to 255 (ignore)
  car: area_label=0               # Lost to 255 (ignore)
```

## Fix Applied

Commit `f6576fe` (Jan 10, 2026) introduced proper per-dataset label handling:

1. Added `DATASET_LABEL_CONFIG` dictionary with per-dataset configuration
2. Added `process_label_for_dataset()` function that:
   - For ACDC: Applies `CityscapesLabelIdToTrainId` (correct - ACDC uses labelIDs)
   - For IDD-AW: Already trainID format - NO conversion (fixed)
   - For BDD10k, BDD100k, Cityscapes: Already trainID format - NO conversion
   - For MapillaryVistas: RGB color decode to native 66 classes
   - For OUTSIDE15k: Keep native 24 classes

## Affected Test Results

Any `test_results_detailed` generated before Jan 10, 2026 for these datasets are INVALID:
- **BDD10k**: All 17+ models
- **BDD100k**: If tested (none found currently)
- **Cityscapes**: If tested with color-encoded labels

## Re-Testing Required

### Using the Re-Test Script

```bash
# Show what would be done (dry run)
python scripts/retest_bdd10k_fine_grained.py --dry-run

# Generate SLURM job scripts
python scripts/retest_bdd10k_fine_grained.py

# Submit all jobs
python scripts/retest_bdd10k_fine_grained.py --submit-all

# Submit for specific strategy only
python scripts/retest_bdd10k_fine_grained.py --submit-strategy baseline
```

### Manually Submitting Jobs

```bash
# Submit all generated jobs
for f in scripts/bdd10k_retest_jobs/*.sh; do sbatch "$f"; done
```

### Job Output Location
- Logs: `${AWARE_DATA_ROOT}/LOGS/retest_bdd10k/`
- Results: `<model_dir>/test_results_detailed_fixed/`

## Additional Issues Found

### Cityscapes Color Labels
The Cityscapes labels in `FINAL_SPLITS` are **color-encoded** (`gtFine_color.png`), not trainID format:
- Shape: (1024, 2048, 4) - RGBA
- Values are RGB colors, not class IDs

This requires RGB-to-trainID decoding similar to MapillaryVistas handling. This is a **separate issue** that needs to be addressed before running Cityscapes fine-grained tests.

### BDD100k JSON Labels
BDD100k labels are stored as JSON files, not PNG images. This requires different loading logic that isn't currently implemented in `fine_grained_test.py`.

## Verification

After re-testing, verify the results show:
1. mIoU values around 30-40% (matching training validation)
2. All 19 classes have `area_label > 0`
3. Per-class IoU values are reasonable:
   - road: ~90%
   - building: ~70%
   - car: ~80%
   - vegetation: ~85%

## Timeline
- **Jan 8, 2026**: Tests run with buggy code
- **Jan 10, 2026**: Bug fixed in commit `f6576fe`
- **Current**: Re-testing scripts created, jobs need to be submitted

## Related Files
- `/home/chge7185/repositories/PROVE/fine_grained_test.py` - Fixed evaluation script
- `/home/chge7185/repositories/PROVE/scripts/retest_bdd10k_fine_grained.py` - Re-test job generator
- `/home/chge7185/repositories/PROVE/scripts/bdd10k_retest_jobs/` - Generated SLURM job scripts
