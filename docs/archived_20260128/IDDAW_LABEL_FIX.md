# IDD-AW Label Transform Fix

## Issue Discovered
Date: 2025-01-10

### Problem
IDD-AW training was producing models where approximately half of the classes (8-18) showed `nan` in per-class validation metrics.

### Root Cause Analysis

1. **Observation**: Per-class validation showed classes 0-7 with reasonable IoU values (40-90%), but classes 8-18 all showed `nan`.

2. **Investigation**: Checked the actual label values in IDD-AW label files:
   - ACDC labels: `[0, 7, 8, 11, 12, 17, 20, 21, 22, 23, 24, 26, 31]` → **labelID format** (values > 18 present)
   - IDD-AW labels: `[0, 2, 5, 8, 10, 11, 12, 13, 17, 255]` → **trainID format** (all values ≤ 18)

3. **Bug Location**: In `unified_training_config.py`, IDD-AW was incorrectly included in `CITYSCAPES_LABEL_ID_DATASETS`:
   ```python
   # Line 1183 (multi-dataset)
   CITYSCAPES_LABEL_ID_DATASETS = {'ACDC', 'IDD-AW'}  # WRONG!
   
   # Line 1890 (single-dataset)
   CITYSCAPES_LABEL_ID_DATASETS = {'ACDC', 'Cityscapes', 'IDD-AW'}  # WRONG!
   ```

4. **Corruption Mechanism**: The `CityscapesLabelIdToTrainId` transform applied the wrong mapping:
   
   | IDD-AW Label (trainID) | Meaning | Mapped To | Result |
   |------------------------|---------|-----------|--------|
   | 0 | road | 255 (IGNORE) | ❌ Road becomes ignored |
   | 1 | sidewalk | 255 (IGNORE) | ❌ Sidewalk becomes ignored |
   | 7 | traffic sign | 0 (road) | ❌ Wrong class |
   | 8 | vegetation | 1 (sidewalk) | ❌ Wrong class |
   | ... | ... | ... | ... |
   
   The transform expects labelID format (where 7=road, 8=sidewalk, 11=building, etc.) but IDD-AW already has trainID format (where 0=road, 1=sidewalk, 2=building, etc.).

### Fix Applied

Removed `IDD-AW` from `CITYSCAPES_LABEL_ID_DATASETS` in both locations:

```python
# Line 1183 (multi-dataset) - FIXED
CITYSCAPES_LABEL_ID_DATASETS = {'ACDC'}  # Use Cityscapes label ID format (7=road, 8=sidewalk, etc.)
# NOTE: IDD-AW removed - its labels are already in trainID format (0-18), not labelID format (0-33)

# Line 1892 (single-dataset) - FIXED  
CITYSCAPES_LABEL_ID_DATASETS = {'ACDC', 'Cityscapes'}
# IDD-AW removed - its labels are already in trainID format (0-18), not labelID format (0-33)
```

Also updated comments at lines 1180, 1897, 1921 to correctly document which datasets need the transform.

### Label Format Summary

| Dataset | Label Format | Transform Needed |
|---------|-------------|------------------|
| ACDC | labelID (0-33) | ✅ CityscapesLabelIdToTrainId |
| Cityscapes | labelID (0-33) | ✅ CityscapesLabelIdToTrainId |
| IDD-AW | trainID (0-18) | ❌ No transform (already correct) |
| BDD10k | trainID (0-18) | ❌ No transform (already correct) |
| MapillaryVistas | RGB color-encoded | MapillaryRGBToClassId → MapillaryToTrainId |
| OUTSIDE15k | Native 24 classes | Outside15kLabelTransform |

### Retraining Required

All existing IDD-AW models were trained with corrupted labels and need to be retrained.

**Affected Models:**
- 13 strategies × 3 models = ~39 single-dataset IDD-AW models
- 3 multi-dataset models (MapillaryVistas+IDD-AW+BDD10k)
  - Note: ACDC was removed from multi-dataset training due to label format complexity

**Retraining Script:**
```bash
# Dry run (view what will be submitted)
python scripts/retrain_iddaw_fixed_labels.py --dry-run

# Submit all retraining jobs
python scripts/retrain_iddaw_fixed_labels.py
```

### Files Changed
- `unified_training_config.py`: Fixed CITYSCAPES_LABEL_ID_DATASETS and updated comments
- `unified_training.py`: Removed ACDC from multi-dataset configuration
- `scripts/retrain_iddaw_fixed_labels.py`: Created retraining script (new)
- `docs/IDDAW_LABEL_FIX.md`: This documentation (new)
