# Pipeline Comparison: PROVE vs Standard mmsegmentation

**Date:** 2026-01-31
**Purpose:** Document the critical differences between PROVE training pipeline and standard mmsegmentation pipelines

---

## Summary of Critical Differences

| Aspect | PROVE Pipeline | Standard mmsegmentation | Impact |
|--------|---------------|------------------------|--------|
| **Scale Variation** | ❌ Fixed Resize THEN RandomCrop | ✅ RandomResize THEN RandomCrop | **CRITICAL** |
| **Scale Range** | N/A | 0.5x - 2.0x | Multi-scale learning |
| **Reference Scale** | 512x512 fixed | 2048x1024 base | Resolution consistency |
| **Baseline Augmentation** | ❌ None (just Resize) | ✅ Full pipeline | **CRITICAL** |

---

## Detailed Pipeline Comparison

### PROVE Training Pipeline (unified_training_config.py)

```python
# _build_trainid_training_pipeline (for BDD10k, IDD-AW)
pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='ReduceToSingleChannel'),
    dict(type='Resize', scale=(1024, 512), keep_ratio=True),  # FIXED size
]

# For non-baseline strategies:
if not is_baseline:
    pipeline.extend([
        dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),  # Same size as resize!
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion'),
    ])
```

**Problems:**
1. `Resize(1024, 512)` → `RandomCrop(512, 512)` = crops from a fixed-size image
2. No multi-scale variation - model only sees one scale
3. Baseline strategies get NO augmentation at all (just resize)

### Standard mmsegmentation Pipeline (Cityscapes replication)

```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(2048, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
```

**Why it works:**
1. `RandomResize(0.5-2.0x)` scales image to 50%-200% of original size
2. `RandomCrop(512, 512)` then extracts a random 512x512 patch
3. This creates **true multi-scale training** - model sees objects at many scales
4. ALL training runs (including baseline) get full augmentation pipeline

---

## Visual Explanation

### PROVE Pipeline (WRONG)
```
Original Image (1920x1080)
     ↓
Resize to (1024, 512)     # Fixed size
     ↓
RandomCrop (512, 512)     # Crops half the image - ALWAYS same scale
     ↓
Model sees objects at SINGLE SCALE
```

### Standard Pipeline (CORRECT)
```
Original Image (2048x1024)
     ↓
RandomResize (0.5x to 2.0x)   # Varies from (1024, 512) to (4096, 2048)
     ↓
RandomCrop (512, 512)         # Crops from DIFFERENT scaled images
     ↓
Model sees objects at MANY SCALES (0.5x to 2.0x variation)
```

---

## Impact on Results

### Observed Performance
| Model | PROVE (BDD10k) | Expected (Cityscapes) | Gap |
|-------|---------------|----------------------|-----|
| SegFormer MIT-B5 | 45.69% | ~82% | **-36%** |
| DeepLabV3+ R50 | 38.42% | ~77% | **-39%** |
| PSPNet R50 | 37.40% | ~76% | **-39%** |

### Why This Matters
1. **Multi-scale training is essential** for semantic segmentation
2. Objects appear at many scales in real images (cars nearby vs far away)
3. Without scale variation, model can't generalize to different object sizes
4. This explains why ALL augmentation strategies perform similarly (~45% ±0.5%)

---

## Recommended Fix for PROVE

### Option 1: Full Fix (Recommended)

Update `_build_trainid_training_pipeline` in unified_training_config.py:

```python
def _build_trainid_training_pipeline(self, dataset_cfg, is_baseline=False):
    crop_size = (512, 512)
    
    pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='ReduceToSingleChannel'),
        # CHANGE: RandomResize with scale variation
        dict(type='RandomResize', scale=(2048, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
        dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='PackSegInputs'),
    ]
    
    return pipeline
```

**Note:** Remove the `is_baseline` branch - baseline should still use full augmentation pipeline.

### Option 2: Minimal Fix

At minimum, change:
```python
# BEFORE (broken):
dict(type='Resize', scale=(1024, 512), keep_ratio=True),

# AFTER (working):
dict(type='RandomResize', scale=(2048, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
```

---

## Verification Strategy

1. **Cityscapes Replication (In Progress)**
   - 6 jobs submitted with standard pipeline
   - Expected results: 76-82% mIoU
   - If achieved: Confirms pipeline bug is the issue

2. **PROVE Pipeline Fix**
   - After verification, update unified_training_config.py
   - Re-run all training with fixed pipeline
   - Compare results

---

## Files to Modify

| File | Change Needed |
|------|--------------|
| `unified_training_config.py` | Update all `_build_*_training_pipeline` methods |
| `unified_training_config.py` | Remove `is_baseline` conditional for augmentation |

---

## References

- [mmsegmentation Cityscapes config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/_base_/datasets/cityscapes.py)
- PROVE Cityscapes replication: `cityscapes_replication/configs/`
