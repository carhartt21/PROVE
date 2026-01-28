# Bug Report: MixedDataLoader Never Implemented - Generated Images Not Used

**Date Discovered:** 2026-01-28  
**Date Re-Analyzed:** 2026-01-28  
**Severity:** **CRITICAL** (but different from original report)  
**Status:** **UNFIXED - MixedDataLoader not connected**

## Executive Summary

**The original "cross-dataset contamination bug" was a FALSE POSITIVE**, but the investigation revealed a **MUCH MORE SERIOUS ISSUE:**

1. **Generated images are NEVER loaded during training** - the MixedDataLoader infrastructure exists but isn't wired up
2. **The `real_gen_ratio` parameter has NO EFFECT** - all ratio ablation results are invalid
3. **All "gen_*" strategies train identically** - the only difference is pipeline augmentations, not generative models
4. **gen_* vs baseline comparison is invalid** - it compares augmentation strength, not generative methods

## Investigation Timeline

### Initial Analysis (Incorrect)

The initial investigation found that `GeneratedAugmentedDataset.load_data_list()` had no dataset filtering, which **would** load all datasets from the manifest if it were used.

### Re-Analysis (Correct)

Upon reviewing actual training configurations and logs:

```python
# ACTUAL train_dataloader in training_config.py
train_dataloader = dict(
    dataset=dict(
        type='CityscapesDataset',  # NOT GeneratedAugmentedDataset!
        data_prefix=dict(
            img_path='train/images/BDD10k/clear_day',
            seg_map_path='train/labels/BDD10k/clear_day'),
        ...
    ))
```

The `mixed_dataloader` section in the config is **metadata only** - it describes intended behavior but MMEngine's `Runner.from_cfg()` uses `train_dataloader`, not `mixed_dataloader`.

## Proof That Bug Doesn't Exist

If cross-dataset contamination had occurred, training would have **crashed** due to:

1. **MapillaryVistas labels** have RGB-encoded colors. When `ReduceToSingleChannel` extracts the R channel, values like 70, 128, 170 appear. These would cause:
   ```
   IndexError: Target 128 is out of bounds.
   ```

2. **ACDC labels** use Cityscapes labelIDs (0-33) not trainIDs (0-18). Values 20-33 would cause IndexErrors.

3. **The training completed successfully** with reasonable mIoU (50.71%), proving only valid BDD10k labels were used.

## What The Code Actually Does

### Config Generation (`unified_training_config.py`)
- Creates both `train_dataloader` (used) and `mixed_dataloader` (informational) sections
- `train_dataloader` points directly to the target dataset's images and labels
- `mixed_dataloader` was designed for future functionality that was never implemented

### Training (`unified_training.py`)
- Generates a simple training script: `Runner.from_cfg(cfg).train()`
- MMEngine only uses `train_dataloader` from the config
- The `mixed_dataloader` section is ignored by MMEngine

### GeneratedAugmentedDataset (`generated_images_dataset.py`)
- A valid dataset class that COULD mix generated images
- **BUT it's never actually instantiated** during training
- The `dataset_filter` parameter added in commit ecb9721 is dead code

## The "Fix" Applied (Unnecessary)

Commit `ecb9721` added `dataset_filter` to `GeneratedAugmentedDataset`, but since this class is never used, the fix has no effect on actual training.

## Why gen_stargan_v2 Performs Well

The original question was "why does gen_stargan_v2 perform well even at ratio 0.0?"

**CRITICAL FINDING: Generated images are NEVER loaded!**

After thorough investigation, the truth is:
1. **MixedDataLoader is NOT connected** - `_setup_mixed_training()` only prints config values
2. **`real_gen_ratio` has NO EFFECT** - the parameter only sets unused config metadata
3. **gen_* strategies differ ONLY in the training pipeline:**

| Strategy | Pipeline |
|----------|----------|
| `baseline` | `Resize(512,512)` → `PackSegInputs` |
| `gen_*` | `Resize(1024,512)` → `RandomCrop` → `RandomFlip` → **`PhotoMetricDistortion`** → `PackSegInputs` |

**The performance difference comes from `PhotoMetricDistortion`** - a standard color/brightness augmentation that simulates some weather-like effects. This is NOT generative augmentation - it's classical data augmentation!

### Proof from Config Files

**Baseline:**
```python
train_dataloader.dataset.pipeline=[
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='ReduceToSingleChannel'),
    dict(keep_ratio=False, scale=(512, 512), type='Resize'),
    dict(type='PackSegInputs'),
]
```

**gen_stargan_v2 (and ALL gen_* strategies):**
```python
train_dataloader.dataset.pipeline=[
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='ReduceToSingleChannel'),
    dict(keep_ratio=True, scale=(1024, 512), type='Resize'),
    dict(cat_max_ratio=0.75, crop_size=(512, 512), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),  # ← THE REAL DIFFERENCE
    dict(type='PackSegInputs'),
]
```

## Implications

1. **ALL ratio ablation results are INVALID** - ratio 0.0 vs 0.5 vs 1.0 all trained identically
2. **All "gen_*" strategy comparisons are INVALID** - they all get the same pipeline
3. **gen_* vs baseline comparison is actually augmentation_strong vs augmentation_none**
4. **No generated images were ever used in any training**

## Recommendations

1. **URGENT: Implement MixedDataLoader properly** - wire it up in the training script
2. **Re-run all ratio ablation studies** - current results are meaningless
3. **Re-evaluate all gen_* strategy comparisons** - they're not comparing generative methods
4. **Consider fair baseline** - add `PhotoMetricDistortion` to baseline for proper comparison

## Lessons Learned

1. Always verify assumptions by checking actual runtime behavior, not just code analysis
2. Check training logs and configs from completed runs
3. Consider what errors would occur if a bug existed - crashes are evidence of absence

---

*This bug report has been corrected based on thorough investigation of actual training behavior.*
