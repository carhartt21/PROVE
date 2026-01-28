# Bug Report: Cross-Dataset Contamination in Generative Strategy Training

**Date Discovered:** 2026-01-28
**Severity:** CRITICAL
**Status:** FIXED (commit ecb9721)

## Summary

A critical bug in `GeneratedAugmentedDataset` caused training for generative strategies to load ALL images from the manifest across ALL datasets, instead of filtering to the target dataset only.

## Impact

- **865 trained models affected** (all generative strategy models)
- Training data was **18x larger than intended**
- Results from generative strategies **cannot be fairly compared** to baselines or non-generative strategies

### Affected Model Counts by Location

| Location | Affected Models |
|----------|----------------|
| Stage 1 (WEIGHTS) | 252 |
| Stage 2 (WEIGHTS_STAGE_2) | 253 |
| Ratio Ablation Stage 1 | 186 |
| Ratio Ablation Stage 2 | 102 |
| Extended Training | 37 |
| Combinations | 35 |
| **TOTAL AFFECTED** | **865** |
| Total Unaffected (baseline, std_*) | 171 |

## Root Cause

The `GeneratedAugmentedDataset.load_data_list()` method in `generated_images_dataset.py` was loading ALL entries from the manifest without filtering by dataset name.

### Example: Training BDD10k with gen_stargan_v2

**Expected behavior:**
- Load ~10,218 generated images from BDD10k

**Actual behavior (bug):**
- Loaded 187,398 images from ALL datasets:
  - ACDC: 4,206 (2.2%)
  - BDD100k: 79,992 (42.7%)
  - BDD10k: 10,218 (5.5%)  ← target dataset was only 5.5% of training!
  - IDD-AW: 23,082 (12.3%)
  - MapillaryVistas: 45,162 (24.1%)
  - OUTSIDE15k: 24,738 (13.2%)

## Affected Strategies

All 24 generative strategies were affected:
- gen_cycleGAN, gen_cyclediffusion, gen_IP2P, gen_flux_kontext
- gen_stargan_v2, gen_LANIT, gen_step1x_new, gen_step1x_v1p2
- gen_Attribute_Hallucination, gen_Img2Img, gen_CUT, gen_TSIT
- gen_SUSTechGAN, gen_StyleID, gen_EDICT, gen_UniControl
- gen_VisualCloze, gen_Qwen-Image-Edit, gen_albumentations_weather
- gen_automold, gen_Weather_Effect_Generator, gen_augmenters
- gen_CNetSeg, gen_flux2

## Fix Applied

Added `dataset_filter` parameter to `GeneratedAugmentedDataset`:

```python
# generated_images_dataset.py
class GeneratedAugmentedDataset(BaseSegDataset):
    def __init__(self, ..., dataset_filter: str = None, ...):
        self.dataset_filter = dataset_filter
    
    def load_data_list(self):
        for gen_path, original_path in condition_entries:
            # CRITICAL: Filter by dataset if specified
            if self.dataset_filter and self.dataset_filter not in original_path:
                continue  # Skip cross-dataset entries
            ...
```

```python
# unified_training_config.py
config['mixed_dataloader']['generated_dataset'] = {
    ...
    'dataset_filter': dataset_cfg.name,  # NEW: Filter to target dataset
}
```

## Verification

After fix:
- Loading for BDD10k now correctly loads only 10,218 images
- Skips 177,180 cross-dataset images
- Warning printed if no dataset_filter specified

## Implications for Research

1. **Unfair comparison**: Generative strategy models were trained on ~18x more data than baselines
2. **Invalid ablation results**: Ratio ablation results (real_gen_ratio) don't reflect actual ratios
3. **Unexpected performance**: Models like stargan_v2 performed well despite low semantic consistency because they were learning from high-quality images from other datasets (especially BDD100k)

## Recommendations

### Option A: Retrain All Affected Models
- Most accurate but resource-intensive
- Required for publication-quality results

### Option B: Interpret Results with Caveat
- Document that generative models were trained on multi-dataset data
- Can still analyze relative performance among generative strategies
- Cannot fairly compare to baseline/std_* strategies

### Option C: Selective Retraining
- Retrain only key configurations for paper figures
- Use existing results as preliminary exploration

## Related Investigation

This bug was discovered while investigating why `gen_stargan_v2` performed well at ratio 0.0 (100% generated images) despite having only 4.84% semantic consistency in quality pretests. The unexpected performance was due to the model actually training on diverse high-quality images from multiple datasets.
