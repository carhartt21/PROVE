# Archived Invalid Results

**Archived:** 2026-01-28

## Why These Files Are Archived

These analysis files contain results from training runs where **generated images were NEVER loaded**. 
The MixedDataLoader infrastructure existed but was never connected, so all `gen_*` strategies 
trained identically to baseline (with only pipeline augmentation differences).

## Bug Details

See [../BUG_REPORT_CROSS_DATASET_CONTAMINATION.md](../BUG_REPORT_CROSS_DATASET_CONTAMINATION.md)

## Archived Files

| File | Reason |
|------|--------|
| `DOMAIN_ADAPTATION_ABLATION.md` | gen_* domain adaptation comparisons invalid |
| `STAGE_COMPARISON_ANALYSIS.md` | Stage 1 vs 2 gen_* comparisons invalid |

## Status

- **Bug Status:** ✅ FIXED (Jan 28, 2026)
- **Retraining:** ⏳ Required to generate valid results
- **These files:** May be regenerated after retraining

## What Was Invalid

1. All `gen_*` vs `baseline` comparisons (only compared PhotoMetricDistortion)
2. All ratio ablation results (ratio parameter had no effect)
3. All "best strategy" rankings involving gen_* strategies

## What Remains Valid

- `baseline` results (no generated images expected)
- `std_*` strategy results (use pipeline augmentation only)
- `std_std_std_photometric_distort` results
