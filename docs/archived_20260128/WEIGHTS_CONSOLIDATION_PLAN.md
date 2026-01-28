# WEIGHTS Directory Consolidation Plan

*Generated: 2026-01-15*

## Summary of Issues

### 1. Naming Inconsistencies (NOT data corruption)

The directory naming has inconsistencies:
- `idd-aw_cd` vs `iddaw_cd` (hyphen vs no hyphen)
- Unsuffixed directories (should have `_cd` or `_ad` suffix)

**Standard Naming Convention:**
- Use `idd-aw` (with hyphen) for IDD-AW dataset
- `_cd` suffix = Stage 1 (domain_filter='clear_day')
- `_ad` suffix = Stage 2 (no domain_filter / all domains)

### 2. Class Count Configuration

Different datasets use different class counts:
| Dataset | Classes | Notes |
|---------|---------|-------|
| BDD10k | 19 | Unified Cityscapes classes |
| IDD-AW | 19 | Unified Cityscapes classes |
| MapillaryVistas | 66 | Native MapillaryVistas classes |
| OUTSIDE15k | 24 | Native OUTSIDE15k classes |

This is **intentional** - models are trained with native classes for some datasets.

### 3. Test Report Class Name Issue

The test reports for MapillaryVistas/OUTSIDE15k show:
- Cityscapes class names (road, sidewalk, building, etc.)
- Plus additional `class_19`, `class_20`, etc. for native classes

This is a **display issue** in the per-class report - the overall metrics (mIoU) are correct.

## Consolidation Actions

### Directories That Need Renaming

**Safe Renames (target doesn't exist):**
```
baseline/bdd10k → baseline/bdd10k_ad
baseline/idd-aw → baseline/idd-aw_ad
baseline/mapillaryvistas → baseline/mapillaryvistas_ad
baseline/outside15k → baseline/outside15k_ad
gen_albumentations_weather/bdd10k → gen_albumentations_weather/bdd10k_ad
gen_albumentations_weather/idd-aw → gen_albumentations_weather/idd-aw_ad
gen_albumentations_weather/mapillaryvistas → gen_albumentations_weather/mapillaryvistas_ad
gen_albumentations_weather/outside15k → gen_albumentations_weather/outside15k_ad
gen_augmenters/bdd10k → gen_augmenters/bdd10k_ad
gen_augmenters/idd-aw → gen_augmenters/idd-aw_ad
gen_augmenters/mapillaryvistas → gen_augmenters/mapillaryvistas_ad
gen_augmenters/outside15k → gen_augmenters/outside15k_ad
gen_automold/bdd10k → gen_automold/bdd10k_ad
gen_automold/idd-aw → gen_automold/idd-aw_ad
gen_automold/mapillaryvistas → gen_automold/mapillaryvistas_ad
gen_automold/outside15k → gen_automold/outside15k_ad
gen_CNetSeg/bdd10k → gen_CNetSeg/bdd10k_ad
gen_CNetSeg/idd-aw → gen_CNetSeg/idd-aw_ad
gen_CNetSeg/mapillaryvistas → gen_CNetSeg/mapillaryvistas_ad
gen_CNetSeg/outside15k → gen_CNetSeg/outside15k_ad
gen_CUT/bdd10k → gen_CUT/bdd10k_ad
gen_cycleGAN/bdd10k → gen_cycleGAN/bdd10k_ad
gen_cycleGAN/idd-aw → gen_cycleGAN/idd-aw_ad
gen_cycleGAN/mapillaryvistas → gen_cycleGAN/mapillaryvistas_ad
gen_cycleGAN/outside15k → gen_cycleGAN/outside15k_ad
gen_EDICT/bdd10k → gen_EDICT/bdd10k_ad
gen_Img2Img/bdd10k → gen_Img2Img/bdd10k_ad
gen_IP2P/bdd10k → gen_IP2P/bdd10k_ad
gen_LANIT/bdd10k → gen_LANIT/bdd10k_ad
gen_SUSTechGAN/bdd10k → gen_SUSTechGAN/bdd10k_ad
gen_SUSTechGAN/idd-aw → gen_SUSTechGAN/idd-aw_ad
gen_SUSTechGAN/mapillaryvistas → gen_SUSTechGAN/mapillaryvistas_ad
gen_SUSTechGAN/outside15k → gen_SUSTechGAN/outside15k_ad
gen_TSIT/bdd10k → gen_TSIT/bdd10k_ad
gen_UniControl/bdd10k → gen_UniControl/bdd10k_ad
```

**Requires Merging (target exists):**
```
CONFLICT: baseline/iddaw_ad (0 ckpts) + baseline/idd-aw → baseline/idd-aw_ad
CONFLICT: baseline/iddaw_cd (0 ckpts) → MERGE INTO baseline/idd-aw_cd (1 ckpts)
CONFLICT: gen_Attribute_Hallucination/iddaw_ad → MERGE INTO gen_Attribute_Hallucination/idd-aw_ad (if exists)
CONFLICT: gen_Attribute_Hallucination/iddaw_cd (0 ckpts) → MERGE INTO gen_Attribute_Hallucination/idd-aw_cd (1 ckpts)
CONFLICT: gen_CNetSeg/iddaw_ad → MERGE INTO gen_CNetSeg/idd-aw_ad
CONFLICT: gen_CNetSeg/iddaw_cd (0 ckpts) → MERGE INTO gen_CNetSeg/idd-aw_cd (1 ckpts)
CONFLICT: gen_CUT/iddaw_ad (3 ckpts) → MERGE INTO gen_CUT/idd-aw_cd (0 ckpts) - source is Stage2 (_ad)
CONFLICT: gen_CUT/iddaw_cd (0 ckpts) → MERGE INTO gen_CUT/idd-aw_cd (0 ckpts)
CONFLICT: gen_cyclediffusion/iddaw_cd (9 ckpts) → MERGE INTO gen_cyclediffusion/idd-aw_cd (3 ckpts)
CONFLICT: gen_cycleGAN/iddaw_cd (14 ckpts) → MERGE INTO gen_cycleGAN/idd-aw_cd (3 ckpts)
CONFLICT: gen_flux_kontext/iddaw_ad (2 ckpts) → CREATE gen_flux_kontext/idd-aw_ad
CONFLICT: gen_flux_kontext/iddaw_cd (3 ckpts) → CREATE gen_flux_kontext/idd-aw_cd
CONFLICT: gen_IP2P/iddaw_ad (2 ckpts) → CREATE gen_IP2P/idd-aw_ad
CONFLICT: gen_IP2P/iddaw_cd (0 ckpts) → MERGE INTO gen_IP2P/idd-aw_cd (1 ckpts)
CONFLICT: gen_Qwen_Image_Edit/idd-aw (1 ckpts) → MERGE/CHECK
CONFLICT: gen_Qwen_Image_Edit/iddaw_ad (1 ckpts) → CREATE gen_Qwen_Image_Edit/idd-aw_ad
CONFLICT: gen_Qwen_Image_Edit/mapillaryvistas (1 ckpts) → MERGE INTO gen_Qwen_Image_Edit/mapillaryvistas_ad (1 ckpts)
CONFLICT: gen_Qwen_Image_Edit/outside15k (1 ckpts) → MERGE INTO gen_Qwen_Image_Edit/outside15k_ad (1 ckpts)
CONFLICT: gen_stargan_v2/iddaw_cd (10 ckpts) → MERGE INTO gen_stargan_v2/idd-aw_cd (2 ckpts)
CONFLICT: gen_step1x_v1p2/iddaw_cd (1 ckpts) → MERGE INTO gen_step1x_v1p2/idd-aw_cd (2 ckpts)
CONFLICT: gen_TSIT/idd-aw (1 ckpts) → CREATE gen_TSIT/idd-aw_ad
CONFLICT: gen_TSIT/iddaw_ad (1 ckpts) → CREATE gen_TSIT/idd-aw_ad
CONFLICT: gen_TSIT/mapillaryvistas (1 ckpts) → MERGE INTO gen_TSIT/mapillaryvistas_ad (1 ckpts)
CONFLICT: gen_TSIT/outside15k (1 ckpts) → MERGE INTO gen_TSIT/outside15k_ad (1 ckpts)
CONFLICT: gen_UniControl/bdd10k (1 ckpts) → MERGE INTO gen_UniControl/bdd10k_ad (1 ckpts)
CONFLICT: gen_UniControl/iddaw_ad (2 ckpts) → CREATE gen_UniControl/idd-aw_ad
CONFLICT: gen_VisualCloze/iddaw_ad (2 ckpts) → CREATE gen_VisualCloze/idd-aw_ad
CONFLICT: std_std_photometric_distort/iddaw_cd (0 ckpts) → MERGE INTO std_std_photometric_distort/idd-aw_cd (5 ckpts)
CONFLICT: std_minimal/iddaw_ad (2 ckpts) → CREATE std_minimal/idd-aw_ad
```

### Special Note: Ratio Ablation Directories

Some `iddaw_cd` directories contain **ratio ablation models** (different ratios), not standard models:
- `gen_cyclediffusion/iddaw_cd/`: ratio models (r0.00, r0.12, etc.)
- `gen_cycleGAN/iddaw_cd/`: ratio models
- `gen_stargan_v2/iddaw_cd/`: ratio models

These should be moved as-is to `idd-aw_cd/` and will contain both standard and ratio models.

## Recommended Execution Order

1. **First Pass**: Safe renames (no conflicts)
2. **Second Pass**: Merge conflicts (copy subdirectories, don't overwrite)
3. **Third Pass**: Clean up empty source directories
4. **Fourth Pass**: Verify all checkpoints are accessible

## Script Location

`scripts/cleanup_corrupted_weights.sh` (needs to be updated with merge logic)
