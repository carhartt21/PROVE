# Testing Coverage Report

**Generated:** 2026-02-13 23:12

## Summary

| Status | Count | Percentage |
|--------|------:|----------:|
| ✅ Complete (valid mIoU) | 339 | 97.7% |
| 🔄 Running | 0 | 0.0% |
| ⏳ Pending (in queue) | 0 | 0.0% |
| ⚠️ Buggy (mIoU < 5%) | 0 | 0.0% |
| 🔃 Stale (wrong checkpoint) | 0 | 0.0% |
| ❌ Missing (no results) | 8 | 2.3% |
| **Total** | **347** | **100%** |

## Per-Dataset Breakdown

### BDD10k
- Complete: 67/112 (59.8%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 0
- Missing (no results): 3

### IDD-AW
- Complete: 60/106 (56.6%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 0
- Missing (no results): 3

### MapillaryVistas
- Complete: 106/112 (94.6%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 0
- Missing (no results): 1

### OUTSIDE15k
- Complete: 106/112 (94.6%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 0
- Missing (no results): 1

## Running Configurations

*No test jobs currently running.*

## Pending Configurations (in queue)

*No configurations pending in queue.*

## Buggy Configurations (need retesting)

*No buggy configurations - all tests have valid mIoU.*

## Stale Configurations (need retesting)

*No stale configurations - all tests use correct checkpoints.*

## Missing Configurations (no test results)

| Strategy | Dataset | Model | Issue |
|----------|---------|-------|-------|
| baseline | idd-aw | SegFormer | missing |
| gen_LANIT | mapillaryvistas | Mask2Former | missing |
| gen_LANIT | outside15k | Mask2Former | missing |
| gen_albumentations_weather | bdd10k | PSPNet | missing |
| gen_albumentations_weather | idd-aw | PSPNet | missing |
| gen_albumentations_weather | idd-aw | SegNeXt | missing |
| gen_automold | bdd10k | PSPNet | missing |
| gen_automold | bdd10k | SegNeXt | missing |

## Complete Configurations

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k |
|----------|--------|--------|-----------------|------------|
| gen_Attribute_Hallucination | ✅ SF | ✅ SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_augmenters | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_automold | ✅ SF | ✅ SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_CNetSeg | ⏳ | ⏳ | ✅ PSP, SF | ✅ PSP, SF |
| gen_CUT | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_cyclediffusion | ✅ SF | ✅ SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_cycleGAN | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_flux_kontext | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_Img2Img | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_IP2P | ✅ SF | ✅ SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_LANIT | ✅ SF | ✅ SF | ✅ SF | ✅ SF |
| gen_Qwen_Image_Edit | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_stargan_v2 | ⏳ | ⏳ | ✅ PSP, SF | ✅ PSP, SF |
| gen_step1x_new | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_step1x_v1p2 | ✅ SF | ✅ SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_SUSTechGAN | ✅ SF | ✅ SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_TSIT | ⏳ | ⏳ | ✅ PSP, SF | ✅ PSP, SF |
| gen_UniControl | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_VisualCloze | ✅ SF | ✅ SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_Weather_Effect_Generator | ⏳ | ⏳ | ✅ PSP, SF | ✅ PSP, SF |
| gen_albumentations_weather | ✅ SF | ✅ SF | ✅ PSP, SF | ✅ PSP, SF |
| baseline | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| std_minimal | ⏳ | ⏳ | ⏳ | ⏳ |
| std_photometric_distort | ⏳ | ⏳ | ⏳ | ⏳ |
| std_autoaugment | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| std_cutmix | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| std_mixup | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| std_randaugment | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
