# Testing Coverage Report

**Generated:** 2026-02-12 13:07

## Summary

| Status | Count | Percentage |
|--------|------:|----------:|
| ✅ Complete (valid mIoU) | 187 | 95.9% |
| 🔄 Running | 0 | 0.0% |
| ⏳ Pending (in queue) | 0 | 0.0% |
| ⚠️ Buggy (mIoU < 5%) | 0 | 0.0% |
| 🔃 Stale (wrong checkpoint) | 0 | 0.0% |
| ❌ Missing (no results) | 8 | 4.1% |
| **Total** | **195** | **100%** |

## Per-Dataset Breakdown

### BDD10k
- Complete: 52/112 (46.4%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 0
- Missing (no results): 2

### IDD-AW
- Complete: 46/106 (43.4%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 0
- Missing (no results): 1

### MapillaryVistas
- Complete: 45/111 (40.5%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 0
- Missing (no results): 3

### OUTSIDE15k
- Complete: 44/111 (39.6%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 0
- Missing (no results): 2

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
| gen_Qwen_Image_Edit | mapillaryvistas | Mask2Former | missing |
| gen_Qwen_Image_Edit | outside15k | Mask2Former | missing |
| gen_UniControl | bdd10k | Mask2Former | missing |
| gen_UniControl | mapillaryvistas | PSPNet | missing |
| gen_UniControl | mapillaryvistas | SegNeXt | missing |
| gen_UniControl | outside15k | PSPNet | missing |
| gen_augmenters | bdd10k | PSPNet | missing |

## Complete Configurations

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k |
|----------|--------|--------|-----------------|------------|
| gen_Attribute_Hallucination | ✅ SF | ✅ SF | ✅ SF | ✅ SF |
| gen_augmenters | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_automold | ✅ SF | ✅ SF | ✅ SF | ✅ SF |
| gen_CNetSeg | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_CUT | ✅ SF | ✅ SF | ✅ SF | ✅ SF |
| gen_cyclediffusion | ✅ SF | ✅ SF | ✅ SF | ✅ SF |
| gen_cycleGAN | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_flux_kontext | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_Img2Img | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_IP2P | ✅ SF | ✅ SF | ✅ SF | ✅ SF |
| gen_LANIT | ✅ SF | ✅ SF | ✅ SF | ✅ SF |
| gen_Qwen_Image_Edit | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_stargan_v2 | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_step1x_new | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_step1x_v1p2 | ✅ SF | ✅ SF | ✅ SF | ✅ SF |
| gen_SUSTechGAN | ✅ SF | ✅ SF | ✅ SF | ✅ SF |
| gen_TSIT | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_UniControl | ✅ PSP, SF | ✅ PSP, SF | ✅ SF | ✅ SF |
| gen_VisualCloze | ✅ SF | ✅ SF | ✅ SF | ✅ SF |
| gen_Weather_Effect_Generator | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_albumentations_weather | ✅ SF | ✅ SF | ✅ SF | ✅ SF |
| baseline | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| std_minimal | ⏳ | ⏳ | ⏳ | ⏳ |
| std_photometric_distort | ⏳ | ⏳ | ⏳ | ⏳ |
| std_autoaugment | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| std_cutmix | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| std_mixup | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| std_randaugment | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
