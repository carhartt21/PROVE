# Testing Coverage Report

**Generated:** 2026-02-13 15:47

## Summary

| Status | Count | Percentage |
|--------|------:|----------:|
| ✅ Complete (valid mIoU) | 287 | 95.3% |
| 🔄 Running | 0 | 0.0% |
| ⏳ Pending (in queue) | 0 | 0.0% |
| ⚠️ Buggy (mIoU < 5%) | 0 | 0.0% |
| 🔃 Stale (wrong checkpoint) | 0 | 0.0% |
| ❌ Missing (no results) | 14 | 4.7% |
| **Total** | **301** | **100%** |

## Per-Dataset Breakdown

### BDD10k
- Complete: 65/112 (58.0%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 0
- Missing (no results): 0

### IDD-AW
- Complete: 59/106 (55.7%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 0
- Missing (no results): 1

### MapillaryVistas
- Complete: 90/111 (81.1%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 0
- Missing (no results): 11

### OUTSIDE15k
- Complete: 73/111 (65.8%)
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
| gen_Attribute_Hallucination | mapillaryvistas | Mask2Former | missing |
| gen_CNetSeg | mapillaryvistas | Mask2Former | missing |
| gen_SUSTechGAN | mapillaryvistas | Mask2Former | missing |
| gen_TSIT | mapillaryvistas | segformer_mit-b3_ratio0p50 | missing |
| gen_VisualCloze | outside15k | PSPNet | missing |
| gen_Weather_Effect_Generator | mapillaryvistas | PSPNet | missing |
| gen_Weather_Effect_Generator | mapillaryvistas | segformer_mit-b3_ratio0p50 | missing |
| gen_Weather_Effect_Generator | mapillaryvistas | SegNeXt | missing |
| gen_albumentations_weather | mapillaryvistas | Mask2Former | no_json |
| gen_automold | mapillaryvistas | Mask2Former | missing |
| gen_cyclediffusion | mapillaryvistas | Mask2Former | missing |
| gen_step1x_v1p2 | mapillaryvistas | Mask2Former | no_json |
| gen_step1x_v1p2 | outside15k | PSPNet | missing |

## Complete Configurations

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k |
|----------|--------|--------|-----------------|------------|
| gen_Attribute_Hallucination | ✅ SF | ✅ SF | ✅ PSP, SF | ✅ SF |
| gen_augmenters | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_automold | ✅ SF | ✅ SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_CNetSeg | ⏳ | ⏳ | ✅ PSP, SF | ⏳ |
| gen_CUT | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_cyclediffusion | ✅ SF | ✅ SF | ✅ PSP, SF | ✅ SF |
| gen_cycleGAN | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_flux_kontext | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_Img2Img | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_IP2P | ✅ SF | ✅ SF | ✅ PSP, SF | ✅ SF |
| gen_LANIT | ✅ SF | ✅ SF | ✅ SF | ✅ SF |
| gen_Qwen_Image_Edit | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_stargan_v2 | ⏳ | ⏳ | ✅ PSP, SF | ⏳ |
| gen_step1x_new | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_step1x_v1p2 | ✅ SF | ✅ SF | ✅ PSP, SF | ✅ SF |
| gen_SUSTechGAN | ✅ SF | ✅ SF | ✅ PSP, SF | ✅ SF |
| gen_TSIT | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_UniControl | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_VisualCloze | ✅ SF | ✅ SF | ✅ PSP, SF | ✅ SF |
| gen_Weather_Effect_Generator | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_albumentations_weather | ✅ SF | ✅ SF | ✅ PSP, SF | ✅ PSP, SF |
| baseline | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| std_minimal | ⏳ | ⏳ | ⏳ | ⏳ |
| std_photometric_distort | ⏳ | ⏳ | ⏳ | ⏳ |
| std_autoaugment | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| std_cutmix | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| std_mixup | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| std_randaugment | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
