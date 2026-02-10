# Testing Coverage Report

**Generated:** 2026-02-10 09:52

## Summary

| Status | Count | Percentage |
|--------|------:|----------:|
| ✅ Complete (valid mIoU) | 163 | 87.6% |
| 🔄 Running | 0 | 0.0% |
| ⏳ Pending (in queue) | 0 | 0.0% |
| ⚠️ Buggy (mIoU < 5%) | 13 | 7.0% |
| 🔃 Stale (wrong checkpoint) | 0 | 0.0% |
| ❌ Missing (no results) | 10 | 5.4% |
| **Total** | **186** | **100%** |

## Per-Dataset Breakdown

### Cityscapes
- Complete: 72/100 (72.0%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 13
- Missing (no results): 8

### ACDC (cross-domain)
- Complete: 91/100 (91.0%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 0
- Missing (no results): 2

## Running Configurations

*No test jobs currently running.*

## Pending Configurations (in queue)

*No configurations pending in queue.*

## Buggy Configurations (need retesting)

These configurations have test results with mIoU < 5%, indicating a bug in the test.

| Strategy | Dataset | Model | mIoU |
|----------|---------|-------|-----:|
| gen_Attribute_Hallucination | cityscapes | Mask2Former | N/A |
| gen_CNetSeg | cityscapes | PSPNet | N/A |
| gen_CUT | cityscapes | Mask2Former | N/A |
| gen_CUT | cityscapes | PSPNet | N/A |
| gen_CUT | cityscapes | SegNeXt | N/A |
| gen_IP2P | cityscapes | Mask2Former | N/A |
| gen_Img2Img | cityscapes | Mask2Former | N/A |
| gen_Img2Img | cityscapes | PSPNet | N/A |
| gen_Qwen_Image_Edit | cityscapes | Mask2Former | N/A |
| gen_Qwen_Image_Edit | cityscapes | PSPNet | N/A |
| gen_UniControl | cityscapes | Mask2Former | N/A |
| gen_Weather_Effect_Generator | cityscapes | PSPNet | N/A |
| gen_stargan_v2 | cityscapes | PSPNet | N/A |

## Stale Configurations (need retesting)

*No stale configurations - all tests use correct checkpoints.*

## Missing Configurations (no test results)

| Strategy | Dataset | Model | Issue |
|----------|---------|-------|-------|
| gen_Attribute_Hallucination | cityscapes | PSPNet | missing |
| gen_Attribute_Hallucination | cityscapes | SegNeXt | missing |
| gen_CNetSeg | acdc | Mask2Former | missing |
| gen_CNetSeg | cityscapes | Mask2Former | missing |
| gen_SUSTechGAN | cityscapes | Mask2Former | missing |
| gen_TSIT | acdc | PSPNet | missing |
| gen_TSIT | cityscapes | PSPNet | missing |
| gen_UniControl | cityscapes | PSPNet | missing |
| gen_UniControl | cityscapes | SegNeXt | missing |
| gen_cyclediffusion | cityscapes | Mask2Former | missing |

## Complete Configurations

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k |
|----------|--------|--------|-----------------|------------|
| gen_Attribute_Hallucination | ✅ SF | ✅ PSP, SF |
| gen_augmenters | ✅ SF | ✅ SF |
| gen_automold | ✅ PSP, SF | ✅ PSP, SF |
| gen_CNetSeg | ✅ SF | ✅ PSP, SF |
| gen_CUT | ✅ SF | ✅ PSP, SF |
| gen_cyclediffusion | ✅ PSP, SF | ✅ PSP, SF |
| gen_cycleGAN | ✅ SF | ✅ SF |
| gen_flux_kontext | ✅ PSP, SF | ✅ PSP, SF |
| gen_Img2Img | ✅ SF | ✅ PSP, SF |
| gen_IP2P | ✅ PSP, SF | ✅ PSP, SF |
| gen_LANIT | ⏳ | ⏳ |
| gen_Qwen_Image_Edit | ✅ SF | ✅ PSP, SF |
| gen_stargan_v2 | ✅ SF | ✅ PSP, SF |
| gen_step1x_new | ✅ PSP, SF | ✅ PSP, SF |
| gen_step1x_v1p2 | ✅ PSP, SF | ✅ PSP, SF |
| gen_SUSTechGAN | ✅ PSP, SF | ✅ PSP, SF |
| gen_TSIT | ✅ SF | ✅ SF |
| gen_UniControl | ✅ SF | ✅ PSP, SF |
| gen_VisualCloze | ✅ PSP, SF | ✅ PSP, SF |
| gen_Weather_Effect_Generator | ✅ SF | ✅ PSP, SF |
| gen_albumentations_weather | ✅ PSP, SF | ✅ PSP, SF |
| baseline | ✅ PSP, SF | ✅ PSP, SF |
| std_minimal | ⏳ | ⏳ |
| std_photometric_distort | ⏳ | ⏳ |
| std_autoaugment | ✅ PSP, SF | ✅ PSP, SF |
| std_cutmix | ✅ PSP, SF | ✅ PSP, SF |
| std_mixup | ✅ PSP, SF | ✅ PSP, SF |
| std_randaugment | ✅ PSP, SF | ✅ PSP, SF |
