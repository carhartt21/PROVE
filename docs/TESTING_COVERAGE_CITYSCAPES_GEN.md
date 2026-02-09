# Testing Coverage Report

**Generated:** 2026-02-09 21:48

## Summary

| Status | Count | Percentage |
|--------|------:|----------:|
| ✅ Complete (valid mIoU) | 129 | 93.5% |
| 🔄 Running | 0 | 0.0% |
| ⏳ Pending (in queue) | 0 | 0.0% |
| ⚠️ Buggy (mIoU < 5%) | 3 | 2.2% |
| 🔃 Stale (wrong checkpoint) | 0 | 0.0% |
| ❌ Missing (no results) | 6 | 4.3% |
| **Total** | **138** | **100%** |

## Per-Dataset Breakdown

### Cityscapes
- Complete: 63/100 (63.0%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 3
- Missing (no results): 3

### ACDC (cross-domain)
- Complete: 66/100 (66.0%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 0
- Missing (no results): 3

## Running Configurations

*No test jobs currently running.*

## Pending Configurations (in queue)

*No configurations pending in queue.*

## Buggy Configurations (need retesting)

These configurations have test results with mIoU < 5%, indicating a bug in the test.

| Strategy | Dataset | Model | mIoU |
|----------|---------|-------|-----:|
| gen_Attribute_Hallucination | cityscapes | PSPNet | N/A |
| gen_SUSTechGAN | cityscapes | Mask2Former | N/A |
| gen_cyclediffusion | cityscapes | Mask2Former | N/A |

## Stale Configurations (need retesting)

*No stale configurations - all tests use correct checkpoints.*

## Missing Configurations (no test results)

| Strategy | Dataset | Model | Issue |
|----------|---------|-------|-------|
| gen_Attribute_Hallucination | acdc | SegNeXt | missing |
| gen_Attribute_Hallucination | cityscapes | SegNeXt | missing |
| gen_UniControl | acdc | PSPNet | missing |
| gen_UniControl | acdc | SegNeXt | missing |
| gen_UniControl | cityscapes | PSPNet | missing |
| gen_UniControl | cityscapes | SegNeXt | missing |

## Complete Configurations

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k |
|----------|--------|--------|-----------------|------------|
| gen_Attribute_Hallucination | ✅ SF | ✅ PSP, SF |
| gen_augmenters | ✅ SF | ✅ SF |
| gen_automold | ✅ PSP, SF | ✅ PSP, SF |
| gen_CNetSeg | ✅ SF | ✅ SF |
| gen_CUT | ✅ SF | ✅ SF |
| gen_cyclediffusion | ✅ PSP, SF | ✅ PSP, SF |
| gen_cycleGAN | ⏳ | ⏳ |
| gen_flux_kontext | ✅ PSP, SF | ✅ PSP, SF |
| gen_Img2Img | ✅ SF | ✅ SF |
| gen_IP2P | ✅ PSP, SF | ✅ PSP, SF |
| gen_LANIT | ⏳ | ⏳ |
| gen_Qwen_Image_Edit | ✅ SF | ✅ SF |
| gen_stargan_v2 | ✅ SF | ✅ SF |
| gen_step1x_new | ✅ PSP, SF | ✅ PSP, SF |
| gen_step1x_v1p2 | ✅ PSP, SF | ✅ PSP, SF |
| gen_SUSTechGAN | ✅ PSP, SF | ✅ PSP, SF |
| gen_TSIT | ✅ SF | ✅ SF |
| gen_UniControl | ✅ SF | ✅ SF |
| gen_VisualCloze | ✅ PSP, SF | ✅ PSP, SF |
| gen_Weather_Effect_Generator | ✅ SF | ✅ SF |
| gen_albumentations_weather | ✅ PSP, SF | ✅ PSP, SF |
| baseline | ✅ PSP, SF | ✅ PSP, SF |
| std_minimal | ⏳ | ⏳ |
| std_photometric_distort | ⏳ | ⏳ |
| std_autoaugment | ✅ PSP, SF | ✅ PSP, SF |
| std_cutmix | ✅ PSP, SF | ✅ PSP, SF |
| std_mixup | ✅ PSP, SF | ✅ PSP, SF |
| std_randaugment | ✅ PSP, SF | ✅ PSP, SF |
