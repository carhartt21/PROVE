# Testing Coverage Report

**Generated:** 2026-01-17 21:56

## Summary

| Status | Count | Percentage |
|--------|------:|----------:|
| ✅ Complete (valid mIoU) | 321 | 99.1% |
| 🔄 Running | 3 | 0.9% |
| ⏳ Pending (in queue) | 0 | 0.0% |
| ⚠️ Buggy (mIoU < 5%) | 0 | 0.0% |
| ❌ Missing (no results) | 0 | 0.0% |
| **Total** | **324** | **100%** |

## Per-Dataset Breakdown

### BDD10k
- Complete: 81/82 (98.8%)
- Running: 0
- Pending (in queue): 1
- Buggy (mIoU < 5%): 0
- Missing (no results): 10

### IDD-AW
- Complete: 81/81 (100.0%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 0
- Missing (no results): 0

### MapillaryVistas
- Complete: 78/81 (96.3%)
- Running: 3
- Pending (in queue): 0
- Buggy (mIoU < 5%): 0
- Missing (no results): 0

### OUTSIDE15k
- Complete: 81/81 (100.0%)
- Running: 0
- Pending (in queue): 1
- Buggy (mIoU < 5%): 0
- Missing (no results): 52

## Running Configurations

| Strategy | Dataset | Model |
|----------|---------|-------|
| gen_Weather_Effect_Generator | mapillaryvistas | PSPNet |
| gen_cyclediffusion | mapillaryvistas | DeepLabV3+ |
| gen_cyclediffusion | mapillaryvistas | PSPNet |

## Pending Configurations (in queue)

| Strategy | Dataset | Model |
|----------|---------|-------|
| gen_Qwen_Image_Edit | mapillaryvistas | PSPNet |
| gen_Qwen_Image_Edit | outside15k | PSPNet |
| gen_UniControl | bdd10k | DeepLabV3+ |
| gen_Weather_Effect_Generator | mapillaryvistas | PSPNet |
| gen_albumentations_weather | mapillaryvistas | DeepLabV3+ |

## Buggy Configurations (need retesting)

*No buggy configurations - all tests have valid mIoU.*

## Missing Configurations (no test results)

*No missing configurations.*

## Complete Configurations

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k |
|----------|--------|--------|-----------------|------------|
| gen_Attribute_Hallucination | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_augmenters | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_automold | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_CNetSeg | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_CUT | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_cyclediffusion | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ SF | ✅ DLV3+, PSP, SF |
| gen_cycleGAN | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_flux_kontext | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_Img2Img | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_IP2P | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_LANIT | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_Qwen_Image_Edit | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_stargan_v2 | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_step1x_new | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_step1x_v1p2 | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_SUSTechGAN | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_TSIT | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_UniControl | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_VisualCloze | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_Weather_Effect_Generator | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, SF | ✅ DLV3+, PSP, SF |
| gen_albumentations_weather | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| baseline | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| photometric_distort | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| std_autoaugment | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| std_cutmix | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| std_mixup | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| std_randaugment | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
