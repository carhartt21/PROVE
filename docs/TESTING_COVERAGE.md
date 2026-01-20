# Testing Coverage Report

**Generated:** 2026-01-20 14:13

## Summary

| Status | Count | Percentage |
|--------|------:|----------:|
| ✅ Complete (valid mIoU) | 312 | 96.9% |
| 🔄 Running | 0 | 0.0% |
| ⏳ Pending (in queue) | 0 | 0.0% |
| ⚠️ Buggy (mIoU < 5%) | 0 | 0.0% |
| ❌ Missing (no results) | 10 | 3.1% |
| **Total** | **322** | **100%** |

## Per-Dataset Breakdown

### BDD10k
- Complete: 80/82 (97.6%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 0
- Missing (no results): 1

### IDD-AW
- Complete: 81/81 (100.0%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 0
- Missing (no results): 0

### MapillaryVistas
- Complete: 73/81 (90.1%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 0
- Missing (no results): 8

### OUTSIDE15k
- Complete: 78/80 (97.5%)
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

## Missing Configurations (no test results)

| Strategy | Dataset | Model | Issue |
|----------|---------|-------|-------|
| gen_Weather_Effect_Generator | mapillaryvistas | PSPNet | missing |
| std_autoaugment | mapillaryvistas | DeepLabV3+ | empty |
| std_autoaugment | outside15k | PSPNet | empty |
| std_cutmix | bdd10k | PSPNet | empty |
| std_cutmix | mapillaryvistas | DeepLabV3+ | empty |
| std_mixup | mapillaryvistas | DeepLabV3+ | empty |
| std_mixup | mapillaryvistas | PSPNet | empty |
| std_mixup | mapillaryvistas | SegFormer | empty |
| std_randaugment | mapillaryvistas | DeepLabV3+ | empty |
| std_randaugment | mapillaryvistas | PSPNet | empty |

## Complete Configurations

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k |
|----------|--------|--------|-----------------|------------|
| gen_Attribute_Hallucination | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_augmenters | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_automold | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_CNetSeg | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_CUT | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_cyclediffusion | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
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
| photometric_distort | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP |
| std_autoaugment | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ PSP, SF | ✅ DLV3+ |
| std_cutmix | ✅ DLV3+, SF | ✅ DLV3+, PSP, SF | ✅ PSP, SF | ✅ DLV3+, PSP, SF |
| std_mixup | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ⏳ | ✅ DLV3+, PSP, SF |
| std_randaugment | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ SF | ✅ DLV3+, PSP, SF |
