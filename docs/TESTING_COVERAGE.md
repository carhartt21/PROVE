# Testing Coverage Report

**Generated:** 2026-01-16 10:28

## Summary

| Status | Count | Percentage |
|--------|------:|----------:|
| ✅ Complete (valid mIoU) | 312 | 93.4% |
| 🔄 Running | 0 | 0.0% |
| ⏳ Pending (in queue) | 17 | 5.1% |
| ⚠️ Buggy (mIoU < 5%) | 0 | 0.0% |
| ❌ Missing (no results) | 5 | 1.5% |
| **Total** | **334** | **100%** |

## Per-Dataset Breakdown

### BDD10k
- Complete: 82/85 (96.5%)
- Running: 0
- Pending (in queue): 1
- Buggy (mIoU < 5%): 0
- Missing (no results): 1

### IDD-AW
- Complete: 70/84 (83.3%)
- Running: 0
- Pending (in queue): 8
- Buggy (mIoU < 5%): 0
- Missing (no results): 4

### MapillaryVistas
- Complete: 76/84 (90.5%)
- Running: 0
- Pending (in queue): 8
- Buggy (mIoU < 5%): 0
- Missing (no results): 0

### OUTSIDE15k
- Complete: 84/84 (100.0%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 0
- Missing (no results): 0

## Running Configurations

*No test jobs currently running.*

## Pending Configurations (in queue)

| Strategy | Dataset | Model |
|----------|---------|-------|
| gen_Attribute_Hallucination | mapillaryvistas | PSPNet |
| gen_CNetSeg | idd-aw | DeepLabV3+ |
| gen_CNetSeg | idd-aw | PSPNet |
| gen_CNetSeg | idd-aw | SegFormer |
| gen_IP2P | idd-aw | PSPNet |
| gen_IP2P | idd-aw | SegFormer |
| gen_IP2P | mapillaryvistas | PSPNet |
| gen_Weather_Effect_Generator | bdd10k | SegFormer |
| gen_Weather_Effect_Generator | mapillaryvistas | PSPNet |
| gen_albumentations_weather | mapillaryvistas | DeepLabV3+ |
| gen_augmenters | mapillaryvistas | DeepLabV3+ |
| gen_automold | mapillaryvistas | DeepLabV3+ |
| gen_cycleGAN | mapillaryvistas | PSPNet |
| gen_flux_kontext | mapillaryvistas | PSPNet |
| photometric_distort | idd-aw | DeepLabV3+ |
| photometric_distort | idd-aw | PSPNet |
| photometric_distort | idd-aw | SegFormer |

## Buggy Configurations (need retesting)

*No buggy configurations - all tests have valid mIoU.*

## Missing Configurations (no test results)

| Strategy | Dataset | Model | Issue |
|----------|---------|-------|-------|
| std_minimal | bdd10k | SegFormer | missing |
| std_minimal | idd-aw | SegFormer | missing |
| std_randaugment | idd-aw | DeepLabV3+ | missing |
| std_randaugment | idd-aw | PSPNet | missing |
| std_randaugment | idd-aw | SegFormer | missing |

## Complete Configurations

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k |
|----------|--------|--------|-----------------|------------|
| gen_Attribute_Hallucination | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, SF | ✅ DLV3+, PSP, SF |
| gen_augmenters | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ PSP, SF | ✅ DLV3+, PSP, SF |
| gen_automold | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ PSP, SF | ✅ DLV3+, PSP, SF |
| gen_CNetSeg | ✅ DLV3+, PSP, SF | ⏳ | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_CUT | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_cyclediffusion | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_cycleGAN | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, SF | ✅ DLV3+, PSP, SF |
| gen_flux_kontext | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, SF | ✅ DLV3+, PSP, SF |
| gen_Img2Img | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_IP2P | ✅ DLV3+, PSP, SF | ⏳ | ✅ DLV3+, SF | ✅ DLV3+, PSP, SF |
| gen_LANIT | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_Qwen_Image_Edit | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_stargan_v2 | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_step1x_new | ✅ DLV3+, PSP, SF | ✅ DLV3+, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_step1x_v1p2 | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_SUSTechGAN | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_TSIT | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_UniControl | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_VisualCloze | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| gen_Weather_Effect_Generator | ✅ DLV3+, PSP | ✅ DLV3+, PSP, SF | ✅ DLV3+, SF | ✅ DLV3+, PSP, SF |
| gen_albumentations_weather | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ PSP, SF | ✅ DLV3+, PSP, SF |
| baseline | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| photometric_distort | ✅ DLV3+, PSP, SF | ⏳ | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| std_minimal | ✅ DLV3+, PSP | ✅ DLV3+, PSP | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| std_autoaugment | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| std_cutmix | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| std_mixup | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| std_randaugment | ✅ DLV3+, PSP, SF | ⏳ | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
