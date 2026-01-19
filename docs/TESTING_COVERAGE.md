# Testing Coverage Report

**Generated:** 2026-01-17 02:02

## Summary

| Status | Count | Percentage |
|--------|------:|----------:|
| ✅ Complete (valid mIoU) | 183 | 65.1% |
| 🔄 Running | 0 | 0.0% |
| ⏳ Pending (in queue) | 5 | 1.8% |
| ⚠️ Buggy (mIoU < 5%) | 0 | 0.0% |
| ❌ Missing (no results) | 93 | 33.1% |
| **Total** | **281** | **100%** |

## Per-Dataset Breakdown

### BDD10k
- Complete: 70/82 (85.4%)
- Running: 0
- Pending (in queue): 1
- Buggy (mIoU < 5%): 0
- Missing (no results): 10

### IDD-AW
- Complete: 45/81 (55.6%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 0
- Missing (no results): 14

### MapillaryVistas
- Complete: 55/75 (73.3%)
- Running: 0
- Pending (in queue): 3
- Buggy (mIoU < 5%): 0
- Missing (no results): 17

### OUTSIDE15k
- Complete: 13/66 (19.7%)
- Running: 0
- Pending (in queue): 1
- Buggy (mIoU < 5%): 0
- Missing (no results): 52

## Running Configurations

*No test jobs currently running.*

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

| Strategy | Dataset | Model | Issue |
|----------|---------|-------|-------|
| baseline | mapillaryvistas | DeepLabV3+ | empty |
| baseline | outside15k | PSPNet | no_json |
| baseline | outside15k | SegFormer | no_json |
| gen_Attribute_Hallucination | mapillaryvistas | DeepLabV3+ | empty |
| gen_Attribute_Hallucination | mapillaryvistas | PSPNet | empty |
| gen_Attribute_Hallucination | mapillaryvistas | SegFormer | empty |
| gen_Attribute_Hallucination | outside15k | DeepLabV3+ | no_json |
| gen_Attribute_Hallucination | outside15k | PSPNet | no_json |
| gen_Attribute_Hallucination | outside15k | SegFormer | no_json |
| gen_CNetSeg | idd-aw | DeepLabV3+ | empty |
| gen_CNetSeg | idd-aw | PSPNet | empty |
| gen_CNetSeg | idd-aw | SegFormer | empty |
| gen_CNetSeg | mapillaryvistas | PSPNet | empty |
| gen_CNetSeg | mapillaryvistas | SegFormer | empty |
| gen_CNetSeg | outside15k | DeepLabV3+ | no_json |
| gen_CNetSeg | outside15k | PSPNet | no_json |
| gen_CNetSeg | outside15k | SegFormer | no_json |
| gen_CUT | outside15k | DeepLabV3+ | no_json |
| gen_CUT | outside15k | PSPNet | no_json |
| gen_CUT | outside15k | SegFormer | no_json |
| gen_IP2P | idd-aw | PSPNet | empty |
| gen_IP2P | idd-aw | SegFormer | empty |
| gen_IP2P | mapillaryvistas | PSPNet | empty |
| gen_IP2P | outside15k | DeepLabV3+ | no_json |
| gen_IP2P | outside15k | PSPNet | no_json |
| gen_IP2P | outside15k | SegFormer | no_json |
| gen_Img2Img | outside15k | DeepLabV3+ | no_json |
| gen_Img2Img | outside15k | PSPNet | no_json |
| gen_Img2Img | outside15k | SegFormer | no_json |
| gen_LANIT | outside15k | DeepLabV3+ | no_json |
| ... | ... | ... | (63 more) |

## Complete Configurations

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k |
|----------|--------|--------|-----------------|------------|
| gen_Attribute_Hallucination | ✅ DLV3+, PSP, SF | ⏳ | ⏳ | ⏳ |
| gen_augmenters | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ PSP | ✅ SF |
| gen_automold | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ SF | ✅ SF |
| gen_CNetSeg | ✅ DLV3+, PSP, SF | ⏳ | ✅ DLV3+ | ⏳ |
| gen_CUT | ✅ DLV3+, PSP, SF | ⏳ | ✅ DLV3+, PSP, SF | ⏳ |
| gen_cyclediffusion | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ⏳ | ⏳ |
| gen_cycleGAN | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ SF | ✅ SF |
| gen_flux_kontext | ⏳ | ⏳ | ✅ DLV3+, SF | ⏳ |
| gen_Img2Img | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ⏳ |
| gen_IP2P | ✅ DLV3+, PSP, SF | ⏳ | ✅ DLV3+, SF | ⏳ |
| gen_LANIT | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ⏳ |
| gen_Qwen_Image_Edit | ⏳ | ✅ DLV3+, PSP, SF | ✅ DLV3+, SF | ⏳ |
| gen_stargan_v2 | ✅ DLV3+, PSP, SF | ✅ DLV3+, SF | ✅ DLV3+, PSP, SF | ✅ PSP |
| gen_step1x_new | ⏳ | ✅ DLV3+, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+ |
| gen_step1x_v1p2 | ✅ DLV3+, PSP, SF | ✅ DLV3+, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+ |
| gen_SUSTechGAN | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ⏳ |
| gen_TSIT | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ⏳ | ⏳ |
| gen_UniControl | ✅ PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ⏳ |
| gen_VisualCloze | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ⏳ |
| gen_Weather_Effect_Generator | ✅ DLV3+, PSP | ✅ DLV3+, PSP, SF | ✅ SF | ✅ PSP |
| gen_albumentations_weather | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ SF | ⏳ |
| baseline | ✅ DLV3+, PSP, SF | ⏳ | ✅ PSP, SF | ✅ DLV3+ |
| photometric_distort | ✅ DLV3+, PSP, SF | ⏳ | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP |
| std_autoaugment | ✅ DLV3+, PSP, SF | ⏳ | ✅ DLV3+, PSP, SF | ✅ DLV3+ |
| std_cutmix | ✅ DLV3+, PSP, SF | ⏳ | ✅ DLV3+, PSP, SF | ⏳ |
| std_mixup | ✅ DLV3+, PSP, SF | ⏳ | ✅ DLV3+, PSP, SF | ⏳ |
| std_randaugment | ✅ DLV3+, PSP, SF | ⏳ | ✅ DLV3+, PSP | ✅ DLV3+, PSP |
