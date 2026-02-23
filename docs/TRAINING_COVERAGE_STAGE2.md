# Training Coverage Report

**Generated:** 2026-02-09 21:50

## Summary

| Status | Count | Percentage |
|--------|------:|----------:|
| ✅ Complete | 89 | 14.8% |
| 🔄 Running | 3 | 0.5% |
| ⏳ Pending (in queue) | 31 | 5.2% |
| ⚠️ Missing (not started) | 477 | 79.5% |
| ❌ Failed | 0 | 0.0% |
| **Total** | **600** | **100%** |

## Per-Dataset Breakdown

### BDD10K
- Complete: 24/150 (16.0%)
- Running: 1
- Pending (in queue): 4
- Missing (not started): 121
- Failed: 0

### IDD-AW
- Complete: 23/150 (15.3%)
- Running: 2
- Pending (in queue): 4
- Missing (not started): 121
- Failed: 0

### MAPILLARYVISTAS
- Complete: 21/150 (14.0%)
- Running: 0
- Pending (in queue): 11
- Missing (not started): 118
- Failed: 0

### OUTSIDE15K
- Complete: 21/150 (14.0%)
- Running: 0
- Pending (in queue): 12
- Missing (not started): 117
- Failed: 0


## Running Configurations

| Strategy | Dataset | Model | User |
|----------|---------|-------|------|
| gen_VisualCloze | idd-aw | Mask2Former (0.5) | ${USER} |
| gen_albumentations_weather | bdd10k | Mask2Former (0.5) | ${USER} |
| gen_albumentations_weather | idd-aw | Mask2Former (0.5) | ${USER} |

## Pending Configurations (in queue)

| Strategy | Dataset | Model | User |
|----------|---------|-------|------|
| gen_Attribute_Hallucination | mapillaryvistas | Mask2Former (0.5) | ${USER} |
| gen_IP2P | mapillaryvistas | Mask2Former (0.5) | ${USER} |
| gen_IP2P | outside15k | Mask2Former (0.5) | ${USER} |
| gen_Img2Img | bdd10k | PSPNet (0.5) | ${USER} |
| gen_Img2Img | bdd10k | SegNeXt (0.5) | ${USER} |
| gen_Img2Img | idd-aw | PSPNet (0.5) | ${USER} |
| gen_Img2Img | idd-aw | SegNeXt (0.5) | ${USER} |
| gen_Img2Img | mapillaryvistas | PSPNet (0.5) | ${USER} |
| gen_Img2Img | outside15k | PSPNet (0.5) | ${USER} |
| gen_Img2Img | outside15k | SegNeXt (0.5) | ${USER} |
| gen_SUSTechGAN | mapillaryvistas | Mask2Former (0.5) | ${USER} |
| gen_SUSTechGAN | outside15k | Mask2Former (0.5) | ${USER} |
| gen_UniControl | bdd10k | PSPNet (0.5) | ${USER} |
| gen_UniControl | bdd10k | SegNeXt (0.5) | ${USER} |
| gen_UniControl | idd-aw | PSPNet (0.5) | ${USER} |
| gen_UniControl | idd-aw | SegNeXt (0.5) | ${USER} |
| gen_UniControl | mapillaryvistas | PSPNet (0.5) | ${USER} |
| gen_UniControl | outside15k | PSPNet (0.5) | ${USER} |
| gen_UniControl | outside15k | SegNeXt (0.5) | ${USER} |
| gen_VisualCloze | mapillaryvistas | Mask2Former (0.5) | ${USER} |
| gen_VisualCloze | outside15k | Mask2Former (0.5) | ${USER} |
| gen_albumentations_weather | mapillaryvistas | Mask2Former (0.5) | ${USER} |
| gen_albumentations_weather | mapillaryvistas | PSPNet (0.5) | ${USER} |
| gen_albumentations_weather | mapillaryvistas | SegNeXt (0.5) | ${USER} |
| gen_albumentations_weather | outside15k | Mask2Former (0.5) | ${USER} |
| gen_albumentations_weather | outside15k | PSPNet (0.5) | ${USER} |
| gen_albumentations_weather | outside15k | SegNeXt (0.5) | ${USER} |
| gen_automold | mapillaryvistas | Mask2Former (0.5) | ${USER} |
| gen_automold | outside15k | Mask2Former (0.5) | ${USER} |
| gen_step1x_v1p2 | mapillaryvistas | Mask2Former (0.5) | ${USER} |
| gen_step1x_v1p2 | outside15k | Mask2Former (0.5) | ${USER} |

## Missing Configurations (not started)

| Strategy | Dataset | Model |
|----------|---------|-------|
| baseline | bdd10k | HRNet |
| baseline | idd-aw | HRNet |
| baseline | idd-aw | PSPNet |
| baseline | mapillaryvistas | DeepLabV3+ |
| baseline | mapillaryvistas | HRNet |
| baseline | mapillaryvistas | Mask2Former |
| baseline | outside15k | DeepLabV3+ |
| baseline | outside15k | HRNet |
| baseline | outside15k | Mask2Former |
| gen_Attribute_Hallucination | bdd10k | DeepLabV3+ (0.5) |
| gen_Attribute_Hallucination | bdd10k | HRNet (0.5) |
| gen_Attribute_Hallucination | bdd10k | Mask2Former (0.5) |
| gen_Attribute_Hallucination | bdd10k | PSPNet (0.5) |
| gen_Attribute_Hallucination | bdd10k | SegNeXt (0.5) |
| gen_Attribute_Hallucination | idd-aw | DeepLabV3+ (0.5) |
| gen_Attribute_Hallucination | idd-aw | HRNet (0.5) |
| gen_Attribute_Hallucination | idd-aw | Mask2Former (0.5) |
| gen_Attribute_Hallucination | idd-aw | PSPNet (0.5) |
| gen_Attribute_Hallucination | idd-aw | SegNeXt (0.5) |
| gen_Attribute_Hallucination | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_Attribute_Hallucination | mapillaryvistas | HRNet (0.5) |
| gen_Attribute_Hallucination | mapillaryvistas | PSPNet (0.5) |
| gen_Attribute_Hallucination | mapillaryvistas | SegNeXt (0.5) |
| gen_Attribute_Hallucination | outside15k | DeepLabV3+ (0.5) |
| gen_Attribute_Hallucination | outside15k | HRNet (0.5) |
| gen_Attribute_Hallucination | outside15k | Mask2Former (0.5) |
| gen_Attribute_Hallucination | outside15k | PSPNet (0.5) |
| gen_Attribute_Hallucination | outside15k | SegNeXt (0.5) |
| gen_CNetSeg | bdd10k | DeepLabV3+ (0.5) |
| gen_CNetSeg | bdd10k | HRNet (0.5) |
| gen_CNetSeg | bdd10k | Mask2Former (0.5) |
| gen_CNetSeg | bdd10k | PSPNet (0.5) |
| gen_CNetSeg | bdd10k | SegFormer (0.5) |
| gen_CNetSeg | bdd10k | SegNeXt (0.5) |
| gen_CNetSeg | idd-aw | DeepLabV3+ (0.5) |
| gen_CNetSeg | idd-aw | HRNet (0.5) |
| gen_CNetSeg | idd-aw | Mask2Former (0.5) |
| gen_CNetSeg | idd-aw | PSPNet (0.5) |
| gen_CNetSeg | idd-aw | SegFormer (0.5) |
| gen_CNetSeg | idd-aw | SegNeXt (0.5) |
| gen_CNetSeg | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_CNetSeg | mapillaryvistas | HRNet (0.5) |
| gen_CNetSeg | mapillaryvistas | Mask2Former (0.5) |
| gen_CNetSeg | mapillaryvistas | PSPNet (0.5) |
| gen_CNetSeg | mapillaryvistas | SegFormer (0.5) |
| gen_CNetSeg | mapillaryvistas | SegNeXt (0.5) |
| gen_CNetSeg | outside15k | DeepLabV3+ (0.5) |
| gen_CNetSeg | outside15k | HRNet (0.5) |
| gen_CNetSeg | outside15k | Mask2Former (0.5) |
| gen_CNetSeg | outside15k | PSPNet (0.5) |
| gen_CNetSeg | outside15k | SegFormer (0.5) |
| gen_CNetSeg | outside15k | SegNeXt (0.5) |
| gen_CUT | bdd10k | DeepLabV3+ (0.5) |
| gen_CUT | bdd10k | HRNet (0.5) |
| gen_CUT | bdd10k | Mask2Former (0.5) |
| gen_CUT | bdd10k | PSPNet (0.5) |
| gen_CUT | bdd10k | SegNeXt (0.5) |
| gen_CUT | idd-aw | DeepLabV3+ (0.5) |
| gen_CUT | idd-aw | HRNet (0.5) |
| gen_CUT | idd-aw | Mask2Former (0.5) |
| gen_CUT | idd-aw | PSPNet (0.5) |
| gen_CUT | idd-aw | SegNeXt (0.5) |
| gen_CUT | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_CUT | mapillaryvistas | HRNet (0.5) |
| gen_CUT | mapillaryvistas | Mask2Former (0.5) |
| gen_CUT | mapillaryvistas | PSPNet (0.5) |
| gen_CUT | mapillaryvistas | SegNeXt (0.5) |
| gen_CUT | outside15k | DeepLabV3+ (0.5) |
| gen_CUT | outside15k | HRNet (0.5) |
| gen_CUT | outside15k | Mask2Former (0.5) |
| gen_CUT | outside15k | PSPNet (0.5) |
| gen_CUT | outside15k | SegNeXt (0.5) |
| gen_IP2P | bdd10k | DeepLabV3+ (0.5) |
| gen_IP2P | bdd10k | HRNet (0.5) |
| gen_IP2P | bdd10k | Mask2Former (0.5) |
| gen_IP2P | bdd10k | PSPNet (0.5) |
| gen_IP2P | bdd10k | SegNeXt (0.5) |
| gen_IP2P | idd-aw | DeepLabV3+ (0.5) |
| gen_IP2P | idd-aw | HRNet (0.5) |
| gen_IP2P | idd-aw | Mask2Former (0.5) |
| gen_IP2P | idd-aw | PSPNet (0.5) |
| gen_IP2P | idd-aw | SegNeXt (0.5) |
| gen_IP2P | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_IP2P | mapillaryvistas | HRNet (0.5) |
| gen_IP2P | mapillaryvistas | PSPNet (0.5) |
| gen_IP2P | mapillaryvistas | SegNeXt (0.5) |
| gen_IP2P | outside15k | DeepLabV3+ (0.5) |
| gen_IP2P | outside15k | HRNet (0.5) |
| gen_IP2P | outside15k | PSPNet (0.5) |
| gen_IP2P | outside15k | SegNeXt (0.5) |
| gen_Img2Img | bdd10k | DeepLabV3+ (0.5) |
| gen_Img2Img | bdd10k | HRNet (0.5) |
| gen_Img2Img | bdd10k | Mask2Former (0.5) |
| gen_Img2Img | idd-aw | DeepLabV3+ (0.5) |
| gen_Img2Img | idd-aw | HRNet (0.5) |
| gen_Img2Img | idd-aw | Mask2Former (0.5) |
| gen_Img2Img | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_Img2Img | mapillaryvistas | HRNet (0.5) |
| gen_Img2Img | mapillaryvistas | Mask2Former (0.5) |
| gen_Img2Img | mapillaryvistas | SegNeXt (0.5) |
| gen_Img2Img | outside15k | DeepLabV3+ (0.5) |
| gen_Img2Img | outside15k | HRNet (0.5) |
| gen_Img2Img | outside15k | Mask2Former (0.5) |
| gen_LANIT | bdd10k | DeepLabV3+ (0.5) |
| gen_LANIT | bdd10k | HRNet (0.5) |
| gen_LANIT | bdd10k | Mask2Former (0.5) |
| gen_LANIT | bdd10k | PSPNet (0.5) |
| gen_LANIT | bdd10k | SegNeXt (0.5) |
| gen_LANIT | idd-aw | DeepLabV3+ (0.5) |
| gen_LANIT | idd-aw | HRNet (0.5) |
| gen_LANIT | idd-aw | Mask2Former (0.5) |
| gen_LANIT | idd-aw | PSPNet (0.5) |
| gen_LANIT | idd-aw | SegNeXt (0.5) |
| gen_LANIT | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_LANIT | mapillaryvistas | HRNet (0.5) |
| gen_LANIT | mapillaryvistas | Mask2Former (0.5) |
| gen_LANIT | mapillaryvistas | PSPNet (0.5) |
| gen_LANIT | mapillaryvistas | SegNeXt (0.5) |
| gen_LANIT | outside15k | DeepLabV3+ (0.5) |
| gen_LANIT | outside15k | HRNet (0.5) |
| gen_LANIT | outside15k | Mask2Former (0.5) |
| gen_LANIT | outside15k | PSPNet (0.5) |
| gen_LANIT | outside15k | SegNeXt (0.5) |
| gen_Qwen_Image_Edit | bdd10k | DeepLabV3+ (0.5) |
| gen_Qwen_Image_Edit | bdd10k | HRNet (0.5) |
| gen_Qwen_Image_Edit | bdd10k | Mask2Former (0.5) |
| gen_Qwen_Image_Edit | bdd10k | PSPNet (0.5) |
| gen_Qwen_Image_Edit | bdd10k | SegFormer (0.5) |
| gen_Qwen_Image_Edit | bdd10k | SegNeXt (0.5) |
| gen_Qwen_Image_Edit | idd-aw | DeepLabV3+ (0.5) |
| gen_Qwen_Image_Edit | idd-aw | HRNet (0.5) |
| gen_Qwen_Image_Edit | idd-aw | Mask2Former (0.5) |
| gen_Qwen_Image_Edit | idd-aw | PSPNet (0.5) |
| gen_Qwen_Image_Edit | idd-aw | SegFormer (0.5) |
| gen_Qwen_Image_Edit | idd-aw | SegNeXt (0.5) |
| gen_Qwen_Image_Edit | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_Qwen_Image_Edit | mapillaryvistas | HRNet (0.5) |
| gen_Qwen_Image_Edit | mapillaryvistas | Mask2Former (0.5) |
| gen_Qwen_Image_Edit | mapillaryvistas | PSPNet (0.5) |
| gen_Qwen_Image_Edit | mapillaryvistas | SegFormer (0.5) |
| gen_Qwen_Image_Edit | mapillaryvistas | SegNeXt (0.5) |
| gen_Qwen_Image_Edit | outside15k | DeepLabV3+ (0.5) |
| gen_Qwen_Image_Edit | outside15k | HRNet (0.5) |
| gen_Qwen_Image_Edit | outside15k | Mask2Former (0.5) |
| gen_Qwen_Image_Edit | outside15k | PSPNet (0.5) |
| gen_Qwen_Image_Edit | outside15k | SegFormer (0.5) |
| gen_Qwen_Image_Edit | outside15k | SegNeXt (0.5) |
| gen_SUSTechGAN | bdd10k | DeepLabV3+ (0.5) |
| gen_SUSTechGAN | bdd10k | HRNet (0.5) |
| gen_SUSTechGAN | bdd10k | Mask2Former (0.5) |
| gen_SUSTechGAN | bdd10k | PSPNet (0.5) |
| gen_SUSTechGAN | bdd10k | SegNeXt (0.5) |
| gen_SUSTechGAN | idd-aw | DeepLabV3+ (0.5) |
| gen_SUSTechGAN | idd-aw | HRNet (0.5) |
| gen_SUSTechGAN | idd-aw | Mask2Former (0.5) |
| gen_SUSTechGAN | idd-aw | PSPNet (0.5) |
| gen_SUSTechGAN | idd-aw | SegNeXt (0.5) |
| gen_SUSTechGAN | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_SUSTechGAN | mapillaryvistas | HRNet (0.5) |
| gen_SUSTechGAN | mapillaryvistas | PSPNet (0.5) |
| gen_SUSTechGAN | mapillaryvistas | SegNeXt (0.5) |
| gen_SUSTechGAN | outside15k | DeepLabV3+ (0.5) |
| gen_SUSTechGAN | outside15k | HRNet (0.5) |
| gen_SUSTechGAN | outside15k | PSPNet (0.5) |
| gen_SUSTechGAN | outside15k | SegNeXt (0.5) |
| gen_TSIT | bdd10k | DeepLabV3+ (0.5) |
| gen_TSIT | bdd10k | HRNet (0.5) |
| gen_TSIT | bdd10k | Mask2Former (0.5) |
| gen_TSIT | bdd10k | PSPNet (0.5) |
| gen_TSIT | bdd10k | SegFormer (0.5) |
| gen_TSIT | bdd10k | SegNeXt (0.5) |
| gen_TSIT | idd-aw | DeepLabV3+ (0.5) |
| gen_TSIT | idd-aw | HRNet (0.5) |
| gen_TSIT | idd-aw | Mask2Former (0.5) |
| gen_TSIT | idd-aw | PSPNet (0.5) |
| gen_TSIT | idd-aw | SegFormer (0.5) |
| gen_TSIT | idd-aw | SegNeXt (0.5) |
| gen_TSIT | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_TSIT | mapillaryvistas | HRNet (0.5) |
| gen_TSIT | mapillaryvistas | Mask2Former (0.5) |
| gen_TSIT | mapillaryvistas | PSPNet (0.5) |
| gen_TSIT | mapillaryvistas | SegFormer (0.5) |
| gen_TSIT | mapillaryvistas | SegNeXt (0.5) |
| gen_TSIT | outside15k | DeepLabV3+ (0.5) |
| gen_TSIT | outside15k | HRNet (0.5) |
| gen_TSIT | outside15k | Mask2Former (0.5) |
| gen_TSIT | outside15k | PSPNet (0.5) |
| gen_TSIT | outside15k | SegFormer (0.5) |
| gen_TSIT | outside15k | SegNeXt (0.5) |
| gen_UniControl | bdd10k | DeepLabV3+ (0.5) |
| gen_UniControl | bdd10k | HRNet (0.5) |
| gen_UniControl | bdd10k | Mask2Former (0.5) |
| gen_UniControl | idd-aw | DeepLabV3+ (0.5) |
| gen_UniControl | idd-aw | HRNet (0.5) |
| gen_UniControl | idd-aw | Mask2Former (0.5) |
| gen_UniControl | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_UniControl | mapillaryvistas | HRNet (0.5) |
| gen_UniControl | mapillaryvistas | Mask2Former (0.5) |
| gen_UniControl | mapillaryvistas | SegNeXt (0.5) |
| gen_UniControl | outside15k | DeepLabV3+ (0.5) |
| gen_UniControl | outside15k | HRNet (0.5) |
| gen_UniControl | outside15k | Mask2Former (0.5) |
| gen_VisualCloze | bdd10k | DeepLabV3+ (0.5) |
| gen_VisualCloze | bdd10k | HRNet (0.5) |
| gen_VisualCloze | bdd10k | Mask2Former (0.5) |
| gen_VisualCloze | bdd10k | PSPNet (0.5) |
| gen_VisualCloze | bdd10k | SegNeXt (0.5) |
| gen_VisualCloze | idd-aw | DeepLabV3+ (0.5) |
| gen_VisualCloze | idd-aw | HRNet (0.5) |
| gen_VisualCloze | idd-aw | PSPNet (0.5) |
| gen_VisualCloze | idd-aw | SegNeXt (0.5) |
| gen_VisualCloze | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_VisualCloze | mapillaryvistas | HRNet (0.5) |
| gen_VisualCloze | mapillaryvistas | PSPNet (0.5) |
| gen_VisualCloze | mapillaryvistas | SegNeXt (0.5) |
| gen_VisualCloze | outside15k | DeepLabV3+ (0.5) |
| gen_VisualCloze | outside15k | HRNet (0.5) |
| gen_VisualCloze | outside15k | PSPNet (0.5) |
| gen_VisualCloze | outside15k | SegNeXt (0.5) |
| gen_Weather_Effect_Generator | bdd10k | DeepLabV3+ (0.5) |
| gen_Weather_Effect_Generator | bdd10k | HRNet (0.5) |
| gen_Weather_Effect_Generator | bdd10k | Mask2Former (0.5) |
| gen_Weather_Effect_Generator | bdd10k | PSPNet (0.5) |
| gen_Weather_Effect_Generator | bdd10k | SegFormer (0.5) |
| gen_Weather_Effect_Generator | bdd10k | SegNeXt (0.5) |
| gen_Weather_Effect_Generator | idd-aw | DeepLabV3+ (0.5) |
| gen_Weather_Effect_Generator | idd-aw | HRNet (0.5) |
| gen_Weather_Effect_Generator | idd-aw | Mask2Former (0.5) |
| gen_Weather_Effect_Generator | idd-aw | PSPNet (0.5) |
| gen_Weather_Effect_Generator | idd-aw | SegFormer (0.5) |
| gen_Weather_Effect_Generator | idd-aw | SegNeXt (0.5) |
| gen_Weather_Effect_Generator | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_Weather_Effect_Generator | mapillaryvistas | HRNet (0.5) |
| gen_Weather_Effect_Generator | mapillaryvistas | Mask2Former (0.5) |
| gen_Weather_Effect_Generator | mapillaryvistas | PSPNet (0.5) |
| gen_Weather_Effect_Generator | mapillaryvistas | SegFormer (0.5) |
| gen_Weather_Effect_Generator | mapillaryvistas | SegNeXt (0.5) |
| gen_Weather_Effect_Generator | outside15k | DeepLabV3+ (0.5) |
| gen_Weather_Effect_Generator | outside15k | HRNet (0.5) |
| gen_Weather_Effect_Generator | outside15k | Mask2Former (0.5) |
| gen_Weather_Effect_Generator | outside15k | PSPNet (0.5) |
| gen_Weather_Effect_Generator | outside15k | SegFormer (0.5) |
| gen_Weather_Effect_Generator | outside15k | SegNeXt (0.5) |
| gen_albumentations_weather | bdd10k | DeepLabV3+ (0.5) |
| gen_albumentations_weather | bdd10k | HRNet (0.5) |
| gen_albumentations_weather | bdd10k | PSPNet (0.5) |
| gen_albumentations_weather | bdd10k | SegNeXt (0.5) |
| gen_albumentations_weather | idd-aw | DeepLabV3+ (0.5) |
| gen_albumentations_weather | idd-aw | HRNet (0.5) |
| gen_albumentations_weather | idd-aw | PSPNet (0.5) |
| gen_albumentations_weather | idd-aw | SegNeXt (0.5) |
| gen_albumentations_weather | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_albumentations_weather | mapillaryvistas | HRNet (0.5) |
| gen_albumentations_weather | outside15k | DeepLabV3+ (0.5) |
| gen_albumentations_weather | outside15k | HRNet (0.5) |
| gen_augmenters | bdd10k | DeepLabV3+ (0.5) |
| gen_augmenters | bdd10k | HRNet (0.5) |
| gen_augmenters | bdd10k | Mask2Former (0.5) |
| gen_augmenters | bdd10k | PSPNet (0.5) |
| gen_augmenters | bdd10k | SegFormer (0.5) |
| gen_augmenters | bdd10k | SegNeXt (0.5) |
| gen_augmenters | idd-aw | DeepLabV3+ (0.5) |
| gen_augmenters | idd-aw | HRNet (0.5) |
| gen_augmenters | idd-aw | Mask2Former (0.5) |
| gen_augmenters | idd-aw | PSPNet (0.5) |
| gen_augmenters | idd-aw | SegFormer (0.5) |
| gen_augmenters | idd-aw | SegNeXt (0.5) |
| gen_augmenters | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_augmenters | mapillaryvistas | HRNet (0.5) |
| gen_augmenters | mapillaryvistas | Mask2Former (0.5) |
| gen_augmenters | mapillaryvistas | PSPNet (0.5) |
| gen_augmenters | mapillaryvistas | SegFormer (0.5) |
| gen_augmenters | mapillaryvistas | SegNeXt (0.5) |
| gen_augmenters | outside15k | DeepLabV3+ (0.5) |
| gen_augmenters | outside15k | HRNet (0.5) |
| gen_augmenters | outside15k | Mask2Former (0.5) |
| gen_augmenters | outside15k | PSPNet (0.5) |
| gen_augmenters | outside15k | SegFormer (0.5) |
| gen_augmenters | outside15k | SegNeXt (0.5) |
| gen_automold | bdd10k | DeepLabV3+ (0.5) |
| gen_automold | bdd10k | HRNet (0.5) |
| gen_automold | bdd10k | Mask2Former (0.5) |
| gen_automold | bdd10k | PSPNet (0.5) |
| gen_automold | bdd10k | SegNeXt (0.5) |
| gen_automold | idd-aw | DeepLabV3+ (0.5) |
| gen_automold | idd-aw | HRNet (0.5) |
| gen_automold | idd-aw | Mask2Former (0.5) |
| gen_automold | idd-aw | PSPNet (0.5) |
| gen_automold | idd-aw | SegNeXt (0.5) |
| gen_automold | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_automold | mapillaryvistas | HRNet (0.5) |
| gen_automold | mapillaryvistas | PSPNet (0.5) |
| gen_automold | mapillaryvistas | SegNeXt (0.5) |
| gen_automold | outside15k | DeepLabV3+ (0.5) |
| gen_automold | outside15k | HRNet (0.5) |
| gen_automold | outside15k | PSPNet (0.5) |
| gen_automold | outside15k | SegNeXt (0.5) |
| gen_cycleGAN | bdd10k | DeepLabV3+ (0.5) |
| gen_cycleGAN | bdd10k | HRNet (0.5) |
| gen_cycleGAN | bdd10k | Mask2Former (0.5) |
| gen_cycleGAN | bdd10k | PSPNet (0.5) |
| gen_cycleGAN | bdd10k | SegNeXt (0.5) |
| gen_cycleGAN | idd-aw | DeepLabV3+ (0.5) |
| gen_cycleGAN | idd-aw | HRNet (0.5) |
| gen_cycleGAN | idd-aw | Mask2Former (0.5) |
| gen_cycleGAN | idd-aw | PSPNet (0.5) |
| gen_cycleGAN | idd-aw | SegNeXt (0.5) |
| gen_cycleGAN | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_cycleGAN | mapillaryvistas | HRNet (0.5) |
| gen_cycleGAN | mapillaryvistas | Mask2Former (0.5) |
| gen_cycleGAN | mapillaryvistas | PSPNet (0.5) |
| gen_cycleGAN | mapillaryvistas | SegNeXt (0.5) |
| gen_cycleGAN | outside15k | DeepLabV3+ (0.5) |
| gen_cycleGAN | outside15k | HRNet (0.5) |
| gen_cycleGAN | outside15k | Mask2Former (0.5) |
| gen_cycleGAN | outside15k | PSPNet (0.5) |
| gen_cycleGAN | outside15k | SegNeXt (0.5) |
| gen_flux_kontext | bdd10k | DeepLabV3+ (0.5) |
| gen_flux_kontext | bdd10k | HRNet (0.5) |
| gen_flux_kontext | bdd10k | Mask2Former (0.5) |
| gen_flux_kontext | bdd10k | PSPNet (0.5) |
| gen_flux_kontext | bdd10k | SegNeXt (0.5) |
| gen_flux_kontext | idd-aw | DeepLabV3+ (0.5) |
| gen_flux_kontext | idd-aw | HRNet (0.5) |
| gen_flux_kontext | idd-aw | Mask2Former (0.5) |
| gen_flux_kontext | idd-aw | PSPNet (0.5) |
| gen_flux_kontext | idd-aw | SegNeXt (0.5) |
| gen_flux_kontext | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_flux_kontext | mapillaryvistas | HRNet (0.5) |
| gen_flux_kontext | mapillaryvistas | Mask2Former (0.5) |
| gen_flux_kontext | mapillaryvistas | PSPNet (0.5) |
| gen_flux_kontext | mapillaryvistas | SegNeXt (0.5) |
| gen_flux_kontext | outside15k | DeepLabV3+ (0.5) |
| gen_flux_kontext | outside15k | HRNet (0.5) |
| gen_flux_kontext | outside15k | Mask2Former (0.5) |
| gen_flux_kontext | outside15k | PSPNet (0.5) |
| gen_flux_kontext | outside15k | SegNeXt (0.5) |
| gen_stargan_v2 | bdd10k | DeepLabV3+ (0.5) |
| gen_stargan_v2 | bdd10k | HRNet (0.5) |
| gen_stargan_v2 | bdd10k | Mask2Former (0.5) |
| gen_stargan_v2 | bdd10k | PSPNet (0.5) |
| gen_stargan_v2 | bdd10k | SegFormer (0.5) |
| gen_stargan_v2 | bdd10k | SegNeXt (0.5) |
| gen_stargan_v2 | idd-aw | DeepLabV3+ (0.5) |
| gen_stargan_v2 | idd-aw | HRNet (0.5) |
| gen_stargan_v2 | idd-aw | Mask2Former (0.5) |
| gen_stargan_v2 | idd-aw | PSPNet (0.5) |
| gen_stargan_v2 | idd-aw | SegFormer (0.5) |
| gen_stargan_v2 | idd-aw | SegNeXt (0.5) |
| gen_stargan_v2 | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_stargan_v2 | mapillaryvistas | HRNet (0.5) |
| gen_stargan_v2 | mapillaryvistas | Mask2Former (0.5) |
| gen_stargan_v2 | mapillaryvistas | PSPNet (0.5) |
| gen_stargan_v2 | mapillaryvistas | SegFormer (0.5) |
| gen_stargan_v2 | mapillaryvistas | SegNeXt (0.5) |
| gen_stargan_v2 | outside15k | DeepLabV3+ (0.5) |
| gen_stargan_v2 | outside15k | HRNet (0.5) |
| gen_stargan_v2 | outside15k | Mask2Former (0.5) |
| gen_stargan_v2 | outside15k | PSPNet (0.5) |
| gen_stargan_v2 | outside15k | SegFormer (0.5) |
| gen_stargan_v2 | outside15k | SegNeXt (0.5) |
| gen_step1x_new | bdd10k | DeepLabV3+ (0.5) |
| gen_step1x_new | bdd10k | HRNet (0.5) |
| gen_step1x_new | bdd10k | Mask2Former (0.5) |
| gen_step1x_new | bdd10k | PSPNet (0.5) |
| gen_step1x_new | bdd10k | SegNeXt (0.5) |
| gen_step1x_new | idd-aw | DeepLabV3+ (0.5) |
| gen_step1x_new | idd-aw | HRNet (0.5) |
| gen_step1x_new | idd-aw | Mask2Former (0.5) |
| gen_step1x_new | idd-aw | PSPNet (0.5) |
| gen_step1x_new | idd-aw | SegNeXt (0.5) |
| gen_step1x_new | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_step1x_new | mapillaryvistas | HRNet (0.5) |
| gen_step1x_new | mapillaryvistas | Mask2Former (0.5) |
| gen_step1x_new | mapillaryvistas | PSPNet (0.5) |
| gen_step1x_new | mapillaryvistas | SegNeXt (0.5) |
| gen_step1x_new | outside15k | DeepLabV3+ (0.5) |
| gen_step1x_new | outside15k | HRNet (0.5) |
| gen_step1x_new | outside15k | Mask2Former (0.5) |
| gen_step1x_new | outside15k | PSPNet (0.5) |
| gen_step1x_new | outside15k | SegNeXt (0.5) |
| gen_step1x_v1p2 | bdd10k | DeepLabV3+ (0.5) |
| gen_step1x_v1p2 | bdd10k | HRNet (0.5) |
| gen_step1x_v1p2 | bdd10k | Mask2Former (0.5) |
| gen_step1x_v1p2 | bdd10k | PSPNet (0.5) |
| gen_step1x_v1p2 | bdd10k | SegNeXt (0.5) |
| gen_step1x_v1p2 | idd-aw | DeepLabV3+ (0.5) |
| gen_step1x_v1p2 | idd-aw | HRNet (0.5) |
| gen_step1x_v1p2 | idd-aw | Mask2Former (0.5) |
| gen_step1x_v1p2 | idd-aw | PSPNet (0.5) |
| gen_step1x_v1p2 | idd-aw | SegNeXt (0.5) |
| gen_step1x_v1p2 | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_step1x_v1p2 | mapillaryvistas | HRNet (0.5) |
| gen_step1x_v1p2 | mapillaryvistas | PSPNet (0.5) |
| gen_step1x_v1p2 | mapillaryvistas | SegNeXt (0.5) |
| gen_step1x_v1p2 | outside15k | DeepLabV3+ (0.5) |
| gen_step1x_v1p2 | outside15k | HRNet (0.5) |
| gen_step1x_v1p2 | outside15k | PSPNet (0.5) |
| gen_step1x_v1p2 | outside15k | SegNeXt (0.5) |
| std_autoaugment | bdd10k | DeepLabV3+ |
| std_autoaugment | bdd10k | HRNet |
| std_autoaugment | idd-aw | DeepLabV3+ |
| std_autoaugment | idd-aw | HRNet |
| std_autoaugment | mapillaryvistas | DeepLabV3+ |
| std_autoaugment | mapillaryvistas | HRNet |
| std_autoaugment | mapillaryvistas | Mask2Former |
| std_autoaugment | outside15k | DeepLabV3+ |
| std_autoaugment | outside15k | HRNet |
| std_autoaugment | outside15k | Mask2Former |
| std_minimal | bdd10k | DeepLabV3+ |
| std_minimal | bdd10k | HRNet |
| std_minimal | bdd10k | Mask2Former |
| std_minimal | bdd10k | PSPNet |
| std_minimal | bdd10k | SegFormer |
| std_minimal | bdd10k | SegNeXt |
| std_minimal | idd-aw | DeepLabV3+ |
| std_minimal | idd-aw | HRNet |
| std_minimal | idd-aw | Mask2Former |
| std_minimal | idd-aw | PSPNet |
| std_minimal | idd-aw | SegFormer |
| std_minimal | idd-aw | SegNeXt |
| std_minimal | mapillaryvistas | DeepLabV3+ |
| std_minimal | mapillaryvistas | HRNet |
| std_minimal | mapillaryvistas | Mask2Former |
| std_minimal | mapillaryvistas | PSPNet |
| std_minimal | mapillaryvistas | SegFormer |
| std_minimal | mapillaryvistas | SegNeXt |
| std_minimal | outside15k | DeepLabV3+ |
| std_minimal | outside15k | HRNet |
| std_minimal | outside15k | Mask2Former |
| std_minimal | outside15k | PSPNet |
| std_minimal | outside15k | SegFormer |
| std_minimal | outside15k | SegNeXt |
| std_photometric_distort | bdd10k | DeepLabV3+ |
| std_photometric_distort | bdd10k | HRNet |
| std_photometric_distort | bdd10k | Mask2Former |
| std_photometric_distort | bdd10k | PSPNet |
| std_photometric_distort | bdd10k | SegFormer |
| std_photometric_distort | bdd10k | SegNeXt |
| std_photometric_distort | idd-aw | DeepLabV3+ |
| std_photometric_distort | idd-aw | HRNet |
| std_photometric_distort | idd-aw | Mask2Former |
| std_photometric_distort | idd-aw | PSPNet |
| std_photometric_distort | idd-aw | SegFormer |
| std_photometric_distort | idd-aw | SegNeXt |
| std_photometric_distort | mapillaryvistas | DeepLabV3+ |
| std_photometric_distort | mapillaryvistas | HRNet |
| std_photometric_distort | mapillaryvistas | Mask2Former |
| std_photometric_distort | mapillaryvistas | PSPNet |
| std_photometric_distort | mapillaryvistas | SegFormer |
| std_photometric_distort | mapillaryvistas | SegNeXt |
| std_photometric_distort | outside15k | DeepLabV3+ |
| std_photometric_distort | outside15k | HRNet |
| std_photometric_distort | outside15k | Mask2Former |
| std_photometric_distort | outside15k | PSPNet |
| std_photometric_distort | outside15k | SegFormer |
| std_photometric_distort | outside15k | SegNeXt |
| std_randaugment | bdd10k | DeepLabV3+ |
| std_randaugment | bdd10k | HRNet |
| std_randaugment | bdd10k | Mask2Former |
| std_randaugment | bdd10k | PSPNet |
| std_randaugment | bdd10k | SegNeXt |
| std_randaugment | idd-aw | DeepLabV3+ |
| std_randaugment | idd-aw | HRNet |
| std_randaugment | idd-aw | Mask2Former |
| std_randaugment | idd-aw | PSPNet |
| std_randaugment | idd-aw | SegNeXt |
| std_randaugment | mapillaryvistas | DeepLabV3+ |
| std_randaugment | mapillaryvistas | HRNet |
| std_randaugment | mapillaryvistas | Mask2Former |
| std_randaugment | mapillaryvistas | PSPNet |
| std_randaugment | mapillaryvistas | SegNeXt |
| std_randaugment | outside15k | DeepLabV3+ |
| std_randaugment | outside15k | HRNet |
| std_randaugment | outside15k | Mask2Former |
| std_randaugment | outside15k | PSPNet |
| std_randaugment | outside15k | SegNeXt |

## Failed Configurations

*No failed configurations.*


## Complete Configurations

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k |
|----------|------ | ------ | --------------- | ----------|
| gen_Attribute_Hallucination | ✅ SegFormer | ✅ SegFormer | ✅ SegFormer | ✅ SegFormer |
| gen_augmenters | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_automold | ✅ SegFormer | ✅ SegFormer | ✅ SegFormer | ✅ SegFormer |
| gen_CNetSeg | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_CUT | ✅ SegFormer | ✅ SegFormer | ✅ SegFormer | ✅ SegFormer |
| gen_cyclediffusion | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_cycleGAN | ✅ SegFormer | ✅ SegFormer | ✅ SegFormer | ✅ SegFormer |
| gen_flux_kontext | ✅ SegFormer | ✅ SegFormer | ✅ SegFormer | ✅ SegFormer |
| gen_Img2Img | ✅ SegFormer | ✅ SegFormer | ✅ SegFormer | ✅ SegFormer |
| gen_IP2P | ✅ SegFormer | ✅ SegFormer | ✅ SegFormer | ✅ SegFormer |
| gen_LANIT | ✅ SegFormer | ✅ SegFormer | ✅ SegFormer | ✅ SegFormer |
| gen_Qwen_Image_Edit | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_stargan_v2 | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_step1x_new | ✅ SegFormer | ✅ SegFormer | ✅ SegFormer | ✅ SegFormer |
| gen_step1x_v1p2 | ✅ SegFormer | ✅ SegFormer | ✅ SegFormer | ✅ SegFormer |
| gen_SUSTechGAN | ✅ SegFormer | ✅ SegFormer | ✅ SegFormer | ✅ SegFormer |
| gen_TSIT | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_UniControl | ✅ SegFormer | ✅ SegFormer | ✅ SegFormer | ✅ SegFormer |
| gen_VisualCloze | ✅ SegFormer | ✅ SegFormer | ✅ SegFormer | ✅ SegFormer |
| gen_Weather_Effect_Generator | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_albumentations_weather | ✅ SegFormer | ✅ SegFormer | ✅ SegFormer | ✅ SegFormer |
| baseline | ✅ DeepLabV3+, Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ DeepLabV3+, Mask2Former, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt |
| std_minimal | ⏳ | ⏳ | ⏳ | ⏳ |
| std_photometric_distort | ⏳ | ⏳ | ⏳ | ⏳ |
| std_autoaugment | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt |
| std_cutmix | ⏳ | ⏳ | ⏳ | ⏳ |
| std_mixup | ⏳ | ⏳ | ⏳ | ⏳ |
| std_randaugment | ✅ SegFormer | ✅ SegFormer | ✅ SegFormer | ✅ SegFormer |