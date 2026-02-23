# Training Coverage Report

**Generated:** 2026-02-09 21:50

## Summary

| Status | Count | Percentage |
|--------|------:|----------:|
| ✅ Complete | 340 | 50.6% |
| 🔄 Running | 3 | 0.4% |
| ⏳ Pending (in queue) | 26 | 3.9% |
| ⚠️ Missing (not started) | 303 | 45.1% |
| ❌ Failed | 0 | 0.0% |
| **Total** | **672** | **100%** |

## Per-Dataset Breakdown

### BDD10K
- Complete: 90/168 (53.6%)
- Running: 1
- Pending (in queue): 3
- Missing (not started): 74
- Failed: 0

### IDD-AW
- Complete: 97/168 (57.7%)
- Running: 2
- Pending (in queue): 3
- Missing (not started): 66
- Failed: 0

### MAPILLARYVISTAS
- Complete: 77/168 (45.8%)
- Running: 0
- Pending (in queue): 10
- Missing (not started): 81
- Failed: 0

### OUTSIDE15K
- Complete: 76/168 (45.2%)
- Running: 0
- Pending (in queue): 10
- Missing (not started): 82
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
| gen_UniControl | mapillaryvistas | PSPNet (0.5) | ${USER} |
| gen_UniControl | outside15k | PSPNet (0.5) | ${USER} |
| gen_VisualCloze | mapillaryvistas | Mask2Former (0.5) | ${USER} |
| gen_VisualCloze | outside15k | Mask2Former (0.5) | ${USER} |
| gen_albumentations_weather | mapillaryvistas | Mask2Former (0.5) | ${USER} |
| gen_albumentations_weather | outside15k | Mask2Former (0.5) | ${USER} |
| gen_automold | mapillaryvistas | Mask2Former (0.5) | ${USER} |
| gen_automold | outside15k | Mask2Former (0.5) | ${USER} |
| gen_cyclediffusion | bdd10k | Mask2Former (0.5) | ${USER} |
| gen_cyclediffusion | idd-aw | Mask2Former (0.5) | ${USER} |
| gen_cyclediffusion | mapillaryvistas | Mask2Former (0.5) | ${USER} |
| gen_cyclediffusion | outside15k | Mask2Former (0.5) | ${USER} |
| gen_step1x_v1p2 | mapillaryvistas | Mask2Former (0.5) | ${USER} |
| gen_step1x_v1p2 | outside15k | Mask2Former (0.5) | ${USER} |

## Missing Configurations (not started)

| Strategy | Dataset | Model |
|----------|---------|-------|
| baseline | bdd10k | DeepLabV3+ |
| baseline | bdd10k | HRNet |
| baseline | bdd10k | PSPNet |
| baseline | idd-aw | DeepLabV3+ |
| baseline | idd-aw | SegNeXt |
| baseline | mapillaryvistas | DeepLabV3+ |
| baseline | mapillaryvistas | Mask2Former |
| baseline | outside15k | DeepLabV3+ |
| baseline | outside15k | Mask2Former |
| gen_Attribute_Hallucination | bdd10k | DeepLabV3+ (0.5) |
| gen_Attribute_Hallucination | bdd10k | HRNet (0.5) |
| gen_Attribute_Hallucination | idd-aw | DeepLabV3+ (0.5) |
| gen_Attribute_Hallucination | idd-aw | HRNet (0.5) |
| gen_Attribute_Hallucination | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_Attribute_Hallucination | mapillaryvistas | HRNet (0.5) |
| gen_Attribute_Hallucination | outside15k | DeepLabV3+ (0.5) |
| gen_Attribute_Hallucination | outside15k | HRNet (0.5) |
| gen_Attribute_Hallucination | outside15k | Mask2Former (0.5) |
| gen_CNetSeg | bdd10k | DeepLabV3+ (0.5) |
| gen_CNetSeg | bdd10k | HRNet (0.5) |
| gen_CNetSeg | idd-aw | DeepLabV3+ (0.5) |
| gen_CNetSeg | idd-aw | HRNet (0.5) |
| gen_CNetSeg | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_CNetSeg | mapillaryvistas | HRNet (0.5) |
| gen_CNetSeg | mapillaryvistas | Mask2Former (0.5) |
| gen_CNetSeg | outside15k | DeepLabV3+ (0.5) |
| gen_CNetSeg | outside15k | HRNet (0.5) |
| gen_CNetSeg | outside15k | Mask2Former (0.5) |
| gen_CUT | bdd10k | DeepLabV3+ (0.5) |
| gen_CUT | bdd10k | HRNet (0.5) |
| gen_CUT | idd-aw | DeepLabV3+ (0.5) |
| gen_CUT | idd-aw | HRNet (0.5) |
| gen_CUT | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_CUT | mapillaryvistas | HRNet (0.5) |
| gen_CUT | mapillaryvistas | Mask2Former (0.5) |
| gen_CUT | outside15k | DeepLabV3+ (0.5) |
| gen_CUT | outside15k | HRNet (0.5) |
| gen_CUT | outside15k | Mask2Former (0.5) |
| gen_IP2P | bdd10k | DeepLabV3+ (0.5) |
| gen_IP2P | bdd10k | HRNet (0.5) |
| gen_IP2P | idd-aw | DeepLabV3+ (0.5) |
| gen_IP2P | idd-aw | HRNet (0.5) |
| gen_IP2P | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_IP2P | mapillaryvistas | HRNet (0.5) |
| gen_IP2P | outside15k | DeepLabV3+ (0.5) |
| gen_IP2P | outside15k | HRNet (0.5) |
| gen_Img2Img | bdd10k | DeepLabV3+ (0.5) |
| gen_Img2Img | bdd10k | HRNet (0.5) |
| gen_Img2Img | bdd10k | Mask2Former (0.5) |
| gen_Img2Img | idd-aw | DeepLabV3+ (0.5) |
| gen_Img2Img | idd-aw | HRNet (0.5) |
| gen_Img2Img | idd-aw | Mask2Former (0.5) |
| gen_Img2Img | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_Img2Img | mapillaryvistas | HRNet (0.5) |
| gen_Img2Img | mapillaryvistas | Mask2Former (0.5) |
| gen_Img2Img | outside15k | DeepLabV3+ (0.5) |
| gen_Img2Img | outside15k | HRNet (0.5) |
| gen_Img2Img | outside15k | Mask2Former (0.5) |
| gen_LANIT | bdd10k | DeepLabV3+ (0.5) |
| gen_LANIT | bdd10k | HRNet (0.5) |
| gen_LANIT | bdd10k | SegNeXt (0.5) |
| gen_LANIT | idd-aw | DeepLabV3+ (0.5) |
| gen_LANIT | idd-aw | HRNet (0.5) |
| gen_LANIT | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_LANIT | mapillaryvistas | HRNet (0.5) |
| gen_LANIT | mapillaryvistas | Mask2Former (0.5) |
| gen_LANIT | outside15k | DeepLabV3+ (0.5) |
| gen_LANIT | outside15k | HRNet (0.5) |
| gen_LANIT | outside15k | Mask2Former (0.5) |
| gen_Qwen_Image_Edit | bdd10k | DeepLabV3+ (0.5) |
| gen_Qwen_Image_Edit | bdd10k | HRNet (0.5) |
| gen_Qwen_Image_Edit | idd-aw | DeepLabV3+ (0.5) |
| gen_Qwen_Image_Edit | idd-aw | HRNet (0.5) |
| gen_Qwen_Image_Edit | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_Qwen_Image_Edit | mapillaryvistas | HRNet (0.5) |
| gen_Qwen_Image_Edit | mapillaryvistas | Mask2Former (0.5) |
| gen_Qwen_Image_Edit | outside15k | DeepLabV3+ (0.5) |
| gen_Qwen_Image_Edit | outside15k | HRNet (0.5) |
| gen_Qwen_Image_Edit | outside15k | Mask2Former (0.5) |
| gen_SUSTechGAN | bdd10k | DeepLabV3+ (0.5) |
| gen_SUSTechGAN | bdd10k | HRNet (0.5) |
| gen_SUSTechGAN | idd-aw | DeepLabV3+ (0.5) |
| gen_SUSTechGAN | idd-aw | HRNet (0.5) |
| gen_SUSTechGAN | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_SUSTechGAN | mapillaryvistas | HRNet (0.5) |
| gen_SUSTechGAN | outside15k | DeepLabV3+ (0.5) |
| gen_SUSTechGAN | outside15k | HRNet (0.5) |
| gen_TSIT | bdd10k | DeepLabV3+ (0.5) |
| gen_TSIT | bdd10k | HRNet (0.5) |
| gen_TSIT | idd-aw | DeepLabV3+ (0.5) |
| gen_TSIT | idd-aw | HRNet (0.5) |
| gen_TSIT | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_TSIT | mapillaryvistas | HRNet (0.5) |
| gen_TSIT | mapillaryvistas | Mask2Former (0.5) |
| gen_TSIT | outside15k | DeepLabV3+ (0.5) |
| gen_TSIT | outside15k | HRNet (0.5) |
| gen_TSIT | outside15k | Mask2Former (0.5) |
| gen_UniControl | bdd10k | DeepLabV3+ (0.5) |
| gen_UniControl | bdd10k | HRNet (0.5) |
| gen_UniControl | bdd10k | Mask2Former (0.5) |
| gen_UniControl | idd-aw | DeepLabV3+ (0.5) |
| gen_UniControl | idd-aw | HRNet (0.5) |
| gen_UniControl | idd-aw | Mask2Former (0.5) |
| gen_UniControl | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_UniControl | mapillaryvistas | HRNet (0.5) |
| gen_UniControl | mapillaryvistas | Mask2Former (0.5) |
| gen_UniControl | outside15k | DeepLabV3+ (0.5) |
| gen_UniControl | outside15k | HRNet (0.5) |
| gen_UniControl | outside15k | Mask2Former (0.5) |
| gen_VisualCloze | bdd10k | DeepLabV3+ (0.5) |
| gen_VisualCloze | bdd10k | HRNet (0.5) |
| gen_VisualCloze | bdd10k | SegNeXt (0.5) |
| gen_VisualCloze | idd-aw | DeepLabV3+ (0.5) |
| gen_VisualCloze | idd-aw | HRNet (0.5) |
| gen_VisualCloze | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_VisualCloze | mapillaryvistas | HRNet (0.5) |
| gen_VisualCloze | outside15k | DeepLabV3+ (0.5) |
| gen_VisualCloze | outside15k | HRNet (0.5) |
| gen_Weather_Effect_Generator | bdd10k | DeepLabV3+ (0.5) |
| gen_Weather_Effect_Generator | bdd10k | HRNet (0.5) |
| gen_Weather_Effect_Generator | idd-aw | DeepLabV3+ (0.5) |
| gen_Weather_Effect_Generator | idd-aw | HRNet (0.5) |
| gen_Weather_Effect_Generator | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_Weather_Effect_Generator | mapillaryvistas | HRNet (0.5) |
| gen_Weather_Effect_Generator | mapillaryvistas | Mask2Former (0.5) |
| gen_Weather_Effect_Generator | outside15k | DeepLabV3+ (0.5) |
| gen_Weather_Effect_Generator | outside15k | HRNet (0.5) |
| gen_Weather_Effect_Generator | outside15k | Mask2Former (0.5) |
| gen_albumentations_weather | bdd10k | DeepLabV3+ (0.5) |
| gen_albumentations_weather | bdd10k | HRNet (0.5) |
| gen_albumentations_weather | bdd10k | SegNeXt (0.5) |
| gen_albumentations_weather | idd-aw | DeepLabV3+ (0.5) |
| gen_albumentations_weather | idd-aw | HRNet (0.5) |
| gen_albumentations_weather | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_albumentations_weather | mapillaryvistas | HRNet (0.5) |
| gen_albumentations_weather | outside15k | DeepLabV3+ (0.5) |
| gen_albumentations_weather | outside15k | HRNet (0.5) |
| gen_augmenters | bdd10k | DeepLabV3+ (0.5) |
| gen_augmenters | bdd10k | HRNet (0.5) |
| gen_augmenters | idd-aw | DeepLabV3+ (0.5) |
| gen_augmenters | idd-aw | HRNet (0.5) |
| gen_augmenters | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_augmenters | mapillaryvistas | HRNet (0.5) |
| gen_augmenters | mapillaryvistas | Mask2Former (0.5) |
| gen_augmenters | outside15k | DeepLabV3+ (0.5) |
| gen_augmenters | outside15k | HRNet (0.5) |
| gen_augmenters | outside15k | Mask2Former (0.5) |
| gen_automold | bdd10k | DeepLabV3+ (0.5) |
| gen_automold | bdd10k | HRNet (0.5) |
| gen_automold | bdd10k | SegNeXt (0.5) |
| gen_automold | idd-aw | DeepLabV3+ (0.5) |
| gen_automold | idd-aw | HRNet (0.5) |
| gen_automold | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_automold | mapillaryvistas | HRNet (0.5) |
| gen_automold | outside15k | DeepLabV3+ (0.5) |
| gen_automold | outside15k | HRNet (0.5) |
| gen_cycleGAN | bdd10k | DeepLabV3+ (0.5) |
| gen_cycleGAN | bdd10k | HRNet (0.5) |
| gen_cycleGAN | idd-aw | DeepLabV3+ (0.5) |
| gen_cycleGAN | idd-aw | HRNet (0.5) |
| gen_cycleGAN | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_cycleGAN | mapillaryvistas | HRNet (0.5) |
| gen_cycleGAN | mapillaryvistas | Mask2Former (0.5) |
| gen_cycleGAN | outside15k | DeepLabV3+ (0.5) |
| gen_cycleGAN | outside15k | HRNet (0.5) |
| gen_cycleGAN | outside15k | Mask2Former (0.5) |
| gen_cyclediffusion | bdd10k | DeepLabV3+ (0.5) |
| gen_cyclediffusion | bdd10k | HRNet (0.5) |
| gen_cyclediffusion | idd-aw | DeepLabV3+ (0.5) |
| gen_cyclediffusion | idd-aw | HRNet (0.5) |
| gen_cyclediffusion | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_cyclediffusion | mapillaryvistas | HRNet (0.5) |
| gen_cyclediffusion | outside15k | DeepLabV3+ (0.5) |
| gen_cyclediffusion | outside15k | HRNet (0.5) |
| gen_flux_kontext | bdd10k | DeepLabV3+ (0.5) |
| gen_flux_kontext | bdd10k | HRNet (0.5) |
| gen_flux_kontext | bdd10k | SegNeXt (0.5) |
| gen_flux_kontext | idd-aw | DeepLabV3+ (0.5) |
| gen_flux_kontext | idd-aw | HRNet (0.5) |
| gen_flux_kontext | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_flux_kontext | mapillaryvistas | HRNet (0.5) |
| gen_flux_kontext | mapillaryvistas | Mask2Former (0.5) |
| gen_flux_kontext | outside15k | DeepLabV3+ (0.5) |
| gen_flux_kontext | outside15k | HRNet (0.5) |
| gen_flux_kontext | outside15k | Mask2Former (0.5) |
| gen_stargan_v2 | bdd10k | DeepLabV3+ (0.5) |
| gen_stargan_v2 | bdd10k | HRNet (0.5) |
| gen_stargan_v2 | idd-aw | DeepLabV3+ (0.5) |
| gen_stargan_v2 | idd-aw | HRNet (0.5) |
| gen_stargan_v2 | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_stargan_v2 | mapillaryvistas | HRNet (0.5) |
| gen_stargan_v2 | mapillaryvistas | Mask2Former (0.5) |
| gen_stargan_v2 | outside15k | DeepLabV3+ (0.5) |
| gen_stargan_v2 | outside15k | HRNet (0.5) |
| gen_stargan_v2 | outside15k | Mask2Former (0.5) |
| gen_step1x_new | bdd10k | DeepLabV3+ (0.5) |
| gen_step1x_new | bdd10k | HRNet (0.5) |
| gen_step1x_new | bdd10k | SegNeXt (0.5) |
| gen_step1x_new | idd-aw | DeepLabV3+ (0.5) |
| gen_step1x_new | idd-aw | HRNet (0.5) |
| gen_step1x_new | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_step1x_new | mapillaryvistas | HRNet (0.5) |
| gen_step1x_new | mapillaryvistas | Mask2Former (0.5) |
| gen_step1x_new | outside15k | DeepLabV3+ (0.5) |
| gen_step1x_new | outside15k | HRNet (0.5) |
| gen_step1x_new | outside15k | Mask2Former (0.5) |
| gen_step1x_v1p2 | bdd10k | DeepLabV3+ (0.5) |
| gen_step1x_v1p2 | bdd10k | HRNet (0.5) |
| gen_step1x_v1p2 | bdd10k | SegNeXt (0.5) |
| gen_step1x_v1p2 | idd-aw | DeepLabV3+ (0.5) |
| gen_step1x_v1p2 | idd-aw | HRNet (0.5) |
| gen_step1x_v1p2 | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_step1x_v1p2 | mapillaryvistas | HRNet (0.5) |
| gen_step1x_v1p2 | outside15k | DeepLabV3+ (0.5) |
| gen_step1x_v1p2 | outside15k | HRNet (0.5) |
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
| std_cutmix | bdd10k | DeepLabV3+ |
| std_cutmix | bdd10k | HRNet |
| std_cutmix | idd-aw | DeepLabV3+ |
| std_cutmix | idd-aw | HRNet |
| std_cutmix | mapillaryvistas | DeepLabV3+ |
| std_cutmix | mapillaryvistas | HRNet |
| std_cutmix | mapillaryvistas | Mask2Former |
| std_cutmix | outside15k | DeepLabV3+ |
| std_cutmix | outside15k | HRNet |
| std_cutmix | outside15k | Mask2Former |
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
| std_mixup | bdd10k | DeepLabV3+ |
| std_mixup | bdd10k | HRNet |
| std_mixup | idd-aw | DeepLabV3+ |
| std_mixup | idd-aw | HRNet |
| std_mixup | mapillaryvistas | DeepLabV3+ |
| std_mixup | mapillaryvistas | HRNet |
| std_mixup | mapillaryvistas | Mask2Former |
| std_mixup | outside15k | DeepLabV3+ |
| std_mixup | outside15k | HRNet |
| std_mixup | outside15k | Mask2Former |
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
| std_randaugment | idd-aw | DeepLabV3+ |
| std_randaugment | idd-aw | HRNet |
| std_randaugment | mapillaryvistas | DeepLabV3+ |
| std_randaugment | mapillaryvistas | HRNet |
| std_randaugment | mapillaryvistas | Mask2Former |
| std_randaugment | outside15k | DeepLabV3+ |
| std_randaugment | outside15k | HRNet |
| std_randaugment | outside15k | Mask2Former |

## Failed Configurations

*No failed configurations.*


## Complete Configurations

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k |
|----------|------ | ------ | --------------- | ----------|
| gen_Attribute_Hallucination | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt |
| gen_augmenters | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt |
| gen_automold | ✅ Mask2Former, PSPNet, SegFormer | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt |
| gen_CNetSeg | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt |
| gen_CUT | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt |
| gen_cyclediffusion | ✅ PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt |
| gen_cycleGAN | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt |
| gen_flux_kontext | ✅ Mask2Former, PSPNet, SegFormer | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt |
| gen_Img2Img | ✅ SegFormer | ✅ SegFormer | ✅ SegFormer, SegNeXt | ✅ SegFormer |
| gen_IP2P | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt |
| gen_LANIT | ✅ Mask2Former, PSPNet, SegFormer | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt |
| gen_Qwen_Image_Edit | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt |
| gen_stargan_v2 | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt |
| gen_step1x_new | ✅ Mask2Former, PSPNet, SegFormer | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt |
| gen_step1x_v1p2 | ✅ Mask2Former, PSPNet, SegFormer | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt |
| gen_SUSTechGAN | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt |
| gen_TSIT | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt |
| gen_UniControl | ✅ PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt | ✅ SegFormer, SegNeXt | ✅ SegFormer, SegNeXt |
| gen_VisualCloze | ✅ Mask2Former, PSPNet, SegFormer | ✅ PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt |
| gen_Weather_Effect_Generator | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt |
| gen_albumentations_weather | ✅ PSPNet, SegFormer | ✅ PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt |
| baseline | ✅ Mask2Former, SegFormer, SegNeXt | ✅ HRNet, Mask2Former, PSPNet, SegFormer | ✅ HRNet, PSPNet, SegFormer, SegNeXt | ✅ HRNet, PSPNet, SegFormer, SegNeXt |
| std_minimal | ⏳ | ⏳ | ⏳ | ⏳ |
| std_photometric_distort | ⏳ | ⏳ | ⏳ | ⏳ |
| std_autoaugment | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt |
| std_cutmix | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt |
| std_mixup | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt |
| std_randaugment | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ Mask2Former, PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt | ✅ PSPNet, SegFormer, SegNeXt |