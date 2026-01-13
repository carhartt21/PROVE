# Training Coverage Report

**Generated:** 2026-01-13 14:53

## Summary

| Status | Count | Percentage |
|--------|------:|----------:|
| ✅ Complete | 271 | 83.6% |
| 🔄 Running | 12 | 3.7% |
| ⏳ Pending | 41 | 12.7% |
| ❌ Failed | 0 | 0.0% |
| **Total** | **324** | **100%** |

## Per-Dataset Breakdown

### BDD10K
- Complete: 60/81 (74.1%)
- Running: 9
- Pending: 12
- Failed: 0

### IDD-AW
- Complete: 73/81 (90.1%)
- Running: 3
- Pending: 5
- Failed: 0

### MAPILLARYVISTAS
- Complete: 78/81 (96.3%)
- Running: 0
- Pending: 3
- Failed: 0

### OUTSIDE15K
- Complete: 60/81 (74.1%)
- Running: 0
- Pending: 21
- Failed: 0


## Running Configurations

| Strategy | Dataset | Model |
|----------|---------|-------|
| gen_Weather_Effect_Generator | bdd10k | DeepLabV3+ (0.5) |
| gen_Weather_Effect_Generator | bdd10k | PSPNet (0.5) |
| gen_Weather_Effect_Generator | bdd10k | SegFormer (0.5) |
| gen_step1x_new | idd-aw | DeepLabV3+ (0.5) |
| gen_step1x_new | idd-aw | PSPNet (0.5) |
| gen_step1x_new | idd-aw | SegFormer (0.5) |
| photometric_distort | bdd10k | DeepLabV3+ |
| photometric_distort | bdd10k | PSPNet |
| photometric_distort | bdd10k | SegFormer |
| std_autoaugment | bdd10k | DeepLabV3+ |
| std_autoaugment | bdd10k | PSPNet |
| std_autoaugment | bdd10k | SegFormer |

## Pending Configurations

| Strategy | Dataset | Model |
|----------|---------|-------|
| gen_Qwen_Image_Edit | bdd10k | DeepLabV3+ (0.5) |
| gen_Qwen_Image_Edit | bdd10k | PSPNet (0.5) |
| gen_Qwen_Image_Edit | bdd10k | SegFormer (0.5) |
| gen_cyclediffusion | mapillaryvistas | DeepLabV3+ (0.5) |
| gen_cyclediffusion | mapillaryvistas | PSPNet (0.5) |
| gen_cyclediffusion | mapillaryvistas | SegFormer (0.5) |
| gen_cyclediffusion | outside15k | DeepLabV3+ (0.5) |
| gen_cyclediffusion | outside15k | PSPNet (0.5) |
| gen_cyclediffusion | outside15k | SegFormer (0.5) |
| gen_flux_kontext | bdd10k | DeepLabV3+ (0.5) |
| gen_flux_kontext | bdd10k | PSPNet (0.5) |
| gen_flux_kontext | bdd10k | SegFormer (0.5) |
| gen_flux_kontext | idd-aw | DeepLabV3+ (0.5) |
| gen_flux_kontext | idd-aw | PSPNet (0.5) |
| gen_flux_kontext | idd-aw | SegFormer (0.5) |
| gen_flux_kontext | outside15k | DeepLabV3+ (0.5) |
| gen_flux_kontext | outside15k | PSPNet (0.5) |
| gen_flux_kontext | outside15k | SegFormer (0.5) |
| gen_stargan_v2 | idd-aw | PSPNet (0.5) |
| gen_stargan_v2 | outside15k | SegFormer (0.5) |
| gen_step1x_new | bdd10k | DeepLabV3+ (0.5) |
| gen_step1x_new | bdd10k | PSPNet (0.5) |
| gen_step1x_new | bdd10k | SegFormer (0.5) |
| gen_step1x_new | outside15k | PSPNet (0.5) |
| gen_step1x_new | outside15k | SegFormer (0.5) |
| gen_step1x_v1p2 | idd-aw | PSPNet (0.5) |
| gen_step1x_v1p2 | outside15k | PSPNet (0.5) |
| gen_step1x_v1p2 | outside15k | SegFormer (0.5) |
| photometric_distort | outside15k | SegFormer |
| std_autoaugment | outside15k | PSPNet |
| std_autoaugment | outside15k | SegFormer |
| std_cutmix | bdd10k | SegFormer |
| std_cutmix | outside15k | DeepLabV3+ |
| std_cutmix | outside15k | PSPNet |
| std_cutmix | outside15k | SegFormer |
| std_mixup | bdd10k | SegFormer |
| std_mixup | outside15k | DeepLabV3+ |
| std_mixup | outside15k | PSPNet |
| std_mixup | outside15k | SegFormer |
| std_randaugment | bdd10k | SegFormer |
| std_randaugment | outside15k | SegFormer |

## Failed Configurations

*No failed configurations.*


## Complete Configurations

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k |
|----------|--------|--------|-----------------|------------|
| gen_Attribute_Hallucination | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_augmenters | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_automold | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_CNetSeg | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_CUT | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_cyclediffusion | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ⏳ | ⏳ |
| gen_cycleGAN | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_flux_kontext | ⏳ | ⏳ | ✅ DeepLabV3+, PSPNet, SegFormer | ⏳ |
| gen_Img2Img | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_IP2P | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_LANIT | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_Qwen_Image_Edit | ⏳ | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_stargan_v2 | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet |
| gen_step1x_new | ⏳ | ⏳ | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+ |
| gen_step1x_v1p2 | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+ |
| gen_SUSTechGAN | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_TSIT | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_UniControl | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_VisualCloze | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_Weather_Effect_Generator | ⏳ | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_albumentations_weather | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| baseline | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| photometric_distort | ⏳ | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet |
| std_autoaugment | ⏳ | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+ |
| std_cutmix | ✅ DeepLabV3+, PSPNet | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ⏳ |
| std_mixup | ✅ DeepLabV3+, PSPNet | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ⏳ |
| std_randaugment | ✅ DeepLabV3+, PSPNet | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet |