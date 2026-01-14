# Training Coverage Report

**Generated:** 2026-01-14 12:30

## Summary

| Status | Count | Percentage |
|--------|------:|----------:|
| ✅ Complete | 318 | 94.6% |
| 🔄 Running | 9 | 2.7% |
| ⏳ Pending (in queue) | 0 | 0.0% |
| ⚠️ Missing (not started) | 9 | 2.7% |
| ❌ Failed | 0 | 0.0% |
| **Total** | **336** | **100%** |

## Per-Dataset Breakdown

### BDD10K
- Complete: 79/84 (94.0%)
- Running: 3
- Pending (in queue): 0
- Missing (not started): 2
- Failed: 0

### IDD-AW
- Complete: 81/84 (96.4%)
- Running: 2
- Pending (in queue): 0
- Missing (not started): 1
- Failed: 0

### MAPILLARYVISTAS
- Complete: 81/84 (96.4%)
- Running: 0
- Pending (in queue): 0
- Missing (not started): 3
- Failed: 0

### OUTSIDE15K
- Complete: 77/84 (91.7%)
- Running: 4
- Pending (in queue): 0
- Missing (not started): 3
- Failed: 0


## Running Configurations

| Strategy | Dataset | Model | User |
|----------|---------|-------|------|
| gen_cyclediffusion | outside15k | DeepLabV3+ (0.5) | mima2416 |
| gen_cyclediffusion | outside15k | PSPNet (0.5) | mima2416 |
| gen_cyclediffusion | outside15k | SegFormer (0.5) | mima2416 |
| gen_flux_kontext | bdd10k | DeepLabV3+ (0.5) | mima2416 |
| gen_flux_kontext | bdd10k | PSPNet (0.5) | mima2416 |
| gen_flux_kontext | bdd10k | SegFormer (0.5) | mima2416 |
| gen_flux_kontext | idd-aw | SegFormer (0.5) | mima2416 |
| gen_flux_kontext | outside15k | SegFormer (0.5) | mima2416 |
| gen_step1x_new | idd-aw | PSPNet (0.5) | mima2416 |

## Pending Configurations (in queue)

*No configurations pending in queue.*


## Missing Configurations (not started)

| Strategy | Dataset | Model |
|----------|---------|-------|
| gen_Weather_Effect_Generator | bdd10k | SegFormer (0.5) |
| std_minimal | bdd10k | SegFormer |
| std_minimal | idd-aw | SegFormer |
| std_minimal | mapillaryvistas | DeepLabV3+ |
| std_minimal | mapillaryvistas | PSPNet |
| std_minimal | mapillaryvistas | SegFormer |
| std_minimal | outside15k | DeepLabV3+ |
| std_minimal | outside15k | PSPNet |
| std_minimal | outside15k | SegFormer |

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
| gen_cyclediffusion | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ⏳ |
| gen_cycleGAN | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_flux_kontext | ⏳ | ✅ DeepLabV3+, PSPNet | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet |
| gen_Img2Img | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_IP2P | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_LANIT | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_Qwen_Image_Edit | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_stargan_v2 | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_step1x_new | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_step1x_v1p2 | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_SUSTechGAN | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_TSIT | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_UniControl | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_VisualCloze | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_Weather_Effect_Generator | ✅ DeepLabV3+, PSPNet | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_albumentations_weather | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| baseline | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| photometric_distort | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| std_minimal | ✅ DeepLabV3+, PSPNet | ✅ DeepLabV3+, PSPNet | ⏳ | ⏳ |
| std_autoaugment | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| std_cutmix | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| std_mixup | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| std_randaugment | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |