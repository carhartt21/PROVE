# Training Coverage Report

**Generated:** 2026-01-13 16:38

## Summary

| Status | Count | Percentage |
|--------|------:|----------:|
| ✅ Complete | 285 | 84.8% |
| 🔄 Running | 8 | 2.4% |
| ⏳ Pending (in queue) | 35 | 10.4% |
| ⚠️ Missing (not started) | 8 | 2.4% |
| ❌ Failed | 0 | 0.0% |
| **Total** | **336** | **100%** |

## Per-Dataset Breakdown

### BDD10K
- Complete: 69/84 (82.1%)
- Running: 4
- Pending (in queue): 10
- Missing (not started): 1
- Failed: 0

### IDD-AW
- Complete: 77/84 (91.7%)
- Running: 1
- Pending (in queue): 5
- Missing (not started): 1
- Failed: 0

### MAPILLARYVISTAS
- Complete: 79/84 (94.0%)
- Running: 2
- Pending (in queue): 0
- Missing (not started): 3
- Failed: 0

### OUTSIDE15K
- Complete: 60/84 (71.4%)
- Running: 1
- Pending (in queue): 20
- Missing (not started): 3
- Failed: 0


## Running Configurations

| Strategy | Dataset | Model | User |
|----------|---------|-------|------|
| gen_cyclediffusion | mapillaryvistas | PSPNet (0.5) | chge7185 |
| gen_cyclediffusion | mapillaryvistas | SegFormer (0.5) | chge7185 |
| gen_stargan_v2 | outside15k | SegFormer (0.5) | chge7185 |
| gen_step1x_new | idd-aw | PSPNet (0.5) | mima2416 |
| std_autoaugment | bdd10k | SegFormer | chge7185 |
| std_cutmix | bdd10k | SegFormer | chge7185 |
| std_mixup | bdd10k | SegFormer | chge7185 |
| std_randaugment | bdd10k | SegFormer | chge7185 |

## Pending Configurations (in queue)

| Strategy | Dataset | Model | User |
|----------|---------|-------|------|
| gen_Qwen_Image_Edit | bdd10k | DeepLabV3+ (0.5) | mima2416 |
| gen_Qwen_Image_Edit | bdd10k | PSPNet (0.5) | mima2416 |
| gen_Qwen_Image_Edit | bdd10k | SegFormer (0.5) | mima2416 |
| gen_Weather_Effect_Generator | bdd10k | SegFormer (0.5) | mima2416 |
| gen_cyclediffusion | outside15k | DeepLabV3+ (0.5) | mima2416 |
| gen_cyclediffusion | outside15k | PSPNet (0.5) | mima2416 |
| gen_cyclediffusion | outside15k | SegFormer (0.5) | mima2416 |
| gen_flux_kontext | bdd10k | DeepLabV3+ (0.5) | mima2416 |
| gen_flux_kontext | bdd10k | PSPNet (0.5) | mima2416 |
| gen_flux_kontext | bdd10k | SegFormer (0.5) | mima2416 |
| gen_flux_kontext | idd-aw | DeepLabV3+ (0.5) | mima2416 |
| gen_flux_kontext | idd-aw | PSPNet (0.5) | mima2416 |
| gen_flux_kontext | idd-aw | SegFormer (0.5) | mima2416 |
| gen_flux_kontext | outside15k | DeepLabV3+ (0.5) | mima2416 |
| gen_flux_kontext | outside15k | PSPNet (0.5) | mima2416 |
| gen_flux_kontext | outside15k | SegFormer (0.5) | mima2416 |
| gen_stargan_v2 | idd-aw | PSPNet (0.5) | mima2416 |
| gen_step1x_new | bdd10k | DeepLabV3+ (0.5) | mima2416 |
| gen_step1x_new | bdd10k | PSPNet (0.5) | mima2416 |
| gen_step1x_new | bdd10k | SegFormer (0.5) | mima2416 |
| gen_step1x_new | outside15k | PSPNet (0.5) | chge7185 |
| gen_step1x_new | outside15k | SegFormer (0.5) | chge7185 |
| gen_step1x_v1p2 | idd-aw | PSPNet (0.5) | mima2416 |
| gen_step1x_v1p2 | outside15k | PSPNet (0.5) | chge7185 |
| gen_step1x_v1p2 | outside15k | SegFormer (0.5) | chge7185 |
| photometric_distort | outside15k | SegFormer | chge7185 |
| std_autoaugment | outside15k | PSPNet | chge7185 |
| std_autoaugment | outside15k | SegFormer | chge7185 |
| std_cutmix | outside15k | DeepLabV3+ | chge7185 |
| std_cutmix | outside15k | PSPNet | chge7185 |
| std_cutmix | outside15k | SegFormer | chge7185 |
| std_mixup | outside15k | DeepLabV3+ | chge7185 |
| std_mixup | outside15k | PSPNet | chge7185 |
| std_mixup | outside15k | SegFormer | chge7185 |
| std_randaugment | outside15k | SegFormer | chge7185 |

## Missing Configurations (not started)

| Strategy | Dataset | Model |
|----------|---------|-------|
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
| gen_cyclediffusion | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+ | ⏳ |
| gen_cycleGAN | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_flux_kontext | ⏳ | ⏳ | ✅ DeepLabV3+, PSPNet, SegFormer | ⏳ |
| gen_Img2Img | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_IP2P | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_LANIT | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_Qwen_Image_Edit | ⏳ | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_stargan_v2 | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet |
| gen_step1x_new | ⏳ | ✅ DeepLabV3+, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+ |
| gen_step1x_v1p2 | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+ |
| gen_SUSTechGAN | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_TSIT | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_UniControl | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_VisualCloze | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_Weather_Effect_Generator | ✅ DeepLabV3+, PSPNet | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_albumentations_weather | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| baseline | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| photometric_distort | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet |
| std_minimal | ✅ DeepLabV3+, PSPNet | ✅ DeepLabV3+, PSPNet | ⏳ | ⏳ |
| std_autoaugment | ✅ DeepLabV3+, PSPNet | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+ |
| std_cutmix | ✅ DeepLabV3+, PSPNet | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ⏳ |
| std_mixup | ✅ DeepLabV3+, PSPNet | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ⏳ |
| std_randaugment | ✅ DeepLabV3+, PSPNet | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet |