# Training Coverage Report

**Generated:** 2026-01-15 13:26

## Summary

| Status | Count | Percentage |
|--------|------:|----------:|
| ✅ Complete | 296 | 88.1% |
| 🔄 Running | 3 | 0.9% |
| ⏳ Pending (in queue) | 37 | 11.0% |
| ⚠️ Missing (not started) | 0 | 0.0% |
| ❌ Failed | 0 | 0.0% |
| **Total** | **336** | **100%** |

## Per-Dataset Breakdown

### BDD10K
- Complete: 82/84 (97.6%)
- Running: 0
- Pending (in queue): 2
- Missing (not started): 0
- Failed: 0

### IDD-AW
- Complete: 52/84 (61.9%)
- Running: 3
- Pending (in queue): 29
- Missing (not started): 0
- Failed: 0

### MAPILLARYVISTAS
- Complete: 81/84 (96.4%)
- Running: 0
- Pending (in queue): 3
- Missing (not started): 0
- Failed: 0

### OUTSIDE15K
- Complete: 81/84 (96.4%)
- Running: 0
- Pending (in queue): 3
- Missing (not started): 0
- Failed: 0


## Running Configurations

| Strategy | Dataset | Model | User |
|----------|---------|-------|------|
| gen_IP2P | idd-aw | DeepLabV3+ (0.5) | mima2416 |
| gen_IP2P | idd-aw | PSPNet (0.5) | mima2416 |
| gen_IP2P | idd-aw | SegFormer (0.5) | mima2416 |

## Pending Configurations (in queue)

| Strategy | Dataset | Model | User |
|----------|---------|-------|------|
| baseline | idd-aw | DeepLabV3+ | mima2416 |
| baseline | idd-aw | PSPNet | mima2416 |
| baseline | idd-aw | SegFormer | mima2416 |
| gen_Attribute_Hallucination | idd-aw | DeepLabV3+ (0.5) | mima2416 |
| gen_Attribute_Hallucination | idd-aw | PSPNet (0.5) | mima2416 |
| gen_Attribute_Hallucination | idd-aw | SegFormer (0.5) | mima2416 |
| gen_CNetSeg | idd-aw | DeepLabV3+ (0.5) | mima2416 |
| gen_CNetSeg | idd-aw | PSPNet (0.5) | mima2416 |
| gen_CNetSeg | idd-aw | SegFormer (0.5) | mima2416 |
| gen_CUT | idd-aw | DeepLabV3+ (0.5) | mima2416 |
| gen_CUT | idd-aw | PSPNet (0.5) | mima2416 |
| gen_CUT | idd-aw | SegFormer (0.5) | mima2416 |
| gen_Weather_Effect_Generator | bdd10k | SegFormer (0.5) | mima2416 |
| gen_step1x_new | idd-aw | PSPNet (0.5) | mima2416 |
| photometric_distort | idd-aw | DeepLabV3+ | mima2416 |
| photometric_distort | idd-aw | PSPNet | mima2416 |
| photometric_distort | idd-aw | SegFormer | mima2416 |
| std_autoaugment | idd-aw | DeepLabV3+ | mima2416 |
| std_autoaugment | idd-aw | PSPNet | mima2416 |
| std_autoaugment | idd-aw | SegFormer | mima2416 |
| std_cutmix | idd-aw | DeepLabV3+ | mima2416 |
| std_cutmix | idd-aw | PSPNet | mima2416 |
| std_cutmix | idd-aw | SegFormer | mima2416 |
| std_minimal | bdd10k | SegFormer | mima2416 |
| std_minimal | idd-aw | SegFormer | mima2416 |
| std_minimal | mapillaryvistas | DeepLabV3+ | mima2416 |
| std_minimal | mapillaryvistas | PSPNet | mima2416 |
| std_minimal | mapillaryvistas | SegFormer | mima2416 |
| std_minimal | outside15k | DeepLabV3+ | mima2416 |
| std_minimal | outside15k | PSPNet | mima2416 |
| std_minimal | outside15k | SegFormer | mima2416 |
| std_mixup | idd-aw | DeepLabV3+ | mima2416 |
| std_mixup | idd-aw | PSPNet | mima2416 |
| std_mixup | idd-aw | SegFormer | mima2416 |
| std_randaugment | idd-aw | DeepLabV3+ | mima2416 |
| std_randaugment | idd-aw | PSPNet | mima2416 |
| std_randaugment | idd-aw | SegFormer | mima2416 |

## Missing Configurations (not started)

*No configurations missing.*


## Failed Configurations

*No failed configurations.*


## Complete Configurations

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k |
|----------|--------|--------|-----------------|------------|
| gen_Attribute_Hallucination | ✅ DeepLabV3+, PSPNet, SegFormer | ⏳ | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_augmenters | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_automold | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_CNetSeg | ✅ DeepLabV3+, PSPNet, SegFormer | ⏳ | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_CUT | ✅ DeepLabV3+, PSPNet, SegFormer | ⏳ | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_cyclediffusion | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_cycleGAN | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_flux_kontext | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_Img2Img | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| gen_IP2P | ✅ DeepLabV3+, PSPNet, SegFormer | ⏳ | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
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
| baseline | ✅ DeepLabV3+, PSPNet, SegFormer | ⏳ | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| photometric_distort | ✅ DeepLabV3+, PSPNet, SegFormer | ⏳ | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| std_minimal | ✅ DeepLabV3+, PSPNet | ✅ DeepLabV3+, PSPNet | ⏳ | ⏳ |
| std_autoaugment | ✅ DeepLabV3+, PSPNet, SegFormer | ⏳ | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| std_cutmix | ✅ DeepLabV3+, PSPNet, SegFormer | ⏳ | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| std_mixup | ✅ DeepLabV3+, PSPNet, SegFormer | ⏳ | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |
| std_randaugment | ✅ DeepLabV3+, PSPNet, SegFormer | ⏳ | ✅ DeepLabV3+, PSPNet, SegFormer | ✅ DeepLabV3+, PSPNet, SegFormer |