# Testing Coverage Report

**Generated:** 2026-02-09 21:41

## Summary

| Status | Count | Percentage |
|--------|------:|----------:|
| ✅ Complete (valid mIoU) | 351 | 97.5% |
| 🔄 Running | 0 | 0.0% |
| ⏳ Pending (in queue) | 2 | 0.6% |
| ⚠️ Buggy (mIoU < 5%) | 0 | 0.0% |
| 🔃 Stale (wrong checkpoint) | 0 | 0.0% |
| ❌ Missing (no results) | 7 | 1.9% |
| **Total** | **360** | **100%** |

## Per-Dataset Breakdown

### BDD10k
- Complete: 95/105 (90.5%)
- Running: 0
- Pending (in queue): 2
- Buggy (mIoU < 5%): 0
- Missing (no results): 5

### IDD-AW
- Complete: 103/105 (98.1%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 0
- Missing (no results): 0

### MapillaryVistas
- Complete: 77/105 (73.3%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 0
- Missing (no results): 1

### OUTSIDE15k
- Complete: 76/105 (72.4%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 0
- Missing (no results): 1

## Running Configurations

*No test jobs currently running.*

## Pending Configurations (in queue)

| Strategy | Dataset | Model |
|----------|---------|-------|
| gen_step1x_new | bdd10k | SegNeXt |
| gen_step1x_v1p2 | bdd10k | SegNeXt |

## Buggy Configurations (need retesting)

*No buggy configurations - all tests have valid mIoU.*

## Stale Configurations (need retesting)

*No stale configurations - all tests use correct checkpoints.*

## Missing Configurations (no test results)

| Strategy | Dataset | Model | Issue |
|----------|---------|-------|-------|
| gen_LANIT | bdd10k | SegNeXt | missing |
| gen_UniControl | mapillaryvistas | PSPNet | missing |
| gen_UniControl | outside15k | PSPNet | missing |
| gen_VisualCloze | bdd10k | SegNeXt | missing |
| gen_albumentations_weather | bdd10k | SegNeXt | missing |
| gen_automold | bdd10k | SegNeXt | missing |
| gen_flux_kontext | bdd10k | SegNeXt | missing |

## Complete Configurations

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k |
|----------|--------|--------|-----------------|------------|
| gen_Attribute_Hallucination | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_augmenters | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_automold | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_CNetSeg | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_CUT | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_cyclediffusion | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_cycleGAN | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_flux_kontext | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_Img2Img | ✅ SF | ✅ SF | ✅ SF | ✅ SF |
| gen_IP2P | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_LANIT | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_Qwen_Image_Edit | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_stargan_v2 | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_step1x_new | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_step1x_v1p2 | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_SUSTechGAN | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_TSIT | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_UniControl | ✅ PSP, SF | ✅ PSP, SF | ✅ SF | ✅ SF |
| gen_VisualCloze | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_Weather_Effect_Generator | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| gen_albumentations_weather | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| baseline | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| std_minimal | ⏳ | ⏳ | ⏳ | ⏳ |
| std_photometric_distort | ⏳ | ⏳ | ⏳ | ⏳ |
| std_autoaugment | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| std_cutmix | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| std_mixup | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| std_randaugment | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
