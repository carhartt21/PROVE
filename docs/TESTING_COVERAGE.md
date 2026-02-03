# Testing Coverage Report

**Generated:** 2026-02-03 07:00

## Summary

| Status | Count | Percentage |
|--------|------:|----------:|
| ✅ Complete (valid mIoU) | 68 | 84.0% |
| 🔄 Running | 0 | 0.0% |
| ⏳ Pending (in queue) | 0 | 0.0% |
| ⚠️ Buggy (mIoU < 5%) | 0 | 0.0% |
| 🔃 Stale (wrong checkpoint) | 0 | 0.0% |
| ❌ Missing (no results) | 13 | 16.0% |
| **Total** | **81** | **100%** |

## Per-Dataset Breakdown

### BDD10k
- Complete: 32/132 (24.2%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 0
- Missing (no results): 13

### MapillaryVistas
- Complete: 25/105 (23.8%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 0
- Missing (no results): 0

### OUTSIDE15k
- Complete: 11/105 (10.5%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 0
- Missing (no results): 0

## Running Configurations

*No test jobs currently running.*

## Pending Configurations (in queue)

*No configurations pending in queue.*

## Buggy Configurations (need retesting)

*No buggy configurations - all tests have valid mIoU.*

## Stale Configurations (need retesting)

*No stale configurations - all tests use correct checkpoints.*

## Missing Configurations (no test results)

| Strategy | Dataset | Model | Issue |
|----------|---------|-------|-------|
| gen_LANIT | bdd10k | SegFormer | missing |
| gen_LANIT | bdd10k | SegNeXt | missing |
| gen_SUSTechGAN | bdd10k | SegNeXt | missing |
| gen_VisualCloze | bdd10k | SegFormer | missing |
| gen_VisualCloze | bdd10k | SegNeXt | missing |
| gen_albumentations_weather | bdd10k | SegFormer | missing |
| gen_albumentations_weather | bdd10k | SegNeXt | missing |
| gen_automold | bdd10k | SegFormer | missing |
| gen_automold | bdd10k | SegNeXt | missing |
| gen_flux_kontext | bdd10k | SegNeXt | missing |
| gen_step1x_new | bdd10k | SegNeXt | missing |
| gen_step1x_v1p2 | bdd10k | SegFormer | missing |
| gen_step1x_v1p2 | bdd10k | SegNeXt | missing |

## Complete Configurations

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k |
|----------|--------|--------|-----------------|------------|
| gen_Attribute_Hallucination | ⏳ | ⏳ | ⏳ | ✅ PSP |
| gen_augmenters | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_automold | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_CNetSeg | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_CUT | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_cyclediffusion | ✅ PSP | ⏳ | ✅ SF | ⏳ |
| gen_cycleGAN | ✅ SF | ⏳ | ⏳ | ⏳ |
| gen_flux_kontext | ✅ SF | ⏳ | ⏳ | ⏳ |
| gen_Img2Img | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_IP2P | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_LANIT | ✅ SF | ⏳ | ⏳ | ⏳ |
| gen_Qwen_Image_Edit | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_stargan_v2 | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_step1x_new | ✅ SF | ⏳ | ✅ PSP | ✅ PSP, SF |
| gen_step1x_v1p2 | ✅ SF | ⏳ | ⏳ | ⏳ |
| gen_SUSTechGAN | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_TSIT | ⏳ | ⏳ | ✅ SF | ⏳ |
| gen_UniControl | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_VisualCloze | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_Weather_Effect_Generator | ⏳ | ⏳ | ✅ SF | ⏳ |
| gen_albumentations_weather | ⏳ | ⏳ | ⏳ | ⏳ |
| baseline | ✅ PSP | ⏳ | ✅ PSP, SF | ✅ PSP, SF |
| std_minimal | ✅ SF | ⏳ | ⏳ | ⏳ |
| std_photometric_distort | ✅ SF | ⏳ | ⏳ | ⏳ |
| std_autoaugment | ✅ SF | ⏳ | ✅ PSP, SF | ⏳ |
| std_cutmix | ✅ SF | ⏳ | ✅ PSP, SF | ✅ SF |
| std_mixup | ✅ SF | ⏳ | ✅ SF | ✅ PSP, SF |
| std_randaugment | ✅ SF | ⏳ | ⏳ | ⏳ |
