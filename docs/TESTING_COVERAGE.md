# Testing Coverage Report

**Generated:** 2026-02-08 08:37

## Summary

| Status | Count | Percentage |
|--------|------:|----------:|
| ✅ Complete (valid mIoU) | 237 | 95.2% |
| 🔄 Running | 0 | 0.0% |
| ⏳ Pending (in queue) | 0 | 0.0% |
| ⚠️ Buggy (mIoU < 5%) | 0 | 0.0% |
| 🔃 Stale (wrong checkpoint) | 0 | 0.0% |
| ❌ Missing (no results) | 12 | 4.8% |
| **Total** | **249** | **100%** |

## Per-Dataset Breakdown

### BDD10k
- Complete: 97/132 (73.5%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 0
- Missing (no results): 12

### MapillaryVistas
- Complete: 71/105 (67.6%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 0
- Missing (no results): 0

### OUTSIDE15k
- Complete: 69/105 (65.7%)
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
| gen_Attribute_Hallucination | ✅ PSP, SF | ⏳ | ✅ PSP, SF | ✅ PSP, SF |
| gen_augmenters | ✅ PSP, SF | ⏳ | ✅ PSP, SF | ✅ PSP, SF |
| gen_automold | ✅ PSP, SF | ⏳ | ✅ PSP, SF | ✅ PSP, SF |
| gen_CNetSeg | ✅ PSP, SF | ⏳ | ✅ PSP, SF | ✅ PSP, SF |
| gen_CUT | ✅ PSP, SF | ⏳ | ✅ PSP, SF | ✅ PSP, SF |
| gen_cyclediffusion | ✅ PSP | ⏳ | ✅ SF | ⏳ |
| gen_cycleGAN | ✅ PSP, SF | ⏳ | ✅ PSP, SF | ✅ PSP, SF |
| gen_flux_kontext | ✅ PSP, SF | ⏳ | ✅ PSP, SF | ✅ PSP, SF |
| gen_Img2Img | ✅ SF | ⏳ | ✅ SF | ✅ SF |
| gen_IP2P | ✅ PSP, SF | ⏳ | ✅ PSP, SF | ✅ PSP, SF |
| gen_LANIT | ✅ PSP, SF | ⏳ | ✅ PSP, SF | ✅ PSP, SF |
| gen_Qwen_Image_Edit | ✅ PSP, SF | ⏳ | ✅ PSP, SF | ✅ PSP, SF |
| gen_stargan_v2 | ✅ PSP, SF | ⏳ | ✅ PSP, SF | ✅ PSP, SF |
| gen_step1x_new | ✅ PSP, SF | ⏳ | ✅ PSP, SF | ✅ PSP, SF |
| gen_step1x_v1p2 | ✅ PSP, SF | ⏳ | ✅ PSP, SF | ✅ PSP, SF |
| gen_SUSTechGAN | ✅ PSP, SF | ⏳ | ✅ PSP, SF | ✅ PSP, SF |
| gen_TSIT | ✅ PSP, SF | ⏳ | ✅ PSP, SF | ✅ PSP, SF |
| gen_UniControl | ⏳ | ⏳ | ⏳ | ✅ SF |
| gen_VisualCloze | ✅ PSP, SF | ⏳ | ✅ PSP, SF | ✅ PSP, SF |
| gen_Weather_Effect_Generator | ✅ PSP, SF | ⏳ | ✅ PSP, SF | ✅ PSP, SF |
| gen_albumentations_weather | ⏳ | ⏳ | ⏳ | ⏳ |
| baseline | ✅ PSP, SF | ⏳ | ✅ PSP, SF | ✅ PSP, SF |
| std_minimal | ✅ SF | ⏳ | ⏳ | ⏳ |
| std_photometric_distort | ✅ SF | ⏳ | ⏳ | ⏳ |
| std_autoaugment | ✅ PSP, SF | ⏳ | ✅ PSP, SF | ✅ PSP, SF |
| std_cutmix | ✅ PSP, SF | ⏳ | ✅ PSP, SF | ✅ PSP, SF |
| std_mixup | ✅ PSP, SF | ⏳ | ✅ PSP, SF | ✅ PSP, SF |
| std_randaugment | ✅ PSP, SF | ⏳ | ✅ PSP, SF | ✅ PSP, SF |
