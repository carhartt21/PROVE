# Testing Coverage Report

**Generated:** 2026-02-08 08:38

## Summary

| Status | Count | Percentage |
|--------|------:|----------:|
| ✅ Complete (valid mIoU) | 0 | 0.0% |
| 🔄 Running | 0 | 0.0% |
| ⏳ Pending (in queue) | 0 | 0.0% |
| ⚠️ Buggy (mIoU < 5%) | 9 | 100.0% |
| 🔃 Stale (wrong checkpoint) | 0 | 0.0% |
| ❌ Missing (no results) | 0 | 0.0% |
| **Total** | **9** | **100%** |

## Per-Dataset Breakdown

### Cityscapes
- Complete: 0/96 (0.0%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 9
- Missing (no results): 0

## Running Configurations

*No test jobs currently running.*

## Pending Configurations (in queue)

*No configurations pending in queue.*

## Buggy Configurations (need retesting)

These configurations have test results with mIoU < 5%, indicating a bug in the test.

| Strategy | Dataset | Model | mIoU |
|----------|---------|-------|-----:|
| baseline | cityscapes | segformer_mit-b3 | N/A |
| gen_albumentations_weather | cityscapes | segformer_mit-b3_ratio0p50 | N/A |
| gen_automold | cityscapes | segformer_mit-b3_ratio0p50 | N/A |
| gen_flux_kontext | cityscapes | segformer_mit-b3_ratio0p50 | N/A |
| gen_step1x_new | cityscapes | segformer_mit-b3_ratio0p50 | N/A |
| std_autoaugment | cityscapes | segformer_mit-b3 | N/A |
| std_cutmix | cityscapes | segformer_mit-b3 | N/A |
| std_mixup | cityscapes | segformer_mit-b3 | N/A |
| std_randaugment | cityscapes | segformer_mit-b3 | N/A |

## Stale Configurations (need retesting)

*No stale configurations - all tests use correct checkpoints.*

## Missing Configurations (no test results)

*No missing configurations.*

## Complete Configurations

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k |
|----------|--------|--------|-----------------|------------|
| gen_Attribute_Hallucination | ⏳ |
| gen_augmenters | ⏳ |
| gen_automold | ⏳ |
| gen_CNetSeg | ⏳ |
| gen_CUT | ⏳ |
| gen_cyclediffusion | ⏳ |
| gen_cycleGAN | ⏳ |
| gen_flux_kontext | ⏳ |
| gen_Img2Img | ⏳ |
| gen_IP2P | ⏳ |
| gen_LANIT | ⏳ |
| gen_Qwen_Image_Edit | ⏳ |
| gen_stargan_v2 | ⏳ |
| gen_step1x_new | ⏳ |
| gen_step1x_v1p2 | ⏳ |
| gen_SUSTechGAN | ⏳ |
| gen_TSIT | ⏳ |
| gen_UniControl | ⏳ |
| gen_VisualCloze | ⏳ |
| gen_Weather_Effect_Generator | ⏳ |
| gen_albumentations_weather | ⏳ |
| baseline | ⏳ |
| std_minimal | ⏳ |
| std_photometric_distort | ⏳ |
| std_autoaugment | ⏳ |
| std_cutmix | ⏳ |
| std_mixup | ⏳ |
| std_randaugment | ⏳ |
