# Testing Coverage Report

**Generated:** 2026-02-08 20:48

## Summary

| Status | Count | Percentage |
|--------|------:|----------:|
| ✅ Complete (valid mIoU) | 79 | 97.5% |
| 🔄 Running | 0 | 0.0% |
| ⏳ Pending (in queue) | 1 | 1.2% |
| ⚠️ Buggy (mIoU < 5%) | 0 | 0.0% |
| 🔃 Stale (wrong checkpoint) | 0 | 0.0% |
| ❌ Missing (no results) | 1 | 1.2% |
| **Total** | **81** | **100%** |

## Per-Dataset Breakdown

### BDD10k
- Complete: 24/112 (21.4%)
- Running: 0
- Pending (in queue): 1
- Buggy (mIoU < 5%): 0
- Missing (no results): 0

### IDD-AW
- Complete: 18/106 (17.0%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 0
- Missing (no results): 1

### MapillaryVistas
- Complete: 20/111 (18.0%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 0
- Missing (no results): 0

### OUTSIDE15k
- Complete: 17/111 (15.3%)
- Running: 0
- Pending (in queue): 0
- Buggy (mIoU < 5%): 0
- Missing (no results): 0

## Running Configurations

*No test jobs currently running.*

## Pending Configurations (in queue)

| Strategy | Dataset | Model |
|----------|---------|-------|
| gen_step1x_new | bdd10k | segformer_mit-b3_ratio0p50 |

## Buggy Configurations (need retesting)

*No buggy configurations - all tests have valid mIoU.*

## Stale Configurations (need retesting)

*No stale configurations - all tests use correct checkpoints.*

## Missing Configurations (no test results)

| Strategy | Dataset | Model | Issue |
|----------|---------|-------|-------|
| baseline | idd-aw | SegFormer | missing |

## Complete Configurations

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k |
|----------|--------|--------|-----------------|------------|
| gen_Attribute_Hallucination | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_augmenters | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_automold | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_CNetSeg | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_CUT | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_cyclediffusion | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_cycleGAN | ✅ SF | ✅ SF | ✅ SF | ✅ SF |
| gen_flux_kontext | ✅ SF | ✅ SF | ✅ SF | ⏳ |
| gen_Img2Img | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_IP2P | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_LANIT | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_Qwen_Image_Edit | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_stargan_v2 | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_step1x_new | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_step1x_v1p2 | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_SUSTechGAN | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_TSIT | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_UniControl | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_VisualCloze | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_Weather_Effect_Generator | ⏳ | ⏳ | ⏳ | ⏳ |
| gen_albumentations_weather | ⏳ | ⏳ | ⏳ | ⏳ |
| baseline | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF | ✅ DLV3+, PSP, SF |
| std_minimal | ⏳ | ⏳ | ⏳ | ⏳ |
| std_photometric_distort | ⏳ | ⏳ | ⏳ | ⏳ |
| std_autoaugment | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| std_cutmix | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF |
| std_mixup | ✅ PSP, SF | ✅ PSP, SF | ✅ PSP, SF | ✅ SF |
| std_randaugment | ✅ SF | ✅ SF | ✅ SF | ✅ SF |
