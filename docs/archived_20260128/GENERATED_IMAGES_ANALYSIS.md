# Generated Images Analysis

**Generated:** January 9, 2025  
**Last Updated:** January 9, 2025 (Manifests refreshed, scripts added)  
**Base Directory:** `${AWARE_DATA_ROOT}/GENERATED_IMAGES/`

## Overview

This document provides a comprehensive analysis of the generated images available for training, comparing them with the strategies currently used in the retraining scripts.

## Summary Statistics

| Category | Count |
|----------|-------|
| Total image generation methods | 25 |
| Methods with valid manifests | 23 |
| Methods in retrain scripts | 28 (22 gen + 4 std + baseline + photometric) |
| **Methods EXCLUDED** | 4 (EDICT, StyleID, flux2, AOD-Net) |
| **Manifests Status** | ✅ All updated (Jan 8, 2025) |

### Excluded Methods

The following methods are **excluded** from training due to insufficient training dataset coverage (0/4 training datasets):

| Method | Reason |
|--------|--------|
| EDICT | Only ACDC/BDD100k coverage |
| StyleID | Only ACDC/BDD100k coverage |
| flux2 | Only ACDC/BDD100k coverage |
| AOD-Net | No manifest available |

---

## All Retraining Strategies (28 total)

### Generative Methods with Full Coverage (20)

| Strategy | Directory | Total Images | Status |
|----------|-----------|--------------|--------|
| gen_Attribute_Hallucination | Attribute_Hallucination | 191,400 | ✅ Running |
| gen_augmenters | augmenters | 159,500 | ✅ Running |
| gen_automold | automold | 95,700 | ✅ Running |
| gen_CUT | CUT | 191,400 | ✅ Running |
| gen_cycleGAN | cycleGAN | 187,398 | ✅ Running |
| gen_cyclediffusion | cyclediffusion | 180,783 | ✅ Running |
| gen_flux1_kontext | flux_kontext | 69,900 | ⚠️ 2/4 datasets |
| gen_Img2Img | Img2Img | 187,398 | ✅ Running |
| gen_IP2P | IP2P | 187,398 | ✅ Running |
| gen_LANIT | LANIT | 223,300 | ✅ Running |
| gen_NST | N/A | N/A | Uses different source |
| gen_Qwen_Image_Edit | Qwen-Image-Edit | 52,682 | ✅ Running |
| gen_stargan_v2 | stargan_v2 | 187,398 | ✅ Running |
| gen_step1x_new | step1x_new | 91,186 | ✅ Running |
| gen_step1x_v1p2 | step1x_v1p2 | 112,307 | ✅ Running |
| gen_SUSTechGAN | SUSTechGAN | 127,700 | ✅ Running |
| gen_TSIT | TSIT | 191,400 | ✅ Running |
| gen_UniControl | UniControl | 187,398 | ✅ Running |
| gen_VisualCloze | VisualCloze | 104,427 | ✅ Running |
| gen_Weather_Effect_Generator | Weather_Effect_Generator | 82,179 | ✅ Running |
| gen_albumentations_weather | albumentations_weather | 95,700 | ✅ Running |
| gen_CNetSeg | CNetSeg | 187,398 | ✅ Running |

### Standard Augmentation Methods (4)
| Strategy | Description | Status |
|----------|-------------|--------|
| std_autoaugment | AutoAugment | ✅ Running |
| std_cutmix | CutMix | ✅ Running |
| std_mixup | MixUp | ✅ Running |
| std_randaugment | RandAugment | ✅ Running |

### Other (2)
| Strategy | Description | Status |
|----------|-------------|--------|
| baseline | No augmentation | ✅ Running |
| std_std_photometric_distort | Photometric distortion | ✅ Running |

---

## Per-Dataset Image Counts (Used Methods)

### ACDC
| Method | Images |
|--------|--------|
| gen_LANIT | 4,907 |
| gen_Attribute_Hallucination | 4,206 |
| gen_CUT | 4,206 |
| gen_TSIT | 4,206 |
| gen_cycleGAN | 4,206 |
| gen_Img2Img | 4,206 |
| gen_IP2P | 4,206 |
| gen_UniControl | 4,206 |
| gen_EDICT | 4,206 |
| gen_step1x_new | 4,206 |
| gen_StyleID | 4,206 |
| gen_augmenters | 3,505 |
| gen_SUSTechGAN | 2,904 |
| gen_Weather_Effect_Generator | 2,103 |
| gen_automold | 2,103 |
| gen_Qwen_Image_Edit | 152 |

### BDD10k
| Method | Images |
|--------|--------|
| gen_Attribute_Hallucination | 14,220 |
| gen_CUT | 14,220 |
| gen_LANIT | 11,921 |
| gen_augmenters | 11,850 |
| gen_cycleGAN | 10,218 |
| gen_Img2Img | 10,218 |
| gen_IP2P | 10,218 |
| gen_TSIT | 10,218 |
| gen_UniControl | 10,218 |
| gen_stargan_v2 | 10,218 |
| gen_Weather_Effect_Generator | 7,110 |
| gen_automold | 7,110 |
| gen_SUSTechGAN | 6,812 |
| gen_Qwen_Image_Edit | 0 |
| gen_step1x_new | 0 |

### IDD-AW
| Method | Images |
|--------|--------|
| gen_LANIT | 26,929 |
| gen_Attribute_Hallucination | 23,082 |
| gen_CUT | 23,082 |
| gen_TSIT | 23,082 |
| gen_cycleGAN | 23,082 |
| gen_Img2Img | 23,082 |
| gen_IP2P | 23,082 |
| gen_UniControl | 23,082 |
| gen_stargan_v2 | 23,082 |
| gen_step1x_new | 20,952 |
| gen_augmenters | 19,235 |
| gen_SUSTechGAN | 15,388 |
| gen_Qwen_Image_Edit | 15,349 |
| gen_automold | 11,541 |
| gen_Weather_Effect_Generator | 9,670 |

### MapillaryVistas
| Method | Images |
|--------|--------|
| gen_LANIT | 52,689 |
| gen_Attribute_Hallucination | 45,162 |
| gen_CUT | 45,162 |
| gen_TSIT | 45,162 |
| gen_cycleGAN | 45,162 |
| gen_Img2Img | 45,162 |
| gen_IP2P | 45,162 |
| gen_UniControl | 45,162 |
| gen_stargan_v2 | 45,162 |
| gen_flux1_kontext | 45,162 |
| gen_augmenters | 37,635 |
| gen_SUSTechGAN | 30,108 |
| gen_automold | 22,581 |
| gen_step1x_new | 20,959 |
| gen_Weather_Effect_Generator | 15,054 |
| gen_Qwen_Image_Edit | 3,753 |

### OUTSIDE15k
| Method | Images |
|--------|--------|
| gen_LANIT | 28,861 |
| gen_Attribute_Hallucination | 24,738 |
| gen_CUT | 24,738 |
| gen_TSIT | 24,738 |
| gen_cycleGAN | 24,738 |
| gen_Img2Img | 24,738 |
| gen_IP2P | 24,738 |
| gen_UniControl | 24,738 |
| gen_stargan_v2 | 24,738 |
| gen_flux1_kontext | 24,738 |
| gen_step1x_new | 20,901 |
| gen_augmenters | 20,615 |
| gen_SUSTechGAN | 16,492 |
| gen_Qwen_Image_Edit | 15,092 |
| gen_automold | 12,369 |
| gen_Weather_Effect_Generator | 8,246 |

---

## Naming Conventions

### Script to Directory Mapping
| Script Strategy | Directory Name | Notes |
|----------------|----------------|-------|
| gen_flux1_kontext | flux_kontext | "1" removed in directory |
| gen_Qwen_Image_Edit | Qwen-Image-Edit | Underscore vs hyphen |
| gen_NST | N/A | No generated images (uses different method) |

---

## Recommendations

### Immediate Actions
1. ✅ **Manifests Updated**: All 5 outdated manifests have been regenerated (Jan 8, 2026)
2. ✅ **New Strategies Added**: 6 new methods added to retraining (CNetSeg, VisualCloze, albumentations_weather, cyclediffusion, step1x_v1p2, AOD_Net)
3. ✅ **Jobs Submitted**: 25 retraining jobs submitted and running

### Remaining Tasks
1. **Create Manifests**: Generate manifests for AOD-Net and flux2 directories
2. **Monitor Training**: Check job progress for newly added methods
3. **Validate Results**: Review test metrics after retraining completes

### Note on Dataset Label Changes
With the native class label implementation:
- **OUTSIDE15k**: Now uses 24 native classes (was incorrectly mapped to 19)
- **MapillaryVistas**: Now uses 66 native classes (was unified to 19)
- All models for these datasets need retraining with correct class counts

---

## File Locations

- **Generated Images**: `${AWARE_DATA_ROOT}/GENERATED_IMAGES/`
- **Manifest Files**: `${AWARE_DATA_ROOT}/GENERATED_IMAGES/*/manifest.json`
- **Retrain Scripts**: `${HOME}/repositories/PROVE/scripts/retrain_jobs/`
- **Label Handling Docs**: `${HOME}/repositories/PROVE/docs/LABEL_HANDLING.md`
