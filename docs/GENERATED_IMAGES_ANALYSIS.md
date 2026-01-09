# Generated Images Analysis

**Generated:** January 8, 2026  
**Base Directory:** `/scratch/aaa_exchange/AWARE/GENERATED_IMAGES/`

## Overview

This document provides a comprehensive analysis of the generated images available for training, comparing them with the strategies currently used in the `retrain_affected_models.py` script.

## Summary Statistics

| Category | Count |
|----------|-------|
| Total methods available | 25 |
| Methods with manifests | 23 |
| Methods used in retrain script | 18 |
| Methods NOT used | 7 |
| Manifests requiring update | 5 |

---

## Methods Used in Retrain Script

| Script Strategy | Image Directory | Total Images | Per-Dataset Coverage |
|----------------|-----------------|--------------|---------------------|
| gen_Attribute_Hallucination | Attribute_Hallucination | 191,400 | All 6 datasets |
| gen_augmenters | augmenters | 159,500 | All 6 datasets |
| gen_automold | automold | 95,700 | All 6 datasets |
| gen_CUT | CUT | 191,400 | All 6 datasets |
| gen_cycleGAN | cycleGAN | 187,398 | All 6 datasets |
| gen_EDICT | EDICT | 64,187 | ACDC, BDD100k only |
| gen_flux1_kontext | flux_kontext | 69,900 | MapillaryVistas, OUTSIDE15k only |
| gen_Img2Img | Img2Img | 187,398 | All 6 datasets |
| gen_IP2P | IP2P | 187,398 | All 6 datasets |
| gen_LANIT | LANIT | 223,300 | All 6 datasets |
| gen_NST | **NOT FOUND** | N/A | Uses different source |
| gen_Qwen_Image_Edit | Qwen-Image-Edit | 55,645* | Missing BDD10k |
| gen_stargan_v2 | stargan_v2 | 187,398 | All 6 datasets |
| gen_step1x_new | step1x_new | 103,627* | Missing BDD10k |
| gen_StyleID | StyleID | 15,533 | ACDC, BDD100k only |
| gen_SUSTechGAN | SUSTechGAN | 127,700 | All 6 datasets |
| gen_TSIT | TSIT | 191,400 | All 6 datasets |
| gen_UniControl | UniControl | 187,398 | All 6 datasets |
| gen_Weather_Effect_Generator | Weather_Effect_Generator | 82,179 | All 6 datasets |

*Manifest count outdated - actual count shown

---

## Methods NOT Used in Retrain Script

These methods have generated images available but are not included in the current retraining:

| Method | Total Images | Manifest Status | Notes |
|--------|-------------|-----------------|-------|
| AOD-Net | Unknown | NO_MANIFEST | Dehazing/restoration method |
| CNetSeg | 187,398 | ✓ Valid | ControlNet-based segmentation |
| VisualCloze | 104,414* | ⚠ Outdated | Visual completion/inpainting |
| albumentations_weather | 95,700 | ✓ Valid | Albumentations weather augmentation |
| cyclediffusion | 180,783* | ⚠ Outdated | CycleDiffusion image editing |
| flux2 | Unknown | NO_MANIFEST | FLUX model version 2 |
| step1x_v1p2 | 112,305* | ⚠ Outdated | Step1X model version 1.2 |

*Manifest count outdated - actual count shown

---

## Manifests Requiring Update

The following manifest files have outdated image counts (more images exist than recorded):

| Method | Manifest Count | Actual Count | Difference | % Increase |
|--------|----------------|--------------|------------|------------|
| Qwen-Image-Edit | 41,718 | 55,645 | +13,927 | +33.4% |
| VisualCloze | 65,006 | 104,414 | +39,408 | +60.6% |
| cyclediffusion | 110,883 | 180,783 | +69,900 | +63.0% |
| step1x_new | 77,343 | 103,627 | +26,284 | +34.0% |
| step1x_v1p2 | 94,002 | 112,305 | +18,303 | +19.5% |

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

1. **Update Manifests**: Run manifest regeneration for the 5 outdated manifests
2. **Add Missing Strategies**: Consider adding unused methods (CNetSeg, albumentations_weather, etc.)
3. **Create Manifests**: Generate manifests for AOD-Net and flux2
4. **Review gen_NST**: Clarify the source/method for NST augmentation

---

## File Locations

- **Generated Images**: `/scratch/aaa_exchange/AWARE/GENERATED_IMAGES/`
- **Manifest Files**: `/scratch/aaa_exchange/AWARE/GENERATED_IMAGES/*/manifest.json`
- **Retrain Script**: `/home/mima2416/repositories/PROVE/scripts/retrain_affected_models.py`
