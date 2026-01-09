# Dataset Coverage Analysis

This document provides an overview of which image generation methods cover which training datasets.

## Training Datasets
- **BDD10k**: 7,110 training images
- **IDD-AW**: 11,541 training images  
- **MapillaryVistas**: 22,581 training images
- **OUTSIDE15k**: 12,369 training images

## Testing Datasets (not in this analysis)
- **ACDC**: 4 domains (fog, night, rain, snow)
- **BDD100k**: 3 domains (clear, foggy, rainy)

## Coverage Table

| Method | Total | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Coverage |
|--------|------:|-------:|-------:|----------------:|-----------:|----------|
| Attribute_Hallucination | 191,400 | 14,220 | 23,082 | 45,162 | 24,738 | ✅ Full |
| CNetSeg | 187,398 | 10,218 | 23,082 | 45,162 | 24,738 | ✅ Full |
| CUT | 191,400 | 14,220 | 23,082 | 45,162 | 24,738 | ✅ Full |
| IP2P | 187,398 | 10,218 | 23,082 | 45,162 | 24,738 | ✅ Full |
| Img2Img | 187,398 | 10,218 | 23,082 | 45,162 | 24,738 | ✅ Full |
| LANIT | 223,300 | 11,921 | 26,929 | 52,689 | 28,861 | ✅ Full |
| Qwen-Image-Edit | 52,682 | 2,219 | 15,349 | 3,753 | 16,564 | ✅ Full |
| SUSTechGAN | 127,700 | 6,812 | 15,388 | 30,108 | 16,492 | ✅ Full |
| TSIT | 191,400 | 10,218 | 23,082 | 45,162 | 24,738 | ✅ Full |
| UniControl | 187,398 | 10,218 | 23,082 | 45,162 | 24,738 | ✅ Full |
| VisualCloze | 104,427 | 14,220 | 23,082 | 5,026 | 18,472 | ✅ Full |
| Weather_Effect_Generator | 82,179 | 7,110 | 9,670 | 15,054 | 8,246 | ✅ Full |
| albumentations_weather | 95,700 | 7,110 | 11,541 | 22,581 | 12,369 | ✅ Full |
| augmenters | 159,500 | 11,850 | 19,235 | 37,635 | 20,615 | ✅ Full |
| automold | 95,700 | 7,110 | 11,541 | 22,581 | 12,369 | ✅ Full |
| cycleGAN | 187,398 | 10,218 | 23,082 | 45,162 | 24,738 | ✅ Full |
| cyclediffusion | 180,783 | 14,220 | 12,465 | 45,162 | 24,738 | ✅ Full |
| flux_kontext | 69,900 | 0 | 0 | 45,162 | 24,738 | ⚠️ 2/4 |
| stargan_v2 | 187,398 | 10,218 | 23,082 | 45,162 | 24,738 | ✅ Full |
| step1x_new | 91,186 | 1,212 | 20,952 | 20,959 | 20,901 | ✅ Full |
| step1x_v1p2 | 112,820 | 14,220 | 23,082 | 26,727 | 24,562 | ✅ Full |

## Summary

### By Coverage Level

| Coverage | Count | Methods |
|----------|-------|---------|
| ✅ Full (4/4) | 20 | Attribute_Hallucination, CNetSeg, CUT, IP2P, Img2Img, LANIT, Qwen-Image-Edit, SUSTechGAN, TSIT, UniControl, VisualCloze, Weather_Effect_Generator, albumentations_weather, augmenters, automold, cycleGAN, cyclediffusion, stargan_v2, step1x_new, step1x_v1p2 |
| ⚠️ Partial (2/4) | 1 | flux_kontext |

## Excluded Methods

The following methods have been excluded from training experiments due to insufficient training dataset coverage:

| Method | Total Images | Reason | Status |
|--------|-------------|--------|--------|
| EDICT | 64,187 | Only ACDC/BDD100k coverage (0/4 training datasets) | Excluded |
| StyleID | 15,533 | Only ACDC/BDD100k coverage (0/4 training datasets) | Excluded |
| flux2 | 29,540 | Only ACDC/BDD100k coverage (0/4 training datasets) | Excluded |
| AOD-Net | ~547 | No manifest available (permission denied) | Excluded |

### Partial Coverage Details

| Method | Missing | Available |
|--------|---------|-----------|
| flux_kontext | BDD10k, IDD-AW | MapillaryVistas, OUTSIDE15k |

### Notes

- **Qwen-Image-Edit**: Manifest regenerated - now shows full 4/4 coverage with 2,219 BDD10k images
- **cyclediffusion**: Manifest regenerated - now shows full 4/4 coverage (was missing MapillaryVistas & OUTSIDE15k)
- **step1x_new**: Created BDD10k folder by extracting 1,212 images from BDD100k that match BDD10k image IDs (202 images × 6 domains). Now 4/4 coverage.
- **flux_kontext**: Only has MapillaryVistas and OUTSIDE15k - missing BDD10k and IDD-AW
- **NST** (Neural Style Transfer) is a special baseline using style transfer, not from GENERATED_IMAGES

Last updated: January 2025
