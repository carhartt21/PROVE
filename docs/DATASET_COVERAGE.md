# Dataset Coverage Analysis

**Last Updated:** 2026-01-12

This document provides an overview of which image generation methods cover which training datasets.

## Recent Changes (2026-01-12)
- **flux_kontext**: Now has full 4/4 coverage (added BDD10k: 11,344 and IDD-AW: 5,318 images)
- All 21 methods with full coverage confirmed
- Submitted 27 training jobs for newly available configurations
- Updated tracking scripts to handle new `train_` prefix jobs

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
| Qwen-Image-Edit | 56,784 | 6,321 | 15,349 | 3,753 | 16,564 | ✅ Full |
| SUSTechGAN | 127,700 | 6,812 | 15,388 | 30,108 | 16,492 | ✅ Full |
| TSIT | 191,400 | 10,218 | 23,082 | 45,162 | 24,738 | ✅ Full |
| UniControl | 187,398 | 10,218 | 23,082 | 45,162 | 24,738 | ✅ Full |
| VisualCloze | 108,032 | 14,220 | 23,082 | 5,026 | 24,738 | ✅ Full |
| Weather_Effect_Generator | 82,179 | 7,110 | 9,670 | 15,054 | 8,246 | ✅ Full |
| albumentations_weather | 95,700 | 7,110 | 11,541 | 22,581 | 12,369 | ✅ Full |
| augmenters | 159,500 | 11,850 | 19,235 | 37,635 | 20,615 | ✅ Full |
| automold | 95,700 | 7,110 | 11,541 | 22,581 | 12,369 | ✅ Full |
| cycleGAN | 187,398 | 10,218 | 23,082 | 45,162 | 24,738 | ✅ Full |
| cyclediffusion | 180,783 | 14,220 | 12,465 | 45,162 | 24,738 | ✅ Full |
| flux_kontext | 86,562 | 11,344 | 5,318 | 45,162 | 24,738 | ✅ Full |
| stargan_v2 | 187,398 | 10,218 | 23,082 | 45,162 | 24,738 | ✅ Full |
| step1x_new | 85,270 | 7,927 | 20,952 | 20,959 | 20,901 | ✅ Full |
| step1x_v1p2 | 105,075 | 14,220 | 23,082 | 26,727 | 24,562 | ✅ Full |

## Summary

### By Coverage Level

| Coverage | Count | Methods |
|----------|-------|---------|
| ✅ Full (4/4) | 21 | All methods listed above |
| ⚠️ Partial | 0 | None |

## Excluded Methods

The following methods have been excluded from training experiments due to insufficient training dataset coverage:

| Method | Total Images | Reason | Status |
|--------|-------------|--------|--------|
| EDICT | 64,187 | Only ACDC/BDD100k coverage (0/4 training datasets) | Excluded |
| StyleID | 15,533 | Only ACDC/BDD100k coverage (0/4 training datasets) | Excluded |
| flux2 | 29,540 | Only ACDC/BDD100k coverage (0/4 training datasets) | Excluded |
| AOD-Net | 547 | Image restoration method, minimal images | Excluded |

### Notes

- **flux_kontext**: Now has full 4/4 coverage with 11,344 BDD10k images and 5,318 IDD-AW images
- **Qwen-Image-Edit**: 56,784 total images with full 4/4 coverage
- **cyclediffusion**: 180,783 images with full 4/4 coverage
- **step1x_new**: 85,270 images including 7,927 BDD10k images
- All generative methods now have full training dataset coverage

Last updated: January 2026
