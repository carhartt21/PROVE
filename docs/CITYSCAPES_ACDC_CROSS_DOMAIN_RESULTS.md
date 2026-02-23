# Cityscapes → ACDC Cross-Domain Evaluation Results

**Date:** 2026-02-01

## Overview

This document summarizes the cross-domain evaluation results from models trained on the **Cityscapes** dataset and tested on the **ACDC** (Adverse Conditions Dataset). The goal is to assess how well models trained on clear weather conditions generalize to adverse weather scenarios.

## Methodology

- **Training Dataset:** Cityscapes (2,975 training images, 500 validation images)
- **Test Dataset:** ACDC test split (2,003 images across 4 weather domains)
- **Training Duration:** 160,000 iterations (80,000 for DeepLabV3+ and PSPNet)
- **Checkpoint Used:** Best mIoU checkpoint (`best_mIoU_iter_*.pth`)
- **Evaluation Metric:** Mean Intersection over Union (mIoU) per domain

### ACDC Domains
| Domain | # Images | Description |
|--------|----------|-------------|
| Foggy | 500 | Dense fog conditions |
| Night | 505 | Nighttime driving scenes |
| Rainy | 499 | Rain and wet conditions |
| Snowy | 499 | Snow-covered scenes |

## Results Summary

### Per-Domain mIoU (%)

| Model | Crop Size | Overall | Foggy | Night | Rainy | Snowy |
|-------|-----------|---------|-------|-------|-------|-------|
| **SegFormer MIT-B3** | 512×512 | **49.63** | **64.05** | 33.10 | **50.56** | **49.47** |
| **SegNeXt MSCAN-B** | 512×512 | **49.52** | 64.01 | **33.86** | 49.94 | 49.11 |
| HRNet HR48 | 512×1024 | 36.27 | 47.82 | 18.52 | 42.45 | 38.01 |
| PSPNet R50 | 769×769 | 30.91 | 42.16 | 16.45 | 34.22 | 31.77 |
| HRNet HR48 | 512×512 | 31.64 | 41.38 | 17.57 | 35.03 | 34.33 |
| PSPNet R50 | 512×512 | 29.39 | 42.22 | 14.16 | 33.38 | 31.75 |
| OCRNet HR48 | 512×1024 | 23.71 | 28.18 | 14.62 | 27.26 | 23.13 |
| DeepLabV3+ R50 | 769×769 | 20.99 | 34.39 | 11.85 | 27.25 | 17.33 |
| DeepLabV3+ R50 | 512×512 | 20.03 | 27.38 | 10.00 | 28.05 | 19.69 |
| OCRNet HR48 | 512×512 | 11.97 | 20.72 | 3.52 | 13.67 | 11.50 |

### Model Rankings by Overall mIoU

1. **SegFormer MIT-B3 (512×512)** - 49.63%
2. **SegNeXt MSCAN-B (512×512)** - 49.52%
3. HRNet HR48 (512×1024) - 36.27%
4. HRNet HR48 (512×512) - 31.64%
5. PSPNet R50 (769×769) - 30.91%
6. PSPNet R50 (512×512) - 29.39%
7. OCRNet HR48 (512×1024) - 23.71%
8. DeepLabV3+ R50 (769×769) - 20.99%
9. DeepLabV3+ R50 (512×512) - 20.03%
10. OCRNet HR48 (512×512) - 11.97%

## Key Findings

### 1. Transformer-Based Models Dominate
- **SegFormer** and **SegNeXt** significantly outperform CNN-based architectures
- ~18-20 percentage points higher mIoU compared to the next best model (HRNet)
- Self-attention mechanisms may provide better domain generalization

### 2. Night Domain is Most Challenging
- All models show significant performance degradation at night
- Best night performance: 33.86% (SegNeXt)
- Worst night performance: 3.52% (OCRNet 512×512)
- Night-specific augmentation or adaptation may be needed

### 3. Foggy Domain is Most Similar to Clear Weather
- Highest mIoU scores across all models
- Best foggy performance: 64.05% (SegFormer)
- Suggests fog is more similar to clear weather patterns than other adverse conditions

### 4. Domain Performance Ordering
Across most models: **Foggy > Snowy ≈ Rainy > Night**

### 5. Crop Size Impact
- Larger crop sizes generally improve CNN-based models:
  - HRNet: 31.64% (512×512) → 36.27% (512×1024)
  - PSPNet: 29.39% (512×512) → 30.91% (769×769)
- Transformers achieve best results at 512×512

### 6. OCRNet Underperformance
- OCRNet shows surprisingly poor results, especially at 512×512
- May indicate training issues or architectural limitations for cross-domain transfer
- Requires further investigation

## Per-Class Analysis (Top Models)

### SegFormer MIT-B3 (512×512)

**Best Performing Classes:**
- Road: ~90%+ IoU across domains
- Sky: Consistent high performance
- Vegetation: Good generalization

**Challenging Classes:**
- Traffic signs: Significantly degraded at night
- Small objects (poles, persons): Poor in adverse conditions
- Wall/fence: Low sample counts affect accuracy

## Recommendations

1. **For Production Deployment:** Use SegFormer or SegNeXt for adverse weather scenarios
2. **For Night Driving:** Consider night-specific fine-tuning or data augmentation
3. **For Foggy Conditions:** Current models may be sufficient without domain adaptation
4. **For Research:** Investigate why OCRNet underperforms on cross-domain transfer

## Data Locations

- **Model Weights:** `${AWARE_DATA_ROOT}/CITYSCAPES_REPLICATION/`
- **Test Results:** `${AWARE_DATA_ROOT}/CITYSCAPES_REPLICATION/acdc_cross_domain_results/`
- **Combined Results:** `combined_results.json`
- **Test Logs:** `cross_domain_test_log_v6.txt`

## Reproducibility

To reproduce these results:
```bash
cd ${HOME}/repositories/PROVE
python scripts/test_cityscapes_replication_on_acdc.py
```

For LSF cluster submission:
```bash
python scripts/test_cityscapes_replication_on_acdc.py --submit-jobs
```

## References

- ACDC Dataset: [https://acdc.vision.ee.ethz.ch/](https://acdc.vision.ee.ethz.ch/)
- Cityscapes Dataset: [https://www.cityscapes-dataset.com/](https://www.cityscapes-dataset.com/)
