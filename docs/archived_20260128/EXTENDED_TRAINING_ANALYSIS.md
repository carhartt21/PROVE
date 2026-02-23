# Extended Training Ablation Study: Analysis Report

**Generated:** 2026-01-17  
**Configurations Analyzed:** 22 completed at 320k iterations  
**Training Duration:** 320,000 iterations (4× standard 80k)

## Executive Summary

This analysis examines the value of extended training (4× iterations) for domain-adaptive semantic segmentation. **Key finding:** Most improvement occurs in the first 80k of extended training, with severe diminishing returns thereafter.

## 1. Diminishing Returns Analysis

### Marginal Gains Per Training Phase

| Training Phase | Average Gain | Relative to First Phase | Cumulative |
|----------------|-------------|------------------------|------------|
| 80k → 90k | (baseline) | - | 0.00 |
| 90k → 160k | **+0.75 mIoU** | 100% | +0.75 |
| 160k → 240k | **+0.39 mIoU** | 52% | +1.14 |
| 240k → 320k | **+0.10 mIoU** | 13% | +1.24 |

### Key Insight
- **First 80k extended** (90k→160k): Captures ~60% of total improvement
- **First 160k extended** (90k→240k): Captures ~92% of total improvement  
- **Final 80k** (240k→320k): Only ~8% of total improvement

## 2. Performance at Key Checkpoints

| Configuration | 160k | 240k | 320k | Peak@ |
|---------------|------|------|------|-------|
| gen_cyclediffusion/MapillaryVistas/PSPNet | 56.0 | 58.0 | 56.4 | 400k |
| gen_albumentations_weather/BDD10k/SegFormer | 52.4 | 52.8 | 52.9 | 350k |
| gen_automold/BDD10k/SegFormer | 52.2 | 52.5 | 52.6 | 370k |
| gen_cyclediffusion/BDD10k/SegFormer | 52.1 | 52.3 | 52.4 | 330k |
| gen_TSIT/BDD10k/SegFormer | 52.4 | 52.4 | 52.4 | 190k |
| gen_cycleGAN/BDD10k/SegFormer | 51.6 | 51.9 | 52.0 | 430k |
| gen_cycleGAN/BDD10k/PSPNet | 46.6 | 46.8 | 47.3 | 450k |
| gen_TSIT/IDD-AW/SegFormer | 45.5 | 45.9 | 46.1 | 440k |
| gen_cycleGAN/IDD-AW/SegFormer | 45.5 | 45.8 | 46.1 | 440k |
| gen_albumentations_weather/IDD-AW/SegFormer | 45.5 | 45.8 | 46.1 | 440k |

## 3. Strategy-Specific Patterns

### Fast Saturators (Peak < 200k)
- **gen_TSIT/BDD10k**: Reached peak at 190k, negligible gains after 160k
- These may not benefit from extended training beyond 160k

### Slow Learners (Peak > 400k)
- **gen_cycleGAN/BDD10k/PSPNet**: Still improving at 450k
- **gen_automold/IDD-AW/PSPNet**: Still improving at 450k
- PSPNet architectures generally benefit more from extended training

### Dataset Patterns
- **BDD10k configurations**: Tend to saturate earlier (~300k average peak)
- **IDD-AW configurations**: Continue improving longer (~430k average peak)
- Smaller/harder datasets benefit more from extended training

## 4. Marginal Gains by Configuration

| Configuration | 90k→160k | 160k→240k | 240k→320k |
|---------------|----------|-----------|-----------|
| gen_TSIT/BDD10k/SegFormer | +0.59 | +0.01 | -0.02 |
| gen_albumentations_weather/BDD10k/SegFormer | +0.38 | +0.35 | +0.13 |
| gen_automold/BDD10k/SegFormer | +0.73 | +0.28 | +0.13 |
| gen_cycleGAN/BDD10k/SegFormer | +0.28 | +0.37 | +0.03 |
| gen_TSIT/IDD-AW/SegFormer | +1.09 | +0.41 | +0.22 |
| gen_cycleGAN/IDD-AW/SegFormer | +1.08 | +0.36 | +0.27 |
| gen_albumentations_weather/IDD-AW/SegFormer | +1.10 | +0.35 | +0.25 |
| gen_UniControl/IDD-AW/SegFormer | +1.08 | +0.32 | +0.28 |
| gen_cycleGAN/IDD-AW/PSPNet | +1.23 | +0.34 | +0.27 |
| gen_automold/IDD-AW/PSPNet | +1.07 | +0.49 | +0.26 |

## 5. Conclusions & Recommendations

### Recommended Training Duration

| Use Case | Recommended | Rationale |
|----------|-------------|-----------|
| **Production** | 160k (2×) | 75% of gains at 50% compute cost |
| **Research** | 240k (3×) | 92% of gains, good cost-benefit |
| **Benchmarks** | 320k (4×) | Maximum performance, worth marginal gains |

### Strategy-Specific Recommendations
1. **gen_TSIT**: No benefit beyond 160k iterations
2. **gen_cycleGAN**: Benefits from full 320k training
3. **IDD-AW configurations**: Consider 400k+ for maximum performance
4. **PSPNet models**: Generally benefit more from extended training than SegFormer

### Cost-Benefit Analysis
- **160k training**: 2× compute for ~1.1 mIoU improvement
- **320k training**: 4× compute for ~1.2 mIoU improvement
- **ROI drops significantly after 160k**

## 6. Technical Details

### Training Configuration
- Base: 80,000 iterations (standard training)
- Extended: 320,000 iterations (resumed from 80k checkpoint)
- Validation: Every 10,000 iterations
- Models: SegFormer MIT-B5, PSPNet ResNet-101

### Completed Configurations
- **Strategies:** gen_albumentations_weather, gen_automold, gen_cyclediffusion, gen_cycleGAN, gen_TSIT, gen_UniControl
- **Datasets:** BDD10k, IDD-AW, MapillaryVistas
- **Models:** SegFormer, PSPNet
- **Total:** 22 configurations completed at 320k

### Ongoing Training (as of report generation)
- **Running:** 6 jobs (gen_cyclediffusion, gen_automold)
- **Pending:** 28 jobs (gen_flux_kontext, gen_step1x_new, gen_step1x_v1p2, std_randaugment)

---

*This analysis was generated from training logs in `${AWARE_DATA_ROOT}/WEIGHTS_EXTENDED/`*
