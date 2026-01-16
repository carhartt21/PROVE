# Comprehensive Baseline Analysis Report

Generated: 2026-01-08 13:06:04

**Note:** Results from domains with fewer than 50 test images are excluded to ensure reliable metrics.

## 1. Full Baseline Analysis

Models trained on ALL domains (not just clear_day).

### Overall Statistics

- **Average mIoU:** 58.49% ± 15.75
- **Normal Conditions mIoU:** 61.57%
- **Adverse Conditions mIoU:** 55.33%
- **Domain Gap (Normal - Adverse):** +6.24%

### Per-Domain Performance

| Domain | Type | mIoU | Std | Avg Images |
|--------|------|------|-----|------------|
| clear_day | NORMAL | 60.84% | ±16.52 | 1915 |
| cloudy | NORMAL | 62.30% | ±15.60 | 517 |
| dawn_dusk | ADVERSE | 55.20% | ±15.10 | 91 |
| foggy | NORMAL | 81.23% | ±8.80 | 290 |
| night | ADVERSE | 51.19% | ±16.38 | 88 |
| rainy | ADVERSE | 56.92% | ±13.03 | 190 |
| snowy | ADVERSE | 57.96% | ±15.30 | 180 |

### Per-Dataset Performance

| Dataset | Overall mIoU | Normal | Adverse | Gap |
|---------|--------------|--------|---------|-----|
| bdd10k | 53.74% | 58.43% | 51.40% | +7.03% |
| idd-aw | 73.59% | 77.43% | 69.76% | +7.67% |
| mapillaryvistas | 44.57% | 44.65% | 44.52% | +0.12% |
| outside15k | 57.22% | 65.78% | 52.93% | +12.85% |

### Per-Model Performance

| Model | Overall mIoU | Normal | Adverse | Gap |
|-------|--------------|--------|---------|-----|
| deeplabv3plus_r50 | 50.72% | 54.80% | 47.01% | +7.80% |
| pspnet_r50 | 55.09% | 57.36% | 52.28% | +5.08% |
| segformer_mit-b5 | 69.66% | 72.55% | 66.70% | +5.85% |

### Per-Configuration Performance

| Dataset | Model | Clear Day | Normal | Adverse | Overall | Gap |
|---------|-------|-----------|--------|---------|---------|-----|
| idd-aw | segformer_mit-b5 | 81.8% | 82.1% | 78.0% | 81.0% | +4.1% |
| idd-aw | pspnet_r50 | 76.9% | 76.5% | 67.0% | 71.4% | +9.5% |
| outside15k | segformer_mit-b5 | 76.4% | 76.0% | 66.7% | 69.8% | +9.2% |
| idd-aw | deeplabv3plus_r50 | 73.0% | 73.7% | 64.3% | 68.3% | +9.5% |
| mapillaryvistas | segformer_mit-b5 | 66.9% | 66.6% | 62.2% | 63.9% | +4.4% |
| bdd10k | segformer_mit-b5 | 63.2% | 65.6% | 58.8% | 61.0% | +6.8% |
| outside15k | pspnet_r50 | 66.0% | 64.9% | 51.5% | 56.0% | +13.3% |
| bdd10k | pspnet_r50 | 55.0% | 59.2% | 52.2% | 54.5% | +7.0% |
| outside15k | deeplabv3plus_r50 | 57.8% | 56.5% | 40.5% | 45.9% | +16.0% |
| bdd10k | deeplabv3plus_r50 | 49.0% | 50.6% | 43.2% | 45.7% | +7.3% |
| mapillaryvistas | deeplabv3plus_r50 | 34.8% | 38.4% | 37.6% | 37.9% | +0.8% |
| mapillaryvistas | pspnet_r50 | 29.4% | 29.0% | 33.8% | 31.9% | -4.8% |

## 2. Clear Day Baseline Analysis

Models trained ONLY on clear_day images (limited training).

### Overall Statistics

- **Average mIoU:** 55.23% ± 15.07
- **Normal Conditions mIoU:** 60.73%
- **Adverse Conditions mIoU:** 51.52%
- **Domain Gap (Normal - Adverse):** +9.21%

### Per-Domain Performance

| Domain | Type | mIoU | Std |
|--------|------|------|-----|
| clear_day | NORMAL | 61.68% | ±18.01 |
| cloudy | NORMAL | 59.78% | ±17.63 |
| dawn_dusk | ADVERSE | 51.19% | ±13.31 |
| foggy | NORMAL | 67.03% | ±5.68 |
| night | ADVERSE | 49.57% | ±11.92 |
| rainy | ADVERSE | 53.82% | ±11.62 |
| snowy | ADVERSE | 51.42% | ±15.69 |

## 3. Full vs Clear Day Baseline Comparison

| Metric | Full Baseline | Clear Day Baseline | Difference |
|--------|---------------|-------------------|------------|
| Overall mIoU | 58.49% | 55.23% | +3.25% |
| Normal mIoU | 61.57% | 60.73% | +0.84% |
| Adverse mIoU | 55.33% | 51.52% | +3.81% |
| Domain Gap | +6.24% | +9.21% | -2.97% |

## 4. Key Insights

### Best and Worst Configurations

- **Best:** idd-aw / segformer_mit-b5 (81.0% overall)
- **Worst:** mapillaryvistas / pspnet_r50 (31.9% overall)

### Model Ranking (by overall mIoU)

1. **segformer_mit-b5:** 69.7%
2. **pspnet_r50:** 55.1%
3. **deeplabv3plus_r50:** 50.7%

### Dataset Ranking (by overall mIoU)

1. **idd-aw:** 73.6%
2. **outside15k:** 57.2%
3. **bdd10k:** 53.7%
4. **mapillaryvistas:** 44.6%
