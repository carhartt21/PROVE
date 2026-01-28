# Comprehensive Baseline Analysis Report

Generated: 2026-01-27 15:25:14

**Note:** Results from domains with fewer than 50 test images are excluded to ensure reliable metrics.

## 1. Stage 1 Baseline Analysis (Clear Day Training)

Models trained ONLY on clear_day images.

**Source:** `/scratch/aaa_exchange/AWARE/WEIGHTS/baseline/`

### Overall Statistics

- **Average mIoU:** 33.21% ± 10.04
- **Normal Conditions mIoU:** 39.18%
- **Adverse Conditions mIoU:** 30.49%
- **Domain Gap (Normal - Adverse):** +8.68%

### Per-Domain Performance

| Domain | Type | mIoU | Std | Avg Images |
|--------|------|------|-----|------------|
| clear_day | NORMAL | 41.05% | ±7.00 | 1915 |
| cloudy | NORMAL | 37.30% | ±8.56 | 517 |
| dawn_dusk | ADVERSE | 29.82% | ±10.40 | 91 |
| foggy | NORMAL | 26.34% | ±7.46 | 290 |
| night | ADVERSE | 26.20% | ±7.45 | 88 |
| rainy | ADVERSE | 33.52% | ±8.94 | 190 |
| snowy | ADVERSE | 32.27% | ±11.93 | 180 |

### Per-Dataset Performance

| Dataset | Overall mIoU | Normal | Adverse | Gap |
|---------|--------------|--------|---------|-----|
| bdd10k | 41.32% | 47.01% | 38.47% | +8.54% |
| idd-aw | 29.75% | 38.39% | 26.28% | +12.11% |
| mapillaryvistas | 32.09% | 34.65% | 30.39% | +4.27% |
| outside15k | 30.09% | 36.65% | 26.81% | +9.84% |

### Per-Model Performance

| Model | Overall mIoU | Normal | Adverse | Gap |
|-------|--------------|--------|---------|-----|
| deeplabv3plus_r50 | 28.20% | 34.87% | 25.13% | +9.74% |
| pspnet_r50 | 31.10% | 37.34% | 28.30% | +9.04% |
| segformer_mit-b5 | 40.34% | 45.32% | 38.05% | +7.27% |

### Per-Configuration Performance

| Dataset | Model | Clear Day | Normal | Adverse | Overall | Gap |
|---------|-------|-----------|--------|---------|---------|-----|
| bdd10k | segformer_mit-b5 | 50.9% | 51.8% | 43.9% | 46.6% | +7.9% |
| outside15k | segformer_mit-b5 | 50.9% | 48.0% | 39.0% | 42.0% | +9.0% |
| bdd10k | pspnet_r50 | 44.5% | 46.3% | 37.6% | 40.5% | +8.7% |
| mapillaryvistas | segformer_mit-b5 | 40.4% | 39.3% | 35.5% | 37.0% | +3.8% |
| bdd10k | deeplabv3plus_r50 | 42.0% | 42.9% | 33.9% | 36.9% | +9.0% |
| idd-aw | segformer_mit-b5 | 46.0% | 42.1% | 33.1% | 35.9% | +9.0% |
| mapillaryvistas | deeplabv3plus_r50 | 32.3% | 32.8% | 27.9% | 29.8% | +4.9% |
| mapillaryvistas | pspnet_r50 | 31.8% | 31.8% | 27.8% | 29.4% | +4.1% |
| outside15k | pspnet_r50 | 38.4% | 34.9% | 24.7% | 28.1% | +10.3% |
| idd-aw | pspnet_r50 | 42.3% | 36.3% | 23.1% | 26.9% | +13.2% |
| idd-aw | deeplabv3plus_r50 | 43.4% | 36.7% | 22.6% | 26.4% | +14.1% |
| outside15k | deeplabv3plus_r50 | 29.6% | 27.0% | 16.8% | 20.2% | +10.3% |

## 2. Stage 2 Baseline Analysis (All Domains Training)

Models trained on ALL domains (clear_day + adverse).

**Source:** `/scratch/aaa_exchange/AWARE/WEIGHTS_STAGE_2/baseline/`

### Overall Statistics

- **Average mIoU:** 40.00% ± 9.70
- **Normal Conditions mIoU:** 41.95%
- **Adverse Conditions mIoU:** 37.71%
- **Domain Gap (Normal - Adverse):** +4.23%

### Per-Domain Performance

| Domain | Type | mIoU | Std |
|--------|------|------|-----|
| clear_day | NORMAL | 41.85% | ±6.60 |
| cloudy | NORMAL | 42.05% | ±7.81 |
| dawn_dusk | ADVERSE | 37.26% | ±11.31 |
| foggy | NORMAL | 58.71% | ±4.41 |
| night | ADVERSE | 33.37% | ±9.02 |
| rainy | ADVERSE | 39.45% | ±7.96 |
| snowy | ADVERSE | 40.66% | ±10.16 |

### Per-Dataset Performance

| Dataset | Overall mIoU | Normal | Adverse | Gap |
|---------|--------------|--------|---------|-----|
| bdd10k | 43.87% | 47.64% | 41.99% | +5.64% |
| idd-aw | 46.91% | 45.78% | 44.52% | +1.25% |
| mapillaryvistas | 33.95% | 35.78% | 32.73% | +3.05% |
| outside15k | 33.11% | 38.59% | 30.36% | +8.23% |

### Per-Model Performance

| Model | Overall mIoU | Normal | Adverse | Gap |
|-------|--------------|--------|---------|-----|
| deeplabv3plus_r50 | 35.87% | 38.49% | 33.10% | +5.38% |
| pspnet_r50 | 37.87% | 39.99% | 35.54% | +4.45% |
| segformer_mit-b5 | 46.26% | 47.37% | 44.50% | +2.87% |

## 3. Stage 1 vs Stage 2 Baseline Comparison

| Metric | Stage 1 (Clear Day) | Stage 2 (All Domains) | Difference |
|--------|---------------------|----------------------|------------|
| Overall mIoU | 33.21% | 40.00% | +6.79% |
| Normal mIoU | 39.18% | 41.95% | +2.77% |
| Adverse mIoU | 30.49% | 37.71% | +7.22% |
| Domain Gap | +8.68% | +4.23% | -4.45% |

### Key Insights

- **Stage 1** models are trained only on clear_day, testing cross-domain robustness
- **Stage 2** models are trained on all domains, testing domain-inclusive performance
- Stage 2 outperforms Stage 1 by **+6.79%** overall mIoU
- Stage 2 has **smaller domain gap** (4.23% vs 8.68%)

## 4. Key Insights

### Best and Worst Configurations (Stage 1)

- **Best:** bdd10k / segformer_mit-b5 (46.6% overall)
- **Worst:** outside15k / deeplabv3plus_r50 (20.2% overall)

### Model Ranking (by overall mIoU, Stage 1)

1. **segformer_mit-b5:** 40.3%
2. **pspnet_r50:** 31.1%
3. **deeplabv3plus_r50:** 28.2%

### Dataset Ranking (by overall mIoU, Stage 1)

1. **bdd10k:** 41.3%
2. **mapillaryvistas:** 32.1%
3. **outside15k:** 30.1%
4. **idd-aw:** 29.7%
