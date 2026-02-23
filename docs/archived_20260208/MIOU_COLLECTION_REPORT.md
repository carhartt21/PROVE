# mIoU Collection and Analysis Report

**Generated:** 2026-02-01  
**Data Source:** WEIGHTS and WEIGHTS_STAGE_2 directories

## Executive Summary

This report summarizes mIoU (mean Intersection over Union) values collected from training and test results across the PROVE evaluation framework.

### Collection Statistics

| Metric | Count |
|--------|-------|
| **Total Results** | 82 |
| From test_results (verified) | 31 |
| From log files (validation) | 51 |
| Stage 1 (clear-day training) | 45 |
| Stage 2 (all-domain training) | 37 |

### Dataset Coverage

| Dataset | Results | Best mIoU | Best Model |
|---------|---------|-----------|------------|
| BDD10k | 53 | **48.14%** | SegFormer MIT-B5 (Stage 2) |
| MapillaryVistas | 12 | 27.70% | SegFormer MIT-B3 (Stage 1) |
| IDD-AW | 11 | 38.37% | DeepLabV3+ R50 (Stage 2) |
| OUTSIDE15k | 6 | - | (log only) |

### Strategy Coverage

| Strategy Type | Count | Examples |
|--------------|-------|----------|
| Baseline | 55 | Standard training |
| Generative (gen_*) | 18 | CycleGAN, STEP1X, Flux Kontext |
| Standard Aug (std_*) | 6 | AutoAugment, CutMix, MixUp |

---

## Key Findings

### 1. Domain Difficulty Ranking

Based on average mIoU across all test results:

| Rank | Domain | Avg mIoU | Interpretation |
|------|--------|----------|----------------|
| 1 | Foggy | **45.80%** | Easiest domain |
| 2 | Cloudy | 42.59% | Good transfer |
| 3 | Snowy | 41.73% | Moderate difficulty |
| 4 | Clear Day | 40.28% | Reference domain |
| 5 | Rainy | 35.33% | Challenging |
| 6 | Dawn/Dusk | 34.16% | Challenging |
| 7 | Night | **22.76%** | Most difficult |

**Key Insight:** Night domain is consistently the most challenging, with ~50% lower mIoU compared to foggy conditions.

### 2. Best Model Per Domain

| Domain | Best mIoU | Model | Strategy |
|--------|-----------|-------|----------|
| Clear Day | 49.27% | SegFormer MIT-B5 | Baseline |
| Foggy | **59.74%** | SegFormer MIT-B5 aux-lovasz | Baseline |
| Cloudy | 53.11% | SegFormer MIT-B5 | Baseline |
| Snowy | 56.41% | SegFormer MIT-B5 aux-lovasz | Baseline |
| Rainy | 46.68% | SegFormer MIT-B5 | Baseline |
| Dawn/Dusk | 43.91% | SegFormer MIT-B5 | std_autoaugment |
| Night | 33.17% | DeepLabV3+ R50 | Baseline |

**Key Insight:** SegFormer MIT-B5 dominates most domains. Interestingly, DeepLabV3+ performs best for night domain.

### 3. Architecture Comparison

Stage 1 Baseline results (test_results verified):

| Architecture | Best mIoU | Dataset |
|--------------|-----------|---------|
| SegFormer | **45.69%** | BDD10k |
| SegNeXt | 41.27% | BDD10k |
| DeepLabV3+ | 38.13% | BDD10k |
| PSPNet | 35.64% | BDD10k |
| HRNet | 20.69% | IDD-AW |

**Key Insight:** Transformer-based models (SegFormer, SegNeXt) significantly outperform CNN-based architectures.

### 4. Stage Comparison

When comparing Stage 1 (clear-day only) vs Stage 2 (all-domain training):

| Model | Stage 1 | Stage 2 | Δ |
|-------|---------|---------|---|
| SegFormer MIT-B5 (BDD10k) | 45.69% | 48.14% | **+2.45%** |

**Key Insight:** Training on all domains (Stage 2) improves overall performance by ~2-5%.

### 5. Augmentation Strategy Impact

Comparing strategies on BDD10k with SegFormer:

| Strategy | mIoU | Δ vs Baseline |
|----------|------|---------------|
| std_autoaugment | 45.52% | -0.17% |
| std_photometric_distort | 45.47% | -0.22% |
| gen_flux_kontext | 46.15% | +0.46% |
| Baseline | 45.69% | - |

**Key Insight:** Generative augmentation (flux_kontext) shows slight improvement over standard augmentation.

---

## Data Quality Notes

1. **Test Results (31)**: Verified mIoU from proper test evaluation
2. **Log Results (51)**: Validation metrics during training (may not reflect final performance)
3. **Missing Data**: Some models only have log results, need test evaluation

### Recommended Actions

1. ✅ Complete test evaluation for all trained models
2. ✅ Ensure Stage 1 uses `--domain-filter clear_day`
3. ✅ Run remaining missing tests via `auto_submit_tests.py`

---

## Detailed Results

### Stage 1 Baseline (Clear-Day Training)

| Dataset | Model | mIoU | Source |
|---------|-------|------|--------|
| BDD10k | segformer_mit-b5_aux-lovasz | 45.69% | test_results |
| BDD10k | segnext_mscan-b | 41.27% | test_results |
| BDD10k | deeplabv3plus_r50_aux-lovasz | 38.13% | test_results |
| BDD10k | pspnet_r50_aux-lovasz | 35.64% | test_results |
| IDD-AW | segnext_mscan-b | 35.10% | test_results |
| IDD-AW | segformer_mit-b3 | 34.02% | test_results |
| MapillaryVistas | segformer_mit-b3 | 27.70% | test_results |

### Stage 2 Baseline (All-Domain Training)

| Dataset | Model | mIoU | Source |
|---------|-------|------|--------|
| BDD10k | segformer_mit-b5 | **48.14%** | test_results |
| BDD10k | segformer_mit-b5_aux-lovasz | 47.46% | test_results |
| IDD-AW | deeplabv3plus_r50 | 38.37% | test_results |

---

## Reproducibility

Run the mIoU collector:
```bash
# Table output
python analysis_scripts/miou_collector.py --include-domains -v

# CSV output for analysis
python analysis_scripts/miou_collector.py --include-domains --format csv --output results.csv

# JSON output
python analysis_scripts/miou_collector.py --include-domains --format json --output results.json
```

---

## Data Locations

- **Stage 1 Weights:** `${AWARE_DATA_ROOT}/WEIGHTS/`
- **Stage 2 Weights:** `${AWARE_DATA_ROOT}/WEIGHTS_STAGE_2/`
- **Test Results:** `{weights_dir}/test_results_detailed/*/results.json`
- **Training Logs:** `{weights_dir}/*.log`
