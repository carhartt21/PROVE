# PROVE Experiment Status Overview

**Last Updated:** 2026-02-02 12:00

---

## Executive Summary

| Metric | Stage 1 (clear_day) | Stage 2 (all domains) | Loss Ablation |
|--------|---------------------|----------------------|---------------|
| **Total Checkpoints** | 701 | 396 | 212 |
| **Fully Trained (80k)** | 15 | 23 | TBD |
| **Tested Results** | 39 | 25 | 8 (Stage 1) + 3 (Stage 2) |
| **Cluster Jobs Active** | 0 | 0 | 0 |

### Quick Stats
- **Total Strategies:** 28 (1 baseline + 6 std_* + 21 gen_*)
- **Models (6):** `deeplabv3plus_r50`, `pspnet_r50`, `segformer_mit-b3`, `segnext_mscan-b`, `hrnet_hr48`, `mask2former_swin-b`
- **Average mIoU:** Stage 1: 36.80%, Stage 2: 40.64%, Loss Ablation: 38.17%
- **Best Result (Stage 1):** gen_flux_kontext/bdd10k/segformer_mit-b5 → **46.15% mIoU**
- **Best Result (Stage 2):** baseline/bdd10k/segnext_mscan-b_ratio1p0 → **49.84% mIoU**

---

## Stage 1: Cross-Domain Robustness (clear_day training)

### Purpose
Train models ONLY on `clear_day` domain, test on ALL domains to evaluate cross-domain robustness.

### Strategies Available (28 total)

#### Baseline (1)
- `baseline`

#### Standard Augmentation (6)
- `std_autoaugment`
- `std_cutmix`
- `std_minimal`
- `std_mixup`
- `std_photometric_distort`
- `std_randaugment`

#### Generative Augmentation (21)
- `gen_albumentations_weather`
- `gen_Attribute_Hallucination`
- `gen_augmenters`
- `gen_automold`
- `gen_CNetSeg`
- `gen_CUT`
- `gen_cyclediffusion`
- `gen_cycleGAN`
- `gen_flux_kontext`
- `gen_Img2Img`
- `gen_IP2P`
- `gen_LANIT`
- `gen_Qwen_Image_Edit`
- `gen_stargan_v2`
- `gen_step1x_new`
- `gen_step1x_v1p2`
- `gen_SUSTechGAN`
- `gen_TSIT`
- `gen_UniControl`
- `gen_VisualCloze`
- `gen_Weather_Effect_Generator`

### Training Status by Dataset

| Dataset | Baseline | gen_* | std_* | Total Checkpoints |
|---------|----------|-------|-------|-------------------|
| **BDD10k** | 16 models | 45+ configs | 7 configs | ~70 |
| **IDD-AW** | 7 models | 3-5 per strategy | 3-4 per strategy | ~48 |
| **MapillaryVistas** | 7 models | 3-5 per strategy | 3 per strategy | ~35 |
| **OUTSIDE15k** | 7 models | 3-5 per strategy | 3-7 per strategy | ~49 |

### Fully Trained (80k iterations) - Stage 1

| Configuration | Status |
|--------------|--------|
| baseline/bdd10k/segnext_mscan-b | ✅ Tested |
| baseline/iddaw/hrnet_hr48 | ✅ Tested |
| baseline/iddaw/segformer_mit-b3 | ✅ Tested |
| baseline/mapillaryvistas/hrnet_hr48 | ✅ Tested |
| baseline/mapillaryvistas/segformer_mit-b3 | ✅ Tested |
| baseline/outside15k/hrnet_hr48 | ✅ Tested |
| baseline/outside15k/segformer_mit-b3 | ✅ Tested |
| baseline/outside15k/segnext_mscan-b | ✅ Tested |
| gen_cycleGAN/bdd10k/segformer_mit-b3_ratio0p50 | ✅ Tested |
| gen_cycleGAN/bdd10k/segnext_mscan-b_ratio0p50 | ✅ Tested |
| gen_flux_kontext/bdd10k/segformer_mit-b3_ratio0p50 | ✅ Tested |
| gen_LANIT/bdd10k/segformer_mit-b3_ratio0p50 | ✅ Tested |
| gen_step1x_new/bdd10k/segformer_mit-b3_ratio0p50 | ✅ Tested |
| gen_step1x_v1p2/bdd10k/segformer_mit-b3_ratio0p50 | ✅ Tested |
| gen_TSIT/mapillaryvistas/segformer_mit-b3_ratio0p50 | ✅ Tested |

### Test Results Summary (39 results)

*Note: Loss ablation results (aux-lovasz, aux-boundary, aux-focal) now tracked separately in WEIGHTS_LOSS_ABLATION*

#### Top 5 Configurations by mIoU
| Rank | Strategy | Dataset | Model | mIoU |
|------|----------|---------|-------|------|
| 1 | gen_flux_kontext | bdd10k | segformer_mit-b5_ratio0p50 | **46.15%** |
| 2 | std_autoaugment | bdd10k | segformer_mit-b5 | 45.52% |
| 3 | std_photometric_distort | bdd10k | segformer_mit-b5 | 45.47% |
| 4 | std_cutmix | bdd10k | segformer_mit-b5 | 45.29% |
| 5 | std_randaugment | bdd10k | segformer_mit-b5 | 45.25% |

#### Best per Dataset
| Dataset | Best Strategy | Best Model | mIoU |
|---------|--------------|------------|------|
| BDD10k | gen_flux_kontext | segformer_mit-b5_ratio0p50 | 46.15% |
| IDD-AW | baseline | segnext_mscan-b | 35.10% |
| MapillaryVistas | gen_UniControl | segnext_mscan-b_ratio0p50 | 34.98% |
| OUTSIDE15k | gen_step1x_v1p2 | segnext_mscan-b_ratio0p50 | 42.67% |

#### Strategy Performance (Average mIoU)
| Strategy Type | Avg mIoU | Max mIoU | Count |
|--------------|----------|----------|-------|
| std_autoaugment | 45.52% | 45.52% | 1 |
| std_photometric_distort | 45.47% | 45.47% | 1 |
| std_cutmix | 45.29% | 45.29% | 1 |
| std_randaugment | 45.25% | 45.25% | 1 |
| std_minimal | 44.91% | 44.91% | 1 |
| std_mixup | 43.81% | 44.78% | 2 |
| gen_flux_kontext | 43.19% | 46.15% | 2 |
| gen_step1x_v1p2 | 41.65% | 42.67% | 2 |
| gen_cycleGAN | 40.56% | 42.59% | 3 |
| gen_step1x_new | 41.09% | 42.47% | 2 |
| gen_LANIT | 39.80% | 39.80% | 1 |
| baseline | 29.95% | 41.27% | 10 |

#### Results by Dataset
| Dataset | Count | Avg mIoU | Best mIoU |
|---------|-------|----------|-----------|
| BDD10k | 17 | 42.24% | 46.15% |
| MapillaryVistas | 14 | 33.46% | 34.98% |
| OUTSIDE15k | 5 | 36.18% | 42.67% |
| IDD-AW | 3 | 29.94% | 35.10% |
| gen_flux_kontext | 43.19% | 46.15% | 2 |
| gen_step1x_v1p2 | 41.65% | 42.67% | 2 |
| gen_cycleGAN | 40.56% | 42.59% | 3 |
| baseline | 32.22% | 45.69% | 13 |

---

## Stage 2: Domain-Inclusive Training (all domains)

### Purpose
Train models on ALL domains including adverse weather, evaluate overall performance.

### Training Status

| Dataset | Baseline | gen_* | std_* |
|---------|----------|-------|-------|
| **BDD10k** | 22 models (6 tested) | 0 | 4 (0 ckpts) |
| **IDD-AW** | 6 models (2 tested) | 0 | 0 |
| **MapillaryVistas** | 11 models (0 tested) | 0 | 0 |
| **OUTSIDE15k** | 11 models (0 tested) | 0 | 0 |

### Fully Trained (80k iterations) - Stage 2

| Configuration | Status |
|--------------|--------|
| baseline/bdd10k/deeplabv3plus_r50 | ✅ Tested |
| baseline/bdd10k/deeplabv3plus_r50 | ✅ Tested |
| baseline/bdd10k/pspnet_r50 | ✅ Tested |
| baseline/bdd10k/deeplabv3plus_r50_ratio1p0 | ⏳ Untested |
| baseline/bdd10k/hrnet_hr48_ratio1p0 | ⏳ Untested |
| baseline/bdd10k/pspnet_r50_ratio1p0 | ⏳ Untested |
| baseline/bdd10k/segformer_mit-b3_ratio1p0 | ⏳ Untested |
| baseline/bdd10k/segnext_mscan-b_ratio1p0 | ⏳ Untested |
| baseline/iddaw/deeplabv3plus_r50 | ✅ Tested |
| baseline/idd-aw/deeplabv3plus_r50_ratio1p0 | ⏳ Untested |
| baseline/idd-aw/hrnet_hr48_ratio1p0 | ⏳ Untested |
| baseline/idd-aw/pspnet_r50_ratio1p0 | ⏳ Untested |
| baseline/idd-aw/segformer_mit-b3_ratio1p0 | ⏳ Untested |
| baseline/idd-aw/segnext_mscan-b_ratio1p0 | ⏳ Untested |
| baseline/mapillaryvistas/*_ratio1p0 (5 models) | ⏳ Untested |
| baseline/outside15k/*_ratio1p0 (5 models) | ⏳ Untested |

### Test Results Summary (5 results)

| Rank | Strategy | Dataset | Model | mIoU |
|------|----------|---------|-------|------|
| 1 | baseline | bdd10k | segformer_mit-b5 | **48.14%** |
| 2 | baseline | iddaw | deeplabv3plus_r50 | 38.37% |
| 3 | baseline | bdd10k | pspnet_r50 | 37.05% |
| 4 | baseline | iddaw | pspnet_r50 | 33.47% |
| 5 | baseline | bdd10k | deeplabv3plus_r50 | 30.80% |

*Note: Loss ablation results (aux-lovasz, etc.) now in separate WEIGHTS_LOSS_ABLATION directory*

---

## Cityscapes Replication (Pipeline Verification)

### Purpose
Verify training pipeline on standard Cityscapes benchmark before full experiments.

### Status
- Directory created: `${AWARE_DATA_ROOT}/WEIGHTS_CITYSCAPES/`
- Training started for: segformer_mit-b3, segnext_mscan-b
- **No checkpoints completed yet** (training in progress or failed)
- **No test results available**

---

## Coverage Gaps Analysis

### Stage 1 Critical Gaps

#### IDD-AW Dataset (only 3 baseline tested)
- Missing: gen_* strategies (only 1-2 checkpoints each, training at early stages)
- Missing: std_* strategies (only 2k-5k iters)
- **Priority:** Complete gen_step1x_new, gen_flux_kontext, gen_cycleGAN training

#### OUTSIDE15k Dataset (only 3 baseline + 1 gen tested)
- Missing: Most gen_* strategies
- Missing: All std_* strategies testing
- **Priority:** Complete remaining training runs

#### MapillaryVistas Dataset (2 baseline + 12 gen tested)
- Best coverage among non-BDD10k datasets
- Most gen_* strategies at 15k iterations (partial)
- **Priority:** Run tests on completed checkpoints

### Stage 2 Critical Gaps

#### ALL Datasets (ONLY baseline tested)
- **No gen_* strategies trained** (directories don't exist)
- **No std_* strategies trained** (empty directories with 0 checkpoints)
- 17 untested 80k checkpoints (mostly ratio1p0 variants)
- **Priority:** Submit tests for completed baseline checkpoints

---

## Models Used

| Model | Short Name | Architecture |
|-------|-----------|--------------|
| deeplabv3plus_r50 | DeepLabV3+ | ResNet-50 backbone |
| pspnet_r50 | PSPNet | ResNet-50 backbone |
| segformer_mit-b3 | SegFormer-B3 | MIT-B3 transformer |
| segformer_mit-b5 | SegFormer-B5 | MIT-B5 transformer |
| segnext_mscan-b | SegNeXt-B | MSCAN-B backbone |
| hrnet_hr48 | HRNet-W48 | High-Resolution Net |

---

## Key Findings

### Performance Insights
1. **Best overall:** gen_flux_kontext achieves 46.15% mIoU on BDD10k (Stage 1)
2. **std_* competitive:** Standard augmentation strategies (45.25-45.52%) nearly match best generative
3. **Model impact:** SegFormer-B5 consistently top performer across strategies
4. **Dataset difficulty:** MapillaryVistas hardest (~27-35% mIoU), OUTSIDE15k moderate (~37-43%)

### Training Progress
1. **Stage 1:** 15/784 checkpoints at 80k iterations (1.9%)
2. **Stage 2:** 25/522 checkpoints at 80k iterations (4.8%)
3. **Many early-stage runs:** Most at 2k-15k iterations (need completion)

### Testing Backlog
1. **Stage 1:** 31 untested checkpoints (tested 15/46 complete)
2. **Stage 2:** 17 untested 80k checkpoints
3. **Priority:** Run auto_submit_tests scripts to clear backlog

---

## Recommended Actions

### Immediate (This Week)
1. ⏳ Run `auto_submit_tests_stage2.py` for Stage 2 baseline testing
2. ⏳ Monitor running job: `s1_std_mixup_outside15k_segformer` 
3. ⏳ Submit tests for MapillaryVistas 15k checkpoints

### Short-term (Next 2 Weeks)
1. Complete IDD-AW and OUTSIDE15k gen_* training for Stage 1
2. Submit Stage 2 gen_* training jobs (none exist currently)
3. Run comprehensive testing on all 80k checkpoints

### Medium-term (Next Month)
1. Analyze cross-domain robustness patterns
2. Generate publication-ready figures
3. Complete Cityscapes replication for validation

---

## File Locations

| Type | Path |
|------|------|
| Stage 1 Weights | `${AWARE_DATA_ROOT}/WEIGHTS/` |
| Stage 2 Weights | `${AWARE_DATA_ROOT}/WEIGHTS_STAGE_2/` |
| **Loss Ablation** | `${AWARE_DATA_ROOT}/WEIGHTS_LOSS_ABLATION/` |
| Cityscapes | `${AWARE_DATA_ROOT}/WEIGHTS_CITYSCAPES/` |
| Ratio Ablation | `${AWARE_DATA_ROOT}/WEIGHTS_RATIO_ABLATION/` |
| Extended Training | `${AWARE_DATA_ROOT}/WEIGHTS_EXTENDED/` |

---

## Loss Function Ablation Study

**Location:** `${AWARE_DATA_ROOT}/WEIGHTS_LOSS_ABLATION/`

This directory contains experiments testing different loss function configurations, separated from the main experiments for clarity.

### Loss Types Evaluated

| Loss Type | Suffix | Description |
|-----------|--------|-------------|
| Auxiliary Lovasz-Softmax | `_aux-lovasz` | Standard CE loss + Lovasz-Softmax auxiliary |
| Auxiliary Boundary | `_aux-boundary` | Standard CE loss + Boundary loss auxiliary |
| Auxiliary Focal | `_aux-focal` | Standard CE loss + Focal loss auxiliary |
| Main Lovasz-Softmax | `_loss-lovasz` | Lovasz-Softmax as main loss (replaces CE) |

### Checkpoint Summary

| Stage | Checkpoints | Test Results | Best mIoU |
|-------|-------------|--------------|-----------|
| Stage 1 | 86 | 8 | 45.69% (baseline/segformer_mit-b5_aux-lovasz) |
| Stage 2 | 126 | 3 | 47.46% (baseline/segformer_mit-b5_aux-lovasz) |

### Top Loss Ablation Results (Stage 1)

| Strategy | Model | Loss Type | mIoU |
|----------|-------|-----------|------|
| baseline | segformer_mit-b5_aux-lovasz | aux-lovasz | **45.69%** |
| gen_step1x_new | segformer_mit-b5_ratio0p50_aux-lovasz | aux-lovasz | 45.34% |
| gen_stargan_v2 | segformer_mit-b5_ratio0p50_aux-lovasz | aux-lovasz | 45.14% |
| baseline | deeplabv3plus_r50_aux-lovasz | aux-lovasz | 38.13% |
| baseline | pspnet_r50_aux-lovasz | aux-lovasz | 35.64% |

### Directory Structure

```
WEIGHTS_LOSS_ABLATION/
├── stage1/
│   ├── baseline/
│   │   ├── bdd10k/     (12 model variants)
│   │   ├── iddaw/      (3 model variants)
│   │   ├── mapillaryvistas/ (3 model variants)
│   │   └── outside15k/ (3 model variants)
│   ├── gen_stargan_v2/bdd10k/ (6 model variants)
│   ├── gen_step1x_new/bdd10k/ (6 model variants)
│   └── std_*/bdd10k/   (3 models each × 6 strategies)
└── stage2/
    ├── baseline/       (same structure as stage1)
    └── std_*/bdd10k/   (3 models each × 6 strategies)
```

### Models with Loss Variants

- `deeplabv3plus_r50` (supports aux_head)
- `pspnet_r50` (supports aux_head)  
- `segformer_mit-b5` (no aux_head - uses loss replacement)

---

*Generated automatically from PROVE experiment status analysis*
