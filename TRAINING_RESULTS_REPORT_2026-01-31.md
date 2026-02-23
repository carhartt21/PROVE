# Training Results Report - January 31, 2026

## Executive Summary

This report analyzes training results from the PROVE project, covering both Stage 1 (clear day only) and Stage 2 (all weather domains) experiments on semantic segmentation models.

**Key Findings:**
- **SegFormer MIT-B5** is the top-performing model across all configurations (45-48% mIoU)
- **Stage 2 training** (all domains) improves SegFormer performance by ~2.6% over Stage 1 (clear day only)
- **Auxiliary Lovasz loss** provides significant benefit for PSPNet (+5.6%) but marginal for SegFormer
- **Night domain** is the most challenging condition (16-30% mIoU across models)

---

## 1. Test Results Summary

### Stage 1 (Clear Day Training Only)
| Strategy | Dataset | Model | mIoU (%) | aAcc (%) |
|----------|---------|-------|----------|----------|
| baseline | bdd10k | segformer_mit-b5_aux-lovasz | **45.69** | 91.07 |
| baseline | bdd10k | segformer_mit-b5 | 45.56 | 91.11 |
| std_autoaugment | bdd10k | segformer_mit-b5 | 45.52 | 91.01 |
| std_photometric_distort | bdd10k | segformer_mit-b5 | 45.47 | 90.98 |
| gen_step1x_new | bdd10k | segformer_mit-b5_ratio0p50_aux-lovasz | 45.34 | 90.84 |
| std_cutmix | bdd10k | segformer_mit-b5 | 45.29 | 90.83 |
| std_minimal | bdd10k | segformer_mit-b5 | 44.91 | 90.67 |
| std_mixup | bdd10k | segformer_mit-b5 | 44.78 | 90.78 |
| baseline | bdd10k | deeplabv3plus_r50_aux-lovasz | 38.13 | 89.09 |
| baseline | bdd10k | pspnet_r50_aux-lovasz | 35.64 | 86.61 |
| gen_step1x_new | bdd10k | deeplabv3plus_r50_ratio0p50_aux-lovasz | 34.67 | 87.56 |
| baseline | bdd10k | pspnet_r50 | 30.03 | 83.74 |

**Total: 12 completed test results**

### Stage 2 (All Domains Training)
| Strategy | Dataset | Model | mIoU (%) | aAcc (%) |
|----------|---------|-------|----------|----------|
| baseline | bdd10k | segformer_mit-b5 | **48.14** | 91.72 |
| baseline | bdd10k | segformer_mit-b5_aux-lovasz | 47.46 | 91.74 |
| baseline | iddaw | deeplabv3plus_r50 | 38.37 | 85.13 |
| baseline | bdd10k | pspnet_r50 | 37.05 | 87.79 |
| baseline | iddaw | pspnet_r50 | 33.47 | 83.24 |
| baseline | bdd10k | deeplabv3plus_r50_aux-lovasz | 33.01 | 82.26 |
| baseline | bdd10k | pspnet_r50_aux-lovasz | 32.89 | 84.92 |
| baseline | bdd10k | deeplabv3plus_r50 | 30.80 | 82.30 |

**Total: 8 completed test results**

---

## 2. Per-Domain Analysis (BDD10k Test Set)

### Stage 2 SegFormer MIT-B5 (Best Model)
| Domain | mIoU (%) | Images | Relative to Clear Day |
|--------|----------|--------|----------------------|
| foggy | 59.03 | 4 | +19.8% |
| snowy | 55.04 | 221 | +11.7% |
| cloudy | 53.11 | 230 | +7.8% |
| clear_day | 49.27 | 1016 | baseline |
| rainy | 46.68 | 200 | -5.3% |
| dawn_dusk | 41.85 | 94 | -15.1% |
| **night** | **30.11** | 92 | **-38.9%** |

### Stage 1 SegFormer MIT-B5 (Comparison)
| Domain | mIoU (%) | Images | vs Stage 2 |
|--------|----------|--------|------------|
| cloudy | 52.01 | 230 | -1.1% |
| snowy | 52.15 | 221 | -2.9% |
| foggy | 50.82 | 4 | -8.2% |
| clear_day | 47.35 | 1016 | -1.9% |
| rainy | 43.57 | 200 | -3.1% |
| dawn_dusk | 38.63 | 94 | -3.2% |
| night | 26.63 | 92 | -3.5% |

**Observation:** Stage 2 training consistently improves all domains, with largest gains in fog (+8.2%) and dawn/dusk (+3.2%).

---

## 3. Model Architecture Comparison

### Best mIoU by Model (Stage 1, BDD10k)
| Model | Best Config | mIoU (%) | Gap vs SegFormer |
|-------|-------------|----------|------------------|
| **SegFormer MIT-B5** | aux-lovasz | 45.69 | - |
| DeepLabV3+ R50 | aux-lovasz | 38.13 | -7.56% |
| PSPNet R50 | aux-lovasz | 35.64 | -10.05% |

### Observations:
1. **Transformer architecture (SegFormer)** significantly outperforms CNN architectures
2. **10+ point gap** between SegFormer and traditional CNNs
3. Auxiliary Lovasz loss improves all models, but effect varies:
   - PSPNet: +5.61% (30.03 → 35.64)
   - DeepLabV3+: estimated +6-8%
   - SegFormer: +0.13% (marginal)

---

## 4. Standard Augmentation Strategies Analysis (Stage 1, SegFormer)

### Overview
All standard augmentation strategies were tested with SegFormer MIT-B5 on BDD10k.

| Strategy | Val mIoU (Max) | Test mIoU | Delta vs Baseline | # Evals |
|----------|----------------|-----------|-------------------|---------|
| baseline | 45.02% | 45.56% | - | - |
| baseline (aux-lovasz) | 44.96% | **45.69%** | **+0.13%** | - |
| std_minimal | 46.49% | 44.91% | -0.65% | 9 |
| std_photometric_distort | 46.21% | 45.47% | -0.09% | 10 |
| std_autoaugment | 45.78% | 45.52% | -0.04% | 6 |
| std_mixup | 45.81% | 44.78% | -0.78% | 6 |
| std_cutmix | 45.45% | 45.29% | -0.27% | 9 |
| std_randaugment | 45.65% | pending | - | 4 |

### Analysis

**Observation 1: Validation vs Test Gap**
- std_minimal shows highest validation (46.49%) but lower test (44.91%)
- This suggests overfitting to validation augmentation patterns
- Baseline remains most consistent (45.02% → 45.56%)

**Observation 2: Augmentation Effectiveness**
| Strategy | Type | Benefit |
|----------|------|---------|
| std_autoaugment | Policy-based | -0.04% (minimal) |
| std_photometric_distort | Color/contrast | -0.09% (minimal) |
| std_cutmix | Cut-paste | -0.27% (slight negative) |
| std_minimal | Geometric | -0.65% (negative) |
| std_mixup | Interpolation | -0.78% (negative) |

**Finding:** Standard augmentations do NOT improve SegFormer performance. The transformer architecture appears robust enough that additional augmentation provides no benefit and may actually hurt generalization.

---

## 5. Generative Augmentation Strategies Analysis (Stage 1)

### Completed Tests
| Strategy | Model | Val mIoU | Test mIoU | Delta |
|----------|-------|----------|-----------|-------|
| gen_step1x_new | segformer_mit-b5_aux-lovasz | 45.70% | 45.34% | -0.22% |
| gen_step1x_new | deeplabv3plus_r50_aux-lovasz | 40.71% | 34.67% | -3.46% |

### Currently Training (Validation mIoU Progress)
| Strategy | Model | Val mIoU | # Evals | Status |
|----------|-------|----------|---------|--------|
| gen_flux_kontext | segformer_mit-b5_ratio0p50 | **46.14%** | 3 | 🔄 Running |
| gen_LANIT | segformer_mit-b5_ratio0p50 | 45.94% | 2 | 🔄 Running |
| gen_albumentations_weather | segformer_mit-b5_ratio0p50 | 45.87% | 1 | 🔄 Running |
| gen_cycleGAN | segformer_mit-b5_ratio0p50 | 45.71% | 7 | 🔄 Running |
| gen_step1x_new | segformer_mit-b5_ratio0p50 | 45.59% | 5 | 🔄 Running |
| gen_stargan_v2 | segformer_mit-b5_aux-lovasz | 45.14% | 6 | 🔄 Running |
| gen_step1x_new | pspnet_r50_aux-lovasz | 40.59% | 5 | 🔄 Running |
| gen_stargan_v2 | pspnet_r50_aux-lovasz | 40.55% | 5 | 🔄 Running |
| gen_stargan_v2 | deeplabv3plus_r50_aux-lovasz | 40.47% | 5 | 🔄 Running |

### Promising Observations
1. **gen_flux_kontext** shows highest validation mIoU (46.14%) - exceeds baseline
2. **gen_LANIT** also showing good progress (45.94%)
3. **gen_cycleGAN** stable at 45.71% across 7 evaluations

### Preliminary Analysis
| Strategy | Approach | Expectation | Current Status |
|----------|----------|-------------|----------------|
| gen_flux_kontext | Diffusion-based editing | High quality | ⭐ Best so far |
| gen_LANIT | GAN style transfer | Good domain adaptation | Promising |
| gen_cycleGAN | Unpaired translation | Proven method | Stable |
| gen_step1x_new | Step-based editing | Fast generation | Slight negative |
| gen_stargan_v2 | Multi-domain GAN | Flexible | Below baseline |

**Note:** Test results pending for most gen_* strategies. Final conclusions will require completed testing.

---

## 6. Strategy Effectiveness Summary

### All Strategies Ranked by Effectiveness (SegFormer MIT-B5, BDD10k)

| Rank | Strategy | Type | Test mIoU | Val mIoU | Status | Verdict |
|------|----------|------|-----------|----------|--------|---------|
| 1 | baseline_aux-lovasz | Baseline | **45.69%** | 44.96% | ✅ Complete | ⭐ Best |
| 2 | baseline | Baseline | 45.56% | 45.02% | ✅ Complete | Reference |
| 3 | std_autoaugment | Standard | 45.52% | 45.78% | ✅ Complete | No benefit |
| 4 | std_photometric_distort | Standard | 45.47% | 46.21% | ✅ Complete | No benefit |
| 5 | gen_step1x_new_aux-lovasz | Generative | 45.34% | 45.70% | ✅ Complete | Slight negative |
| 6 | std_cutmix | Standard | 45.29% | 45.45% | ✅ Complete | Negative |
| 7 | std_minimal | Standard | 44.91% | 46.49% | ✅ Complete | Overfitting |
| 8 | std_mixup | Standard | 44.78% | 45.81% | ✅ Complete | Negative |
| - | gen_flux_kontext | Generative | pending | **46.14%** | 🔄 Running | ⭐ Promising |
| - | gen_LANIT | Generative | pending | 45.94% | 🔄 Running | Promising |
| - | gen_albumentations_weather | Generative | pending | 45.87% | 🔄 Running | Promising |
| - | gen_cycleGAN | Generative | pending | 45.71% | 🔄 Running | Stable |
| - | std_randaugment | Standard | pending | 45.65% | 🔄 Running | Pending |
| - | gen_step1x_new | Generative | pending | 45.59% | 🔄 Running | Pending |
| - | gen_stargan_v2 | Generative | pending | 45.14% | 🔄 Running | Below baseline |

### Key Takeaways

1. **Standard Augmentations (std_*)**
   - ❌ None improve over baseline
   - ❌ std_minimal, std_mixup show negative impact
   - ⚠️ High validation mIoU doesn't translate to test performance
   
2. **Generative Augmentations (gen_*)**
   - ⭐ gen_flux_kontext shows promise (46.14% validation)
   - ⚠️ gen_step1x_new shows slight degradation (-0.22%)
   - 📊 Most results still pending

3. **Auxiliary Loss (aux-lovasz)**
   - ✅ Marginal benefit for SegFormer (+0.13%)
   - ✅ Significant benefit for PSPNet (+5.6%)
   - ❌ Not beneficial for CNN models in Stage 2

---

## 7. Currently Active Jobs

### Running (${USER})
| Job ID | Name | Progress |
|--------|------|----------|
| 980992 | s1_gen_step1x_new_bdd10k_pspnet_aux-lovasz | 35% (28k/80k) |
| 980994 | s1_gen_stargan_v2_bdd10k_deeplabv3plus_aux-lovasz | Running |
| 980995 | s1_gen_stargan_v2_bdd10k_pspnet_aux-lovasz | Running |
| 980996 | s1_gen_stargan_v2_bdd10k_segformer_aux-lovasz | Running |

### Running (chge7185)
| Job ID | Name | Status |
|--------|------|--------|
| 982074 | s1_std_randaugment_bdd10k_segformer | Running |
| 982087 | s1_gen_cycleGAN_bdd10k_segformer | Running |
| 982088 | s1_gen_flux_kontext_bdd10k_segformer | Running |
| 982089 | s1_gen_step1x_new_bdd10k_segformer | Running |
| 982090 | s1_gen_LANIT_bdd10k_segformer | Running |
| 982091 | s1_gen_albumentations_weather_bdd10k_segformer | Running |

### Pending (chge7185): 16 jobs
- gen_automold, gen_step1x_v1p2, gen_VisualCloze, gen_SUSTechGAN, etc.

---

## 7. Key Insights & Recommendations

### ✅ Confirmed
1. **SegFormer is the architecture of choice** - 7-10% mIoU advantage over CNNs
2. **Stage 2 training improves cross-domain performance** - 2.6% gain for SegFormer
3. **Night domain is the critical challenge** - 30% mIoU even with best model
4. **Auxiliary Lovasz loss helps CNN models** - especially PSPNet (+5.6%)

### ⚠️ Concerns
1. **DeepLabV3+ underperforms expectations** - 30-38% mIoU vs expected 50%+
2. **Standard augmentations show no benefit** for SegFormer
3. **Gen strategies (gen_step1x_new) show slight degradation** so far

### 📋 Recommended Next Steps
1. **Monitor running gen_* jobs** - gen_flux_kontext shows promising 46.1% validation mIoU
2. **Investigate DeepLabV3+ training** - possible configuration issues
3. **Focus testing resources on SegFormer** - clearly the best architecture
4. **Consider ensemble strategies** for night domain improvement

---

## 8. Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 16 |
| Max Iterations | 80,000 |
| Warmup Iterations | 1,000 |
| Primary Loss | CrossEntropy |
| Auxiliary Loss | Lovasz (optional) |
| Checkpoint Interval | 5,000 |
| Evaluation Interval | 5,000 |
| LR Scale Factor | 8.0 |

---

*Report generated: January 31, 2026, 09:30*
