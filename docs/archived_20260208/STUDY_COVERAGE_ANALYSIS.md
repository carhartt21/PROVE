# Study Coverage Analysis

**Last Updated:** 2026-01-28 (13:30)

---

## ⚠️ CRITICAL WARNING: gen_* Results Invalid

> **MixedDataLoader Bug Discovered (Jan 28, 2026):** Generated images were **NEVER LOADED** during training.
> All `gen_*` strategy comparisons below are **INVALID** - they compared pipeline augmentation only.
> 
> **Status:** Bug is **FIXED**. Retraining required to generate valid results.
> 
> See [BUG_REPORT](BUG_REPORT_CROSS_DATASET_CONTAMINATION.md) for details.

### What's Valid vs Invalid

| Category | Status | Notes |
|----------|--------|-------|
| `baseline` results | ✅ **VALID** | No generated images expected |
| `std_*` strategy results | ✅ **VALID** | Use pipeline augmentation, not generative |
| `std_photometric_distort` results | ✅ **VALID** | Pipeline augmentation only |
| `gen_*` strategy results | ❌ **INVALID** | Generated images never loaded |
| Ratio ablation results | ❌ **INVALID** | Ratio parameter had no effect |
| gen_* vs baseline comparisons | ❌ **INVALID** | Compared PhotoMetricDistortion, not generative |

---

## Executive Summary (OUTDATED - DO NOT CITE)

This document provides comprehensive coverage analysis and key findings from the PROVE semantic segmentation evaluation study. The study evaluates augmentation strategies for improving model robustness under adverse weather conditions using MMSegmentation across 4 datasets, 3 models, and 27 strategies.

### High-Level Findings (⚠️ gen_* findings INVALID)

| Study | Key Takeaway | Best Performer | Status |
|-------|--------------|----------------|--------|
| **Stage 1** | gen_Attribute_Hallucination best (+1.4 mIoU) | gen_Attribute_Hallucination | ❌ INVALID |
| **Stage 2** | Gains compress when training includes all domains | gen_stargan_v2 | ❌ INVALID |
| **Ratio Ablation** | Optimal synthetic ratio is 12–38% | 0.75 ratio (25% gen) | ❌ INVALID |
| **Extended Training** | 77% configs benefit; baseline overfits | gen_cyclediffusion | ⚠️ Baseline only valid |
| **Combinations** | std_photometric_distort combos dominate | std_mixup+photo | ⚠️ Partially valid |
| **Domain Adaptation** | ALL 15/15 strategies beat baseline | gen_stargan_v2 | ❌ INVALID |

---

## Summary

| Study | Path | Checkpoints | Test Results | Status |
|-------|------|-------------|--------------|--------|
| **Stage 1** | `WEIGHTS/` | 324 | 324 | ⚠️ gen_* invalid |
| **Stage 2** | `WEIGHTS_STAGE_2/` | 325 | 325 | ⚠️ gen_* invalid |
| **Ratio Ablation** | `WEIGHTS_RATIO_ABLATION/` | 284 | 279 (98%) | ❌ **ALL INVALID** |
| **Extended Training** | `WEIGHTS_EXTENDED/` | 970 | 764 (79%) | ⚠️ Baseline only valid |
| **Combinations** | `WEIGHTS_COMBINATIONS/` | 53 | 53 | ⚠️ std_* only valid |
| **Domain Adaptation** | `WEIGHTS/domain_adaptation_ablation/` | N/A (reuses S1) | 64 | ❌ **INVALID** |

### Mismatch Analysis (Jan 28, 2026)

| Study | Checkpoints | Tests | Missing | Cause |
|-------|-------------|-------|---------|-------|
| Stage 1 | 324 | 324 | 0 | — |
| Stage 2 | 325 | 325 | 0 | — |
| Ratio Ablation | 284 | 279 | 5 | Permission: `chge7185` checkpoints have `-rw-------` |
| Extended Training | 970 | 764 | 206 | Early checkpoints (10k-70k) not prioritized |
| Combinations | 53 | 53 | 0 | — |

#### Blocked Checkpoints (Ratio Ablation)
Require `chge7185` to run `chmod 644 iter_80000.pth`:
```
gen_flux_kontext/iddaw/segformer_mit-b5_ratio0p00
gen_flux_kontext/iddaw/segformer_mit-b5_ratio0p25
gen_flux_kontext/iddaw/segformer_mit-b5_ratio0p62
gen_flux_kontext/iddaw/segformer_mit-b5_ratio0p75
gen_flux_kontext/iddaw/segformer_mit-b5_ratio0p88
```

### Directory Naming Convention

| Directory | Dataset Naming | Notes |
|-----------|---------------|-------|
| `WEIGHTS/`, `WEIGHTS_STAGE_2/` | `idd-aw` (hyphen) | Main directories |
| `WEIGHTS_RATIO_ABLATION/` | `iddaw` (no hyphen) | Has `stage1/`, `stage2/` subdirs |
| `WEIGHTS_EXTENDED/` | `iddaw` (no hyphen) | All Stage 2 training |
| `WEIGHTS_COMBINATIONS/` | `iddaw` (no hyphen) | All Stage 2 training |

**Common datasets:** `bdd10k`, `iddaw` or `idd-aw`, `mapillaryvistas`, `outside15k`

---

## 1. Ratio Ablation Study

**Path:** `${AWARE_DATA_ROOT}/WEIGHTS_RATIO_ABLATION/`
**Status:** 🔄 **IN PROGRESS** (275/281 checkpoints, 3 training jobs running)

### Ratio Definition

The **ratio** (`real_gen_ratio`) represents the proportion of **REAL images** in the training data:
- **Ratio 1.00** = 100% real images, 0% generated (baseline - from WEIGHTS/)
- **Ratio 0.50** = 50% real images, 50% generated (standard gen training)
- **Ratio 0.25** = 25% real images, 75% generated
- **Ratio 0.00** = 0% real images, 100% generated

### Final Results (2026-01-27)

**Training:** 🔄 275/281 complete + 3 jobs running
**Testing:** ✅ 275/275 existing tested (100%)
**Analysis:** ✅ 467 total results, 108 globally common (2 configs × 6 strategies × 9 ratios)

### Key Finding: Best Ratio is 0.75 (25% Generated!)

**Globally Common Comparison** (2 identical (dataset, model) pairs across ALL strategies):
- bdd10k/pspnet_r50
- iddaw/pspnet_r50

| Ratio | Description | Avg mIoU | Count | Δ vs Baseline |
|-------|-------------|----------|-------|---------------|
| **1.00** | **100% real (baseline)** | **39.90%** | **12** | **—** |
| 0.88 | 88% real, 12% gen | 41.04% | 12 | +1.14% |
| **0.75** | **75% real, 25% gen** | **41.46%** | **12** | **+1.56%** |
| 0.62 | 62% real, 38% gen | 41.28% | 12 | +1.38% |
| 0.50 | 50% real, 50% gen | 41.39% | 12 | +1.49% |
| 0.38 | 38% real, 62% gen | 41.36% | 12 | +1.46% |
| 0.25 | 25% real, 75% gen | 41.00% | 12 | +1.10% |
| 0.12 | 12% real, 88% gen | 40.83% | 12 | +0.93% |
| 0.00 | 0% real, 100% gen | 41.25% | 12 | +1.35% |

**Key Insight:** All generated data ratios outperform baseline by +0.9-1.6% mIoU. The optimal ratio is **0.75** (75% real + 25% generated) achieving **+1.56% mIoU improvement**. Baseline is now **identical** across all strategies (39.90%).

**Exclusions for Valid Comparison:**
- MapillaryVistas: different num_classes between WEIGHTS and WEIGHTS_RATIO_ABLATION
- OUTSIDE15k: excluded to focus on the two main datasets (BDD10k and IDD-AW)
- deeplabv3plus_r50: only exists at ratio 0.50 and 1.00, not in full ablation range
- gen_step1x_*: **stage mismatch** - ablation is stage 2 (all domains), but ratio 0.50/1.00 from WEIGHTS/ are stage 1 (clear_day only)

### Strategy Coverage (in Globally Common Comparison)

| Strategy | Checkpoints | Tests | In Global Common |
|----------|-------------|-------|------------------|
| gen_Attribute_Hallucination | 28 | 28 | ✅ 2 configs |
| gen_cycleGAN | 42 | 42 | ✅ 2 configs |
| gen_cyclediffusion | 28 | 28 | ✅ 2 configs |
| gen_flux_kontext | 28 | 28 | ✅ 2 configs |
| gen_Img2Img | 28 | 28 | ✅ 2 configs |
| gen_stargan_v2 | 28 | 28 | ✅ 2 configs |
| gen_step1x_new | 56 | 56 | ❌ Stage mismatch |
| gen_step1x_v1p2 | 37 | 37 | ❌ Stage mismatch |
| **Total** | **275** | **275** | **12 configs/ratio** |

### Visualizations
- [figures/ratio_ablation/](../figures/ratio_ablation/) - 7 figures generated

### Analysis
```bash
python analysis_scripts/analyze_ratio_ablation.py --verbose
python analysis_scripts/visualize_ratio_ablation.py
```

---

## 2. Extended Training Study

**Path:** `${AWARE_DATA_ROOT}/WEIGHTS_EXTENDED/`
**Status:** 🔄 **BASELINE TRAINING IN PROGRESS** (714 gen results + 4 baseline jobs running)

### Baseline Extended Training (Jan 28, 2026)

Training baseline models from 80k→320k iterations for fair comparison with generative strategies:

| Job ID | Configuration | Status | Output |
|--------|---------------|--------|--------|
| 443834 | bdd10k/pspnet_r50 | **RUN** | `baseline/bdd10k/pspnet_r50/` |
| 443835 | bdd10k/segformer_mit-b5 | **RUN** | `baseline/bdd10k/segformer_mit-b5/` |
| 443836 | iddaw/pspnet_r50 | **RUN** | `baseline/iddaw/pspnet_r50/` |
| 443837 | iddaw/segformer_mit-b5 | **RUN** | `baseline/iddaw/segformer_mit-b5/` |

**Test Submission Script:** `python scripts/submit_baseline_extended_tests.py --submit`

### Key Findings (Generative Strategies Only)

| Metric | Value |
|--------|-------|
| **Total improvement** | +12.09% mIoU (37.7% → 49.79%) |
| **Configs improved** | 77.4% (24/31) |
| **Mean improvement** | +1.41 mIoU from 80k to optimal |
| **Max improvement** | +4.10 mIoU |
| **Best strategy** | gen_cyclediffusion (53.81% at 320k) |

**Diminishing Returns:** 75% of gains achieved by 160k iterations

### Performance by Iteration

| Iteration | Avg mIoU | Count |
|-----------|----------|-------|
| 10k | 37.70% | 26 |
| 40k | 42.62% | 26 |
| 80k | 45.01% | 25 |
| 160k | 49.29% | 21 |
| 320k | 49.79% | 21 |

### Visualizations
- [result_figures/extended_training/](../result_figures/extended_training/) - 9 figures generated

### Analysis
```bash
python analysis_scripts/analyze_extended_training.py
python analysis_scripts/visualize_extended_training.py
```

---

## 3. Strategy Combinations Study

**Path:** `${AWARE_DATA_ROOT}/WEIGHTS_COMBINATIONS/`
**Status:** ✅ **COMPLETE** (53/53 tested)

### Key Findings

| Metric | Value |
|--------|-------|
| **Generative+Standard avg** | 40.1% mIoU |
| **Standard+Standard avg** | 39.7% mIoU |

### Top Combinations by mIoU

| Combination | Mean mIoU |
|-------------|-----------|
| std_mixup+std_photometric_distort | 45.22% |
| std_autoaugment+std_photometric_distort | 45.18% |
| gen_step1x_new+std_photometric_distort | 45.18% |
| gen_Attribute_Hallucination+std_photometric_distort | 45.17% |
| gen_stargan_v2+std_photometric_distort | 45.17% |

**Insight:** Photometric distortion is the key booster for all strategy combinations.

### Coverage

| Combination Type | Strategies | Checkpoints | Tests |
|------------------|------------|-------------|-------|
| gen_* + std_photometric_distort | 4 | 8 | 8 |
| gen_flux_kontext + std_* | 5 | 10 | 10 |
| gen_Qwen_Image_Edit + std_* | 5 | 10 | 10 |
| gen_step1x_new + std_* | 5 | 9 | 9 |
| std_* + std_* | 8 | 16 | 16 |
| **Total** | **27** | **53** | **53** |

### Limitations
- **Dataset:** iddaw only (IDD-AW)
- **Models:** pspnet_r50, segformer_mit-b5
- **Missing:** bdd10k, mapillaryvistas, outside15k

### Visualizations
- [result_figures/combination_ablation/](../result_figures/combination_ablation/)

---

## 4. Domain Adaptation Testing

**Path:** `${AWARE_DATA_ROOT}/WEIGHTS/domain_adaptation_ablation/`
**Status:** ✅ **COMPLETE** (64 configurations tested)

### Study Design

Cross-dataset evaluation using **Stage 1 checkpoints** tested on **ACDC** (unseen dataset with weather domains).

**Key Concept:** Models trained on BDD10k/IDD-AW with clear_day domain are evaluated on ACDC's adverse weather conditions (foggy, rainy, snowy, night, clear_day) to measure domain generalization.

### Fair Comparison (bdd10k + idd-aw × pspnet_r50 + segformer_mit-b5)

Using 4 consistent configurations for valid comparison (2 datasets × 2 models):

| Rank | Strategy | Mean mIoU | Δ Baseline |
|------|----------|-----------|------------|
| 1 | gen_stargan_v2 | 26.94% | **+1.96%** |
| 2 | std_photometric_distort | 26.87% | +1.88% |
| 3 | gen_cycleGAN | 26.70% | +1.72% |
| 4 | std_autoaugment | 26.64% | +1.66% |
| 5 | gen_step1x_v1p2 | 26.61% | +1.62% |
| 6 | gen_cyclediffusion | 26.59% | +1.60% |
| 7 | gen_UniControl | 26.57% | +1.58% |
| 8 | gen_albumentations_weather | 26.52% | +1.53% |
| 9 | gen_flux_kontext | 26.33% | +1.35% |
| 10 | std_mixup | 26.29% | +1.31% |
| 11 | gen_TSIT | 26.27% | +1.28% |
| 12 | std_cutmix | 26.26% | +1.28% |
| 13 | gen_step1x_new | 26.18% | +1.20% |
| 14 | std_randaugment | 26.04% | +1.06% |
| 15 | gen_automold | 26.01% | +1.03% |
| 16 | **baseline** | **24.98%** | **—** |

### Per-Domain Performance (Top 5 Strategies)

| Strategy | clear_day | foggy | night | rainy | snowy |
|----------|-----------|-------|-------|-------|-------|
| gen_stargan_v2 | 32.1% | 28.1% | 12.4% | 26.4% | 26.6% |
| std_photometric_distort | 32.5% | 28.1% | 12.6% | 26.5% | 26.2% |
| gen_cycleGAN | 31.8% | 28.3% | 13.0% | 26.6% | 26.7% |
| std_autoaugment | 32.9% | 27.8% | 11.2% | 26.6% | 25.6% |
| gen_step1x_v1p2 | 32.5% | 27.8% | 12.3% | 26.1% | 25.3% |

### Key Findings

1. **ALL 15/15 strategies beat baseline** on domain adaptation (+1.03% to +1.96%)
2. **Best performer:** gen_stargan_v2 (26.94% mIoU, +1.96% over baseline)
3. **Best family:** Photometric (26.87% mean mIoU)
4. **Generative strategies average:** 26.47% mIoU (+1.49% over baseline)
5. **Standard augmentations average:** 26.42% mIoU (+1.44% over baseline)
6. **Most challenging domain:** night (12-13% mIoU across all strategies)
7. **Generative and standard provide similar benefits** for cross-dataset transfer

### Visualizations
- [result_figures/domain_adaptation/strategy_ranking.png](../result_figures/domain_adaptation/strategy_ranking.png)
- [result_figures/domain_adaptation/baseline_delta.png](../result_figures/domain_adaptation/baseline_delta.png)
- [result_figures/domain_adaptation/per_domain_heatmap.png](../result_figures/domain_adaptation/per_domain_heatmap.png)

### Analysis
```bash
python analysis_scripts/analyze_domain_adaptation_strategies.py
```

---

---

## Active Jobs & Next Actions

### Active Jobs (Jan 28, 2026 08:25)

**Baseline Extended Training:**
- 443834: ext_baseline_bdd10k_pspnet - **RUN**
- 443835: ext_baseline_bdd10k_segformer - **RUN**
- 443836: ext_baseline_iddaw_pspnet - **RUN**
- 443837: ext_baseline_iddaw_segformer - **RUN**

**Ratio Ablation Training:**
- 291472: gen_cyclediffusion_BDD10k_segformer_mit-b5_ratio0p00 - **RUN**
- 291559: gen_cyclediffusion_BDD10k_segformer_mit-b5_ratio0p25 - **RUN**
- 291560: gen_flux_kontext_BDD10k_segformer_mit-b5_ratio0p62 - **RUN**

### Pending Actions

1. **Extended Training Baseline:** Wait for training (~4-6 hours), then:
   ```bash
   python scripts/submit_baseline_extended_tests.py --submit
   python analysis_scripts/analyze_extended_training.py
   python analysis_scripts/visualize_extended_training.py
   ```

2. **Ratio Ablation:** Wait for training, then submit tests and update analysis

### Analysis Scripts
| Study | Script | Status |
|-------|--------|--------|
| Ratio Ablation | `analysis_scripts/analyze_ratio_ablation.py` | ✅ Complete |
| Extended Training | `analysis_scripts/analyze_extended_training.py` | ✅ Complete |
| Combinations | `analysis_scripts/analyze_combination_ablation.py` | ✅ Complete |
| Baseline | `analysis_scripts/analyze_baseline_consolidated.py` | ✅ Complete |
| Stage 1 | `analysis_scripts/generate_stage1_leaderboard.py` | ✅ Complete |
| Stage 2 | `analysis_scripts/generate_stage2_leaderboard.py` | ✅ Complete |
| Domain Adaptation | `analysis_scripts/analyze_domain_adaptation_strategies.py` | ✅ Complete |

---

## Study Completion Summary

| Study | Training | Testing | Analysis | Figures |
|-------|:--------:|:-------:|:--------:|:-------:|
| Stage 1 | ✅ | ✅ | ✅ | ✅ |
| Stage 2 | ✅ | ✅ | ✅ | ✅ |
| Ratio Ablation | ✅ | ✅ | ✅ | ✅ |
| Extended Training | ✅ | ✅ | ✅ | ✅ |
| Combinations | ✅ | ✅ | ✅ | ✅ |
| Domain Adaptation | N/A | ✅ | ✅ | ✅ |

**All major studies complete!**

### Extended Training Test Distribution

| Iteration | Tests | Iteration | Tests |
|-----------|-------|-----------|-------|
| 10k | 26 | 170k | 21 |
| 20k | 26 | 180k | 21 |
| 30k | 26 | 190k | 21 |
| 40k | 26 | 200k | 21 |
| 50k | 26 | 210k | 21 |
| 60k | 25 | 220k | 21 |
| 70k | 25 | 230k | 21 |
| 80k | 25 | 240k | 21 |
| 90k | 21 | 250k | 21 |
| 100k | 22 | 260k | 21 |
| 110k | 22 | 270k | 21 |
| 120k | 22 | 280k | 21 |
| 130k | 22 | 290k | 21 |
| 140k | 22 | 300k | 21 |
| 150k | 21 | 310k | 21 |
| 160k | 21 | 320k | 21 |

### Analysis Findings (Jan 24)

**Learning Curve:**
- Early phase (10k-30k): 37.7% → 42.0% mIoU (+4.3%)
- Mid phase (40k-80k): 42.6% → 45.0% mIoU (+2.4%)
- Extended (90k-320k): 48.6% → 49.8% mIoU (+1.2%)

**Key Metrics:**
- **77.4%** of configs improve with extended training
- **Mean improvement: +1.41 mIoU** from 80k to optimal
- **160k captures ~75%** of extended gains at 50% compute
- Best strategy: gen_cyclediffusion (53.8% at 320k)

### Top 5 Stage 1 Strategies Ablation Coverage

| Strategy | mIoU | Ratio Ablation | Extended Training |
|----------|------|----------------|-------------------|
| **gen_Attribute_Hallucination** | 39.83% | ✅ 28 configs | ❌ Not covered |
| **gen_cycleGAN** | 39.60% | ✅ 42 configs | ✅ 96 ckpts |
| **gen_Img2Img** | 39.58% | ✅ 9 configs | ❌ Not covered |
| **gen_stargan_v2** | 39.55% | ✅ 28 configs | ❌ Not covered |
| **gen_flux_kontext** | 39.54% | ❌ Not covered | ✅ 96 ckpts |

**Update (Jan 27):** 4 of 5 top strategies now have ratio ablation coverage.

### MapillaryVistas BGR/RGB Bug Status

The BGR/RGB bug in `custom_transforms.py` affected all MapillaryVistas training. All buggy checkpoints have been handled:

| Study | MV Status | Notes |
|-------|-----------|-------|
| **Stage 1** | ✅ **COMPLETE** | All 81/81 retrained + tested |
| **Stage 2** | ✅ **COMPLETE** | All 81/81 retrained + tested (by user chge7185) |
| **Ratio Ablation** | 📦 Backed up | `WEIGHTS_BACKUP_BUGGY_MAPILLARY/ratio_ablation/` |
| **Extended Training** | 📦 Backed up | `WEIGHTS_BACKUP_BUGGY_MAPILLARY/extended_training/` |
| **Combinations** | 📦 Backed up | `WEIGHTS_BACKUP_BUGGY_MAPILLARY/combinations/` |

**Note:** Backed up checkpoints are INVALID and cannot be used.

---

## Stage 1: Clear Day Training

**Path:** `${AWARE_DATA_ROOT}/WEIGHTS/`
**Status:** ✅ **COMPLETE** (Training 405/405 + Testing 405/405)

### Baseline Analysis Results

Publication-ready analysis available at `result_figures/baseline_consolidated/stage1_baseline_output/`:

| Metric | Value |
|--------|-------|
| Overall mIoU | 33.3% |
| Domain Gap | 10.1% |
| Most Robust Model | SegFormer (8.7% gap) |
| Hardest Domain | Night (-14.9% from Clear Day) |
| Largest Dataset Gap | IDD-AW (17.6%) |
| Smallest Dataset Gap | Mapillary (2.6%) |

### Coverage Matrix

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Total |
|----------|:------:|:------:|:---------------:|:----------:|------:|
| baseline | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_Attribute_Hallucination | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_CNetSeg | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_CUT | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_IP2P | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_Img2Img | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_LANIT | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_Qwen_Image_Edit | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_SUSTechGAN | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_TSIT | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_UniControl | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_VisualCloze | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_Weather_Effect_Generator | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_albumentations_weather | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_augmenters | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_automold | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_cycleGAN | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_cyclediffusion | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_flux_kontext | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_stargan_v2 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_step1x_new | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| gen_step1x_v1p2 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| std_photometric_distort | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| std_autoaugment | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| std_cutmix | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| std_mixup | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |
| std_randaugment | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 |

**Total:** 405 checkpoints (27 strategies × 4 datasets × 3 models) ✅

---

## Stage 2: All Domains Training

**Path:** `${AWARE_DATA_ROOT}/WEIGHTS_STAGE_2/`
**Status:** ✅ **COMPLETE** (Training 325/325 + Testing 324/324)

### Leaderboard (Top 10)
| Rank | Strategy | mIoU | Gain vs Baseline |
|------|----------|------|------------------|
| 1 | gen_stargan_v2 | 41.73% | +0.38 |
| 2 | gen_CNetSeg | 41.61% | +0.26 |
| 3 | gen_Weather_Effect_Generator | 41.60% | +0.25 |
| 4 | gen_automold | 41.59% | +0.24 |
| 5 | gen_Attribute_Hallucination | 41.56% | +0.21 |
| 6 | std_autoaugment | 41.52% | +0.17 |
| 7 | gen_cyclediffusion | 41.52% | +0.17 |
| 8 | gen_Img2Img | 41.51% | +0.16 |
| 9 | gen_augmenters | 41.50% | +0.15 |
| 10 | gen_cycleGAN | 41.48% | +0.13 |

### Coverage Matrix - ALL COMPLETE ✅

| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Total |
|----------|:------:|:------:|:---------------:|:----------:|------:|
| All 27 strategies | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | ✅ 3/3 | 12 each |

**Total:** 324/324 tests complete (27 strategies × 4 datasets × 3 models)

---

## Ratio Ablation Study

**Path:** `${AWARE_DATA_ROOT}/WEIGHTS_RATIO_ABLATION/`
**Owner:** ${USER}
**Status:** 🔄 Testing in progress (141 jobs submitted 2026-01-24)

### Purpose
Test the impact of different real/generated image mixing ratios on model performance.

### Ratios Tested
0.00, 0.125, 0.25, 0.375, 0.50*, 0.625, 0.75, 0.875, 1.00

*Note: Ratio 0.50 is the standard training in `WEIGHTS/`

### Training & Testing Status

| Strategy | Trained | Tested | Test Jobs | Notes |
|----------|---------|--------|-----------|-------|
| **gen_cycleGAN** | ✅ 28/28 | ✅ 28/28 | N/A | **COMPLETE** |
| gen_cyclediffusion | ✅ 10/11 | ✅ 9 | +1 | 1 incomplete training |
| gen_stargan_v2 | ⚠️ 9/12 | ✅ 9 | +3 | 3 incomplete training |
| gen_step1x_new | ✅ 56/56 | ⏳ 0/56 | 56 | Testing queued |
| gen_step1x_v1p2 | ✅ 46/46 | ⏳ 0/46 | 46 | Testing queued |
| gen_TSIT | ⚠️ 42/45 | ⏳ 0/42 | 42 | 3 incomplete training |

**Total:** 191 checkpoints (141 need testing)

### Notes
- MapillaryVistas checkpoints backed up to `WEIGHTS_BACKUP_BUGGY_MAPILLARY/ratio_ablation/`
- Analysis script: `analysis_scripts/analyze_ratio_ablation.py`
- Visualization: `analysis_scripts/visualize_ratio_ablation.py`
- Test submission script: `scripts/submit_ablation_tests.py`

---

## Extended Training Study

**Path:** `${AWARE_DATA_ROOT}/WEIGHTS_EXTENDED/`
**Owner:** chge7185
**Status:** 🔄 Testing in progress (39 jobs submitted 2026-01-24 for 320k iteration)

### Purpose
Evaluate training convergence and performance at extended iteration milestones.

### Iterations Tested
90k, 100k, 110k, 120k, 130k, 140k, 150k, 160k, 170k, 180k, 190k, 200k, 210k, 220k, 230k, 240k, 250k, 260k, 270k, 280k, 290k, 300k, 310k, 320k

(24 checkpoints per configuration)

### Training & Testing Status

| Strategy | Trained Configs | 320k Tests | All Tests | Notes |
|----------|----------------|------------|-----------|-------|
| gen_cyclediffusion | ✅ 8/8 | ⏳ 8 | ⏳ 192 | Complete training |
| gen_cycleGAN | ⚠️ 4/6 | ⏳ 4 | ⏳ 96 | 2 OUTSIDE15k missing |
| gen_flux_kontext | ⚠️ 4/6 | ⏳ 4 | ⏳ 96 | 2 MV missing |
| gen_step1x_new | ⚠️ 5/7 | ⏳ 5 | ⏳ 120 | 2 configs missing |
| gen_TSIT | ⚠️ 4/6 | ⏳ 4 | ⏳ 96 | 2 OUTSIDE15k missing |
| gen_UniControl | ⚠️ 4/6 | ⏳ 4 | ⏳ 96 | 2 OUTSIDE15k missing |
| gen_albumentations_weather | ⚠️ 4/6 | ⏳ 4 | ⏳ 96 | 2 OUTSIDE15k missing |
| gen_automold | ⚠️ 4/8 | ⏳ 4 | ⏳ 95 | 4 configs missing |
| std_randaugment | ⚠️ 3/8 | ⏳ 3 | ⏳ 72 | 5 configs missing |

**Total:** 39 complete configs × 24 iterations = 936 checkpoints
**Testing:** Currently testing 320k iteration only (39 jobs)

### Key Findings (from earlier analysis)
- **160k iterations** captures ~75% of gains at 50% compute cost
- Performance plateaus vary by strategy

### Notes
- MapillaryVistas directories backed up (logs/configs only, no checkpoints saved)
- Analysis: `analysis_scripts/analyze_extended_training.py`
- Report: [docs/EXTENDED_TRAINING_ANALYSIS.md](EXTENDED_TRAINING_ANALYSIS.md)
- Test submission script: `scripts/submit_extended_tests.py`

---

## Combination Strategies Study

**Path:** `${AWARE_DATA_ROOT}/WEIGHTS_COMBINATIONS/`
**Owner:** chge7185
**Status:** ✅ Valid (IDD-AW only) | 📦 MV backed up (54 checkpoints)

### Purpose
Test combining generative augmentation with standard augmentation techniques.

### Coverage Summary

| Category | Combinations | Checkpoints |
|----------|-------------|-------------|
| gen_flux_kontext + std_* | 5 | 10 |
| gen_Qwen_Image_Edit + std_* | 5 | 10 |
| gen_step1x_new + std_* | 5 | 9 |
| gen_stargan_v2 + photometric | 1 | 2 |
| gen_Attribute_Hallucination + photometric | 1 | 2 |
| std_* + std_* | 10 | 20 |
| **Total** | **27** | **53** |

### Notes
- All valid checkpoints are IDD-AW only
- Models: pspnet_r50, segformer_mit-b5
- MapillaryVistas backed up to `WEIGHTS_BACKUP_BUGGY_MAPILLARY/combinations/`
- Missing: BDD10k, OUTSIDE15k coverage

---

## Domain Adaptation Ablation - Details

**Path:** `${AWARE_DATA_ROOT}/WEIGHTS/domain_adaptation_ablation/`
**Status:** ✅ **COMPLETE** (55 configurations)

### Purpose
Evaluate **cross-dataset domain generalization** using Stage 1 models tested on ACDC.

### Coverage by Strategy

| Strategy | Configs | Datasets | Status |
|----------|---------|----------|--------|
| bdd10k (baseline) | 2 | BDD10k | ✅ |
| std_autoaugment | 2 | BDD10k, IDD-AW | ✅ |
| std_mixup | 2 | BDD10k, IDD-AW | ✅ |
| std_cutmix | 2 | BDD10k, IDD-AW | ✅ |
| std_randaugment | 3 | BDD10k, IDD-AW | ✅ |
| gen_step1x_new | 3 | BDD10k, IDD-AW | ✅ |
| gen_stargan_v2 | 4 | BDD10k, IDD-AW | ✅ |
| std_photometric_distort | 4 | BDD10k, IDD-AW | ✅ |
| gen_cycleGAN | 4 | BDD10k, IDD-AW | ✅ |
| gen_step1x_v1p2 | 4 | BDD10k, IDD-AW | ✅ |
| gen_cyclediffusion | 4 | BDD10k, IDD-AW | ✅ |
| gen_UniControl | 4 | BDD10k, IDD-AW | ✅ |
| gen_albumentations_weather | 4 | BDD10k, IDD-AW | ✅ |
| gen_flux_kontext | 4 | BDD10k, IDD-AW | ✅ |
| gen_TSIT | 4 | BDD10k, IDD-AW | ✅ |
| gen_automold | 4 | BDD10k, IDD-AW | ✅ |
| gen_Attribute_Hallucination | 1 | BDD10k | ✅ |
| **Total** | **55** | | ✅ |

### Analysis
```bash
# Run domain adaptation analysis
python analysis_scripts/analyze_domain_adaptation_ablation.py
```

**Note:** The existing analysis script expects a different directory structure. The actual results are stored at:
`${AWARE_DATA_ROOT}/WEIGHTS/domain_adaptation_ablation/{strategy}/{dataset}/{model}/domain_adaptation_evaluation.json`

---

## Comprehensive Findings Analysis

### Stage 1: Clear-Day Training (Cross-Domain Robustness)

**Summary:** Generative augmentation strategies consistently outperform both baseline and standard augmentation when training only on clear-day conditions. The top-ranked strategies achieve +1.0–1.4 mIoU gain over baseline.

**Key Findings:**
- **Best performer:** gen_Attribute_Hallucination (39.83 mIoU, +1.36 vs baseline 38.47)
- **Top 5 all generative:** gen_cycleGAN (39.60), gen_Img2Img (39.58), gen_stargan_v2 (39.55), gen_flux_kontext (39.54)
- **Best standard aug:** std_autoaugment (39.41 mIoU, +0.94) outperforms mid-tier generative methods
- **Domain gap range:** 5.84 (std_autoaugment) to 7.03 (gen_LANIT) - lower is better
- **std_autoaugment achieves smallest domain gap** while maintaining good mIoU

**Per-Dataset Insights:**
- BDD10k: Largest gains from generative aug (+1.4-2.4 mIoU over baseline)
- IDD-AW: Consistent +0.7-1.5 mIoU improvement across strategies
- MapillaryVistas: Smaller gains (+0.2-0.8 mIoU), some strategies underperform baseline
- OUTSIDE15k: Mixed results, some strategies show negative gain

---

### Stage 2: All-Domains Training (Domain-Inclusive Performance)

**Summary:** When training includes all weather conditions, the performance gap narrows significantly. Baseline improves +2.88 mIoU and domain gap shrinks 5-6×. Strategy gains compress to +0.3-0.4 mIoU.

**Key Findings:**
- **Best performer:** gen_stargan_v2 (41.73 mIoU, +0.38 vs baseline 41.35)
- **Baseline improvement:** 38.47 → 41.35 mIoU (+2.88 from including adverse domains in training)
- **Domain gap reduction:** 6.81 → 0.66 (10× smaller gap when training on all conditions)
- **Rank changes:** gen_Attribute_Hallucination drops from #1 to #6; gen_stargan_v2 rises to #1
- **Some strategies underperform:** gen_Weather_Effect_Generator (-0.64), std_cutmix (-0.54) worse than baseline

**Insight:** Explicit weather augmentation becomes partially redundant when training data already contains adverse conditions. The value proposition of generative augmentation is highest in domain-shift scenarios (Stage 1).

---

### Ratio Ablation Study (Real/Synthetic Data Mix)

**Summary:** Performance peaks at moderate synthetic ratios (12–38%), not at 50% or higher. Optimal ratio varies by configuration but moderate mixing consistently beats extremes.

**Key Findings:**
- **Optimal ratio:** 0.75 (75% real + 25% synthetic) achieves +1.56 mIoU over pure real baseline
- **Sweet spot range:** 12-38% synthetic data
- **Diminishing returns:** Performance drops slightly at 50%+ synthetic
- **100% synthetic (ratio 0.00):** Still outperforms baseline (+1.35 mIoU) but suboptimal
- **All ratios beat baseline:** Every synthetic data ratio improves over pure real data training

**Ratio Performance Table:**
| Ratio | Description | Avg mIoU | Δ vs Baseline |
|-------|-------------|----------|---------------|
| 0.75 | 75% real, 25% gen | 41.46% | **+1.56%** |
| 0.50 | 50% real, 50% gen | 41.39% | +1.49% |
| 0.38 | 38% real, 62% gen | 41.36% | +1.46% |
| 0.00 | 0% real, 100% gen | 41.25% | +1.35% |
| 1.00 | 100% real (baseline) | 39.90% | — |

---

### Extended Training Study (80K → 320K Iterations)

**Summary:** Extended training benefits the majority of configurations (77%) with average +1.4 mIoU improvement. Optimal performance typically occurs at 310K–320K iterations, but 75% of gains are captured by 160K.

**Key Findings:**
- **Improvement rate:** 77.4% of configs improve with extended training
- **Mean improvement:** +1.41 mIoU from 80K to optimal iteration
- **Maximum improvement:** +4.10 mIoU (specific configs)
- **Total improvement:** +12.09 mIoU from 10K to 320K (37.7% → 49.79%)
- **Best strategy at 320K:** gen_cyclediffusion (53.81% mIoU)

**⚠️ Critical Finding - Baseline Overfitting:**
- Baseline performance **degrades after 90K** iterations (46.11 → 43.47 mIoU)
- Generative strategies do NOT show this degradation pattern
- **Implication:** Synthetic data provides implicit regularization preventing overfitting

**Efficiency Analysis:**
- 160K iterations captures **~75% of gains** at 50% compute cost
- Early phase (10K-30K): Rapid improvement (+4.3%)
- Mid phase (40K-80K): Steady gains (+2.4%)
- Extended (90K-320K): Diminishing returns (+1.2%)

---

### Combination Strategies Study (Strategy Stacking)

**Summary:** Combinations with std_photometric_distort consistently dominate (~45 mIoU). Strategy stacking does not provide additive benefits—combined performance rarely exceeds best individual component.

**Key Findings:**
- **Best combination:** std_mixup+std_photometric_distort (45.22% mIoU)
- **All +std_photometric_distort combos:** 44.9-45.2% mIoU (regardless of partner strategy)
- **Generative+Standard avg:** 40.1% mIoU (no synergy observed)
- **Standard+Standard avg:** 39.7% mIoU

**Top Combinations:**
| Combination | mIoU |
|-------------|------|
| std_mixup+std_photometric_distort | 45.22% |
| std_autoaugment+std_photometric_distort | 45.18% |
| gen_step1x_new+std_photometric_distort | 45.18% |
| gen_Attribute_Hallucination+std_photometric_distort | 45.17% |
| gen_stargan_v2+std_photometric_distort | 45.17% |

**Insight:** The dominant factor is std_photometric_distort, not the generative component. Combining multiple augmentation strategies shows diminishing returns rather than synergistic improvement.

---

### Domain Adaptation Study (Cross-Dataset Transfer)

**Summary:** All 15 tested strategies beat baseline for cross-dataset domain adaptation (+1.03% to +1.96%). Generative and standard augmentations provide similar benefits for transfer learning.

**Key Findings:**
- **All strategies beat baseline:** 15/15 strategies outperform baseline on ACDC transfer
- **Best performer:** gen_stargan_v2 (26.94% mIoU, +1.96% over baseline 24.98%)
- **Best family:** Photometric augmentation (26.87% mean mIoU)
- **Generative avg:** 26.47% mIoU (+1.49% over baseline)
- **Standard aug avg:** 26.42% mIoU (+1.44% over baseline)
- **Most challenging domain:** Night (12-13% mIoU across all strategies)

**Insight:** For cross-dataset transfer to unseen domains (ACDC), both generative and standard augmentation provide similar robustness benefits. The choice between them can be based on compute/data availability rather than performance.

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ 3/3 | All 3 models complete (deeplabv3plus_r50, pspnet_r50, segformer_mit-b5) |
| 🔶 2/3 | 2 of 3 models complete |
| 🔶 1/3 | 1 of 3 models complete |
| 🔄 | Retraining in progress |
| ⏳ | Pending/queued |
| ⏸️ | Never trained (lower priority, generated data available) |
| ❌ | Not available / no data |
| 📦 | Backed up (buggy checkpoints, INVALID, awaiting retrain) |
| ⚠️ INVALID | Trained with bug, cannot use results |
