# PROVE Results Reference Document

**Purpose:** Comprehensive reference for all PROVE evaluation results, data formats, and locations. Intended for setting up a separate analysis/visualization repository.

**Last Updated:** 2026-02-12

---

## 1. Project Overview

PROVE (PRObabilistic Visual Evaluation) evaluates the impact of generative data augmentation on semantic segmentation model robustness under adverse weather conditions. The project trains segmentation models on clear-day images, then tests cross-domain robustness on adverse weather (rain, snow, fog, night, dawn/dusk, cloudy).

### Core Question
> Do generative augmentation strategies improve semantic segmentation robustness under domain shift compared to standard augmentation and baseline training?

### Experimental Stages

| Stage | Training Data | Weights Directory | Purpose | Status |
|-------|--------------|-------------------|---------|--------|
| **Stage 1 (S1)** | Clear-day only | `WEIGHTS/` | Cross-domain robustness evaluation | 89.1% complete |
| **Stage 2 (S2)** | All domains | `WEIGHTS_STAGE_2/` | Domain-inclusive training evaluation | 35.5% complete |
| **Cityscapes Replication (CS)** | Cityscapes full | `WEIGHTS_CITYSCAPES/` | Pipeline verification against reference | ✅ Complete |
| **Cityscapes-Gen (CG)** | Cityscapes + gen | `WEIGHTS_CITYSCAPES_GEN/` | Strategy comparison on standard benchmark | ✅ 100% complete |

### External Root Path
All data lives under: `${AWARE_DATA_ROOT}/`

---

## 2. Datasets

### 2.1 Source Datasets

Data from 6 source datasets, unified into `FINAL_SPLITS/` with 70/30 train/test split.

| Dataset | Total Images | Train | Test | Domains | Classes | Label Format |
|---------|-------------|-------|------|---------|---------|--------------|
| ACDC | 3,595 | 2,483 | 1,112 | 7 | 19 | Cityscapes trainIds |
| BDD100k | 60,185 | 42,130 | 18,055 | 7 | 19 | Cityscapes trainIds |
| BDD10k | 5,857 | 4,100 | 1,757 | 7 | 19 | Cityscapes trainIds |
| IDD-AW | 9,531 | 6,672 | 2,859 | 7 | 19 | Cityscapes trainIds |
| MapillaryVistas | 16,263 | 11,384 | 4,879 | 7 | 66 | RGB-encoded |
| OUTSIDE15k | 7,772 | 5,540 | 2,232 | 7 | 24 | RGB-encoded |
| **Total** | **103,203** | **71,671** | **31,532** | — | — | — |

**File:** `${AWARE_DATA_ROOT}/FINAL_SPLITS/split_statistics.json`

### 2.2 Weather Domains

7 weather/lighting conditions across all datasets:

| Domain | Total Images | Train | Test |
|--------|-------------|-------|------|
| clear_day | 45,574 | 31,900 | 13,674 |
| night | 25,601 | 17,793 | 7,808 |
| cloudy | 11,973 | 8,379 | 3,594 |
| rainy | 8,693 | 5,944 | 2,749 |
| snowy | 7,091 | 4,869 | 2,222 |
| dawn_dusk | 2,756 | 1,818 | 938 |
| foggy | 1,515 | 968 | 547 |

### 2.3 Cityscapes (Separate Pipeline)

- Standard Cityscapes dataset for pipeline verification and CG experiments
- 2,975 train / 500 test images
- 19 classes, native trainIds
- **File:** `${AWARE_DATA_ROOT}/CITYSCAPES/`
- **ACDC cross-domain test:** 4 domains (fog, night, rain, snow)

---

## 3. Augmentation Strategies

### 3.1 Generative Strategies (21 total)

Each strategy generates synthetic adverse-weather images from clear-day source images.

| Strategy | Method Type | Num Generated | Match Rate |
|----------|-----------|---------------|------------|
| gen_Attribute_Hallucination | Attribute editing | 191,400 | 100% |
| gen_CNetSeg | ControlNet conditioning | 205,248 | 100% |
| gen_CUT | Contrastive unpaired translation | 209,250 | varies |
| gen_cycleGAN | Cycle-consistent GAN | 187,398 | 100% |
| gen_cyclediffusion | Diffusion-based translation | 110,883 | varies |
| gen_flux_kontext | Flux-based context editing | 69,900 | varies |
| gen_Img2Img | Image-to-image diffusion | 124,932 | varies |
| gen_IP2P | InstructPix2Pix | 187,398 | 100% |
| gen_LANIT | Language-guided translation | 223,300 | varies |
| gen_Qwen_Image_Edit | Qwen VL editing | 41,718 | varies |
| gen_stargan_v2 | StarGAN v2 multi-domain | varies | varies |
| gen_step1x_new | Step1X new version | 77,343 | varies |
| gen_step1x_v1p2 | Step1X v1.2 | 119,050 | varies |
| gen_SUSTechGAN | SUSTech weather GAN | 127,699 | 100% |
| gen_TSIT | TSIT translation | varies | varies |
| gen_UniControl | Universal ControlNet | 187,398 | 100% |
| gen_VisualCloze | Visual cloze completion | 65,006 | varies |
| gen_Weather_Effect_Generator | Weather effects | varies | varies |
| gen_albumentations_weather | Albumentations weather | 95,700 | varies |
| gen_augmenters | Imgaug augmenters | varies | varies |
| gen_automold | Automold weather sim | 95,700 | varies |

**Manifest files:** `PROVE/generated_manifests/{method}_manifest.json`
**All manifests summary:** `${AWARE_DATA_ROOT}/GENERATED_IMAGES/all_manifests_summary.json`

### 3.2 Standard Augmentation Strategies (5 total)

| Strategy | Description |
|----------|-------------|
| std_autoaugment | AutoAugment policy |
| std_cutmix | CutMix augmentation |
| std_mixup | MixUp augmentation |
| std_randaugment | RandAugment policy |
| std_photometric_distort | Photometric distortion (excluded from CG) |

### 3.3 Strategy Exclusions by Stage

| Stage | Excluded Strategies | Reason |
|-------|-------------------|--------|
| S1 | None | All 26 strategies (21 gen + 4 std + 1 baseline) |
| S2 | std_cutmix, std_mixup, gen_cyclediffusion | Deliberate exclusion (25 strategies) |
| CG | gen_LANIT, std_minimal, std_photometric_distort | gen_LANIT has no Cityscapes images; others deliberate (25 strategies) |

---

## 4. Models

### 4.1 Architecture Summary

| Model | Architecture | Backbone | Parameters |
|-------|-------------|----------|-----------|
| deeplabv3plus_r50 | DeepLabV3+ | ResNet-50 | ~43M |
| pspnet_r50 | PSPNet | ResNet-50 | ~49M |
| segformer_mit-b3 | SegFormer | MiT-B3 | ~47M |
| segnext_mscan-b | SegNeXt | MSCAN-B | ~28M |
| hrnet_hr48 | HRNet | HRNet-W48 | ~66M |
| mask2former_swin-b | Mask2Former | Swin-B | ~107M |

### 4.2 Model Usage by Stage

| Stage | Models Used | Count | Notes |
|-------|------------|-------|-------|
| S1 | pspnet, segformer, segnext, mask2former | 4 | hrnet excluded from leaderboard (legacy) |
| S2 | pspnet, segformer, segnext, mask2former | 4 | Same 4 core models |
| CS | All 5 | 5 | Pipeline verification |
| CG | pspnet, segformer, segnext, mask2former (+ bonus deeplabv3plus) | 4+1 | Most strategies also trained deeplabv3plus |

---

## 5. Result Data Files

### 5.1 Downstream Results CSVs (Main Aggregated Results)

These are the primary analysis input files, auto-generated by `generate_strategy_leaderboard.py`.

| File | Stage | Location |
|------|-------|----------|
| `downstream_results.csv` | S1 | `PROVE/downstream_results.csv` |
| `downstream_results_stage2.csv` | S2 | `PROVE/downstream_results_stage2.csv` |
| `downstream_results_cityscapes_gen.csv` | CG | `PROVE/downstream_results_cityscapes_gen.csv` |

**CSV Schema:**
```
strategy, dataset, model, test_type, result_type, result_dir, timestamp,
mIoU, mAcc, aAcc, fwIoU, num_images,
has_per_domain, has_per_class, per_domain_metrics, per_class_metrics
```

- `per_domain_metrics`: JSON string with per-domain mIoU/mAcc/aAcc/fwIoU
- `per_class_metrics`: JSON string with per-class IoU/Acc (19 classes for most datasets)
- `test_type`: `test_results_detailed` (standard), `test_results_acdc` (cross-domain)

### 5.2 Test Result JSONs (Per-Model Raw Results)

Each completed test produces a `results.json`:

**Location pattern:**
```
WEIGHTS/{strategy}/{dataset}/{model}/test_results_detailed/{timestamp}/results.json
```

**JSON Schema:**
```json
{
  "config": "path/to/training_config.py",
  "overall": {
    "aAcc": 91.15, "mIoU": 46.25, "mAcc": 53.61, "fwIoU": 83.87,
    "num_images": 1857
  },
  "per_domain": {
    "clear_day": {"mIoU": 47.32, "mAcc": 54.64, "aAcc": 91.62, "fwIoU": 85.15},
    "cloudy": {"mIoU": 49.61, ...},
    "dawn_dusk": {"mIoU": 39.83, ...},
    "foggy": {"mIoU": 46.81, ...},
    "night": {"mIoU": 24.61, ...},
    "rainy": {"mIoU": 46.24, ...},
    "snowy": {"mIoU": 50.10, ...}
  },
  "per_class": {
    "road": {"IoU": 91.71, "Acc": 96.85, "area_intersect": ..., "area_union": ...},
    "sidewalk": {"IoU": 46.10, ...},
    ...  // 19 Cityscapes classes
  }
}
```

### 5.3 Leaderboard Files

Auto-generated by `analysis_scripts/generate_strategy_leaderboard.py`:

| File | Content |
|------|---------|
| `result_figures/leaderboard/STRATEGY_LEADERBOARD_STAGE1_MIOU.md` | S1 ranked table |
| `result_figures/leaderboard/STRATEGY_LEADERBOARD_STAGE2_MIOU.md` | S2 ranked table |
| `result_figures/leaderboard/STRATEGY_LEADERBOARD_CITYSCAPES_GEN_MIOU.md` | CG ranked table |

**Breakdown CSVs** (in `result_figures/leaderboard/breakdowns/`):

| File | Content |
|------|---------|
| `strategy_leaderboard_{stage}.csv` | Strategy × mIoU/Std/Gain/NormalMIoU/AdverseMIoU/DomainGap |
| `per_dataset_breakdown_{stage}.csv` | Strategy × Dataset mIoU matrix |
| `per_domain_breakdown_{stage}.csv` | Strategy × Domain mIoU matrix |
| `per_model_breakdown_{stage}.csv` | Strategy × Model mIoU matrix |
| `DETAILED_GAINS_{stage}_MIOU.md` | Markdown tables with gain analysis |

---

## 6. Generative Quality Evaluation

### 6.1 Composite Quality Score (CQS)

**File:** `PROVE/results/generative_quality/generative_quality.csv`

**CSV Schema:**
```
Rank, Method, CQS, FID, LPIPS, SSIM, PSNR, Pixel_Accuracy, mIoU, fw_IoU, Num_Images
```

**Metrics:**
- **CQS** (Composite Quality Score): Combined metric ranking generative methods
- **FID**: Fréchet Inception Distance (lower = more realistic)
- **LPIPS**: Learned Perceptual Image Patch Similarity (lower = more similar)
- **SSIM**: Structural Similarity Index (higher = more similar)
- **PSNR**: Peak Signal-to-Noise Ratio (higher = less distortion)
- **Pixel_Accuracy / mIoU / fw_IoU**: Semantic consistency metrics (segmentation quality on generated images)

**Key Rankings (CQS, lower = higher quality):**
1. cycleGAN (CQS: -0.78, best semantic consistency)
2. flux_kontext (CQS: -0.66)
3. step1x_new (CQS: -0.47)
4. LANIT (CQS: -0.29)
...
20. CNetSeg (CQS: 1.43, worst quality)

### 6.2 Per-Strategy Per-Domain Stats

**Location:** `${AWARE_DATA_ROOT}/STATS/{method}/`

Each strategy has per-domain JSON files:
- `{domain}_stats.json` — Aggregate statistics
- `{domain}_per_image.json` — Per-image metrics

**Stats JSON Schema:**
```json
{
  "domain": "cloudy",
  "generated": "/path/to/generated/images",
  "original": "/path/to/original/images",
  "num_pairs": 31900,
  "metrics": {
    "ssim": {"count": ..., "mean": 0.379, "std": 0.099, "median": ..., "min": ..., "max": ...},
    "lpips": {"count": ..., "mean": ..., ...},
    "semantic_pixel_accuracy": {...},
    "semantic_mIoU": {...},
    "semantic_fw_IoU": {...},
    "fid": ...
  },
  "semantic_consistency": {
    "enabled": true,
    "model_variant": "...",
    "model_name": "...",
    "summary": {...},
    "per_image_details": [...]
  }
}
```

### 6.3 FID Reference Statistics

**Location:** `${AWARE_DATA_ROOT}/STATS/`
- `clear_day_fid.npz`, `cloudy_fid.npz`, `night_fid.npz`, etc.
- Pre-computed Inception features for FID calculation against real domain images

---

## 7. Ablation Studies

### 7.1 Ratio Ablation (Real/Generated Mix Ratio)

| Property | Value |
|----------|-------|
| **Location** | `${AWARE_DATA_ROOT}/WEIGHTS_RATIO_ABLATION/` |
| **Ratios** | 0.00, 0.12, 0.25, 0.38, 0.50, 0.62, 0.75, 0.88 |
| **Models** | pspnet_r50, segformer_mit-b3 |
| **Datasets** | BDD10k, IDD-AW |
| **Analysis** | `analysis_scripts/analyze_ratio_ablation.py` |
| **Results** | `results/ratio_ablation_full.csv`, `results/ratio_ablation_consolidated.csv` |
| **Figures** | `result_figures/ratio_ablation/` |

**Ratio CSV Schema:**
```
strategy, dataset, model, ratio, mIoU, mAcc, aAcc, fwIoU, timestamp, checkpoint_iter, stage
```

### 7.2 Cityscapes-Ratio Ablation

| Property | Value |
|----------|-------|
| **Location** | `${AWARE_DATA_ROOT}/WEIGHTS_CITYSCAPES_RATIO/` |
| **Strategies** | gen_flux_kontext, gen_step1x_v1p2, gen_TSIT, gen_VisualCloze |
| **Ratios** | 0.00, 0.12, 0.25, 0.38, 0.50, 0.62, 0.75, 0.88 |
| **Status** | 37/60 trained, all tested on CS + ACDC |

### 7.3 Extended Training Study

| Property | Value |
|----------|-------|
| **Location** | `${AWARE_DATA_ROOT}/WEIGHTS_EXTENDED/` |
| **Iterations** | 40k, 60k, 80k, 100k, 120k, 140k, 160k, 320k |
| **Analysis** | `analysis_scripts/analyze_extended_training.py` |
| **Results** | `results/extended_training_analysis.csv` |
| **Figures** | `result_figures/extended_training/` |
| **Key Finding** | 160k iterations captures 75% of max gains at 50% compute cost |

**Extended Training CSV Schema:**
```
strategy, dataset, model, iteration, mIoU, pixel_acc
```

### 7.4 Batch Size Ablation

| Property | Value |
|----------|-------|
| **Location** | `${AWARE_DATA_ROOT}/WEIGHTS_BATCH_SIZE_ABLATION/` |
| **Batch Sizes** | 2, 4, 8, 16 |
| **LR Scaling** | Linear (BS=2→LR=0.01, BS=16→LR=0.08) |

### 7.5 Loss Function Ablation

| Property | Value |
|----------|-------|
| **Location** | `${AWARE_DATA_ROOT}/WEIGHTS_LOSS_ABLATION/` |
| **Variants** | aux-lovasz, aux-boundary, aux-focal, loss-lovasz |
| **Stages** | 1 and 2 |

### 7.6 Combination Ablation (Strategy Combinations)

| Property | Value |
|----------|-------|
| **Results** | `result_figures/combination_ablation/` |
| **Analysis** | `analysis_scripts/analyze_combination_ablation.py` |

---

## 8. Generated Figures Inventory

### 8.1 Leaderboard (`result_figures/leaderboard/`)
- Strategy leaderboard tables (MD + CSV) for S1, S2, CG

### 8.2 Baseline Analysis (`result_figures/baseline_consolidated/`)
- `fig8_domain_shift.png` — Domain shift visualization
- `fig9_stage_comparison_radar.png` — Stage 1 vs 2 radar chart
- `fig10_segformer_per_dataset_radar.png` — Per-dataset radar chart
- Per-domain CSVs (`clear_day_baseline_per_domain.csv`, etc.)
- IEEE publication figures in `ieee_figures/`, `publication_output/`

### 8.3 Ratio Ablation (`result_figures/ratio_ablation/`)
- `miou_vs_ratio_by_strategy.png` — mIoU curves by strategy
- `miou_vs_ratio_by_dataset.png` — mIoU curves by dataset
- `miou_vs_ratio_by_model.png` — mIoU curves by model
- `heatmap_strategy_ratio.png` — Strategy × ratio heatmap
- `optimal_ratio_distribution.png` — Optimal ratio histogram
- IEEE figures in `ieee_figures/`

### 8.4 Extended Training (`result_figures/extended_training/`)
- `convergence_analysis.png` — Iteration convergence curves
- `diminishing_returns.png` — Compute efficiency analysis
- `improvement_by_strategy.png` — Strategy improvement over iterations
- `baseline_comparison.png` — Extended vs baseline comparison
- Learning curves by dataset/model/strategy

### 8.5 Stage Comparison (`result_figures/stage_comparison/`)
- `fig1_overall_comparison.png` — S1 vs S2 overall
- `fig2_rank_change.png` — Strategy rank changes between stages
- `fig3_per_dataset_heatmap.png` — Per-dataset heatmap
- `fig4_gain_comparison.png` — Gain comparison
- `fig5_scatter_stages.png` — S1 vs S2 scatter
- `fig6_improvement_by_type.png` — Gen vs std improvement

### 8.6 Domain Analysis (`result_figures/domain_adaptation/`)
- `domain_gap_vs_miou_gains.png` — Domain gap correlation
- `baseline_delta.png` — Baseline deltas
- Domain adaptation CSVs and reports

### 8.7 Weather Analysis (`result_figures/weather_analysis_stage1/`)
- Per-domain heatmaps, radar plots, comparison charts
- `top_strategies_per_domain.csv`
- `weather_domain_results.csv`

### 8.8 Unified Domain Gap (`result_figures/unified_domain_gap/`)
- `all_domain_results.csv` — Complete domain-level results
- Normal vs adverse scatter plots
- Strategy ranking charts

### 8.9 t-SNE Visualizations (`result_figures/tsne/`)
- Feature space visualizations for select strategies

### 8.10 Crop Size Analysis (`result_figures/crop_size_analysis/`)
- ASPP kernel coverage, receptive field comparisons

---

## 9. Key Findings Summary

### 9.1 Stage 1 Leaderboard (413 results, top-5)

| Rank | Strategy | mIoU | Gain vs Baseline | Domain Gap |
|------|----------|------|------------------|------------|
| 1 | gen_automold | 40.45% | +2.84 pp | 5.66 |
| 2 | gen_UniControl | 40.37% | +2.76 pp | 5.29 |
| 3 | gen_albumentations_weather | 40.35% | +2.73 pp | 5.50 |
| 4 | gen_cyclediffusion | 40.12% | +2.51 pp | 5.67 |
| 5 | gen_Qwen_Image_Edit | 40.12% | +2.51 pp | 5.66 |
| — | baseline | 37.61% | — | 5.48 |

**All 25 strategies beat baseline** (gains +1.42 to +2.84 pp).

### 9.2 Stage 2 Leaderboard (195 results, top-5)

| Rank | Strategy | mIoU | Gain vs Baseline |
|------|----------|------|------------------|
| 1 | gen_augmenters | 42.05% | +1.26 pp |
| 2 | gen_IP2P | 41.98% | +1.18 pp |
| 3 | gen_VisualCloze | 41.90% | +1.11 pp |
| 4 | gen_albumentations_weather | 41.87% | +1.07 pp |
| 5 | gen_cyclediffusion | 41.85% | +1.05 pp |
| — | baseline | 40.80% | — |

**13/21 strategies beat baseline** in Stage 2; standard augmentations underperform.

### 9.3 CG Leaderboard (248 results)

| Rank | Strategy | mIoU | Gain vs Baseline |
|------|----------|------|------------------|
| 1 | gen_Attribute_Hallucination | 54.13% | +1.49 pp |
| 2 | gen_Img2Img | 52.87% | +0.22 pp |
| 3 | gen_augmenters | 52.82% | +0.18 pp |
| — | baseline | 52.65% | — |

Only 3 strategies beat baseline on Cityscapes (smaller gains than cross-dataset).

---

## 10. Analysis Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `generate_strategy_leaderboard.py` | Generate ranked leaderboards | CSVs + MD files |
| `analyze_ratio_ablation.py` | Ratio ablation analysis | Figures + CSVs |
| `analyze_extended_training.py` | Extended training convergence | Figures + CSVs |
| `analyze_baseline_consolidated.py` | Baseline per-domain analysis | Publication figures |
| `generate_stage_comparison_figures.py` | S1 vs S2 comparison | 6 publication figures |
| `analyze_domain_gap_corrected.py` | Domain gap analysis | Figures |
| `analyze_weather_domains.py` | Per-weather-domain analysis | Figures + CSVs |
| `analyze_unified_domain_gap.py` | Unified domain gap | Figures + CSVs |
| `analyze_strategy_families.py` | Strategy family grouping | Figures |
| `analyze_family_domains.py` | Family × domain analysis | Figures |
| `analyze_combination_ablation.py` | Strategy combination analysis | Figures |
| `analyze_mask2former_paradox.py` | Mask2Former behavior analysis | Report |
| `miou_collector.py` | Collect mIoU from weight dirs | Data extraction |
| `analyze_strategies_for_tsne.py` | t-SNE feature analysis | Figures |
| `visualize_ratio_ablation.py` | Ratio ablation visualization | Figures |
| `visualize_extended_training.py` | Extended training visualization | Figures |
| `generate_ieee_figures_extended_training.py` | IEEE-format figures | Publication figures |

---

## 11. Data Export Guide

### For the New Analysis Repository

To set up analysis on a different machine, you need these files:

#### Required (core results):
1. `downstream_results.csv` — S1 per-model results with per-domain/per-class metrics
2. `downstream_results_stage2.csv` — S2 per-model results
3. `downstream_results_cityscapes_gen.csv` — CG per-model results
4. `results/generative_quality/generative_quality.csv` — Generative quality metrics
5. `results/ratio_ablation_full.csv` — Ratio ablation results
6. `results/ratio_ablation_consolidated.csv` — Ratio ablation summary
7. `results/extended_training_analysis.csv` — Extended training iteration curves
8. `result_figures/leaderboard/breakdowns/*.csv` — All leaderboard breakdowns

#### Recommended (dataset metadata):
9. `${AWARE_DATA_ROOT}/FINAL_SPLITS/split_statistics.json` — Dataset split info
10. `${AWARE_DATA_ROOT}/GENERATED_IMAGES/all_manifests_summary.json` — Generation summary
11. `generated_manifests/*.json` — Per-strategy manifests (24 files)

#### Optional (per-strategy quality stats):
12. `${AWARE_DATA_ROOT}/STATS/{method}/{domain}_stats.json` — Per-method quality metrics

---

## 12. Naming Conventions

| Entity | Convention | Example |
|--------|-----------|---------|
| Datasets in dirs | lowercase, no hyphens | `bdd10k`, `iddaw`, `mapillaryvistas` |
| Strategies | prefix + method | `gen_cycleGAN`, `std_autoaugment` |
| Models | architecture_backbone | `segformer_mit-b3`, `mask2former_swin-b` |
| Ratio suffix | `_ratio{X}p{YZ}` | `pspnet_r50_ratio0p50` (50% gen) |
| Weight files | `iter_{N}.pth` | `iter_80000.pth` (Stage 1), `iter_20000.pth` (CG) |
| Test dirs | `test_results_detailed/` | Standard test output |
| Cross-domain test | `test_results_acdc/` | ACDC cross-domain |
