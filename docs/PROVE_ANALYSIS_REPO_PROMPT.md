# Initial Prompt for PROVE Analysis Repository

Use this prompt in a new chat window to set up the analysis/visualization repository.

---

## Prompt

```
I want to create a new Python analysis and visualization repository that combines results from a large-scale semantic segmentation robustness evaluation project (PROVE). The project tests whether generative data augmentation improves model robustness under adverse weather conditions.

### Goal
A self-contained analysis repository that:
1. Imports and processes segmentation performance results (CSV files with per-dataset, per-model, per-domain, per-class mIoU metrics)
2. Imports generative quality metrics (FID, LPIPS, SSIM, semantic consistency scores per generation method)
3. Imports dataset statistics (image counts per domain/dataset, train/test splits)
4. Correlates generative image quality with downstream segmentation performance
5. Generates consistent publication-quality IEEE-format figures and LaTeX tables

### Data Sources (strucured by directory)
-The file names below are just examples and could differ
#### PROVE
**Segmentation Results (3 CSVs, each with columns: strategy, dataset, model, test_type, mIoU, mAcc, aAcc, fwIoU, per_domain_metrics [JSON], per_class_metrics [JSON]):**
- `downstream_results.csv` — Stage 1: Clear-day training, cross-domain robustness (413 test results, 26 strategies, 4 datasets, 4 models)
- `downstream_results_stage2.csv` — Stage 2: All-domain training (195 results, 22 strategies)
- `downstream_results_cityscapes_gen.csv` — Cityscapes-Gen: Standard benchmark (248 results, 25 strategies, Cityscapes + ACDC cross-domain)

**Leaderboard Breakdowns (CSVs with strategy × dimension matrices):**
- `strategy_leaderboard_{stage}.csv` — Strategy rankings with mIoU, Std, Gain, Normal/Adverse mIoU, Domain Gap
- `per_dataset_breakdown_{stage}.csv` — Strategy × Dataset mIoU
- `per_domain_breakdown_{stage}.csv` — Strategy × Domain mIoU (clear_day, cloudy, dawn_dusk, foggy, night, rainy, snowy)
- `per_model_breakdown_{stage}.csv` — Strategy × Model mIoU

**Ablation Studies:**
- `ratio_ablation_full.csv` — Real/generated mix ratios (0.00–0.88) × strategies × datasets × models with mIoU
- `extended_training_analysis.csv` — Iteration convergence (40k–320k) with mIoU per checkpoint

#### PRISM
**Generative Quality:**
- `generative_quality.csv` — Per-method: CQS, FID, LPIPS, SSIM, PSNR, semantic Pixel_Accuracy, mIoU, fw_IoU, Num_Images (20 methods)

#### DATA
**Dataset Metadata:**
- `split_statistics.json` — 6 datasets, 7 weather domains, train/test split counts (103,203 total images)
- `all_manifests_summary.json` — Per-generation-method: total generated, total matched, match rate, domains

#### SWIFT
**Domain distribution**

### Key Analysis Dimensions
1. **Strategy Effectiveness**: Which augmentation strategies improve robustness? (leaderboard analysis)
2. **Quality vs Performance Correlation**: Does higher generative quality (lower CQS/FID) predict better downstream mIoU?
3. **Domain-Specific Impact**: Which strategies help most for specific adverse conditions (night, fog, rain, snow)?
4. **Model Architecture Sensitivity**: Do different architectures benefit differently from augmentation?
5. **Training Stage Comparison**: How do S1 (clear-only) vs S2 (all-domain) results differ?
6. **Ratio Optimization**: What's the optimal real/generated data mix ratio?
7. **Compute Efficiency**: At what training iteration do gains plateau?

### Technical Requirements
- Python 3.10+
- pandas, numpy, matplotlib, seaborn, scipy
- Plotly for interactive visualizations
- LaTeX table generation (preferably with pandas styler or booktabs format)
- IEEE double-column figure sizing (3.5" single column, 7" double column)
- Consistent color scheme for strategies (gen = blues/greens, std = oranges, baseline = gray)
- Reproducible figures with fixed random seeds

### Repository Structure
```
PROVE-Analysis/
├── data/                    # Raw data files (CSVs, JSONs)
│   ├── PROVE/        # downstream_results*.csv + breakdowns and ablation result CSVs
│   ├── PRISM/             # generative_quality.csv
│   ├── ablatio/            # 
│   ├── leaderboard/         # strategy leaderboard breakdown CSVs
│   └── metadata/            # split_statistics.json, manifests
├── src/                     # Analysis modules
│   ├── data_loader.py       # Unified data loading with caching
│   ├── preprocessing.py     # Parse JSON columns, compute derived metrics
│   ├── correlation.py       # Quality vs performance correlation analysis
│   ├── domain_analysis.py   # Per-domain breakdown and gap analysis
│   ├── ablation_analysis.py # Ratio and extended training analysis
│   ├── plotting.py          # Shared plotting utilities (IEEE styling)
│   └── latex_export.py      # LaTeX table generation
├── figures/                 # Generated figures
│   ├── ieee/                # Publication-ready IEEE format
│   └── exploration/         # Interactive/exploratory figures
├── tables/                  # Generated LaTeX tables
├── pyproject.toml
└── README.md
```

### Key Relationships to Analyze
- 21 generative methods appear in both `generative_quality.csv` (with CQS, FID, SSIM, LPIPS) AND `downstream_results.csv` (with mIoU). The strategy names map as: `gen_{method}` in downstream ↔ `{method}` in quality CSV (e.g., `gen_cycleGAN` ↔ `cycleGAN`).
- 7 weather domains: clear_day, cloudy, dawn_dusk, foggy, night, rainy, snowy
- 4 test datasets: BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k (+ Cityscapes for CG stage)
- 4 core models: pspnet_r50, segformer_mit-b3, segnext_mscan-b, mask2former_swin-b
- Domain gap = Normal mIoU − Adverse mIoU (lower = more robust)

Please set up the repository structure, create the core analysis modules, and implement the first notebook (data overview) that loads all data sources and provides summary statistics.
```

---

## How to Use

1. Copy this prompt into a new Copilot chat window
2. Open the new workspace/directory where you want the repo created
3. Export data from PROVE using the automated export script:

```bash
# From PROVE repository root:

# Preview what will be exported (recommended first):
python scripts/export_analysis_data.py /path/to/PROVE-Analysis/data --dry-run

# Export core data (67 files, ~4 MB):
python scripts/export_analysis_data.py /path/to/PROVE-Analysis/data

# Export with per-strategy quality stats (214 files, ~5 MB):
python scripts/export_analysis_data.py /path/to/PROVE-Analysis/data --include-stats

# Export with FID reference features too (221 files, ~235 MB):
python scripts/export_analysis_data.py /path/to/PROVE-Analysis/data --include-stats --include-fid
```

The export script:
- Copies all downstream results CSVs, leaderboard breakdowns, quality metrics
- Extracts lightweight aggregate stats from large per-image quality JSONs
- Copies dataset metadata (split statistics, generation manifests)
- Writes `_export_manifest.json` for data provenance tracking
