# Initial Prompt for PROVE Analysis Repository

Use this prompt in a new chat window to set up the analysis/visualization repository.

---

## Prompt

```
I want to create a new Python analysis and visualization repository called "PROVE-Analysis" that combines results from a large-scale semantic segmentation robustness evaluation project (PROVE). The project tests whether generative data augmentation improves model robustness under adverse weather conditions.

### Goal
A self-contained analysis repository that:
1. Imports and processes segmentation performance results (CSV files with per-model, per-domain, per-class mIoU metrics)
2. Imports generative quality metrics (FID, LPIPS, SSIM, semantic consistency scores per generation method)
3. Imports dataset statistics (image counts per domain/dataset, train/test splits)
4. Correlates generative image quality with downstream segmentation performance
5. Generates publication-quality IEEE-format figures and LaTeX tables
6. Provides interactive exploration via Jupyter notebooks

### Data Sources (to be copied into the repo's `data/` directory)

**Segmentation Results (3 CSVs, each with columns: strategy, dataset, model, test_type, mIoU, mAcc, aAcc, fwIoU, per_domain_metrics [JSON], per_class_metrics [JSON]):**
- `downstream_results.csv` — Stage 1: Clear-day training, cross-domain robustness (413 test results, 26 strategies, 4 datasets, 4 models)
- `downstream_results_stage2.csv` — Stage 2: All-domain training (195 results, 22 strategies)
- `downstream_results_cityscapes_gen.csv` — Cityscapes-Gen: Standard benchmark (248 results, 25 strategies, Cityscapes + ACDC cross-domain)

**Leaderboard Breakdowns (CSVs with strategy × dimension matrices):**
- `strategy_leaderboard_{stage}.csv` — Strategy rankings with mIoU, Std, Gain, Normal/Adverse mIoU, Domain Gap
- `per_dataset_breakdown_{stage}.csv` — Strategy × Dataset mIoU
- `per_domain_breakdown_{stage}.csv` — Strategy × Domain mIoU (clear_day, cloudy, dawn_dusk, foggy, night, rainy, snowy)
- `per_model_breakdown_{stage}.csv` — Strategy × Model mIoU

**Generative Quality:**
- `generative_quality.csv` — Per-method: CQS, FID, LPIPS, SSIM, PSNR, semantic Pixel_Accuracy, mIoU, fw_IoU, Num_Images (20 methods)

**Ablation Studies:**
- `ratio_ablation_full.csv` — Real/generated mix ratios (0.00–0.88) × strategies × datasets × models with mIoU
- `extended_training_analysis.csv` — Iteration convergence (40k–320k) with mIoU per checkpoint

**Dataset Metadata:**
- `split_statistics.json` — 6 datasets, 7 weather domains, train/test split counts (103,203 total images)
- `all_manifests_summary.json` — Per-generation-method: total generated, total matched, match rate, domains

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
│   ├── segmentation/        # downstream_results*.csv
│   ├── quality/             # generative_quality.csv
│   ├── ablation/            # ratio and extended training CSVs
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
├── notebooks/               # Analysis notebooks
│   ├── 01_data_overview.ipynb
│   ├── 02_strategy_leaderboard.ipynb
│   ├── 03_quality_vs_performance.ipynb
│   ├── 04_domain_analysis.ipynb
│   ├── 05_ablation_studies.ipynb
│   └── 06_publication_figures.ipynb
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
3. After setup, copy the data files from PROVE into the `data/` directory:

```bash
# From PROVE repository root:
cp downstream_results.csv /path/to/PROVE-Analysis/data/segmentation/
cp downstream_results_stage2.csv /path/to/PROVE-Analysis/data/segmentation/
cp downstream_results_cityscapes_gen.csv /path/to/PROVE-Analysis/data/segmentation/
cp results/generative_quality/generative_quality.csv /path/to/PROVE-Analysis/data/quality/
cp results/ratio_ablation_full.csv /path/to/PROVE-Analysis/data/ablation/
cp results/ratio_ablation_consolidated.csv /path/to/PROVE-Analysis/data/ablation/
cp results/extended_training_analysis.csv /path/to/PROVE-Analysis/data/ablation/
cp result_figures/leaderboard/breakdowns/*.csv /path/to/PROVE-Analysis/data/leaderboard/

# From external data:
cp /scratch/aaa_exchange/AWARE/FINAL_SPLITS/split_statistics.json /path/to/PROVE-Analysis/data/metadata/
cp /scratch/aaa_exchange/AWARE/GENERATED_IMAGES/all_manifests_summary.json /path/to/PROVE-Analysis/data/metadata/
```
