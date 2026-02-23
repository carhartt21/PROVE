# PROVE Ablation Study Evaluation Scripts

This document describes the evaluation and visualization scripts for the PROVE ablation studies.

## Overview

The PROVE project includes two ablation studies:
1. **Ratio Ablation**: Varying the ratio of generated to real images (0.0 - 1.0)
2. **Extended Training**: Extending training beyond 80,000 iterations

Each study has an analyzer script (data collection/statistics) and a visualizer script (plots/figures).

## Ratio Ablation Study

### Data Sources

The ratio ablation results come from two sources:

| Source | Ratio Values | Description |
|--------|--------------|-------------|
| `WEIGHTS_RATIO_ABLATION` | 0.125, 0.25, 0.375, 0.625, 0.75, 0.875, 1.0 | Dedicated ablation experiments |
| `WEIGHTS` (regular) | 0.0, 0.5 | Baseline (no gen) and standard training (50/50 split) |

**Key insight**: Ratio 0.0 represents **baseline** training (no generated images), and ratio 0.5 represents **standard** gen_* strategy training, both of which are already in the regular `WEIGHTS` folder.

### analyze_ratio_ablation.py

Analyzes results from the ratio ablation study.

```bash
# Basic analysis (includes baseline and standard from regular WEIGHTS)
python analyze_ratio_ablation.py

# Verbose output showing all found results
python analyze_ratio_ablation.py --verbose

# Only analyze ablation results (exclude regular WEIGHTS)
python analyze_ratio_ablation.py --no-regular

# Export to CSV
python analyze_ratio_ablation.py --output results.csv --format csv

# Export to JSON
python analyze_ratio_ablation.py --output results.json --format json

# Custom weights directories
python analyze_ratio_ablation.py \
    --weights-root /path/to/ablation/weights \
    --regular-weights-root /path/to/regular/weights
```

#### Output

The analyzer produces:
- **Summary by Ratio**: Average mIoU/mAcc/aAcc at each ratio value
- **Summary by Strategy**: mIoU per ratio for each strategy
- **Optimal Ratios**: Best performing ratio for each strategy/dataset/model combination

### visualize_ratio_ablation.py

Generates visualizations for the ratio ablation study.

```bash
# Generate all plots
python visualize_ratio_ablation.py

# Generate specific plots
python visualize_ratio_ablation.py --plots line heatmap bar

# Specify output directory
python visualize_ratio_ablation.py --output-dir ./figures/ratio_ablation

# Exclude regular WEIGHTS (baseline/0.5)
python visualize_ratio_ablation.py --no-regular
```

#### Available Plot Types

| Plot | Flag | Description |
|------|------|-------------|
| Line plots | `line` | mIoU vs ratio for each strategy |
| Heatmap | `heatmap` | Strategy × ratio performance matrix |
| Bar chart | `bar` | Distribution of optimal ratios |
| Box plot | `box` | mIoU variance at each ratio |
| By dataset | `dataset` | Per-dataset learning curves |
| By model | `model` | Per-model architecture comparison |
| Improvement | `improvement` | Relative improvement vs baseline |
| Dashboard | `dashboard` | Comprehensive summary dashboard |
| All | `all` | Generate all plots (default) |

#### Generated Files

```
figures/ratio_ablation/
├── miou_vs_ratio_by_strategy.png
├── heatmap_strategy_ratio.png
├── optimal_ratio_distribution.png
├── miou_boxplot_by_ratio.png
├── miou_vs_ratio_by_dataset.png
├── miou_vs_ratio_by_model.png
├── relative_improvement_vs_baseline.png
├── ratio_ablation_dashboard.png
└── *.pdf (PDF versions of all plots)
```

---

## Extended Training Study

### Data Sources

Extended training results are stored in `WEIGHTS_EXTENDED` with test results at different iteration checkpoints (80k, 120k, 160k, etc.).

### analyze_extended_training.py

Analyzes results from extended training experiments.

```bash
# Basic analysis
python analyze_extended_training.py

# Verbose output
python analyze_extended_training.py --verbose

# Filter by strategy
python analyze_extended_training.py --strategy gen_LANIT

# Filter by dataset
python analyze_extended_training.py --dataset ACDC

# Export results
python analyze_extended_training.py --export-csv results.csv
python analyze_extended_training.py --export-json results.json

# Custom weights directory
python analyze_extended_training.py --weights-root /path/to/extended/weights
```

#### Output

The analyzer produces:
- **Summary by Iteration**: Average mIoU at each training checkpoint
- **Convergence Analysis**: When each configuration reaches peak performance
- **Improvement Statistics**: Impact of extended training vs. standard 80k iterations

### visualize_extended_training.py

Generates visualizations for the extended training study.

```bash
# Generate all plots
python visualize_extended_training.py

# Generate specific plots
python visualize_extended_training.py --plots learning convergence improvement

# Specify output directory
python visualize_extended_training.py --output-dir ./figures/extended_training
```

#### Available Plot Types

| Plot | Flag | Description |
|------|------|-------------|
| Learning curves | `learning` | mIoU vs iterations for each strategy |
| Improvement | `improvement` | Histogram of improvements |
| Convergence | `convergence` | When training peaks |
| By strategy | `strategy` | Improvement per strategy |
| By dataset | `dataset` | Per-dataset training curves |
| By model | `model` | Per-model comparison |
| Heatmap | `heatmap` | Strategy × iteration matrix |
| Diminishing returns | `diminishing` | Marginal improvement per step |
| Dashboard | `dashboard` | Comprehensive summary |
| All | `all` | Generate all plots (default) |

#### Generated Files

```
figures/extended_training/
├── learning_curves_by_strategy.png
├── improvement_histogram.png
├── convergence_analysis.png
├── improvement_by_strategy.png
├── learning_curves_by_dataset.png
├── learning_curves_by_model.png
├── heatmap_strategy_iteration.png
├── diminishing_returns.png
├── extended_training_dashboard.png
└── *.pdf (PDF versions of all plots)
```

---

## Dependencies

```bash
# Required for analysis
pip install tabulate pandas

# Required for visualization
pip install matplotlib seaborn numpy pandas
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PROVE_WEIGHTS_ROOT` | Root directory for weights | `${AWARE_DATA_ROOT}/WEIGHTS` |

### Default Paths

| Path | Purpose |
|------|---------|
| `${AWARE_DATA_ROOT}/WEIGHTS` | Regular training results |
| `${AWARE_DATA_ROOT}/WEIGHTS_RATIO_ABLATION` | Ratio ablation results |
| `${AWARE_DATA_ROOT}/WEIGHTS_EXTENDED` | Extended training results |

## Examples

### Complete Ratio Ablation Analysis

```bash
# Full analysis workflow
python analyze_ratio_ablation.py --detailed --output ratio_results.csv
python visualize_ratio_ablation.py --output-dir ./figures/ratio_ablation
```

### Complete Extended Training Analysis

```bash
# Full analysis workflow
python analyze_extended_training.py --verbose --export-csv extended_results.csv
python visualize_extended_training.py --output-dir ./figures/extended_training
```

### Filtered Analysis

```bash
# Analyze only gen_LANIT on ACDC
python analyze_ratio_ablation.py --verbose 2>&1 | grep "gen_LANIT/ACDC"

# Analyze extended training for specific configuration
python analyze_extended_training.py --strategy gen_LANIT --dataset ACDC
```

## Programmatic Usage

Both analyzers can be used as Python modules:

```python
from analyze_ratio_ablation import RatioAblationAnalyzer

# Create analyzer
analyzer = RatioAblationAnalyzer()
analyzer.scan_results(verbose=True, include_regular=True)

# Get summaries
ratio_summary = analyzer.get_summary_by_ratio()
strategy_summary = analyzer.get_summary_by_strategy()
optimal_ratios = analyzer.get_optimal_ratios()

# Access raw results
for result in analyzer.results:
    print(f"{result.strategy}/{result.dataset}: ratio={result.ratio}, mIoU={result.miou}")
```

```python
from analyze_extended_training import ExtendedTrainingAnalyzer

# Create analyzer
analyzer = ExtendedTrainingAnalyzer()
analyzer.scan_results(verbose=True)

# Get summaries
iter_summary = analyzer.get_summary_by_iteration()
convergence = analyzer.get_convergence_analysis()
stats = analyzer.get_improvement_stats()

# Filter results
filtered = analyzer.filter_results(strategy='gen_LANIT', dataset='ACDC')
```

## See Also

- [RATIO_ABLATION.md](RATIO_ABLATION.md) - Job submission documentation for ratio ablation
- [EXTENDED_TRAINING.md](EXTENDED_TRAINING.md) - Job submission documentation for extended training
- [RESULT_VISUALIZATION.md](RESULT_VISUALIZATION.md) - General result visualization documentation
