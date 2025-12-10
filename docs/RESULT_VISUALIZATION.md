# PROVE Test Result Visualization

This document describes how to generate visualizations from PROVE test results using `test_result_visualizer.py`.

## Overview

The test result visualizer generates publication-quality figures from the detailed test results produced by `test_unified.sh detailed`. It supports multiple plot types and can compare results across different models and strategies.

## Prerequisites

```bash
# Install visualization dependencies
pip install matplotlib seaborn pandas
```

## Quick Start

```bash
# Generate all visualizations for a test result
python test_result_visualizer.py --results-dir /path/to/test_results_detailed/timestamp/

# Generate specific plot types
python test_result_visualizer.py --results-dir /path/to/results --plots domain class

# Compare multiple models
python test_result_visualizer.py --compare --results-dirs result1 result2 --labels "Model A" "Model B"
```

## Input Data Structure

The visualizer expects the output from `test_unified.sh detailed`:

```
test_results_detailed/
└── 20251210_094301/          # Timestamped folder
    ├── metrics_summary.json   # Overall metrics
    ├── metrics_per_domain.json # Per-domain breakdown
    ├── metrics_per_class.json  # Per-class metrics
    ├── metrics_full.json       # Complete detailed metrics
    ├── per_domain_metrics.csv  # CSV format
    └── per_class_metrics.csv   # CSV format
```

## Plot Types

### 1. Domain Metrics Bar Chart (`domain`)

Grouped bar chart showing mIoU and fwIoU for each weather domain.

```bash
python test_result_visualizer.py --results-dir /path/to/results --plots domain
```

**Output**: `figures/domain_metrics.png`

![Domain Metrics Example](examples/domain_metrics.png)

### 2. Per-Class IoU Chart (`class`)

Horizontal bar chart of per-class IoU, color-coded by performance (red-yellow-green gradient).

```bash
python test_result_visualizer.py --results-dir /path/to/results --plots class
```

**Output**: `figures/class_iou.png`, `figures/class_acc.png`

![Per-Class IoU Example](examples/class_iou.png)

### 3. Domain Radar Chart (`radar`)

Polar chart comparing all domains across multiple metrics (mIoU, fwIoU, mAcc, aAcc).

```bash
python test_result_visualizer.py --results-dir /path/to/results --plots radar
```

**Output**: `figures/domain_radar.png`

![Domain Radar Example](examples/domain_radar.png)

### 4. Per-Domain Per-Class Heatmap (`heatmap`)

2D heatmap showing IoU for each class within each domain. Requires full metrics data.

```bash
python test_result_visualizer.py --results-dir /path/to/results --plots heatmap
```

**Output**: `figures/heatmap_iou.png`

### 5. Summary Dashboard (`dashboard`)

Comprehensive 6-panel dashboard with:
- Overall metrics bar chart
- Per-domain performance comparison
- Per-class IoU (sorted)
- Domain radar chart
- Image distribution pie chart
- Best/worst classes comparison

```bash
python test_result_visualizer.py --results-dir /path/to/results --plots dashboard
```

**Output**: `figures/dashboard.png`

![Dashboard Example](examples/dashboard.png)

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--results-dir <path>` | Path to test results directory | Required |
| `--output-dir <path>` | Output directory for figures | `results_dir/figures` |
| `--plots <types>` | Plot types: `domain`, `class`, `radar`, `heatmap`, `dashboard`, `all` | `all` |
| `--format <fmt>` | Output format: `png`, `pdf`, `svg` | `png` |
| `--dpi <int>` | Figure resolution | `150` |
| `--show` | Display figures interactively | Disabled |

### Comparison Mode

| Option | Description |
|--------|-------------|
| `--compare` | Enable comparison mode |
| `--results-dirs <paths>` | List of result directories to compare |
| `--labels <names>` | Labels for each result (default: directory names) |

## Python API

```python
from test_result_visualizer import TestResultVisualizer, compare_results

# Single result visualization
visualizer = TestResultVisualizer(
    results_dir='/path/to/test_results_detailed/timestamp/',
    output_dir='/path/to/figures',  # Optional
    figsize=(12, 8),
    dpi=150
)

# Generate individual plots
visualizer.plot_domain_metrics(metrics=['mIoU', 'fwIoU'])
visualizer.plot_class_metrics(metric='IoU', sort_by='value')
visualizer.plot_radar_chart()
visualizer.plot_heatmap(metric='IoU')
visualizer.plot_summary_dashboard()

# Generate all plots
visualizer.generate_all_plots()

# Compare multiple results
compare_results(
    results_dirs=['/path/to/result1', '/path/to/result2'],
    output_dir='/path/to/comparison',
    labels=['Baseline', 'CycleGAN']
)
```

## Customization

### Color Palettes

Domain colors can be customized in the `DOMAIN_COLORS` dictionary:

```python
DOMAIN_COLORS = {
    'clear_day': '#FFD700',      # Gold
    'cloudy': '#A9A9A9',         # Dark Gray
    'dawn_dusk': '#FF8C00',      # Dark Orange
    'foggy': '#B0C4DE',          # Light Steel Blue
    'night': '#191970',          # Midnight Blue
    'rainy': '#4682B4',          # Steel Blue
    'snowy': '#F0F8FF',          # Alice Blue
}
```

### Figure Styles

Available matplotlib styles:
- `seaborn-v0_8-whitegrid` (default)
- `seaborn-v0_8-darkgrid`
- `ggplot`
- `default`

```python
visualizer = TestResultVisualizer(
    results_dir='/path/to/results',
    style='ggplot'
)
```

## Example Workflow

```bash
# 1. Run detailed test
./test_unified.sh detailed --dataset ACDC --model deeplabv3plus_r50 --strategy baseline

# 2. Find the timestamped results folder
ls /scratch/aaa_exchange/AWARE/WEIGHTS/baseline/acdc/deeplabv3plus_r50/test_results_detailed/

# 3. Generate visualizations
python test_result_visualizer.py \
    --results-dir /scratch/aaa_exchange/AWARE/WEIGHTS/baseline/acdc/deeplabv3plus_r50/test_results_detailed/20251210_094301/

# 4. View generated figures
ls /scratch/aaa_exchange/AWARE/WEIGHTS/baseline/acdc/deeplabv3plus_r50/test_results_detailed/20251210_094301/figures/
```

## Integration with test_unified.sh

You can add visualization generation directly to your testing workflow:

```bash
# Run detailed test and generate visualizations
./test_unified.sh detailed --dataset ACDC --model deeplabv3plus_r50 --strategy baseline && \
python test_result_visualizer.py --results-dir $(ls -td /scratch/aaa_exchange/AWARE/WEIGHTS/baseline/acdc/deeplabv3plus_r50/test_results_detailed/*/ | head -1)
```

## Output Examples

### Per-Domain Performance (ACDC Dataset)

| Domain | mIoU | fwIoU | Images |
|--------|------|-------|--------|
| foggy | 68.78% | 96.18% | 144 |
| rainy | 58.16% | 91.60% | 162 |
| snowy | 54.72% | 90.11% | 190 |
| cloudy | 42.45% | 86.50% | 217 |
| night | 41.43% | 82.74% | 183 |
| clear_day | 25.28% | 69.83% | 301 |
| dawn_dusk | 20.38% | 65.94% | 16 |

### Per-Class IoU (Top 5)

| Class | IoU |
|-------|-----|
| road | 92.71% |
| sky | 92.60% |
| vegetation | 75.20% |
| building | 72.90% |
| sidewalk | 66.44% |

### Per-Class IoU (Bottom 5)

| Class | IoU |
|-------|-----|
| motorcycle | 5.36% |
| rider | 12.50% |
| bicycle | 19.46% |
| person | 22.60% |
| pole | 28.84% |

## See Also

- [UNIFIED_TESTING.md](UNIFIED_TESTING.md) - Testing documentation
- [UNIFIED_TRAINING.md](UNIFIED_TRAINING.md) - Training documentation
- [fine_grained_test.py](../fine_grained_test.py) - Detailed test script
