# Testing and Evaluation Guide

This document covers all aspects of testing and evaluating models with PROVE, including fine-grained per-domain/per-class analysis, cross-domain testing, result visualization, and analysis tools.

**Quick links:** [Datasets](DATASETS.md) | [Training](TRAINING.md) | [Advanced](ADVANCED.md) | [README](../README.md)

---

## Using test_unified.sh (Recommended)

The unified testing script provides a streamlined interface for evaluating trained models:

```bash
# Test a single trained model
./scripts/test_unified.sh single --dataset ACDC --model deeplabv3plus_r50 --strategy baseline

# Test with validation split
./scripts/test_unified.sh single --dataset ACDC --model deeplabv3plus_r50 --strategy baseline --test-split val

# Find available checkpoints
./scripts/test_unified.sh find --all

# Batch test all models on a dataset
./scripts/test_unified.sh batch --dataset ACDC --all-seg-models --strategy baseline --dry-run

# Submit test job to LSF cluster
./scripts/test_unified.sh submit --dataset ACDC --model deeplabv3plus_r50 --strategy baseline

# Test multi-dataset trained model (e.g., ACDC+Mapillary)
./scripts/test_unified.sh single-multi --datasets ACDC MapillaryVistas --model deeplabv3plus_r50

# View test results
./scripts/test_unified.sh results --dataset ACDC
```

**Test Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | Dataset name | Required |
| `--model` | Model name | Required |
| `--strategy` | Augmentation strategy used in training | `baseline` |
| `--ratio` | Real-to-generated ratio used in training | `1.0` |
| `--checkpoint` | Path to checkpoint (auto-detected if not specified) | Auto |
| `--test-split` | Test split: `val`, `test` | `test` |
| `--output-dir` | Output directory for results | Auto |

**Output Metrics (Segmentation):**
- `aAcc` - Average accuracy (overall pixel accuracy)
- `mIoU` - Mean Intersection over Union
- `mAcc` - Mean per-class accuracy  
- `fwIoU` - Frequency-weighted IoU

**Output Metrics (Detection):**
- `mAP` - Mean Average Precision
- `mAP_50` / `mAP_75` - mAP at IoU thresholds 0.50/0.75
- `mAP_s` / `mAP_m` / `mAP_l` - mAP by object size

## Fine-Grained Testing (Per-Domain/Per-Class)

For detailed analysis of model performance across weather domains and semantic classes:

```bash
# Run full detailed testing (per-domain and per-class metrics)
./scripts/test_unified.sh detailed --dataset ACDC --model deeplabv3plus_r50 --strategy baseline

# Batch detailed testing for all models
./scripts/test_unified.sh detailed-batch --all-seg-models --dataset ACDC --strategy baseline --dry-run

# Submit detailed test to LSF cluster
./scripts/test_unified.sh submit-detailed --dataset ACDC --model deeplabv3plus_r50 --strategy baseline

# Submit batch detailed tests
./scripts/test_unified.sh submit-detailed-batch --all-seg-datasets --all-seg-models --strategy baseline --dry-run
```

## Using fine_grained_test.py Directly

The `fine_grained_test.py` script provides fine-grained per-domain and per-class evaluation:

```bash
# Basic usage (all required arguments)
python fine_grained_test.py \
    --config /path/to/config.py \
    --checkpoint /path/to/checkpoint.pth \
    --dataset BDD10k \
    --output-dir results/my_experiment

# With custom batch size (larger = faster, uses more GPU memory)
python fine_grained_test.py \
    --config /path/to/config.py \
    --checkpoint /path/to/checkpoint.pth \
    --dataset MapillaryVistas \
    --output-dir results/mapillary_test \
    --batch-size 8

# Test on validation split instead of test split
python fine_grained_test.py \
    --config /path/to/config.py \
    --checkpoint /path/to/checkpoint.pth \
    --dataset ACDC \
    --output-dir results/acdc_val \
    --test-split val
```

**Required Arguments:**

| Argument | Description |
|----------|-------------|
| `--config` | Path to the MMSeg config file (usually in `configs/` subdirectory of weights folder) |
| `--checkpoint` | Path to the checkpoint file (e.g., `iter_80000.pth`) |
| `--dataset` | Dataset name: `ACDC`, `BDD10k`, `IDD-AW`, `MapillaryVistas`, `OUTSIDE15k` |
| `--output-dir` | Directory where results will be saved |

**Optional Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-root` | Auto | Override the data root directory |
| `--test-split` | `test` | Use `val` for validation split or `test` for test split |
| `--batch-size` | `10` | Batch size for inference (reduce if OOM errors occur) |

**Output Files:**
- `results.json` - Complete metrics (overall, per-domain, per-class)
- `test_report.txt` - Human-readable summary

**Structure of `results.json`:**
```json
{
  "overall": {
    "aAcc": 85.2,
    "mIoU": 62.5,
    "mAcc": 73.1,
    "fwIoU": 78.9
  },
  "per_domain": {
    "clear_day": {"aAcc": 91.2, "mIoU": 71.8, "...": "..."},
    "foggy": {"aAcc": 78.4, "mIoU": 55.2, "...": "..."}
  },
  "per_class": {
    "road": {"IoU": 95.2, "Acc": 97.1},
    "sidewalk": {"IoU": 72.4, "Acc": 85.3}
  }
}
```

**Available Domains by Dataset:**

| Dataset | Domains |
|---------|---------|
| ACDC | foggy, night, rainy, snowy |
| BDD10k | clear_day, cloudy, dawn_dusk, foggy, night, rainy, snowy |
| BDD100k | clear_day, cloudy, dawn_dusk, foggy, night, rainy, snowy |
| IDD-AW | clear_day, cloudy, dawn_dusk, foggy, night, rainy, snowy |
| MapillaryVistas | clear_day, cloudy, dawn_dusk, foggy, night, rainy, snowy |
| OUTSIDE15k | clear_day, cloudy, dawn_dusk, foggy, night, rainy, snowy |

## Cross-Domain Testing (Cityscapes → ACDC)

Test Cityscapes-trained models on ACDC adverse weather conditions:

```bash
# Preview available models and what would be tested
python scripts/test_cityscapes_replication_on_acdc.py --dry-run

# Test all models with per-domain breakdown
python scripts/test_cityscapes_replication_on_acdc.py

# Test specific models only
python scripts/test_cityscapes_replication_on_acdc.py --models segformer_b3 segnext_mscan_b

# Submit as LSF cluster jobs (parallel)
python scripts/test_cityscapes_replication_on_acdc.py --submit-jobs
```

**Output:** Per-domain mIoU breakdown (foggy, night, rainy, snowy) saved to `CITYSCAPES_REPLICATION/acdc_cross_domain_results/`

See [UNIFIED_TESTING.md](UNIFIED_TESTING.md) for comprehensive testing documentation.

## Result Visualization

Generate publication-quality visualizations from test results:

```bash
# Generate all visualizations for test results (includes insights by default)
python test_result_visualizer.py --results-dir /path/to/test_results_detailed/timestamp/

# Generate specific plot types
python test_result_visualizer.py --results-dir /path/to/results --plots domain class radar

# Compare multiple models
python test_result_visualizer.py --compare --results-dirs baseline_results cycleGAN_results --labels "Baseline" "CycleGAN"

# Skip insights printout
python test_result_visualizer.py --results-dir /path/to/results --no-insights
```

**Automatic Insights:** By default, the visualizer prints high-level insights about:
- Overall performance metrics (mIoU, fwIoU, mAcc, aAcc)
- Domain performance ranking with difficulty indicators (🟢 Easy, 🟡 Medium, 🔴 Hard)
- Top 3 and bottom 3 performing semantic classes
- Failing classes (IoU < 10%) that need attention

**Visualization Types:**
| Plot | Description |
|------|-------------|
| `domain_metrics.png` | Bar chart of mIoU/fwIoU by weather domain |
| `class_iou.png` | Per-class IoU horizontal bar chart |
| `domain_radar.png` | Radar chart comparing domains |
| `heatmap_iou.png` | Per-domain per-class IoU heatmap |
| `dashboard.png` | Comprehensive 6-panel summary |

See [RESULT_VISUALIZATION.md](RESULT_VISUALIZATION.md) for comprehensive visualization documentation.

## Result Analysis

Analyze test results across all configurations and generate comprehensive performance reports:

```bash
# Analyze all results with comprehensive summary
python test_result_analyzer.py

# Generate comprehensive summary with top 10 performers
python test_result_analyzer.py --comprehensive --top-n 10

# Show per-dataset insights
python test_result_analyzer.py --dataset-insights

# Show per-domain (weather condition) insights
python test_result_analyzer.py --domain-insights

# Show all insights (comprehensive + dataset + domain)
python test_result_analyzer.py --all-insights

# Filter by strategy or dataset
python test_result_analyzer.py --strategy baseline --dataset ACDC

# Output as JSON for programmatic processing
python test_result_analyzer.py --format json
```

**Analysis Options:**

| Option | Description |
|--------|-------------|
| `--comprehensive` | Top performers and strategy comparisons |
| `--dataset-insights` | Per-dataset performance analysis |
| `--domain-insights` | Per-domain (weather) performance analysis |
| `--all-insights` | All insights combined |
| `--top-n N` | Number of top configurations to show (default: 5) |
| `--domain-breakdown` | Detailed per-domain metrics table |

**Per-Dataset Insights** (`--dataset-insights`) provides:
- Overall statistics (avg, best, worst, std, spread)
- Best/worst configuration for each dataset
- Strategy effectiveness ranking per dataset
- Model architecture comparison per dataset
- Key recommendations

**Per-Domain Insights** (`--domain-insights`) provides:
- Domain difficulty ranking (easiest to hardest conditions)
- Performance gap analysis between domains
- Best configuration per weather domain
- Strategy effectiveness matrix (strategy × domain)
- Improvement potential and variability analysis

**Example Domain Insights Output:**
```
📊 DOMAIN DIFFICULTY RANKING (by average mIoU)
Rank  Domain               Avg mIoU    Best      Worst
1     foggy                  80.96%   100.00%    33.88%
2     snowy                  61.20%    82.40%    22.94%
3     rainy                  59.77%    76.76%    26.72%
...
✅ Easiest Domain: foggy (avg mIoU: 80.96%)
❌ Hardest Domain: dawn_dusk (avg mIoU: 40.59%)
📏 Domain Performance Gap: 40.36% mIoU

💡 KEY DOMAIN INSIGHTS
  foggy:
    • Best strategy: std_randaugment (avg 82.66%)
    • High variability (std=14.5%): Strategy choice matters significantly
```

## Baseline and Domain Gap Analysis

Comprehensive analysis scripts for understanding baseline performance and domain gaps:

```bash
# Baseline clear_day analysis (models trained only on clear weather)
python analyze_baseline_clear_day.py

# Class distribution analysis (explains fwIoU vs mIoU discrepancy)
python analyze_class_distribution.py

# Corrected domain gap analysis (uses mIoU, filters low-sample domains)
python analyze_domain_gap_corrected.py

# Full baseline analysis with mIoU focus
python analyze_baseline_miou.py
```

**Key Findings from Baseline Analysis:**

| Metric | Normal Conditions | Adverse Conditions | Domain Gap |
|--------|-------------------|-------------------|------------|
| mIoU | 54.96% | 47.49% | **7.46%** drop |
| fwIoU | 81.76% | 81.65% | 0.11% (misleading) |

**Per-Dataset Domain Gap (mIoU, reliable data only):**
| Dataset | Normal → Adverse Gap |
|---------|---------------------|
| IDD-AW | -16.20% |
| Outside15k | -15.36% |
| BDD10k | -3.60% |
| ACDC | -0.41% |

*Note: Domains with <50 test images excluded from analysis.*

**Strategy Effectiveness (Fair Comparison Group):**
| Strategy | Overall mIoU | Domain Gap | Δ Overall |
|----------|--------------|------------|-----------|
| **gen_StyleID** | 59.3% | -2.0% | +6.0% |
| std_photometric_distort | 57.3% | -3.5% | +4.0% |
| std_randaugment | 56.5% | -4.8% | +3.3% |
| baseline_clear_day | 53.2% | +7.5% | - |

*All augmentation strategies reduce domain gap and improve overall performance. gen_StyleID achieves best balance.*

See [DOMAIN_GAP_ANALYSIS.md](DOMAIN_GAP_ANALYSIS.md) for comprehensive domain gap analysis findings.

## Strategy Family Analysis

Analyze strategy performance grouped by families (2D Rendering, CNN/GAN, Style Transfer, Diffusion, etc.):

```bash
# Main family analysis (excludes combination strategies)
python analyze_strategy_families.py

# Family-domain cross-analysis (requires per-domain results)
python analyze_family_domains.py

# Combination strategy ablation study (WEIGHTS_COMBINATIONS)
python analyze_combination_ablation.py
```

**Strategy Families:**

| Family | Example Strategies |
|--------|-------------------|
| 2D Rendering | gen_automold, gen_imgaug_weather |
| CNN/GAN | gen_CUT, gen_cycleGAN, gen_SUSTechGAN |
| Style Transfer | gen_NST, gen_LANIT, gen_StyleID |
| Diffusion | gen_Img2Img, gen_IP2P, gen_UniControl |
| Multimodal Diffusion | gen_flux1_kontext, gen_step1x_new |
| Standard Augmentation | std_autoaugment, std_randaugment |
| Standard Mixing | std_cutmix, std_mixup |

See [FAMILY_ANALYSIS.md](FAMILY_ANALYSIS.md) for comprehensive family analysis documentation.

See [COMBINATION_ABLATION.md](COMBINATION_ABLATION.md) for combination strategy ablation documentation.

## Legacy Testing (prove.py)

Evaluate your trained model:

```bash
# Test object detection model
python prove.py test \
    --config-path prove_object_detection_bdd100k_json_config.py \
    --checkpoint-path ./work_dirs/od_experiment_001/latest.pth \
    --output-path ./results/od_results/

# Test semantic segmentation model
python prove.py test \
    --config-path prove_semantic_segmentation_cityscapes_config.py \
    --checkpoint-path ./work_dirs/seg_experiment_001/latest.pth \
    --output-path ./results/seg_results/
```
