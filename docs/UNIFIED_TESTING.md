# PROVE Unified Testing Documentation

## Overview

The `test_unified.sh` script provides a unified interface for testing trained models using checkpoints created by the PROVE training pipeline. It is analogous to `train_unified.sh` but focused on evaluation and testing.

## Features

- **Automatic Checkpoint Discovery**: Automatically finds checkpoints from the training work directory
- **Batch Testing**: Test multiple configurations at once
- **LSF Cluster Support**: Submit test jobs to LSF cluster with `submit` and `submit-batch` commands
- **Results Aggregation**: View and compare test results across configurations
- **Flexible Output**: Save metrics to JSON and visualize predictions

## Quick Start

```bash
# Test a single trained model
./test_unified.sh single --dataset ACDC --model deeplabv3plus_r50 --strategy baseline

# Fine-grained per-domain testing
./test_unified.sh detailed --dataset ACDC --model deeplabv3plus_r50 --strategy baseline --mode per-domain

# Find available checkpoints
./test_unified.sh find --all

# Batch test all segmentation models on ACDC
./test_unified.sh batch --dataset ACDC --all-seg-models --strategy baseline --dry-run

# Submit test job to LSF cluster
./test_unified.sh submit --dataset ACDC --model deeplabv3plus_r50 --strategy baseline
```

## Commands

### `detailed` - Fine-Grained Per-Domain/Per-Class Metrics

Run detailed testing with metrics breakdown by domain and class:

```bash
./test_unified.sh detailed --dataset ACDC --model deeplabv3plus_r50 --strategy baseline --mode per-domain
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset <name>` | Dataset name | Required |
| `--model <name>` | Model name | Required |
| `--strategy <name>` | Training strategy | `baseline` |
| `--ratio <ratio>` | Real-to-generated ratio | `1.0` |
| `--mode <mode>` | Testing mode: `per-domain`, `per-class`, `both` | `per-domain` |
| `--test-split <split>` | Test split | `test` |

**Output Files:**

Results are saved in a timestamped folder under `test_results_detailed/`:

| File | Description |
|------|-------------|
| `metrics_summary.json` | Overall metrics and configuration |
| `metrics_per_domain.json` | Metrics broken down by weather domain |
| `metrics_per_class.json` | IoU and Accuracy per semantic class |
| `metrics_full.json` | Complete metrics with per-domain per-class breakdown |
| `test_report.txt` | Human-readable text report |
| `per_domain_metrics.csv` | CSV for easy import into spreadsheets |
| `per_class_metrics.csv` | CSV with per-class metrics |

**Example Output:**

```
FINAL TEST RESULTS SUMMARY
============================================================
  aAcc: 90.46
  mIoU: 45.83
  mAcc: 53.64
  fwIoU: 83.39
  num_images: 1213

Per-Domain Results:
  clear_day:  mIoU=25.28, fwIoU=69.83 (301 images)
  cloudy:     mIoU=42.45, fwIoU=86.50 (217 images)
  foggy:      mIoU=68.78, fwIoU=96.18 (144 images)
  night:      mIoU=41.43, fwIoU=82.74 (183 images)
  rainy:      mIoU=58.16, fwIoU=91.60 (162 images)
  snowy:      mIoU=54.72, fwIoU=90.11 (190 images)
```

**Available Domains by Dataset:**

| Dataset | Domains |
|---------|---------|
| ACDC | clear_day, cloudy, dawn_dusk, foggy, night, rainy, snowy |
| BDD10k | daytime, night, dawn_dusk |
| BDD100k | daytime, night, dawn_dusk |
| IDD-AW | clear, rainy, foggy, hazy, low_light |
| MapillaryVistas | all |
| OUTSIDE15k | all |

---

### `single` - Test Single Checkpoint

Test a single trained checkpoint:

```bash
./test_unified.sh single --dataset ACDC --model deeplabv3plus_r50 --strategy baseline
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset <name>` | Dataset name (ACDC, BDD10k, BDD100k, IDD-AW, MapillaryVistas, OUTSIDE15k) | Required |
| `--model <name>` | Model name (deeplabv3plus_r50, pspnet_r50, segformer_mit-b5, etc.) | Required |
| `--strategy <name>` | Augmentation strategy used during training | `baseline` |
| `--ratio <ratio>` | Real-to-generated ratio used during training | `1.0` |
| `--checkpoint <path>` | Path to checkpoint file | Auto-detected |
| `--checkpoint-type <type>` | Checkpoint type: `best`, `latest` | `best` |
| `--work-dir <path>` | Work directory root | `$PROVE_WEIGHTS_ROOT` |
| `--output-dir <path>` | Output directory for results | `work_dir/test_results` |
| `--test-split <split>` | Test split: `val`, `test` | `test` |
| `--show` | Visualize results | Disabled |
| `--show-dir <path>` | Directory to save visualizations | - |

**Examples:**

```bash
# Basic test with baseline strategy
./test_unified.sh single --dataset ACDC --model deeplabv3plus_r50 --strategy baseline

# Test with generative augmentation model
./test_unified.sh single --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN --ratio 0.5

# Test on validation split
./test_unified.sh single --dataset ACDC --model deeplabv3plus_r50 --strategy baseline --test-split val

# Test with custom checkpoint
./test_unified.sh single --checkpoint /path/to/checkpoint.pth --dataset ACDC --model deeplabv3plus_r50
```

### `batch` - Batch Testing

Test multiple configurations:

```bash
./test_unified.sh batch --all-seg-datasets --all-seg-models --strategy baseline --dry-run
```

**Options:**

| Option | Description |
|--------|-------------|
| `--datasets <names...>` | List of datasets |
| `--models <names...>` | List of models |
| `--strategies <names...>` | List of strategies |
| `--ratios <values...>` | List of ratios |
| `--all-seg-datasets` | Use all segmentation datasets |
| `--all-det-datasets` | Use all detection datasets |
| `--all-seg-models` | Use all segmentation models |
| `--all-det-models` | Use all detection models |
| `--dry-run` | Show commands without executing |

**Examples:**

```bash
# Preview all segmentation tests
./test_unified.sh batch --all-seg-datasets --all-seg-models --strategy baseline --dry-run

# Test all models on a single dataset
./test_unified.sh batch --dataset ACDC --all-seg-models --strategies baseline gen_cycleGAN

# Test multiple strategies on all datasets
./test_unified.sh batch --all-seg-datasets --model deeplabv3plus_r50 --strategies baseline std_cutmix gen_cycleGAN
```

### `submit` - Submit to LSF Cluster

Submit a single test job to LSF cluster:

```bash
./test_unified.sh submit --dataset ACDC --model deeplabv3plus_r50 --strategy baseline
```

**LSF Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--queue <name>` | `BatchGPU` | LSF queue name |
| `--gpu-mem <size>` | `16G` | GPU memory requirement |
| `--num-cpus <n>` | `4` | Number of CPUs per job |
| `--dry-run` | - | Show bsub command without executing |

### `submit-batch` - Submit Batch to LSF

Submit multiple test jobs to LSF cluster:

```bash
./test_unified.sh submit-batch --all-seg-datasets --all-seg-models --strategy baseline --dry-run
```

### `find` - Find Checkpoints

Search for available checkpoints:

```bash
# Find all checkpoints
./test_unified.sh find --all

# Find checkpoints for specific dataset
./test_unified.sh find --dataset ACDC

# Find checkpoints for specific configuration
./test_unified.sh find --dataset ACDC --model deeplabv3plus_r50 --strategy baseline
```

### `results` - View Results

Display test results summary:

```bash
# View all results
./test_unified.sh results

# View results for specific dataset
./test_unified.sh results --dataset ACDC

# View results for specific model
./test_unified.sh results --model deeplabv3plus_r50
```

### `list` - List Options

Display available datasets, models, and strategies:

```bash
./test_unified.sh list
```

## Output Structure

Test results are saved in the following structure:

```
{WEIGHTS_ROOT}/{strategy}/{dataset}/{model}/
├── iter_80000.pth           # Trained checkpoint
├── configs/
│   └── training_config.py   # Training configuration
├── logs/                    # Training logs
└── test_results/
    └── test/                # Test split results
        ├── metrics.json     # Test metrics
        └── vis/             # Visualizations (if --show enabled)
```

## Metrics

### Segmentation Metrics

For semantic segmentation tasks, the following metrics are computed:

| Metric | Description |
|--------|-------------|
| `aAcc` | Average Accuracy - overall pixel accuracy |
| `mIoU` | Mean Intersection over Union - average IoU across all classes |
| `mAcc` | Mean Accuracy - average per-class accuracy |
| `fwIoU` | Frequency Weighted IoU - IoU weighted by class frequency |

**Example Output:**
```
+---------------+-------+-------+
|     Class     |  IoU  |  Acc  |
+---------------+-------+-------+
|      road     | 92.71 | 96.86 |
|    sidewalk   | 66.44 | 80.35 |
|    building   | 72.90 | 85.47 |
...
+---------------+-------+-------+

test/aAcc: 90.46%
test/mIoU: 45.83%
test/mAcc: 53.64%
test/fwIoU: 83.39%
```

### Detection Metrics

For object detection tasks:

| Metric | Description |
|--------|-------------|
| `mAP` | Mean Average Precision |
| `mAP_50` | mAP at IoU threshold 0.50 |
| `mAP_75` | mAP at IoU threshold 0.75 |
| `mAP_s` | mAP for small objects |
| `mAP_m` | mAP for medium objects |
| `mAP_l` | mAP for large objects |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PROVE_WEIGHTS_ROOT` | Root directory for checkpoints | `/scratch/aaa_exchange/AWARE/WEIGHTS` |
| `PROVE_DATA_ROOT` | Root directory for datasets | `/scratch/aaa_exchange/AWARE/FINAL_SPLITS` |

## Custom Transforms

The test script automatically registers custom transforms required for some datasets:

- **ReduceToSingleChannel**: Converts 3-channel label PNGs to single channel
- **CityscapesLabelIdToTrainId**: Maps Cityscapes label IDs (0-33) to trainIds (0-18)
- **FWIoUMetric**: Custom metric for computing frequency-weighted IoU

## Troubleshooting

### Common Issues

**1. Checkpoint not found**
```bash
# Check if checkpoint exists
./test_unified.sh find --dataset ACDC --model deeplabv3plus_r50 --strategy baseline

# Use explicit checkpoint path
./test_unified.sh single --checkpoint /path/to/iter_80000.pth --dataset ACDC --model deeplabv3plus_r50
```

**2. Config file missing**
The test script looks for `configs/training_config.py` in the work directory. Ensure the training was run with the unified training pipeline which saves configs.

**3. CUDA out of memory**
Reduce batch size or use a GPU with more memory:
```bash
./test_unified.sh submit --dataset ACDC --model deeplabv3plus_r50 --gpu-mem 32G
```

**4. Custom transforms not registered**
The script imports `custom_transforms.py` automatically. Ensure you're running from the PROVE project directory.

## Integration with Training

The test script is designed to work seamlessly with checkpoints from `train_unified.sh`:

```bash
# Train model
./train_unified.sh single --dataset ACDC --model deeplabv3plus_r50 --strategy baseline

# Test trained model
./test_unified.sh single --dataset ACDC --model deeplabv3plus_r50 --strategy baseline
```

The checkpoint path is automatically inferred from the same directory structure used by training.

## See Also

- [UNIFIED_TRAINING.md](UNIFIED_TRAINING.md) - Training documentation
- [README_standard_augmentations.md](../tools/README_standard_augmentations.md) - Augmentation strategies
