# Training Guide

This document covers all aspects of training with PROVE, including the unified training system, batch submission, combined strategies, and cluster job management.

**Quick links:** [Datasets](DATASETS.md) | [Testing](TESTING.md) | [Advanced](ADVANCED.md) | [README](../README.md)

---

## Unified Training System

PROVE includes a unified training system that simplifies configuration management and supports mixed real/generated image training:

```bash
# Basic training with baseline (no augmentation)
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy baseline

# Training on specific domain (e.g., clear weather only)
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy baseline --domain-filter clear_day

# Training with generated image augmentation (cycleGAN)
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN

# Mixed training with 50% real, 50% generated images
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN --real-gen-ratio 0.5

# Training with standard augmentation (CutMix, MixUp, etc.)
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy std_cutmix

# Combined strategies: generative augmentation + standard augmentation
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN --std-strategy std_cutmix

# Specify custom cache directory for pretrained weights
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy baseline --cache-dir /path/to/cache

# Batch training for multiple configurations
python unified_training.py --batch --datasets ACDC BDD10k --strategies baseline gen_cycleGAN

# Batch training for all segmentation datasets and models
python unified_training.py --batch --all-seg-datasets --all-seg-models --strategies baseline

# Batch training for all detection datasets and models
python unified_training.py --batch --all-det-datasets --all-det-models --strategies baseline

# Dry run to preview batch training commands
python unified_training.py --batch --all-seg-datasets --all-seg-models --dry-run

# List all available options
python unified_training.py --list
```

## Training Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | Dataset name (ACDC, BDD10k, BDD100k, IDD-AW, MapillaryVistas, OUTSIDE15k) | Required |
| `--model` | Model name (deeplabv3plus_r50, pspnet_r50, segformer_mit-b5, segnext_mscan-b, etc.) | Required |
| `--strategy` | Main augmentation strategy (baseline, std_cutmix, gen_cycleGAN, etc.) | baseline |
| `--std-strategy` | Standard augmentation to combine with main strategy (see Combined Strategies) | None |
| `--real-gen-ratio` | Ratio of real to generated images (0.0 to 1.0) | 1.0 |
| `--domain-filter` | Filter training data to specific domain (e.g., clear_day for Stage 1 training) | None |
| `--dataset-layout` | Dataset directory layout: `standard` (original) or `stratified` (SWIFT domain-split) | stratified |
| `--work-dir` | Output directory for checkpoints and logs | Auto-generated |
| `--cache-dir` | Directory for caching pretrained weights and checkpoints | ~/.cache/torch |
| `--load-from` | Path to pretrained weights to initialize model | None |
| `--resume-from` | Path to checkpoint to resume training from | None |
| `--max-iters` | Maximum training iterations | 15000 (seg) / 10000 (det) |
| `--checkpoint-interval` | Save checkpoint every N iterations | 2000 |
| `--eval-interval` | Run validation every N iterations | 2000 |
| `--batch-size` | Training batch size (LR auto-scales with linear scaling rule) | 16 |
| `--no-early-stop` | Disable early stopping (stops when no improvement for 5 validations) | Enabled |
| `--early-stop-patience` | Number of validations without improvement before stopping | 5 |
| `--use-native-classes` | Use native labels (66 for Mapillary, 24 for OUTSIDE15k) instead of Cityscapes 19 | False |
| `--aux-loss` | Auxiliary loss to add alongside CrossEntropyLoss (`focal`, `lovasz`, `boundary`) | None |
| `--save-val-predictions` | Save validation visualizations (Input \| GT \| Prediction side-by-side) | False |
| `--max-val-samples` | Maximum number of samples to visualize per validation epoch | 5 |

## Important Training Modes

```bash
# Stage 1 Training: Clear-day only (domain-filtered)
python unified_training.py --dataset BDD10k --model segformer_mit-b5 \
    --strategy gen_cycleGAN --domain-filter clear_day

# Stage 2 Training: All domains  
python unified_training.py --dataset BDD10k --model segformer_mit-b5 \
    --strategy gen_cycleGAN

# Extended Training: Longer training with no early stopping
python unified_training.py --dataset BDD10k --model segformer_mit-b5 \
    --strategy gen_cycleGAN --max-iters 320000 --no-early-stop

# Resume from checkpoint
python unified_training.py --dataset BDD10k --model segformer_mit-b5 \
    --strategy gen_cycleGAN --resume-from /path/to/iter_80000.pth --max-iters 160000

# Training with auxiliary loss (CE remains primary)
python unified_training.py --dataset BDD10k --model deeplabv3plus_r50 \
    --strategy baseline --aux-loss focal

# Training with validation visualization (saves Input | GT | Prediction images)
python unified_training.py --dataset BDD10k --model deeplabv3plus_r50 \
    --strategy baseline --save-val-predictions --max-val-samples 10
```

## Combined Strategies

You can combine standard augmentation strategies (std_*) with generative augmentation strategies (gen_*) or baseline using the `--std-strategy` option. This enables applying both types of augmentation during training.

**Available Standard Augmentations for `--std-strategy`:**
- `std_cutmix` - CutMix augmentation
- `std_mixup` - MixUp augmentation
- `std_autoaugment` - AutoAugment policy
- `std_randaugment` - RandAugment augmentation

**Usage Examples:**

```bash
# Combine generative augmentation (cycleGAN) with CutMix
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 \
    --strategy gen_cycleGAN --std-strategy std_cutmix

# Combine baseline training with MixUp augmentation
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 \
    --strategy baseline --std-strategy std_mixup

# Multi-dataset training with combined strategies
python unified_training.py --multi-dataset --datasets ACDC MapillaryVistas \
    --model deeplabv3plus_r50 --strategy gen_CUT --std-strategy std_autoaugment
```

**Output Directory Structure:**
When using combined strategies, the work directory includes both strategy names:
- `gen_cycleGAN+std_cutmix/acdc/deeplabv3plus_r50/`
- `baseline+std_mixup/acdc/deeplabv3plus_r50/`

**Batch Combination Training:**

For systematic ablation studies combining top gen and std strategies:

```bash
# List all combinations (top 3 gen × top 3 std × datasets × models)
./scripts/submit_combination_training.sh --list

# Preview bsub commands without submitting
./scripts/submit_combination_training.sh --dry-run

# Submit all combination training jobs
./scripts/submit_combination_training.sh

# Submit with limit
./scripts/submit_combination_training.sh --limit 10
```

**Default Configuration:**
- Gen strategies: gen_cyclediffusion, gen_TSIT, gen_cycleGAN
- Std strategies: std_randaugment, std_mixup, std_cutmix
- Datasets: MapillaryVistas, IDD-AW
- Models: SegFormer, PSPNet
- Total: 36 combinations (3 × 3 × 2 × 2)

**Shell Script Usage:**
```bash
# Using train_unified.sh
./scripts/train_unified.sh single --dataset ACDC --model deeplabv3plus_r50 \
    --strategy gen_cycleGAN --std-strategy std_cutmix

# Submit to LSF cluster
./scripts/train_unified.sh submit --dataset ACDC --model deeplabv3plus_r50 \
    --strategy gen_cycleGAN --std-strategy std_mixup
```

## Early Stopping

Early stopping is enabled by default to prevent overfitting and save training time. It monitors:
- **Segmentation**: `val/mIoU` (validation mean Intersection over Union)
- **Detection**: `coco/bbox_mAP` (COCO bounding box mean Average Precision)

Training stops when the monitored metric doesn't improve by at least 0.001 for 5 consecutive validation steps.

```bash
# Disable early stopping
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --no-early-stop

# Custom patience (stop after 10 validations without improvement)
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --early-stop-patience 10
```

## Evaluation Metrics

For segmentation tasks, the following metrics are computed during validation and testing:

**Primary Metric: mIoU (Mean Intersection over Union)**
- Standard metric giving equal weight to all classes
- **Recommended for domain robustness analysis**
- Not biased by class frequency distribution

**Secondary Metric: fwIoU (Frequency Weighted IoU)**
- Weights each class IoU by its pixel frequency: `fwIoU = Σ(freq_i × IoU_i)`
- **NOT recommended for cross-domain comparison** (see note below)
- Useful for overall scene understanding when class distribution is relevant

**⚠️ Important: Why mIoU for Domain Analysis**

Our analysis revealed that fwIoU can be misleading for domain gap analysis:
- Adverse weather (fog, rain) naturally occludes small objects (people, cyclists)
- This shifts class distribution toward "easy" classes (road, sky)
- fwIoU artificially inflates performance on adverse domains
- **Example**: Foggy images showed +13% higher fwIoU than clear_day, but only -8.5% lower mIoU

**Baseline Domain Gap (Clear_Day Trained Models, mIoU)**:
| Condition | mIoU | Gap |
|-----------|------|-----|
| Normal (clear_day, cloudy) | 54.96% | - |
| Adverse (foggy, rainy, snowy, night) | 47.49% | -7.46% |

*Data filtered to domains with ≥50 test images for reliability.*

## Batch Training Options

| Option | Description |
|--------|-------------|
| `--batch` | Enable batch training mode |
| `--datasets` | List of datasets for batch training |
| `--models` | List of models for batch training |
| `--all-seg-datasets` | Use all segmentation datasets (ACDC, BDD10k, IDD-AW, MapillaryVistas, OUTSIDE15k) |
| `--all-det-datasets` | Use all detection datasets (BDD100k) |
| `--all-seg-models` | Use all segmentation models (deeplabv3plus_r50, pspnet_r50, segformer_mit-b5) |
| `--all-det-models` | Use all detection models (faster_rcnn_r50_fpn_1x, yolox_l, rtmdet_l) |
| `--strategies` | List of augmentation strategies for batch training |
| `--ratios` | List of real-to-generated ratios for batch training |
| `--parallel` | Run batch jobs in parallel |
| `--dry-run` | Preview commands without executing |

## Multi-Dataset Joint Training

Train on multiple datasets simultaneously (e.g., ACDC + MapillaryVistas) with automatic label unification:

```bash
# Joint training on ACDC + Mapillary (labels unified automatically)
./scripts/train_unified.sh single-multi --datasets ACDC MapillaryVistas --model deeplabv3plus_r50

# With custom sampling weights (70% ACDC, 30% Mapillary)
./scripts/train_unified.sh single-multi --datasets ACDC MapillaryVistas --weights 0.7 0.3 --model deeplabv3plus_r50

# Generate config only (no training)
./scripts/train_unified.sh single-multi --datasets ACDC MapillaryVistas --model deeplabv3plus_r50 --config-only

# Python CLI alternative
python unified_training.py --multi-dataset --datasets ACDC MapillaryVistas --model deeplabv3plus_r50
```

**How Label Unification Works:**
- Datasets using Cityscapes format (ACDC, BDD10k, etc.) use labels as-is
- Mapillary datasets automatically get `MapillaryLabelTransform` applied, mapping 66 classes to 19 Cityscapes trainIDs
- All datasets are combined using `ConcatDataset` with configurable sampling weights

**Multi-Dataset Options:**

| Option | Description |
|--------|-------------|
| `--datasets` | Space-separated list of datasets to train jointly |
| `--weights` | Optional sampling weights per dataset (must sum to 1.0) |
| `--model` | Model architecture to use |
| `--strategy` | Augmentation strategy (default: baseline) |
| `--config-only` | Generate config without training |

## Using train_unified.sh (Alternative)

```bash
# Single training run
bash train_unified.sh single --dataset ACDC --model deeplabv3plus_r50 --strategy baseline

# With domain filter
bash train_unified.sh single --dataset ACDC --model deeplabv3plus_r50 --strategy baseline --domain-filter clear_day

# Multi-dataset training
bash train_unified.sh single-multi --datasets ACDC MapillaryVistas --model deeplabv3plus_r50

# Batch training for all segmentation
bash train_unified.sh batch --all-seg-datasets --all-seg-models --strategy baseline --dry-run
```

## LSF Cluster Submission

Submit training jobs directly to LSF cluster:

```bash
# Submit single job
bash train_unified.sh submit --dataset ACDC --model deeplabv3plus_r50 --strategy baseline

# Submit with custom queue and GPU memory
bash train_unified.sh submit --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN \
    --queue BatchGPU --gpu-mem 32G --num-cpus 8

# Preview bsub command without submitting (dry run)
bash train_unified.sh submit --dataset ACDC --model deeplabv3plus_r50 --strategy baseline --dry-run
```

**LSF Submit Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--queue` | BatchGPU | LSF queue name |
| `--gpu-mem` | 24G | GPU memory requirement |
| `--num-cpus` | 8 | Number of CPUs per job |
| `--dry-run` | - | Show bsub command without executing |

See [UNIFIED_TRAINING.md](UNIFIED_TRAINING.md) for comprehensive documentation.

## Batch Training Submission (Preferred for Large-Scale Experiments)

For systematic experiments across multiple configurations, use the batch submission script:

```bash
# Stage 1: Train on clear-day domain only (cross-domain robustness evaluation)
python scripts/batch_training_submission.py --stage 1 --dry-run  # Preview jobs
python scripts/batch_training_submission.py --stage 1 -y         # Submit all

# Stage 2: Train on all domains (domain-inclusive training)
python scripts/batch_training_submission.py --stage 2 --dry-run

# Cityscapes: Pipeline verification (160k iterations, all 5 models)
python scripts/batch_training_submission.py --stage cityscapes --dry-run

# Filter by dataset, model, or strategy
python scripts/batch_training_submission.py --stage 1 --datasets BDD10k IDD-AW \
    --models segformer_mit-b3 --strategies baseline --dry-run

# Custom training duration with frequent checkpoints
python scripts/batch_training_submission.py --stage 1 --max-iters 20000 \
    --checkpoint-interval 2000 --eval-interval 2000 --dry-run
```

**Batch Submission Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--stage` | Training stage (1, 2, cityscapes, ratio, extended, combinations) | Required |
| `--datasets` | List of datasets to train on | All for stage |
| `--models` | List of models to train | All 5 models |
| `--strategies` | List of augmentation strategies | baseline |
| `--ratios` | Real/gen ratios for generative strategies | 0.5 |
| `--max-iters` | Maximum training iterations | 15k (20k for Cityscapes) |
| `--checkpoint-interval` | Save checkpoint every N iterations | 2000 |
| `--eval-interval` | Run validation every N iterations | 2000 |
| `--aux-loss` | Auxiliary loss (focal, lovasz, boundary) | None |
| `--limit` | Maximum number of jobs to submit | None |
| `--dry-run` | Preview jobs without submitting | False |
| `-y, --yes` | Skip confirmation prompt | False |

**Available Stages:**
| Stage | Domain Filter | Output Directory | Purpose |
|-------|--------------|------------------|---------|
| `1` | `clear_day` | `WEIGHTS/` | Train clear-only, test cross-domain |
| `2` | None (all) | `WEIGHTS_STAGE_2/` | Train all conditions |
| `cityscapes` | None | `WEIGHTS_CITYSCAPES/` | Pipeline verification on standard benchmark |

## Domain Adaptation Ablation

Evaluate cross-dataset domain adaptation by testing models trained on BDD10k/IDD-AW/MapillaryVistas on:
- **Cityscapes** (clear_day condition)
- **ACDC** (adverse weather: foggy, night, rainy, snowy)

### Model Variants

Two training configurations are compared:
1. **Full Dataset Models** - Trained on all available data
2. **Clear Day Baseline Models** (`_clear_day` suffix) - Trained only on clear_day subset

### Job Submission

```bash
# Submit all 18 evaluation jobs (9 full + 9 clear_day baseline)
./scripts/submit_domain_adaptation_ablation.sh --all

# Submit only full dataset models (9 jobs)
./scripts/submit_domain_adaptation_ablation.sh --all-full

# Submit only clear_day baseline models (9 jobs)
./scripts/submit_domain_adaptation_ablation.sh --all-clear-day

# List available checkpoints
./scripts/submit_domain_adaptation_ablation.sh --list
```

See [DOMAIN_ADAPTATION_ABLATION.md](DOMAIN_ADAPTATION_ABLATION.md) for comprehensive documentation.

## Extended Training and Trajectory Testing

### Extended Training Ablation

Investigate model performance beyond the standard 80,000 iterations (e.g., 160k, 240k, 320k).

```bash
# Submit extended training jobs (resumes from 80k)
./scripts/submit_extended_training.sh --max-iters 160000
```

See [EXTENDED_TRAINING.md](EXTENDED_TRAINING.md) for more details.

### Automated Trajectory Testing

Evaluate performance across the entire training trajectory by testing all available checkpoints sequentially.

```bash
# Submit sequential tests for all checkpoints of a strategy
./scripts/submit_all_tests.sh gen_LANIT
```

**Features:**
- Automatically detects all `iter_*.pth` checkpoints.
- Groups tests by model to run **sequentially** in a single LSF job.
- Optimized for cluster usage to minimize queue length.
- Results are saved to `test_results/` within each model's directory.
