# PROVE Unified Training System

This document describes the unified training configuration system for the PROVE pipeline. The system simplifies training by eliminating redundant config files and providing a single entry point for all training configurations.

## Overview

The unified training system consists of three main components:

1. **`unified_training_config.py`** - Configuration generator that creates training configs from parameters
2. **`mixed_dataloader.py`** - Dual dataloader system for mixing real and generated images
3. **`unified_training.py`** - Training orchestrator with CLI interface

## Quick Start

### Train a Single Model

```bash
# Baseline training (no augmentation)
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy baseline

# With cycleGAN augmentation
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN

# Mixed training (50% real, 50% generated)
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN --real-gen-ratio 0.5
```

### Generate Config Only

```bash
# Generate config file without training
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN --config-only
```

### Batch Training

```bash
# Train multiple configurations
python unified_training.py --batch --datasets ACDC BDD10k --strategies baseline gen_cycleGAN --parallel
```

## Configuration Parameters

### Datasets

| Dataset | Task | Format |
|---------|------|--------|
| ACDC | Segmentation | Cityscapes |
| BDD10k | Segmentation | Cityscapes |
| BDD100k | Detection | COCO JSON |
| IDD-AW | Segmentation | Cityscapes |
| MapillaryVistas | Segmentation | Mapillary |
| OUTSIDE15k | Segmentation | Cityscapes |

### Models

**Segmentation Models:**
- `deeplabv3plus_r50` - DeepLabV3+ with ResNet-50
- `pspnet_r50` - PSPNet with ResNet-50
- `segformer_mit-b5` - SegFormer with MiT-B5

**Detection Models:**
- `faster_rcnn_r50_fpn_1x` - Faster R-CNN with ResNet-50 FPN
- `yolox_l` - YOLOX Large
- `rtmdet_l` - RTMDet Large

### Augmentation Strategies

| Strategy | Description |
|----------|-------------|
| `baseline` | No augmentation |
| `photometric_distort` | Random brightness, contrast, saturation |
| `gen_cycleGAN` | Generated images from CycleGAN |
| `gen_CUT` | Generated images from CUT |
| `gen_stargan_v2` | Generated images from StarGAN v2 |
| ... | (and more generative models) |

### Real-to-Generated Ratio

The `--real-gen-ratio` parameter controls the proportion of real vs. generated images in training:

- `1.0` - 100% real images (default)
- `0.75` - 75% real, 25% generated
- `0.5` - 50% real, 50% generated
- `0.25` - 25% real, 75% generated
- `0.0` - 100% generated images

## Mixed Dataloader System

The mixed dataloader enables fine-grained control over training data composition.

### Sampling Strategies

1. **`batch_split`** (recommended) - Each batch contains a fixed number of real and generated samples
2. **`ratio`** - Probabilistically selects from real or generated dataset per sample
3. **`alternating`** - Alternates between real and generated batches

### Example Configuration

```python
from mixed_dataloader import MixedDataLoader

# Create mixed dataloader with 50% real, 50% generated
loader = MixedDataLoader(
    real_dataset=real_dataset,
    generated_dataset=gen_dataset,
    real_gen_ratio=0.5,
    batch_size=4,
    sampling_strategy='batch_split'
)

# Each batch will have 2 real and 2 generated samples
print(loader.get_batch_composition())
# {'total': 4, 'real': 2, 'generated': 2}
```

## Python API

### UnifiedTrainingConfig

```python
from unified_training_config import UnifiedTrainingConfig

# Initialize config builder
config_builder = UnifiedTrainingConfig()

# Build configuration
config = config_builder.build(
    dataset='ACDC',
    model='deeplabv3plus_r50',
    strategy='gen_cycleGAN',
    real_gen_ratio=0.5,
)

# Save to file
config_builder.save_config(config, 'my_config.py')
```

### UnifiedTrainer

```python
from unified_training import UnifiedTrainer

# Create trainer
trainer = UnifiedTrainer(
    dataset='ACDC',
    model='deeplabv3plus_r50',
    strategy='gen_cycleGAN',
    real_gen_ratio=0.5,
)

# Get training summary
print(trainer.get_training_summary())

# Run training
trainer.train()
```

## Job Submission

### LSF (Load Sharing Facility)

```bash
# Generate LSF job script
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN --generate-job-script lsf

# Submit job
bsub < job_ACDC_deeplabv3plus_r50_gen_cycleGAN.sh
```

### SLURM

```bash
# Generate SLURM job script
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN --generate-job-script slurm

# Submit job
sbatch job_ACDC_deeplabv3plus_r50_gen_cycleGAN.sh
```

## Shell Helper Script

The `train_unified.sh` script provides convenient shortcuts:

```bash
# Single training
./train_unified.sh single --dataset ACDC --model deeplabv3plus_r50 --strategy baseline

# Batch training
./train_unified.sh batch --datasets ACDC BDD10k --strategy gen_cycleGAN

# Ratio experiment
./train_unified.sh ratio-exp --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN

# Generate configs
./train_unified.sh generate --strategy baseline --all

# List options
./train_unified.sh list
```

## Directory Structure

After training, outputs are organized as:

```
/scratch/aaa_exchange/AWARE/WEIGHTS/
├── baseline/
│   ├── acdc/
│   │   ├── deeplabv3plus_r50/
│   │   ├── pspnet_r50/
│   │   └── segformer_mit-b5/
│   └── ...
├── gen_cycleGAN/
│   ├── acdc/
│   │   ├── deeplabv3plus_r50/
│   │   ├── deeplabv3plus_r50_ratio0p50/  # With 50% real ratio
│   │   └── ...
│   └── ...
└── ...
```

## Environment Variables

The system respects the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `PROVE_DATA_ROOT` | Root directory for datasets | `/scratch/aaa_exchange/AWARE/FINAL_SPLITS` |
| `PROVE_GEN_ROOT` | Root directory for generated images | `/scratch/aaa_exchange/AWARE/GENERATED_IMAGES` |
| `PROVE_WEIGHTS_ROOT` | Root directory for model weights | `/scratch/aaa_exchange/AWARE/WEIGHTS` |
| `PROVE_CONFIG_ROOT` | Root directory for config files | `./multi_model_configs` |

## Migration from Old Config System

If you have existing training scripts using the old config files:

### Old Way (deprecated)
```bash
python tools/train.py multi_model_configs/gen_cycleGAN/ACDC/acdc_deeplabv3plus_r50_config.py
```

### New Way (recommended)
```bash
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy gen_cycleGAN
```

The new system automatically generates equivalent configurations on-the-fly.

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'mmengine'**
   - Install mmsegmentation: `pip install mmsegmentation`

2. **Config not found errors**
   - Ensure `PROVE_DATA_ROOT` points to your data directory
   - Check that the dataset exists

3. **CUDA out of memory**
   - Reduce batch size: Add `--batch-size 1` (requires modifying config)
   - Use smaller model: Try `pspnet_r50` instead of `segformer_mit-b5`

### Debug Mode

```bash
# Print config without training
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --config-only --print

# Dry run batch training
python unified_training.py --batch --datasets ACDC --strategies baseline --dry-run
```

## Contributing

To add a new model or dataset:

1. Add entry to `DATASET_CONFIGS` or `SEGMENTATION_MODELS`/`DETECTION_MODELS` in `unified_training_config.py`
2. Test with `--config-only` flag first
3. Update this documentation
