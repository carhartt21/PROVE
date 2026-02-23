# PROVE Unified Training System

This document describes the unified training configuration system for the PROVE pipeline. The system simplifies training by eliminating redundant config files and providing a single entry point for all training configurations.

## Overview

The unified training system consists of the following components:

1. **`unified_training_config.py`** - Configuration generator that creates MMSeg training configs from parameters
2. **`unified_training.py`** - Training orchestrator with CLI interface and LSF job submission
3. **`scripts/batch_training_submission.py`** - **Preferred** batch job submission (handles locks, pre-flight checks, parameters)
4. **`fine_grained_test.py`** - Per-domain/per-class evaluation with optimized inference
5. **`scripts/batch_test_submission.py`** - Batch test job submission

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
| ACDC | Segmentation | Cityscapes trainIds (19 classes) |
| BDD10k | Segmentation | Cityscapes trainIds (19 classes) |
| IDD-AW | Segmentation | Cityscapes trainIds (19 classes) |
| MapillaryVistas | Segmentation | RGB-encoded (66 native classes, `--use-native-classes`) |
| OUTSIDE15k | Segmentation | RGB-encoded (24 native classes, `--use-native-classes`) |
| Cityscapes | Segmentation | Native trainIds (19 classes) — pipeline verification only |

### Models (6 segmentation architectures)

| Model | Architecture | Backbone | Notes |
|-------|-------------|----------|-------|
| `deeplabv3plus_r50` | DeepLabV3+ | ResNet-50 | CNN, encoder-decoder with ASPP |
| `pspnet_r50` | PSPNet | ResNet-50 | CNN, pyramid pooling module |
| `segformer_mit-b3` | SegFormer | MiT-B3 | Transformer-based |
| `segnext_mscan-b` | SegNeXt | MSCAN-B | Transformer-like, multi-scale conv attention |
| `hrnet_hr48` | HRNet | HRNet-W48 | CNN, high-resolution representations |
| `mask2former_swin-b` | Mask2Former | Swin-B | Transformer, mask classification |

### Augmentation Strategies

| Strategy | Description |
|----------|-------------|
| **Baseline** | |
| `baseline` | No augmentation (control) |
| **Standard Augmentations (4)** | |
| `std_autoaugment` | AutoAugment augmentation |
| `std_cutmix` | CutMix augmentation |
| `std_mixup` | MixUp augmentation |
| `std_randaugment` | RandAugment augmentation |
| **Generative Strategies (21)** | |
| `gen_albumentations_weather` | Albumentations weather effects |
| `gen_Attribute_Hallucination` | Attribute hallucination |
| `gen_augmenters` | General augmenters |
| `gen_automold` | Automold weather simulation |
| `gen_CNetSeg` | ControlNet segmentation |
| `gen_CUT` | Contrastive Unpaired Translation |
| `gen_cyclediffusion` | Cycle diffusion |
| `gen_cycleGAN` | CycleGAN image translation |
| `gen_flux_kontext` | Flux Kontext editing |
| `gen_Img2Img` | Image-to-image diffusion |
| `gen_IP2P` | InstructPix2Pix |
| `gen_LANIT` | LANIT translation |
| `gen_Qwen_Image_Edit` | Qwen Image Edit |
| `gen_stargan_v2` | StarGAN v2 |
| `gen_step1x_new` | Step1X (new version) |
| `gen_step1x_v1p2` | Step1X v1.2 |
| `gen_SUSTechGAN` | SUSTechGAN |
| `gen_TSIT` | TSIT translation |
| `gen_UniControl` | UniControl |
| `gen_VisualCloze` | VisualCloze |
| `gen_Weather_Effect_Generator` | Weather effect generator |

**Note:** `std_photometric_distort` is applied to all strategies as default pipeline augmentation and is not counted as a separate strategy.

### Real-to-Generated Ratio

The `--real-gen-ratio` parameter controls the proportion of real vs. generated images in training:

- `1.0` - 100% real images (default)
- `0.75` - 75% real, 25% generated
- `0.5` - 50% real, 50% generated
- `0.25` - 25% real, 75% generated
- `0.0` - 100% generated images

### Domain Filtering

The `--domain-filter` parameter allows training on a specific weather/lighting domain within a dataset:

```bash
# Train only on clear day images from ACDC
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy baseline --domain-filter clear_day

# Train only on foggy images
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy baseline --domain-filter foggy
```

**Available domains by dataset:**

| Dataset | Available Domains |
|---------|-------------------|
| ACDC | `clear_day`, `cloudy`, `dawn_dusk`, `foggy`, `night`, `rainy`, `snowy` |
| BDD10k | `clear`, `overcast`, `partly_cloudy`, `rainy`, `snowy`, `foggy` |
| BDD100k | `clear`, `overcast`, `partly_cloudy`, `rainy`, `snowy`, `foggy`, `undefined` |
| IDD-AW | `clear`, `foggy`, `rainy` |
| MapillaryVistas | (no domain split) |
| OUTSIDE15k | (no domain split) |

**Use cases for domain filtering:**
- Domain-specific model training
- Ablation studies on weather robustness
- Understanding per-domain model performance
- Creating domain-specific baselines

## Mixed Dataloader System

The mixed dataloader injects generated images into the training `data_list` at runtime. For gen_* strategies, both real and generated images are combined based on the `--real-gen-ratio` parameter.

**How it works:**
1. Training script is generated by `unified_training.py` with `_generate_mixed_training_script()`
2. Generated images from the manifest CSV are injected into `data_list` before training
3. `serialize_data=False` is set to allow runtime modification of `data_list`
4. The ratio of real:generated samples is controlled by `--real-gen-ratio` (default 0.5)

> **Historical note:** The original MixedDataLoader was found to be non-functional (Jan 28, 2026). See [BUG_REPORT](BUG_REPORT_CROSS_DATASET_CONTAMINATION.md). The current implementation injects generated image paths directly into the dataset's `data_list`.

## Job Submission

### LSF (Load Sharing Facility) — Single Job

```bash
# Submit a single job to LSF
python unified_training.py --dataset BDD10k --model deeplabv3plus_r50 \
    --strategy gen_cycleGAN --domain-filter clear_day --submit-job
```

## Batch Job Submission (Preferred)

Use `batch_training_submission.py` for production training (handles locks, pre-flight checks, parameters):

```bash
# Stage 1: Clear-day training (ALWAYS dry-run first!)
python scripts/batch_training_submission.py --stage 1 --dry-run
python scripts/batch_training_submission.py --stage 1 --strategies baseline

# Stage 2: All-domains training
python scripts/batch_training_submission.py --stage 2 --dry-run

# Cityscapes pipeline verification
python scripts/batch_training_submission.py --stage cityscapes --dry-run

# Filter by dataset/model
python scripts/batch_training_submission.py --stage 1 --datasets BDD10k --models deeplabv3plus_r50

# Resume interrupted training
python scripts/batch_training_submission.py --stage 1 --resume --dry-run

# Limit number of jobs
python scripts/batch_training_submission.py --stage 1 --limit 10
```

## Directory Structure

After training, outputs are organized as:

```
${AWARE_DATA_ROOT}/WEIGHTS/
├── baseline/
│   ├── bdd10k/
│   │   ├── deeplabv3plus_r50/
│   │   ├── pspnet_r50/
│   │   ├── segformer_mit-b3/
│   │   ├── segnext_mscan-b/
│   │   ├── hrnet_hr48/
│   │   └── mask2former_swin-b/
│   └── ...
├── gen_cycleGAN/
│   ├── bdd10k/
│   │   ├── pspnet_r50/
│   │   ├── segformer_mit-b3/
│   │   ├── segnext_mscan-b/
│   │   └── mask2former_swin-b/
│   └── ...
└── ...
```

## Environment Variables

The system respects the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `PROVE_DATA_ROOT` | Root directory for datasets | `${AWARE_DATA_ROOT}/FINAL_SPLITS` |
| `PROVE_GEN_ROOT` | Root directory for generated images | `${AWARE_DATA_ROOT}/GENERATED_IMAGES` |
| `PROVE_WEIGHTS_ROOT` | Root directory for model weights | `${AWARE_DATA_ROOT}/WEIGHTS` |
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
   - Use smaller model: Try `pspnet_r50` instead of `segformer_mit-b3`

### Debug Mode

```bash
# Print config without training
python unified_training.py --dataset BDD10k --model deeplabv3plus_r50 --config-only --print

# Dry run batch submission
python scripts/batch_training_submission.py --stage 1 --dry-run
```

## Contributing

To add a new model or dataset:

1. Add entry to `DATASET_CONFIGS` or `SEGMENTATION_MODELS` in `unified_training_config.py`
2. Test with `--config-only` flag first
3. Update this documentation
