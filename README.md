<p align="center">
  <img src="assets/PROVE.png" alt="PROVE Logo" width="400">
</p>

# PROVE: Pipeline for Recognition & Object Vision Evaluation

[![Project Page](https://img.shields.io/badge/Project-AWARE-blue)](https://aware.github.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{AWARE2026,
  title     = {TODO: Paper Title},
  author    = {TODO: Authors},
  journal   = {IEEE Access},
  year      = {2026},
  doi       = {TODO: DOI}
}
```

## Overview

PROVE (Pipeline for Recognition & Object Vision Evaluation) is a comprehensive, streamlined pipeline for training and testing semantic segmentation approaches using the MMSegmentation framework. This repository is part of the [AWARE project](https://aware.github.io), which evaluates domain adaptation through weather-based data augmentation.

**Related Repositories:**
- [PRISM](https://github.com/carhartt21/PRISM) - Image quality evaluation metrics pipeline
- [SWIFT](https://github.com/carhartt21/SWIFT) - Domain-stratified dataset preparation

## Features

- **Semantic Segmentation & Object Detection** with MMSegmentation/MMDetection
- **Two dataset layout modes**: standard (original datasets) and stratified (domain-split via [SWIFT](https://github.com/carhartt21/SWIFT))
- **Generative augmentation strategies**: cycleGAN, CUT, IP2P, Flux, and more
- **Standard augmentation strategies**: CutMix, MixUp, AutoAugment, RandAugment
- **Combined strategies**: mix generative + standard augmentations
- **Multi-dataset joint training** with automatic label unification
- **Per-domain/per-class evaluation** for weather robustness analysis
- **Batch job submission** for LSF clusters
- **Reproducible experiments** with config-driven approach

## Documentation

| Document | Description |
|----------|-------------|
| [Datasets](docs/DATASETS.md) | Dataset setup, layout modes, label formats, environment variables |
| [Training](docs/TRAINING.md) | Training commands, strategies, batch submission, LSF cluster |
| [Testing](docs/TESTING.md) | Evaluation, fine-grained testing, result analysis, visualization |
| [Advanced](docs/ADVANCED.md) | Configuration, performance optimization, troubleshooting, utilities |

## Installation

### Quick Setup with Mamba (Recommended)

```bash
# Clone repository
git clone https://github.com/carhartt21/PROVE.git
cd PROVE

# Create mamba environment
mamba env create -f environment.yml

# Activate environment
mamba activate prove

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import mmseg; print(f'MMSeg: {mmseg.__version__}')"
```

### Manual Installation

```bash
# Create environment
mamba create -n prove python=3.10 -y
mamba activate prove

# Install PyTorch with CUDA 11.8
mamba install pytorch=2.1.2 torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install MMCV with compiled extensions (requires conda/mamba)
mamba install -c conda-forge mmcv=2.1.0 -y

# Install MMEngine and OpenMMLab packages
pip install mmengine==0.10.7
pip install mmsegmentation==1.2.2 mmdet==3.3.0

# Install additional dependencies
pip install ftfy regex tqdm
```

**Tested Working Versions:** PyTorch 2.1.2, CUDA 11.8, MMCV 2.1.0, MMEngine 0.10.7, MMSegmentation 1.2.2, MMDetection 3.3.0

### Verify Installation

```bash
python -c "
import torch, mmcv, mmseg, mmdet
print('All imports successful!')
print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
print(f'MMCV: {mmcv.__version__}, MMSeg: {mmseg.__version__}, MMDet: {mmdet.__version__}')
"
```

## Repository Structure

```
PROVE/
├── prove.py                  # Main CLI entry point
├── unified_training.py       # Unified training pipeline
├── prove_config.py           # Configuration management
├── config_paths.py           # Path configuration (edit for your setup)
├── utils/                    # Utility modules
│   ├── custom_transforms.py  # Data augmentation transforms
│   ├── custom_losses.py      # Custom loss functions
│   ├── unified_datasets.py   # Dataset loaders
│   ├── mixed_dataloader.py   # Real/generated image mixing
│   └── label_unification.py  # Cross-dataset label mapping
├── tools/                    # Utility scripts
│   ├── fine_grained_test.py  # Per-domain/per-class evaluation
│   └── standard_augmentations.py
├── scripts/                  # Batch processing scripts
├── cityscapes_replication/   # Cityscapes training configs
└── docs/                     # Documentation
```

## Quick Start

### 1. Set Up Data

PROVE supports two dataset layout modes. See [Datasets](docs/DATASETS.md) for full details.

**Standard layout** (use datasets as-is):
```bash
export PROVE_DATA_ROOT=/path/to/your/data
# Or per-dataset: export PROVE_MAPILLARY_ROOT=/path/to/mapillary_vistas
```

**Stratified layout** (domain-split via [SWIFT](https://github.com/carhartt21/SWIFT)):
```bash
export PROVE_ROOT=/path/to/project/root
# Data expected at ${PROVE_ROOT}/FINAL_SPLITS/{train,test}/{images,labels}/{DATASET}/{condition}/
```

### 2. Train

```bash
# Standard layout - train on MapillaryVistas
python unified_training.py --dataset MapillaryVistas --model deeplabv3plus_r50 \
    --strategy baseline --dataset-layout standard

# Stratified layout - Stage 1: clear-day only
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 \
    --strategy baseline --domain-filter clear_day

# With generative augmentation
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 \
    --strategy gen_cycleGAN --real-gen-ratio 0.5

# Batch training for large-scale experiments
python scripts/batch_training_submission.py --stage 1 --dry-run
```

See [Training Guide](docs/TRAINING.md) for all options.

### 3. Test

```bash
# Fine-grained per-domain/per-class evaluation
python tools/fine_grained_test.py \
    --config /path/to/config.py \
    --checkpoint /path/to/checkpoint.pth \
    --dataset ACDC \
    --output-dir results/acdc_test

# Batch test submission
python scripts/batch_test_submission.py --stage 1 --dry-run
```

See [Testing Guide](docs/TESTING.md) for all options.

### 4. Analyze

```bash
# Comprehensive result analysis
python test_result_analyzer.py --all-insights

# Publication-quality visualizations
python test_result_visualizer.py --results-dir /path/to/results
```

## Supported Models

### Semantic Segmentation
- **DeepLabV3+**: `deeplabv3plus_r50`
- **PSPNet**: `pspnet_r50`
- **SegFormer**: `segformer_mit-b3`
- **SegNeXt**: `segnext_mscan-b`
- **HRNet**: `hrnet_hr48`
- **Mask2Former**: `mask2former_swin-b`

### Object Detection
- **Faster R-CNN**: `faster_rcnn_r50_fpn_1x`
- **YOLOX**: `yolox_l`, `yolox_m`, `yolox_s`
- **RTMDet**: `rtmdet_l`, `rtmdet_m`, `rtmdet_s`

## Supported Datasets

| Dataset | Classes | Label Format | Conditions |
|---------|---------|--------------|------------|
| Cityscapes | 19 | Native trainIds | Urban scenes |
| ACDC | 19 | Cityscapes train IDs | fog, night, rain, snow |
| BDD10k | 19 | Cityscapes train IDs | Various weather |
| IDD-AW | 19 | Cityscapes train IDs | Indian driving, adverse weather |
| MapillaryVistas | 66 / 19 | RGB-encoded | Global street-level |
| OUTSIDE15k | 24 / 19 | RGB-encoded | Outdoor scenes |

See [Datasets Guide](docs/DATASETS.md) for setup instructions and label handling details.

## Augmentation Strategies

| Type | Strategies |
|------|-----------|
| **Generative** | `gen_cycleGAN`, `gen_CUT`, `gen_IP2P`, `gen_flux_kontext`, `gen_step1x_new`, `gen_StyleID`, ... |
| **Standard** | `std_cutmix`, `std_mixup`, `std_autoaugment`, `std_randaugment`, `std_photometric_distort` |
| **Combined** | Any gen + std combination (e.g., `--strategy gen_cycleGAN --std-strategy std_cutmix`) |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `PROVE_ROOT` | Base directory for all project data |
| `PROVE_DATA_ROOT` | Stratified data splits (`${PROVE_ROOT}/FINAL_SPLITS`) |
| `PROVE_GEN_ROOT` | Generated images (`${PROVE_ROOT}/GENERATED_IMAGES`) |
| `PROVE_WEIGHTS_ROOT` | Model weights (`${PROVE_ROOT}/WEIGHTS`) |
| `PROVE_MAPILLARY_ROOT` | MapillaryVistas root (standard layout) |
| `PROVE_CITYSCAPES_ROOT` | Cityscapes root (standard layout) |
| `PROVE_BDD_ROOT` | BDD root (standard layout) |
| `PROVE_ACDC_ROOT` | ACDC root (standard layout) |

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Support

For questions and support:
- Create an issue on GitHub
- Check [Troubleshooting](docs/ADVANCED.md#troubleshooting)
- Review [MMSegmentation docs](https://mmsegmentation.readthedocs.io/)

## Acknowledgments

- Built on top of OpenMMLab's [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [MMDetection](https://github.com/open-mmlab/mmdetection) frameworks
- Domain stratification via [SWIFT](https://github.com/carhartt21/SWIFT)
