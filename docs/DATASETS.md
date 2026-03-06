# Dataset Guide

This document covers dataset setup, layout modes, label formats, label unification, and supported datasets in PROVE.

**Quick links:** [Training](TRAINING.md) | [Testing](TESTING.md) | [Advanced](ADVANCED.md) | [README](../README.md)

---

## Dataset Layouts

PROVE supports two dataset layout modes:

| Layout | Description | Use Case |
|--------|-------------|----------|
| **`standard`** | Original dataset directory structure | Using datasets as downloaded (MapillaryVistas, Cityscapes, BDD, ACDC) |
| **`stratified`** | Domain-stratified layout from [SWIFT](https://github.com/carhartt21/SWIFT) | Per-domain weather evaluation, domain filtering |

## Standard Layout (No Domain Stratification Required)

You can use PROVE directly with standard datasets without any domain stratification preprocessing. Set the dataset root via environment variable and use `--dataset-layout standard`:

```bash
# Set your data root (parent directory containing dataset folders)
export PROVE_DATA_ROOT=/path/to/your/data

# Or set per-dataset roots
export PROVE_MAPILLARY_ROOT=/path/to/mapillary_vistas
export PROVE_CITYSCAPES_ROOT=/path/to/cityscapes
export PROVE_BDD_ROOT=/path/to/bdd100k
export PROVE_ACDC_ROOT=/path/to/ACDC
```

### Expected Directory Structures

```
# MapillaryVistas (v2.0)
mapillary_vistas/
├── training/
│   ├── images/          # .jpg images
│   └── v2.0/
│       └── labels/      # .png label masks
└── validation/
    ├── images/
    └── v2.0/
        └── labels/

# Cityscapes
cityscapes/
├── leftImg8bit/
│   ├── train/{city}/    # *_leftImg8bit.png
│   └── val/{city}/
└── gtFine/
    ├── train/{city}/    # *_gtFine_labelTrainIds.png
    └── val/{city}/

# BDD10k
bdd100k/
├── images/10k/
│   ├── train/           # .jpg images
│   └── val/
└── labels/sem_seg/masks/
    ├── train/           # .png label masks
    └── val/

# ACDC
ACDC/
├── rgb_anon/
│   ├── train/           # .png images (may have subdirectories)
│   └── val/
└── gt/
    ├── train/           # .png label masks
    └── val/
```

### Training Examples (Standard Layout)

**MapillaryVistas:**

```bash
# Train with native 66 classes
python unified_training.py \
    --dataset MapillaryVistas \
    --model deeplabv3plus_r50 \
    --strategy baseline \
    --dataset-layout standard

# Train with Cityscapes 19-class mapping
python unified_training.py \
    --dataset MapillaryVistas \
    --model deeplabv3plus_r50 \
    --strategy baseline \
    --dataset-layout standard \
    --no-native-classes
```

**Cityscapes:**

```bash
python unified_training.py \
    --dataset Cityscapes \
    --model segformer_mit-b3 \
    --strategy baseline \
    --dataset-layout standard
```

**Testing with standard layout:**

```bash
python tools/fine_grained_test.py \
    --config /path/to/config.py \
    --checkpoint /path/to/checkpoint.pth \
    --dataset MapillaryVistas \
    --data-root /path/to/mapillary_vistas \
    --output-dir results/mapillary_test \
    --dataset-layout standard
```

> **Note:** Per-domain weather evaluation (e.g., separate metrics for foggy, rainy, night) requires domain-stratified data. Use [SWIFT](https://github.com/carhartt21/SWIFT) to create domain-stratified splits from standard datasets, then use `--dataset-layout stratified` (the default).

## Stratified Layout (Domain-Stratified via SWIFT)

For weather-domain evaluation, PROVE uses domain-stratified datasets created by [SWIFT](https://github.com/carhartt21/SWIFT). This is the default layout (`--dataset-layout stratified`).

```
FINAL_SPLITS/
├── train/
│   ├── images/{DATASET}/{condition}/   # e.g., ACDC/clear_day/, ACDC/foggy/
│   └── labels/{DATASET}/{condition}/
└── test/
    ├── images/{DATASET}/{condition}/
    └── labels/{DATASET}/{condition}/
```

With stratified data, you can train on specific domains and evaluate per-domain robustness:

```bash
# Stage 1: Train only on clear-day images (cross-domain robustness)
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 \
    --strategy baseline --domain-filter clear_day

# Stage 2: Train on all domains
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 \
    --strategy baseline
```

### Stratified Directory Structure

```
FINAL_SPLITS/
├── train/
│   ├── images/
│   │   ├── ACDC/
│   │   │   ├── clear_day/
│   │   │   ├── fog/
│   │   │   ├── night/
│   │   │   ├── rainy/
│   │   │   └── snowy/
│   │   ├── BDD10k/
│   │   ├── BDD100k/
│   │   ├── IDD-AW/
│   │   ├── MapillaryVistas/
│   │   └── OUTSIDE15k/
│   └── labels/
│       ├── ACDC/
│       │   ├── clear_day/
│       │   └── ...
│       └── ...
└── test/
    ├── images/
    └── labels/
```

**Label Format Notes:**
- Labels are stored as PNG images with Cityscapes label IDs (0-33)
- The pipeline automatically converts to trainIds (0-18) using `CityscapesLabelIdToTrainId` transform
- 3-channel label PNGs are supported via `ReduceToSingleChannel` transform

## Supported Datasets

| Dataset | Task | Classes | Label Format | Conditions |
|---------|------|---------|--------------|------------|
| ACDC | Segmentation | 19 | Cityscapes train IDs | clear_day, fog, night, rain, snow |
| BDD10k | Segmentation | 19 | Cityscapes train IDs | Various weather conditions |
| BDD100k | Segmentation/Detection | 19 | Cityscapes train IDs | Large-scale diverse conditions |
| IDD-AW | Segmentation | 19 | Cityscapes train IDs | Indian driving with adverse weather |
| MapillaryVistas | Segmentation | 66 (native) / 19 (mapped) | RGB-encoded | Global street-level imagery |
| OUTSIDE15k | Segmentation | 24 (native) / 19 (mapped) | RGB-encoded | Outdoor scenes |
| Cityscapes | Segmentation | 19 | Native trainIds | Urban street scenes |

## Label Unification System

PROVE includes a comprehensive label unification module (`label_unification.py`) that enables seamless joint training of Cityscapes and Mapillary Vistas datasets.

### Label Space Options

1. **Cityscapes (19 classes)**: Maps Mapillary's 66 classes to Cityscapes' 19 evaluation classes
   - Best for: Benchmarking on Cityscapes test set
   - Pros: Compatible with existing Cityscapes benchmarks
   - Cons: Some Mapillary-specific classes are merged or ignored

2. **Unified (42 classes)**: Extended label space preserving more semantic granularity
   - Best for: Maximum information preservation during training
   - Pros: Retains more fine-grained distinctions from both datasets
   - Cons: Requires mapping back to Cityscapes for standard benchmarking

### Class Mapping Overview

| Unified Class | Cityscapes | Mapillary Sources |
|--------------|------------|-------------------|
| road | road | Road, Bike Lane, Service Lane |
| sidewalk | sidewalk | Sidewalk, Curb Cut, Pedestrian Area |
| building | building | Building |
| wall | wall | Wall, Barrier |
| fence | fence | Fence |
| pole | pole | Pole, Traffic Sign Frame |
| traffic light | traffic light | Traffic Light |
| traffic sign | traffic sign | Traffic Sign (Front/Back) |
| vegetation | vegetation | Vegetation |
| terrain | terrain | Terrain, Sand |
| sky | sky | Sky |
| person | person | Person |
| rider | rider | Bicyclist, Motorcyclist, Other Rider |
| car | car | Car |
| truck | truck | Truck |
| bus | bus | Bus |
| train | train | On Rails |
| motorcycle | motorcycle | Motorcycle |
| bicycle | bicycle | Bicycle |

### Programmatic Usage

```python
from label_unification import LabelUnificationManager, MapillarytoCityscapes

# Initialize manager
manager = LabelUnificationManager()

# Transform Mapillary label to Cityscapes format
cityscapes_label = manager.transform_label(mapillary_label, 'mapillary', 'cityscapes')

# Transform to unified format
unified_label = manager.transform_label(mapillary_label, 'mapillary', 'unified')

# Get class names and palettes
classes = manager.get_cityscapes_classes()  # or get_unified_classes()
palette = manager.get_cityscapes_palette()  # or get_unified_palette()
```

## Dataset Format Specifications

### BDD100k JSON Format
```json
{
  "name": "image_name.jpg",
  "labels": [
    {
      "category": "car",
      "box2d": {
        "x1": 100,
        "y1": 200,
        "x2": 300,
        "y2": 400
      }
    }
  ]
}
```

### Cityscapes Format
```
cityscapes/
├── leftImg8bit/
│   ├── train/
│   ├── val/
│   └── test/
└── gtFine/
    ├── train/
    ├── val/
    └── test/
```

## Available Domains by Dataset

| Dataset | Domains |
|---------|---------|
| ACDC | foggy, night, rainy, snowy |
| BDD10k | clear_day, cloudy, dawn_dusk, foggy, night, rainy, snowy |
| BDD100k | clear_day, cloudy, dawn_dusk, foggy, night, rainy, snowy |
| IDD-AW | clear_day, cloudy, dawn_dusk, foggy, night, rainy, snowy |
| MapillaryVistas | clear_day, cloudy, dawn_dusk, foggy, night, rainy, snowy |
| OUTSIDE15k | clear_day, cloudy, dawn_dusk, foggy, night, rainy, snowy |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PROVE_ROOT` | Base directory for all project data | (must be set for stratified layout) |
| `PROVE_DATA_ROOT` | Stratified data splits | `${PROVE_ROOT}/FINAL_SPLITS` |
| `PROVE_GEN_ROOT` | Generated images | `${PROVE_ROOT}/GENERATED_IMAGES` |
| `PROVE_WEIGHTS_ROOT` | Model weights | `${PROVE_ROOT}/WEIGHTS` |
| `PROVE_MAPILLARY_ROOT` | MapillaryVistas dataset root (standard layout) | — |
| `PROVE_CITYSCAPES_ROOT` | Cityscapes dataset root (standard layout) | — |
| `PROVE_BDD_ROOT` | BDD100k/BDD10k dataset root (standard layout) | — |
| `PROVE_ACDC_ROOT` | ACDC dataset root (standard layout) | — |

## Manifest Generator for Generated Images

PROVE includes a standalone tool for creating manifest files that map generated images to their original counterparts. These manifests are required for training with generative augmentation strategies.

```bash
# Check manifest status for all methods
python tools/generate_manifests.py --status

# Generate manifests for all methods missing them (writable directories only)
python tools/generate_manifests.py --all-missing

# Generate manifest for a specific method
python tools/generate_manifests.py --method cycleGAN

# Regenerate all manifests (even existing ones)
python tools/generate_manifests.py --all

# Preview what would be done without writing files
python tools/generate_manifests.py --all-missing --dry-run

# Verbose output showing detailed progress
python tools/generate_manifests.py --method SUSTechGAN --verbose
```

**Output Files:**
Each method directory gets two manifest files:
- `manifest.csv`: CSV mapping generated images to originals (gen_path, original_path, name, domain, dataset, target_domain)
- `manifest.json`: Summary statistics including total images, match rate, domain/dataset breakdown

**Domain Name Handling:**
The tool normalizes various domain naming conventions:
- `fog` → `foggy`, `rain` → `rainy`, `snow` → `snowy`
- `sunny` → `clear_day`, `overcast` → `cloudy`
- `clear_day2foggy` or `clear_day_to_foggy` → recognized as source→target translation
