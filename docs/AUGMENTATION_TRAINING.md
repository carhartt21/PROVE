# PROVE Augmentation Training Framework

## Overview

This framework provides an efficient configuration setup for training and testing different data augmentation strategies, including:

1. **Baseline** - No augmentation (reference)
2. **PhotoMetricDistort** - Classical image augmentation
3. **Generated Images** - Augmentation using images from various generative models

## Directory Structure

```
multi_model_configs/
├── baseline/                      # No augmentation (reference)
│   ├── ACDC/
│   ├── BDD10K/
│   ├── BDD100K/
│   ├── IDD-AW/
│   ├── MAPILLARYVISTAS/
│   └── OUTSIDE15K/
│
├── photometric_distort/           # PhotoMetricDistortion augmentation
│   ├── ACDC/
│   ├── BDD10K/
│   ├── BDD100K/
│   ├── IDD-AW/
│   ├── MAPILLARYVISTAS/
│   └── OUTSIDE15K/
│
├── gen_cycleGAN/                  # CycleGAN generated images (7x augmentation)
│   ├── ACDC/
│   ├── BDD10K/
│   ├── IDD-AW/
│   ├── MAPILLARYVISTAS/
│   └── OUTSIDE15K/
│
├── gen_<other_model>/             # Other generative models (when manifest available)
│   └── ...
│
├── train_photometric_distort.sh   # Training script for PhotoMetricDistort
└── train_gen_cycleGAN.sh          # Training script for CycleGAN augmentation
```

## Generated Images Augmentation

### How It Works

1. **Original clear_day images** from training datasets are transformed into 6 adverse conditions:
   - cloudy
   - dawn_dusk
   - fog
   - night
   - rainy
   - snowy

2. **7x Augmentation**: For each clear_day image, you get:
   - 1 original image
   - 6 generated images (one per condition)

3. **Label Preservation**: Generated images use the same segmentation labels as their source images (style transfer preserves semantics)

### Available Generative Models

| Model | Status | Images | Manifest |
|-------|--------|--------|----------|
| cycleGAN | ✅ Ready | 187,398 | ✅ |
| CUT | ⏳ Pending | - | ❌ |
| stargan_v2 | ⏳ Pending | - | ❌ |
| EDICT | ⏳ Pending | - | ❌ |
| ... (20+ more) | ⏳ Pending | - | ❌ |

### Creating Manifests for Other Models

To enable a generative model, create a `manifest.csv` in its directory:

```csv
gen_path,original_path,name,domain,target_domain
/path/to/generated/image.png,/path/to/original/image.jpg,image_name,clear_day2fog,fog
...
```

## Usage

### 1. List Available Strategies

```bash
python augmentation_training.py --list
```

### 2. Generate Configs for PhotoMetricDistort

```bash
python augmentation_training.py --generate-configs --strategy photometric_distort
```

### 3. Generate Configs for a Specific Generative Model

```bash
python augmentation_training.py --generate-configs --strategy gen_cycleGAN
```

### 4. Generate Configs for All Generative Models (with manifests)

```bash
python augmentation_training.py --generate-configs --all-generative
```

### 5. Train Models

```bash
# Train all models with PhotoMetricDistort
./multi_model_configs/train_photometric_distort.sh

# Train all models with CycleGAN augmentation
./multi_model_configs/train_gen_cycleGAN.sh
```

## Training Configuration

All strategies use consistent training settings:

| Parameter | Value |
|-----------|-------|
| max_iters | 40,000 |
| batch_size | 2 |
| checkpoint_interval | 5,000 |
| eval_interval | 3,333 |
| image_size | 512x512 |

## Experiment Comparison

### Recommended Evaluation Order

1. **Baseline vs PhotoMetricDistort**
   - Quick sanity check
   - Establishes whether augmentation helps at all

2. **Baseline vs gen_cycleGAN**
   - Tests if generated images improve robustness
   - 7x more training data

3. **PhotoMetricDistort vs gen_cycleGAN**
   - Compares classical vs generative augmentation

4. **gen_cycleGAN vs gen_<other_model>**
   - Compares different generative models
   - Requires manifests for other models

### Expected Results Structure

```
/scratch/aaa_exchange/AWARE/WEIGHTS/
├── baseline/
│   ├── acdc/
│   │   ├── deeplabv3plus_r50/
│   │   ├── pspnet_r50/
│   │   └── segformer_mit-b5/
│   └── ...
│
├── photometric_distort/
│   └── (same structure)
│
└── gen_cycleGAN/
    └── (same structure)
```

## Key Files

| File | Purpose |
|------|---------|
| `augmentation_training.py` | Main CLI for generating configs and scripts |
| `augmentation_config_generator.py` | Configuration generation logic |
| `generated_images_dataset.py` | Custom dataset class for generated images |
| `multi_model_training.py` | Original multi-model config generator |

## Next Steps

1. **Run baseline training** to establish reference metrics
2. **Run PhotoMetricDistort training** for classical augmentation comparison
3. **Run gen_cycleGAN training** for generative augmentation comparison
4. **Create manifests** for other generative models as they become ready
5. **Compare results** across all strategies

## Notes

- BDD100k (object detection) is excluded from generated image augmentation since the generated images have segmentation labels
- All configs use 40,000 iterations for fair comparison regardless of dataset size
- Segmentation models: DeepLabV3+, PSPNet, SegFormer
- Detection models: Faster R-CNN, YOLOX, RTMDet
