# PROVE Configurations
# Dataset path: /scratch/aaa_exchange/AWARE/FINAL_SPLITS/

This folder contains configuration files for training and testing semantic segmentation models
on different datasets and label spaces.

## Configuration Files

### Single Dataset Configurations

| Config File | Dataset | Label Space | Classes |
|-------------|---------|-------------|---------|
| `cityscapes_config.py` | Cityscapes | Cityscapes | 19 |
| `mapillary_vistas_config.py` | Mapillary Vistas | Cityscapes | 19 |
| `mapillary_vistas_unified_config.py` | Mapillary Vistas | Unified | 42 |

### Joint Training Configurations

| Config File | Datasets | Label Space | Classes |
|-------------|----------|-------------|---------|
| `joint_cityscapes_mapillary_config.py` | Cityscapes + Mapillary | Cityscapes | 19 |
| `joint_unified_config.py` | Cityscapes + Mapillary | Unified | 42 |

## Dataset Paths

All configurations use datasets from:
```
/scratch/aaa_exchange/AWARE/FINAL_SPLITS/
├── cityscapes/
│   ├── leftImg8bit/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── gtFine/
│       ├── train/
│       ├── val/
│       └── test/
└── mapillary_vistas/
    ├── training/
    │   ├── images/
    │   └── v1.2/labels/
    └── validation/
        ├── images/
        └── v1.2/labels/
```

## Usage

### Training

```bash
# Train on Cityscapes only
python prove.py train --config-path configs/cityscapes_config.py --work-dir ./work_dirs/cityscapes/

# Train on Mapillary Vistas (mapped to Cityscapes labels)
python prove.py train --config-path configs/mapillary_vistas_config.py --work-dir ./work_dirs/mapillary_cs/

# Joint training (Cityscapes label space - 19 classes)
python prove.py train --config-path configs/joint_cityscapes_mapillary_config.py --work-dir ./work_dirs/joint_cs/

# Joint training (Unified label space - 42 classes)
python prove.py train --config-path configs/joint_unified_config.py --work-dir ./work_dirs/joint_unified/
```

### Testing

```bash
# Test trained model
python prove.py test --config-path configs/cityscapes_config.py --checkpoint-path ./work_dirs/cityscapes/latest.pth
```

## Label Spaces

### Cityscapes (19 classes)
Standard Cityscapes evaluation classes. Best for benchmarking.

### Unified (42 classes)
Extended label space preserving more semantic information from both datasets.
Includes additional classes like: parking, bike lane, guard rail, bridge, tunnel, 
water, snow, mountain, etc.

## Notes

- All configs use DeepLabV3+ with ResNet-50 backbone by default
- Mapillary labels are automatically transformed to the target label space
- Validation is always performed on Cityscapes for consistent benchmarking
