# Advanced Usage

This document covers advanced configuration, performance optimization, troubleshooting, utility tools, and contributing to PROVE.

**Quick links:** [Datasets](DATASETS.md) | [Training](TRAINING.md) | [Testing](TESTING.md) | [README](../README.md)

---

## Configuration System

PROVE uses a hierarchical configuration system that allows for easy customization and reproducibility:

```python
from prove_config import PROVEConfig

# Initialize configuration generator
config_gen = PROVEConfig()

# Generate custom configuration
config = config_gen.generate_config(
    task_type='object_detection',
    dataset_format='bdd100k_json',
    dataset_path='./data/bdd100k/',
    model_name='yolox_l'
)
```

## Supported Models

### Semantic Segmentation Models
- **DeepLabV3+**: `deeplabv3plus_r50`, `deeplabv3plus_r101`
- **PSPNet**: `pspnet_r50`, `pspnet_r101`
- **SegFormer**: `segformer_mit-b5`
- **UperNet**: `upernet_swin`

### Object Detection Models
- **Faster R-CNN**: `faster_rcnn_r50_fpn_1x`
- **YOLOX**: `yolox_l`, `yolox_m`, `yolox_s`
- **RTMDet**: `rtmdet_l`, `rtmdet_m`, `rtmdet_s`
- **DETR**: `detr_r50`
- **Mask R-CNN**: `mask_rcnn_r50_fpn_1x`

## Advanced Configuration

### Custom Training Parameters

```python
# Modify training configuration
config = {
    'training': {
        'gpu_ids': [0, 1],  # Multi-GPU training
        'seed': 42,
        'lr': 0.01,
        'max_epochs': 24,
        'samples_per_gpu': 4,
        'workers_per_gpu': 8
    }
}
```

### Learning Rate Scheduling

```python
# Custom learning rate policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-7,
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001
)
```

### Data Augmentation

```python
# Training pipeline with augmentation
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
```

## Standard Augmentation Methods

PROVE includes 4 SOTA standard augmentation methods as optional baselines for comparison with weather-specific augmentations:

| Strategy | Method | Reference | Expected Improvement |
|----------|--------|-----------|---------------------|
| `std_cutmix` | CutMix | ICCV'19 | +3.9% mIoU |
| `std_mixup` | MixUp | ICLR'18 | +3.4% mIoU |
| `std_autoaugment` | AutoAugment | CVPR'19 | +2.8% mIoU |
| `std_randaugment` | RandAugment | CVPR'20 | +2.3% mIoU |

## Custom Transforms

PROVE includes custom transforms for handling various label formats:

| Transform | Purpose |
|-----------|--------|
| `ReduceToSingleChannel` | Converts 3-channel label PNGs to single channel |
| `CityscapesLabelIdToTrainId` | Maps Cityscapes label IDs (0-33) to trainIds (0-18) |

These transforms are automatically applied in the training pipeline when using datasets with non-standard label formats.

### Usage with Unified Training

```bash
# Train with CutMix augmentation
python unified_training.py --dataset ACDC --model deeplabv3plus_r50 --strategy std_cutmix

# Train with RandAugment
python unified_training.py --dataset BDD10k --model pspnet_r50 --strategy std_randaugment
```

### Programmatic Usage

```python
from tools.standard_augmentations import StandardAugmentationFamily

# Initialize augmentation
aug = StandardAugmentationFamily(method='cutmix', p_aug=0.5)

# Apply to batch (512x512 segmentation)
images, labels = aug(images, labels)
```

## Reproducibility Features

### Deterministic Training
- Fixed random seeds across all components
- Deterministic CUDA operations
- Version-controlled configurations

### Experiment Tracking
- Automatic logging of hyperparameters
- Model checkpoint versioning
- Performance metrics logging
- TensorBoard integration

### Configuration Management
```bash
# Save experiment configuration
python prove.py config \
    --task-type object_detection \
    --dataset-format bdd100k_json \
    --dataset-path ./data/bdd100k/ \
    --config-path ./experiments/exp_001_config.py
```

## Performance Optimization

### Multi-GPU Training

```bash
# Distributed training across multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    prove.py train \
    --config-path config.py \
    --launcher pytorch
```

### Mixed Precision Training

```python
# Enable mixed precision in config
fp16 = dict(loss_scale=512.)
```

### Memory Optimization

```python
# Gradient checkpointing for large models
model = dict(
    backbone=dict(
        with_cp=True  # Enable gradient checkpointing
    )
)
```

### Speed Optimization

```bash
# Use mixed precision training
export CUDA_VISIBLE_DEVICES=0
python unified_training.py --fp16

# Optimize data loading
export NUM_WORKERS=4  # Adjust based on CPU cores
python unified_training.py --workers-per-gpu 4
```

## Troubleshooting

### Version Compatibility Issues

**Problem**: `ModuleNotFoundError: No module named 'mmcv._ext'` or import errors with MMCV.

**Cause**: MMCV 2.x requires compiled CUDA extensions that aren't included in pip packages. This affects operations like focal loss, deformable convolutions, etc.

**Solutions**:
1. **Use conda (recommended)**:
   ```bash
   conda install -c conda-forge mmcv=2.2.0
   ```

2. **Use compatible older versions**:
   ```bash
   pip install mmcv-full==1.7.2 mmsegmentation==0.30.0
   ```

3. **Use subprocess training** (avoids import conflicts):
   ```bash
   python unified_training.py --train-method subprocess
   ```

**Problem**: `AssertionError: MMCV==X.X.X is used but incompatible`

**Cause**: Different versions of MMSegmentation/MMDetection require specific MMCV versions.

**Solutions**:
- MMSegmentation 1.2.x → MMCV 2.2.0
- MMSegmentation 1.1.x → MMCV 2.0.x  
- MMSegmentation 0.30.0 → MMCV 1.7.x

### CUDA Out of Memory
```python
# Reduce batch size
data = dict(
    samples_per_gpu=1,  # Reduce from default 2
    workers_per_gpu=2   # Reduce workers if needed
)
```

### Dataset Loading Errors
```bash
# Verify dataset format
python prove.py prepare \
    --dataset-path ./data/your_dataset/ \
    --dataset-format your_format \
    --output-path ./data/converted/ \
    --verbose
```

### Configuration Errors
```python
# Validate configuration
from mmcv import Config
cfg = Config.fromfile('your_config.py')
print(cfg.pretty_text)
```

### Debug Mode

```bash
# Run with debug logging
export PYTHONPATH=$PWD:$PYTHONPATH
python prove.py train \
    --config-path config.py \
    --work-dir ./debug_run/ \
    --debug
```

### MMSegmentation Deprecation Warnings

**Problem**: `UserWarning: ``build_loss`` would be deprecated soon, please use ``mmseg.registry.MODELS.build()`` instead.`

**Cause**: MMSegmentation 1.2.2 includes deprecated internal API usage that will be removed in future versions.

**Solution**: This warning has been suppressed in PROVE scripts. The functionality remains unaffected, and training/testing will continue normally. The warning appears in logs but doesn't impact performance.

### Custom Transform Registration Errors

**Problem**: `KeyError: 'MapillaryLabelTransform is not in the mmseg::transform registry'`

**Cause**: Custom transforms from `unified_datasets.py` need to be imported before the config is loaded to register them with MMSegmentation.

**Solution**: The training scripts automatically import `custom_transforms` and `unified_datasets` before loading configs. If running training manually, ensure you import these modules first:

```python
import sys
sys.path.insert(0, '/path/to/PROVE')
import custom_transforms
import unified_datasets

# Now load and use config
from mmengine.config import Config
cfg = Config.fromfile('config.py')
```

### Mapillary Label IndexError

**Problem**: `IndexError: index 70 is out of bounds for axis 0 with size 66` when training with Mapillary Vistas dataset.

**Cause**: While Mapillary Vistas officially has 66 classes (IDs 0-65), some label images contain higher values (up to 70 or more) due to dataset artifacts or annotation errors.

**Solution**: The label transformation lookup tables are sized to 256 (covering all possible uint8 values). Any label values beyond the official 66 classes are automatically mapped to the ignore index (255). No action needed - this is handled automatically.

## Utilities

### Weights Directory Analyzer

Analyze and summarize all training configurations and checkpoints:

```bash
# Display formatted table with summary statistics
python weights_analyzer.py

# Show only summary statistics
python weights_analyzer.py --summary-only

# Export to JSON for programmatic processing
python weights_analyzer.py --format json --output weights_summary.json

# Export to CSV for spreadsheet analysis
python weights_analyzer.py --format csv --output weights_summary.csv

# Analyze a custom directory
python weights_analyzer.py --root /path/to/weights/
```

### Submit Untested Tests

Automatically identify and submit test jobs for configurations that haven't been tested yet:

```bash
# Preview untested configurations (dry-run)
./scripts/submit_untested_tests.sh --dry-run

# Submit all untested standard configurations
./scripts/submit_untested_tests.sh

# Include clear_day domain-filtered variants
./scripts/submit_untested_tests.sh --include-clear-day

# Filter by strategy
./scripts/submit_untested_tests.sh --strategy std_photometric_distort

# Limit number of jobs to submit
./scripts/submit_untested_tests.sh --limit 10
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--dry-run` | Preview commands without submitting | off |
| `--include-clear-day` | Include domain-filtered model variants | off |
| `--include-multi` | Include multi-dataset configurations | off |
| `--strategy <name>` | Filter by specific strategy | all |
| `--dataset <name>` | Filter by specific dataset | all |
| `--model <name>` | Filter by specific model | all |
| `--queue <name>` | LSF queue name | BatchGPU |
| `--gpu-mem <size>` | GPU memory requirement | 24G |
| `--limit <n>` | Maximum number of jobs | unlimited |
| `--detailed` | Submit detailed (fine-grained) tests instead of basic tests | off |
| `--missing-detailed` | Filter for configs with basic tests but missing detailed tests | off |

## Examples

### Complete Object Detection Workflow

```bash
# 1. Prepare BDD100k dataset
python prove.py prepare \
    --dataset-path ./data/bdd100k/labels/bdd100k_labels_images_train.json \
    --dataset-format bdd100k_json \
    --output-path ./data/bdd100k_coco/

# 2. Generate configuration
python prove.py config \
    --task-type object_detection \
    --dataset-format bdd100k_json \
    --dataset-path ./data/bdd100k_coco/ \
    --model-name faster_rcnn_r50_fpn_1x \
    --config-path ./configs/bdd100k_faster_rcnn.py

# 3. Train model
python prove.py train \
    --config-path ./configs/bdd100k_faster_rcnn.py \
    --work-dir ./work_dirs/bdd100k_faster_rcnn/ \
    --load-from ./checkpoints/faster_rcnn_r50_fpn_1x_coco.pth

# 4. Evaluate model
python prove.py test \
    --config-path ./configs/bdd100k_faster_rcnn.py \
    --checkpoint-path ./work_dirs/bdd100k_faster_rcnn/latest.pth \
    --output-path ./results/bdd100k_evaluation/

# 5. Run inference
python prove.py inference \
    --config-path ./configs/bdd100k_faster_rcnn.py \
    --checkpoint-path ./work_dirs/bdd100k_faster_rcnn/latest.pth \
    --image-path ./test_images/driving_scene.jpg \
    --output-path ./results/detection_result.jpg
```

### Complete Semantic Segmentation Workflow

```bash
# 1. Prepare Cityscapes dataset
python prove.py prepare \
    --dataset-path ./data/cityscapes/ \
    --dataset-format cityscapes \
    --output-path ./data/cityscapes_processed/

# 2. Generate configuration
python prove.py config \
    --task-type semantic_segmentation \
    --dataset-format cityscapes \
    --dataset-path ./data/cityscapes_processed/ \
    --model-name deeplabv3plus_r50 \
    --config-path ./configs/cityscapes_deeplabv3plus.py

# 3. Train model
python prove.py train \
    --config-path ./configs/cityscapes_deeplabv3plus.py \
    --work-dir ./work_dirs/cityscapes_deeplabv3plus/ \
    --load-from ./checkpoints/deeplabv3plus_r50-d8_cityscapes.pth

# 4. Evaluate model
python prove.py test \
    --config-path ./configs/cityscapes_deeplabv3plus.py \
    --checkpoint-path ./work_dirs/cityscapes_deeplabv3plus/latest.pth \
    --output-path ./results/cityscapes_evaluation/
```

## Contributing

### Adding New Dataset Formats

1. Implement converter in `DatasetConverter` class:
```python
@staticmethod
def convert_your_format_to_coco(input_path: str, output_path: str) -> bool:
    # Implementation here
    pass
```

2. Update supported formats in configuration:
```python
'supported_formats': {
    'object_detection': ['bdd100k_json', 'coco_json', 'your_format'],
    # ...
}
```

3. Add format-specific configuration:
```python
def _get_your_format_config(self, base_config, dataset_path):
    # Implementation here
    pass
```

### Adding New Models

1. Update model configurations:
```python
'models': {
    'object_detection': {
        'available': [
            'existing_models',
            'your_new_model'
        ]
    }
}
```

2. Add model-specific configuration:
```python
def _get_detection_config(self, model_name):
    configs = {
        'your_new_model': {
            '_base_': ['path/to/your/model/config.py']
        }
    }
    # ...
```
