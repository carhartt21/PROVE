# Standard Augmentation Family

SOTA standard augmentations as control baselines for weather augmentation experiments.

## Methods

| Method | Reference | Expected Improvement |
|--------|-----------|---------------------|
| **CutMix** | ICCV'19 | +3.9% mIoU |
| **MixUp** | ICLR'18 | +3.4% mIoU |
| **AutoAugment** | CVPR'19 | +2.8% mIoU |
| **RandAugment** | CVPR'20 | +2.3% mIoU |

## Usage

### Basic Usage (Standalone)

```python
from tools.standard_augmentations import StandardAugmentationFamily

# Initialize with desired method
aug = StandardAugmentationFamily(method='cutmix', p_aug=0.5)

# Apply to batch (512x512 segmentation)
images, labels = aug(images, labels)
```

### Training Integration (Automatic via Hook)

When using `std_*` strategies in the PROVE pipeline, the `StandardAugmentationHook` 
is automatically added to the training configuration. This hook applies batch-level
augmentations during training.

```bash
# Train with standard augmentation
python unified_training.py \
    --dataset BDD10k \
    --model deeplabv3plus_r50 \
    --strategy std_cutmix \
    --domain-filter clear_day
```

The hook is automatically registered and applies augmentation in `before_train_iter`.

### Visualization

Generate visualizations showing how each augmentation transforms images:

```bash
# Generate comparison visualizations
python tools/visualize_std_augmentations.py

# With custom settings
python tools/visualize_std_augmentations.py \
    --data-root /path/to/FINAL_SPLITS \
    --output-dir result_figures/std_augmentation_visualization \
    --num-samples 4 \
    --seed 42
```

Output files:
- `comparison_grid.png` - Side-by-side comparison of all methods
- `cutmix_visualization.png` - CutMix detailed visualization
- `mixup_visualization.png` - MixUp detailed visualization
- `autoaugment_visualization.png` - AutoAugment detailed visualization
- `randaugment_visualization.png` - RandAugment detailed visualization
- `class_legend.png` - Cityscapes class color legend

## CLI Usage

```bash
# Run tests
python tools/standard_augmentations.py
```

## Files

| File | Description |
|------|-------------|
| `standard_augmentations.py` | Core augmentation implementations |
| `standard_augmentation_hook.py` | MMEngine Hook for training integration |
| `visualize_std_augmentations.py` | Visualization script |

## Parameters

| Argument | Description | Default |
|----------|-------------|---------|
| `method` | `cutmix`, `mixup`, `autoaugment`, `randaugment` | `cutmix` |
| `p_aug` | Augmentation probability | `0.5` |

### Method-specific

- **CutMix**: `alpha=1.0` (Beta distribution)
- **MixUp**: `alpha=0.4`, `soft_labels=False`
- **AutoAugment**: ImageNet policy adapted for segmentation
- **RandAugment**: `n=2` transforms, `m=9` magnitude

## Compatibility

- ✓ 512x512 semantic segmentation
- ✓ 19 Cityscapes classes
- ✓ Batch processing
- ✓ Online augmentation (no pre-generation)
