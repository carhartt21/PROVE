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

```python
from tools.standard_augmentations import StandardAugmentationFamily

# Initialize with desired method
aug = StandardAugmentationFamily(method='cutmix', p_aug=0.5)

# Apply to batch (512x512 segmentation)
images, labels = aug(images, labels)
```

## CLI Usage

```bash
# Run tests
python tools/standard_augmentations.py
```

## Integration with Training

```python
# In training loop
for images, labels in dataloader:
    images, labels = aug(images, labels)
    output = model(images)
    loss = criterion(output, labels)
```

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
