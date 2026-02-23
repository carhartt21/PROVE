# Label Handling in PROVE

This document describes how labels are handled for different datasets and training modes in the PROVE system.

## Dataset Native Class Labels

Each dataset uses its own native label format:

| Dataset | Native Classes | Format Description |
|---------|---------------|-------------------|
| **Cityscapes** | 19 | trainId (0-18), labelId needs conversion (0-33) |
| **ACDC** | 19 | Uses Cityscapes labelId format (7=road, 8=sidewalk, etc.) |
| **BDD10k** | 19 | Already in Cityscapes trainId format (0-18) |
| **IDD-AW** | 19 | Already in Cityscapes trainId format (0-18) |
| **MapillaryVistas** | 66 | Native Mapillary classes (0-65) |
| **OUTSIDE15k** | 24 | Custom 24-class format (0-23) |

### OUTSIDE15k Classes (24 classes)
```
0: unlabeled, 1: animal, 2: barrier, 3: bicycle, 4: boat, 5: bridge,
6: building, 7: grass, 8: ground, 9: mountain, 10: object, 11: person,
12: pole, 13: road, 14: sand, 15: sidewalk, 16: sign, 17: sky,
18: street light, 19: traffic light, 20: tunnel, 21: vegetation,
22: vehicle, 23: water
```

### MapillaryVistas Classes (66 classes)
```
0: Bird, 1: Ground Animal, 2: Curb, 3: Fence, 4: Guard Rail, 5: Barrier,
6: Wall, 7: Bike Lane, 8: Crosswalk - Plain, 9: Curb Cut, 10: Parking,
...
64: Ego Vehicle, 65: Unlabeled
```
(See `label_unification.py` for full class list)

## Training Modes

### Single-Dataset Training (Default)

When training on a single dataset, models use **native class labels**:

```python
# Example: Training on OUTSIDE15k
python unified_training.py --dataset OUTSIDE15k --model deeplabv3plus_r50 --strategy baseline
# Model outputs: 24 classes (OUTSIDE15k native)

# Example: Training on MapillaryVistas
python unified_training.py --dataset MapillaryVistas --model deeplabv3plus_r50 --strategy baseline
# Model outputs: 66 classes (Mapillary native)

# Example: Training on BDD10k
python unified_training.py --dataset BDD10k --model deeplabv3plus_r50 --strategy baseline
# Model outputs: 19 classes (Cityscapes trainId)
```

**Label transforms applied in single-dataset mode:**
- **ACDC/Cityscapes**: `CityscapesLabelIdToTrainId` (converts labelId to trainId)
- **BDD10k/IDD-AW**: No transform (already trainId format)
- **MapillaryVistas**: No transform (uses native 66 classes)
- **OUTSIDE15k**: No transform (uses native 24 classes)

### Multi-Dataset Training (Unified Labels)

When training on multiple datasets together, all labels are **unified to Cityscapes 19-class format**:

```python
# Example: Multi-dataset training
python unified_training.py --datasets ACDC BDD10k MapillaryVistas --model deeplabv3plus_r50 --strategy baseline
# All datasets mapped to 19 Cityscapes classes
```

**Label transforms applied in multi-dataset mode:**
- **ACDC/Cityscapes**: `CityscapesLabelIdToTrainId`
- **BDD10k/IDD-AW**: No transform (already trainId)
- **MapillaryVistas**: `MapillaryLabelTransform` (66 → 19 classes)
- **OUTSIDE15k**: `Outside15kLabelTransform` (24 → 19 classes)

### Domain Adaptation

For domain adaptation experiments where models trained on one dataset are evaluated on another, unified labels should be used to ensure comparability.

## Implementation Details

### Configuration in `unified_training_config.py`

The `_add_training_pipeline` method accepts a `use_unified_labels` parameter:

```python
def _add_training_pipeline(
    self,
    config: Dict[str, Any],
    task: str,
    aug_strategy: AugmentationStrategy,
    std_strategy: Optional[str] = None,
    dataset: Optional[str] = None,
    use_unified_labels: bool = False,  # Key parameter
) -> Dict[str, Any]:
```

- `use_unified_labels=False` (default): Use native class labels
- `use_unified_labels=True`: Map all labels to Cityscapes 19-class

### DatasetConfig Class Numbers

```python
DATASET_CONFIGS = {
    'ACDC': DatasetConfig(num_classes=19, classes=CITYSCAPES_CLASSES, ...),
    'BDD10k': DatasetConfig(num_classes=19, classes=CITYSCAPES_CLASSES, ...),
    'IDD-AW': DatasetConfig(num_classes=19, classes=CITYSCAPES_CLASSES, ...),
    'MapillaryVistas': DatasetConfig(num_classes=66, classes=MAPILLARY_CLASSES, ...),
    'OUTSIDE15k': DatasetConfig(num_classes=24, classes=OUTSIDE15K_CLASSES, ...),
}
```

### Label Transforms (in `custom_transforms.py`)

1. **CityscapesLabelIdToTrainId**: Converts Cityscapes labelId (0-33) to trainId (0-18)
2. **MapillaryLabelTransform**: Converts Mapillary 66 classes to Cityscapes 19 classes
3. **Outside15kLabelTransform**: Converts OUTSIDE15k 24 classes to Cityscapes 19 classes

## Bug Fixes (January 2025)

### Previous Bug 1: BDD10k/IDD-AW Label Transform
- **Issue**: BDD10k and IDD-AW were incorrectly processed with `CityscapesLabelIdToTrainId`
- **Fix**: These datasets already use trainId format; no transform needed
- **Affected**: All BDD10k and IDD-AW models

### Previous Bug 2: OUTSIDE15k/MapillaryVistas Class Unification
- **Issue**: OUTSIDE15k (24 classes) and MapillaryVistas (66 classes) were incorrectly mapped to 19 classes for single-dataset training
- **Fix**: Single-dataset training now uses native class labels
- **Affected**: All OUTSIDE15k and MapillaryVistas models

### Retraining
All affected models are being retrained with the corrected label handling. See `scripts/retrain_affected_models.py` for details.
