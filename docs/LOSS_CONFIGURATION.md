# Loss Configuration in PROVE

**Last Updated:** 2026-02-11 (model table updated to 6 models)

## Overview

PROVE uses a multi-loss training approach where CrossEntropyLoss (CE) serves as the primary loss function, with optional auxiliary losses for improved segmentation quality.

---

## Architecture Components

### Segmentation Model Structure

All segmentation models use an **EncoderDecoder** architecture:

| Model | Backbone | Decode Head | Auxiliary Head |
|-------|----------|-------------|----------------|
| DeepLabV3+ | ResNet50-V1c | DepthwiseSeparableASPPHead | FCNHead ✓ |
| PSPNet | ResNet50-V1c | PSPHead | FCNHead ✓ |
| SegFormer | MiT-B3 | SegFormerHead | None |
| SegNeXt | MSCAN-B | LightDecodeHead | None |
| HRNet | HRNet-W48 | FCNHead | FCNHead ✓ |
| Mask2Former | Swin-B | Mask2FormerHead | None |

**Note:** SegFormer, SegNeXt, and Mask2Former have no auxiliary head — only the main decode head.

---

## Base Loss Configuration (CrossEntropy Only)

### Decode Head (Main Prediction)
```python
'loss_decode': {
    'type': 'CrossEntropyLoss',
    'use_sigmoid': False,      # Multi-class softmax classification
    'loss_weight': 1.0,        # Full weight for primary head
    'avg_non_ignore': True,    # Average over valid pixels only (ignore_index=255)
}
```

### Auxiliary Head (Intermediate Supervision)
```python
'loss_decode': {
    'type': 'CrossEntropyLoss',
    'use_sigmoid': False,
    'loss_weight': 0.4,        # Reduced weight for auxiliary supervision
    'avg_non_ignore': True,
}
```

### Total Loss Calculation (CE Only)

For models with auxiliary head (DeepLabV3+, PSPNet):
```
Total = L_decode × 1.0 + L_auxiliary × 0.4
```

For SegFormer (no auxiliary head):
```
Total = L_decode × 1.0
```

---

## Auxiliary Loss Configuration

When using `--aux-loss <loss>`, an additional loss is **appended** to each head's loss list.

### Usage
```bash
python unified_training.py --dataset BDD10k --model deeplabv3plus_r50 \
    --strategy baseline --aux-loss focal
```

### Available Auxiliary Losses

| Loss | Type | Weight | Key Parameters | Purpose |
|------|------|--------|----------------|---------|
| **Focal** | FocalLoss | 0.3 | `gamma=2.0, alpha=0.5` | Down-weights easy examples, focuses on hard pixels |
| **Lovasz** | LovaszLoss | 0.3 | `classes='all', per_image=True` | Directly optimizes IoU metric |
| **Boundary** | SegBoundaryLoss | 0.3 | `ignore_index=255` | Penalizes boundary prediction errors |

### Modified Loss Configuration (with aux-loss)

**Decode Head:**
```python
'loss_decode': [
    {'type': 'CrossEntropyLoss', 'loss_weight': 1.0, ...},   # Primary (unchanged)
    {'type': '<AuxLoss>', 'loss_weight': 0.3, ...}           # Added auxiliary
]
```

**Auxiliary Head:**
```python
'loss_decode': [
    {'type': 'CrossEntropyLoss', 'loss_weight': 0.4, ...},   # Primary (unchanged)
    {'type': '<AuxLoss>', 'loss_weight': 0.3, ...}           # Added auxiliary
]
```

### Total Loss Calculation (CE + Auxiliary)

For DeepLabV3+/PSPNet:
```
Total = (CE_decode × 1.0 + AuxLoss_decode × 0.3) + (CE_auxiliary × 0.4 + AuxLoss_auxiliary × 0.3)
      = CE_decode + 0.3 × AuxLoss_decode + 0.4 × CE_auxiliary + 0.3 × AuxLoss_auxiliary
```

For SegFormer:
```
Total = CE_decode × 1.0 + AuxLoss_decode × 0.3
```

---

## Loss Function Details

### CrossEntropyLoss (Primary)
```python
{
    'type': 'CrossEntropyLoss',
    'use_sigmoid': False,       # Use softmax for multi-class
    'loss_weight': 1.0,         # or 0.4 for auxiliary head
    'avg_non_ignore': True,     # Ignore pixels with label 255
}
```
- Standard pixel-wise classification loss
- Computes negative log-likelihood for correct class
- `avg_non_ignore=True` ensures proper handling of ignore regions

### FocalLoss
```python
{
    'type': 'FocalLoss',
    'use_sigmoid': False,       # Multi-class mode
    'gamma': 2.0,               # Focusing parameter
    'alpha': 0.5,               # Class weighting
    'reduction': 'mean',
    'loss_weight': 0.3,
}
```
- Down-weights easy examples: $(1 - p_t)^\gamma \times CE$
- `gamma=2.0` reduces contribution of well-classified pixels significantly
- Helps with class imbalance by focusing on hard examples

### LovaszLoss
```python
{
    'type': 'LovaszLoss',
    'loss_type': 'multi_class',
    'classes': 'all',           # Compute for all classes
    'per_image': True,          # Average per image then batch
    'reduction': 'mean',
    'loss_weight': 0.3,
}
```
- Directly optimizes IoU through Lovász extension of submodular functions
- `classes='all'` ensures all classes contribute to gradient
- `per_image=True` computes loss per image before batch averaging
- More stable than `classes='present'` which can cause gradient issues

### SegBoundaryLoss
```python
{
    'type': 'SegBoundaryLoss',
    'loss_weight': 0.3,
    'ignore_index': 255,
}
```
- Custom loss implemented in `custom_losses.py`
- Computes boundary maps from predictions and labels using Sobel operators
- Binary cross-entropy between predicted and ground-truth boundaries
- Improves edge sharpness and object delineation

---

## Early Stopping Configuration

```python
{
    'type': 'EarlyStoppingHook',
    'monitor': 'val/mIoU',
    'rule': 'greater',
    'patience': 5,              # 5 validations without improvement
    'min_delta': 0.1,           # Must improve by 0.1 mIoU
    'strict': False,
    'check_finite': True,
}
```

With `eval_interval=5000` iterations:
- Early stopping triggers after 25,000 iterations (5 × 5000) without ≥0.1 mIoU improvement

---

## Possible Improvements

### 1. **Dice Loss Addition**
```python
{
    'type': 'DiceLoss',
    'loss_weight': 0.3,
    'use_sigmoid': False,
    'ignore_index': 255,
}
```
- Complements CE by directly optimizing F1-score
- Often used alongside CE for medical/satellite segmentation
- MMSegmentation supports this natively

### 2. **Class-Weighted CrossEntropy**
```python
{
    'type': 'CrossEntropyLoss',
    'class_weight': [1.0, 2.5, 1.0, 10.0, ...],  # Per-class weights
    'loss_weight': 1.0,
}
```
- Weight rare classes higher (person, bicycle, motorcycle)
- Can be computed from training set class frequencies
- Addresses severe class imbalance in driving datasets

### 3. **OHEM (Online Hard Example Mining)**
```python
{
    'type': 'CrossEntropyLoss',
    'loss_weight': 1.0,
    'sampler': {
        'type': 'OHEMPixelSampler',
        'thresh': 0.7,
        'min_kept': 100000,
    }
}
```
- Only backpropagate through hardest K% of pixels
- Focuses training on challenging regions
- Particularly useful for boundary refinement

### 4. **Label Smoothing**
```python
{
    'type': 'CrossEntropyLoss',
    'loss_weight': 1.0,
    'label_smoothing': 0.1,
}
```
- Prevents overconfident predictions
- Improves calibration and generalization
- Simple modification: soft labels instead of hard 0/1

### 5. **Auxiliary Loss Weight Tuning**
Current weight (0.3) may not be optimal. Consider:
- **Higher (0.5)**: Stronger auxiliary signal, may help with Lovasz oscillation
- **Lower (0.1)**: Let CE dominate more, use aux as regularizer
- **Scheduled decay**: Start high (0.5) and decay to (0.1) during training

### 6. **Boundary-Aware Focal Loss**
Combine boundary detection with focal weighting:
```python
# Compute boundary weight map
boundary_weight = sobel_edges(labels)
# Apply to focal loss
weighted_focal = focal_loss * (1 + boundary_weight)
```

### 7. **Multi-Scale Loss (already implemented differently)**
Apply loss at multiple decoder scales if architecture supports it.

---

## Recommendations for Current Study

### For Loss Ablation (Current Jobs)
The current configuration tests:
1. CE only (baseline)
2. CE + Focal (hard example mining)
3. CE + Lovasz (direct IoU optimization)
4. CE + Boundary (edge refinement)

**Expected outcomes:**
- Focal may help with rare class performance
- Lovasz should improve overall mIoU but may show oscillation
- Boundary should improve edge quality metrics (if measured)

### For Production Use
Based on literature and initial observations:
1. **Start with CE + Focal (0.3)** - Most stable, good hard example mining
2. **Add Lovasz (0.2) for IoU boost** - Use lower weight to reduce oscillation
3. **Consider class weighting** - Address class imbalance directly

---

## References

- [Lovasz-Softmax Loss](https://arxiv.org/abs/1705.08790) - IoU optimization
- [Focal Loss](https://arxiv.org/abs/1708.02002) - Hard example mining
- [Boundary Loss](https://arxiv.org/abs/1812.07032) - Distance-based boundary loss
- [MMSegmentation Losses](https://mmsegmentation.readthedocs.io/en/latest/api.html#module-mmseg.models.losses)
