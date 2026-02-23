# std_minimal Strategy

## Overview

The `std_minimal` strategy represents a **minimal standard augmentation baseline** that applies basic augmentation techniques (RandomCrop, RandomFlip, and 1x PhotoMetricDistortion) to training images.

## Origin

These models were created when the `generate_training_jobs.py` script accidentally used `real_gen_ratio=1.0` instead of `0.5` for generative strategies. This resulted in training on **100% real images** with standard augmentations but **no generated images**.

## Training Pipeline

```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='ReduceToSingleChannel'),
    dict(keep_ratio=False, scale=(512, 512), type='Resize'),
    dict(cat_max_ratio=0.75, crop_size=(512, 512), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),  # Applied once
    dict(type='PackSegInputs'),
]
```

## Comparison with Other Strategies

| Strategy | RandomCrop | RandomFlip | PhotoMetricDistortion |
|----------|------------|------------|----------------------|
| **baseline** | ❌ | ❌ | ❌ |
| **std_minimal** | ✅ | ✅ | 1x |
| **std_std_photometric_distort** | ✅ | ✅ | 2x |

## Purpose

`std_minimal` serves as an **ablation baseline** to answer:
- "What is the effect of standard augmentation pipeline alone (without generated images)?"
- "How much of the performance gain comes from augmentation vs. generated data?"

## Coverage

| Dataset | Models |
|---------|--------|
| idd-aw_cd | deeplabv3plus_r50, pspnet_r50 (38 total - 19 strategies × 2 models) |
| bdd10k_cd | deeplabv3plus_r50, pspnet_r50 (4 total - 2 strategies × 2 models) |

Note: Only deeplabv3plus_r50 and pspnet_r50 models were created due to the bug in the original script. No segformer models exist.

## Date Created

2025-01-12 to 2025-01-13 (originally as gen_* strategies with wrong ratio)
2025-01-XX (moved to std_minimal)
