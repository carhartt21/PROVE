# Cityscapes Replication Experiment

## Purpose

This experiment replicates the standard mmsegmentation Cityscapes training pipeline to verify that our training infrastructure can achieve published results.

**Motivation:** Our PROVE experiments currently achieve only 45-48% mIoU on BDD10k with SegFormer MIT-B5, while the published results on Cityscapes are 82.25%. This experiment will determine if the issue is:
1. **Pipeline bug** - our training pipeline is incorrect
2. **Dataset difference** - BDD10k is fundamentally harder than Cityscapes
3. **Configuration issue** - hyperparameters are suboptimal

## Key Difference: The Pipeline Bug

Our current PROVE pipeline uses:
```python
# CURRENT (WRONG)
Resize(512, 512)  # Fixed resize
RandomCrop(512, 512)  # SAME SIZE - NO EFFECT!
RandomFlip(0.5)
```

The standard mmsegmentation pipeline uses:
```python
# STANDARD (CORRECT)
RandomResize(scale=(2048,1024), ratio_range=(0.5, 2.0))  # CRITICAL!
RandomCrop(769, 769)  # Now meaningful - extracts random crop from resized image
RandomFlip(0.5)
PhotoMetricDistortion()
```

The `RandomResize` step is **critical** - it provides:
- Scale augmentation (0.5x to 2.0x)
- Multi-scale training capability
- Proper context variation for learning

## Selected Models

| Model | Expected mIoU | Iterations | Crop Size | Notes |
|-------|---------------|------------|-----------|-------|
| SegFormer MIT-B5 | 82.25% | 160k | 1024×1024 | Transformer backbone |
| DeepLabV3+ R50 | 79.61% | 80k | 769×769 | ASPP decoder |
| PSPNet R50 | 78.55% | 80k | 769×769 | PPM pooling |

## Usage

```bash
# Preview jobs (dry run)
python submit_jobs.py --dry-run

# Submit all jobs
python submit_jobs.py

# Submit single model
python submit_jobs.py --model segformer

# Manual training (single GPU, for debugging)
cd /home/mima2416/repositories/PROVE/cityscapes_replication
python train.py configs/pspnet_r50_cityscapes_769x769.py --work-dir ./debug_run

# Distributed training (4 GPUs)
torchrun --nproc_per_node=4 train.py configs/pspnet_r50_cityscapes_769x769.py --work-dir ./work_dirs/pspnet
```

## Directory Structure

```
cityscapes_replication/
├── README.md                           # This file
├── train.py                            # Training script using MMEngine Runner
├── submit_jobs.py                      # LSF job submission script
├── generate_configs.py                 # Config generation script
└── configs/
    ├── segformer_mit-b5_cityscapes_1024x1024.py
    ├── deeplabv3plus_r50_cityscapes_769x769.py
    └── pspnet_r50_cityscapes_769x769.py
```

## Expected Results

If replication succeeds (achieves ~78-82% mIoU):
- **Confirms pipeline bug** in unified_training_config.py
- **Action:** Fix the pipeline to include RandomResize

If replication fails (achieves ~45% mIoU similar to PROVE):
- **Indicates other issue** (data, environment, or dependencies)
- **Action:** Debug installation and dependencies

## Data Location

Cityscapes data is at: `/scratch/aaa_exchange/AWARE/CITYSCAPES/`
- Images: `leftImg8bit/{train,val,test}/`
- Labels: `gtFine/{train,val,test}/`

## Pipeline Comparison

| Component | Standard Cityscapes | Current PROVE |
|-----------|---------------------|---------------|
| RandomResize | ✅ ratio_range=(0.5, 2.0) | ❌ Fixed Resize |
| RandomCrop | ✅ After RandomResize | ⚠️ After Fixed Resize (no effect) |
| PhotoMetricDistortion | ✅ Included | ⚠️ Only in some strategies |
| Crop Size | 512x1024 or 1024x1024 | 512x512 |
| Iterations | 80k-160k | 80k |

## Timeline

Created: 2026-01-31
Branch: cityscapes-replication
