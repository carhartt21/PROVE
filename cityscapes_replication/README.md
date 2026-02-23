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
| SegFormer MIT-B3 | 81.94% | 160k | 1024×1024 | Transformer backbone, lighter than B5 |
| HRNet HR48 | 80.65% | 160k | 512×1024 | High-resolution multi-scale features |
| OCRNet HR48 | 81.35% | 160k | 512×1024 | Object-contextual representations |
| DeepLabV3+ R50 | 79.61% | 80k | 769×769 | ASPP decoder |
| PSPNet R50 | 78.55% | 80k | 769×769 | PPM pooling |
| SegNeXt MSCAN-B | ~79% (est) | 160k | 512×1024 | MSCAN attention, not officially benchmarked |

**Note:** SegNeXt MSCAN-B is adapted from the ADE20K config as it's not officially benchmarked on Cityscapes.

## Usage

```bash
# Preview jobs (dry run)
python submit_jobs.py --dry-run

# Submit all jobs
python submit_jobs.py

# Submit single model
python submit_jobs.py --model segformer

# Manual training (single GPU, for debugging)
cd ${HOME}/repositories/PROVE/cityscapes_replication
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

Cityscapes data is at: `${AWARE_DATA_ROOT}/CITYSCAPES/`
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

## Current Training Status (2026-01-31)

**All 6 models are training successfully!**

| Model | Progress | ETA | Node | Status |
|-------|----------|-----|------|--------|
| deeplabv3plus_r50 | 1,700/80k (2%) | ~1h 09m | makalu95 | ✅ Training |
| pspnet_r50 | 1,100/80k (1%) | ~1h 11m | makalu95 | ✅ Training |
| hrnet_hr48 | 46,550/160k (29%) | ~3h 17m | makalu94 | ✅ Training |
| segformer_b3 | 29,950/160k (19%) | ~5h 24m | makalu94 | ✅ Training |
| ocrnet_hr48 | 1,100/160k (0.7%) | ~6h 21m | makalu94 | ✅ Training |
| segnext_mscan_b | 21,350/160k (13%) | ~7h 49m | makalu94 | ✅ Training |

**Expected Completion:** All models within ~8 hours (by ~22:30 on 2026-01-31)

### Job IDs
- cs_hrnet_hr48: 983234
- cs_segformer_b3: 983239
- cs_segnext_mscan_b: 983271
- cs_ocrnet_hr48: 983368
- cs_deeplabv3plus_r50: 983372
- cs_pspnet_r50: 983373

### Setup Notes
- All models use **512x512 crop size** to match PROVE training conditions
- Using **single GPU** execution (changed from 4-GPU torchrun)
- Pretrained weights downloaded to `~/.cache/torch/hub/checkpoints/` (cluster nodes have no internet)
- labelTrainIds files generated using `prepare_cityscapes.py`

## Timeline

Created: 2026-01-31
Branch: cityscapes-replication
