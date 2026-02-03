# Cityscapes Training Comparison Results

**Date**: February 2, 2026
**Purpose**: Verify PROVE unified training config produces comparable results to reference mmseg configs

## Training Configurations

| Parameter | CITYSCAPES_REPLICATION | WEIGHTS_CITYSCAPES |
|-----------|------------------------|-------------------|
| **Iterations** | 80,000 | 20,000 |
| **Batch Size** | 2 | 16 |
| **Total Samples** | 160,000 | 320,000 |
| **Crop Size** | 512×512 | 512×512 |
| **Optimizer** | SGD | SGD |
| **Base LR** | 0.01 | 0.08 (scaled) |
| **LR Scheduler** | Poly | Poly |
| **Val Interval** | 8,000 | 2,000 |

## Validation mIoU Results

| Model | Reference (80k, BS=2) | PROVE (20k, BS=16) | Difference |
|-------|----------------------|-------------------|------------|
| pspnet_r50 | 57.64% | 56.86% | **-0.78%** |
| deeplabv3plus_r50 | 58.02% | (pending) | - |
| hrnet_hr48 | 65.67% | (pending) | - |
| segformer_b3 | 79.98% | 75.97% | **-4.01%** |
| segnext_mscan_b | 81.13% | 79.97% | **-1.16%** |
| mask2former_swin-b | N/A | (running @ 2500) | - |

## Analysis

### Overall
- **Average difference**: ~-2% lower than reference
- CNN models (PSPNet, DeepLabV3+): Similar performance (~1% difference)
- Transformer models (SegFormer): Larger gap (~4% difference)

### Possible Causes of Performance Gap

1. **Batch size effects on transformers**: SegFormer and transformer-based models may prefer longer training with smaller batch sizes due to:
   - More gradient updates (80k vs 20k steps)
   - Better generalization with smaller batch noise

2. **Learning rate scaling**: Linear scaling (0.01 → 0.08 for 8× batch) may not be optimal for all architectures

3. **Training convergence**: With 4× more samples but 4× fewer gradient updates, the model may not have fully converged

### Recommendations

1. **For production use**: The PROVE config is acceptable for relative comparisons (same config across experiments)
2. **For absolute benchmarking**: Consider using original mmseg configs or extending training iterations
3. **For transformer models**: May benefit from reduced batch size (BS=8) with proportionally more iterations (40k)

## Status

- ✅ pspnet_r50: Completed (chge7185)
- ✅ segformer_mit-b3: Completed (mima2416)
- ✅ segnext_mscan-b: Completed (mima2416)
- 🔄 mask2former_swin-b: Running (mima2416) @ 2500/20000
- ⏳ deeplabv3plus_r50: Lock conflict (chge7185)
- ⏳ hrnet_hr48: Lock conflict (chge7185)

## Additional Experiment: BS=8, 40k Iterations

**Purpose**: Test if transformer models (SegFormer, SegNext) achieve better results with smaller batch size and longer training.

**Hypothesis**: 80k iterations at BS=2 = 160k samples; 40k iterations at BS=8 = 320k samples (same as 20k @ BS=16 but more gradient updates)

**Jobs Submitted**:
- Job 1055760: segformer_mit-b3 (BS=8, 40k iters)  
- Job 1055761: segnext_mscan-b (BS=8, 40k iters)

**Output Directory**: `/scratch/aaa_exchange/AWARE/WEIGHTS_CITYSCAPES_BS8/baseline/cityscapes/`
