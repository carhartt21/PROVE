#!/usr/bin/env python3
"""
Investigate why segmentation performance is almost identical across different inputs.

Hypothesis: The model is learning label structure (spatial patterns) without using image features.
This could be due to:
1. Very weak image normalization/preprocessing
2. Strong label prior dominating learning
3. Architecture bias favoring label-only learning
4. Loss function weighting issue
"""

from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import sys

sys.path.insert(0, '/home/mima2416/repositories/PROVE')

from mmengine.config import Config
from mmseg.registry import DATASETS

def analyze_label_statistics():
    """Analyze if labels have strong structural patterns."""
    print("\n" + "="*60)
    print("LABEL STATISTICS ANALYSIS")
    print("="*60)
    
    cfg_path = '/scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION/gen_stargan_v2/iddaw/pspnet_r50_ratio0p00/configs/training_config.py'
    cfg = Config.fromfile(cfg_path)
    
    # Build dataset
    dataset_cfg = cfg.train_dataloader.dataset
    dataset = DATASETS.build(dataset_cfg)
    dataset.full_init()
    
    # Sample labels
    print(f"\nDataset size: {len(dataset)}")
    
    label_counts = {i: 0 for i in range(19)}
    label_entropy = []
    
    for i in range(min(100, len(dataset))):
        data = dataset[i]
        seg_map = data['gt_sem_seg'].data.numpy().flatten()
        
        # Count class distribution
        unique, counts = np.unique(seg_map, return_counts=True)
        for u, c in zip(unique, counts):
            if u < 19:
                label_counts[u] += c
        
        # Calculate entropy
        probs = counts / seg_map.size
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        label_entropy.append(entropy)
    
    print("\nLabel distribution (first 100 samples):")
    total = sum(label_counts.values())
    for cls, count in sorted(label_counts.items()):
        pct = 100 * count / total if total > 0 else 0
        print(f"  Class {cls:2d}: {pct:6.2f}% ({count} pixels)")
    
    print(f"\nLabel entropy per sample: mean={np.mean(label_entropy):.3f}, std={np.std(label_entropy):.3f}")
    print(f"  Range: [{np.min(label_entropy):.3f}, {np.max(label_entropy):.3f}]")


def analyze_image_statistics():
    """Analyze if images have meaningful variation."""
    print("\n" + "="*60)
    print("IMAGE STATISTICS ANALYSIS")
    print("="*60)
    
    cfg_path = '/scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION/gen_stargan_v2/iddaw/pspnet_r50_ratio0p00/configs/training_config.py'
    cfg = Config.fromfile(cfg_path)
    
    # Build dataset WITHOUT preprocessing pipeline
    dataset_cfg = cfg.train_dataloader.dataset
    dataset_cfg['pipeline'] = []  # Remove pipeline to get raw images
    dataset = DATASETS.build(dataset_cfg)
    dataset.full_init()
    
    print(f"\nAnalyzing {min(50, len(dataset))} samples...")
    
    img_means = []
    img_stds = []
    img_ranges = []
    
    for i in range(min(50, len(dataset))):
        try:
            data = dataset[i]
            if 'img_path' in data:
                img_path = data['img_path']
                img = Image.open(f'/scratch/aaa_exchange/AWARE/FINAL_SPLITS/{img_path}')
                img_arr = np.array(img, dtype=np.float32)
                
                if img_arr.ndim == 3:
                    mean = np.mean(img_arr)
                    std = np.std(img_arr)
                    img_min = np.min(img_arr)
                    img_max = np.max(img_arr)
                    
                    img_means.append(mean)
                    img_stds.append(std)
                    img_ranges.append((img_min, img_max))
        except Exception as e:
            pass
    
    if img_means:
        print(f"\nImage intensity statistics:")
        print(f"  Mean brightness: {np.mean(img_means):.2f} ± {np.std(img_means):.2f}")
        print(f"  Mean std dev: {np.mean(img_stds):.2f} ± {np.std(img_stds):.2f}")
        print(f"  Range: [{np.mean([r[0] for r in img_ranges]):.2f}, {np.mean([r[1] for r in img_ranges]):.2f}]")


def check_model_feature_extraction():
    """Check if model can actually extract image features vs just learning from labels."""
    print("\n" + "="*60)
    print("MODEL FEATURE EXTRACTION ANALYSIS")
    print("="*60)
    
    # Load a trained model
    cfg_path = '/scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION/gen_stargan_v2/iddaw/pspnet_r50_ratio0p00/configs/training_config.py'
    cfg = Config.fromfile(cfg_path)
    
    checkpoint_path = '/scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION/gen_stargan_v2/iddaw/pspnet_r50_ratio0p00/iter_80000.pth'
    
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    from mmengine.runner import Runner
    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(checkpoint_path)
    
    model = runner.model
    backbone = model.backbone
    
    print(f"\nBackbone architecture: {type(backbone).__name__}")
    
    # Check if input gradient is zero (would indicate frozen backbone)
    try:
        import torch.nn as nn
        total_params = sum(p.numel() for p in backbone.parameters())
        trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        print(f"\nBackbone parameters: {total_params:,} ({trainable_params:,} trainable)")
        
        if trainable_params == 0:
            print("  WARNING: Backbone is NOT trainable (frozen)!")
        elif trainable_params < total_params / 2:
            print(f"  WARNING: Only {100*trainable_params/total_params:.1f}% of backbone is trainable")
    except Exception as e:
        print(f"  Could not analyze trainability: {e}")


def check_label_leakage():
    """Check if labels might be leaking into training somehow."""
    print("\n" + "="*60)
    print("LABEL LEAKAGE / ARCHITECTURE BIAS ANALYSIS")
    print("="*60)
    
    print("\nPossible issues:")
    print("1. Check if seg_map_suffix is accidentally matching image files")
    print("2. Check if DataLoader is mixing up img and label paths")
    print("3. Check model decode head - maybe it has a skip connection directly to labels")
    print("4. Check if validation uses same data as training (data leakage)")
    
    # Load config and check paths
    cfg_path = '/scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION/gen_stargan_v2/iddaw/pspnet_r50_ratio0p00/configs/training_config.py'
    cfg = Config.fromfile(cfg_path)
    
    print("\nDataset configuration:")
    ds_cfg = cfg.train_dataloader.dataset
    print(f"  img_suffix: {ds_cfg.get('img_suffix', 'NOT SET')}")
    print(f"  seg_map_suffix: {ds_cfg.get('seg_map_suffix', 'NOT SET')}")
    print(f"  img_dir/img_path: {ds_cfg.get('data_prefix', {}).get('img_path', 'NOT SET')}")
    print(f"  seg_dir/seg_path: {ds_cfg.get('data_prefix', {}).get('seg_map_path', 'NOT SET')}")
    
    print("\nValidation data config:")
    val_ds_cfg = cfg.val_dataloader.dataset
    print(f"  img_path: {val_ds_cfg.get('data_prefix', {}).get('img_path', 'NOT SET')}")
    print(f"  seg_map_path: {val_ds_cfg.get('data_prefix', {}).get('seg_map_path', 'NOT SET')}")
    
    # Check if train and val share paths
    train_img = ds_cfg.get('data_prefix', {}).get('img_path', '')
    val_img = val_ds_cfg.get('data_prefix', {}).get('img_path', '')
    train_seg = ds_cfg.get('data_prefix', {}).get('seg_map_path', '')
    val_seg = val_ds_cfg.get('data_prefix', {}).get('seg_map_path', '')
    
    if 'clear_day' in train_img and 'clear_day' not in val_img:
        print("\n✓ Train/val split looks correct (train=clear_day, val=all)")
    else:
        print(f"\n⚠ WARNING: Train/val split may be wrong!")
        print(f"   Train img: {train_img}")
        print(f"   Val img: {val_img}")


if __name__ == '__main__':
    try:
        analyze_label_statistics()
    except Exception as e:
        print(f"Error in label analysis: {e}")
    
    try:
        analyze_image_statistics()
    except Exception as e:
        print(f"Error in image analysis: {e}")
    
    try:
        check_label_leakage()
    except Exception as e:
        print(f"Error in leakage check: {e}")
    
    try:
        check_model_feature_extraction()
    except Exception as e:
        print(f"Error in feature extraction check: {e}")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("""
1. Verify images are actually being processed:
   - Print first layer activations with real vs noise inputs
   - Compare attention maps between real and noise inputs
   
2. Test backbone freezing:
   - Train with frozen backbone (should perform poorly)
   - Compare to unfrozen backbone
   
3. Check gradient flow:
   - Monitor gradients at each layer
   - See if image gradients reach zero
   
4. Ablation test - train on LABELS ONLY:
   - Create fake "image" data (all zeros or fixed pattern)
   - Train segmentation model
   - If mIoU is still ~22%, labels are dominating
    
5. Check loss function:
   - Verify CrossEntropyLoss is being applied correctly
   - Check if logits are being used or pre-softmaxed outputs
    """)
