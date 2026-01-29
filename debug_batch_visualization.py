#!/usr/bin/env python3
"""
Debug script to visualize batches from a running/completed training job.

This script loads the same dataloader configuration as the training job
and visualizes the first few batches to verify what data is being used.

Usage:
    python debug_batch_visualization.py --job-id 815489 --num-batches 5
    
    Or with explicit parameters:
    python debug_batch_visualization.py \\
        --dataset IDD-AW \\
        --model pspnet_r50 \\
        --strategy gen_stargan_v2 \\
        --real-gen-ratio 0.0 \\
        --num-batches 5
"""

import sys
import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import json
import cv2

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from unified_training_config import UnifiedTrainingConfig
import warnings
warnings.filterwarnings('ignore')


def denormalize_image(img_tensor):
    """Convert normalized tensor to uint8 RGB image"""
    # ImageNet stats (used by MMSeg)
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
    img = img * std + mean
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # BGR to RGB (MMSeg uses BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_cityscapes_palette():
    """Cityscapes color palette"""
    return np.array([
        [128, 64, 128],   # road
        [244, 35, 232],   # sidewalk
        [70, 70, 70],     # building
        [102, 102, 156],  # wall
        [190, 153, 153],  # fence
        [153, 153, 153],  # pole
        [250, 170, 30],   # traffic light
        [220, 220, 0],    # traffic sign
        [107, 142, 35],   # vegetation
        [152, 251, 152],  # terrain
        [70, 130, 180],   # sky
        [220, 20, 60],    # person
        [255, 0, 0],      # rider
        [0, 0, 142],      # car
        [0, 0, 70],       # truck
        [0, 60, 100],     # bus
        [0, 80, 100],     # train
        [0, 0, 230],      # motorcycle
        [119, 11, 32],    # bicycle
    ], dtype=np.uint8)


def colorize_label(label, palette):
    """Convert label to RGB using palette"""
    h, w = label.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(len(palette)):
        mask = label == class_id
        colored[mask] = palette[class_id]
    return colored


def main():
    parser = argparse.ArgumentParser(description="Visualize training batches")
    parser.add_argument('--job-id', type=int, help="LSF job ID to analyze")
    parser.add_argument('--dataset', type=str, help="Dataset name (e.g., IDD-AW)")
    parser.add_argument('--model', type=str, help="Model name (e.g., pspnet_r50)")
    parser.add_argument('--strategy', type=str, help="Strategy name (e.g., gen_stargan_v2)")
    parser.add_argument('--real-gen-ratio', type=float, default=0.0, help="Real/gen ratio")
    parser.add_argument('--num-batches', type=int, default=5, help="Number of batches to visualize")
    parser.add_argument('--output-dir', type=str, default='./batch_debug', help="Output directory")
    
    args = parser.parse_args()
    
    # If job-id provided, parse from bjobs
    if args.job_id:
        print(f"TODO: Parse job parameters from bjobs {args.job_id}")
        print("For now, please provide --dataset, --model, --strategy manually")
        return
    
    if not all([args.dataset, args.model, args.strategy]):
        print("Error: Must provide --dataset, --model, and --strategy")
        print("Or use --job-id to auto-detect (not yet implemented)")
        return
    
    print("=" * 80)
    print("BATCH VISUALIZATION DEBUG")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Strategy: {args.strategy}")
    print(f"Real-gen ratio: {args.real_gen_ratio}")
    print(f"Batches to visualize: {args.num_batches}")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "labels").mkdir(exist_ok=True)
    (output_dir / "overlays").mkdir(exist_ok=True)
    
    # Generate config
    print("Generating training config...")
    config_gen = UnifiedTrainingConfig(
        dataset=args.dataset,
        model=args.model,
        strategy=args.strategy,
        real_gen_ratio=args.real_gen_ratio,
        domain_filter='clear_day',  # Stage 1 default
    )
    
    config_path = output_dir / "debug_config.py"
    config_gen.generate(str(config_path))
    
    # Load config
    print("Loading config...")
    from mmengine.config import Config
    cfg = Config.fromfile(str(config_path))
    
    # Build dataloader
    print("Building dataloader...")
    from mmengine.runner import Runner
    runner = Runner.from_cfg(cfg)
    
    # Get palette
    palette = get_cityscapes_palette()
    
    # Iterate through batches
    print(f"\nProcessing first {args.num_batches} batches...")
    print("=" * 80)
    
    dataloader = runner.train_dataloader
    metadata_list = []
    
    for batch_idx, batch_data in enumerate(dataloader):
        if batch_idx >= args.num_batches:
            break
        
        # Extract data
        inputs = batch_data['inputs']
        data_samples = batch_data['data_samples']
        batch_size = inputs.shape[0]
        
        print(f"\nBatch {batch_idx}:")
        print(f"  Batch size: {batch_size}")
        print(f"  Input shape: {inputs.shape}")
        
        # Extract image paths
        img_paths = []
        for sample in data_samples:
            if hasattr(sample, 'img_path'):
                path = sample.img_path
            elif hasattr(sample, 'metainfo') and 'img_path' in sample.metainfo:
                path = sample.metainfo['img_path']
            else:
                path = "unknown"
            img_paths.append(path)
        
        # Count real vs generated
        real_count = sum(1 for p in img_paths if isinstance(p, str) and 'generated' not in p.lower() and 'gen_' not in p.lower())
        gen_count = batch_size - real_count
        
        print(f"  Composition: {real_count} real + {gen_count} generated")
        print(f"  Paths:")
        for i, path in enumerate(img_paths):
            path_type = "GEN" if 'generated' in path.lower() or 'gen_' in path.lower() else "REAL"
            # Truncate long paths
            if len(path) > 80:
                path_display = "..." + path[-77:]
            else:
                path_display = path
            print(f"    [{path_type}] {path_display}")
        
        # Save metadata
        metadata = {
            'batch_idx': batch_idx,
            'batch_size': batch_size,
            'real_count': real_count,
            'gen_count': gen_count,
            'composition': f"{real_count} real + {gen_count} generated",
            'img_paths': img_paths,
        }
        metadata_list.append(metadata)
        
        # Save images and labels
        for i in range(min(batch_size, 4)):  # Save up to 4 samples per batch
            # Denormalize image
            img = denormalize_image(inputs[i])
            
            # Save image
            img_path = output_dir / "images" / f"batch{batch_idx:02d}_sample{i:02d}.jpg"
            Image.fromarray(img).save(img_path, quality=95)
            
            # Extract and save label
            sample = data_samples[i]
            if hasattr(sample, 'gt_sem_seg'):
                label = sample.gt_sem_seg.data.cpu().numpy().squeeze()
            elif hasattr(sample, 'gt_semantic_seg'):
                label = sample.gt_semantic_seg.data.cpu().numpy().squeeze()
            else:
                print(f"    Warning: No label found for sample {i}")
                continue
            
            # Save raw label
            label_path = output_dir / "labels" / f"batch{batch_idx:02d}_sample{i:02d}.png"
            Image.fromarray(label.astype(np.uint8)).save(label_path)
            
            # Save colored label
            colored = colorize_label(label, palette)
            colored_path = output_dir / "labels" / f"batch{batch_idx:02d}_sample{i:02d}_colored.png"
            Image.fromarray(colored).save(colored_path)
            
            # Save overlay
            overlay = (img * 0.5 + colored * 0.5).astype(np.uint8)
            overlay_path = output_dir / "overlays" / f"batch{batch_idx:02d}_sample{i:02d}.jpg"
            Image.fromarray(overlay).save(overlay_path, quality=95)
    
    # Save summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'model': args.model,
            'strategy': args.strategy,
            'real_gen_ratio': args.real_gen_ratio,
            'num_batches': args.num_batches,
            'batches': metadata_list,
        }, f, indent=2)
    
    print(f"\n✓ Visualization complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Images: {output_dir}/images/")
    print(f"  Labels: {output_dir}/labels/")
    print(f"  Overlays: {output_dir}/overlays/")
    print(f"  Summary: {summary_path}")
    
    # Print composition summary
    print("\nBatch Composition Summary:")
    total_real = sum(b['real_count'] for b in metadata_list)
    total_gen = sum(b['gen_count'] for b in metadata_list)
    total_samples = total_real + total_gen
    print(f"  Total samples: {total_samples}")
    print(f"  Real: {total_real} ({total_real/total_samples*100:.1f}%)")
    print(f"  Generated: {total_gen} ({total_gen/total_samples*100:.1f}%)")
    
    if args.real_gen_ratio == 0.0 and total_real > 0:
        print("\n⚠️  WARNING: Found real images when ratio=0.0 (should be 100% generated)!")
    elif args.real_gen_ratio == 1.0 and total_gen > 0:
        print("\n⚠️  WARNING: Found generated images when ratio=1.0 (should be 100% real)!")
    else:
        expected_real_pct = args.real_gen_ratio * 100
        actual_real_pct = total_real / total_samples * 100
        diff = abs(expected_real_pct - actual_real_pct)
        if diff > 5:
            print(f"\n⚠️  WARNING: Composition mismatch!")
            print(f"  Expected: {expected_real_pct:.1f}% real")
            print(f"  Actual: {actual_real_pct:.1f}% real")
            print(f"  Difference: {diff:.1f}%")
        else:
            print(f"\n✓ Composition matches expected ratio (within {diff:.1f}%)")


if __name__ == '__main__':
    main()
