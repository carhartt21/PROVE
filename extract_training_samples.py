#!/usr/bin/env python3
"""
Extract and visualize actual training batches from GeneratedAugmentedDataset

This script directly loads the dataset and samples batches to verify
what images and labels are being used in training.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import shutil

sys.path.insert(0, str(Path(__file__).parent))

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

from generated_images_dataset import GeneratedAugmentedDataset


def get_cityscapes_palette():
    """Cityscapes 19-class color palette"""
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
    """Convert label map to RGB"""
    h, w = label.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(len(palette)):
        mask = label == class_id
        colored[mask] = palette[class_id]
    return colored


def main():
    print("=" * 80)
    print("EXTRACTING ACTUAL TRAINING BATCHES")
    print("=" * 80)
    
    # Configuration (matching job 815489)
    data_root = '/scratch/aaa_exchange/AWARE/FINAL_SPLITS'
    manifest_path = '/scratch/aaa_exchange/AWARE/GENERATED_IMAGES/stargan_v2/manifest.csv'
    gen_root = '/scratch/aaa_exchange/AWARE/GENERATED_IMAGES/stargan_v2'
    
    dataset = 'IDD-AW'
    domain_filter = 'clear_day'
    
    print(f"\nDataset: {dataset}")
    print(f"Domain filter: {domain_filter}")
    print(f"Manifest: {manifest_path}")
    print(f"Real/gen ratio: 0.0 (100% generated)")
    
    # Create output directory
    output_dir = Path('/tmp/batch_samples_ratio0')
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    (output_dir / 'images').mkdir()
    (output_dir / 'labels').mkdir()
    (output_dir / 'overlays').mkdir()
    
    print(f"\nOutput directory: {output_dir}")
    
    # Create dataset
    print("\nCreating GeneratedAugmentedDataset...")
    dataset_obj = GeneratedAugmentedDataset(
        data_root=data_root,
        generated_root=gen_root,
        manifest_path=manifest_path,
        dataset_filter=dataset,
        include_original=False,  # Only generated images (ratio=0.0)
        img_suffix='.png',
        seg_map_suffix='.png',
        pipeline=[],  # No transforms for visualization
        serialize_data=False,
    )
    
    print(f"  Total samples: {len(dataset_obj)}")
    
    # Get palette
    palette = get_cityscapes_palette()
    
    # Sample first N items
    num_samples = min(20, len(dataset_obj))
    print(f"\nSampling first {num_samples} items...")
    
    metadata_file = output_dir / 'samples_metadata.txt'
    with open(metadata_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAINING BATCH SAMPLES - METADATA\n")
        f.write("=" * 80 + "\n\n")
        
        for idx in range(num_samples):
            print(f"\nSample {idx}:")
            
            # Get sample
            sample = dataset_obj[idx]
            
            # Extract data
            img_path = sample.get('img_path', 'unknown')
            seg_map_path = sample.get('seg_map_path', 'unknown')
            
            # Determine if generated or real
            is_generated = 'generated' in img_path.lower() or 'gen_' in img_path.lower() or 'GENERATED_IMAGES' in img_path
            sample_type = "GENERATED" if is_generated else "REAL"
            
            print(f"  Type: {sample_type}")
            print(f"  Image: ...{img_path[-80:]}" if len(img_path) > 80 else f"  Image: {img_path}")
            print(f"  Label: ...{seg_map_path[-80:]}" if len(seg_map_path) > 80 else f"  Label: {seg_map_path}")
            
            # Write to metadata
            f.write(f"Sample {idx}:\n")
            f.write(f"  Type: {sample_type}\n")
            f.write(f"  Image: {img_path}\n")
            f.write(f"  Label: {seg_map_path}\n\n")
            
            # Load and save image
            try:
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img)
                
                # Save image
                img_save_path = output_dir / 'images' / f'sample{idx:03d}_{sample_type}.jpg'
                img.save(img_save_path, quality=95)
                
                # Load and save label
                label = Image.open(seg_map_path)
                label_array = np.array(label)
                
                # Save raw label
                label_save_path = output_dir / 'labels' / f'sample{idx:03d}_{sample_type}.png'
                label.save(label_save_path)
                
                # Save colored label
                colored = colorize_label(label_array, palette)
                colored_save_path = output_dir / 'labels' / f'sample{idx:03d}_{sample_type}_colored.png'
                Image.fromarray(colored).save(colored_save_path)
                
                # Save overlay
                overlay = (img_array * 0.5 + colored * 0.5).astype(np.uint8)
                overlay_save_path = output_dir / 'overlays' / f'sample{idx:03d}_{sample_type}.jpg'
                Image.fromarray(overlay).save(overlay_save_path, quality=95)
                
                print(f"  ✓ Saved")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                f.write(f"  ERROR: {e}\n\n")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Count sample types
    with open(metadata_file, 'r') as f:
        content = f.read()
        gen_count = content.count("Type: GENERATED")
        real_count = content.count("Type: REAL")
    
    print(f"\nSamples extracted: {num_samples}")
    print(f"  Generated: {gen_count} ({gen_count/num_samples*100:.1f}%)")
    print(f"  Real: {real_count} ({real_count/num_samples*100:.1f}%)")
    
    if real_count > 0:
        print("\n⚠️  WARNING: Found REAL images when ratio=0.0 (should be 100% generated)!")
    else:
        print("\n✓ All samples are generated (as expected for ratio=0.0)")
    
    print(f"\nOutput directory: {output_dir}")
    print(f"  Images: {output_dir}/images/")
    print(f"  Labels: {output_dir}/labels/")
    print(f"  Overlays: {output_dir}/overlays/")
    print(f"  Metadata: {metadata_file}")
    
    print("\n✓ Extraction complete!")


if __name__ == '__main__':
    main()
