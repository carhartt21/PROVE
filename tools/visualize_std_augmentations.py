#!/usr/bin/env python3
"""
Visualize Standard Augmentation Strategies

This script generates and saves example training images showing how each
std_* augmentation strategy (CutMix, MixUp, AutoAugment, RandAugment) transforms
the input data during training.

Usage:
    python visualize_std_augmentations.py --output-dir /path/to/output

    # With custom images
    python visualize_std_augmentations.py --image-dir /path/to/images --output-dir /path/to/output

Output:
    - Original images with segmentation masks
    - Augmented images for each strategy
    - Comparison grids showing before/after
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import torch
import torch.nn.functional as F
from PIL import Image
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.standard_augmentations import (
    StandardAugmentationFamily,
    CutMix,
    MixUp,
    AutoAugmentSegmentation,
    RandAugmentSegmentation,
)


# Cityscapes color palette (19 classes)
CITYSCAPES_COLORS = np.array([
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

CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]


def label_to_color(label: np.ndarray) -> np.ndarray:
    """Convert label indices to RGB color image."""
    H, W = label.shape
    color = np.zeros((H, W, 3), dtype=np.uint8)
    
    for class_id in range(len(CITYSCAPES_COLORS)):
        mask = label == class_id
        color[mask] = CITYSCAPES_COLORS[class_id]
    
    return color


def load_sample_data(data_root: str, num_samples: int = 4) -> tuple:
    """
    Load sample images and labels from the dataset.
    
    Returns:
        images: (B, C, H, W) tensor normalized to [0, 1]
        labels: (B, H, W) tensor with class indices
    """
    img_dir = Path(data_root) / 'train/images/BDD10k/clear_day'
    lbl_dir = Path(data_root) / 'train/labels/BDD10k/clear_day'
    
    if not img_dir.exists():
        print(f"Image directory not found: {img_dir}")
        print("Creating synthetic sample data...")
        return create_synthetic_data(num_samples)
    
    img_files = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))[:num_samples * 2]
    
    if len(img_files) < num_samples:
        print(f"Not enough images found. Creating synthetic data...")
        return create_synthetic_data(num_samples)
    
    # Randomly select samples
    selected = random.sample(img_files, num_samples)
    
    images = []
    labels = []
    
    for img_path in selected:
        # Load image
        img = Image.open(img_path).convert('RGB')
        img = img.resize((512, 512), Image.BILINEAR)
        img_np = np.array(img).astype(np.float32) / 255.0
        images.append(torch.from_numpy(img_np).permute(2, 0, 1))
        
        # Load corresponding label
        lbl_name = img_path.stem + '.png'
        lbl_path = lbl_dir / lbl_name
        
        if lbl_path.exists():
            lbl = Image.open(lbl_path)
            lbl = lbl.resize((512, 512), Image.NEAREST)
            lbl_np = np.array(lbl)
            # Handle 3-channel labels (take first channel)
            if lbl_np.ndim == 3:
                lbl_np = lbl_np[:, :, 0]
            # Clamp to valid range
            lbl_np = np.clip(lbl_np, 0, 18)
            labels.append(torch.from_numpy(lbl_np).long())
        else:
            # Create synthetic label
            labels.append(create_synthetic_label((512, 512)))
    
    images = torch.stack(images)
    labels = torch.stack(labels)
    
    print(f"Loaded {num_samples} samples from {data_root}")
    return images, labels


def create_synthetic_data(num_samples: int = 4) -> tuple:
    """Create synthetic sample data for visualization."""
    H, W = 512, 512
    
    images = []
    labels = []
    
    # Create diverse synthetic scenes
    for i in range(num_samples):
        # Create base image with gradients
        img = torch.zeros(3, H, W)
        
        # Sky region (top)
        sky_h = H // 3
        img[:, :sky_h, :] = torch.tensor([0.5, 0.7, 0.9]).view(3, 1, 1)  # Light blue
        
        # Building region (middle)
        build_h = H // 3
        img[:, sky_h:sky_h+build_h, :] = torch.tensor([0.3, 0.3, 0.35]).view(3, 1, 1)  # Gray
        
        # Road region (bottom)
        img[:, sky_h+build_h:, :] = torch.tensor([0.3, 0.25, 0.3]).view(3, 1, 1)  # Dark
        
        # Add some variation
        noise = torch.rand(3, H, W) * 0.1
        img = (img + noise).clamp(0, 1)
        
        # Add random colored patches (simulate objects)
        for _ in range(random.randint(3, 8)):
            x = random.randint(0, W-50)
            y = random.randint(0, H-50)
            w = random.randint(20, 80)
            h = random.randint(20, 80)
            color = torch.rand(3)
            img[:, y:y+h, x:x+w] = color.view(3, 1, 1) * 0.8 + img[:, y:y+h, x:x+w] * 0.2
        
        images.append(img)
        labels.append(create_synthetic_label((H, W)))
    
    images = torch.stack(images)
    labels = torch.stack(labels)
    
    print(f"Created {num_samples} synthetic samples")
    return images, labels


def create_synthetic_label(size: tuple) -> torch.Tensor:
    """Create a synthetic segmentation label."""
    H, W = size
    label = torch.zeros(H, W, dtype=torch.long)
    
    # Sky (top third)
    label[:H//3, :] = 10  # sky
    
    # Buildings (middle third)
    label[H//3:2*H//3, :] = 2  # building
    
    # Road (bottom third)
    label[2*H//3:, :] = 0  # road
    
    # Add some sidewalk
    label[2*H//3:, :W//4] = 1  # sidewalk
    label[2*H//3:, 3*W//4:] = 1  # sidewalk
    
    # Add random objects
    for _ in range(random.randint(5, 15)):
        x = random.randint(0, W-40)
        y = random.randint(H//3, H-40)
        w = random.randint(10, 50)
        h = random.randint(10, 50)
        cls = random.choice([11, 12, 13, 14, 15])  # person, rider, car, truck, bus
        label[y:y+h, x:x+w] = cls
    
    return label


def visualize_augmentation(
    images: torch.Tensor,
    labels: torch.Tensor,
    method: str,
    output_path: str,
    p_aug: float = 1.0,
) -> None:
    """
    Visualize augmentation effect and save to file.
    
    Args:
        images: (B, C, H, W) tensor normalized to [0, 1]
        labels: (B, H, W) tensor with class indices
        method: Augmentation method name
        output_path: Path to save the visualization
        p_aug: Probability of augmentation (default 1.0 to force application)
    """
    # Create augmentation
    aug = StandardAugmentationFamily(method=method, p_aug=p_aug)
    
    # Apply augmentation
    aug_images, aug_labels = aug(images.clone(), labels.clone())
    
    # Create figure
    B = images.shape[0]
    fig = plt.figure(figsize=(20, 5 * B))
    gs = GridSpec(B, 4, figure=fig, hspace=0.3, wspace=0.1)
    
    for i in range(B):
        # Original image
        ax1 = fig.add_subplot(gs[i, 0])
        img = images[i].permute(1, 2, 0).numpy()
        ax1.imshow(img)
        ax1.set_title(f'Original Image {i+1}', fontsize=12)
        ax1.axis('off')
        
        # Original label
        ax2 = fig.add_subplot(gs[i, 1])
        lbl = labels[i].numpy()
        lbl_color = label_to_color(lbl)
        ax2.imshow(lbl_color)
        ax2.set_title(f'Original Label {i+1}', fontsize=12)
        ax2.axis('off')
        
        # Augmented image
        ax3 = fig.add_subplot(gs[i, 2])
        aug_img = aug_images[i].permute(1, 2, 0).numpy()
        aug_img = np.clip(aug_img, 0, 1)  # Ensure valid range
        ax3.imshow(aug_img)
        ax3.set_title(f'{method.upper()} Image {i+1}', fontsize=12)
        ax3.axis('off')
        
        # Augmented label
        ax4 = fig.add_subplot(gs[i, 3])
        aug_lbl = aug_labels[i].numpy()
        aug_lbl_color = label_to_color(aug_lbl)
        ax4.imshow(aug_lbl_color)
        ax4.set_title(f'{method.upper()} Label {i+1}', fontsize=12)
        ax4.axis('off')
    
    # Add overall title
    expected_gain = {
        'cutmix': '+3.9%',
        'mixup': '+3.4%',
        'autoaugment': '+2.8%',
        'randaugment': '+2.3%',
    }
    gain = expected_gain.get(method, 'N/A')
    plt.suptitle(f'{method.upper()} Augmentation (Expected Improvement: {gain} mIoU)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_comparison_grid(
    images: torch.Tensor,
    labels: torch.Tensor,
    output_path: str,
) -> None:
    """
    Create a grid comparing all augmentation methods.
    
    Args:
        images: (B, C, H, W) tensor normalized to [0, 1]
        labels: (B, H, W) tensor with class indices
        output_path: Path to save the visualization
    """
    methods = ['cutmix', 'mixup', 'autoaugment', 'randaugment']
    expected_gains = {
        'cutmix': '+3.9%',
        'mixup': '+3.4%',
        'autoaugment': '+2.8%',
        'randaugment': '+2.3%',
    }
    
    # Use first sample only for comparison
    img = images[0:1]
    lbl = labels[0:1]
    
    # Create augmented versions
    augmented = {}
    for method in methods:
        aug = StandardAugmentationFamily(method=method, p_aug=1.0)
        aug_img, aug_lbl = aug(img.clone(), lbl.clone())
        augmented[method] = (aug_img[0], aug_lbl[0])
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    
    # Row 1: Images
    # Original
    axes[0, 0].imshow(img[0].permute(1, 2, 0).numpy())
    axes[0, 0].set_title('Original\n(Baseline)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Augmented
    for i, method in enumerate(methods):
        aug_img = augmented[method][0].permute(1, 2, 0).numpy()
        aug_img = np.clip(aug_img, 0, 1)
        axes[0, i+1].imshow(aug_img)
        axes[0, i+1].set_title(f'{method.upper()}\n({expected_gains[method]} expected)', 
                               fontsize=12, fontweight='bold')
        axes[0, i+1].axis('off')
    
    # Row 2: Labels
    # Original
    lbl_color = label_to_color(lbl[0].numpy())
    axes[1, 0].imshow(lbl_color)
    axes[1, 0].set_title('Original Label', fontsize=11)
    axes[1, 0].axis('off')
    
    # Augmented labels
    for i, method in enumerate(methods):
        aug_lbl = augmented[method][1].numpy()
        aug_lbl_color = label_to_color(aug_lbl)
        axes[1, i+1].imshow(aug_lbl_color)
        axes[1, i+1].set_title(f'{method.upper()} Label', fontsize=11)
        axes[1, i+1].axis('off')
    
    plt.suptitle('Standard Augmentation Strategy Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_legend_figure(output_path: str) -> None:
    """Create a legend figure showing all class colors."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create patches
    patches = []
    for i, (name, color) in enumerate(zip(CITYSCAPES_CLASSES, CITYSCAPES_COLORS)):
        patch = mpatches.Patch(color=color/255.0, label=f'{i}: {name}')
        patches.append(patch)
    
    ax.legend(handles=patches, loc='center', ncol=2, fontsize=10)
    ax.axis('off')
    ax.set_title('Cityscapes Class Legend (19 classes)', fontsize=14, fontweight='bold')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Standard Augmentation Strategies')
    parser.add_argument('--data-root', type=str, 
                        default='${AWARE_DATA_ROOT}/FINAL_SPLITS',
                        help='Path to dataset root')
    parser.add_argument('--output-dir', type=str, 
                        default='result_figures/std_augmentation_visualization',
                        help='Output directory for visualizations')
    parser.add_argument('--num-samples', type=int, default=4,
                        help='Number of samples to visualize')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Standard Augmentation Visualization")
    print("=" * 60)
    
    # Load or create sample data
    images, labels = load_sample_data(args.data_root, args.num_samples)
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Create visualizations for each method
    methods = ['cutmix', 'mixup', 'autoaugment', 'randaugment']
    
    for method in methods:
        print(f"\nVisualizing {method}...")
        output_path = output_dir / f'{method}_visualization.png'
        visualize_augmentation(images, labels, method, str(output_path))
    
    # Create comparison grid
    print("\nCreating comparison grid...")
    comparison_path = output_dir / 'comparison_grid.png'
    create_comparison_grid(images, labels, str(comparison_path))
    
    # Create legend
    print("\nCreating legend...")
    legend_path = output_dir / 'class_legend.png'
    create_legend_figure(str(legend_path))
    
    # Create multiple runs to show stochasticity
    print("\nCreating multiple run comparison...")
    for run in range(3):
        comparison_path = output_dir / f'comparison_grid_run{run+1}.png'
        create_comparison_grid(images, labels, str(comparison_path))
    
    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {output_dir}")
    print("=" * 60)
    
    # Print summary
    print("\nFiles created:")
    for f in sorted(output_dir.glob('*.png')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
