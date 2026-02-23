#!/usr/bin/env python3
"""
Visualization of Crop Size Impact on CNN-based Semantic Segmentation

This script creates visualizations showing:
1. PSPNet PPM cell sizes at different crop sizes
2. DeepLabV3+ ASPP kernel coverage at different crop sizes  
3. Feature map utilization comparison
4. Performance vs crop size relationship

Author: GitHub Copilot
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import os

# Create output directory
OUTPUT_DIR = '/home/chge7185/repositories/PROVE/result_figures/crop_size_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.dpi'] = 150


def visualize_ppm_pool_scales():
    """
    Visualize how PSPNet PPM pooling cells cover the input at different crop sizes.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle('PSPNet Pyramid Pooling Module: Cell Coverage at Different Crop Sizes', 
                 fontsize=14, fontweight='bold')
    
    crop_sizes = [512, 769, 1024]
    pool_scales_configs = [
        ('Original: (1,2,3,6)', (1, 2, 3, 6)),
        ('Modified: (1,2,4,8)', (1, 2, 4, 8))
    ]
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    
    for row, (config_name, pool_scales) in enumerate(pool_scales_configs):
        for col, crop_size in enumerate(crop_sizes):
            ax = axes[row, col]
            feature_map_size = crop_size // 8  # ResNet output stride = 8
            
            ax.set_xlim(0, crop_size)
            ax.set_ylim(0, crop_size)
            ax.set_aspect('equal')
            
            # Draw feature map boundary
            rect = patches.Rectangle((0, 0), crop_size, crop_size, 
                                     linewidth=2, edgecolor='black', 
                                     facecolor='lightgray', alpha=0.3)
            ax.add_patch(rect)
            
            # Draw pooling cells for scale-6 (or scale-8 for modified)
            scale = pool_scales[-1]  # Largest scale (finest grid)
            cell_pixels = crop_size / scale
            
            for i in range(scale):
                for j in range(scale):
                    x = j * cell_pixels
                    y = i * cell_pixels
                    color_idx = (i + j) % len(colors)
                    rect = patches.Rectangle((x, y), cell_pixels, cell_pixels,
                                           linewidth=1, edgecolor='white',
                                           facecolor=colors[color_idx], alpha=0.5)
                    ax.add_patch(rect)
            
            # Add grid lines
            for i in range(1, scale):
                ax.axhline(y=i * cell_pixels, color='white', linewidth=1, linestyle='--', alpha=0.7)
                ax.axvline(x=i * cell_pixels, color='white', linewidth=1, linestyle='--', alpha=0.7)
            
            ax.set_title(f'{crop_size}×{crop_size}\nFM: {feature_map_size}×{feature_map_size}, '
                        f'Scale-{scale} cells: {cell_pixels:.0f}×{cell_pixels:.0f}px')
            ax.set_xlabel('Width (pixels)')
            if col == 0:
                ax.set_ylabel(f'{config_name}\nHeight (pixels)')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/ppm_cell_coverage.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/ppm_cell_coverage.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/ppm_cell_coverage.png")
    plt.close()


def visualize_aspp_kernel_coverage():
    """
    Visualize ASPP dilated convolution kernel coverage and overflow.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle('DeepLabV3+ ASPP: Dilated Convolution Kernel Coverage', 
                 fontsize=14, fontweight='bold')
    
    crop_sizes = [512, 769, 1024]
    aspp_configs = [
        ('Original: rates=(1,12,24,36)', [1, 12, 24, 36]),
        ('Modified: rates=(1,6,12,18)', [1, 6, 12, 18])
    ]
    
    for row, (config_name, dilations) in enumerate(aspp_configs):
        for col, crop_size in enumerate(crop_sizes):
            ax = axes[row, col]
            feature_map_size = crop_size // 8
            
            ax.set_xlim(-10, feature_map_size + 10)
            ax.set_ylim(-10, feature_map_size + 10)
            ax.set_aspect('equal')
            
            # Draw feature map
            rect = patches.Rectangle((0, 0), feature_map_size, feature_map_size,
                                     linewidth=2, edgecolor='black',
                                     facecolor='lightblue', alpha=0.3)
            ax.add_patch(rect)
            
            # Center position on feature map
            center = feature_map_size // 2
            
            # Draw each dilation rate's effective receptive field
            colors = ['#27ae60', '#3498db', '#f39c12', '#e74c3c']
            
            max_rate = dilations[-1]
            effective_kernel_size = (3 - 1) * max_rate + 1
            half_kernel = effective_kernel_size // 2
            
            # Check for overflow
            overflow = half_kernel > center
            overflow_pct = 0
            if overflow:
                # Calculate what percentage of positions would overflow
                overflow_positions = 0
                total_positions = feature_map_size * feature_map_size
                for x in range(feature_map_size):
                    for y in range(feature_map_size):
                        if (x - half_kernel < 0 or x + half_kernel >= feature_map_size or
                            y - half_kernel < 0 or y + half_kernel >= feature_map_size):
                            overflow_positions += 1
                overflow_pct = overflow_positions / total_positions * 100
            
            for i, (rate, color) in enumerate(zip(reversed(dilations), reversed(colors))):
                kernel_size = (3 - 1) * rate + 1
                half = kernel_size // 2
                
                rect = patches.Rectangle((center - half, center - half), 
                                        kernel_size, kernel_size,
                                        linewidth=2, edgecolor=color,
                                        facecolor=color, alpha=0.2,
                                        label=f'rate={rate} ({kernel_size}×{kernel_size})')
                ax.add_patch(rect)
            
            # Mark center
            ax.plot(center, center, 'ko', markersize=8)
            
            status = "✓ Fits" if not overflow else f"⚠ {overflow_pct:.0f}% overflow"
            ax.set_title(f'{crop_size}×{crop_size} → FM: {feature_map_size}×{feature_map_size}\n'
                        f'Max kernel: {effective_kernel_size}×{effective_kernel_size} | {status}')
            ax.set_xlabel('Feature Map Width')
            if col == 0:
                ax.set_ylabel(f'{config_name}\nFeature Map Height')
            
            ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/aspp_kernel_coverage.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/aspp_kernel_coverage.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/aspp_kernel_coverage.png")
    plt.close()


def visualize_performance_vs_crop_size():
    """
    Visualize performance degradation curves.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Semantic Segmentation Performance vs Crop Size (Cityscapes)', 
                 fontsize=14, fontweight='bold')
    
    # Experimental data points
    crop_sizes = [512, 769, 1024]
    
    # Actual experimental results
    pspnet_miou = [57.64, 72.50, None]  # 1024 not tested yet
    deeplabv3_miou = [58.02, 66.57, None]
    segformer_miou = [79.98, None, None]  # Only tested at 512
    segnext_miou = [81.22, None, None]
    
    # Left plot: CNN models
    ax1 = axes[0]
    ax1.plot([512, 769], [57.64, 72.50], 'o-', color='#e74c3c', linewidth=2, 
             markersize=10, label='PSPNet R50')
    ax1.plot([512, 769], [58.02, 66.57], 's-', color='#3498db', linewidth=2,
             markersize=10, label='DeepLabV3+ R50')
    
    # Add annotations
    ax1.annotate('+14.86%', xy=(640, 65), fontsize=11, color='#e74c3c', fontweight='bold')
    ax1.annotate('+8.55%', xy=(640, 59), fontsize=11, color='#3498db', fontweight='bold')
    
    ax1.set_xlabel('Crop Size (pixels)')
    ax1.set_ylabel('mIoU (%)')
    ax1.set_title('CNN-based Models\n(Fixed Spatial Hyperparameters)')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(450, 850)
    ax1.set_ylim(50, 85)
    
    # Right plot: Transformer models (hypothetical curve showing stability)
    ax2 = axes[1]
    
    # Transformers maintain performance (theoretical)
    crop_range = np.linspace(256, 1024, 100)
    segformer_theoretical = 79.98 * np.ones_like(crop_range)  # Stable
    segnext_theoretical = 81.22 * np.ones_like(crop_range)  # Stable
    
    ax2.plot(crop_range, segformer_theoretical, '-', color='#9b59b6', linewidth=2,
             label='SegFormer B3 (theoretical)')
    ax2.plot(crop_range, segnext_theoretical, '-', color='#2ecc71', linewidth=2,
             label='SegNeXt MSCAN-B (theoretical)')
    ax2.plot([512], [79.98], 'o', color='#9b59b6', markersize=10)
    ax2.plot([512], [81.22], 'o', color='#2ecc71', markersize=10)
    
    ax2.set_xlabel('Crop Size (pixels)')
    ax2.set_ylabel('mIoU (%)')
    ax2.set_title('Transformer-based Models\n(Global Self-Attention)')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(200, 1100)
    ax2.set_ylim(50, 85)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/performance_vs_crop_size.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/performance_vs_crop_size.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/performance_vs_crop_size.png")
    plt.close()


def visualize_receptive_field_comparison():
    """
    Compare effective receptive fields of different architectures.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Effective Receptive Field Comparison at 512×512 Input', 
                 fontsize=14, fontweight='bold')
    
    feature_map_size = 64  # 512/8
    
    # PSPNet PPM
    ax1 = axes[0]
    ax1.set_xlim(0, feature_map_size)
    ax1.set_ylim(0, feature_map_size)
    ax1.set_aspect('equal')
    
    # Background
    rect = patches.Rectangle((0, 0), feature_map_size, feature_map_size,
                             facecolor='lightgray', alpha=0.3)
    ax1.add_patch(rect)
    
    # PPM has global pooling (scale-1) which sees everything
    pool_colors = {'1': '#e74c3c', '2': '#3498db', '3': '#2ecc71', '6': '#9b59b6'}
    for scale, color in pool_colors.items():
        s = int(scale)
        cell_size = feature_map_size / s
        for i in range(s):
            for j in range(s):
                rect = patches.Rectangle((j*cell_size, i*cell_size), 
                                        cell_size, cell_size,
                                        linewidth=0.5, edgecolor='white',
                                        facecolor=color, alpha=0.2)
                ax1.add_patch(rect)
    
    ax1.set_title('PSPNet PPM\npool_scales=(1,2,3,6)\n✓ Global context via scale-1')
    ax1.set_xlabel('Feature Map Width')
    ax1.set_ylabel('Feature Map Height')
    
    # DeepLabV3+ ASPP
    ax2 = axes[1]
    ax2.set_xlim(-5, feature_map_size + 5)
    ax2.set_ylim(-5, feature_map_size + 5)
    ax2.set_aspect('equal')
    
    rect = patches.Rectangle((0, 0), feature_map_size, feature_map_size,
                             facecolor='lightgray', alpha=0.3, edgecolor='black', linewidth=2)
    ax2.add_patch(rect)
    
    # Show rate-36 kernel overflow
    center = feature_map_size // 2
    kernel_size = 73  # (3-1)*36+1
    half = kernel_size // 2
    
    # Draw kernel (parts outside FM are overflow)
    rect = patches.Rectangle((center - half, center - half), kernel_size, kernel_size,
                             linewidth=2, edgecolor='#e74c3c', facecolor='#e74c3c', 
                             alpha=0.2, linestyle='--')
    ax2.add_patch(rect)
    
    # Highlight overflow regions
    ax2.axhline(y=0, color='red', linewidth=2)
    ax2.axhline(y=feature_map_size, color='red', linewidth=2)
    ax2.axvline(x=0, color='red', linewidth=2)
    ax2.axvline(x=feature_map_size, color='red', linewidth=2)
    
    ax2.plot(center, center, 'ko', markersize=8)
    ax2.annotate('Overflow\nRegion', xy=(-3, center), fontsize=9, color='red', ha='right')
    
    ax2.set_title('DeepLabV3+ ASPP\nrate-36 kernel: 73×73\n⚠ Overflows 64×64 FM')
    ax2.set_xlabel('Feature Map Width')
    ax2.set_ylabel('Feature Map Height')
    
    # SegFormer Self-Attention
    ax3 = axes[2]
    ax3.set_xlim(0, feature_map_size)
    ax3.set_ylim(0, feature_map_size)
    ax3.set_aspect('equal')
    
    # Global attention - every position attends to every other
    rect = patches.Rectangle((0, 0), feature_map_size, feature_map_size,
                             facecolor='#2ecc71', alpha=0.4, edgecolor='black', linewidth=2)
    ax3.add_patch(rect)
    
    # Draw attention connections from center
    center = feature_map_size // 2
    for i in range(0, feature_map_size, 8):
        for j in range(0, feature_map_size, 8):
            if i == center and j == center:
                continue
            ax3.plot([center, j], [center, i], 'b-', alpha=0.1, linewidth=0.5)
    
    ax3.plot(center, center, 'ro', markersize=10)
    ax3.set_title('SegFormer Self-Attention\nGlobal receptive field\n✓ Scale-invariant')
    ax3.set_xlabel('Feature Map Width')
    ax3.set_ylabel('Feature Map Height')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/receptive_field_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/receptive_field_comparison.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/receptive_field_comparison.png")
    plt.close()


def visualize_modification_hypothesis():
    """
    Visualize the hypothesis: modified parameters should improve 512x512 performance.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Hypothesis: Modified Parameters Should Improve 512×512 Performance', 
                 fontsize=14, fontweight='bold')
    
    # Left: PSPNet PPM comparison
    ax1 = axes[0]
    
    configs = ['Original\n(1,2,3,6)', 'Modified\n(1,2,4,8)']
    cell_sizes_512 = [512/6, 512/8]  # 85.3 vs 64 pixels
    cell_sizes_769 = [769/6, 769/8]  # 128 vs 96 pixels
    
    x = np.arange(len(configs))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, cell_sizes_512, width, label='512×512', color='#e74c3c', alpha=0.7)
    bars2 = ax1.bar(x + width/2, cell_sizes_769, width, label='769×769', color='#3498db', alpha=0.7)
    
    # Add reference line for "optimal" cell size
    ax1.axhline(y=128, color='green', linestyle='--', linewidth=2, label='Target: ~128px (scale-6 @ 769)')
    
    ax1.set_ylabel('Scale-max Cell Size (pixels)')
    ax1.set_title('PSPNet PPM: Finest Grid Cell Size')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs)
    ax1.legend()
    ax1.bar_label(bars1, fmt='%.0f', padding=3)
    ax1.bar_label(bars2, fmt='%.0f', padding=3)
    
    # Annotation
    ax1.annotate('Modified @ 512\napproaches\nOriginal @ 769', 
                xy=(0.7, 90), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Right: DeepLabV3+ ASPP comparison
    ax2 = axes[1]
    
    configs = ['Original\n(1,12,24,36)', 'Modified\n(1,6,12,18)']
    max_kernels = [73, 37]  # (3-1)*36+1 vs (3-1)*18+1
    fm_size = 64
    
    bars = ax2.bar(configs, max_kernels, color=['#e74c3c', '#2ecc71'], alpha=0.7)
    
    # Add reference line for feature map size
    ax2.axhline(y=fm_size, color='blue', linestyle='--', linewidth=2, 
               label=f'Feature Map Size: {fm_size}×{fm_size}')
    
    ax2.set_ylabel('Max Kernel Size (pixels)')
    ax2.set_title('DeepLabV3+ ASPP: Largest Dilated Kernel')
    ax2.legend()
    ax2.bar_label(bars, fmt='%d', padding=3)
    
    # Add overflow indicators
    ax2.annotate('⚠ Overflow!', xy=(0, 73), xytext=(0.3, 80),
                fontsize=11, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red'))
    ax2.annotate('✓ Fits!', xy=(1, 37), xytext=(0.7, 45),
                fontsize=11, color='green', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green'))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/modification_hypothesis.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/modification_hypothesis.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/modification_hypothesis.png")
    plt.close()


def create_summary_figure():
    """
    Create a comprehensive summary figure for the analysis.
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid spec for complex layout
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    
    fig.suptitle('Crop Size Impact on CNN-based Semantic Segmentation: A Mechanistic Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Top row: Experimental results
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_title('A) Experimental Results: Performance vs Crop Size', fontweight='bold')
    
    # Bar chart of results
    models = ['PSPNet\nR50', 'DeepLabV3+\nR50', 'SegFormer\nB3', 'SegNeXt\nMSCAN-B']
    miou_512 = [57.64, 58.02, 79.98, 81.22]
    miou_769 = [72.50, 66.57, None, None]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, miou_512, width, label='512×512', color='#e74c3c', alpha=0.7)
    bars2 = ax1.bar([0, 1], [72.50, 66.57], width, bottom=0, 
                   color='#3498db', alpha=0.7, label='769×769')
    
    # Position 769 bars correctly
    ax1.bar(x[0] + width/2, 72.50, width, color='#3498db', alpha=0.7)
    ax1.bar(x[1] + width/2, 66.57, width, color='#3498db', alpha=0.7)
    
    ax1.set_ylabel('mIoU (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.set_ylim(0, 90)
    ax1.bar_label(bars1, fmt='%.1f', padding=3, fontsize=9)
    
    # Top right: Root cause summary
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.set_title('B) Root Cause: Fixed Spatial Hyperparameters', fontweight='bold')
    ax2.axis('off')
    
    text = """
    PSPNet Pyramid Pooling Module (PPM):
    • pool_scales = (1, 2, 3, 6) - fixed grid sizes
    • At 512×512: scale-6 creates 85×85 pixel cells (too small)
    • At 769×769: scale-6 creates 128×128 pixel cells (optimal)
    
    DeepLabV3+ ASPP:
    • dilations = (1, 12, 24, 36) - fixed dilation rates
    • Rate-36 kernel: 73×73 pixels
    • At 512×512: Feature map = 64×64, kernel OVERFLOWS!
    • At 769×769: Feature map = 96×96, kernel fits
    
    Transformers (SegFormer, SegNeXt):
    • Global self-attention: no fixed spatial hyperparameters
    • Receptive field adapts to any input size
    • Scale-invariant by design
    """
    ax2.text(0.05, 0.95, text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Middle row: Visualizations
    # PPM visualization
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_title('C) PSPNet PPM: Pool Scale-6 Cell Coverage', fontweight='bold')
    
    # Draw two side-by-side feature maps
    for i, (crop, color, label) in enumerate([(512, '#e74c3c', '512×512'), 
                                               (769, '#3498db', '769×769')]):
        offset = i * 100
        fm_size = crop // 8
        scale = 6
        cell_pixels = crop / scale
        
        # Draw outline
        rect = patches.Rectangle((offset, 0), 80, 80,
                                 linewidth=2, edgecolor=color,
                                 facecolor=color, alpha=0.2)
        ax3.add_patch(rect)
        
        # Draw grid
        for j in range(1, scale):
            y = j * 80 / scale
            ax3.axhline(y=y, xmin=offset/200, xmax=(offset+80)/200, 
                       color=color, linewidth=1, alpha=0.5)
            ax3.axvline(x=offset + j * 80 / scale, ymin=0, ymax=80/100,
                       color=color, linewidth=1, alpha=0.5)
        
        ax3.text(offset + 40, -10, f'{label}\nCell: {cell_pixels:.0f}px', 
                ha='center', fontsize=10, color=color)
    
    ax3.set_xlim(-10, 200)
    ax3.set_ylim(-25, 100)
    ax3.axis('off')
    
    # ASPP visualization
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.set_title('D) DeepLabV3+ ASPP: Rate-36 Kernel vs Feature Map', fontweight='bold')
    
    # 512x512 case - overflow
    ax4_left = ax4.inset_axes([0.05, 0.1, 0.4, 0.8])
    ax4_left.set_xlim(-10, 80)
    ax4_left.set_ylim(-10, 80)
    ax4_left.set_aspect('equal')
    
    # Feature map
    rect = patches.Rectangle((0, 0), 64, 64, linewidth=2, edgecolor='black',
                             facecolor='lightblue', alpha=0.3)
    ax4_left.add_patch(rect)
    
    # Kernel (73x73, centered at 32,32)
    rect = patches.Rectangle((32-36, 32-36), 73, 73, linewidth=2, edgecolor='red',
                             facecolor='red', alpha=0.2, linestyle='--')
    ax4_left.add_patch(rect)
    ax4_left.set_title('512×512\n⚠ 73×73 > 64×64', fontsize=10, color='red')
    ax4_left.axis('off')
    
    # 769x769 case - fits
    ax4_right = ax4.inset_axes([0.55, 0.1, 0.4, 0.8])
    ax4_right.set_xlim(-10, 110)
    ax4_right.set_ylim(-10, 110)
    ax4_right.set_aspect('equal')
    
    # Feature map
    rect = patches.Rectangle((0, 0), 96, 96, linewidth=2, edgecolor='black',
                             facecolor='lightgreen', alpha=0.3)
    ax4_right.add_patch(rect)
    
    # Kernel (73x73, centered at 48,48)
    rect = patches.Rectangle((48-36, 48-36), 73, 73, linewidth=2, edgecolor='green',
                             facecolor='green', alpha=0.2)
    ax4_right.add_patch(rect)
    ax4_right.set_title('769×769\n✓ 73×73 < 96×96', fontsize=10, color='green')
    ax4_right.axis('off')
    
    ax4.axis('off')
    
    # Bottom row: Hypothesis and expected results
    ax5 = fig.add_subplot(gs[2, :])
    ax5.set_title('E) Verification Experiment: Modified Hyperparameters at 512×512', fontweight='bold')
    ax5.axis('off')
    
    # Create table of expected results
    table_data = [
        ['Model', 'Original Config', 'Modified Config', 'Expected Improvement'],
        ['PSPNet R50', 'pool_scales=(1,2,3,6)\n57.64% mIoU', 'pool_scales=(1,2,4,8)\nPending...', 
         '~10-15% mIoU gain'],
        ['DeepLabV3+ R50', 'dilations=(1,12,24,36)\n58.02% mIoU', 'dilations=(1,6,12,18)\nPending...',
         '~5-8% mIoU gain']
    ]
    
    table = ax5.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.15, 0.3, 0.3, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style header row
    for j in range(4):
        table[(0, j)].set_facecolor('#4a86e8')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    plt.savefig(f'{OUTPUT_DIR}/crop_size_analysis_summary.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/crop_size_analysis_summary.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/crop_size_analysis_summary.png")
    plt.close()


if __name__ == '__main__':
    print("=" * 60)
    print("Generating Crop Size Impact Visualizations")
    print("=" * 60)
    
    visualize_ppm_pool_scales()
    visualize_aspp_kernel_coverage()
    visualize_performance_vs_crop_size()
    visualize_receptive_field_comparison()
    visualize_modification_hypothesis()
    create_summary_figure()
    
    print("\n" + "=" * 60)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 60)
