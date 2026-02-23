#!/usr/bin/env python3
"""
Generate Domain Gap vs mIoU Gains Plot for Domain Adaptation Study

This script creates a scatter plot showing the trade-off between performance (mIoU)
and robustness (domain gap reduction) relative to baseline.

Usage:
    python analysis_scripts/plot_domain_gap_vs_miou_gains.py
    
Output:
    result_figures/domain_adaptation/domain_gap_vs_miou_gains.png
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'result_figures' / 'domain_adaptation'


def main():
    # Load cleaned data
    csv_path = OUTPUT_DIR / 'domain_adaptation_analysis_cleaned.csv'
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run regenerate_domain_adaptation_figures.py first.")
        return 1
    
    df = pd.read_csv(csv_path)
    
    # Get baseline values
    baseline = df[df['strategy'] == 'baseline'].iloc[0]
    
    # Calculate gains relative to baseline
    df['miou_gain'] = df['overall_mIoU'] - baseline['overall_mIoU']
    df['gap_reduction'] = baseline['domain_gap'] - df['domain_gap']  # Positive = more robust
    
    # Create plot with baseline at center
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color by strategy category
    for i, row in df.iterrows():
        if row['strategy'] == 'baseline':
            color = '#e74c3c'  # Red
            marker = '*'
            size = 300
            zorder = 10
        elif row['strategy'].startswith('std_'):
            color = '#3498db'  # Blue
            marker = 'o'
            size = 150
            zorder = 5
        elif row['strategy'].startswith('gen_'):
            color = '#2ecc71'  # Green
            marker = 's'
            size = 150
            zorder = 5
        else:
            color = '#95a5a6'  # Gray
            marker = '^'
            size = 150
            zorder = 5
        
        ax.scatter(row['gap_reduction'], row['miou_gain'], 
                   c=color, s=size, alpha=0.7, edgecolors='black', linewidth=0.5,
                   marker=marker, zorder=zorder)
    
    # Add labels for all strategies (excluding baseline)
    for i, row in df.iterrows():
        if row['strategy'] != 'baseline':
            label = row['strategy'].replace('gen_', '').replace('std_', '')
            # Adjust position based on crowding
            offset_x = 0.08
            offset_y = 0.1
            ax.annotate(label, 
                        (row['gap_reduction'], row['miou_gain']),
                        xytext=(offset_x, offset_y), textcoords='offset fontsize',
                        fontsize=8, alpha=0.8)
    
    # Draw quadrant lines through origin (baseline)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
    
    # Add quadrant labels with background
    props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
    ax.text(1.0, 1.8, '✓ Better mIoU\n✓ More Robust', fontsize=11, ha='center', va='center',
            color='#27ae60', fontweight='bold', bbox=props)
    ax.text(-0.5, 1.8, '✓ Better mIoU\n✗ Less Robust', fontsize=11, ha='center', va='center',
            color='#f39c12', fontweight='bold', bbox=props)
    ax.text(1.0, -0.8, '✗ Worse mIoU\n✓ More Robust', fontsize=11, ha='center', va='center',
            color='#f39c12', fontweight='bold', bbox=props)
    ax.text(-0.5, -0.8, '✗ Worse mIoU\n✗ Less Robust', fontsize=11, ha='center', va='center',
            color='#e74c3c', fontweight='bold', bbox=props)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#e74c3c', markersize=15, 
               label='Baseline (center)', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=10, 
               label='Standard Augmentation', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#2ecc71', markersize=10, 
               label='Generative Augmentation', markeredgecolor='black', markeredgewidth=0.5),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # Labels and title
    ax.set_xlabel('Domain Gap Reduction (positive = more robust)', fontsize=12)
    ax.set_ylabel('mIoU Gain over Baseline (%)', fontsize=12)
    ax.set_title('Domain Adaptation: Gains over Baseline\n(Baseline at Origin)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Set limits to center the plot nicely
    x_margin = max(abs(df['gap_reduction'].min()), abs(df['gap_reduction'].max())) * 1.2
    y_margin = max(abs(df['miou_gain'].min()), abs(df['miou_gain'].max())) * 1.2
    ax.set_xlim(-x_margin, x_margin)
    ax.set_ylim(-y_margin, y_margin)
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'domain_gap_vs_miou_gains.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'Saved: {output_path}')
    
    # Print summary statistics
    print(f'\n{"="*60}')
    print('Summary Statistics')
    print(f'{"="*60}')
    print(f'Baseline: mIoU={baseline["overall_mIoU"]:.2f}%, Gap={baseline["domain_gap"]:.2f}')
    
    print('\nStrategies in UPPER-RIGHT quadrant (best - better mIoU AND more robust):')
    upper_right = df[(df['miou_gain'] > 0) & (df['gap_reduction'] > 0) & (df['strategy'] != 'baseline')]
    if len(upper_right) > 0:
        for _, row in upper_right.sort_values('miou_gain', ascending=False).iterrows():
            print(f'  {row["strategy"]}: +{row["miou_gain"]:.2f}% mIoU, +{row["gap_reduction"]:.2f} robustness')
    else:
        print('  (none)')
    
    print('\nStrategies in UPPER-LEFT quadrant (better mIoU, less robust):')
    upper_left = df[(df['miou_gain'] > 0) & (df['gap_reduction'] <= 0) & (df['strategy'] != 'baseline')]
    if len(upper_left) > 0:
        for _, row in upper_left.sort_values('miou_gain', ascending=False).head(5).iterrows():
            print(f'  {row["strategy"]}: +{row["miou_gain"]:.2f}% mIoU, {row["gap_reduction"]:.2f} robustness')
    else:
        print('  (none)')
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
