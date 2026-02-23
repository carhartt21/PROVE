#!/usr/bin/env python3
"""
Generate Stage 1 vs Stage 2 comparison figures.
This script creates comprehensive visualizations comparing:
- Overall mIoU by strategy
- Per-dataset performance
- Rank changes
- Domain gap analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

# Paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / 'result_figures' / 'stage_comparison'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Strategy type mapping
STRATEGY_TYPES = {
    'baseline': 'Baseline',
    'std_photometric_distort': 'Augmentation',
    'std_autoaugment': 'Standard Aug',
    'std_cutmix': 'Standard Aug',
    'std_mixup': 'Standard Aug',
    'std_randaugment': 'Standard Aug',
}
# All gen_* strategies are Generative
def get_strategy_type(strategy):
    if strategy.startswith('gen_'):
        return 'Generative'
    return STRATEGY_TYPES.get(strategy, 'Other')

# Colors for strategy types
TYPE_COLORS = {
    'Baseline': '#2c3e50',
    'Standard Aug': '#3498db',
    'Augmentation': '#9b59b6',
    'Generative': '#27ae60',
}

def load_data():
    """Load Stage 1 and Stage 2 results."""
    s1 = pd.read_csv(BASE_DIR / 'downstream_results.csv')
    s2 = pd.read_csv(BASE_DIR / 'downstream_results_stage2.csv')
    return s1, s2

def compute_strategy_stats(df, stage_name):
    """Compute per-strategy statistics."""
    stats = df.groupby('strategy').agg({
        'mIoU': ['mean', 'std', 'count']
    }).reset_index()
    stats.columns = ['strategy', 'mIoU_mean', 'mIoU_std', 'count']
    stats['rank'] = stats['mIoU_mean'].rank(ascending=False).astype(int)
    stats['stage'] = stage_name
    stats['type'] = stats['strategy'].apply(get_strategy_type)
    return stats

def compute_dataset_stats(df, stage_name):
    """Compute per-dataset statistics."""
    stats = df.groupby(['strategy', 'dataset']).agg({
        'mIoU': 'mean'
    }).reset_index()
    stats['stage'] = stage_name
    return stats

def figure1_overall_comparison(s1_stats, s2_stats):
    """Bar chart comparing Stage 1 vs Stage 2 overall mIoU."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Merge stats
    merged = s1_stats.merge(s2_stats, on='strategy', suffixes=('_s1', '_s2'))
    merged = merged.sort_values('mIoU_mean_s1', ascending=False)
    
    x = np.arange(len(merged))
    width = 0.35
    
    # Bars
    bars1 = ax.bar(x - width/2, merged['mIoU_mean_s1'], width, 
                   label='Stage 1 (Clear Day)', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, merged['mIoU_mean_s2'], width,
                   label='Stage 2 (All Domains)', color='#e74c3c', alpha=0.8)
    
    # Error bars
    ax.errorbar(x - width/2, merged['mIoU_mean_s1'], yerr=merged['mIoU_std_s1'],
                fmt='none', color='black', capsize=2, alpha=0.5)
    ax.errorbar(x + width/2, merged['mIoU_mean_s2'], yerr=merged['mIoU_std_s2'],
                fmt='none', color='black', capsize=2, alpha=0.5)
    
    # Labels
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Mean mIoU (%)')
    ax.set_title('Stage 1 vs Stage 2: Overall mIoU Comparison\n(Sorted by Stage 1 Performance)')
    ax.set_xticks(x)
    ax.set_xticklabels(merged['strategy'], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(30, 50)
    
    # Add baseline reference line
    baseline_s1 = merged[merged['strategy'] == 'baseline']['mIoU_mean_s1'].values[0]
    baseline_s2 = merged[merged['strategy'] == 'baseline']['mIoU_mean_s2'].values[0]
    ax.axhline(y=baseline_s1, color='#3498db', linestyle='--', alpha=0.5, label=f'S1 baseline ({baseline_s1:.1f}%)')
    ax.axhline(y=baseline_s2, color='#e74c3c', linestyle='--', alpha=0.5, label=f'S2 baseline ({baseline_s2:.1f}%)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_overall_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig1_overall_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: fig1_overall_comparison.png/pdf")

def figure2_rank_change(s1_stats, s2_stats):
    """Horizontal bar chart showing rank changes between stages."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    merged = s1_stats.merge(s2_stats, on='strategy', suffixes=('_s1', '_s2'))
    merged['rank_change'] = merged['rank_s1'] - merged['rank_s2']
    merged = merged.sort_values('rank_change', ascending=True)
    
    y = np.arange(len(merged))
    colors = ['#27ae60' if x > 0 else '#e74c3c' if x < 0 else '#95a5a6' 
              for x in merged['rank_change']]
    
    bars = ax.barh(y, merged['rank_change'], color=colors, alpha=0.8)
    
    ax.set_yticks(y)
    ax.set_yticklabels(merged['strategy'])
    ax.set_xlabel('Rank Change (positive = improved in Stage 2)')
    ax.set_title('Strategy Rank Changes: Stage 1 → Stage 2')
    ax.axvline(x=0, color='black', linewidth=0.8)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, merged['rank_change'])):
        if val != 0:
            label = f'+{int(val)}' if val > 0 else str(int(val))
            ax.text(val + (0.3 if val > 0 else -0.3), i, label, 
                   va='center', ha='left' if val > 0 else 'right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_rank_change.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig2_rank_change.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: fig2_rank_change.png/pdf")

def figure3_per_dataset(s1, s2):
    """Per-dataset comparison heatmaps."""
    datasets = ['bdd10k', 'iddaw', 'mapillaryvistas', 'outside15k']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    for idx, (df, stage, ax) in enumerate([(s1, 'Stage 1', axes[0]), (s2, 'Stage 2', axes[1])]):
        # Compute mean mIoU per strategy/dataset
        pivot = df.pivot_table(values='mIoU', index='strategy', columns='dataset', aggfunc='mean')
        
        # Sort by mean across datasets
        pivot['mean'] = pivot.mean(axis=1)
        pivot = pivot.sort_values('mean', ascending=False)
        pivot = pivot.drop('mean', axis=1)
        
        # Reorder columns
        pivot = pivot[[d for d in datasets if d in pivot.columns]]
        
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax,
                   vmin=30, vmax=55, cbar_kws={'label': 'mIoU (%)'})
        ax.set_title(f'{stage}: mIoU by Strategy and Dataset')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Strategy')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_per_dataset_heatmap.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig3_per_dataset_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: fig3_per_dataset_heatmap.png/pdf")

def figure4_gain_comparison(s1_stats, s2_stats):
    """Compare gain over baseline between stages."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get baseline values
    baseline_s1 = s1_stats[s1_stats['strategy'] == 'baseline']['mIoU_mean'].values[0]
    baseline_s2 = s2_stats[s2_stats['strategy'] == 'baseline']['mIoU_mean'].values[0]
    
    # Compute gains
    s1_stats = s1_stats.copy()
    s2_stats = s2_stats.copy()
    s1_stats['gain'] = s1_stats['mIoU_mean'] - baseline_s1
    s2_stats['gain'] = s2_stats['mIoU_mean'] - baseline_s2
    
    # Merge and sort
    merged = s1_stats[['strategy', 'gain', 'type']].merge(
        s2_stats[['strategy', 'gain']], on='strategy', suffixes=('_s1', '_s2'))
    merged = merged[merged['strategy'] != 'baseline']
    merged = merged.sort_values('gain_s1', ascending=False)
    
    x = np.arange(len(merged))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, merged['gain_s1'], width,
                   label='Stage 1 Gain', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, merged['gain_s2'], width,
                   label='Stage 2 Gain', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Gain over Baseline (%)')
    ax.set_title('Gain over Baseline: Stage 1 vs Stage 2')
    ax.set_xticks(x)
    ax.set_xticklabels(merged['strategy'], rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0, color='black', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_gain_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig4_gain_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: fig4_gain_comparison.png/pdf")

def figure5_scatter_stages(s1_stats, s2_stats):
    """Scatter plot of Stage 1 vs Stage 2 performance."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    merged = s1_stats.merge(s2_stats, on='strategy', suffixes=('_s1', '_s2'))
    
    # Color by type
    for stype in TYPE_COLORS:
        mask = merged['type_s1'] == stype
        if mask.any():
            ax.scatter(merged.loc[mask, 'mIoU_mean_s1'], 
                      merged.loc[mask, 'mIoU_mean_s2'],
                      c=TYPE_COLORS[stype], label=stype, s=100, alpha=0.7)
    
    # Add diagonal line (y=x would mean same performance)
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.3, label='Same performance')
    
    # Add labels for outliers
    for _, row in merged.iterrows():
        if abs(row['rank_s1'] - row['rank_s2']) > 10:  # Big rank changes
            ax.annotate(row['strategy'].replace('gen_', '').replace('std_', ''),
                       (row['mIoU_mean_s1'], row['mIoU_mean_s2']),
                       fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Stage 1 mIoU (%)')
    ax.set_ylabel('Stage 2 mIoU (%)')
    ax.set_title('Stage 1 vs Stage 2 Performance by Strategy')
    ax.legend(loc='lower right')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_scatter_stages.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig5_scatter_stages.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: fig5_scatter_stages.png/pdf")

def figure6_improvement_by_type(s1_stats, s2_stats):
    """Box plot showing improvement by strategy type."""
    merged = s1_stats.merge(s2_stats, on='strategy', suffixes=('_s1', '_s2'))
    merged['improvement'] = merged['mIoU_mean_s2'] - merged['mIoU_mean_s1']
    merged['type'] = merged['type_s1']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    types_order = ['Baseline', 'Standard Aug', 'Augmentation', 'Generative']
    data_by_type = [merged[merged['type'] == t]['improvement'].values for t in types_order]
    
    bp = ax.boxplot(data_by_type, labels=types_order, patch_artist=True)
    
    for patch, stype in zip(bp['boxes'], types_order):
        patch.set_facecolor(TYPE_COLORS[stype])
        patch.set_alpha(0.7)
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Strategy Type')
    ax.set_ylabel('mIoU Improvement (Stage 2 - Stage 1) %')
    ax.set_title('Performance Improvement by Strategy Type')
    
    # Add individual points
    for i, (t, data) in enumerate(zip(types_order, data_by_type)):
        x = np.random.normal(i+1, 0.04, len(data))
        ax.scatter(x, data, alpha=0.5, color=TYPE_COLORS[t], s=30)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_improvement_by_type.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig6_improvement_by_type.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: fig6_improvement_by_type.png/pdf")

def main():
    print("="*60)
    print("STAGE COMPARISON FIGURE GENERATOR")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    s1, s2 = load_data()
    print(f"  Stage 1: {len(s1)} results")
    print(f"  Stage 2: {len(s2)} results")
    
    # Compute stats
    s1_stats = compute_strategy_stats(s1, 'Stage 1')
    s2_stats = compute_strategy_stats(s2, 'Stage 2')
    
    print(f"\nGenerating figures to: {OUTPUT_DIR}")
    
    # Generate figures
    figure1_overall_comparison(s1_stats, s2_stats)
    figure2_rank_change(s1_stats, s2_stats)
    figure3_per_dataset(s1, s2)
    figure4_gain_comparison(s1_stats, s2_stats)
    figure5_scatter_stages(s1_stats, s2_stats)
    figure6_improvement_by_type(s1_stats, s2_stats)
    
    print(f"\nDone! All figures saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
