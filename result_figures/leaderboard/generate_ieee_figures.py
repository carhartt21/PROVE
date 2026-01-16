#!/usr/bin/env python3
"""
IEEE Publication-Ready Figures Generator for Strategy Leaderboard

Generates high-quality figures for the PROVE strategy leaderboard analysis.
Follows IEEE formatting guidelines with proper fonts, sizes, and color schemes.

Author: PROVE Project
Date: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# IEEE Styling Configuration
# =============================================================================

IEEE_SINGLE_COL = 3.5
IEEE_DOUBLE_COL = 7.16

FONT_SIZE_TITLE = 10
FONT_SIZE_LABEL = 9
FONT_SIZE_TICK = 8
FONT_SIZE_LEGEND = 8
FONT_SIZE_ANNOTATION = 7

# Strategy type colors
COLORS_TYPE = {
    'Baseline Clear Day': '#666666',
    'Baseline Full': '#333333',
    'Generative': '#2E86AB',
    'Standard Aug': '#E07A5F',
    'Augmentation': '#3D405B'
}

# Top strategies for highlighting
TOP_STRATEGIES_COLORS = {
    'gen_automold': '#2E86AB',
    'gen_NST': '#457B9D',
    'photometric_distort': '#3D405B',
    'gen_SUSTechGAN': '#1D3557',
    'std_randaugment': '#E07A5F',
    'baseline': '#666666'
}

def setup_ieee_style():
    """Configure matplotlib for IEEE publication style."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': FONT_SIZE_TICK,
        'axes.titlesize': FONT_SIZE_TITLE,
        'axes.labelsize': FONT_SIZE_LABEL,
        'xtick.labelsize': FONT_SIZE_TICK,
        'ytick.labelsize': FONT_SIZE_TICK,
        'legend.fontsize': FONT_SIZE_LEGEND,
        'figure.titlesize': FONT_SIZE_TITLE,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.linewidth': 0.5,
        'axes.grid': False,
        'grid.linewidth': 0.3,
        'lines.linewidth': 1.0,
        'patch.linewidth': 0.5,
        'legend.frameon': False,
    })

def load_data(base_path):
    """Load all CSV data files."""
    data = {}
    data['leaderboard_full'] = pd.read_csv(base_path / 'strategy_leaderboard_full.csv')
    data['leaderboard_clear'] = pd.read_csv(base_path / 'strategy_leaderboard_clear_day.csv')
    data['per_domain_full'] = pd.read_csv(base_path / 'per_domain_gains_full.csv')
    data['per_dataset_full'] = pd.read_csv(base_path / 'per_dataset_gains_full.csv')
    return data

def format_strategy_name(name):
    """Format strategy name for display."""
    replacements = {
        'gen_': '',
        'std_': '',
        'baseline_clear_day': 'Baseline (Clear Day)',
        'baseline': 'Baseline (Full)',
        'photometric_distort': 'Photometric',
        '_': ' '
    }
    result = name
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result.title()

# =============================================================================
# Figure 1: Strategy Leaderboard (Top 15)
# =============================================================================

def create_leaderboard_figure(data, output_path):
    """Create horizontal bar chart of top strategies by overall mIoU gain."""
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 3.2))
    
    df = data['leaderboard_full'].copy()
    # Filter out incomplete results and sort by gain
    df = df[df['Num Results'] >= 10].head(15)
    df = df.sort_values('Gain vs Clear Day', ascending=True)
    
    y = np.arange(len(df))
    colors = [COLORS_TYPE.get(t, '#888888') for t in df['Type']]
    
    bars = ax.barh(y, df['Gain vs Clear Day'], color=colors, edgecolor='black', linewidth=0.3)
    
    # Add value annotations
    for i, (bar, val) in enumerate(zip(bars, df['Gain vs Clear Day'])):
        offset = 0.1 if val >= 0 else -0.3
        ha = 'left' if val >= 0 else 'right'
        ax.annotate(f'{val:+.1f}%', xy=(val + offset, i), va='center', ha=ha,
                   fontsize=FONT_SIZE_ANNOTATION)
    
    ax.set_yticks(y)
    ax.set_yticklabels([format_strategy_name(s) for s in df['Strategy']])
    ax.set_xlabel('mIoU Gain vs Clear Day Baseline (%)')
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-1, 4)
    
    # Legend
    legend_handles = [
        mpatches.Patch(color=COLORS_TYPE['Generative'], label='Generative'),
        mpatches.Patch(color=COLORS_TYPE['Standard Aug'], label='Standard Aug'),
        mpatches.Patch(color=COLORS_TYPE['Augmentation'], label='Photometric'),
        mpatches.Patch(color=COLORS_TYPE['Baseline Full'], label='Baseline'),
    ]
    ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.15),
              ncol=4, fontsize=FONT_SIZE_LEGEND-1, frameon=False)
    
    ax.set_title('Strategy Ranking by Overall mIoU Gain', fontsize=FONT_SIZE_TITLE, pad=25)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    fig.savefig(output_path / 'fig1_strategy_leaderboard.png', dpi=300)
    plt.close(fig)
    print("✓ Figure 1: Strategy Leaderboard saved")

# =============================================================================
# Figure 2: Normal vs Adverse Gain Comparison
# =============================================================================

def create_normal_adverse_comparison(data, output_path):
    """Create scatter plot comparing normal vs adverse gains."""
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, IEEE_SINGLE_COL))
    
    df = data['leaderboard_full'].copy()
    df = df[df['Num Results'] >= 10]
    df = df.dropna(subset=['Normal Gain', 'Adverse Gain'])
    
    # Color by type
    for stype, color in COLORS_TYPE.items():
        mask = df['Type'] == stype
        if mask.any():
            ax.scatter(df[mask]['Normal Gain'], df[mask]['Adverse Gain'], 
                      c=color, label=stype, s=40, alpha=0.8, edgecolors='black', linewidth=0.3)
    
    # Add labels for top strategies
    top_strategies = ['gen_automold', 'gen_NST', 'photometric_distort', 'baseline', 
                     'gen_StyleID', 'gen_CUT', 'std_randaugment']
    for _, row in df.iterrows():
        if row['Strategy'] in top_strategies:
            ax.annotate(format_strategy_name(row['Strategy']), 
                       xy=(row['Normal Gain'], row['Adverse Gain']),
                       xytext=(3, 3), textcoords='offset points',
                       fontsize=FONT_SIZE_ANNOTATION-1, alpha=0.8)
    
    # Reference lines
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Diagonal line (equal improvement)
    lims = [-2, 7]
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel('Normal Conditions Gain (%)')
    ax.set_ylabel('Adverse Conditions Gain (%)')
    ax.set_xlim(-2, 6)
    ax.set_ylim(-1, 7)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=3, 
              fontsize=FONT_SIZE_LEGEND-1, frameon=False)
    
    ax.set_title('Normal vs Adverse Performance Gains', fontsize=FONT_SIZE_TITLE, pad=25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    
    fig.savefig(output_path / 'fig2_normal_adverse_scatter.png', dpi=300)
    plt.close(fig)
    print("✓ Figure 2: Normal vs Adverse Comparison saved")

# =============================================================================
# Figure 3: Per-Domain Performance Gains (Top Strategies)
# =============================================================================

def create_domain_gains_heatmap(data, output_path):
    """Create heatmap showing per-domain gains for top strategies."""
    fig, ax = plt.subplots(figsize=(IEEE_DOUBLE_COL, 2.8))
    
    df = data['per_domain_full'].copy()
    
    # Select top strategies (excluding problematic ones)
    top_strategies = ['baseline', 'gen_automold', 'gen_NST', 'photometric_distort', 
                      'gen_SUSTechGAN', 'gen_UniControl', 'std_randaugment', 'gen_LANIT',
                      'gen_TSIT', 'gen_StyleID']
    
    df_top = df[df['Strategy'].isin(top_strategies)].copy()
    df_top = df_top.set_index('Strategy').loc[top_strategies]
    
    domains = ['clear_day', 'cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy']
    heatmap_data = df_top[domains].values
    
    # Create heatmap
    cmap = LinearSegmentedColormap.from_list('gains', ['#E76F51', '#F4F1DE', '#2A9D8F'])
    im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=-5, vmax=10)
    
    # Add annotations
    for i in range(len(df_top)):
        for j in range(len(domains)):
            val = heatmap_data[i, j]
            color = 'white' if abs(val) > 6 else 'black'
            ax.text(j, i, f'{val:+.1f}', ha='center', va='center',
                   fontsize=FONT_SIZE_ANNOTATION-1, color=color)
    
    ax.set_xticks(np.arange(len(domains)))
    ax.set_yticks(np.arange(len(df_top)))
    domain_labels = ['Clear', 'Cloudy', 'Dawn', 'Foggy', 'Night', 'Rainy', 'Snowy']
    ax.set_xticklabels(domain_labels, rotation=45, ha='right')
    ax.set_yticklabels([format_strategy_name(s) for s in df_top.index])
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('mIoU Gain (%)', fontsize=FONT_SIZE_LABEL)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)
    
    ax.set_title('Per-Domain Performance Gains vs Clear Day Baseline', fontsize=FONT_SIZE_TITLE, pad=10)
    
    plt.tight_layout()
    
    fig.savefig(output_path / 'fig3_domain_gains_heatmap.png', dpi=300)
    plt.close(fig)
    print("✓ Figure 3: Domain Gains Heatmap saved")

# =============================================================================
# Figure 4: Per-Dataset Performance Gains
# =============================================================================

def create_dataset_gains_figure(data, output_path):
    """Create grouped bar chart of per-dataset gains for top strategies."""
    fig, ax = plt.subplots(figsize=(IEEE_DOUBLE_COL, 2.5))
    
    df = data['per_dataset_full'].copy()
    
    # Select top strategies
    top_strategies = ['baseline', 'gen_automold', 'gen_NST', 'photometric_distort', 
                      'gen_SUSTechGAN', 'std_randaugment', 'std_mixup']
    
    df_top = df[df['Strategy'].isin(top_strategies)].copy()
    df_top = df_top.set_index('Strategy').loc[top_strategies]
    
    datasets = ['bdd10k', 'idd-aw', 'mapillaryvistas', 'outside15k']
    dataset_labels = ['BDD10K', 'IDD-AW', 'Mapillary', 'Outside15K']
    
    x = np.arange(len(datasets))
    width = 0.12
    n_strategies = len(df_top)
    
    colors = ['#333333', '#2E86AB', '#457B9D', '#3D405B', '#1D3557', '#E07A5F', '#F4A261']
    
    for i, (strategy, row) in enumerate(df_top.iterrows()):
        values = [row[ds] if pd.notna(row[ds]) else 0 for ds in datasets]
        offset = (i - n_strategies/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=format_strategy_name(strategy),
                     color=colors[i], edgecolor='black', linewidth=0.3)
    
    ax.set_ylabel('mIoU Gain (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=4,
              fontsize=FONT_SIZE_LEGEND-1, frameon=False)
    
    ax.set_title('Per-Dataset Performance Gains vs Clear Day Baseline', fontsize=FONT_SIZE_TITLE, pad=30)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.78)
    
    fig.savefig(output_path / 'fig4_dataset_gains.png', dpi=300)
    plt.close(fig)
    print("✓ Figure 4: Dataset Gains saved")

# =============================================================================
# Figure 5: Full vs Clear Day Training Comparison
# =============================================================================

def create_training_type_comparison(data, output_path):
    """Compare full-trained vs clear-day-trained strategies."""
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 3.0))
    
    df_full = data['leaderboard_full'].copy()
    df_clear = data['leaderboard_clear'].copy()
    
    # Get common strategies
    common = set(df_full['Strategy']) & set(df_clear['Strategy'])
    common = [s for s in common if s not in ['baseline_clear_day', 'baseline']]
    
    # Select top strategies
    top_strategies = ['gen_automold', 'gen_NST', 'photometric_distort', 'gen_SUSTechGAN',
                      'gen_UniControl', 'std_randaugment', 'gen_LANIT', 'gen_TSIT']
    top_strategies = [s for s in top_strategies if s in common]
    
    full_gains = []
    clear_gains = []
    
    for s in top_strategies:
        full_row = df_full[df_full['Strategy'] == s]
        clear_row = df_clear[df_clear['Strategy'] == s]
        if len(full_row) > 0 and len(clear_row) > 0:
            full_gains.append(full_row['Gain vs Clear Day'].values[0])
            clear_gains.append(clear_row['Gain vs Clear Day'].values[0])
    
    y = np.arange(len(top_strategies))
    height = 0.35
    
    bars1 = ax.barh(y - height/2, full_gains, height, label='Full Training',
                    color='#2A9D8F', edgecolor='black', linewidth=0.3)
    bars2 = ax.barh(y + height/2, clear_gains, height, label='Clear Day Training',
                    color='#E76F51', edgecolor='black', linewidth=0.3)
    
    ax.set_yticks(y)
    ax.set_yticklabels([format_strategy_name(s) for s in top_strategies])
    ax.set_xlabel('mIoU Gain vs Baseline (%)')
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2,
              fontsize=FONT_SIZE_LEGEND, frameon=False)
    
    ax.set_title('Training Strategy Impact on Augmentation Gains', fontsize=FONT_SIZE_TITLE, pad=20)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    fig.savefig(output_path / 'fig5_training_type_comparison.png', dpi=300)
    plt.close(fig)
    print("✓ Figure 5: Training Type Comparison saved")

# =============================================================================
# Figure 6: Domain Gap Reduction
# =============================================================================

def create_gap_reduction_figure(data, output_path):
    """Create figure showing domain gap reduction by strategy."""
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 3.0))
    
    df = data['leaderboard_full'].copy()
    df = df[df['Num Results'] >= 10]
    df = df.dropna(subset=['Gap Reduction vs Clear Day'])
    
    # Sort by gap reduction
    df = df.sort_values('Gap Reduction vs Clear Day', ascending=True).head(15)
    
    y = np.arange(len(df))
    colors = [COLORS_TYPE.get(t, '#888888') for t in df['Type']]
    
    bars = ax.barh(y, df['Gap Reduction vs Clear Day'], color=colors, 
                   edgecolor='black', linewidth=0.3)
    
    # Add value annotations
    for i, (bar, val) in enumerate(zip(bars, df['Gap Reduction vs Clear Day'])):
        offset = 0.1 if val >= 0 else -0.3
        ha = 'left' if val >= 0 else 'right'
        ax.annotate(f'{val:+.1f}%', xy=(val + offset, i), va='center', ha=ha,
                   fontsize=FONT_SIZE_ANNOTATION)
    
    ax.set_yticks(y)
    ax.set_yticklabels([format_strategy_name(s) for s in df['Strategy']])
    ax.set_xlabel('Domain Gap Reduction (%)')
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_title('Domain Gap Reduction vs Baseline', fontsize=FONT_SIZE_TITLE, pad=10)
    
    plt.tight_layout()
    
    fig.savefig(output_path / 'fig6_gap_reduction.png', dpi=300)
    plt.close(fig)
    print("✓ Figure 6: Gap Reduction saved")

# =============================================================================
# Figure 7: Strategy Type Comparison
# =============================================================================

def create_strategy_type_summary(data, output_path):
    """Create summary figure comparing strategy types."""
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_DOUBLE_COL, 2.5))
    
    df = data['leaderboard_full'].copy()
    df = df[df['Num Results'] >= 10]
    df = df.dropna(subset=['Overall mIoU', 'Adverse Gain'])
    
    # Group by type
    types = ['Generative', 'Standard Aug', 'Augmentation', 'Baseline Full']
    type_stats = []
    
    for t in types:
        mask = df['Type'] == t
        if mask.any():
            type_stats.append({
                'Type': t,
                'Overall mIoU': df[mask]['Overall mIoU'].mean(),
                'Overall Std': df[mask]['Overall mIoU'].std(),
                'Adverse Gain': df[mask]['Adverse Gain'].mean(),
                'Adverse Std': df[mask]['Adverse Gain'].std(),
                'Count': mask.sum()
            })
    
    type_df = pd.DataFrame(type_stats)
    
    # Panel A: Overall mIoU
    ax1 = axes[0]
    x = np.arange(len(type_df))
    bars = ax1.bar(x, type_df['Overall mIoU'], yerr=type_df['Overall Std'], 
                   capsize=3, color=[COLORS_TYPE.get(t, '#888') for t in type_df['Type']],
                   edgecolor='black', linewidth=0.3, error_kw={'linewidth': 0.8})
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Generative', 'Std Aug', 'Photom.', 'Baseline'], rotation=0)
    ax1.set_ylabel('Overall mIoU (%)')
    ax1.set_ylim(58, 63)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title('(a) Overall Performance', fontsize=FONT_SIZE_TITLE)
    
    # Panel B: Adverse Gain
    ax2 = axes[1]
    bars = ax2.bar(x, type_df['Adverse Gain'], yerr=type_df['Adverse Std'],
                   capsize=3, color=[COLORS_TYPE.get(t, '#888') for t in type_df['Type']],
                   edgecolor='black', linewidth=0.3, error_kw={'linewidth': 0.8})
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Generative', 'Std Aug', 'Photom.', 'Baseline'], rotation=0)
    ax2.set_ylabel('Adverse Gain (%)')
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('(b) Adverse Weather Improvement', fontsize=FONT_SIZE_TITLE)
    
    plt.tight_layout()
    
    fig.savefig(output_path / 'fig7_strategy_type_summary.png', dpi=300)
    plt.close(fig)
    print("✓ Figure 7: Strategy Type Summary saved")

# =============================================================================
# Figure 8: Top 5 Strategy Comparison
# =============================================================================

def create_top5_comparison(data, output_path):
    """Create detailed comparison of top 5 strategies."""
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 2.8))
    
    df = data['leaderboard_full'].copy()
    df = df[df['Num Results'] >= 10]
    df = df.dropna(subset=['Normal Gain', 'Adverse Gain', 'Gap Reduction vs Clear Day'])
    
    # Top 5 by overall gain
    top5 = df.nlargest(5, 'Gain vs Clear Day')
    
    metrics = ['Normal Gain', 'Adverse Gain', 'Gap Reduction vs Clear Day']
    metric_labels = ['Normal', 'Adverse', 'Gap Red.']
    
    x = np.arange(len(metrics))
    width = 0.15
    
    colors = ['#2E86AB', '#457B9D', '#3D405B', '#E07A5F', '#F4A261']
    
    for i, (_, row) in enumerate(top5.iterrows()):
        values = [row[m] for m in metrics]
        offset = (i - 2) * width
        ax.bar(x + offset, values, width, label=format_strategy_name(row['Strategy']),
               color=colors[i], edgecolor='black', linewidth=0.3)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel('Gain (%)')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2,
              fontsize=FONT_SIZE_LEGEND-1, frameon=False)
    
    ax.set_title('Top 5 Strategies: Detailed Comparison', fontsize=FONT_SIZE_TITLE, pad=35)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.75)
    
    fig.savefig(output_path / 'fig8_top5_comparison.png', dpi=300)
    plt.close(fig)
    print("✓ Figure 8: Top 5 Comparison saved")

# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Generate all IEEE publication-ready figures for leaderboard."""
    print("=" * 60)
    print("IEEE Publication Figure Generator - Strategy Leaderboard")
    print("=" * 60)
    
    base_path = Path(__file__).parent
    output_path = base_path / 'ieee_figures'
    output_path.mkdir(exist_ok=True)
    
    setup_ieee_style()
    
    print("\nLoading data...")
    data = load_data(base_path)
    print(f"  - Full leaderboard: {len(data['leaderboard_full'])} strategies")
    print(f"  - Clear day leaderboard: {len(data['leaderboard_clear'])} strategies")
    
    print("\nGenerating figures...")
    print("-" * 40)
    
    create_leaderboard_figure(data, output_path)
    create_normal_adverse_comparison(data, output_path)
    create_domain_gains_heatmap(data, output_path)
    create_dataset_gains_figure(data, output_path)
    create_training_type_comparison(data, output_path)
    create_gap_reduction_figure(data, output_path)
    create_strategy_type_summary(data, output_path)
    create_top5_comparison(data, output_path)
    
    print("-" * 40)
    print(f"\n✓ All figures saved to: {output_path}")
    print("\nGenerated files:")
    for f in sorted(output_path.glob('fig*.png')):
        print(f"  - {f.name}")
    
    print("\n" + "=" * 60)
    print("Figure generation complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()
