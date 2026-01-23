#!/usr/bin/env python3
"""
IEEE Publication-Ready Figures Generator for Baseline Analysis

Generates high-quality figures suitable for IEEE journal publications.
Follows IEEE formatting guidelines with proper fonts, sizes, and color schemes.

Author: PROVE Project
Date: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# IEEE Styling Configuration
# =============================================================================

# IEEE recommends figures that fit in one or two columns
# Single column: 3.5 inches (88.9 mm)
# Double column: 7.16 inches (181.9 mm)

IEEE_SINGLE_COL = 3.5
IEEE_DOUBLE_COL = 7.16
IEEE_TEXT_WIDTH = 7.16

# Font sizes for IEEE (generally 8-10 pt for figures)
FONT_SIZE_TITLE = 10
FONT_SIZE_LABEL = 9
FONT_SIZE_TICK = 8
FONT_SIZE_LEGEND = 8
FONT_SIZE_ANNOTATION = 7

# Color palettes - using colorblind-friendly colors
COLORS_MODELS = {
    'deeplabv3plus_r50': '#2E86AB',    # Deep blue
    'pspnet_r50': '#E07A5F',            # Terracotta
    'segformer_mit-b5': '#3D405B'       # Dark slate
}

COLORS_DATASETS = {
    'bdd10k': '#457B9D',
    'idd-aw': '#E63946', 
    'mapillaryvistas': '#2A9D8F',
    'outside15k': '#F4A261'
}

COLORS_DOMAINS = {
    'clear_day': '#2E86AB',
    'cloudy': '#A8DADC',
    'dawn_dusk': '#F4A261',
    'foggy': '#E9C46A',
    'night': '#264653',
    'rainy': '#457B9D',
    'snowy': '#E9ECEF'
}

COLORS_CONDITION = {
    'normal': '#2A9D8F',
    'adverse': '#E76F51'
}

# Model display names (cleaner for figures)
MODEL_NAMES = {
    'deeplabv3plus_r50': 'DeepLabV3+',
    'pspnet_r50': 'PSPNet',
    'segformer_mit-b5': 'SegFormer'
}

DATASET_NAMES = {
    'bdd10k': 'BDD10K',
    'idd-aw': 'IDD-AW',
    'mapillaryvistas': 'Mapillary',
    'outside15k': 'Outside15K'
}

DOMAIN_NAMES = {
    'clear_day': 'Clear Day',
    'cloudy': 'Cloudy',
    'dawn_dusk': 'Dawn/Dusk',
    'foggy': 'Foggy',
    'night': 'Night',
    'rainy': 'Rainy',
    'snowy': 'Snowy'
}

# Define normal vs adverse domains
NORMAL_DOMAINS = {'clear_day', 'cloudy', 'foggy'}
ADVERSE_DOMAINS = {'dawn_dusk', 'night', 'rainy', 'snowy'}

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
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'legend.frameon': False,
        'legend.borderpad': 0.3,
        'legend.handletextpad': 0.3,
        'text.usetex': False,  # Set True if LaTeX is available
    })

def load_data(base_path):
    """Load all CSV data files.
    
    Updated for new naming convention:
    - stage1_baseline_* : Clear day training (Stage 1)
    - stage2_baseline_* : All domains training (Stage 2)
    
    Falls back to old naming (full_baseline_*, clear_day_baseline_*) for compatibility.
    """
    data = {}
    
    # Try new naming convention first, then fall back to old
    # Stage 1 = Clear day training (was "clear_day_baseline")
    stage1_domain_path = base_path / 'stage1_baseline_per_domain.csv'
    if not stage1_domain_path.exists():
        stage1_domain_path = base_path / 'clear_day_baseline_per_domain.csv'
    if stage1_domain_path.exists():
        data['clear_domain'] = pd.read_csv(stage1_domain_path)
    
    # Stage 2 = All domains training (was "full_baseline")
    stage2_domain_path = base_path / 'stage2_baseline_per_domain.csv'
    if not stage2_domain_path.exists():
        stage2_domain_path = base_path / 'full_baseline_per_domain.csv'
    if stage2_domain_path.exists():
        data['full_domain'] = pd.read_csv(stage2_domain_path)
    
    # Config files
    stage1_config_path = base_path / 'stage1_baseline_per_config.csv'
    if not stage1_config_path.exists():
        stage1_config_path = base_path / 'full_baseline_per_config.csv'
    if stage1_config_path.exists():
        data['config'] = pd.read_csv(stage1_config_path)
    
    # Also load Stage 2 config if available
    stage2_config_path = base_path / 'stage2_baseline_per_config.csv'
    if stage2_config_path.exists():
        data['config_stage2'] = pd.read_csv(stage2_config_path)
    
    return data

# =============================================================================
# Figure 1: Model Architecture Comparison
# =============================================================================

def create_model_comparison_figure(data, output_path):
    """
    Create a grouped bar chart comparing model architectures across datasets.
    Shows overall mIoU performance.
    """
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 2.4))
    
    config_df = data['config']
    
    # Prepare data
    datasets = ['bdd10k', 'idd-aw', 'mapillaryvistas', 'outside15k']
    models = ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5']
    
    x = np.arange(len(datasets))
    width = 0.25
    
    for i, model in enumerate(models):
        model_data = config_df[config_df['model'] == model]
        values = [model_data[model_data['dataset'] == ds]['overall_mIoU'].values[0] 
                  for ds in datasets]
        
        bars = ax.bar(x + i * width, values, width, 
                      label=MODEL_NAMES[model],
                      color=COLORS_MODELS[model],
                      edgecolor='black',
                      linewidth=0.3)
    
    ax.set_ylabel('mIoU (%)')
    ax.set_xticks(x + width)
    ax.set_xticklabels([DATASET_NAMES[d] for d in datasets])
    # Legend outside plot area at top
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3, 
              frameon=False, fontsize=FONT_SIZE_LEGEND-0.5, columnspacing=0.8)
    ax.set_ylim(0, 90)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    
    # Save PNG only
    fig.savefig(output_path / 'fig1_model_comparison.png', dpi=300)
    plt.close(fig)
    
    print("✓ Figure 1: Model Architecture Comparison saved")

# =============================================================================
# Figure 2: Domain Gap Heatmap
# =============================================================================

def create_domain_gap_heatmap(data, output_path):
    """
    Create a heatmap showing domain gap (Normal - Adverse) across configurations.
    """
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 2.5))
    
    config_df = data['config']
    
    # Pivot for heatmap
    datasets = ['bdd10k', 'idd-aw', 'mapillaryvistas', 'outside15k']
    models = ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5']
    
    heatmap_data = np.zeros((len(datasets), len(models)))
    
    for i, ds in enumerate(datasets):
        for j, model in enumerate(models):
            row = config_df[(config_df['dataset'] == ds) & (config_df['model'] == model)]
            if len(row) > 0:
                heatmap_data[i, j] = row['domain_gap'].values[0]
    
    # Create custom colormap (green = small gap, red = large gap)
    cmap = LinearSegmentedColormap.from_list('gap', ['#2A9D8F', '#F4F1DE', '#E76F51'])
    
    im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=-5, vmax=17)
    
    # Add text annotations
    for i in range(len(datasets)):
        for j in range(len(models)):
            val = heatmap_data[i, j]
            color = 'white' if abs(val) > 10 else 'black'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center', 
                   fontsize=FONT_SIZE_ANNOTATION, color=color, fontweight='bold')
    
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_xticklabels([MODEL_NAMES[m] for m in models], rotation=45, ha='right')
    ax.set_yticklabels([DATASET_NAMES[d] for d in datasets])
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Domain Gap (%)', fontsize=FONT_SIZE_LABEL)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)
    
    ax.set_title('Performance Gap: Normal vs Adverse', fontsize=FONT_SIZE_TITLE, pad=10)
    
    plt.tight_layout()
    
    fig.savefig(output_path / 'fig2_domain_gap_heatmap.png', dpi=300)
    plt.close(fig)
    
    print("✓ Figure 2: Domain Gap Heatmap saved")

# =============================================================================
# Figure 3: Weather Domain Performance (Radar/Spider Chart)
# =============================================================================

def create_domain_radar_chart(data, output_path):
    """
    Create a radar chart showing model performance across weather domains.
    Averaged across datasets.
    """
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, IEEE_SINGLE_COL), 
                           subplot_kw=dict(projection='polar'))
    
    domain_df = data['full_domain']
    
    # Filter domains with sufficient samples (exclude foggy due to anomalies)
    domains = ['clear_day', 'cloudy', 'dawn_dusk', 'night', 'rainy', 'snowy']
    models = ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5']
    
    # Calculate average mIoU per model per domain (excluding small sample sizes)
    domain_df_filtered = domain_df[domain_df['num_images'] >= 50]
    
    angles = np.linspace(0, 2*np.pi, len(domains), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    for model in models:
        values = []
        for domain in domains:
            model_domain = domain_df_filtered[(domain_df_filtered['model'] == model) & 
                                              (domain_df_filtered['domain'] == domain)]
            if len(model_domain) > 0:
                values.append(model_domain['mIoU'].mean())
            else:
                values.append(0)
        values += values[:1]  # Complete the loop
        
        ax.plot(angles, values, 'o-', linewidth=1.2, label=MODEL_NAMES[model],
               color=COLORS_MODELS[model], markersize=3)
        ax.fill(angles, values, alpha=0.15, color=COLORS_MODELS[model])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([DOMAIN_NAMES[d] for d in domains], fontsize=FONT_SIZE_TICK)
    
    ax.set_ylim(30, 85)
    ax.set_yticks([40, 50, 60, 70, 80])
    ax.set_yticklabels(['40', '50', '60', '70', '80'], fontsize=FONT_SIZE_ANNOTATION)
    
    # Legend below the chart
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, 
              frameon=False, fontsize=FONT_SIZE_LEGEND)
    ax.set_title('Performance Across Weather Domains', fontsize=FONT_SIZE_TITLE, pad=12)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    
    fig.savefig(output_path / 'fig3_domain_radar.png', dpi=300)
    plt.close(fig)
    
    print("✓ Figure 3: Domain Radar Chart saved")

# =============================================================================
# Figure 4: Training Strategy Comparison (Full vs Clear Day)
# =============================================================================

def create_training_comparison_figure(data, output_path):
    """
    Compare Full baseline (trained on all domains) vs Clear Day baseline.
    Shows the impact of diverse training data.
    """
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_DOUBLE_COL, 2.5))
    
    full_df = data['full_domain']
    clear_df = data['clear_domain']
    
    # Filter for sufficient samples
    full_df = full_df[full_df['num_images'] >= 50]
    clear_df = clear_df[clear_df['num_images'] >= 50]
    
    # Calculate averages
    domains = ['clear_day', 'cloudy', 'dawn_dusk', 'night', 'rainy', 'snowy']
    
    # Panel A: Per-domain comparison
    ax1 = axes[0]
    
    full_means = [full_df[full_df['domain'] == d]['mIoU'].mean() for d in domains]
    clear_means = [clear_df[clear_df['domain'] == d]['mIoU'].mean() for d in domains]
    
    x = np.arange(len(domains))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, full_means, width, label='Full Training',
                    color='#2A9D8F', edgecolor='black', linewidth=0.3)
    bars2 = ax1.bar(x + width/2, clear_means, width, label='Clear Day Only',
                    color='#E76F51', edgecolor='black', linewidth=0.3)
    
    ax1.set_ylabel('mIoU (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([DOMAIN_NAMES[d] for d in domains], rotation=45, ha='right')
    ax1.set_ylim(0, 80)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title('(a) Per-Domain Performance', fontsize=FONT_SIZE_TITLE)
    
    # Panel B: Normal vs Adverse summary
    ax2 = axes[1]
    
    # Calculate normal/adverse averages
    normal_domains = ['clear_day', 'cloudy']
    adverse_domains = ['dawn_dusk', 'night', 'rainy', 'snowy']
    
    full_normal = full_df[full_df['domain'].isin(normal_domains)]['mIoU'].mean()
    full_adverse = full_df[full_df['domain'].isin(adverse_domains)]['mIoU'].mean()
    clear_normal = clear_df[clear_df['domain'].isin(normal_domains)]['mIoU'].mean()
    clear_adverse = clear_df[clear_df['domain'].isin(adverse_domains)]['mIoU'].mean()
    
    categories = ['Normal\nConditions', 'Adverse\nConditions', 'Domain\nGap']
    full_values = [full_normal, full_adverse, full_normal - full_adverse]
    clear_values = [clear_normal, clear_adverse, clear_normal - clear_adverse]
    
    x = np.arange(len(categories))
    
    bars1 = ax2.bar(x - width/2, full_values, width, label='Full Training',
                    color='#2A9D8F', edgecolor='black', linewidth=0.3)
    bars2 = ax2.bar(x + width/2, clear_values, width, label='Clear Day Only',
                    color='#E76F51', edgecolor='black', linewidth=0.3)
    
    # Add value annotations
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 2),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=FONT_SIZE_ANNOTATION)
    
    ax2.set_ylabel('mIoU (%) / Gap (%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('(b) Aggregated Performance', fontsize=FONT_SIZE_TITLE)
    
    # Shared legend at top center
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), 
               ncol=2, frameon=False, fontsize=FONT_SIZE_LEGEND)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    fig.savefig(output_path / 'fig4_training_comparison.png', dpi=300)
    plt.close(fig)
    
    print("✓ Figure 4: Training Strategy Comparison saved")

# =============================================================================
# Figure 5: Dataset Difficulty Analysis
# =============================================================================

def create_dataset_difficulty_figure(data, output_path):
    """
    Create a figure showing dataset difficulty with normal/adverse breakdown.
    """
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 2.5))
    
    config_df = data['config']
    
    datasets = ['idd-aw', 'outside15k', 'bdd10k', 'mapillaryvistas']  # Sorted by difficulty
    
    # Average across models
    normal_means = []
    adverse_means = []
    
    for ds in datasets:
        ds_data = config_df[config_df['dataset'] == ds]
        normal_means.append(ds_data['normal_mIoU'].mean())
        adverse_means.append(ds_data['adverse_mIoU'].mean())
    
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax.barh(x - width/2, normal_means, width, label='Normal',
                    color=COLORS_CONDITION['normal'], edgecolor='black', linewidth=0.3)
    bars2 = ax.barh(x + width/2, adverse_means, width, label='Adverse',
                    color=COLORS_CONDITION['adverse'], edgecolor='black', linewidth=0.3)
    
    # Add gap annotations
    for i, (n, a) in enumerate(zip(normal_means, adverse_means)):
        gap = n - a
        ax.annotate(f'Δ={gap:.1f}%',
                   xy=(max(n, a) + 1, i),
                   va='center', ha='left',
                   fontsize=FONT_SIZE_ANNOTATION,
                   color='#666666')
    
    ax.set_xlabel('mIoU (%)')
    ax.set_yticks(x)
    ax.set_yticklabels([DATASET_NAMES[d] for d in datasets])
    # Legend at top
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2, 
              frameon=False, fontsize=FONT_SIZE_LEGEND)
    ax.set_xlim(0, 95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_title('Normal vs Adverse Performance', fontsize=FONT_SIZE_TITLE, pad=20)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    
    fig.savefig(output_path / 'fig5_dataset_difficulty.png', dpi=300)
    plt.close(fig)
    
    print("✓ Figure 5: Dataset Difficulty Analysis saved")

# =============================================================================
# Figure 6: Comprehensive Performance Matrix
# =============================================================================

def create_performance_matrix(data, output_path):
    """
    Create a comprehensive matrix showing per-domain performance for each model-dataset combo.
    """
    fig, axes = plt.subplots(1, 3, figsize=(IEEE_DOUBLE_COL, 3.0))
    
    domain_df = data['full_domain']
    domain_df = domain_df[domain_df['num_images'] >= 50]
    
    models = ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5']
    datasets = ['bdd10k', 'idd-aw', 'mapillaryvistas', 'outside15k']
    domains = ['clear_day', 'cloudy', 'dawn_dusk', 'night', 'rainy', 'snowy']
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        
        # Create heatmap data
        heatmap_data = np.zeros((len(datasets), len(domains)))
        mask = np.zeros((len(datasets), len(domains)), dtype=bool)
        
        for i, ds in enumerate(datasets):
            for j, domain in enumerate(domains):
                row = domain_df[(domain_df['model'] == model) & 
                               (domain_df['dataset'] == ds) & 
                               (domain_df['domain'] == domain)]
                if len(row) > 0:
                    heatmap_data[i, j] = row['mIoU'].values[0]
                else:
                    mask[i, j] = True
        
        # Create heatmap
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=25, vmax=85)
        
        # Add text annotations
        for i in range(len(datasets)):
            for j in range(len(domains)):
                if not mask[i, j]:
                    val = heatmap_data[i, j]
                    color = 'white' if val < 45 or val > 75 else 'black'
                    ax.text(j, i, f'{val:.0f}', ha='center', va='center', 
                           fontsize=FONT_SIZE_ANNOTATION-1, color=color)
        
        ax.set_xticks(np.arange(len(domains)))
        ax.set_xticklabels([DOMAIN_NAMES[d][:3] for d in domains], rotation=45, ha='right')
        
        # Always show y-axis labels
        ax.set_yticks(np.arange(len(datasets)))
        ax.set_yticklabels([DATASET_NAMES[d] for d in datasets])
        
        ax.set_title(MODEL_NAMES[model], fontsize=FONT_SIZE_TITLE)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.93, 0.18, 0.015, 0.65])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('mIoU (%)', fontsize=FONT_SIZE_LABEL)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)
    
    fig.suptitle('Per-Domain Performance Matrix', fontsize=FONT_SIZE_TITLE, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.95])
    
    fig.savefig(output_path / 'fig6_performance_matrix.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("✓ Figure 6: Performance Matrix saved")

# =============================================================================
# Figure 7: Model Ranking Summary
# =============================================================================

def create_model_ranking_figure(data, output_path):
    """
    Create a summary figure showing model rankings with error bars.
    """
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 2.0))
    
    config_df = data['config']
    
    models = ['segformer_mit-b5', 'pspnet_r50', 'deeplabv3plus_r50']  # Sorted by performance
    
    overall_means = []
    overall_stds = []
    
    for model in models:
        model_data = config_df[config_df['model'] == model]['overall_mIoU']
        overall_means.append(model_data.mean())
        overall_stds.append(model_data.std())
    
    y = np.arange(len(models))
    
    bars = ax.barh(y, overall_means, xerr=overall_stds, capsize=3,
                   color=[COLORS_MODELS[m] for m in models],
                   edgecolor='black', linewidth=0.3, error_kw={'linewidth': 0.8})
    
    # Add value annotations
    for i, (mean, std) in enumerate(zip(overall_means, overall_stds)):
        ax.annotate(f'{mean:.1f}±{std:.1f}',
                   xy=(mean + std + 2, i),
                   va='center', ha='left',
                   fontsize=FONT_SIZE_ANNOTATION)
    
    ax.set_xlabel('Overall mIoU (%)')
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_NAMES[m] for m in models])
    ax.set_xlim(0, 90)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_title('Model Architecture Ranking', fontsize=FONT_SIZE_TITLE)
    
    plt.tight_layout()
    
    fig.savefig(output_path / 'fig7_model_ranking.png', dpi=300)
    plt.close(fig)
    
    print("✓ Figure 7: Model Ranking saved")

# =============================================================================
# Figure 8: Domain Shift Impact Visualization (Per Dataset)
# =============================================================================

def create_domain_shift_figure(data, output_path):
    """
    Create a visualization showing performance drop from clear_day to adverse domains.
    Shows per-dataset performance to reveal dataset-specific patterns.
    
    IMPORTANT: Uses Stage 1 data (clear_domain) - models trained ONLY on clear_day.
    This measures cross-domain robustness: how well models generalize to unseen conditions.
    """
    # Use Stage 1 data (clear_day training only) to measure domain shift
    domain_df = data['clear_domain']
    domain_df = domain_df[domain_df['num_images'] >= 50]
    
    datasets = ['bdd10k', 'idd-aw', 'mapillaryvistas', 'outside15k']
    adverse_domains = ['dawn_dusk', 'night', 'rainy', 'snowy']  # Removed foggy (only in IDD-AW)
    
    # Create figure with subplots for each dataset
    fig, axes = plt.subplots(2, 2, figsize=(IEEE_DOUBLE_COL, 4.0))
    axes = axes.flatten()
    
    models = ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5']
    
    for ds_idx, dataset in enumerate(datasets):
        ax = axes[ds_idx]
        ds_data = domain_df[domain_df['dataset'] == dataset]
        
        # Get domains available for this dataset
        available_domains = [d for d in adverse_domains 
                            if len(ds_data[ds_data['domain'] == d]) > 0]
        
        if len(available_domains) == 0:
            ax.text(0.5, 0.5, 'No adverse\ndomains', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=FONT_SIZE_TICK)
            ax.set_title(DATASET_NAMES[dataset], fontsize=FONT_SIZE_TITLE)
            ax.axis('off')
            continue
        
        x = np.arange(len(available_domains))
        width = 0.25
        
        for i, model in enumerate(models):
            model_data = ds_data[ds_data['model'] == model]
            clear_day_row = model_data[model_data['domain'] == 'clear_day']
            
            if len(clear_day_row) == 0:
                continue
                
            clear_day_miou = clear_day_row['mIoU'].values[0]
            
            changes = []
            for domain in available_domains:
                domain_row = model_data[model_data['domain'] == domain]
                if len(domain_row) > 0:
                    change = domain_row['mIoU'].values[0] - clear_day_miou
                    changes.append(change)
                else:
                    changes.append(0)
            
            offset = (i - 1) * width
            ax.bar(x + offset, changes, width, label=MODEL_NAMES[model],
                   color=COLORS_MODELS[model], edgecolor='black', linewidth=0.3)
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_ylabel('mIoU Change (%)', fontsize=FONT_SIZE_LABEL)
        ax.set_xticks(x)
        ax.set_xticklabels([DOMAIN_NAMES[d] for d in available_domains], 
                          fontsize=FONT_SIZE_TICK, rotation=15, ha='right')
        ax.set_title(DATASET_NAMES[dataset], fontsize=FONT_SIZE_TITLE)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Set consistent y-axis limits
        ax.set_ylim(-30, 10)
    
    # Add legend to first subplot
    axes[0].legend(loc='lower left', fontsize=FONT_SIZE_LEGEND-1, ncol=1)
    
    # Overall title
    fig.suptitle('Performance Change from Clear Day (Stage 1: Clear Day Training Only)', 
                 fontsize=FONT_SIZE_TITLE, y=1.02)
    
    plt.tight_layout()
    
    fig.savefig(output_path / 'fig8_domain_shift.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("✓ Figure 8: Domain Shift Impact (Per Dataset) saved")

# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Generate all IEEE publication-ready figures."""
    print("=" * 60)
    print("IEEE Publication Figure Generator")
    print("=" * 60)
    
    # Setup paths
    base_path = Path(__file__).parent
    output_path = base_path / 'ieee_figures'
    output_path.mkdir(exist_ok=True)
    
    # Setup style
    setup_ieee_style()
    
    # Load data
    print("\nLoading data...")
    data = load_data(base_path)
    print(f"  - Full domain data: {len(data['full_domain'])} rows")
    print(f"  - Clear domain data: {len(data['clear_domain'])} rows")
    print(f"  - Config data: {len(data['config'])} rows")
    
    # Generate figures
    print("\nGenerating figures...")
    print("-" * 40)
    
    create_model_comparison_figure(data, output_path)
    create_domain_gap_heatmap(data, output_path)
    create_domain_radar_chart(data, output_path)
    create_training_comparison_figure(data, output_path)
    create_dataset_difficulty_figure(data, output_path)
    create_performance_matrix(data, output_path)
    create_model_ranking_figure(data, output_path)
    create_domain_shift_figure(data, output_path)
    
    print("-" * 40)
    print(f"\n✓ All figures saved to: {output_path}")
    print("\nGenerated files:")
    for f in sorted(output_path.glob('*')):
        print(f"  - {f.name}")
    
    print("\n" + "=" * 60)
    print("Figure generation complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()
