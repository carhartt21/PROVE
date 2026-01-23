#!/usr/bin/env python3
"""
Stage 1 Baseline Analysis - Publication Figures and Tables Generator

Generates 4 figures and 4 tables for IEEE publication analyzing Stage 1 ONLY.
Stage 1 = Clear Day Training → Test on ALL domains
This establishes the TRUE BASELINE for cross-domain robustness evaluation.

Author: PROVE Project
Date: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

COLORS_MODELS = {
    'deeplabv3plus_r50': '#2E86AB',
    'pspnet_r50': '#E07A5F',
    'segformer_mit-b5': '#3D405B'
}

COLORS_DATASETS = {
    'bdd10k': '#457B9D',
    'idd-aw': '#E63946', 
    'mapillaryvistas': '#2A9D8F',
    'outside15k': '#F4A261'
}

COLORS_DOMAINS = {
    'clear_day': '#2A9D8F',
    'cloudy': '#A8DADC',
    'dawn_dusk': '#F4A261',
    'night': '#264653',
    'rainy': '#457B9D',
    'snowy': '#E9ECEF',
    'foggy': '#E9C46A'
}

MODEL_NAMES = {
    'deeplabv3plus_r50': 'DeepLabV3+',
    'pspnet_r50': 'PSPNet',
    'segformer_mit-b5': 'SegFormer'
}

DATASET_NAMES = {
    'bdd10k': 'BDD10K',
    'idd-aw': 'IDD-AW',
    'mapillaryvistas': 'Mapillary',
    'outside15k': 'OUTSIDE15K'
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

ADVERSE_DOMAINS = ['dawn_dusk', 'night', 'rainy', 'snowy']


def setup_ieee_style():
    """Configure matplotlib for IEEE publication style."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': FONT_SIZE_TICK,
        'axes.titlesize': FONT_SIZE_TITLE,
        'axes.labelsize': FONT_SIZE_LABEL,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': 0.5,
        'axes.grid': False,
        'legend.frameon': False,
    })


def load_data(base_path):
    """Load Stage 1 baseline data."""
    data = {}
    data['config'] = pd.read_csv(base_path / 'stage1_baseline_per_config.csv')
    data['domain'] = pd.read_csv(base_path / 'stage1_baseline_per_domain.csv')
    
    # Filter domain data by image count
    data['domain'] = data['domain'][data['domain']['num_images'] >= 45]
    
    return data


# =============================================================================
# TABLE 1: Overall Baseline Performance
# =============================================================================

def generate_table1(data, output_path):
    """Generate Table 1: Overall Baseline Performance."""
    config_df = data['config']
    domain_df = data['domain']
    
    rows = []
    for _, row in config_df.iterrows():
        dataset = row['dataset']
        model = row['model']
        
        # Get domain-specific values
        ds_model_data = domain_df[(domain_df['dataset'] == dataset) & 
                                   (domain_df['model'] == model)]
        
        clear_day = ds_model_data[ds_model_data['domain'] == 'clear_day']['mIoU'].values
        clear_day = clear_day[0] if len(clear_day) > 0 else np.nan
        
        adverse = ds_model_data[ds_model_data['domain'].isin(ADVERSE_DOMAINS)]['mIoU'].mean()
        gap = clear_day - adverse if not np.isnan(adverse) else np.nan
        
        rows.append({
            'Dataset': DATASET_NAMES[dataset],
            'Model': MODEL_NAMES[model],
            'Overall mIoU': row['overall_mIoU'],
            'Clear Day': clear_day,
            'Adverse Avg': adverse,
            'Domain Gap': gap
        })
    
    df = pd.DataFrame(rows)
    
    # Add summary
    avg_row = {
        'Dataset': 'AVERAGE',
        'Model': '',
        'Overall mIoU': df['Overall mIoU'].mean(),
        'Clear Day': df['Clear Day'].mean(),
        'Adverse Avg': df['Adverse Avg'].mean(),
        'Domain Gap': df['Domain Gap'].mean()
    }
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    
    df.to_csv(output_path / 'table1_overall_baseline.csv', index=False)
    
    print("✓ Table 1: Overall Baseline Performance saved")
    return df


# =============================================================================
# TABLE 2: Model Architecture Robustness
# =============================================================================

def generate_table2(data, output_path):
    """Generate Table 2: Model Architecture Robustness."""
    config_df = data['config']
    domain_df = data['domain']
    
    models = ['segformer_mit-b5', 'pspnet_r50', 'deeplabv3plus_r50']
    
    rows = []
    for model in models:
        overall = config_df[config_df['model'] == model]['overall_mIoU'].mean()
        overall_std = config_df[config_df['model'] == model]['overall_mIoU'].std()
        
        clear = domain_df[(domain_df['model'] == model) & 
                          (domain_df['domain'] == 'clear_day')]['mIoU'].mean()
        adverse = domain_df[(domain_df['model'] == model) & 
                            (domain_df['domain'].isin(ADVERSE_DOMAINS))]['mIoU'].mean()
        gap = clear - adverse
        
        rows.append({
            'Model': MODEL_NAMES[model],
            'Overall': f'{overall:.1f} ± {overall_std:.1f}',
            'Clear Day': clear,
            'Adverse Avg': adverse,
            'Gap': gap
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path / 'table2_model_robustness.csv', index=False)
    
    print("✓ Table 2: Model Architecture Robustness saved")
    return df


# =============================================================================
# TABLE 3: Per-Domain Degradation
# =============================================================================

def generate_table3(data, output_path):
    """Generate Table 3: Per-Domain Degradation."""
    domain_df = data['domain']
    
    domains = ['clear_day', 'dawn_dusk', 'night', 'rainy', 'snowy']
    clear_avg = domain_df[domain_df['domain'] == 'clear_day']['mIoU'].mean()
    
    rows = []
    for domain in domains:
        miou = domain_df[domain_df['domain'] == domain]['mIoU'].mean()
        drop = clear_avg - miou
        n_images = domain_df[domain_df['domain'] == domain]['num_images'].sum()
        total_images = domain_df['num_images'].sum()
        pct = (n_images / total_images) * 100
        
        rows.append({
            'Domain': DOMAIN_NAMES[domain],
            'mIoU': miou,
            'Drop from Clear': f'{-drop:+.1f}' if domain != 'clear_day' else '—',
            'Test Images': n_images,
            '% of Data': pct
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path / 'table3_domain_degradation.csv', index=False)
    
    print("✓ Table 3: Per-Domain Degradation saved")
    return df


# =============================================================================
# TABLE 4: Dataset Challenge Levels
# =============================================================================

def generate_table4(data, output_path):
    """Generate Table 4: Dataset Challenge Levels."""
    config_df = data['config']
    domain_df = data['domain']
    
    datasets = ['bdd10k', 'idd-aw', 'mapillaryvistas', 'outside15k']
    
    rows = []
    for ds in datasets:
        overall = config_df[config_df['dataset'] == ds]['overall_mIoU'].mean()
        
        ds_data = domain_df[domain_df['dataset'] == ds]
        clear_day = ds_data[ds_data['domain'] == 'clear_day']['mIoU'].mean()
        
        # Find worst and best domains
        domain_means = ds_data.groupby('domain')['mIoU'].mean()
        if len(domain_means) > 0:
            worst_domain = domain_means.idxmin()
            worst_miou = domain_means.min()
            best_domain = domain_means.idxmax()
            best_miou = domain_means.max()
            gap_range = best_miou - worst_miou
        else:
            worst_domain = '—'
            worst_miou = np.nan
            best_domain = '—'
            best_miou = np.nan
            gap_range = np.nan
        
        rows.append({
            'Dataset': DATASET_NAMES[ds],
            'Overall': overall,
            'Clear Day': clear_day,
            'Worst Domain': f'{DOMAIN_NAMES.get(worst_domain, worst_domain)} ({worst_miou:.1f}%)',
            'Best Domain': f'{DOMAIN_NAMES.get(best_domain, best_domain)} ({best_miou:.1f}%)',
            'Gap Range': gap_range
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path / 'table4_dataset_challenge.csv', index=False)
    
    print("✓ Table 4: Dataset Challenge Levels saved")
    return df


# =============================================================================
# FIGURE 1: Cross-Domain Robustness Overview
# =============================================================================

def create_figure1(data, output_path):
    """Create Figure 1: Cross-Domain Robustness (grouped bar chart)."""
    domain_df = data['domain']
    
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL + 0.5, 2.5))
    
    domains = ['clear_day', 'dawn_dusk', 'night', 'rainy', 'snowy']
    models = ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5']
    
    x = np.arange(len(domains))
    width = 0.25
    
    for i, model in enumerate(models):
        values = []
        for domain in domains:
            val = domain_df[(domain_df['model'] == model) & 
                           (domain_df['domain'] == domain)]['mIoU'].mean()
            values.append(val)
        
        bars = ax.bar(x + i * width, values, width, label=MODEL_NAMES[model],
                     color=COLORS_MODELS[model], edgecolor='black', linewidth=0.3)
    
    ax.set_ylabel('mIoU (%)')
    ax.set_xlabel('Weather Domain')
    ax.set_xticks(x + width)
    ax.set_xticklabels([DOMAIN_NAMES[d][:6] for d in domains], fontsize=FONT_SIZE_TICK)
    ax.set_ylim(0, 55)
    ax.legend(loc='upper right', fontsize=FONT_SIZE_LEGEND - 0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add reference line at clear_day average
    clear_avg = domain_df[domain_df['domain'] == 'clear_day']['mIoU'].mean()
    ax.axhline(y=clear_avg, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.text(4.5, clear_avg + 1, f'Clear Day Avg: {clear_avg:.1f}%', 
           fontsize=FONT_SIZE_ANNOTATION, ha='right')
    
    ax.set_title('Cross-Domain Robustness (Stage 1 Baseline)', fontsize=FONT_SIZE_TITLE)
    
    plt.tight_layout()
    fig.savefig(output_path / 'figure1_cross_domain_robustness.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("✓ Figure 1: Cross-Domain Robustness saved")


# =============================================================================
# FIGURE 2: Dataset × Domain Heatmap
# =============================================================================

def create_figure2(data, output_path):
    """Create Figure 2: Dataset × Domain Heatmap."""
    domain_df = data['domain']
    
    datasets = ['bdd10k', 'idd-aw', 'mapillaryvistas', 'outside15k']
    domains = ['clear_day', 'cloudy', 'dawn_dusk', 'night', 'rainy', 'snowy']
    
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL + 0.5, 2.5))
    
    # Build heatmap data (averaged across models)
    heatmap_data = np.full((len(datasets), len(domains)), np.nan)
    
    for ds_idx, dataset in enumerate(datasets):
        for dom_idx, domain in enumerate(domains):
            values = domain_df[(domain_df['dataset'] == dataset) & 
                              (domain_df['domain'] == domain)]['mIoU']
            if len(values) > 0:
                heatmap_data[ds_idx, dom_idx] = values.mean()
    
    # Create heatmap
    cmap = plt.cm.RdYlGn
    im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=15, vmax=55)
    
    # Add text annotations
    for i in range(len(datasets)):
        for j in range(len(domains)):
            if not np.isnan(heatmap_data[i, j]):
                value = heatmap_data[i, j]
                text_color = 'white' if value < 25 or value > 45 else 'black'
                ax.text(j, i, f'{value:.0f}', ha='center', va='center',
                       fontsize=FONT_SIZE_ANNOTATION, color=text_color, fontweight='bold')
            else:
                ax.text(j, i, '—', ha='center', va='center',
                       fontsize=FONT_SIZE_ANNOTATION, color='gray')
    
    ax.set_xticks(np.arange(len(domains)))
    ax.set_xticklabels([DOMAIN_NAMES[d][:5] for d in domains], fontsize=FONT_SIZE_TICK - 1,
                       rotation=30, ha='right')
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_yticklabels([DATASET_NAMES[d] for d in datasets], fontsize=FONT_SIZE_TICK)
    
    # Grid lines
    ax.set_xticks(np.arange(len(domains) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(datasets) + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', length=0)
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.9, pad=0.02)
    cbar.set_label('mIoU (%)', fontsize=FONT_SIZE_LABEL)
    
    ax.set_title('Stage 1 Baseline: Dataset × Domain Performance', fontsize=FONT_SIZE_TITLE)
    
    plt.tight_layout()
    fig.savefig(output_path / 'figure2_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("✓ Figure 2: Dataset × Domain Heatmap saved")


# =============================================================================
# FIGURE 3: Domain Gap Analysis
# =============================================================================

def create_figure3(data, output_path):
    """Create Figure 3: Domain Gap Analysis (horizontal bar chart)."""
    domain_df = data['domain']
    
    datasets = ['mapillaryvistas', 'bdd10k', 'outside15k', 'idd-aw']  # Sorted by gap
    
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 2.2))
    
    gaps = []
    colors = []
    for ds in datasets:
        ds_data = domain_df[domain_df['dataset'] == ds]
        clear = ds_data[ds_data['domain'] == 'clear_day']['mIoU'].mean()
        adverse = ds_data[ds_data['domain'].isin(ADVERSE_DOMAINS)]['mIoU'].mean()
        gap = clear - adverse
        gaps.append(gap)
        
        # Color by severity
        if gap < 5:
            colors.append('#2A9D8F')  # Low gap - green
        elif gap < 12:
            colors.append('#F4A261')  # Medium gap - orange
        else:
            colors.append('#E63946')  # High gap - red
    
    y = np.arange(len(datasets))
    bars = ax.barh(y, gaps, color=colors, edgecolor='black', linewidth=0.3)
    
    # Add value labels
    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
               f'{gap:.1f}%', ha='left', va='center', fontsize=FONT_SIZE_ANNOTATION)
    
    ax.set_xlabel('Domain Gap (Clear Day − Adverse Avg)')
    ax.set_yticks(y)
    ax.set_yticklabels([DATASET_NAMES[d] for d in datasets], fontsize=FONT_SIZE_TICK)
    ax.set_xlim(0, 22)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_title('Domain Gap by Dataset', fontsize=FONT_SIZE_TITLE)
    
    plt.tight_layout()
    fig.savefig(output_path / 'figure3_domain_gap.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("✓ Figure 3: Domain Gap Analysis saved")


# =============================================================================
# FIGURE 4: Performance Distribution
# =============================================================================

def create_figure4(data, output_path):
    """Create Figure 4: Performance Distribution (box plot)."""
    domain_df = data['domain']
    
    models = ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5']
    
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 2.5))
    
    box_data = []
    positions = []
    colors = []
    
    for i, model in enumerate(models):
        model_data = domain_df[domain_df['model'] == model]['mIoU'].values
        box_data.append(model_data)
        positions.append(i)
        colors.append(COLORS_MODELS[model])
    
    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Style the medians
    for median in bp['medians']:
        median.set_color('white')
        median.set_linewidth(2)
    
    ax.set_ylabel('mIoU (%)')
    ax.set_xticks(positions)
    ax.set_xticklabels([MODEL_NAMES[m] for m in models], fontsize=FONT_SIZE_TICK)
    ax.set_ylim(0, 60)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add mean annotations
    for i, model in enumerate(models):
        mean_val = domain_df[domain_df['model'] == model]['mIoU'].mean()
        ax.scatter([i], [mean_val], marker='D', color='white', s=30, zorder=3,
                  edgecolor='black', linewidth=0.5)
    
    ax.set_title('Performance Distribution Across Domains', fontsize=FONT_SIZE_TITLE)
    
    plt.tight_layout()
    fig.savefig(output_path / 'figure4_distribution.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("✓ Figure 4: Performance Distribution saved")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Generate all Stage 1 baseline figures and tables."""
    print("=" * 60)
    print("Stage 1 Baseline Analysis Generator")
    print("=" * 60)
    
    base_path = Path(__file__).parent
    output_path = base_path / 'stage1_baseline_output'
    output_path.mkdir(exist_ok=True)
    
    setup_ieee_style()
    
    print("\nLoading Stage 1 data...")
    data = load_data(base_path)
    print(f"  - Config data: {len(data['config'])} rows")
    print(f"  - Domain data: {len(data['domain'])} rows")
    
    print("\nGenerating tables...")
    print("-" * 40)
    table1 = generate_table1(data, output_path)
    table2 = generate_table2(data, output_path)
    table3 = generate_table3(data, output_path)
    table4 = generate_table4(data, output_path)
    
    print("\nGenerating figures...")
    print("-" * 40)
    create_figure1(data, output_path)
    create_figure2(data, output_path)
    create_figure3(data, output_path)
    create_figure4(data, output_path)
    
    print("\n" + "=" * 60)
    print("Generation complete!")
    print(f"Output saved to: {output_path}")
    print("=" * 60)
    
    # Print key insights
    print("\n### KEY INSIGHTS ###")
    avg_overall = table1[table1['Dataset'] == 'AVERAGE']['Overall mIoU'].values[0]
    avg_gap = table1[table1['Dataset'] == 'AVERAGE']['Domain Gap'].values[0]
    print(f"1. Stage 1 baseline overall: {avg_overall:.1f}% mIoU")
    print(f"2. Average domain gap: {avg_gap:.1f}%")
    print(f"3. Most robust model: SegFormer (gap {table2[table2['Model'] == 'SegFormer']['Gap'].values[0]:.1f}%)")
    print(f"4. Hardest domain: Night ({table3[table3['Domain'] == 'Night']['Drop from Clear'].values[0]})")
    print(f"5. Largest dataset gap: IDD-AW (17.6%)")


if __name__ == '__main__':
    main()
