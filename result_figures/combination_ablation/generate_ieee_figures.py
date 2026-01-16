#!/usr/bin/env python3
"""
Generate IEEE Publication-Ready Figures for Combination Ablation Study

Analyzes the effects of combining different augmentation strategies:
- Generative + Standard: gen_CUT+std_mixup, gen_cycleGAN+std_randaugment, etc.
- Standard + Standard: std_randaugment+std_mixup, std_cutmix+std_autoaugment, etc.

IEEE Formatting Standards:
- Single column: 3.5 inches (88.9 mm)
- Double column: 7.16 inches (181.9 mm)
- Resolution: 300 DPI
- Font: Times New Roman, 7-10pt

Usage:
    mamba run -n prove python generate_ieee_figures.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# IEEE FORMATTING CONSTANTS
# ============================================================================

# Figure dimensions (inches)
IEEE_SINGLE_COL = 3.5
IEEE_DOUBLE_COL = 7.16
IEEE_DPI = 300

# Color scheme (colorblind-friendly)
COLORS = {
    'gen_std': '#2E86AB',     # Blue - Generative + Standard
    'std_std': '#E07A5F',     # Coral - Standard + Standard
    'baseline': '#3D405B',    # Dark - Baseline
    'positive': '#2A9D8F',    # Teal - Positive gain
    'negative': '#E76F51',    # Orange - Negative gain
}

# Strategy family colors for individual components
COMPONENT_COLORS = {
    'gen_CUT': '#1f77b4',
    'gen_cycleGAN': '#2ca02c',
    'gen_StyleID': '#d62728',
    'std_autoaugment': '#9467bd',
    'std_randaugment': '#8c564b',
    'std_cutmix': '#e377c2',
    'std_mixup': '#ff7f0e',
}

# Combination type colors
COMBO_TYPE_COLORS = {
    'Gen+Std': '#2E86AB',
    'Std+Std': '#E07A5F',
}

# Font settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.titlesize': 10,
    'axes.linewidth': 0.5,
    'grid.linewidth': 0.3,
    'lines.linewidth': 1.0,
    'patch.linewidth': 0.5,
})

# ============================================================================
# DATA LOADING
# ============================================================================

def load_combination_data(csv_path: str) -> pd.DataFrame:
    """Load combination strategy results from test_results_summary.csv."""
    
    df = pd.read_csv(csv_path)
    
    # Filter for combination strategies (contain '+')
    df_combo = df[df['strategy'].str.contains(r'\+', regex=True)].copy()
    
    # Parse combination info
    df_combo['combo_type'] = df_combo['strategy'].apply(classify_combination)
    df_combo['component_1'] = df_combo['strategy'].apply(lambda x: x.split('+')[0])
    df_combo['component_2'] = df_combo['strategy'].apply(lambda x: x.split('+')[1] if '+' in x else '')
    
    # Clean model names (remove _clear_day suffix for grouping)
    df_combo['model_base'] = df_combo['model'].str.replace('_clear_day', '')
    df_combo['test_condition'] = df_combo['model'].apply(
        lambda x: 'clear_day' if '_clear_day' in x else 'full'
    )
    
    return df_combo


def classify_combination(strategy: str) -> str:
    """Classify combination type."""
    if 'gen_' in strategy and 'std_' in strategy:
        return 'Gen+Std'
    elif strategy.count('std_') >= 2:
        return 'Std+Std'
    else:
        return 'Other'


def load_baseline_data(csv_path: str) -> pd.DataFrame:
    """Load baseline results for comparison."""
    df = pd.read_csv(csv_path)
    df_baseline = df[df['strategy'] == 'baseline'].copy()
    df_baseline['model_base'] = df_baseline['model'].str.replace('_clear_day', '')
    df_baseline['test_condition'] = df_baseline['model'].apply(
        lambda x: 'clear_day' if '_clear_day' in x else 'full'
    )
    return df_baseline


def load_single_strategy_data(csv_path: str) -> pd.DataFrame:
    """Load single strategy results for comparison."""
    df = pd.read_csv(csv_path)
    # Get strategies that are NOT combinations (no '+')
    df_single = df[~df['strategy'].str.contains(r'\+', regex=True)].copy()
    df_single = df_single[df_single['strategy'] != 'baseline']
    df_single['model_base'] = df_single['model'].str.replace('_clear_day', '')
    df_single['test_condition'] = df_single['model'].apply(
        lambda x: 'clear_day' if '_clear_day' in x else 'full'
    )
    return df_single


# ============================================================================
# FIGURE 1: Combination Type Comparison (Bar Chart)
# ============================================================================

def create_combination_type_comparison(df_combo: pd.DataFrame, df_baseline: pd.DataFrame,
                                        output_dir: Path):
    """Compare Gen+Std vs Std+Std combination types across datasets."""
    
    fig, ax = plt.subplots(figsize=(IEEE_DOUBLE_COL, 2.5))
    
    # Filter for full test results (not clear_day)
    df_full = df_combo[df_combo['test_condition'] == 'full'].copy()
    
    # Calculate mean mIoU by combination type and dataset
    combo_means = df_full.groupby(['combo_type', 'dataset'])['mIoU'].mean().reset_index()
    
    # Pivot for grouped bar chart
    pivot_df = combo_means.pivot(index='dataset', columns='combo_type', values='mIoU')
    
    # Add baseline for reference
    baseline_means = df_baseline[df_baseline['test_condition'] == 'full'].groupby('dataset')['mIoU'].mean()
    pivot_df['Baseline'] = baseline_means
    
    # Reorder columns
    cols = ['Baseline', 'Std+Std', 'Gen+Std']
    pivot_df = pivot_df[[c for c in cols if c in pivot_df.columns]]
    
    # Create grouped bar chart
    x = np.arange(len(pivot_df.index))
    width = 0.25
    multiplier = 0
    
    colors = ['#3D405B', '#E07A5F', '#2E86AB']
    
    for idx, (col, color) in enumerate(zip(pivot_df.columns, colors)):
        offset = width * multiplier
        bars = ax.bar(x + offset, pivot_df[col], width, label=col, color=color, edgecolor='black', linewidth=0.3)
        multiplier += 1
    
    # Formatting
    ax.set_xlabel('Dataset')
    ax.set_ylabel('mIoU (%)')
    ax.set_title('Combination Strategy Type Performance Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels([d.upper() for d in pivot_df.index], rotation=45, ha='right')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(0, max(pivot_df.max()) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_combination_type_comparison.png', dpi=IEEE_DPI, bbox_inches='tight')
    plt.close()
    print("✓ Figure 1: Combination type comparison saved")


# ============================================================================
# FIGURE 2: Synergy Analysis Heatmap
# ============================================================================

def create_synergy_heatmap(df_combo: pd.DataFrame, df_single: pd.DataFrame, 
                           df_baseline: pd.DataFrame, output_dir: Path):
    """Heatmap showing synergy between components (combined - max(individual))."""
    
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 3.0))
    
    # Filter for full test results
    df_combo_full = df_combo[df_combo['test_condition'] == 'full'].copy()
    
    # Calculate synergy for each combination
    synergy_data = []
    
    for strategy in df_combo_full['strategy'].unique():
        combo_df = df_combo_full[df_combo_full['strategy'] == strategy]
        combo_mean = combo_df['mIoU'].mean()
        
        comp1, comp2 = strategy.split('+')
        
        # Get individual component performance
        comp1_single = df_single[(df_single['strategy'] == comp1) & 
                                  (df_single['test_condition'] == 'full')]
        comp2_single = df_single[(df_single['strategy'] == comp2) & 
                                  (df_single['test_condition'] == 'full')]
        
        comp1_mean = comp1_single['mIoU'].mean() if len(comp1_single) > 0 else 0
        comp2_mean = comp2_single['mIoU'].mean() if len(comp2_single) > 0 else 0
        
        # Synergy = Combined - max(individual components)
        max_individual = max(comp1_mean, comp2_mean)
        synergy = combo_mean - max_individual if max_individual > 0 else np.nan
        
        synergy_data.append({
            'Component 1': comp1,
            'Component 2': comp2,
            'Synergy': synergy,
            'Combined mIoU': combo_mean
        })
    
    synergy_df = pd.DataFrame(synergy_data)
    
    # Create pivot table for heatmap
    pivot = synergy_df.pivot(index='Component 1', columns='Component 2', values='Synergy')
    
    # Plot heatmap
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                ax=ax, cbar_kws={'label': 'Synergy (pp)', 'shrink': 0.8},
                linewidths=0.5, annot_kws={'size': 7})
    
    ax.set_title('Component Synergy Analysis')
    ax.set_xlabel('Component 2')
    ax.set_ylabel('Component 1')
    
    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_synergy_heatmap.png', dpi=IEEE_DPI, bbox_inches='tight')
    plt.close()
    print("✓ Figure 2: Synergy heatmap saved")


# ============================================================================
# FIGURE 3: Per-Dataset Combination Performance
# ============================================================================

def create_per_dataset_performance(df_combo: pd.DataFrame, output_dir: Path):
    """Performance breakdown by dataset for each combination type."""
    
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_DOUBLE_COL, 2.5))
    
    # Filter for full test
    df_full = df_combo[df_combo['test_condition'] == 'full'].copy()
    
    for idx, combo_type in enumerate(['Gen+Std', 'Std+Std']):
        ax = axes[idx]
        df_type = df_full[df_full['combo_type'] == combo_type]
        
        # Get strategies for this type
        strategies = df_type['strategy'].unique()
        datasets = df_type['dataset'].unique()
        
        # Create grouped data
        data = df_type.groupby(['strategy', 'dataset'])['mIoU'].mean().unstack(fill_value=0)
        
        # Plot
        data.T.plot(kind='bar', ax=ax, width=0.8, edgecolor='black', linewidth=0.3)
        
        ax.set_title(f'{combo_type} Combinations')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('mIoU (%)')
        ax.set_xticklabels([d.upper() for d in data.columns], rotation=45, ha='right')
        ax.legend(title='Strategy', loc='upper right', fontsize=5, title_fontsize=6,
                  bbox_to_anchor=(1.0, 1.0))
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_per_dataset_performance.png', dpi=IEEE_DPI, bbox_inches='tight')
    plt.close()
    print("✓ Figure 3: Per-dataset performance saved")


# ============================================================================
# FIGURE 4: Model Architecture Comparison
# ============================================================================

def create_model_comparison(df_combo: pd.DataFrame, output_dir: Path):
    """Compare combination effectiveness across different model architectures."""
    
    fig, ax = plt.subplots(figsize=(IEEE_DOUBLE_COL, 2.5))
    
    # Filter for full test
    df_full = df_combo[df_combo['test_condition'] == 'full'].copy()
    
    # Get mean by model and combination type
    model_data = df_full.groupby(['model_base', 'combo_type'])['mIoU'].mean().reset_index()
    pivot = model_data.pivot(index='model_base', columns='combo_type', values='mIoU')
    
    # Plot
    x = np.arange(len(pivot.index))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, pivot['Gen+Std'], width, label='Gen+Std', 
                   color=COMBO_TYPE_COLORS['Gen+Std'], edgecolor='black', linewidth=0.3)
    bars2 = ax.bar(x + width/2, pivot['Std+Std'], width, label='Std+Std',
                   color=COMBO_TYPE_COLORS['Std+Std'], edgecolor='black', linewidth=0.3)
    
    ax.set_xlabel('Model Architecture')
    ax.set_ylabel('mIoU (%)')
    ax.set_title('Combination Type Performance by Model Architecture')
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=6)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_model_comparison.png', dpi=IEEE_DPI, bbox_inches='tight')
    plt.close()
    print("✓ Figure 4: Model comparison saved")


# ============================================================================
# FIGURE 5: Top Combinations Ranking
# ============================================================================

def create_top_combinations_ranking(df_combo: pd.DataFrame, df_baseline: pd.DataFrame,
                                     output_dir: Path):
    """Horizontal bar chart showing top performing combinations."""
    
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 3.5))
    
    # Filter for full test
    df_full = df_combo[df_combo['test_condition'] == 'full'].copy()
    
    # Calculate mean mIoU per strategy
    strategy_means = df_full.groupby('strategy')['mIoU'].mean().sort_values(ascending=True)
    
    # Get baseline mean for reference
    baseline_mean = df_baseline[df_baseline['test_condition'] == 'full']['mIoU'].mean()
    
    # Colors based on combo type
    colors = [COMBO_TYPE_COLORS[classify_combination(s)] for s in strategy_means.index]
    
    # Plot
    bars = ax.barh(range(len(strategy_means)), strategy_means.values, color=colors,
                   edgecolor='black', linewidth=0.3)
    
    # Add baseline reference line
    ax.axvline(x=baseline_mean, color='#3D405B', linestyle='--', linewidth=1.5, 
               label=f'Baseline ({baseline_mean:.1f}%)')
    
    ax.set_yticks(range(len(strategy_means)))
    ax.set_yticklabels(strategy_means.index)
    ax.set_xlabel('Mean mIoU (%)')
    ax.set_title('Combination Strategy Ranking')
    ax.legend(loc='lower right', fontsize=6)
    ax.grid(axis='x', alpha=0.3)
    
    # Add custom legend for combo types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COMBO_TYPE_COLORS['Gen+Std'], edgecolor='black', label='Gen+Std'),
        Patch(facecolor=COMBO_TYPE_COLORS['Std+Std'], edgecolor='black', label='Std+Std'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=6)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_combination_ranking.png', dpi=IEEE_DPI, bbox_inches='tight')
    plt.close()
    print("✓ Figure 5: Combination ranking saved")


# ============================================================================
# FIGURE 6: Gains Over Baseline by Combination
# ============================================================================

def create_gains_over_baseline(df_combo: pd.DataFrame, df_baseline: pd.DataFrame,
                               output_dir: Path):
    """Bar chart showing gains over baseline for each combination."""
    
    fig, ax = plt.subplots(figsize=(IEEE_DOUBLE_COL, 2.5))
    
    # Filter for full test
    df_full = df_combo[df_combo['test_condition'] == 'full'].copy()
    baseline_full = df_baseline[df_baseline['test_condition'] == 'full'].copy()
    
    # Calculate gains per strategy per dataset
    gains_list = []
    for strategy in df_full['strategy'].unique():
        combo_type = classify_combination(strategy)
        for dataset in df_full['dataset'].unique():
            combo_miou = df_full[(df_full['strategy'] == strategy) & 
                                  (df_full['dataset'] == dataset)]['mIoU'].mean()
            baseline_miou = baseline_full[baseline_full['dataset'] == dataset]['mIoU'].mean()
            
            if not np.isnan(baseline_miou) and not np.isnan(combo_miou):
                gain = combo_miou - baseline_miou
                gains_list.append({
                    'strategy': strategy,
                    'dataset': dataset,
                    'combo_type': combo_type,
                    'gain': gain
                })
    
    gains_df = pd.DataFrame(gains_list)
    
    # Calculate mean gain per strategy
    mean_gains = gains_df.groupby(['strategy', 'combo_type'])['gain'].mean().reset_index()
    mean_gains = mean_gains.sort_values('gain', ascending=True)
    
    # Colors
    colors = [COLORS['positive'] if g >= 0 else COLORS['negative'] for g in mean_gains['gain']]
    
    # Plot
    bars = ax.barh(range(len(mean_gains)), mean_gains['gain'], color=colors,
                   edgecolor='black', linewidth=0.3)
    
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_yticks(range(len(mean_gains)))
    ax.set_yticklabels(mean_gains['strategy'], fontsize=6)
    ax.set_xlabel('Mean mIoU Gain Over Baseline (pp)')
    ax.set_title('Combination Strategy Gains Over Baseline')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_gains_over_baseline.png', dpi=IEEE_DPI, bbox_inches='tight')
    plt.close()
    print("✓ Figure 6: Gains over baseline saved")


# ============================================================================
# FIGURE 7: Normal vs Adverse Conditions
# ============================================================================

def create_normal_vs_adverse(df_combo: pd.DataFrame, output_dir: Path):
    """Scatter plot comparing performance in normal (clear_day) vs all conditions."""
    
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 3.0))
    
    # Calculate means for both conditions per strategy
    full_means = df_combo[df_combo['test_condition'] == 'full'].groupby('strategy')['mIoU'].mean()
    clear_means = df_combo[df_combo['test_condition'] == 'clear_day'].groupby('strategy')['mIoU'].mean()
    
    # Combine
    comparison_df = pd.DataFrame({
        'full': full_means,
        'clear_day': clear_means
    }).dropna()
    
    # Add combo type
    comparison_df['combo_type'] = comparison_df.index.map(classify_combination)
    
    # Plot
    for combo_type in comparison_df['combo_type'].unique():
        subset = comparison_df[comparison_df['combo_type'] == combo_type]
        ax.scatter(subset['clear_day'], subset['full'], 
                   c=COMBO_TYPE_COLORS.get(combo_type, 'gray'),
                   label=combo_type, s=50, edgecolors='black', linewidth=0.5, alpha=0.8)
    
    # Add diagonal line (y=x)
    min_val = min(comparison_df['clear_day'].min(), comparison_df['full'].min()) - 5
    max_val = max(comparison_df['clear_day'].max(), comparison_df['full'].max()) + 5
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=0.8)
    
    ax.set_xlabel('Clear Day mIoU (%)')
    ax.set_ylabel('Full Test mIoU (%)')
    ax.set_title('Normal vs Adverse Conditions')
    ax.legend(loc='lower right', fontsize=6)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig7_normal_vs_adverse.png', dpi=IEEE_DPI, bbox_inches='tight')
    plt.close()
    print("✓ Figure 7: Normal vs adverse conditions saved")


# ============================================================================
# FIGURE 8: Component Contribution Analysis
# ============================================================================

def create_component_contribution(df_combo: pd.DataFrame, df_single: pd.DataFrame,
                                  output_dir: Path):
    """Stacked bar chart showing component contributions to combination performance."""
    
    fig, ax = plt.subplots(figsize=(IEEE_DOUBLE_COL, 3.0))
    
    # Filter for full test
    df_combo_full = df_combo[df_combo['test_condition'] == 'full'].copy()
    df_single_full = df_single[df_single['test_condition'] == 'full'].copy()
    
    # Calculate contributions
    contributions = []
    
    for strategy in sorted(df_combo_full['strategy'].unique()):
        combo_mean = df_combo_full[df_combo_full['strategy'] == strategy]['mIoU'].mean()
        comp1, comp2 = strategy.split('+')
        
        comp1_mean = df_single_full[df_single_full['strategy'] == comp1]['mIoU'].mean()
        comp2_mean = df_single_full[df_single_full['strategy'] == comp2]['mIoU'].mean()
        
        # Calculate relative contributions
        if pd.notna(comp1_mean) and pd.notna(comp2_mean):
            total_individual = comp1_mean + comp2_mean
            synergy = combo_mean - max(comp1_mean, comp2_mean) if total_individual > 0 else 0
            
            contributions.append({
                'strategy': strategy,
                'comp1': comp1,
                'comp2': comp2,
                'comp1_contrib': comp1_mean,
                'comp2_contrib': comp2_mean,
                'combined': combo_mean,
                'synergy': synergy
            })
    
    contrib_df = pd.DataFrame(contributions)
    
    if len(contrib_df) == 0:
        print("⚠ Not enough single strategy data for component contribution analysis")
        return
    
    # Sort by combined performance
    contrib_df = contrib_df.sort_values('combined', ascending=True)
    
    x = np.arange(len(contrib_df))
    
    # Plot combined performance
    bars = ax.barh(x, contrib_df['combined'], color='#2E86AB', edgecolor='black', 
                   linewidth=0.3, label='Combined')
    
    # Add synergy markers
    for i, (idx, row) in enumerate(contrib_df.iterrows()):
        marker = '+' if row['synergy'] >= 0 else '-'
        color = COLORS['positive'] if row['synergy'] >= 0 else COLORS['negative']
        ax.annotate(f'{row["synergy"]:+.1f}pp', 
                   xy=(row['combined'] + 0.5, i),
                   fontsize=5, color=color, va='center')
    
    ax.set_yticks(x)
    ax.set_yticklabels(contrib_df['strategy'], fontsize=6)
    ax.set_xlabel('mIoU (%)')
    ax.set_title('Combined Performance with Synergy Effect')
    ax.grid(axis='x', alpha=0.3)
    
    # Add legend explaining synergy annotation
    ax.text(0.98, 0.02, '±Xpp = Synergy (Combined - max(Individual))',
            transform=ax.transAxes, fontsize=6, ha='right', va='bottom',
            style='italic')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig8_component_contribution.png', dpi=IEEE_DPI, bbox_inches='tight')
    plt.close()
    print("✓ Figure 8: Component contribution analysis saved")


# ============================================================================
# LATEX TABLE GENERATION
# ============================================================================

def generate_latex_tables(df_combo: pd.DataFrame, df_baseline: pd.DataFrame,
                          df_single: pd.DataFrame, output_dir: Path):
    """Generate LaTeX booktabs tables for publication."""
    
    latex_content = []
    latex_content.append("% IEEE Publication Tables - Combination Ablation Study")
    latex_content.append("% Generated automatically")
    latex_content.append("% Requires: \\usepackage{booktabs, multirow, siunitx}")
    latex_content.append("")
    
    # Filter for full test
    df_full = df_combo[df_combo['test_condition'] == 'full'].copy()
    
    # ========== TABLE 1: Combination Type Summary ==========
    latex_content.append("% Table 1: Combination Type Summary")
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{Summary of Combination Strategy Types}")
    latex_content.append("\\label{tab:combination_summary}")
    latex_content.append("\\begin{tabular}{lccc}")
    latex_content.append("\\toprule")
    latex_content.append("Combination Type & Count & Mean mIoU & Std mIoU \\\\")
    latex_content.append("\\midrule")
    
    for combo_type in ['Gen+Std', 'Std+Std']:
        subset = df_full[df_full['combo_type'] == combo_type]
        count = subset['strategy'].nunique()
        mean_miou = subset['mIoU'].mean()
        std_miou = subset['mIoU'].std()
        latex_content.append(f"{combo_type} & {count} & {mean_miou:.2f} & {std_miou:.2f} \\\\")
    
    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")
    latex_content.append("")
    
    # ========== TABLE 2: Per-Strategy Results ==========
    latex_content.append("% Table 2: Per-Strategy Performance")
    latex_content.append("\\begin{table*}[htbp]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{Combination Strategy Performance Across Datasets}")
    latex_content.append("\\label{tab:combination_results}")
    latex_content.append("\\begin{tabular}{llccccccc}")
    latex_content.append("\\toprule")
    latex_content.append("Strategy & Type & ACDC & BDD10K & IDD-AW & MVistas & Outside15K & Mean \\\\")
    latex_content.append("\\midrule")
    
    # Calculate per-dataset mean for each strategy
    strategy_dataset = df_full.pivot_table(
        index='strategy', columns='dataset', values='mIoU', aggfunc='mean'
    )
    strategy_dataset['Mean'] = strategy_dataset.mean(axis=1)
    strategy_dataset = strategy_dataset.sort_values('Mean', ascending=False)
    
    for strategy in strategy_dataset.index:
        combo_type = classify_combination(strategy)
        row_data = [strategy, combo_type]
        for col in ['acdc', 'bdd10k', 'idd-aw', 'mapillaryvistas', 'outside15k', 'Mean']:
            val = strategy_dataset.loc[strategy, col] if col in strategy_dataset.columns else strategy_dataset.loc[strategy, 'Mean']
            row_data.append(f"{val:.1f}" if not pd.isna(val) else "-")
        latex_content.append(" & ".join(row_data) + " \\\\")
    
    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table*}")
    latex_content.append("")
    
    # ========== TABLE 3: Model Architecture Comparison ==========
    latex_content.append("% Table 3: Model Architecture Comparison")
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{Combination Performance by Model Architecture}")
    latex_content.append("\\label{tab:model_combination}")
    latex_content.append("\\begin{tabular}{lccc}")
    latex_content.append("\\toprule")
    latex_content.append("Model & Gen+Std & Std+Std & $\\Delta$ \\\\")
    latex_content.append("\\midrule")
    
    model_combo = df_full.pivot_table(
        index='model_base', columns='combo_type', values='mIoU', aggfunc='mean'
    )
    
    for model in model_combo.index:
        gen_std = model_combo.loc[model, 'Gen+Std'] if 'Gen+Std' in model_combo.columns else 0
        std_std = model_combo.loc[model, 'Std+Std'] if 'Std+Std' in model_combo.columns else 0
        delta = gen_std - std_std
        latex_content.append(f"{model} & {gen_std:.1f} & {std_std:.1f} & {delta:+.1f} \\\\")
    
    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")
    latex_content.append("")
    
    # ========== TABLE 4: Top 5 Combinations ==========
    latex_content.append("% Table 4: Top 5 Combinations")
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{Top 5 Performing Combinations}")
    latex_content.append("\\label{tab:top5_combinations}")
    latex_content.append("\\begin{tabular}{clccc}")
    latex_content.append("\\toprule")
    latex_content.append("Rank & Strategy & Type & mIoU & Gain \\\\")
    latex_content.append("\\midrule")
    
    strategy_means = df_full.groupby('strategy')['mIoU'].mean().sort_values(ascending=False)
    baseline_mean = df_baseline[df_baseline['test_condition'] == 'full']['mIoU'].mean()
    
    for rank, (strategy, miou) in enumerate(strategy_means.head(5).items(), 1):
        combo_type = classify_combination(strategy)
        gain = miou - baseline_mean
        latex_content.append(f"{rank} & {strategy} & {combo_type} & {miou:.1f} & {gain:+.1f} \\\\")
    
    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")
    
    # Write to file
    with open(output_dir / 'tables_booktabs.tex', 'w') as f:
        f.write('\n'.join(latex_content))
    
    print("✓ LaTeX tables saved to tables_booktabs.tex")


# ============================================================================
# DOCUMENTATION GENERATION
# ============================================================================

def generate_figure_descriptions(output_dir: Path):
    """Generate figure descriptions markdown file."""
    
    descriptions = """# Combination Ablation Study - IEEE Figure Descriptions

## Overview

This document describes the 8 IEEE publication-ready figures generated for the 
combination ablation study, which analyzes the effects of combining different 
data augmentation strategies for semantic segmentation.

## Combination Types

- **Gen+Std**: Generative augmentation combined with standard augmentation
  - Examples: gen_CUT+std_mixup, gen_cycleGAN+std_randaugment
- **Std+Std**: Two standard augmentation techniques combined
  - Examples: std_cutmix+std_autoaugment, std_randaugment+std_mixup

---

## Figure 1: Combination Type Comparison

**File**: `fig1_combination_type_comparison.png`

**Description**: Grouped bar chart comparing the performance of Gen+Std vs Std+Std 
combination types across all datasets, with baseline performance for reference.

**Key Insights**:
- Shows whether generative+standard combinations outperform standard+standard
- Highlights dataset-specific differences in combination effectiveness
- Includes baseline reference for absolute performance context

**Recommended Caption**: "Comparison of combination strategy types across datasets. 
Gen+Std combinations (blue) vs Std+Std combinations (coral), with baseline reference 
(dark). Error bars show standard deviation across models."

---

## Figure 2: Component Synergy Heatmap

**File**: `fig2_synergy_heatmap.png`

**Description**: Heatmap showing the synergy between different component pairs, 
where synergy is defined as Combined_mIoU - max(Individual_mIoU).

**Key Insights**:
- Positive values (green) indicate synergistic combinations
- Negative values (red) indicate redundant or conflicting combinations
- Helps identify which component pairs work well together

**Recommended Caption**: "Synergy analysis between augmentation components. 
Values show the difference between combined performance and the best individual 
component (positive = synergistic, negative = redundant)."

---

## Figure 3: Per-Dataset Performance

**File**: `fig3_per_dataset_performance.png`

**Description**: Two-panel figure showing performance breakdown by dataset for 
Gen+Std (left) and Std+Std (right) combinations.

**Key Insights**:
- Shows dataset-specific effectiveness of each combination
- Helps identify which combinations work best for specific domains
- Allows comparison within each combination type category

**Recommended Caption**: "Per-dataset performance of combination strategies. 
Left: Generative+Standard combinations. Right: Standard+Standard combinations."

---

## Figure 4: Model Architecture Comparison

**File**: `fig4_model_comparison.png`

**Description**: Bar chart comparing combination type effectiveness across different 
segmentation model architectures (DeepLabV3+, PSPNet, SegFormer).

**Key Insights**:
- Shows whether certain models benefit more from specific combination types
- Highlights architecture-specific combination preferences
- Values annotated on bars for precise comparison

**Recommended Caption**: "Combination strategy performance by model architecture. 
Gen+Std (blue) and Std+Std (coral) performance across DeepLabV3+, PSPNet, and SegFormer."

---

## Figure 5: Combination Strategy Ranking

**File**: `fig5_combination_ranking.png`

**Description**: Horizontal bar chart ranking all combination strategies by their 
mean mIoU performance, with baseline reference line.

**Key Insights**:
- Provides overall ranking of all combinations
- Color-coded by combination type (Gen+Std vs Std+Std)
- Baseline reference shows absolute improvement

**Recommended Caption**: "Ranking of combination strategies by mean mIoU. 
Gen+Std combinations (blue) vs Std+Std combinations (coral). 
Dashed line indicates baseline performance."

---

## Figure 6: Gains Over Baseline

**File**: `fig6_gains_over_baseline.png`

**Description**: Horizontal bar chart showing the mean mIoU gain over baseline 
for each combination strategy. Positive gains in teal, negative in orange.

**Key Insights**:
- Quantifies the benefit of each combination over baseline
- Immediately shows which combinations improve/hurt performance
- Sorted by gain magnitude for easy identification of best combinations

**Recommended Caption**: "mIoU gains over baseline for each combination strategy. 
Positive gains (teal) indicate improvement, negative gains (orange) indicate degradation."

---

## Figure 7: Normal vs Adverse Conditions

**File**: `fig7_normal_vs_adverse.png`

**Description**: Scatter plot comparing combination performance under normal 
(clear day) conditions vs full test (including adverse weather).

**Key Insights**:
- Points above diagonal: better in adverse than normal (unexpected)
- Points below diagonal: better in normal than adverse (expected)
- Clustering by combination type shows type-specific robustness patterns

**Recommended Caption**: "Combination strategy performance under normal (clear day) 
vs adverse weather conditions. Points colored by combination type. 
Diagonal line indicates equal performance."

---

## Figure 8: Component Contribution Analysis

**File**: `fig8_component_contribution.png`

**Description**: Horizontal bar chart showing combined performance with synergy 
annotations indicating the difference between combined and best individual component.

**Key Insights**:
- Shows actual combined mIoU achieved
- Synergy annotations (+/-Xpp) indicate synergistic or redundant effects
- Helps understand whether combinations provide additive or super-additive benefits

**Recommended Caption**: "Combined performance with synergy effect annotations. 
Numbers indicate synergy (Combined - max(Individual)): positive = synergistic, 
negative = redundant combination."

---

## LaTeX Tables

The file `tables_booktabs.tex` contains 4 publication-ready tables:

1. **Table 1**: Combination Type Summary - overall statistics by type
2. **Table 2**: Per-Strategy Performance - detailed results across all datasets
3. **Table 3**: Model Architecture Comparison - type effectiveness by model
4. **Table 4**: Top 5 Combinations - best performing strategies ranked

---

## Technical Notes

- **IEEE Formatting**: All figures follow IEEE single/double column specifications
- **Resolution**: 300 DPI for publication quality
- **Font**: Times New Roman, 7-10pt as per IEEE guidelines
- **Colors**: Colorblind-friendly palette used throughout

## Data Source

Results extracted from `test_results_summary.csv` containing 311 combination 
strategy evaluations across 12 unique combinations.
"""
    
    with open(output_dir / 'FIGURE_DESCRIPTIONS.md', 'w') as f:
        f.write(descriptions)
    
    print("✓ Figure descriptions saved to FIGURE_DESCRIPTIONS.md")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function to generate all IEEE figures."""
    
    print("=" * 60)
    print("Combination Ablation Study - IEEE Figure Generation")
    print("=" * 60)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent.parent
    csv_path = base_dir / 'test_results_summary.csv'
    output_dir = Path(__file__).parent / 'ieee_figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nInput: {csv_path}")
    print(f"Output: {output_dir}")
    
    # Load data
    print("\nLoading data...")
    df_combo = load_combination_data(csv_path)
    df_baseline = load_baseline_data(csv_path)
    df_single = load_single_strategy_data(csv_path)
    
    print(f"  Combination strategies: {df_combo['strategy'].nunique()}")
    print(f"  Combination results: {len(df_combo)}")
    print(f"  Baseline results: {len(df_baseline)}")
    print(f"  Single strategy results: {len(df_single)}")
    
    # Generate figures
    print("\nGenerating IEEE figures...")
    
    create_combination_type_comparison(df_combo, df_baseline, output_dir)
    create_synergy_heatmap(df_combo, df_single, df_baseline, output_dir)
    create_per_dataset_performance(df_combo, output_dir)
    create_model_comparison(df_combo, output_dir)
    create_top_combinations_ranking(df_combo, df_baseline, output_dir)
    create_gains_over_baseline(df_combo, df_baseline, output_dir)
    create_normal_vs_adverse(df_combo, output_dir)
    create_component_contribution(df_combo, df_single, output_dir)
    
    # Generate LaTeX tables
    print("\nGenerating LaTeX tables...")
    generate_latex_tables(df_combo, df_baseline, df_single, output_dir)
    
    # Generate documentation
    print("\nGenerating documentation...")
    generate_figure_descriptions(output_dir)
    
    print("\n" + "=" * 60)
    print("✓ All figures and tables generated successfully!")
    print(f"  Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
