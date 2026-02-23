#!/usr/bin/env python3
"""
Strategy Family Analysis Script (fwIoU-based with Gains/Losses)

Analyzes segmentation performance grouped by strategy families using:
- Primary Metric: fwIoU (frequency-weighted IoU)
- Secondary Metrics: mIoU, PA (where relevant)
- Comparisons: Gains/Losses relative to baseline (not absolute values)

Strategy Families:
1. 2D Rendering (imgaug_weather, automold, Weather_Effect_Generator)
2. CNN/GAN (Attribute_Hallucination, cycleGAN, CUT, stargan_v2, SUSTechGAN)
3. Style Transfer (NST, LANIT, TSIT)
4. Diffusion (Img2Img, IP2P, UniControl)
5. Multimodal Diffusion (flux1_kontext, step1x_new, Qwen_Image_Edit)
6. Standard Augmentation (autoaugment, randaugment, std_photometric_distort)
7. Standard Mixing (cutmix, mixup)

Excluded methods (0/4 training dataset coverage): StyleID, EDICT, AOD-Net

Note: Combined strategies (with + in name) are analyzed separately in 
analyze_combination_ablation.py and stored in WEIGHTS_COMBINATIONS.

Usage:
    mamba run -n prove python analyze_strategy_families.py

Output:
    result_figures/family_analysis/
    
See docs/FAMILY_ANALYSIS.md for detailed documentation.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Output directory
OUTPUT_DIR = Path("result_figures/family_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Strategy family definitions based on the provided table
STRATEGY_FAMILIES = {
    "2D Rendering": [
        "gen_albumentations_weather",
        "gen_automold", 
        "gen_Weather_Effect_Generator",
        "gen_augmenters",
    ],
    "CNN/GAN": [
        "gen_Attribute_Hallucination",
        "gen_cycleGAN",
        "gen_CUT",
        "gen_stargan_v2",
        "gen_SUSTechGAN"
    ],
    "Style Transfer": [
        "gen_LANIT",
        "gen_TSIT",
    ],
    "Diffusion I2I": [
        "gen_Img2Img",
        "gen_IP2P",
        "gen_cyclediffusion",
    ],
    "Instruct/Edit": [
        "gen_UniControl",
        "gen_Qwen_Image_Edit",
        "gen_Attribute_Hallucination",
    ],
    "Modern Diffusion": [
        "gen_flux_kontext",
        "gen_step1x_new",
        "gen_step1x_v1p2",
        "gen_VisualCloze",
        "gen_CNetSeg",
    ],
    "Standard Augmentation": [
        "std_autoaugment",
        "std_randaugment",
    ],
    "Standard Mixing": [
        "std_cutmix",
        "std_mixup"
    ],
    "Baseline": [
        "baseline"
    ]
}

# Create reverse mapping: strategy -> family
STRATEGY_TO_FAMILY = {}
for family, strategies in STRATEGY_FAMILIES.items():
    for strategy in strategies:
        STRATEGY_TO_FAMILY[strategy] = family

# Weather domains
WEATHER_DOMAINS = ['clear_day', 'cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy']
ADVERSE_DOMAINS = ['foggy', 'night', 'rainy', 'snowy']
NORMAL_DOMAINS = ['clear_day', 'cloudy', 'dawn_dusk']


def load_baseline_reference(df: pd.DataFrame = None):
    """Load baseline reference values for computing gains/losses.
    
    First tries the reference JSON file; if not found, builds reference
    from baseline rows in the provided dataframe.
    """
    ref_path = Path("result_figures/baseline/baseline_reference.json")
    if ref_path.exists():
        with open(ref_path) as f:
            return json.load(f)
    
    # Fallback: build from dataframe
    if df is not None:
        baseline_df = df[df['strategy'] == 'baseline']
        ref = {}
        for _, row in baseline_df.iterrows():
            key = f"{row['dataset']}_{row['model']}"
            ref[key] = {
                'fwIoU': row.get('fwIoU', 0),
                'mIoU': row.get('mIoU', 0),
            }
        return ref
    
    return {}


def get_family(strategy: str) -> str:
    """Get the family for a strategy."""
    if '+' in strategy:
        return "Combined"
    return STRATEGY_TO_FAMILY.get(strategy, "Unknown")


def load_all_results(downstream_csv: str) -> pd.DataFrame:
    """Load results from downstream_results.csv and enrich with family info."""
    
    df = pd.read_csv(downstream_csv)
    
    # EXCLUDE combination strategies (analyzed separately)
    df = df[~df['strategy'].str.contains(r'\+', regex=True)]
    
    # Add family information
    df['family'] = df['strategy'].apply(get_family)
    
    return df


def compute_gains_losses(df: pd.DataFrame, baseline_ref: dict) -> pd.DataFrame:
    """Compute gains/losses relative to baseline for fwIoU."""
    
    results = []
    
    for _, row in df.iterrows():
        if row['family'] == 'Baseline':
            result = row.to_dict()
            result['fwIoU_gain'] = 0.0
            result['mIoU_gain'] = 0.0
            results.append(result)
            continue
        
        # Find matching baseline
        key = f"{row['dataset']}_{row['model']}"
        
        if key in baseline_ref:
            baseline = baseline_ref[key]
            baseline_fwIoU = baseline.get('fwIoU', 0)
            baseline_mIoU = baseline.get('mIoU', 0)
            
            result = row.to_dict()
            result['baseline_fwIoU'] = baseline_fwIoU
            result['baseline_mIoU'] = baseline_mIoU
            result['fwIoU_gain'] = row['fwIoU'] - baseline_fwIoU
            result['mIoU_gain'] = row['mIoU'] - baseline_mIoU
            results.append(result)
        else:
            # No baseline match - use strategy comparison within same config
            result = row.to_dict()
            result['fwIoU_gain'] = np.nan
            result['mIoU_gain'] = np.nan
            results.append(result)
    
    return pd.DataFrame(results)


def create_family_gain_analysis(df: pd.DataFrame, output_dir: Path):
    """Create family gain/loss analysis plots using fwIoU as primary metric."""
    
    # Filter out baseline and unknown
    analysis_df = df[~df['family'].isin(['Baseline', 'Unknown'])]
    
    if len(analysis_df) == 0 or 'fwIoU_gain' not in analysis_df.columns:
        print("No gain data available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot 1: Family fwIoU Gain ranking
    ax1 = axes[0, 0]
    family_gains = analysis_df.groupby('family')['fwIoU_gain'].agg(['mean', 'std', 'count']).reset_index()
    family_gains = family_gains.sort_values('mean', ascending=True)
    
    colors = ['green' if x > 0 else 'red' for x in family_gains['mean']]
    bars = ax1.barh(range(len(family_gains)), family_gains['mean'], 
                    xerr=family_gains['std'], capsize=5, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(family_gains)))
    ax1.set_yticklabels(family_gains['family'])
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax1.set_xlabel('fwIoU Gain over Baseline (%)')
    ax1.set_title('Family fwIoU Gain Ranking', fontsize=12, fontweight='bold')
    
    # Add count annotations
    for i, (mean, count) in enumerate(zip(family_gains['mean'], family_gains['count'])):
        ax1.annotate(f'n={count}', xy=(mean + family_gains['std'].iloc[i] + 0.1, i), va='center', fontsize=9)
    
    # Plot 2: Gain distribution by family
    ax2 = axes[0, 1]
    family_order = analysis_df.groupby('family')['fwIoU_gain'].mean().sort_values(ascending=False).index
    try:
        sns.boxplot(data=analysis_df, x='fwIoU_gain', y='family', order=family_order, ax=ax2,
                    hue='family', legend=False, palette='RdYlGn')
    except (UnboundLocalError, ValueError):
        # Fallback for seaborn 0.13.x bug with single-element groups
        data_by_family = [analysis_df[analysis_df['family'] == f]['fwIoU_gain'].values for f in family_order]
        bp = ax2.boxplot(data_by_family, vert=False, patch_artist=True, tick_labels=list(family_order))
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('fwIoU Gain over Baseline (%)')
    ax2.set_ylabel('')
    ax2.set_title('fwIoU Gain Distribution by Family', fontsize=12, fontweight='bold')
    
    # Plot 3: Strategies with largest gains
    ax3 = axes[1, 0]
    strategy_gains = analysis_df.groupby('strategy')['fwIoU_gain'].mean().sort_values(ascending=False)
    top_gainers = strategy_gains.head(10)
    
    colors = ['green' if x > 0 else 'red' for x in top_gainers.values]
    ax3.barh(range(len(top_gainers)), top_gainers.values, color=colors, alpha=0.7)
    ax3.set_yticks(range(len(top_gainers)))
    ax3.set_yticklabels(top_gainers.index)
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('fwIoU Gain over Baseline (%)')
    ax3.set_title('Top 10 Strategies by fwIoU Gain', fontsize=12, fontweight='bold')
    
    # Plot 4: Positive vs Negative gains per family
    ax4 = axes[1, 1]
    positive_counts = analysis_df[analysis_df['fwIoU_gain'] > 0].groupby('family').size()
    negative_counts = analysis_df[analysis_df['fwIoU_gain'] <= 0].groupby('family').size()
    
    families = sorted(set(positive_counts.index) | set(negative_counts.index))
    pos_vals = [positive_counts.get(f, 0) for f in families]
    neg_vals = [negative_counts.get(f, 0) for f in families]
    
    x = np.arange(len(families))
    width = 0.35
    ax4.bar(x - width/2, pos_vals, width, label='Gain > 0', color='green', alpha=0.7)
    ax4.bar(x + width/2, neg_vals, width, label='Gain ≤ 0', color='red', alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels(families, rotation=45, ha='right')
    ax4.set_ylabel('Number of Configurations')
    ax4.set_title('Positive vs Negative fwIoU Gains', fontsize=12, fontweight='bold')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "family_fwIoU_gains.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'family_fwIoU_gains.png'}")


def create_intra_family_gains(df: pd.DataFrame, family: str, output_dir: Path):
    """Create intra-family gain analysis plot."""
    
    family_df = df[df['family'] == family]
    
    if len(family_df) == 0 or 'fwIoU_gain' not in family_df.columns:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Strategy gain comparison
    ax1 = axes[0]
    strategy_gains = family_df.groupby('strategy')['fwIoU_gain'].agg(['mean', 'std']).reset_index()
    strategy_gains = strategy_gains.sort_values('mean', ascending=False)
    
    colors = ['green' if x > 0 else 'red' for x in strategy_gains['mean']]
    ax1.barh(range(len(strategy_gains)), strategy_gains['mean'], 
             xerr=strategy_gains['std'], capsize=3, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(strategy_gains)))
    ax1.set_yticklabels(strategy_gains['strategy'])
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('fwIoU Gain over Baseline (%)')
    ax1.set_title(f'{family}: Strategy fwIoU Gains', fontweight='bold')
    
    # Plot 2: Gain by dataset (if multiple)
    ax2 = axes[1]
    if family_df['dataset'].nunique() > 1:
        dataset_strategy_gains = family_df.pivot_table(values='fwIoU_gain', index='strategy', 
                                                       columns='dataset', aggfunc='mean')
        if not dataset_strategy_gains.empty:
            sns.heatmap(dataset_strategy_gains, annot=True, fmt='.2f', cmap='RdYlGn', 
                       center=0, ax=ax2, cbar_kws={'label': 'fwIoU Gain (%)'})
            ax2.set_title(f'{family}: Gains by Dataset', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'Single dataset', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f'{family}: Gains by Dataset', fontweight='bold')
    
    plt.tight_layout()
    safe_name = family.replace(' ', '_').replace('/', '_')
    plt.savefig(output_dir / f"intra_{safe_name}_gains.png", dpi=150, bbox_inches='tight')
    plt.close()


def create_gain_summary_table(df: pd.DataFrame, output_path: Path):
    """Create summary table showing gains/losses per family."""
    
    summary = []
    
    for family in df['family'].unique():
        if family in ['Unknown']:
            continue
        
        family_df = df[df['family'] == family]
        
        # Determine the gain column to use
        gain_col = 'fwIoU_gain' if 'fwIoU_gain' in family_df.columns else 'fwIoU'
        miou_gain_col = 'mIoU_gain' if 'mIoU_gain' in family_df.columns else 'mIoU'
        
        row = {
            'Family': family,
            'N_Strategies': family_df['strategy'].nunique(),
            'N_Results': len(family_df),
            'fwIoU_Gain_Mean': family_df[gain_col].mean() if gain_col in family_df.columns else np.nan,
            'fwIoU_Gain_Std': family_df[gain_col].std() if gain_col in family_df.columns else np.nan,
            'mIoU_Gain_Mean': family_df[miou_gain_col].mean() if miou_gain_col in family_df.columns else np.nan,
            'Best_Strategy': family_df.groupby('strategy')[gain_col].mean().idxmax() if len(family_df) > 0 else 'N/A',
            'Best_Gain': family_df.groupby('strategy')[gain_col].mean().max() if len(family_df) > 0 else np.nan,
            'Strategies_Improving': (family_df.groupby('strategy')[gain_col].mean() > 0).sum() if gain_col in family_df.columns else 0
        }
        summary.append(row)
    
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values('fwIoU_Gain_Mean', ascending=False)
    summary_df.to_csv(output_path, index=False)
    
    return summary_df


def create_publication_figure(df: pd.DataFrame, output_path: Path):
    """Create publication-ready summary figure using gains/losses."""
    
    analysis_df = df[~df['family'].isin(['Baseline', 'Unknown'])]
    
    if len(analysis_df) == 0:
        return
    
    fig = plt.figure(figsize=(18, 12))
    
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Family gain ranking (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    if 'fwIoU_gain' in analysis_df.columns:
        family_gains = analysis_df.groupby('family')['fwIoU_gain'].mean().sort_values(ascending=True)
        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in family_gains.values]
        ax1.barh(range(len(family_gains)), family_gains.values, color=colors, alpha=0.8)
        ax1.set_yticks(range(len(family_gains)))
        ax1.set_yticklabels(family_gains.index)
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax1.set_xlabel('Mean fwIoU Gain (%)', fontsize=10)
        ax1.set_title('(a) Family fwIoU Gain Ranking', fontsize=12, fontweight='bold')
    
    # 2. Strategy count per family (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    strategy_counts = analysis_df.groupby('family')['strategy'].nunique().sort_values(ascending=True)
    ax2.barh(range(len(strategy_counts)), strategy_counts.values, color='steelblue', alpha=0.7)
    ax2.set_yticks(range(len(strategy_counts)))
    ax2.set_yticklabels(strategy_counts.index)
    ax2.set_xlabel('Number of Strategies', fontsize=10)
    ax2.set_title('(b) Strategies per Family', fontsize=12, fontweight='bold')
    
    # 3. Win rate (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    if 'fwIoU_gain' in analysis_df.columns:
        win_rates = analysis_df.groupby('family').apply(
            lambda x: (x['fwIoU_gain'] > 0).sum() / len(x) * 100
        ).sort_values(ascending=True)
        colors = plt.cm.RdYlGn(win_rates.values / 100)
        ax3.barh(range(len(win_rates)), win_rates.values, color=colors)
        ax3.set_yticks(range(len(win_rates)))
        ax3.set_yticklabels(win_rates.index)
        ax3.axvline(x=50, color='black', linestyle='--', linewidth=1)
        ax3.set_xlabel('Win Rate (%)', fontsize=10)
        ax3.set_title('(c) % Configs Beating Baseline', fontsize=12, fontweight='bold')
    
    # 4. Top strategies by gain (bottom, spans 2 cols)
    ax4 = fig.add_subplot(gs[1, :2])
    if 'fwIoU_gain' in analysis_df.columns:
        strategy_gains = analysis_df.groupby(['family', 'strategy'])['fwIoU_gain'].mean().reset_index()
        strategy_gains = strategy_gains.sort_values('fwIoU_gain', ascending=False).head(15)
        
        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in strategy_gains['fwIoU_gain']]
        y_labels = [f"{row['strategy']} ({row['family'][:10]})" for _, row in strategy_gains.iterrows()]
        
        ax4.barh(range(len(strategy_gains)), strategy_gains['fwIoU_gain'].values, color=colors, alpha=0.7)
        ax4.set_yticks(range(len(strategy_gains)))
        ax4.set_yticklabels(y_labels, fontsize=9)
        ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_xlabel('fwIoU Gain over Baseline (%)', fontsize=10)
        ax4.set_title('(d) Top 15 Strategies by fwIoU Gain', fontsize=12, fontweight='bold')
    
    # 5. Summary statistics (bottom right)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    if 'fwIoU_gain' in analysis_df.columns:
        stats_text = "Summary Statistics\n"
        stats_text += "─" * 30 + "\n\n"
        stats_text += f"Total Strategies: {analysis_df['strategy'].nunique()}\n"
        stats_text += f"Total Configurations: {len(analysis_df)}\n\n"
        stats_text += f"Mean fwIoU Gain: {analysis_df['fwIoU_gain'].mean():.2f}%\n"
        stats_text += f"Configs > Baseline: {(analysis_df['fwIoU_gain'] > 0).sum()}\n"
        stats_text += f"Configs ≤ Baseline: {(analysis_df['fwIoU_gain'] <= 0).sum()}\n\n"
        
        best = analysis_df.groupby('strategy')['fwIoU_gain'].mean().idxmax()
        best_gain = analysis_df.groupby('strategy')['fwIoU_gain'].mean().max()
        worst = analysis_df.groupby('strategy')['fwIoU_gain'].mean().idxmin()
        worst_gain = analysis_df.groupby('strategy')['fwIoU_gain'].mean().min()
        
        stats_text += f"Best: {best}\n  +{best_gain:.2f}% fwIoU\n\n"
        stats_text += f"Worst: {worst}\n  {worst_gain:.2f}% fwIoU"
        
        ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Strategy Family Analysis: fwIoU Gains/Losses vs Baseline', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Main function."""
    
    print("=" * 60)
    print("Strategy Family Analysis (fwIoU-based)")
    print("=" * 60)
    
    # Paths
    downstream_csv = "${HOME}/repositories/PROVE/downstream_results.csv"
    
    # Load results
    print("\nLoading downstream results...")
    df = load_all_results(downstream_csv)
    print(f"Loaded {len(df)} results from {df['strategy'].nunique()} strategies")
    
    # Load baseline reference (using dataframe as fallback)
    print("\nLoading baseline reference...")
    baseline_ref = load_baseline_reference(df)
    print(f"Loaded {len(baseline_ref)} baseline reference points")
    
    # Print family distribution
    print("\nFamily distribution:")
    for family in sorted(df['family'].unique()):
        count = df[df['family'] == family]['strategy'].nunique()
        strategies = list(df[df['family'] == family]['strategy'].unique())[:3]
        print(f"  {family}: {count} strategies - {', '.join(strategies)}...")
    
    # Compute gains/losses relative to baseline
    print("\nComputing gains/losses relative to baseline...")
    df = compute_gains_losses(df, baseline_ref)
    
    if 'fwIoU_gain' in df.columns:
        valid_gains = df[df['fwIoU_gain'].notna()]
        print(f"Computed gains for {len(valid_gains)} configurations")
        print(f"Mean fwIoU Gain: {valid_gains['fwIoU_gain'].mean():.2f}%")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Family gain analysis
    print("  - Family gain analysis...")
    create_family_gain_analysis(df, OUTPUT_DIR)
    
    # 2. Intra-family gains
    print("  - Intra-family gains...")
    for family in df['family'].unique():
        if family not in ['Unknown', 'Baseline']:
            create_intra_family_gains(df, family, OUTPUT_DIR)
    
    # 3. Publication figure
    print("  - Publication figure...")
    create_publication_figure(df, OUTPUT_DIR / "family_gains_publication.png")
    
    # Create summary table
    print("\nCreating summary table...")
    summary_df = create_gain_summary_table(df, OUTPUT_DIR / "family_gains_summary.csv")
    
    # Print summary
    print("\n" + "=" * 60)
    print("FAMILY GAINS SUMMARY (fwIoU)")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    
    # Save report
    with open(OUTPUT_DIR / "family_gains_report.txt", 'w') as f:
        f.write("Strategy Family Analysis Report (fwIoU-based)\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("PRIMARY METRIC: fwIoU (frequency-weighted IoU)\n")
        f.write("COMPARISON: Gains/Losses relative to baseline\n\n")
        
        f.write("FAMILY PERFORMANCE SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(summary_df.to_string(index=False))
        
        if 'fwIoU_gain' in df.columns:
            f.write("\n\nKEY FINDINGS\n")
            f.write("-" * 40 + "\n")
            
            analysis_df = df[~df['family'].isin(['Baseline', 'Unknown'])]
            valid = analysis_df[analysis_df['fwIoU_gain'].notna()]
            
            f.write(f"Total Configurations: {len(valid)}\n")
            f.write(f"Configurations Improving: {(valid['fwIoU_gain'] > 0).sum()} ({(valid['fwIoU_gain'] > 0).mean()*100:.1f}%)\n")
            f.write(f"Mean fwIoU Gain: {valid['fwIoU_gain'].mean():.2f}%\n")
            
            best = valid.groupby('strategy')['fwIoU_gain'].mean().idxmax()
            best_gain = valid.groupby('strategy')['fwIoU_gain'].mean().max()
            f.write(f"\nBest Strategy: {best} (+{best_gain:.2f}% fwIoU)\n")
    
    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
