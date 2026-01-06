#!/usr/bin/env python3
"""
Strategy Family Analysis Script

Analyzes segmentation performance grouped by strategy families:
1. 2D Rendering (imgaug_weather, automold, Weather_Effect_Generator)
2. CNN/GAN (Attribute_Hallucination, cycleGAN, CUT, stargan_v2, SUSTechGAN)
3. Style Transfer (NST, LANIT, TSIT, StyleID)
4. Diffusion (Img2Img, IP2P, EDICT, UniControl)
5. Multimodal Diffusion (flux1_kontext, step1x_new, Qwen_Image_Edit)
6. Standard Augmentation (autoaugment, randaugment, photometric_distort)
7. Standard Mixing (cutmix, mixup)

Note: Combined strategies (with + in name) are analyzed separately in 
analyze_combination_ablation.py and stored in WEIGHTS_COMBINATIONS.

Provides:
- Intra-family analysis (comparing methods within each family)
- Inter-family analysis (comparing families against each other)

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
        "gen_imgaug_weather",
        "gen_automold", 
        "gen_Weather_Effect_Generator",
        "photometric_distort"
    ],
    "CNN/GAN": [
        "gen_Attribute_Hallucination",
        "gen_cycleGAN",
        "gen_CUT",
        "gen_stargan_v2",
        "gen_SUSTechGAN"
    ],
    "Style Transfer": [
        "gen_NST",
        "gen_LANIT",
        "gen_TSIT",
        "gen_StyleID"
    ],
    "Diffusion": [
        "gen_Img2Img",
        "gen_IP2P",
        "gen_EDICT",
        "gen_UniControl"
    ],
    "Multimodal Diffusion": [
        "gen_flux1_kontext",
        "gen_step1x_new",
        "gen_Qwen_Image_Edit"
    ],
    "Standard Augmentation": [
        "std_autoaugment",
        "std_randaugment",
        "photometric_distort"
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


def load_all_results(weights_root: str, downstream_csv: str) -> pd.DataFrame:
    """Load results from downstream_results.csv and enrich with family info."""
    
    # Load main results
    df = pd.read_csv(downstream_csv)
    
    # EXCLUDE combination strategies (they are analyzed separately)
    df = df[~df['strategy'].str.contains(r'\+', regex=True)]
    
    # Add family information
    df['family'] = df['strategy'].apply(lambda x: get_family(x))
    
    # Identify combined strategies (should be none after filtering)
    df['is_combined'] = df['strategy'].apply(lambda x: '+' in x)
    
    # Get component families for combined strategies
    df['component_families'] = df['strategy'].apply(get_component_families)
    
    return df


def get_family(strategy: str) -> str:
    """Get the family for a strategy, handling combined strategies."""
    if '+' in strategy:
        return "Combined"
    return STRATEGY_TO_FAMILY.get(strategy, "Unknown")


def get_component_families(strategy: str) -> list:
    """Get component families for combined strategies."""
    if '+' not in strategy:
        return [get_family(strategy)]
    
    components = strategy.split('+')
    families = []
    for comp in components:
        comp = comp.strip()
        family = STRATEGY_TO_FAMILY.get(comp, "Unknown")
        if family not in families:
            families.append(family)
    return families


def load_per_domain_results(weights_root: str) -> pd.DataFrame:
    """Load per-domain results from metrics_per_domain.json and test_report.txt files."""
    
    results = []
    weights_path = Path(weights_root)
    
    for strategy_dir in weights_path.iterdir():
        if not strategy_dir.is_dir():
            continue
        strategy = strategy_dir.name
        
        for dataset_dir in strategy_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name
            
            for model_dir in dataset_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model = model_dir.name
                
                # Try metrics_per_domain.json first
                metrics_file = model_dir / "metrics_per_domain.json"
                if metrics_file.exists():
                    try:
                        with open(metrics_file) as f:
                            data = json.load(f)
                        for domain, metrics in data.items():
                            if domain in WEATHER_DOMAINS:
                                results.append({
                                    'strategy': strategy,
                                    'family': get_family(strategy),
                                    'dataset': dataset,
                                    'model': model,
                                    'domain': domain,
                                    'mIoU': metrics.get('mIoU', metrics.get('miou', None)),
                                    'mAcc': metrics.get('mAcc', metrics.get('macc', None)),
                                    'source': 'metrics_per_domain.json'
                                })
                    except Exception as e:
                        print(f"Error loading {metrics_file}: {e}")
                        continue
                
                # Try test_report.txt
                report_file = model_dir / "test_report.txt"
                if report_file.exists():
                    try:
                        domain_metrics = parse_test_report(report_file)
                        for domain, metrics in domain_metrics.items():
                            if domain in WEATHER_DOMAINS:
                                results.append({
                                    'strategy': strategy,
                                    'family': get_family(strategy),
                                    'dataset': dataset,
                                    'model': model,
                                    'domain': domain,
                                    'mIoU': metrics.get('mIoU'),
                                    'mAcc': metrics.get('mAcc'),
                                    'source': 'test_report.txt'
                                })
                    except Exception as e:
                        continue
    
    df = pd.DataFrame(results)
    if not df.empty:
        # Drop duplicates keeping first (metrics_per_domain.json preferred)
        df = df.drop_duplicates(subset=['strategy', 'dataset', 'model', 'domain'], keep='first')
    
    return df


def parse_test_report(filepath: Path) -> dict:
    """Parse test_report.txt to extract per-domain metrics."""
    domain_metrics = {}
    current_domain = None
    
    with open(filepath) as f:
        content = f.read()
    
    for line in content.split('\n'):
        line = line.strip()
        
        # Check for domain header
        for domain in WEATHER_DOMAINS:
            if domain.lower() in line.lower() and ':' in line:
                current_domain = domain
                domain_metrics[current_domain] = {}
                break
        
        # Parse metrics
        if current_domain and 'mIoU' in line:
            try:
                # Extract mIoU value
                parts = line.split(':')
                if len(parts) >= 2:
                    value = float(parts[1].strip().replace('%', ''))
                    domain_metrics[current_domain]['mIoU'] = value
            except:
                pass
        
        if current_domain and 'mAcc' in line:
            try:
                parts = line.split(':')
                if len(parts) >= 2:
                    value = float(parts[1].strip().replace('%', ''))
                    domain_metrics[current_domain]['mAcc'] = value
            except:
                pass
    
    return domain_metrics


def compute_family_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute statistics for each family."""
    
    # Group by family
    family_stats = []
    
    for family in df['family'].unique():
        family_df = df[df['family'] == family]
        
        stats = {
            'family': family,
            'n_strategies': family_df['strategy'].nunique(),
            'n_results': len(family_df),
            'mean_mIoU': family_df['mIoU'].mean(),
            'std_mIoU': family_df['mIoU'].std(),
            'min_mIoU': family_df['mIoU'].min(),
            'max_mIoU': family_df['mIoU'].max(),
            'strategies': list(family_df['strategy'].unique())
        }
        
        family_stats.append(stats)
    
    return pd.DataFrame(family_stats)


def compute_baseline_relative(df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics relative to baseline for each configuration."""
    
    results = []
    
    for _, row in df.iterrows():
        if row['family'] == 'Baseline':
            continue
        
        # Find matching baseline
        baseline_match = baseline_df[
            (baseline_df['dataset'] == row['dataset']) &
            (baseline_df['model'] == row['model'])
        ]
        
        if 'domain' in df.columns and 'domain' in baseline_df.columns:
            baseline_match = baseline_match[baseline_match['domain'] == row['domain']]
        
        if len(baseline_match) > 0:
            baseline_mIoU = baseline_match['mIoU'].iloc[0]
            
            result = row.to_dict()
            result['baseline_mIoU'] = baseline_mIoU
            result['improvement'] = row['mIoU'] - baseline_mIoU
            result['relative_improvement'] = (row['mIoU'] - baseline_mIoU) / baseline_mIoU * 100 if baseline_mIoU > 0 else 0
            results.append(result)
    
    return pd.DataFrame(results)


def create_intra_family_comparison(df: pd.DataFrame, family: str, output_path: Path):
    """Create visualization comparing strategies within a family."""
    
    family_df = df[df['family'] == family]
    
    if len(family_df) == 0:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: mIoU by strategy
    ax1 = axes[0]
    strategy_means = family_df.groupby('strategy')['mIoU'].agg(['mean', 'std']).reset_index()
    strategy_means = strategy_means.sort_values('mean', ascending=False)
    
    bars = ax1.barh(range(len(strategy_means)), strategy_means['mean'], 
                    xerr=strategy_means['std'], capsize=3, color=plt.cm.Set3(np.linspace(0, 1, len(strategy_means))))
    ax1.set_yticks(range(len(strategy_means)))
    ax1.set_yticklabels(strategy_means['strategy'])
    ax1.set_xlabel('mIoU')
    ax1.set_title(f'{family}: Strategy Comparison')
    ax1.axvline(x=family_df['mIoU'].mean(), color='red', linestyle='--', label='Family Mean')
    ax1.legend()
    
    # Plot 2: Improvement over baseline (if available)
    if 'improvement' in family_df.columns:
        ax2 = axes[1]
        improvement_means = family_df.groupby('strategy')['improvement'].agg(['mean', 'std']).reset_index()
        improvement_means = improvement_means.sort_values('mean', ascending=False)
        
        colors = ['green' if x > 0 else 'red' for x in improvement_means['mean']]
        ax2.barh(range(len(improvement_means)), improvement_means['mean'],
                 xerr=improvement_means['std'], capsize=3, color=colors)
        ax2.set_yticks(range(len(improvement_means)))
        ax2.set_yticklabels(improvement_means['strategy'])
        ax2.set_xlabel('Improvement over Baseline (mIoU)')
        ax2.set_title(f'{family}: Improvement Analysis')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    else:
        axes[1].text(0.5, 0.5, 'No baseline comparison available', 
                     ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title(f'{family}: Improvement Analysis')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_inter_family_comparison(df: pd.DataFrame, output_path: Path):
    """Create visualization comparing families against each other."""
    
    # Exclude baseline and unknown
    family_df = df[~df['family'].isin(['Baseline', 'Unknown', 'Combined'])]
    
    if len(family_df) == 0:
        print("No data for inter-family comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot 1: Family mIoU comparison
    ax1 = axes[0, 0]
    family_stats = family_df.groupby('family')['mIoU'].agg(['mean', 'std', 'count']).reset_index()
    family_stats = family_stats.sort_values('mean', ascending=True)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(family_stats)))
    bars = ax1.barh(range(len(family_stats)), family_stats['mean'],
                    xerr=family_stats['std'], capsize=5, color=colors)
    ax1.set_yticks(range(len(family_stats)))
    ax1.set_yticklabels(family_stats['family'])
    ax1.set_xlabel('Mean mIoU')
    ax1.set_title('Inter-Family Comparison: Mean mIoU')
    
    # Add count annotations
    for i, (mean, count) in enumerate(zip(family_stats['mean'], family_stats['count'])):
        ax1.annotate(f'n={count}', xy=(mean + family_stats['std'].iloc[i] + 0.5, i),
                     va='center', fontsize=8)
    
    # Plot 2: Improvement distribution by family
    ax2 = axes[0, 1]
    if 'improvement' in family_df.columns:
        order = family_df.groupby('family')['improvement'].mean().sort_values(ascending=False).index
        sns.boxplot(data=family_df, x='improvement', y='family', order=order, ax=ax2, palette='Set2')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=1)
        ax2.set_xlabel('Improvement over Baseline (mIoU)')
        ax2.set_title('Inter-Family: Improvement Distribution')
    else:
        ax2.text(0.5, 0.5, 'No improvement data', ha='center', va='center', transform=ax2.transAxes)
    
    # Plot 3: Family performance by dataset
    ax3 = axes[1, 0]
    if 'dataset' in family_df.columns:
        pivot = family_df.pivot_table(values='mIoU', index='family', columns='dataset', aggfunc='mean')
        if not pivot.empty:
            sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax3, center=pivot.values.mean())
            ax3.set_title('Family Performance by Dataset')
    
    # Plot 4: Family performance variability
    ax4 = axes[1, 1]
    family_var = family_df.groupby('family')['mIoU'].agg(['mean', 'std', 'min', 'max']).reset_index()
    family_var['range'] = family_var['max'] - family_var['min']
    family_var = family_var.sort_values('mean', ascending=False)
    
    x = range(len(family_var))
    ax4.bar(x, family_var['range'], color='lightblue', label='Range (Max-Min)')
    ax4.bar(x, family_var['std'] * 2, color='darkblue', alpha=0.7, label='2×Std Dev')
    ax4.set_xticks(x)
    ax4.set_xticklabels(family_var['family'], rotation=45, ha='right')
    ax4.set_ylabel('mIoU Variability')
    ax4.set_title('Family Performance Variability')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_domain_family_analysis(domain_df: pd.DataFrame, output_path: Path):
    """Create analysis of family performance across weather domains."""
    
    if len(domain_df) == 0:
        print("No domain-level data available")
        return
    
    # Exclude baseline
    family_df = domain_df[~domain_df['family'].isin(['Baseline', 'Unknown'])]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot 1: Heatmap of family x domain performance
    ax1 = axes[0, 0]
    pivot = family_df.pivot_table(values='mIoU', index='family', columns='domain', aggfunc='mean')
    if not pivot.empty:
        # Reorder columns
        col_order = [d for d in WEATHER_DOMAINS if d in pivot.columns]
        pivot = pivot[col_order]
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax1)
        ax1.set_title('Family Performance by Weather Domain')
    
    # Plot 2: Adverse vs Normal conditions by family
    ax2 = axes[0, 1]
    family_df['condition'] = family_df['domain'].apply(
        lambda x: 'Adverse' if x in ADVERSE_DOMAINS else 'Normal'
    )
    condition_pivot = family_df.pivot_table(values='mIoU', index='family', columns='condition', aggfunc='mean')
    if not condition_pivot.empty:
        condition_pivot.plot(kind='bar', ax=ax2, color=['#2ecc71', '#e74c3c'])
        ax2.set_ylabel('Mean mIoU')
        ax2.set_title('Family: Adverse vs Normal Conditions')
        ax2.legend(title='Condition')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 3: Robustness score (adverse/normal ratio)
    ax3 = axes[1, 0]
    if not condition_pivot.empty and 'Adverse' in condition_pivot.columns and 'Normal' in condition_pivot.columns:
        robustness = (condition_pivot['Adverse'] / condition_pivot['Normal']).sort_values(ascending=True)
        colors = plt.cm.RdYlGn((robustness - robustness.min()) / (robustness.max() - robustness.min()))
        ax3.barh(range(len(robustness)), robustness.values, color=colors)
        ax3.set_yticks(range(len(robustness)))
        ax3.set_yticklabels(robustness.index)
        ax3.axvline(x=1.0, color='black', linestyle='--', linewidth=1)
        ax3.set_xlabel('Robustness Score (Adverse/Normal)')
        ax3.set_title('Family Robustness to Adverse Conditions')
    
    # Plot 4: Best family per domain
    ax4 = axes[1, 1]
    best_family = pivot.idxmax()
    domain_counts = best_family.value_counts()
    ax4.pie(domain_counts.values, labels=domain_counts.index, autopct='%1.1f%%',
            colors=plt.cm.Set3(np.linspace(0, 1, len(domain_counts))))
    ax4.set_title('Best Family by Domain (# domains won)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_combined_strategy_analysis(df: pd.DataFrame, output_path: Path):
    """Analyze combined strategies and their component families."""
    
    combined_df = df[df['is_combined']]
    
    if len(combined_df) == 0:
        print("No combined strategies found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot 1: Combined strategy performance
    ax1 = axes[0, 0]
    strategy_means = combined_df.groupby('strategy')['mIoU'].mean().sort_values(ascending=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(strategy_means)))
    ax1.barh(range(len(strategy_means)), strategy_means.values, color=colors)
    ax1.set_yticks(range(len(strategy_means)))
    ax1.set_yticklabels(strategy_means.index)
    ax1.set_xlabel('Mean mIoU')
    ax1.set_title('Combined Strategies Performance')
    
    # Plot 2: Combined vs single strategy comparison
    ax2 = axes[0, 1]
    single_df = df[~df['is_combined'] & (df['family'] != 'Baseline')]
    
    data_to_plot = []
    if len(single_df) > 0:
        data_to_plot.append({'Type': 'Single Strategy', 'mIoU': single_df['mIoU'].mean()})
    if len(combined_df) > 0:
        data_to_plot.append({'Type': 'Combined Strategy', 'mIoU': combined_df['mIoU'].mean()})
    
    if data_to_plot:
        plot_df = pd.DataFrame(data_to_plot)
        ax2.bar(plot_df['Type'], plot_df['mIoU'], color=['#3498db', '#9b59b6'])
        ax2.set_ylabel('Mean mIoU')
        ax2.set_title('Single vs Combined Strategy Comparison')
    
    # Plot 3: Improvement by combination type
    ax3 = axes[1, 0]
    if 'improvement' in combined_df.columns and len(combined_df) > 0:
        improvement_data = combined_df.groupby('strategy')['improvement'].mean().sort_values(ascending=False)
        colors = ['green' if x > 0 else 'red' for x in improvement_data.values]
        ax3.barh(range(len(improvement_data)), improvement_data.values, color=colors)
        ax3.set_yticks(range(len(improvement_data)))
        ax3.set_yticklabels(improvement_data.index)
        ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_xlabel('Improvement over Baseline')
        ax3.set_title('Combined Strategy Improvement')
    
    # Plot 4: Distribution comparison
    ax4 = axes[1, 1]
    if len(single_df) > 0 and len(combined_df) > 0:
        ax4.hist(single_df['mIoU'], bins=20, alpha=0.5, label='Single', color='#3498db')
        ax4.hist(combined_df['mIoU'], bins=20, alpha=0.5, label='Combined', color='#9b59b6')
        ax4.axvline(single_df['mIoU'].mean(), color='#3498db', linestyle='--', linewidth=2)
        ax4.axvline(combined_df['mIoU'].mean(), color='#9b59b6', linestyle='--', linewidth=2)
        ax4.set_xlabel('mIoU')
        ax4.set_ylabel('Frequency')
        ax4.set_title('mIoU Distribution: Single vs Combined')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_table(df: pd.DataFrame, domain_df: pd.DataFrame = None) -> pd.DataFrame:
    """Create summary table with family statistics."""
    
    summary = []
    
    families = [f for f in df['family'].unique() if f not in ['Unknown']]
    
    for family in families:
        family_data = df[df['family'] == family]
        
        row = {
            'Family': family,
            'N_Strategies': family_data['strategy'].nunique(),
            'N_Results': len(family_data),
            'Mean_mIoU': family_data['mIoU'].mean(),
            'Std_mIoU': family_data['mIoU'].std(),
            'Best_Strategy': family_data.groupby('strategy')['mIoU'].mean().idxmax() if len(family_data) > 0 else 'N/A',
            'Best_mIoU': family_data.groupby('strategy')['mIoU'].mean().max() if len(family_data) > 0 else 'N/A'
        }
        
        # Add improvement stats if available
        if 'improvement' in family_data.columns:
            row['Mean_Improvement'] = family_data['improvement'].mean()
            row['Strategies_Improving'] = (family_data.groupby('strategy')['improvement'].mean() > 0).sum()
        
        # Add domain stats if available
        if domain_df is not None and len(domain_df) > 0:
            fam_domain = domain_df[domain_df['family'] == family]
            if len(fam_domain) > 0:
                adverse = fam_domain[fam_domain['domain'].isin(ADVERSE_DOMAINS)]['mIoU'].mean()
                normal = fam_domain[fam_domain['domain'].isin(NORMAL_DOMAINS)]['mIoU'].mean()
                row['Adverse_mIoU'] = adverse
                row['Normal_mIoU'] = normal
                row['Robustness'] = adverse / normal if normal > 0 else 0
        
        summary.append(row)
    
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values('Mean_mIoU', ascending=False)
    
    return summary_df


def create_publication_figure(df: pd.DataFrame, domain_df: pd.DataFrame, output_path: Path):
    """Create publication-ready summary figure."""
    
    fig = plt.figure(figsize=(20, 16))
    
    # Create grid for subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Filter data
    analysis_df = df[~df['family'].isin(['Baseline', 'Unknown'])]
    
    # 1. Family performance ranking (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    family_perf = analysis_df.groupby('family')['mIoU'].mean().sort_values(ascending=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(family_perf)))
    ax1.barh(range(len(family_perf)), family_perf.values, color=colors)
    ax1.set_yticks(range(len(family_perf)))
    ax1.set_yticklabels(family_perf.index)
    ax1.set_xlabel('Mean mIoU', fontsize=10)
    ax1.set_title('(a) Family Performance Ranking', fontsize=12, fontweight='bold')
    
    # 2. Improvement distribution (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    if 'improvement' in analysis_df.columns:
        family_order = analysis_df.groupby('family')['improvement'].mean().sort_values(ascending=False).index
        sns.boxplot(data=analysis_df, x='improvement', y='family', order=family_order,
                    ax=ax2, palette='RdYlGn')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
        ax2.set_xlabel('Improvement over Baseline (mIoU)', fontsize=10)
        ax2.set_ylabel('')
        ax2.set_title('(b) Improvement Distribution', fontsize=12, fontweight='bold')
    
    # 3. Strategy count per family (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    strategy_counts = analysis_df.groupby('family')['strategy'].nunique().sort_values(ascending=True)
    ax3.barh(range(len(strategy_counts)), strategy_counts.values, color='steelblue')
    ax3.set_yticks(range(len(strategy_counts)))
    ax3.set_yticklabels(strategy_counts.index)
    ax3.set_xlabel('Number of Strategies', fontsize=10)
    ax3.set_title('(c) Strategies per Family', fontsize=12, fontweight='bold')
    
    # 4. Domain heatmap (middle row, spans 2 columns)
    ax4 = fig.add_subplot(gs[1, :2])
    if domain_df is not None and len(domain_df) > 0:
        domain_analysis = domain_df[~domain_df['family'].isin(['Baseline', 'Unknown'])]
        pivot = domain_analysis.pivot_table(values='mIoU', index='family', columns='domain', aggfunc='mean')
        if not pivot.empty:
            col_order = [d for d in WEATHER_DOMAINS if d in pivot.columns]
            pivot = pivot[col_order]
            # Sort by mean performance
            pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
            sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax4,
                       cbar_kws={'label': 'mIoU'})
            ax4.set_title('(d) Family Performance by Weather Domain', fontsize=12, fontweight='bold')
    
    # 5. Robustness comparison (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    if domain_df is not None and len(domain_df) > 0:
        domain_analysis = domain_df[~domain_df['family'].isin(['Baseline', 'Unknown'])]
        domain_analysis['condition'] = domain_analysis['domain'].apply(
            lambda x: 'Adverse' if x in ADVERSE_DOMAINS else 'Normal'
        )
        cond_pivot = domain_analysis.pivot_table(values='mIoU', index='family', columns='condition', aggfunc='mean')
        if 'Adverse' in cond_pivot.columns and 'Normal' in cond_pivot.columns:
            robustness = (cond_pivot['Adverse'] / cond_pivot['Normal']).sort_values(ascending=True)
            colors = plt.cm.RdYlGn((robustness - robustness.min()) / (robustness.max() - robustness.min() + 0.01))
            ax5.barh(range(len(robustness)), robustness.values, color=colors)
            ax5.set_yticks(range(len(robustness)))
            ax5.set_yticklabels(robustness.index)
            ax5.axvline(x=1.0, color='red', linestyle='--', linewidth=1.5)
            ax5.set_xlabel('Robustness (Adverse/Normal)', fontsize=10)
            ax5.set_title('(e) Weather Robustness', fontsize=12, fontweight='bold')
    
    # 6. Best strategies per family (bottom row)
    ax6 = fig.add_subplot(gs[2, :])
    best_strategies = analysis_df.groupby('family').apply(
        lambda x: x.groupby('strategy')['mIoU'].mean().nlargest(3)
    ).reset_index()
    best_strategies.columns = ['family', 'strategy', 'mIoU']
    
    families = best_strategies['family'].unique()
    x_positions = np.arange(len(families))
    width = 0.25
    
    for i, rank in enumerate([0, 1, 2]):
        values = []
        labels = []
        for family in families:
            family_best = best_strategies[best_strategies['family'] == family]
            if len(family_best) > rank:
                values.append(family_best.iloc[rank]['mIoU'])
                labels.append(family_best.iloc[rank]['strategy'][:15])
            else:
                values.append(0)
                labels.append('')
        
        bars = ax6.bar(x_positions + (i - 1) * width, values, width, 
                       label=f'Rank {rank+1}', alpha=0.8)
    
    ax6.set_xticks(x_positions)
    ax6.set_xticklabels(families, rotation=45, ha='right')
    ax6.set_ylabel('mIoU', fontsize=10)
    ax6.set_title('(f) Top 3 Strategies per Family', fontsize=12, fontweight='bold')
    ax6.legend(loc='upper right')
    
    plt.suptitle('Strategy Family Analysis Summary', fontsize=16, fontweight='bold', y=1.02)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # Skip PDF to avoid timeout issues
    # plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Publication figure saved to {output_path}")


def main():
    """Main function to run family analysis."""
    
    print("=" * 60)
    print("Strategy Family Analysis")
    print("=" * 60)
    
    # Paths
    weights_root = "/scratch/aaa_exchange/AWARE/WEIGHTS"
    downstream_csv = "/home/mima2416/repositories/PROVE/downstream_results.csv"
    
    # Load main results
    print("\nLoading downstream results...")
    df = load_all_results(weights_root, downstream_csv)
    print(f"Loaded {len(df)} results from {df['strategy'].nunique()} strategies")
    
    # Print family distribution
    print("\nFamily distribution:")
    family_counts = df.groupby('family')['strategy'].nunique()
    for family, count in family_counts.items():
        strategies = df[df['family'] == family]['strategy'].unique()[:5]
        print(f"  {family}: {count} strategies - {', '.join(strategies)}")
    
    # Separate baseline for comparison
    baseline_df = df[df['family'] == 'Baseline']
    
    # Compute baseline-relative metrics
    print("\nComputing baseline-relative metrics...")
    if len(baseline_df) > 0:
        df_with_improvement = compute_baseline_relative(df, baseline_df)
        # Merge back
        df = pd.merge(df, df_with_improvement[['strategy', 'dataset', 'model', 'improvement', 'relative_improvement']],
                      on=['strategy', 'dataset', 'model'], how='left')
    
    # Load per-domain results
    print("\nLoading per-domain results...")
    domain_df = load_per_domain_results(weights_root)
    print(f"Loaded {len(domain_df)} domain-level results")
    
    if len(domain_df) > 0:
        # Compute domain-level baseline relative
        domain_baseline = domain_df[domain_df['family'] == 'Baseline']
        if len(domain_baseline) > 0:
            domain_df_improved = compute_baseline_relative(domain_df, domain_baseline)
            domain_df = pd.merge(domain_df, 
                                domain_df_improved[['strategy', 'dataset', 'model', 'domain', 'improvement']],
                                on=['strategy', 'dataset', 'model', 'domain'], how='left')
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # 1. Inter-family comparison
    print("  - Inter-family comparison...")
    create_inter_family_comparison(df, OUTPUT_DIR / "inter_family_comparison.png")
    
    # 2. Intra-family comparisons
    print("  - Intra-family comparisons...")
    for family in df['family'].unique():
        if family not in ['Unknown', 'Baseline']:
            safe_name = family.replace(' ', '_').replace('/', '_')
            create_intra_family_comparison(df, family, OUTPUT_DIR / f"intra_{safe_name}.png")
    
    # 3. Domain-family analysis
    if len(domain_df) > 0:
        print("  - Domain-family analysis...")
        create_domain_family_analysis(domain_df, OUTPUT_DIR / "domain_family_analysis.png")
    
    # 4. Combined strategy analysis
    print("  - Combined strategy analysis...")
    create_combined_strategy_analysis(df, OUTPUT_DIR / "combined_strategy_analysis.png")
    
    # 5. Publication figure
    print("  - Publication figure...")
    create_publication_figure(df, domain_df, OUTPUT_DIR / "family_publication_summary.png")
    
    # Create summary table
    print("\nCreating summary table...")
    summary_df = create_summary_table(df, domain_df)
    summary_df.to_csv(OUTPUT_DIR / "family_summary.csv", index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("FAMILY SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    
    # Save detailed text summary
    with open(OUTPUT_DIR / "family_analysis_report.txt", 'w') as f:
        f.write("Strategy Family Analysis Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. FAMILY DEFINITIONS\n")
        f.write("-" * 40 + "\n")
        for family, strategies in STRATEGY_FAMILIES.items():
            f.write(f"\n{family}:\n")
            for s in strategies:
                f.write(f"  - {s}\n")
        
        f.write("\n\n2. FAMILY PERFORMANCE SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(summary_df.to_string(index=False))
        
        if 'improvement' in df.columns:
            f.write("\n\n3. IMPROVEMENT ANALYSIS\n")
            f.write("-" * 40 + "\n")
            improvement_by_family = df.groupby('family')['improvement'].agg(['mean', 'std', 'min', 'max'])
            improvement_by_family = improvement_by_family.sort_values('mean', ascending=False)
            f.write(improvement_by_family.to_string())
        
        if len(domain_df) > 0:
            f.write("\n\n4. DOMAIN PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            domain_pivot = domain_df.pivot_table(values='mIoU', index='family', columns='domain', aggfunc='mean')
            f.write(domain_pivot.to_string())
    
    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("Generated files:")
    for f in OUTPUT_DIR.iterdir():
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
