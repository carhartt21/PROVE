#!/usr/bin/env python3
"""
Comprehensive Baseline Analysis Script (Updated for mIoU)

Analyzes baseline performance on a per-domain and per-dataset basis.
Establishes the reference point for all subsequent comparisons.

This script generates:
1. Baseline performance summary (per-dataset, per-model, per-domain)
2. Reference values for computing gains/losses
3. Baseline variance analysis
4. Performance consistency across domains

Primary Metric: mIoU (mean Intersection over Union)
    - Recommended for domain robustness analysis
    - Equal weight to all classes (not biased by class frequency)
    
Secondary Metrics: fwIoU (frequency-weighted IoU), PA (Pixel Accuracy)
    - fwIoU gives more weight to dominant classes (road, sky)
    - Can be misleading for cross-domain comparison due to class distribution shifts

NOTE: fwIoU was previously used but is NOT recommended for domain gap analysis
because foggy/rainy scenes naturally have fewer small objects, making them
appear "easier" when using fwIoU. mIoU provides fairer comparison.

Usage:
    mamba run -n prove python analyze_baseline_miou.py

Output:
    result_figures/baseline_miou/
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
OUTPUT_DIR = Path("result_figures/baseline_miou")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Weather domains
WEATHER_DOMAINS = ['clear_day', 'cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy']
ADVERSE_DOMAINS = ['foggy', 'night', 'rainy', 'snowy', 'dawn_dusk']
NORMAL_DOMAINS = ['clear_day', 'cloudy']

# Datasets
DATASETS = ['acdc', 'bdd10k', 'idd-aw', 'mapillaryvistas', 'outside15k']

# Models
MODELS = ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5']


def load_baseline_results(downstream_csv: str) -> pd.DataFrame:
    """Load baseline results from downstream_results.csv."""
    
    df = pd.read_csv(downstream_csv)
    
    # Filter for baseline only
    baseline_df = df[df['strategy'] == 'baseline'].copy()
    
    # Filter out domain-specific models (e.g., deeplabv3plus_r50_clear_day)
    # These are from domain-specific training
    standard_models = ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5']
    baseline_df = baseline_df[baseline_df['model'].isin(standard_models)]
    
    return baseline_df


def load_baseline_domain_results(weights_root: str) -> pd.DataFrame:
    """Load per-domain baseline results from test_report.txt files."""
    
    results = []
    baseline_dir = Path(weights_root) / "baseline"
    
    if not baseline_dir.exists():
        print(f"Baseline directory not found: {baseline_dir}")
        return pd.DataFrame()
    
    for dataset_dir in baseline_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        
        for model_dir in dataset_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            
            # Skip domain-specific models
            if any(domain in model for domain in WEATHER_DOMAINS):
                continue
            
            # Look for test_results_detailed
            detailed_dir = model_dir / "test_results_detailed"
            if not detailed_dir.exists():
                continue
            
            # Find most recent test report
            for subdir in sorted(detailed_dir.iterdir(), reverse=True):
                if not subdir.is_dir():
                    continue
                
                report_file = subdir / "test_report.txt"
                if report_file.exists():
                    domain_metrics = parse_test_report(report_file)
                    
                    for domain, metrics in domain_metrics.items():
                        results.append({
                            'dataset': dataset,
                            'model': model,
                            'domain': domain,
                            'mIoU': metrics.get('mIoU', metrics.get('miou')),
                            'fwIoU': metrics.get('fwIoU', metrics.get('fwiou')),
                            'aAcc': metrics.get('aAcc', metrics.get('PA')),
                            'mAcc': metrics.get('mAcc'),
                            'num_images': metrics.get('num_images', 0)
                        })
                    break  # Use most recent report
    
    return pd.DataFrame(results)


def parse_test_report(filepath: Path) -> dict:
    """Parse test_report.txt to extract per-domain metrics."""
    
    domain_metrics = {}
    current_domain = None
    in_per_domain = False
    
    content = filepath.read_text()
    
    for line in content.split('\n'):
        line = line.strip()
        
        if 'PER-DOMAIN METRICS' in line:
            in_per_domain = True
            continue
        
        if 'PER-CLASS METRICS' in line:
            in_per_domain = False
            continue
        
        if in_per_domain:
            # Check for domain header
            for domain in WEATHER_DOMAINS:
                if f'--- {domain} ---' in line.lower() or domain.lower() in line.lower():
                    if '---' in line:
                        current_domain = domain
                        domain_metrics[current_domain] = {}
                        break
            
            if current_domain and ':' in line:
                # Parse metric
                parts = line.split(':')
                if len(parts) >= 2:
                    key = parts[0].strip()
                    try:
                        value = float(parts[1].strip().split()[0])
                        domain_metrics[current_domain][key] = value
                    except:
                        pass
    
    return domain_metrics


def create_baseline_summary(df: pd.DataFrame, output_path: Path):
    """Create baseline performance summary using mIoU as primary metric."""
    
    summary = []
    
    # Per-dataset summary
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        
        row = {
            'Level': 'Dataset',
            'Name': dataset.upper(),
            'mIoU_mean': dataset_df['mIoU'].mean(),
            'mIoU_std': dataset_df['mIoU'].std(),
            'mIoU_min': dataset_df['mIoU'].min(),
            'mIoU_max': dataset_df['mIoU'].max(),
            'fwIoU_mean': dataset_df['fwIoU'].mean() if 'fwIoU' in dataset_df.columns else np.nan,
            'aAcc_mean': dataset_df['aAcc'].mean() if 'aAcc' in dataset_df.columns else np.nan,
            'N_Models': dataset_df['model'].nunique()
        }
        summary.append(row)
    
    # Per-model summary (across all datasets)
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        
        row = {
            'Level': 'Model',
            'Name': model,
            'mIoU_mean': model_df['mIoU'].mean(),
            'mIoU_std': model_df['mIoU'].std(),
            'mIoU_min': model_df['mIoU'].min(),
            'mIoU_max': model_df['mIoU'].max(),
            'fwIoU_mean': model_df['fwIoU'].mean() if 'fwIoU' in model_df.columns else np.nan,
            'aAcc_mean': model_df['aAcc'].mean() if 'aAcc' in model_df.columns else np.nan,
            'N_Models': model_df['dataset'].nunique()
        }
        summary.append(row)
    
    # Overall summary
    overall_row = {
        'Level': 'Overall',
        'Name': 'ALL',
        'mIoU_mean': df['mIoU'].mean(),
        'mIoU_std': df['mIoU'].std(),
        'mIoU_min': df['mIoU'].min(),
        'mIoU_max': df['mIoU'].max(),
        'fwIoU_mean': df['fwIoU'].mean() if 'fwIoU' in df.columns else np.nan,
        'aAcc_mean': df['aAcc'].mean() if 'aAcc' in df.columns else np.nan,
        'N_Models': len(df)
    }
    summary.append(overall_row)
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_path, index=False)
    
    return summary_df


def create_domain_gap_analysis(domain_df: pd.DataFrame, output_path: Path):
    """Create domain gap analysis using mIoU as primary metric."""
    
    if domain_df.empty:
        print("No domain data available for domain gap analysis")
        return None
    
    analysis = []
    
    # Compute baseline mIoU per domain (averaged across datasets/models)
    domain_means = domain_df.groupby('domain')['mIoU'].agg(['mean', 'std', 'count']).reset_index()
    domain_means.columns = ['domain', 'mIoU_mean', 'mIoU_std', 'count']
    
    # Reference: clear_day performance
    clear_day_miou = domain_means[domain_means['domain'] == 'clear_day']['mIoU_mean'].values
    if len(clear_day_miou) > 0:
        clear_day_miou = clear_day_miou[0]
    else:
        clear_day_miou = domain_means['mIoU_mean'].mean()
    
    # Compute domain gap (drop from clear_day)
    domain_means['domain_gap'] = clear_day_miou - domain_means['mIoU_mean']
    domain_means['gap_percentage'] = (domain_means['domain_gap'] / clear_day_miou) * 100
    
    # Categorize domains
    domain_means['category'] = domain_means['domain'].apply(
        lambda x: 'normal' if x in NORMAL_DOMAINS else 'adverse'
    )
    
    domain_means.to_csv(output_path, index=False)
    
    return domain_means


def create_visualizations(domain_df: pd.DataFrame, summary_df: pd.DataFrame):
    """Create comprehensive visualizations using mIoU."""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Domain Performance Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    domain_means = domain_df.groupby('domain')['mIoU'].mean().reindex(WEATHER_DOMAINS)
    
    colors = ['#2ecc71' if d in NORMAL_DOMAINS else '#e74c3c' for d in domain_means.index]
    
    bars = ax.bar(range(len(domain_means)), domain_means.values, color=colors, 
                  alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(domain_means)))
    ax.set_xticklabels(domain_means.index, rotation=45, ha='right')
    ax.set_ylabel('mIoU (%)')
    ax.set_title('Baseline mIoU Performance by Weather Domain\n(Green=Normal, Red=Adverse)')
    
    # Add reference line
    overall_mean = domain_means.mean()
    ax.axhline(y=overall_mean, color='gray', linestyle='--', 
               label=f'Average: {overall_mean:.1f}%')
    ax.legend()
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, domain_means.values)):
        ax.annotate(f'{val:.1f}', (bar.get_x() + bar.get_width()/2, val + 0.5),
                   ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'domain_miou_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Domain Gap Chart (relative to clear_day)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    clear_day_miou = domain_means.get('clear_day', domain_means.mean())
    domain_gaps = clear_day_miou - domain_means
    
    colors = ['#3498db' if gap <= 0 else '#e74c3c' for gap in domain_gaps.values]
    
    bars = ax.bar(range(len(domain_gaps)), domain_gaps.values, color=colors,
                  alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(domain_gaps)))
    ax.set_xticklabels(domain_gaps.index, rotation=45, ha='right')
    ax.set_ylabel('mIoU Drop from Clear_Day (%)')
    ax.set_title('Domain Gap Analysis (Baseline)\nmIoU Drop Relative to Clear_Day')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, domain_gaps.values)):
        offset = 0.5 if val >= 0 else -1.5
        ax.annotate(f'{val:.1f}', (bar.get_x() + bar.get_width()/2, val + offset),
                   ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'domain_gap_miou.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Dataset x Domain Heatmap
    if len(domain_df) > 0:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        pivot = domain_df.pivot_table(values='mIoU', index='dataset', 
                                       columns='domain', aggfunc='mean')
        pivot = pivot.reindex(columns=WEATHER_DOMAINS)
        
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn',
                   ax=ax, vmin=0, vmax=80, cbar_kws={'label': 'mIoU (%)'})
        ax.set_title('Baseline mIoU: Dataset × Domain')
        ax.set_xlabel('Weather Domain')
        ax.set_ylabel('Dataset')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'dataset_domain_heatmap_miou.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Model x Domain Comparison
    if len(domain_df) > 0:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        pivot = domain_df.pivot_table(values='mIoU', index='model',
                                       columns='domain', aggfunc='mean')
        pivot = pivot.reindex(columns=WEATHER_DOMAINS)
        
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn',
                   ax=ax, vmin=0, vmax=80, cbar_kws={'label': 'mIoU (%)'})
        ax.set_title('Baseline mIoU: Model × Domain')
        ax.set_xlabel('Weather Domain')
        ax.set_ylabel('Model')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'model_domain_heatmap_miou.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Adverse vs Normal Summary
    fig, ax = plt.subplots(figsize=(10, 6))
    
    adverse_df = domain_df[domain_df['domain'].isin(ADVERSE_DOMAINS)]
    normal_df = domain_df[domain_df['domain'].isin(NORMAL_DOMAINS)]
    
    data = {
        'Normal\n(clear_day, cloudy)': normal_df['mIoU'].mean() if len(normal_df) > 0 else 0,
        'Adverse\n(foggy, night, rainy, snowy, dawn_dusk)': adverse_df['mIoU'].mean() if len(adverse_df) > 0 else 0
    }
    
    bars = ax.bar(data.keys(), data.values(), color=['#2ecc71', '#e74c3c'], 
                  alpha=0.8, edgecolor='black')
    ax.set_ylabel('mIoU (%)')
    ax.set_title('Baseline Performance: Normal vs Adverse Conditions')
    
    for bar, val in zip(bars, data.values()):
        ax.annotate(f'{val:.1f}%', (bar.get_x() + bar.get_width()/2, val + 0.5),
                   ha='center', fontsize=12, fontweight='bold')
    
    # Compute domain gap
    if data['Normal\n(clear_day, cloudy)'] > 0:
        gap = data['Normal\n(clear_day, cloudy)'] - data['Adverse\n(foggy, night, rainy, snowy, dawn_dusk)']
        ax.text(0.5, 0.95, f'Domain Gap: {gap:.1f}% mIoU drop',
               transform=ax.transAxes, ha='center', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'adverse_vs_normal_miou.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Publication Summary Figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Top-left: Domain performance
    ax1 = fig.add_subplot(gs[0, 0])
    domain_means_plot = domain_df.groupby('domain')['mIoU'].mean().reindex(WEATHER_DOMAINS)
    colors = ['#2ecc71' if d in NORMAL_DOMAINS else '#e74c3c' for d in domain_means_plot.index]
    ax1.bar(range(len(domain_means_plot)), domain_means_plot.values, color=colors, alpha=0.8)
    ax1.set_xticks(range(len(domain_means_plot)))
    ax1.set_xticklabels(domain_means_plot.index, rotation=45, ha='right')
    ax1.set_ylabel('mIoU (%)')
    ax1.set_title('Baseline by Domain')
    ax1.axhline(y=domain_means_plot.mean(), color='gray', linestyle='--')
    
    # Top-right: Domain gap
    ax2 = fig.add_subplot(gs[0, 1])
    clear_day_val = domain_means_plot.get('clear_day', domain_means_plot.mean())
    gaps = clear_day_val - domain_means_plot
    colors = ['#3498db' if g <= 0 else '#e74c3c' for g in gaps.values]
    ax2.bar(range(len(gaps)), gaps.values, color=colors, alpha=0.8)
    ax2.set_xticks(range(len(gaps)))
    ax2.set_xticklabels(gaps.index, rotation=45, ha='right')
    ax2.set_ylabel('mIoU Drop (%)')
    ax2.set_title('Domain Gap (vs Clear_Day)')
    ax2.axhline(y=0, color='black', linestyle='-')
    
    # Bottom-left: Dataset comparison
    ax3 = fig.add_subplot(gs[1, 0])
    dataset_means = domain_df.groupby('dataset')['mIoU'].mean()
    ax3.barh(dataset_means.index, dataset_means.values, color='#3498db', alpha=0.8)
    ax3.set_xlabel('mIoU (%)')
    ax3.set_title('Baseline by Dataset')
    for i, v in enumerate(dataset_means.values):
        ax3.annotate(f'{v:.1f}', (v + 0.5, i), va='center')
    
    # Bottom-right: Model comparison
    ax4 = fig.add_subplot(gs[1, 1])
    model_means = domain_df.groupby('model')['mIoU'].mean()
    ax4.barh(model_means.index, model_means.values, color='#9b59b6', alpha=0.8)
    ax4.set_xlabel('mIoU (%)')
    ax4.set_title('Baseline by Model')
    for i, v in enumerate(model_means.values):
        ax4.annotate(f'{v:.1f}', (v + 0.5, i), va='center')
    
    fig.suptitle('Baseline Performance Analysis (mIoU)', fontsize=14, fontweight='bold')
    
    plt.savefig(OUTPUT_DIR / 'publication_summary_miou.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {OUTPUT_DIR}/")


def generate_text_report(domain_df: pd.DataFrame, summary_df: pd.DataFrame):
    """Generate a comprehensive text report."""
    
    report = []
    report.append("=" * 80)
    report.append("BASELINE PERFORMANCE ANALYSIS (mIoU Primary Metric)")
    report.append("=" * 80)
    report.append("")
    
    report.append("METRIC CHOICE RATIONALE:")
    report.append("-" * 40)
    report.append("Primary Metric: mIoU (mean Intersection over Union)")
    report.append("  - Equal weight to all classes")
    report.append("  - Not biased by class frequency distribution")
    report.append("  - Recommended for domain robustness analysis")
    report.append("")
    report.append("Why NOT fwIoU?")
    report.append("  - fwIoU weights by class frequency (pixel count)")
    report.append("  - Foggy/adverse scenes have fewer small objects")
    report.append("  - This makes adverse domains appear 'easier' with fwIoU")
    report.append("  - mIoU gives fairer cross-domain comparison")
    report.append("")
    
    # Overall summary
    if domain_df is not None and len(domain_df) > 0:
        report.append("OVERALL BASELINE PERFORMANCE")
        report.append("-" * 40)
        report.append(f"Overall mIoU: {domain_df['mIoU'].mean():.2f}% (±{domain_df['mIoU'].std():.2f})")
        if 'fwIoU' in domain_df.columns:
            report.append(f"Overall fwIoU: {domain_df['fwIoU'].mean():.2f}%")
        if 'aAcc' in domain_df.columns:
            report.append(f"Overall aAcc: {domain_df['aAcc'].mean():.2f}%")
        report.append("")
        
        # Per-domain analysis
        report.append("PER-DOMAIN PERFORMANCE (mIoU)")
        report.append("-" * 40)
        domain_means = domain_df.groupby('domain')['mIoU'].agg(['mean', 'std', 'count'])
        clear_day_miou = domain_means.loc['clear_day', 'mean'] if 'clear_day' in domain_means.index else domain_means['mean'].mean()
        
        report.append(f"{'Domain':<15} {'mIoU':>8} {'Std':>8} {'Gap':>8} {'N':>6}")
        report.append("-" * 50)
        for domain in WEATHER_DOMAINS:
            if domain in domain_means.index:
                row = domain_means.loc[domain]
                gap = clear_day_miou - row['mean']
                marker = '⬇' if gap > 0 else '⬆' if gap < 0 else ''
                report.append(f"{domain:<15} {row['mean']:>8.2f} {row['std']:>8.2f} {gap:>+8.2f}{marker} {int(row['count']):>6}")
        report.append("")
        
        # Adverse vs Normal
        adverse_df = domain_df[domain_df['domain'].isin(ADVERSE_DOMAINS)]
        normal_df = domain_df[domain_df['domain'].isin(NORMAL_DOMAINS)]
        
        report.append("ADVERSE VS NORMAL CONDITIONS")
        report.append("-" * 40)
        normal_mean = normal_df['mIoU'].mean() if len(normal_df) > 0 else 0
        adverse_mean = adverse_df['mIoU'].mean() if len(adverse_df) > 0 else 0
        gap = normal_mean - adverse_mean
        
        report.append(f"Normal conditions (clear_day, cloudy):  {normal_mean:.2f}% mIoU")
        report.append(f"Adverse conditions (foggy, night, etc): {adverse_mean:.2f}% mIoU")
        report.append(f"Domain Gap: {gap:.2f}% mIoU drop on adverse conditions")
        report.append("")
        
        # Per-dataset analysis
        report.append("PER-DATASET PERFORMANCE (mIoU)")
        report.append("-" * 40)
        dataset_means = domain_df.groupby('dataset')['mIoU'].agg(['mean', 'std', 'count'])
        report.append(f"{'Dataset':<20} {'mIoU':>8} {'Std':>8} {'N':>6}")
        report.append("-" * 45)
        for dataset in dataset_means.index:
            row = dataset_means.loc[dataset]
            report.append(f"{dataset:<20} {row['mean']:>8.2f} {row['std']:>8.2f} {int(row['count']):>6}")
        report.append("")
        
        # Per-model analysis
        report.append("PER-MODEL PERFORMANCE (mIoU)")
        report.append("-" * 40)
        model_means = domain_df.groupby('model')['mIoU'].agg(['mean', 'std', 'count'])
        report.append(f"{'Model':<25} {'mIoU':>8} {'Std':>8} {'N':>6}")
        report.append("-" * 50)
        for model in model_means.index:
            row = model_means.loc[model]
            report.append(f"{model:<25} {row['mean']:>8.2f} {row['std']:>8.2f} {int(row['count']):>6}")
    
    report_text = '\n'.join(report)
    
    # Save report
    report_path = OUTPUT_DIR / 'baseline_report_miou.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    return report_text


def main():
    print("=" * 60)
    print("Baseline Analysis (mIoU Primary Metric)")
    print("=" * 60)
    print()
    
    # Load domain results from WEIGHTS directory
    weights_root = "/scratch/aaa_exchange/AWARE/WEIGHTS"
    
    print("Loading per-domain baseline results...")
    domain_df = load_baseline_domain_results(weights_root)
    
    if len(domain_df) == 0:
        print("No domain data found. Trying alternative paths...")
        # Try alternative paths
        for alt_path in ["/scratch/aaa_exchange/AWARE/WEIGHTS"]:
            domain_df = load_baseline_domain_results(alt_path)
            if len(domain_df) > 0:
                print(f"Found data in {alt_path}")
                break
    
    print(f"Loaded {len(domain_df)} domain results")
    
    # Create summary
    print("\nGenerating summary statistics...")
    summary_df = create_baseline_summary(domain_df, OUTPUT_DIR / 'baseline_summary_miou.csv')
    
    # Create domain gap analysis
    print("Analyzing domain gaps...")
    domain_gap_df = create_domain_gap_analysis(domain_df, OUTPUT_DIR / 'domain_gap_analysis_miou.csv')
    
    # Generate text report
    print("\nGenerating text report...")
    generate_text_report(domain_df, summary_df)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(domain_df, summary_df)
    
    # Save raw data
    domain_df.to_csv(OUTPUT_DIR / 'baseline_domain_results_miou.csv', index=False)
    print(f"\nRaw data saved to {OUTPUT_DIR}/baseline_domain_results_miou.csv")
    
    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
