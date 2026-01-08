#!/usr/bin/env python3
"""
Comprehensive Baseline Analysis Script

Analyzes baseline performance on a per-domain and per-dataset basis.
Establishes the reference point for all subsequent comparisons.

This script generates:
1. Baseline performance summary (per-dataset, per-model, per-domain)
2. Reference values for computing gains/losses
3. Baseline variance analysis
4. Performance consistency across domains

Primary Metric: fwIoU (frequency-weighted IoU)
Secondary Metrics: mIoU, PA (Pixel Accuracy) where relevant

Usage:
    mamba run -n prove python analyze_baseline.py

Output:
    result_figures/baseline/
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
OUTPUT_DIR = Path("result_figures/baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Weather domains
WEATHER_DOMAINS = ['clear_day', 'cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy']
ADVERSE_DOMAINS = ['foggy', 'night', 'rainy', 'snowy']
NORMAL_DOMAINS = ['clear_day', 'cloudy', 'dawn_dusk']

# Datasets
DATASETS = ['acdc', 'cityscapes', 'mapillary']

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
                            'fwIoU': metrics.get('fwIoU', metrics.get('fwiou')),
                            'mIoU': metrics.get('mIoU', metrics.get('miou')),
                            'aAcc': metrics.get('aAcc', metrics.get('PA')),
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
    """Create baseline performance summary."""
    
    summary = []
    
    # Per-dataset summary
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        
        row = {
            'Level': 'Dataset',
            'Name': dataset.upper(),
            'fwIoU_mean': dataset_df['fwIoU'].mean(),
            'fwIoU_std': dataset_df['fwIoU'].std(),
            'fwIoU_min': dataset_df['fwIoU'].min(),
            'fwIoU_max': dataset_df['fwIoU'].max(),
            'mIoU_mean': dataset_df['mIoU'].mean(),
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
            'fwIoU_mean': model_df['fwIoU'].mean(),
            'fwIoU_std': model_df['fwIoU'].std(),
            'fwIoU_min': model_df['fwIoU'].min(),
            'fwIoU_max': model_df['fwIoU'].max(),
            'mIoU_mean': model_df['mIoU'].mean(),
            'aAcc_mean': model_df['aAcc'].mean() if 'aAcc' in model_df.columns else np.nan,
            'N_Models': model_df['dataset'].nunique()
        }
        summary.append(row)
    
    # Overall
    row = {
        'Level': 'Overall',
        'Name': 'BASELINE',
        'fwIoU_mean': df['fwIoU'].mean(),
        'fwIoU_std': df['fwIoU'].std(),
        'fwIoU_min': df['fwIoU'].min(),
        'fwIoU_max': df['fwIoU'].max(),
        'mIoU_mean': df['mIoU'].mean(),
        'aAcc_mean': df['aAcc'].mean() if 'aAcc' in df.columns else np.nan,
        'N_Models': len(df)
    }
    summary.append(row)
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_path, index=False)
    
    return summary_df


def create_baseline_domain_summary(domain_df: pd.DataFrame, output_path: Path):
    """Create baseline domain performance summary."""
    
    if len(domain_df) == 0:
        return pd.DataFrame()
    
    summary = []
    
    # Per-domain summary
    for domain in WEATHER_DOMAINS:
        dom_data = domain_df[domain_df['domain'] == domain]
        if len(dom_data) == 0:
            continue
        
        condition = 'Adverse' if domain in ADVERSE_DOMAINS else 'Normal'
        
        row = {
            'Domain': domain,
            'Condition': condition,
            'fwIoU_mean': dom_data['fwIoU'].mean(),
            'fwIoU_std': dom_data['fwIoU'].std(),
            'fwIoU_min': dom_data['fwIoU'].min(),
            'fwIoU_max': dom_data['fwIoU'].max(),
            'mIoU_mean': dom_data['mIoU'].mean() if 'mIoU' in dom_data.columns else np.nan,
            'N_Results': len(dom_data)
        }
        summary.append(row)
    
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values('fwIoU_mean', ascending=False)
    summary_df.to_csv(output_path, index=False)
    
    return summary_df


def create_baseline_visualizations(df: pd.DataFrame, domain_df: pd.DataFrame, output_dir: Path):
    """Create baseline visualization plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot 1: Baseline fwIoU by dataset and model
    ax1 = axes[0, 0]
    pivot = df.pivot_table(values='fwIoU', index='model', columns='dataset', aggfunc='mean')
    pivot.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_ylabel('fwIoU (%)')
    ax1.set_title('Baseline fwIoU by Model and Dataset', fontsize=12, fontweight='bold')
    ax1.legend(title='Dataset')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.axhline(y=df['fwIoU'].mean(), color='red', linestyle='--', label=f"Mean: {df['fwIoU'].mean():.1f}%")
    
    # Plot 2: mIoU vs fwIoU scatter
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df['mIoU'], df['fwIoU'], c=df['dataset'].astype('category').cat.codes,
                          cmap='Set1', s=100, alpha=0.7)
    ax2.set_xlabel('mIoU (%)')
    ax2.set_ylabel('fwIoU (%)')
    ax2.set_title('mIoU vs fwIoU Relationship', fontsize=12, fontweight='bold')
    
    # Add diagonal line
    lims = [min(ax2.get_xlim()[0], ax2.get_ylim()[0]), max(ax2.get_xlim()[1], ax2.get_ylim()[1])]
    ax2.plot(lims, lims, 'k--', alpha=0.5, label='y=x')
    ax2.legend()
    
    # Plot 3: Per-domain baseline performance
    ax3 = axes[1, 0]
    if len(domain_df) > 0:
        domain_means = domain_df.groupby('domain')['fwIoU'].mean().sort_values(ascending=True)
        colors = ['#e74c3c' if d in ADVERSE_DOMAINS else '#2ecc71' for d in domain_means.index]
        ax3.barh(range(len(domain_means)), domain_means.values, color=colors)
        ax3.set_yticks(range(len(domain_means)))
        ax3.set_yticklabels(domain_means.index)
        ax3.set_xlabel('fwIoU (%)')
        ax3.set_title('Baseline fwIoU by Weather Domain', fontsize=12, fontweight='bold')
        ax3.axvline(x=domain_df['fwIoU'].mean(), color='black', linestyle='--', 
                    label=f"Mean: {domain_df['fwIoU'].mean():.1f}%")
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No per-domain data available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Baseline fwIoU by Weather Domain', fontsize=12, fontweight='bold')
    
    # Plot 4: Performance variance
    ax4 = axes[1, 1]
    if len(domain_df) > 0:
        # Box plot by domain
        domain_order = domain_df.groupby('domain')['fwIoU'].mean().sort_values(ascending=False).index
        sns.boxplot(data=domain_df, x='domain', y='fwIoU', order=domain_order, ax=ax4, palette='Set2')
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
        ax4.set_ylabel('fwIoU (%)')
        ax4.set_title('Baseline fwIoU Distribution by Domain', fontsize=12, fontweight='bold')
    else:
        # Use dataset instead
        sns.boxplot(data=df, x='dataset', y='fwIoU', ax=ax4, palette='Set2')
        ax4.set_ylabel('fwIoU (%)')
        ax4.set_title('Baseline fwIoU Distribution by Dataset', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "baseline_overview.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'baseline_overview.png'}")


def create_reference_table(df: pd.DataFrame, domain_df: pd.DataFrame, output_path: Path):
    """Create reference table for computing gains/losses."""
    
    reference = {}
    
    # Per dataset-model combination
    for _, row in df.iterrows():
        key = f"{row['dataset']}_{row['model']}"
        reference[key] = {
            'dataset': row['dataset'],
            'model': row['model'],
            'fwIoU': row['fwIoU'],
            'mIoU': row['mIoU'],
            'aAcc': row.get('aAcc', np.nan)
        }
    
    # Per dataset-model-domain combination
    if len(domain_df) > 0:
        for _, row in domain_df.iterrows():
            key = f"{row['dataset']}_{row['model']}_{row['domain']}"
            reference[key] = {
                'dataset': row['dataset'],
                'model': row['model'],
                'domain': row['domain'],
                'fwIoU': row['fwIoU'],
                'mIoU': row.get('mIoU', np.nan),
                'aAcc': row.get('aAcc', np.nan)
            }
    
    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(reference, f, indent=2)
    
    print(f"Saved: {output_path}")
    return reference


def main():
    """Main function."""
    
    print("=" * 60)
    print("Comprehensive Baseline Analysis")
    print("=" * 60)
    
    # Paths
    downstream_csv = "/home/mima2416/repositories/PROVE/downstream_results.csv"
    weights_root = "/scratch/aaa_exchange/AWARE/WEIGHTS"
    
    # Load baseline results
    print("\nLoading baseline results...")
    df = load_baseline_results(downstream_csv)
    print(f"Loaded {len(df)} baseline results")
    
    if len(df) == 0:
        print("No baseline results found!")
        return
    
    # Print baseline overview
    print(f"\nBaseline Overview:")
    print(f"  Datasets: {df['dataset'].unique().tolist()}")
    print(f"  Models: {df['model'].unique().tolist()}")
    print(f"  Overall fwIoU: {df['fwIoU'].mean():.2f}% ± {df['fwIoU'].std():.2f}%")
    print(f"  Overall mIoU: {df['mIoU'].mean():.2f}% ± {df['mIoU'].std():.2f}%")
    
    # Load per-domain results
    print("\nLoading per-domain results...")
    domain_df = load_baseline_domain_results(weights_root)
    print(f"Loaded {len(domain_df)} domain-level results")
    
    # Create summaries
    print("\nGenerating summaries...")
    
    # Overall summary
    summary_df = create_baseline_summary(df, OUTPUT_DIR / "baseline_summary.csv")
    print("\nBaseline Summary:")
    print(summary_df.to_string(index=False))
    
    # Domain summary
    if len(domain_df) > 0:
        domain_summary_df = create_baseline_domain_summary(domain_df, OUTPUT_DIR / "baseline_domain_summary.csv")
        print("\nDomain Summary:")
        print(domain_summary_df.to_string(index=False))
    
    # Create reference table
    print("\nCreating reference table...")
    reference = create_reference_table(df, domain_df, OUTPUT_DIR / "baseline_reference.json")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_baseline_visualizations(df, domain_df, OUTPUT_DIR)
    
    # Save detailed text report
    with open(OUTPUT_DIR / "baseline_report.txt", 'w') as f:
        f.write("COMPREHENSIVE BASELINE ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. OVERVIEW\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Baseline Results: {len(df)}\n")
        f.write(f"Datasets: {', '.join(df['dataset'].unique())}\n")
        f.write(f"Models: {', '.join(df['model'].unique())}\n\n")
        
        f.write("2. PRIMARY METRIC: fwIoU\n")
        f.write("-" * 40 + "\n")
        f.write(f"Overall Mean: {df['fwIoU'].mean():.2f}%\n")
        f.write(f"Overall Std: {df['fwIoU'].std():.2f}%\n")
        f.write(f"Range: [{df['fwIoU'].min():.2f}%, {df['fwIoU'].max():.2f}%]\n\n")
        
        f.write("3. SECONDARY METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"mIoU Mean: {df['mIoU'].mean():.2f}%\n")
        if 'aAcc' in df.columns:
            f.write(f"PA (aAcc) Mean: {df['aAcc'].mean():.2f}%\n")
        f.write("\n")
        
        f.write("4. PER-DATASET BASELINE (fwIoU)\n")
        f.write("-" * 40 + "\n")
        for dataset in df['dataset'].unique():
            ds_df = df[df['dataset'] == dataset]
            f.write(f"  {dataset.upper()}: {ds_df['fwIoU'].mean():.2f}% ± {ds_df['fwIoU'].std():.2f}%\n")
        f.write("\n")
        
        f.write("5. PER-MODEL BASELINE (fwIoU)\n")
        f.write("-" * 40 + "\n")
        for model in df['model'].unique():
            m_df = df[df['model'] == model]
            f.write(f"  {model}: {m_df['fwIoU'].mean():.2f}% ± {m_df['fwIoU'].std():.2f}%\n")
        f.write("\n")
        
        if len(domain_df) > 0:
            f.write("6. PER-DOMAIN BASELINE (fwIoU)\n")
            f.write("-" * 40 + "\n")
            domain_means = domain_df.groupby('domain')['fwIoU'].mean().sort_values(ascending=False)
            for domain, fwiou in domain_means.items():
                condition = 'Adverse' if domain in ADVERSE_DOMAINS else 'Normal'
                f.write(f"  {domain} [{condition}]: {fwiou:.2f}%\n")
            
            f.write("\n7. WEATHER CONDITION SUMMARY\n")
            f.write("-" * 40 + "\n")
            adverse_mean = domain_df[domain_df['domain'].isin(ADVERSE_DOMAINS)]['fwIoU'].mean()
            normal_mean = domain_df[domain_df['domain'].isin(NORMAL_DOMAINS)]['fwIoU'].mean()
            f.write(f"  Normal Conditions: {normal_mean:.2f}%\n")
            f.write(f"  Adverse Conditions: {adverse_mean:.2f}%\n")
            f.write(f"  Gap: {normal_mean - adverse_mean:.2f}%\n")
    
    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("Generated files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
