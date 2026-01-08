#!/usr/bin/env python3
"""
Comprehensive Baseline Analysis Script v2

Analyzes baseline performance using CLEAR_DAY trained models as the reference.
This provides a better baseline for comparing how augmentation strategies help
with adverse weather conditions.

Reference Baseline: Models trained ONLY on clear_day images
- This represents the "naive" baseline without any weather diversity
- Gains show how much strategies improve over this limited training

Primary Metric: fwIoU (frequency-weighted IoU)
Secondary Metrics: mIoU, PA (Pixel Accuracy) where relevant

Usage:
    mamba run -n prove python analyze_baseline_v2.py

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

# Datasets with domain annotations
DOMAIN_DATASETS = ['acdc', 'bdd10k', 'idd-aw']

# Standard models (without domain suffix)
MODELS = ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5']


def load_baseline_results(downstream_csv: str) -> tuple:
    """Load both full baseline and clear_day baseline results."""
    
    df = pd.read_csv(downstream_csv)
    
    # Filter for baseline strategy
    baseline_df = df[df['strategy'] == 'baseline'].copy()
    
    # Separate full baseline (standard models) and clear_day baseline
    full_baseline = baseline_df[baseline_df['model'].isin(MODELS)].copy()
    
    # Clear_day baseline: models with _clear_day suffix
    clear_day_models = [f"{m}_clear_day" for m in MODELS]
    clear_day_baseline = baseline_df[baseline_df['model'].isin(clear_day_models)].copy()
    
    # Normalize model names in clear_day_baseline for comparison
    clear_day_baseline['model_base'] = clear_day_baseline['model'].str.replace('_clear_day', '')
    
    return full_baseline, clear_day_baseline


def load_domain_results(weights_root: str, use_clear_day_baseline: bool = True) -> pd.DataFrame:
    """Load per-domain results from test_report.txt files.
    
    Args:
        weights_root: Path to WEIGHTS directory
        use_clear_day_baseline: If True, load from *_clear_day model directories
    """
    
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
            
            # Filter based on baseline type
            if use_clear_day_baseline:
                # Only use clear_day trained models
                if not model.endswith('_clear_day'):
                    continue
                model_base = model.replace('_clear_day', '')
            else:
                # Only use full baseline models (no domain suffix)
                if any(domain in model for domain in WEATHER_DOMAINS):
                    continue
                model_base = model
            
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
                            'model': model_base,
                            'model_full': model,
                            'domain': domain,
                            'baseline_type': 'clear_day' if use_clear_day_baseline else 'full',
                            'fwIoU': metrics.get('fwIoU', 0),
                            'mIoU': metrics.get('mIoU', 0),
                            'aAcc': metrics.get('aAcc', 0),
                            'num_images': metrics.get('num_images', 0)
                        })
                    break
    
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
                if f'--- {domain} ---' in line.lower() or f'--- {domain}' in line:
                    current_domain = domain
                    domain_metrics[current_domain] = {}
                    break
            
            if current_domain and ':' in line:
                parts = line.split(':')
                if len(parts) >= 2:
                    key = parts[0].strip()
                    try:
                        value = float(parts[1].strip().split()[0])
                        domain_metrics[current_domain][key] = value
                    except:
                        pass
    
    return domain_metrics


def create_baseline_reference(full_baseline: pd.DataFrame, 
                              clear_day_baseline: pd.DataFrame,
                              clear_day_domain_df: pd.DataFrame,
                              output_path: Path):
    """Create baseline reference JSON for gains/losses computation.
    
    Uses clear_day trained models as the reference baseline.
    """
    
    reference = {}
    
    # Create reference from clear_day baseline overall results
    for _, row in clear_day_baseline.iterrows():
        key = f"{row['dataset']}_{row['model_base']}"
        reference[key] = {
            'dataset': row['dataset'],
            'model': row['model_base'],
            'baseline_type': 'clear_day',
            'fwIoU': row['fwIoU'],
            'mIoU': row['mIoU'],
            'aAcc': row.get('aAcc', 0)
        }
    
    # Also add per-domain reference if available
    for _, row in clear_day_domain_df.iterrows():
        key = f"{row['dataset']}_{row['model']}_{row['domain']}"
        reference[key] = {
            'dataset': row['dataset'],
            'model': row['model'],
            'domain': row['domain'],
            'baseline_type': 'clear_day',
            'fwIoU': row['fwIoU'],
            'mIoU': row['mIoU'],
            'aAcc': row.get('aAcc', 0),
            'num_images': row.get('num_images', 0)
        }
    
    with open(output_path, 'w') as f:
        json.dump(reference, f, indent=2)
    
    print(f"Saved baseline reference: {output_path}")
    return reference


def create_comparison_visualization(full_domain_df: pd.DataFrame, 
                                   clear_day_domain_df: pd.DataFrame,
                                   output_dir: Path):
    """Create visualization comparing full vs clear_day baseline performance."""
    
    # Merge dataframes for comparison
    merged = pd.merge(
        full_domain_df[['dataset', 'model', 'domain', 'fwIoU', 'mIoU', 'num_images']],
        clear_day_domain_df[['dataset', 'model', 'domain', 'fwIoU', 'mIoU']],
        on=['dataset', 'model', 'domain'],
        suffixes=('_full', '_clearday')
    )
    
    if len(merged) == 0:
        print("No matching data for comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. Comparison scatter plot
    ax1 = axes[0, 0]
    colors = merged['domain'].map({
        'foggy': 'gray', 'rainy': 'blue', 'snowy': 'cyan', 'night': 'purple',
        'clear_day': 'green', 'cloudy': 'yellow', 'dawn_dusk': 'orange'
    })
    ax1.scatter(merged['fwIoU_clearday'], merged['fwIoU_full'], c=colors, alpha=0.7, s=100)
    ax1.plot([50, 100], [50, 100], 'k--', alpha=0.5)
    ax1.set_xlabel('fwIoU (Clear_day Baseline)')
    ax1.set_ylabel('fwIoU (Full Baseline)')
    ax1.set_title('Full vs Clear_day Baseline Comparison', fontweight='bold')
    
    # Add legend
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=d) 
                      for d, c in [('foggy', 'gray'), ('rainy', 'blue'), ('snowy', 'cyan'), 
                                   ('night', 'purple'), ('clear_day', 'green'), ('cloudy', 'yellow'), 
                                   ('dawn_dusk', 'orange')]]
    ax1.legend(handles=legend_handles, loc='lower right')
    
    # 2. Gap analysis by domain
    ax2 = axes[0, 1]
    merged['gap'] = merged['fwIoU_full'] - merged['fwIoU_clearday']
    domain_gaps = merged.groupby('domain')['gap'].mean().sort_values(ascending=False)
    
    colors = ['#2ecc71' if g > 0 else '#e74c3c' for g in domain_gaps.values]
    ax2.barh(range(len(domain_gaps)), domain_gaps.values, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(domain_gaps)))
    ax2.set_yticklabels(domain_gaps.index)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('fwIoU Gap (Full - Clear_day)')
    ax2.set_title('Training Diversity Benefit by Domain', fontweight='bold')
    
    # 3. Clear_day baseline per-domain performance
    ax3 = axes[1, 0]
    domain_perf = clear_day_domain_df.groupby('domain')['fwIoU'].mean().sort_values(ascending=True)
    
    # Color by adverse vs normal
    colors = ['#e74c3c' if d in ADVERSE_DOMAINS else '#2ecc71' for d in domain_perf.index]
    ax3.barh(range(len(domain_perf)), domain_perf.values, color=colors, alpha=0.7)
    ax3.set_yticks(range(len(domain_perf)))
    ax3.set_yticklabels(domain_perf.index)
    ax3.set_xlabel('Mean fwIoU (%)')
    ax3.set_title('Clear_day Baseline Performance by Domain', fontweight='bold')
    ax3.legend([plt.Line2D([0], [0], color='#e74c3c', linewidth=4), 
                plt.Line2D([0], [0], color='#2ecc71', linewidth=4)], 
               ['Adverse', 'Normal'], loc='lower right')
    
    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate summary stats
    adverse_clearday = clear_day_domain_df[clear_day_domain_df['domain'].isin(ADVERSE_DOMAINS)]['fwIoU'].mean()
    normal_clearday = clear_day_domain_df[clear_day_domain_df['domain'].isin(NORMAL_DOMAINS)]['fwIoU'].mean()
    adverse_full = full_domain_df[full_domain_df['domain'].isin(ADVERSE_DOMAINS)]['fwIoU'].mean()
    normal_full = full_domain_df[full_domain_df['domain'].isin(NORMAL_DOMAINS)]['fwIoU'].mean()
    
    stats_text = "BASELINE COMPARISON SUMMARY\n"
    stats_text += "=" * 40 + "\n\n"
    stats_text += "CLEAR_DAY BASELINE (Reference):\n"
    stats_text += f"  Adverse Conditions: {adverse_clearday:.2f}%\n"
    stats_text += f"  Normal Conditions: {normal_clearday:.2f}%\n"
    stats_text += f"  Gap (Adverse-Normal): {adverse_clearday - normal_clearday:.2f}%\n\n"
    stats_text += "FULL BASELINE (Comparison):\n"
    stats_text += f"  Adverse Conditions: {adverse_full:.2f}%\n"
    stats_text += f"  Normal Conditions: {normal_full:.2f}%\n"
    stats_text += f"  Gap (Adverse-Normal): {adverse_full - normal_full:.2f}%\n\n"
    stats_text += "TRAINING DIVERSITY BENEFIT:\n"
    stats_text += f"  Adverse: +{adverse_full - adverse_clearday:.2f}%\n"
    stats_text += f"  Normal: +{normal_full - normal_clearday:.2f}%\n"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Baseline Analysis: Clear_day vs Full Baseline', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "baseline_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'baseline_comparison.png'}")


def create_clearday_baseline_summary(clear_day_baseline: pd.DataFrame,
                                     clear_day_domain_df: pd.DataFrame,
                                     output_dir: Path):
    """Create summary visualizations for clear_day baseline."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. Overall performance by dataset
    ax1 = axes[0, 0]
    dataset_perf = clear_day_baseline.groupby('dataset')['fwIoU'].mean().sort_values(ascending=False)
    ax1.bar(range(len(dataset_perf)), dataset_perf.values, color='steelblue', alpha=0.7)
    ax1.set_xticks(range(len(dataset_perf)))
    ax1.set_xticklabels(dataset_perf.index, rotation=45, ha='right')
    ax1.set_ylabel('fwIoU (%)')
    ax1.set_title('Clear_day Baseline: Per-Dataset Performance', fontweight='bold')
    
    # Add values on bars
    for i, v in enumerate(dataset_perf.values):
        ax1.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    # 2. Per-domain performance heatmap
    ax2 = axes[0, 1]
    if len(clear_day_domain_df) > 0:
        pivot = clear_day_domain_df.pivot_table(values='fwIoU', index='domain', columns='dataset', aggfunc='mean')
        if not pivot.empty:
            sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax2, 
                       cbar_kws={'label': 'fwIoU (%)'})
            ax2.set_title('Clear_day Baseline: Domain × Dataset', fontweight='bold')
    
    # 3. Model comparison
    ax3 = axes[1, 0]
    model_perf = clear_day_baseline.groupby('model_base')['fwIoU'].mean().sort_values(ascending=False)
    ax3.bar(range(len(model_perf)), model_perf.values, color='coral', alpha=0.7)
    ax3.set_xticks(range(len(model_perf)))
    ax3.set_xticklabels(model_perf.index, rotation=45, ha='right')
    ax3.set_ylabel('fwIoU (%)')
    ax3.set_title('Clear_day Baseline: Per-Model Performance', fontweight='bold')
    
    for i, v in enumerate(model_perf.values):
        ax3.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    # 4. Domain vulnerability
    ax4 = axes[1, 1]
    if len(clear_day_domain_df) > 0:
        domain_perf = clear_day_domain_df.groupby('domain')['fwIoU'].mean().sort_values(ascending=True)
        
        colors = ['#e74c3c' if d in ADVERSE_DOMAINS else '#2ecc71' for d in domain_perf.index]
        ax4.barh(range(len(domain_perf)), domain_perf.values, color=colors, alpha=0.7)
        ax4.set_yticks(range(len(domain_perf)))
        ax4.set_yticklabels(domain_perf.index)
        ax4.set_xlabel('fwIoU (%)')
        ax4.set_title('Domain Vulnerability (Clear_day Baseline)', fontweight='bold')
    
    plt.suptitle('Clear_day Baseline Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "clearday_baseline_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'clearday_baseline_summary.png'}")


def main():
    """Main function."""
    
    print("=" * 60)
    print("Baseline Analysis v2: Using Clear_day as Reference")
    print("=" * 60)
    
    # Paths
    downstream_csv = "/home/mima2416/repositories/PROVE/downstream_results.csv"
    weights_root = "/scratch/aaa_exchange/AWARE/WEIGHTS"
    
    # Load overall baseline results
    print("\nLoading baseline results...")
    full_baseline, clear_day_baseline = load_baseline_results(downstream_csv)
    print(f"Full baseline: {len(full_baseline)} results")
    print(f"Clear_day baseline: {len(clear_day_baseline)} results")
    
    # Load per-domain results for both baseline types
    print("\nLoading per-domain results...")
    full_domain_df = load_domain_results(weights_root, use_clear_day_baseline=False)
    clear_day_domain_df = load_domain_results(weights_root, use_clear_day_baseline=True)
    print(f"Full baseline domains: {len(full_domain_df)} results")
    print(f"Clear_day baseline domains: {len(clear_day_domain_df)} results")
    
    # Create baseline reference JSON (using clear_day as reference)
    print("\nCreating baseline reference...")
    reference = create_baseline_reference(
        full_baseline, clear_day_baseline, clear_day_domain_df,
        OUTPUT_DIR / "baseline_reference.json"
    )
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    if len(full_domain_df) > 0 and len(clear_day_domain_df) > 0:
        create_comparison_visualization(full_domain_df, clear_day_domain_df, OUTPUT_DIR)
    
    if len(clear_day_baseline) > 0:
        create_clearday_baseline_summary(clear_day_baseline, clear_day_domain_df, OUTPUT_DIR)
    
    # Generate report
    print("\nGenerating report...")
    report_path = OUTPUT_DIR / "baseline_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("BASELINE ANALYSIS REPORT v2\n")
        f.write("=" * 60 + "\n\n")
        f.write("REFERENCE: Clear_day Trained Models\n")
        f.write("(Models trained ONLY on clear_day images)\n\n")
        
        f.write("1. CLEAR_DAY BASELINE OVERALL RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total configs: {len(clear_day_baseline)}\n")
        f.write(f"Mean fwIoU: {clear_day_baseline['fwIoU'].mean():.2f}%\n")
        f.write(f"Mean mIoU: {clear_day_baseline['mIoU'].mean():.2f}%\n\n")
        
        f.write("Per-Dataset:\n")
        for dataset in sorted(clear_day_baseline['dataset'].unique()):
            subset = clear_day_baseline[clear_day_baseline['dataset'] == dataset]
            f.write(f"  {dataset}: fwIoU={subset['fwIoU'].mean():.2f}%, mIoU={subset['mIoU'].mean():.2f}%\n")
        
        f.write("\n2. CLEAR_DAY BASELINE PER-DOMAIN RESULTS\n")
        f.write("-" * 40 + "\n")
        
        if len(clear_day_domain_df) > 0:
            for domain in WEATHER_DOMAINS:
                domain_data = clear_day_domain_df[clear_day_domain_df['domain'] == domain]
                if len(domain_data) > 0:
                    condition = "[Adverse]" if domain in ADVERSE_DOMAINS else "[Normal]"
                    num_imgs = domain_data['num_images'].sum()
                    f.write(f"  {domain} {condition}: fwIoU={domain_data['fwIoU'].mean():.2f}%, ")
                    f.write(f"mIoU={domain_data['mIoU'].mean():.2f}% (n={num_imgs})\n")
        
        f.write("\n3. TRAINING DIVERSITY BENEFIT\n")
        f.write("-" * 40 + "\n")
        
        if len(full_domain_df) > 0 and len(clear_day_domain_df) > 0:
            adverse_clearday = clear_day_domain_df[clear_day_domain_df['domain'].isin(ADVERSE_DOMAINS)]['fwIoU'].mean()
            adverse_full = full_domain_df[full_domain_df['domain'].isin(ADVERSE_DOMAINS)]['fwIoU'].mean()
            f.write(f"Adverse conditions: Clear_day={adverse_clearday:.2f}%, Full={adverse_full:.2f}%, Δ={adverse_full-adverse_clearday:.2f}%\n")
            
            normal_clearday = clear_day_domain_df[clear_day_domain_df['domain'].isin(NORMAL_DOMAINS)]['fwIoU'].mean()
            normal_full = full_domain_df[full_domain_df['domain'].isin(NORMAL_DOMAINS)]['fwIoU'].mean()
            f.write(f"Normal conditions: Clear_day={normal_clearday:.2f}%, Full={normal_full:.2f}%, Δ={normal_full-normal_clearday:.2f}%\n")
        
        f.write("\n4. INTERPRETATION\n")
        f.write("-" * 40 + "\n")
        f.write("The clear_day baseline represents models trained WITHOUT any weather\n")
        f.write("diversity. When comparing augmentation strategies, gains relative\n")
        f.write("to this baseline show how much they help adapt to adverse conditions.\n")
    
    print(f"\nReport saved: {report_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nClear_day Baseline (Reference):")
    print(f"  Overall fwIoU: {clear_day_baseline['fwIoU'].mean():.2f}%")
    print(f"  Overall mIoU: {clear_day_baseline['mIoU'].mean():.2f}%")
    
    if len(clear_day_domain_df) > 0:
        adverse_fwIoU = clear_day_domain_df[clear_day_domain_df['domain'].isin(ADVERSE_DOMAINS)]['fwIoU'].mean()
        normal_fwIoU = clear_day_domain_df[clear_day_domain_df['domain'].isin(NORMAL_DOMAINS)]['fwIoU'].mean()
        print(f"\n  Adverse conditions: {adverse_fwIoU:.2f}% fwIoU")
        print(f"  Normal conditions: {normal_fwIoU:.2f}% fwIoU")
    
    print(f"\nOutput saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
