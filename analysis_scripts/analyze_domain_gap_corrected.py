#!/usr/bin/env python3
"""
Corrected Domain Gap Analysis (mIoU Primary Metric)

This script creates a proper domain gap analysis by:
1. Using mIoU as primary metric (not fwIoU)
2. Excluding datasets with insufficient sample sizes (<50 images per domain)
3. Computing weighted averages based on sample size
4. Separating analysis for reliable vs unreliable data

Key Findings:
- BDD10k foggy (4 images) and MapillaryVistas foggy (27 images) are unreliable
- ACDC, IDD-AW, and Outside15k have sufficient samples for all domains
- When excluding low-sample domains, the expected pattern emerges:
  normal conditions > adverse conditions in mIoU
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
OUTPUT_DIR = Path("result_figures/domain_gap_analysis_corrected")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Minimum sample size for reliable metrics
MIN_SAMPLE_SIZE = 50

# Weather domains
WEATHER_DOMAINS = ['clear_day', 'cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy']
ADVERSE_DOMAINS = ['foggy', 'night', 'rainy', 'snowy', 'dawn_dusk']
NORMAL_DOMAINS = ['clear_day', 'cloudy']


def load_domain_results():
    """Load domain results from baseline_clear_day analysis."""
    df = pd.read_csv('result_figures/baseline_clear_day_analysis/baseline_clear_day_results.csv')
    return df[df['type'] == 'domain'].copy()


def flag_reliable_data(df, min_samples=MIN_SAMPLE_SIZE):
    """Flag data points with sufficient sample size."""
    df['reliable'] = df['num_images'] >= min_samples
    return df


def compute_weighted_average(df, metric='mIoU', weight_col='num_images'):
    """Compute weighted average of a metric."""
    valid = df[df[metric].notna() & df[weight_col].notna()]
    if len(valid) == 0:
        return np.nan
    weights = valid[weight_col].values
    values = valid[metric].values
    return np.average(values, weights=weights)


def create_analysis():
    """Create comprehensive domain gap analysis."""
    
    print("=" * 70)
    print("CORRECTED DOMAIN GAP ANALYSIS (mIoU Primary Metric)")
    print("=" * 70)
    print()
    
    # Load data
    df = load_domain_results()
    df = flag_reliable_data(df)
    
    # Report data reliability
    print("DATA RELIABILITY ASSESSMENT")
    print("-" * 50)
    print(f"Minimum sample size threshold: {MIN_SAMPLE_SIZE} images")
    print()
    
    unreliable = df[~df['reliable']]
    print("UNRELIABLE DATA POINTS (< 50 images):")
    if len(unreliable) > 0:
        for _, row in unreliable.drop_duplicates(subset=['dataset', 'domain']).iterrows():
            print(f"  {row['dataset']:20s} - {row['domain']:10s}: {int(row['num_images'])} images")
    else:
        print("  None")
    print()
    
    # Separate reliable and all data
    reliable_df = df[df['reliable']]
    
    print("=" * 70)
    print("ANALYSIS 1: ALL DATA (Including Low Sample Sizes)")
    print("=" * 70)
    
    # All data analysis
    domain_all = df.groupby('domain').agg({
        'mIoU': ['mean', 'std'],
        'fwIoU': 'mean',
        'num_images': 'sum'
    }).round(2)
    domain_all.columns = ['mIoU_mean', 'mIoU_std', 'fwIoU_mean', 'total_images']
    
    clear_day_all = domain_all.loc['clear_day', 'mIoU_mean'] if 'clear_day' in domain_all.index else 0
    domain_all['gap_from_clear_day'] = (clear_day_all - domain_all['mIoU_mean']).round(2)
    
    print("\nPer-Domain Performance (All Data):")
    print(domain_all.to_string())
    print()
    
    # Adverse vs Normal - All data
    adverse_df_all = df[df['domain'].isin(ADVERSE_DOMAINS)]
    normal_df_all = df[df['domain'].isin(NORMAL_DOMAINS)]
    
    adverse_all = compute_weighted_average(adverse_df_all, 'mIoU')
    normal_all = compute_weighted_average(normal_df_all, 'mIoU')
    gap_all = normal_all - adverse_all
    
    print(f"Normal conditions (weighted avg):  {normal_all:.2f}% mIoU")
    print(f"Adverse conditions (weighted avg): {adverse_all:.2f}% mIoU")
    print(f"Domain Gap: {gap_all:.2f}% mIoU")
    print()
    
    print("=" * 70)
    print("ANALYSIS 2: RELIABLE DATA ONLY (≥50 images per domain)")
    print("=" * 70)
    
    # Reliable data analysis
    domain_reliable = reliable_df.groupby('domain').agg({
        'mIoU': ['mean', 'std'],
        'fwIoU': 'mean',
        'num_images': 'sum'
    }).round(2)
    domain_reliable.columns = ['mIoU_mean', 'mIoU_std', 'fwIoU_mean', 'total_images']
    
    clear_day_rel = domain_reliable.loc['clear_day', 'mIoU_mean'] if 'clear_day' in domain_reliable.index else 0
    domain_reliable['gap_from_clear_day'] = (clear_day_rel - domain_reliable['mIoU_mean']).round(2)
    
    print("\nPer-Domain Performance (Reliable Data Only):")
    print(domain_reliable.to_string())
    print()
    
    # Adverse vs Normal - Reliable data
    adverse_df_rel = reliable_df[reliable_df['domain'].isin(ADVERSE_DOMAINS)]
    normal_df_rel = reliable_df[reliable_df['domain'].isin(NORMAL_DOMAINS)]
    
    adverse_rel = compute_weighted_average(adverse_df_rel, 'mIoU')
    normal_rel = compute_weighted_average(normal_df_rel, 'mIoU')
    gap_rel = normal_rel - adverse_rel
    
    print(f"Normal conditions (weighted avg):  {normal_rel:.2f}% mIoU")
    print(f"Adverse conditions (weighted avg): {adverse_rel:.2f}% mIoU")
    print(f"Domain Gap: {gap_rel:.2f}% mIoU")
    print()
    
    print("=" * 70)
    print("ANALYSIS 3: PER-DATASET DOMAIN GAP")
    print("=" * 70)
    
    for dataset in df['dataset'].unique():
        ds_df = reliable_df[reliable_df['dataset'] == dataset]
        
        if len(ds_df) == 0:
            print(f"\n{dataset.upper()}: No reliable data")
            continue
        
        print(f"\n{dataset.upper()}:")
        
        for domain in WEATHER_DOMAINS:
            domain_data = ds_df[ds_df['domain'] == domain]
            if len(domain_data) > 0:
                miou = domain_data['mIoU'].mean()
                n = int(domain_data['num_images'].iloc[0])
                print(f"  {domain:15s}: {miou:5.1f}% mIoU ({n:5d} images)")
        
        # Dataset-level gap
        ds_adverse = ds_df[ds_df['domain'].isin(ADVERSE_DOMAINS)]
        ds_normal = ds_df[ds_df['domain'].isin(NORMAL_DOMAINS)]
        
        if len(ds_adverse) > 0 and len(ds_normal) > 0:
            ds_adv_miou = compute_weighted_average(ds_adverse, 'mIoU')
            ds_norm_miou = compute_weighted_average(ds_normal, 'mIoU')
            ds_gap = ds_norm_miou - ds_adv_miou
            print(f"  => Domain Gap: {ds_gap:.2f}% mIoU (normal - adverse)")
    
    # Create visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    
    create_visualizations(df, reliable_df, domain_all, domain_reliable)
    
    # Save summary
    summary = {
        'all_data': {
            'normal_miou': normal_all,
            'adverse_miou': adverse_all,
            'domain_gap': gap_all
        },
        'reliable_data': {
            'normal_miou': normal_rel,
            'adverse_miou': adverse_rel,
            'domain_gap': gap_rel
        }
    }
    
    pd.DataFrame(summary).to_csv(OUTPUT_DIR / 'domain_gap_summary.csv')
    domain_all.to_csv(OUTPUT_DIR / 'domain_analysis_all.csv')
    domain_reliable.to_csv(OUTPUT_DIR / 'domain_analysis_reliable.csv')
    
    print(f"\nResults saved to {OUTPUT_DIR}/")
    
    return df, reliable_df, summary


def create_visualizations(df, reliable_df, domain_all, domain_reliable):
    """Create comprehensive visualizations."""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Comparison: All vs Reliable Data
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # All data
    ax = axes[0]
    domain_order = WEATHER_DOMAINS
    values_all = [domain_all.loc[d, 'mIoU_mean'] if d in domain_all.index else 0 for d in domain_order]
    colors = ['#2ecc71' if d in NORMAL_DOMAINS else '#e74c3c' for d in domain_order]
    
    bars = ax.bar(range(len(domain_order)), values_all, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(domain_order)))
    ax.set_xticklabels(domain_order, rotation=45, ha='right')
    ax.set_ylabel('mIoU (%)')
    ax.set_title('All Data (Including Low Sample Sizes)')
    ax.axhline(y=np.mean(values_all), color='gray', linestyle='--', alpha=0.7)
    
    # Add sample size warning
    for i, d in enumerate(domain_order):
        if d in domain_all.index:
            n = int(domain_all.loc[d, 'total_images'])
            if n < 200:
                ax.annotate('⚠️', (i, values_all[i] + 2), ha='center', fontsize=12)
    
    # Reliable data
    ax = axes[1]
    values_rel = [domain_reliable.loc[d, 'mIoU_mean'] if d in domain_reliable.index else 0 for d in domain_order]
    
    bars = ax.bar(range(len(domain_order)), values_rel, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(domain_order)))
    ax.set_xticklabels(domain_order, rotation=45, ha='right')
    ax.set_ylabel('mIoU (%)')
    ax.set_title('Reliable Data Only (≥50 images)')
    ax.axhline(y=np.mean([v for v in values_rel if v > 0]), color='gray', linestyle='--', alpha=0.7)
    
    fig.suptitle('Domain Gap Analysis: Impact of Sample Size Filtering', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'all_vs_reliable_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Sample Size Heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pivot_samples = df.drop_duplicates(subset=['dataset', 'domain']).pivot_table(
        values='num_images', index='dataset', columns='domain', aggfunc='first'
    )
    pivot_samples = pivot_samples.reindex(columns=WEATHER_DOMAINS)
    
    # Create annotation with warning for low samples
    annot = pivot_samples.copy()
    for col in annot.columns:
        annot[col] = annot[col].apply(lambda x: f'{int(x)}⚠️' if pd.notna(x) and x < MIN_SAMPLE_SIZE else (f'{int(x)}' if pd.notna(x) else 'N/A'))
    
    sns.heatmap(pivot_samples, annot=annot.values, fmt='', cmap='YlGnBu',
               ax=ax, cbar_kws={'label': 'Number of Images'})
    ax.set_title(f'Sample Size by Dataset × Domain\n(⚠️ = fewer than {MIN_SAMPLE_SIZE} images)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sample_size_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Per-Dataset Domain Gap (Reliable Data Only)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    dataset_gaps = []
    for dataset in df['dataset'].unique():
        ds_df = reliable_df[reliable_df['dataset'] == dataset]
        
        ds_adverse = ds_df[ds_df['domain'].isin(ADVERSE_DOMAINS)]
        ds_normal = ds_df[ds_df['domain'].isin(NORMAL_DOMAINS)]
        
        if len(ds_adverse) > 0 and len(ds_normal) > 0:
            ds_adv_miou = compute_weighted_average(ds_adverse, 'mIoU')
            ds_norm_miou = compute_weighted_average(ds_normal, 'mIoU')
            ds_gap = ds_norm_miou - ds_adv_miou
            dataset_gaps.append({'dataset': dataset, 'gap': ds_gap, 
                                'normal': ds_norm_miou, 'adverse': ds_adv_miou})
    
    if dataset_gaps:
        gap_df = pd.DataFrame(dataset_gaps)
        colors = ['#2ecc71' if g > 0 else '#e74c3c' for g in gap_df['gap']]
        
        ax.barh(gap_df['dataset'], gap_df['gap'], color=colors, alpha=0.8, edgecolor='black')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Domain Gap (mIoU: Normal - Adverse)')
        ax.set_title('Per-Dataset Domain Gap (Reliable Data Only)')
        
        for i, row in gap_df.iterrows():
            offset = 0.5 if row['gap'] >= 0 else -3
            ax.annotate(f'{row["gap"]:.1f}%', (row['gap'] + offset, i), va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'per_dataset_domain_gap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Publication Summary
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Top-left: Domain performance (reliable)
    ax1 = fig.add_subplot(gs[0, 0])
    values = [domain_reliable.loc[d, 'mIoU_mean'] if d in domain_reliable.index else 0 for d in WEATHER_DOMAINS]
    colors = ['#2ecc71' if d in NORMAL_DOMAINS else '#e74c3c' for d in WEATHER_DOMAINS]
    ax1.bar(range(len(WEATHER_DOMAINS)), values, color=colors, alpha=0.8)
    ax1.set_xticks(range(len(WEATHER_DOMAINS)))
    ax1.set_xticklabels(WEATHER_DOMAINS, rotation=45, ha='right')
    ax1.set_ylabel('mIoU (%)')
    ax1.set_title('Per-Domain Performance (mIoU)')
    
    # Top-right: Domain gap from clear_day
    ax2 = fig.add_subplot(gs[0, 1])
    clear_day_val = domain_reliable.loc['clear_day', 'mIoU_mean'] if 'clear_day' in domain_reliable.index else 0
    gaps = [clear_day_val - (domain_reliable.loc[d, 'mIoU_mean'] if d in domain_reliable.index else 0) for d in WEATHER_DOMAINS]
    colors = ['#3498db' if g <= 0 else '#e74c3c' for g in gaps]
    ax2.bar(range(len(WEATHER_DOMAINS)), gaps, color=colors, alpha=0.8)
    ax2.set_xticks(range(len(WEATHER_DOMAINS)))
    ax2.set_xticklabels(WEATHER_DOMAINS, rotation=45, ha='right')
    ax2.set_ylabel('mIoU Drop (%)')
    ax2.set_title('Domain Gap vs Clear_Day')
    ax2.axhline(y=0, color='black', linestyle='-')
    
    # Bottom-left: Normal vs Adverse
    ax3 = fig.add_subplot(gs[1, 0])
    adverse_df_rel = reliable_df[reliable_df['domain'].isin(ADVERSE_DOMAINS)]
    normal_df_rel = reliable_df[reliable_df['domain'].isin(NORMAL_DOMAINS)]
    
    normal_val = compute_weighted_average(normal_df_rel, 'mIoU')
    adverse_val = compute_weighted_average(adverse_df_rel, 'mIoU')
    
    bars = ax3.bar(['Normal\n(clear_day, cloudy)', 'Adverse\n(foggy, night, etc.)'], 
                   [normal_val, adverse_val], color=['#2ecc71', '#e74c3c'], alpha=0.8)
    ax3.set_ylabel('mIoU (%)')
    ax3.set_title('Normal vs Adverse Conditions')
    
    for bar, val in zip(bars, [normal_val, adverse_val]):
        ax3.annotate(f'{val:.1f}%', (bar.get_x() + bar.get_width()/2, val + 0.5),
                    ha='center', fontsize=12, fontweight='bold')
    
    # Add gap annotation
    gap = normal_val - adverse_val
    ax3.text(0.5, 0.95, f'Domain Gap: {gap:.1f}% mIoU',
            transform=ax3.transAxes, ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Bottom-right: Summary text
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    summary_text = f"""
    DOMAIN GAP ANALYSIS SUMMARY
    ===========================
    
    Primary Metric: mIoU (mean IoU)
    Data Filtering: ≥{MIN_SAMPLE_SIZE} images per domain
    
    BASELINE PERFORMANCE:
    • Normal conditions:  {normal_val:.1f}% mIoU
    • Adverse conditions: {adverse_val:.1f}% mIoU
    • Domain Gap: {gap:.1f}% mIoU drop
    
    KEY INSIGHT:
    Models trained on clear_day images show
    {gap:.1f}% mIoU degradation on adverse
    weather conditions.
    
    This establishes the baseline for evaluating
    data augmentation strategies.
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('Corrected Domain Gap Analysis (Clear_Day Baseline, mIoU Metric)',
                fontsize=14, fontweight='bold')
    
    plt.savefig(OUTPUT_DIR / 'publication_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    df, reliable_df, summary = create_analysis()
