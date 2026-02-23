#!/usr/bin/env python3
"""
Unified Domain Gap Analysis

This script provides a comprehensive analysis combining:
1. Baseline clear_day performance
2. Strategy-augmented performance  
3. Domain gap reduction measurement
4. Frequency-weighted averaging for aggregates

Supports both Stage 1 and Stage 2:
- Stage 1 (WEIGHTS/): Models trained only on clear_day
- Stage 2 (WEIGHTS_STAGE_2/): Models trained on all domains

Primary Metric: mIoU (mean Intersection over Union)
- Recommended for domain robustness analysis
- Equal weight to all classes
- Not biased by class frequency distribution

Data Quality:
- Excludes domains with <50 test images (configurable)
- Uses frequency-weighted averaging for aggregates
- Reports individual values only for reliable data

Usage:
    python analyze_unified_domain_gap.py              # Stage 1 (default)
    python analyze_unified_domain_gap.py --stage 2   # Stage 2
    python analyze_unified_domain_gap.py --weights-root /path/to/weights

Output:
    result_figures/unified_domain_gap/
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
OUTPUT_DIR = Path("result_figures/unified_domain_gap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Stage-specific weights directories
WEIGHTS_ROOT_STAGE1 = "${AWARE_DATA_ROOT}/WEIGHTS"
WEIGHTS_ROOT_STAGE2 = "${AWARE_DATA_ROOT}/WEIGHTS_STAGE_2"

# Minimum sample size for reliable metrics
MIN_SAMPLE_SIZE = 50

# Weather domains
WEATHER_DOMAINS = ['clear_day', 'cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy']
ADVERSE_DOMAINS = ['foggy', 'night', 'rainy', 'snowy', 'dawn_dusk']
NORMAL_DOMAINS = ['clear_day', 'cloudy']

# Datasets and Models
DATASETS = ['acdc', 'bdd10k', 'iddaw', 'mapillaryvistas', 'outside15k']
MODELS = ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5']


def parse_test_report(filepath):
    """Parse test_report.txt to extract per-domain metrics."""
    domain_metrics = {}
    current_domain = None
    in_per_domain = False
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception:
        return {}
    
    for line in content.split('\n'):
        line = line.strip()
        
        if 'PER-DOMAIN METRICS' in line:
            in_per_domain = True
            continue
        
        if 'PER-CLASS METRICS' in line:
            in_per_domain = False
            continue
        
        if in_per_domain:
            for domain in WEATHER_DOMAINS:
                if f'--- {domain} ---' in line:
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


def load_strategy_domain_results(weights_root, strategy='baseline'):
    """Load per-domain results for a specific strategy."""
    results = []
    
    strategy_dir = Path(weights_root) / strategy
    if not strategy_dir.exists():
        return pd.DataFrame()
    
    for dataset_dir in strategy_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        
        for model_dir in dataset_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            
            # Skip domain-specific models (e.g., deeplabv3plus_r50_clear_day)
            if any(d in model for d in WEATHER_DOMAINS):
                continue
            
            # Handle models with ratio suffix (gen_* strategies use _ratio0p50)
            # For domain gap analysis, we want to include these but normalize the model name
            if '_ratio' in model:
                # Only include _ratio0p50 (standard generative strategy ratio)
                # Skip other ratios (e.g., _ratio0p25) as they're from ablation studies
                if '_ratio0p50' not in model:
                    continue
                # Normalize model name by removing the ratio suffix
                model_normalized = model.replace('_ratio0p50', '')
            else:
                model_normalized = model
            
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
                            'strategy': strategy,
                            'dataset': dataset,
                            'model': model_normalized,  # Use normalized name (without _ratio suffix)
                            'domain': domain,
                            'mIoU': metrics.get('mIoU', metrics.get('miou')),
                            'fwIoU': metrics.get('fwIoU', metrics.get('fwiou')),
                            'aAcc': metrics.get('aAcc'),
                            'mAcc': metrics.get('mAcc'),
                            'num_images': metrics.get('num_images', 0)
                        })
                    break
    
    return pd.DataFrame(results)


def load_clear_day_baseline(weights_root):
    """Load clear_day trained baseline results.
    
    NEW: Uses dataset suffix (_cd) instead of model suffix (_clear_day).
    """
    CLEAR_DAY_DATASET_SUFFIX = '_cd'
    results = []
    baseline_dir = Path(weights_root) / "baseline"
    
    if not baseline_dir.exists():
        return pd.DataFrame()
    
    for dataset_dir in baseline_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        
        # Only include _cd suffix datasets for clear_day trained models
        if not dataset.endswith(CLEAR_DAY_DATASET_SUFFIX):
            continue
        
        # Get base dataset name (remove _cd suffix)
        base_dataset = dataset[:-len(CLEAR_DAY_DATASET_SUFFIX)]
        
        for model_dir in dataset_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            
            detailed_dir = model_dir / "test_results_detailed"
            if not detailed_dir.exists():
                continue
            
            for subdir in sorted(detailed_dir.iterdir(), reverse=True):
                if not subdir.is_dir():
                    continue
                
                report_file = subdir / "test_report.txt"
                if report_file.exists():
                    domain_metrics = parse_test_report(report_file)
                    
                    for domain, metrics in domain_metrics.items():
                        results.append({
                            'strategy': 'baseline_clear_day',
                            'dataset': base_dataset,
                            'model': model,
                            'domain': domain,
                            'mIoU': metrics.get('mIoU'),
                            'fwIoU': metrics.get('fwIoU'),
                            'aAcc': metrics.get('aAcc'),
                            'mAcc': metrics.get('mAcc'),
                            'num_images': metrics.get('num_images', 0)
                        })
                    break
    
    return pd.DataFrame(results)


def filter_reliable_data(df, min_samples=MIN_SAMPLE_SIZE):
    """Filter to reliable data points with sufficient sample size."""
    if len(df) == 0:
        return df
    return df[df['num_images'] >= min_samples].copy()


def compute_weighted_average(df, metric='mIoU', weight_col='num_images'):
    """Compute frequency-weighted average."""
    valid = df[df[metric].notna() & df[weight_col].notna() & (df[weight_col] > 0)]
    if len(valid) == 0:
        return np.nan
    weights = valid[weight_col].values.astype(float)
    values = valid[metric].values.astype(float)
    return np.average(values, weights=weights)


def compute_domain_gap(df, metric='mIoU'):
    """Compute domain gap (normal - adverse) using weighted average."""
    adverse = df[df['domain'].isin(ADVERSE_DOMAINS)]
    normal = df[df['domain'].isin(NORMAL_DOMAINS)]
    
    adverse_avg = compute_weighted_average(adverse, metric)
    normal_avg = compute_weighted_average(normal, metric)
    
    if pd.isna(adverse_avg) or pd.isna(normal_avg):
        return np.nan
    
    return normal_avg - adverse_avg


def analyze_all_strategies(weights_root: str = WEIGHTS_ROOT_STAGE1):
    """Analyze domain gap for all available strategies.
    
    Args:
        weights_root: Root directory for weights (WEIGHTS or WEIGHTS_STAGE_2)
    """
    
    # Load clear_day baseline
    print("Loading baseline results...")
    baseline_clear_day = load_clear_day_baseline(weights_root)
    baseline_clear_day_reliable = filter_reliable_data(baseline_clear_day)
    
    if len(baseline_clear_day) > 0:
        print(f"  Loaded {len(baseline_clear_day)} raw results, {len(baseline_clear_day_reliable)} reliable")
    
    # Find all strategy directories
    weights_path = Path(weights_root)
    strategies = []
    for d in weights_path.iterdir():
        if d.is_dir() and not d.name.startswith('.'):
            strategies.append(d.name)
    
    print(f"\nFound {len(strategies)} strategy directories")
    
    # Load results for each strategy
    all_results = []
    strategy_summaries = []
    
    for strategy in sorted(strategies):
        print(f"  Processing: {strategy}...")
        
        df = load_strategy_domain_results(weights_root, strategy)
        if len(df) == 0:
            continue
        
        df_reliable = filter_reliable_data(df)
        
        if len(df_reliable) == 0:
            continue
        
        # Compute metrics
        overall_miou = compute_weighted_average(df_reliable, 'mIoU')
        normal_miou = compute_weighted_average(df_reliable[df_reliable['domain'].isin(NORMAL_DOMAINS)], 'mIoU')
        adverse_miou = compute_weighted_average(df_reliable[df_reliable['domain'].isin(ADVERSE_DOMAINS)], 'mIoU')
        domain_gap = compute_domain_gap(df_reliable, 'mIoU')
        
        strategy_summaries.append({
            'strategy': strategy,
            'overall_mIoU': overall_miou,
            'normal_mIoU': normal_miou,
            'adverse_mIoU': adverse_miou,
            'domain_gap': domain_gap,
            'n_reliable_results': len(df_reliable),
            'n_total_results': len(df)
        })
        
        # Add individual results
        for _, row in df_reliable.iterrows():
            all_results.append(row.to_dict())
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(strategy_summaries)
    all_df = pd.DataFrame(all_results)
    
    # Add clear_day baseline
    if len(baseline_clear_day_reliable) > 0:
        baseline_gap = compute_domain_gap(baseline_clear_day_reliable, 'mIoU')
        baseline_summary = {
            'strategy': 'baseline_clear_day',
            'overall_mIoU': compute_weighted_average(baseline_clear_day_reliable, 'mIoU'),
            'normal_mIoU': compute_weighted_average(baseline_clear_day_reliable[baseline_clear_day_reliable['domain'].isin(NORMAL_DOMAINS)], 'mIoU'),
            'adverse_mIoU': compute_weighted_average(baseline_clear_day_reliable[baseline_clear_day_reliable['domain'].isin(ADVERSE_DOMAINS)], 'mIoU'),
            'domain_gap': baseline_gap,
            'n_reliable_results': len(baseline_clear_day_reliable),
            'n_total_results': len(baseline_clear_day)
        }
        summary_df = pd.concat([pd.DataFrame([baseline_summary]), summary_df], ignore_index=True)
    
    # Sort by overall mIoU
    summary_df = summary_df.sort_values('overall_mIoU', ascending=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("STRATEGY PERFORMANCE SUMMARY (sorted by mIoU)")
    print("=" * 70)
    print(f"{'Strategy':<30} {'mIoU':>8} {'Normal':>8} {'Adverse':>8} {'Gap':>8} {'N':>6}")
    print("-" * 70)
    
    for _, row in summary_df.iterrows():
        print(f"{row['strategy']:<30} {row['overall_mIoU']:>8.2f} {row['normal_mIoU']:>8.2f} {row['adverse_mIoU']:>8.2f} {row['domain_gap']:>+8.2f} {row['n_reliable_results']:>6}")
    
    # Compute domain gap reduction relative to baseline
    if 'baseline_clear_day' in summary_df['strategy'].values:
        baseline_gap = summary_df[summary_df['strategy'] == 'baseline_clear_day']['domain_gap'].values[0]
        summary_df['gap_reduction'] = baseline_gap - summary_df['domain_gap']
        summary_df['gap_reduction_pct'] = (summary_df['gap_reduction'] / baseline_gap) * 100
    
    # Save results
    summary_df.to_csv(OUTPUT_DIR / 'strategy_summary.csv', index=False)
    all_df.to_csv(OUTPUT_DIR / 'all_domain_results.csv', index=False)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(summary_df, all_df, baseline_clear_day_reliable)
    
    print(f"\nResults saved to {OUTPUT_DIR}/")
    
    return summary_df, all_df


def create_visualizations(summary_df, all_df, baseline_df):
    """Create comprehensive visualizations."""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Strategy comparison (top 15 by mIoU)
    fig, ax = plt.subplots(figsize=(14, 8))
    
    top_strategies = summary_df.head(15).copy()
    
    colors = ['#2ecc71' if s == 'baseline_clear_day' else '#3498db' 
              for s in top_strategies['strategy']]
    
    y_pos = range(len(top_strategies))
    ax.barh(y_pos, top_strategies['overall_mIoU'], color=colors, alpha=0.8, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_strategies['strategy'])
    ax.set_xlabel('mIoU (%)')
    ax.set_title('Strategy Performance Ranking (Top 15, mIoU)')
    ax.invert_yaxis()
    
    # Add value labels
    for i, v in enumerate(top_strategies['overall_mIoU']):
        ax.annotate(f'{v:.1f}', (v + 0.5, i), va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'strategy_ranking.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Domain gap comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Sort by domain gap
    gap_sorted = summary_df.sort_values('domain_gap', ascending=True).head(15)
    
    colors = ['#e74c3c' if g > 0 else '#2ecc71' for g in gap_sorted['domain_gap']]
    
    y_pos = range(len(gap_sorted))
    ax.barh(y_pos, gap_sorted['domain_gap'], color=colors, alpha=0.8, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(gap_sorted['strategy'])
    ax.set_xlabel('Domain Gap (mIoU: Normal - Adverse)')
    ax.set_title('Domain Gap by Strategy\n(Negative = Adverse > Normal, indicating robustness)')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'domain_gap_ranking.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Normal vs Adverse scatter
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.scatter(summary_df['normal_mIoU'], summary_df['adverse_mIoU'], 
              s=100, alpha=0.7, c='#3498db', edgecolor='black')
    
    # Add diagonal line (equal performance)
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), 
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Equal Performance')
    
    # Highlight baseline
    if 'baseline_clear_day' in summary_df['strategy'].values:
        baseline_row = summary_df[summary_df['strategy'] == 'baseline_clear_day'].iloc[0]
        ax.scatter([baseline_row['normal_mIoU']], [baseline_row['adverse_mIoU']],
                  s=200, c='#e74c3c', marker='*', edgecolor='black', label='Baseline (clear_day)')
    
    ax.set_xlabel('Normal Conditions mIoU (%)')
    ax.set_ylabel('Adverse Conditions mIoU (%)')
    ax.set_title('Strategy Performance: Normal vs Adverse Conditions')
    ax.legend()
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'normal_vs_adverse_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Publication summary
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Top-left: Strategy ranking
    ax1 = fig.add_subplot(gs[0, 0])
    top10 = summary_df.head(10)
    colors = ['#2ecc71' if s == 'baseline_clear_day' else '#3498db' for s in top10['strategy']]
    ax1.barh(range(len(top10)), top10['overall_mIoU'], color=colors, alpha=0.8)
    ax1.set_yticks(range(len(top10)))
    ax1.set_yticklabels(top10['strategy'])
    ax1.set_xlabel('mIoU (%)')
    ax1.set_title('Top 10 Strategies by mIoU')
    ax1.invert_yaxis()
    
    # Top-right: Domain gap
    ax2 = fig.add_subplot(gs[0, 1])
    top10_gap = summary_df.sort_values('domain_gap').head(10)
    colors = ['#e74c3c' if g > 0 else '#2ecc71' for g in top10_gap['domain_gap']]
    ax2.barh(range(len(top10_gap)), top10_gap['domain_gap'], color=colors, alpha=0.8)
    ax2.set_yticks(range(len(top10_gap)))
    ax2.set_yticklabels(top10_gap['strategy'])
    ax2.set_xlabel('Domain Gap (Normal - Adverse)')
    ax2.set_title('Smallest Domain Gaps')
    ax2.axvline(x=0, color='black', linestyle='-')
    ax2.invert_yaxis()
    
    # Bottom-left: Domain performance heatmap (for top strategies)
    ax3 = fig.add_subplot(gs[1, 0])
    if len(all_df) > 0:
        top5_strategies = summary_df.head(5)['strategy'].tolist()
        heatmap_data = all_df[all_df['strategy'].isin(top5_strategies)].pivot_table(
            values='mIoU', index='strategy', columns='domain', aggfunc='mean'
        )
        heatmap_data = heatmap_data.reindex(columns=[d for d in WEATHER_DOMAINS if d in heatmap_data.columns])
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn',
                   ax=ax3, vmin=20, vmax=80, cbar_kws={'label': 'mIoU (%)'})
        ax3.set_title('Top 5 Strategies: Per-Domain Performance')
    
    # Bottom-right: Summary text
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Get baseline stats
    baseline_row = summary_df[summary_df['strategy'] == 'baseline_clear_day']
    if len(baseline_row) > 0:
        baseline_gap = baseline_row['domain_gap'].values[0]
        baseline_miou = baseline_row['overall_mIoU'].values[0]
    else:
        baseline_gap = summary_df['domain_gap'].mean()
        baseline_miou = summary_df['overall_mIoU'].mean()
    
    best_strategy = summary_df.iloc[0]
    best_gap = summary_df.sort_values('domain_gap').iloc[0]
    
    summary_text = f"""
    UNIFIED DOMAIN GAP ANALYSIS SUMMARY
    ====================================
    
    Total Strategies Analyzed: {len(summary_df)}
    
    BASELINE (Clear_Day Trained):
    • Overall mIoU: {baseline_miou:.1f}%
    • Domain Gap: {baseline_gap:.1f}% (Normal - Adverse)
    
    BEST OVERALL (by mIoU):
    • Strategy: {best_strategy['strategy']}
    • mIoU: {best_strategy['overall_mIoU']:.1f}%
    • Domain Gap: {best_strategy['domain_gap']:.1f}%
    
    SMALLEST DOMAIN GAP:
    • Strategy: {best_gap['strategy']}
    • Domain Gap: {best_gap['domain_gap']:.1f}%
    • mIoU: {best_gap['overall_mIoU']:.1f}%
    
    METRIC: mIoU (mean IoU)
    DATA: Filtered to ≥{MIN_SAMPLE_SIZE} images/domain
    AGGREGATION: Frequency-weighted average
    """
    
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('Unified Domain Gap Analysis', fontsize=14, fontweight='bold')
    
    plt.savefig(OUTPUT_DIR / 'publication_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {OUTPUT_DIR}/")


def main():
    parser = argparse.ArgumentParser(description='Unified Domain Gap Analysis')
    parser.add_argument('--stage', type=int, choices=[1, 2], default=1,
                        help='Stage to analyze (1=clear_day training, 2=all domains training)')
    parser.add_argument('--weights-root', type=str, default=None,
                        help='Override weights root directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory')
    args = parser.parse_args()
    
    # Determine weights root
    if args.weights_root:
        weights_root = args.weights_root
    elif args.stage == 1:
        weights_root = WEIGHTS_ROOT_STAGE1
    else:
        weights_root = WEIGHTS_ROOT_STAGE2
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = OUTPUT_DIR / f"stage{args.stage}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stage_name = f"Stage {args.stage}"
    stage_desc = "Clear Day Training" if args.stage == 1 else "All Domains Training"
    
    print("=" * 70)
    print(f"UNIFIED DOMAIN GAP ANALYSIS - {stage_name}")
    print(f"Training: {stage_desc}")
    print(f"Weights: {weights_root}")
    print("=" * 70)
    
    summary_df, all_df = analyze_all_strategies(weights_root)
    
    # Generate text report
    report = []
    report.append("=" * 80)
    report.append(f"UNIFIED DOMAIN GAP ANALYSIS REPORT - {stage_name}")
    report.append(f"Training: {stage_desc}")
    report.append(f"Weights: {weights_root}")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")
    report.append("METHODOLOGY:")
    report.append("-" * 40)
    report.append(f"• Primary Metric: mIoU (mean Intersection over Union)")
    report.append(f"• Data Filtering: ≥{MIN_SAMPLE_SIZE} images per domain")
    report.append(f"• Aggregation: Frequency-weighted average (by num_images)")
    report.append(f"• Normal Domains: {', '.join(NORMAL_DOMAINS)}")
    report.append(f"• Adverse Domains: {', '.join(ADVERSE_DOMAINS)}")
    report.append("")
    
    report.append("STRATEGY RANKING (by Overall mIoU):")
    report.append("-" * 80)
    report.append(f"{'Rank':<5} {'Strategy':<30} {'mIoU':>8} {'Normal':>8} {'Adverse':>8} {'Gap':>8}")
    report.append("-" * 80)
    
    for i, (_, row) in enumerate(summary_df.iterrows(), 1):
        report.append(f"{i:<5} {row['strategy']:<30} {row['overall_mIoU']:>8.2f} {row['normal_mIoU']:>8.2f} {row['adverse_mIoU']:>8.2f} {row['domain_gap']:>+8.2f}")
    
    report_text = '\n'.join(report)
    
    with open(output_dir / 'analysis_report.txt', 'w') as f:
        f.write(report_text)
    
    # Save dataframes
    if not summary_df.empty:
        summary_df.to_csv(output_dir / 'strategy_summary.csv', index=False)
    if not all_df.empty:
        all_df.to_csv(output_dir / 'all_results.csv', index=False)
    
    print("\n" + report_text)
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
