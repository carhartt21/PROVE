#!/usr/bin/env python3
"""
Regenerate Domain Adaptation Analysis Figures

This script generates updated figures from the domain adaptation ablation results
using the cleaned data after removing duplicates and quick-tests.

Usage:
    python analysis_scripts/regenerate_domain_adaptation_figures.py
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
WEIGHTS_ROOT = Path('${AWARE_DATA_ROOT}/WEIGHTS')
OUTPUT_DIR = Path(__file__).parent.parent / 'result_figures' / 'domain_adaptation'


def load_all_results():
    """Load all domain adaptation results from the weights directory."""
    results = []
    for f in WEIGHTS_ROOT.rglob('**/domain_adaptation/*/results.json'):
        try:
            d = json.load(open(f))
            # Skip quick tests
            foggy_images = d.get('domains', {}).get('foggy', {}).get('num_images', 500)
            if foggy_images < 100:
                continue
            
            results.append({
                'strategy': d['strategy'],
                'source_dataset': d['source_dataset'],
                'model': d['model'],
                'model_num_classes': d.get('model_num_classes', 19),
                'clear_day': d['domains']['clear_day']['mIoU'],
                'foggy': d['domains']['foggy']['mIoU'],
                'night': d['domains']['night']['mIoU'],
                'rainy': d['domains']['rainy']['mIoU'],
                'snowy': d['domains']['snowy']['mIoU'],
            })
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    return pd.DataFrame(results)


def aggregate_by_strategy(df):
    """Aggregate results by strategy (averaging over datasets/models)."""
    agg = df.groupby('strategy').agg({
        'clear_day': 'mean',
        'foggy': 'mean',
        'night': 'mean',
        'rainy': 'mean',
        'snowy': 'mean',
    }).reset_index()
    
    agg['overall_mIoU'] = agg[['clear_day', 'foggy', 'night', 'rainy', 'snowy']].mean(axis=1)
    agg['adverse_avg'] = agg[['foggy', 'night', 'rainy', 'snowy']].mean(axis=1)
    agg['domain_gap'] = agg['clear_day'] - agg['adverse_avg']
    agg['n_samples'] = df.groupby('strategy').size().values
    
    return agg.sort_values('overall_mIoU', ascending=False)


def plot_strategy_comparison(df, output_dir):
    """Plot strategy comparison bar chart."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    strategies = df['strategy'].values
    x = np.arange(len(strategies))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df['clear_day'], width, label='Clear Day', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, df['adverse_avg'], width, label='Adverse Weather Avg', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Strategy', fontsize=12)
    ax.set_ylabel('mIoU (%)', fontsize=12)
    ax.set_title('Domain Adaptation: Clear Day vs Adverse Weather Performance', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.set_ylim(0, 40)
    ax.grid(axis='y', alpha=0.3)
    
    # Add baseline line
    baseline_row = df[df['strategy'] == 'baseline'].iloc[0]
    ax.axhline(y=baseline_row['overall_mIoU'], color='black', linestyle='--', alpha=0.5, label='Baseline Avg')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'strategy_comparison_cleaned.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'strategy_comparison_cleaned.png'}")


def plot_per_domain_heatmap(df, output_dir):
    """Plot heatmap of per-domain performance."""
    # Prepare data
    domains = ['clear_day', 'foggy', 'night', 'rainy', 'snowy']
    heatmap_data = df[['strategy'] + domains].set_index('strategy')
    
    fig, ax = plt.subplots(figsize=(10, 14))
    
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn',
                ax=ax, vmin=5, vmax=35, cbar_kws={'label': 'mIoU (%)'})
    
    ax.set_title('Per-Domain mIoU by Strategy', fontsize=14)
    ax.set_xlabel('Weather Domain', fontsize=12)
    ax.set_ylabel('Strategy', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_domain_heatmap_cleaned.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'per_domain_heatmap_cleaned.png'}")


def plot_domain_gap(df, output_dir):
    """Plot domain gap analysis."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort by domain gap
    df_sorted = df.sort_values('domain_gap', ascending=True)
    
    colors = ['#27ae60' if s == 'baseline' else '#3498db' for s in df_sorted['strategy']]
    
    bars = ax.barh(df_sorted['strategy'], df_sorted['domain_gap'], color=colors, alpha=0.8)
    
    ax.set_xlabel('Domain Gap (Clear Day - Adverse Avg) in mIoU %', fontsize=12)
    ax.set_ylabel('Strategy', fontsize=12)
    ax.set_title('Domain Gap: Lower is Better (More Robust)', fontsize=14)
    ax.axvline(x=df[df['strategy'] == 'baseline']['domain_gap'].values[0], 
               color='red', linestyle='--', alpha=0.7, label='Baseline')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'domain_gap_cleaned.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'domain_gap_cleaned.png'}")


def plot_gain_over_baseline(df, output_dir):
    """Plot gain over baseline."""
    baseline_miou = df[df['strategy'] == 'baseline']['overall_mIoU'].values[0]
    
    df_nonbaseline = df[df['strategy'] != 'baseline'].copy()
    df_nonbaseline['gain'] = df_nonbaseline['overall_mIoU'] - baseline_miou
    df_nonbaseline = df_nonbaseline.sort_values('gain', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#27ae60' if g > 0 else '#e74c3c' for g in df_nonbaseline['gain']]
    
    bars = ax.barh(df_nonbaseline['strategy'], df_nonbaseline['gain'], color=colors, alpha=0.8)
    
    ax.set_xlabel('Gain over Baseline (mIoU %)', fontsize=12)
    ax.set_ylabel('Strategy', fontsize=12)
    ax.set_title(f'Performance Gain over Baseline ({baseline_miou:.2f}%)', fontsize=14)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gain_over_baseline_cleaned.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'gain_over_baseline_cleaned.png'}")


def plot_source_dataset_comparison(df_raw, output_dir):
    """Plot comparison across source datasets."""
    # Aggregate by strategy and source
    agg = df_raw.groupby(['strategy', 'source_dataset'])['clear_day'].mean().reset_index()
    pivot = agg.pivot(index='strategy', columns='source_dataset', values='clear_day')
    
    # Sort by overall mean
    pivot['mean'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('mean', ascending=False).drop('mean', axis=1)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    pivot.plot(kind='barh', ax=ax, width=0.8, alpha=0.8)
    
    ax.set_xlabel('Clear Day mIoU (%)', fontsize=12)
    ax.set_ylabel('Strategy', fontsize=12)
    ax.set_title('Clear Day Performance by Source Dataset', fontsize=14)
    ax.legend(title='Source Dataset')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'source_dataset_comparison_cleaned.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'source_dataset_comparison_cleaned.png'}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Regenerating Domain Adaptation Analysis Figures")
    print("=" * 60)
    print()
    
    # Load data
    print("Loading results...")
    df_raw = load_all_results()
    print(f"Loaded {len(df_raw)} results")
    
    # Aggregate
    print("\nAggregating by strategy...")
    df_agg = aggregate_by_strategy(df_raw)
    
    # Save updated CSV
    csv_path = OUTPUT_DIR / 'domain_adaptation_analysis_cleaned.csv'
    df_agg.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    # Generate figures
    print("\nGenerating figures...")
    plot_strategy_comparison(df_agg, OUTPUT_DIR)
    plot_per_domain_heatmap(df_agg, OUTPUT_DIR)
    plot_domain_gap(df_agg, OUTPUT_DIR)
    plot_gain_over_baseline(df_agg, OUTPUT_DIR)
    plot_source_dataset_comparison(df_raw, OUTPUT_DIR)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nTop 5 Strategies:")
    for i, row in df_agg.head(5).iterrows():
        print(f"  {row['strategy']}: {row['overall_mIoU']:.2f}% (n={row['n_samples']})")
    
    baseline = df_agg[df_agg['strategy'] == 'baseline'].iloc[0]
    print(f"\nBaseline: {baseline['overall_mIoU']:.2f}% (rank {df_agg[df_agg['strategy'] == 'baseline'].index[0] + 1}/{len(df_agg)})")
    
    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
