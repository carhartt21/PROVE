#!/usr/bin/env python3
"""
Analyze Domain Adaptation Ablation Results

This script analyzes the results from the domain adaptation ablation study,
which evaluates models trained on BDD10k/IDD-AW/MapillaryVistas against ACDC.

Generates:
- Cross-dataset performance heatmap
- Per-domain (weather condition) breakdown
- Architecture comparison plots
- Per-class IoU analysis
- Summary statistics and tables

Usage:
    python analyze_domain_adaptation_ablation.py
    python analyze_domain_adaptation_ablation.py --output-dir ./figures/domain_adaptation
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import warnings

import numpy as np
import pandas as pd

# Try importing visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib/seaborn not available. Visualization disabled.")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Configuration
# ============================================================================

WEIGHTS_ROOT = Path(os.environ.get('PROVE_WEIGHTS_ROOT', '${AWARE_DATA_ROOT}/WEIGHTS'))
RESULTS_ROOT = WEIGHTS_ROOT / 'domain_adaptation_ablation'

SOURCE_DATASETS = ['BDD10k', 'IDD-AW', 'MapillaryVistas']
MODELS = ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5']
ACDC_DOMAINS = ['foggy', 'rainy', 'snowy', 'night', 'cloudy']

# Friendly names for plotting
MODEL_NAMES = {
    'deeplabv3plus_r50': 'DeepLabV3+',
    'pspnet_r50': 'PSPNet',
    'segformer_mit-b5': 'SegFormer',
}

CITYSCAPES_CLASSES = (
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle',
)


# ============================================================================
# Data Loading
# ============================================================================

def load_results() -> Dict:
    """Load all domain adaptation ablation results."""
    results = {}
    
    for source in SOURCE_DATASETS:
        for model in MODELS:
            result_file = RESULTS_ROOT / source.lower().replace('-', '') / model / 'acdc_evaluation.json'
            
            if result_file.exists():
                with open(result_file, 'r') as f:
                    results[f"{source}_{model}"] = json.load(f)
                print(f"  Loaded: {source}/{model}")
            else:
                print(f"  Missing: {source}/{model}")
    
    return results


def results_to_dataframe(results: Dict) -> pd.DataFrame:
    """Convert results to a pandas DataFrame for analysis."""
    rows = []
    
    for key, data in results.items():
        source, model = key.rsplit('_', 1)
        # Handle model names with underscores
        parts = key.split('_')
        if 'mit' in key:
            model = 'segformer_mit-b5'
            source = '_'.join(parts[:-2])
        else:
            model = parts[-1] if parts[-1] in ['deeplabv3plus', 'pspnet'] else '_'.join(parts[-2:])
            source = '_'.join(parts[:-2 if '_r50' in key else -1])
        
        # Actually parse correctly
        for src in SOURCE_DATASETS:
            for mdl in MODELS:
                if key == f"{src}_{mdl}":
                    source = src
                    model = mdl
                    break
        
        overall_miou = data['overall']['mIoU'] * 100
        
        # Overall metrics
        rows.append({
            'source_dataset': source,
            'model': model,
            'model_friendly': MODEL_NAMES.get(model, model),
            'domain': 'Overall',
            'mIoU': overall_miou,
        })
        
        # Per-domain metrics
        for domain, metrics in data.get('per_domain', {}).items():
            rows.append({
                'source_dataset': source,
                'model': model,
                'model_friendly': MODEL_NAMES.get(model, model),
                'domain': domain.capitalize(),
                'mIoU': metrics['mIoU'] * 100,
            })
    
    return pd.DataFrame(rows)


# ============================================================================
# Analysis Functions
# ============================================================================

def compute_summary_statistics(df: pd.DataFrame) -> Dict:
    """Compute summary statistics from results."""
    stats = {}
    
    # Overall best
    overall = df[df['domain'] == 'Overall']
    best_idx = overall['mIoU'].idxmax()
    stats['best_overall'] = {
        'source': overall.loc[best_idx, 'source_dataset'],
        'model': overall.loc[best_idx, 'model'],
        'mIoU': overall.loc[best_idx, 'mIoU'],
    }
    
    # Best by source dataset
    stats['by_source'] = {}
    for source in SOURCE_DATASETS:
        source_data = overall[overall['source_dataset'] == source]
        if not source_data.empty:
            mean_miou = source_data['mIoU'].mean()
            best_model = source_data.loc[source_data['mIoU'].idxmax(), 'model']
            stats['by_source'][source] = {
                'mean_mIoU': mean_miou,
                'best_model': best_model,
                'best_mIoU': source_data['mIoU'].max(),
            }
    
    # Best by model
    stats['by_model'] = {}
    for model in MODELS:
        model_data = overall[overall['model'] == model]
        if not model_data.empty:
            mean_miou = model_data['mIoU'].mean()
            best_source = model_data.loc[model_data['mIoU'].idxmax(), 'source_dataset']
            stats['by_model'][model] = {
                'mean_mIoU': mean_miou,
                'best_source': best_source,
                'best_mIoU': model_data['mIoU'].max(),
            }
    
    # Per-domain analysis
    stats['by_domain'] = {}
    for domain in ['Foggy', 'Rainy', 'Snowy', 'Night', 'Cloudy']:
        domain_data = df[df['domain'] == domain]
        if not domain_data.empty:
            best_idx = domain_data['mIoU'].idxmax()
            stats['by_domain'][domain] = {
                'mean_mIoU': domain_data['mIoU'].mean(),
                'std_mIoU': domain_data['mIoU'].std(),
                'best_source': domain_data.loc[best_idx, 'source_dataset'],
                'best_model': domain_data.loc[best_idx, 'model'],
                'best_mIoU': domain_data['mIoU'].max(),
                'worst_mIoU': domain_data['mIoU'].min(),
            }
    
    return stats


def generate_markdown_report(df: pd.DataFrame, stats: Dict, output_file: Path):
    """Generate a markdown report with results."""
    
    lines = [
        "# Domain Adaptation Ablation Study Results",
        "",
        "## Overview",
        "",
        "This study evaluates cross-dataset domain adaptation by testing models trained on",
        "BDD10k, IDD-AW, and MapillaryVistas against the ACDC adverse weather benchmark.",
        "",
        "**ACDC Domains Evaluated**: Foggy, Rainy, Snowy, Night, Cloudy",
        "",
        "**Excluded from ACDC**: clear_day and dawn_dusk (all reference images with mismatched labels)",
        "",
        "---",
        "",
        "## Summary Statistics",
        "",
        f"**Best Overall Configuration**: {stats['best_overall']['source']} + {stats['best_overall']['model']}",
        f"  - mIoU: **{stats['best_overall']['mIoU']:.2f}%**",
        "",
        "### By Source Dataset",
        "",
        "| Source Dataset | Mean mIoU | Best Model | Best mIoU |",
        "|----------------|-----------|------------|-----------|",
    ]
    
    for source, data in stats['by_source'].items():
        lines.append(f"| {source} | {data['mean_mIoU']:.2f}% | {MODEL_NAMES.get(data['best_model'], data['best_model'])} | {data['best_mIoU']:.2f}% |")
    
    lines.extend([
        "",
        "### By Model Architecture",
        "",
        "| Model | Mean mIoU | Best Source | Best mIoU |",
        "|-------|-----------|-------------|-----------|",
    ])
    
    for model, data in stats['by_model'].items():
        lines.append(f"| {MODEL_NAMES.get(model, model)} | {data['mean_mIoU']:.2f}% | {data['best_source']} | {data['best_mIoU']:.2f}% |")
    
    lines.extend([
        "",
        "### By Weather Domain",
        "",
        "| Domain | Mean mIoU | Std | Best Source | Best Model | Best mIoU |",
        "|--------|-----------|-----|-------------|------------|-----------|",
    ])
    
    for domain, data in stats['by_domain'].items():
        lines.append(f"| {domain} | {data['mean_mIoU']:.2f}% | ±{data['std_mIoU']:.2f} | {data['best_source']} | {MODEL_NAMES.get(data['best_model'], data['best_model'])} | {data['best_mIoU']:.2f}% |")
    
    lines.extend([
        "",
        "---",
        "",
        "## Detailed Results",
        "",
        "### Overall mIoU by Source Dataset × Model",
        "",
    ])
    
    # Create pivot table for overall results
    overall = df[df['domain'] == 'Overall'].pivot(index='source_dataset', columns='model_friendly', values='mIoU')
    
    # Format as markdown table
    lines.append("| Source Dataset | " + " | ".join(overall.columns) + " |")
    lines.append("|" + "---|" * (len(overall.columns) + 1))
    
    for source in overall.index:
        row_values = [f"{overall.loc[source, col]:.2f}%" for col in overall.columns]
        lines.append(f"| {source} | " + " | ".join(row_values) + " |")
    
    lines.extend([
        "",
        "### Per-Domain mIoU Breakdown",
        "",
    ])
    
    # Per-domain tables
    for source in SOURCE_DATASETS:
        lines.append(f"#### {source}")
        lines.append("")
        
        source_data = df[(df['source_dataset'] == source) & (df['domain'] != 'Overall')]
        if source_data.empty:
            lines.append("*No data available*")
            lines.append("")
            continue
        
        pivot = source_data.pivot(index='domain', columns='model_friendly', values='mIoU')
        
        lines.append("| Domain | " + " | ".join(pivot.columns) + " |")
        lines.append("|" + "---|" * (len(pivot.columns) + 1))
        
        for domain in pivot.index:
            row_values = [f"{pivot.loc[domain, col]:.2f}%" if not pd.isna(pivot.loc[domain, col]) else "N/A" for col in pivot.columns]
            lines.append(f"| {domain} | " + " | ".join(row_values) + " |")
        
        lines.append("")
    
    lines.extend([
        "---",
        "",
        "## Key Findings",
        "",
        "1. **Source Dataset Effect**: TBD after results are collected",
        "2. **Architecture Effect**: TBD after results are collected",
        "3. **Domain Difficulty**: TBD after results are collected",
        "",
        "---",
        "",
        "*Report generated automatically by analyze_domain_adaptation_ablation.py*",
    ])
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Report saved to: {output_file}")


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_heatmap(df: pd.DataFrame, output_dir: Path):
    """Plot heatmap of source dataset × model performance."""
    if not HAS_MATPLOTLIB:
        return
    
    overall = df[df['domain'] == 'Overall'].pivot(
        index='source_dataset', 
        columns='model_friendly', 
        values='mIoU'
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(overall, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'mIoU (%)'})
    ax.set_title('Cross-Dataset Domain Adaptation to ACDC\n(Overall mIoU %)')
    ax.set_xlabel('Model Architecture')
    ax.set_ylabel('Training Dataset')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'domain_adaptation_heatmap.png', dpi=150)
    plt.savefig(output_dir / 'domain_adaptation_heatmap.pdf')
    plt.close()
    
    print(f"Saved: {output_dir / 'domain_adaptation_heatmap.png'}")


def plot_per_domain_breakdown(df: pd.DataFrame, output_dir: Path):
    """Plot per-domain performance breakdown."""
    if not HAS_MATPLOTLIB:
        return
    
    domains = ['Foggy', 'Rainy', 'Snowy', 'Night', 'Cloudy']
    domain_data = df[df['domain'].isin(domains)]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, source in enumerate(SOURCE_DATASETS):
        ax = axes[idx]
        source_data = domain_data[domain_data['source_dataset'] == source]
        
        pivot = source_data.pivot(index='domain', columns='model_friendly', values='mIoU')
        pivot = pivot.reindex(domains)
        
        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title(f'Trained on: {source}')
        ax.set_xlabel('ACDC Weather Domain')
        ax.set_ylabel('mIoU (%)')
        ax.set_xticklabels(domains, rotation=45, ha='right')
        ax.legend(title='Model')
        ax.set_ylim(0, 100)
    
    plt.suptitle('Domain Adaptation Performance by Weather Condition', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'per_domain_breakdown.png', dpi=150)
    plt.savefig(output_dir / 'per_domain_breakdown.pdf')
    plt.close()
    
    print(f"Saved: {output_dir / 'per_domain_breakdown.png'}")


def plot_source_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot comparison of source datasets."""
    if not HAS_MATPLOTLIB:
        return
    
    overall = df[df['domain'] == 'Overall']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(SOURCE_DATASETS))
    width = 0.25
    
    for idx, model in enumerate(MODELS):
        model_data = overall[overall['model'] == model]
        values = [model_data[model_data['source_dataset'] == src]['mIoU'].values[0] 
                  if len(model_data[model_data['source_dataset'] == src]) > 0 else 0 
                  for src in SOURCE_DATASETS]
        ax.bar(x + idx * width, values, width, label=MODEL_NAMES.get(model, model))
    
    ax.set_xlabel('Training Dataset')
    ax.set_ylabel('Overall mIoU on ACDC (%)')
    ax.set_title('Cross-Dataset Domain Adaptation Performance')
    ax.set_xticks(x + width)
    ax.set_xticklabels(SOURCE_DATASETS)
    ax.legend()
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'source_comparison.png', dpi=150)
    plt.savefig(output_dir / 'source_comparison.pdf')
    plt.close()
    
    print(f"Saved: {output_dir / 'source_comparison.png'}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze domain adaptation ablation results"
    )
    parser.add_argument('--output-dir', type=str, 
                        default=str(PROJECT_ROOT / 'result_figures' / 'domain_adaptation'),
                        help='Output directory for figures and reports')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Domain Adaptation Ablation Analysis")
    print("=" * 70)
    print()
    
    # Load results
    print("Loading results...")
    results = load_results()
    
    if not results:
        print("\nNo results found. Run the evaluation jobs first:")
        print("  ./scripts/submit_domain_adaptation_ablation.sh --all")
        return
    
    print(f"\nLoaded {len(results)} configurations")
    
    # Convert to DataFrame
    df = results_to_dataframe(results)
    
    # Save raw data
    csv_file = output_dir / 'domain_adaptation_results.csv'
    df.to_csv(csv_file, index=False)
    print(f"Raw data saved to: {csv_file}")
    
    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_summary_statistics(df)
    
    # Generate report
    report_file = output_dir / 'DOMAIN_ADAPTATION_RESULTS.md'
    generate_markdown_report(df, stats, report_file)
    
    # Generate plots
    if not args.no_plots and HAS_MATPLOTLIB:
        print("\nGenerating visualizations...")
        plot_heatmap(df, output_dir)
        plot_per_domain_breakdown(df, output_dir)
        plot_source_comparison(df, output_dir)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Best Configuration: {stats['best_overall']['source']} + {stats['best_overall']['model']}")
    print(f"Best Overall mIoU: {stats['best_overall']['mIoU']:.2f}%")
    print()
    print("By Source Dataset:")
    for source, data in stats['by_source'].items():
        print(f"  {source}: {data['mean_mIoU']:.2f}% (avg)")
    print()
    print("By Model:")
    for model, data in stats['by_model'].items():
        print(f"  {MODEL_NAMES.get(model, model)}: {data['mean_mIoU']:.2f}% (avg)")
    print()
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
