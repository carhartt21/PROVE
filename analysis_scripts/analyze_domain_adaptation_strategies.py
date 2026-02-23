#!/usr/bin/env python3
"""
Analyze Domain Adaptation Strategy Comparison Results

This script analyzes which training strategies produce models that generalize
best to the ACDC adverse weather benchmark.

Results are organized by:
- strategy/dataset/model/domain_adaptation_evaluation.json

Usage:
    python analyze_domain_adaptation_strategies.py
    python analyze_domain_adaptation_strategies.py --output-dir ./figures/domain_adaptation
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from datetime import datetime

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try importing visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ============================================================================
# Configuration
# ============================================================================

WEIGHTS_ROOT = Path(os.environ.get('PROVE_WEIGHTS_ROOT', '${AWARE_DATA_ROOT}/WEIGHTS'))
RESULTS_ROOT = WEIGHTS_ROOT / 'domain_adaptation_ablation'
OUTPUT_ROOT = PROJECT_ROOT / 'result_figures' / 'domain_adaptation'

ACDC_DOMAINS = ['clear_day', 'foggy', 'night', 'rainy', 'snowy']

MODEL_NAMES = {
    'deeplabv3plus_r50': 'DeepLabV3+',
    'pspnet_r50': 'PSPNet',
    'segformer_mit-b5': 'SegFormer',
}

DATASET_NAMES = {
    'bdd10k': 'BDD10k',
    'idd-aw': 'IDD-AW',
}

STRATEGY_FAMILIES = {
    'baseline': 'Baseline',
    'std': 'Standard Aug',
    'gen': 'Generative',
    'photometric': 'Photometric',
}

def get_strategy_family(strategy: str) -> str:
    """Get the family for a strategy."""
    if strategy == 'baseline':
        return 'Baseline'
    elif strategy.startswith('std_'):
        return 'Standard Aug'
    elif strategy.startswith('gen_'):
        return 'Generative'
    elif strategy.startswith('photometric'):
        return 'Photometric'
    return 'Other'


# ============================================================================
# Data Loading
# ============================================================================

def load_all_results() -> List[Dict]:
    """Load all domain adaptation evaluation results."""
    results = []
    
    for result_file in RESULTS_ROOT.rglob('domain_adaptation_evaluation.json'):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Parse path: strategy/dataset/model/...
            rel_path = result_file.relative_to(RESULTS_ROOT)
            parts = rel_path.parts
            
            if len(parts) < 3:
                continue
            
            strategy = parts[0]
            dataset = parts[1]
            model_dir = parts[2]
            
            # Handle model names with ratio suffix
            model = model_dir.replace('_ratio0p50', '')
            has_ratio = '_ratio' in model_dir
            
            # Extract overall mIoU
            overall_miou = data.get('overall', {}).get('mIoU', 0) * 100
            
            # Extract per-domain mIoU
            per_domain = {}
            for domain, metrics in data.get('per_domain', {}).items():
                domain_miou = metrics.get('mIoU', 0)
                # Handle NaN values
                if isinstance(domain_miou, float) and np.isnan(domain_miou):
                    domain_miou = 0
                per_domain[domain] = domain_miou * 100
            
            results.append({
                'strategy': strategy,
                'dataset': dataset,
                'model': model,
                'model_dir': model_dir,
                'has_ratio': has_ratio,
                'overall_miou': overall_miou,
                'per_domain': per_domain,
                'family': get_strategy_family(strategy),
                'file_path': str(result_file),
            })
            
        except Exception as e:
            print(f"  Error loading {result_file}: {e}")
    
    return results


# ============================================================================
# Analysis Functions
# ============================================================================

def compute_strategy_summary(results: List[Dict]) -> Dict:
    """Compute summary statistics by strategy."""
    strategy_data = defaultdict(list)
    
    for r in results:
        strategy_data[r['strategy']].append(r['overall_miou'])
    
    summary = {}
    for strategy, mious in strategy_data.items():
        summary[strategy] = {
            'mean_miou': np.mean(mious),
            'std_miou': np.std(mious),
            'min_miou': np.min(mious),
            'max_miou': np.max(mious),
            'count': len(mious),
        }
    
    return summary


def compute_baseline_delta(results: List[Dict]) -> Dict:
    """Compute delta from baseline for each strategy."""
    # Get baseline results
    baseline_results = {}
    for r in results:
        if r['strategy'] == 'baseline':
            key = (r['dataset'], r['model'])
            baseline_results[key] = r['overall_miou']
    
    # Compute deltas
    strategy_deltas = defaultdict(list)
    for r in results:
        if r['strategy'] != 'baseline':
            key = (r['dataset'], r['model'])
            if key in baseline_results:
                delta = r['overall_miou'] - baseline_results[key]
                strategy_deltas[r['strategy']].append(delta)
    
    # Summarize
    deltas = {}
    for strategy, delta_list in strategy_deltas.items():
        if delta_list:
            deltas[strategy] = {
                'mean_delta': np.mean(delta_list),
                'std_delta': np.std(delta_list),
                'min_delta': np.min(delta_list),
                'max_delta': np.max(delta_list),
                'all_positive': all(d > 0 for d in delta_list),
                'count': len(delta_list),
            }
    
    return deltas


# ============================================================================
# Report Generation
# ============================================================================

def generate_markdown_report(results: List[Dict], output_file: Path):
    """Generate a comprehensive markdown report."""
    
    summary = compute_strategy_summary(results)
    deltas = compute_baseline_delta(results)
    
    # Sort strategies by mean mIoU
    sorted_strategies = sorted(summary.items(), key=lambda x: x[1]['mean_miou'], reverse=True)
    
    lines = [
        "# Domain Adaptation Strategy Comparison",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "## Overview",
        "",
        "This analysis evaluates which training strategies produce models that generalize",
        "best to the ACDC adverse weather benchmark.",
        "",
        f"**Total Results**: {len(results)} configurations",
        f"**Strategies Evaluated**: {len(summary)}",
        f"**Datasets**: {', '.join(set(r['dataset'] for r in results))}",
        f"**Models**: {', '.join(set(r['model'] for r in results))}",
        "",
        "---",
        "",
        "## Strategy Rankings (by Mean mIoU)",
        "",
        "| Rank | Strategy | Mean mIoU | Δ Baseline | Family | N |",
        "|------|----------|-----------|------------|--------|---|",
    ]
    
    baseline_mean = summary.get('baseline', {}).get('mean_miou', 0)
    
    for rank, (strategy, stats) in enumerate(sorted_strategies, 1):
        delta_info = deltas.get(strategy, {})
        mean_delta = delta_info.get('mean_delta', 0) if strategy != 'baseline' else 0
        delta_str = f"+{mean_delta:.2f}%" if mean_delta > 0 else f"{mean_delta:.2f}%" if mean_delta < 0 else "—"
        
        family = get_strategy_family(strategy)
        
        lines.append(f"| {rank} | {strategy} | {stats['mean_miou']:.2f}% | {delta_str} | {family} | {stats['count']} |")
    
    # Per-domain breakdown
    lines.extend([
        "",
        "---",
        "",
        "## Per-Domain Performance",
        "",
        "### By Strategy (Mean mIoU per Domain)",
        "",
        "| Strategy |" + " | ".join(ACDC_DOMAINS) + " |",
        "|----------|" + "|".join(["-----------" for _ in ACDC_DOMAINS]) + "|",
    ])
    
    # Compute per-domain means
    for strategy, _ in sorted_strategies[:10]:  # Top 10
        strategy_results = [r for r in results if r['strategy'] == strategy]
        domain_means = {}
        for domain in ACDC_DOMAINS:
            domain_values = [r['per_domain'].get(domain, 0) for r in strategy_results]
            domain_means[domain] = np.mean(domain_values) if domain_values else 0
        
        row = f"| {strategy} |"
        for domain in ACDC_DOMAINS:
            row += f" {domain_means[domain]:.1f}% |"
        lines.append(row)
    
    # Key findings
    lines.extend([
        "",
        "---",
        "",
        "## Key Findings",
        "",
    ])
    
    # Best strategy overall
    best_strategy = sorted_strategies[0][0]
    best_miou = sorted_strategies[0][1]['mean_miou']
    lines.append(f"1. **Best Strategy**: `{best_strategy}` with {best_miou:.2f}% mean mIoU")
    
    # Count strategies beating baseline
    strategies_beating_baseline = [s for s, d in deltas.items() if d.get('mean_delta', 0) > 0]
    lines.append(f"2. **Strategies Beating Baseline**: {len(strategies_beating_baseline)}/{len(deltas)}")
    
    # Best family
    family_means = defaultdict(list)
    for strategy, stats in summary.items():
        family = get_strategy_family(strategy)
        family_means[family].append(stats['mean_miou'])
    
    best_family = max(family_means.items(), key=lambda x: np.mean(x[1]))
    lines.append(f"3. **Best Strategy Family**: {best_family[0]} ({np.mean(best_family[1]):.2f}% mean)")
    
    # Worst domain overall
    domain_means = defaultdict(list)
    for r in results:
        for domain, miou in r['per_domain'].items():
            domain_means[domain].append(miou)
    
    worst_domain = min(domain_means.items(), key=lambda x: np.mean(x[1]))
    lines.append(f"4. **Most Challenging Domain**: {worst_domain[0]} ({np.mean(worst_domain[1]):.1f}% mean)")
    
    # Write report
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"\nReport saved to: {output_file}")
    return '\n'.join(lines)


# ============================================================================
# Visualization
# ============================================================================

def plot_strategy_comparison(results: List[Dict], output_dir: Path):
    """Create strategy comparison visualizations."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping visualizations")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = compute_strategy_summary(results)
    deltas = compute_baseline_delta(results)
    
    # Sort by mean mIoU
    sorted_strategies = sorted(summary.items(), key=lambda x: x[1]['mean_miou'], reverse=True)
    
    # 1. Bar chart of strategy rankings
    fig, ax = plt.subplots(figsize=(14, 8))
    
    strategies = [s[0] for s in sorted_strategies]
    mious = [s[1]['mean_miou'] for s in sorted_strategies]
    stds = [s[1]['std_miou'] for s in sorted_strategies]
    
    # Color by family
    colors = []
    family_colors = {
        'Baseline': '#2ecc71',
        'Standard Aug': '#3498db',
        'Generative': '#e74c3c',
        'Photometric': '#9b59b6',
        'Other': '#95a5a6',
    }
    for s in strategies:
        family = get_strategy_family(s)
        colors.append(family_colors.get(family, '#95a5a6'))
    
    bars = ax.bar(range(len(strategies)), mious, yerr=stds, capsize=3, color=colors, alpha=0.8)
    
    # Add baseline reference line
    baseline_miou = summary.get('baseline', {}).get('mean_miou', 0)
    ax.axhline(y=baseline_miou, color='green', linestyle='--', linewidth=2, label=f'Baseline ({baseline_miou:.1f}%)')
    
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean mIoU on ACDC (%)', fontsize=12)
    ax.set_title('Domain Adaptation: Strategy Comparison', fontsize=14)
    ax.legend(loc='upper right')
    
    # Add legend for families
    legend_patches = [mpatches.Patch(color=c, label=f) for f, c in family_colors.items() if f != 'Other']
    ax.legend(handles=legend_patches, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'strategy_ranking.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'strategy_ranking.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_dir / 'strategy_ranking.png'}")
    
    # 2. Delta from baseline
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sorted_deltas = sorted(deltas.items(), key=lambda x: x[1]['mean_delta'], reverse=True)
    strategies = [s[0] for s in sorted_deltas]
    mean_deltas = [s[1]['mean_delta'] for s in sorted_deltas]
    
    colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in mean_deltas]
    ax.barh(range(len(strategies)), mean_deltas, color=colors, alpha=0.8)
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies, fontsize=9)
    ax.set_xlabel('Δ mIoU from Baseline (%)', fontsize=12)
    ax.set_title('Domain Adaptation: Improvement over Baseline', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_delta.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'baseline_delta.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_dir / 'baseline_delta.png'}")
    
    # 3. Per-domain heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Build matrix
    top_strategies = [s[0] for s in sorted_strategies[:12]]
    matrix = []
    for strategy in top_strategies:
        strategy_results = [r for r in results if r['strategy'] == strategy]
        row = []
        for domain in ACDC_DOMAINS:
            values = [r['per_domain'].get(domain, 0) for r in strategy_results]
            row.append(np.mean(values) if values else 0)
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    sns.heatmap(matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=[d.replace('_', '\n') for d in ACDC_DOMAINS],
                yticklabels=top_strategies, ax=ax,
                vmin=0, vmax=50)
    ax.set_title('Per-Domain mIoU by Strategy', fontsize=14)
    ax.set_xlabel('ACDC Domain', fontsize=12)
    ax.set_ylabel('Strategy', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_domain_heatmap.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'per_domain_heatmap.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_dir / 'per_domain_heatmap.png'}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze domain adaptation strategy comparison')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_ROOT,
                        help='Output directory for figures and reports')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Domain Adaptation Strategy Comparison Analysis")
    print("=" * 70)
    
    print("\nLoading results...")
    results = load_all_results()
    
    if not results:
        print("\nNo results found!")
        print(f"Expected results in: {RESULTS_ROOT}")
        return 1
    
    print(f"\nLoaded {len(results)} configurations")
    print(f"  Strategies: {len(set(r['strategy'] for r in results))}")
    print(f"  Datasets: {set(r['dataset'] for r in results)}")
    print(f"  Models: {set(r['model'] for r in results)}")
    
    # Generate report
    print("\nGenerating report...")
    report_file = args.output_dir / 'domain_adaptation_strategy_report.md'
    report = generate_markdown_report(results, report_file)
    
    # Print summary to console
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    summary = compute_strategy_summary(results)
    deltas = compute_baseline_delta(results)
    
    sorted_strategies = sorted(summary.items(), key=lambda x: x[1]['mean_miou'], reverse=True)
    
    print(f"\n{'Rank':<5} {'Strategy':<30} {'Mean mIoU':<12} {'Δ Baseline':<12}")
    print("-" * 65)
    
    for rank, (strategy, stats) in enumerate(sorted_strategies, 1):
        delta = deltas.get(strategy, {}).get('mean_delta', 0)
        delta_str = f"+{delta:.2f}%" if delta > 0 else f"{delta:.2f}%" if delta < 0 else "—"
        print(f"{rank:<5} {strategy:<30} {stats['mean_miou']:>8.2f}%   {delta_str:>10}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_strategy_comparison(results, args.output_dir)
    
    print(f"\nAll outputs saved to: {args.output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
