#!/usr/bin/env python3
"""
Generate Stage 2 strategy leaderboard from test results.

Stage 2: Models trained on ALL domains (not just clear_day)
Located in: /scratch/aaa_exchange/AWARE/WEIGHTS_STAGE_2/

Usage:
    python generate_stage2_leaderboard.py               # Auto-refresh results
    python generate_stage2_leaderboard.py --no-refresh  # Use cached results
"""

import os
import json
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def find_test_results(weights_root: str) -> list:
    """Find all test results in the weights directory.
    
    For each model, only return the LATEST test result.
    Also verify the checkpoint_path in results.json actually points to Stage 2.
    """
    results = []
    weights_path = Path(weights_root)
    
    # Group results by model to find the latest
    model_results = defaultdict(list)
    
    for results_file in weights_path.rglob("results.json"):
        try:
            with open(results_file) as f:
                data = json.load(f)
            
            # Parse path: strategy/dataset/model/test_results_detailed/timestamp/results.json
            parts = results_file.relative_to(weights_path).parts
            if len(parts) >= 5:
                strategy = parts[0]
                dataset = parts[1]
                model = parts[2]
                timestamp = parts[4]  # e.g., 20260117_232233
                
                # Verify the checkpoint is from Stage 2 (WEIGHTS_STAGE_2)
                config = data.get('config', {})
                checkpoint_path = config.get('checkpoint_path', '')
                if 'WEIGHTS_STAGE_2' not in checkpoint_path and checkpoint_path:
                    # Skip tests that were run against Stage 1 checkpoints
                    continue
                
                model_key = f"{strategy}/{dataset}/{model}"
                model_results[model_key].append({
                    'strategy': strategy,
                    'dataset': dataset,
                    'model': model,
                    'timestamp': timestamp,
                    'results_path': str(results_file),
                    'data': data
                })
        except Exception as e:
            print(f"Error reading {results_file}: {e}")
    
    # For each model, keep only the latest result
    for model_key, result_list in model_results.items():
        # Sort by timestamp (descending) and take the first
        result_list.sort(key=lambda x: x['timestamp'], reverse=True)
        results.append(result_list[0])
    
    return results


def extract_metrics(result: dict) -> dict:
    """Extract key metrics from a test result."""
    data = result['data']
    
    metrics = {
        'strategy': result['strategy'],
        'dataset': result['dataset'],
        'model': result['model'],
    }
    
    # Overall metrics
    if 'overall' in data:
        metrics['mIoU'] = data['overall'].get('mIoU', 0)
        metrics['aAcc'] = data['overall'].get('aAcc', 0)
    elif 'mIoU' in data:
        metrics['mIoU'] = data['mIoU']
        metrics['aAcc'] = data.get('aAcc', 0)
    else:
        metrics['mIoU'] = 0
        metrics['aAcc'] = 0
    
    # Per-domain metrics for Stage 2 (all weather conditions in training data)
    per_domain = data.get('per_domain', {})
    
    # Define normal vs adverse domains
    normal_domains = ['clear_day']
    adverse_domains = ['cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy']
    
    # Helper to get mIoU from per_domain - handles both formats:
    # - per_domain[domain]['mIoU'] (old format)
    # - per_domain[domain]['summary']['mIoU'] (new format)
    def get_domain_miou(domain_data):
        if 'summary' in domain_data:
            return domain_data['summary'].get('mIoU')
        return domain_data.get('mIoU')
    
    # Calculate normal mIoU
    normal_values = []
    for domain in normal_domains:
        if domain in per_domain:
            val = get_domain_miou(per_domain[domain])
            if val is not None:
                normal_values.append(val)
    metrics['normal_mIoU'] = sum(normal_values) / len(normal_values) if normal_values else metrics['mIoU']
    
    # Calculate adverse mIoU
    adverse_values = []
    for domain in adverse_domains:
        if domain in per_domain:
            val = get_domain_miou(per_domain[domain])
            if val is not None:
                adverse_values.append(val)
    metrics['adverse_mIoU'] = sum(adverse_values) / len(adverse_values) if adverse_values else 0
    
    # Domain gap
    metrics['domain_gap'] = metrics['normal_mIoU'] - metrics['adverse_mIoU']
    
    # Store all domain values
    for domain in normal_domains + adverse_domains:
        if domain in per_domain:
            val = get_domain_miou(per_domain[domain])
            if val is not None:
                metrics[f'domain_{domain}'] = val
    
    return metrics


def get_strategy_type(strategy: str) -> str:
    """Determine strategy type."""
    if strategy == 'baseline':
        return 'Baseline'
    elif strategy.startswith('std_'):
        return 'Standard Aug'
    elif strategy.startswith('gen_'):
        return 'Generative'
    elif strategy == 'std_photometric_distort':
        return 'Augmentation'
    else:
        return 'Other'


def generate_leaderboard(df: pd.DataFrame, output_dir: str):
    """Generate the leaderboard markdown files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate per-strategy averages
    strategy_stats = []
    
    for strategy in df['strategy'].unique():
        strategy_df = df[df['strategy'] == strategy]
        
        stats = {
            'Strategy': strategy,
            'Type': get_strategy_type(strategy),
            'mIoU': strategy_df['mIoU'].mean(),
            'Normal mIoU': strategy_df['normal_mIoU'].mean(),
            'Adverse mIoU': strategy_df['adverse_mIoU'].mean(),
            'Domain Gap': strategy_df['domain_gap'].mean(),
            'Num Models': len(strategy_df),
            'Datasets': ', '.join(sorted(strategy_df['dataset'].unique())),
        }
        strategy_stats.append(stats)
    
    # Create strategy leaderboard DataFrame
    leaderboard_df = pd.DataFrame(strategy_stats)
    
    # Get baseline mIoU for gain calculation
    baseline_miou = leaderboard_df[leaderboard_df['Strategy'] == 'baseline']['mIoU'].values
    baseline_miou = baseline_miou[0] if len(baseline_miou) > 0 else 0
    
    leaderboard_df['Gain vs Baseline'] = leaderboard_df['mIoU'] - baseline_miou
    
    # Sort by mIoU
    leaderboard_df = leaderboard_df.sort_values('mIoU', ascending=False)
    
    # Save CSV
    leaderboard_df.to_csv(output_path / 'strategy_leaderboard_stage2.csv', index=False)
    
    # Generate markdown
    md_content = generate_markdown_leaderboard(leaderboard_df, df, baseline_miou)
    with open(output_path / 'STRATEGY_LEADERBOARD_STAGE2.md', 'w') as f:
        f.write(md_content)
    
    # Generate detailed gains
    detailed_content = generate_detailed_gains(df, baseline_miou)
    with open(output_path / 'DETAILED_GAINS_STAGE2.md', 'w') as f:
        f.write(detailed_content)
    
    return leaderboard_df


def generate_markdown_leaderboard(leaderboard_df: pd.DataFrame, df: pd.DataFrame, baseline_miou: float) -> str:
    """Generate markdown content for leaderboard."""
    lines = [
        "# Stage 2 Strategy Leaderboard",
        "",
        f"**Training:** All domains (not filtered to clear_day)",
        f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Baseline mIoU:** {baseline_miou:.2f}%",
        f"**Total Test Results:** {len(df)}",
        f"**Strategies Evaluated:** {len(leaderboard_df)}",
        "",
        "## Overall Rankings",
        "",
        "| Rank | Strategy | Type | mIoU | Gain | Normal | Adverse | Gap | Models |",
        "|------|----------|------|------|------|--------|---------|-----|--------|",
    ]
    
    for rank, (idx, row) in enumerate(leaderboard_df.iterrows(), 1):
        gain = row['Gain vs Baseline']
        gain_str = f"+{gain:.2f}" if gain > 0 else f"{gain:.2f}"
        if row['Strategy'] == 'baseline':
            gain_str = '-'
        
        lines.append(
            f"| {rank} | {row['Strategy']} | {row['Type']} | {row['mIoU']:.2f} | {gain_str} | "
            f"{row['Normal mIoU']:.2f} | {row['Adverse mIoU']:.2f} | {row['Domain Gap']:.2f} | {row['Num Models']} |"
        )
    
    # Add per-dataset breakdown
    lines.extend([
        "",
        "## Per-Dataset Breakdown",
        "",
    ])
    
    datasets = sorted(df['dataset'].unique())
    for dataset in datasets:
        lines.append(f"### {dataset}")
        lines.append("")
        lines.append("| Strategy | mIoU | Models |")
        lines.append("|----------|------|--------|")
        
        dataset_df = df[df['dataset'] == dataset]
        dataset_stats = dataset_df.groupby('strategy').agg({
            'mIoU': 'mean',
            'model': 'count'
        }).reset_index()
        dataset_stats = dataset_stats.sort_values('mIoU', ascending=False)
        
        for _, row in dataset_stats.iterrows():
            lines.append(f"| {row['strategy']} | {row['mIoU']:.2f} | {row['model']} |")
        
        lines.append("")
    
    return "\n".join(lines)


def generate_detailed_gains(df: pd.DataFrame, baseline_miou: float) -> str:
    """Generate detailed per-dataset and per-domain analysis."""
    lines = [
        "# Stage 2 Detailed Per-Dataset and Per-Domain Analysis",
        "",
        "## Per-Dataset mIoU by Strategy",
        "",
    ]
    
    # Per-dataset table
    datasets = sorted(df['dataset'].unique())
    
    # Get per-strategy, per-dataset averages
    pivot = df.pivot_table(
        values='mIoU',
        index='strategy',
        columns='dataset',
        aggfunc='mean'
    ).reset_index()
    
    # Calculate baseline per dataset
    baseline_row = pivot[pivot['strategy'] == 'baseline']
    
    # Build header
    header = "| Strategy | Type |"
    separator = "|---|---|"
    for ds in datasets:
        header += f" {ds} | Δ{ds} |"
        separator += "---:|---:|"
    header += " Avg |"
    separator += "---:|"
    
    lines.append(header)
    lines.append(separator)
    
    # Add data rows
    for _, row in pivot.iterrows():
        strategy = row['strategy']
        stype = get_strategy_type(strategy)
        line = f"| {strategy} | {stype} |"
        
        gains = []
        for ds in datasets:
            val = row.get(ds, 0)
            if pd.isna(val):
                line += " - | - |"
            else:
                baseline_val = baseline_row[ds].values[0] if len(baseline_row) > 0 and ds in baseline_row else 0
                gain = val - baseline_val if not pd.isna(baseline_val) else 0
                gains.append(gain)
                sign = "+" if gain >= 0 else ""
                line += f" {val:.1f} | {sign}{gain:.1f} |"
        
        avg_gain = sum(gains) / len(gains) if gains else 0
        sign = "+" if avg_gain >= 0 else ""
        line += f" {sign}{avg_gain:.2f} |"
        lines.append(line)
    
    # Per-domain breakdown
    lines.extend([
        "",
        "## Per-Domain mIoU by Strategy",
        "",
    ])
    
    domains = ['clear_day', 'cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy']
    domain_cols = [f'domain_{d}' for d in domains]
    
    # Check which domain columns exist
    available_domains = [d for d in domains if f'domain_{d}' in df.columns]
    
    if available_domains:
        domain_pivot = df.groupby('strategy')[[f'domain_{d}' for d in available_domains]].mean().reset_index()
        
        header = "| Strategy | Type |"
        separator = "|---|---|"
        for d in available_domains:
            header += f" {d} |"
            separator += "---:|"
        header += " Normal Avg | Adverse Avg | Gap |"
        separator += "---:|---:|---:|"
        
        lines.append(header)
        lines.append(separator)
        
        for _, row in domain_pivot.iterrows():
            strategy = row['strategy']
            stype = get_strategy_type(strategy)
            line = f"| {strategy} | {stype} |"
            
            normal_vals = []
            adverse_vals = []
            
            for d in available_domains:
                col = f'domain_{d}'
                val = row.get(col, 0)
                if pd.isna(val):
                    line += " - |"
                else:
                    line += f" {val:.2f} |"
                    if d == 'clear_day':
                        normal_vals.append(val)
                    else:
                        adverse_vals.append(val)
            
            normal_avg = sum(normal_vals) / len(normal_vals) if normal_vals else 0
            adverse_avg = sum(adverse_vals) / len(adverse_vals) if adverse_vals else 0
            gap = normal_avg - adverse_avg
            
            line += f" {normal_avg:.2f} | {adverse_avg:.2f} | {gap:.2f} |"
            lines.append(line)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Generate Stage 2 strategy leaderboard')
    parser.add_argument('--weights-root', type=str, 
                       default='/scratch/aaa_exchange/AWARE/WEIGHTS_STAGE_2',
                       help='Root directory for Stage 2 weights')
    parser.add_argument('--output-dir', type=str,
                       default='result_figures/leaderboard',
                       help='Output directory for leaderboard files')
    parser.add_argument('--no-refresh', action='store_true',
                       help='Use cached results instead of re-scanning (default: auto-refresh)')
    
    args = parser.parse_args()
    
    # Default behavior: always scan for fresh results unless --no-refresh
    if args.no_refresh:
        print("Using cached results (--no-refresh specified)")
        # For Stage 2, we don't have a separate cache file - always scan
        # This flag could be extended to support caching if needed
    
    print(f"Scanning {args.weights_root} for test results...")
    results = find_test_results(args.weights_root)
    print(f"Found {len(results)} test results")
    
    if not results:
        print("No test results found!")
        return
    
    # Extract metrics
    metrics_list = []
    for r in results:
        try:
            metrics = extract_metrics(r)
            metrics_list.append(metrics)
        except Exception as e:
            print(f"Error extracting metrics from {r['results_path']}: {e}")
    
    df = pd.DataFrame(metrics_list)
    print(f"Processed {len(df)} test results from {df['strategy'].nunique()} strategies")
    
    # Show strategy counts
    print("\nStrategies found:")
    for strategy, count in df['strategy'].value_counts().items():
        print(f"  {strategy}: {count} results")
    
    # Generate leaderboard
    leaderboard_df = generate_leaderboard(df, args.output_dir)
    
    print(f"\nLeaderboard saved to {args.output_dir}/")
    print("\nTop 10 Strategies (Stage 2 - All Domain Training):")
    print("-" * 80)
    for idx, row in leaderboard_df.head(10).iterrows():
        gain = row['Gain vs Baseline']
        gain_str = f"+{gain:.2f}" if gain > 0 else f"{gain:.2f}"
        if row['Strategy'] == 'baseline':
            gain_str = '(baseline)'
        print(f"{idx+1:2}. {row['Strategy']:<35} mIoU: {row['mIoU']:.2f}%  {gain_str}")


if __name__ == '__main__':
    main()
