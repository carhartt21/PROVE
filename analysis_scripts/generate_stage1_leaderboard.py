#!/usr/bin/env python3
"""
Generate Stage 1 Strategy Leaderboard

Stage 1: All models are trained with clear_day domain filter only.
Reference baseline is 'baseline' (which is trained on clear_day in Stage 1).

This script generates:
1. Overall strategy ranking by mIoU
2. Per-dataset breakdown
3. Per-domain breakdown (Normal vs Adverse performance)
4. Gain over baseline calculations

Usage:
    python generate_stage1_leaderboard.py
    python generate_stage1_leaderboard.py --refresh  # Re-extract from WEIGHTS
    python generate_stage1_leaderboard.py --weights-root /path/to/WEIGHTS
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import TestResultAnalyzer
try:
    from test_result_analyzer import TestResultAnalyzer
    TEST_ANALYZER_AVAILABLE = True
except ImportError:
    TEST_ANALYZER_AVAILABLE = False
    print("Warning: test_result_analyzer.py not found")

# Configuration
WEIGHTS_ROOT = Path(os.environ.get('PROVE_WEIGHTS_ROOT', '/scratch/aaa_exchange/AWARE/WEIGHTS'))
RESULTS_CSV = PROJECT_ROOT / 'downstream_results.csv'
OUTPUT_DIR = PROJECT_ROOT / 'result_figures' / 'leaderboard'

# Weather domain classifications
NORMAL_DOMAINS = ['clear_day', 'cloudy']
ADVERSE_DOMAINS = ['foggy', 'night', 'rainy', 'snowy']
TRANSITION_DOMAINS = ['dawn_dusk']
ALL_DOMAINS = NORMAL_DOMAINS + TRANSITION_DOMAINS + ADVERSE_DOMAINS

# Strategy type mapping
STRATEGY_TYPES = {
    'baseline': 'Baseline',
    'photometric_distort': 'Augmentation',
    'std_cutmix': 'Standard Aug',
    'std_mixup': 'Standard Aug',
    'std_autoaugment': 'Standard Aug',
    'std_randaugment': 'Standard Aug',
    'std_cutout': 'Standard Aug',
    'std_minimal': 'Standard Aug',
}

MODELS = ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5']
DATASETS = ['bdd10k', 'idd-aw', 'mapillaryvistas', 'outside15k']


def extract_results(weights_root: Path, verbose: bool = False) -> pd.DataFrame:
    """Extract test results from WEIGHTS directory."""
    if not TEST_ANALYZER_AVAILABLE:
        raise RuntimeError("TestResultAnalyzer not available")
    
    print(f"Extracting results from: {weights_root}")
    analyzer = TestResultAnalyzer(str(weights_root))
    analyzer.scan_directory(verbose=verbose)
    analyzer.deduplicate_results()
    
    if not analyzer.test_results:
        print("Warning: No test results found!")
        return pd.DataFrame()
    
    df = pd.DataFrame(analyzer.test_results)
    print(f"Extracted {len(df)} test results")
    return df


def get_strategy_type(strategy: str) -> str:
    """Get the type/category of a strategy."""
    if strategy in STRATEGY_TYPES:
        return STRATEGY_TYPES[strategy]
    elif strategy.startswith('gen_'):
        return 'Generative'
    elif strategy.startswith('std_'):
        return 'Standard Aug'
    else:
        return 'Other'


def parse_per_domain_metrics(row) -> Dict[str, float]:
    """Parse per-domain metrics from a row."""
    if pd.isna(row.get('per_domain_metrics')) or row.get('per_domain_metrics') == '':
        return {}
    
    try:
        metrics = row['per_domain_metrics']
        if isinstance(metrics, str):
            import json
            metrics = json.loads(metrics)  # Convert JSON string to dict
        
        result = {}
        for domain, data in metrics.items():
            if isinstance(data, dict) and 'mIoU' in data:
                result[domain] = data['mIoU']
        return result
    except:
        return {}


def compute_normal_adverse_miou(domain_metrics: Dict[str, float]) -> Tuple[Optional[float], Optional[float]]:
    """Compute Normal and Adverse mIoU from per-domain metrics."""
    normal_vals = [domain_metrics.get(d) for d in NORMAL_DOMAINS if d in domain_metrics]
    adverse_vals = [domain_metrics.get(d) for d in ADVERSE_DOMAINS if d in domain_metrics]
    
    normal_miou = np.mean([v for v in normal_vals if v is not None]) if normal_vals else None
    adverse_miou = np.mean([v for v in adverse_vals if v is not None]) if adverse_vals else None
    
    return normal_miou, adverse_miou


def generate_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """Generate the main strategy leaderboard."""
    
    # Get baseline mIoU for comparison
    baseline_df = df[df['strategy'] == 'baseline']
    baseline_miou = baseline_df['mIoU'].mean() if not baseline_df.empty else None
    
    # Group by strategy
    records = []
    for strategy in sorted(df['strategy'].unique()):
        strat_df = df[df['strategy'] == strategy]
        
        overall_miou = strat_df['mIoU'].mean()
        overall_std = strat_df['mIoU'].std()
        num_results = len(strat_df)
        
        # Compute Normal/Adverse mIoU from per-domain metrics
        normal_mious = []
        adverse_mious = []
        
        for _, row in strat_df.iterrows():
            domain_metrics = parse_per_domain_metrics(row)
            if domain_metrics:
                normal, adverse = compute_normal_adverse_miou(domain_metrics)
                if normal is not None:
                    normal_mious.append(normal)
                if adverse is not None:
                    adverse_mious.append(adverse)
        
        normal_miou = np.mean(normal_mious) if normal_mious else None
        adverse_miou = np.mean(adverse_mious) if adverse_mious else None
        domain_gap = (normal_miou - adverse_miou) if (normal_miou and adverse_miou) else None
        
        # Compute gain vs baseline
        gain = (overall_miou - baseline_miou) if baseline_miou else None
        
        records.append({
            'Strategy': strategy,
            'Type': get_strategy_type(strategy),
            'mIoU': round(overall_miou, 2),
            'Std': round(overall_std, 2),
            'Gain': round(gain, 2) if gain else None,
            'Normal mIoU': round(normal_miou, 2) if normal_miou else None,
            'Adverse mIoU': round(adverse_miou, 2) if adverse_miou else None,
            'Domain Gap': round(domain_gap, 2) if domain_gap else None,
            'Num Tests': num_results,
        })
    
    result_df = pd.DataFrame(records)
    result_df = result_df.sort_values('mIoU', ascending=False).reset_index(drop=True)
    return result_df


def generate_per_dataset_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate per-dataset mIoU breakdown."""
    baseline_df = df[df['strategy'] == 'baseline']
    baseline_by_ds = baseline_df.groupby('dataset')['mIoU'].mean().to_dict()
    
    records = []
    for strategy in sorted(df['strategy'].unique()):
        strat_df = df[df['strategy'] == strategy]
        row = {'Strategy': strategy, 'Type': get_strategy_type(strategy)}
        
        for dataset in DATASETS:
            ds_df = strat_df[strat_df['dataset'] == dataset]
            if not ds_df.empty:
                miou = ds_df['mIoU'].mean()
                baseline = baseline_by_ds.get(dataset)
                gain = (miou - baseline) if baseline else None
                row[dataset] = f"{miou:.2f}"
                row[f"{dataset}_gain"] = f"{gain:+.2f}" if gain else "-"
            else:
                row[dataset] = "-"
                row[f"{dataset}_gain"] = "-"
        
        records.append(row)
    
    result_df = pd.DataFrame(records)
    # Sort by average mIoU
    for ds in DATASETS:
        result_df[f'{ds}_num'] = pd.to_numeric(result_df[ds], errors='coerce')
    result_df['avg'] = result_df[[f'{ds}_num' for ds in DATASETS]].mean(axis=1)
    result_df = result_df.sort_values('avg', ascending=False).reset_index(drop=True)
    result_df = result_df.drop(columns=[f'{ds}_num' for ds in DATASETS] + ['avg'])
    
    return result_df


def generate_per_domain_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate per-domain mIoU breakdown."""
    # Aggregate per-domain metrics by strategy
    domain_data = defaultdict(lambda: defaultdict(list))
    
    for _, row in df.iterrows():
        strategy = row['strategy']
        metrics = parse_per_domain_metrics(row)
        for domain, miou in metrics.items():
            if miou is not None:
                domain_data[strategy][domain].append(miou)
    
    # Get baseline per-domain
    baseline_by_domain = {}
    if 'baseline' in domain_data:
        for domain, vals in domain_data['baseline'].items():
            baseline_by_domain[domain] = np.mean(vals)
    
    records = []
    for strategy in sorted(domain_data.keys()):
        row = {'Strategy': strategy, 'Type': get_strategy_type(strategy)}
        
        normal_vals = []
        adverse_vals = []
        
        for domain in ALL_DOMAINS:
            if domain in domain_data[strategy]:
                miou = np.mean(domain_data[strategy][domain])
                baseline = baseline_by_domain.get(domain)
                gain = (miou - baseline) if baseline else None
                row[domain] = f"{miou:.2f}"
                
                if domain in NORMAL_DOMAINS:
                    normal_vals.append(miou)
                elif domain in ADVERSE_DOMAINS:
                    adverse_vals.append(miou)
            else:
                row[domain] = "-"
        
        # Summary columns
        row['Normal Avg'] = f"{np.mean(normal_vals):.2f}" if normal_vals else "-"
        row['Adverse Avg'] = f"{np.mean(adverse_vals):.2f}" if adverse_vals else "-"
        if normal_vals and adverse_vals:
            row['Gap'] = f"{np.mean(normal_vals) - np.mean(adverse_vals):.2f}"
        else:
            row['Gap'] = "-"
        
        records.append(row)
    
    result_df = pd.DataFrame(records)
    return result_df


def format_markdown_table(df: pd.DataFrame, title: str, description: str = "") -> str:
    """Format a DataFrame as a markdown table."""
    lines = [f"## {title}", ""]
    if description:
        lines.append(description)
        lines.append("")
    
    # Header
    cols = df.columns.tolist()
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    
    # Rows
    for _, row in df.iterrows():
        values = [str(row[c]) if pd.notna(row[c]) else "-" for c in cols]
        lines.append("| " + " | ".join(values) + " |")
    
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate Stage 1 Strategy Leaderboard")
    parser.add_argument('--refresh', action='store_true', 
                       help='Re-extract results from WEIGHTS directory')
    parser.add_argument('--weights-root', type=str, default=None,
                       help='Override WEIGHTS directory path')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = Path(args.weights_root) if args.weights_root else WEIGHTS_ROOT
    
    # Load data
    print("\n" + "=" * 70)
    print("STAGE 1 STRATEGY LEADERBOARD GENERATOR")
    print("=" * 70)
    print(f"Weights root: {weights_path}")
    print("=" * 70)
    
    if args.refresh or not RESULTS_CSV.exists():
        df = extract_results(weights_path, verbose=args.verbose)
        if not df.empty:
            df.to_csv(RESULTS_CSV, index=False)
            print(f"Cached to: {RESULTS_CSV}")
    else:
        print(f"Loading from cache: {RESULTS_CSV}")
        df = pd.read_csv(RESULTS_CSV)
    
    if df.empty:
        print("ERROR: No results found!")
        return
    
    print(f"\nLoaded {len(df)} test results")
    print(f"Strategies: {len(df['strategy'].unique())}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    
    # Generate leaderboard
    print("\n1. Generating main leaderboard...")
    leaderboard_df = generate_leaderboard(df)
    
    print("\n" + "=" * 70)
    print("STRATEGY LEADERBOARD (Stage 1 - Clear Day Training)")
    print("=" * 70)
    print(leaderboard_df.to_string(index=False))
    
    # Get baseline info
    baseline_row = leaderboard_df[leaderboard_df['Strategy'] == 'baseline']
    if not baseline_row.empty:
        baseline_miou = baseline_row.iloc[0]['mIoU']
        print(f"\nBaseline mIoU: {baseline_miou}%")
        
        strategies_above = len(leaderboard_df[leaderboard_df['mIoU'] > baseline_miou])
        print(f"Strategies beating baseline: {strategies_above}/{len(leaderboard_df) - 1}")
    
    # Generate per-dataset table
    print("\n2. Generating per-dataset breakdown...")
    per_dataset_df = generate_per_dataset_table(df)
    
    # Generate per-domain table
    print("\n3. Generating per-domain breakdown...")
    per_domain_df = generate_per_domain_table(df)
    
    # Save markdown
    output_file = OUTPUT_DIR / 'STRATEGY_LEADERBOARD.md'
    with open(output_file, 'w') as f:
        f.write("# Stage 1 Strategy Leaderboard\n\n")
        f.write("**Stage 1**: All models trained with `clear_day` domain filter only.\n\n")
        f.write(f"**Total Results**: {len(df)} test results from {len(df['strategy'].unique())} strategies\n\n")
        f.write("---\n\n")
        
        # Main leaderboard
        f.write(format_markdown_table(
            leaderboard_df,
            "Overall Strategy Ranking",
            "Sorted by mIoU. Gain = improvement over baseline. Domain Gap = Normal mIoU - Adverse mIoU (lower is better)."
        ))
        
        f.write("---\n\n")
        
        # Per-dataset
        f.write(format_markdown_table(
            per_dataset_df,
            "Per-Dataset Breakdown",
            "mIoU performance on each dataset."
        ))
        
        f.write("---\n\n")
        
        # Per-domain
        f.write(format_markdown_table(
            per_domain_df,
            "Per-Domain Breakdown",
            "mIoU performance on each weather domain. Normal = clear_day, cloudy. Adverse = foggy, night, rainy, snowy."
        ))
    
    print(f"\nSaved to: {output_file}")
    
    # Save detailed gains
    detailed_file = OUTPUT_DIR / 'DETAILED_GAINS.md'
    with open(detailed_file, 'w') as f:
        f.write("# Detailed Per-Dataset and Per-Domain Analysis\n\n")
        f.write("## Per-Dataset mIoU by Strategy\n\n")
        
        # Detailed per-dataset with gains
        baseline_df = df[df['strategy'] == 'baseline']
        baseline_by_ds = baseline_df.groupby('dataset')['mIoU'].mean().to_dict()
        
        f.write("| Strategy | Type |")
        for ds in DATASETS:
            f.write(f" {ds} | Δ{ds} |")
        f.write(" Avg |\n")
        
        f.write("|---|---|")
        for _ in DATASETS:
            f.write("---:|---:|")
        f.write("---:|\n")
        
        for _, row in leaderboard_df.iterrows():
            strategy = row['Strategy']
            strat_df = df[df['strategy'] == strategy]
            
            f.write(f"| {strategy} | {row['Type']} |")
            gains = []
            for ds in DATASETS:
                ds_df = strat_df[strat_df['dataset'] == ds]
                if not ds_df.empty:
                    miou = ds_df['mIoU'].mean()
                    baseline = baseline_by_ds.get(ds, 0)
                    gain = miou - baseline
                    gains.append(gain)
                    gain_str = f"+{gain:.1f}" if gain >= 0 else f"{gain:.1f}"
                    f.write(f" {miou:.1f} | {gain_str} |")
                else:
                    f.write(" - | - |")
            
            avg_gain = np.mean(gains) if gains else 0
            f.write(f" {avg_gain:+.2f} |\n")
        
        f.write("\n## Per-Domain mIoU by Strategy\n\n")
        per_domain_df.to_markdown(f, index=False)
    
    print(f"Saved to: {detailed_file}")
    
    # Save CSVs
    leaderboard_df.to_csv(OUTPUT_DIR / 'strategy_leaderboard.csv', index=False)
    per_dataset_df.to_csv(OUTPUT_DIR / 'per_dataset_breakdown.csv', index=False)
    per_domain_df.to_csv(OUTPUT_DIR / 'per_domain_breakdown.csv', index=False)
    print(f"CSVs saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
