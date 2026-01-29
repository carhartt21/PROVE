#!/usr/bin/env python3
"""
Generate Stage 1 Strategy Leaderboard

Stage 1: All models are trained with clear_day domain filter only.
Reference baseline is 'baseline' (which is trained on clear_day in Stage 1).

This script generates:
1. Overall strategy ranking by selected metric (mIoU, aAcc, mAcc, fwIoU)
2. Per-dataset breakdown
3. Per-domain breakdown (Normal vs Adverse performance)
4. Gain over baseline calculations

Usage:
    python generate_stage1_leaderboard.py               # Auto-refresh results
    python generate_stage1_leaderboard.py --no-refresh  # Use cached results
    python generate_stage1_leaderboard.py --metric aAcc # Use pixel accuracy instead of mIoU
    python generate_stage1_leaderboard.py --weights-root /path/to/WEIGHTS

Available metrics:
    mIoU  - Mean Intersection over Union (default)
    aAcc  - Pixel Accuracy (overall accuracy)
    mAcc  - Mean Class Accuracy
    fwIoU - Frequency-weighted IoU
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

# Valid metrics for leaderboard generation
VALID_METRICS = ['mIoU', 'aAcc', 'mAcc', 'fwIoU']
METRIC_DESCRIPTIONS = {
    'mIoU': 'Mean Intersection over Union',
    'aAcc': 'Pixel Accuracy (Overall Accuracy)',
    'mAcc': 'Mean Class Accuracy',
    'fwIoU': 'Frequency-Weighted IoU'
}

# Weather domain classifications
NORMAL_DOMAINS = ['clear_day', 'cloudy']
ADVERSE_DOMAINS = ['foggy', 'night', 'rainy', 'snowy']
TRANSITION_DOMAINS = ['dawn_dusk']
ALL_DOMAINS = NORMAL_DOMAINS + TRANSITION_DOMAINS + ADVERSE_DOMAINS

# Strategy type mapping
STRATEGY_TYPES = {
    'baseline': 'Baseline',
    'std_photometric_distort': 'Augmentation',
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


def parse_per_domain_metrics(row, metric: str = 'mIoU') -> Dict[str, float]:
    """Parse per-domain metrics from a row.
    
    Args:
        row: DataFrame row with per_domain_metrics
        metric: Which metric to extract ('mIoU', 'aAcc', 'mAcc', 'fwIoU')
    
    Returns:
        Dict mapping domain name to metric value
    """
    if pd.isna(row.get('per_domain_metrics')) or row.get('per_domain_metrics') == '':
        return {}
    
    try:
        metrics = row['per_domain_metrics']
        if isinstance(metrics, str):
            import ast
            # Use ast.literal_eval for Python dict syntax (single quotes)
            # Fall back to json.loads for JSON format (double quotes)
            try:
                metrics = ast.literal_eval(metrics)
            except (ValueError, SyntaxError):
                import json
                metrics = json.loads(metrics)
        
        result = {}
        for domain, data in metrics.items():
            if isinstance(data, dict) and metric in data:
                result[domain] = data[metric]
        return result
    except:
        return {}


def compute_normal_adverse_metric(domain_metrics: Dict[str, float]) -> Tuple[Optional[float], Optional[float]]:
    """Compute Normal and Adverse metric values from per-domain metrics.
    
    Works with any metric (mIoU, aAcc, mAcc, fwIoU).
    """
    normal_vals = [domain_metrics.get(d) for d in NORMAL_DOMAINS if d in domain_metrics]
    adverse_vals = [domain_metrics.get(d) for d in ADVERSE_DOMAINS if d in domain_metrics]
    
    normal_val = np.mean([v for v in normal_vals if v is not None]) if normal_vals else None
    adverse_val = np.mean([v for v in adverse_vals if v is not None]) if adverse_vals else None
    
    return normal_val, adverse_val


def generate_leaderboard(df: pd.DataFrame, metric: str = 'mIoU') -> pd.DataFrame:
    """Generate the main strategy leaderboard.
    
    Args:
        df: DataFrame with test results
        metric: Which metric to use for ranking ('mIoU', 'aAcc', 'mAcc', 'fwIoU')
    
    Returns:
        DataFrame with strategy leaderboard sorted by selected metric
    """
    # Validate metric exists in dataframe
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in results. Available: {list(df.columns)}")
    
    # Get baseline metric value for comparison
    baseline_df = df[df['strategy'] == 'baseline']
    baseline_val = baseline_df[metric].mean() if not baseline_df.empty else None
    
    # Group by strategy
    records = []
    for strategy in sorted(df['strategy'].unique()):
        strat_df = df[df['strategy'] == strategy]
        
        overall_val = strat_df[metric].mean()
        overall_std = strat_df[metric].std()
        num_results = len(strat_df)
        
        # Compute Normal/Adverse values from per-domain metrics
        normal_vals = []
        adverse_vals = []
        
        for _, row in strat_df.iterrows():
            domain_metrics = parse_per_domain_metrics(row, metric)
            if domain_metrics:
                normal, adverse = compute_normal_adverse_metric(domain_metrics)
                if normal is not None:
                    normal_vals.append(normal)
                if adverse is not None:
                    adverse_vals.append(adverse)
        
        normal_avg = np.mean(normal_vals) if normal_vals else None
        adverse_avg = np.mean(adverse_vals) if adverse_vals else None
        domain_gap = (normal_avg - adverse_avg) if (normal_avg and adverse_avg) else None
        
        # Compute gain vs baseline
        gain = (overall_val - baseline_val) if baseline_val else None
        
        records.append({
            'Strategy': strategy,
            'Type': get_strategy_type(strategy),
            metric: round(overall_val, 2),
            'Std': round(overall_std, 2),
            'Gain': round(gain, 2) if gain else None,
            f'Normal {metric}': round(normal_avg, 2) if normal_avg else None,
            f'Adverse {metric}': round(adverse_avg, 2) if adverse_avg else None,
            'Domain Gap': round(domain_gap, 2) if domain_gap else None,
            'Num Tests': num_results,
        })
    
    result_df = pd.DataFrame(records)
    result_df = result_df.sort_values(metric, ascending=False).reset_index(drop=True)
    return result_df


def generate_per_dataset_table(df: pd.DataFrame, metric: str = 'mIoU') -> pd.DataFrame:
    """Generate per-dataset metric breakdown.
    
    Args:
        df: DataFrame with test results
        metric: Which metric to use ('mIoU', 'aAcc', 'mAcc', 'fwIoU')
    """
    baseline_df = df[df['strategy'] == 'baseline']
    baseline_by_ds = baseline_df.groupby('dataset')[metric].mean().to_dict()
    
    records = []
    for strategy in sorted(df['strategy'].unique()):
        strat_df = df[df['strategy'] == strategy]
        row = {'Strategy': strategy, 'Type': get_strategy_type(strategy)}
        
        for dataset in DATASETS:
            ds_df = strat_df[strat_df['dataset'] == dataset]
            if not ds_df.empty:
                val = ds_df[metric].mean()
                baseline = baseline_by_ds.get(dataset)
                gain = (val - baseline) if baseline else None
                row[dataset] = f"{val:.2f}"
                row[f"{dataset}_gain"] = f"{gain:+.2f}" if gain else "-"
            else:
                row[dataset] = "-"
                row[f"{dataset}_gain"] = "-"
        
        records.append(row)
    
    result_df = pd.DataFrame(records)
    # Sort by average metric value
    for ds in DATASETS:
        result_df[f'{ds}_num'] = pd.to_numeric(result_df[ds], errors='coerce')
    result_df['avg'] = result_df[[f'{ds}_num' for ds in DATASETS]].mean(axis=1)
    result_df = result_df.sort_values('avg', ascending=False).reset_index(drop=True)
    result_df = result_df.drop(columns=[f'{ds}_num' for ds in DATASETS] + ['avg'])
    
    return result_df


def generate_per_domain_table(df: pd.DataFrame, metric: str = 'mIoU') -> pd.DataFrame:
    """Generate per-domain metric breakdown.
    
    Args:
        df: DataFrame with test results
        metric: Which metric to use ('mIoU', 'aAcc', 'mAcc', 'fwIoU')
    """
    # Aggregate per-domain metrics by strategy
    domain_data = defaultdict(lambda: defaultdict(list))
    
    for _, row in df.iterrows():
        strategy = row['strategy']
        metrics = parse_per_domain_metrics(row, metric)
        for domain, val in metrics.items():
            if val is not None:
                domain_data[strategy][domain].append(val)
    
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
                val = np.mean(domain_data[strategy][domain])
                baseline = baseline_by_domain.get(domain)
                gain = (val - baseline) if baseline else None
                row[domain] = f"{val:.2f}"
                
                if domain in NORMAL_DOMAINS:
                    normal_vals.append(val)
                elif domain in ADVERSE_DOMAINS:
                    adverse_vals.append(val)
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


def filter_to_complete_configs(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Filter DataFrame to only include dataset+model configurations where ALL strategies have results.
    
    This ensures fair comparison by requiring equal coverage across all strategies.
    
    Args:
        df: DataFrame with test results
        verbose: Print details about filtering
        
    Returns:
        Filtered DataFrame with only complete configurations
    """
    all_strategies = set(df['strategy'].unique())
    
    # Normalize model names by stripping ratio suffix (e.g., "_ratio0p50")
    # This ensures baseline/std_* and gen_* strategies are grouped together
    df = df.copy()
    df['model_normalized'] = df['model'].str.replace(r'_ratio\d+p\d+$', '', regex=True)
    
    # Find all configs using normalized model names
    configs = df.groupby(['dataset', 'model_normalized']).apply(
        lambda x: set(x['strategy'].unique())
    ).to_dict()
    
    # Identify complete configs (have all strategies)
    complete_configs = [
        config for config, strategies in configs.items()
        if strategies == all_strategies
    ]
    
    if verbose or True:  # Always show this info
        print(f"\n{'='*70}")
        print(f"FAIR COMPARISON MODE - Filtering to Complete Configurations")
        print(f"{'='*70}")
        print(f"Total strategies: {len(all_strategies)}")
        print(f"Total configurations: {len(configs)}")
        print(f"Complete configurations: {len(complete_configs)}")
        
        if complete_configs:
            print(f"\nUsing configurations:")
            for i, (dataset, model) in enumerate(sorted(complete_configs), 1):
                print(f"  {i}. {dataset} + {model}")
        else:
            print("\n⚠️  WARNING: No configurations have complete strategy coverage!")
            print("    Showing coverage for each configuration:")
            for (dataset, model), strategies in sorted(configs.items()):
                coverage = len(strategies) / len(all_strategies) * 100
                missing = all_strategies - strategies
                print(f"    {dataset} + {model}: {len(strategies)}/{len(all_strategies)} ({coverage:.0f}%)")
                if missing and verbose:
                    print(f"      Missing: {', '.join(sorted(missing)[:5])}" + 
                          (f" + {len(missing)-5} more" if len(missing) > 5 else ""))
        print(f"{'='*70}\n")
    
    if not complete_configs:
        print("ERROR: Cannot create fair leaderboard - no complete configurations found")
        print("Recommendation: Wait for all MapillaryVistas/OUTSIDE15k tests to complete")
        return pd.DataFrame()
    
    # Filter to complete configs using normalized model names
    filtered_df = df[df.apply(lambda r: (r['dataset'], r['model_normalized']) in complete_configs, axis=1)]
    
    # Drop the temporary normalized column
    filtered_df = filtered_df.drop(columns=['model_normalized'])
    
    print(f"Filtered {len(df)} → {len(filtered_df)} test results")
    print(f"Each strategy has exactly {len(complete_configs)} test results\n")
    
    return filtered_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate Stage 1 Strategy Leaderboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available metrics:
  mIoU  - Mean Intersection over Union (default)
  aAcc  - Pixel Accuracy (overall accuracy)
  mAcc  - Mean Class Accuracy
  fwIoU - Frequency-weighted IoU

Examples:
  %(prog)s                        # Generate leaderboard with mIoU
  %(prog)s --metric aAcc          # Use pixel accuracy instead
  %(prog)s --metric mAcc --no-refresh  # Use mean class accuracy from cache
"""
    )
    parser.add_argument('--no-refresh', action='store_true', 
                       help='Use cached results instead of re-extracting from WEIGHTS')
    parser.add_argument('--weights-root', type=str, default=None,
                       help='Override WEIGHTS directory path')
    parser.add_argument('--metric', type=str, default='mIoU', choices=VALID_METRICS,
                       help='Metric to use for ranking (default: mIoU)')
    parser.add_argument('--fair', action='store_true',
                       help='Use only configurations with complete strategy coverage for fair comparison')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    metric = args.metric
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = Path(args.weights_root) if args.weights_root else WEIGHTS_ROOT
    
    # Load data
    print("\n" + "=" * 70)
    print("STAGE 1 STRATEGY LEADERBOARD GENERATOR")
    print("=" * 70)
    print(f"Weights root: {weights_path}")
    print(f"Metric: {metric} ({METRIC_DESCRIPTIONS.get(metric, '')})")
    print("=" * 70)
    
    # Default behavior: refresh unless --no-refresh is specified
    if not args.no_refresh or not RESULTS_CSV.exists():
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
    
    # Apply fair filtering if requested
    if args.fair:
        df = filter_to_complete_configs(df, verbose=args.verbose)
        if df.empty:
            return
    
    # Generate leaderboard
    print(f"\n1. Generating main leaderboard by {metric}...")
    leaderboard_df = generate_leaderboard(df, metric)
    
    print("\n" + "=" * 70)
    print(f"STRATEGY LEADERBOARD (Stage 1 - Clear Day Training) - Ranked by {metric}")
    print("=" * 70)
    print(leaderboard_df.to_string(index=False))
    
    # Get baseline info
    baseline_row = leaderboard_df[leaderboard_df['Strategy'] == 'baseline']
    if not baseline_row.empty:
        baseline_val = baseline_row.iloc[0][metric]
        print(f"\nBaseline {metric}: {baseline_val}%")
        
        strategies_above = len(leaderboard_df[leaderboard_df[metric] > baseline_val])
        print(f"Strategies beating baseline: {strategies_above}/{len(leaderboard_df) - 1}")
    
    # Generate per-dataset table
    print(f"\n2. Generating per-dataset breakdown by {metric}...")
    per_dataset_df = generate_per_dataset_table(df, metric)
    
    # Generate per-domain table
    print(f"\n3. Generating per-domain breakdown by {metric}...")
    per_domain_df = generate_per_domain_table(df, metric)
    
    # Save markdown
    mode_suffix = '_FAIR' if args.fair else ''
    output_file = OUTPUT_DIR / f'STRATEGY_LEADERBOARD_{metric.upper()}{mode_suffix}.md'
    with open(output_file, 'w') as f:
        title_mode = " (Fair Comparison)" if args.fair else ""
        f.write(f"# Stage 1 Strategy Leaderboard{title_mode} (by {metric})\n\n")
        f.write("**Stage 1**: All models trained with `clear_day` domain filter only.\n\n")
        f.write(f"**Metric**: {metric} ({METRIC_DESCRIPTIONS.get(metric, '')})\n\n")
        
        if args.fair:
            f.write("**Fair Comparison Mode**: Only includes dataset+model configurations where ALL strategies have test results.\n")
            f.write("This ensures equal coverage and prevents incomplete results from skewing rankings.\n\n")
        
        f.write(f"**Total Results**: {len(df)} test results from {len(df['strategy'].unique())} strategies\n\n")
        f.write("---\n\n")
        
        # Main leaderboard
        f.write(format_markdown_table(
            leaderboard_df,
            "Overall Strategy Ranking",
            f"Sorted by {metric}. Gain = improvement over baseline. Domain Gap = Normal {metric} - Adverse {metric} (lower is better)."
        ))
        
        f.write("---\n\n")
        
        # Per-dataset
        f.write(format_markdown_table(
            per_dataset_df,
            "Per-Dataset Breakdown",
            f"{metric} performance on each dataset."
        ))
        
        f.write("---\n\n")
        
        # Per-domain
        f.write(format_markdown_table(
            per_domain_df,
            "Per-Domain Breakdown",
            f"{metric} performance on each weather domain. Normal = clear_day, cloudy. Adverse = foggy, night, rainy, snowy."
        ))
    
    print(f"\nSaved to: {output_file}")
    
    # Save detailed gains
    detailed_file = OUTPUT_DIR / f'DETAILED_GAINS_{metric.upper()}.md'
    with open(detailed_file, 'w') as f:
        f.write(f"# Detailed Per-Dataset and Per-Domain Analysis ({metric})\n\n")
        f.write(f"## Per-Dataset {metric} by Strategy\n\n")
        
        # Detailed per-dataset with gains
        baseline_df = df[df['strategy'] == 'baseline']
        baseline_by_ds = baseline_df.groupby('dataset')[metric].mean().to_dict()
        
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
                    val = ds_df[metric].mean()
                    baseline = baseline_by_ds.get(ds, 0)
                    gain = val - baseline
                    gains.append(gain)
                    gain_str = f"+{gain:.1f}" if gain >= 0 else f"{gain:.1f}"
                    f.write(f" {val:.1f} | {gain_str} |")
                else:
                    f.write(" - | - |")
            
            avg_gain = np.mean(gains) if gains else 0
            f.write(f" {avg_gain:+.2f} |\n")
        
        f.write(f"\n## Per-Domain {metric} by Strategy\n\n")
        per_domain_df.to_markdown(f, index=False)
    
    print(f"Saved to: {detailed_file}")
    
    # Save CSVs (include metric in filename for non-default metrics)
    suffix = f"_{metric.lower()}" if metric != 'mIoU' else ""
    leaderboard_df.to_csv(OUTPUT_DIR / f'strategy_leaderboard{suffix}.csv', index=False)
    per_dataset_df.to_csv(OUTPUT_DIR / f'per_dataset_breakdown{suffix}.csv', index=False)
    per_domain_df.to_csv(OUTPUT_DIR / f'per_domain_breakdown{suffix}.csv', index=False)
    print(f"CSVs saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
