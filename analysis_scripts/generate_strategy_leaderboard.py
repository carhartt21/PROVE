#!/usr/bin/env python3
"""
Comprehensive Strategy Leaderboard Analysis

Generates a leaderboard table comparing all augmentation strategies with:
- Overall mIoU
- Normal weather mIoU  
- Adverse weather mIoU
- Domain Gap (Δ)
- Gap Reduction vs baseline

Primary metric: mIoU (recommended for domain gap analysis)

Usage:
    python generate_strategy_leaderboard.py
    python generate_strategy_leaderboard.py --output leaderboard.md
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
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# Configuration
# ============================================================================

# Weather domain classifications
WEATHER_DOMAINS = ['clear_day', 'cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy']
NORMAL_DOMAINS = ['clear_day', 'cloudy']  # Good weather conditions
ADVERSE_DOMAINS = ['foggy', 'night', 'rainy', 'snowy']  # Challenging conditions
TRANSITION_DOMAINS = ['dawn_dusk']  # Low light but not adverse

# Model types
MODELS = ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5']

# Dataset paths
WEIGHTS_ROOT = Path(os.environ.get('PROVE_WEIGHTS_ROOT', '/scratch/aaa_exchange/AWARE/WEIGHTS'))
RESULTS_CSV = PROJECT_ROOT / 'downstream_results.csv'
OUTPUT_DIR = PROJECT_ROOT / 'result_figures' / 'leaderboard'

# Strategy type mapping
STRATEGY_TYPES = {
    'baseline': 'Baseline',
    'baseline_clear_day': 'Baseline',
    'photometric_distort': 'Augmentation',
    'std_cutmix': 'Standard Aug',
    'std_mixup': 'Standard Aug',
    'std_autoaugment': 'Standard Aug',
    'std_randaugment': 'Standard Aug',
    'std_cutout': 'Standard Aug',
}


# ============================================================================
# Data Loading
# ============================================================================

def load_results_csv(csv_path: Path) -> pd.DataFrame:
    """Load results from CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} results from {csv_path}")
    return df


def load_per_domain_results(weights_root: Path, strategy: str = 'baseline') -> Dict:
    """Load per-domain results from test_report.txt files."""
    
    results = {}
    strategy_dir = weights_root / strategy
    
    if not strategy_dir.exists():
        print(f"Strategy directory not found: {strategy_dir}")
        return results
    
    for dataset_dir in strategy_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        
        for model_dir in dataset_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            
            # Look for test_report.txt (per-domain results)
            report_file = model_dir / 'test_report.txt'
            if not report_file.exists():
                continue
            
            # Parse per-domain results
            try:
                domain_results = parse_test_report(report_file)
                if domain_results:
                    key = f"{dataset}_{model}"
                    results[key] = domain_results
            except Exception as e:
                print(f"Error parsing {report_file}: {e}")
    
    return results


def parse_test_report(report_file: Path) -> Dict:
    """Parse per-domain results from test_report.txt."""
    results = {}
    
    with open(report_file, 'r') as f:
        content = f.read()
    
    # Look for domain-specific mIoU values
    lines = content.split('\n')
    for line in lines:
        for domain in WEATHER_DOMAINS:
            if domain in line.lower() and 'miou' in line.lower():
                try:
                    # Extract mIoU value - format varies
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'miou' in part.lower() and i + 1 < len(parts):
                            value = float(parts[i + 1].strip('%,'))
                            results[domain] = value
                            break
                except:
                    pass
    
    return results


# ============================================================================
# Metrics Calculation
# ============================================================================

def calculate_domain_gap(normal_miou: float, adverse_miou: float) -> float:
    """Calculate domain gap (difference between normal and adverse performance)."""
    return normal_miou - adverse_miou


def calculate_gap_reduction(strategy_gap: float, baseline_gap: float) -> float:
    """Calculate gap reduction compared to baseline."""
    if baseline_gap == 0:
        return 0
    return baseline_gap - strategy_gap


def aggregate_strategy_metrics(df: pd.DataFrame, strategy: str) -> Dict:
    """Calculate aggregated metrics for a strategy across all datasets and models."""
    
    strategy_df = df[df['strategy'] == strategy]
    
    if strategy_df.empty:
        return None
    
    # Filter for standard models (not clear_day variants)
    strategy_df = strategy_df[strategy_df['model'].isin(MODELS)]
    
    if strategy_df.empty:
        return None
    
    metrics = {
        'strategy': strategy,
        'overall_miou': strategy_df['mIoU'].mean(),
        'overall_fwiou': strategy_df['fwIoU'].mean() if 'fwIoU' in strategy_df.columns else None,
        'num_results': len(strategy_df),
        'datasets': list(strategy_df['dataset'].unique()),
        'models': list(strategy_df['model'].unique()),
    }
    
    return metrics


def get_per_domain_metrics(df: pd.DataFrame, strategy: str) -> Dict:
    """Get per-domain metrics for a strategy if available."""
    
    # Check if per-domain data is available
    strategy_df = df[(df['strategy'] == strategy) & (df['has_per_domain'] == True)]
    
    if strategy_df.empty:
        return None
    
    # This would require additional per-domain data loading
    return None


# ============================================================================
# Leaderboard Generation
# ============================================================================

def generate_leaderboard(df: pd.DataFrame, per_domain_results: Dict = None) -> pd.DataFrame:
    """Generate strategy leaderboard with all metrics."""
    
    strategies = df['strategy'].unique()
    
    # Calculate baseline metrics first
    baseline_metrics = aggregate_strategy_metrics(df, 'baseline')
    
    leaderboard_data = []
    
    for strategy in sorted(strategies):
        metrics = aggregate_strategy_metrics(df, strategy)
        
        if metrics is None:
            continue
        
        # Determine strategy type
        if strategy.startswith('gen_'):
            strategy_type = 'Generative'
        elif strategy.startswith('std_'):
            strategy_type = 'Standard Aug'
        elif strategy in STRATEGY_TYPES:
            strategy_type = STRATEGY_TYPES[strategy]
        else:
            strategy_type = 'Other'
        
        row = {
            'Strategy': strategy,
            'Type': strategy_type,
            'Overall mIoU': metrics['overall_miou'],
            'Normal mIoU': None,  # Needs per-domain data
            'Adverse mIoU': None,  # Needs per-domain data
            'Domain Gap (Δ)': None,  # Needs per-domain data
            'Gap Reduction': None,  # Needs per-domain data
            'Num Results': metrics['num_results'],
        }
        
        leaderboard_data.append(row)
    
    leaderboard_df = pd.DataFrame(leaderboard_data)
    
    # Sort by Overall mIoU (descending)
    leaderboard_df = leaderboard_df.sort_values('Overall mIoU', ascending=False)
    
    return leaderboard_df


def load_unified_domain_results() -> pd.DataFrame:
    """Load results from unified_domain_gap analysis if available."""
    
    results_dir = PROJECT_ROOT / 'result_figures' / 'unified_domain_gap'
    
    # Try loading strategy summary
    summary_file = results_dir / 'strategy_summary.csv'
    if summary_file.exists():
        return pd.read_csv(summary_file)
    
    # Try loading all domain results
    all_results_file = results_dir / 'all_domain_results.csv'
    if all_results_file.exists():
        return pd.read_csv(all_results_file)
    
    return None


def generate_comprehensive_leaderboard() -> pd.DataFrame:
    """Generate comprehensive leaderboard using all available data sources."""
    
    print("=" * 70)
    print("Generating Strategy Leaderboard")
    print("=" * 70)
    
    # Load main results
    main_df = load_results_csv(RESULTS_CSV)
    
    # Try to load unified domain gap results
    unified_df = load_unified_domain_results()
    
    if unified_df is not None and 'strategy' in unified_df.columns:
        print(f"Using unified domain gap results ({len(unified_df)} entries)")
        
        # Check what columns are available
        print(f"Available columns: {unified_df.columns.tolist()}")
        
        # Use the unified results for the leaderboard
        leaderboard_data = []
        
        for strategy in unified_df['strategy'].unique():
            strategy_data = unified_df[unified_df['strategy'] == strategy]
            
            # Calculate metrics
            overall_miou = strategy_data['mIoU'].mean() if 'mIoU' in strategy_data.columns else None
            
            # Check for domain-specific columns or aggregate by domain type
            normal_miou = None
            adverse_miou = None
            
            if 'domain' in strategy_data.columns:
                normal_data = strategy_data[strategy_data['domain'].isin(NORMAL_DOMAINS)]
                adverse_data = strategy_data[strategy_data['domain'].isin(ADVERSE_DOMAINS)]
                
                if not normal_data.empty:
                    normal_miou = normal_data['mIoU'].mean()
                if not adverse_data.empty:
                    adverse_miou = adverse_data['mIoU'].mean()
            
            # Determine strategy type
            if strategy.startswith('gen_'):
                strategy_type = 'Generative'
            elif strategy.startswith('std_'):
                strategy_type = 'Standard Aug'
            elif strategy == 'baseline':
                strategy_type = 'Baseline'
            elif strategy == 'baseline_clear_day':
                strategy_type = 'Clear-Day'
            elif strategy == 'photometric_distort':
                strategy_type = 'Augmentation'
            else:
                strategy_type = 'Other'
            
            domain_gap = None
            if normal_miou is not None and adverse_miou is not None:
                domain_gap = normal_miou - adverse_miou
            
            row = {
                'Strategy': strategy,
                'Type': strategy_type,
                'Overall mIoU': round(overall_miou, 2) if overall_miou else None,
                'Normal mIoU': round(normal_miou, 2) if normal_miou else None,
                'Adverse mIoU': round(adverse_miou, 2) if adverse_miou else None,
                'Domain Gap (Δ)': round(domain_gap, 2) if domain_gap else None,
                'Gap Reduction': None,  # Calculated after baseline is known
            }
            
            leaderboard_data.append(row)
        
        leaderboard_df = pd.DataFrame(leaderboard_data)
        
        # Calculate gap reduction vs baseline
        baseline_row = leaderboard_df[leaderboard_df['Strategy'] == 'baseline']
        if not baseline_row.empty:
            baseline_gap = baseline_row['Domain Gap (Δ)'].values[0]
            if baseline_gap is not None:
                leaderboard_df['Gap Reduction'] = leaderboard_df['Domain Gap (Δ)'].apply(
                    lambda x: round(baseline_gap - x, 2) if x is not None else None
                )
        
        # Sort by Overall mIoU
        leaderboard_df = leaderboard_df.sort_values('Overall mIoU', ascending=False)
        
        return leaderboard_df
    
    else:
        print("Unified domain gap results not available, using main CSV only")
        return generate_leaderboard(main_df)


def format_leaderboard_markdown(df: pd.DataFrame, title: str = "Strategy Leaderboard") -> str:
    """Format leaderboard as markdown table."""
    
    lines = [
        f"# {title}",
        "",
        f"Generated from PROVE domain gap analysis pipeline.",
        "",
        "**Metrics:**",
        "- **Overall mIoU**: Mean Intersection over Union across all domains",
        "- **Normal mIoU**: Performance on clear_day + cloudy conditions",
        "- **Adverse mIoU**: Performance on foggy, rainy, snowy, night conditions",
        "- **Domain Gap (Δ)**: Normal - Adverse (positive = worse on adverse)",
        "- **Gap Reduction**: Improvement in domain gap vs baseline",
        "",
        "---",
        "",
    ]
    
    # Add table
    cols = ['Strategy', 'Type', 'Overall mIoU', 'Normal mIoU', 'Adverse mIoU', 'Domain Gap (Δ)', 'Gap Reduction']
    available_cols = [c for c in cols if c in df.columns]
    
    # Header
    lines.append("| " + " | ".join(available_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(available_cols)) + "|")
    
    # Rows
    for _, row in df.iterrows():
        values = []
        for col in available_cols:
            val = row[col]
            if pd.isna(val) or val is None:
                values.append("-")
            elif isinstance(val, float):
                if 'Gap' in col:
                    values.append(f"{val:+.1f}%" if val != 0 else "0.0%")
                else:
                    values.append(f"{val:.1f}%")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    
    return "\n".join(lines)


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_baseline(df: pd.DataFrame) -> Dict:
    """Analyze baseline strategy performance."""
    
    baseline_df = df[df['strategy'] == 'baseline']
    baseline_df = baseline_df[baseline_df['model'].isin(MODELS)]
    
    analysis = {
        'overall_miou': baseline_df['mIoU'].mean(),
        'overall_std': baseline_df['mIoU'].std(),
        'by_dataset': {},
        'by_model': {},
    }
    
    for dataset in baseline_df['dataset'].unique():
        ds_data = baseline_df[baseline_df['dataset'] == dataset]
        analysis['by_dataset'][dataset] = {
            'miou': ds_data['mIoU'].mean(),
            'std': ds_data['mIoU'].std(),
        }
    
    for model in baseline_df['model'].unique():
        model_data = baseline_df[baseline_df['model'] == model]
        analysis['by_model'][model] = {
            'miou': model_data['mIoU'].mean(),
            'std': model_data['mIoU'].std(),
        }
    
    return analysis


def analyze_baseline_clear_day(df: pd.DataFrame) -> Dict:
    """Analyze baseline_clear_day (models trained only on clear day) performance."""
    
    clear_day_models = [f"{m}_clear_day" for m in MODELS]
    clearday_df = df[df['model'].isin(clear_day_models)]
    
    if clearday_df.empty:
        return None
    
    analysis = {
        'overall_miou': clearday_df['mIoU'].mean(),
        'overall_std': clearday_df['mIoU'].std(),
        'by_dataset': {},
        'by_model': {},
    }
    
    for dataset in clearday_df['dataset'].unique():
        ds_data = clearday_df[clearday_df['dataset'] == dataset]
        analysis['by_dataset'][dataset] = {
            'miou': ds_data['mIoU'].mean(),
            'std': ds_data['mIoU'].std(),
        }
    
    for model in clear_day_models:
        model_data = clearday_df[clearday_df['model'] == model]
        if not model_data.empty:
            analysis['by_model'][model] = {
                'miou': model_data['mIoU'].mean(),
                'std': model_data['mIoU'].std(),
            }
    
    return analysis


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate strategy leaderboard")
    parser.add_argument('--output', type=str, default=None, help='Output markdown file')
    parser.add_argument('--csv', type=str, default=None, help='Output CSV file')
    
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load main results
    print("\n1. Loading baseline analysis...")
    df = load_results_csv(RESULTS_CSV)
    
    # Analyze baseline
    print("\n2. Analyzing baseline performance...")
    baseline_analysis = analyze_baseline(df)
    print(f"   Baseline Overall mIoU: {baseline_analysis['overall_miou']:.2f}%")
    print(f"   By Dataset:")
    for ds, data in baseline_analysis['by_dataset'].items():
        print(f"     {ds}: {data['miou']:.2f}% ± {data['std']:.2f}")
    
    # Analyze baseline_clear_day
    print("\n3. Analyzing baseline_clear_day performance...")
    clearday_analysis = analyze_baseline_clear_day(df)
    if clearday_analysis:
        print(f"   Clear-Day Baseline mIoU: {clearday_analysis['overall_miou']:.2f}%")
        print(f"   By Dataset:")
        for ds, data in clearday_analysis['by_dataset'].items():
            print(f"     {ds}: {data['miou']:.2f}% ± {data['std']:.2f}")
    else:
        print("   No clear_day baseline results found")
    
    # Generate leaderboard
    print("\n4. Generating comprehensive leaderboard...")
    leaderboard_df = generate_comprehensive_leaderboard()
    
    # Print leaderboard
    print("\n" + "=" * 70)
    print("STRATEGY LEADERBOARD")
    print("=" * 70)
    print(leaderboard_df.to_string(index=False))
    
    # Save outputs
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = OUTPUT_DIR / 'STRATEGY_LEADERBOARD.md'
    
    md_content = format_leaderboard_markdown(leaderboard_df)
    with open(output_file, 'w') as f:
        f.write(md_content)
    print(f"\nLeaderboard saved to: {output_file}")
    
    csv_file = args.csv if args.csv else OUTPUT_DIR / 'strategy_leaderboard.csv'
    leaderboard_df.to_csv(csv_file, index=False)
    print(f"CSV saved to: {csv_file}")
    
    # Also save analysis summary
    summary = {
        'baseline': baseline_analysis,
        'baseline_clear_day': clearday_analysis,
        'leaderboard_rows': len(leaderboard_df),
    }
    
    summary_file = OUTPUT_DIR / 'analysis_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Analysis summary saved to: {summary_file}")


if __name__ == '__main__':
    main()
