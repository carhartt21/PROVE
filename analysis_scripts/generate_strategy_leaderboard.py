#!/usr/bin/env python3
"""
Comprehensive Strategy Leaderboard Analysis

Generates a leaderboard table comparing all augmentation strategies with:
- Overall mIoU
- Normal weather mIoU  
- Adverse weather mIoU
- Domain Gap (Δ)
- Gap Reduction vs baseline_clear_day (models trained only on clear_day data)

Primary metric: mIoU (recommended for domain gap analysis)

**Baseline Terminology:**
- baseline_clear_day: Models trained only on clear_day subset (THE REFERENCE BASELINE)
- baseline_full: Models trained on all weather conditions (formerly just 'baseline')

Gap reduction is calculated relative to baseline_clear_day to measure improvement
over models that never saw adverse weather during training.

Uses test_result_analyzer.py to extract fresh results from /scratch/aaa_exchange/AWARE/WEIGHTS

Usage:
    python generate_strategy_leaderboard.py
    python generate_strategy_leaderboard.py --output leaderboard.md
    python generate_strategy_leaderboard.py --refresh  # Re-extract from WEIGHTS
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
PROJECT_ROOT = Path(__file__).parent.parent  # Go up from analysis_scripts to PROVE
sys.path.insert(0, str(PROJECT_ROOT))

# Import TestResultAnalyzer from test_result_analyzer.py
try:
    from test_result_analyzer import TestResultAnalyzer
    TEST_ANALYZER_AVAILABLE = True
except ImportError:
    TEST_ANALYZER_AVAILABLE = False
    print("Warning: test_result_analyzer.py not found, using CSV-based loading only")

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
# baseline_clear_day: trained only on clear_day data (REFERENCE BASELINE)
# baseline_full: trained on all weather domains (formerly 'baseline')
STRATEGY_TYPES = {
    'baseline': 'Baseline Full',  # Legacy name for full baseline
    'baseline_full': 'Baseline Full',  # Explicit full baseline name
    'baseline_clear_day': 'Baseline Clear Day',  # Reference baseline
    'photometric_distort': 'Augmentation',
    'std_cutmix': 'Standard Aug',
    'std_mixup': 'Standard Aug',
    'std_autoaugment': 'Standard Aug',
    'std_randaugment': 'Standard Aug',
    'std_cutout': 'Standard Aug',
}

# Baseline configurations
BASELINE_CLEAR_DAY = 'baseline_clear_day'  # Reference baseline (trained on clear_day only)
BASELINE_FULL = 'baseline'  # Full baseline (trained on all conditions) - maps to 'baseline' in data


# ============================================================================
# Data Loading
# ============================================================================

def extract_results_from_weights(weights_root: Path = WEIGHTS_ROOT, verbose: bool = False) -> pd.DataFrame:
    """
    Extract test results directly from WEIGHTS directory using TestResultAnalyzer.
    
    This is the preferred method as it reads fresh results from disk.
    
    Args:
        weights_root: Path to the WEIGHTS directory
        verbose: Print verbose output during scanning
        
    Returns:
        DataFrame with all test results
    """
    if not TEST_ANALYZER_AVAILABLE:
        raise RuntimeError("TestResultAnalyzer not available. Please ensure test_result_analyzer.py exists.")
    
    print(f"Extracting results from: {weights_root}")
    
    # Create analyzer and scan directory
    analyzer = TestResultAnalyzer(str(weights_root))
    analyzer.scan_directory(verbose=verbose)
    analyzer.deduplicate_results()
    
    if not analyzer.test_results:
        print("Warning: No test results found!")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(analyzer.test_results)
    print(f"Extracted {len(df)} test results from WEIGHTS directory")
    
    return df


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
    """Generate strategy leaderboard with all metrics.
    
    Gap reduction is calculated relative to baseline_clear_day (models trained only on clear_day).
    This shows how much each strategy improves over a model that never saw adverse weather.
    """
    
    strategies = df['strategy'].unique()
    
    # Calculate baseline_clear_day metrics first (the reference)
    baseline_clearday_metrics = aggregate_strategy_metrics(df, BASELINE_CLEAR_DAY)
    
    # Also get baseline_full for comparison
    baseline_full_metrics = aggregate_strategy_metrics(df, BASELINE_FULL)
    
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
    """Generate comprehensive leaderboard using all available data sources.
    
    Uses main test results for overall mIoU across all strategies,
    and supplements with unified domain gap results for per-domain metrics when available.
    """
    
    print("=" * 70)
    print("Generating Strategy Leaderboard")
    print("=" * 70)
    
    # Load main results (this is the primary source with all strategies)
    main_df = load_results_csv(RESULTS_CSV)
    
    # Try to load unified domain gap results for per-domain metrics
    unified_df = load_unified_domain_results()
    unified_data = {}
    
    if unified_df is not None and 'strategy' in unified_df.columns:
        print(f"Loaded per-domain data for {len(unified_df)} strategies")
        
        # Index unified data by strategy for quick lookup
        for _, row in unified_df.iterrows():
            strategy = row['strategy']
            unified_data[strategy] = {
                'normal_mIoU': row.get('normal_mIoU'),
                'adverse_mIoU': row.get('adverse_mIoU'),
                'domain_gap': row.get('domain_gap'),
                'gap_reduction': row.get('gap_reduction'),
            }
    
    # Generate leaderboard from main results
    print(f"Generating leaderboard from {len(main_df)} test results")
    
    strategies = main_df['strategy'].unique()
    leaderboard_data = []
    
    for strategy in sorted(strategies):
        metrics = aggregate_strategy_metrics(main_df, strategy)
        
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
        
        # Get per-domain metrics from unified data if available
        normal_miou = None
        adverse_miou = None
        domain_gap = None
        gap_reduction = None
        
        if strategy in unified_data:
            ud = unified_data[strategy]
            normal_miou = ud.get('normal_mIoU')
            adverse_miou = ud.get('adverse_mIoU')
            domain_gap = ud.get('domain_gap')
            gap_reduction = ud.get('gap_reduction')
        
        row = {
            'Strategy': strategy,
            'Type': strategy_type,
            'Overall mIoU': round(metrics['overall_miou'], 2) if metrics['overall_miou'] else None,
            'Normal mIoU': round(normal_miou, 2) if normal_miou is not None else None,
            'Adverse mIoU': round(adverse_miou, 2) if adverse_miou is not None else None,
            'Domain Gap (Δ)': round(domain_gap, 2) if domain_gap is not None else None,
            'Gap Reduction': round(gap_reduction, 2) if gap_reduction is not None else None,
            'Num Results': metrics['num_results'],
        }
        
        leaderboard_data.append(row)
    
    leaderboard_df = pd.DataFrame(leaderboard_data)
    
    # Sort by Overall mIoU (descending)
    leaderboard_df = leaderboard_df.sort_values('Overall mIoU', ascending=False)
    
    print(f"Generated leaderboard with {len(leaderboard_df)} strategies")
    
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
    parser.add_argument('--refresh', action='store_true', 
                       help='Re-extract results from WEIGHTS directory instead of using cached CSV')
    parser.add_argument('--weights-root', type=str, default=None,
                       help='Override WEIGHTS directory path')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output during extraction')
    
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Determine data source
    weights_path = Path(args.weights_root) if args.weights_root else WEIGHTS_ROOT
    
    # Load results - either fresh from WEIGHTS or from cached CSV
    print("\n1. Loading test results...")
    if args.refresh or not RESULTS_CSV.exists():
        if TEST_ANALYZER_AVAILABLE:
            print(f"   Extracting fresh results from: {weights_path}")
            df = extract_results_from_weights(weights_path, verbose=args.verbose)
            
            # Optionally save to CSV for caching
            if not df.empty:
                df.to_csv(RESULTS_CSV, index=False)
                print(f"   Cached results to: {RESULTS_CSV}")
        else:
            print("   ERROR: Cannot refresh - test_result_analyzer.py not available")
            if RESULTS_CSV.exists():
                df = load_results_csv(RESULTS_CSV)
            else:
                print("   ERROR: No results available!")
                return
    else:
        print(f"   Loading cached results from: {RESULTS_CSV}")
        df = load_results_csv(RESULTS_CSV)
    
    if df.empty:
        print("   ERROR: No results found!")
        return
    
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
