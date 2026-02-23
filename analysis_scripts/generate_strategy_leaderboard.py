#!/usr/bin/env python3
"""
Unified Strategy Leaderboard Generator

Consolidated script replacing generate_stage1_leaderboard.py and
generate_stage2_leaderboard.py. Generates strategy leaderboards for any
training stage.

Stages:
  1             - Stage 1: Clear day training only, cross-domain testing
  2             - Stage 2: All domains training, full evaluation
  cityscapes-gen - Cityscapes generative augmentation + ACDC cross-domain

Usage:
    python generate_strategy_leaderboard.py --stage 1
    python generate_strategy_leaderboard.py --stage 2
    python generate_strategy_leaderboard.py --stage cityscapes-gen
    python generate_strategy_leaderboard.py --stage all
    python generate_strategy_leaderboard.py --stage 1 --metric aAcc --fair
    python generate_strategy_leaderboard.py --stage 2 --no-refresh --verbose
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime

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

# ============================================================================
# Stage configurations
# ============================================================================

STAGE_CONFIGS = {
    '1': {
        'name': 'Stage 1',
        'description': 'Clear Day Training — cross-domain robustness evaluation',
        'weights_root': '${AWARE_DATA_ROOT}/WEIGHTS',
        'datasets': ['bdd10k', 'iddaw', 'mapillaryvistas', 'outside15k'],
        'models': ['pspnet_r50', 'segformer_mit-b3',
                    'segnext_mscan-b', 'mask2former_swin-b'],
        'exclude_models': ['hrnet_hr48'],
        'cache_csv': 'downstream_results.csv',
        'output_prefix': 'STAGE1',
        'checkpoint_filter': None,  # No checkpoint path filter needed
        'domain_classification': {
            'normal': ['clear_day', 'cloudy'],
            'adverse': ['foggy', 'night', 'rainy', 'snowy'],
            'transition': ['dawn_dusk'],
        },
        'cross_domain_datasets': {},  # No cross-domain testing
    },
    '2': {
        'name': 'Stage 2',
        'description': 'All Domains Training — domain-inclusive evaluation',
        'weights_root': '${AWARE_DATA_ROOT}/WEIGHTS_STAGE_2',
        'datasets': ['bdd10k', 'iddaw', 'mapillaryvistas', 'outside15k'],
        'models': ['pspnet_r50', 'segformer_mit-b3',
                    'segnext_mscan-b', 'mask2former_swin-b'],
        'exclude_models': ['hrnet_hr48'],
        'cache_csv': 'downstream_results_stage2.csv',
        'output_prefix': 'STAGE2',
        'checkpoint_filter': 'WEIGHTS_STAGE_2',  # Verify checkpoint path
        'domain_classification': {
            'normal': ['clear_day'],
            'adverse': ['cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy'],
            'transition': [],
        },
        'cross_domain_datasets': {},
    },
    'cityscapes-gen': {
        'name': 'Cityscapes-Gen',
        'description': 'Cityscapes Generative Augmentation + ACDC cross-domain testing',
        'weights_root': '${AWARE_DATA_ROOT}/WEIGHTS_CITYSCAPES_GEN',
        'datasets': ['cityscapes'],
        'models': ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b3',
                    'segnext_mscan-b', 'hrnet_hr48', 'mask2former_swin-b'],
        'exclude_models': ['hrnet_hr48'],
        'cache_csv': 'downstream_results_cityscapes_gen.csv',
        'output_prefix': 'CITYSCAPES_GEN',
        'checkpoint_filter': 'WEIGHTS_CITYSCAPES_GEN',
        'domain_classification': {
            'normal': ['clear_day'],  # Cityscapes val is clear/day
            'adverse': ['foggy', 'night', 'rainy', 'snowy'],  # ACDC domains
            'transition': [],
        },
        # For Normal/Adverse computation: use dataset-level overall mIoU
        # Cityscapes val = all clear/day → Normal, ACDC = adverse weather → Adverse
        'normal_adverse_from_datasets': {
            'normal': ['cityscapes'],  # Overall Cityscapes mIoU = Normal mIoU
            'adverse': ['acdc'],       # Overall ACDC mIoU = Adverse mIoU
        },
        # ACDC cross-domain results are in test_results_acdc/ instead of test_results_detailed/
        'cross_domain_datasets': {
            'acdc': {
                'test_dir': 'test_results_acdc',
                'display_name': 'ACDC (cross-domain)',
            },
        },
    },
}

# Valid metrics for leaderboard generation
VALID_METRICS = ['mIoU', 'aAcc', 'mAcc', 'fwIoU']
METRIC_DESCRIPTIONS = {
    'mIoU': 'Mean Intersection over Union',
    'aAcc': 'Pixel Accuracy (Overall Accuracy)',
    'mAcc': 'Mean Class Accuracy',
    'fwIoU': 'Frequency-Weighted IoU',
}

# Strategy type mapping
STRATEGY_TYPES = {
    'baseline': 'Baseline',
    'std_photometric_distort': 'Standard Aug',
    'std_cutmix': 'Standard Aug',
    'std_mixup': 'Standard Aug',
    'std_autoaugment': 'Standard Aug',
    'std_randaugment': 'Standard Aug',
    'std_cutout': 'Standard Aug',
    'std_minimal': 'Standard Aug',
}


# ============================================================================
# Result extraction
# ============================================================================

def extract_results_with_analyzer(weights_root: Path, verbose: bool = False) -> pd.DataFrame:
    """Extract test results using TestResultAnalyzer.

    This picks up results from all test_results* directories automatically.
    """
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


def extract_results_rglob(weights_root: Path, checkpoint_filter: Optional[str] = None,
                           verbose: bool = False) -> pd.DataFrame:
    """Extract test results using rglob (fallback when TestResultAnalyzer not available,
    or when checkpoint path verification is needed).

    Args:
        weights_root: Root directory containing strategy/dataset/model structure
        checkpoint_filter: If set, only include results where checkpoint_path contains
                          this string (e.g., 'WEIGHTS_STAGE_2')
        verbose: Print progress details
    """
    weights_path = Path(weights_root)
    model_results = defaultdict(list)

    for results_file in weights_path.rglob("results.json"):
        try:
            with open(results_file) as f:
                data = json.load(f)

            parts = results_file.relative_to(weights_path).parts
            if len(parts) < 5:
                continue

            strategy = parts[0]
            dataset = parts[1]
            model = parts[2]
            test_dir_name = parts[3]  # e.g., test_results_detailed, test_results_acdc
            timestamp = parts[4]

            # Optionally verify checkpoint path
            if checkpoint_filter:
                config = data.get('config', {})
                checkpoint_path = config.get('checkpoint_path', '')
                if checkpoint_filter not in checkpoint_path and checkpoint_path:
                    continue

            # Parse overall metrics
            overall = data.get('overall', {})

            # Parse per-domain metrics
            per_domain = data.get('per_domain', {})
            domain_metrics = {}
            for domain, domain_data in per_domain.items():
                metrics = domain_data.get('summary', domain_data) if isinstance(domain_data, dict) else {}
                domain_metrics[domain] = {
                    'mIoU': metrics.get('mIoU'),
                    'mAcc': metrics.get('mAcc'),
                    'aAcc': metrics.get('aAcc'),
                    'fwIoU': metrics.get('fwIoU'),
                }

            model_key = f"{strategy}/{dataset}/{model}/{test_dir_name}"
            model_results[model_key].append({
                'strategy': strategy,
                'dataset': dataset,
                'model': model,
                'test_type': test_dir_name,
                'result_type': 'detailed',
                'result_dir': str(results_file.parent),
                'timestamp': timestamp,
                'mIoU': overall.get('mIoU'),
                'mAcc': overall.get('mAcc'),
                'aAcc': overall.get('aAcc'),
                'fwIoU': overall.get('fwIoU'),
                'num_images': overall.get('num_images', 0),
                'has_per_domain': bool(per_domain),
                'has_per_class': bool(data.get('per_class')),
                'per_domain_metrics': domain_metrics if domain_metrics else None,
            })
        except Exception as e:
            if verbose:
                print(f"  Error reading {results_file}: {e}")

    # Deduplicate: keep latest per model_key
    results = []
    for model_key, result_list in model_results.items():
        result_list.sort(key=lambda x: x['timestamp'], reverse=True)
        results.append(result_list[0])

    if not results:
        print("Warning: No test results found!")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    print(f"Extracted {len(df)} test results (rglob)")
    return df


def extract_cross_domain_results(weights_root: Path, cross_domain_config: dict,
                                  checkpoint_filter: Optional[str] = None,
                                  verbose: bool = False) -> pd.DataFrame:
    """Extract cross-domain test results (e.g., ACDC results for Cityscapes-Gen).

    Cross-domain results are stored in a different directory (e.g., test_results_acdc/)
    and need to be mapped to a different dataset name.

    Args:
        weights_root: Root directory containing strategy/dataset/model structure
        cross_domain_config: Dict mapping dataset_name -> {test_dir, display_name}
        checkpoint_filter: If set, verify checkpoint_path contains this string
        verbose: Print progress details
    """
    weights_path = Path(weights_root)
    all_results = []

    for cd_dataset, cd_config in cross_domain_config.items():
        test_dir_name = cd_config['test_dir']
        model_results = defaultdict(list)

        # Search for results.json files within the specific test directory
        for results_file in weights_path.rglob(f"{test_dir_name}/*/results.json"):
            try:
                with open(results_file) as f:
                    data = json.load(f)

                parts = results_file.relative_to(weights_path).parts
                if len(parts) < 5:
                    continue

                strategy = parts[0]
                _orig_dataset = parts[1]  # e.g., 'cityscapes' — the training dataset
                model = parts[2]
                timestamp = parts[4]

                # Optionally verify checkpoint path
                if checkpoint_filter:
                    config = data.get('config', {})
                    checkpoint_path = config.get('checkpoint_path', '')
                    if checkpoint_filter not in checkpoint_path and checkpoint_path:
                        continue

                overall = data.get('overall', {})
                per_domain = data.get('per_domain', {})
                domain_metrics = {}
                for domain, domain_data in per_domain.items():
                    metrics = domain_data.get('summary', domain_data) if isinstance(domain_data, dict) else {}
                    domain_metrics[domain] = {
                        'mIoU': metrics.get('mIoU'),
                        'mAcc': metrics.get('mAcc'),
                        'aAcc': metrics.get('aAcc'),
                        'fwIoU': metrics.get('fwIoU'),
                    }

                model_key = f"{strategy}/{cd_dataset}/{model}"
                model_results[model_key].append({
                    'strategy': strategy,
                    'dataset': cd_dataset,  # Use the cross-domain dataset name (e.g., 'acdc')
                    'model': model,
                    'test_type': test_dir_name,
                    'result_type': 'detailed',
                    'result_dir': str(results_file.parent),
                    'timestamp': timestamp,
                    'mIoU': overall.get('mIoU'),
                    'mAcc': overall.get('mAcc'),
                    'aAcc': overall.get('aAcc'),
                    'fwIoU': overall.get('fwIoU'),
                    'num_images': overall.get('num_images', 0),
                    'has_per_domain': bool(per_domain),
                    'has_per_class': bool(data.get('per_class')),
                    'per_domain_metrics': domain_metrics if domain_metrics else None,
                })
            except Exception as e:
                if verbose:
                    print(f"  Error reading {results_file}: {e}")

        # Deduplicate: keep latest per model_key
        for model_key, result_list in model_results.items():
            result_list.sort(key=lambda x: x['timestamp'], reverse=True)
            all_results.append(result_list[0])

        if verbose:
            print(f"  Cross-domain {cd_dataset}: found {len(model_results)} results")

    if not all_results:
        return pd.DataFrame()

    return pd.DataFrame(all_results)


def load_results(stage_config: dict, weights_root: Path,
                 no_refresh: bool = False, verbose: bool = False) -> pd.DataFrame:
    """Load or extract results for a given stage.

    Uses TestResultAnalyzer when available, with rglob as fallback.
    Handles cross-domain datasets and checkpoint path verification.
    """
    cache_csv = PROJECT_ROOT / stage_config['cache_csv']
    checkpoint_filter = stage_config.get('checkpoint_filter')
    cross_domain = stage_config.get('cross_domain_datasets', {})

    if no_refresh and cache_csv.exists():
        print(f"Loading from cache: {cache_csv}")
        df = pd.read_csv(cache_csv)
        return df

    # Extract main results
    if TEST_ANALYZER_AVAILABLE and not checkpoint_filter:
        # Use TestResultAnalyzer for stages without checkpoint verification
        df = extract_results_with_analyzer(weights_root, verbose=verbose)
    else:
        # Use rglob for stages needing checkpoint path verification or when
        # TestResultAnalyzer is not available
        df = extract_results_rglob(weights_root, checkpoint_filter=checkpoint_filter,
                                    verbose=verbose)

    # Extract cross-domain results if configured
    if cross_domain:
        # Remove cross-domain test results from the main DataFrame to avoid
        # double-counting. rglob picks up test_results_acdc/ with dataset=cityscapes,
        # but cross-domain extraction correctly maps them to dataset=acdc.
        cd_test_dirs = {cd_config['test_dir'] for cd_config in cross_domain.values()}
        if not df.empty and 'test_type' in df.columns:
            before = len(df)
            df = df[~df['test_type'].isin(cd_test_dirs)]
            removed = before - len(df)
            if removed > 0:
                print(f"Removed {removed} duplicate cross-domain entries from main results")

        cd_df = extract_cross_domain_results(weights_root, cross_domain,
                                              checkpoint_filter=checkpoint_filter,
                                              verbose=verbose)
        if not cd_df.empty:
            print(f"Found {len(cd_df)} cross-domain test results")
            df = pd.concat([df, cd_df], ignore_index=True)

    # Cache results
    if not df.empty:
        df.to_csv(cache_csv, index=False)
        print(f"Cached to: {cache_csv}")

    # Filter out excluded models
    exclude_models = stage_config.get('exclude_models', [])
    if exclude_models and not df.empty:
        before = len(df)
        df = df[~df['model'].isin(exclude_models)]
        excluded = before - len(df)
        if excluded > 0:
            print(f"Excluded {excluded} results from models: {exclude_models}")

    return df


# ============================================================================
# Analysis helpers
# ============================================================================

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
    """Parse per-domain metrics from a DataFrame row.

    Handles multiple formats:
    - Dict of {domain: {metric: value}} (from per_domain_metrics column)
    - String representation of above (from CSV cache)
    """
    raw = row.get('per_domain_metrics')
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return {}
    if isinstance(raw, str) and raw == '':
        return {}

    try:
        metrics = raw
        if isinstance(metrics, str):
            import ast
            try:
                metrics = ast.literal_eval(metrics)
            except (ValueError, SyntaxError):
                metrics = json.loads(metrics)

        result = {}
        for domain, data in metrics.items():
            if isinstance(data, dict) and metric in data:
                val = data[metric]
                if val is not None:
                    result[domain] = val
        return result
    except Exception:
        return {}


def compute_normal_adverse_metric(domain_metrics: Dict[str, float],
                                   domain_classification: dict) -> Tuple[Optional[float], Optional[float]]:
    """Compute Normal and Adverse metric averages from per-domain metrics.

    Uses stage-specific domain classification.
    """
    normal_domains = domain_classification.get('normal', [])
    adverse_domains = domain_classification.get('adverse', [])

    normal_vals = [domain_metrics[d] for d in normal_domains if d in domain_metrics]
    adverse_vals = [domain_metrics[d] for d in adverse_domains if d in domain_metrics]

    normal_val = np.mean(normal_vals) if normal_vals else None
    adverse_val = np.mean(adverse_vals) if adverse_vals else None

    return normal_val, adverse_val


# ============================================================================
# Leaderboard generation
# ============================================================================

def generate_leaderboard(df: pd.DataFrame, metric: str, domain_classification: dict,
                         normal_adverse_from_datasets: dict = None) -> pd.DataFrame:
    """Generate the main strategy leaderboard.

    Args:
        df: DataFrame with test results
        metric: Which metric to use for ranking ('mIoU', 'aAcc', 'mAcc', 'fwIoU')
        domain_classification: Stage-specific normal/adverse domain mapping
        normal_adverse_from_datasets: Optional dict with 'normal' and 'adverse'
            keys mapping to dataset names. When provided, Normal/Adverse mIoU
            is computed from dataset-level overall metrics instead of per-domain
            breakdown. E.g., {'normal': ['cityscapes'], 'adverse': ['acdc']}.

    Returns:
        DataFrame with strategy leaderboard sorted by selected metric
    """
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in results. Available: {list(df.columns)}")

    baseline_df = df[df['strategy'] == 'baseline']
    baseline_val = baseline_df[metric].mean() if not baseline_df.empty else None

    records = []
    for strategy in sorted(df['strategy'].unique()):
        strat_df = df[df['strategy'] == strategy]

        overall_val = strat_df[metric].mean()
        overall_std = strat_df[metric].std()
        num_results = len(strat_df)

        normal_avg = None
        adverse_avg = None

        if normal_adverse_from_datasets:
            # Compute Normal/Adverse from dataset-level overall metrics
            # e.g., Cityscapes overall mIoU = Normal, ACDC overall mIoU = Adverse
            normal_datasets = normal_adverse_from_datasets.get('normal', [])
            adverse_datasets = normal_adverse_from_datasets.get('adverse', [])

            normal_df = strat_df[strat_df['dataset'].isin(normal_datasets)]
            adverse_df = strat_df[strat_df['dataset'].isin(adverse_datasets)]

            normal_avg = normal_df[metric].mean() if not normal_df.empty else None
            adverse_avg = adverse_df[metric].mean() if not adverse_df.empty else None
        else:
            # Compute Normal/Adverse from per-domain metrics within each test result
            normal_vals = []
            adverse_vals = []

            for _, row in strat_df.iterrows():
                domain_metrics = parse_per_domain_metrics(row, metric)
                if domain_metrics:
                    normal, adverse = compute_normal_adverse_metric(domain_metrics, domain_classification)
                    if normal is not None:
                        normal_vals.append(normal)
                    if adverse is not None:
                        adverse_vals.append(adverse)

            normal_avg = np.mean(normal_vals) if normal_vals else None
            adverse_avg = np.mean(adverse_vals) if adverse_vals else None

        domain_gap = (normal_avg - adverse_avg) if (normal_avg is not None and adverse_avg is not None) else None

        gain = (overall_val - baseline_val) if baseline_val is not None else None

        records.append({
            'Strategy': strategy,
            'Type': get_strategy_type(strategy),
            metric: round(overall_val, 2),
            'Std': round(overall_std, 2) if not np.isnan(overall_std) else 0.0,
            'Gain': round(gain, 2) if gain is not None else None,
            f'Normal {metric}': round(normal_avg, 2) if normal_avg is not None else None,
            f'Adverse {metric}': round(adverse_avg, 2) if adverse_avg is not None else None,
            'Domain Gap': round(domain_gap, 2) if domain_gap is not None else None,
            'Num Tests': num_results,
        })

    result_df = pd.DataFrame(records)
    result_df = result_df.sort_values(metric, ascending=False).reset_index(drop=True)
    return result_df


def generate_per_dataset_table(df: pd.DataFrame, metric: str, datasets: list) -> pd.DataFrame:
    """Generate per-dataset metric breakdown.

    Args:
        df: DataFrame with test results
        metric: Which metric to use
        datasets: List of dataset names for this stage
    """
    baseline_df = df[df['strategy'] == 'baseline']
    baseline_by_ds = baseline_df.groupby('dataset')[metric].mean().to_dict()

    records = []
    for strategy in sorted(df['strategy'].unique()):
        strat_df = df[df['strategy'] == strategy]
        row = {'Strategy': strategy, 'Type': get_strategy_type(strategy)}

        for dataset in datasets:
            ds_df = strat_df[strat_df['dataset'] == dataset]
            if not ds_df.empty:
                val = ds_df[metric].mean()
                baseline = baseline_by_ds.get(dataset)
                gain = (val - baseline) if baseline is not None else None
                row[dataset] = f"{val:.2f}"
                row[f"{dataset}_gain"] = f"{gain:+.2f}" if gain is not None else "-"
            else:
                row[dataset] = "-"
                row[f"{dataset}_gain"] = "-"

        records.append(row)

    result_df = pd.DataFrame(records)

    # Sort by average metric across available datasets
    for ds in datasets:
        result_df[f'{ds}_num'] = pd.to_numeric(result_df[ds], errors='coerce')
    num_cols = [f'{ds}_num' for ds in datasets]
    result_df['avg'] = result_df[num_cols].mean(axis=1)
    result_df = result_df.sort_values('avg', ascending=False).reset_index(drop=True)
    result_df = result_df.drop(columns=num_cols + ['avg'])

    return result_df


def generate_per_domain_table(df: pd.DataFrame, metric: str,
                               domain_classification: dict) -> pd.DataFrame:
    """Generate per-domain metric breakdown.

    Args:
        df: DataFrame with test results
        metric: Which metric to use
        domain_classification: Stage-specific domain classification
    """
    normal_domains = domain_classification.get('normal', [])
    adverse_domains = domain_classification.get('adverse', [])
    transition_domains = domain_classification.get('transition', [])
    all_domains = normal_domains + transition_domains + adverse_domains

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

        for domain in all_domains:
            if domain in domain_data[strategy]:
                val = np.mean(domain_data[strategy][domain])
                row[domain] = f"{val:.2f}"

                if domain in normal_domains:
                    normal_vals.append(val)
                elif domain in adverse_domains:
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

    return pd.DataFrame(records)


def generate_per_model_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Generate per-model metric breakdown.

    Shows each strategy's performance on each base model architecture,
    normalizing model names by stripping ratio suffixes (e.g. _ratio0p50).

    Args:
        df: DataFrame with test results
        metric: Which metric to use
    """
    df = df.copy()
    df['model_base'] = df['model'].str.replace(r'_ratio\d+p\d+$', '', regex=True)

    base_models = sorted(df['model_base'].unique())

    baseline_df = df[df['strategy'] == 'baseline']
    baseline_by_model = baseline_df.groupby('model_base')[metric].mean().to_dict()

    records = []
    for strategy in sorted(df['strategy'].unique()):
        strat_df = df[df['strategy'] == strategy]
        row = {'Strategy': strategy, 'Type': get_strategy_type(strategy)}

        for model in base_models:
            model_df = strat_df[strat_df['model_base'] == model]
            if not model_df.empty:
                val = model_df[metric].mean()
                baseline = baseline_by_model.get(model)
                gain = (val - baseline) if baseline is not None else None
                row[model] = f"{val:.2f}"
                row[f"{model}_gain"] = f"{gain:+.2f}" if gain is not None else "-"
            else:
                row[model] = "-"
                row[f"{model}_gain"] = "-"

        records.append(row)

    result_df = pd.DataFrame(records)

    # Sort by average metric across available models
    for m in base_models:
        result_df[f'{m}_num'] = pd.to_numeric(result_df[m], errors='coerce')
    num_cols = [f'{m}_num' for m in base_models]
    result_df['avg'] = result_df[num_cols].mean(axis=1)
    result_df = result_df.sort_values('avg', ascending=False).reset_index(drop=True)
    result_df = result_df.drop(columns=num_cols + ['avg'])

    return result_df


# ============================================================================
# Fair comparison filtering
# ============================================================================

def filter_to_complete_configs(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Filter to only include dataset+model configs where ALL strategies have results.

    Ensures fair comparison by requiring equal coverage across all strategies.
    """
    all_strategies = set(df['strategy'].unique())

    df = df.copy()
    df['model_normalized'] = df['model'].str.replace(r'_ratio\d+p\d+$', '', regex=True)

    configs = df.groupby(['dataset', 'model_normalized']).apply(
        lambda x: set(x['strategy'].unique())
    ).to_dict()

    complete_configs = [
        config for config, strategies in configs.items()
        if strategies == all_strategies
    ]

    print(f"\n{'='*70}")
    print("FAIR COMPARISON MODE - Filtering to Complete Configurations")
    print(f"{'='*70}")
    print(f"Total strategies: {len(all_strategies)}")
    print(f"Total configurations: {len(configs)}")
    print(f"Complete configurations: {len(complete_configs)}")

    if complete_configs:
        print("\nUsing configurations:")
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
        return pd.DataFrame()

    filtered_df = df[df.apply(lambda r: (r['dataset'], r['model_normalized']) in complete_configs, axis=1)]
    filtered_df = filtered_df.drop(columns=['model_normalized'])

    print(f"Filtered {len(df)} → {len(filtered_df)} test results")
    print(f"Each strategy has exactly {len(complete_configs)} test results\n")

    return filtered_df


# ============================================================================
# Markdown formatting
# ============================================================================

def format_markdown_table(df: pd.DataFrame, title: str, description: str = "") -> str:
    """Format a DataFrame as a markdown table section."""
    lines = [f"## {title}", ""]
    if description:
        lines.append(description)
        lines.append("")

    cols = df.columns.tolist()
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, row in df.iterrows():
        values = [str(row[c]) if pd.notna(row[c]) else "-" for c in cols]
        lines.append("| " + " | ".join(values) + " |")

    lines.append("")
    return "\n".join(lines)


def generate_output(df: pd.DataFrame, leaderboard_df: pd.DataFrame,
                    per_dataset_df: pd.DataFrame, per_domain_df: pd.DataFrame,
                    stage_config: dict, metric: str, fair: bool,
                    output_dir: Path, per_model_df: pd.DataFrame = None) -> None:
    """Generate all output files (markdown, CSV).

    Args:
        df: Full results DataFrame
        leaderboard_df: Main leaderboard DataFrame
        per_dataset_df: Per-dataset breakdown DataFrame
        per_domain_df: Per-domain breakdown DataFrame
        stage_config: Stage configuration dict
        metric: Selected metric
        fair: Whether fair comparison mode was used
        output_dir: Output directory path
        per_model_df: Per-model breakdown DataFrame (optional)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_name = stage_config['name']
    prefix = stage_config['output_prefix']
    datasets = stage_config['datasets']
    cross_domain = stage_config.get('cross_domain_datasets', {})

    # Include cross-domain datasets in the dataset list for output
    all_datasets = list(datasets)
    for cd_name in cross_domain:
        if cd_name not in all_datasets:
            all_datasets.append(cd_name)

    # Get baseline info
    baseline_row = leaderboard_df[leaderboard_df['Strategy'] == 'baseline']
    baseline_val = baseline_row.iloc[0][metric] if not baseline_row.empty else None

    mode_suffix = '_FAIR' if fair else ''

    # --- Main leaderboard markdown ---
    output_file = output_dir / f'STRATEGY_LEADERBOARD_{prefix}_{metric.upper()}{mode_suffix}.md'
    with open(output_file, 'w') as f:
        title_mode = " (Fair Comparison)" if fair else ""
        f.write(f"# {stage_name} Strategy Leaderboard{title_mode} (by {metric})\n\n")
        f.write(f"**{stage_name}**: {stage_config['description']}\n\n")
        f.write(f"**Metric**: {metric} ({METRIC_DESCRIPTIONS.get(metric, '')})\n\n")

        if fair:
            f.write("**Fair Comparison Mode**: Only includes dataset+model configurations "
                    "where ALL strategies have test results.\n")
            f.write("This ensures equal coverage and prevents incomplete results from "
                    "skewing rankings.\n\n")

        f.write(f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        if baseline_val is not None:
            f.write(f"**Baseline {metric}**: {baseline_val:.2f}%\n")
        f.write(f"**Total Results**: {len(df)} test results from "
                f"{len(df['strategy'].unique())} strategies\n\n")
        f.write("---\n\n")

        # Main leaderboard table
        f.write(format_markdown_table(
            leaderboard_df,
            "Overall Strategy Ranking",
            f"Sorted by {metric}. Gain = improvement over baseline. "
            f"Domain Gap = Normal {metric} - Adverse {metric} (lower is better)."
        ))
        f.write("---\n\n")

        # Per-dataset breakdown
        f.write(format_markdown_table(
            per_dataset_df,
            "Per-Dataset Breakdown",
            f"{metric} performance on each dataset."
        ))
        f.write("---\n\n")

        # Per-domain breakdown
        domain_desc = _domain_description(stage_config)
        f.write(format_markdown_table(
            per_domain_df,
            "Per-Domain Breakdown",
            f"{metric} performance on each weather domain. {domain_desc}"
        ))
        f.write("---\n\n")

        # Per-model breakdown
        if per_model_df is not None and not per_model_df.empty:
            f.write(format_markdown_table(
                per_model_df,
                "Per-Model Breakdown",
                f"{metric} performance on each model architecture. "
                f"Gain columns show improvement over baseline per model."
            ))

    print(f"Saved to: {output_file}")

    # --- Breakdowns subfolder ---
    breakdowns_dir = output_dir / 'breakdowns'
    breakdowns_dir.mkdir(parents=True, exist_ok=True)

    # --- Detailed gains markdown ---
    detailed_file = breakdowns_dir / f'DETAILED_GAINS_{prefix}_{metric.upper()}.md'
    _write_detailed_gains(detailed_file, df, leaderboard_df, per_domain_df,
                          stage_config, metric, all_datasets)
    print(f"Saved to: {detailed_file}")

    # --- CSV files (in breakdowns subfolder) ---
    suffix = f"_{metric.lower()}" if metric != 'mIoU' else ""
    leaderboard_df.to_csv(breakdowns_dir / f'strategy_leaderboard_{prefix.lower()}{suffix}.csv', index=False)
    per_dataset_df.to_csv(breakdowns_dir / f'per_dataset_breakdown_{prefix.lower()}{suffix}.csv', index=False)
    per_domain_df.to_csv(breakdowns_dir / f'per_domain_breakdown_{prefix.lower()}{suffix}.csv', index=False)
    if per_model_df is not None and not per_model_df.empty:
        per_model_df.to_csv(breakdowns_dir / f'per_model_breakdown_{prefix.lower()}{suffix}.csv', index=False)
    print(f"Breakdowns saved to: {breakdowns_dir}")


def _domain_description(stage_config: dict) -> str:
    """Build a human-readable domain description for markdown output."""
    dc = stage_config['domain_classification']
    normal = dc.get('normal', [])
    adverse = dc.get('adverse', [])
    parts = []
    if normal:
        parts.append(f"Normal = {', '.join(normal)}")
    if adverse:
        parts.append(f"Adverse = {', '.join(adverse)}")
    return '. '.join(parts) + '.' if parts else ''


def _write_detailed_gains(filepath: Path, df: pd.DataFrame, leaderboard_df: pd.DataFrame,
                           per_domain_df: pd.DataFrame, stage_config: dict,
                           metric: str, datasets: list) -> None:
    """Write detailed per-dataset and per-domain analysis."""
    stage_name = stage_config['name']

    baseline_df_full = df[df['strategy'] == 'baseline']
    baseline_by_ds = baseline_df_full.groupby('dataset')[metric].mean().to_dict()

    with open(filepath, 'w') as f:
        f.write(f"# {stage_name} Detailed Per-Dataset and Per-Domain Analysis ({metric})\n\n")
        f.write(f"## Per-Dataset {metric} by Strategy\n\n")

        # Header
        f.write("| Strategy | Type |")
        for ds in datasets:
            f.write(f" {ds} | Δ{ds} |")
        f.write(" Avg |\n")

        f.write("|---|---|")
        for _ in datasets:
            f.write("---:|---:|")
        f.write("---:|\n")

        # Rows
        for _, row in leaderboard_df.iterrows():
            strategy = row['Strategy']
            strat_df = df[df['strategy'] == strategy]

            f.write(f"| {strategy} | {row['Type']} |")
            gains = []
            for ds in datasets:
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


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Strategy Leaderboard Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  1              Stage 1: Clear day training, cross-domain robustness
  2              Stage 2: All domains training, inclusive evaluation
  cityscapes-gen Cityscapes generative augmentation + ACDC cross-domain

Metrics:
  mIoU  - Mean Intersection over Union (default)
  aAcc  - Pixel Accuracy (overall accuracy)
  mAcc  - Mean Class Accuracy
  fwIoU - Frequency-weighted IoU

Examples:
  %(prog)s --stage 1                           # Stage 1 mIoU leaderboard
  %(prog)s --stage 2 --metric aAcc             # Stage 2 pixel accuracy
  %(prog)s --stage cityscapes-gen              # Cityscapes-Gen leaderboard
  %(prog)s --stage 1 --fair --verbose          # Fair comparison mode
  %(prog)s --stage 2 --no-refresh              # Use cached results
"""
    )
    parser.add_argument('--stage', required=True, choices=['1', '2', 'cityscapes-gen', 'all'],
                        help='Training stage to generate leaderboard for (all=run all stages)')
    parser.add_argument('--metric', type=str, default='mIoU', choices=VALID_METRICS,
                        help='Metric for ranking (default: mIoU)')
    parser.add_argument('--fair', action='store_true',
                        help='Only use configurations with complete strategy coverage')
    parser.add_argument('--no-refresh', action='store_true',
                        help='Use cached results instead of re-extracting')
    parser.add_argument('--weights-root', type=str, default=None,
                        help='Override the weights directory path')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory (default: result_figures/leaderboard)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    all_stages = ['1', '2', 'cityscapes-gen']
    stages = all_stages if args.stage == 'all' else [args.stage]
    
    for stage in stages:
        if len(stages) > 1:
            print(f"\n{'#' * 70}")
            print(f"# STAGE {stage.upper()}")
            print(f"{'#' * 70}")
        run_stage(stage, args)


def run_stage(stage, args):
    """Run leaderboard generation for a single stage."""
    metric = args.metric

    # Resolve stage config
    stage_config = STAGE_CONFIGS[stage]
    weights_root = Path(args.weights_root) if args.weights_root else Path(stage_config['weights_root'])
    output_dir = Path(args.output_dir) if args.output_dir else (PROJECT_ROOT / 'result_figures' / 'leaderboard')

    # Banner
    print("\n" + "=" * 70)
    print(f"STRATEGY LEADERBOARD GENERATOR — {stage_config['name']}")
    print("=" * 70)
    print(f"Description : {stage_config['description']}")
    print(f"Weights root: {weights_root}")
    print(f"Metric      : {metric} ({METRIC_DESCRIPTIONS.get(metric, '')})")
    if stage_config.get('cross_domain_datasets'):
        cd_names = ', '.join(stage_config['cross_domain_datasets'].keys())
        print(f"Cross-domain: {cd_names}")
    print("=" * 70)

    # Load data
    df = load_results(stage_config, weights_root,
                      no_refresh=args.no_refresh, verbose=args.verbose)

    if df.empty:
        print("ERROR: No results found!")
        return

    print(f"\nLoaded {len(df)} test results")
    print(f"Strategies: {sorted(df['strategy'].unique())}")
    print(f"Datasets  : {sorted(df['dataset'].unique())}")
    if 'model' in df.columns:
        print(f"Models    : {sorted(df['model'].unique())}")

    # Fair filtering
    if args.fair:
        df = filter_to_complete_configs(df, verbose=args.verbose)
        if df.empty:
            return

    # Determine datasets list (include cross-domain datasets present in results)
    all_datasets = list(stage_config['datasets'])
    cross_domain = stage_config.get('cross_domain_datasets', {})
    for cd_name in cross_domain:
        if cd_name in df['dataset'].values and cd_name not in all_datasets:
            all_datasets.append(cd_name)

    domain_classification = stage_config['domain_classification']
    normal_adverse_from_datasets = stage_config.get('normal_adverse_from_datasets')

    # Generate tables
    print(f"\n1. Generating main leaderboard by {metric}...")
    leaderboard_df = generate_leaderboard(df, metric, domain_classification,
                                          normal_adverse_from_datasets=normal_adverse_from_datasets)

    print("\n" + "=" * 70)
    print(f"STRATEGY LEADERBOARD ({stage_config['name']}) — Ranked by {metric}")
    print("=" * 70)
    print(leaderboard_df.to_string(index=False))

    # Baseline summary
    baseline_row = leaderboard_df[leaderboard_df['Strategy'] == 'baseline']
    if not baseline_row.empty:
        baseline_val = baseline_row.iloc[0][metric]
        print(f"\nBaseline {metric}: {baseline_val}%")
        strategies_above = len(leaderboard_df[leaderboard_df[metric] > baseline_val])
        print(f"Strategies beating baseline: {strategies_above}/{len(leaderboard_df) - 1}")

    print(f"\n2. Generating per-dataset breakdown by {metric}...")
    per_dataset_df = generate_per_dataset_table(df, metric, all_datasets)

    print(f"\n3. Generating per-domain breakdown by {metric}...")
    per_domain_df = generate_per_domain_table(df, metric, domain_classification)

    print(f"\n4. Generating per-model breakdown by {metric}...")
    per_model_df = generate_per_model_table(df, metric)

    print("\n" + "=" * 70)
    print(f"PER-MODEL BREAKDOWN ({stage_config['name']}) — {metric}")
    print("=" * 70)
    print(per_model_df.to_string(index=False))

    # Write output
    generate_output(df, leaderboard_df, per_dataset_df, per_domain_df,
                    stage_config, metric, args.fair, output_dir,
                    per_model_df=per_model_df)


if __name__ == '__main__':
    main()
