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
def _normalize_domain_name(name: str) -> Optional[str]:
    """Normalize various domain labels to canonical WEATHER_DOMAINS."""
    if not name:
        return None
    s = name.strip().lower().replace('-', '_').replace(' ', '_')
    # direct matches
    if s in WEATHER_DOMAINS:
        return s
    # synonyms and substrings
    if 'clear' in s or ('day' in s and 'night' not in s):
        return 'clear_day'
    if 'cloud' in s:
        return 'cloudy'
    if 'dawn' in s or 'dusk' in s or 'twilight' in s or 'sunrise' in s or 'sunset' in s:
        return 'dawn_dusk'
    if 'fog' in s or 'mist' in s:
        return 'foggy'
    if 'night' in s or 'dark' in s:
        return 'night'
    if 'rain' in s or 'wet' in s:
        return 'rainy'
    if 'snow' in s or 'ice' in s:
        return 'snowy'
    return None

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
# Note: baseline_clear_day is NOT a separate strategy label; it is baseline with dataset suffix "_cd".
# OLD: WEIGHTS/strategy/dataset/model_clear_day/...
# NEW: WEIGHTS/strategy/dataset_cd/model/...
BASELINE_CLEAR_DAY_LABEL = 'baseline_clear_day'  # For display/reference only
BASELINE_FULL = 'baseline'  # Full baseline (trained on all conditions)
CLEAR_DAY_DATASET_SUFFIX = '_cd'  # Dataset suffix for clear_day trained models


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


def _find_latest_detailed_run_dir(detailed_root: Path) -> Optional[Path]:
    """Find the latest timestamped subdirectory under test_results_detailed."""
    if not detailed_root.exists():
        return None
    subdirs = [p for p in detailed_root.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    # Timestamps are lexically sortable (YYYYMMDD_HHMMSS)
    return sorted(subdirs)[-1]

def _read_per_domain_metrics_from_run(run_dir: Path) -> Optional[Dict[str, float]]:
    """Read per-domain metrics from JSON/CSV/test_report.txt inside a detailed run directory."""
    # Try JSON first
    json_file = run_dir / 'metrics_per_domain.json'
    if json_file.exists():
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            # Newer format stores metrics under 'per_domain'
            if isinstance(data, dict) and 'per_domain' in data:
                per_domain = data['per_domain']
            else:
                per_domain = data
            parsed = {}
            if isinstance(per_domain, dict):
                for k, v in per_domain.items():
                    dom = _normalize_domain_name(k)
                    if not dom:
                        continue
                    if isinstance(v, dict):
                        if 'mIoU' in v:
                            parsed[dom] = float(v['mIoU'])
                        elif 'miou' in v:
                            parsed[dom] = float(v['miou'])
                        elif 'mIoU (%)' in v:
                            parsed[dom] = float(v['mIoU (%)'])
                        else:
                            # Try any float value present
                            for subk, subv in v.items():
                                if isinstance(subv, (int, float)):
                                    parsed[dom] = float(subv)
                                    break
                    elif isinstance(v, (int, float)):
                        parsed[dom] = float(v)
            return parsed if parsed else None
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    # Try unified results.json produced by fine_grained_test.py
    results_json = run_dir / 'results.json'
    if results_json.exists():
        try:
            with open(results_json, 'r') as f:
                data = json.load(f)
            parsed = {}
            # Expect structure: { 'per_domain': { domain: { 'summary': { 'mIoU': ... }, ... } } }
            per_domain = data.get('per_domain', {}) if isinstance(data, dict) else {}
            if isinstance(per_domain, dict):
                for k, v in per_domain.items():
                    dom = _normalize_domain_name(k)
                    if not dom:
                        continue
                    miou_val = None
                    if isinstance(v, dict):
                        summary = v.get('summary') if 'summary' in v else v
                        if isinstance(summary, dict):
                            if 'mIoU' in summary:
                                miou_val = summary['mIoU']
                            elif 'miou' in summary:
                                miou_val = summary['miou']
                            elif 'mIoU (%)' in summary:
                                miou_val = summary['mIoU (%)']
                    if isinstance(miou_val, (int, float)):
                        parsed[dom] = float(miou_val)
            return parsed if parsed else None
        except Exception as e:
            print(f"Error reading {results_json}: {e}")

    # Try CSV fallback
    csv_file = run_dir / 'per_domain_metrics.csv'
    if csv_file.exists():
        try:
            df = pd.read_csv(csv_file)
            # Expect columns like 'domain' and 'mIoU'
            domain_col = 'domain' if 'domain' in df.columns else 'Domain'
            miou_col = 'mIoU' if 'mIoU' in df.columns else 'miou'
            if domain_col in df.columns and miou_col in df.columns:
                out = {}
                for _, row in df.iterrows():
                    dom = _normalize_domain_name(str(row[domain_col]))
                    if dom:
                        try:
                            out[dom] = float(row[miou_col])
                        except Exception:
                            pass
                return out if out else None
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    # Try parsing test_report.txt
    report_file = run_dir / 'test_report.txt'
    if report_file.exists():
        try:
            with open(report_file, 'r') as f:
                content = f.read()
            metrics = {}
            for line in content.split('\n'):
                lower = line.lower()
                if 'miou' in lower:
                    for domain in WEATHER_DOMAINS:
                        if domain in lower:
                            # Extract the first float in the line
                            tokens = line.replace('%', '').replace(',', ' ').split()
                            floats = []
                            for t in tokens:
                                try:
                                    floats.append(float(t))
                                except:
                                    pass
                            # Heuristic: last number is mIoU or the only number present
                            if floats:
                                dom = _normalize_domain_name(domain)
                                if dom:
                                    metrics[dom] = floats[-1]
                            break
            return metrics if metrics else None
        except Exception as e:
            print(f"Error parsing {report_file}: {e}")

    return None

def load_per_domain_results(weights_root: Path, strategy: str, scope: str = 'full') -> Dict:
    """Load per-domain results by locating latest detailed run and reading metrics.

    Returns a dict keyed by (dataset, model) -> per-domain metrics dict.
    
    Args:
        weights_root: Path to WEIGHTS directory
        strategy: Strategy name
        scope: 'full' to exclude _cd datasets, 'clear_day' to only include _cd datasets
    """
    results = {}
    strategy_dir = weights_root / strategy

    if not strategy_dir.exists():
        return results

    for dataset_dir in strategy_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        
        # Filter based on scope
        is_cd_dataset = dataset.endswith(CLEAR_DAY_DATASET_SUFFIX)
        if scope == 'clear_day' and not is_cd_dataset:
            continue
        if scope == 'full' and is_cd_dataset:
            continue

        for model_dir in dataset_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model = model_dir.name

            detailed_root = model_dir / 'test_results_detailed'
            run_dir = _find_latest_detailed_run_dir(detailed_root)
            if run_dir is None:
                continue
            domain_metrics = _read_per_domain_metrics_from_run(run_dir)
            if domain_metrics:
                results[(dataset, model)] = domain_metrics

    return results


def aggregate_normal_adverse(domain_metrics: Dict) -> Tuple[Optional[float], Optional[float]]:
    """Aggregate normal and adverse mIoU from a per-domain metrics dict.

    The metrics in JSON are expected to be percentages (e.g., 45.8). If some domains
    are missing for a dataset, averages are computed over available domains in that group.
    """
    normal_vals = []
    adverse_vals = []

    for d in NORMAL_DOMAINS:
        if d in domain_metrics and isinstance(domain_metrics[d], (int, float)):
            normal_vals.append(domain_metrics[d])

    for d in ADVERSE_DOMAINS:
        if d in domain_metrics and isinstance(domain_metrics[d], (int, float)):
            adverse_vals.append(domain_metrics[d])

    normal_miou = float(np.mean(normal_vals)) if normal_vals else None
    adverse_miou = float(np.mean(adverse_vals)) if adverse_vals else None

    return normal_miou, adverse_miou


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
    """Calculate aggregated metrics for a strategy across all datasets and models.
    
    Excludes _cd suffix datasets (those are for clear_day trained models).
    """
    
    strategy_df = df[df['strategy'] == strategy]
    
    if strategy_df.empty:
        return None
    
    # Filter for standard models and exclude _cd suffix datasets
    strategy_df = strategy_df[strategy_df['model'].isin(MODELS)]
    strategy_df = strategy_df[~strategy_df['dataset'].str.endswith(CLEAR_DAY_DATASET_SUFFIX)]
    
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

def aggregate_strategy_metrics_for_models(df: pd.DataFrame, strategy: str, model_names: List[str]) -> Optional[Dict]:
    """Aggregate metrics for a strategy restricted to specific model names."""
    strategy_df = df[(df['strategy'] == strategy) & (df['model'].isin(model_names))]
    if strategy_df.empty:
        return None
    return {
        'strategy': strategy,
        'overall_miou': strategy_df['mIoU'].mean(),
        'overall_fwiou': strategy_df['fwIoU'].mean() if 'fwIoU' in strategy_df.columns else None,
        'num_results': len(strategy_df),
        'datasets': list(strategy_df['dataset'].unique()),
        'models': list(strategy_df['model'].unique()),
    }


def aggregate_strategy_metrics_for_clearday(df: pd.DataFrame, strategy: str) -> Optional[Dict]:
    """Aggregate metrics for a strategy restricted to _cd suffix datasets.
    
    NEW: This replaces the old model-based filtering with dataset-based filtering.
    """
    # Filter for _cd suffix datasets and standard models
    strategy_df = df[(df['strategy'] == strategy) & 
                     (df['dataset'].str.endswith(CLEAR_DAY_DATASET_SUFFIX)) &
                     (df['model'].isin(MODELS))]
    if strategy_df.empty:
        return None
    return {
        'strategy': strategy,
        'overall_miou': strategy_df['mIoU'].mean(),
        'overall_fwiou': strategy_df['fwIoU'].mean() if 'fwIoU' in strategy_df.columns else None,
        'num_results': len(strategy_df),
        'datasets': list(strategy_df['dataset'].unique()),
        'models': list(strategy_df['model'].unique()),
    }


def compute_strategy_domain_aggregates(weights_root: Path, strategy: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute aggregated Normal mIoU, Adverse mIoU, and Domain Gap for a strategy.

    Scans WEIGHTS/{strategy}/{dataset}/{model}/test_results_detailed/metrics_per_domain.json
    Only includes standard models defined in MODELS.
    Excludes _cd suffix datasets (those are for clear_day trained models).
    """
    per_config_normals = []
    per_config_adverses = []

    # Pass scope='full' to exclude _cd datasets
    per_domain_all = load_per_domain_results(weights_root, strategy, scope='full')
    for (dataset, model), domain_metrics in per_domain_all.items():
        if model not in MODELS:
            # Skip non-standard models
            continue
        normal_miou, adverse_miou = aggregate_normal_adverse(domain_metrics)
        if normal_miou is not None and adverse_miou is not None:
            per_config_normals.append(normal_miou)
            per_config_adverses.append(adverse_miou)

    if not per_config_normals or not per_config_adverses:
        return None, None, None

    normal_avg = float(np.mean(per_config_normals))
    adverse_avg = float(np.mean(per_config_adverses))
    gap = calculate_domain_gap(normal_avg, adverse_avg)
    return normal_avg, adverse_avg, gap

def compute_baseline_clearday_domain_gap(weights_root: Path) -> Optional[float]:
    """Compute domain gap for baseline clear_day models.

    Scans WEIGHTS/baseline/{dataset}_cd/{model}/test_results_detailed/metrics_per_domain.json
    (NEW: dataset-level suffix instead of model-level suffix)
    """
    strategy = BASELINE_FULL
    per_config_normals = []
    per_config_adverses = []

    strategy_dir = weights_root / strategy
    if not strategy_dir.exists():
        return None

    for dataset_dir in strategy_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        # Only process _cd suffixed datasets for clear_day models
        if not dataset.endswith(CLEAR_DAY_DATASET_SUFFIX):
            continue

        for model in MODELS:
            model_dir = dataset_dir / model
            detailed_root = model_dir / 'test_results_detailed'
            run_dir = _find_latest_detailed_run_dir(detailed_root)
            if run_dir is None:
                continue
            domain_metrics = _read_per_domain_metrics_from_run(run_dir)
            if not domain_metrics:
                continue
            # Normalize keys to lower-case
            domain_metrics = {k.lower(): v for k, v in domain_metrics.items()}
            normal_miou, adverse_miou = aggregate_normal_adverse(domain_metrics)
            if normal_miou is not None and adverse_miou is not None:
                per_config_normals.append(normal_miou)
                per_config_adverses.append(adverse_miou)

    if not per_config_normals or not per_config_adverses:
        return None

    normal_avg = float(np.mean(per_config_normals))
    adverse_avg = float(np.mean(per_config_adverses))
    return calculate_domain_gap(normal_avg, adverse_avg)

def compute_baseline_clearday_domain_aggregates(weights_root: Path) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute Normal mIoU, Adverse mIoU, and Gap for baseline clear_day models.

    Scans WEIGHTS/baseline/{dataset}_cd/{model}/test_results_detailed/metrics_per_domain.json
    (NEW: dataset-level suffix instead of model-level suffix)
    and aggregates across standard models.
    """
    strategy = BASELINE_FULL
    per_config_normals = []
    per_config_adverses = []

    strategy_dir = weights_root / strategy
    if not strategy_dir.exists():
        return None, None, None

    for dataset_dir in strategy_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        # Only process _cd suffixed datasets for clear_day models
        if not dataset_dir.name.endswith(CLEAR_DAY_DATASET_SUFFIX):
            continue
        for model in MODELS:
            model_dir = dataset_dir / model
            detailed_root = model_dir / 'test_results_detailed'
            run_dir = _find_latest_detailed_run_dir(detailed_root)
            if run_dir is None:
                continue
            domain_metrics = _read_per_domain_metrics_from_run(run_dir)
            if not domain_metrics:
                continue
            domain_metrics = {k.lower(): v for k, v in domain_metrics.items()}
            normal_miou, adverse_miou = aggregate_normal_adverse(domain_metrics)
            if normal_miou is not None and adverse_miou is not None:
                per_config_normals.append(normal_miou)
                per_config_adverses.append(adverse_miou)

    if not per_config_normals or not per_config_adverses:
        return None, None, None

    normal_avg = float(np.mean(per_config_normals))
    adverse_avg = float(np.mean(per_config_adverses))
    gap = calculate_domain_gap(normal_avg, adverse_avg)
    return normal_avg, adverse_avg, gap

def compute_strategy_clearday_domain_aggregates(weights_root: Path, strategy: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute Normal/Adverse/Gap for a strategy's clear_day-trained models.

    Scans WEIGHTS/{strategy}/{dataset}_cd/{model}/...
    (NEW: dataset-level suffix instead of model-level suffix)
    """
    per_config_normals = []
    per_config_adverses = []

    strategy_dir = weights_root / strategy
    if not strategy_dir.exists():
        return None, None, None

    for dataset_dir in strategy_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        # Only process _cd suffixed datasets for clear_day models
        if not dataset_dir.name.endswith(CLEAR_DAY_DATASET_SUFFIX):
            continue
        for model in MODELS:
            model_dir = dataset_dir / model
            detailed_root = model_dir / 'test_results_detailed'
            run_dir = _find_latest_detailed_run_dir(detailed_root)
            if run_dir is None:
                continue
            domain_metrics = _read_per_domain_metrics_from_run(run_dir)
            if not domain_metrics:
                continue
            domain_metrics = {k.lower(): v for k, v in domain_metrics.items()}
            normal_miou, adverse_miou = aggregate_normal_adverse(domain_metrics)
            if normal_miou is not None and adverse_miou is not None:
                per_config_normals.append(normal_miou)
                per_config_adverses.append(adverse_miou)

    if not per_config_normals or not per_config_adverses:
        return None, None, None

    normal_avg = float(np.mean(per_config_normals))
    adverse_avg = float(np.mean(per_config_adverses))
    gap = calculate_domain_gap(normal_avg, adverse_avg)
    return normal_avg, adverse_avg, gap

def audit_per_domain_coverage(weights_root: Path) -> pd.DataFrame:
    """Audit which domains are present in detailed results per strategy/dataset/model.

    Produces columns: strategy, dataset, model, has_clear_day, has_cloudy, has_dawn_dusk,
    has_foggy, has_night, has_rainy, has_snowy, normal_domains_present, adverse_domains_present,
    total_domains_present.
    """
    records = []
    weights_path = Path(weights_root)
    for strategy_dir in weights_path.iterdir():
        if not strategy_dir.is_dir():
            continue
        strategy = strategy_dir.name
        for dataset_dir in strategy_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name
            for model_dir in dataset_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model = model_dir.name
                detailed_root = model_dir / 'test_results_detailed'
                run_dir = _find_latest_detailed_run_dir(detailed_root)
                if run_dir is None:
                    continue
                metrics = _read_per_domain_metrics_from_run(run_dir)
                domains_present = set(metrics.keys()) if metrics else set()
                rec = {
                    'strategy': strategy,
                    'dataset': dataset,
                    'model': model,
                    'has_clear_day': 'clear_day' in domains_present,
                    'has_cloudy': 'cloudy' in domains_present,
                    'has_dawn_dusk': 'dawn_dusk' in domains_present,
                    'has_foggy': 'foggy' in domains_present,
                    'has_night': 'night' in domains_present,
                    'has_rainy': 'rainy' in domains_present,
                    'has_snowy': 'snowy' in domains_present,
                }
                rec['normal_domains_present'] = int(rec['has_clear_day']) + int(rec['has_cloudy'])
                rec['adverse_domains_present'] = int(rec['has_foggy']) + int(rec['has_night']) + int(rec['has_rainy']) + int(rec['has_snowy']) + int(rec['has_dawn_dusk'])
                rec['total_domains_present'] = rec['normal_domains_present'] + rec['adverse_domains_present']
                records.append(rec)
    return pd.DataFrame(records)


# ============================================================================
# Per-Dataset and Per-Domain Detailed Analysis
# ============================================================================

def compute_per_dataset_miou(weights_root: Path, scope: str = 'full') -> pd.DataFrame:
    """Compute mIoU per strategy and dataset from downstream_results.csv.
    
    Args:
        weights_root: Path to WEIGHTS directory (not used, kept for API compatibility)
        scope: 'full' for standard models, 'clear_day' for _cd suffix datasets
    
    Returns:
        DataFrame with columns: Strategy, Dataset, mIoU, Num_Configs
    """
    # Load from downstream_results.csv
    if not RESULTS_CSV.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(RESULTS_CSV)
    
    # Filter for the appropriate scope
    # NEW: clear_day filtering is now based on dataset suffix (_cd), not model suffix
    if scope == 'clear_day':
        # Filter for datasets ending with _cd suffix
        df = df[df['dataset'].str.endswith(CLEAR_DAY_DATASET_SUFFIX)]
        # Keep only standard model names (not _clear_day suffixed)
        df = df[df['model'].isin(MODELS)]
    else:
        # Full scope: exclude _cd suffix datasets, use standard models
        df = df[~df['dataset'].str.endswith(CLEAR_DAY_DATASET_SUFFIX)]
        df = df[df['model'].isin(MODELS)]
    
    if df.empty:
        return pd.DataFrame()
    
    # Group by strategy and dataset, compute mean mIoU
    records = []
    for (strategy, dataset), group in df.groupby(['strategy', 'dataset']):
        records.append({
            'Strategy': strategy,
            'Dataset': dataset,
            'mIoU': group['mIoU'].mean(),
            'Num_Configs': len(group)
        })
    
    return pd.DataFrame(records)


def compute_per_domain_miou(weights_root: Path, scope: str = 'full') -> pd.DataFrame:
    """Compute mIoU per strategy and weather domain.
    
    Args:
        weights_root: Path to WEIGHTS directory
        scope: 'full' for standard models, 'clear_day' for _cd suffix datasets
    
    Returns:
        DataFrame with columns: Strategy, Domain, mIoU, Num_Configs
    """
    # NEW: filtering is now based on dataset suffix (_cd), not model suffix
    target_models = MODELS  # Always use standard model names
    
    # Collect per-domain metrics across all configs
    domain_data = defaultdict(lambda: defaultdict(list))  # strategy -> domain -> [mIoU values]
    
    weights_path = Path(weights_root)
    
    for strategy_dir in weights_path.iterdir():
        if not strategy_dir.is_dir():
            continue
        strategy = strategy_dir.name
        
        for dataset_dir in strategy_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            # Filter based on scope: clear_day looks for _cd suffix, full excludes it
            is_cd_dataset = dataset_dir.name.endswith(CLEAR_DAY_DATASET_SUFFIX)
            if scope == 'clear_day' and not is_cd_dataset:
                continue
            if scope == 'full' and is_cd_dataset:
                continue
            
            for model_dir in dataset_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model = model_dir.name
                if model not in target_models:
                    continue
                
                # Read per-domain metrics
                detailed_root = model_dir / 'test_results_detailed'
                run_dir = _find_latest_detailed_run_dir(detailed_root)
                if run_dir is None:
                    continue
                
                domain_metrics = _read_per_domain_metrics_from_run(run_dir)
                if not domain_metrics:
                    continue
                
                # Normalize domain names and collect
                for domain_name, miou in domain_metrics.items():
                    normalized = _normalize_domain_name(domain_name)
                    if normalized and isinstance(miou, (int, float)):
                        domain_data[strategy][normalized].append(miou)
    
    # Build records
    records = []
    for strategy, domains in domain_data.items():
        for domain, mious in domains.items():
            if mious:
                records.append({
                    'Strategy': strategy,
                    'Domain': domain,
                    'mIoU': np.mean(mious),
                    'Num_Configs': len(mious)
                })
    
    return pd.DataFrame(records)


def generate_per_dataset_gains_table(weights_root: Path, scope: str = 'full') -> pd.DataFrame:
    """Generate table showing mIoU gain per strategy per dataset vs baseline_clear_day.
    
    Returns:
        DataFrame with columns: Strategy, Type, dataset columns, Avg Gain, Normal Gain, Adverse Gain
    """
    per_dataset_df = compute_per_dataset_miou(weights_root, scope)
    
    if per_dataset_df.empty:
        return pd.DataFrame()
    
    # Get baseline_clear_day per-dataset mIoU (always use clear_day scope for baseline)
    baseline_df = compute_per_dataset_miou(weights_root, 'clear_day')
    baseline_df = baseline_df[baseline_df['Strategy'] == BASELINE_FULL]
    
    baseline_miou_by_dataset = {}
    for _, row in baseline_df.iterrows():
        baseline_miou_by_dataset[row['Dataset']] = row['mIoU']
    
    # Get per-domain gains for Normal/Adverse summaries
    per_domain_gains = generate_per_domain_gains_table(weights_root, scope)
    domain_gains_by_strategy = {}
    if not per_domain_gains.empty:
        for _, row in per_domain_gains.iterrows():
            domain_gains_by_strategy[row['Strategy']] = {
                'Normal Gain': row.get('Normal Gain'),
                'Adverse Gain': row.get('Adverse Gain')
            }
    
    # Pivot to have datasets as columns
    datasets = sorted(per_dataset_df['Dataset'].unique())
    
    # Group by strategy and compute gains
    records = []
    for strategy in sorted(per_dataset_df['Strategy'].unique()):
        strategy_data = per_dataset_df[per_dataset_df['Strategy'] == strategy]
        
        # Determine type
        if strategy.startswith('gen_'):
            strategy_type = 'Generative'
        elif strategy.startswith('std_'):
            strategy_type = 'Standard Aug'
        elif strategy in STRATEGY_TYPES:
            strategy_type = STRATEGY_TYPES[strategy]
        else:
            strategy_type = 'Other'
        
        row = {'Strategy': strategy, 'Type': strategy_type}
        
        overall_gains = []
        for dataset in datasets:
            ds_data = strategy_data[strategy_data['Dataset'] == dataset]
            if not ds_data.empty and dataset in baseline_miou_by_dataset:
                miou = ds_data.iloc[0]['mIoU']
                baseline = baseline_miou_by_dataset[dataset]
                gain = miou - baseline
                row[dataset] = round(gain, 2)
                overall_gains.append(gain)
            else:
                row[dataset] = None
        
        # Add average gain
        row['Avg Gain'] = round(np.mean(overall_gains), 2) if overall_gains else None
        
        # Add Normal Gain and Adverse Gain from per-domain analysis
        if strategy in domain_gains_by_strategy:
            row['Normal Gain'] = domain_gains_by_strategy[strategy].get('Normal Gain')
            row['Adverse Gain'] = domain_gains_by_strategy[strategy].get('Adverse Gain')
        else:
            row['Normal Gain'] = None
            row['Adverse Gain'] = None
        
        records.append(row)
    
    result_df = pd.DataFrame(records)
    
    # Sort: baselines first, then by Avg Gain descending
    def sort_key(row):
        if row['Strategy'] == 'baseline_clear_day' or (row['Strategy'] == BASELINE_FULL and scope == 'clear_day'):
            return (0, 0)
        if row['Strategy'] == BASELINE_FULL:
            return (1, -row['Avg Gain'] if row['Avg Gain'] is not None else 999)
        return (2, -row['Avg Gain'] if row['Avg Gain'] is not None else 999)
    
    result_df['_sort'] = result_df.apply(sort_key, axis=1)
    result_df = result_df.sort_values('_sort').drop(columns=['_sort'])
    
    # Reorder columns: Strategy, Type, datasets, Avg Gain, Normal Gain, Adverse Gain
    col_order = ['Strategy', 'Type'] + datasets + ['Avg Gain', 'Normal Gain', 'Adverse Gain']
    result_df = result_df[[c for c in col_order if c in result_df.columns]]
    
    return result_df


def generate_per_domain_gains_table(weights_root: Path, scope: str = 'full') -> pd.DataFrame:
    """Generate table showing mIoU gain per strategy per domain vs baseline_clear_day.
    
    Returns:
        DataFrame with columns: Strategy, Type, and one column per domain showing gain
    """
    per_domain_df = compute_per_domain_miou(weights_root, scope)
    
    if per_domain_df.empty:
        return pd.DataFrame()
    
    # Get baseline_clear_day per-domain mIoU (always use clear_day scope for baseline)
    baseline_df = compute_per_domain_miou(weights_root, 'clear_day')
    baseline_df = baseline_df[baseline_df['Strategy'] == BASELINE_FULL]
    
    baseline_miou_by_domain = {}
    for _, row in baseline_df.iterrows():
        baseline_miou_by_domain[row['Domain']] = row['mIoU']
    
    # Use canonical domain order
    domain_order = NORMAL_DOMAINS + TRANSITION_DOMAINS + ADVERSE_DOMAINS
    available_domains = [d for d in domain_order if d in per_domain_df['Domain'].unique()]
    
    # Group by strategy and compute gains
    records = []
    for strategy in sorted(per_domain_df['Strategy'].unique()):
        strategy_data = per_domain_df[per_domain_df['Strategy'] == strategy]
        
        # Determine type
        if strategy.startswith('gen_'):
            strategy_type = 'Generative'
        elif strategy.startswith('std_'):
            strategy_type = 'Standard Aug'
        elif strategy in STRATEGY_TYPES:
            strategy_type = STRATEGY_TYPES[strategy]
        else:
            strategy_type = 'Other'
        
        row = {'Strategy': strategy, 'Type': strategy_type}
        
        normal_gains = []
        adverse_gains = []
        all_gains = []
        
        for domain in available_domains:
            dom_data = strategy_data[strategy_data['Domain'] == domain]
            if not dom_data.empty and domain in baseline_miou_by_domain:
                miou = dom_data.iloc[0]['mIoU']
                baseline = baseline_miou_by_domain[domain]
                gain = miou - baseline
                row[domain] = round(gain, 2)
                all_gains.append(gain)
                if domain in NORMAL_DOMAINS:
                    normal_gains.append(gain)
                elif domain in ADVERSE_DOMAINS:
                    adverse_gains.append(gain)
            else:
                row[domain] = None
        
        # Add group averages (renamed to Gain for consistency)
        row['Normal Gain'] = round(np.mean(normal_gains), 2) if normal_gains else None
        row['Adverse Gain'] = round(np.mean(adverse_gains), 2) if adverse_gains else None
        row['Overall Avg'] = round(np.mean(all_gains), 2) if all_gains else None
        records.append(row)
    
    result_df = pd.DataFrame(records)
    
    # Sort: baselines first, then by Overall Avg descending
    def sort_key(row):
        if row['Strategy'] == 'baseline_clear_day' or (row['Strategy'] == BASELINE_FULL and scope == 'clear_day'):
            return (0, 0)
        if row['Strategy'] == BASELINE_FULL:
            return (1, -row['Overall Avg'] if row['Overall Avg'] is not None else 999)
        return (2, -row['Overall Avg'] if row['Overall Avg'] is not None else 999)
    
    result_df['_sort'] = result_df.apply(sort_key, axis=1)
    result_df = result_df.sort_values('_sort').drop(columns=['_sort'])
    
    # Reorder columns: Strategy, Type, domains in order, then aggregates
    col_order = ['Strategy', 'Type'] + available_domains + ['Normal Gain', 'Adverse Gain', 'Overall Avg']
    result_df = result_df[[c for c in col_order if c in result_df.columns]]
    
    return result_df


def format_gains_table_markdown(df: pd.DataFrame, title: str, description: str = "") -> str:
    """Format a gains table as markdown."""
    lines = [
        f"## {title}",
        "",
    ]
    if description:
        lines.append(description)
        lines.append("")
    
    # Format header
    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join(["---" if c in ['Strategy', 'Type'] else "---:" for c in df.columns]) + " |"
    lines.append(header)
    lines.append(separator)
    
    # Format rows
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            val = row[col]
            if pd.isna(val) or val is None:
                values.append("-")
            elif isinstance(val, float):
                # Format gains with sign
                if col not in ['Strategy', 'Type']:
                    values.append(f"{val:+.2f}" if val != 0 else "0.00")
                else:
                    values.append(str(val))
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    
    return "\n".join(lines)


# ============================================================================
# Leaderboard Generation
# ============================================================================

def _build_unified_fallback_map() -> Dict[str, Dict[str, Optional[float]]]:
    """Load unified domain results and map by strategy name for fallback."""
    unified_df = load_unified_domain_results()
    unified_data: Dict[str, Dict[str, Optional[float]]] = {}
    if unified_df is not None and 'strategy' in unified_df.columns:
        for _, row in unified_df.iterrows():
            s = row['strategy']
            unified_data[s] = {
                'normal_mIoU': row.get('normal_mIoU') if 'normal_mIoU' in unified_df.columns else row.get('normal_miou'),
                'adverse_mIoU': row.get('adverse_mIoU') if 'adverse_mIoU' in unified_df.columns else row.get('adverse_miou'),
                'domain_gap': row.get('domain_gap')
            }
    return unified_data


def generate_leaderboard_for_scope(main_df: pd.DataFrame, weights_root: Path, scope: str,
                                   baseline_clearday_miou: Optional[float], baseline_clearday_gap: Optional[float],
                                   unified_fallback: Optional[Dict[str, Dict[str, Optional[float]]]] = None) -> pd.DataFrame:
    """Generate leaderboard for a given scope: 'full' or 'clear_day'."""
    strategies = sorted(main_df['strategy'].unique())
    rows: List[Dict] = []
    
    # Get baseline_clear_day Normal and Adverse mIoU for computing gains
    baseline_cd_normal, baseline_cd_adverse, _ = compute_baseline_clearday_domain_aggregates(weights_root)

    def add_row(strategy: str, strategy_type: str, overall: Optional[float], normal: Optional[float], adverse: Optional[float], domain_gap: Optional[float], num_results: int):
        gain_vs_clearday = None
        if baseline_clearday_miou is not None and overall is not None:
            gain_vs_clearday = overall - baseline_clearday_miou
        gap_reduction = None
        if domain_gap is not None and baseline_clearday_gap is not None:
            gap_reduction = baseline_clearday_gap - domain_gap
        # Compute Normal and Adverse gains vs baseline_clear_day
        normal_gain = None
        if baseline_cd_normal is not None and normal is not None:
            normal_gain = normal - baseline_cd_normal
        adverse_gain = None
        if baseline_cd_adverse is not None and adverse is not None:
            adverse_gain = adverse - baseline_cd_adverse
        rows.append({
            'Strategy': strategy,
            'Type': strategy_type,
            'Overall mIoU': round(overall, 2) if overall is not None else None,
            'Gain vs Clear Day': round(gain_vs_clearday, 2) if gain_vs_clearday is not None else None,
            'Normal mIoU': round(normal, 2) if normal is not None else None,
            'Normal Gain': round(normal_gain, 2) if normal_gain is not None else None,
            'Adverse mIoU': round(adverse, 2) if adverse is not None else None,
            'Adverse Gain': round(adverse_gain, 2) if adverse_gain is not None else None,
            'Domain Gap (Δ)': round(domain_gap, 2) if domain_gap is not None else None,
            'Gap Reduction vs Clear Day': round(gap_reduction, 2) if gap_reduction is not None else None,
            'Num Results': num_results,
        })

    # Baselines first
    if scope == 'clear_day':
        cd_normal, cd_adverse, cd_gap = compute_baseline_clearday_domain_aggregates(weights_root)
        # NEW: Use dataset-based filtering instead of model-based
        cd_metrics = aggregate_strategy_metrics_for_clearday(main_df, BASELINE_FULL)
        add_row('baseline_clear_day', STRATEGY_TYPES.get('baseline_clear_day', 'Baseline Clear Day'),
                cd_metrics['overall_miou'] if cd_metrics else None, cd_normal, cd_adverse, cd_gap, cd_metrics['num_results'] if cd_metrics else 0)
    else:
        # Include clear-day baseline as reference
        cd_normal, cd_adverse, cd_gap = compute_baseline_clearday_domain_aggregates(weights_root)
        # NEW: Use dataset-based filtering instead of model-based
        cd_metrics = aggregate_strategy_metrics_for_clearday(main_df, BASELINE_FULL)
        add_row('baseline_clear_day', STRATEGY_TYPES.get('baseline_clear_day', 'Baseline Clear Day'),
                cd_metrics['overall_miou'] if cd_metrics else None, cd_normal, cd_adverse, cd_gap, cd_metrics['num_results'] if cd_metrics else 0)
        # Baseline full row
        bf_metrics = aggregate_strategy_metrics(main_df, BASELINE_FULL)
        bf_normal, bf_adverse, bf_gap = compute_strategy_domain_aggregates(weights_root, BASELINE_FULL)
        if (bf_normal is None or bf_adverse is None or bf_gap is None) and unified_fallback is not None and BASELINE_FULL in unified_fallback:
            uf = unified_fallback[BASELINE_FULL]
            bf_normal = bf_normal if bf_normal is not None else uf.get('normal_mIoU')
            bf_adverse = bf_adverse if bf_adverse is not None else uf.get('adverse_mIoU')
            if bf_gap is None and bf_normal is not None and bf_adverse is not None:
                bf_gap = calculate_domain_gap(bf_normal, bf_adverse)
        add_row('baseline', STRATEGY_TYPES.get('baseline', 'Baseline Full'),
                bf_metrics['overall_miou'] if bf_metrics else None, bf_normal, bf_adverse, bf_gap, bf_metrics['num_results'] if bf_metrics else 0)

    # Strategy rows
    for strategy in strategies:
        # Avoid duplicating baseline rows we already added
        if strategy == BASELINE_FULL:
            continue
        stype = 'Generative' if strategy.startswith('gen_') else ('Standard Aug' if strategy.startswith('std_') else STRATEGY_TYPES.get(strategy, 'Other'))
        if scope == 'clear_day':
            # NEW: Use dataset-based filtering instead of model-based
            met = aggregate_strategy_metrics_for_clearday(main_df, strategy)
            if met is None:
                continue
            nrm, adv, gap = compute_strategy_clearday_domain_aggregates(weights_root, strategy)
            add_row(strategy, stype, met['overall_miou'], nrm, adv, gap, met['num_results'])
        else:
            met = aggregate_strategy_metrics(main_df, strategy)
            if met is None:
                continue
            nrm, adv, gap = compute_strategy_domain_aggregates(weights_root, strategy)
            if (nrm is None or adv is None or gap is None) and unified_fallback is not None and strategy in unified_fallback:
                uf = unified_fallback[strategy]
                if nrm is None:
                    nrm = uf.get('normal_mIoU')
                if adv is None:
                    adv = uf.get('adverse_mIoU')
                if gap is None and nrm is not None and adv is not None:
                    gap = calculate_domain_gap(nrm, adv)
            add_row(strategy, stype, met['overall_miou'], nrm, adv, gap, met['num_results'])

    df = pd.DataFrame(rows)
    # Baselines at the top
    order = ['baseline_clear_day'] if scope == 'clear_day' else ['baseline_clear_day', 'baseline']
    df['__order'] = df['Strategy'].apply(lambda s: order.index(s) if s in order else len(order))
    df = df.sort_values(['__order', 'Overall mIoU'], ascending=[True, False]).drop(columns='__order')
    return df


def generate_comprehensive_leaderboards(weights_root: Path = WEIGHTS_ROOT) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate two leaderboards: full-trained and clear_day-trained."""
    print("=" * 70)
    print("Generating Strategy Leaderboards (Full and Clear-Day)")
    print("=" * 70)
    print("Reference Baseline: baseline_clear_day (trained on clear_day only)")
    print("=" * 70)

    main_df = load_results_csv(RESULTS_CSV)
    clearday_analysis = analyze_baseline_clear_day(main_df)
    baseline_clearday_miou = clearday_analysis['overall_miou'] if clearday_analysis else None
    baseline_full_metrics = aggregate_strategy_metrics(main_df, BASELINE_FULL)
    baseline_full_miou = baseline_full_metrics['overall_miou'] if baseline_full_metrics else None
    print(f"  Baseline Clear Day mIoU: {baseline_clearday_miou:.2f}%" if baseline_clearday_miou else "  Baseline Clear Day: NOT FOUND")
    print(f"  Baseline Full mIoU: {baseline_full_miou:.2f}%" if baseline_full_miou else "  Baseline Full: NOT FOUND")
    baseline_clearday_gap = compute_baseline_clearday_domain_gap(weights_root)
    unified_fallback = _build_unified_fallback_map()

    full_df = generate_leaderboard_for_scope(main_df, weights_root, 'full', baseline_clearday_miou, baseline_clearday_gap, unified_fallback)
    clear_df = generate_leaderboard_for_scope(main_df, weights_root, 'clear_day', baseline_clearday_miou, baseline_clearday_gap, None)
    print(f"Generated leaderboards: full={len(full_df)} strategies, clear_day={len(clear_df)} strategies")
    return full_df, clear_df


def load_unified_domain_results() -> Optional[pd.DataFrame]:
    """Load results from unified_domain_gap analysis if available."""
    results_dir = PROJECT_ROOT / 'result_figures' / 'unified_domain_gap'
    summary_file = results_dir / 'strategy_summary.csv'
    if summary_file.exists():
        try:
            return pd.read_csv(summary_file)
        except Exception:
            return None
    all_results_file = results_dir / 'all_domain_results.csv'
    if all_results_file.exists():
        try:
            return pd.read_csv(all_results_file)
        except Exception:
            return None
    return None


def format_leaderboard_markdown(df: pd.DataFrame, title: str = "Strategy Leaderboard") -> str:
    """Format leaderboard as markdown table."""
    
    lines = [
        f"# {title}",
        "",
        f"Generated from PROVE domain gap analysis pipeline.",
        "",
        "**Reference Baseline: `baseline_clear_day`** (models trained only on clear_day data)",
        "",
        "This is the proper baseline for measuring augmentation effectiveness, as it represents",
        "models that never saw adverse weather conditions during training.",
        "",
        "**Metrics:**",
        "- **Overall mIoU**: Mean Intersection over Union across all domains",
        "- **Gain vs Clear Day**: Overall mIoU improvement vs baseline_clear_day (positive = better)",
        "- **Normal mIoU**: Performance on clear_day + cloudy conditions",
        "- **Normal Gain**: Normal mIoU improvement vs baseline_clear_day (positive = better)",
        "- **Adverse mIoU**: Performance on foggy, rainy, snowy, night conditions",
        "- **Adverse Gain**: Adverse mIoU improvement vs baseline_clear_day (positive = better)",
        "- **Domain Gap (Δ)**: Normal - Adverse (positive = worse on adverse)",
        "- **Gap Reduction vs Clear Day**: Domain gap improvement vs baseline_clear_day (positive = smaller gap)",
        "",
        "**Baseline Types:**",
        "- `baseline_clear_day`: Trained only on clear_day data (THE REFERENCE)",
        "- `baseline` / `baseline_full`: Trained on all weather conditions",
        "",
        "---",
        "",
    ]
    
    # Add table - include Normal Gain and Adverse Gain columns
    cols = ['Strategy', 'Type', 'Overall mIoU', 'Gain vs Clear Day', 'Normal mIoU', 'Normal Gain',
            'Adverse mIoU', 'Adverse Gain', 'Domain Gap (Δ)', 'Gap Reduction vs Clear Day']
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
                if 'Gain' in col or 'Gap' in col:
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
    """Analyze baseline_clear_day (models trained only on clear day) performance.
    
    NEW: Uses dataset suffix (_cd) instead of model suffix (_clear_day).
    """
    
    # Restrict to baseline strategy with _cd suffix datasets and standard models
    clearday_df = df[(df['strategy'] == BASELINE_FULL) & 
                     (df['dataset'].str.endswith(CLEAR_DAY_DATASET_SUFFIX)) &
                     (df['model'].isin(MODELS))]
    
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
    
    for model in MODELS:
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
    
    # Analyze baseline_full (trained on all conditions)
    print("\n2. Analyzing baseline_full performance (trained on all conditions)...")
    baseline_analysis = analyze_baseline(df)
    if baseline_analysis:
        print(f"   Baseline Full Overall mIoU: {baseline_analysis['overall_miou']:.2f}%")
        print(f"   By Dataset:")
        for ds, data in baseline_analysis['by_dataset'].items():
            print(f"     {ds}: {data['miou']:.2f}% ± {data['std']:.2f}")
    else:
        print("   No baseline_full results found")
    
    # Analyze baseline_clear_day (REFERENCE BASELINE - trained only on clear_day)
    print("\n3. Analyzing baseline_clear_day performance (THE REFERENCE BASELINE)...")
    clearday_analysis = analyze_baseline_clear_day(df)
    if clearday_analysis:
        print(f"   *** Clear-Day Baseline mIoU: {clearday_analysis['overall_miou']:.2f}% ***")
        print(f"   (This is the reference for computing gains/losses)")
        print(f"   By Dataset:")
        for ds, data in clearday_analysis['by_dataset'].items():
            print(f"     {ds}: {data['miou']:.2f}% ± {data['std']:.2f}")
    else:
        print("   WARNING: No clear_day baseline results found!")
        print("   Gap reduction calculations will be unavailable.")
    
    # Generate leaderboards (Full + Clear-Day)
    print("\n4. Generating comprehensive leaderboards (full and clear-day)...")
    full_df, clear_df = generate_comprehensive_leaderboards(weights_path)

    # Print summaries
    print("\n" + "=" * 70)
    print("FULL-TRAINED STRATEGY LEADERBOARD")
    print("Reference Baseline: baseline_clear_day")
    print("=" * 70)
    print(full_df.to_string(index=False))
    print("\n" + "=" * 70)
    print("CLEAR-DAY TRAINED STRATEGY LEADERBOARD")
    print("Reference Baseline: baseline_clear_day")
    print("=" * 70)
    print(clear_df.to_string(index=False))
    
    # Save outputs (Markdown with two sections)
    output_file = Path(args.output) if args.output else OUTPUT_DIR / 'STRATEGY_LEADERBOARD.md'
    md_sections = []
    md_sections.append(format_leaderboard_markdown(full_df, title="Strategy Leaderboard (Full-Trained)"))
    md_sections.append("")
    md_sections.append(format_leaderboard_markdown(clear_df, title="Strategy Leaderboard (Clear-Day Trained)"))
    with open(output_file, 'w') as f:
        f.write("\n\n".join(md_sections))
    print(f"\nLeaderboards saved to: {output_file}")
    
    # Generate per-dataset and per-domain gains tables
    print("\n5. Generating per-dataset and per-domain gains tables...")
    
    # Per-Dataset Gains (Full-Trained)
    per_dataset_full = generate_per_dataset_gains_table(weights_path, scope='full')
    if not per_dataset_full.empty:
        print(f"   Per-Dataset Gains (Full): {len(per_dataset_full)} strategies")
    
    # Per-Dataset Gains (Clear-Day Trained)
    per_dataset_clear = generate_per_dataset_gains_table(weights_path, scope='clear_day')
    if not per_dataset_clear.empty:
        print(f"   Per-Dataset Gains (Clear-Day): {len(per_dataset_clear)} strategies")
    
    # Per-Domain Gains (Full-Trained)
    per_domain_full = generate_per_domain_gains_table(weights_path, scope='full')
    if not per_domain_full.empty:
        print(f"   Per-Domain Gains (Full): {len(per_domain_full)} strategies")
    
    # Per-Domain Gains (Clear-Day Trained)
    per_domain_clear = generate_per_domain_gains_table(weights_path, scope='clear_day')
    if not per_domain_clear.empty:
        print(f"   Per-Domain Gains (Clear-Day): {len(per_domain_clear)} strategies")
    
    # Sanity check: warn about large discrepancies between Avg Gain and Normal/Adverse Gain
    print("\n6. Sanity check - checking for data coverage issues...")
    for scope_name, ds_df in [('Full', per_dataset_full), ('Clear-Day', per_dataset_clear)]:
        if ds_df.empty:
            continue
        for _, row in ds_df.iterrows():
            avg_gain = row.get('Avg Gain')
            normal_gain = row.get('Normal Gain')
            adverse_gain = row.get('Adverse Gain')
            strategy = row['Strategy']
            
            # Check for sign mismatch: Avg Gain positive but both Normal/Adverse negative (or vice versa)
            if avg_gain is not None and normal_gain is not None and adverse_gain is not None:
                if (avg_gain > 0 and normal_gain < -1 and adverse_gain < -1) or \
                   (avg_gain < -1 and normal_gain > 0 and adverse_gain > 0):
                    print(f"   ⚠️  {scope_name} {strategy}: Avg Gain ({avg_gain:+.2f}) vs Normal ({normal_gain:+.2f}) / Adverse ({adverse_gain:+.2f}) - different data coverage")
    
    # Save detailed gains tables to separate markdown file
    detailed_output_file = OUTPUT_DIR / 'DETAILED_GAINS.md'
    detailed_sections = [
        "# Detailed Per-Dataset and Per-Domain mIoU Gains",
        "",
        "All gains are computed relative to **baseline_clear_day** (trained only on clear_day data).",
        "",
        "**Important Notes:**",
        "- **Avg Gain** = Average improvement across available datasets (from overall mIoU)",
        "- **Normal Gain / Adverse Gain** = Improvement from per-domain detailed test results",
        "- These metrics may come from different data subsets for strategies with incomplete coverage",
        "- Strategies with '-' entries have missing data for those columns",
        "",
        "---",
        ""
    ]
    
    if not per_dataset_full.empty:
        detailed_sections.append(format_gains_table_markdown(
            per_dataset_full,
            title="Per-Dataset mIoU Gains (Full-Trained Models)",
            description="Shows mIoU improvement over baseline_clear_day for each dataset. Positive = better than baseline.\n\n**Columns:** Dataset gains from overall mIoU; Normal/Adverse Gain from per-domain metrics (may have different coverage)."
        ))
        detailed_sections.append("")
        detailed_sections.append("---")
        detailed_sections.append("")
    
    if not per_dataset_clear.empty:
        detailed_sections.append(format_gains_table_markdown(
            per_dataset_clear,
            title="Per-Dataset mIoU Gains (Clear-Day Trained Models)",
            description="Shows mIoU improvement over baseline_clear_day for models trained only on clear_day data."
        ))
        detailed_sections.append("")
        detailed_sections.append("---")
        detailed_sections.append("")
    
    if not per_domain_full.empty:
        detailed_sections.append(format_gains_table_markdown(
            per_domain_full,
            title="Per-Domain mIoU Gains (Full-Trained Models)",
            description="Shows mIoU improvement over baseline_clear_day for each weather domain. Normal = clear_day + cloudy. Adverse = foggy, night, rainy, snowy."
        ))
        detailed_sections.append("")
        detailed_sections.append("---")
        detailed_sections.append("")
    
    if not per_domain_clear.empty:
        detailed_sections.append(format_gains_table_markdown(
            per_domain_clear,
            title="Per-Domain mIoU Gains (Clear-Day Trained Models)",
            description="Shows mIoU improvement over baseline_clear_day for models trained only on clear_day data."
        ))
    
    with open(detailed_output_file, 'w') as f:
        f.write("\n".join(detailed_sections))
    print(f"Detailed gains saved to: {detailed_output_file}")
    
    # Save CSVs for detailed tables
    if not per_dataset_full.empty:
        per_dataset_full.to_csv(OUTPUT_DIR / 'per_dataset_gains_full.csv', index=False)
    if not per_dataset_clear.empty:
        per_dataset_clear.to_csv(OUTPUT_DIR / 'per_dataset_gains_clear_day.csv', index=False)
    if not per_domain_full.empty:
        per_domain_full.to_csv(OUTPUT_DIR / 'per_domain_gains_full.csv', index=False)
    if not per_domain_clear.empty:
        per_domain_clear.to_csv(OUTPUT_DIR / 'per_domain_gains_clear_day.csv', index=False)
    print(f"Detailed gains CSVs saved to: {OUTPUT_DIR}")
    
    # Save CSVs
    csv_full = OUTPUT_DIR / 'strategy_leaderboard_full.csv'
    csv_clear = OUTPUT_DIR / 'strategy_leaderboard_clear_day.csv'
    full_df.to_csv(csv_full, index=False)
    clear_df.to_csv(csv_clear, index=False)
    # Backward-compatibility: keep primary CSV as full
    csv_file = Path(args.csv) if args.csv else OUTPUT_DIR / 'strategy_leaderboard.csv'
    full_df.to_csv(csv_file, index=False)
    print(f"CSV saved to: {csv_file}\nFull CSV: {csv_full}\nClear-Day CSV: {csv_clear}")
    
    # Also save analysis summary
    summary = {
        'baseline_full': baseline_analysis,
        'baseline_clear_day': clearday_analysis,  # This is the reference
        'reference_baseline': 'baseline_clear_day',
        'leaderboard_rows_full': len(full_df),
        'leaderboard_rows_clear': len(clear_df),
    }
    
    summary_file = OUTPUT_DIR / 'analysis_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Analysis summary saved to: {summary_file}")

    # Produce per-domain coverage audit
    coverage_df = audit_per_domain_coverage(weights_path)
    coverage_file = OUTPUT_DIR / 'per_domain_coverage.csv'
    coverage_df.to_csv(coverage_file, index=False)
    print(f"Coverage audit saved to: {coverage_file} ({len(coverage_df)} rows)")


if __name__ == '__main__':
    main()
