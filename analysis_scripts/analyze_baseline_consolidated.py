#!/usr/bin/env python3
"""
Comprehensive Baseline Analysis Script (Consolidated)

This script consolidates analyze_baseline.py, analyze_baseline_v2.py,
analyze_baseline_clear_day.py, and analyze_baseline_miou.py into a single
comprehensive analysis tool.

Features:
1. Stage 1 baseline analysis (models trained only on clear_day) - WEIGHTS/
2. Stage 2 baseline analysis (models trained on all domains) - WEIGHTS_STAGE_2/
3. Per-domain performance breakdown
4. Per-dataset and per-model comparisons
5. Domain gap analysis (normal vs adverse conditions)
6. Both mIoU and fwIoU metrics

Primary Metric: mIoU (mean Intersection over Union)
    - Recommended for domain robustness analysis
    - Equal weight to all classes

Usage:
    python analysis_scripts/analyze_baseline_consolidated.py [--output-dir DIR]

Output:
    result_figures/baseline_consolidated/
    - BASELINE_ANALYSIS_REPORT.md
    - baseline_summary.csv
    - domain_performance.csv
    - Various visualization figures (optional)
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# Configuration
# ==============================================================================

# Stage 1 = Clear day only training, Stage 2 = All domains training
WEIGHTS_ROOT_STAGE1 = "${AWARE_DATA_ROOT}/WEIGHTS"
WEIGHTS_ROOT_STAGE2 = "${AWARE_DATA_ROOT}/WEIGHTS_STAGE_2"

# Weather domains
ALL_DOMAINS = ['clear_day', 'cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy']
ADVERSE_DOMAINS = ['dawn_dusk', 'night', 'rainy', 'snowy']
NORMAL_DOMAINS = ['clear_day', 'cloudy']

# Datasets
DATASETS = ['bdd10k', 'iddaw', 'mapillaryvistas', 'outside15k']

# Models
MODELS = ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5']

# Minimum number of images for a domain to be considered valid
# Small test sets (e.g., BDD10k foggy=4 images) produce unreliable metrics
MIN_IMAGES_THRESHOLD = 50


# ==============================================================================
# Data Loading Functions
# ==============================================================================

def load_results_json(filepath: Path) -> dict:
    """Load results from a JSON file."""
    try:
        with open(filepath) as f:
            return json.load(f)
    except Exception as e:
        return None


def find_latest_results(detailed_dir: Path) -> Path:
    """Find the latest results.json in a test_results_detailed directory."""
    if not detailed_dir.exists():
        return None
    
    # Get all subdirectories and sort by timestamp (reversed)
    subdirs = sorted([d for d in detailed_dir.iterdir() if d.is_dir()], reverse=True)
    
    for subdir in subdirs:
        results_file = subdir / "results.json"
        if results_file.exists():
            return results_file
    
    return None


def load_baseline_results(weights_root: str, stage_name: str = "stage1") -> pd.DataFrame:
    """Load baseline results from the weights directory.
    
    Args:
        weights_root: Root directory for weights (WEIGHTS or WEIGHTS_STAGE_2)
        stage_name: Name for reporting ('stage1' or 'stage2')
        
    Returns:
        DataFrame with per-domain results for all configurations
    """
    results = []
    strategy_dir = Path(weights_root) / "baseline"
    
    if not strategy_dir.exists():
        print(f"Strategy directory not found: {strategy_dir}")
        return pd.DataFrame()
    
    for dataset_dir in strategy_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name.lower()
        
        # Skip any legacy _cd or _ad suffixes
        if dataset_name.endswith('_cd') or dataset_name.endswith('_ad'):
            continue
        
        for model_dir in dataset_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            
            # Skip non-standard models
            if model not in MODELS:
                # Skip other domain-specific models
                if any(domain in model for domain in ALL_DOMAINS):
                    continue
            
            # Find detailed results
            detailed_dir = model_dir / "test_results_detailed"
            results_file = find_latest_results(detailed_dir)
            
            if not results_file:
                continue
            
            data = load_results_json(results_file)
            if not data:
                continue
            
            per_domain = data.get('per_domain', {})
            if len(per_domain) < 4:
                # Skip incomplete results
                continue
            
            # Extract per-domain metrics
            for domain, metrics in per_domain.items():
                summary = metrics.get('summary', metrics)
                results.append({
                    'dataset': dataset_name,
                    'model': model,
                    'stage': stage_name,
                    'domain': domain,
                    'mIoU': summary.get('mIoU', 0),
                    'aAcc': summary.get('aAcc', 0),
                    'fwIoU': summary.get('fwIoU', 0),
                    'num_images': summary.get('num_images', 0)
                })
    
    return pd.DataFrame(results)


# ==============================================================================
# Analysis Functions
# ==============================================================================

def filter_small_test_sets(df: pd.DataFrame, min_images: int = MIN_IMAGES_THRESHOLD, verbose: bool = False) -> pd.DataFrame:
    """Filter out results from small test sets that may produce unreliable metrics.
    
    Args:
        df: DataFrame with per-domain results
        min_images: Minimum number of images required for a domain to be included
        verbose: Whether to print details about filtered results
        
    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df
    
    before = len(df)
    filtered = df[df['num_images'] >= min_images].copy()
    after = len(filtered)
    
    if before != after and verbose:
        excluded = df[df['num_images'] < min_images]
        print(f"  Filtered out {before - after} results with < {min_images} images:")
        for _, row in excluded.iterrows():
            print(f"    - {row['dataset']}/{row['model']}/{row['domain']}: {row['num_images']} images")
    
    return filtered
    
    return filtered


def compute_aggregates(df: pd.DataFrame, filter_small: bool = True) -> dict:
    """Compute aggregate statistics from per-domain results.
    
    Args:
        df: DataFrame with per-domain results
        filter_small: If True, exclude small test sets
    """
    
    if df.empty:
        return {}
    
    # Optionally filter small test sets
    if filter_small:
        df = filter_small_test_sets(df)
    
    if df.empty:
        return {}
    
    # Calculate normal/adverse averages per config
    normal_df = df[df['domain'].isin(NORMAL_DOMAINS)]
    adverse_df = df[df['domain'].isin(ADVERSE_DOMAINS)]
    
    results = {
        'overall_miou': df['mIoU'].mean(),
        'overall_std': df['mIoU'].std(),
        'normal_miou': normal_df['mIoU'].mean() if not normal_df.empty else 0,
        'adverse_miou': adverse_df['mIoU'].mean() if not adverse_df.empty else 0,
        'domain_gap': 0
    }
    
    if results['normal_miou'] > 0 and results['adverse_miou'] > 0:
        results['domain_gap'] = results['normal_miou'] - results['adverse_miou']
    
    return results


def per_domain_analysis(df: pd.DataFrame, filter_small: bool = True) -> pd.DataFrame:
    """Analyze performance per domain across all configs.
    
    Args:
        df: DataFrame with per-domain results
        filter_small: If True, exclude small test sets
    """
    
    if filter_small:
        df = filter_small_test_sets(df)
    
    domain_stats = []
    for domain in ALL_DOMAINS:
        domain_df = df[df['domain'] == domain]
        if domain_df.empty:
            continue
        
        domain_type = "ADVERSE" if domain in ADVERSE_DOMAINS else "NORMAL"
        domain_stats.append({
            'domain': domain,
            'type': domain_type,
            'mIoU_mean': domain_df['mIoU'].mean(),
            'mIoU_std': domain_df['mIoU'].std(),
            'fwIoU_mean': domain_df['fwIoU'].mean(),
            'num_configs': len(domain_df),
            'total_images': domain_df['num_images'].sum(),
            'avg_images': domain_df['num_images'].mean()
        })
    
    return pd.DataFrame(domain_stats)


def per_dataset_analysis(df: pd.DataFrame, filter_small: bool = True) -> pd.DataFrame:
    """Analyze performance per dataset across all configs."""
    
    if filter_small:
        df = filter_small_test_sets(df)
    
    dataset_stats = []
    for dataset in DATASETS:
        dataset_df = df[df['dataset'] == dataset]
        if dataset_df.empty:
            continue
        
        normal = dataset_df[dataset_df['domain'].isin(NORMAL_DOMAINS)]
        adverse = dataset_df[dataset_df['domain'].isin(ADVERSE_DOMAINS)]
        
        dataset_stats.append({
            'dataset': dataset,
            'overall_mIoU': dataset_df['mIoU'].mean(),
            'overall_std': dataset_df['mIoU'].std(),
            'normal_mIoU': normal['mIoU'].mean() if not normal.empty else 0,
            'adverse_mIoU': adverse['mIoU'].mean() if not adverse.empty else 0,
            'domain_gap': (normal['mIoU'].mean() - adverse['mIoU'].mean()) if not normal.empty and not adverse.empty else 0,
            'num_samples': len(dataset_df)
        })
    
    return pd.DataFrame(dataset_stats)


def per_model_analysis(df: pd.DataFrame, filter_small: bool = True) -> pd.DataFrame:
    """Analyze performance per model across all configs."""
    
    if filter_small:
        df = filter_small_test_sets(df)
    
    model_stats = []
    for model in MODELS:
        model_df = df[df['model'] == model]
        if model_df.empty:
            continue
        
        normal = model_df[model_df['domain'].isin(NORMAL_DOMAINS)]
        adverse = model_df[model_df['domain'].isin(ADVERSE_DOMAINS)]
        
        model_stats.append({
            'model': model,
            'overall_mIoU': model_df['mIoU'].mean(),
            'overall_std': model_df['mIoU'].std(),
            'normal_mIoU': normal['mIoU'].mean() if not normal.empty else 0,
            'adverse_mIoU': adverse['mIoU'].mean() if not adverse.empty else 0,
            'domain_gap': (normal['mIoU'].mean() - adverse['mIoU'].mean()) if not normal.empty and not adverse.empty else 0,
            'num_samples': len(model_df)
        })
    
    return pd.DataFrame(model_stats)


def per_config_analysis(df: pd.DataFrame, filter_small: bool = True) -> pd.DataFrame:
    """Analyze performance per configuration (dataset + model)."""
    
    if filter_small:
        df = filter_small_test_sets(df)
    
    config_stats = []
    for dataset in DATASETS:
        for model in MODELS:
            config_df = df[(df['dataset'] == dataset) & (df['model'] == model)]
            if config_df.empty:
                continue
            
            # Get clear_day domain performance
            clear_day = config_df[config_df['domain'] == 'clear_day']['mIoU'].values
            clear_day_miou = clear_day[0] if len(clear_day) > 0 else 0
            
            normal = config_df[config_df['domain'].isin(NORMAL_DOMAINS)]
            adverse = config_df[config_df['domain'].isin(ADVERSE_DOMAINS)]
            
            config_stats.append({
                'dataset': dataset,
                'model': model,
                'clear_day_mIoU': clear_day_miou,
                'normal_mIoU': normal['mIoU'].mean() if not normal.empty else 0,
                'adverse_mIoU': adverse['mIoU'].mean() if not adverse.empty else 0,
                'overall_mIoU': config_df['mIoU'].mean(),
                'domain_gap': (normal['mIoU'].mean() - adverse['mIoU'].mean()) if not normal.empty and not adverse.empty else 0
            })
    
    return pd.DataFrame(config_stats)


# ==============================================================================
# Report Generation
# ==============================================================================

def generate_report(stage1_df: pd.DataFrame, stage2_df: pd.DataFrame, output_dir: Path) -> str:
    """Generate comprehensive markdown report."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    lines = []
    lines.append("# Comprehensive Baseline Analysis Report")
    lines.append(f"\nGenerated: {timestamp}")
    lines.append(f"\n**Note:** Results from domains with fewer than {MIN_IMAGES_THRESHOLD} test images are excluded to ensure reliable metrics.\n")
    
    # ============== Stage 1 Baseline Analysis ==============
    lines.append("## 1. Stage 1 Baseline Analysis (Clear Day Training)")
    lines.append("\nModels trained ONLY on clear_day images.")
    lines.append(f"\n**Source:** `{WEIGHTS_ROOT_STAGE1}/baseline/`\n")
    
    if stage1_df.empty:
        lines.append("**No Stage 1 baseline results found.**\n")
    else:
        agg = compute_aggregates(stage1_df)
        lines.append(f"### Overall Statistics\n")
        lines.append(f"- **Average mIoU:** {agg['overall_miou']:.2f}% ± {agg['overall_std']:.2f}")
        lines.append(f"- **Normal Conditions mIoU:** {agg['normal_miou']:.2f}%")
        lines.append(f"- **Adverse Conditions mIoU:** {agg['adverse_miou']:.2f}%")
        lines.append(f"- **Domain Gap (Normal - Adverse):** {agg['domain_gap']:+.2f}%\n")
        
        # Per-domain breakdown
        lines.append("### Per-Domain Performance\n")
        domain_df = per_domain_analysis(stage1_df)
        if not domain_df.empty:
            lines.append("| Domain | Type | mIoU | Std | Avg Images |")
            lines.append("|--------|------|------|-----|------------|")
            for _, row in domain_df.iterrows():
                lines.append(f"| {row['domain']} | {row['type']} | {row['mIoU_mean']:.2f}% | ±{row['mIoU_std']:.2f} | {row['avg_images']:.0f} |")
            lines.append("")
        
        # Per-dataset breakdown
        lines.append("### Per-Dataset Performance\n")
        dataset_df = per_dataset_analysis(stage1_df)
        if not dataset_df.empty:
            lines.append("| Dataset | Overall mIoU | Normal | Adverse | Gap |")
            lines.append("|---------|--------------|--------|---------|-----|")
            for _, row in dataset_df.iterrows():
                lines.append(f"| {row['dataset']} | {row['overall_mIoU']:.2f}% | {row['normal_mIoU']:.2f}% | {row['adverse_mIoU']:.2f}% | {row['domain_gap']:+.2f}% |")
            lines.append("")
        
        # Per-model breakdown
        lines.append("### Per-Model Performance\n")
        model_df = per_model_analysis(stage1_df)
        if not model_df.empty:
            lines.append("| Model | Overall mIoU | Normal | Adverse | Gap |")
            lines.append("|-------|--------------|--------|---------|-----|")
            for _, row in model_df.iterrows():
                lines.append(f"| {row['model']} | {row['overall_mIoU']:.2f}% | {row['normal_mIoU']:.2f}% | {row['adverse_mIoU']:.2f}% | {row['domain_gap']:+.2f}% |")
            lines.append("")
        
        # Per-config breakdown
        lines.append("### Per-Configuration Performance\n")
        config_df = per_config_analysis(stage1_df)
        if not config_df.empty:
            config_df = config_df.sort_values('overall_mIoU', ascending=False)
            lines.append("| Dataset | Model | Clear Day | Normal | Adverse | Overall | Gap |")
            lines.append("|---------|-------|-----------|--------|---------|---------|-----|")
            for _, row in config_df.iterrows():
                lines.append(f"| {row['dataset']} | {row['model']} | {row['clear_day_mIoU']:.1f}% | {row['normal_mIoU']:.1f}% | {row['adverse_mIoU']:.1f}% | {row['overall_mIoU']:.1f}% | {row['domain_gap']:+.1f}% |")
            lines.append("")
    
    # ============== Stage 2 Baseline Analysis ==============
    lines.append("## 2. Stage 2 Baseline Analysis (All Domains Training)")
    lines.append("\nModels trained on ALL domains (clear_day + adverse).")
    lines.append(f"\n**Source:** `{WEIGHTS_ROOT_STAGE2}/baseline/`\n")
    
    if stage2_df.empty:
        lines.append("**No Stage 2 baseline results found.**\n")
    else:
        agg = compute_aggregates(stage2_df)
        lines.append(f"### Overall Statistics\n")
        lines.append(f"- **Average mIoU:** {agg['overall_miou']:.2f}% ± {agg['overall_std']:.2f}")
        lines.append(f"- **Normal Conditions mIoU:** {agg['normal_miou']:.2f}%")
        lines.append(f"- **Adverse Conditions mIoU:** {agg['adverse_miou']:.2f}%")
        lines.append(f"- **Domain Gap (Normal - Adverse):** {agg['domain_gap']:+.2f}%\n")
        
        # Per-domain breakdown
        lines.append("### Per-Domain Performance\n")
        domain_df = per_domain_analysis(stage2_df)
        if not domain_df.empty:
            lines.append("| Domain | Type | mIoU | Std |")
            lines.append("|--------|------|------|-----|")
            for _, row in domain_df.iterrows():
                lines.append(f"| {row['domain']} | {row['type']} | {row['mIoU_mean']:.2f}% | ±{row['mIoU_std']:.2f} |")
            lines.append("")
        
        # Per-dataset breakdown
        lines.append("### Per-Dataset Performance\n")
        dataset_df = per_dataset_analysis(stage2_df)
        if not dataset_df.empty:
            lines.append("| Dataset | Overall mIoU | Normal | Adverse | Gap |")
            lines.append("|---------|--------------|--------|---------|-----|")
            for _, row in dataset_df.iterrows():
                lines.append(f"| {row['dataset']} | {row['overall_mIoU']:.2f}% | {row['normal_mIoU']:.2f}% | {row['adverse_mIoU']:.2f}% | {row['domain_gap']:+.2f}% |")
            lines.append("")
        
        # Per-model breakdown
        lines.append("### Per-Model Performance\n")
        model_df = per_model_analysis(stage2_df)
        if not model_df.empty:
            lines.append("| Model | Overall mIoU | Normal | Adverse | Gap |")
            lines.append("|-------|--------------|--------|---------|-----|")
            for _, row in model_df.iterrows():
                lines.append(f"| {row['model']} | {row['overall_mIoU']:.2f}% | {row['normal_mIoU']:.2f}% | {row['adverse_mIoU']:.2f}% | {row['domain_gap']:+.2f}% |")
            lines.append("")
    
    # ============== Comparison ==============
    if not stage1_df.empty and not stage2_df.empty:
        lines.append("## 3. Stage 1 vs Stage 2 Baseline Comparison\n")
        
        s1_agg = compute_aggregates(stage1_df)
        s2_agg = compute_aggregates(stage2_df)
        
        lines.append("| Metric | Stage 1 (Clear Day) | Stage 2 (All Domains) | Difference |")
        lines.append("|--------|---------------------|----------------------|------------|")
        lines.append(f"| Overall mIoU | {s1_agg['overall_miou']:.2f}% | {s2_agg['overall_miou']:.2f}% | {s2_agg['overall_miou'] - s1_agg['overall_miou']:+.2f}% |")
        lines.append(f"| Normal mIoU | {s1_agg['normal_miou']:.2f}% | {s2_agg['normal_miou']:.2f}% | {s2_agg['normal_miou'] - s1_agg['normal_miou']:+.2f}% |")
        lines.append(f"| Adverse mIoU | {s1_agg['adverse_miou']:.2f}% | {s2_agg['adverse_miou']:.2f}% | {s2_agg['adverse_miou'] - s1_agg['adverse_miou']:+.2f}% |")
        lines.append(f"| Domain Gap | {s1_agg['domain_gap']:+.2f}% | {s2_agg['domain_gap']:+.2f}% | {s2_agg['domain_gap'] - s1_agg['domain_gap']:+.2f}% |")
        lines.append("")
        
        lines.append("### Key Insights\n")
        lines.append("- **Stage 1** models are trained only on clear_day, testing cross-domain robustness")
        lines.append("- **Stage 2** models are trained on all domains, testing domain-inclusive performance")
        if s2_agg['overall_miou'] > s1_agg['overall_miou']:
            lines.append(f"- Stage 2 outperforms Stage 1 by **{s2_agg['overall_miou'] - s1_agg['overall_miou']:+.2f}%** overall mIoU")
        if s2_agg['domain_gap'] < s1_agg['domain_gap']:
            lines.append(f"- Stage 2 has **smaller domain gap** ({s2_agg['domain_gap']:.2f}% vs {s1_agg['domain_gap']:.2f}%)")
        lines.append("")
    
    # ============== Key Insights ==============
    lines.append("## 4. Key Insights\n")
    
    if not stage1_df.empty:
        # Best/worst configs
        config_df = per_config_analysis(stage1_df)
        if not config_df.empty:
            best = config_df.loc[config_df['overall_mIoU'].idxmax()]
            worst = config_df.loc[config_df['overall_mIoU'].idxmin()]
            
            lines.append("### Best and Worst Configurations (Stage 1)\n")
            lines.append(f"- **Best:** {best['dataset']} / {best['model']} ({best['overall_mIoU']:.1f}% overall)")
            lines.append(f"- **Worst:** {worst['dataset']} / {worst['model']} ({worst['overall_mIoU']:.1f}% overall)\n")
        
        # Model ranking
        model_df = per_model_analysis(stage1_df)
        if not model_df.empty:
            model_df = model_df.sort_values('overall_mIoU', ascending=False)
            lines.append("### Model Ranking (by overall mIoU, Stage 1)\n")
            for i, (_, row) in enumerate(model_df.iterrows(), 1):
                lines.append(f"{i}. **{row['model']}:** {row['overall_mIoU']:.1f}%")
            lines.append("")
        
        # Dataset ranking
        dataset_df = per_dataset_analysis(stage1_df)
        if not dataset_df.empty:
            dataset_df = dataset_df.sort_values('overall_mIoU', ascending=False)
            lines.append("### Dataset Ranking (by overall mIoU, Stage 1)\n")
            for i, (_, row) in enumerate(dataset_df.iterrows(), 1):
                lines.append(f"{i}. **{row['dataset']}:** {row['overall_mIoU']:.1f}%")
            lines.append("")
    
    return '\n'.join(lines)
    return '\n'.join(lines)


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Baseline Analysis')
    parser.add_argument('--output-dir', type=str, default='result_figures/baseline_consolidated',
                        help='Output directory for results')
    parser.add_argument('--stage1-root', type=str, default=WEIGHTS_ROOT_STAGE1,
                        help='Root directory for Stage 1 weights (clear_day training)')
    parser.add_argument('--stage2-root', type=str, default=WEIGHTS_ROOT_STAGE2,
                        help='Root directory for Stage 2 weights (all domains training)')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("COMPREHENSIVE BASELINE ANALYSIS")
    print("=" * 80)
    print(f"\nStage 1 root (clear_day): {args.stage1_root}")
    print(f"Stage 2 root (all domains): {args.stage2_root}")
    print(f"Output dir: {output_dir}\n")
    
    # Load Stage 1 baseline results (clear_day training)
    print("Loading Stage 1 baseline results (clear_day training)...")
    stage1_df = load_baseline_results(args.stage1_root, "stage1")
    print(f"  Found {len(stage1_df)} per-domain results")
    
    # Load Stage 2 baseline results (all domains training)
    print("Loading Stage 2 baseline results (all domains training)...")
    stage2_df = load_baseline_results(args.stage2_root, "stage2")
    print(f"  Found {len(stage2_df)} per-domain results")
    
    # Print summary
    print("\n" + "-" * 80)
    print("QUICK SUMMARY")
    print("-" * 80)
    
    # Print filtered results once
    if not stage1_df.empty:
        excluded = stage1_df[stage1_df['num_images'] < MIN_IMAGES_THRESHOLD]
        if len(excluded) > 0:
            print(f"\n  Note: {len(excluded)} Stage 1 results excluded (< {MIN_IMAGES_THRESHOLD} images):")
            for _, row in excluded.iterrows():
                print(f"    - {row['dataset']}/{row['model']}/{row['domain']}: {row['num_images']} images")
    
    if not stage1_df.empty:
        agg = compute_aggregates(stage1_df)
        print(f"\nStage 1 Baseline (Clear Day Training):")
        print(f"  Overall mIoU: {agg['overall_miou']:.2f}% ± {agg['overall_std']:.2f}")
        print(f"  Normal: {agg['normal_miou']:.2f}% | Adverse: {agg['adverse_miou']:.2f}% | Gap: {agg['domain_gap']:+.2f}%")
    
    if not stage2_df.empty:
        agg = compute_aggregates(stage2_df)
        print(f"\nStage 2 Baseline (All Domains Training):")
        print(f"  Overall mIoU: {agg['overall_miou']:.2f}% ± {agg['overall_std']:.2f}")
        print(f"  Normal: {agg['normal_miou']:.2f}% | Adverse: {agg['adverse_miou']:.2f}% | Gap: {agg['domain_gap']:+.2f}%")
    
    # Print per-domain table for Stage 1
    if not stage1_df.empty:
        print("\n" + "-" * 80)
        print("PER-DOMAIN PERFORMANCE (Stage 1 - Clear Day Training)")
        print("-" * 80)
        domain_df = per_domain_analysis(stage1_df)
        for _, row in domain_df.iterrows():
            print(f"  {row['domain']:12s} [{row['type']:7s}]: {row['mIoU_mean']:5.2f}% ± {row['mIoU_std']:.2f}")
    
    # Print per-model table
    if not stage1_df.empty:
        print("\n" + "-" * 80)
        print("PER-MODEL PERFORMANCE (Stage 1 - Clear Day Training)")
        print("-" * 80)
        model_df = per_model_analysis(stage1_df)
        model_df = model_df.sort_values('overall_mIoU', ascending=False)
        for _, row in model_df.iterrows():
            print(f"  {row['model']:25s}: {row['overall_mIoU']:5.2f}% (N:{row['normal_mIoU']:.1f}% A:{row['adverse_mIoU']:.1f}% Gap:{row['domain_gap']:+.1f}%)")
    
    # Print per-dataset table
    if not stage1_df.empty:
        print("\n" + "-" * 80)
        print("PER-DATASET PERFORMANCE (Stage 1 - Clear Day Training)")
        print("-" * 80)
        dataset_df = per_dataset_analysis(stage1_df)
        dataset_df = dataset_df.sort_values('overall_mIoU', ascending=False)
        for _, row in dataset_df.iterrows():
            print(f"  {row['dataset']:20s}: {row['overall_mIoU']:5.2f}% (N:{row['normal_mIoU']:.1f}% A:{row['adverse_mIoU']:.1f}% Gap:{row['domain_gap']:+.1f}%)")
    
    # Generate report
    print("\n" + "-" * 80)
    print("GENERATING REPORT")
    print("-" * 80)
    
    report = generate_report(stage1_df, stage2_df, output_dir)
    report_path = output_dir / "BASELINE_ANALYSIS_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Report saved to: {report_path}")
    
    # Save CSV data
    if not stage1_df.empty:
        stage1_df.to_csv(output_dir / "stage1_baseline_per_domain.csv", index=False)
        print(f"  Stage 1 baseline data saved to: {output_dir / 'stage1_baseline_per_domain.csv'}")
        
        config_df = per_config_analysis(stage1_df)
        config_df.to_csv(output_dir / "stage1_baseline_per_config.csv", index=False)
        print(f"  Stage 1 config summary saved to: {output_dir / 'stage1_baseline_per_config.csv'}")
    
    if not stage2_df.empty:
        stage2_df.to_csv(output_dir / "stage2_baseline_per_domain.csv", index=False)
        print(f"  Stage 2 baseline data saved to: {output_dir / 'stage2_baseline_per_domain.csv'}")
        
        config_df = per_config_analysis(stage2_df)
        config_df.to_csv(output_dir / "stage2_baseline_per_config.csv", index=False)
        print(f"  Stage 2 config summary saved to: {output_dir / 'stage2_baseline_per_config.csv'}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
