#!/usr/bin/env python3
"""
Extract Ratio Ablation Results from Test Result JSON Files

This script extracts mIoU values directly from the results.json files in the
WEIGHTS_RATIO_ABLATION directory structure.

Directory structure:
/scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION/
    gen_STRATEGY/
        DATASET/
            MODEL[_ratioXpXX]/
                test_results/
                    TIMESTAMP/
                        results.json

Output: ratio_ablation_results.csv

Usage:
    mamba run -n prove python extract_from_json.py
"""

import os
import re
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Base directory
BASE_DIR = Path("/scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION")
OUTPUT_DIR = Path("/home/mima2416/repositories/PROVE/result_figures/ratio_ablation")

# Known values for validation
VALID_STRATEGIES = ['gen_LANIT', 'gen_step1x_new', 'gen_automold', 'gen_TSIT', 'gen_NST']
VALID_DATASETS = ['acdc', 'bdd10k', 'idd-aw', 'mapillaryvistas', 'outside15k']
VALID_MODELS = ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5']


def parse_model_dir(model_dir: str) -> tuple:
    """Parse model directory name to extract model and ratio.
    
    Examples:
        - 'deeplabv3plus_r50' -> ('deeplabv3plus_r50', 0.0)
        - 'deeplabv3plus_r50_ratio0p25' -> ('deeplabv3plus_r50', 0.25)
        - 'segformer_mit-b5_ratio0p38' -> ('segformer_mit-b5', 0.38)
    """
    # Check for ratio suffix
    ratio_match = re.search(r'_ratio(\d+p\d+)$', model_dir)
    
    if ratio_match:
        ratio_str = ratio_match.group(1)
        ratio = float(ratio_str.replace('p', '.'))
        model = model_dir[:ratio_match.start()]
    else:
        ratio = 0.0
        model = model_dir
    
    return model, ratio


def extract_iteration_from_checkpoint(checkpoint_path: str) -> int:
    """Extract iteration number from checkpoint path."""
    match = re.search(r'iter_(\d+)\.pth', checkpoint_path)
    if match:
        return int(match.group(1))
    return 0


def process_results_json(json_path: Path) -> dict:
    """Extract relevant data from a results.json file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Error reading {json_path}: {e}")
        return None
    
    config = data.get('config', {})
    overall = data.get('overall', {})
    per_domain = data.get('per_domain', {})
    
    # Check if we have actual results
    if not overall or 'mIoU' not in overall:
        return None
    
    # Extract checkpoint path to get iteration
    checkpoint_path = config.get('checkpoint_path', '')
    iteration = extract_iteration_from_checkpoint(checkpoint_path)
    
    # Extract domain results
    domain_mious = {}
    for domain, domain_data in per_domain.items():
        if 'summary' in domain_data and 'mIoU' in domain_data['summary']:
            domain_mious[domain] = domain_data['summary']['mIoU']
    
    return {
        'iteration': iteration,
        'mIoU': overall.get('mIoU', 0),
        'aAcc': overall.get('aAcc', 0),
        'fwIoU': overall.get('fwIoU', 0),
        'num_images': overall.get('num_images', 0),
        'domain_mious': domain_mious,
        'dataset_from_config': config.get('dataset', ''),
        'timestamp': config.get('timestamp', ''),
    }


def scan_directory() -> pd.DataFrame:
    """Scan the entire directory structure and collect results."""
    
    all_results = []
    
    print(f"Scanning: {BASE_DIR}")
    
    # Iterate through strategies
    for strategy_dir in sorted(BASE_DIR.iterdir()):
        if not strategy_dir.is_dir():
            continue
        
        strategy = strategy_dir.name
        print(f"\n  Strategy: {strategy}")
        
        # Iterate through datasets
        for dataset_dir in sorted(strategy_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            
            dataset = dataset_dir.name.lower()
            
            # Iterate through model configurations
            for model_dir in sorted(dataset_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                
                model, ratio = parse_model_dir(model_dir.name)
                
                # Check for test_results directory
                test_results_dir = model_dir / 'test_results'
                if not test_results_dir.exists():
                    continue
                
                # Iterate through timestamp directories
                for timestamp_dir in sorted(test_results_dir.iterdir()):
                    if not timestamp_dir.is_dir():
                        continue
                    
                    results_json = timestamp_dir / 'results.json'
                    if not results_json.exists():
                        continue
                    
                    # Extract results
                    result = process_results_json(results_json)
                    
                    if result is None:
                        continue
                    
                    # Build row
                    row = {
                        'strategy': strategy,
                        'dataset': dataset,
                        'model': model,
                        'ratio': ratio,
                        'iteration': result['iteration'],
                        'mIoU': result['mIoU'],
                        'aAcc': result['aAcc'],
                        'fwIoU': result['fwIoU'],
                        'num_images': result['num_images'],
                        'timestamp': result['timestamp'],
                    }
                    
                    # Add per-domain mIoU
                    for domain, miou in result['domain_mious'].items():
                        row[f'mIoU_{domain}'] = miou
                    
                    all_results.append(row)
    
    return pd.DataFrame(all_results)


def add_best_iteration_markers(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns indicating best mIoU for each configuration."""
    
    # Group by configuration
    groupby_cols = ['strategy', 'dataset', 'model', 'ratio']
    
    # Find best mIoU
    best_results = df.groupby(groupby_cols)['mIoU'].max().reset_index()
    best_results.columns = groupby_cols + ['best_mIoU']
    
    # Merge back
    df = df.merge(best_results, on=groupby_cols)
    df['is_best'] = df['mIoU'] == df['best_mIoU']
    
    return df


def create_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create a summary table with best results per configuration."""
    
    best_df = df[df['is_best'] == True].copy()
    
    # Take latest if multiple have same mIoU
    best_df = best_df.sort_values('iteration', ascending=False).drop_duplicates(
        subset=['strategy', 'dataset', 'model', 'ratio']
    )
    
    return best_df.sort_values(['strategy', 'dataset', 'model', 'ratio'])


def main():
    """Main extraction function."""
    
    print("=" * 70)
    print("Extracting Ratio Ablation Results from JSON Files")
    print("=" * 70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Scan directories
    df = scan_directory()
    
    if len(df) == 0:
        print("\nNo results found!")
        return
    
    print(f"\n{'=' * 70}")
    print(f"Extracted {len(df)} results")
    print(f"{'=' * 70}")
    
    # Statistics
    print(f"\nStrategies: {df['strategy'].nunique()} - {sorted(df['strategy'].unique())}")
    print(f"Datasets: {df['dataset'].nunique()} - {sorted(df['dataset'].unique())}")
    print(f"Models: {df['model'].nunique()} - {sorted(df['model'].unique())}")
    print(f"Ratios: {sorted(df['ratio'].unique())}")
    print(f"Iterations: {sorted(df['iteration'].unique())}")
    
    # Add best iteration markers
    df = add_best_iteration_markers(df)
    
    # Save full results
    full_output = OUTPUT_DIR / 'ratio_ablation_full_results.csv'
    df.to_csv(full_output, index=False)
    print(f"\nSaved full results to: {full_output}")
    
    # Create and save summary
    summary_df = create_summary(df)
    summary_output = OUTPUT_DIR / 'ratio_ablation_summary.csv'
    summary_df.to_csv(summary_output, index=False)
    print(f"Saved summary to: {summary_output}")
    
    # Print summary statistics
    print(f"\n{'=' * 70}")
    print("Key Statistics")
    print(f"{'=' * 70}")
    
    # Results per dataset
    print("\nResults per dataset:")
    for dataset in sorted(df['dataset'].unique()):
        dataset_df = df[df['dataset'] == dataset]
        print(f"  {dataset}: {len(dataset_df)} results, "
              f"mIoU range: {dataset_df['mIoU'].min():.1f} - {dataset_df['mIoU'].max():.1f}")
    
    # Best ratio per strategy-dataset combination
    print("\nBest ratio per strategy-dataset:")
    best_df = df[df['is_best'] == True]
    for strategy in sorted(df['strategy'].unique()):
        print(f"\n  {strategy}:")
        for dataset in sorted(df['dataset'].unique()):
            config_df = best_df[(best_df['strategy'] == strategy) & (best_df['dataset'] == dataset)]
            if len(config_df) > 0:
                best_row = config_df.loc[config_df['mIoU'].idxmax()]
                print(f"    {dataset}: ratio={best_row['ratio']:.2f}, mIoU={best_row['mIoU']:.1f}")
    
    print(f"\n{'=' * 70}")
    print("✓ Extraction complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
