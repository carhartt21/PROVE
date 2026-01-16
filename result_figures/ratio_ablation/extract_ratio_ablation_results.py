#!/usr/bin/env python3
"""
Extract Ratio Ablation Results from Log Files

This script parses the test log files from the ratio ablation study and extracts
mIoU values for each ratio, iteration, dataset, and model configuration.

The ratio ablation study tests different ratios of synthetic:real data mixing:
- Ratios: 0, 12, 25, 37, 62, 75, 87, 100 (percentage of synthetic data)
- Strategies: gen_LANIT, gen_step1x_new, gen_automold, gen_TSIT, gen_NST
- Datasets: ACDC, BDD10K, IDD-AW, MapillaryVistas, OUTSIDE15k
- Models: deeplabv3plus_r50, pspnet_r50, segformer_mit-b5

Output: ratio_ablation_results.csv

Usage:
    mamba run -n prove python extract_ratio_ablation_results.py
"""

import os
import re
import pandas as pd
from pathlib import Path
from collections import defaultdict
import glob

# Log directory
LOGS_DIR = Path("/home/mima2416/repositories/PROVE/logs")
OUTPUT_DIR = Path("/home/mima2416/repositories/PROVE/result_figures/ratio_ablation")


def parse_ratio_value(ratio_str: str) -> float:
    """Convert ratio string to float (e.g., '0p25' -> 0.25, '25' -> 0.25)."""
    if 'p' in ratio_str:
        return float(ratio_str.replace('p', '.'))
    else:
        return float(ratio_str) / 100.0


def parse_log_filename(filename: str) -> dict:
    """Parse ratio ablation log filename to extract metadata.
    
    Format: test_ratio_gen_STRATEGY_DATASET_MODEL_[ratioXpXX_]JOBID.log
    Examples:
        - test_ratio_gen_LANIT_acdc_deeplabv3plus_r50_9223355.log
        - test_ratio_gen_LANIT_acdc_deeplabv3plus_r50_ratio0p25_9223356.log
        - test_ratio_gen_LANIT_idd-aw_segformer_mit-b5_ratio0p38_9223404.log
    """
    # Known model suffixes
    model_patterns = [
        'deeplabv3plus_r50',
        'pspnet_r50',
        'segformer_mit-b5'
    ]
    
    # Known datasets
    known_datasets = ['acdc', 'bdd10k', 'idd-aw', 'mapillaryvistas', 'outside15k']
    
    # Remove .log extension
    base = filename.replace('.log', '')
    
    # Check for ratio pattern
    ratio = 0.0  # Default for no ratio in filename
    ratio_match = re.search(r'_ratio(\d+p\d+)_', base)
    if ratio_match:
        ratio = parse_ratio_value(ratio_match.group(1))
        # Remove ratio from base for further parsing
        base = base.replace(f'_ratio{ratio_match.group(1)}', '')
    
    # Extract job ID (last number)
    job_match = re.search(r'_(\d+)$', base)
    if not job_match:
        return None
    job_id = job_match.group(1)
    base = base[:-(len(job_id) + 1)]  # Remove _JOBID
    
    # Now base should be: test_ratio_gen_STRATEGY_DATASET_MODEL
    # Remove prefix
    if not base.startswith('test_ratio_gen_'):
        return None
    base = base[len('test_ratio_gen_'):]
    
    # Find model (from the end)
    model = None
    for mp in model_patterns:
        if base.endswith(mp):
            model = mp
            base = base[:-(len(mp) + 1)]  # Remove _MODEL
            break
    
    if not model:
        return None
    
    # Now base should be: STRATEGY_DATASET
    # Find dataset (search from known datasets)
    dataset = None
    for ds in known_datasets:
        if base.endswith(ds):
            dataset = ds
            base = base[:-(len(ds) + 1)]  # Remove _DATASET
            break
    
    if not dataset:
        # Try to find dataset in the middle (for hyphenated datasets like idd-aw)
        for ds in known_datasets:
            if f'_{ds}_' in f'_{base}_' or base.endswith(ds):
                dataset = ds
                idx = base.rfind(ds)
                if idx > 0:
                    base = base[:idx-1]
                break
    
    if not dataset:
        return None
    
    # Strategy is what remains
    strategy = f"gen_{base}"
    
    return {
        'strategy': strategy,
        'dataset': dataset,
        'model': model,
        'ratio': ratio,
        'job_id': job_id
    }


def extract_results_from_log(log_path: Path) -> list:
    """Extract all mIoU results from a log file.
    
    Returns list of dicts with keys: iteration, mIoU, per_domain_results
    """
    results = []
    
    try:
        with open(log_path, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"  Error reading {log_path}: {e}")
        return results
    
    # Split by iteration testing sections
    iter_sections = re.split(r'Testing iter (\d+)\.\.\.', content)
    
    # iter_sections alternates: [preamble, iter1, content1, iter2, content2, ...]
    i = 1
    while i < len(iter_sections) - 1:
        iteration = int(iter_sections[i])
        section_content = iter_sections[i + 1]
        
        # Find the FINAL TEST RESULTS SUMMARY for this iteration
        final_match = re.search(
            r'FINAL TEST RESULTS SUMMARY[\s\S]*?mIoU:\s*(\d+\.?\d*)',
            section_content
        )
        
        if final_match:
            miou = float(final_match.group(1))
            
            # Extract per-domain results
            per_domain = {}
            domain_matches = re.findall(
                r'Testing domain:\s*(\w+)[\s\S]*?mIoU:\s*(\d+\.?\d*)',
                section_content
            )
            for domain, domain_miou in domain_matches:
                per_domain[domain] = float(domain_miou)
            
            results.append({
                'iteration': iteration,
                'mIoU': miou,
                'per_domain': per_domain
            })
        
        i += 2
    
    return results


def process_all_logs() -> pd.DataFrame:
    """Process all ratio ablation log files and create a DataFrame."""
    
    all_results = []
    
    # Find all test_ratio log files
    log_files = list(LOGS_DIR.glob("test_ratio_*.log"))
    print(f"Found {len(log_files)} ratio ablation log files")
    
    parsed_count = 0
    skipped_count = 0
    
    for log_path in sorted(log_files):
        filename = log_path.name
        
        # Parse filename
        metadata = parse_log_filename(filename)
        
        if metadata is None:
            skipped_count += 1
            continue
        
        parsed_count += 1
        
        # Extract results from log
        log_results = extract_results_from_log(log_path)
        
        if not log_results:
            continue
        
        # Add metadata and append to all_results
        for result in log_results:
            row = {
                'strategy': metadata['strategy'],
                'dataset': metadata['dataset'],
                'model': metadata['model'],
                'ratio': metadata['ratio'],
                'iteration': result['iteration'],
                'mIoU': result['mIoU'],
                'job_id': metadata['job_id'],
                'log_file': filename
            }
            
            # Add per-domain results
            for domain, domain_miou in result['per_domain'].items():
                row[f'mIoU_{domain}'] = domain_miou
            
            all_results.append(row)
    
    print(f"  Parsed: {parsed_count}, Skipped: {skipped_count}")
    
    return pd.DataFrame(all_results)


def add_best_iteration_results(df: pd.DataFrame) -> pd.DataFrame:
    """Add a column indicating best mIoU per configuration and which iteration achieved it."""
    
    # Group by configuration (strategy, dataset, model, ratio)
    groupby_cols = ['strategy', 'dataset', 'model', 'ratio']
    
    # Find best mIoU for each configuration
    best_results = df.groupby(groupby_cols)['mIoU'].max().reset_index()
    best_results.columns = groupby_cols + ['best_mIoU']
    
    # Merge back
    df = df.merge(best_results, on=groupby_cols)
    df['is_best'] = df['mIoU'] == df['best_mIoU']
    
    return df


def create_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a summary table showing best mIoU per strategy/dataset/model/ratio."""
    
    # Get best iteration for each configuration
    best_df = df[df['is_best']].copy()
    
    # If multiple iterations have the same best mIoU, take the latest
    best_df = best_df.sort_values('iteration', ascending=False).drop_duplicates(
        subset=['strategy', 'dataset', 'model', 'ratio']
    )
    
    # Select columns for summary
    summary_cols = ['strategy', 'dataset', 'model', 'ratio', 'iteration', 'mIoU']
    domain_cols = [c for c in best_df.columns if c.startswith('mIoU_')]
    
    return best_df[summary_cols + domain_cols].sort_values(
        ['strategy', 'dataset', 'model', 'ratio']
    )


def main():
    """Main function to extract and save ratio ablation results."""
    
    print("=" * 60)
    print("Extracting Ratio Ablation Results")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process all logs
    print("\nProcessing log files...")
    df = process_all_logs()
    
    if len(df) == 0:
        print("No results extracted!")
        return
    
    print(f"\nExtracted {len(df)} results:")
    print(f"  Strategies: {df['strategy'].nunique()} - {sorted(df['strategy'].unique())}")
    print(f"  Datasets: {df['dataset'].nunique()} - {sorted(df['dataset'].unique())}")
    print(f"  Models: {df['model'].nunique()} - {sorted(df['model'].unique())}")
    print(f"  Ratios: {sorted(df['ratio'].unique())}")
    print(f"  Iterations: {sorted(df['iteration'].unique())}")
    
    # Add best iteration markers
    df = add_best_iteration_results(df)
    
    # Save full results
    full_output = OUTPUT_DIR / 'ratio_ablation_full_results.csv'
    df.to_csv(full_output, index=False)
    print(f"\nSaved full results to: {full_output}")
    
    # Create and save summary table
    summary_df = create_summary_table(df)
    summary_output = OUTPUT_DIR / 'ratio_ablation_summary.csv'
    summary_df.to_csv(summary_output, index=False)
    print(f"Saved summary to: {summary_output}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    
    # Best ratio per strategy
    print("\nBest ratio per strategy (by mean mIoU across all configs):")
    strategy_ratio = df.groupby(['strategy', 'ratio'])['mIoU'].mean().reset_index()
    for strategy in df['strategy'].unique():
        strategy_data = strategy_ratio[strategy_ratio['strategy'] == strategy]
        best_idx = strategy_data['mIoU'].idxmax()
        best_row = strategy_data.loc[best_idx]
        print(f"  {strategy}: ratio={best_row['ratio']:.2f} (mean mIoU={best_row['mIoU']:.1f})")
    
    # Overall best configuration
    print("\nTop 5 configurations by mIoU:")
    top5 = df.nlargest(5, 'mIoU')[['strategy', 'dataset', 'model', 'ratio', 'iteration', 'mIoU']]
    for _, row in top5.iterrows():
        print(f"  {row['strategy']} / {row['dataset']} / {row['model']} / "
              f"ratio={row['ratio']:.2f} / iter={row['iteration']} => mIoU={row['mIoU']:.1f}")
    
    print("\n" + "=" * 60)
    print("✓ Extraction complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
