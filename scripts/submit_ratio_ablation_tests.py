#!/usr/bin/env python3
"""Submit fine-grained test jobs for ratio ablation models.

This script submits test jobs for models with non-0.50 ratios in the WEIGHTS directory.
"""

import os
import re
import subprocess
import sys
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).parent.parent
WEIGHTS_ROOT = Path('/scratch/aaa_exchange/AWARE/WEIGHTS')
LOG_DIR = PROJECT_ROOT / 'logs'

DATASET_DISPLAY = {
    'bdd10k': 'BDD10k',
    'idd-aw': 'IDD-AW',
    'iddaw': 'IDD-AW',
    'mapillaryvistas': 'MapillaryVistas',
    'outside15k': 'OUTSIDE15k',
}


def get_running_jobs():
    """Get list of running/pending test job names."""
    running = set()
    try:
        result = subprocess.run(['bjobs', '-w'], capture_output=True, text=True, timeout=30)
        for line in result.stdout.strip().split('\n')[1:]:
            parts = line.split()
            if len(parts) >= 7:
                stat = parts[2]
                job_name = parts[6]
                if stat in ('RUN', 'PEND') and job_name.startswith('fg_'):
                    running.add(job_name.lower())
    except Exception as e:
        print(f"Warning: Could not get running jobs: {e}")
    return running


def find_ratio_ablation_configs():
    """Find all ratio ablation configs needing tests."""
    configs = []
    
    for strategy_path in WEIGHTS_ROOT.iterdir():
        if not strategy_path.is_dir():
            continue
        strategy = strategy_path.name
        
        for dataset_path in strategy_path.iterdir():
            if not dataset_path.is_dir():
                continue
            dataset = dataset_path.name
            
            for model_path in dataset_path.iterdir():
                if not model_path.is_dir():
                    continue
                if '_backup' in model_path.name:
                    continue
                
                # Only include non-0.50 ratios
                if 'ratio0p50' in model_path.name:
                    continue
                if 'ratio' not in model_path.name:
                    continue
                
                checkpoint = model_path / 'iter_80000.pth'
                config = model_path / 'training_config.py'
                
                if not checkpoint.exists() or not config.exists():
                    continue
                
                # Check if test results exist
                test_dir = model_path / 'test_results_detailed'
                has_results = False
                if test_dir.exists():
                    for subdir in test_dir.iterdir():
                        if subdir.is_dir() and (subdir / 'results.json').exists():
                            has_results = True
                            break
                
                if not has_results:
                    configs.append({
                        'strategy': strategy,
                        'dataset': dataset,
                        'model': model_path.name,
                        'weights_dir': model_path,
                        'config_path': config,
                        'checkpoint_path': checkpoint,
                    })
    
    return configs


def submit_job(config, dry_run=False):
    """Submit a test job."""
    strategy = config['strategy']
    dataset = config['dataset']
    model = config['model']
    weights_dir = config['weights_dir']
    config_path = config['config_path']
    checkpoint_path = config['checkpoint_path']
    
    dataset_display = DATASET_DISPLAY.get(dataset, dataset)
    
    # Short job name
    short_strategy = strategy.replace('gen_', 'g').replace('std_', 's')[:8]
    short_dataset = dataset[:4]
    short_model = model[:3]
    # Extract ratio
    ratio_match = re.search(r'ratio(0p\d+)', model)
    ratio_str = ratio_match.group(1) if ratio_match else ''
    job_name = f"fg_{short_strategy}_{short_dataset}_{short_model}_{ratio_str}"
    
    output_dir = weights_dir / 'test_results_detailed'
    
    cmd = [
        'bsub',
        '-J', job_name,
        '-q', 'BatchGPU',
        '-n', '10',
        '-R', 'span[hosts=1]',
        '-R', 'rusage[mem=8000]',
        '-gpu', 'num=1:gmem=8G:mode=shared',
        '-W', '0:30',
        '-o', f'{LOG_DIR}/{job_name}_%J.out',
        '-e', f'{LOG_DIR}/{job_name}_%J.err',
        f'source ~/.bashrc && conda activate prove && cd {PROJECT_ROOT} && python fine_grained_test.py --config {config_path} --checkpoint {checkpoint_path} --dataset {dataset_display} --output-dir {output_dir} --batch-size 8'
    ]
    
    if dry_run:
        print(f"  Would submit: {strategy}/{dataset}/{model}")
        return True
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            match = re.search(r'Job <(\d+)>', result.stdout)
            job_id = match.group(1) if match else 'unknown'
            print(f"  Submitted: {strategy}/{dataset}/{model} (Job {job_id})")
            return True
        else:
            print(f"  Failed: {strategy}/{dataset}/{model}")
            print(f"    Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"  Error: {strategy}/{dataset}/{model}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Submit ratio ablation test jobs')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be submitted')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of jobs')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Submit Ratio Ablation Test Jobs")
    print("=" * 60)
    print(f"Dry run: {args.dry_run}")
    if args.limit:
        print(f"Limit: {args.limit} jobs")
    print()
    
    print("Checking running jobs...")
    running_jobs = get_running_jobs()
    print(f"Found {len(running_jobs)} running/pending test jobs")
    print()
    
    print("Scanning for ratio ablation configs...")
    configs = find_ratio_ablation_configs()
    print(f"Found {len(configs)} configurations needing tests")
    print()
    
    if not configs:
        print("No configurations need testing!")
        return
    
    print("Submitting test jobs...")
    submitted = 0
    skipped = 0
    failed = 0
    
    for config in configs:
        if args.limit and submitted >= args.limit:
            print(f"\nReached limit of {args.limit} jobs")
            break
        
        # Check if job is already running
        short_strategy = config['strategy'].replace('gen_', 'g').replace('std_', 's')[:8]
        short_dataset = config['dataset'][:4]
        short_model = config['model'][:3]
        ratio_match = re.search(r'ratio(0p\d+)', config['model'])
        ratio_str = ratio_match.group(1) if ratio_match else ''
        job_name = f"fg_{short_strategy}_{short_dataset}_{short_model}_{ratio_str}".lower()
        
        if job_name in running_jobs:
            skipped += 1
            continue
        
        if submit_job(config, dry_run=args.dry_run):
            submitted += 1
        else:
            failed += 1
    
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Submitted: {submitted}")
    print(f"Skipped (already running): {skipped}")
    print(f"Failed: {failed}")
    print(f"Remaining: {len(configs) - submitted - skipped - failed}")


if __name__ == '__main__':
    main()
