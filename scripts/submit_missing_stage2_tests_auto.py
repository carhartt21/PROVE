#!/usr/bin/env python3
"""
Submit missing Stage 2 test jobs.

Scans WEIGHTS_STAGE_2 for models with iter_80000.pth that need testing
and submits fine-grained test jobs for them.

Usage:
    python scripts/submit_missing_stage2_tests.py --dry-run   # Preview
    python scripts/submit_missing_stage2_tests.py --limit 20  # Submit up to 20
    python scripts/submit_missing_stage2_tests.py             # Submit all
"""

import os
import subprocess
import argparse
import re
import json
from pathlib import Path
from datetime import datetime

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WEIGHTS_ROOT = Path('/scratch/aaa_exchange/AWARE/WEIGHTS_STAGE_2')
LOG_DIR = PROJECT_ROOT / 'logs'

DATASETS = ['bdd10k', 'idd-aw', 'mapillaryvistas', 'outside15k']
DATASET_DISPLAY = {
    'bdd10k': 'BDD10k',
    'idd-aw': 'IDD-AW',
    'mapillaryvistas': 'MapillaryVistas',
    'outside15k': 'OUTSIDE15k',
}

# Stage 2 strategies (excluding cutmix, mixup, cyclediffusion)
STRATEGIES = [
    'gen_Attribute_Hallucination', 'gen_augmenters', 'gen_automold', 'gen_CNetSeg',
    'gen_CUT', 'gen_cycleGAN', 'gen_flux_kontext', 'gen_Img2Img', 'gen_IP2P',
    'gen_LANIT', 'gen_Qwen_Image_Edit', 'gen_stargan_v2', 'gen_step1x_new',
    'gen_step1x_v1p2', 'gen_SUSTechGAN', 'gen_TSIT', 'gen_UniControl',
    'gen_VisualCloze', 'gen_Weather_Effect_Generator', 'gen_albumentations_weather',
    'baseline', 'std_photometric_distort', 'std_autoaugment', 'std_randaugment'
]


def get_running_jobs():
    """Get list of running test jobs."""
    running_jobs = set()
    try:
        result = subprocess.run(
            ['bjobs', '-w', '-u', 'all'],
            capture_output=True, text=True, timeout=30
        )
        for line in result.stdout.split('\n')[1:]:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 7:
                stat = parts[2]
                if stat in ('RUN', 'PEND'):
                    job_name = parts[6].lower()
                    running_jobs.add(job_name)
    except Exception as e:
        print(f"Warning: Could not get running jobs: {e}")
    return running_jobs


def find_configs_needing_tests():
    """Find all Stage 2 configurations that need testing."""
    configs = []
    
    for strategy in STRATEGIES:
        strategy_path = WEIGHTS_ROOT / strategy
        if not strategy_path.exists():
            continue
        
        for dataset in DATASETS:
            dataset_path = strategy_path / dataset
            if not dataset_path.exists():
                continue
            
            try:
                for model_dir in dataset_path.iterdir():
                    if not model_dir.is_dir():
                        continue
                    
                    # Skip backup directories
                    if 'backup' in model_dir.name.lower() or 'old' in model_dir.name.lower():
                        continue
                    
                    checkpoint = model_dir / 'iter_80000.pth'
                    config_file = model_dir / 'training_config.py'
                    
                    if not checkpoint.exists() or not config_file.exists():
                        continue
                    
                    # Check if test results exist
                    has_results = False
                    for test_dir_name in ['test_results_detailed', 'test_results_detailed_fixed']:
                        test_dir = model_dir / test_dir_name
                        if test_dir.exists():
                            result_files = list(test_dir.glob('*/results.json'))
                            for rf in result_files:
                                try:
                                    with open(rf) as f:
                                        data = json.load(f)
                                    miou = data.get('overall', {}).get('mIoU')
                                    if miou is not None and miou > 0.05:
                                        has_results = True
                                        break
                                except:
                                    pass
                            if has_results:
                                break
                    
                    if not has_results:
                        configs.append({
                            'strategy': strategy,
                            'dataset': dataset,
                            'model': model_dir.name,
                            'weights_dir': model_dir,
                            'config_path': config_file,
                            'checkpoint_path': checkpoint,
                        })
            except (PermissionError, OSError) as e:
                print(f"Warning: Error scanning {dataset_path}: {e}")
    
    return configs


def submit_test_job(config, dry_run=False):
    """Submit a fine-grained test job."""
    strategy = config['strategy']
    dataset = config['dataset']
    model = config['model']
    weights_dir = config['weights_dir']
    config_path = config['config_path']
    checkpoint_path = config['checkpoint_path']
    
    dataset_display = DATASET_DISPLAY.get(dataset, dataset)
    
    # Create short job name
    short_strategy = strategy.replace('gen_', 'g').replace('std_', 's')[:10]
    short_dataset = dataset[:4]
    short_model = model[:3]
    job_name = f"fg2_{short_strategy}_{short_dataset}_{short_model}"  # fg2 = fine-grained Stage 2
    
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
    parser = argparse.ArgumentParser(description='Submit Stage 2 test jobs')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be submitted without actually submitting')
    parser.add_argument('--limit', type=int, default=None,
                       help='Maximum number of jobs to submit')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show verbose output')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Stage 2 Test Job Submission")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Get running jobs
    print("\nChecking running jobs...")
    running_jobs = get_running_jobs()
    print(f"Found {len(running_jobs)} running/pending jobs")
    
    # Find configs needing tests
    print("\nScanning for configurations needing tests...")
    configs = find_configs_needing_tests()
    print(f"Found {len(configs)} configurations needing tests")
    
    if not configs:
        print("\nNo configurations need testing!")
        return
    
    # Filter out configs with jobs already running
    configs_to_submit = []
    for c in configs:
        short_strategy = c['strategy'].replace('gen_', 'g').replace('std_', 's')[:10]
        short_dataset = c['dataset'][:4]
        short_model = c['model'][:3]
        job_name = f"fg2_{short_strategy}_{short_dataset}_{short_model}".lower()
        
        if job_name not in running_jobs:
            configs_to_submit.append(c)
    
    print(f"After filtering running jobs: {len(configs_to_submit)} to submit")
    
    if args.limit:
        configs_to_submit = configs_to_submit[:args.limit]
        print(f"Limited to: {len(configs_to_submit)} jobs")
    
    # Group by strategy for display
    from collections import defaultdict
    by_strategy = defaultdict(list)
    for c in configs_to_submit:
        by_strategy[c['strategy']].append(c)
    
    print(f"\nTo submit by strategy:")
    for strategy in sorted(by_strategy.keys()):
        print(f"  {strategy}: {len(by_strategy[strategy])}")
    
    # Submit jobs
    print(f"\n{'DRY RUN: ' if args.dry_run else ''}Submitting {len(configs_to_submit)} jobs...")
    
    submitted = 0
    failed = 0
    for config in configs_to_submit:
        if submit_test_job(config, dry_run=args.dry_run):
            submitted += 1
        else:
            failed += 1
    
    print(f"\n{'Would submit' if args.dry_run else 'Submitted'}: {submitted}")
    if failed > 0:
        print(f"Failed: {failed}")
    
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
