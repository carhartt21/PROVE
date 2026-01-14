#!/usr/bin/env python3
"""
Auto-submit fine-grained test jobs for completed training checkpoints.

Scans the weights directory for models with iter_80000.pth that need testing
and submits test jobs for them.

Usage:
    python scripts/auto_submit_tests.py                    # Submit all missing tests
    python scripts/auto_submit_tests.py --main-only        # Only main datasets  
    python scripts/auto_submit_tests.py --dry-run          # Show what would be submitted
    python scripts/auto_submit_tests.py --limit 10         # Submit max 10 jobs
    python scripts/auto_submit_tests.py --stage testing    # Filter by stage
    python scripts/auto_submit_tests.py --shared-gpu       # Use shared GPU mode
"""

import os
import subprocess
import argparse
import time
import re
from pathlib import Path
from datetime import datetime

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WEIGHTS_ROOT = Path(os.environ.get('PROVE_WEIGHTS_ROOT', '/scratch/aaa_exchange/AWARE/WEIGHTS'))
LOG_DIR = PROJECT_ROOT / 'logs'

DATASETS = ['bdd10k', 'idd-aw', 'mapillaryvistas', 'outside15k']
DATASET_DISPLAY = {
    'bdd10k': 'BDD10k',
    'idd-aw': 'IDD-AW',
    'iddaw': 'IDD-AW',
    'mapillaryvistas': 'MapillaryVistas',
    'outside15k': 'OUTSIDE15k',
}

MAIN_DATASETS = ['bdd10k', 'mapillaryvistas']  # Primary evaluation datasets

# Strategies
GENERATIVE_STRATEGIES = [
    'gen_Attribute_Hallucination',
    'gen_augmenters',
    'gen_automold',
    'gen_CNetSeg',
    'gen_CUT',
    'gen_cyclediffusion',
    'gen_cycleGAN',
    'gen_flux_kontext',
    'gen_Img2Img',
    'gen_IP2P',
    'gen_LANIT',
    'gen_Qwen_Image_Edit',
    'gen_stargan_v2',
    'gen_step1x_new',
    'gen_step1x_v1p2',
    'gen_SUSTechGAN',
    'gen_TSIT',
    'gen_UniControl',
    'gen_VisualCloze',
    'gen_Weather_Effect_Generator',
    'gen_albumentations_weather',
]

STANDARD_STRATEGIES = [
    'baseline',
    'photometric_distort',
    'std_minimal',
    'std_autoaugment',
    'std_cutmix',
    'std_mixup',
    'std_randaugment',
]

ALL_STRATEGIES = GENERATIVE_STRATEGIES + STANDARD_STRATEGIES


def get_running_test_jobs():
    """Get list of currently running/pending test jobs.
    
    Returns:
        set: Set of (strategy, dataset, model) tuples for jobs that are RUN or PEND
    """
    running_jobs = set()
    
    try:
        # Use bjobs -w for full job names, filter by RUN and PEND only
        result = subprocess.run(
            ['bjobs', '-u', 'mima2416', '-w'],
            capture_output=True,
            text=True,
            timeout=15
        )
        
        for line in result.stdout.strip().split('\n')[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 3:
                job_name = parts[6] if len(parts) > 6 else parts[2]  # Job name is usually index 6 in -w output
                stat = parts[2]  # Status is at index 2
                
                # Only count RUN and PEND jobs (not DONE, EXIT, etc.)
                if stat not in ('RUN', 'PEND'):
                    continue
                
                # Check for fine-grained test jobs (fg_ prefix)
                if job_name.startswith('fg_') or 'retest' in job_name.lower():
                    # Try to extract strategy, dataset, model
                    for strat in ALL_STRATEGIES:
                        short_strat = strat.replace('gen_', 'g').replace('std_', 's')[:10]
                        if short_strat in job_name:
                            for ds in DATASETS + ['iddaw']:
                                short_ds = ds[:4]
                                if short_ds in job_name:
                                    running_jobs.add((strat, ds, job_name))
                                    break
                            break
    except subprocess.TimeoutExpired:
        print("Warning: bjobs timed out, assuming no running jobs")
    except Exception as e:
        print(f"Warning: Could not get running jobs: {e}")
    
    return running_jobs


def find_configs_needing_tests(main_only=False, stage=None):
    """Find all configurations that have checkpoints but need testing.
    
    Args:
        main_only: Only consider main datasets (BDD10k, MapillaryVistas)
        stage: Filter by training stage ('training', 'testing', None for all)
    
    Returns:
        list: List of (strategy, dataset, model, weights_dir) tuples needing tests
    """
    configs_needing_tests = []
    
    datasets_to_check = MAIN_DATASETS if main_only else DATASETS
    seen_configs = set()  # Track to avoid duplicates
    
    for strategy in ALL_STRATEGIES:
        strategy_path = WEIGHTS_ROOT / strategy
        if not strategy_path.exists():
            continue
        
        for dataset in datasets_to_check:
            # Handle different directory naming conventions
            ds_dir_candidates = [f"{dataset}_cd"]
            # Add alternate naming for idd-aw -> iddaw
            if '-' in dataset:
                ds_dir_candidates.append(f"{dataset.replace('-', '')}_cd")
            
            for ds_dir_name in ds_dir_candidates:
                dataset_path = strategy_path / ds_dir_name
                if not dataset_path.exists():
                    continue
                
                # Scan model directories
                try:
                    for model_dir in dataset_path.iterdir():
                        if not model_dir.is_dir():
                            continue
                        if model_dir.name.endswith('_backup'):
                            continue
                        
                        checkpoint_path = model_dir / 'iter_80000.pth'
                        config_path = model_dir / 'training_config.py'
                        
                        if not checkpoint_path.exists():
                            continue
                        if not config_path.exists():
                            continue
                        
                        # Check if test results exist
                        test_results_dir = model_dir / 'test_results_detailed'
                        has_valid_results = False
                        
                        if test_results_dir.exists():
                            # Check for results.json with valid mIoU
                            for result_subdir in test_results_dir.iterdir():
                                if result_subdir.is_dir() and result_subdir.name.startswith('202'):
                                    results_json = result_subdir / 'results.json'
                                    if results_json.exists():
                                        try:
                                            import json
                                            with open(results_json) as f:
                                                data = json.load(f)
                                            miou = data.get('overall', {}).get('mIoU')
                                            if miou is not None and miou > 0.05:  # > 5% mIoU is valid
                                                has_valid_results = True
                                                break
                                        except:
                                            pass
                        
                        if not has_valid_results:
                            # Create unique key to avoid duplicates
                            config_key = (strategy, dataset, model_dir.name)
                            if config_key in seen_configs:
                                continue
                            seen_configs.add(config_key)
                            
                            configs_needing_tests.append({
                                'strategy': strategy,
                                'dataset': dataset,
                                'model': model_dir.name,
                                'weights_dir': model_dir,
                                'config_path': config_path,
                                'checkpoint_path': checkpoint_path,
                            })
                except PermissionError:
                    continue
                except OSError as e:
                    print(f"Warning: Error scanning {dataset_path}: {e}")
                    continue
    
    return configs_needing_tests


def submit_test_job(config, dry_run=False, shared_gpu=True):
    """Submit a fine-grained test job for a configuration.
    
    Args:
        config: Dict with strategy, dataset, model, weights_dir, etc.
        dry_run: If True, just print what would be submitted
        shared_gpu: If True, use shared GPU mode for better scheduling
    
    Returns:
        bool: True if job was submitted (or would be in dry-run), False otherwise
    """
    strategy = config['strategy']
    dataset = config['dataset']
    model = config['model']
    weights_dir = config['weights_dir']
    config_path = config['config_path']
    checkpoint_path = config['checkpoint_path']
    
    # Get display name for dataset
    dataset_display = DATASET_DISPLAY.get(dataset, dataset)
    
    # Create short job name
    short_strategy = strategy.replace('gen_', 'g').replace('std_', 's')[:10]
    short_dataset = dataset[:4]
    short_model = model[:3]
    job_name = f"fg_{short_strategy}_{short_dataset}_{short_model}"
    
    output_dir = weights_dir / 'test_results_detailed'
    
    # Build bsub command
    gpu_spec = "num=1:gmem=8G:mode=shared" if shared_gpu else "num=1:gmem=8G"
    
    cmd = [
        'bsub',
        '-J', job_name,
        '-q', 'BatchGPU',
        '-n', '2',  # 2 CPU cores for I/O parallelism
        '-R', 'span[hosts=1]',
        '-R', 'rusage[mem=8000]',
        '-gpu', gpu_spec,
        '-W', '0:30',  # 30 minutes walltime (optimized batch inference)
        '-o', f'{LOG_DIR}/{job_name}_%J.out',
        '-e', f'{LOG_DIR}/{job_name}_%J.err',
        f'source ~/.bashrc && conda activate prove && cd {PROJECT_ROOT} && python fine_grained_test.py --config {config_path} --checkpoint {checkpoint_path} --dataset {dataset_display} --output-dir {output_dir} --batch-size 8'
    ]
    
    if dry_run:
        print(f"  Would submit: {strategy}/{dataset}/{model}")
        print(f"    Job name: {job_name}")
        return True
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            # Extract job ID from output
            match = re.search(r'Job <(\d+)>', result.stdout)
            job_id = match.group(1) if match else 'unknown'
            print(f"  Submitted: {strategy}/{dataset}/{model} (Job {job_id})")
            return True
        else:
            print(f"  Failed: {strategy}/{dataset}/{model}")
            print(f"    Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  Timeout submitting: {strategy}/{dataset}/{model}")
        return False
    except Exception as e:
        print(f"  Error: {strategy}/{dataset}/{model}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Auto-submit test jobs for completed training')
    parser.add_argument('--main-only', action='store_true',
                       help='Only process main datasets (BDD10k, MapillaryVistas)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be submitted without actually submitting')
    parser.add_argument('--limit', type=int, default=None,
                       help='Maximum number of jobs to submit')
    parser.add_argument('--stage', type=str, choices=['training', 'testing'],
                       help='Filter by stage')
    parser.add_argument('--shared-gpu', action='store_true', default=True,
                       help='Use shared GPU mode (default: True)')
    parser.add_argument('--no-shared-gpu', dest='shared_gpu', action='store_false',
                       help='Use exclusive GPU mode')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show verbose output')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Auto-Submit Fine-Grained Test Jobs")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Main datasets only: {args.main_only}")
    print(f"Dry run: {args.dry_run}")
    print(f"Shared GPU: {args.shared_gpu}")
    if args.limit:
        print(f"Limit: {args.limit} jobs")
    print()
    
    # Get currently running jobs
    print("Checking for running jobs...")
    running_jobs = get_running_test_jobs()
    print(f"Found {len(running_jobs)} running/pending test jobs")
    print()
    
    # Find configs needing tests
    print("Scanning for configurations needing tests...")
    configs = find_configs_needing_tests(main_only=args.main_only, stage=args.stage)
    print(f"Found {len(configs)} configurations needing tests")
    print()
    
    if not configs:
        print("No configurations need testing!")
        return
    
    # Submit jobs
    print("Submitting test jobs...")
    submitted = 0
    skipped = 0
    failed = 0
    
    for config in configs:
        if args.limit and submitted >= args.limit:
            print(f"\nReached limit of {args.limit} jobs")
            break
        
        # Skip if job is already running for this config
        strategy = config['strategy']
        dataset = config['dataset']
        model = config['model']
        
        # Check if already running (simplified check)
        is_running = any(
            strategy in job[0] and dataset in job[1]
            for job in running_jobs
        )
        
        if is_running:
            if args.verbose:
                print(f"  Skip (running): {strategy}/{dataset}/{model}")
            skipped += 1
            continue
        
        success = submit_test_job(config, dry_run=args.dry_run, shared_gpu=args.shared_gpu)
        if success:
            submitted += 1
        else:
            failed += 1
        
        # Small delay to avoid overwhelming scheduler
        if not args.dry_run:
            time.sleep(0.3)
    
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
