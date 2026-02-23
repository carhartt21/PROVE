#!/usr/bin/env python3
"""
DEPRECATED: Use scripts/auto_submit_tests.py --stage 2 instead.

This script has been replaced by the unified auto_submit_tests.py which supports
all stages (1, 2, cityscapes, cityscapes-gen) via the --stage argument.

Examples:
    python scripts/auto_submit_tests.py --stage 2 --dry-run
    python scripts/auto_submit_tests.py --stage 2 --include-ratio1p0
    python scripts/auto_submit_tests.py --stage 2 --limit 10
"""
import sys
print("DEPRECATED: Use 'python scripts/auto_submit_tests.py --stage 2' instead.", file=sys.stderr)
sys.exit(1)

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
WEIGHTS_ROOT_STAGE2 = Path(os.environ.get('PROVE_WEIGHTS_ROOT_STAGE2', '${AWARE_DATA_ROOT}/WEIGHTS_STAGE_2'))
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
    'std_photometric_distort',
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
        set: Set of job names (lowercase) for jobs that are RUN or PEND
    """
    running_jobs = set()
    
    try:
        # Use bjobs -w for full job names, filter by RUN and PEND only
        result = subprocess.run(
            ['bjobs', '-u', '${USER}', '-w'],
            capture_output=True,
            text=True,
            timeout=15
        )
        
        for line in result.stdout.strip().split('\n')[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 7:
                stat = parts[2]  # Status is at index 2
                job_name = parts[6]  # Job name is at index 6 in -w output
                
                # Only count RUN and PEND jobs (not DONE, EXIT, etc.)
                if stat not in ('RUN', 'PEND'):
                    continue
                
                # Check for Stage 2 fine-grained test jobs (fg2_ prefix)
                if job_name.startswith('fg2_'):
                    running_jobs.add(job_name.lower())
    except subprocess.TimeoutExpired:
        print("Warning: bjobs timed out, assuming no running jobs")
    except Exception as e:
        print(f"Warning: Could not get running jobs: {e}")
    
    return running_jobs


def find_configs_needing_tests(main_only=False, include_ratio1p0=False):
    """Find all Stage 2 configurations that have checkpoints but need testing.
    
    Args:
        main_only: Only consider main datasets (BDD10k, MapillaryVistas)
        include_ratio1p0: Include ratio1p0 models (real-only training)
    
    Returns:
        list: List of dicts with strategy, dataset, model, weights_dir, etc.
    """
    configs_needing_tests = []
    
    datasets_to_check = MAIN_DATASETS if main_only else DATASETS
    seen_configs = set()  # Track to avoid duplicates
    
    for strategy in ALL_STRATEGIES:
        strategy_path = WEIGHTS_ROOT_STAGE2 / strategy
        if not strategy_path.exists():
            continue
        
        for dataset in datasets_to_check:
            # Directory name is just the dataset (no _ad suffix anymore)
            ds_dir_candidates = [dataset]
            # Add alternate naming for idd-aw -> iddaw
            if '-' in dataset:
                ds_dir_candidates.append(dataset.replace('-', ''))
            
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
                        
                        # Filter by ratio type
                        # - Always include models without "ratio" in name (standard models)
                        # - Always include ratio0p50 (mixed real+gen training)
                        # - Include ratio1p0 only if flag is set (real-only training)
                        if 'ratio' in model_dir.name:
                            if 'ratio0p50' in model_dir.name:
                                pass  # Always include
                            elif 'ratio1p0' in model_dir.name and include_ratio1p0:
                                pass  # Include if flag set
                            else:
                                continue  # Skip other ratios
                        
                        config_path = model_dir / 'training_config.py'
                        
                        if not config_path.exists():
                            continue
                        
                        # Extract expected max_iters from training config
                        expected_max_iters = None
                        try:
                            with open(config_path, 'r') as f:
                                config_content = f.read()
                            # Parse max_iters from train_cfg = dict(max_iters=15000, ...)
                            import re
                            match = re.search(r'max_iters\s*=\s*(\d+)', config_content)
                            if match:
                                expected_max_iters = int(match.group(1))
                        except Exception as e:
                            pass
                        
                        if expected_max_iters is None:
                            continue  # Can't determine expected checkpoint
                        
                        # Look for the final checkpoint matching expected_max_iters
                        checkpoint_path = model_dir / f'iter_{expected_max_iters}.pth'
                        
                        if not checkpoint_path.exists():
                            continue  # Training not complete
                        
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
    """Submit a fine-grained test job for a Stage 2 configuration.
    
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
    
    # Create short job name with fg2_ prefix for Stage 2
    short_strategy = strategy.replace('gen_', 'g').replace('std_', 's')[:10]
    short_dataset = dataset[:4]
    short_model = model[:3]
    job_name = f"fg2_{short_strategy}_{short_dataset}_{short_model}"
    
    output_dir = weights_dir / 'test_results_detailed'
    
    # Build bsub command
    gpu_spec = "num=1:gmem=16G:mode=shared" if shared_gpu else "num=1:gmem=16G"
    
    cmd = [
        'bsub',
        '-J', job_name,
        '-q', 'BatchGPU',
        '-n', '10',  # 10 CPU cores for I/O parallelism
        '-gpu', gpu_spec,
        '-W', '0:30',  # 30 minutes walltime (optimized batch inference)
        '-o', f'{LOG_DIR}/{job_name}_%J.out',
        '-e', f'{LOG_DIR}/{job_name}_%J.err',
        f'source ~/.bashrc && mamba activate prove && cd {PROJECT_ROOT} && python fine_grained_test.py --config {config_path} --checkpoint {checkpoint_path} --dataset {dataset_display} --output-dir {output_dir} --batch-size 10'
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
    parser = argparse.ArgumentParser(description='Auto-submit Stage 2 test jobs for completed training')
    parser.add_argument('--main-only', action='store_true',
                       help='Only process main datasets (BDD10k, MapillaryVistas)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be submitted without actually submitting')
    parser.add_argument('--limit', type=int, default=None,
                       help='Maximum number of jobs to submit')
    parser.add_argument('--shared-gpu', action='store_true', default=True,
                       help='Use shared GPU mode (default: True)')
    parser.add_argument('--no-shared-gpu', dest='shared_gpu', action='store_false',
                       help='Use exclusive GPU mode')
    parser.add_argument('--include-ratio1p0', action='store_true',
                       help='Include ratio1p0 models (real-only training)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show verbose output')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Auto-Submit Stage 2 Fine-Grained Test Jobs")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Weights root: {WEIGHTS_ROOT_STAGE2}")
    print(f"Main datasets only: {args.main_only}")
    print(f"Include ratio1p0: {args.include_ratio1p0}")
    print(f"Dry run: {args.dry_run}")
    print(f"Shared GPU: {args.shared_gpu}")
    print(f"Main datasets only: {args.main_only}")
    print(f"Dry run: {args.dry_run}")
    print(f"Shared GPU: {args.shared_gpu}")
    if args.limit:
        print(f"Limit: {args.limit} jobs")
    print()
    
    # Get currently running jobs
    print("Checking for running jobs...")
    running_jobs = get_running_test_jobs()
    print(f"Found {len(running_jobs)} running/pending Stage 2 test jobs")
    print()
    
    # Find configs needing tests
    print("Scanning for configurations needing tests...")
    configs = find_configs_needing_tests(
        main_only=args.main_only,
        include_ratio1p0=args.include_ratio1p0
    )
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
        
        # Generate normalized key for matching
        # Jobs have pattern: fg2_<strat_prefix>_<dataset_prefix>_<model_prefix>
        strat_key = strategy.replace('gen_', 'g').replace('std_', 's').lower()[:8]
        dataset_key = dataset[:4].lower()
        model_key = model[:3].lower()
        
        # Check if any running job matches this config
        # Match if the job name contains all the key parts
        is_running = any(
            strat_key in job and dataset_key in job and model_key in job
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
