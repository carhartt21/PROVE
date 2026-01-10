#!/usr/bin/env python3
"""
Re-run fine-grained tests for BDD10k models with the fixed label handling.

Background:
- The old version of fine_grained_test.py (before Jan 10, 2026) had a bug where
  ALL datasets were processed with CityscapesLabelIdToTrainId conversion
- This corrupted BDD10k labels (which are already in trainID format):
  - trainID 0 (road) → 255 (ignored)
  - trainID 7 (traffic sign) → trainID 0 (road)
  - trainID 8 (vegetation) → trainID 1 (sidewalk)
  - trainID 13+ (car, etc.) → 255 (ignored)
- The bug was fixed in commit f6576fe (Jan 10, 2026) with proper per-dataset
  label handling via process_label_for_dataset()

This script re-runs fine-grained tests with the fixed code to get correct metrics.

Usage:
    # Dry run - show what would be done
    python scripts/retest_bdd10k_fine_grained.py --dry-run

    # Generate LSF jobs
    python scripts/retest_bdd10k_fine_grained.py

    # Submit specific strategy
    python scripts/retest_bdd10k_fine_grained.py --submit-strategy baseline
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Configuration
WEIGHTS_ROOT = Path('/scratch/aaa_exchange/AWARE/WEIGHTS')
DATA_ROOT = '/scratch/aaa_exchange/AWARE/FINAL_SPLITS'
PROVE_DIR = Path('/home/chge7185/repositories/PROVE')
JOBS_DIR = PROVE_DIR / 'scripts' / 'bdd10k_retest_jobs'
LOGS_DIR = '/scratch/aaa_exchange/AWARE/LOGS/retest_bdd10k'

# LSF configuration
LSF_TEMPLATE = """#!/bin/bash
#BSUB -J {job_name}
#BSUB -o {logs_dir}/{job_name}_%J.out
#BSUB -e {logs_dir}/{job_name}_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=8000]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=20G"
#BSUB -W 2:00
#BSUB -q BatchGPU

# BDD10k Fine-Grained Re-Test Job
# Strategy: {strategy}
# Model: {model_dir}
# Generated: {timestamp}

echo "Starting BDD10k fine-grained re-test"
echo "Strategy: {strategy}"
echo "Model: {model_dir}"
echo "Timestamp: $(date)"

# Activate environment
source /home/chge7185/.bashrc
mamba activate prove

# Change to PROVE directory
cd {prove_dir}

# Run fine-grained test
python fine_grained_test.py \\
    --config "{config_path}" \\
    --checkpoint "{checkpoint_path}" \\
    --output-dir "{output_dir}" \\
    --dataset BDD10k \\
    --data-root {data_root} \\
    --test-split test

echo "Re-test completed at $(date)"
"""


def find_bdd10k_models(skip_completed=True):
    """Find all BDD10k model directories.
    
    Args:
        skip_completed: If True, skip models that already have fixed test results
    """
    models = []
    skipped = []
    
    # Find all bdd10k model directories
    for strategy_dir in WEIGHTS_ROOT.iterdir():
        if not strategy_dir.is_dir():
            continue
        strategy = strategy_dir.name
        
        bdd10k_dir = strategy_dir / 'bdd10k'
        try:
            if not bdd10k_dir.exists():
                continue
        except PermissionError:
            print(f"  Warning: Permission denied for {bdd10k_dir}")
            continue
        
        try:
            model_dirs = list(bdd10k_dir.iterdir())
        except PermissionError:
            print(f"  Warning: Permission denied for {bdd10k_dir}")
            continue
            
        for model_dir in model_dirs:
            if not model_dir.is_dir():
                continue
            
            # Check if fixed results already exist
            fixed_results_dir = model_dir / 'test_results_detailed_fixed'
            if skip_completed and fixed_results_dir.exists():
                # Check if results.json exists in any subdirectory
                results_files = list(fixed_results_dir.glob('*/results.json'))
                if results_files:
                    skipped.append(f"{strategy}/{model_dir.name}")
                    continue
            
            # Find config and checkpoint
            config_path = model_dir / 'training_config.py'
            checkpoint_path = model_dir / 'iter_80000.pth'
            
            # Also check for best checkpoint
            if not checkpoint_path.exists():
                checkpoint_path = model_dir / 'best_mIoU_iter_*.pth'
                checkpoints = list(model_dir.glob('best_mIoU_iter_*.pth'))
                if checkpoints:
                    checkpoint_path = checkpoints[0]
                else:
                    checkpoints = list(model_dir.glob('iter_*.pth'))
                    if checkpoints:
                        checkpoint_path = sorted(checkpoints)[-1]
                    else:
                        print(f"  Warning: No checkpoint found in {model_dir}")
                        continue
            
            if not config_path.exists():
                print(f"  Warning: No config found in {model_dir}")
                continue
            
            models.append({
                'strategy': strategy,
                'model_name': model_dir.name,
                'model_dir': model_dir,
                'config_path': config_path,
                'checkpoint_path': checkpoint_path,
            })
    
    if skipped:
        print(f"\nSkipped {len(skipped)} models with existing fixed results:")
        for s in skipped:
            print(f"  {s}")
    
    return models


def generate_job_script(model_info, timestamp):
    """Generate an LSF job script for a model."""
    strategy = model_info['strategy']
    model_name = model_info['model_name']
    model_dir = model_info['model_dir']
    config_path = model_info['config_path']
    checkpoint_path = model_info['checkpoint_path']
    
    job_name = f"retest_bdd10k_{strategy}_{model_name}"
    output_dir = model_dir / 'test_results_detailed_fixed'
    
    script_content = LSF_TEMPLATE.format(
        job_name=job_name,
        logs_dir=LOGS_DIR,
        strategy=strategy,
        model_dir=model_dir,
        timestamp=timestamp,
        prove_dir=PROVE_DIR,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        data_root=DATA_ROOT,
    )
    
    return job_name, script_content


def main():
    parser = argparse.ArgumentParser(description='Re-test BDD10k models with fixed fine_grained_test.py')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without creating jobs')
    parser.add_argument('--submit-strategy', type=str, help='Submit jobs for a specific strategy')
    parser.add_argument('--submit-all', action='store_true', help='Submit all jobs immediately')
    parser.add_argument('--force', action='store_true', help='Re-run even if fixed results already exist')
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("Finding BDD10k models...")
    skip_completed = not args.force
    models = find_bdd10k_models(skip_completed=skip_completed)
    print(f"Found {len(models)} BDD10k models to re-test\n")
    
    if args.dry_run:
        print("DRY RUN - would generate jobs for:")
        for model in models:
            print(f"  {model['strategy']}/{model['model_name']}")
            print(f"    Config: {model['config_path']}")
            print(f"    Checkpoint: {model['checkpoint_path']}")
        return
    
    # Create jobs directory
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create logs directory
    logs_dir = Path('/scratch/aaa_exchange/AWARE/LOGS/retest_bdd10k')
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate job scripts
    job_scripts = []
    for model in models:
        job_name, script_content = generate_job_script(model, timestamp)
        script_path = JOBS_DIR / f"{job_name}.sh"
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)
        
        job_scripts.append({
            'strategy': model['strategy'],
            'model_name': model['model_name'],
            'script_path': script_path,
        })
        print(f"Created: {script_path}")
    
    print(f"\nGenerated {len(job_scripts)} job scripts in {JOBS_DIR}")
    
    # Submit jobs if requested
    if args.submit_all or args.submit_strategy:
        jobs_to_submit = job_scripts
        if args.submit_strategy:
            jobs_to_submit = [j for j in job_scripts if j['strategy'] == args.submit_strategy]
            print(f"\nSubmitting {len(jobs_to_submit)} jobs for strategy: {args.submit_strategy}")
        else:
            print(f"\nSubmitting all {len(jobs_to_submit)} jobs...")
        
        submitted = 0
        for job in jobs_to_submit:
            try:
                # Submit using LSF bsub
                with open(job['script_path'], 'r') as f:
                    script_content = f.read()
                result = subprocess.run(
                    ['bsub'],
                    input=script_content,
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print(f"  Submitted: {job['strategy']}/{job['model_name']}")
                    submitted += 1
                else:
                    print(f"  Failed: {job['strategy']}/{job['model_name']} - {result.stderr}")
            except FileNotFoundError:
                print("  Error: bsub not found. Are you on a node with LSF access?")
                break
        
        print(f"\nSubmitted {submitted} jobs")
    else:
        print("\nTo submit jobs:")
        print(f"  Submit all: python {sys.argv[0]} --submit-all")
        print(f"  Submit specific: python {sys.argv[0]} --submit-strategy baseline")
        print("\nOr manually submit:")
        print(f"  for f in {JOBS_DIR}/*.sh; do bsub < \"$f\"; done")


if __name__ == '__main__':
    main()
