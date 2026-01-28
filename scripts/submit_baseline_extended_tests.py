#!/usr/bin/env python3
"""
Submit Extended Training Tests for Baseline Models

This script checks for completed baseline extended training checkpoints
and submits test jobs for any untested iterations.

Usage:
    python scripts/submit_baseline_extended_tests.py --dry-run
    python scripts/submit_baseline_extended_tests.py --submit
    python scripts/submit_baseline_extended_tests.py --submit --limit 20
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
WEIGHTS_EXTENDED = Path('/scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED')
BASELINE_DIR = WEIGHTS_EXTENDED / 'baseline'
JOBS_DIR = PROJECT_ROOT / 'jobs' / 'baseline_extended_tests'
LOGS_DIR = PROJECT_ROOT / 'logs'

# Baseline configurations matching the extended training study
CONFIGS = [
    ('bdd10k', 'pspnet_r50', 'BDD10k'),
    ('bdd10k', 'segformer_mit-b5', 'BDD10k'),
    ('iddaw', 'pspnet_r50', 'IDD-AW'),
    ('iddaw', 'segformer_mit-b5', 'IDD-AW'),
]

# Iterations to test (90k-320k, every 10k)
ITERATIONS = list(range(90000, 330000, 10000))


def get_tested_iterations(ckpt_dir: Path) -> set:
    """Get set of iterations that have been tested."""
    tested = set()
    test_results_dir = ckpt_dir / 'test_results_detailed'
    if test_results_dir.exists():
        for result_dir in test_results_dir.iterdir():
            # Check if results.json exists
            if (result_dir / 'results.json').exists():
                # Try to get iteration from checkpoint info in results
                # For now, assume all checkpoints in dir have been tested
                pass
    
    # Alternative: check for result files with iteration in name
    for result_file in ckpt_dir.glob('test_results_*/*/results.json'):
        # Parse iteration from result directory name if possible
        pass
    
    return tested


def get_available_checkpoints(ckpt_dir: Path) -> dict:
    """Get available checkpoints and their paths."""
    checkpoints = {}
    for ckpt in ckpt_dir.glob('iter_*.pth'):
        # Extract iteration number
        try:
            iter_num = int(ckpt.stem.split('_')[1])
            checkpoints[iter_num] = ckpt
        except (IndexError, ValueError):
            continue
    return checkpoints


def generate_test_job_script(dataset_dir: str, model: str, dataset_name: str, 
                             checkpoint: Path, iteration: int) -> str:
    """Generate LSF job script for testing a checkpoint."""
    
    job_name = f"test_baseline_{dataset_dir}_{model.replace('-', '_')}_{iteration//1000}k"
    output_dir = checkpoint.parent / 'test_results_detailed' / f'iter_{iteration}'
    config_path = checkpoint.parent / 'training_config.py'
    
    script = f"""#!/bin/bash
#BSUB -J {job_name}
#BSUB -o {LOGS_DIR}/{job_name}_%J.out
#BSUB -e {LOGS_DIR}/{job_name}_%J.err
#BSUB -q BatchGPU
#BSUB -W 2:00
#BSUB -n 4
#BSUB -R "rusage[mem=32000]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=24G"

# Test baseline extended training checkpoint
# Dataset: {dataset_name}
# Model: {model}
# Iteration: {iteration}

source /home/mima2416/miniconda3/etc/profile.d/conda.sh
conda activate prove

cd {PROJECT_ROOT}

echo "=========================================="
echo "Testing baseline extended checkpoint"
echo "=========================================="
echo "Dataset: {dataset_name}"
echo "Model: {model}"
echo "Iteration: {iteration}"
echo "Checkpoint: {checkpoint}"
echo "Output: {output_dir}"
echo "=========================================="

# Create output directory
mkdir -p "{output_dir}"

python fine_grained_test.py \\
    --config "{config_path}" \\
    --checkpoint "{checkpoint}" \\
    --dataset {dataset_name} \\
    --output-dir "{output_dir}"

echo "Testing complete!"
"""
    return script


def main():
    parser = argparse.ArgumentParser(description='Submit baseline extended training tests')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    parser.add_argument('--submit', action='store_true', help='Actually submit jobs')
    parser.add_argument('--limit', type=int, default=None, help='Max jobs to submit')
    parser.add_argument('--iterations', nargs='+', type=int, default=None,
                        help='Specific iterations to test (e.g., 160000 320000)')
    args = parser.parse_args()
    
    print("="*70)
    print("Baseline Extended Training - Test Job Submission")
    print("="*70)
    print(f"Baseline dir: {BASELINE_DIR}")
    print(f"Jobs dir: {JOBS_DIR}")
    
    # Create directories
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Determine iterations to test
    iterations_to_test = args.iterations if args.iterations else ITERATIONS
    
    # Find checkpoints needing tests
    jobs_to_submit = []
    
    for dataset_dir, model, dataset_name in CONFIGS:
        ckpt_dir = BASELINE_DIR / dataset_dir / model
        
        if not ckpt_dir.exists():
            print(f"\n⏳ Waiting for: {dataset_dir}/{model} (training not complete)")
            continue
        
        # Get available checkpoints
        checkpoints = get_available_checkpoints(ckpt_dir)
        
        if not checkpoints:
            print(f"\n⏳ No checkpoints yet: {dataset_dir}/{model}")
            continue
        
        print(f"\n✅ {dataset_dir}/{model}: {len(checkpoints)} checkpoints found")
        
        # Check each iteration
        for iteration in iterations_to_test:
            if iteration not in checkpoints:
                continue
            
            checkpoint = checkpoints[iteration]
            
            # Check if already tested - look for any results.json inside iteration dir
            iter_output_dir = ckpt_dir / 'test_results_detailed' / f'iter_{iteration}'
            has_results = False
            if iter_output_dir.exists():
                for timestamp_dir in iter_output_dir.iterdir():
                    if timestamp_dir.is_dir() and (timestamp_dir / 'results.json').exists():
                        has_results = True
                        break
            
            if has_results:
                print(f"   ✓ iter_{iteration}: already tested")
                continue
            
            # Generate job script
            script_content = generate_test_job_script(
                dataset_dir, model, dataset_name, checkpoint, iteration
            )
            
            job_file = JOBS_DIR / f'test_baseline_{dataset_dir}_{model}_{iteration}.sh'
            jobs_to_submit.append((job_file, script_content, iteration, dataset_dir, model))
    
    print(f"\n{'='*70}")
    print(f"Jobs to submit: {len(jobs_to_submit)}")
    print("="*70)
    
    if not jobs_to_submit:
        print("\nNo new tests needed!")
        return 0
    
    # Apply limit
    if args.limit and len(jobs_to_submit) > args.limit:
        print(f"Limiting to {args.limit} jobs")
        jobs_to_submit = jobs_to_submit[:args.limit]
    
    # Show or submit jobs
    submitted = 0
    for job_file, script_content, iteration, dataset_dir, model in jobs_to_submit:
        if args.dry_run:
            print(f"  Would submit: {job_file.name}")
        elif args.submit:
            # Write script
            with open(job_file, 'w') as f:
                f.write(script_content)
            
            # Submit job
            result = subprocess.run(['bsub'], input=script_content, capture_output=True, text=True)
            if result.returncode == 0:
                job_id = result.stdout.strip()
                print(f"  ✅ Submitted: {job_file.name} -> {job_id}")
                submitted += 1
            else:
                print(f"  ❌ Failed: {job_file.name}")
                print(f"     Error: {result.stderr}")
        else:
            print(f"  Pending: {job_file.name}")
    
    if args.submit:
        print(f"\n✅ Submitted {submitted} jobs")
    elif args.dry_run:
        print(f"\nDry run complete. Use --submit to actually submit jobs.")
    else:
        print(f"\nRun with --dry-run to preview or --submit to submit jobs.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
