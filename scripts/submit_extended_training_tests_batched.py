#!/usr/bin/env python3
"""
Submit extended training tests in batched form.

This script groups test jobs to minimize LSF submissions while maximizing efficiency.

Strategy:
- Group tests by model directory (each directory has 24 checkpoints)
- Each job tests all iterations for ONE model directory
- Total: ~40 jobs (one per config)

Alternative: --batch-by-iteration mode
- Group tests by iteration value
- Each job tests ALL configs at ONE iteration
- Total: ~24 jobs

Usage:
    # Dry run to see what would be submitted
    python submit_extended_training_tests_batched.py --dry-run
    
    # Submit with model-based batching (40 jobs)
    python submit_extended_training_tests_batched.py --batch-by-model
    
    # Submit with iteration-based batching (24 jobs)  
    python submit_extended_training_tests_batched.py --batch-by-iteration
    
    # Limit number of jobs
    python submit_extended_training_tests_batched.py --limit 10
"""

import os
import sys
import re
import json
import subprocess
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Configuration
WEIGHTS_ROOT = Path("/scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED")
PROVE_ROOT = Path("/home/mima2416/repositories/PROVE")
LOGS_DIR = PROVE_ROOT / "logs" / "extended_tests_batched"

# Dataset name mapping for fine_grained_test.py
DATASET_MAP = {
    "bdd10k": "BDD10k",
    "bdd10k_ad": "BDD10k",
    "idd-aw": "IDD-AW",
    "iddaw_ad": "IDD-AW",
    "mapillaryvistas_ad": "MapillaryVistas",
    "outside15k_ad": "OUTSIDE15k",
}


def find_all_checkpoints():
    """Find all checkpoints and their test status."""
    checkpoints = []
    
    for ckpt in WEIGHTS_ROOT.rglob("iter_*.pth"):
        match = re.search(r"iter_(\d+)\.pth", ckpt.name)
        if not match:
            continue
        iteration = int(match.group(1))
        
        model_dir = ckpt.parent
        config_path = model_dir / "training_config.py"
        
        if not config_path.exists():
            continue
        
        relative = model_dir.relative_to(WEIGHTS_ROOT)
        parts = list(relative.parts)
        if len(parts) < 3:
            continue
        
        strategy = parts[0]
        dataset_dir = parts[1]
        model = parts[2]
        
        # Check if this specific iteration has been tested
        # Look for test results that reference this checkpoint
        tested = False
        test_results_dir = model_dir / "test_results_detailed"
        if test_results_dir.exists():
            for result_file in test_results_dir.rglob("results.json"):
                try:
                    with open(result_file) as f:
                        data = json.load(f)
                    ckpt_path = data.get("config", {}).get("checkpoint_path", "")
                    if f"iter_{iteration}" in ckpt_path:
                        tested = True
                        break
                except:
                    continue
        
        if not tested:
            checkpoints.append({
                "strategy": strategy,
                "dataset_dir": dataset_dir,
                "dataset": DATASET_MAP.get(dataset_dir, dataset_dir),
                "model": model,
                "iteration": iteration,
                "checkpoint": str(ckpt),
                "config": str(config_path),
                "model_dir": str(model_dir),
            })
    
    return checkpoints


def group_by_model(checkpoints):
    """Group checkpoints by model directory."""
    groups = defaultdict(list)
    for ckpt in checkpoints:
        key = (ckpt["strategy"], ckpt["dataset_dir"], ckpt["model"])
        groups[key].append(ckpt)
    return groups


def group_by_iteration(checkpoints):
    """Group checkpoints by iteration value."""
    groups = defaultdict(list)
    for ckpt in checkpoints:
        groups[ckpt["iteration"]].append(ckpt)
    return groups


def generate_batch_script(tests, batch_id, mode):
    """Generate a batch job script that runs multiple tests sequentially."""
    
    if mode == "model":
        # All tests for one model - use first test's info for naming
        first = tests[0]
        job_name = f"ext_batch_{first['strategy'][:8]}_{first['dataset_dir'][:4]}_{first['model'].split('_')[0][:4]}"
        iterations = sorted([t["iteration"] for t in tests])
        iter_str = f"{len(iterations)}iters"
    else:
        # All tests for one iteration
        iteration = tests[0]["iteration"]
        job_name = f"ext_iter_{iteration // 1000}k_{len(tests)}cfgs"
        iter_str = f"iter{iteration // 1000}k"
    
    # Build test commands
    test_commands = []
    for test in tests:
        # Include iteration in output dir name to avoid collisions
        # fine_grained_test.py creates timestamp subdirs, but checkpoint path
        # is recorded in the result, so we can distinguish results
        output_dir = f"{test['model_dir']}/test_results_detailed"
        
        cmd = f"""echo "Testing {test['strategy']}/{test['dataset_dir']}/{test['model']} iter_{test['iteration']}"
python fine_grained_test.py \\
    --config "{test['config']}" \\
    --checkpoint "{test['checkpoint']}" \\
    --dataset {test['dataset']} \\
    --output-dir "{output_dir}" || echo "FAILED: iter_{test['iteration']}"
"""
        test_commands.append(cmd)
    
    # Combine into single script with sequential execution
    test_block = "\n".join(test_commands)
    
    # Estimate time: ~5 min per test
    estimated_time = len(tests) * 5
    hours = estimated_time // 60 + 1
    
    script = f"""#!/bin/bash
#BSUB -J {job_name}
#BSUB -q BatchGPU
#BSUB -n 10
#BSUB -gpu "num=1:gmem=16G"
#BSUB -W {hours}:00
#BSUB -o {LOGS_DIR}/{job_name}_%J.out
#BSUB -e {LOGS_DIR}/{job_name}_%J.err

cd {PROVE_ROOT}
source ~/.bashrc
mamba activate prove

echo "Starting batch of {len(tests)} tests at $(date)"
echo "=========================================="

{test_block}

echo "=========================================="
echo "Batch completed at $(date)"
"""
    
    return script, job_name


def submit_jobs(groups, mode, dry_run=False, limit=None):
    """Submit batch jobs."""
    
    # Create logs directory
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    jobs = list(groups.items())
    if limit:
        jobs = jobs[:limit]
    
    submitted = 0
    total_tests = 0
    
    for i, (key, tests) in enumerate(jobs):
        script, job_name = generate_batch_script(tests, i, mode)
        total_tests += len(tests)
        
        if dry_run:
            if mode == "model":
                print(f"[DRY-RUN] {job_name}: {len(tests)} iterations")
                print(f"  Config: {key[0]}/{key[1]}/{key[2]}")
            else:
                print(f"[DRY-RUN] {job_name}: {len(tests)} configs at iter_{key}")
            continue
        
        # Write and submit
        script_path = LOGS_DIR / f"batch_{i}.lsf"
        with open(script_path, "w") as f:
            f.write(script)
        
        try:
            result = subprocess.run(
                f"bsub < {script_path}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            if "is submitted" in result.stdout:
                print(f"Submitted: {job_name} ({len(tests)} tests)")
                submitted += 1
            else:
                print(f"Failed: {job_name}")
                print(f"  {result.stderr}")
        except Exception as e:
            print(f"Error submitting {job_name}: {e}")
    
    return submitted, total_tests


def main():
    parser = argparse.ArgumentParser(description="Submit extended training tests in batched form")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be submitted")
    parser.add_argument("--batch-by-model", action="store_true", 
                        help="Group by model directory (40 jobs, ~24 tests each)")
    parser.add_argument("--batch-by-iteration", action="store_true",
                        help="Group by iteration (24 jobs, ~40 tests each)")
    parser.add_argument("--limit", type=int, help="Limit number of jobs to submit")
    args = parser.parse_args()
    
    # Default to model-based batching
    if not args.batch_by_model and not args.batch_by_iteration:
        args.batch_by_model = True
    
    print("Scanning WEIGHTS_EXTENDED for untested checkpoints...")
    checkpoints = find_all_checkpoints()
    print(f"Found {len(checkpoints)} untested checkpoints")
    
    if not checkpoints:
        print("All checkpoints have been tested!")
        return 0
    
    if args.batch_by_model:
        print("\nGrouping by model directory...")
        groups = group_by_model(checkpoints)
        mode = "model"
    else:
        print("\nGrouping by iteration...")
        groups = group_by_iteration(checkpoints)
        mode = "iteration"
    
    print(f"Created {len(groups)} batch groups")
    
    # Show summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total untested checkpoints: {len(checkpoints)}")
    print(f"Number of batch jobs: {len(groups)}")
    print(f"Average tests per job: {len(checkpoints) / len(groups):.1f}")
    
    if args.limit:
        print(f"Limiting to {args.limit} jobs")
    
    print("="*60 + "\n")
    
    submitted, total_tests = submit_jobs(groups, mode, args.dry_run, args.limit)
    
    if args.dry_run:
        print(f"\n[DRY-RUN] Would submit {len(groups) if not args.limit else args.limit} jobs")
        print(f"[DRY-RUN] Total tests: {total_tests}")
    else:
        print(f"\nSubmitted {submitted} jobs covering {total_tests} tests")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
