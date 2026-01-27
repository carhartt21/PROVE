#!/usr/bin/env python3
"""
Submit tests for ALL iterations of extended training models (90k-320k).
This completes the extended training evaluation across all checkpoints.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import re
from datetime import datetime

# Configuration
WEIGHTS_ROOT = Path("/scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED")
PROVE_ROOT = Path("/home/mima2416/repositories/PROVE")

# Extended training iterations (90k to 320k, every 10k)
ITERATIONS = list(range(90000, 330000, 10000))

# Dataset mappings
DATASET_MAPPING = {
    "bdd10k": "BDD10k",
    "idd-aw": "IDD-AW",
    "outside15k": "OUTSIDE15k",
    "mapillaryvistas": "MapillaryVistas",
    "bdd10k_ad": "BDD10k",
    "iddaw_ad": "IDD-AW",
    "idd-aw_ad": "IDD-AW",
    "outside15k_ad": "OUTSIDE15k",
    "mapillaryvistas_ad": "MapillaryVistas",
}

NATIVE_CLASS_DATASETS = ["MapillaryVistas", "OUTSIDE15k"]


def find_missing_tests():
    """Find all extended training checkpoints that need testing."""
    missing_tests = []
    
    # Find all model directories with checkpoints
    model_dirs = set()
    for ckpt in WEIGHTS_ROOT.rglob("iter_*.pth"):
        model_dirs.add(ckpt.parent)
    
    for model_dir in sorted(model_dirs):
        # Parse path: strategy/dataset/model_ratio0p50/
        relative = model_dir.relative_to(WEIGHTS_ROOT)
        parts = list(relative.parts)
        
        if len(parts) < 3:
            continue
        
        strategy = parts[0]
        dataset_dir = parts[1]
        model_dir_name = parts[2]
        
        # Extract model name (remove ratio suffix)
        model_match = re.match(r"([\w_-]+?)(?:_ratio[\dp]+)?$", model_dir_name)
        model = model_match.group(1) if model_match else model_dir_name
        
        dataset = DATASET_MAPPING.get(dataset_dir, dataset_dir)
        
        # Check for training_config.py
        config_path = model_dir / "training_config.py"
        if not config_path.exists():
            continue
        
        for iter_val in ITERATIONS:
            checkpoint = model_dir / f"iter_{iter_val}.pth"
            if not checkpoint.exists():
                continue
            
            # Check for existing test results (with timestamp subdir)
            result_dir = model_dir / f"test_results_iter_{iter_val}"
            result_exists = False
            
            if result_dir.exists():
                for subdir in result_dir.iterdir():
                    if subdir.is_dir() and (subdir / "results.json").exists():
                        result_exists = True
                        break
            
            if not result_exists:
                missing_tests.append({
                    "checkpoint": str(checkpoint),
                    "config": str(config_path),
                    "strategy": strategy,
                    "dataset": dataset,
                    "dataset_dir": dataset_dir,
                    "model": model,
                    "model_dir": str(model_dir),
                    "iter": iter_val,
                    "output_dir": str(result_dir),
                })
    
    return missing_tests


def generate_job_script(test_info, logs_dir):
    """Generate LSF job script for a test."""
    strategy = test_info["strategy"]
    dataset = test_info["dataset"]
    model = test_info["model"]
    iter_val = test_info["iter"]
    checkpoint = test_info["checkpoint"]
    config = test_info["config"]
    output_dir = test_info["output_dir"]
    
    # Short names for job
    strategy_short = strategy[:10].replace("_", "")
    model_short = "psp" if "pspnet" in model else "seg" if "segformer" in model else "dlv3"
    iter_short = f"{iter_val // 1000}k"
    
    job_name = f"ext_{strategy_short}_{dataset[:4]}_{model_short}_{iter_short}"
    
    # Build test command
    # Note: --use-native-classes is NOT needed - fine_grained_test.py auto-detects
    # model num_classes from config via detect_model_num_classes()
    
    script_content = f"""#!/bin/bash
#BSUB -J {job_name}
#BSUB -q BatchGPU
#BSUB -n 1
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "select[ngpus>0] rusage[ngpus_excl_p=1]"
#BSUB -W 00:30
#BSUB -o {logs_dir}/{job_name}_%J.out
#BSUB -e {logs_dir}/{job_name}_%J.err

cd {PROVE_ROOT}
source ~/.bashrc
conda activate prove

python fine_grained_test.py \\
    --config "{config}" \\
    --checkpoint "{checkpoint}" \\
    --dataset {dataset} \\
    --output-dir "{output_dir}"
"""
    return job_name, script_content


def submit_tests(missing_tests, dry_run=False, limit=None):
    """Submit test jobs for missing tests."""
    logs_dir = PROVE_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    jobs_dir = PROVE_ROOT / "jobs" / "extended_tests_all"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply limit if specified
    if limit:
        missing_tests = missing_tests[:limit]
    
    submitted = 0
    for test_info in missing_tests:
        job_name, script_content = generate_job_script(test_info, logs_dir)
        
        script_path = jobs_dir / f"{job_name}.sh"
        
        if dry_run:
            print(f"[DRY RUN] Would submit: {job_name}")
            print(f"  Checkpoint: {test_info['checkpoint']}")
            print(f"  Output: {test_info['output_dir']}")
        else:
            # Write script
            with open(script_path, "w") as f:
                f.write(script_content)
            
            # Submit
            result = subprocess.run(
                f"bsub < {script_path}",
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✓ Submitted: {job_name}")
                submitted += 1
            else:
                print(f"✗ Failed: {job_name}")
                print(f"  Error: {result.stderr}")
    
    return submitted


def main():
    parser = argparse.ArgumentParser(description="Submit extended training tests (90k-320k)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be submitted")
    parser.add_argument("--limit", type=int, help="Limit number of jobs")
    args = parser.parse_args()
    
    print("Finding missing extended training tests...")
    missing = find_missing_tests()
    print(f"Found {len(missing)} missing tests")
    print()
    
    if not missing:
        print("All extended training tests are complete!")
        return
    
    # Summary by iteration
    by_iter = {}
    for test in missing:
        iter_val = test["iter"]
        by_iter[iter_val] = by_iter.get(iter_val, 0) + 1
    
    print("Missing tests by iteration:")
    for iter_val in sorted(by_iter.keys()):
        print(f"  iter_{iter_val}: {by_iter[iter_val]}")
    print()
    
    # Summary by strategy
    by_strategy = {}
    for test in missing:
        s = test["strategy"]
        by_strategy[s] = by_strategy.get(s, 0) + 1
    
    print("Missing tests by strategy:")
    for s in sorted(by_strategy.keys()):
        print(f"  {s}: {by_strategy[s]}")
    print()
    
    if args.dry_run:
        print(f"\n[DRY RUN] Would submit {len(missing)} jobs")
        print("\nFirst 10 jobs:")
        submit_tests(missing[:10], dry_run=True)
    else:
        submitted = submit_tests(missing, limit=args.limit)
        print(f"\nSubmitted {submitted} jobs")


if __name__ == "__main__":
    main()
