#!/usr/bin/env python3
"""
Re-evaluate MapillaryVistas and OUTSIDE15k models with correct native class evaluation.

This script re-runs fine_grained_test.py on existing checkpoints that were trained
with native classes (66 for MapillaryVistas, 24 for OUTSIDE15k) but were incorrectly
evaluated with 19 Cityscapes classes.

Usage:
    # Dry run - show what would be evaluated
    python scripts/reeval_native_class_models.py --dry-run
    
    # Generate job submission script
    python scripts/reeval_native_class_models.py --output-script jobs_reeval.sh
    
    # Submit jobs directly
    python scripts/reeval_native_class_models.py --submit
"""

import os
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple


WEIGHTS_BASE = "/scratch/aaa_exchange/AWARE/WEIGHTS"


def find_checkpoints(dataset_pattern: str) -> List[Tuple[str, str, str]]:
    """
    Find all iter_80000.pth checkpoints for a dataset pattern.
    
    Returns list of (checkpoint_path, config_path, dataset_name)
    """
    results = []
    
    # Find all checkpoint files matching the pattern
    # Use 2>/dev/null to suppress permission denied errors
    cmd = f"find {WEIGHTS_BASE} -path '*{dataset_pattern}*' -name 'iter_80000.pth' -type f 2>/dev/null"
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    output = result.stdout
    
    for checkpoint_path in output.strip().split('\n'):
        if not checkpoint_path:
            continue
            
        # Derive config path and dataset name
        model_dir = os.path.dirname(checkpoint_path)
        
        # Check for training_config.py or configs/training_config.py
        config_path = os.path.join(model_dir, "training_config.py")
        if not os.path.exists(config_path):
            config_path = os.path.join(model_dir, "configs", "training_config.py")
        
        if not os.path.exists(config_path):
            print(f"Warning: No config found for {checkpoint_path}")
            continue
        
        # Determine dataset name
        if 'mapillaryvistas' in checkpoint_path.lower():
            dataset_name = 'MapillaryVistas'
        elif 'outside15k' in checkpoint_path.lower():
            dataset_name = 'OUTSIDE15k'
        else:
            print(f"Warning: Unknown dataset for {checkpoint_path}")
            continue
        
        results.append((checkpoint_path, config_path, dataset_name))
    
    return results


def generate_reeval_command(checkpoint_path: str, config_path: str, 
                           dataset_name: str, output_dir: str = None) -> str:
    """Generate the fine_grained_test.py command for re-evaluation."""
    
    if output_dir is None:
        model_dir = os.path.dirname(checkpoint_path)
        output_dir = os.path.join(model_dir, "test_results_detailed_fixed")
    
    cmd = f"""python fine_grained_test.py \\
    --checkpoint {checkpoint_path} \\
    --config {config_path} \\
    --dataset {dataset_name} \\
    --output-dir {output_dir}"""
    
    return cmd


def generate_job_script(checkpoints: List[Tuple[str, str, str]], 
                       job_name: str = "reeval_native_classes") -> str:
    """Generate a bash job submission script."""
    
    script = f"""#!/bin/bash
#BSUB -J {job_name}
#BSUB -o /home/chge7185/repositories/PROVE/logs/retrain/{job_name}_%J.out
#BSUB -e /home/chge7185/repositories/PROVE/logs/retrain/{job_name}_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=20G"
#BSUB -W 4:00
#BSUB -q BatchGPU

# Activate environment
source ~/.bashrc
mamba activate prove

cd /home/chge7185/repositories/PROVE

echo "========================================"
echo "Re-evaluation job: {job_name}"
echo "Started: $(date)"
echo "========================================"

"""
    
    for i, (checkpoint_path, config_path, dataset_name) in enumerate(checkpoints, 1):
        model_dir = os.path.dirname(checkpoint_path)
        relative_path = model_dir.replace(WEIGHTS_BASE + "/", "")
        output_dir = os.path.join(model_dir, "test_results_detailed_fixed")
        
        script += f"""
echo ""
echo "----------------------------------------"
echo "[{i}/{len(checkpoints)}] Re-evaluating: {relative_path}"
echo "Dataset: {dataset_name}"
echo "----------------------------------------"

python fine_grained_test.py \\
    --checkpoint {checkpoint_path} \\
    --config {config_path} \\
    --dataset {dataset_name} \\
    --output-dir {output_dir}

echo "Finished: {relative_path}"
"""
    
    script += """
echo ""
echo "========================================"
echo "All re-evaluations completed: $(date)"
echo "========================================"
"""
    
    return script


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate models with native class settings")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be evaluated without doing anything")
    parser.add_argument("--output-script", type=str, 
                       help="Output path for job submission script")
    parser.add_argument("--submit", action="store_true",
                       help="Submit the job directly")
    parser.add_argument("--dataset", type=str, choices=["mapillaryvistas", "outside15k", "all"],
                       default="all", help="Which dataset(s) to re-evaluate")
    args = parser.parse_args()
    
    # Find checkpoints
    all_checkpoints = []
    
    if args.dataset in ["mapillaryvistas", "all"]:
        # Include single-dataset MapillaryVistas only (not multi-dataset which uses Cityscapes classes)
        mv_checkpoints = find_checkpoints("mapillaryvistas_cd")
        all_checkpoints.extend(mv_checkpoints)
        print(f"Found {len(mv_checkpoints)} MapillaryVistas checkpoints (single-dataset)")
    
    if args.dataset in ["outside15k", "all"]:
        outside_checkpoints = find_checkpoints("outside15k")
        all_checkpoints.extend(outside_checkpoints)
        print(f"Found {len(outside_checkpoints)} OUTSIDE15k checkpoints")
    
    if not all_checkpoints:
        print("No checkpoints found!")
        return
    
    print(f"\nTotal checkpoints to re-evaluate: {len(all_checkpoints)}")
    
    if args.dry_run:
        print("\n=== DRY RUN - Commands that would be executed ===\n")
        for i, (checkpoint_path, config_path, dataset_name) in enumerate(all_checkpoints, 1):
            relative_path = checkpoint_path.replace(WEIGHTS_BASE + "/", "")
            print(f"[{i}] {dataset_name}: {relative_path}")
            cmd = generate_reeval_command(checkpoint_path, config_path, dataset_name)
            print(cmd)
            print()
        return
    
    # Generate job script
    script_content = generate_job_script(all_checkpoints)
    
    if args.output_script:
        output_path = args.output_script
    else:
        output_path = "/home/chge7185/repositories/PROVE/scripts/jobs_reeval_native_classes.sh"
    
    with open(output_path, 'w') as f:
        f.write(script_content)
    os.chmod(output_path, 0o755)
    print(f"\nJob script written to: {output_path}")
    
    if args.submit:
        print("\nSubmitting job...")
        result = subprocess.run(["bsub", "<", output_path], shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
    else:
        print(f"\nTo submit the job, run:")
        print(f"  bsub < {output_path}")


if __name__ == "__main__":
    main()
