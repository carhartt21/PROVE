#!/usr/bin/env python3
"""
Submit tests for all ablation study models (ratio ablation and extended training).

This script finds all trained models in WEIGHTS_RATIO_ABLATION and WEIGHTS_EXTENDED
that don't have test results yet and submits fine_grained_test.py jobs for them.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import re

# Configuration
WEIGHTS_ROOT = "/scratch/aaa_exchange/AWARE"
PROVE_ROOT = "/home/mima2416/repositories/PROVE"

# Dataset mappings (directory name -> proper dataset name for testing)
DATASET_MAPPING = {
    # Standard datasets
    "bdd10k": "BDD10k",
    "idd-aw": "IDD-AW",
    "outside15k": "OUTSIDE15k",
    "mapillaryvistas": "MapillaryVistas",
    # _ad suffix datasets (all-domain training)
    "bdd10k_ad": "BDD10k",
    "iddaw_ad": "IDD-AW",
    "idd-aw_ad": "IDD-AW",
    "outside15k_ad": "OUTSIDE15k",
    "mapillaryvistas_ad": "MapillaryVistas",
}

# Models that need --use-native-classes
NATIVE_CLASS_DATASETS = ["MapillaryVistas", "OUTSIDE15k"]


def find_untested_models(weights_dir: str, checkpoint_pattern: str = "iter_80000.pth") -> list:
    """Find all trained models without test results."""
    untested = []
    
    weights_path = Path(weights_dir)
    if not weights_path.exists():
        print(f"Warning: {weights_dir} does not exist")
        return []
    
    # Find all checkpoints matching the pattern
    for ckpt in weights_path.rglob(checkpoint_pattern):
        model_dir = ckpt.parent
        
        # Check if test results exist
        test_results = list(model_dir.glob("test_results_detailed/*/results.json"))
        
        if not test_results:
            # Parse the path to get stage/strategy/dataset/model
            # Structure: WEIGHTS_RATIO_ABLATION/stage1/strategy/dataset/model_ratioX/
            relative = model_dir.relative_to(weights_path)
            parts = list(relative.parts)
            
            # Handle both structures:
            # - stage1/strategy/dataset/model (4 parts)
            # - strategy/dataset/model (3 parts - old structure)
            if len(parts) >= 4 and parts[0] in ("stage1", "stage2"):
                stage = parts[0]
                strategy = parts[1]
                dataset_dir = parts[2]
                model_dir_name = parts[3]
            elif len(parts) >= 3:
                stage = "unknown"
                strategy = parts[0]
                dataset_dir = parts[1]
                model_dir_name = parts[2]
            else:
                continue
                
            # Extract model name (remove ratio suffix if present)
            model_match = re.match(r"([\w_-]+?)(?:_ratio[\dp]+)?$", model_dir_name)
            if model_match:
                model = model_match.group(1)
            else:
                model = model_dir_name
            
            # Map dataset directory to proper name
            dataset = DATASET_MAPPING.get(dataset_dir, dataset_dir)
            
            untested.append({
                "checkpoint": str(ckpt),
                "config": str(model_dir / "training_config.py"),
                "stage": stage,
                "strategy": strategy,
                "dataset": dataset,
                "dataset_dir": dataset_dir,
                "model": model,
                "model_dir": str(model_dir),
                "iter": int(checkpoint_pattern.replace("iter_", "").replace(".pth", "")),
            })
    
    return untested


def find_extended_training_models(weights_dir: str) -> list:
    """Find all extended training checkpoints."""
    models = []
    
    weights_path = Path(weights_dir)
    if not weights_path.exists():
        print(f"Warning: {weights_dir} does not exist")
        return []
    
    # Find all iter_*.pth checkpoints
    for ckpt in sorted(weights_path.rglob("iter_*.pth")):
        # Extract iteration number
        match = re.search(r"iter_(\d+)\.pth", ckpt.name)
        if not match:
            continue
        iteration = int(match.group(1))
        
        model_dir = ckpt.parent
        
        # Check if test results exist for this specific checkpoint
        # Test results are stored per checkpoint in subdirectories
        test_results_dir = model_dir / "test_results_detailed"
        has_test = False
        if test_results_dir.exists():
            # Look for results matching this iteration
            for result_dir in test_results_dir.iterdir():
                if result_dir.is_dir():
                    results_file = result_dir / "results.json"
                    if results_file.exists():
                        # Check if this result is for this checkpoint
                        # We need to determine this - for now assume no tests exist
                        has_test = False
                        break
        
        if not has_test:
            # Parse the path
            relative = model_dir.relative_to(weights_path)
            parts = list(relative.parts)
            
            if len(parts) >= 3:
                strategy = parts[0]
                dataset_dir = parts[1]
                model_dir_name = parts[2]
                
                # Extract model name
                model_match = re.match(r"([\w_-]+?)(?:_ratio[\dp]+)?$", model_dir_name)
                model = model_match.group(1) if model_match else model_dir_name
                
                dataset = DATASET_MAPPING.get(dataset_dir, dataset_dir)
                
                models.append({
                    "checkpoint": str(ckpt),
                    "config": str(model_dir / "training_config.py"),
                    "strategy": strategy,
                    "dataset": dataset,
                    "dataset_dir": dataset_dir,
                    "model": model,
                    "model_dir": str(model_dir),
                    "iter": iteration,
                })
    
    return models


def generate_job_script(model_info: dict, job_id: int) -> tuple:
    """Generate an LSF job submission script for testing."""
    
    strategy = model_info["strategy"]
    dataset = model_info["dataset"]
    model = model_info["model"]
    iteration = model_info["iter"]
    stage = model_info.get("stage", "s1")
    ckpt = model_info["checkpoint"]
    config = model_info["config"]
    output_dir = model_info["model_dir"]
    
    # Determine job name
    iter_str = f"{iteration // 1000}k" if iteration >= 1000 else str(iteration)
    stage_short = stage[:2] if stage else "s1"
    job_name = f"abl_{stage_short}_{strategy[:8]}_{dataset[:4]}_{model[:6]}_{iter_str}"
    
    # Build test command
    # Note: --use-native-classes is NOT needed for testing - fine_grained_test.py
    # auto-detects model num_classes from config via detect_model_num_classes()
    
    test_cmd = f"""cd {PROVE_ROOT}
source ~/.bashrc
mamba activate prove

python fine_grained_test.py \\
    --config "{config}" \\
    --checkpoint "{ckpt}" \\
    --dataset {dataset} \\
    --output-dir "{output_dir}/test_results_detailed"
"""
    
    job_script = f"""#!/bin/bash
#BSUB -J {job_name}
#BSUB -q BatchGPU
#BSUB -n 10
#BSUB -gpu "num=1:gmem=16G"
#BSUB -W 0:30
#BSUB -o {PROVE_ROOT}/logs/ablation_test_{job_id}_%J.out
#BSUB -e {PROVE_ROOT}/logs/ablation_test_{job_id}_%J.err

{test_cmd}
"""
    
    return job_script, job_name


def submit_jobs(models: list, dry_run: bool = False, limit: int = None) -> int:
    """Submit test jobs for models."""
    
    if limit:
        models = models[:limit]
    
    submitted = 0
    
    for i, model in enumerate(models):
        job_script, job_name = generate_job_script(model, i)
        
        if dry_run:
            print(f"[DRY-RUN] Would submit: {job_name}")
            print(f"  Config: {model['config']}")
            print(f"  Checkpoint: {model['checkpoint']}")
            print(f"  Dataset: {model['dataset']}")
            print()
        else:
            # Write temp script
            script_path = f"/tmp/ablation_test_{i}.lsf"
            with open(script_path, "w") as f:
                f.write(job_script)
            
            # Submit job using shell to handle redirection
            try:
                result = subprocess.run(
                    f"bsub < {script_path}",
                    shell=True,
                    capture_output=True,
                    text=True
                )
                if "is submitted" in result.stdout:
                    print(f"Submitted: {job_name}")
                    submitted += 1
                else:
                    print(f"Failed to submit {job_name}: {result.stdout} {result.stderr}")
            except Exception as e:
                print(f"Error submitting {job_name}: {e}")
            
            # Clean up
            os.remove(script_path)
    
    return submitted


def main():
    parser = argparse.ArgumentParser(description="Submit ablation study test jobs")
    parser.add_argument("--dry-run", action="store_true", help="Print jobs without submitting")
    parser.add_argument("--limit", type=int, help="Maximum number of jobs to submit")
    parser.add_argument("--ratio-only", action="store_true", help="Only submit ratio ablation tests")
    parser.add_argument("--extended-only", action="store_true", help="Only submit extended training tests")
    parser.add_argument("--strategy", type=str, help="Filter by strategy name")
    parser.add_argument("--dataset", type=str, help="Filter by dataset name")
    
    args = parser.parse_args()
    
    all_models = []
    
    # Find ratio ablation models
    if not args.extended_only:
        print("Scanning WEIGHTS_RATIO_ABLATION...")
        ratio_models = find_untested_models(
            os.path.join(WEIGHTS_ROOT, "WEIGHTS_RATIO_ABLATION")
        )
        print(f"  Found {len(ratio_models)} untested ratio ablation models")
        all_models.extend(ratio_models)
    
    # Find extended training models
    if not args.ratio_only:
        print("Scanning WEIGHTS_EXTENDED...")
        extended_models = find_extended_training_models(
            os.path.join(WEIGHTS_ROOT, "WEIGHTS_EXTENDED")
        )
        print(f"  Found {len(extended_models)} extended training checkpoints")
        all_models.extend(extended_models)
    
    # Apply filters
    if args.strategy:
        all_models = [m for m in all_models if args.strategy in m["strategy"]]
        print(f"After strategy filter '{args.strategy}': {len(all_models)} models")
    
    if args.dataset:
        all_models = [m for m in all_models if args.dataset.lower() in m["dataset"].lower()]
        print(f"After dataset filter '{args.dataset}': {len(all_models)} models")
    
    print(f"\nTotal models to test: {len(all_models)}")
    
    if not all_models:
        print("No models to test!")
        return
    
    # Submit jobs
    submitted = submit_jobs(all_models, dry_run=args.dry_run, limit=args.limit)
    
    if args.dry_run:
        print(f"\n[DRY-RUN] Would submit {len(all_models)} jobs (limit: {args.limit})")
    else:
        print(f"\nSubmitted {submitted} jobs")


if __name__ == "__main__":
    main()
