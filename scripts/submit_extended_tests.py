#!/usr/bin/env python3
"""
Submit tests for extended training models (320k iteration only for now).
Tests are submitted for the final checkpoint (iter_320000.pth) of each model.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import re

# Configuration
WEIGHTS_ROOT = Path("/scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED")
PROVE_ROOT = Path("/home/mima2416/repositories/PROVE")

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


def find_extended_models(iteration: int = 320000):
    """Find all extended training checkpoints at a specific iteration."""
    models = []
    checkpoint_name = f"iter_{iteration}.pth"
    
    for ckpt in WEIGHTS_ROOT.rglob(checkpoint_name):
        model_dir = ckpt.parent
        
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
            print(f"Warning: No config for {model_dir}")
            continue
        
        models.append({
            "checkpoint": str(ckpt),
            "config": str(config_path),
            "strategy": strategy,
            "dataset": dataset,
            "dataset_dir": dataset_dir,
            "model": model,
            "model_dir": str(model_dir),
            "iter": iteration,
        })
    
    return models


def submit_tests(models: list, dry_run: bool = False, limit: int = None):
    """Submit test jobs for extended training models."""
    
    if limit:
        models = models[:limit]
    
    submitted = 0
    
    for model in models:
        strategy = model["strategy"]
        dataset = model["dataset"]
        model_name = model["model"]
        iteration = model["iter"]
        ckpt = model["checkpoint"]
        config = model["config"]
        output_dir = Path(model["model_dir"]) / "test_results_detailed"
        
        # Create job name
        iter_str = f"{iteration // 1000}k"
        job_name = f"ext_{strategy[:8]}_{dataset[:4]}_{model_name[:6]}_{iter_str}"
        
        # Build test command
        # Note: --use-native-classes is NOT needed - fine_grained_test.py auto-detects
        # model num_classes from config via detect_model_num_classes()
        
        cmd = [
            'bsub',
            '-J', job_name,
            '-q', 'BatchGPU',
            '-n', '10',
            '-gpu', 'num=1:gmem=16G',
            '-W', '0:30',
            '-o', f'{PROVE_ROOT}/logs/{job_name}_%J.out',
            '-e', f'{PROVE_ROOT}/logs/{job_name}_%J.err',
            f'source ~/.bashrc && mamba activate prove && cd {PROVE_ROOT} && python fine_grained_test.py --config {config} --checkpoint {ckpt} --dataset {dataset} --output-dir {output_dir} --batch-size 10'
        ]
        
        if dry_run:
            print(f"[DRY-RUN] {job_name}")
            print(f"  Config: {config}")
            print(f"  Checkpoint: {ckpt}")
            continue
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                match = re.search(r'Job <(\d+)>', result.stdout)
                job_id = match.group(1) if match else 'unknown'
                print(f"Submitted: {job_name} (Job {job_id})")
                submitted += 1
            else:
                print(f"Failed: {job_name}")
                print(f"  Error: {result.stderr}")
        except Exception as e:
            print(f"Error: {job_name} - {e}")
    
    return submitted


def main():
    parser = argparse.ArgumentParser(description="Submit extended training tests")
    parser.add_argument("--dry-run", action="store_true", help="Print without submitting")
    parser.add_argument("--limit", type=int, help="Maximum jobs to submit")
    parser.add_argument("--iteration", type=int, default=320000, help="Iteration to test (default: 320000)")
    parser.add_argument("--strategy", type=str, help="Filter by strategy")
    
    args = parser.parse_args()
    
    print(f"Finding extended training models at iteration {args.iteration}...")
    models = find_extended_models(args.iteration)
    
    if args.strategy:
        models = [m for m in models if args.strategy in m["strategy"]]
    
    print(f"Found {len(models)} models to test")
    
    if not models:
        return
    
    submitted = submit_tests(models, dry_run=args.dry_run, limit=args.limit)
    
    if args.dry_run:
        print(f"\n[DRY-RUN] Would submit {len(models)} jobs")
    else:
        print(f"\nSubmitted {submitted} jobs")


if __name__ == "__main__":
    main()
