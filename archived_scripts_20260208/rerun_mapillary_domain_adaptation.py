#!/usr/bin/env python3
"""
Re-run domain adaptation tests for MapillaryVistas models after BGR/RGB fix.

The original domain adaptation evaluations for MapillaryVistas were done with
buggy models that had wrong class mappings. This script re-runs those tests
using the fixed (retrained) models.

Usage:
    python scripts/rerun_mapillary_domain_adaptation.py --dry-run
    python scripts/rerun_mapillary_domain_adaptation.py
"""

import os
import subprocess
import argparse
from pathlib import Path

WEIGHTS_ROOT = Path("${AWARE_DATA_ROOT}/WEIGHTS")
RESULTS_ROOT = WEIGHTS_ROOT / "domain_adaptation_ablation"
REPO_ROOT = Path("${HOME}/repositories/PROVE")

# Strategies that have MapillaryVistas domain adaptation results to re-run
STRATEGIES = [
    "baseline",
    "gen_albumentations_weather",
    "gen_automold",
    "gen_cyclediffusion", 
    "gen_cycleGAN",
    "gen_flux_kontext",
    "gen_stargan_v2",
    "gen_step1x_new",
    "gen_step1x_v1p2",
    "gen_TSIT",
    "gen_UniControl",
    "std_photometric_distort",
    "std_autoaugment",
    "std_cutmix",
    "std_mixup",
    "std_randaugment",
]

# Models to test
MODELS = ["pspnet_r50", "segformer_mit-b5"]


def find_models_to_retest():
    """Find MapillaryVistas models that need domain adaptation re-testing."""
    models_to_test = []
    
    for strategy in STRATEGIES:
        for model in MODELS:
            # Check if the retrained model exists
            model_name = model if strategy == "baseline" else f"{model}_ratio0p50"
            model_path = WEIGHTS_ROOT / strategy / "mapillaryvistas" / model_name
            checkpoint = model_path / "iter_80000.pth"
            config = model_path / "training_config.py"
            
            if not checkpoint.exists() or not config.exists():
                # Try without ratio suffix for non-gen strategies
                model_name = model
                model_path = WEIGHTS_ROOT / strategy / "mapillaryvistas" / model_name
                checkpoint = model_path / "iter_80000.pth"
                config = model_path / "training_config.py"
                
                if not checkpoint.exists() or not config.exists():
                    continue
            
            # Check if old evaluation exists (to be replaced)
            old_result = RESULTS_ROOT / strategy / "mapillaryvistas" / model_name / "domain_adaptation_evaluation.json"
            old_result_baseline = RESULTS_ROOT / "mapillaryvistas" / f"{model}_clear_day" / "domain_adaptation_evaluation.json"
            
            has_old = old_result.exists() or old_result_baseline.exists()
            
            models_to_test.append({
                "strategy": strategy,
                "model": model,
                "model_name": model_name,
                "checkpoint": str(checkpoint),
                "config": str(config),
                "has_old_result": has_old,
            })
    
    return models_to_test


def generate_job_script(model_info):
    """Generate LSF job script for domain adaptation testing."""
    strategy = model_info["strategy"]
    model = model_info["model"]
    
    job_name = f"da_mv_{strategy[:8]}_{model[:10]}"
    
    script = f"""#!/bin/bash
#BSUB -J {job_name}
#BSUB -q BatchGPU
#BSUB -gpu "num=1:gmem=16G"
#BSUB -n 10
#BSUB -R "span[hosts=1]"
#BSUB -W 01:00
#BSUB -o {REPO_ROOT}/logs/{job_name}_%J.out
#BSUB -e {REPO_ROOT}/logs/{job_name}_%J.err

cd {REPO_ROOT}
source ~/.bashrc
mamba activate prove

python scripts/run_domain_adaptation_tests.py \\
    --source-dataset mapillaryvistas \\
    --model {model} \\
    --strategy {strategy} \\
    --regenerate

echo "Domain adaptation test completed for {strategy}/{model}"
"""
    return script, job_name


def submit_jobs(models_to_test, dry_run=False, limit=None):
    """Submit domain adaptation test jobs."""
    submitted = 0
    
    if limit:
        models_to_test = models_to_test[:limit]
    
    for model_info in models_to_test:
        script, job_name = generate_job_script(model_info)
        
        if dry_run:
            old = " (replaces old)" if model_info["has_old_result"] else ""
            print(f"[DRY RUN] Would submit: {job_name}{old}")
            print(f"  Checkpoint: {model_info['checkpoint']}")
            submitted += 1
        else:
            script_path = f"/tmp/{job_name}.sh"
            with open(script_path, 'w') as f:
                f.write(script)
            
            result = subprocess.run(
                f"bsub < {script_path}",
                shell=True,
                capture_output=True,
                text=True
            )
            
            if "is submitted" in result.stdout:
                print(f"✓ Submitted: {job_name}")
                submitted += 1
            else:
                print(f"✗ Failed: {job_name}")
                print(f"  Error: {result.stderr}")
    
    return submitted


def main():
    parser = argparse.ArgumentParser(description="Re-run MapillaryVistas domain adaptation tests")
    parser.add_argument("--dry-run", action="store_true", help="Print jobs without submitting")
    parser.add_argument("--limit", type=int, help="Limit number of jobs")
    args = parser.parse_args()
    
    print("Finding MapillaryVistas models to re-test...")
    models_to_test = find_models_to_retest()
    
    print(f"Found {len(models_to_test)} models to test")
    
    if not models_to_test:
        print("No models to test!")
        return
    
    # Summary
    has_old = sum(1 for m in models_to_test if m["has_old_result"])
    print(f"  {has_old} have old results to replace")
    print(f"  {len(models_to_test) - has_old} are new")
    
    print()
    submitted = submit_jobs(models_to_test, dry_run=args.dry_run, limit=args.limit)
    
    if args.dry_run:
        print(f"\n[DRY RUN] Would submit {submitted} jobs")
    else:
        print(f"\nSubmitted {submitted} jobs")


if __name__ == "__main__":
    main()
