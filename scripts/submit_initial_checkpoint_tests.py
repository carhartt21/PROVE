#!/usr/bin/env python3
"""
Submit tests for initial checkpoints (10k-80k) from WEIGHTS/ to complement the extended training study.
Results are saved to WEIGHTS_EXTENDED/{strategy}/{dataset}/{model}/test_results_iter_{iteration}/

This provides the full learning curve from 10k to 320k iterations for the extended training analysis.
"""

import os
import subprocess
import argparse
from pathlib import Path

WEIGHTS_ROOT = "/scratch/aaa_exchange/AWARE/WEIGHTS"
EXTENDED_ROOT = "/scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED"
REPO_ROOT = "/home/mima2416/repositories/PROVE"

# Strategies that are in the extended training study (excluding std_randaugment which doesn't have ratio models in WEIGHTS)
EXTENDED_STRATEGIES = [
    "gen_albumentations_weather",
    "gen_automold", 
    "gen_cyclediffusion",
    "gen_cycleGAN",
    "gen_flux_kontext",
    "gen_step1x_new",
    "gen_TSIT",
    "gen_UniControl",
]

# Checkpoints to test (10k to 80k, every 10k)
CHECKPOINTS = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]

# Datasets - normalized names (without _ad suffix)
DATASETS = ["bdd10k", "idd-aw", "mapillaryvistas", "outside15k"]

# Models in extended training study
MODELS = ["pspnet_r50_ratio0p50", "segformer_mit-b5_ratio0p50"]

# Dataset class mappings
NATIVE_CLASS_DATASETS = {"mapillaryvistas", "outside15k"}


def get_dataset_display_name(dataset):
    """Get proper display name for dataset."""
    mapping = {
        "bdd10k": "BDD10k",
        "idd-aw": "IDD_AW",
        "mapillaryvistas": "MapillaryVistas",
        "outside15k": "OUTSIDE15k",
    }
    return mapping.get(dataset, dataset)


def find_models_to_test():
    """Find all models that need testing for initial checkpoints."""
    models_to_test = []
    
    for strategy in EXTENDED_STRATEGIES:
        for dataset in DATASETS:
            for model in MODELS:
                # Check if this model exists in WEIGHTS_EXTENDED
                extended_path = Path(EXTENDED_ROOT) / strategy / dataset / model
                
                # Also check for _ad suffix variants
                if not extended_path.exists():
                    # Try with _ad suffix
                    for suffix in ["_ad", ""]:
                        test_path = Path(EXTENDED_ROOT) / strategy / f"{dataset}{suffix}" / model
                        if test_path.exists():
                            extended_path = test_path
                            break
                
                if not extended_path.exists():
                    continue
                
                # Check if source model exists in WEIGHTS
                weights_path = Path(WEIGHTS_ROOT) / strategy / dataset / model
                if not weights_path.exists():
                    continue
                
                # Check each checkpoint
                for iteration in CHECKPOINTS:
                    checkpoint_file = weights_path / f"iter_{iteration}.pth"
                    config_file = weights_path / "training_config.py"
                    
                    if not checkpoint_file.exists() or not config_file.exists():
                        continue
                    
                    # Check if test result already exists
                    # Results can be in either:
                    # - test_results_iter_{iteration}/results.json (direct)
                    # - test_results_iter_{iteration}/{timestamp}/results.json (with timestamp subdir)
                    result_dir = extended_path / f"test_results_iter_{iteration}"
                    result_file = result_dir / "results.json"
                    
                    result_exists = result_file.exists()
                    if not result_exists and result_dir.exists():
                        # Check for timestamp subdirectories
                        for item in result_dir.iterdir():
                            if item.is_dir() and (item / "results.json").exists():
                                result_exists = True
                                break
                    
                    if result_exists:
                        continue
                    
                    models_to_test.append({
                        "strategy": strategy,
                        "dataset": dataset,
                        "model": model,
                        "iteration": iteration,
                        "checkpoint": str(checkpoint_file),
                        "config": str(config_file),
                        "output_dir": str(result_dir),
                        "extended_path": str(extended_path),
                    })
    
    return models_to_test


def generate_job_script(model_info):
    """Generate LSF job script for testing."""
    strategy = model_info["strategy"]
    dataset = model_info["dataset"]
    model = model_info["model"]
    iteration = model_info["iteration"]
    checkpoint = model_info["checkpoint"]
    config = model_info["config"]
    output_dir = model_info["output_dir"]
    
    # Get proper dataset display name
    dataset_display = get_dataset_display_name(dataset)
    
    # Build test command
    # Note: --use-native-classes is NOT needed - fine_grained_test.py auto-detects
    # model num_classes from config via detect_model_num_classes()
    
    job_name = f"init_{strategy[:10]}_{dataset}_{model.split('_')[0]}_iter{iteration//1000}k"
    
    # Create output directory first
    os.makedirs(output_dir, exist_ok=True)
    
    script = f"""#!/bin/bash
#BSUB -J {job_name}
#BSUB -q BatchGPU
#BSUB -gpu "num=1:gmem=16G"
#BSUB -n 10
#BSUB -R "span[hosts=1]"
#BSUB -W 00:30
#BSUB -o {REPO_ROOT}/logs/{job_name}_%J.out
#BSUB -e {REPO_ROOT}/logs/{job_name}_%J.err

cd {REPO_ROOT}
source ~/.bashrc
mamba activate prove

python fine_grained_test.py \\
    --config {config} \\
    --checkpoint {checkpoint} \\
    --dataset {dataset_display} \\
    --output-dir {output_dir}

echo "Test completed for {strategy}/{dataset}/{model} at iteration {iteration}"
"""
    return script, job_name


def submit_jobs(models_to_test, dry_run=False, limit=None):
    """Submit test jobs for all models."""
    submitted = 0
    
    if limit:
        models_to_test = models_to_test[:limit]
    
    for model_info in models_to_test:
        script, job_name = generate_job_script(model_info)
        
        if dry_run:
            print(f"[DRY RUN] Would submit: {job_name}")
            print(f"  Checkpoint: {model_info['checkpoint']}")
            print(f"  Output: {model_info['output_dir']}")
            submitted += 1
        else:
            # Write script to temp file and submit
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
    parser = argparse.ArgumentParser(description="Submit initial checkpoint tests for extended training study")
    parser.add_argument("--dry-run", action="store_true", help="Print jobs without submitting")
    parser.add_argument("--limit", type=int, help="Limit number of jobs to submit")
    parser.add_argument("--strategy", type=str, help="Filter by specific strategy")
    parser.add_argument("--dataset", type=str, help="Filter by specific dataset")
    parser.add_argument("--iteration", type=int, help="Filter by specific iteration")
    args = parser.parse_args()
    
    print("Finding models to test...")
    models_to_test = find_models_to_test()
    
    # Apply filters
    if args.strategy:
        models_to_test = [m for m in models_to_test if args.strategy in m["strategy"]]
    if args.dataset:
        models_to_test = [m for m in models_to_test if args.dataset in m["dataset"]]
    if args.iteration:
        models_to_test = [m for m in models_to_test if m["iteration"] == args.iteration]
    
    print(f"Found {len(models_to_test)} models needing tests")
    
    if not models_to_test:
        print("No models to test!")
        return
    
    # Group by strategy for summary
    by_strategy = {}
    for m in models_to_test:
        strategy = m["strategy"]
        if strategy not in by_strategy:
            by_strategy[strategy] = []
        by_strategy[strategy].append(m)
    
    print("\nSummary by strategy:")
    for strategy, models in sorted(by_strategy.items()):
        print(f"  {strategy}: {len(models)} tests")
    
    # Group by iteration
    by_iteration = {}
    for m in models_to_test:
        iteration = m["iteration"]
        if iteration not in by_iteration:
            by_iteration[iteration] = 0
        by_iteration[iteration] += 1
    
    print("\nSummary by iteration:")
    for iteration, count in sorted(by_iteration.items()):
        print(f"  iter_{iteration}: {count} tests")
    
    print()
    submitted = submit_jobs(models_to_test, dry_run=args.dry_run, limit=args.limit)
    
    if args.dry_run:
        print(f"\n[DRY RUN] Would submit {submitted} jobs")
    else:
        print(f"\nSubmitted {submitted} jobs")


if __name__ == "__main__":
    main()
