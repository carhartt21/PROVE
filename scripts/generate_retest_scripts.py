#!/usr/bin/env python3
"""
Generate test re-run scripts for affected configurations.

This script identifies all tests that need to be re-run after the 
fine_grained_test.py bug fix (Jan 10, 2026) and generates SLURM job scripts.
"""

import os
import pandas as pd
from pathlib import Path
from datetime import datetime

# Configuration
WEIGHTS_DIR = "/scratch/aaa_exchange/AWARE/WEIGHTS"
DATA_ROOT = "/scratch/aaa_exchange/AWARE/FINAL_SPLITS"
SCRIPTS_DIR = Path("/home/mima2416/repositories/PROVE/scripts/retest_jobs")

# SLURM job template
SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=retest_{job_name}
#SBATCH --output=/home/mima2416/repositories/PROVE/logs/retest_{job_name}_%j.out
#SBATCH --error=/home/mima2416/repositories/PROVE/logs/retest_{job_name}_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00

# Activate environment
source /home/mima2416/miniconda3/bin/activate prove
cd /home/mima2416/repositories/PROVE

# Run fine-grained test
python fine_grained_test.py \\
    --config "{config_path}" \\
    --checkpoint "{checkpoint_path}" \\
    --output-dir "{output_dir}/test_results_detailed_fixed" \\
    --dataset {dataset} \\
    --data-root {data_root}

echo "Test completed at $(date)"
"""


def get_tests_to_rerun():
    """Identify tests that need to be re-run."""
    df = pd.read_csv('/home/mima2416/repositories/PROVE/downstream_results.csv')
    
    # Tests with mIoU <= 10% need re-running (buggy results)
    bad = df[df['mIoU'] <= 10.0].copy()
    
    # Exclude tests that already have _fixed in the result_dir
    bad = bad[~bad['result_dir'].str.contains('_fixed', na=False)]
    
    return bad


def generate_job_script(row):
    """Generate a SLURM job script for a single test."""
    result_dir = Path(row['result_dir'])
    base_dir = result_dir.parent.parent
    
    # Determine config and checkpoint paths
    config_path = base_dir / "training_config.py"
    checkpoint_path = base_dir / "iter_80000.pth"
    
    # Check if files exist
    if not config_path.exists():
        print(f"  Warning: Config not found: {config_path}")
        return None
    if not checkpoint_path.exists():
        # Try iter_60000 or iter_40000
        for alt in ["iter_60000.pth", "iter_40000.pth"]:
            alt_path = base_dir / alt
            if alt_path.exists():
                checkpoint_path = alt_path
                break
        else:
            print(f"  Warning: Checkpoint not found: {checkpoint_path}")
            return None
    
    # Parse dataset name
    dataset = row['dataset']
    # Map dataset names to folder names
    dataset_map = {
        'bdd10k': 'BDD10k',
        'bdd10k_cd': 'BDD10k',  # Same test data
        'idd-aw_cd': 'IDD-AW',
        'mapillaryvistas_cd': 'MapillaryVistas',
        'outside15k_cd': 'OUTSIDE15k',
    }
    dataset_folder = dataset_map.get(dataset, dataset)
    
    # Generate job name
    job_name = f"{row['strategy']}_{row['dataset']}_{row['model']}"
    job_name = job_name.replace('/', '_').replace(' ', '_')[:50]
    
    script_content = SLURM_TEMPLATE.format(
        job_name=job_name,
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        output_dir=str(base_dir),
        dataset=dataset_folder,
        data_root=DATA_ROOT,
    )
    
    return {
        'job_name': job_name,
        'content': script_content,
        'base_dir': str(base_dir),
        'dataset': dataset,
    }


def main():
    print("=" * 70)
    print("GENERATING TEST RE-RUN SCRIPTS")
    print("=" * 70)
    
    # Get tests to re-run
    tests = get_tests_to_rerun()
    print(f"\nFound {len(tests)} tests to re-run")
    
    # Create output directory
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Group by dataset for separate submission
    by_dataset = {}
    
    for _, row in tests.iterrows():
        job = generate_job_script(row)
        if job:
            ds = job['dataset']
            if ds not in by_dataset:
                by_dataset[ds] = []
            by_dataset[ds].append(job)
    
    # Generate individual job scripts
    all_jobs = []
    for ds, jobs in by_dataset.items():
        print(f"\n{ds}: {len(jobs)} jobs")
        ds_dir = SCRIPTS_DIR / ds
        ds_dir.mkdir(exist_ok=True)
        
        for job in jobs:
            script_path = ds_dir / f"retest_{job['job_name']}.sh"
            with open(script_path, 'w') as f:
                f.write(job['content'])
            script_path.chmod(0o755)
            all_jobs.append(str(script_path))
    
    # Generate master submission script
    master_script = SCRIPTS_DIR / "submit_all_retests.sh"
    with open(master_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Master script to submit all re-test jobs\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total jobs: {len(all_jobs)}\n\n")
        
        for ds, jobs in by_dataset.items():
            f.write(f"\n# {ds} ({len(jobs)} jobs)\n")
            ds_dir = SCRIPTS_DIR / ds
            f.write(f"for script in {ds_dir}/*.sh; do\n")
            f.write("    sbatch \"$script\"\n")
            f.write("    sleep 1  # Rate limit submissions\n")
            f.write("done\n")
    
    master_script.chmod(0o755)
    
    print(f"\n{'=' * 70}")
    print("GENERATED SCRIPTS")
    print("=" * 70)
    print(f"Total job scripts: {len(all_jobs)}")
    print(f"Master submission script: {master_script}")
    print(f"\nTo submit all jobs:")
    print(f"  {master_script}")
    
    # Also create per-dataset submission scripts
    for ds in by_dataset:
        ds_submit = SCRIPTS_DIR / f"submit_{ds}_retests.sh"
        with open(ds_submit, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"for script in {SCRIPTS_DIR}/{ds}/*.sh; do\n")
            f.write("    sbatch \"$script\"\n")
            f.write("    sleep 1\n")
            f.write("done\n")
        ds_submit.chmod(0o755)
        print(f"  {ds_submit}")


if __name__ == '__main__':
    main()
