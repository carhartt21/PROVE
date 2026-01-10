#!/usr/bin/env python3
"""
Retrain all IDD-AW models with fixed label transform.

ISSUE: IDD-AW labels were discovered to already be in Cityscapes trainID format (0-18),
but they were incorrectly included in CITYSCAPES_LABEL_ID_DATASETS, causing the
CityscapesLabelIdToTrainId transform to corrupt the labels.

For example:
- trainID 0 (road) was mapped to 255 (IGNORE)
- trainID 7 (traffic sign) was mapped to 0 (road)
- trainID 8 (vegetation) was mapped to 1 (sidewalk)

This resulted in ~half the classes (8-18) showing nan in per-class validation metrics.

FIX: Removed IDD-AW from CITYSCAPES_LABEL_ID_DATASETS in unified_training_config.py

This script generates SLURM jobs to retrain all IDD-AW models with the corrected label handling.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Tuple

# Models that need retraining
MODELS_TO_RETRAIN = [
    # Strategy, dataset, model, ratio (None for no ratio suffix)
    # Baseline
    ('baseline', 'IDD-AW', 'deeplabv3plus_r50', None),
    ('baseline', 'IDD-AW', 'pspnet_r50', None),
    ('baseline', 'IDD-AW', 'segformer_mit-b5', None),
    
    # Standard augmentation strategies
    ('std_autoaugment', 'IDD-AW', 'deeplabv3plus_r50', None),
    ('std_autoaugment', 'IDD-AW', 'pspnet_r50', None),
    ('std_autoaugment', 'IDD-AW', 'segformer_mit-b5', None),
    
    ('std_cutmix', 'IDD-AW', 'deeplabv3plus_r50', None),
    ('std_cutmix', 'IDD-AW', 'pspnet_r50', None),
    ('std_cutmix', 'IDD-AW', 'segformer_mit-b5', None),
    
    ('std_mixup', 'IDD-AW', 'deeplabv3plus_r50', None),
    ('std_mixup', 'IDD-AW', 'pspnet_r50', None),
    ('std_mixup', 'IDD-AW', 'segformer_mit-b5', None),
    
    ('std_randaugment', 'IDD-AW', 'deeplabv3plus_r50', None),
    ('std_randaugment', 'IDD-AW', 'pspnet_r50', None),
    ('std_randaugment', 'IDD-AW', 'segformer_mit-b5', None),
    
    ('photometric_distort', 'IDD-AW', 'deeplabv3plus_r50', None),
    ('photometric_distort', 'IDD-AW', 'pspnet_r50', None),
    ('photometric_distort', 'IDD-AW', 'segformer_mit-b5', None),
    
    # Generative strategies (with ratio)
    ('gen_Attribute_Hallucination', 'IDD-AW', 'deeplabv3plus_r50', 0.5),
    ('gen_Attribute_Hallucination', 'IDD-AW', 'pspnet_r50', 0.5),
    ('gen_Attribute_Hallucination', 'IDD-AW', 'segformer_mit-b5', 0.5),
    
    ('gen_CNetSeg', 'IDD-AW', 'deeplabv3plus_r50', 0.5),
    ('gen_CNetSeg', 'IDD-AW', 'pspnet_r50', 0.5),
    ('gen_CNetSeg', 'IDD-AW', 'segformer_mit-b5', 0.5),
    
    ('gen_CUT', 'IDD-AW', 'deeplabv3plus_r50', 0.5),
    ('gen_CUT', 'IDD-AW', 'pspnet_r50', 0.5),
    ('gen_CUT', 'IDD-AW', 'segformer_mit-b5', 0.5),
    
    ('gen_IP2P', 'IDD-AW', 'deeplabv3plus_r50', 0.5),
    ('gen_IP2P', 'IDD-AW', 'pspnet_r50', 0.5),
    ('gen_IP2P', 'IDD-AW', 'segformer_mit-b5', 0.5),
]

# Multi-dataset training that includes IDD-AW
# Note: Removed ACDC from multi-dataset - ACDC uses Cityscapes label IDs which adds complexity
MULTI_DATASET_MODELS = [
    ('baseline', ['MapillaryVistas', 'IDD-AW', 'BDD10k'], 'deeplabv3plus_r50'),
    ('baseline', ['MapillaryVistas', 'IDD-AW', 'BDD10k'], 'pspnet_r50'),
    ('baseline', ['MapillaryVistas', 'IDD-AW', 'BDD10k'], 'segformer_mit-b5'),
]

WEIGHTS_ROOT = '/scratch/aaa_exchange/AWARE/WEIGHTS'
LOGS_DIR = '/home/chge7185/repositories/PROVE/logs/retrain'


def generate_lsf_script(strategy: str, dataset: str, model: str, ratio: float = None,
                        backup_existing: bool = True) -> str:
    """Generate an LSF job script for retraining with pre-flight lock mechanism."""
    
    ratio_arg = f'--real-gen-ratio {ratio}' if ratio is not None else ''
    ratio_suffix = f'_ratio{ratio:.2f}'.replace('.', 'p') if ratio else ''
    
    job_name = f'retrain_{strategy}_{dataset.lower()}_{model}'
    if ratio:
        job_name += f'_{int(ratio*100)}'
    
    # Backup existing checkpoint
    existing_dir = f'{WEIGHTS_ROOT}/{strategy}/{dataset.lower()}_cd/{model}{ratio_suffix}'
    backup_cmd = ''
    if backup_existing:
        backup_cmd = f'''
# Backup existing checkpoint (corrupted labels)
if [ -d "{existing_dir}" ]; then
    mv "{existing_dir}" "{existing_dir}_corrupted_labels_backup"
    echo "Backed up corrupted checkpoint to {existing_dir}_corrupted_labels_backup"
fi
'''
    
    ratio_lock_arg = f'--ratio {ratio}' if ratio is not None else ''
    
    # Build training command with or without ratio
    if ratio is not None:
        train_cmd = f'''python unified_training.py \\
    --strategy {strategy} \\
    --dataset {dataset} \\
    --model {model} \\
    --real-gen-ratio {ratio} \\
    --domain-filter clear_day'''
    else:
        train_cmd = f'''python unified_training.py \\
    --strategy {strategy} \\
    --dataset {dataset} \\
    --model {model} \\
    --domain-filter clear_day'''
    
    script = f'''#!/bin/bash
#BSUB -J {job_name}
#BSUB -o {LOGS_DIR}/{job_name}_%J.out
#BSUB -e {LOGS_DIR}/{job_name}_%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=8000]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=20G"
#BSUB -W 48:00
#BSUB -q BatchGPU

echo "Starting IDD-AW retraining with fixed labels"
echo "Job ID: $LSB_JOBID"
echo "Strategy: {strategy}"
echo "Dataset: {dataset}"
echo "Model: {model}"
echo "Ratio: {ratio if ratio else 'N/A'}"

cd {os.getcwd()}
source ~/.bashrc
mamba activate prove

# Pre-flight check: Ensure no duplicate training is running
echo "Running pre-flight lock check..."
if ! python training_lock.py check --strategy {strategy} --dataset {dataset} --model {model} {ratio_lock_arg}; then
    echo "Pre-flight check FAILED: Another job is training this configuration"
    echo "Exiting to avoid duplicate work"
    exit 0
fi
echo "Pre-flight check PASSED"
{backup_cmd}
# Acquire training lock and run training
echo "Acquiring training lock..."
python -c "
import sys, os, subprocess
sys.path.insert(0, '{os.getcwd()}')
from training_lock import TrainingLock

lock = TrainingLock('{strategy}', '{dataset}', '{model}', {ratio if ratio else None})
if not lock.acquire():
    print('Failed to acquire training lock - another job started training')
    sys.exit(0)
print('Training lock acquired successfully')

try:
    # Run training
    result = subprocess.run('{train_cmd}'.split(), check=False)
    exit_code = result.returncode
finally:
    lock.release()
    print('Training lock released')

sys.exit(exit_code)
"

echo "Training completed"
'''
    return script


def generate_multi_dataset_lsf_script(strategy: str, datasets: List[str], model: str,
                                      backup_existing: bool = True) -> str:
    """Generate an LSF job script for multi-dataset retraining with pre-flight lock."""
    
    datasets_str = '+'.join(d.lower() for d in datasets)
    job_name = f'retrain_{strategy}_multi_{model}'
    datasets_lock = '_'.join(d.lower() for d in datasets)
    
    # Backup existing checkpoint (with old ACDC-included name)
    old_existing_dir = f'{WEIGHTS_ROOT}/{strategy}/multi_acdc+{datasets_str}/{model}'
    backup_cmd = ''
    if backup_existing:
        backup_cmd = f'''
# Backup existing checkpoint (old version with ACDC)
if [ -d "{old_existing_dir}" ]; then
    mv "{old_existing_dir}" "{old_existing_dir}_with_acdc_backup"
    echo "Backed up old checkpoint with ACDC"
fi
'''
    
    train_cmd = f"python unified_training.py --strategy {strategy} --multi-dataset --datasets {' '.join(datasets)} --model {model}"
    
    script = f'''#!/bin/bash
#BSUB -J {job_name}
#BSUB -o {LOGS_DIR}/{job_name}_%J.out
#BSUB -e {LOGS_DIR}/{job_name}_%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=8000]"
#BSUB -gpu "num=1:mode=exclusive_process:gmem=20G"
#BSUB -W 96:00
#BSUB -q BatchGPU

echo "Starting multi-dataset retraining with fixed IDD-AW labels"
echo "Job ID: $LSB_JOBID"
echo "Strategy: {strategy}"
echo "Datasets: {', '.join(datasets)} (removed ACDC)"
echo "Model: {model}"

cd {os.getcwd()}
source ~/.bashrc
mamba activate prove

# Pre-flight check: Ensure no duplicate training is running
echo "Running pre-flight lock check..."
if ! python training_lock.py check --strategy {strategy} --dataset multi_{datasets_lock} --model {model}; then
    echo "Pre-flight check FAILED: Another job is training this configuration"
    echo "Exiting to avoid duplicate work"
    exit 0
fi
echo "Pre-flight check PASSED"
{backup_cmd}
# Acquire training lock and run training
echo "Acquiring training lock..."
python -c "
import sys, os, subprocess
sys.path.insert(0, '{os.getcwd()}')
from training_lock import TrainingLock

lock = TrainingLock('{strategy}', 'multi_{datasets_lock}', '{model}')
if not lock.acquire():
    print('Failed to acquire training lock - another job started training')
    sys.exit(0)
print('Training lock acquired successfully')

try:
    # Run training
    result = subprocess.run('{train_cmd}'.split(), check=False)
    exit_code = result.returncode
finally:
    lock.release()
    print('Training lock released')

sys.exit(exit_code)
"

echo "Training completed"
'''
    return script


def submit_job(script_content: str, script_name: str, dry_run: bool = False) -> int:
    """Submit an LSF job and return the job ID."""
    
    # Create logs directory
    os.makedirs('logs/retrain', exist_ok=True)
    
    # Write script to temp file
    script_path = f'/tmp/{script_name}.sh'
    with open(script_path, 'w') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)
    
    if dry_run:
        print(f"[DRY RUN] Would submit: {script_name}")
        print(f"Script saved to: {script_path}")
        return -1
    
    # Submit job using LSF bsub
    result = subprocess.run(
        ['bsub'],
        input=script_content,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error submitting job: {result.stderr}")
        return -1
    
    # Extract job ID from LSF output (e.g., "Job <12345> is submitted to queue <BatchGPU>.")
    import re
    match = re.search(r'<(\d+)>', result.stdout)
    if match:
        job_id = int(match.group(1))
        print(f"Submitted job {job_id}: {script_name}")
        return job_id
    else:
        print(f"Submitted: {script_name} (could not parse job ID)")
        return -1


def main():
    parser = argparse.ArgumentParser(description='Retrain IDD-AW models with fixed label transform')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing')
    parser.add_argument('--generate-only', action='store_true', help='Generate job scripts to files without submitting')
    parser.add_argument('--no-backup', action='store_true', help='Skip backing up existing checkpoints')
    parser.add_argument('--single-dataset-only', action='store_true', help='Only retrain single-dataset models')
    parser.add_argument('--multi-dataset-only', action='store_true', help='Only retrain multi-dataset models')
    parser.add_argument('--strategy', type=str, help='Only retrain specific strategy')
    parser.add_argument('--model', type=str, help='Only retrain specific model')
    args = parser.parse_args()
    
    backup_existing = not args.no_backup
    job_ids = []
    
    # Create jobs directory if generating scripts
    jobs_dir = Path('scripts/iddaw_retrain_jobs')
    if args.generate_only:
        jobs_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("IDD-AW Label Fix Retraining Script (LSF)")
    print("=" * 70)
    print()
    print("ISSUE: IDD-AW labels were incorrectly transformed because they were")
    print("       included in CITYSCAPES_LABEL_ID_DATASETS, but they are already")
    print("       in trainID format (0-18), not labelID format (0-33).")
    print()
    print("FIX: Removed IDD-AW from CITYSCAPES_LABEL_ID_DATASETS")
    print()
    print("=" * 70)
    
    # Single-dataset models
    if not args.multi_dataset_only:
        print("\n=== Single-Dataset IDD-AW Models ===")
        for strategy, dataset, model, ratio in MODELS_TO_RETRAIN:
            if args.strategy and strategy != args.strategy:
                continue
            if args.model and model != args.model:
                continue
            
            script = generate_lsf_script(strategy, dataset, model, ratio, backup_existing)
            script_name = f'retrain_iddaw_{strategy}_{model}'
            if ratio:
                script_name += f'_{int(ratio*100)}'
            
            if args.generate_only:
                script_path = jobs_dir / f'{script_name}.sh'
                with open(script_path, 'w') as f:
                    f.write(script)
                os.chmod(script_path, 0o755)
                print(f"  Created: {script_path}")
            else:
                job_id = submit_job(script, script_name, args.dry_run)
                if job_id > 0:
                    job_ids.append(job_id)
    
    # Multi-dataset models
    if not args.single_dataset_only:
        print("\n=== Multi-Dataset Models (including IDD-AW) ===")
        for strategy, datasets, model in MULTI_DATASET_MODELS:
            if args.strategy and strategy != args.strategy:
                continue
            if args.model and model != args.model:
                continue
            
            script = generate_multi_dataset_lsf_script(strategy, datasets, model, backup_existing)
            script_name = f'retrain_multi_{strategy}_{model}'
            
            if args.generate_only:
                script_path = jobs_dir / f'{script_name}.sh'
                with open(script_path, 'w') as f:
                    f.write(script)
                os.chmod(script_path, 0o755)
                print(f"  Created: {script_path}")
            else:
                job_id = submit_job(script, script_name, args.dry_run)
                if job_id > 0:
                    job_ids.append(job_id)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if args.generate_only:
        total_scripts = len(MODELS_TO_RETRAIN) + len(MULTI_DATASET_MODELS)
        print(f"Generated {total_scripts} job scripts in {jobs_dir}")
        print(f"\nTo submit all jobs:")
        print(f"  for f in {jobs_dir}/*.sh; do bsub < \"$f\"; done")
    elif args.dry_run:
        print(f"[DRY RUN] Would submit {len(MODELS_TO_RETRAIN) + len(MULTI_DATASET_MODELS)} jobs")
    else:
        print(f"Submitted {len(job_ids)} jobs")
        if job_ids:
            print(f"Job IDs: {', '.join(map(str, job_ids))}")
            print(f"\nMonitor with: bjobs")
            print(f"Logs in: logs/retrain/")


if __name__ == '__main__':
    main()
