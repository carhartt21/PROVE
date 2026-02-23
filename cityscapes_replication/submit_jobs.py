#!/usr/bin/env python3
"""
Submit Cityscapes replication training jobs to LSF.

Usage:
    python submit_jobs.py --dry-run     # Preview jobs without submitting
    python submit_jobs.py               # Submit all jobs
    python submit_jobs.py --model segformer_b3  # Submit only SegFormer B3
"""

import argparse
import subprocess
import os
from pathlib import Path
from datetime import datetime

# Configuration
CONFIG_DIR = Path('/home/chge7185/repositories/PROVE/cityscapes_replication/configs')
WORK_DIR_BASE = Path('${AWARE_DATA_ROOT}/CITYSCAPES_REPLICATION')
CONDA_ENV = 'prove'
REPO_DIR = Path('/home/chge7185/repositories/PROVE')

# Job definitions - 512x512 crop size for comparison with PROVE
# Using single GPU - hours increased from original 4-GPU estimates
JOBS = {
    'segformer_b3': {
        'config': 'segformer_mit-b3_cityscapes_512x512.py',
        'gpus': 1,
        'mem': '48000',
        'hours': 96,  # 160k iterations, single GPU
        'expected_miou': 80.0,  # Lower than 1024x1024 due to smaller crop
    },
    'hrnet_hr48': {
        'config': 'hrnet_hr48_cityscapes_512x512.py',
        'gpus': 1,
        'mem': '48000',
        'hours': 96,
        'expected_miou': 78.0,
    },
    'ocrnet_hr48': {
        'config': 'ocrnet_hr48_cityscapes_512x512.py',
        'gpus': 1,
        'mem': '48000',
        'hours': 96,
        'expected_miou': 79.0,
    },
    'deeplabv3plus_r50': {
        'config': 'deeplabv3plus_r50_cityscapes_512x512.py',
        'gpus': 1,
        'mem': '48000',
        'hours': 48,  # 80k iterations is faster
        'expected_miou': 77.0,
    },
    'pspnet_r50': {
        'config': 'pspnet_r50_cityscapes_512x512.py',
        'gpus': 1,
        'mem': '48000',
        'hours': 48,
        'expected_miou': 76.0,
    },
    'segnext_mscan_b': {
        'config': 'segnext_mscan-b_cityscapes_512x512.py',
        'gpus': 1,
        'mem': '48000',
        'hours': 96,  # 160k iterations
        'expected_miou': 77.0,  # Estimated
    },
    # PROPER CROP SIZE JOBS - should match published results
    'deeplabv3plus_r50_769': {
        'config': 'deeplabv3plus_r50_cityscapes_769x769.py',
        'gpus': 1,
        'mem': '64000',  # Larger crop needs more memory
        'hours': 72,  # 80k iterations with larger crop
        'expected_miou': 79.6,  # Official published mIoU
    },
    'pspnet_r50_769': {
        'config': 'pspnet_r50_cityscapes_769x769.py',
        'gpus': 1,
        'mem': '64000',
        'hours': 72,
        'expected_miou': 78.5,  # Official published mIoU
    },
    'hrnet_hr48_1024': {
        'config': 'hrnet_hr48_cityscapes_512x1024.py',
        'gpus': 1,
        'mem': '64000',
        'hours': 120,  # 160k iterations with larger crop
        'expected_miou': 80.6,  # Official published mIoU
    },
    'ocrnet_hr48_1024': {
        'config': 'ocrnet_hr48_cityscapes_512x1024.py',
        'gpus': 1,
        'mem': '64000',
        'hours': 120,
        'expected_miou': 81.3,  # Official published mIoU
    },
}

LSF_TEMPLATE = '''#!/bin/bash
#BSUB -J cs_{model}
#BSUB -q BatchGPU
#BSUB -o {work_dir}/train_%J.out
#BSUB -e {work_dir}/train_%J.err
#BSUB -W {hours}:00
#BSUB -n 1
#BSUB -gpu "num=1:gmem=36GB"

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $LSB_JOBID"
echo "Config: {config}"
echo "Work dir: {work_dir}"
echo "Expected mIoU: {expected_miou}%"

# Set permissions
umask 002

# Activate environment
source ~/.bashrc
mamba activate {conda_env}

# Create work directory
mkdir -p {work_dir}
cd {repo_dir}/cityscapes_replication

# Run training with single GPU (no torchrun needed for single GPU)
python train.py \\
    {config_path} \\
    --work-dir {work_dir}

echo "Job completed at $(date)"
'''


def submit_job(model: str, job_info: dict, dry_run: bool = False):
    """Submit a single training job."""
    work_dir = WORK_DIR_BASE / model
    config_path = CONFIG_DIR / job_info['config']
    
    # Create work directory
    if not dry_run:
        work_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate job script
    job_script = LSF_TEMPLATE.format(
        model=model,
        config=job_info['config'],
        work_dir=work_dir,
        config_path=config_path,
        gpus=job_info['gpus'],
        mem=job_info['mem'],
        hours=job_info['hours'],
        expected_miou=job_info['expected_miou'],
        conda_env=CONDA_ENV,
        repo_dir=REPO_DIR,
    )
    
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"Config: {config_path}")
    print(f"Work dir: {work_dir}")
    print(f"GPUs: {job_info['gpus']}")
    print(f"Expected mIoU: {job_info['expected_miou']}%")
    print(f"{'='*60}")
    
    if dry_run:
        print("DRY RUN - Would submit job")
        return None
    
    # Write job script
    job_file = work_dir / 'train_job.sh'
    job_file.write_text(job_script)
    
    # Submit job
    result = subprocess.run(
        ['bsub'],
        capture_output=True,
        text=True,
        input=job_script
    )
    
    if result.returncode == 0:
        print(f"✅ Submitted: {result.stdout.strip()}")
        return result.stdout.strip()
    else:
        print(f"❌ Failed: {result.stderr}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Submit Cityscapes replication jobs')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Preview jobs without submitting')
    parser.add_argument('--model', choices=list(JOBS.keys()),
                       help='Submit only a specific model')
    args = parser.parse_args()
    
    print(f"Cityscapes Replication Training Jobs")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config dir: {CONFIG_DIR}")
    print(f"Work dir base: {WORK_DIR_BASE}")
    
    if args.dry_run:
        print("\n*** DRY RUN MODE - No jobs will be submitted ***")
    
    # Select jobs to submit
    if args.model:
        jobs_to_submit = {args.model: JOBS[args.model]}
    else:
        jobs_to_submit = JOBS
    
    # Submit jobs
    results = {}
    for model, job_info in jobs_to_submit.items():
        results[model] = submit_job(model, job_info, dry_run=args.dry_run)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    submitted = sum(1 for r in results.values() if r is not None)
    failed = len(results) - submitted
    
    if args.dry_run:
        print(f"Would submit: {len(results)} jobs")
    else:
        print(f"Submitted: {submitted} jobs")
        if failed > 0:
            print(f"Failed: {failed} jobs")
    
    print("\nExpected Results Comparison (all 512x512 crop):")
    print("-" * 60)
    print(f"{'Model':<25} {'PROVE (Current)':<15} {'Expected':<15}")
    print("-" * 60)
    print(f"{'SegFormer MIT-B3':<25} {'~45%':<15} {'~80%':<15}")
    print(f"{'HRNet HR48':<25} {'N/A':<15} {'~78%':<15}")
    print(f"{'OCRNet HR48':<25} {'N/A':<15} {'~79%':<15}")
    print(f"{'DeepLabV3+ R50':<25} {'~38%':<15} {'~77%':<15}")
    print(f"{'PSPNet R50':<25} {'~35%':<15} {'~76%':<15}")
    print(f"{'SegNeXt MSCAN-B':<25} {'N/A':<15} {'~77% (est)':<15}")
    print("-" * 60)
    print("\nIf replication achieves expected results, the pipeline bug is confirmed.")
    print("NOTE: All configs use 512x512 crop size to match PROVE training.")


if __name__ == '__main__':
    main()
