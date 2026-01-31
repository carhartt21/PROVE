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
CONFIG_DIR = Path('/home/mima2416/repositories/PROVE/cityscapes_replication/configs')
WORK_DIR_BASE = Path('/scratch/aaa_exchange/AWARE/CITYSCAPES_REPLICATION')
CONDA_ENV = 'prove'

# Job definitions - updated with new models
JOBS = {
    'segformer_b3': {
        'config': 'segformer_mit-b3_cityscapes_1024x1024.py',
        'gpus': 4,
        'mem': '32000',
        'hours': 48,  # 160k iterations
        'expected_miou': 81.94,
    },
    'hrnet_hr48': {
        'config': 'hrnet_hr48_cityscapes_512x1024.py',
        'gpus': 4,
        'mem': '32000',
        'hours': 48,
        'expected_miou': 80.65,
    },
    'ocrnet_hr48': {
        'config': 'ocrnet_hr48_cityscapes_512x1024.py',
        'gpus': 4,
        'mem': '32000',
        'hours': 48,
        'expected_miou': 81.35,
    },
    'deeplabv3plus_r50': {
        'config': 'deeplabv3plus_r50_cityscapes_769x769.py',
        'gpus': 4,
        'mem': '32000',
        'hours': 24,  # 80k iterations is faster
        'expected_miou': 79.61,
    },
    'pspnet_r50': {
        'config': 'pspnet_r50_cityscapes_769x769.py',
        'gpus': 4,
        'mem': '32000',
        'hours': 24,
        'expected_miou': 78.55,
    },
    'segnext_mscan_b': {
        'config': 'segnext_mscan-b_cityscapes_512x1024.py',
        'gpus': 4,
        'mem': '32000',
        'hours': 48,  # 160k iterations
        'expected_miou': 79.0,  # Estimated, not officially benchmarked
    },
}

LSF_TEMPLATE = '''#!/bin/bash
#BSUB -J cs_{model}
#BSUB -q BatchGPU
#BSUB -o {work_dir}/train_%J.out
#BSUB -e {work_dir}/train_%J.err
#BSUB -W {hours}:00
#BSUB -n {gpus}
#BSUB -R "span[hosts=1]"
#BSUB -M {mem}
#BSUB -R "rusage[mem={mem}]"
#BSUB -gpu "num={gpus}:mode=exclusive_process"

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Config: {config}"
echo "Work dir: {work_dir}"
echo "Expected mIoU: {expected_miou}%"

# Activate environment
source /home/mima2416/miniconda3/etc/profile.d/conda.sh
conda activate {conda_env}

# Set up distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=$((29500 + RANDOM % 100))

# Run training with torchrun (modern distributed launcher)
cd /home/mima2416/repositories/PROVE/cityscapes_replication

torchrun --nproc_per_node={gpus} --master_port=$MASTER_PORT \\
    train.py \\
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
    
    print("\nExpected Results Comparison:")
    print("-" * 60)
    print(f"{'Model':<25} {'PROVE (Current)':<15} {'Expected':<15}")
    print("-" * 60)
    print(f"{'SegFormer MIT-B3':<25} {'~45%':<15} {'81.94%':<15}")
    print(f"{'HRNet HR48':<25} {'N/A':<15} {'80.65%':<15}")
    print(f"{'OCRNet HR48':<25} {'N/A':<15} {'81.35%':<15}")
    print(f"{'DeepLabV3+ R50':<25} {'~38%':<15} {'79.61%':<15}")
    print(f"{'PSPNet R50':<25} {'~35%':<15} {'78.55%':<15}")
    print(f"{'SegNeXt MSCAN-B':<25} {'N/A':<15} {'~79% (est)':<15}")
    print("-" * 60)
    print("\nIf replication achieves expected results, the pipeline bug is confirmed.")
    print("NOTE: SegNeXt is not officially benchmarked on Cityscapes (estimate only).")


if __name__ == '__main__':
    main()
