#!/usr/bin/env python3
"""
Cityscapes Replication Training Script

This script runs standard mmsegmentation configs to replicate official Cityscapes results.
It verifies that our training infrastructure can achieve published benchmarks.

Usage:
    python cityscapes_train.py --model segformer_mit-b5 --submit
    python cityscapes_train.py --model all --dry-run
"""

import argparse
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

# Define the 10 models to replicate
MODELS = {
    # SegFormer family (best performing)
    'segformer_mit-b5': {
        'config': 'configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py',
        'expected_miou': 82.25,
        'crop_size': (1024, 1024),
        'batch_size': 1,  # Per GPU
        'iters': 160000,
    },
    'segformer_mit-b4': {
        'config': 'configs/segformer/segformer_mit-b4_8xb1-160k_cityscapes-1024x1024.py',
        'expected_miou': 81.89,
        'crop_size': (1024, 1024),
        'batch_size': 1,
        'iters': 160000,
    },
    'segformer_mit-b3': {
        'config': 'configs/segformer/segformer_mit-b3_8xb1-160k_cityscapes-1024x1024.py',
        'expected_miou': 81.94,
        'crop_size': (1024, 1024),
        'batch_size': 1,
        'iters': 160000,
    },
    'segformer_mit-b2': {
        'config': 'configs/segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py',
        'expected_miou': 81.08,
        'crop_size': (1024, 1024),
        'batch_size': 1,
        'iters': 160000,
    },
    # DeepLabV3+ family
    'deeplabv3plus_r101': {
        'config': 'configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb2-80k_cityscapes-769x769.py',
        'expected_miou': 80.98,
        'crop_size': (769, 769),
        'batch_size': 2,
        'iters': 80000,
    },
    'deeplabv3plus_r50': {
        'config': 'configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb2-80k_cityscapes-769x769.py',
        'expected_miou': 79.61,
        'crop_size': (769, 769),
        'batch_size': 2,
        'iters': 80000,
    },
    # PSPNet family
    'pspnet_r101': {
        'config': 'configs/pspnet/pspnet_r101-d8_4xb2-80k_cityscapes-769x769.py',
        'expected_miou': 79.76,
        'crop_size': (769, 769),
        'batch_size': 2,
        'iters': 80000,
    },
    'pspnet_r50': {
        'config': 'configs/pspnet/pspnet_r50-d8_4xb2-80k_cityscapes-769x769.py',
        'expected_miou': 78.55,
        'crop_size': (769, 769),
        'batch_size': 2,
        'iters': 80000,
    },
    # OCRNet
    'ocrnet_hr48': {
        'config': 'configs/ocrnet/ocrnet_hr48_4xb2-80k_cityscapes-512x1024.py',
        'expected_miou': 80.70,
        'crop_size': (512, 1024),
        'batch_size': 2,
        'iters': 80000,
    },
    # HRNet
    'fcn_hr48': {
        'config': 'configs/hrnet/fcn_hr48_4xb2-80k_cityscapes-512x1024.py',
        'expected_miou': 80.65,
        'crop_size': (512, 1024),
        'batch_size': 2,
        'iters': 80000,
    },
}

# LSF job template
LSF_TEMPLATE = """#!/bin/bash
#BSUB -J cityscapes_{model_name}
#BSUB -o ${AWARE_DATA_ROOT}/CITYSCAPES_REPLICATION/{model_name}/lsf_%J.out
#BSUB -e ${AWARE_DATA_ROOT}/CITYSCAPES_REPLICATION/{model_name}/lsf_%J.err
#BSUB -q BatchGPU
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=8:mode=exclusive_process"
#BSUB -W 48:00
#BSUB -R "rusage[mem=32000]"

# Load conda environment
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate prove

# Navigate to mmsegmentation
cd ${HOME}/repositories/mmsegmentation

# Create output directory
mkdir -p ${AWARE_DATA_ROOT}/CITYSCAPES_REPLICATION/{model_name}

# Run training with distributed launcher
PYTHONPATH="$(dirname $0)/..:$PYTHONPATH" \\
python -m torch.distributed.launch \\
    --nproc_per_node=8 \\
    --master_port={port} \\
    tools/train.py \\
    {config} \\
    --work-dir ${AWARE_DATA_ROOT}/CITYSCAPES_REPLICATION/{model_name} \\
    --launcher pytorch \\
    --cfg-options \\
        data_root=${AWARE_DATA_ROOT}/CITYSCAPES_DATA

echo "Training complete for {model_name}"
"""


def get_available_port(base_port: int = 29500) -> int:
    """Get a random available port for distributed training."""
    import random
    return base_port + random.randint(0, 1000)


def create_job_script(model_name: str, model_cfg: Dict, output_dir: Path) -> Path:
    """Create an LSF job script for training."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    job_content = LSF_TEMPLATE.format(
        model_name=model_name,
        config=model_cfg['config'],
        port=get_available_port(),
    )
    
    job_file = output_dir / f'{model_name}.job'
    job_file.write_text(job_content)
    return job_file


def submit_job(job_file: Path, dry_run: bool = False) -> Optional[str]:
    """Submit an LSF job."""
    if dry_run:
        print(f"[DRY-RUN] Would submit: {job_file}")
        return None
    
    result = subprocess.run(
        ['bsub', '<', str(job_file)],
        shell=True,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        # Extract job ID from output
        import re
        match = re.search(r'Job <(\d+)>', result.stdout)
        if match:
            return match.group(1)
    else:
        print(f"Error submitting {job_file}: {result.stderr}")
    return None


def main():
    parser = argparse.ArgumentParser(description='Cityscapes Replication Training')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all'] + list(MODELS.keys()),
                       help='Model to train (default: all)')
    parser.add_argument('--submit', action='store_true',
                       help='Submit jobs to LSF')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without submitting')
    parser.add_argument('--output-dir', type=Path,
                       default=Path('${HOME}/repositories/PROVE/cityscapes_replication/jobs'),
                       help='Directory to save job scripts')
    args = parser.parse_args()
    
    # Select models to train
    if args.model == 'all':
        models_to_train = MODELS
    else:
        models_to_train = {args.model: MODELS[args.model]}
    
    print("=" * 70)
    print("Cityscapes Replication Training Setup")
    print("=" * 70)
    print(f"\nModels to train: {len(models_to_train)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'SUBMIT' if args.submit else 'GENERATE ONLY'}")
    print()
    
    # Show model details
    print("Model Configurations:")
    print("-" * 70)
    for name, cfg in models_to_train.items():
        print(f"  {name}:")
        print(f"    Expected mIoU: {cfg['expected_miou']}%")
        print(f"    Crop size: {cfg['crop_size']}")
        print(f"    Iterations: {cfg['iters']:,}")
        print(f"    Config: {cfg['config']}")
    print()
    
    # Create job scripts
    job_files = []
    for name, cfg in models_to_train.items():
        job_file = create_job_script(name, cfg, args.output_dir)
        job_files.append((name, job_file))
        print(f"Created: {job_file}")
    
    # Submit if requested
    if args.submit or args.dry_run:
        print("\n" + "=" * 70)
        print("Job Submission")
        print("=" * 70)
        
        for name, job_file in job_files:
            job_id = submit_job(job_file, dry_run=args.dry_run)
            if job_id:
                print(f"  {name}: Job {job_id} submitted")
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total jobs: {len(job_files)}")
    print("\nTo submit all jobs manually:")
    print(f"  cd {args.output_dir} && for f in *.job; do bsub < $f; done")
    print()


if __name__ == '__main__':
    main()
