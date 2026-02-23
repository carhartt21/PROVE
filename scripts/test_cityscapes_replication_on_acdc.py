#!/usr/bin/env python3
"""
Test Cityscapes Replication Models on ACDC Dataset

This script evaluates the Cityscapes replication models on the ACDC dataset
to measure cross-domain performance. It provides per-domain breakdowns for:
- foggy
- night
- rainy
- snowy

Usage:
    # Dry run - show what would be tested
    python scripts/test_cityscapes_replication_on_acdc.py --dry-run
    
    # Test all available models
    python scripts/test_cityscapes_replication_on_acdc.py
    
    # Test specific models
    python scripts/test_cityscapes_replication_on_acdc.py --models segformer_b3 segnext_mscan_b
    
    # Submit as LSF jobs
    python scripts/test_cityscapes_replication_on_acdc.py --submit-jobs
    
    # Use specific checkpoint (best or final)
    python scripts/test_cityscapes_replication_on_acdc.py --checkpoint-type best
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
CITYSCAPES_REPLICATION_ROOT = Path('${AWARE_DATA_ROOT}/CITYSCAPES_REPLICATION')
DATA_ROOT = Path('${AWARE_DATA_ROOT}/FINAL_SPLITS')
OUTPUT_ROOT = CITYSCAPES_REPLICATION_ROOT / 'acdc_cross_domain_results'

# Model configurations
# Maps folder name to display name (checkpoints and configs auto-discovered)
REPLICATION_MODELS = {
    # Standard 512x512 crop models
    'segformer_b3': {
        'display_name': 'SegFormer MIT-B3 (512×512)',
    },
    'segnext_mscan_b': {
        'display_name': 'SegNeXt MSCAN-B (512×512)',
    },
    'hrnet_hr48': {
        'display_name': 'HRNet HR48 (512×512)',
    },
    'deeplabv3plus_r50': {
        'display_name': 'DeepLabV3+ R50 (512×512)',
    },
    'pspnet_r50': {
        'display_name': 'PSPNet R50 (512×512)',
    },
    'ocrnet_hr48': {
        'display_name': 'OCRNet HR48 (512×512)',
    },
    # Special crop size variants
    'deeplabv3plus_r50_769': {
        'display_name': 'DeepLabV3+ R50 (769×769)',
    },
    'pspnet_r50_769': {
        'display_name': 'PSPNet R50 (769×769)',
    },
    'hrnet_hr48_1024': {
        'display_name': 'HRNet HR48 (512×1024)',
    },
    'ocrnet_hr48_1024': {
        'display_name': 'OCRNet HR48 (512×1024)',
    },
}

# ACDC domains
ACDC_DOMAINS = ['foggy', 'night', 'rainy', 'snowy']


def find_available_models() -> Dict[str, dict]:
    """Find which models have trained checkpoints available."""
    available = {}
    
    for model_name, config in REPLICATION_MODELS.items():
        model_dir = CITYSCAPES_REPLICATION_ROOT / model_name
        
        if not model_dir.exists():
            continue
            
        # Check for config file
        config_files = list(model_dir.glob('*.py'))
        if not config_files:
            continue
            
        # Auto-discover checkpoints
        # Look for best_mIoU_iter_*.pth (best) and iter_*.pth (final/latest)
        best_checkpoints = sorted(model_dir.glob('best_mIoU_iter_*.pth'), 
                                   key=lambda p: int(p.stem.split('_')[-1]))
        final_checkpoints = sorted([p for p in model_dir.glob('iter_*.pth') 
                                    if 'best' not in p.stem],
                                    key=lambda p: int(p.stem.split('_')[-1]))
        
        best_ckpt = best_checkpoints[-1] if best_checkpoints else None
        final_ckpt = final_checkpoints[-1] if final_checkpoints else None
        
        if final_ckpt or best_ckpt:
            available[model_name] = {
                'dir': model_dir,
                'config': config_files[0],  # Use first config found
                'final_checkpoint': final_ckpt,
                'best_checkpoint': best_ckpt,
                'display_name': config['display_name'],
            }
    
    return available


def run_test(
    model_name: str,
    model_info: dict,
    checkpoint_type: str = 'best',
    dry_run: bool = False,
) -> Optional[dict]:
    """
    Run fine-grained test on ACDC for a single model.
    
    Args:
        model_name: Name of the model
        model_info: Model configuration dictionary
        checkpoint_type: 'best' or 'final'
        dry_run: If True, only print command without executing
    
    Returns:
        Results dictionary if successful, None otherwise
    """
    # Select checkpoint
    if checkpoint_type == 'best' and model_info['best_checkpoint']:
        checkpoint = model_info['best_checkpoint']
    elif model_info['final_checkpoint']:
        checkpoint = model_info['final_checkpoint']
    else:
        print(f"  ERROR: No checkpoint found for {model_name}")
        return None
    
    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = OUTPUT_ROOT / model_name / f'{checkpoint_type}_{timestamp}'
    
    # Build command
    cmd = [
        'python', str(PROJECT_ROOT / 'fine_grained_test.py'),
        '--config', str(model_info['config']),
        '--checkpoint', str(checkpoint),
        '--output-dir', str(output_dir),
        '--dataset', 'ACDC',
        '--data-root', str(DATA_ROOT),
        '--test-split', 'test',
        '--batch-size', '10',
    ]
    
    if dry_run:
        print(f"  Would run: {' '.join(cmd)}")
        return None
    
    print(f"  Running: {' '.join(cmd)}")
    
    # Execute
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        
        if result.returncode != 0:
            print(f"  ERROR: Test failed with return code {result.returncode}")
            print(f"  STDERR: {result.stderr[:500]}")
            return None
        
        # Load results - fine_grained_test creates a timestamp subdirectory
        # Look for results.json in output_dir or its subdirectories
        results_file = output_dir / 'results.json'
        if not results_file.exists():
            # Check for timestamped subdirectory
            subdirs = sorted([d for d in output_dir.iterdir() if d.is_dir()], 
                           key=lambda x: x.name, reverse=True)
            if subdirs:
                results_file = subdirs[0] / 'results.json'
        
        if results_file.exists():
            with open(results_file) as f:
                return json.load(f)
        else:
            print(f"  ERROR: Results file not found in {output_dir}")
            return None
            
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def generate_lsf_script(
    model_name: str,
    model_info: dict,
    checkpoint_type: str = 'best',
) -> str:
    """Generate LSF job script for a model test."""
    # Select checkpoint
    if checkpoint_type == 'best' and model_info['best_checkpoint']:
        checkpoint = model_info['best_checkpoint']
    else:
        checkpoint = model_info['final_checkpoint']
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = OUTPUT_ROOT / model_name / f'{checkpoint_type}_{timestamp}'
    
    script = f'''#!/bin/bash
#BSUB -J acdc_test_{model_name}
#BSUB -q BatchGPU
#BSUB -o {output_dir}/test_%J.out
#BSUB -e {output_dir}/test_%J.err
#BSUB -n 4
#BSUB -gpu "num=1"

echo "=========================================="
echo "Testing {model_info['display_name']} on ACDC"
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "=========================================="

# Create output directory
mkdir -p {output_dir}

# Activate environment
source ~/.bashrc
mamba activate prove

# Run test
python {PROJECT_ROOT}/fine_grained_test.py \\
    --config {model_info['config']} \\
    --checkpoint {checkpoint} \\
    --output-dir {output_dir} \\
    --dataset ACDC \\
    --data-root {DATA_ROOT} \\
    --test-split test \\
    --batch-size 10

echo ""
echo "=========================================="
echo "Test completed: $(date)"
echo "Results: {output_dir}/results.json"
echo "=========================================="
'''
    return script


def submit_lsf_job(script: str, model_name: str) -> Optional[int]:
    """Submit an LSF job and return the job ID."""
    try:
        result = subprocess.run(
            ['bsub'],
            input=script,
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0:
            # Parse job ID from "Job <12345> is submitted..."
            import re
            match = re.search(r'Job <(\d+)>', result.stdout)
            if match:
                return int(match.group(1))
        
        print(f"  ERROR: Failed to submit job for {model_name}")
        print(f"  STDERR: {result.stderr}")
        return None
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def print_results_summary(results: Dict[str, dict]):
    """Print a formatted summary of cross-domain results."""
    print("\n" + "="*80)
    print("CITYSCAPES REPLICATION → ACDC CROSS-DOMAIN RESULTS")
    print("="*80)
    
    # Header
    print(f"\n{'Model':<35} {'Overall':>10} {'Foggy':>10} {'Night':>10} {'Rainy':>10} {'Snowy':>10}")
    print("-"*85)
    
    for model_name, result in results.items():
        if result is None:
            continue
            
        display_name = REPLICATION_MODELS.get(model_name, {}).get('display_name', model_name)
        overall = result.get('overall', {}).get('mIoU', 0)
        # Only scale if value is in 0-1 range (some results may be pre-scaled)
        if isinstance(overall, float) and overall <= 1:
            overall *= 100
        
        domain_values = []
        for domain in ACDC_DOMAINS:
            domain_data = result.get('per_domain', {}).get(domain, {})
            miou = domain_data.get('mIoU', domain_data.get('summary', {}).get('mIoU', 0))
            if isinstance(miou, float) and miou <= 1:
                miou *= 100
            domain_values.append(miou)
        
        print(f"{display_name:<35} {overall:>10.2f} {domain_values[0]:>10.2f} {domain_values[1]:>10.2f} {domain_values[2]:>10.2f} {domain_values[3]:>10.2f}")
    
    print("-"*85)
    print(f"\nNote: Values are mIoU percentages (0-100)")
    print(f"Models trained on Cityscapes clear weather, tested on ACDC adverse conditions")


def main():
    parser = argparse.ArgumentParser(
        description='Test Cityscapes replication models on ACDC dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    # Dry run - show what would be tested
    python scripts/test_cityscapes_replication_on_acdc.py --dry-run
    
    # Test all available models
    python scripts/test_cityscapes_replication_on_acdc.py
    
    # Test specific models
    python scripts/test_cityscapes_replication_on_acdc.py --models segformer_b3 segnext_mscan_b
    
    # Submit as LSF jobs
    python scripts/test_cityscapes_replication_on_acdc.py --submit-jobs
        ''',
    )
    
    parser.add_argument('--models', nargs='+', choices=list(REPLICATION_MODELS.keys()),
                       help='Specific models to test (default: all available)')
    parser.add_argument('--checkpoint-type', choices=['best', 'final'], default='best',
                       help='Which checkpoint to use (default: best)')
    parser.add_argument('--submit-jobs', action='store_true',
                       help='Submit as LSF cluster jobs instead of running locally')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print commands without executing')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Custom output directory (default: uses CITYSCAPES_REPLICATION/acdc_cross_domain_results)')
    
    args = parser.parse_args()
    
    # Update output root if specified
    global OUTPUT_ROOT
    if args.output_dir:
        OUTPUT_ROOT = Path(args.output_dir)
    
    print("\n" + "="*60)
    print("Cityscapes Replication → ACDC Cross-Domain Testing")
    print("="*60)
    
    # Find available models
    available_models = find_available_models()
    
    if not available_models:
        print("ERROR: No trained models found in", CITYSCAPES_REPLICATION_ROOT)
        return 1
    
    print(f"\nFound {len(available_models)} trained models:")
    for name, info in available_models.items():
        best_status = "✓" if info['best_checkpoint'] else "✗"
        final_status = "✓" if info['final_checkpoint'] else "✗"
        print(f"  - {info['display_name']:<35} [best: {best_status}, final: {final_status}]")
    
    # Filter to requested models
    if args.models:
        models_to_test = {k: v for k, v in available_models.items() if k in args.models}
        if not models_to_test:
            print(f"\nERROR: None of the requested models are available")
            return 1
    else:
        models_to_test = available_models
    
    print(f"\nTesting {len(models_to_test)} models on ACDC (4 domains)...")
    print(f"Checkpoint type: {args.checkpoint_type}")
    print(f"Output directory: {OUTPUT_ROOT}")
    
    if args.dry_run:
        print("\n[DRY RUN - No tests will be executed]\n")
    
    # Run tests
    results = {}
    
    if args.submit_jobs:
        # Submit LSF jobs
        print("\nSubmitting LSF jobs...")
        job_ids = {}
        
        for model_name, model_info in models_to_test.items():
            print(f"\n  {model_info['display_name']}:")
            
            if args.dry_run:
                script = generate_lsf_script(model_name, model_info, args.checkpoint_type)
                print(f"    Would submit job script:\n{script[:300]}...")
            else:
                script = generate_lsf_script(model_name, model_info, args.checkpoint_type)
                job_id = submit_lsf_job(script, model_name)
                if job_id:
                    job_ids[model_name] = job_id
                    print(f"    Submitted job {job_id}")
        
        if not args.dry_run and job_ids:
            print(f"\n\nSubmitted {len(job_ids)} jobs:")
            for name, jid in job_ids.items():
                print(f"  {name}: Job {jid}")
            print(f"\nMonitor with: bjobs -w | grep acdc_test")
        
    else:
        # Run locally
        for model_name, model_info in models_to_test.items():
            print(f"\n{'='*40}")
            print(f"Testing: {model_info['display_name']}")
            print(f"{'='*40}")
            
            result = run_test(model_name, model_info, args.checkpoint_type, args.dry_run)
            results[model_name] = result
        
        # Print summary
        if not args.dry_run and any(r is not None for r in results.values()):
            print_results_summary(results)
            
            # Save combined results
            combined_output = OUTPUT_ROOT / 'combined_results.json'
            combined_output.parent.mkdir(parents=True, exist_ok=True)
            
            combined = {
                'timestamp': datetime.now().isoformat(),
                'checkpoint_type': args.checkpoint_type,
                'results': {k: v for k, v in results.items() if v is not None},
            }
            
            with open(combined_output, 'w') as f:
                json.dump(combined, f, indent=2)
            
            print(f"\nCombined results saved to: {combined_output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
