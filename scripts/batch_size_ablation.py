#!/usr/bin/env python3
"""
Batch Size Ablation Study for PROVE

This script runs a minimal ablation study to quantify the impact of batch size
on training speed and model performance for the PROVE project.

Study Design:
- 1 model (DeepLabv3+ R50 - most used in the project)
- 1 dataset (BDD10k - good size and representative)
- 1 strategy (baseline - no extra complexity)
- 4 batch sizes: 2 (current), 4, 8, 16 (if memory allows)

For fair comparison, we use the Linear Scaling Rule:
- LR_new = LR_base × (BS_new / BS_base)

Total: 4 jobs × ~1 hour each = ~4 hours of GPU time

Usage:
    # Dry run to see commands
    python scripts/batch_size_ablation.py --dry-run
    
    # Submit all jobs
    python scripts/batch_size_ablation.py
    
    # Analyze results after completion
    python scripts/batch_size_ablation.py --analyze
"""

import argparse
import os
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Ablation configuration
ABLATION_CONFIG = {
    'output_base': '/scratch/aaa_exchange/AWARE/WEIGHTS_BATCH_SIZE_ABLATION',
    'dataset': 'BDD10k',
    'model': 'deeplabv3plus_r50',
    'strategy': 'baseline',
    'max_iters': 40000,  # Reduced from 80k to speed up ablation (still valid comparison)
    'domain_filter': 'clear_day',  # Consistent with Stage 1
    'seed': 42,
    
    # Batch sizes to test with corresponding learning rates (linear scaling)
    # Base: BS=2, LR=0.01
    'configurations': [
        {'batch_size': 2,  'lr': 0.01,  'warmup_iters': 500},
        {'batch_size': 4,  'lr': 0.02,  'warmup_iters': 500},
        {'batch_size': 8,  'lr': 0.04,  'warmup_iters': 1000},  # Extended warmup for larger LR
        {'batch_size': 16, 'lr': 0.08,  'warmup_iters': 1500},  # Extended warmup
    ],
    
    # LSF job settings
    'queue': 'BatchGPU',
    'gpu_mem': '24G',
    'wall_time': '4:00',  # 4 hours should be plenty for 40k iters
    'num_cpus': 8,
}


def generate_training_command(config: dict, batch_config: dict, work_dir: str) -> str:
    """Generate the training command for a specific batch size configuration."""
    
    cmd_parts = [
        'python', '/home/mima2416/repositories/PROVE/unified_training.py',
        '--dataset', config['dataset'],
        '--model', config['model'],
        '--strategy', config['strategy'],
        '--domain-filter', config['domain_filter'],
        '--max-iters', str(config['max_iters']),
        '--seed', str(config['seed']),
        '--work-dir', work_dir,
    ]
    
    return ' '.join(cmd_parts)


def generate_lsf_script(config: dict, batch_config: dict, job_name: str, work_dir: str) -> str:
    """Generate LSF job submission script."""
    
    bs = batch_config['batch_size']
    lr = batch_config['lr']
    warmup = batch_config['warmup_iters']
    
    # We need to modify the config to use different batch size/LR
    # This requires modifying unified_training.py to accept these parameters
    # For now, we'll create a custom config file
    
    script = f'''#!/bin/bash
#BSUB -J {job_name}
#BSUB -q {config['queue']}
#BSUB -gpu "num=1:mode=shared:gmem={config['gpu_mem']}"
#BSUB -n {config['num_cpus']}
#BSUB -W {config['wall_time']}
#BSUB -o {work_dir}/job_%J.out
#BSUB -e {work_dir}/job_%J.err

# Batch Size Ablation Job
# Batch Size: {bs}
# Learning Rate: {lr}
# Warmup Iterations: {warmup}

cd /home/mima2416/repositories/PROVE

# Activate conda environment
source ~/.bashrc
conda activate prove

# Create work directory
mkdir -p {work_dir}

# Log start time
echo "Job started at $(date)" > {work_dir}/timing.log
START_TIME=$(date +%s)

# Run training with custom batch size configuration
python unified_training.py \\
    --dataset {config['dataset']} \\
    --model {config['model']} \\
    --strategy {config['strategy']} \\
    --domain-filter {config['domain_filter']} \\
    --max-iters {config['max_iters']} \\
    --seed {config['seed']} \\
    --work-dir {work_dir} \\
    --batch-size {bs} \\
    --lr {lr} \\
    --warmup-iters {warmup}

# Log end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Job finished at $(date)" >> {work_dir}/timing.log
echo "Total duration: $DURATION seconds" >> {work_dir}/timing.log
echo "Duration in hours: $(echo "scale=2; $DURATION / 3600" | bc)" >> {work_dir}/timing.log

# Save configuration for analysis
cat > {work_dir}/ablation_config.json << EOF
{{
    "batch_size": {bs},
    "learning_rate": {lr},
    "warmup_iters": {warmup},
    "max_iters": {config['max_iters']},
    "dataset": "{config['dataset']}",
    "model": "{config['model']}",
    "strategy": "{config['strategy']}",
    "training_time_seconds": $DURATION
}}
EOF

echo "Batch size ablation job completed."
'''
    return script


def submit_ablation_jobs(dry_run: bool = True):
    """Submit all batch size ablation jobs."""
    
    config = ABLATION_CONFIG
    output_base = Path(config['output_base'])
    
    print("=" * 70)
    print("BATCH SIZE ABLATION STUDY FOR PROVE")
    print("=" * 70)
    print(f"\nDataset: {config['dataset']}")
    print(f"Model: {config['model']}")
    print(f"Strategy: {config['strategy']}")
    print(f"Max iterations: {config['max_iters']}")
    print(f"Output directory: {output_base}")
    print()
    
    jobs_to_submit = []
    
    for batch_config in config['configurations']:
        bs = batch_config['batch_size']
        lr = batch_config['lr']
        
        job_name = f"bs_ablation_bs{bs}"
        work_dir = output_base / f"batch_size_{bs}"
        
        print(f"\n--- Batch Size {bs} ---")
        print(f"  Learning Rate: {lr}")
        print(f"  Warmup Iters: {batch_config['warmup_iters']}")
        print(f"  Work Dir: {work_dir}")
        
        # Generate LSF script
        script_content = generate_lsf_script(config, batch_config, job_name, str(work_dir))
        script_path = output_base / f"submit_bs{bs}.sh"
        
        jobs_to_submit.append({
            'name': job_name,
            'batch_size': bs,
            'script_path': script_path,
            'script_content': script_content,
            'work_dir': work_dir,
        })
    
    if dry_run:
        print("\n" + "=" * 70)
        print("DRY RUN - No jobs submitted")
        print("=" * 70)
        print("\nTo submit jobs, run without --dry-run")
        print("\nGenerated scripts would be:")
        for job in jobs_to_submit:
            print(f"  - {job['script_path']}")
        
        # Show first script as example
        print("\n--- Example Script (BS=2) ---")
        print(jobs_to_submit[0]['script_content'][:1500])
        print("...")
        return
    
    # Create output directory
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Write and submit scripts
    print("\n" + "=" * 70)
    print("SUBMITTING JOBS")
    print("=" * 70)
    
    for job in jobs_to_submit:
        # Create work directory
        job['work_dir'].mkdir(parents=True, exist_ok=True)
        
        # Write script
        with open(job['script_path'], 'w') as f:
            f.write(job['script_content'])
        os.chmod(job['script_path'], 0o755)
        
        # Submit job
        result = subprocess.run(
            ['bsub', '<', str(job['script_path'])],
            shell=True,
            capture_output=True,
            text=True,
            cwd=str(output_base)
        )
        
        # Alternative submission method
        result = subprocess.run(
            f"bsub < {job['script_path']}",
            shell=True,
            capture_output=True,
            text=True,
            cwd=str(output_base)
        )
        
        if result.returncode == 0:
            print(f"✓ Submitted: {job['name']}")
            print(f"  {result.stdout.strip()}")
        else:
            print(f"✗ Failed: {job['name']}")
            print(f"  Error: {result.stderr}")
    
    print("\n" + "=" * 70)
    print("MONITORING")
    print("=" * 70)
    print("Check job status with: bjobs -w | grep bs_ablation")
    print(f"Results will be in: {output_base}")


def analyze_results():
    """Analyze completed batch size ablation results."""
    
    output_base = Path(ABLATION_CONFIG['output_base'])
    
    print("=" * 70)
    print("BATCH SIZE ABLATION RESULTS ANALYSIS")
    print("=" * 70)
    
    results = []
    
    for bs_config in ABLATION_CONFIG['configurations']:
        bs = bs_config['batch_size']
        work_dir = output_base / f"batch_size_{bs}"
        
        # Check for ablation config
        config_file = work_dir / 'ablation_config.json'
        if config_file.exists():
            with open(config_file) as f:
                ablation_info = json.load(f)
        else:
            ablation_info = {'training_time_seconds': None}
        
        # Check for checkpoint
        checkpoint = work_dir / 'iter_40000.pth'
        has_checkpoint = checkpoint.exists()
        
        # Check for test results
        test_results_dir = work_dir / 'test_results_detailed'
        has_results = False
        miou = None
        
        if test_results_dir.exists():
            # Find latest results
            result_files = list(test_results_dir.glob('*/results.json'))
            if result_files:
                has_results = True
                with open(sorted(result_files)[-1]) as f:
                    test_data = json.load(f)
                    miou = test_data.get('overall', {}).get('mIoU')
        
        results.append({
            'batch_size': bs,
            'learning_rate': bs_config['lr'],
            'has_checkpoint': has_checkpoint,
            'has_results': has_results,
            'miou': miou,
            'training_time': ablation_info.get('training_time_seconds'),
        })
    
    # Display results table
    print("\n{:<12} {:<10} {:<15} {:<15} {:<12} {:<15}".format(
        "Batch Size", "LR", "Checkpoint", "Test Results", "mIoU", "Train Time"
    ))
    print("-" * 80)
    
    for r in results:
        checkpoint_str = "✓" if r['has_checkpoint'] else "✗"
        results_str = "✓" if r['has_results'] else "✗"
        miou_str = f"{r['miou']:.2f}%" if r['miou'] else "-"
        time_str = f"{r['training_time']/3600:.2f}h" if r['training_time'] else "-"
        
        print("{:<12} {:<10.4f} {:<15} {:<15} {:<12} {:<15}".format(
            r['batch_size'],
            r['learning_rate'],
            checkpoint_str,
            results_str,
            miou_str,
            time_str
        ))
    
    # Analysis
    if all(r['miou'] for r in results):
        print("\n" + "=" * 70)
        print("ANALYSIS")
        print("=" * 70)
        
        base_miou = results[0]['miou']  # BS=2 baseline
        
        print(f"\nBaseline (BS=2): {base_miou:.2f}% mIoU")
        print("\nPerformance vs baseline:")
        
        for r in results[1:]:
            diff = r['miou'] - base_miou
            speedup = results[0]['training_time'] / r['training_time'] if r['training_time'] else None
            
            print(f"  BS={r['batch_size']}: {r['miou']:.2f}% mIoU ({diff:+.2f}%)", end="")
            if speedup:
                print(f", {speedup:.1f}x faster")
            else:
                print()
        
        # Recommendation
        print("\n" + "=" * 70)
        print("RECOMMENDATION")
        print("=" * 70)
        
        # Find best tradeoff (within 0.5% of baseline with best speedup)
        acceptable = [r for r in results if abs(r['miou'] - base_miou) <= 0.5]
        if len(acceptable) > 1 and acceptable[-1]['training_time']:
            best = max(acceptable, key=lambda x: (x['batch_size']))
            print(f"\nRecommended batch size: {best['batch_size']}")
            print(f"  - mIoU difference from baseline: {best['miou'] - base_miou:+.2f}%")
            if best['training_time'] and results[0]['training_time']:
                speedup = results[0]['training_time'] / best['training_time']
                print(f"  - Speed-up: {speedup:.1f}x")
    else:
        print("\n⚠ Not all experiments have completed. Run tests and re-analyze.")
        print("\nTo test completed checkpoints:")
        print("  python fine_grained_test.py --config <config> --checkpoint <checkpoint> ...")


def check_unified_training_support():
    """Check if unified_training.py supports --batch-size, --lr, --warmup-iters arguments."""
    
    result = subprocess.run(
        ['python', '/home/mima2416/repositories/PROVE/unified_training.py', '--help'],
        capture_output=True,
        text=True
    )
    
    help_text = result.stdout
    
    missing_args = []
    if '--batch-size' not in help_text:
        missing_args.append('--batch-size')
    if '--lr' not in help_text and '--learning-rate' not in help_text:
        missing_args.append('--lr')
    if '--warmup-iters' not in help_text and '--warmup' not in help_text:
        missing_args.append('--warmup-iters')
    
    return missing_args


def main():
    parser = argparse.ArgumentParser(
        description='Batch Size Ablation Study for PROVE',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Show commands without submitting jobs')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze completed results')
    parser.add_argument('--check-support', action='store_true',
                       help='Check if unified_training.py supports required arguments')
    
    args = parser.parse_args()
    
    if args.check_support:
        missing = check_unified_training_support()
        if missing:
            print("⚠ unified_training.py is missing these arguments:")
            for arg in missing:
                print(f"  - {arg}")
            print("\nThese need to be added before running the ablation study.")
        else:
            print("✓ unified_training.py supports all required arguments")
        return
    
    if args.analyze:
        analyze_results()
        return
    
    # Check if required arguments are supported
    missing = check_unified_training_support()
    if missing:
        print("=" * 70)
        print("⚠ MISSING ARGUMENT SUPPORT")
        print("=" * 70)
        print(f"\nunified_training.py needs these arguments: {', '.join(missing)}")
        print("\nWould you like me to add support for these arguments?")
        print("This is required before running the batch size ablation.")
        return
    
    submit_ablation_jobs(dry_run=args.dry_run)


if __name__ == '__main__':
    main()
