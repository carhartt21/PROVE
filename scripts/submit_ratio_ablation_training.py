#!/usr/bin/env python3
"""
Submit ratio ablation training jobs for top strategies.

This script trains models with different real/generated ratios for the
best-performing strategies from Stage 1 and Stage 2 evaluations.

Top Strategies by mIoU:
Stage 1 (domain_filter=clear_day):
  1. gen_Attribute_Hallucination (39.83%)
  2. gen_cycleGAN (39.60%)
  3. gen_Img2Img (39.58%)
  4. gen_stargan_v2 (39.55%)
  5. gen_flux_kontext (39.54%)

Stage 2 (no domain_filter):
  1. gen_stargan_v2 (41.73%)
  2. gen_UniControl (41.70%)
  3. gen_CNetSeg (41.69%)
  4. gen_VisualCloze (41.67%)
  5. gen_cycleGAN (41.64%)

Models: pspnet_r50, segformer_mit-b5 (matching existing ablation pattern)
Ratios: 0.00, 0.12, 0.25, 0.38, 0.62, 0.75, 0.88 (0.50 already in main training)
Datasets: BDD10k, IDD-AW (start with 2 for efficiency)
"""

import argparse
import subprocess
import os
from pathlib import Path
from typing import List, Dict, Tuple, Set

# Configuration
RATIOS = [0.00, 0.12, 0.25, 0.38, 0.62, 0.75, 0.88]  # Exclude 0.50 (already in Stage 1/2)
MODELS = ['pspnet_r50', 'segformer_mit-b5']  # Match existing ratio ablation pattern
DATASETS = ['BDD10k', 'IDD-AW']  # Start with 2 datasets

# Stage 1 strategies - Top 5 by mIoU
STAGE1_STRATEGIES = [
    'gen_Attribute_Hallucination',  # #1: 39.83%
    'gen_cycleGAN',                 # #2: 39.60%
    'gen_Img2Img',                  # #3: 39.58%
    'gen_stargan_v2',               # #4: 39.55%
    'gen_flux_kontext',             # #5: 39.54%
]

# Stage 2 strategies - Top 5 by mIoU
STAGE2_STRATEGIES = [
    'gen_stargan_v2',               # #1: 41.73%
    'gen_UniControl',               # #2: 41.70%
    'gen_CNetSeg',                  # #3: 41.69%
    'gen_VisualCloze',              # #4: 41.67%
    'gen_cycleGAN',                 # #5: 41.64%
]

# Output directories
WEIGHTS_DIR = Path('/scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION')


def get_existing_models(strategy: str) -> Set[Tuple[str, str, float]]:
    """Get set of (dataset, model, ratio) tuples that already have trained models."""
    existing = set()
    strategy_dir = WEIGHTS_DIR / strategy
    
    if not strategy_dir.exists():
        return existing
    
    for dataset_dir in strategy_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        
        for model_dir in dataset_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            # Parse model and ratio from directory name
            model_name = model_dir.name
            
            # Check for checkpoint
            checkpoint = model_dir / 'iter_80000.pth'
            if checkpoint.exists():
                # Extract ratio from name like "pspnet_r50_ratio0p50"
                if '_ratio' in model_name:
                    base_model = '_'.join(model_name.split('_ratio')[0].split('_'))
                    ratio_str = model_name.split('_ratio')[1]
                    try:
                        ratio = float(ratio_str.replace('p', '.'))
                        existing.add((dataset, base_model, ratio))
                    except:
                        pass
    
    return existing


def generate_job_script(
    strategy: str,
    dataset: str,
    model: str,
    ratio: float,
    stage: int
) -> str:
    """Generate LSF job submission script."""
    
    ratio_str = f"{ratio:.2f}".replace('.', 'p')
    job_name = f"ratio_{strategy}_{dataset}_{model}_{ratio_str}"
    
    # Use native classes for MapillaryVistas and OUTSIDE15k
    native_flag = ""
    if dataset in ['MapillaryVistas', 'OUTSIDE15k']:
        native_flag = "--use-native-classes"
    
    # Stage 1 uses domain_filter=clear_day, Stage 2 does not
    domain_filter = "--domain-filter clear_day" if stage == 1 else ""
    
    script = f'''#!/bin/bash
#BSUB -J {job_name}
#BSUB -q BatchGPU
#BSUB -n 10
#BSUB -gpu "num=1:gmem=24G"
#BSUB -W 24:00
#BSUB -o logs/ratio_ablation/{job_name}_%J.out
#BSUB -e logs/ratio_ablation/{job_name}_%J.err

mkdir -p logs/ratio_ablation
cd /home/$USER/repositories/PROVE
source ~/.bashrc
mamba activate prove

python unified_training.py \\
    --dataset {dataset} \\
    --model {model} \\
    --strategy {strategy} \\
    --real-gen-ratio {ratio} \\
    {native_flag} \\
    {domain_filter} \\
    --work-dir {WEIGHTS_DIR}

echo "Training completed for {strategy} {dataset} {model} ratio={ratio}"
'''
    return script


def submit_job(script: str, dry_run: bool = True) -> bool:
    """Submit job to LSF."""
    if dry_run:
        print(f"[DRY-RUN] Would submit job")
        return True
    
    try:
        result = subprocess.run(
            ['bsub'],
            input=script,
            text=True,
            capture_output=True
        )
        if result.returncode == 0:
            print(f"  Submitted: {result.stdout.strip()}")
            return True
        else:
            print(f"  Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"  Exception: {e}")
        return False


def preflight_check(stage: int):
    """Comprehensive check for existing weights before starting training."""
    print("=" * 70)
    print(f"PREFLIGHT CHECK - Stage {stage} Ratio Ablation")
    print("=" * 70)
    
    strategies = STAGE1_STRATEGIES if stage == 1 else STAGE2_STRATEGIES
    
    # Check existing weights
    print(f"\n{'='*70}")
    print("EXISTING WEIGHTS IN WEIGHTS_RATIO_ABLATION/")
    print("=" * 70)
    
    total_existing = 0
    total_needed = len(strategies) * len(DATASETS) * len(MODELS) * len(RATIOS)
    
    for strategy in strategies:
        existing = get_existing_models(strategy)
        print(f"\n{strategy}:")
        
        # Check each configuration
        for dataset in DATASETS:
            for model in MODELS:
                found = []
                missing = []
                for ratio in RATIOS:
                    ds_lower = dataset.lower().replace('-', '')
                    # Check various naming patterns
                    matches = [
                        (dataset, model, ratio),
                        (dataset.lower(), model, ratio),
                        (ds_lower, model, ratio),
                        (f"{ds_lower}_ad", model, ratio),
                        (f"{dataset.lower()}_ad", model, ratio),
                        (dataset.lower().replace('-', '-'), model, ratio),  # idd-aw
                    ]
                    found_ratio = any(m in existing for m in matches)
                    if found_ratio:
                        found.append(f"{ratio:.2f}")
                        total_existing += 1
                    else:
                        missing.append(f"{ratio:.2f}")
                
                if found:
                    print(f"  {dataset}/{model}: ✅ {len(found)} found ({', '.join(found[:3])}{'...' if len(found) > 3 else ''})")
                if missing:
                    print(f"  {dataset}/{model}: ❌ {len(missing)} missing ({', '.join(missing[:3])}{'...' if len(missing) > 3 else ''})")
    
    # Check for existing jobs in queue
    print(f"\n{'='*70}")
    print("CURRENT LSF QUEUE STATUS")
    print("=" * 70)
    try:
        user = os.environ.get('USER', 'mima2416')
        result = subprocess.run(
            ['bjobs', '-u', user, '-noheader'],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            print(f"  Total jobs in queue: {len(lines)}")
            
            # Count ratio ablation jobs
            ratio_jobs = [l for l in lines if 'ratio_' in l.lower()]
            print(f"  Ratio ablation jobs: {len(ratio_jobs)}")
        else:
            print("  No jobs currently in queue")
    except Exception as e:
        print(f"  Unable to check queue: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total configurations needed: {total_needed}")
    print(f"  Already trained: {total_existing}")
    print(f"  Need to train: {total_needed - total_existing}")
    print(f"\n  Strategies: {', '.join(strategies)}")
    print(f"  Datasets: {', '.join(DATASETS)}")
    print(f"  Models: {', '.join(MODELS)}")
    print(f"  Ratios: {', '.join([f'{r:.2f}' for r in RATIOS])}")
    
    # Recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print("=" * 70)
    if total_existing == total_needed:
        print("  ✅ All weights already exist! No submission needed.")
    elif total_existing > 0:
        print(f"  ⚠️ Partial weights exist. {total_needed - total_existing} jobs will be submitted.")
        print(f"     Use --dry-run to see exactly what would be submitted.")
    else:
        print(f"  📝 No existing weights. {total_needed} jobs will be submitted.")
    
    print(f"\n  To submit: python {Path(__file__).name} --stage {stage}")
    print(f"  To dry-run: python {Path(__file__).name} --stage {stage} --dry-run")


def main():
    parser = argparse.ArgumentParser(description='Submit ratio ablation training jobs')
    parser.add_argument('--stage', type=int, choices=[1, 2], required=True,
                       help='Training stage (1=clear_day, 2=all domains)')
    parser.add_argument('--strategy', type=str, default=None,
                       help='Specific strategy to train (default: all for stage)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Specific dataset (default: all)')
    parser.add_argument('--model', type=str, default=None,
                       help='Specific model (default: all)')
    parser.add_argument('--ratio', type=float, default=None,
                       help='Specific ratio (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print what would be submitted without actually submitting')
    parser.add_argument('--preflight', action='store_true',
                       help='Check for existing weights and show summary without submitting')
    parser.add_argument('--limit', type=int, default=None,
                       help='Maximum number of jobs to submit')
    args = parser.parse_args()
    
    # Preflight check mode
    if args.preflight:
        preflight_check(args.stage)
        return
    
    # Select strategies based on stage
    if args.stage == 1:
        strategies = STAGE1_STRATEGIES
    else:
        strategies = STAGE2_STRATEGIES
    
    if args.strategy:
        if args.strategy not in strategies:
            print(f"Warning: {args.strategy} not in standard Stage {args.stage} strategies")
        strategies = [args.strategy]
    
    datasets = [args.dataset] if args.dataset else DATASETS
    models = [args.model] if args.model else MODELS
    ratios = [args.ratio] if args.ratio is not None else RATIOS
    
    # Create logs directory
    os.makedirs('logs/ratio_ablation', exist_ok=True)
    
    # Count jobs
    total_jobs = 0
    submitted = 0
    skipped = 0
    
    for strategy in strategies:
        print(f"\n=== Strategy: {strategy} (Stage {args.stage}) ===")
        
        existing = get_existing_models(strategy)
        print(f"  Existing trained models: {len(existing)}")
        
        for dataset in datasets:
            for model in models:
                for ratio in ratios:
                    total_jobs += 1
                    
                    # Check if already trained
                    # Normalize dataset name for comparison
                    ds_normalized = dataset.lower().replace('-', '')
                    model_normalized = model
                    
                    # Check both with and without _ad suffix
                    if (dataset, model, ratio) in existing or \
                       (ds_normalized, model_normalized, ratio) in existing:
                        skipped += 1
                        continue
                    
                    if args.limit and submitted >= args.limit:
                        print(f"  Reached limit of {args.limit} jobs")
                        break
                    
                    print(f"  {dataset}/{model}/ratio{ratio}")
                    script = generate_job_script(strategy, dataset, model, ratio, args.stage)
                    
                    if submit_job(script, dry_run=args.dry_run):
                        submitted += 1
                
                if args.limit and submitted >= args.limit:
                    break
            if args.limit and submitted >= args.limit:
                break
        if args.limit and submitted >= args.limit:
            break
    
    print(f"\n=== Summary ===")
    print(f"Total configurations: {total_jobs}")
    print(f"Skipped (already trained): {skipped}")
    print(f"{'Would submit' if args.dry_run else 'Submitted'}: {submitted}")


if __name__ == '__main__':
    main()
