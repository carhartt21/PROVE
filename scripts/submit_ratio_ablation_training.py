#!/usr/bin/env python3
"""
Submit ratio ablation training jobs for top strategies.

This script trains models with different real/generated ratios for the
best-performing strategies from Stage 1 and Stage 2 evaluations.

**Directory Structure (as of 2026-01-26):**
  WEIGHTS_RATIO_ABLATION/
  ├── stage1/{strategy}/{dataset}/{model}_ratio{ratio}/  # domain_filter=clear_day
  └── stage2/{strategy}/{dataset}/{model}_ratio{ratio}/  # no domain_filter

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
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional

# Add parent directory for training_lock import
sys.path.insert(0, str(Path(__file__).parent.parent))
from training_lock import TrainingLock

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

# Existing strategies (from earlier ratio ablation work)
EXISTING_STRATEGIES_STAGE1 = [
    'gen_cycleGAN',
    'gen_cyclediffusion',
    'gen_stargan_v2',
]

EXISTING_STRATEGIES_STAGE2 = [
    'gen_step1x_new',
    'gen_step1x_v1p2',
]

# Output directories - now with stage separation
WEIGHTS_ROOT = Path('/scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION')


def get_stage_dir(stage: int) -> Path:
    """Get the appropriate stage directory."""
    return WEIGHTS_ROOT / f'stage{stage}'

def normalize_dataset_name(name: str) -> str:
    """Normalize dataset name for consistent path matching."""
    name = name.lower()
    # Remove _ad suffix
    name = name.rstrip('_ad')
    # Standardize idd variations
    if name in ['iddaw', 'idd_aw']:
        name = 'idd-aw'
    return name


def get_existing_models(strategy: str, stage: int) -> Set[Tuple[str, str, float]]:
    """Get set of (dataset, model, ratio) tuples that already have trained models."""
    existing = set()
    stage_dir = get_stage_dir(stage)
    strategy_dir = stage_dir / strategy
    
    if not strategy_dir.exists():
        return existing
    
    for dataset_dir in strategy_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset = normalize_dataset_name(dataset_dir.name)
        
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


def is_config_locked(strategy: str, dataset: str, model: str, ratio: float) -> bool:
    """Check if a training configuration is locked."""
    lock = TrainingLock(strategy, dataset, model, ratio)
    return lock.is_locked()


def generate_job_script(
    strategy: str,
    dataset: str,
    model: str,
    ratio: float,
    stage: int
) -> str:
    """Generate LSF job submission script with training lock."""
    
    ratio_str = f"{ratio:.2f}".replace('.', 'p')
    job_name = f"ratio_{strategy}_{dataset}_{model}_{ratio_str}_s{stage}"
    
    # Use native classes for MapillaryVistas and OUTSIDE15k
    native_flag = ""
    if dataset in ['MapillaryVistas', 'OUTSIDE15k']:
        native_flag = "--use-native-classes"
    
    # Stage 1 uses domain_filter=clear_day, Stage 2 does not
    domain_filter = "--domain-filter clear_day" if stage == 1 else ""
    
    # Build the proper nested work_dir path with stage separation
    # Structure: WEIGHTS_RATIO_ABLATION/stage{N}/{strategy}/{dataset_lower}/{model}_ratio{ratio}
    dataset_lower = normalize_dataset_name(dataset)
    stage_dir = get_stage_dir(stage)
    work_dir = f"{stage_dir}/{strategy}/{dataset_lower}/{model}_ratio{ratio_str}"
    
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

# Training lock to prevent parallel training of same configuration
echo "Acquiring training lock for {strategy}/{dataset}/{model}/ratio={ratio}"
python -c "
import sys
sys.path.insert(0, '/home/$USER/repositories/PROVE')
from training_lock import TrainingLock

lock = TrainingLock('{strategy}', '{dataset}', '{model}', {ratio})
if not lock.acquire():
    holder = lock.get_lock_holder()
    print(f'Configuration locked by job {{holder.get(\"job_id\", \"unknown\")}} since {{holder.get(\"acquired_at\", \"unknown\")}}')
    sys.exit(1)
print('Lock acquired successfully')
" || {{ echo "Failed to acquire lock, exiting"; exit 1; }}

# Run training with explicit work-dir
python unified_training.py \\
    --dataset {dataset} \\
    --model {model} \\
    --strategy {strategy} \\
    --real-gen-ratio {ratio} \\
    {native_flag} \\
    {domain_filter} \\
    --work-dir {work_dir}

# Release training lock
python -c "
import sys
sys.path.insert(0, '/home/$USER/repositories/PROVE')
from training_lock import TrainingLock

lock = TrainingLock('{strategy}', '{dataset}', '{model}', {ratio})
# The lock file should already exist from the first acquire
lock.lock_file.unlink(missing_ok=True) if hasattr(lock.lock_file, 'unlink') else None
print('Lock released')
"

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


def preflight_check(stage: int, use_existing: bool = False):
    """Comprehensive check for existing weights before starting training."""
    print("=" * 70)
    print(f"PREFLIGHT CHECK - Stage {stage} Ratio Ablation")
    print("=" * 70)
    
    if use_existing:
        strategies = EXISTING_STRATEGIES_STAGE1 if stage == 1 else EXISTING_STRATEGIES_STAGE2
        print(f"Using existing strategies: {strategies}")
    else:
        strategies = STAGE1_STRATEGIES if stage == 1 else STAGE2_STRATEGIES
    stage_dir = get_stage_dir(stage)
    
    # Check existing weights
    print(f"\n{'='*70}")
    print(f"EXISTING WEIGHTS IN {stage_dir}/")
    print("=" * 70)
    
    total_existing = 0
    total_locked = 0
    total_needed = len(strategies) * len(DATASETS) * len(MODELS) * len(RATIOS)
    
    for strategy in strategies:
        existing = get_existing_models(strategy, stage)
        print(f"\n{strategy}:")
        
        # Check each configuration
        for dataset in DATASETS:
            ds_normalized = normalize_dataset_name(dataset)
            for model in MODELS:
                found = []
                missing = []
                locked = []
                for ratio in RATIOS:
                    # Check if already trained
                    if (ds_normalized, model, ratio) in existing:
                        found.append(f"{ratio:.2f}")
                        total_existing += 1
                    # Check if locked
                    elif is_config_locked(strategy, dataset, model, ratio):
                        locked.append(f"{ratio:.2f}")
                        total_locked += 1
                    else:
                        missing.append(f"{ratio:.2f}")
                
                if found:
                    print(f"  {dataset}/{model}: ✅ {len(found)} trained ({', '.join(found[:3])}{'...' if len(found) > 3 else ''})")
                if locked:
                    print(f"  {dataset}/{model}: 🔒 {len(locked)} locked ({', '.join(locked[:3])}{'...' if len(locked) > 3 else ''})")
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
    print(f"  Currently locked (in-progress): {total_locked}")
    print(f"  Need to train: {total_needed - total_existing - total_locked}")
    print(f"\n  Strategies: {', '.join(strategies)}")
    print(f"  Datasets: {', '.join(DATASETS)}")
    print(f"  Models: {', '.join(MODELS)}")
    print(f"  Ratios: {', '.join([f'{r:.2f}' for r in RATIOS])}")
    print(f"\n  Stage {stage} directory: {stage_dir}")
    
    # Recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print("=" * 70)
    remaining = total_needed - total_existing - total_locked
    if total_existing == total_needed:
        print("  ✅ All weights already exist! No submission needed.")
    elif remaining == 0:
        print(f"  🔒 All remaining jobs are locked (in-progress). Wait for completion.")
    elif total_existing > 0 or total_locked > 0:
        print(f"  ⚠️ Partial progress. {remaining} jobs will be submitted.")
        print(f"     Use --dry-run to see exactly what would be submitted.")
    else:
        print(f"  📝 No existing weights. {total_needed} jobs will be submitted.")
    
    print(f"\n  To submit: python {Path(__file__).name} --stage {stage}")
    print(f"  To dry-run: python {Path(__file__).name} --stage {stage} --dry-run")


def main():
    parser = argparse.ArgumentParser(
        description='Submit ratio ablation training jobs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check status for Stage 1 (top-5 strategies)
  python submit_ratio_ablation_training.py --stage 1 --preflight

  # Check status for existing Stage 2 strategies (gen_step1x_new, gen_step1x_v1p2)
  python submit_ratio_ablation_training.py --stage 2 --existing-strategies --preflight

  # Submit Stage 1 jobs with limit
  python submit_ratio_ablation_training.py --stage 1 --dry-run --limit 10

  # Submit for a specific strategy
  python submit_ratio_ablation_training.py --stage 1 --strategy gen_cycleGAN
""")
    parser.add_argument('--stage', type=int, choices=[1, 2], required=True,
                       help='Training stage (1=clear_day, 2=all domains)')
    parser.add_argument('--strategy', type=str, default=None,
                       help='Specific strategy to train (default: top-5 for stage)')
    parser.add_argument('--existing-strategies', action='store_true',
                       help='Use existing strategies (gen_step1x_*, gen_cyclediffusion) instead of top-5')
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
        preflight_check(args.stage, use_existing=args.existing_strategies)
        return
    
    # Select strategies based on stage and --existing-strategies flag
    if args.existing_strategies:
        strategies = EXISTING_STRATEGIES_STAGE1 if args.stage == 1 else EXISTING_STRATEGIES_STAGE2
        print(f"Using existing strategies: {strategies}")
    else:
        strategies = STAGE1_STRATEGIES if args.stage == 1 else STAGE2_STRATEGIES
    
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
    skipped_trained = 0
    skipped_locked = 0
    
    stage_dir = get_stage_dir(args.stage)
    print(f"\nStage {args.stage} weights directory: {stage_dir}")
    
    for strategy in strategies:
        print(f"\n=== Strategy: {strategy} (Stage {args.stage}) ===")
        
        existing = get_existing_models(strategy, args.stage)
        print(f"  Existing trained models: {len(existing)}")
        
        for dataset in datasets:
            ds_normalized = normalize_dataset_name(dataset)
            for model in models:
                for ratio in ratios:
                    total_jobs += 1
                    
                    # Check if already trained
                    if (ds_normalized, model, ratio) in existing:
                        skipped_trained += 1
                        continue
                    
                    # Check if locked (another job is training)
                    if is_config_locked(strategy, dataset, model, ratio):
                        print(f"  🔒 {dataset}/{model}/ratio{ratio} (locked - skipping)")
                        skipped_locked += 1
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
    print(f"Skipped (already trained): {skipped_trained}")
    print(f"Skipped (locked/in-progress): {skipped_locked}")
    print(f"{'Would submit' if args.dry_run else 'Submitted'}: {submitted}")


if __name__ == '__main__':
    main()
