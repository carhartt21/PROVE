#!/usr/bin/env python3
"""
Batch Training Submission System

A robust, multi-user training submission system with:
- Pre-flight checks for existing results (won't overwrite completed training)
- Training lock mechanism to prevent duplicate work across multiple users/machines
- Automatic testing after training completes
- Configurable for different evaluation stages
- LSF job submission with proper parameter handling
- File permissions: 775 for directories, 664 for files

Strategies:
    - STD_STRATEGIES (7): baseline, std_minimal, std_photometric_distort,
                          std_autoaugment, std_cutmix, std_mixup, std_randaugment
    - GEN_STRATEGIES (21): gen_cycleGAN, gen_flux_kontext, gen_step1x_new, gen_TSIT, gen_augmenters, ...
    - ALL_STRATEGIES (28): STD_STRATEGIES + GEN_STRATEGIES

Usage:
    # Dry run to see what jobs would be submitted (ALWAYS do this first!)
    python scripts/batch_training_submission.py --stage 1 --dry-run
    
    # Submit Stage 1 jobs (all 336 jobs: 28 strategies × 4 datasets × 3 models)
    python scripts/batch_training_submission.py --stage 1
    
    # Submit Stage 1 jobs with limit
    python scripts/batch_training_submission.py --stage 1 --limit 50
    
    # Submit ONLY baseline + std_* strategies (no generative)
    python scripts/batch_training_submission.py --stage 1 --strategy-type std --dry-run
    
    # Submit ONLY generative strategies
    python scripts/batch_training_submission.py --stage 1 --strategy-type gen --dry-run
    
    # Submit specific strategies
    python scripts/batch_training_submission.py --stage 1 --strategies baseline std_minimal gen_cycleGAN
    
    # Submit jobs for specific dataset/model
    python scripts/batch_training_submission.py --stage 1 --datasets BDD10k --models deeplabv3plus_r50
    
    # Stage 2 (no domain filter, all weather conditions)
    python scripts/batch_training_submission.py --stage 2 --dry-run

Stages:
    Stage 1: Train on clear_day only (--domain-filter clear_day), test cross-domain robustness
    Stage 2: Train on all conditions, test domain-inclusive performance

Pre-flight Checks:
    - Skips if checkpoint already exists (iter_10000.pth)
    - Skips if training lock is held by another job
    - Skips gen_* strategies if generated images don't exist for dataset
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Set
from dataclasses import dataclass
import json

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training_lock import TrainingLock


# ============================================================================
# Configuration
# ============================================================================

# Base paths
WEIGHTS_ROOT_STAGE1 = Path('/scratch/aaa_exchange/AWARE/WEIGHTS')
WEIGHTS_ROOT_STAGE2 = Path('/scratch/aaa_exchange/AWARE/WEIGHTS_STAGE_2')
WEIGHTS_ROOT_RATIO_ABLATION = Path('/scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION')
GENERATED_IMAGES_ROOT = Path('/scratch/aaa_exchange/AWARE/GENERATED_IMAGES')

# All datasets
ALL_DATASETS = ['BDD10k', 'IDD-AW', 'MapillaryVistas', 'OUTSIDE15k']

# All models
ALL_MODELS = ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b3', 'segnext_mscan-b', 'hrnet_hr48']

# 21 gen_* strategies with full dataset coverage
# (excluding gen_EDICT, gen_StyleID, gen_flux2, gen_AOD-Net - no/insufficient coverage)
GEN_STRATEGIES = [
    'gen_cycleGAN',
    'gen_flux_kontext',
    'gen_step1x_new',
    'gen_LANIT',
    'gen_albumentations_weather',
    'gen_automold',
    'gen_step1x_v1p2',
    'gen_VisualCloze',
    'gen_SUSTechGAN',
    'gen_cyclediffusion',
    'gen_IP2P',
    'gen_Attribute_Hallucination',
    'gen_UniControl',
    'gen_CUT',
    'gen_Img2Img',
    'gen_Qwen_Image_Edit',
    'gen_CNetSeg',
    'gen_stargan_v2',
    'gen_Weather_Effect_Generator',
    'gen_TSIT',           # 191,400 images with full coverage
    'gen_augmenters',     # 159,500 images with full coverage
]

# Standard augmentation strategies
STD_STRATEGIES = [
    'baseline',           # No augmentation at all
    # 'std_minimal',        # RandomCrop + RandomFlip only
    # 'std_photometric_distort',  # PhotoMetricDistortion only
    'std_autoaugment',    # AutoAugment (batch-level)
    'std_cutmix',         # CutMix (batch-level)
    'std_mixup',          # MixUp (batch-level)
    'std_randaugment',    # RandAugment (batch-level)
]

# All strategies combined
ALL_STRATEGIES = STD_STRATEGIES + GEN_STRATEGIES

# LSF Job Configuration
@dataclass
class LSFConfig:
    """LSF job configuration"""
    queue: str = 'BatchGPU'
    time_limit: str = '24:00'
    memory: int = 48000  # Memory in MB
    cpu_count: int = 4


# ============================================================================
# Pre-flight Checks
# ============================================================================

def get_checkpoint_path(weights_dir: Path, max_iters: int = 10000) -> Optional[Path]:
    """Get the final checkpoint path if it exists."""
    checkpoint = weights_dir / f'iter_{max_iters}.pth'
    if checkpoint.exists():
        return checkpoint
    return None


def has_valid_results(weights_dir: Path, max_iters: int = 10000) -> bool:
    """Check if valid training results already exist."""
    checkpoint = get_checkpoint_path(weights_dir, max_iters)
    if checkpoint is None:
        return False
    
    # Check if checkpoint is valid (not empty)
    try:
        if checkpoint.stat().st_size < 1000:  # Less than 1KB is suspicious
            return False
    except OSError:
        return False
    
    return True


def has_test_results(weights_dir: Path) -> bool:
    """Check if test results exist."""
    test_results_dir = weights_dir / 'test_results_detailed'
    if not test_results_dir.exists():
        return False
    
    # Check for any results.json file
    for subdir in test_results_dir.iterdir():
        if subdir.is_dir():
            results_json = subdir / 'results.json'
            if results_json.exists():
                return True
    
    return False


def strategy_to_dir_name(strategy: str) -> str:
    """Convert strategy name to directory name (handle hyphens)."""
    # Map underscores back to hyphens for specific directories
    if strategy.startswith('gen_'):
        gen_model = strategy[4:]  # Remove 'gen_' prefix
        hyphen_dirs = {'Qwen_Image_Edit': 'Qwen-Image-Edit'}
        return hyphen_dirs.get(gen_model, gen_model)
    return strategy


def has_generated_images(strategy: str, dataset: str) -> bool:
    """Check if generated images exist for this strategy/dataset combination."""
    if not strategy.startswith('gen_'):
        return True  # Non-generative strategies don't need generated images
    
    gen_dir = strategy_to_dir_name(strategy)
    gen_path = GENERATED_IMAGES_ROOT / gen_dir
    
    if not gen_path.exists():
        return False
    
    # Check manifest for this dataset
    manifest = gen_path / 'manifest.csv'
    if manifest.exists():
        try:
            import csv
            with open(manifest, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('dataset', '') == dataset:
                        return True
            return False
        except Exception:
            pass
    
    # Fallback: check for dataset subdirectory
    dataset_path = gen_path / dataset
    if dataset_path.exists():
        # Check for any images
        for subdir in dataset_path.rglob('*'):
            if subdir.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                return True
    
    return False


# ============================================================================
# Job Generation
# ============================================================================

@dataclass
class TrainingJob:
    """Represents a training job to submit."""
    strategy: str
    dataset: str
    model: str
    stage: int  # Can be 1, 2, or 'ratio' for ratio ablation
    ratio: float = 0.5
    aux_loss: Optional[str] = None
    weights_dir: Optional[Path] = None
    skip_reason: Optional[str] = None
    
    @property
    def job_name(self) -> str:
        """Generate LSF job name with stage prefix."""
        # Stage prefix
        if isinstance(self.stage, int):
            stage_prefix = f's{self.stage}_'
        else:
            stage_prefix = f'{self.stage}_'  # e.g., 'ratio_'
        
        dataset_short = self.dataset.lower().replace('-', '')
        model_short = self.model.split('_')[0]
        aux_tag = f"_aux-{self.aux_loss}" if self.aux_loss else ''
        if self.strategy.startswith('gen_'):
            if self.ratio != 0.5:
                base = f'{self.strategy}_{dataset_short}_{model_short}_{self.ratio:.2f}'.replace('.', 'p')
            else:
                base = f'{self.strategy}_{dataset_short}_{model_short}'
        else:
            base = f'{self.strategy}_{dataset_short}_{model_short}'
        return f'{stage_prefix}{base}{aux_tag}'
    
    @property
    def is_skipped(self) -> bool:
        """Check if job should be skipped."""
        return self.skip_reason is not None


def get_weights_dir(
    strategy: str,
    dataset: str,
    model: str,
    stage: int,
    ratio: float = 0.5,
    aux_loss: Optional[str] = None,
) -> Path:
    """Get the weights directory for a training configuration."""
    # Determine base root based on stage
    if stage == 1:
        base_root = WEIGHTS_ROOT_STAGE1
    elif stage == 2:
        base_root = WEIGHTS_ROOT_STAGE2
    elif stage == 'ratio':
        base_root = WEIGHTS_ROOT_RATIO_ABLATION
    else:
        base_root = WEIGHTS_ROOT_STAGE1
    
    # Dataset directory name (lowercase, handle hyphen)
    dataset_dir = dataset.lower().replace('-', '')  # IDD-AW → iddaw for ratio ablation
    
    # Model directory name
    model_dir = model
    if strategy.startswith('gen_') and ratio != 1.0:
        model_dir = f'{model}_ratio{ratio:.2f}'.replace('.', 'p')
    if aux_loss:
        model_dir = f'{model_dir}_aux-{aux_loss}'
    
    return base_root / strategy / dataset_dir / model_dir


def generate_job_list(
    stage: int,
    strategies: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    ratios: Optional[List[float]] = None,
    aux_loss: Optional[str] = None,
    check_existing: bool = True,
    check_locks: bool = True,
) -> List[TrainingJob]:
    """
    Generate list of training jobs with pre-flight checks.
    
    Args:
        stage: Training stage (1, 2, or 'ratio' for ratio ablation)
        strategies: List of strategies (default: all strategies - baseline, std_*, gen_*)
        datasets: List of datasets (default: all)
        models: List of models (default: all)
        ratios: List of real/gen ratios for generative strategies (default: [0.5])
        check_existing: Skip jobs with existing results
        check_locks: Skip jobs that are currently locked
        
    Returns:
        List of TrainingJob objects
    """
    strategies = strategies or ALL_STRATEGIES  # Now includes baseline + std_* + gen_*
    datasets = datasets or ALL_DATASETS
    models = models or ALL_MODELS
    ratios = ratios or [0.5]  # Default to single ratio of 0.5
    
    jobs = []
    
    for strategy in strategies:
        for dataset in datasets:
            for model in models:
                # For non-generative strategies, only use ratio=1.0 (ignored anyway)
                strategy_ratios = ratios if strategy.startswith('gen_') else [1.0]
                
                for ratio in strategy_ratios:
                    job = TrainingJob(
                        strategy=strategy,
                        dataset=dataset,
                        model=model,
                        stage=stage,
                        ratio=ratio,
                        aux_loss=aux_loss,
                    )
                    
                    # Get weights directory
                    job.weights_dir = get_weights_dir(strategy, dataset, model, stage, ratio, aux_loss)
                    
                    # Pre-flight checks
                    if check_existing and has_valid_results(job.weights_dir):
                        job.skip_reason = 'Results exist'
                    elif strategy.startswith('gen_') and not has_generated_images(strategy, dataset):
                        job.skip_reason = 'No generated images'
                    elif check_locks:
                        lock = TrainingLock(
                            strategy,
                            dataset,
                            model,
                            ratio if strategy.startswith('gen_') else None,
                            aux_loss=aux_loss,
                            stage=stage if isinstance(stage, int) else None,
                        )
                        if lock.is_locked():
                            holder = lock.get_lock_holder()
                            if holder:
                                job.skip_reason = f"Locked by {holder.get('user', 'unknown')}@{holder.get('hostname', 'unknown')}"
                            else:
                                job.skip_reason = 'Locked'
                    
                    jobs.append(job)
    
    return jobs


# ============================================================================
# Job Submission
# ============================================================================

def generate_job_script(
    job: TrainingJob,
    lsf_config: LSFConfig,
    max_iters: Optional[int] = None,
    aux_loss: Optional[str] = None,
) -> str:
    """Generate LSF job script for a training job."""
    work_dir = str(job.weights_dir)
    
    # Build training command
    cmd_parts = [
        'python', str(PROJECT_ROOT / 'unified_training.py'),
        '--dataset', job.dataset,
        '--model', job.model,
        '--strategy', job.strategy,
    ]
    
    # Add domain filter for Stage 1 and ratio ablation (not Stage 2)
    if job.stage in [1, 'ratio']:
        cmd_parts.extend(['--domain-filter', 'clear_day'])
    
    # Add ratio parameter for generative strategies
    if job.strategy.startswith('gen_'):
        cmd_parts.extend(['--real-gen-ratio', str(job.ratio)])
    
    # Add max iterations if specified
    if max_iters is not None:
        cmd_parts.extend(['--max-iters', str(max_iters)])

    # Add auxiliary loss if specified
    if aux_loss:
        cmd_parts.extend(['--aux-loss', aux_loss])
    
    # Add work-dir to ensure proper output location
    cmd_parts.extend(['--work-dir', work_dir])
    
    training_cmd = ' '.join(cmd_parts)
    
    aux_suffix = f"_aux-{aux_loss}" if aux_loss else ''
    # Stage prefix for lock file (same as job name)
    stage_prefix = f's{job.stage}_' if isinstance(job.stage, int) else f'{job.stage}_'
    
    script = f'''#!/bin/bash
#BSUB -J {job.job_name}
#BSUB -q {lsf_config.queue}
#BSUB -o {work_dir}/train_%J.out
#BSUB -e {work_dir}/train_%J.err
#BSUB -n 2,{lsf_config.cpu_count}
#BSUB -gpu "num=1"

# ============================================================================
# Environment setup
# ============================================================================

# Set permissions: 775 for directories, 664 for files
umask 002

# Force Python to not use cached bytecode (always reimport fresh code)
export PYTHONDONTWRITEBYTECODE=1

# ============================================================================
# Pre-flight checks inside the job
# ============================================================================

echo "=========================================="
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "User: $USER"
echo "Started: $(date)"
echo "Strategy: {job.strategy}"
echo "Dataset: {job.dataset}"
echo "Model: {job.model}"
echo "Stage: {job.stage}"
echo "Aux Loss: {aux_loss or 'none'}"
echo "Work Dir: {work_dir}"
echo "=========================================="

# Create work directory with proper permissions
mkdir -p {work_dir}
chmod 775 {work_dir}
cd {work_dir}

# Activate conda environment
source ~/.bashrc
mamba activate prove

# Pre-flight check: verify results don't already exist
CHECKPOINT="{work_dir}/iter_10000.pth"
if [ -f "$CHECKPOINT" ]; then
    SIZE=$(stat -c%s "$CHECKPOINT" 2>/dev/null || echo 0)
    if [ "$SIZE" -gt 1000 ]; then
        echo "WARNING: Checkpoint already exists at $CHECKPOINT (size: $SIZE bytes)"
        echo "Skipping training to avoid overwriting."
        exit 0
    fi
fi

# Acquire training lock
LOCK_DIR="/scratch/aaa_exchange/AWARE/training_locks"
mkdir -p $LOCK_DIR
LOCK_FILE="$LOCK_DIR/{stage_prefix}{job.strategy}_{job.dataset.lower().replace('-', '_')}_{job.model}{aux_suffix}.lock"

# Try to acquire lock (non-blocking)
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    echo "ERROR: Another job is already training this configuration"
    echo "Lock file: $LOCK_FILE"
    cat "$LOCK_FILE" 2>/dev/null || true
    exit 1
fi

# Write lock info
cat > "$LOCK_FILE" << EOF
{{
  "job_id": "$LSB_JOBID",
  "hostname": "$(hostname)",
  "user": "$USER",
  "strategy": "{job.strategy}",
  "dataset": "{job.dataset}",
  "model": "{job.model}",
  "aux_loss": "{aux_loss or ''}",
  "started": "$(date -Iseconds)"
}}
EOF

echo "Lock acquired: $LOCK_FILE"

# ============================================================================
# Training
# ============================================================================

echo ""
echo "Starting training..."
echo "Command: {training_cmd}"
echo ""

{training_cmd}

TRAIN_EXIT_CODE=$?

# ============================================================================
# Testing (if training succeeded)
# ============================================================================

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    CHECKPOINT="{work_dir}/iter_10000.pth"
    CONFIG="{work_dir}/training_config.py"
    TEST_OUTPUT="{work_dir}/test_results_detailed"
    
    if [ -f "$CHECKPOINT" ] && [ -f "$CONFIG" ]; then
        echo ""
        echo "=========================================="
        echo "Starting fine-grained testing..."
        echo "=========================================="
        
        python {PROJECT_ROOT}/fine_grained_test.py \\
            --config "$CONFIG" \\
            --checkpoint "$CHECKPOINT" \\
            --output-dir "$TEST_OUTPUT" \\
            --dataset {job.dataset} \\
            --data-root /scratch/aaa_exchange/AWARE/FINAL_SPLITS \\
            --test-split test \\
            --batch-size 10
        
        TEST_EXIT_CODE=$?
        
        if [ $TEST_EXIT_CODE -eq 0 ]; then
            echo "Testing completed successfully"
        else
            echo "WARNING: Testing failed with exit code: $TEST_EXIT_CODE"
        fi
    else
        echo "WARNING: Checkpoint or config not found, skipping testing"
        echo "  Checkpoint: $CHECKPOINT"
        echo "  Config: $CONFIG"
    fi
else
    echo "Training failed, skipping testing"
fi

# ============================================================================
# Cleanup and permissions
# ============================================================================

# Ensure all created files/directories have proper permissions (775/664)
echo "Setting permissions on output files..."
find {work_dir} -type d -exec chmod 775 {{}} \; 2>/dev/null || true
find {work_dir} -type f -exec chmod 664 {{}} \; 2>/dev/null || true

# Release lock
flock -u 200
rm -f "$LOCK_FILE" 2>/dev/null

echo ""
echo "=========================================="
echo "Training completed with exit code: $TRAIN_EXIT_CODE"
echo "Finished: $(date)"
echo "=========================================="

exit $TRAIN_EXIT_CODE
'''
    return script


def submit_job(
    job: TrainingJob,
    lsf_config: LSFConfig,
    dry_run: bool = False,
    max_iters: Optional[int] = None,
    aux_loss: Optional[str] = None,
) -> bool:
    """
    Submit a training job to LSF.
    
    Args:
        job: TrainingJob to submit
        lsf_config: LSF configuration
        dry_run: If True, just print what would be done
        max_iters: Optional maximum training iterations
        
    Returns:
        True if job was submitted successfully
    """
    if job.is_skipped:
        print(f"  SKIP: {job.job_name} - {job.skip_reason}")
        return False
    
    # Ensure weights directory exists
    if job.weights_dir:
        job.weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate job script
    script = generate_job_script(job, lsf_config, max_iters=max_iters, aux_loss=aux_loss)
    script_path = job.weights_dir / 'submit_job.sh'
    
    if dry_run:
        print(f"  SUBMIT: {job.job_name}")
        print(f"    Dir: {job.weights_dir}")
        return True
    
    # Write script - try weights dir first, fall back to temp file
    import tempfile
    use_temp = False
    try:
        with open(script_path, 'w') as f:
            f.write(script)
        # Try to set permissions - ignore if not owner
        try:
            os.chmod(script_path, 0o755)
        except PermissionError:
            pass  # File might be owned by another user, that's OK
    except PermissionError:
        # Can't write to weights dir (owned by another user), use temp file
        use_temp = True
        fd, temp_path = tempfile.mkstemp(suffix='.sh', prefix=f'{job.job_name}_')
        script_path = Path(temp_path)
        with os.fdopen(fd, 'w') as f:
            f.write(script)
        os.chmod(script_path, 0o755)
    
    # Submit job
    result = subprocess.run(
        f'bsub < {script_path}',
        shell=True,
        capture_output=True,
        text=True
    )
    
    # Clean up temp file if used
    if use_temp:
        try:
            script_path.unlink()
        except:
            pass
    
    if result.returncode == 0:
        print(f"  SUBMIT: {job.job_name} - {result.stdout.strip()}")
        return True
    else:
        print(f"  FAILED: {job.job_name} - {result.stderr.strip()}")
        return False


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Batch Training Submission System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    # Dry run to see what jobs would be submitted
    python batch_training_submission.py --stage 1 --dry-run
    
    # Submit Stage 1 jobs (limit to 50)
    python batch_training_submission.py --stage 1 --limit 50
    
    # Submit specific strategies
    python batch_training_submission.py --stage 1 --strategies gen_cycleGAN gen_flux_kontext
    
    # Submit for specific dataset/model
    python batch_training_submission.py --stage 1 --datasets BDD10k --models deeplabv3plus_r50
'''
    )
    
    # Required arguments
    parser.add_argument('--stage', required=True,
                       help='Training stage: 1 (clear_day only), 2 (all domains), or "ratio" for ratio ablation study')
    
    # Filtering options
    parser.add_argument('--strategy-type', choices=['all', 'std', 'gen'],
                       default='all',
                       help='Strategy type: all (28), std (7 baseline+std_*), gen (21 gen_*)')
    parser.add_argument('--strategies', nargs='+', 
                       help='Specific strategies to train (overrides --strategy-type)')
    parser.add_argument('--datasets', nargs='+', choices=ALL_DATASETS,
                       help='Specific datasets (default: all)')
    parser.add_argument('--models', nargs='+', choices=ALL_MODELS,
                       help='Specific models (default: all)')
    
    # Job control
    parser.add_argument('--limit', type=int, default=None,
                       help='Maximum number of jobs to submit')
    parser.add_argument('--ratios', type=float, nargs='+', default=[0.5],
                       help='Real/gen ratios for generative strategies (default: 0.5). Example: --ratios 0.0 0.25 0.5')
    parser.add_argument('--max-iters', type=int, default=None,
                       help='Maximum training iterations (default: use config default, usually 10000)')
    parser.add_argument('--aux-loss', type=str, default=None,
                       choices=['focal', 'lovasz', 'boundary'],
                       help='Auxiliary loss to add alongside CrossEntropyLoss')
    
    # Pre-flight options
    parser.add_argument('--no-check-existing', action='store_true',
                       help='Don\'t skip jobs with existing results')
    parser.add_argument('--no-check-locks', action='store_true',
                       help='Don\'t check for training locks')
    
    # LSF options
    parser.add_argument('--queue', default='BatchGPU',
                       help='LSF queue (default: BatchGPU)')
    parser.add_argument('--time-limit', default='24:00',
                       help='Job time limit (default: 24:00)')
    parser.add_argument('--memory', type=int, default=48000,
                       help='Memory per process in MB (default: 48000)')
    
    # Execution options
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without submitting')
    parser.add_argument('-y', '--yes', action='store_true',
                       help='Skip confirmation prompt and submit immediately')
    parser.add_argument('--delay', type=float, default=0.5,
                       help='Delay between job submissions in seconds (default: 0.5)')
    
    args = parser.parse_args()
    
    # Create LSF config
    lsf_config = LSFConfig(
        queue=args.queue,
        time_limit=args.time_limit,
        memory=args.memory,
    )
    
    # Validate and parse stage
    stage_map = {'1': 1, '2': 2, 'ratio': 'ratio'}
    stage_input = str(args.stage).lower()
    if stage_input not in stage_map and stage_input not in ['1', '2', 'ratio']:
        print(f"Error: Invalid stage '{args.stage}'. Must be 1, 2, or 'ratio'")
        return
    stage = stage_map.get(stage_input, args.stage if isinstance(args.stage, int) else stage_input)
    
    # Determine strategies based on --strategy-type or --strategies
    strategies = args.strategies
    if strategies is None:
        if args.strategy_type == 'std':
            strategies = STD_STRATEGIES
        elif args.strategy_type == 'gen':
            strategies = GEN_STRATEGIES
        else:  # 'all'
            strategies = ALL_STRATEGIES
    
    # Generate job list
    print(f"\n{'='*60}")
    print(f"Batch Training Submission - Stage {stage}")
    if stage == 'ratio':
        print(f"  Ratios: {args.ratios}")
    print(f"{'='*60}")
    print(f"\nStrategy type: {args.strategy_type}")
    print(f"Strategies: {len(strategies)}")
    print(f"\nGenerating job list...")
    
    jobs = generate_job_list(
        stage=stage,
        strategies=strategies,
        datasets=args.datasets,
        models=args.models,
        ratios=args.ratios,
        aux_loss=args.aux_loss,
        check_existing=not args.no_check_existing,
        check_locks=not args.no_check_locks,
    )
    
    # Summary
    total = len(jobs)
    skipped = sum(1 for j in jobs if j.is_skipped)
    to_submit = total - skipped
    
    print(f"\nJob Summary:")
    print(f"  Total configurations: {total}")
    print(f"  Skipped: {skipped}")
    print(f"  To submit: {to_submit}")
    
    if args.limit and to_submit > args.limit:
        print(f"  Limited to: {args.limit}")
    
    # Skip reason breakdown
    skip_reasons: Dict[str, int] = {}
    for job in jobs:
        if job.skip_reason:
            skip_reasons[job.skip_reason] = skip_reasons.get(job.skip_reason, 0) + 1
    
    if skip_reasons:
        print(f"\nSkip reasons:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            print(f"  - {reason}: {count}")
    
    # Confirm submission
    if not args.dry_run and to_submit > 0 and not args.yes:
        print(f"\n{'='*60}")
        response = input(f"Submit {min(to_submit, args.limit or to_submit)} jobs? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Submit jobs
    print(f"\n{'='*60}")
    print("Submitting jobs..." if not args.dry_run else "Dry run - showing what would be submitted:")
    print(f"{'='*60}\n")
    
    submitted = 0
    for job in jobs:
        if job.is_skipped:
            continue
        
        if args.limit and submitted >= args.limit:
            print(f"\nReached limit of {args.limit} jobs")
            break
        
        if submit_job(
            job,
            lsf_config,
            dry_run=args.dry_run,
            max_iters=args.max_iters,
            aux_loss=args.aux_loss,
        ):
            submitted += 1
            if not args.dry_run and args.delay > 0:
                time.sleep(args.delay)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  {'Would submit' if args.dry_run else 'Submitted'}: {submitted} jobs")
    print(f"  Skipped: {skipped}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
