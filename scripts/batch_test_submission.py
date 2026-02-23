#!/usr/bin/env python3
"""
Batch Test Submission System

A robust test job submission system analogous to batch_training_submission.py:
- Pre-flight checks for existing valid test results
- Duplicate job detection (checks running/pending LSF jobs)
- Supports all stages: 1, 2, cityscapes, cityscapes-gen
- Supports Cityscapes and ACDC cross-domain testing
- LSF job submission with proper GPU/memory configuration
- File permissions: 775 for directories, 664 for files

Usage:
    # Dry run (ALWAYS do this first!)
    python scripts/batch_test_submission.py --stage cityscapes-gen --dry-run

    # Submit all missing cityscapes-gen test jobs
    python scripts/batch_test_submission.py --stage cityscapes-gen -y

    # Submit only Cityscapes tests (no ACDC cross-domain)
    python scripts/batch_test_submission.py --stage cityscapes-gen --test-type cityscapes -y

    # Submit only ACDC cross-domain tests
    python scripts/batch_test_submission.py --stage cityscapes-gen --test-type acdc -y

    # Filter by strategy type
    python scripts/batch_test_submission.py --stage cityscapes-gen --strategy-type gen --dry-run

    # Filter by specific strategies/models
    python scripts/batch_test_submission.py --stage cityscapes-gen --strategies baseline std_cutmix --dry-run
    python scripts/batch_test_submission.py --stage cityscapes-gen --models pspnet_r50 segformer_mit-b3 --dry-run

    # Limit submissions
    python scripts/batch_test_submission.py --stage cityscapes-gen --limit 10 -y

    # Force retest (ignore existing results)
    python scripts/batch_test_submission.py --stage cityscapes-gen --force --dry-run

    # Stage 1 testing
    python scripts/batch_test_submission.py --stage 1 --dry-run

    # Stage 2 testing
    python scripts/batch_test_submission.py --stage 2 --dry-run
"""

import os
import sys
import re
import json
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Set
from dataclasses import dataclass, field

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# Configuration
# ============================================================================

# Base paths
WEIGHTS_ROOTS = {
    1: Path('${AWARE_DATA_ROOT}/WEIGHTS'),
    2: Path('${AWARE_DATA_ROOT}/WEIGHTS_STAGE_2'),
    'cityscapes': Path('${AWARE_DATA_ROOT}/WEIGHTS_CITYSCAPES'),
    'cityscapes-gen': Path('${AWARE_DATA_ROOT}/WEIGHTS_CITYSCAPES_GEN'),
}

DATA_ROOT = '${AWARE_DATA_ROOT}/FINAL_SPLITS'
LOG_DIR = PROJECT_ROOT / 'logs'

# All models (excluding deeplabv3plus_r50 and hrnet_hr48 from active training)
ALL_MODELS = [
    'pspnet_r50',
    'segformer_mit-b3',
    'segnext_mscan-b',
    'mask2former_swin-b',
]

# Datasets per stage
STAGE_DATASETS = {
    1: ['bdd10k', 'idd-aw', 'mapillaryvistas', 'outside15k'],
    2: ['bdd10k', 'idd-aw', 'mapillaryvistas', 'outside15k'],
    'cityscapes': ['cityscapes'],
    'cityscapes-gen': ['cityscapes'],
}

# Dataset display names for fine_grained_test.py --dataset argument
DATASET_DISPLAY = {
    'bdd10k': 'BDD10k',
    'idd-aw': 'IDD-AW',
    'iddaw': 'IDD-AW',
    'mapillaryvistas': 'MapillaryVistas',
    'outside15k': 'OUTSIDE15k',
    'cityscapes': 'Cityscapes',
}

# Max iterations per stage
STAGE_MAX_ITERS = {
    1: 15000,
    2: 15000,
    'cityscapes': 20000,
    'cityscapes-gen': 20000,
}

# Model-specific overrides for max iterations
MODEL_MAX_ITERS = {
    'mask2former_swin-b': 20000,  # Always 20k for mask2former
}

# Strategies
GEN_STRATEGIES = [
    'gen_cycleGAN', 'gen_flux_kontext', 'gen_step1x_new', 'gen_LANIT',
    'gen_albumentations_weather', 'gen_automold', 'gen_step1x_v1p2',
    'gen_VisualCloze', 'gen_SUSTechGAN', 'gen_cyclediffusion', 'gen_IP2P',
    'gen_Attribute_Hallucination', 'gen_UniControl', 'gen_CUT', 'gen_Img2Img',
    'gen_Qwen_Image_Edit', 'gen_CNetSeg', 'gen_stargan_v2',
    'gen_Weather_Effect_Generator', 'gen_TSIT', 'gen_augmenters',
]

STD_STRATEGIES = [
    'baseline', 'std_autoaugment', 'std_cutmix', 'std_mixup', 'std_randaugment',
]

ALL_STRATEGIES = STD_STRATEGIES + GEN_STRATEGIES

# GPU memory per model (for LSF resource requests)
MODEL_GMEM = {
    'pspnet_r50': '16G',
    'segformer_mit-b3': '16G',
    'segnext_mscan-b': '16G',
    'mask2former_swin-b': '32G',
}
DEFAULT_GMEM = '16G'


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class TestJob:
    """Represents a test job to submit."""
    strategy: str
    dataset: str
    model: str
    stage: object  # int or str
    test_type: str  # 'cityscapes', 'acdc', or dataset name
    config_path: Path
    checkpoint_path: Path
    output_dir: Path
    weights_dir: Path
    skip_reason: Optional[str] = None

    @property
    def job_name(self) -> str:
        """Generate LSF job name."""
        stage_str = str(self.stage).replace('-', '')
        model_short = self.model.split('_')[0]
        strat_short = self.strategy[:20]
        test_prefix = 'acdc' if self.test_type == 'acdc' else 'cs'
        return f"test_{test_prefix}_{stage_str}_{strat_short}_{model_short}"[:80]

    @property
    def is_skipped(self) -> bool:
        return self.skip_reason is not None


# ============================================================================
# Pre-flight checks
# ============================================================================

def get_effective_max_iters(stage, model: Optional[str] = None) -> int:
    """Get the target iteration count for a stage/model."""
    if model and model in MODEL_MAX_ITERS:
        return MODEL_MAX_ITERS[model]
    return STAGE_MAX_ITERS.get(stage, 15000)


def find_checkpoint(weights_dir: Path, max_iters: int) -> Optional[Path]:
    """Find the target checkpoint file."""
    ckpt = weights_dir / f'iter_{max_iters}.pth'
    if ckpt.exists() and ckpt.stat().st_size > 1000:
        return ckpt
    return None


def find_config(weights_dir: Path) -> Optional[Path]:
    """Find the training config file."""
    config = weights_dir / 'training_config.py'
    if config.exists():
        return config
    return None


def has_valid_test_result(output_dir: Path, test_type: str = 'cityscapes') -> bool:
    """Check if valid test results already exist.

    Args:
        output_dir: Path to test_results_detailed or test_results_acdc
        test_type: 'cityscapes' or 'acdc' to validate domain keys
    """
    if not output_dir.exists():
        return False

    acdc_domains = {'fog', 'rain', 'snow', 'night'}
    cs_domains = {'frankfurt', 'lindau', 'munster'}

    for ts_dir in output_dir.iterdir():
        if not ts_dir.is_dir() or not ts_dir.name.startswith('202'):
            continue
        rj = ts_dir / 'results.json'
        if not rj.exists():
            continue
        try:
            with open(rj) as f:
                data = json.load(f)
            miou = data.get('overall', {}).get('mIoU', 0)
            if miou is None or miou < 5:
                continue
            domains = set(data.get('per_domain', {}).keys())
            if test_type == 'acdc' and domains & acdc_domains:
                return True
            elif test_type == 'cityscapes' and (domains & cs_domains or not domains):
                # Accept if has CS domains or no domain breakdown (still valid)
                return True
            elif test_type not in ('cityscapes', 'acdc'):
                # For other datasets, just check mIoU > 5
                return True
        except Exception:
            continue
    return False


def get_running_test_jobs() -> Set[str]:
    """Get set of currently running/pending test job names (lowercased)."""
    running = set()
    try:
        result = subprocess.run(
            ['bjobs', '-u', os.environ.get('USER', '${USER}'), '-w'],
            capture_output=True, text=True, timeout=15
        )
        for line in result.stdout.strip().split('\n')[1:]:
            parts = line.split()
            if len(parts) >= 7 and parts[2] in ('RUN', 'PEND'):
                running.add(parts[6].lower())
    except Exception:
        pass
    return running


# ============================================================================
# Job generation
# ============================================================================

def discover_model_dirs(weights_root: Path, strategy: str, dataset: str) -> List[Path]:
    """Discover model directories for a strategy/dataset combination.

    Handles the nested directory structure: strategy/dataset/model[_ratio0p50]/
    """
    strat_dir = weights_root / strategy
    if not strat_dir.is_dir():
        return []

    # Dataset directory (lowercase, handle hyphen variations)
    dataset_lower = dataset.lower().replace('-', '')
    candidates = [dataset_lower, dataset.lower()]

    model_dirs = []
    for ds_name in candidates:
        ds_dir = strat_dir / ds_name
        if ds_dir.is_dir():
            for entry in ds_dir.iterdir():
                if entry.is_dir() and not entry.name.endswith('_backup'):
                    model_dirs.append(entry)
            break  # Found the dataset dir, stop trying alternatives
    return model_dirs


def extract_base_model(model_dir_name: str) -> str:
    """Extract base model name from directory name (strip _ratio0p50 suffix)."""
    # Remove _ratio* suffix
    name = re.sub(r'_ratio\dp\d+$', '', model_dir_name)
    # Remove _aux-* suffix
    name = re.sub(r'_aux-\w+$', '', name)
    return name


def generate_test_jobs(
    stage,
    strategies: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    test_type: str = 'all',
    force: bool = False,
) -> List[TestJob]:
    """Generate list of test jobs with pre-flight checks.

    Args:
        stage: Training stage (1, 2, 'cityscapes', 'cityscapes-gen')
        strategies: Filter by strategies (default: all)
        models: Filter by model names (default: all)
        test_type: 'all', 'cityscapes', or 'acdc'
        force: Skip checking for existing results

    Returns:
        List of TestJob objects
    """
    weights_root = WEIGHTS_ROOTS.get(stage)
    if weights_root is None or not weights_root.exists():
        print(f"ERROR: Weights root not found for stage {stage}: {weights_root}")
        return []

    strategies = strategies or ALL_STRATEGIES
    models_filter = set(models) if models else None
    datasets = STAGE_DATASETS.get(stage, ['cityscapes'])

    jobs = []

    for strategy in strategies:
        for dataset in datasets:
            model_dirs = discover_model_dirs(weights_root, strategy, dataset)
            if not model_dirs:
                continue

            for model_dir in sorted(model_dirs):
                base_model = extract_base_model(model_dir.name)

                # Filter by model if requested
                if models_filter and base_model not in models_filter:
                    continue

                # Check for checkpoint
                max_iters = get_effective_max_iters(stage, base_model)
                checkpoint = find_checkpoint(model_dir, max_iters)
                config = find_config(model_dir)

                if not checkpoint or not config:
                    continue  # Training not complete, skip silently

                # Generate test jobs based on test_type
                test_types_to_generate = []
                if test_type in ('all', 'cityscapes'):
                    test_types_to_generate.append('cityscapes')
                if test_type in ('all', 'acdc') and stage == 'cityscapes-gen':
                    test_types_to_generate.append('acdc')

                # For non-cityscapes stages, test the training dataset
                if stage in (1, 2) and test_type != 'acdc':
                    test_types_to_generate = [dataset]

                for tt in test_types_to_generate:
                    if tt == 'cityscapes':
                        output_dir = model_dir / 'test_results_detailed'
                        test_dataset = 'Cityscapes'
                    elif tt == 'acdc':
                        output_dir = model_dir / 'test_results_acdc'
                        test_dataset = 'ACDC'
                    else:
                        output_dir = model_dir / 'test_results_detailed'
                        test_dataset = DATASET_DISPLAY.get(tt, tt)

                    job = TestJob(
                        strategy=strategy,
                        dataset=dataset,
                        model=model_dir.name,
                        stage=stage,
                        test_type=tt,
                        config_path=config,
                        checkpoint_path=checkpoint,
                        output_dir=output_dir,
                        weights_dir=model_dir,
                    )

                    # Pre-flight: check existing results
                    if not force and has_valid_test_result(output_dir, tt):
                        job.skip_reason = 'Valid results exist'

                    jobs.append(job)

    return jobs


# ============================================================================
# Job submission
# ============================================================================

def generate_job_script(job: TestJob) -> str:
    """Generate LSF job script for a test job."""
    gmem = MODEL_GMEM.get(extract_base_model(job.model), DEFAULT_GMEM)

    # mask2former needs exclusive GPU
    base_model = extract_base_model(job.model)
    if base_model == 'mask2former_swin-b':
        gpu_spec = f'"num=1:mode=exclusive_process:gmem={gmem}"'
        memory = 48000
    else:
        gpu_spec = f'"num=1:gmem={gmem}"'
        memory = 32000

    # Determine test dataset and data root
    if job.test_type == 'acdc':
        test_dataset = 'ACDC'
    elif job.test_type == 'cityscapes':
        test_dataset = 'Cityscapes'
    else:
        test_dataset = DATASET_DISPLAY.get(job.test_type, job.test_type)

    script = f'''#!/bin/bash
#BSUB -J {job.job_name}
#BSUB -q BatchGPU
#BSUB -n 4
#BSUB -M {memory}
#BSUB -R "rusage[mem={memory}]"
#BSUB -gpu {gpu_spec}
#BSUB -W 02:00
#BSUB -o {LOG_DIR}/{job.job_name}_%J.out
#BSUB -e {LOG_DIR}/{job.job_name}_%J.err

# Set permissions
umask 002

# Environment setup
source ~/.bashrc
mamba activate prove
cd {PROJECT_ROOT}

echo "=========================================="
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Strategy: {job.strategy}"
echo "Model: {job.model}"
echo "Test Type: {job.test_type}"
echo "Dataset: {test_dataset}"
echo "Config: {job.config_path}"
echo "Checkpoint: {job.checkpoint_path}"
echo "Output: {job.output_dir}"
echo "=========================================="

python {PROJECT_ROOT}/fine_grained_test.py \\
    --config "{job.config_path}" \\
    --checkpoint "{job.checkpoint_path}" \\
    --output-dir "{job.output_dir}" \\
    --dataset {test_dataset} \\
    --data-root "{DATA_ROOT}" \\
    --test-split test \\
    --batch-size 10

TEST_EXIT=$?

if [ $TEST_EXIT -eq 0 ]; then
    echo "Test completed successfully"
else
    echo "ERROR: Test failed with exit code: $TEST_EXIT"
fi

# Set permissions on output
find "{job.output_dir}" -type d -exec chmod 775 {{}} \\; 2>/dev/null || true
find "{job.output_dir}" -type f -exec chmod 664 {{}} \\; 2>/dev/null || true

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $TEST_EXIT"
echo "=========================================="

exit $TEST_EXIT
'''
    return script


def submit_job(job: TestJob, dry_run: bool = False) -> bool:
    """Submit a test job to LSF."""
    if job.is_skipped:
        return False

    script = generate_job_script(job)

    if dry_run:
        print(f"  SUBMIT: {job.job_name}")
        print(f"    {job.strategy}/{job.dataset}/{job.model} [{job.test_type}]")
        return True

    # Write script to weights dir (avoids /tmp issues)
    script_path = job.weights_dir / f'test_job_{job.test_type}.sh'
    try:
        with open(script_path, 'w') as f:
            f.write(script)
        os.chmod(script_path, 0o755)
    except PermissionError:
        # Fall back to project logs dir
        script_path = LOG_DIR / f'{job.job_name}.sh'
        with open(script_path, 'w') as f:
            f.write(script)
        os.chmod(script_path, 0o755)

    try:
        result = subprocess.run(
            f'bsub < {script_path}',
            shell=True, capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            match = re.search(r'Job <(\d+)>', result.stdout)
            job_id = match.group(1) if match else '?'
            print(f"  ✅ {job.job_name} (Job {job_id})")
            return True
        else:
            print(f"  ❌ {job.job_name}: {result.stderr.strip()[:100]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ❌ {job.job_name}: bsub timeout")
        return False
    except Exception as e:
        print(f"  ❌ {job.job_name}: {e}")
        return False


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Batch Test Submission System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    # Dry run (always do this first!)
    python scripts/batch_test_submission.py --stage cityscapes-gen --dry-run

    # Submit all missing tests
    python scripts/batch_test_submission.py --stage cityscapes-gen -y

    # Only Cityscapes tests (no ACDC)
    python scripts/batch_test_submission.py --stage cityscapes-gen --test-type cityscapes -y

    # Only ACDC cross-domain tests
    python scripts/batch_test_submission.py --stage cityscapes-gen --test-type acdc -y

    # Specific strategies
    python scripts/batch_test_submission.py --stage cityscapes-gen --strategies baseline gen_flux_kontext -y

    # Force retest (ignore existing results)
    python scripts/batch_test_submission.py --stage cityscapes-gen --force --dry-run
'''
    )

    parser.add_argument('--stage', required=True,
                        help='Stage: 1, 2, cityscapes, or cityscapes-gen')
    parser.add_argument('--test-type', choices=['all', 'cityscapes', 'acdc'],
                        default='all',
                        help='Test type (default: all). "all" = Cityscapes + ACDC for cityscapes-gen')
    parser.add_argument('--strategy-type', choices=['all', 'std', 'gen'],
                        default='all',
                        help='Strategy type filter')
    parser.add_argument('--strategies', nargs='+',
                        help='Specific strategies (overrides --strategy-type)')
    parser.add_argument('--models', nargs='+', choices=ALL_MODELS,
                        help='Specific models to test')
    parser.add_argument('--limit', type=int, default=None,
                        help='Maximum number of jobs to submit')
    parser.add_argument('--force', action='store_true',
                        help='Force retest even if valid results exist')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be submitted')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='Skip confirmation prompt')
    parser.add_argument('--delay', type=float, default=0.3,
                        help='Delay between submissions (seconds)')

    args = parser.parse_args()

    # Parse stage
    stage_map = {'1': 1, '2': 2, 'cityscapes': 'cityscapes', 'cityscapes-gen': 'cityscapes-gen'}
    stage = stage_map.get(str(args.stage).lower())
    if stage is None:
        print(f"ERROR: Invalid stage '{args.stage}'. Must be 1, 2, cityscapes, or cityscapes-gen")
        return

    # Determine strategies
    if args.strategies:
        strategies = args.strategies
    elif args.strategy_type == 'std':
        strategies = STD_STRATEGIES
    elif args.strategy_type == 'gen':
        strategies = GEN_STRATEGIES
    else:
        strategies = ALL_STRATEGIES

    # Ensure log dir exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Header
    print(f"\n{'=' * 60}")
    print(f"Batch Test Submission - Stage {stage}")
    print(f"{'=' * 60}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Weights root: {WEIGHTS_ROOTS.get(stage)}")
    print(f"Test type: {args.test_type}")
    print(f"Strategy filter: {args.strategy_type} ({len(strategies)} strategies)")
    if args.models:
        print(f"Model filter: {args.models}")
    if args.force:
        print(f"Force retest: YES")
    print()

    # Check running jobs for duplicate detection
    print("Checking running/pending jobs...")
    running_jobs = get_running_test_jobs()
    print(f"Found {len(running_jobs)} running/pending jobs")
    print()

    # Generate test jobs
    print("Scanning for models needing testing...")
    jobs = generate_test_jobs(
        stage=stage,
        strategies=strategies,
        models=args.models,
        test_type=args.test_type,
        force=args.force,
    )

    # Duplicate detection: skip if job name already in queue
    for job in jobs:
        if not job.is_skipped and job.job_name.lower() in running_jobs:
            job.skip_reason = 'Already in queue'

    # Summary
    total = len(jobs)
    skipped = sum(1 for j in jobs if j.is_skipped)
    to_submit = total - skipped

    print(f"\nJob Summary:")
    print(f"  Total test configurations: {total}")
    print(f"  Skipped: {skipped}")
    print(f"  To submit: {to_submit}")

    if args.limit and to_submit > args.limit:
        actual_submit = min(to_submit, args.limit)
        print(f"  Limited to: {actual_submit}")

    # Skip reason breakdown
    skip_reasons: Dict[str, int] = {}
    for job in jobs:
        if job.skip_reason:
            skip_reasons[job.skip_reason] = skip_reasons.get(job.skip_reason, 0) + 1
    if skip_reasons:
        print(f"\nSkip reasons:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            print(f"  - {reason}: {count}")

    if to_submit == 0:
        print("\nNo jobs to submit!")
        return

    # Show what will be submitted in dry-run
    if not args.dry_run and not args.yes:
        actual = min(to_submit, args.limit) if args.limit else to_submit
        response = input(f"\nSubmit {actual} test jobs? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Submit jobs
    print(f"\n{'=' * 60}")
    print("Submitting jobs..." if not args.dry_run else "Dry run:")
    print(f"{'=' * 60}\n")

    submitted = 0
    failed = 0
    for job in jobs:
        if job.is_skipped:
            continue
        if args.limit and submitted >= args.limit:
            print(f"\nReached limit of {args.limit}")
            break

        if submit_job(job, dry_run=args.dry_run):
            submitted += 1
        else:
            failed += 1

        if not args.dry_run and args.delay > 0:
            time.sleep(args.delay)

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"Summary:")
    print(f"  {'Would submit' if args.dry_run else 'Submitted'}: {submitted}")
    print(f"  Skipped: {skipped}")
    if failed > 0:
        print(f"  Failed: {failed}")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
