#!/usr/bin/env python3
"""
Auto-submit fine-grained test jobs for completed training checkpoints.

Unified script that supports all training stages:
  - Stage 1 (clear-day only training)
  - Stage 2 (all-domain training)
  - Cityscapes (pipeline verification)
  - Cityscapes-gen (generative evaluation on Cityscapes)

Scans weights directories for models with final checkpoints that need testing
and submits test jobs for them. For cityscapes/cityscapes-gen stages, also
checks and submits ACDC cross-domain test jobs.

Usage:
    # Stage 1 (default)
    python scripts/auto_submit_tests.py --stage 1 --dry-run
    python scripts/auto_submit_tests.py --stage 1

    # Stage 2
    python scripts/auto_submit_tests.py --stage 2 --dry-run
    python scripts/auto_submit_tests.py --stage 2 --include-ratio1p0

    # Cityscapes pipeline verification
    python scripts/auto_submit_tests.py --stage cityscapes --dry-run

    # Cityscapes generative evaluation (includes ACDC cross-domain)
    python scripts/auto_submit_tests.py --stage cityscapes-gen --dry-run

    # Common options
    python scripts/auto_submit_tests.py --stage 1 --main-only --limit 20
    python scripts/auto_submit_tests.py --stage 1 --strategies baseline gen_cycleGAN
    python scripts/auto_submit_tests.py --stage cityscapes-gen --models segformer_mit-b3 pspnet_r50
"""

import os
import subprocess
import argparse
import json
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOG_DIR = PROJECT_ROOT / 'logs'

# Weight roots per stage
WEIGHTS_ROOTS = {
    '1': Path('${AWARE_DATA_ROOT}/WEIGHTS'),
    '2': Path('${AWARE_DATA_ROOT}/WEIGHTS_STAGE_2'),
    'cityscapes': Path('${AWARE_DATA_ROOT}/WEIGHTS_CITYSCAPES'),
    'cityscapes-gen': Path('${AWARE_DATA_ROOT}/WEIGHTS_CITYSCAPES_GEN'),
}

# Datasets per stage
STANDARD_DATASETS = ['bdd10k', 'idd-aw', 'mapillaryvistas', 'outside15k']
MAIN_DATASETS = ['bdd10k', 'mapillaryvistas']
CITYSCAPES_DATASETS = ['cityscapes']

# Dataset display name mapping (directory name → CLI dataset name)
DATASET_DISPLAY = {
    'bdd10k': 'BDD10k',
    'idd-aw': 'IDD-AW',
    'iddaw': 'IDD-AW',
    'mapillaryvistas': 'MapillaryVistas',
    'outside15k': 'OUTSIDE15k',
    'cityscapes': 'Cityscapes',
}

# Job name prefixes per stage (to avoid collisions and identify running jobs)
JOB_PREFIXES = {
    '1': 'fg_',
    '2': 'fg2_',
    'cityscapes': 'fgcs_',
    'cityscapes-gen': 'fgcg_',
}

# Strategies
GENERATIVE_STRATEGIES = [
    'gen_Attribute_Hallucination',
    'gen_augmenters',
    'gen_automold',
    'gen_CNetSeg',
    'gen_CUT',
    'gen_cyclediffusion',
    'gen_cycleGAN',
    'gen_flux_kontext',
    'gen_Img2Img',
    'gen_IP2P',
    'gen_LANIT',
    'gen_Qwen_Image_Edit',
    'gen_stargan_v2',
    'gen_step1x_new',
    'gen_step1x_v1p2',
    'gen_SUSTechGAN',
    'gen_TSIT',
    'gen_UniControl',
    'gen_VisualCloze',
    'gen_Weather_Effect_Generator',
    'gen_albumentations_weather',
]

STANDARD_STRATEGIES = [
    'baseline',
    'std_photometric_distort',
    'std_minimal',
    'std_autoaugment',
    'std_cutmix',
    'std_mixup',
    'std_randaugment',
]

ALL_STRATEGIES = STANDARD_STRATEGIES + GENERATIVE_STRATEGIES

# Stage configurations
STAGE_CONFIGS = {
    '1': {
        'label': 'Stage 1 (Clear-Day Only)',
        'datasets': STANDARD_DATASETS,
        'cross_domain': False,
    },
    '2': {
        'label': 'Stage 2 (All Domains)',
        'datasets': STANDARD_DATASETS,
        'cross_domain': False,
    },
    'cityscapes': {
        'label': 'Cityscapes Pipeline Verification',
        'datasets': CITYSCAPES_DATASETS,
        'cross_domain': True,  # Also test on ACDC
    },
    'cityscapes-gen': {
        'label': 'Cityscapes Generative Evaluation',
        'datasets': CITYSCAPES_DATASETS,
        'cross_domain': True,  # Also test on ACDC
    },
}


# ============================================================================
# Job Detection
# ============================================================================

def get_running_test_jobs(stage: str) -> Set[str]:
    """Get set of currently running/pending test job names for a given stage.

    Args:
        stage: Training stage identifier.

    Returns:
        Set of lowercase job names matching the stage's prefix.
    """
    prefix = JOB_PREFIXES.get(stage, 'fg_')
    running_jobs: Set[str] = set()

    try:
        result = subprocess.run(
            ['bjobs', '-u', '${USER}', '-w'],
            capture_output=True, text=True, timeout=15,
        )
        for line in result.stdout.strip().split('\n')[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 7:
                stat = parts[2]
                job_name = parts[6]
                if stat in ('RUN', 'PEND') and job_name.lower().startswith(prefix):
                    running_jobs.add(job_name.lower())
    except subprocess.TimeoutExpired:
        print("Warning: bjobs timed out, assuming no running jobs")
    except Exception as e:
        print(f"Warning: Could not get running jobs: {e}")

    return running_jobs


# ============================================================================
# Configuration Scanner
# ============================================================================

def _has_valid_results(test_dir: Path) -> bool:
    """Check if a test results directory contains valid results (mIoU > 5%).

    Args:
        test_dir: Path to test_results_detailed or test_results_acdc directory.

    Returns:
        True if valid results exist.
    """
    if not test_dir.exists():
        return False

    for result_subdir in test_dir.iterdir():
        if result_subdir.is_dir() and result_subdir.name.startswith('202'):
            results_json = result_subdir / 'results.json'
            if results_json.exists():
                try:
                    with open(results_json) as f:
                        data = json.load(f)
                    miou = data.get('overall', {}).get('mIoU')
                    if miou is not None and miou > 0.05:  # > 5% mIoU is valid
                        return True
                except Exception:
                    pass
    return False


def _get_max_iters_from_config(config_path: Path) -> Optional[int]:
    """Extract max_iters from a training_config.py file.

    Handles both keyword format (max_iters=15000) and dict format ('max_iters': 80000).
    """
    try:
        with open(config_path, 'r') as f:
            content = f.read()
        match = re.search(r"'?max_iters'?\s*[=:]\s*(\d+)", content)
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return None


def find_configs_needing_tests(
    stage: str,
    datasets: Optional[List[str]] = None,
    strategies: Optional[List[str]] = None,
    main_only: bool = False,
    include_ratio1p0: bool = False,
    check_cross_domain: bool = False,
) -> List[Dict]:
    """Find all configurations that have checkpoints but need testing.

    Args:
        stage: Training stage ('1', '2', 'cityscapes', 'cityscapes-gen').
        datasets: Override datasets to check (default: stage-specific).
        strategies: Override strategies to check (default: ALL_STRATEGIES).
        main_only: Only consider main datasets (stages 1/2 only).
        include_ratio1p0: Include ratio1p0 models (real-only training).
        check_cross_domain: Also check for missing ACDC cross-domain tests.

    Returns:
        List of dicts with 'strategy', 'dataset', 'model', 'weights_dir',
        'config_path', 'checkpoint_path', 'test_type' ('main' or 'acdc').
    """
    weights_root = WEIGHTS_ROOTS.get(stage)
    if weights_root is None or not weights_root.exists():
        print(f"Warning: Weights root does not exist: {weights_root}")
        return []

    stage_cfg = STAGE_CONFIGS[stage]

    # Determine datasets
    if datasets:
        datasets_to_check = datasets
    elif main_only and stage in ('1', '2'):
        datasets_to_check = MAIN_DATASETS
    else:
        datasets_to_check = stage_cfg['datasets']

    strategies_to_check = strategies or ALL_STRATEGIES

    configs_needing_tests: List[Dict] = []
    seen_configs: Set[Tuple[str, str, str, str]] = set()  # (strategy, dataset, model, test_type)

    for strategy in strategies_to_check:
        strategy_path = weights_root / strategy
        if not strategy_path.exists():
            continue

        for dataset in datasets_to_check:
            # Handle directory naming variants (idd-aw vs iddaw)
            ds_dir_candidates = [dataset]
            if '-' in dataset:
                ds_dir_candidates.append(dataset.replace('-', ''))

            for ds_dir_name in ds_dir_candidates:
                dataset_path = strategy_path / ds_dir_name
                if not dataset_path.exists():
                    continue

                try:
                    for model_dir in dataset_path.iterdir():
                        if not model_dir.is_dir():
                            continue
                        if model_dir.name.endswith('_backup'):
                            continue

                        # Filter by ratio type
                        if 'ratio' in model_dir.name:
                            if 'ratio0p50' in model_dir.name:
                                pass  # Always include
                            elif 'ratio1p0' in model_dir.name and include_ratio1p0:
                                pass  # Include if flag set
                            else:
                                continue  # Skip other ratios
                        
                        config_path = model_dir / 'training_config.py'
                        if not config_path.exists():
                            continue

                        # Determine expected checkpoint
                        expected_max_iters = _get_max_iters_from_config(config_path)
                        if expected_max_iters is None:
                            continue

                        checkpoint_path = model_dir / f'iter_{expected_max_iters}.pth'
                        if not checkpoint_path.exists():
                            continue  # Training not complete

                        # Check main test results
                        if not _has_valid_results(model_dir / 'test_results_detailed'):
                            key = (strategy, dataset, model_dir.name, 'main')
                            if key not in seen_configs:
                                seen_configs.add(key)
                                configs_needing_tests.append({
                                    'strategy': strategy,
                                    'dataset': dataset,
                                    'model': model_dir.name,
                                    'weights_dir': model_dir,
                                    'config_path': config_path,
                                    'checkpoint_path': checkpoint_path,
                                    'test_type': 'main',
                                })

                        # Check ACDC cross-domain results (cityscapes/cityscapes-gen only)
                        if check_cross_domain and stage_cfg.get('cross_domain', False):
                            if not _has_valid_results(model_dir / 'test_results_acdc'):
                                key = (strategy, dataset, model_dir.name, 'acdc')
                                if key not in seen_configs:
                                    seen_configs.add(key)
                                    configs_needing_tests.append({
                                        'strategy': strategy,
                                        'dataset': dataset,
                                        'model': model_dir.name,
                                        'weights_dir': model_dir,
                                        'config_path': config_path,
                                        'checkpoint_path': checkpoint_path,
                                        'test_type': 'acdc',
                                    })

                except PermissionError:
                    continue
                except OSError as e:
                    print(f"Warning: Error scanning {dataset_path}: {e}")
                    continue

    return configs_needing_tests


# ============================================================================
# Job Submission
# ============================================================================

def submit_test_job(
    config: Dict,
    stage: str,
    dry_run: bool = False,
    shared_gpu: bool = True,
) -> bool:
    """Submit a fine-grained test job for a configuration.

    Args:
        config: Dict with strategy, dataset, model, weights_dir, etc.
        stage: Training stage for job naming.
        dry_run: If True, just print what would be submitted.
        shared_gpu: If True, use shared GPU mode.

    Returns:
        True if job was submitted (or would be in dry-run).
    """
    strategy = config['strategy']
    dataset = config['dataset']
    model = config['model']
    weights_dir = config['weights_dir']
    config_path = config['config_path']
    checkpoint_path = config['checkpoint_path']
    test_type = config['test_type']

    # Get display name for dataset
    if test_type == 'acdc':
        dataset_display = 'ACDC'
        output_dir = weights_dir / 'test_results_acdc'
        extra_args = '--data-root ${AWARE_DATA_ROOT}/FINAL_SPLITS --test-split test'
    else:
        dataset_display = DATASET_DISPLAY.get(dataset, dataset)
        output_dir = weights_dir / 'test_results_detailed'
        extra_args = ''

    # Build short job name
    prefix = JOB_PREFIXES.get(stage, 'fg_')
    short_strategy = strategy.replace('gen_', 'g').replace('std_', 's')[:10]
    short_dataset = dataset[:4]
    short_model = model[:3]
    type_suffix = '_acdc' if test_type == 'acdc' else ''
    job_name = f"{prefix}{short_strategy}_{short_dataset}_{short_model}{type_suffix}"

    if dry_run:
        label = f" [ACDC cross-domain]" if test_type == 'acdc' else ''
        print(f"  Would submit: {strategy}/{dataset}/{model}{label}")
        print(f"    Job name: {job_name}")
        return True

    # Build bsub command
    gpu_spec = "num=1:gmem=16G:mode=shared" if shared_gpu else "num=1:gmem=16G"
    test_cmd = (
        f'source ~/.bashrc && mamba activate prove && cd {PROJECT_ROOT} && '
        f'python fine_grained_test.py '
        f'--config {config_path} '
        f'--checkpoint {checkpoint_path} '
        f'--dataset {dataset_display} '
        f'--output-dir {output_dir} '
        f'--batch-size 10'
    )
    if extra_args:
        test_cmd += f' {extra_args}'

    cmd = [
        'bsub',
        '-J', job_name,
        '-q', 'BatchGPU',
        '-n', '10',
        '-gpu', gpu_spec,
        '-W', '0:30',
        '-o', f'{LOG_DIR}/{job_name}_%J.out',
        '-e', f'{LOG_DIR}/{job_name}_%J.err',
        test_cmd,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            match = re.search(r'Job <(\d+)>', result.stdout)
            job_id = match.group(1) if match else 'unknown'
            label = ' [ACDC]' if test_type == 'acdc' else ''
            print(f"  Submitted: {strategy}/{dataset}/{model}{label} (Job {job_id})")
            return True
        else:
            print(f"  Failed: {strategy}/{dataset}/{model}")
            print(f"    Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  Timeout submitting: {strategy}/{dataset}/{model}")
        return False
    except Exception as e:
        print(f"  Error: {strategy}/{dataset}/{model}: {e}")
        return False


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Auto-submit test jobs for completed training checkpoints (all stages)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --stage 1 --dry-run              # Preview Stage 1 test jobs
  %(prog)s --stage 2 --limit 20             # Submit up to 20 Stage 2 test jobs
  %(prog)s --stage cityscapes-gen --dry-run  # Preview Cityscapes-gen test jobs
  %(prog)s --stage 1 --main-only            # Only BDD10k + MapillaryVistas
  %(prog)s --stage 1 --strategies baseline gen_cycleGAN
""",
    )
    parser.add_argument('--stage', type=str, default='1',
                       choices=['1', '2', 'cityscapes', 'cityscapes-gen'],
                       help='Training stage (default: 1)')
    parser.add_argument('--main-only', action='store_true',
                       help='Only process main datasets (BDD10k, MapillaryVistas) - stages 1/2 only')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be submitted without actually submitting')
    parser.add_argument('--limit', type=int, default=None,
                       help='Maximum number of jobs to submit')
    parser.add_argument('--strategies', nargs='+', default=None,
                       help='Only check specific strategies')
    parser.add_argument('--models', nargs='+', default=None,
                       help='Only check specific model directory name patterns')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Override datasets to check')
    parser.add_argument('--shared-gpu', action='store_true', default=True,
                       help='Use shared GPU mode (default: True)')
    parser.add_argument('--no-shared-gpu', dest='shared_gpu', action='store_false',
                       help='Use exclusive GPU mode')
    parser.add_argument('--include-ratio1p0', action='store_true',
                       help='Include ratio1p0 models (real-only training)')
    parser.add_argument('--skip-cross-domain', action='store_true',
                       help='Skip ACDC cross-domain tests (cityscapes/cityscapes-gen stages)')
    parser.add_argument('--cross-domain-only', action='store_true',
                       help='Only submit ACDC cross-domain tests (cityscapes/cityscapes-gen stages)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show verbose output')

    args = parser.parse_args()
    stage = args.stage
    stage_cfg = STAGE_CONFIGS[stage]

    print("=" * 60)
    print(f"Auto-Submit Test Jobs — {stage_cfg['label']}")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Weights root: {WEIGHTS_ROOTS[stage]}")
    if stage in ('1', '2'):
        print(f"Main datasets only: {args.main_only}")
    if stage == '2':
        print(f"Include ratio1p0: {args.include_ratio1p0}")
    if stage_cfg.get('cross_domain'):
        print(f"Cross-domain (ACDC): {'skip' if args.skip_cross_domain else 'only' if args.cross_domain_only else 'yes'}")
    print(f"Dry run: {args.dry_run}")
    print(f"Shared GPU: {args.shared_gpu}")
    if args.strategies:
        print(f"Strategies filter: {args.strategies}")
    if args.models:
        print(f"Models filter: {args.models}")
    if args.datasets:
        print(f"Datasets filter: {args.datasets}")
    if args.limit:
        print(f"Limit: {args.limit} jobs")
    print()

    # ---- Get running jobs ----
    print("Checking for running jobs...")
    running_jobs = get_running_test_jobs(stage)
    print(f"Found {len(running_jobs)} running/pending test jobs (prefix: {JOB_PREFIXES[stage]})")
    print()

    # ---- Find configs needing tests ----
    print("Scanning for configurations needing tests...")

    all_configs: List[Dict] = []

    # Main tests (unless --cross-domain-only)
    if not args.cross_domain_only:
        main_configs = find_configs_needing_tests(
            stage=stage,
            datasets=args.datasets,
            strategies=args.strategies,
            main_only=args.main_only,
            include_ratio1p0=args.include_ratio1p0,
            check_cross_domain=False,
        )
        all_configs.extend(main_configs)

    # Cross-domain tests (cityscapes/cityscapes-gen only, unless --skip-cross-domain)
    if stage_cfg.get('cross_domain') and not args.skip_cross_domain:
        cross_configs = find_configs_needing_tests(
            stage=stage,
            datasets=args.datasets,
            strategies=args.strategies,
            main_only=False,
            include_ratio1p0=args.include_ratio1p0,
            check_cross_domain=True,
        )
        # Only keep cross-domain entries (test_type == 'acdc')
        cross_configs = [c for c in cross_configs if c['test_type'] == 'acdc']
        all_configs.extend(cross_configs)

    # Apply model filter if specified
    if args.models:
        filtered = []
        for cfg in all_configs:
            model_name = cfg['model']
            if any(m in model_name for m in args.models):
                filtered.append(cfg)
        all_configs = filtered

    # Separate counts
    main_count = sum(1 for c in all_configs if c['test_type'] == 'main')
    acdc_count = sum(1 for c in all_configs if c['test_type'] == 'acdc')
    print(f"Found {len(all_configs)} configurations needing tests")
    if main_count:
        print(f"  Main tests: {main_count}")
    if acdc_count:
        print(f"  ACDC cross-domain tests: {acdc_count}")
    print()

    if not all_configs:
        print("No configurations need testing!")
        return

    # ---- Submit jobs ----
    print("Submitting test jobs...")
    submitted = 0
    skipped_running = 0
    failed = 0

    for config in all_configs:
        if args.limit and submitted >= args.limit:
            print(f"\nReached limit of {args.limit} jobs")
            break

        strategy = config['strategy']
        dataset = config['dataset']
        model = config['model']
        test_type = config['test_type']

        # Check if already running
        prefix = JOB_PREFIXES[stage]
        strat_key = strategy.replace('gen_', 'g').replace('std_', 's').lower()[:8]
        dataset_key = dataset[:4].lower()
        model_key = model[:3].lower()
        type_suffix = '_acdc' if test_type == 'acdc' else ''

        is_running = any(
            strat_key in job and dataset_key in job and model_key in job
            and (type_suffix in job if type_suffix else '_acdc' not in job)
            for job in running_jobs
        )

        if is_running:
            if args.verbose:
                label = ' [ACDC]' if test_type == 'acdc' else ''
                print(f"  Skip (running): {strategy}/{dataset}/{model}{label}")
            skipped_running += 1
            continue

        success = submit_test_job(config, stage=stage, dry_run=args.dry_run, shared_gpu=args.shared_gpu)
        if success:
            submitted += 1
        else:
            failed += 1

        # Small delay to avoid overwhelming scheduler
        if not args.dry_run:
            time.sleep(0.3)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Submitted: {submitted}")
    print(f"Skipped (already running): {skipped_running}")
    print(f"Failed: {failed}")
    remaining = len(all_configs) - submitted - skipped_running - failed
    if remaining > 0:
        print(f"Remaining (limit reached): {remaining}")


if __name__ == '__main__':
    main()
