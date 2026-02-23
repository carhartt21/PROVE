#!/usr/bin/env python3
"""Submit test jobs for loss ablation experiments.

Tests aux-lovasz and aux-boundary models at their latest checkpoint.
All models are in WEIGHTS_LOSS_ABLATION/{stage1,stage2}/.

Usage:
    python scripts/submit_loss_ablation_tests.py --dry-run     # Preview
    python scripts/submit_loss_ablation_tests.py               # Submit all
    python scripts/submit_loss_ablation_tests.py --limit 5     # Submit first 5
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

BASE = Path('${AWARE_DATA_ROOT}/WEIGHTS_LOSS_ABLATION')
PROJECT_ROOT = Path('${HOME}/repositories/PROVE')
LOG_DIR = PROJECT_ROOT / 'logs'

# Dataset display names for fine_grained_test.py
DATASET_DISPLAY = {
    'bdd10k': 'BDD10k',
    'iddaw': 'IDD-AW',
    'mapillaryvistas': 'MapillaryVistas',
    'outside15k': 'OUTSIDE15k',
}


def get_iter_num(path: str) -> int:
    """Extract iteration number from checkpoint path."""
    m = re.search(r'iter_(\d+)\.pth', str(path))
    return int(m.group(1)) if m else 0


def find_jobs():
    """Find all aux-lovasz and aux-boundary models needing testing."""
    jobs = []

    for stage_dir in ['stage1', 'stage2']:
        stage_path = BASE / stage_dir
        if not stage_path.exists():
            continue

        for strategy in sorted(os.listdir(stage_path)):
            strat_path = stage_path / strategy
            if not strat_path.is_dir():
                continue

            for dataset in sorted(os.listdir(strat_path)):
                ds_path = strat_path / dataset
                if not ds_path.is_dir():
                    continue

                for model_dir in sorted(os.listdir(ds_path)):
                    model_path = ds_path / model_dir
                    if not model_path.is_dir():
                        continue

                    # Only aux-lovasz and aux-boundary
                    if 'aux-lovasz' not in model_dir and 'aux-boundary' not in model_dir:
                        continue

                    # Find latest checkpoint (numeric sort)
                    ckpts = list(model_path.glob('iter_*.pth'))
                    if not ckpts:
                        continue

                    latest = max(ckpts, key=lambda p: get_iter_num(str(p)))
                    latest_iter = get_iter_num(str(latest))

                    # Skip if latest is only 5k — too early
                    if latest_iter < 10000:
                        continue

                    # Check existing test results
                    test_dir = model_path / 'test_results_detailed'
                    already_tested_at_latest = False
                    if test_dir.exists():
                        for result_dir in test_dir.iterdir():
                            if result_dir.is_dir():
                                results_json = result_dir / 'results.json'
                                if results_json.exists():
                                    try:
                                        d = json.loads(results_json.read_text())
                                        ckpt_path = d.get('config', {}).get('checkpoint_path', '')
                                        tested_iter = get_iter_num(ckpt_path)
                                        if tested_iter >= latest_iter:
                                            already_tested_at_latest = True
                                            break
                                    except Exception:
                                        pass

                    if already_tested_at_latest:
                        continue

                    # Find training config
                    config_path = model_path / 'training_config.py'
                    if not config_path.exists():
                        # Try configs subdirectory
                        configs_dir = model_path / 'configs'
                        if configs_dir.exists():
                            config_candidates = list(configs_dir.glob('*.py'))
                            if config_candidates:
                                config_path = config_candidates[0]
                            else:
                                continue
                        else:
                            continue

                    # Determine dataset display name
                    dataset_display = DATASET_DISPLAY.get(dataset, dataset)

                    # Determine if native classes needed
                    use_native = dataset in ('mapillaryvistas', 'outside15k')

                    jobs.append({
                        'stage': stage_dir,
                        'strategy': strategy,
                        'dataset': dataset,
                        'dataset_display': dataset_display,
                        'model': model_dir,
                        'model_path': model_path,
                        'config_path': config_path,
                        'checkpoint_path': latest,
                        'latest_iter': latest_iter,
                        'use_native': use_native,
                    })

    return jobs


def submit_job(job, dry_run=False):
    """Submit a single test job."""
    stage = job['stage']
    strategy = job['strategy']
    dataset = job['dataset']
    model = job['model']
    config_path = job['config_path']
    checkpoint_path = job['checkpoint_path']
    dataset_display = job['dataset_display']
    output_dir = job['model_path'] / 'test_results_detailed'
    latest_iter = job['latest_iter']

    # Build short job name
    short_strat = strategy.replace('gen_', 'g').replace('std_', 's')[:8]
    short_model = model[:6]
    job_name = f"la_{stage}_{short_strat}_{dataset[:3]}_{short_model}"

    if dry_run:
        print(f"  Would submit: {stage}/{strategy}/{dataset}/{model} @ iter_{latest_iter}")
        print(f"    Config: {config_path}")
        print(f"    Checkpoint: {checkpoint_path}")
        print(f"    Job name: {job_name}")
        return True

    # Build test command
    test_cmd = (
        f'source ~/.bashrc && mamba activate prove && cd {PROJECT_ROOT} && '
        f'python fine_grained_test.py '
        f'--config {config_path} '
        f'--checkpoint {checkpoint_path} '
        f'--dataset {dataset_display} '
        f'--output-dir {output_dir} '
        f'--batch-size 10'
    )

    cmd = [
        'bsub',
        '-J', job_name,
        '-q', 'BatchGPU',
        '-n', '10',
        '-gpu', 'num=1:gmem=16G:mode=shared',
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
            print(f"  Submitted: {stage}/{strategy}/{dataset}/{model} @ iter_{latest_iter} (Job {job_id})")
            return True
        else:
            print(f"  Failed: {stage}/{strategy}/{dataset}/{model}")
            print(f"    Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Submit loss ablation test jobs')
    parser.add_argument('--dry-run', action='store_true', help='Preview without submitting')
    parser.add_argument('--limit', type=int, default=None, help='Max jobs to submit')
    parser.add_argument('--loss-type', choices=['aux-lovasz', 'aux-boundary', 'all'],
                       default='all', help='Filter by loss type')
    parser.add_argument('--stage', choices=['stage1', 'stage2', 'all'],
                       default='all', help='Filter by stage')
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)

    jobs = find_jobs()

    # Apply filters
    if args.loss_type != 'all':
        jobs = [j for j in jobs if args.loss_type in j['model']]
    if args.stage != 'all':
        jobs = [j for j in jobs if j['stage'] == args.stage]

    print(f"\n{'='*60}")
    print(f"Loss Ablation Test Submission")
    print(f"{'='*60}")
    print(f"Total jobs to {'preview' if args.dry_run else 'submit'}: {len(jobs)}")
    if args.limit:
        jobs = jobs[:args.limit]
        print(f"Limited to: {args.limit}")
    print()

    submitted = 0
    failed = 0
    for job in jobs:
        if submit_job(job, dry_run=args.dry_run):
            submitted += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    action = 'previewed' if args.dry_run else 'submitted'
    print(f"Total {action}: {submitted}, failed: {failed}")
    if args.dry_run and submitted > 0:
        print(f"\nRun without --dry-run to submit these jobs.")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
