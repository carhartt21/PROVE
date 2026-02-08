#!/usr/bin/env python3
"""
Retest cityscapes-gen models that have buggy test results.

The initial cityscapes-gen job submissions used --test-split val and
DATA_ROOT=/scratch/aaa_exchange/AWARE/CITYSCAPES, but Cityscapes test data
is actually in FINAL_SPLITS/test/images/Cityscapes/{frankfurt,lindau,munster}.

This script finds models with empty/buggy results and submits correct retests.

Usage:
    python scripts/retest_cityscapes_gen.py --dry-run       # Preview retests
    python scripts/retest_cityscapes_gen.py                  # Submit retests
    python scripts/retest_cityscapes_gen.py --limit 10       # Submit max 10
    python scripts/retest_cityscapes_gen.py --acdc-only      # Only ACDC retests
"""

import os
import json
import subprocess
import argparse
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WEIGHTS_ROOT = Path('/scratch/aaa_exchange/AWARE/WEIGHTS_CITYSCAPES_GEN')
DATA_ROOT = '/scratch/aaa_exchange/AWARE/FINAL_SPLITS'
LOG_DIR = PROJECT_ROOT / 'logs'


def find_models_needing_retest(acdc_only=False, verbose=False):
    """Find completed models that need retesting.
    
    Returns list of dicts with strategy, model, config_path, checkpoint_path.
    """
    needs_retest = []
    
    for strategy_dir in sorted(WEIGHTS_ROOT.iterdir()):
        if not strategy_dir.is_dir():
            continue
        strategy = strategy_dir.name
        
        cityscapes_dir = strategy_dir / 'cityscapes'
        if not cityscapes_dir.is_dir():
            continue
        
        for model_dir in sorted(cityscapes_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name.endswith('_backup'):
                continue
            
            model = model_dir.name
            
            # Check for completed training (iter_20000.pth)
            checkpoint = model_dir / 'iter_20000.pth'
            config = model_dir / 'training_config.py'
            
            if not checkpoint.exists() or not config.exists():
                continue
            
            # Check Cityscapes test results
            needs_cs_retest = False
            needs_acdc_retest = False
            
            if not acdc_only:
                test_dir = model_dir / 'test_results_detailed'
                if not test_dir.exists():
                    needs_cs_retest = True
                else:
                    # Check if any result has valid mIoU with test_split=test
                    has_valid = False
                    for ts_dir in sorted(test_dir.iterdir()):
                        if not ts_dir.is_dir():
                            continue
                        rj = ts_dir / 'results.json'
                        if rj.exists():
                            try:
                                with open(rj) as f:
                                    data = json.load(f)
                                miou = data.get('overall', {}).get('mIoU')
                                split = data.get('config', {}).get('test_split', '?')
                                if miou and miou > 5 and split == 'test':
                                    has_valid = True
                                    break
                            except Exception:
                                continue
                    if not has_valid:
                        needs_cs_retest = True
            
            # Check ACDC test results
            acdc_dir = model_dir / 'test_results_acdc'
            if not acdc_dir.exists():
                needs_acdc_retest = True
            else:
                has_valid_acdc = False
                for ts_dir in sorted(acdc_dir.iterdir()):
                    if not ts_dir.is_dir():
                        continue
                    rj = ts_dir / 'results.json'
                    if rj.exists():
                        try:
                            with open(rj) as f:
                                data = json.load(f)
                            miou = data.get('overall', {}).get('mIoU')
                            if miou and miou > 5:
                                has_valid_acdc = True
                                break
                        except Exception:
                            continue
                if not has_valid_acdc:
                    needs_acdc_retest = True
            
            if needs_cs_retest or needs_acdc_retest:
                needs_retest.append({
                    'strategy': strategy,
                    'model': model,
                    'config_path': str(config),
                    'checkpoint_path': str(checkpoint),
                    'model_dir': str(model_dir),
                    'needs_cs': needs_cs_retest,
                    'needs_acdc': needs_acdc_retest,
                })
                if verbose:
                    parts = []
                    if needs_cs_retest:
                        parts.append('CS')
                    if needs_acdc_retest:
                        parts.append('ACDC')
                    print(f"  Need retest ({'+'.join(parts)}): {strategy}/{model}")
    
    return needs_retest


def submit_retest_job(config, dry_run=False, acdc_only=False):
    """Submit retest job via LSF."""
    strategy = config['strategy']
    model = config['model']
    config_path = config['config_path']
    checkpoint_path = config['checkpoint_path']
    model_dir = config['model_dir']
    
    job_name = f"rt_csgen_{strategy}_{model}"
    # Truncate if too long
    if len(job_name) > 80:
        job_name = job_name[:80]
    
    # Build test commands
    test_commands = []
    
    if config['needs_cs'] and not acdc_only:
        test_commands.append(f'''
echo "Testing on Cityscapes val (via FINAL_SPLITS/test)..."
python {PROJECT_ROOT}/fine_grained_test.py \\
    --config "{config_path}" \\
    --checkpoint "{checkpoint_path}" \\
    --output-dir "{model_dir}/test_results_detailed" \\
    --dataset Cityscapes \\
    --data-root "{DATA_ROOT}" \\
    --test-split test \\
    --batch-size 10
CS_EXIT=$?
echo "Cityscapes test exit code: $CS_EXIT"
''')
    
    if config['needs_acdc']:
        test_commands.append(f'''
echo "Testing on ACDC (cross-domain)..."
python {PROJECT_ROOT}/fine_grained_test.py \\
    --config "{config_path}" \\
    --checkpoint "{checkpoint_path}" \\
    --output-dir "{model_dir}/test_results_acdc" \\
    --dataset ACDC \\
    --data-root "{DATA_ROOT}" \\
    --test-split test \\
    --batch-size 10
ACDC_EXIT=$?
echo "ACDC test exit code: $ACDC_EXIT"
''')
    
    if not test_commands:
        return False
    
    script = f'''#!/bin/bash
#BSUB -J {job_name}
#BSUB -q BatchGPU
#BSUB -n 8
#BSUB -M 32000
#BSUB -R "rusage[mem=32000]"
#BSUB -gpu "num=1"
#BSUB -W 02:00
#BSUB -o {LOG_DIR}/{job_name}_%J.out
#BSUB -e {LOG_DIR}/{job_name}_%J.err

source ~/.bashrc
mamba activate prove
cd {PROJECT_ROOT}

echo "=========================================="
echo "Retesting: {strategy}/cityscapes/{model}"
echo "=========================================="

{''.join(test_commands)}

echo ""
echo "All retests complete."
'''
    
    if dry_run:
        print(f"  [DRY-RUN] Would submit: {job_name}")
        return True
    
    # Write and submit
    script_path = f'/tmp/retest_{strategy}_{model}.sh'
    with open(script_path, 'w') as f:
        f.write(script)
    os.chmod(script_path, 0o755)
    
    try:
        result = subprocess.run(
            f'bsub < {script_path}',
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print(f"  ✅ Submitted: {job_name}")
            return True
        else:
            print(f"  ❌ Failed: {job_name}: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"  ❌ Error submitting {job_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Retest cityscapes-gen models with buggy results')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be submitted')
    parser.add_argument('--limit', type=int, default=0, help='Max jobs to submit (0=unlimited)')
    parser.add_argument('--acdc-only', action='store_true', help='Only retest ACDC cross-domain')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Cityscapes-Gen Retest Script")
    print("=" * 60)
    print(f"Weights root: {WEIGHTS_ROOT}")
    print(f"Data root: {DATA_ROOT}")
    print(f"Mode: {'ACDC only' if args.acdc_only else 'Cityscapes + ACDC'}")
    print()
    
    configs = find_models_needing_retest(acdc_only=args.acdc_only, verbose=args.verbose)
    
    if not configs:
        print("All models have valid test results!")
        return
    
    cs_count = sum(1 for c in configs if c['needs_cs'])
    acdc_count = sum(1 for c in configs if c['needs_acdc'])
    print(f"\nFound {len(configs)} models needing retest:")
    if not args.acdc_only:
        print(f"  Cityscapes: {cs_count}")
    print(f"  ACDC: {acdc_count}")
    
    if args.limit > 0:
        configs = configs[:args.limit]
        print(f"\nLimited to {args.limit} jobs")
    
    print()
    submitted = 0
    for config in configs:
        success = submit_retest_job(config, dry_run=args.dry_run, acdc_only=args.acdc_only)
        if success:
            submitted += 1
        if not args.dry_run:
            time.sleep(0.3)
    
    print(f"\n{'Would submit' if args.dry_run else 'Submitted'} {submitted}/{len(configs)} jobs")


if __name__ == '__main__':
    main()
