#!/usr/bin/env python3
"""
Update the retraining progress tracker based on current weights.

Usage:
    python scripts/update_retraining_tracker.py
    python scripts/update_retraining_tracker.py --stage 1  # Only check stage 1 (clear_day)
    python scripts/update_retraining_tracker.py --verbose   # Show all status details
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json
import re
import os

# Detect project root from script location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Configuration
WEIGHTS_ROOT = Path(os.environ.get('PROVE_WEIGHTS_ROOT', '/scratch/aaa_exchange/AWARE/WEIGHTS'))
TRACKER_PATH = PROJECT_ROOT / 'docs' / 'RETRAINING_TRACKER.md'
DATASETS = ['bdd10k', 'idd-aw', 'mapillaryvistas', 'outside15k']

# Models - now using new structure: dataset_cd/model_ratio
# Domain filter is part of dataset directory, not model directory
MODELS = {
    'clear_day': {
        'gen': 'deeplabv3plus_r50_ratio0p50',
        'std': 'deeplabv3plus_r50',
        'baseline': 'deeplabv3plus_r50',
    },
    'all_domains': {
        'gen': 'deeplabv3plus_r50_ratio0p50',
        'std': 'deeplabv3plus_r50',
        'baseline': 'deeplabv3plus_r50',
    }
}

# Strategy definitions
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
    # 'gen_NST',  # EXCLUDED: Generated images missing
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
    'photometric_distort',
    'std_autoaugment',
    'std_cutmix',
    'std_mixup',
    'std_randaugment',
]

ALL_STRATEGIES = GENERATIVE_STRATEGIES + STANDARD_STRATEGIES

# Skip combinations
SKIP_COMBOS = {
    ('gen_Qwen_Image_Edit', 'bdd10k'),  # No BDD10k data for Qwen
    ('gen_flux_kontext', 'bdd10k'),    # flux_kontext only has MapillaryVistas and OUTSIDE15k
    ('gen_flux_kontext', 'idd-aw'),    # flux_kontext only has MapillaryVistas and OUTSIDE15k
    ('gen_step1x_new', 'bdd10k'),       # step1x_new has incomplete BDD10k coverage
    ('gen_cyclediffusion', 'outside15k'),  # cyclediffusion has no OUTSIDE15k images
}


def get_status_emoji(status):
    """Convert status to emoji."""
    return {
        'complete': '✅',
        'running': '🔄',
        'pending': '⏳',
        'failed': '❌',
        'skip': '➖',
    }.get(status, '❓')


def check_weight_status(strategy, dataset, domain_filter='clear_day'):
    """Check if weights exist and are valid."""
    # Determine model directory based on strategy type
    if strategy == 'baseline':
        model_dir = MODELS[domain_filter]['baseline']
    elif strategy.startswith('gen_'):
        model_dir = MODELS[domain_filter]['gen']
    else:
        model_dir = MODELS[domain_filter]['std']
    
    # Check if this is a skip combo
    if (strategy, dataset) in SKIP_COMBOS:
        return 'skip', None
    
    # Build dataset directory with domain suffix
    # New structure: strategy/dataset_cd/model_ratio
    domain_abbrev = {'clear_day': 'cd', 'clear_night': 'cn', 'rainy_day': 'rd', 'rainy_night': 'rn', 'fog': 'fg', 'snow': 'sn'}
    domain_suffix = f'_{domain_abbrev.get(domain_filter, domain_filter[:2])}' if domain_filter else ''
    dataset_dir = f'{dataset}{domain_suffix}'
    
    weights_path = WEIGHTS_ROOT / strategy / dataset_dir / model_dir
    checkpoint = weights_path / "iter_80000.pth"
    
    try:
        if checkpoint.exists():
            # Check if it's a valid file (not empty)
            if checkpoint.stat().st_size > 1000:  # At least 1KB
                return 'complete', weights_path
            else:
                return 'failed', weights_path
    except PermissionError:
        # If we can't access it, assume it exists but we can't read it? 
        # Or better: mark as 'pending' but with a note?
        # For now, let's treat as 'complete' if we can see it exists via other means, 
        # but if we get PermissionError on .exists() it's tricky.
        # Let's try to check parent dir.
        try:
            if weights_path.exists():
                 # Check if the file name is visible
                 files = os.listdir(weights_path)
                 if "iter_80000.pth" in files:
                     return 'complete', weights_path
        except:
            pass
        return 'pending', weights_path
    
    return 'pending', weights_path


def get_running_jobs():
    """Get list of currently running retrain jobs from both users."""
    running = {}  # {(strategy, dataset): status}
    
    # Check jobs from multiple users
    users_to_check = ['', '-u chge7185']  # '' means current user
    
    for user_flag in users_to_check:
        try:
            cmd = ['bjobs', '-a', '-o', 'JOBID JOB_NAME STAT']
            if user_flag:
                cmd = ['bjobs'] + user_flag.split() + ['-a', '-o', 'JOBID JOB_NAME STAT']
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 3:
                    job_name = parts[1]
                    stat = parts[2]
                    if 'retrain' in job_name:
                        # Parse job name format: retrain_<strategy>_<dataset>
                        # Handle strategies with underscores
                        name_parts = job_name.replace('retrain_', '').split('_')
                        if len(name_parts) >= 2:
                            dataset = name_parts[-1]  # Last part is dataset
                            strategy = '_'.join(name_parts[:-1])  # Rest is strategy
                            
                            # Store the most "active" status for this config
                            current_stat = running.get((strategy, dataset))
                            if stat == 'RUN':
                                running[(strategy, dataset)] = 'RUN'
                            elif stat == 'PEND' and current_stat != 'RUN':
                                running[(strategy, dataset)] = 'PEND'
                            elif stat in ('DONE', 'EXIT') and current_stat not in ('RUN', 'PEND'):
                                running[(strategy, dataset)] = stat
        except Exception as e:
            print(f"Warning: Could not get jobs for user flag '{user_flag}': {e}")
    
    return running


def collect_status(domain_filter='clear_day', verbose=False):
    """Collect status for all strategy/dataset combinations."""
    running_jobs = get_running_jobs()
    
    status_matrix = {}
    summary = {
        'complete': 0,
        'running': 0,
        'pending': 0,
        'failed': 0,
        'skip': 0,
    }
    
    for strategy in ALL_STRATEGIES:
        status_matrix[strategy] = {}
        
        for dataset in DATASETS:
            status, path = check_weight_status(strategy, dataset, domain_filter)
            
            # Use LSF status to refine the status
            job_status = running_jobs.get((strategy, dataset))
            
            # IMPORTANT: If a job is currently running, mark as running regardless of existing weights
            # This ensures we track retraining jobs that override existing weights
            if job_status == 'RUN':
                status = 'running'
            elif job_status == 'PEND' and status != 'complete':
                status = 'pending'
            elif status == 'pending':
                if job_status == 'PEND':
                    status = 'pending'
                elif job_status in ('DONE', 'EXIT'):
                    # Job finished but weights are still missing -> failure
                    status = 'failed'
                else:
                    # No job in queue/history and no weights -> pending (not started)
                    status = 'pending'
            
            status_matrix[strategy][dataset] = {
                'status': status,
                'path': str(path) if path else None,
                'emoji': get_status_emoji(status),
            }
            
            summary[status] += 1
            
            if verbose:
                print(f"{get_status_emoji(status)} {strategy}/{dataset}")
    
    return status_matrix, summary


def format_status_row(strategy, status_dict, datasets):
    """Format a markdown table row for strategy status."""
    cells = [strategy]
    notes = []
    
    for dataset in datasets:
        data = status_dict.get(dataset, {'emoji': '❓'})
        cells.append(data['emoji'])
    
    # Add notes based on strategy
    if strategy == 'gen_Qwen_Image_Edit':
        notes.append('No BDD10k data')
    
    cells.append(' | '.join(notes) if notes else '')
    
    return '| ' + ' | '.join(cells) + ' |'


def update_tracker(status_matrix, summary, domain_filter='clear_day'):
    """Update the tracker markdown file."""
    # Read current tracker
    with open(TRACKER_PATH, 'r') as f:
        content = f.read()
    
    # Update timestamp
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    content = re.sub(
        r'\*\*Last Updated:\*\* .+',
        f'**Last Updated:** {now}',
        content
    )
    
    # Build new generative strategies table
    gen_rows = ['| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Notes |',
                '|----------|--------|--------|-----------------|------------|-------|']
    for strategy in GENERATIVE_STRATEGIES:
        gen_rows.append(format_status_row(strategy, status_matrix[strategy], DATASETS))
    gen_table = '\n'.join(gen_rows)
    
    # Build new standard strategies table
    std_rows = ['| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k | Notes |',
                '|----------|--------|--------|-----------------|------------|-------|']
    for strategy in STANDARD_STRATEGIES:
        std_rows.append(format_status_row(strategy, status_matrix[strategy], DATASETS))
    std_table = '\n'.join(std_rows)
    
    # Update generative table
    gen_pattern = r'### Generative Image Augmentation Strategies\n\n\|[^\n]+\n\|[-|\s]+\n(?:\|[^\n]+\n)+'
    gen_replacement = f'### Generative Image Augmentation Strategies\n\n{gen_table}\n'
    content = re.sub(gen_pattern, gen_replacement, content)
    
    # Update standard table
    std_pattern = r'### Standard Augmentation Strategies\n\n\|[^\n]+\n\|[-|\s]+\n(?:\|[^\n]+\n)+'
    std_replacement = f'### Standard Augmentation Strategies\n\n{std_table}\n'
    content = re.sub(std_pattern, std_replacement, content)
    
    # Update progress summary
    gen_complete = sum(1 for s in GENERATIVE_STRATEGIES 
                       for d in DATASETS 
                       if status_matrix[s][d]['status'] == 'complete')
    gen_running = sum(1 for s in GENERATIVE_STRATEGIES 
                      for d in DATASETS 
                      if status_matrix[s][d]['status'] == 'running')
    gen_pending = sum(1 for s in GENERATIVE_STRATEGIES 
                      for d in DATASETS 
                      if status_matrix[s][d]['status'] == 'pending')
    gen_failed = sum(1 for s in GENERATIVE_STRATEGIES 
                     for d in DATASETS 
                     if status_matrix[s][d]['status'] == 'failed')
    gen_total = len(GENERATIVE_STRATEGIES) * len(DATASETS) - 1  # -1 for Qwen/BDD10k
    
    std_complete = sum(1 for s in STANDARD_STRATEGIES 
                       for d in DATASETS 
                       if status_matrix[s][d]['status'] == 'complete')
    std_running = sum(1 for s in STANDARD_STRATEGIES 
                      for d in DATASETS 
                      if status_matrix[s][d]['status'] == 'running')
    std_pending = sum(1 for s in STANDARD_STRATEGIES 
                      for d in DATASETS 
                      if status_matrix[s][d]['status'] == 'pending')
    std_failed = sum(1 for s in STANDARD_STRATEGIES 
                     for d in DATASETS 
                     if status_matrix[s][d]['status'] == 'failed')
    std_total = len(STANDARD_STRATEGIES) * len(DATASETS)
    
    total_complete = gen_complete + std_complete
    total_running = gen_running + std_running
    total_pending = gen_pending + std_pending
    total_failed = gen_failed + std_failed
    total = gen_total + std_total
    
    progress_table = f"""| Category | Total | Complete | Running | Pending | Failed |
|----------|-------|----------|---------|---------|--------|
| **Generative (gen_*)** | {gen_total} | {gen_complete} | {gen_running} | {gen_pending} | {gen_failed} |
| **Standard (std_*)** | {std_total} | {std_complete} | {std_running} | {std_pending} | {std_failed} |
| **TOTAL** | {total} | {total_complete} | {total_running} | {total_pending} | {total_failed} |"""
    
    progress_pattern = r'\| Category \| Total \| Complete \| Running \| Pending \| Failed \|\n\|[-|\s]+\n(?:\|[^\n]+\n)+'
    content = re.sub(progress_pattern, progress_table + '\n', content)
    
    # Write updated tracker
    with open(TRACKER_PATH, 'w') as f:
        f.write(content)
    
    return {
        'gen': {'total': gen_total, 'complete': gen_complete, 'running': gen_running, 
                'pending': gen_pending, 'failed': gen_failed},
        'std': {'total': std_total, 'complete': std_complete, 'running': std_running,
                'pending': std_pending, 'failed': std_failed},
        'total': {'total': total, 'complete': total_complete, 'running': total_running,
                  'pending': total_pending, 'failed': total_failed},
    }


def print_summary(stats):
    """Print a summary of the status."""
    print("\n" + "="*60)
    print("RETRAINING PROGRESS SUMMARY")
    print("="*60)
    
    for category, data in stats.items():
        name = {'gen': 'Generative', 'std': 'Standard', 'total': 'TOTAL'}[category]
        pct = (data['complete'] / data['total'] * 100) if data['total'] > 0 else 0
        print(f"\n{name} Strategies:")
        print(f"  Complete: {data['complete']}/{data['total']} ({pct:.1f}%)")
        print(f"  Running:  {data['running']}")
        print(f"  Pending:  {data['pending']}")
        print(f"  Failed:   {data['failed']}")


def main():
    parser = argparse.ArgumentParser(description='Update retraining progress tracker')
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2],
                        help='Stage to check (1=clear_day, 2=all_domains)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed status for all combinations')
    parser.add_argument('--no-update', action='store_true',
                        help='Only show status without updating tracker')
    args = parser.parse_args()
    
    domain_filter = 'clear_day' if args.stage == 1 else 'all_domains'
    
    print(f"Checking Stage {args.stage} ({domain_filter}) status...")
    status_matrix, summary = collect_status(domain_filter, args.verbose)
    
    if not args.no_update:
        print(f"\nUpdating tracker: {TRACKER_PATH}")
        stats = update_tracker(status_matrix, summary, domain_filter)
        print_summary(stats)
        print(f"\nTracker updated: {TRACKER_PATH}")
    else:
        print("\nStatus collected (no update mode)")
        print(f"  Complete: {summary['complete']}")
        print(f"  Running:  {summary['running']}")
        print(f"  Pending:  {summary['pending']}")
        print(f"  Failed:   {summary['failed']}")


if __name__ == '__main__':
    main()
