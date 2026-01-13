#!/usr/bin/env python3
"""
Update the retraining progress tracker based on current weights.

Usage:
    python scripts/update_retraining_tracker.py
    python scripts/update_retraining_tracker.py --stage 1  # Only check stage 1 (clear_day)
    python scripts/update_retraining_tracker.py --verbose   # Show all status details
    python scripts/update_retraining_tracker.py --coverage-report  # Generate detailed coverage report
    python scripts/update_retraining_tracker.py -c -o coverage.md  # Save report to custom file
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

# Skip combinations - UPDATED: All strategies now have full 4/4 coverage after manifest regeneration
SKIP_COMBOS = set()  # No more skip combos needed

# All model variants for detailed coverage report
ALL_MODELS = [
    'deeplabv3plus_r50',
    'deeplabv3plus_r50_ratio0p50',
    'pspnet_r50',
    'pspnet_r50_ratio0p50',
    'segformer_mit-b5',
    'segformer_mit-b5_ratio0p50',
]

# Model display names
MODEL_DISPLAY = {
    'deeplabv3plus_r50': 'DeepLabV3+',
    'deeplabv3plus_r50_ratio0p50': 'DeepLabV3+ (0.5)',
    'pspnet_r50': 'PSPNet',
    'pspnet_r50_ratio0p50': 'PSPNet (0.5)',
    'segformer_mit-b5': 'SegFormer',
    'segformer_mit-b5_ratio0p50': 'SegFormer (0.5)',
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
    """Check if weights exist and are valid. Also checks for training lock files and detailed results."""
    # Determine model directory based on strategy type
    if strategy == 'baseline':
        model_dir = MODELS[domain_filter]['baseline']
    elif strategy.startswith('gen_'):
        model_dir = MODELS[domain_filter]['gen']
    else:
        model_dir = MODELS[domain_filter]['std']
    
    # Check if this is a skip combo
    if (strategy, dataset) in SKIP_COMBOS:
        return 'skip', False, None
    
    # Build dataset directory with domain suffix
    domain_abbrev = {'clear_day': 'cd', 'clear_night': 'cn', 'rainy_day': 'rd', 'rainy_night': 'rn', 'fog': 'fg', 'snow': 'sn'}
    domain_suffix = f'_{domain_abbrev.get(domain_filter, domain_filter[:2])}' if domain_filter else ''
    dataset_dir = f'{dataset}{domain_suffix}'
    
    weights_path = WEIGHTS_ROOT / strategy / dataset_dir / model_dir
    checkpoint = weights_path / "iter_80000.pth"
    lock_file = weights_path / ".training_lock"
    
    has_detailed = False
    detailed_dir = weights_path / "test_results_detailed"
    if detailed_dir.exists():
        # Check if there is any results.json in subdirectories
        for sub in detailed_dir.iterdir():
            if sub.is_dir() and (sub / "results.json").exists():
                has_detailed = True
                break

    try:
        # First check if training is in progress via lock file
        if lock_file.exists():
            return 'running', has_detailed, weights_path
        
        if checkpoint.exists():
            # Check if it's a valid file (not empty)
            if checkpoint.stat().st_size > 1000:  # At least 1KB
                return 'complete', has_detailed, weights_path
            else:
                return 'failed', has_detailed, weights_path
    except PermissionError:
        try:
            if weights_path.exists():
                 # Check if the file name is visible
                 files = os.listdir(weights_path)
                 if ".training_lock" in files:
                     return 'running', has_detailed, weights_path
                 if "iter_80000.pth" in files:
                     return 'complete', has_detailed, weights_path
        except:
            pass
        return 'pending', has_detailed, weights_path
    
    return 'pending', has_detailed, weights_path


def get_running_jobs():
    """Get list of currently running train/retrain jobs from both users."""
    running = {}  # {(strategy, dataset): status}
    
    # Known datasets for matching
    known_datasets = ['bdd10k', 'idd-aw', 'mapillaryvistas', 'outside15k']
    # Also check for _cd suffix variants
    known_datasets_cd = ['bdd10k_cd', 'iddaw_cd', 'mapillaryvistas_cd', 'outside15k_cd']
    # Model suffixes that might appear at the end of job names
    model_suffixes = ['dlv3p', 'pspn', 'segf', 'deeplabv3plus', 'pspnet', 'segformer']
    
    # Check jobs from multiple users
    users_to_check = ['', '-u chge7185']  # '' means current user
    
    # Job name prefixes to check
    job_prefixes = ['retrain_', 'train_']
    
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
                    
                    # Check for both train_ and retrain_ prefixes
                    job_part = None
                    for prefix in job_prefixes:
                        if job_name.startswith(prefix):
                            job_part = job_name[len(prefix):]
                            break
                    
                    if job_part is None:
                        continue  # Not a training job
                    
                    # Parse job name format: <prefix><strategy>_<dataset>[_<model_suffix>]
                    # Find the dataset by looking for known dataset names
                    dataset = None
                    strategy = None
                    
                    # First try to match _cd suffix datasets (from new train_ jobs)
                    for ds_cd, ds in zip(known_datasets_cd, known_datasets):
                        if ds_cd in job_part:
                            dataset = ds
                            # Extract strategy (everything before dataset)
                            idx = job_part.find(ds_cd)
                            strategy = job_part[:idx].rstrip('_')
                            break
                    
                    # If not found, try original dataset matching
                    if dataset is None:
                        for ds in known_datasets:
                            # Check for exact match (with underscores around it or at end)
                            ds_pattern = f'_{ds}_'
                            ds_pattern_end = f'_{ds}'
                            
                            if ds_pattern in job_part or job_part.endswith(ds_pattern_end):
                                dataset = ds
                                # Extract strategy (everything before dataset)
                                if ds_pattern in job_part:
                                    strategy = job_part.split(ds_pattern)[0]
                                else:
                                    # Dataset is at the end
                                    idx = job_part.rfind(f'_{ds}')
                                    strategy = job_part[:idx]
                                break
                    
                    if dataset and strategy:
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
            status, has_detailed, path = check_weight_status(strategy, dataset, domain_filter)
            
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
                'has_detailed': has_detailed,
                'path': str(path) if path else None,
                'emoji': get_status_emoji(status),
            }
            
            summary[status] += 1
            
            if verbose:
                print(f"{get_status_emoji(status)} {strategy}/{dataset} (Detailed: {has_detailed})")
    
    return status_matrix, summary


def check_model_weight_status(strategy, dataset, model, domain_filter='clear_day'):
    """Check weight status for a specific model variant."""
    # Build dataset directory with domain suffix
    domain_abbrev = {'clear_day': 'cd', 'clear_night': 'cn', 'rainy_day': 'rd', 'rainy_night': 'rn', 'fog': 'fg', 'snow': 'sn'}
    domain_suffix = f'_{domain_abbrev.get(domain_filter, domain_filter[:2])}' if domain_filter else ''
    dataset_dir = f'{dataset}{domain_suffix}'
    
    weights_path = WEIGHTS_ROOT / strategy / dataset_dir / model
    checkpoint = weights_path / "iter_80000.pth"
    lock_file = weights_path / ".training_lock"
    
    try:
        # Check if training is in progress via lock file
        if lock_file.exists():
            return 'running', weights_path
        
        if checkpoint.exists():
            if checkpoint.stat().st_size > 1000:  # At least 1KB
                return 'complete', weights_path
            else:
                return 'failed', weights_path
    except PermissionError:
        try:
            if weights_path.exists():
                files = os.listdir(weights_path)
                if ".training_lock" in files:
                    return 'running', weights_path
                if "iter_80000.pth" in files:
                    return 'complete', weights_path
        except:
            pass
        return 'pending', weights_path
    
    return 'pending', weights_path


def get_running_jobs_detailed():
    """Get detailed list of running jobs including model info."""
    jobs = {}  # {(strategy, dataset, model_type): status}
    
    # Model suffix patterns
    model_patterns = {
        'dlv3p': 'deeplabv3plus',
        'pspn': 'pspnet',
        'segf': 'segformer',
        'deeplabv3plus': 'deeplabv3plus',
        'pspnet': 'pspnet',
        'segformer': 'segformer',
    }
    
    known_datasets = ['bdd10k', 'idd-aw', 'mapillaryvistas', 'outside15k']
    known_datasets_cd = ['bdd10k_cd', 'iddaw_cd', 'mapillaryvistas_cd', 'outside15k_cd']
    job_prefixes = ['retrain_', 'train_', 'fix_']
    
    users_to_check = ['', '-u chge7185']
    
    for user_flag in users_to_check:
        try:
            cmd = ['bjobs', '-a', '-o', 'JOBID JOB_NAME STAT']
            if user_flag:
                cmd = ['bjobs'] + user_flag.split() + ['-a', '-o', 'JOBID JOB_NAME STAT']
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            for line in result.stdout.strip().split('\n')[1:]:
                parts = line.split()
                if len(parts) >= 3:
                    job_name = parts[1]
                    stat = parts[2]
                    
                    job_part = None
                    for prefix in job_prefixes:
                        if job_name.startswith(prefix):
                            job_part = job_name[len(prefix):]
                            break
                    
                    if job_part is None:
                        continue
                    
                    # Find dataset
                    dataset = None
                    strategy = None
                    model_type = None
                    
                    # Match _cd suffix datasets first
                    for ds_cd, ds in zip(known_datasets_cd, known_datasets):
                        if ds_cd in job_part:
                            dataset = ds
                            idx = job_part.find(ds_cd)
                            strategy = job_part[:idx].rstrip('_')
                            remaining = job_part[idx + len(ds_cd):]
                            break
                    
                    # Original matching
                    if dataset is None:
                        for ds in known_datasets:
                            ds_pattern = f'_{ds}_'
                            ds_pattern_end = f'_{ds}'
                            if ds_pattern in job_part:
                                dataset = ds
                                strategy = job_part.split(ds_pattern)[0]
                                remaining = job_part.split(ds_pattern)[1] if ds_pattern in job_part else ''
                                break
                            elif job_part.endswith(ds_pattern_end):
                                dataset = ds
                                idx = job_part.rfind(ds_pattern_end)
                                strategy = job_part[:idx]
                                remaining = ''
                                break
                    
                    if dataset and strategy:
                        # Try to extract model type from remaining part
                        remaining = remaining.lstrip('_') if remaining else ''
                        for pattern, model_name in model_patterns.items():
                            if pattern in remaining.lower():
                                model_type = model_name
                                break
                        
                        key = (strategy, dataset, model_type)
                        current = jobs.get(key)
                        if stat == 'RUN':
                            jobs[key] = 'RUN'
                        elif stat == 'PEND' and current != 'RUN':
                            jobs[key] = 'PEND'
        except Exception as e:
            pass
    
    return jobs


def collect_detailed_coverage(domain_filter='clear_day', verbose=False):
    """Collect detailed coverage showing each (strategy, dataset, model) combination."""
    running_jobs = get_running_jobs_detailed()
    
    coverage = []  # List of dicts with strategy, dataset, model, status info
    summary = {'complete': 0, 'running': 0, 'pending': 0, 'failed': 0}
    
    for strategy in ALL_STRATEGIES:
        # Determine which models to check based on strategy type
        if strategy.startswith('gen_'):
            models_to_check = ['deeplabv3plus_r50_ratio0p50', 'pspnet_r50_ratio0p50', 'segformer_mit-b5_ratio0p50']
        else:
            models_to_check = ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5']
        
        for dataset in DATASETS:
            for model in models_to_check:
                status, path = check_model_weight_status(strategy, dataset, model, domain_filter)
                
                # Check job status for refinement
                model_type = 'deeplabv3plus' if 'deeplabv3plus' in model else ('pspnet' if 'pspnet' in model else 'segformer')
                job_key = (strategy, dataset, model_type)
                job_status = running_jobs.get(job_key)
                
                # Also check without model type
                job_key_no_model = (strategy, dataset, None)
                job_status_no_model = running_jobs.get(job_key_no_model)
                
                if job_status == 'RUN' or job_status_no_model == 'RUN':
                    status = 'running'
                elif (job_status == 'PEND' or job_status_no_model == 'PEND') and status != 'complete':
                    status = 'pending'
                
                coverage.append({
                    'strategy': strategy,
                    'dataset': dataset,
                    'model': model,
                    'model_short': MODEL_DISPLAY.get(model, model),
                    'status': status,
                    'emoji': get_status_emoji(status),
                    'path': str(path) if path else None,
                })
                
                summary[status] += 1
                
                if verbose:
                    print(f"{get_status_emoji(status)} {strategy}/{dataset}/{model}")
    
    return coverage, summary


def generate_coverage_report(coverage, summary, output_file=None):
    """Generate a detailed coverage report."""
    lines = []
    lines.append("# Training Coverage Report")
    lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    
    # Summary
    total = sum(summary.values())
    lines.append("## Summary\n")
    lines.append(f"| Status | Count | Percentage |")
    lines.append(f"|--------|------:|----------:|")
    lines.append(f"| ✅ Complete | {summary['complete']} | {summary['complete']/total*100:.1f}% |")
    lines.append(f"| 🔄 Running | {summary['running']} | {summary['running']/total*100:.1f}% |")
    lines.append(f"| ⏳ Pending | {summary['pending']} | {summary['pending']/total*100:.1f}% |")
    lines.append(f"| ❌ Failed | {summary['failed']} | {summary['failed']/total*100:.1f}% |")
    lines.append(f"| **Total** | **{total}** | **100%** |")
    
    # Per-dataset breakdown
    lines.append("\n## Per-Dataset Breakdown\n")
    for dataset in DATASETS:
        dataset_items = [c for c in coverage if c['dataset'] == dataset]
        complete = sum(1 for c in dataset_items if c['status'] == 'complete')
        running = sum(1 for c in dataset_items if c['status'] == 'running')
        pending = sum(1 for c in dataset_items if c['status'] == 'pending')
        failed = sum(1 for c in dataset_items if c['status'] == 'failed')
        total_ds = len(dataset_items)
        
        lines.append(f"### {dataset.upper()}")
        lines.append(f"- Complete: {complete}/{total_ds} ({complete/total_ds*100:.1f}%)")
        lines.append(f"- Running: {running}")
        lines.append(f"- Pending: {pending}")
        lines.append(f"- Failed: {failed}\n")
    
    # Detailed tables by status
    lines.append("\n## Running Configurations\n")
    running_items = [c for c in coverage if c['status'] == 'running']
    if running_items:
        lines.append("| Strategy | Dataset | Model |")
        lines.append("|----------|---------|-------|")
        for item in sorted(running_items, key=lambda x: (x['strategy'], x['dataset'], x['model'])):
            lines.append(f"| {item['strategy']} | {item['dataset']} | {item['model_short']} |")
    else:
        lines.append("*No configurations currently running.*\n")
    
    lines.append("\n## Pending Configurations\n")
    pending_items = [c for c in coverage if c['status'] == 'pending']
    if pending_items:
        lines.append("| Strategy | Dataset | Model |")
        lines.append("|----------|---------|-------|")
        for item in sorted(pending_items, key=lambda x: (x['strategy'], x['dataset'], x['model'])):
            lines.append(f"| {item['strategy']} | {item['dataset']} | {item['model_short']} |")
    else:
        lines.append("*No pending configurations.*\n")
    
    lines.append("\n## Failed Configurations\n")
    failed_items = [c for c in coverage if c['status'] == 'failed']
    if failed_items:
        lines.append("| Strategy | Dataset | Model | Path |")
        lines.append("|----------|---------|-------|------|")
        for item in sorted(failed_items, key=lambda x: (x['strategy'], x['dataset'], x['model'])):
            lines.append(f"| {item['strategy']} | {item['dataset']} | {item['model_short']} | `{item['path']}` |")
    else:
        lines.append("*No failed configurations.*\n")
    
    lines.append("\n## Complete Configurations\n")
    complete_items = [c for c in coverage if c['status'] == 'complete']
    if complete_items:
        # Group by strategy for readability
        by_strategy = defaultdict(list)
        for item in complete_items:
            by_strategy[item['strategy']].append(item)
        
        lines.append("| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k |")
        lines.append("|----------|--------|--------|-----------------|------------|")
        
        for strategy in ALL_STRATEGIES:
            items = by_strategy.get(strategy, [])
            cells = [strategy]
            for dataset in DATASETS:
                ds_items = [i for i in items if i['dataset'] == dataset]
                if ds_items:
                    models = ', '.join(sorted(set(i['model_short'].replace(' (0.5)', '') for i in ds_items)))
                    cells.append(f"✅ {models}")
                else:
                    cells.append("⏳")
            lines.append("| " + " | ".join(cells) + " |")
    else:
        lines.append("*No complete configurations.*\n")
    
    content = '\n'.join(lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(content)
        print(f"Coverage report saved to: {output_file}")
    
    return content


def format_status_row(strategy, status_dict, datasets):
    """Format a markdown table row for strategy status."""
    cells = [strategy]
    
    for dataset in datasets:
        data = status_dict.get(dataset, {'emoji': '❓', 'has_detailed': False})
        cell_content = data['emoji']
        if data.get('has_detailed'):
            cell_content += ' 🎯'
        cells.append(cell_content)
    
    # Add notes based on strategy
    notes = []
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
    parser.add_argument('--coverage-report', '-c', action='store_true',
                        help='Generate detailed coverage report showing each (strategy, dataset, model)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file for coverage report (default: prints to stdout or saves to docs/TRAINING_COVERAGE.md)')
    args = parser.parse_args()
    
    domain_filter = 'clear_day' if args.stage == 1 else 'all_domains'
    
    # Coverage report mode
    if args.coverage_report:
        print(f"Generating detailed coverage report for Stage {args.stage} ({domain_filter})...")
        coverage, summary = collect_detailed_coverage(domain_filter, args.verbose)
        
        output_file = args.output
        if output_file is None:
            output_file = PROJECT_ROOT / 'docs' / 'TRAINING_COVERAGE.md'
        
        report = generate_coverage_report(coverage, summary, output_file)
        
        # Print summary
        total = sum(summary.values())
        print(f"\n{'='*60}")
        print("DETAILED COVERAGE SUMMARY")
        print(f"{'='*60}")
        print(f"Total configurations: {total}")
        print(f"  ✅ Complete: {summary['complete']} ({summary['complete']/total*100:.1f}%)")
        print(f"  🔄 Running:  {summary['running']} ({summary['running']/total*100:.1f}%)")
        print(f"  ⏳ Pending:  {summary['pending']} ({summary['pending']/total*100:.1f}%)")
        print(f"  ❌ Failed:   {summary['failed']} ({summary['failed']/total*100:.1f}%)")
        return
    
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
