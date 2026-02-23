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
WEIGHTS_ROOT = Path(os.environ.get('PROVE_WEIGHTS_ROOT', '${AWARE_DATA_ROOT}/WEIGHTS'))
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
    'std_std_photometric_distort',
    'std_minimal',  # Minimal augmentation baseline (RandomCrop, RandomFlip, 1x PhotoMetricDistortion)
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
        'missing': '⚠️',
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
    
    # Try alternate name for idd-aw
    dirs_to_try = [dataset_dir]
    if dataset == 'idd-aw':
        dirs_to_try.append(f'iddaw{domain_suffix}')
    
    best_status = 'pending'
    best_path = None
    has_detailed = False
    
    for d in dirs_to_try:
        weights_path = WEIGHTS_ROOT / strategy / d / model_dir
        checkpoint = weights_path / "iter_80000.pth"
        lock_file = weights_path / ".training_lock"
        
        current_has_detailed = False
        detailed_dir = weights_path / "test_results_detailed"
        if detailed_dir.exists():
            for sub in detailed_dir.iterdir():
                if sub.is_dir() and (sub / "results.json").exists():
                    current_has_detailed = True
                    break
        
        has_detailed = has_detailed or current_has_detailed
        
        status = 'pending'
        try:
            if lock_file.exists():
                status = 'running'
            elif checkpoint.exists():
                if checkpoint.stat().st_size > 1000:
                    status = 'complete'
                else:
                    status = 'failed'
        except PermissionError:
            try:
                if weights_path.exists():
                    files = os.listdir(weights_path)
                    if ".training_lock" in files: status = 'running'
                    elif "iter_80000.pth" in files: status = 'complete'
            except: pass
            
        if status == 'complete':
            return 'complete', has_detailed, weights_path
        if status == 'running':
            best_status = 'running'
            best_path = weights_path
        elif status == 'failed' and best_status == 'pending':
            best_status = 'failed'
            best_path = weights_path
        elif best_path is None:
            best_path = weights_path
            
    return best_status, has_detailed, best_path


def get_running_jobs():
    """Get list of currently running train/retrain jobs from both users.
    
    Only tracks Stage 1 (clear_day) jobs with _cd suffix.
    Excludes Stage 2 jobs (s2_ prefix) and all_domains jobs (_ad suffix).
    """
    running = {}  # {(strategy, dataset): status}
    
    # Known datasets for matching
    known_datasets = ['bdd10k', 'idd-aw', 'mapillaryvistas', 'outside15k']
    # Also check for _cd suffix variants
    known_datasets_cd = ['bdd10k_cd', 'idd-aw_cd', 'mapillaryvistas_cd', 'outside15k_cd']
    # Model suffixes that might appear at the end of job names
    model_suffixes = ['dlv3p', 'pspn', 'segf', 'deeplabv3plus', 'pspnet', 'segformer']
    
    # Check jobs from multiple users
    users_to_check = ['-u ${USER}', '-u chge7185']  # '' means current user
    
    # Job name prefixes to check for Stage 1
    job_prefixes = ['retrain_', 'train_', 'fix_', 'rt3_', 'rt4_']
    # Skip prefixes for other job types
    skip_prefixes = ['s2_', 'ratio_', 'retest_']
    
    # Mapping from abbreviated job names to full strategy names (for rt3_/rt4_ jobs)
    abbrev_to_strategy = {
        'AttrHall': 'gen_Attribute_Hallucination',
        'baseline': 'baseline',
        'CNetSeg': 'gen_CNetSeg',
        'CUT': 'gen_CUT',
        'IP2P': 'gen_IP2P',
        'autoaug': 'std_autoaugment',
        'cutmix': 'std_cutmix',
        'mixup': 'std_mixup',
        'photom': 'std_std_photometric_distort',
        'randaug': 'std_randaugment',
    }
    
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
                    
                    # Skip non-Stage-1 jobs
                    if any(job_name.startswith(skip) for skip in skip_prefixes):
                        continue
                    
                    # Skip all_domains jobs (_ad suffix)
                    if '_ad' in job_name:
                        continue
                    
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
                    
                    if dataset and strategy and strategy in ALL_STRATEGIES:
                        # Store the most "active" status for this config
                        current_stat = running.get((strategy, dataset))
                        if stat == 'RUN':
                            running[(strategy, dataset)] = 'RUN'
                        elif stat == 'PEND' and current_stat != 'RUN':
                            running[(strategy, dataset)] = 'PEND'
                        elif stat in ('DONE', 'EXIT') and current_stat not in ('RUN', 'PEND'):
                            running[(strategy, dataset)] = stat
                        continue
                    
                    # Handle rt3_/rt4_ job format: <abbrev>_iddaw_<model>
                    # Example: rt4_randaug_iddaw_psp -> std_randaugment / idd-aw
                    if any(job_name.startswith(p) for p in ['rt3_', 'rt4_']):
                        parts = job_part.split('_')
                        if len(parts) >= 2:
                            abbrev = parts[0]
                            # Check for iddaw in second or third position
                            if 'iddaw' in parts:
                                dataset = 'idd-aw'
                                if abbrev in abbrev_to_strategy:
                                    strategy = abbrev_to_strategy[abbrev]
                                    current_stat = running.get((strategy, dataset))
                                    if stat == 'RUN':
                                        running[(strategy, dataset)] = 'RUN'
                                    elif stat == 'PEND' and current_stat != 'RUN':
                                        running[(strategy, dataset)] = 'PEND'
                                    elif stat in ('DONE', 'EXIT') and current_stat not in ('RUN', 'PEND'):
                                        running[(strategy, dataset)] = stat
                                    continue
                        continue

                    # If still not found, try fuzzy/truncated matching for rt_ jobs
                    ds_trunc_map = {'bdd10k': 'bdd10k', 'iddaw_': 'idd-aw', 'mapill': 'mapillaryvistas', 'outsid': 'outside15k'}
                    for trunc_ds, full_ds in ds_trunc_map.items():
                        if f'_{trunc_ds}_' in job_part or job_part.endswith(f'_{trunc_ds}') or f'_{trunc_ds}_' in job_name:
                            dataset = full_ds
                            if f'_{trunc_ds}_' in job_part:
                                strategy_part = job_part.split(f'_{trunc_ds}_')[0]
                            elif job_part.endswith(f'_{trunc_ds}'):
                                strategy_part = job_part[:-len(f'_{trunc_ds}')]
                            else: # Found in job_name but maybe not clearly in job_part
                                idx = job_part.find(trunc_ds)
                                strategy_part = job_part[:idx].rstrip('_')
                            
                            # Find matching full strategy
                            for full_s in ALL_STRATEGIES:
                                if (len(strategy_part) >= 10 and full_s.startswith(strategy_part)) or \
                                   (len(strategy_part) >= 10 and strategy_part.startswith(full_s[:15])):
                                    strategy = full_s
                                    break
                            
                            if dataset and strategy:
                                print(f"DEBUG Match (jobs): {job_name} -> {strategy} / {dataset}")
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
            
            # Use LSF status to refine the status - but ONLY if checkpoint doesn't exist
            # Jobs that train multiple models will skip existing checkpoints via pre-flight checks
            job_status = running_jobs.get((strategy, dataset))
            
            if status != 'complete':
                # Only consider job status if checkpoint doesn't exist
                if job_status == 'RUN':
                    status = 'running'
                elif job_status == 'PEND':
                    status = 'pending'
                elif status == 'pending':
                    if job_status in ('DONE', 'EXIT'):
                        # Job finished but weights are still missing -> failure
                        status = 'failed'
                    # else: No job in queue/history and no weights -> pending (not started)
            
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
    
    # Try alternate name for idd-aw (iddaw)
    dirs_to_try = [dataset_dir]
    if dataset == 'idd-aw':
        dirs_to_try.append(f'iddaw{domain_suffix}')
        
    for d in dirs_to_try:
        weights_path = WEIGHTS_ROOT / strategy / d / model
        checkpoint = weights_path / "iter_80000.pth"
        lock_file = weights_path / ".training_lock"
        
        try:
            if lock_file.exists():
                return 'running', weights_path
            
            if checkpoint.exists():
                if checkpoint.stat().st_size > 1000:
                    return 'complete', weights_path
                else:
                    return 'failed', weights_path
        except PermissionError:
            try:
                if weights_path.exists():
                    files = os.listdir(weights_path)
                    if ".training_lock" in files: return 'running', weights_path
                    if "iter_80000.pth" in files: return 'complete', weights_path
            except: pass
            
    return 'pending', weights_path


def get_running_jobs_detailed(verbose=False):
    """Get detailed list of running jobs including model info and user.
    
    Only tracks Stage 1 (clear_day) jobs with _cd suffix.
    Excludes Stage 2 jobs (s2_ prefix) and all_domains jobs (_ad suffix).
    
    Returns:
        dict: {(strategy, dataset, model_type): {'status': str, 'user': str}}
    """
    jobs = {}  # {(strategy, dataset, model_type): {'status': str, 'user': str}}
    
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
    # Only match Stage 1 job prefixes (not s2_ or ratio ablation jobs)
    job_prefixes = ['rt_', 'retrain_', 'train_', 'fix_']
    # Skip prefixes for other job types
    skip_prefixes = ['s2_', 'ratio_', 'retest_']
    
    users_to_check = ['', '-u chge7185']
    
    for user_flag in users_to_check:
        try:
            cmd = ['bjobs', '-a', '-o', 'JOBID JOB_NAME STAT USER']
            if user_flag:
                cmd = ['bjobs'] + user_flag.split() + ['-a', '-o', 'JOBID JOB_NAME STAT USER']
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            for line in result.stdout.strip().split('\n')[1:]:
                parts = line.split()
                if len(parts) >= 4:
                    job_name = parts[1]
                    stat = parts[2]
                    user = parts[3]
                    
                    if any(job_name.startswith(skip) for skip in skip_prefixes):
                        continue
                    
                    # Skip all_domains jobs (_ad suffix)
                    if '_ad' in job_name:
                        continue
                    
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
                    remaining = ''
                    
                    # 1. Fuzzy/truncated matching for rt_ jobs (highest priority for short names)
                    ds_trunc_map = {'bdd10k': 'bdd10k', 'iddaw_': 'idd-aw', 'mapill': 'mapillaryvistas', 'outsid': 'outside15k'}
                    for trunc_ds, full_ds in ds_trunc_map.items():
                        if f'_{trunc_ds}_' in job_part or job_part.endswith(f'_{trunc_ds}') or f'_{trunc_ds}_' in job_name:
                            dataset = full_ds
                            if f'_{trunc_ds}_' in job_part:
                                strategy_part = job_part.split(f'_{trunc_ds}_')[0]
                                remaining = job_part.split(f'_{trunc_ds}_')[1]
                            elif job_part.endswith(f'_{trunc_ds}'):
                                strategy_part = job_part[:-len(f'_{trunc_ds}')]
                                remaining = ''
                            else:
                                idx = job_part.find(trunc_ds)
                                strategy_part = job_part[:idx].rstrip('_')
                                remaining = job_part[idx + len(trunc_ds):]
                            
                            # Find matching full strategy
                            for full_s in ALL_STRATEGIES:
                                if full_s == strategy_part or \
                                   (len(strategy_part) >= 10 and (full_s.startswith(strategy_part) or strategy_part.startswith(full_s[:15]))):
                                    strategy = full_s
                                    break
                            break
                    
                    # 2. Match _cd suffix datasets (train_gen_xxx_dataset_cd format)
                    if dataset is None:
                        for ds_cd, ds in zip(known_datasets_cd, known_datasets):
                            if ds_cd in job_part:
                                dataset = ds
                                idx = job_part.find(ds_cd)
                                strategy = job_part[:idx].rstrip('_')
                                remaining = job_part[idx + len(ds_cd):]
                                break
                    
                    # 3. Match dataset at the end (fix_gen_xxx_model_dataset format)
                    if dataset is None:
                        for ds in known_datasets:
                            # Check if job_part ends with the dataset name
                            if job_part.endswith(f'_{ds}') or job_part.endswith(ds):
                                dataset = ds
                                # Find where the dataset starts
                                if job_part.endswith(f'_{ds}'):
                                    remaining_part = job_part[:-len(f'_{ds}')]
                                else:
                                    remaining_part = job_part[:-len(ds)].rstrip('_')
                                
                                # Now extract strategy and model from remaining_part
                                found_model = False
                                for model_suffix, model_name in model_patterns.items():
                                    if remaining_part.endswith(f'_{model_suffix}'):
                                        model_type = model_name
                                        strategy = remaining_part[:-len(f'_{model_suffix}')]
                                        found_model = True
                                        break
                                
                                if not found_model:
                                    strategy = remaining_part
                                break
                    
                    # 4. Original middle matching (strategy_dataset_xxx format)
                    if dataset is None:
                        for ds in known_datasets:
                            ds_pattern = f'_{ds}_'
                            if ds_pattern in job_part:
                                dataset = ds
                                strategy = job_part.split(ds_pattern)[0]
                                remaining = job_part.split(ds_pattern)[1]
                                break
                    
                        # Use ALL_STRATEGIES to validate strategy
                    if strategy and strategy not in ALL_STRATEGIES:
                        for full_s in ALL_STRATEGIES:
                            if (len(strategy) >= 10 and (full_s.startswith(strategy) or strategy.startswith(full_s[:15]))):
                                strategy = full_s
                                break

                    if dataset and strategy:
                        # Try to extract model type from remaining part
                        remaining = remaining.lstrip('_') if remaining else ''
                        if not model_type:
                            for pattern, model_name in model_patterns.items():
                                if pattern in remaining.lower():
                                    model_type = model_name
                                    break
                        
                        key = (strategy, dataset, model_type)
                        current = jobs.get(key)
                        current_status = current['status'] if current else None
                        if stat == 'RUN':
                            jobs[key] = {'status': 'RUN', 'user': user}
                        elif stat == 'PEND' and current_status != 'RUN':
                            jobs[key] = {'status': 'PEND', 'user': user}
        except Exception as e:
            pass
    
    return jobs


def collect_detailed_coverage(domain_filter='clear_day', verbose=False):
    """Collect detailed coverage showing each (strategy, dataset, model) combination.
    
    Status definitions:
    - complete: Checkpoint exists and is valid
    - running: Job is actively running (RUN status)
    - pending: Job is in queue waiting (PEND status)
    - missing: No checkpoint AND no job in queue - needs to be started
    - failed: Checkpoint exists but is invalid/corrupted
    """
    running_jobs = get_running_jobs_detailed(verbose)
    
    coverage = []  # List of dicts with strategy, dataset, model, status info
    summary = {'complete': 0, 'running': 0, 'pending': 0, 'missing': 0, 'failed': 0}
    
    for strategy in ALL_STRATEGIES:
        # Determine which models to check based on strategy type
        if strategy.startswith('gen_'):
            models_to_check = ['deeplabv3plus_r50_ratio0p50', 'pspnet_r50_ratio0p50', 'segformer_mit-b5_ratio0p50', 'segnext_mscan-b_ratio0p50']
        else:
            models_to_check = ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5', 'segnext_mscan-b']
        
        for dataset in DATASETS:
            for model in models_to_check:
                status, path = check_model_weight_status(strategy, dataset, model, domain_filter)
                user = None
                
                # Check job status for refinement - but ONLY if checkpoint doesn't already exist
                # Jobs that train multiple models will skip existing checkpoints via pre-flight checks
                if status != 'complete':
                    model_type = 'deeplabv3plus' if 'deeplabv3plus' in model else ('pspnet' if 'pspnet' in model else 'segformer')
                    job_key = (strategy, dataset, model_type)
                    job_info = running_jobs.get(job_key)
                    
                    # Also check without model type
                    job_key_no_model = (strategy, dataset, None)
                    job_info_no_model = running_jobs.get(job_key_no_model)
                    
                    # Determine status and user from job info
                    if job_info and job_info.get('status') == 'RUN':
                        status = 'running'
                        user = job_info.get('user')
                    elif job_info_no_model and job_info_no_model.get('status') == 'RUN':
                        status = 'running'
                        user = job_info_no_model.get('user')
                    elif job_info and job_info.get('status') == 'PEND':
                        status = 'pending'
                        user = job_info.get('user')
                    elif job_info_no_model and job_info_no_model.get('status') == 'PEND':
                        status = 'pending'
                        user = job_info_no_model.get('user')
                    else:
                        # No job in queue and no checkpoint - this is missing/not started
                        status = 'missing'
                
                coverage.append({
                    'strategy': strategy,
                    'dataset': dataset,
                    'model': model,
                    'model_short': MODEL_DISPLAY.get(model, model),
                    'status': status,
                    'emoji': get_status_emoji(status),
                    'path': str(path) if path else None,
                    'user': user,
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
    lines.append(f"| ⏳ Pending (in queue) | {summary['pending']} | {summary['pending']/total*100:.1f}% |")
    lines.append(f"| ⚠️ Missing (not started) | {summary['missing']} | {summary['missing']/total*100:.1f}% |")
    lines.append(f"| ❌ Failed | {summary['failed']} | {summary['failed']/total*100:.1f}% |")
    lines.append(f"| **Total** | **{total}** | **100%** |")
    
    # Per-dataset breakdown
    lines.append("\n## Per-Dataset Breakdown\n")
    for dataset in DATASETS:
        dataset_items = [c for c in coverage if c['dataset'] == dataset]
        complete = sum(1 for c in dataset_items if c['status'] == 'complete')
        running = sum(1 for c in dataset_items if c['status'] == 'running')
        pending = sum(1 for c in dataset_items if c['status'] == 'pending')
        missing = sum(1 for c in dataset_items if c['status'] == 'missing')
        failed = sum(1 for c in dataset_items if c['status'] == 'failed')
        total_ds = len(dataset_items)
        
        lines.append(f"### {dataset.upper()}")
        lines.append(f"- Complete: {complete}/{total_ds} ({complete/total_ds*100:.1f}%)")
        lines.append(f"- Running: {running}")
        lines.append(f"- Pending (in queue): {pending}")
        lines.append(f"- Missing (not started): {missing}")
        lines.append(f"- Failed: {failed}\n")
    
    # Detailed tables by status
    lines.append("\n## Running Configurations\n")
    running_items = [c for c in coverage if c['status'] == 'running']
    if running_items:
        lines.append("| Strategy | Dataset | Model | User |")
        lines.append("|----------|---------|-------|------|")
        for item in sorted(running_items, key=lambda x: (x['strategy'], x['dataset'], x['model'])):
            user = item.get('user', '-') or '-'
            lines.append(f"| {item['strategy']} | {item['dataset']} | {item['model_short']} | {user} |")
    else:
        lines.append("*No configurations currently running.*\n")
    
    lines.append("\n## Pending Configurations (in queue)\n")
    pending_items = [c for c in coverage if c['status'] == 'pending']
    if pending_items:
        lines.append("| Strategy | Dataset | Model | User |")
        lines.append("|----------|---------|-------|------|")
        for item in sorted(pending_items, key=lambda x: (x['strategy'], x['dataset'], x['model'])):
            user = item.get('user', '-') or '-'
            lines.append(f"| {item['strategy']} | {item['dataset']} | {item['model_short']} | {user} |")
    else:
        lines.append("*No configurations pending in queue.*\n")
    
    lines.append("\n## Missing Configurations (not started)\n")
    missing_items = [c for c in coverage if c['status'] == 'missing']
    if missing_items:
        lines.append("| Strategy | Dataset | Model |")
        lines.append("|----------|---------|-------|")
        for item in sorted(missing_items, key=lambda x: (x['strategy'], x['dataset'], x['model'])):
            lines.append(f"| {item['strategy']} | {item['dataset']} | {item['model_short']} |")
    else:
        lines.append("*No configurations missing.*\n")
    
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
        print(f"  ✅ Complete:           {summary['complete']} ({summary['complete']/total*100:.1f}%)")
        print(f"  🔄 Running:            {summary['running']} ({summary['running']/total*100:.1f}%)")
        print(f"  ⏳ Pending (in queue): {summary['pending']} ({summary['pending']/total*100:.1f}%)")
        print(f"  ⚠️ Missing (not started): {summary['missing']} ({summary['missing']/total*100:.1f}%)")
        print(f"  ❌ Failed:             {summary['failed']} ({summary['failed']/total*100:.1f}%)")
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
