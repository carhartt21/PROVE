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
# Stage 1 (clear_day only): WEIGHTS
# Stage 2 (all domains): WEIGHTS_STAGE_2
WEIGHTS_ROOT = Path(os.environ.get('PROVE_WEIGHTS_ROOT', '${AWARE_DATA_ROOT}/WEIGHTS'))
WEIGHTS_ROOT_STAGE2 = Path(os.environ.get('PROVE_WEIGHTS_ROOT_STAGE2', '${AWARE_DATA_ROOT}/WEIGHTS_STAGE_2'))
WEIGHTS_ROOT_CITYSCAPES_GEN = Path('${AWARE_DATA_ROOT}/WEIGHTS_CITYSCAPES_GEN')
TRACKER_PATH = PROJECT_ROOT / 'docs' / 'TRAINING_TRACKER.md'
DATASETS = ['bdd10k', 'idd-aw', 'mapillaryvistas', 'outside15k']
DATASETS_DEFAULT = list(DATASETS)  # Save original for --stage all reset
DATASETS_CITYSCAPES_GEN = ['cityscapes']
DATASET_DISPLAY = {
    'bdd10k': 'BDD10k',
    'idd-aw': 'IDD-AW',
    'mapillaryvistas': 'MapillaryVistas',
    'outside15k': 'OUTSIDE15k',
    'cityscapes': 'Cityscapes',
}

# Models - now using all 4 models per strategy per dataset
# Domain filter is part of dataset directory, not model directory
# Stage 1 (clear_day): Uses 4 models: pspnet, segformer, segnext, mask2former
# Stage 2 (all_domains): Uses same 4 models
MODELS = {
    'clear_day': {
        # Stage 1 now uses 4 models (not 6, excluding deeplabv3plus_r50 and hrnet_hr48)
        'gen': ['pspnet_r50_ratio0p50', 'segformer_mit-b3_ratio0p50', 'segnext_mscan-b_ratio0p50', 'mask2former_swin-b_ratio0p50'],
        'std': ['pspnet_r50', 'segformer_mit-b3', 'segnext_mscan-b', 'mask2former_swin-b'],
        'baseline': ['pspnet_r50', 'segformer_mit-b3', 'segnext_mscan-b', 'mask2former_swin-b'],
    },
    'all_domains': {
        # Stage 2 uses same 4 models as Stage 1
        # Generative strategies use _ratio0p50 suffix (trained with 0.5 real/gen ratio)
        # Standard strategies use base model names (no ratio suffix)
        'gen': ['pspnet_r50_ratio0p50', 'segformer_mit-b3_ratio0p50', 'segnext_mscan-b_ratio0p50', 'mask2former_swin-b_ratio0p50'],
        'std': ['pspnet_r50', 'segformer_mit-b3', 'segnext_mscan-b', 'mask2former_swin-b'],
        'baseline': ['pspnet_r50', 'segformer_mit-b3', 'segnext_mscan-b', 'mask2former_swin-b'],
    },
    'cityscapes_gen': {
        # Cityscapes generative evaluation: 4 models
        'gen': ['pspnet_r50_ratio0p50', 'segformer_mit-b3_ratio0p50', 'segnext_mscan-b_ratio0p50', 'mask2former_swin-b_ratio0p50'],
        'std': ['pspnet_r50', 'segformer_mit-b3', 'segnext_mscan-b', 'mask2former_swin-b'],
        'baseline': ['pspnet_r50', 'segformer_mit-b3', 'segnext_mscan-b', 'mask2former_swin-b'],
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
    'std_minimal',
    'std_photometric_distort',
    'std_autoaugment',
    'std_cutmix',
    'std_mixup',
    'std_randaugment',
]

ALL_STRATEGIES = GENERATIVE_STRATEGIES + STANDARD_STRATEGIES

# Stage 2 excludes these strategies (not part of Stage 2 training plan)
STAGE2_EXCLUDED_STRATEGIES = {
    'std_cutmix',
    'std_mixup',
    'gen_cyclediffusion',
}

# Cityscapes-Gen excludes these strategies (no Cityscapes generated images or near-identical to baseline)
CG_EXCLUDED_STRATEGIES = {
    'gen_LANIT',              # No Cityscapes generated images exist
    'std_minimal',            # RandomCrop + RandomFlip only — essentially same as baseline
    'std_photometric_distort',  # Essentially same as baseline
}

# Skip combinations - UPDATED: All strategies now have full 4/4 coverage after manifest regeneration
SKIP_COMBOS = set()  # No more skip combos needed

# All model variants for detailed coverage report
ALL_MODELS = [
    'deeplabv3plus_r50',
    'deeplabv3plus_r50_ratio0p50',
    'pspnet_r50',
    'pspnet_r50_ratio0p50',
    'segformer_mit-b3',
    'segformer_mit-b3_ratio0p50',
    'segnext_mscan-b',
    'segnext_mscan-b_ratio0p50',
    'hrnet_hr48',
    'hrnet_hr48_ratio0p50',
    'mask2former_swin-b',
    'mask2former_swin-b_ratio0p50',
]

# Model display names
MODEL_DISPLAY = {
    'deeplabv3plus_r50': 'DeepLabV3+',
    'deeplabv3plus_r50_ratio0p50': 'DeepLabV3+ (0.5)',
    'pspnet_r50': 'PSPNet',
    'pspnet_r50_ratio0p50': 'PSPNet (0.5)',
    'segformer_mit-b3': 'SegFormer',
    'segformer_mit-b3_ratio0p50': 'SegFormer (0.5)',
    'segnext_mscan-b': 'SegNeXt',
    'segnext_mscan-b_ratio0p50': 'SegNeXt (0.5)',
    'hrnet_hr48': 'HRNet',
    'hrnet_hr48_ratio0p50': 'HRNet (0.5)',
    'mask2former_swin-b': 'Mask2Former',
    'mask2former_swin-b_ratio0p50': 'Mask2Former (0.5)',
}

# ============================================================================
# LSF Job Status Detection
# ============================================================================

# Map from short model names (used in LSF job names) to model directory prefixes
_MODEL_SHORT_TO_PREFIX = {
    'pspnet': 'pspnet_r50',
    'segformer': 'segformer_mit-b3',
    'segnext': 'segnext_mscan-b',
    'mask2former': 'mask2former_swin-b',
    'hrnet': 'hrnet_hr48',
    'deeplabv3plus': 'deeplabv3plus_r50',
}

# Dataset name normalization for job name parsing
_DATASET_PATTERNS = {
    'bdd10k': 'bdd10k',
    'iddaw': 'idd-aw',
    'mapillaryvistas': 'mapillaryvistas',
    'outside15k': 'outside15k',
    'cityscapes': 'cityscapes',
}


def get_training_jobs():
    """Query bjobs to get currently running/pending training jobs.

    Returns:
        dict: {(strategy, dataset, model_prefix): status} where status is 'RUN' or 'PEND'.
              - strategy: e.g. 'gen_cycleGAN', 'baseline'
              - dataset: e.g. 'bdd10k', 'idd-aw' (normalized with hyphen)
              - model_prefix: e.g. 'pspnet_r50', 'segformer_mit-b3'
    """
    jobs = {}
    try:
        result = subprocess.run(
            ['bjobs', '-u', os.environ.get('USER', '${USER}'), '-o', 'JOBID JOB_NAME STAT'],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in result.stdout.strip().split('\n')[1:]:  # Skip header
            parts = line.split()
            if len(parts) < 3:
                continue
            job_name = parts[1]
            stat = parts[2]
            if stat not in ('RUN', 'PEND'):
                continue

            # Determine stage prefix and strip it
            # Training jobs: s1_<strategy>_<dataset>_<model>, s2_..., cityscapes-gen_..., noise-ablation_...
            # Skip test/retest jobs: fg_, test_, rt_, retest_
            if any(job_name.startswith(p) for p in ('fg_', 'test_', 'rt_', 'retest_')):
                continue

            # Parse stage prefix
            stage_prefix = None
            remainder = None
            for prefix in ('s1_', 's2_', 'cityscapes-gen_', 'noise-ablation_', 'ratio_'):
                if job_name.startswith(prefix):
                    stage_prefix = prefix.rstrip('_')
                    remainder = job_name[len(prefix):]
                    break
            if stage_prefix is None or remainder is None:
                continue

            # Parse model (last segment after final underscore matching known model shorts)
            model_prefix = None
            for short_name, full_name in _MODEL_SHORT_TO_PREFIX.items():
                if remainder.endswith(f'_{short_name}'):
                    model_prefix = full_name
                    remainder = remainder[:-(len(short_name) + 1)]  # Strip _<model>
                    break
            if model_prefix is None:
                continue

            # Parse dataset (last segment of remainder)
            dataset = None
            for ds_short, ds_norm in _DATASET_PATTERNS.items():
                if remainder.endswith(f'_{ds_short}') or remainder == ds_short:
                    dataset = ds_norm
                    strategy = remainder[:-(len(ds_short) + 1)] if remainder.endswith(f'_{ds_short}') else ''
                    break
            if dataset is None:
                continue

            # The remaining string is the strategy
            if not strategy:
                continue

            key = (strategy, dataset, model_prefix)
            # RUN takes priority over PEND
            if stat == 'RUN' or key not in jobs:
                jobs[key] = stat

    except Exception as e:
        print(f"Warning: Could not query bjobs for training status: {e}")

    return jobs

# Global cache for training jobs (populated on first call)
_training_jobs_cache = None


def _get_cached_training_jobs():
    """Get training jobs with caching (queried once per tracker run)."""
    global _training_jobs_cache
    if _training_jobs_cache is None:
        _training_jobs_cache = get_training_jobs()
    return _training_jobs_cache


def get_status_emoji(status, model_info=None):
    """Convert status to emoji. For partial completion, show count."""
    base_emoji = {
        'complete': '✅',
        'partial': '🔶',  # Partial completion
        'running': '🔄',
        'pending': '⏳',
        'missing': '⚠️',
        'failed': '❌',
        'skip': '➖',
    }.get(status, '❓')
    
    # For partial status, append completion count
    if status == 'partial' and model_info:
        return f"{model_info['complete_count']}/{model_info['total_count']}"
    
    return base_emoji


# Default max_iters (new training regime uses 15k iterations)
DEFAULT_MAX_ITERS = 15000


def get_target_iterations(weights_path):
    """Read max_iters from training_config.py in the weights directory.
    
    Returns:
        int: Target iteration count from config, or DEFAULT_MAX_ITERS if not found
    """
    config_file = weights_path / "training_config.py"
    try:
        if config_file.exists():
            content = config_file.read_text()
            # Parse max_iters from both formats:
            #   train_cfg = dict(max_iters=15000, ...)  (keyword format)
            #   train_cfg = {'max_iters': 80000, ...}   (dict literal format)
            match = re.search(r"'?max_iters'?\s*[=:]\s*(\d+)", content)
            if match:
                return int(match.group(1))
    except (PermissionError, IOError):
        pass
    return DEFAULT_MAX_ITERS


def check_weight_status(strategy, dataset, domain_filter='clear_day'):
    """Check if weights exist and are valid. Also checks for training lock files and detailed results.
    
    For all_domains (Stage 2), returns aggregated status across all required models.
    Also returns model-level details for partial completion tracking.
    
    Returns:
        tuple: (status, path, model_details)
            - status: 'complete', 'partial', 'running', 'failed', or 'pending'
            - path: Path to first model directory
            - model_details: dict with complete_count, total_count, and per-model statuses
    """
    # Determine model directory based on strategy type
    if strategy == 'baseline':
        model_dirs = MODELS[domain_filter]['baseline']
    elif strategy.startswith('gen_'):
        model_dirs = MODELS[domain_filter]['gen']
    else:
        model_dirs = MODELS[domain_filter]['std']
    
    # Ensure model_dirs is always a list
    if isinstance(model_dirs, str):
        model_dirs = [model_dirs]
    
    # Check if this is a skip combo
    if (strategy, dataset) in SKIP_COMBOS:
        return 'skip', None, {'complete_count': 0, 'total_count': 0, 'models': {}}
    
    # Determine the appropriate weights root based on domain_filter
    # Stage 1 (clear_day): WEIGHTS_ROOT
    # Stage 2 (all_domains): WEIGHTS_ROOT_STAGE2
    # Cityscapes-gen (cityscapes_gen): WEIGHTS_ROOT_CITYSCAPES_GEN
    if domain_filter == 'all_domains':
        weights_root = WEIGHTS_ROOT_STAGE2
    elif domain_filter == 'cityscapes_gen':
        weights_root = WEIGHTS_ROOT_CITYSCAPES_GEN
    else:
        weights_root = WEIGHTS_ROOT
    
    # Dataset directory - normalize: idd-aw -> iddaw (directory name uses no hyphen)
    dataset_dir = dataset.replace('-', '')
    dirs_to_try = [dataset_dir]
    
    # Track per-model status for Stage 2 partial completion
    model_statuses = []
    best_paths = []
    model_details = {}
    
    for model_dir in model_dirs:
        best_status = 'pending'
        best_path = None
        
        # Try model_dir as-is, and also with _ratio1p0 suffix for legacy compatibility
        model_variants = [model_dir]
        if not model_dir.endswith('_ratio0p50') and not model_dir.endswith('_ratio1p0'):
            # For standard/baseline strategies, also check legacy _ratio1p0 directories
            model_variants.append(f'{model_dir}_ratio1p0')
        
        for d in dirs_to_try:
            for model_variant in model_variants:
                weights_path = weights_root / strategy / d / model_variant
                
                # Get target iterations from config (or use default)
                target_iters = get_target_iterations(weights_path)
                
                # Check for valid checkpoints
                checkpoint = None
                is_complete = False
                highest_iter = 0
                
                # First check for the final checkpoint (target iterations from config)
                final_ckpt = weights_path / f"iter_{target_iters}.pth"
                if final_ckpt.exists():
                    checkpoint = final_ckpt
                    is_complete = True
                else:
                    # Also check common iteration checkpoints to find highest
                    # Include both old (80k) and new (15k) regime checkpoints
                    for iter_num in [80000, 75000, 70000, 65000, 60000, 55000, 50000, 45000, 40000, 
                                     35000, 30000, 25000, 20000, 15000, 14000, 12000, 10000, 8000,
                                     6000, 5000, 4000, 2000, 1250]:
                        ckpt = weights_path / f"iter_{iter_num}.pth"
                        if ckpt.exists():
                            checkpoint = ckpt
                            highest_iter = iter_num
                            # Check if this is actually the target (complete)
                            if iter_num == target_iters:
                                is_complete = True
                            break
                
                lock_file = weights_path / ".training_lock"
                
                status = 'pending'
                try:
                    if lock_file.exists():
                        status = 'running'
                    elif checkpoint and checkpoint.exists():
                        if checkpoint.stat().st_size > 1000:
                            if is_complete:
                                status = 'complete'
                            elif highest_iter >= min(5000, target_iters * 0.3):
                                # Significant progress (at least 30% or 5k iters) - treat as in-progress
                                status = 'running'  # Use 'running' to indicate work in progress
                            else:
                                # Very early checkpoint - likely early OOM
                                status = 'failed'
                        else:
                            status = 'failed'
                except PermissionError:
                    try:
                        if weights_path.exists():
                            files = os.listdir(weights_path)
                            if ".training_lock" in files: 
                                status = 'running'
                            elif f"iter_{target_iters}.pth" in files: 
                                status = 'complete'
                            # Check for significant progress (common iteration numbers)
                            elif any(f"iter_{n}.pth" in files for n in [80000, 75000, 70000, 65000, 60000, 55000, 50000, 
                                    45000, 40000, 35000, 30000, 25000, 20000, 15000, 14000, 12000, 10000, 8000, 6000, 5000]): 
                                status = 'running'  # Significant progress
                            elif any(f"iter_{n}.pth" in files for n in [4000, 2000, 1250]): 
                                status = 'failed'  # Early OOM
                    except: pass
                
                # If filesystem shows pending, check LSF queue for running/pending jobs
                if status == 'pending':
                    training_jobs = _get_cached_training_jobs()
                    # Build model prefix by stripping _ratio* suffix
                    model_base = model_variant.split('_ratio')[0] if '_ratio' in model_variant else model_variant
                    lsf_status = training_jobs.get((strategy, dataset, model_base))
                    if lsf_status == 'RUN':
                        status = 'running'
                    elif lsf_status == 'PEND':
                        status = 'pending'  # Keep as pending but it IS in queue (vs missing)
                    
                if status == 'complete':
                    best_status = 'complete'
                    best_path = weights_path
                    break
                if status == 'running':
                    best_status = 'running'
                    best_path = weights_path
                elif status == 'failed' and best_status == 'pending':
                    best_status = 'failed'
                    best_path = weights_path
                elif best_path is None:
                    best_path = weights_path
            
            # If we found complete, break out of d loop too
            if best_status == 'complete':
                break
        
        model_statuses.append(best_status)
        best_paths.append(best_path)
        model_details[model_dir] = best_status
    
    # Build model_info dict
    complete_count = sum(1 for s in model_statuses if s == 'complete')
    total_count = len(model_dirs)
    model_info = {
        'complete_count': complete_count,
        'total_count': total_count,
        'models': model_details
    }
    
    # For Stage 2 (multiple models), aggregate status:
    # - complete if ALL models are complete
    # - partial if SOME (but not all) models are complete
    # - running if any is running
    # - failed if any failed and none running
    # - pending otherwise
    if len(model_dirs) > 1:
        if all(s == 'complete' for s in model_statuses):
            return 'complete', best_paths[0], model_info
        elif any(s == 'running' for s in model_statuses):
            return 'running', best_paths[0], model_info
        elif any(s == 'failed' for s in model_statuses) and not any(s == 'running' for s in model_statuses):
            return 'failed', best_paths[0], model_info
        elif any(s == 'complete' for s in model_statuses):
            return 'partial', best_paths[0], model_info
        else:
            return 'pending', best_paths[0], model_info
    else:
        return model_statuses[0], best_paths[0], model_info


def get_running_jobs():
    """Get list of currently running train/retrain jobs from both users.
    
    Only tracks Stage 1 (clear_day) jobs with _cd suffix.
    Excludes Stage 2 jobs (s2_ prefix) and all_domains jobs (_ad suffix).
    """
    running = {}  # {(strategy, dataset): status}
    
    # Known datasets for matching (include both 'idd-aw' and 'iddaw' variants)
    known_datasets = ['bdd10k', 'idd-aw', 'mapillaryvistas', 'outside15k']
    # Also match 'iddaw' in job names (maps to 'idd-aw' internally)
    _ds_normalize = {'iddaw': 'idd-aw'}
    # Also check for _cd suffix variants
    known_datasets_cd = ['bdd10k_cd', 'idd-aw_cd', 'iddaw_cd', 'mapillaryvistas_cd', 'outside15k_cd']
    _ds_cd_normalize = {'iddaw_cd': 'idd-aw'}
    # Model suffixes that might appear at the end of job names
    model_suffixes = ['dlv3p', 'pspn', 'segf', 'deeplabv3plus', 'pspnet', 'segformer']
    
    # Check jobs from multiple users
    users_to_check = ['-u ${USER}', '-u chge7185']  # '' means current user
    
    # Job name prefixes to check for Stage 1
    job_prefixes = ['retrain_', 'train_', 'fix_', 'rt3_', 'rt4_', 'rt5_']
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
        'minimal': 'std_minimal',
        'WEG': 'gen_Weather_Effect_Generator',
        'gen_WEG': 'gen_Weather_Effect_Generator',
        'step1x_new': 'gen_step1x_new',
        'step1x_v1p2': 'gen_step1x_v1p2',
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
                    for ds_cd in known_datasets_cd:
                        if ds_cd in job_part:
                            # Normalize: iddaw_cd → idd-aw, idd-aw_cd → idd-aw, etc.
                            dataset = _ds_cd_normalize.get(ds_cd, ds_cd.replace('_cd', ''))
                            # Extract strategy (everything before dataset)
                            idx = job_part.find(ds_cd)
                            strategy = job_part[:idx].rstrip('_')
                            break
                    
                    # If not found, try original dataset matching
                    if dataset is None:
                        for ds in known_datasets + ['iddaw']:
                            # Check for exact match (with underscores around it or at end)
                            ds_pattern = f'_{ds}_'
                            ds_pattern_end = f'_{ds}'
                            
                            if ds_pattern in job_part or job_part.endswith(ds_pattern_end):
                                dataset = _ds_normalize.get(ds, ds)
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
                    if any(job_name.startswith(p) for p in ['rt3_', 'rt4_', 'rt5_']):
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
                    ds_trunc_map = [
                        ('iddaw', 'idd-aw'),
                        ('mapillary', 'mapillaryvistas'),
                        ('mapill', 'mapillaryvistas'),
                        ('outsid', 'outside15k'),
                        ('bdd10k', 'bdd10k'),
                    ]
                    for trunc_ds, full_ds in ds_trunc_map:
                        patterns_to_check = [f'_{trunc_ds}_', f'_{trunc_ds}']
                        for pattern in patterns_to_check:
                            if pattern in job_part:
                                idx = job_part.find(pattern)
                                end_idx = idx + len(pattern)
                                if pattern.endswith('_') or end_idx == len(job_part) or job_part[end_idx] == '_':
                                    dataset = full_ds
                                    strategy_part = job_part[:idx]
                                    
                                    # Find matching full strategy
                                    for full_s in ALL_STRATEGIES:
                                        if full_s == strategy_part or \
                                           (len(strategy_part) >= 10 and (full_s.startswith(strategy_part) or strategy_part.startswith(full_s[:15]))):
                                            strategy = full_s
                                            break
                                    break
                        if dataset:
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
        'partial': 0,  # New status for partial completion
    }
    
    datasets = DATASETS_CITYSCAPES_GEN if domain_filter == 'cityscapes_gen' else DATASETS
    
    # Filter strategies based on domain_filter
    strategies_to_check = ALL_STRATEGIES
    if domain_filter == 'all_domains':
        strategies_to_check = [s for s in ALL_STRATEGIES if s not in STAGE2_EXCLUDED_STRATEGIES]
    elif domain_filter == 'cityscapes_gen':
        strategies_to_check = [s for s in ALL_STRATEGIES if s not in CG_EXCLUDED_STRATEGIES]
    
    for strategy in strategies_to_check:
        status_matrix[strategy] = {}
        
        for dataset in datasets:
            status, path, model_info = check_weight_status(strategy, dataset, domain_filter)
            
            # Use LSF status to refine the status - but ONLY if checkpoint doesn't exist
            # Jobs that train multiple models will skip existing checkpoints via pre-flight checks
            job_status = running_jobs.get((strategy, dataset))
            
            if status not in ('complete', 'partial'):
                # Only consider job status if not all checkpoints exist
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
                'path': str(path) if path else None,
                'emoji': get_status_emoji(status, model_info),
                'model_info': model_info,  # Store for display
            }
            
            summary[status] += 1
            
            if verbose:
                emoji = get_status_emoji(status, model_info)
                print(f"{emoji} {strategy}/{dataset}")
    
    return status_matrix, summary


def check_model_weight_status(strategy, dataset, model, domain_filter='clear_day'):
    """Check weight status for a specific model variant."""
    # Determine the appropriate weights root based on domain_filter
    if domain_filter == 'all_domains':
        weights_root = WEIGHTS_ROOT_STAGE2
    elif domain_filter == 'cityscapes_gen':
        weights_root = WEIGHTS_ROOT_CITYSCAPES_GEN
    else:
        weights_root = WEIGHTS_ROOT
    
    # Dataset directory - normalize: idd-aw -> iddaw (directory name uses no hyphen)
    dataset_dir = dataset.replace('-', '')
    dirs_to_try = [dataset_dir]
        
    for d in dirs_to_try:
        weights_path = weights_root / strategy / d / model
        
        # Get target iterations from config
        target_iters = get_target_iterations(weights_path)
        
        # Check for the final checkpoint first (exact target)
        final_ckpt = weights_path / f"iter_{target_iters}.pth"
        is_complete = False
        checkpoint = None
        highest_iter = 0
        
        if final_ckpt.exists():
            checkpoint = final_ckpt
            is_complete = True
        else:
            # Find highest checkpoint
            for iter_num in [80000, 75000, 70000, 65000, 60000, 55000, 50000, 45000, 40000,
                             35000, 30000, 25000, 20000, 15000, 14000, 12000, 10000, 8750,
                             8000, 7500, 6250, 6000, 5000, 3750, 4000, 2500, 2000, 1250]:
                ckpt = weights_path / f"iter_{iter_num}.pth"
                if ckpt.exists():
                    checkpoint = ckpt
                    highest_iter = iter_num
                    if iter_num == target_iters:
                        is_complete = True
                    break
        
        lock_file = weights_path / ".training_lock"
        
        try:
            if lock_file.exists():
                return 'running', weights_path
            
            if checkpoint and checkpoint.exists():
                if checkpoint.stat().st_size > 1000:
                    if is_complete:
                        return 'complete', weights_path
                    elif highest_iter >= min(5000, target_iters * 0.3):
                        return 'running', weights_path  # Significant progress
                    else:
                        return 'failed', weights_path  # Early failure
                else:
                    return 'failed', weights_path
        except PermissionError:
            try:
                if weights_path.exists():
                    files = os.listdir(weights_path)
                    if ".training_lock" in files: return 'running', weights_path
                    if f"iter_{target_iters}.pth" in files: return 'complete', weights_path
                    if any(f"iter_{n}.pth" in files for n in [80000, 75000, 70000, 65000, 60000, 55000, 50000,
                            45000, 40000, 35000, 30000, 25000, 20000, 15000, 10000, 8000, 6000, 5000]):
                        return 'running', weights_path
                    if any(f"iter_{n}.pth" in files for n in [4000, 2000, 1250]):
                        return 'failed', weights_path
            except: pass
            
    return 'pending', weights_path


def get_running_jobs_detailed(verbose=False):
    """Get detailed list of running/pending training jobs from LSF queue.
    
    Uses the centralized bjobs parser (get_training_jobs) and reformats for
    coverage report compatibility.
    
    Returns:
        dict: {(strategy, dataset, model_type): {'status': str, 'user': str}}
              model_type is short name like 'pspnet', 'segformer', etc.
    """
    jobs = {}
    training_jobs = _get_cached_training_jobs()
    
    # Map from full model prefix to short model type used by coverage report
    prefix_to_short = {
        'pspnet_r50': 'pspnet',
        'segformer_mit-b3': 'segformer',
        'segnext_mscan-b': 'segnext',
        'mask2former_swin-b': 'mask2former',
        'hrnet_hr48': 'hrnet',
        'deeplabv3plus_r50': 'deeplabv3plus',
    }
    
    user = os.environ.get('USER', '${USER}')
    
    for (strategy, dataset, model_prefix), stat in training_jobs.items():
        model_short = prefix_to_short.get(model_prefix, model_prefix)
        key = (strategy, dataset, model_short)
        current = jobs.get(key)
        current_status = current['status'] if current else None
        if stat == 'RUN':
            jobs[key] = {'status': 'RUN', 'user': user}
        elif stat == 'PEND' and current_status != 'RUN':
            jobs[key] = {'status': 'PEND', 'user': user}
    
    if verbose and jobs:
        print(f"  Found {len(jobs)} training jobs in LSF queue")
        run_count = sum(1 for v in jobs.values() if v['status'] == 'RUN')
        pend_count = sum(1 for v in jobs.values() if v['status'] == 'PEND')
        print(f"    RUN: {run_count}, PEND: {pend_count}")
    
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
    
    # For Stage 2 (all_domains), exclude certain strategies not in Stage 2 plan
    # For Cityscapes-Gen, exclude strategies without Cityscapes images or near-baseline
    strategies_to_check = ALL_STRATEGIES
    if domain_filter == 'all_domains':
        strategies_to_check = [s for s in ALL_STRATEGIES if s not in STAGE2_EXCLUDED_STRATEGIES]
    elif domain_filter == 'cityscapes_gen':
        strategies_to_check = [s for s in ALL_STRATEGIES if s not in CG_EXCLUDED_STRATEGIES]
    
    datasets = DATASETS_CITYSCAPES_GEN if domain_filter == 'cityscapes_gen' else DATASETS
    
    for strategy in strategies_to_check:
        # Determine which models to check based on strategy type
        if domain_filter == 'cityscapes_gen':
            # Cityscapes-gen uses 4 models (no deeplabv3plus, no hrnet)
            if strategy.startswith('gen_'):
                models_to_check = ['pspnet_r50_ratio0p50', 'segformer_mit-b3_ratio0p50', 'segnext_mscan-b_ratio0p50', 'mask2former_swin-b_ratio0p50']
            else:
                models_to_check = ['pspnet_r50', 'segformer_mit-b3', 'segnext_mscan-b', 'mask2former_swin-b']
        else:
            if strategy.startswith('gen_'):
                models_to_check = ['deeplabv3plus_r50_ratio0p50', 'pspnet_r50_ratio0p50', 'segformer_mit-b3_ratio0p50', 'segnext_mscan-b_ratio0p50', 'hrnet_hr48_ratio0p50', 'mask2former_swin-b_ratio0p50']
            else:
                models_to_check = ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b3', 'segnext_mscan-b', 'hrnet_hr48', 'mask2former_swin-b']
        
        for dataset in datasets:
            for model in models_to_check:
                status, path = check_model_weight_status(strategy, dataset, model, domain_filter)
                user = None
                
                # Check job status for refinement - but ONLY if checkpoint doesn't already exist
                # Jobs that train multiple models will skip existing checkpoints via pre-flight checks
                if status != 'complete':
                    # Map model name to short type for job lookup
                    model_type_map = {
                        'deeplabv3plus': 'deeplabv3plus',
                        'pspnet': 'pspnet',
                        'segformer': 'segformer',
                        'segnext': 'segnext',
                        'hrnet': 'hrnet',
                        'mask2former': 'mask2former',
                    }
                    model_type = None
                    for key_str, val in model_type_map.items():
                        if key_str in model:
                            model_type = val
                            break
                    if model_type is None:
                        model_type = model.split('_')[0]
                    
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
        
        ds_headers = ' | '.join(DATASET_DISPLAY.get(ds, ds) for ds in DATASETS)
        lines.append(f"| Strategy | {ds_headers} |")
        ds_seps = ' | '.join('---' + '-' * max(0, len(DATASET_DISPLAY.get(ds, ds)) - 3) for ds in DATASETS)
        lines.append(f"|----------|{ds_seps}|")
        
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
        data = status_dict.get(dataset, {'emoji': '❓'})
        cell_content = data['emoji']
        cells.append(cell_content)
    
    # Add notes based on strategy
    notes = []
    if strategy == 'gen_Qwen_Image_Edit':
        notes.append('No BDD10k data')
    
    cells.append(' | '.join(notes) if notes else '')
    
    return '| ' + ' | '.join(cells) + ' |'


def update_tracker(status_matrix, summary, domain_filter='clear_day', tracker_path=None):
    """Update the tracker markdown file."""
    if tracker_path is None:
        tracker_path = TRACKER_PATH
    
    # Determine which strategies to include based on domain_filter
    if domain_filter == 'all_domains':
        excluded = STAGE2_EXCLUDED_STRATEGIES
    elif domain_filter == 'cityscapes_gen':
        excluded = CG_EXCLUDED_STRATEGIES
    else:
        excluded = set()
    
    gen_strategies = [s for s in GENERATIVE_STRATEGIES if s not in excluded]
    std_strategies = [s for s in STANDARD_STRATEGIES if s not in excluded]
    
    # Read current tracker (or create new one if doesn't exist)
    # Build dynamic column headers from DATASETS
    ds_headers = ' | '.join(DATASET_DISPLAY.get(ds, ds) for ds in DATASETS)
    ds_separators = ' | '.join('---' + '-' * max(0, len(DATASET_DISPLAY.get(ds, ds)) - 3) for ds in DATASETS)
    table_header = f'| Strategy | {ds_headers} | Notes |'
    table_separator = f'|----------|{ds_separators}|-------|'
    
    try:
        with open(tracker_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        # Create a basic template if file doesn't exist
        stage_names = {'clear_day': 'Stage 1 (Clear Day)', 'all_domains': 'Stage 2 (All Domains)', 'cityscapes_gen': 'Cityscapes-Gen'}
        stage_name = stage_names.get(domain_filter, domain_filter)
        content = f"""# Training Tracker - {stage_name}

**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Progress Summary

| Category | Total | Complete | Partial | Running | Pending | Failed |
|----------|-------|----------|---------|---------|---------|--------|
| **Generative (gen_*)** | 0 | 0 | 0 | 0 | 0 | 0 |
| **Standard (std_*)** | 0 | 0 | 0 | 0 | 0 | 0 |
| **TOTAL** | 0 | 0 | 0 | 0 | 0 | 0 |

### Generative Image Augmentation Strategies

{table_header}
{table_separator}

### Standard Augmentation Strategies

{table_header}
{table_separator}
"""
    
    # Update timestamp
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    content = re.sub(
        r'\*\*Last Updated:\*\* .+',
        f'**Last Updated:** {now}',
        content
    )
    
    # Build new generative strategies table
    gen_rows = [table_header, table_separator]
    for strategy in gen_strategies:
        gen_rows.append(format_status_row(strategy, status_matrix[strategy], DATASETS))
    gen_table = '\n'.join(gen_rows)
    
    # Build new standard strategies table
    std_rows = [table_header, table_separator]
    for strategy in std_strategies:
        std_rows.append(format_status_row(strategy, status_matrix[strategy], DATASETS))
    std_table = '\n'.join(std_rows)
    
    # Update generative table - handle both empty and filled tables
    gen_pattern = r'### Generative Image Augmentation Strategies\n\n\|[^\n]+\n\|[-|\s]+\n(?:\|[^\n]+\n)*'
    gen_replacement = f'### Generative Image Augmentation Strategies\n\n{gen_table}\n'
    content = re.sub(gen_pattern, gen_replacement, content)
    
    # Update standard table - handle both empty and filled tables
    std_pattern = r'### Standard Augmentation Strategies\n\n\|[^\n]+\n\|[-|\s]+\n(?:\|[^\n]+\n)*'
    std_replacement = f'### Standard Augmentation Strategies\n\n{std_table}\n'
    content = re.sub(std_pattern, std_replacement, content)
    
    # Update progress summary - now counting individual model trainings
    # Each strategy×dataset has 4 models
    num_models = 4  # PSPNet, SegFormer, SegNeXt, Mask2Former
    
    # Count individual model completions from model_info
    gen_model_complete = 0
    gen_model_running = 0
    gen_model_pending = 0
    gen_model_failed = 0
    for s in gen_strategies:
        for d in DATASETS:
            model_info = status_matrix[s][d].get('model_info', {})
            models = model_info.get('models', {})
            for model, status in models.items():
                if status == 'complete':
                    gen_model_complete += 1
                elif status == 'running':
                    gen_model_running += 1
                elif status == 'failed':
                    gen_model_failed += 1
                else:
                    gen_model_pending += 1
    
    std_model_complete = 0
    std_model_running = 0
    std_model_pending = 0
    std_model_failed = 0
    for s in std_strategies:
        for d in DATASETS:
            model_info = status_matrix[s][d].get('model_info', {})
            models = model_info.get('models', {})
            for model, status in models.items():
                if status == 'complete':
                    std_model_complete += 1
                elif status == 'running':
                    std_model_running += 1
                elif status == 'failed':
                    std_model_failed += 1
                else:
                    std_model_pending += 1
    
    # Fallback: count configurations if model_info not available
    gen_complete = sum(1 for s in gen_strategies 
                       for d in DATASETS 
                       if status_matrix[s][d]['status'] == 'complete')
    gen_partial = sum(1 for s in gen_strategies 
                      for d in DATASETS 
                      if status_matrix[s][d]['status'] == 'partial')
    gen_running = sum(1 for s in gen_strategies 
                      for d in DATASETS 
                      if status_matrix[s][d]['status'] == 'running')
    gen_pending = sum(1 for s in gen_strategies 
                      for d in DATASETS 
                      if status_matrix[s][d]['status'] == 'pending')
    gen_failed = sum(1 for s in gen_strategies 
                     for d in DATASETS 
                     if status_matrix[s][d]['status'] == 'failed')
    gen_total = len(gen_strategies) * len(DATASETS)
    
    std_complete = sum(1 for s in std_strategies 
                       for d in DATASETS 
                       if status_matrix[s][d]['status'] == 'complete')
    std_partial = sum(1 for s in std_strategies 
                      for d in DATASETS 
                      if status_matrix[s][d]['status'] == 'partial')
    std_running = sum(1 for s in std_strategies 
                      for d in DATASETS 
                      if status_matrix[s][d]['status'] == 'running')
    std_pending = sum(1 for s in std_strategies 
                      for d in DATASETS 
                      if status_matrix[s][d]['status'] == 'pending')
    std_failed = sum(1 for s in std_strategies 
                     for d in DATASETS 
                     if status_matrix[s][d]['status'] == 'failed')
    std_total = len(std_strategies) * len(DATASETS)
    
    # Individual model totals
    gen_model_total = gen_total * num_models
    std_model_total = std_total * num_models
    
    total_complete = gen_complete + std_complete
    total_partial = gen_partial + std_partial
    total_running = gen_running + std_running
    total_pending = gen_pending + std_pending
    total_failed = gen_failed + std_failed
    total = gen_total + std_total
    
    total_model_complete = gen_model_complete + std_model_complete
    total_model_running = gen_model_running + std_model_running
    total_model_pending = gen_model_pending + std_model_pending
    total_model_failed = gen_model_failed + std_model_failed
    total_model_total = gen_model_total + std_model_total
    
    # Progress table now shows both configurations (4/4 complete) and individual models
    progress_table = f"""| Category | Configs | Complete (4/4) | Partial | Running | Pending | Failed |
|----------|---------|----------------|---------|---------|---------|--------|
| **Generative (gen_*)** | {gen_total} | {gen_complete} | {gen_partial} | {gen_running} | {gen_pending} | {gen_failed} |
| **Standard (std_*)** | {std_total} | {std_complete} | {std_partial} | {std_running} | {std_pending} | {std_failed} |
| **TOTAL** | {total} | {total_complete} | {total_partial} | {total_running} | {total_pending} | {total_failed} |

### Individual Model Trainings

| Category | Total Models | ✅ Complete | 🔄 Running | ⏳ Pending | ❌ Failed |
|----------|-------------|-------------|------------|-----------|----------|
| **Generative (gen_*)** | {gen_model_total} | {gen_model_complete} | {gen_model_running} | {gen_model_pending} | {gen_model_failed} |
| **Standard (std_*)** | {std_model_total} | {std_model_complete} | {std_model_running} | {std_model_pending} | {std_model_failed} |
| **TOTAL** | {total_model_total} | {total_model_complete} | {total_model_running} | {total_model_pending} | {total_model_failed} |"""
    
    # Match either old format (Total | Complete) or new format (Configs | Complete)
    # Also match the Individual Model Trainings section if present
    progress_pattern = r'\| Category \| (?:Total|Configs) \|[^\n]+\n\|[-|\s]+\n(?:\|[^\n]+\n)+(?:\n### Individual Model Trainings\n\n\|[^\n]+\n\|[-|\s]+\n(?:\|[^\n]+\n)+)?'
    content = re.sub(progress_pattern, progress_table + '\n', content)
    
    # Write updated tracker
    with open(tracker_path, 'w') as f:
        f.write(content)
    
    return {
        'gen': {'total': gen_total, 'complete': gen_complete, 'partial': gen_partial,
                'running': gen_running, 'pending': gen_pending, 'failed': gen_failed,
                'model_total': gen_model_total, 'model_complete': gen_model_complete,
                'model_running': gen_model_running, 'model_pending': gen_model_pending,
                'model_failed': gen_model_failed},
        'std': {'total': std_total, 'complete': std_complete, 'partial': std_partial,
                'running': std_running, 'pending': std_pending, 'failed': std_failed,
                'model_total': std_model_total, 'model_complete': std_model_complete,
                'model_running': std_model_running, 'model_pending': std_model_pending,
                'model_failed': std_model_failed},
        'total': {'total': total, 'complete': total_complete, 'partial': total_partial,
                  'running': total_running, 'pending': total_pending, 'failed': total_failed,
                  'model_total': total_model_total, 'model_complete': total_model_complete,
                  'model_running': total_model_running, 'model_pending': total_model_pending,
                  'model_failed': total_model_failed},
    }


def print_summary(stats):
    """Print a summary of the status."""
    print("\n" + "="*60)
    print("RETRAINING PROGRESS SUMMARY")
    print("="*60)
    
    for category, data in stats.items():
        name = {'gen': 'Generative', 'std': 'Standard', 'total': 'TOTAL'}[category]
        pct = (data['complete'] / data['total'] * 100) if data['total'] > 0 else 0
        partial_pct = ((data['complete'] + data['partial']) / data['total'] * 100) if data['total'] > 0 else 0
        model_pct = (data.get('model_complete', 0) / data.get('model_total', 1) * 100) if data.get('model_total', 0) > 0 else 0
        print(f"\n{name} Strategies:")
        print(f"  Configurations (4/4 models):")
        print(f"    Complete: {data['complete']}/{data['total']} ({pct:.1f}%)")
        print(f"    Partial:  {data['partial']} (total {data['complete'] + data['partial']}/{data['total']} = {partial_pct:.1f}%)")
        print(f"    Running:  {data['running']}")
        print(f"    Pending:  {data['pending']}")
        print(f"    Failed:   {data['failed']}")
        if 'model_total' in data:
            print(f"  Individual Models:")
            print(f"    Complete: {data['model_complete']}/{data['model_total']} ({model_pct:.1f}%)")
            print(f"    Running:  {data['model_running']}")
            print(f"    Pending:  {data['model_pending']}")
            print(f"    Failed:   {data['model_failed']}")


def run_stage(stage, verbose=False, no_update=False, coverage_report=False, output=None):
    """Run training tracker for a single stage."""
    stage_map = {'1': 'clear_day', '2': 'all_domains', 'cityscapes-gen': 'cityscapes_gen'}
    domain_filter = stage_map[stage]
    
    # Override global DATASETS for cityscapes-gen
    global DATASETS
    if domain_filter == 'cityscapes_gen':
        DATASETS = DATASETS_CITYSCAPES_GEN
    else:
        DATASETS = DATASETS_DEFAULT
    
    # Coverage report mode
    if coverage_report:
        print(f"Generating detailed coverage report for Stage {stage} ({domain_filter})...")
        coverage, summary = collect_detailed_coverage(domain_filter, verbose)
        
        output_file = output
        if output_file is None:
            # Use stage-specific filenames
            stage_suffix = {'1': 'STAGE1', '2': 'STAGE2', 'cityscapes-gen': 'CITYSCAPES_GEN'}[stage]
            output_file = PROJECT_ROOT / 'docs' / f'TRAINING_COVERAGE_{stage_suffix}.md'
        
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
    
    print(f"Checking Stage {stage} ({domain_filter}) status...")
    status_matrix, summary = collect_status(domain_filter, verbose)
    
    # Use stage-specific tracker file
    stage_suffix = {'1': 'STAGE1', '2': 'STAGE2', 'cityscapes-gen': 'CITYSCAPES_GEN'}[stage]
    tracker_path = PROJECT_ROOT / 'docs' / f'TRAINING_TRACKER_{stage_suffix}.md'
    
    if not no_update:
        print(f"\nUpdating tracker: {tracker_path}")
        stats = update_tracker(status_matrix, summary, domain_filter, tracker_path)
        print_summary(stats)
        print(f"\nTracker updated: {tracker_path}")
    else:
        print("\nStatus collected (no update mode)")
        print(f"  Complete: {summary['complete']}")
        print(f"  Running:  {summary['running']}")
        print(f"  Pending:  {summary['pending']}")
        print(f"  Failed:   {summary['failed']}")


ALL_STAGES = ['1', '2', 'cityscapes-gen']


def main():
    parser = argparse.ArgumentParser(description='Update retraining progress tracker')
    parser.add_argument('--stage', type=str, default='1', choices=['1', '2', 'cityscapes-gen', 'all'],
                        help='Stage to check (1=clear_day, 2=all_domains, cityscapes-gen, all=run all stages)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed status for all combinations')
    parser.add_argument('--no-update', action='store_true',
                        help='Only show status without updating tracker')
    parser.add_argument('--coverage-report', '-c', action='store_true',
                        help='Generate detailed coverage report showing each (strategy, dataset, model)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file for coverage report (default: docs/TRAINING_COVERAGE_STAGE1.md or STAGE2.md)')
    args = parser.parse_args()
    
    stages = ALL_STAGES if args.stage == 'all' else [args.stage]
    
    for i, stage in enumerate(stages):
        if len(stages) > 1:
            print(f"\n{'#' * 70}")
            print(f"# STAGE {stage.upper()}")
            print(f"{'#' * 70}")
        run_stage(stage, verbose=args.verbose, no_update=args.no_update,
                  coverage_report=args.coverage_report, output=args.output)


if __name__ == '__main__':
    main()
