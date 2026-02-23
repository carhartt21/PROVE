#!/usr/bin/env python3
"""
Update the testing progress tracker based on current job status and test results.

Reads results directly from the weights folder (test_results_detailed/*/results.json
or test_results_detailed_fixed/*/results.json for backward compatibility)
to get the latest mIoU values without needing to run the test_result_analyzer first.

Usage:
    python scripts/update_testing_tracker.py              # Stage 1 (default)
    python scripts/update_testing_tracker.py --stage 2    # Stage 2
    python scripts/update_testing_tracker.py --verbose    # Show all status details
"""

import os
import subprocess
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re
import logging

# Set up logging for permission errors (silent by default)
logger = logging.getLogger(__name__)


def safe_iterdir(path, default=None):
    """Safely iterate over a directory, handling permission errors.
    
    Args:
        path: Path object to iterate
        default: Default value to return on permission error (defaults to empty list)
    
    Returns:
        List of Path objects or default value on error
    """
    if default is None:
        default = []
    try:
        return list(path.iterdir())
    except PermissionError as e:
        logger.debug(f"Permission denied accessing {path}: {e}")
        return default
    except OSError as e:
        logger.debug(f"OS error accessing {path}: {e}")
        return default


def safe_glob(path, pattern, default=None):
    """Safely glob a directory, handling permission errors.
    
    Args:
        path: Path object to glob in
        pattern: Glob pattern
        default: Default value to return on permission error (defaults to empty list)
    
    Returns:
        List of matching Path objects or default value on error
    """
    if default is None:
        default = []
    try:
        return list(path.glob(pattern))
    except PermissionError as e:
        logger.debug(f"Permission denied globbing {path}/{pattern}: {e}")
        return default
    except OSError as e:
        logger.debug(f"OS error globbing {path}/{pattern}: {e}")
        return default


def safe_is_dir(path):
    """Safely check if path is a directory, handling permission errors."""
    try:
        return path.is_dir()
    except PermissionError:
        return False
    except OSError:
        return False


def safe_exists(path):
    """Safely check if path exists, handling permission errors."""
    try:
        return path.exists()
    except PermissionError:
        return False
    except OSError:
        return False


# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WEIGHTS_ROOT = Path(os.environ.get('PROVE_WEIGHTS_ROOT', '${AWARE_DATA_ROOT}/WEIGHTS'))
WEIGHTS_ROOT_STAGE2 = Path(os.environ.get('PROVE_WEIGHTS_ROOT_STAGE2', '${AWARE_DATA_ROOT}/WEIGHTS_STAGE_2'))
WEIGHTS_ROOT_CITYSCAPES_GEN = Path('${AWARE_DATA_ROOT}/WEIGHTS_CITYSCAPES_GEN')
TRACKER_PATH = PROJECT_ROOT / 'docs' / 'TESTING_TRACKER.md'
TRACKER_PATH_STAGE2 = PROJECT_ROOT / 'docs' / 'TESTING_TRACKER_STAGE2.md'
TRACKER_PATH_CITYSCAPES_GEN = PROJECT_ROOT / 'docs' / 'TESTING_TRACKER_CITYSCAPES_GEN.md'
COVERAGE_PATH = PROJECT_ROOT / 'docs' / 'TESTING_COVERAGE.md'
COVERAGE_PATH_STAGE2 = PROJECT_ROOT / 'docs' / 'TESTING_COVERAGE_STAGE2.md'
COVERAGE_PATH_CITYSCAPES_GEN = PROJECT_ROOT / 'docs' / 'TESTING_COVERAGE_CITYSCAPES_GEN.md'
TEST_RESULTS_CSV = PROJECT_ROOT / 'test_results_summary.csv'

DATASETS = ['bdd10k', 'idd-aw', 'mapillaryvistas', 'outside15k']
DATASETS_DEFAULT = list(DATASETS)  # Save original for --stage all reset
DATASETS_CITYSCAPES_GEN = ['cityscapes', 'acdc']
DATASET_DISPLAY = {
    'bdd10k': 'BDD10k',
    'idd-aw': 'IDD-AW',
    'mapillaryvistas': 'MapillaryVistas',
    'outside15k': 'OUTSIDE15k',
    'cityscapes': 'Cityscapes',
    'acdc': 'ACDC (cross-domain)',
}

# Cross-domain test mapping: dataset -> (training_dataset_dir, test_results_subdir)
# For cross-domain tests, model dirs are under the training dataset but results
# are in a different subdirectory (e.g., test_results_acdc instead of test_results_detailed)
CROSS_DOMAIN_TEST_MAP = {
    'acdc': ('cityscapes', 'test_results_acdc'),
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
    'std_minimal',
    'std_photometric_distort',
    'std_autoaugment',
    'std_cutmix',
    'std_mixup',
    'std_randaugment',
]

ALL_STRATEGIES = GENERATIVE_STRATEGIES + STANDARD_STRATEGIES

# Cityscapes-Gen excludes these strategies (no Cityscapes generated images or near-identical to baseline)
CG_EXCLUDED_STRATEGIES = {
    'gen_LANIT',              # No Cityscapes generated images exist
    'std_minimal',            # RandomCrop + RandomFlip only — essentially same as baseline
    'std_photometric_distort',  # Essentially same as baseline
}

# Current stage (set by run_stage)
CURRENT_STAGE = '1'

# Skip combinations (no data available)
# NOTE: This list should be empty now that most strategies have full 4/4 coverage
# Only add combinations here if training data is truly unavailable
SKIP_COMBOS = set()  # All strategies now have full dataset coverage

# Models for per-model breakdown
BASE_MODELS = ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5', 'segnext_mscan-b', 'mask2former_swin-b']
MODEL_DISPLAY = {
    'deeplabv3plus_r50': 'DeepLabV3+',
    'pspnet_r50': 'PSPNet',
    'segformer_mit-b5': 'SegFormer',
    'segnext_mscan-b': 'SegNeXt',
    'mask2former_swin-b': 'Mask2Former',
}


def load_per_model_results():
    """Load per-model mIoU results from test_results_summary.csv.
    
    Returns:
        dict: {dataset: {model: {'avg': float, 'count': int}}}
    """
    import pandas as pd
    
    csv_path = TEST_RESULTS_CSV
    if not safe_exists(csv_path):
        return {}
    
    try:
        df = pd.read_csv(csv_path)
        
        # Filter to detailed_fixed test results
        df = df[df['test_type'] == 'test_results_detailed_fixed']
        df = df.dropna(subset=['mIoU'])
        
        # Normalize model names (remove ratio suffix)
        df['base_model'] = df['model'].str.replace('_ratio0p50', '', regex=False)
        
        results = {}
        for ds in DATASETS:
            ds_data = df[df['dataset'] == ds]
            
            results[ds] = {}
            for model in BASE_MODELS:
                model_data = ds_data[ds_data['base_model'] == model]
                if len(model_data) > 0:
                    results[ds][model] = {
                        'avg': model_data['mIoU'].mean(),
                        'count': len(model_data),
                    }
        
        return results
    except Exception as e:
        print(f"Warning: Could not load per-model results: {e}")
        return {}


def resolve_dataset_paths(dataset):
    """Resolve dataset to its filesystem paths.
    
    For standard datasets, the model directory is under the dataset name
    and test results are in test_results_detailed/.
    
    For cross-domain tests (e.g., ACDC), the model directory is under the
    training dataset (cityscapes) but results are in a different subdir
    (test_results_acdc/).
    
    Args:
        dataset: Dataset name (e.g., 'bdd10k', 'cityscapes', 'acdc')
        
    Returns:
        tuple: (dataset_dir, test_results_subdir, fallback_subdir)
            - dataset_dir: Directory name under the strategy dir (e.g., 'cityscapes')
            - test_results_subdir: Primary test results directory name
            - fallback_subdir: Fallback test results directory name (or None)
    """
    if dataset in CROSS_DOMAIN_TEST_MAP:
        training_dataset, test_subdir = CROSS_DOMAIN_TEST_MAP[dataset]
        dataset_dir = training_dataset.replace('-', '')
        return dataset_dir, test_subdir, None
    else:
        dataset_dir = dataset.replace('-', '')
        return dataset_dir, 'test_results_detailed', 'test_results_detailed_fixed'


def load_miou_results():
    """Load mIoU results directly from the weights folder.
    
    Scans all strategy/dataset/model directories for test_results_detailed
    (or test_results_detailed_fixed for backward compatibility)
    and extracts mIoU from results.json files.
    
    Returns:
        dict: {(strategy, dataset): {'best_miou': float, 'best_model': str, 'models': {model: miou}}}
    """
    import json
    
    results = {}
    
    # Scan all strategies
    for strategy in ALL_STRATEGIES:
        for dataset in DATASETS:
            dataset_dir, test_subdir, fallback_subdir = resolve_dataset_paths(dataset)
            strategy_path = WEIGHTS_ROOT / strategy / dataset_dir
            
            if not safe_exists(strategy_path):
                continue
            
            # Get mIoU values per model
            models = {}
            for model_dir in safe_iterdir(strategy_path):
                if not safe_is_dir(model_dir) or model_dir.name.endswith('_backup'):
                    continue
                
                # Look for primary test results dir, then fall back
                results_path = model_dir / test_subdir
                if not safe_exists(results_path) and fallback_subdir:
                    results_path = model_dir / fallback_subdir
                if not safe_exists(results_path):
                    continue
                
                # Find the latest result directory (by timestamp)
                result_dirs = [d for d in safe_iterdir(results_path) 
                              if safe_is_dir(d) and d.name.startswith('2026')]
                if not result_dirs:
                    continue
                
                latest = sorted(result_dirs, key=lambda x: x.name)[-1]
                results_json = latest / 'results.json'
                
                if not safe_exists(results_json):
                    continue
                
                try:
                    with open(results_json) as f:
                        data = json.load(f)
                    miou = data.get('overall', {}).get('mIoU')
                    if miou is not None and miou > 0:
                        # Convert to percentage if in 0-1 range
                        if miou < 1:
                            miou = miou * 100
                        models[model_dir.name] = miou
                except Exception:
                    continue
            
            if models:
                best_model = max(models, key=models.get)
                best_miou = models[best_model]
                results[(strategy, dataset)] = {
                    'best_miou': best_miou,
                    'best_model': best_model,
                    'models': models,
                    'avg_miou': sum(models.values()) / len(models),
                }
    
    return results


def get_retest_jobs():
    """Get list of currently queued/running retest jobs.
    
    Returns:
        tuple: (jobs_by_combo, job_counts_by_dataset)
            - jobs_by_combo: {(strategy, dataset): status} unique combinations
            - job_counts_by_dataset: {dataset: {status: count}} individual job counts
    """
    jobs = {}  # {(strategy, dataset): status}
    job_counts = defaultdict(lambda: defaultdict(int))  # {dataset: {status: count}}
    
    try:
        result = subprocess.run(
            ['bjobs', '-u', '${USER}', '-a', '-o', 'JOBID JOB_NAME STAT'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        for line in result.stdout.strip().split('\n')[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 3:
                job_name = parts[1]
                stat = parts[2]
                if 'retest' in job_name:
                    # Parse job name: retest_<strategy>_<dataset>_<model>
                    # Find dataset
                    dataset = None
                    strategy = None
                    
                    for ds in DATASETS:
                        # Try both hyphenated and non-hyphenated forms for idd-aw/iddaw
                        ds_patterns = [f'_{ds}']
                        if '-' in ds:
                            ds_patterns.append(f'_{ds.replace("-", "")}')
                        matched_pattern = None
                        for ds_pattern in ds_patterns:
                            if ds_pattern in job_name:
                                dataset = ds
                                matched_pattern = ds_pattern
                                break
                        if dataset:
                            # Extract strategy
                            start_idx = job_name.find('retest_') + len('retest_')
                            end_idx = job_name.find(matched_pattern)
                            strategy = job_name[start_idx:end_idx].rstrip('_')
                            break
                    
                    if dataset:
                        # Count individual jobs by dataset
                        job_counts[dataset][stat] += 1
                        
                        if strategy:
                            # Track unique combinations
                            current_stat = jobs.get((strategy, dataset))
                            if stat == 'RUN':
                                jobs[(strategy, dataset)] = 'RUN'
                            elif stat == 'PEND' and current_stat != 'RUN':
                                jobs[(strategy, dataset)] = 'PEND'
                            elif stat in ('DONE', 'EXIT') and current_stat not in ('RUN', 'PEND'):
                                jobs[(strategy, dataset)] = stat
    except Exception as e:
        print(f"Warning: Could not get jobs: {e}")
    
    return jobs, dict(job_counts)


def check_test_results(strategy, dataset, test_dir=None):
    """Check if test results exist for a strategy/dataset combination.
    
    Uses resolve_dataset_paths() to handle cross-domain tests (e.g., ACDC).
    Falls back to test_results_detailed_fixed for backward compatibility.
    """
    dataset_dir, test_subdir, fallback_subdir = resolve_dataset_paths(dataset)
    if test_dir is None:
        test_dir = test_subdir
    
    # Check for any model's test results
    strategy_path = WEIGHTS_ROOT / strategy / dataset_dir
    if not safe_exists(strategy_path):
        return False, False
    
    has_results = False
    has_detailed = False
    
    for model_dir in safe_iterdir(strategy_path):
        if safe_is_dir(model_dir):
            # Check primary test dir, then fallback
            test_path = model_dir / test_dir
            if not safe_exists(test_path) and fallback_subdir:
                test_path = model_dir / fallback_subdir
            if safe_exists(test_path):
                # Check for any timestamped results
                result_dirs = safe_glob(test_path, '*/')
                if result_dirs:
                    has_results = True
                    # Check for detailed results
                    for result_dir in result_dirs:
                        if safe_exists(result_dir / 'per_domain_results.json'):
                            has_detailed = True
                            break
    
    return has_results, has_detailed


def get_status_emoji(status):
    """Convert status to emoji."""
    return {
        'complete': '✅',
        'complete_detailed': '✅ 🎯',
        'running': '🔄',
        'pending': '⏳',
        'failed': '❌',
        'skip': '➖',
    }.get(status, '?')


def collect_test_status(verbose=False):
    """Collect test status for all strategy/dataset combinations."""
    retest_jobs, job_counts = get_retest_jobs()
    miou_results = load_miou_results()
    
    # Filter strategies based on current stage
    strategies_to_check = ALL_STRATEGIES
    if CURRENT_STAGE == 'cityscapes-gen':
        strategies_to_check = [s for s in ALL_STRATEGIES if s not in CG_EXCLUDED_STRATEGIES]
    
    status_matrix = {}
    summary = defaultdict(lambda: defaultdict(int))  # summary[dataset][status] = count
    
    for strategy in strategies_to_check:
        status_matrix[strategy] = {}
        
        for dataset in DATASETS:
            # Check if this combination should be skipped
            if (strategy, dataset) in SKIP_COMBOS:
                status_matrix[strategy][dataset] = {
                    'status': 'skip',
                    'emoji': '➖',
                    'miou': None,
                }
                continue
            
            # Check for existing test results
            has_results, has_detailed = check_test_results(strategy, dataset)
            
            # Get mIoU value if available
            miou_data = miou_results.get((strategy, dataset))
            miou = miou_data['best_miou'] if miou_data else None
            
            # Check job status
            job_status = retest_jobs.get((strategy, dataset))
            
            # Determine status
            # Priority: completed results > running jobs > pending jobs > failed > pending
            if miou is not None:
                # Valid mIoU means actual results (at least one model completed)
                if has_detailed:
                    status = 'complete_detailed'
                    emoji = '✅'
                else:
                    status = 'complete'
                    emoji = '✅'
            elif job_status == 'RUN':
                # Jobs running but no results yet
                status = 'running'
                emoji = '🔄'
            elif job_status == 'PEND':
                # Pending retest - no results yet
                status = 'pending'
                emoji = '⏳'
            elif has_results:
                # Test results exist but no mIoU - likely failed test (path issue)
                status = 'failed'
                emoji = '❌'
            else:
                # No results and no pending job
                status = 'pending'
                emoji = '⏳'
            
            status_matrix[strategy][dataset] = {
                'status': status,
                'emoji': emoji,
                'miou': miou,
            }
            
            # Update summary
            if status not in ('skip',):
                summary[dataset][status] += 1
            
            if verbose:
                miou_str = f" (mIoU: {miou:.2f})" if miou else ""
                print(f"  {strategy}/{dataset}: {status}{miou_str}")
    
    return status_matrix, summary, retest_jobs, job_counts


def update_tracker(status_matrix, summary, retest_jobs, job_counts):
    """Update the testing tracker markdown file."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    # job_counts is now passed in directly (individual job counts by dataset)
    
    # Build markdown content
    lines = []
    lines.append("# Testing Progress Tracker\n")
    lines.append(f"**Last Updated:** {now}\n")
    lines.append("\nThis document tracks the progress of fine-grained testing for trained models.\n")
    
    # Overview section
    lines.append("\n## Overview\n")
    lines.append("\n### Test Job Types\n")
    lines.append("| Type | Description | Output Location |")
    lines.append("|------|-------------|-----------------|")
    lines.append("| **Initial Test** | First test after training completes | `{weights}/test_results_detailed/` |")
    lines.append("| **Retest (Fixed)** | Retest after fine_grained_test.py bug fix | `{weights}/test_results_detailed_fixed/` |")
    
    # Retest job status
    lines.append("\n---\n")
    lines.append("\n## Current Retest Jobs\n")
    
    total_running = sum(job_counts.get(ds, {}).get('RUN', 0) for ds in DATASETS)
    total_pending = sum(job_counts.get(ds, {}).get('PEND', 0) for ds in DATASETS)
    total_done = sum(job_counts.get(ds, {}).get('DONE', 0) for ds in DATASETS)
    
    lines.append("\n### Retest Job Status\n")
    lines.append("| Status | Count | Description |")
    lines.append("|--------|-------|-------------|")
    lines.append(f"| 🔄 Running | {total_running} | Currently testing |")
    lines.append(f"| ⏳ Pending | {total_pending} | Queued, waiting to run |")
    lines.append(f"| ✅ Complete | {total_done} | Test results available |")
    
    lines.append("\n### Retest Jobs by Dataset\n")
    lines.append("| Dataset | Running | Pending | Complete | Total |")
    lines.append("|---------|---------|---------|----------|-------|")
    for ds in DATASETS:
        running = job_counts.get(ds, {}).get('RUN', 0)
        pending = job_counts.get(ds, {}).get('PEND', 0)
        done = job_counts.get(ds, {}).get('DONE', 0)
        total = running + pending + done
        lines.append(f"| {DATASET_DISPLAY[ds]} | {running} | {pending} | {done} | {total} |")
    
    # mIoU Results Table
    lines.append("\n---\n")
    lines.append("\n## mIoU Results (Clear Day Training)\n")
    lines.append("\n*mIoU values shown are the best across all models (deeplabv3plus, pspnet, segformer). Values are percentages.*\n")
    
    # Top 10 Strategies Overview
    lines.append("\n### 🏆 Top 10 Strategies (by Average mIoU)\n")
    
    # Calculate average mIoU for all strategies
    strategy_avgs = []
    for strategy in ALL_STRATEGIES:
        values = []
        miou_by_dataset = {}
        for ds in DATASETS:
            miou = status_matrix[strategy][ds].get('miou')
            if miou is not None:
                values.append(miou)
                miou_by_dataset[ds] = miou
        if values:
            avg = sum(values) / len(values)
            datasets_complete = len(values)
            best_ds = max(miou_by_dataset.keys(), key=lambda ds: miou_by_dataset[ds])
            strategy_avgs.append({
                'strategy': strategy,
                'avg_miou': avg,
                'datasets_complete': datasets_complete,
                'best_dataset': best_ds,
                'best_miou': miou_by_dataset[best_ds],
            })
    
    # Sort by average mIoU descending
    strategy_avgs.sort(key=lambda x: x['avg_miou'], reverse=True)
    
    lines.append("| Rank | Strategy | Avg mIoU | Best Dataset | Best mIoU | Datasets |")
    lines.append("|------|----------|----------|--------------|-----------|----------|")
    
    for i, s in enumerate(strategy_avgs[:10], 1):
        medal = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else f"{i}."))
        lines.append(f"| {medal} | {s['strategy']} | {s['avg_miou']:.1f} | {DATASET_DISPLAY[s['best_dataset']]} | {s['best_miou']:.1f} | {s['datasets_complete']}/4 |")
    
    lines.append("")
    
    # Generative strategies mIoU table
    lines.append("\n### Generative Image Augmentation Strategies\n")
    lines.append("| Strategy | " + " | ".join(DATASET_DISPLAY[ds] for ds in DATASETS) + " | Avg |")
    lines.append("|----------|" + "-------:|" * len(DATASETS) + "-------:|")
    
    for strategy in GENERATIVE_STRATEGIES:
        row = f"| {strategy} |"
        values = []
        for ds in DATASETS:
            miou = status_matrix[strategy][ds].get('miou')
            if miou is not None:
                row += f" {miou:.1f} |"
                values.append(miou)
            elif status_matrix[strategy][ds]['status'] == 'skip':
                row += " - |"
            else:
                row += " ⏳ |"
        # Add average
        if values:
            avg = sum(values) / len(values)
            row += f" {avg:.1f} |"
        else:
            row += " - |"
        lines.append(row)
    
    # Standard strategies mIoU table
    lines.append("\n### Standard Augmentation Strategies\n")
    lines.append("| Strategy | " + " | ".join(DATASET_DISPLAY[ds] for ds in DATASETS) + " | Avg |")
    lines.append("|----------|" + "-------:|" * len(DATASETS) + "-------:|")
    
    for strategy in STANDARD_STRATEGIES:
        row = f"| {strategy} |"
        values = []
        for ds in DATASETS:
            miou = status_matrix[strategy][ds].get('miou')
            if miou is not None:
                row += f" {miou:.1f} |"
                values.append(miou)
            elif status_matrix[strategy][ds]['status'] == 'skip':
                row += " - |"
            else:
                row += " ⏳ |"
        # Add average
        if values:
            avg = sum(values) / len(values)
            row += f" {avg:.1f} |"
        else:
            row += " - |"
        lines.append(row)
    
    # Test result status matrix
    lines.append("\n---\n")
    lines.append("\n## Test Result Status Matrix\n")
    lines.append("\n### Legend")
    lines.append("- ✅ Test results available (mIoU extracted)")
    lines.append("- 🔄 Test in progress")
    lines.append("- ⏳ Pending test/retest")
    lines.append("- ❌ Test failed (path issue, awaiting retest)")
    lines.append("- ➖ Not applicable (no trained model)\n")
    
    # Generative strategies status table
    lines.append("\n### Generative Strategies Status\n")
    lines.append("| Strategy | " + " | ".join(DATASET_DISPLAY[ds] for ds in DATASETS) + " |")
    lines.append("|----------|" + "--------|" * len(DATASETS))
    
    for strategy in GENERATIVE_STRATEGIES:
        row = f"| {strategy} |"
        for ds in DATASETS:
            emoji = status_matrix[strategy][ds]['emoji']
            row += f" {emoji} |"
        lines.append(row)
    
    # Standard strategies status table
    lines.append("\n### Standard Strategies Status\n")
    lines.append("| Strategy | " + " | ".join(DATASET_DISPLAY[ds] for ds in DATASETS) + " |")
    lines.append("|----------|" + "--------|" * len(DATASETS))
    
    for strategy in STANDARD_STRATEGIES:
        row = f"| {strategy} |"
        for ds in DATASETS:
            emoji = status_matrix[strategy][ds]['emoji']
            row += f" {emoji} |"
        lines.append(row)
    
    # Summary
    lines.append("\n---\n")
    lines.append("\n## Test Result Summary\n")
    lines.append("| Dataset | Complete | Running | Pending | Skip |")
    lines.append("|---------|----------|---------|---------|------|")
    for ds in DATASETS:
        complete = summary[ds].get('complete', 0) + summary[ds].get('complete_detailed', 0)
        running = summary[ds].get('running', 0)
        pending = summary[ds].get('pending', 0)
        skip = sum(1 for s in ALL_STRATEGIES if (s, ds) in SKIP_COMBOS)
        lines.append(f"| {DATASET_DISPLAY[ds]} | {complete} | {running} | {pending} | {skip} |")
    
    # Job management section
    lines.append("\n---\n")
    lines.append("\n## Job Management\n")
    lines.append("\n### Check Test Job Status")
    lines.append("```bash")
    lines.append("# List all retest jobs")
    lines.append("bjobs -u ${USER} | grep retest")
    lines.append("")
    lines.append("# Count by status")
    lines.append("bjobs -u ${USER} -o \"JOB_NAME STAT\" | grep retest | awk '{print $2}' | sort | uniq -c")
    lines.append("```")
    
    lines.append("\n### Submit Retest Jobs")
    lines.append("```bash")
    lines.append("cd scripts/retest_jobs_lsf")
    lines.append("bash submit_all_retests.sh")
    lines.append("```\n")
    
    # Add per-model breakdown
    per_model_results = load_per_model_results()
    if per_model_results:
        lines.append("\n---\n")
        lines.append("\n## Per-Model Performance Breakdown\n")
        lines.append("\nThis section shows average mIoU per model architecture to help select which models to focus on for ratio ablation.\n")
        
        # Build summary table - per_model_results is {dataset: {model: {'avg': float, 'count': int}}}
        lines.append("\n### Model Summary (Average mIoU)")
        lines.append("")
        header = "| Model |"
        divider = "|-------|"
        for ds in DATASETS:
            header += f" {DATASET_DISPLAY[ds]} |"
            divider += "------:|"
        header += " Average |"
        divider += "--------:|"
        lines.append(header)
        lines.append(divider)
        
        # Calculate overall averages per model
        model_overall = {}  # model -> list of averages across datasets
        for model in BASE_MODELS:
            model_overall[model] = []
            for ds in DATASETS:
                if ds in per_model_results and model in per_model_results[ds]:
                    model_overall[model].append(per_model_results[ds][model]['avg'])
        
        # Sort by overall average (descending)
        model_avgs = []
        for model in BASE_MODELS:
            if model_overall[model]:
                avg = sum(model_overall[model]) / len(model_overall[model])
                model_avgs.append((model, avg))
        model_avgs.sort(key=lambda x: x[1], reverse=True)
        
        # Output rows
        for model, overall_avg in model_avgs:
            display_name = MODEL_DISPLAY.get(model, model)
            row = f"| {display_name} |"
            for ds in DATASETS:
                if ds in per_model_results and model in per_model_results[ds]:
                    avg = per_model_results[ds][model]['avg']
                    row += f" {avg:.2f} |"
                else:
                    row += " - |"
            row += f" **{overall_avg:.2f}** |"
            lines.append(row)
        
        # Add recommendation
        if len(model_avgs) >= 2:
            lines.append("")
            lines.append("### Recommendation for Ratio Ablation")
            lines.append("")
            lines.append(f"Based on average mIoU performance, recommended models for ratio ablation:")
            best_model, best_avg = model_avgs[0]
            second_model, second_avg = model_avgs[1]
            lines.append(f"1. **{MODEL_DISPLAY.get(best_model, best_model)}** ({best_model}) - avg: {best_avg:.2f}")
            lines.append(f"2. **{MODEL_DISPLAY.get(second_model, second_model)}** ({second_model}) - avg: {second_avg:.2f}")
            lines.append("")
            lines.append("To generate ratio ablation jobs with only these models:")
            lines.append("```bash")
            lines.append(f"python scripts/generate_ratio_ablation_jobs.py --models {best_model} {second_model}")
            lines.append("```")
    
    # Write to file
    content = '\n'.join(lines)
    TRACKER_PATH.write_text(content)
    print(f"Tracker updated: {TRACKER_PATH}")


def get_job_match_key(strategy, dataset, model):
    """Generate the expected job name pattern for matching.
    
    Matches the format: fg_{short_strategy}_{short_dataset}_{short_model}
    where short_strategy = strategy with 'gen_' replaced by 'g', then first 10 chars
    """
    # Match the job naming in submit_all_remaining_tests.sh
    short_strategy = strategy.replace('gen_', 'g')[:10]
    short_dataset = dataset[:4]
    
    # Model short name (3 chars of display name)
    if 'deeplabv3plus' in model:
        short_model = 'dee'
    elif 'pspnet' in model:
        short_model = 'psp'
    elif 'segformer' in model:
        short_model = 'seg'
    else:
        short_model = model[:3]
    
    return f"fg_{short_strategy}_{short_dataset}_{short_model}".lower()


def job_matches_config(job_name, strategy, dataset, model):
    """Check if a job name matches a config using flexible matching.
    
    Matches various job naming conventions used across different submit scripts.
    """
    job_lower = job_name.lower()
    
    # Check for the exact key first
    key = get_job_match_key(strategy, dataset, model)
    if key in job_lower or job_lower in key:
        return True
    
    # More flexible matching: check if key parts are present
    short_strategy = strategy.replace('gen_', 'g')[:8]  # Slightly shorter for flexibility
    short_dataset = dataset[:4]
    
    if 'deeplabv3plus' in model:
        short_model = 'dee'
    elif 'pspnet' in model:
        short_model = 'psp'
    elif 'segformer' in model:
        short_model = 'seg'
    else:
        short_model = model[:3]
    
    # Check if all parts are present in the job name
    if (short_strategy in job_lower and 
        short_dataset in job_lower and 
        short_model in job_lower and
        ('fg_' in job_lower or 'fg2_' in job_lower or 'test_' in job_lower)):
        return True
    
    return False


def get_per_model_test_status():
    """Get detailed per-model test status with mIoU values.
    
    Returns:
        dict: {(strategy, dataset, model): {'status': str, 'miou': float or None, 'is_valid': bool}}
    """
    import json
    
    results = {}
    
    # Also get running test jobs
    running_jobs = set()
    pending_jobs = set()
    try:
        result = subprocess.run(
            ['bjobs', '-u', '${USER}', '-o', 'JOB_NAME STAT'],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.strip().split('\n')[1:]:
            parts = line.split()
            if len(parts) >= 2:
                job_name, stat = parts[0], parts[1]
                if 'fg_' in job_name or 'fg2_' in job_name or 'retest' in job_name:
                    if stat == 'RUN':
                        running_jobs.add(job_name.lower())
                    elif stat == 'PEND':
                        pending_jobs.add(job_name.lower())
    except:
        pass
    
    # Filter strategies based on current stage
    strategies_to_check = ALL_STRATEGIES
    if CURRENT_STAGE == 'cityscapes-gen':
        strategies_to_check = [s for s in ALL_STRATEGIES if s not in CG_EXCLUDED_STRATEGIES]
    
    # Scan all model directories
    for strategy in strategies_to_check:
        for dataset in DATASETS:
            dataset_dir, test_subdir, fallback_subdir = resolve_dataset_paths(dataset)
            strategy_path = WEIGHTS_ROOT / strategy / dataset_dir
            
            if not safe_exists(strategy_path):
                continue
            
            for model_dir in safe_iterdir(strategy_path):
                if not safe_is_dir(model_dir) or model_dir.name.endswith('_backup'):
                    continue
                
                model = model_dir.name
                
                # Check if weights exist (iter_10000.pth for Stage 1, iter_80000.pth for Stage 2)
                weights_path = model_dir / 'iter_10000.pth'
                if not safe_exists(weights_path):
                    weights_path = model_dir / 'iter_80000.pth'  # Fallback for Stage 2
                if not safe_exists(weights_path):
                    results[(strategy, dataset, model)] = {
                        'status': 'no_weights',
                        'miou': None,
                        'is_valid': False
                    }
                    continue
                
                # Check for test results
                test_path = model_dir / test_subdir
                if not safe_exists(test_path) and fallback_subdir:
                    test_path = model_dir / fallback_subdir
                
                if not safe_exists(test_path):
                    # No test results - check if job is running
                    status = 'missing'
                    
                    for job in running_jobs:
                        if job_matches_config(job, strategy, dataset, model):
                            status = 'running'
                            break
                    if status == 'missing':
                        for job in pending_jobs:
                            if job_matches_config(job, strategy, dataset, model):
                                status = 'pending'
                                break
                    
                    results[(strategy, dataset, model)] = {
                        'status': status,
                        'miou': None,
                        'is_valid': False
                    }
                    continue
                
                # Find latest result
                result_dirs = [d for d in safe_iterdir(test_path) 
                              if safe_is_dir(d) and d.name.startswith('2026')]
                if not result_dirs:
                    results[(strategy, dataset, model)] = {
                        'status': 'empty',
                        'miou': None,
                        'is_valid': False
                    }
                    continue
                
                latest = sorted(result_dirs, key=lambda x: x.name)[-1]
                results_json = latest / 'results.json'
                
                if not safe_exists(results_json):
                    # Check if a test job is running/pending for this
                    status = 'no_json'
                    for job in running_jobs:
                        if job_matches_config(job, strategy, dataset, model):
                            status = 'running'
                            break
                    if status == 'no_json':
                        for job in pending_jobs:
                            if job_matches_config(job, strategy, dataset, model):
                                status = 'pending'
                                break
                    results[(strategy, dataset, model)] = {
                        'status': status,
                        'miou': None,
                        'is_valid': False
                    }
                    continue
                
                try:
                    with open(results_json) as f:
                        data = json.load(f)
                    
                    # Check if checkpoint_path matches the expected weights root (Stage verification)
                    config = data.get('config', {})
                    checkpoint_path = config.get('checkpoint_path', '')
                    weights_root_str = str(WEIGHTS_ROOT)
                    
                    # For Stage 2, checkpoint should point to WEIGHTS_STAGE_2
                    # For Stage 1, checkpoint should point to WEIGHTS (not WEIGHTS_STAGE_2)
                    if checkpoint_path and weights_root_str not in checkpoint_path:
                        # Stale test - checkpoint doesn't match current stage
                        results[(strategy, dataset, model)] = {
                            'status': 'stale',
                            'miou': data.get('overall', {}).get('mIoU'),
                            'is_valid': False,
                            'checkpoint_path': checkpoint_path
                        }
                        continue
                    
                    miou = data.get('overall', {}).get('mIoU')
                    
                    if miou is None or miou < 5:  # Buggy test threshold
                        # Check if retest job is running/pending
                        retest_status = 'buggy'
                        for job in running_jobs:
                            if job_matches_config(job, strategy, dataset, model):
                                retest_status = 'retest_running'
                                break
                        if retest_status == 'buggy':
                            for job in pending_jobs:
                                if job_matches_config(job, strategy, dataset, model):
                                    retest_status = 'retest_pending'
                                    break
                        results[(strategy, dataset, model)] = {
                            'status': retest_status,
                            'miou': miou,
                            'is_valid': False
                        }
                    else:
                        results[(strategy, dataset, model)] = {
                            'status': 'complete',
                            'miou': miou,
                            'is_valid': True
                        }
                except Exception as e:
                    results[(strategy, dataset, model)] = {
                        'status': 'error',
                        'miou': None,
                        'is_valid': False
                    }
    
    return results


def generate_testing_coverage(coverage_path=None):
    """Generate detailed TESTING_COVERAGE.md report.
    
    Args:
        coverage_path: Path to write coverage report. Defaults to COVERAGE_PATH.
    """
    from datetime import datetime
    
    if coverage_path is None:
        coverage_path = COVERAGE_PATH
    
    per_model_status = get_per_model_test_status()
    
    # Count statistics
    total_complete = 0
    total_running = 0
    total_pending = 0
    total_buggy = 0
    total_missing = 0
    total_no_weights = 0
    total_stale = 0
    
    per_dataset_stats = {ds: {
        'complete': 0, 'running': 0, 'pending': 0, 
        'buggy': 0, 'missing': 0, 'no_weights': 0, 'stale': 0, 'total': 0
    } for ds in DATASETS}
    
    running_configs = []
    pending_configs = []
    buggy_configs = []
    missing_configs = []
    stale_configs = []
    
    for (strategy, dataset, model), info in per_model_status.items():
        # Skip non-standard models (ratio ablation, backups)
        if 'ratio0p' in model and not model.endswith('ratio0p50'):
            continue
        if '_backup' in model:
            continue
        
        status = info['status']
        miou = info['miou']
        
        per_dataset_stats[dataset]['total'] += 1
        
        if status == 'complete':
            total_complete += 1
            per_dataset_stats[dataset]['complete'] += 1
        elif status == 'running':
            total_running += 1
            per_dataset_stats[dataset]['running'] += 1
            running_configs.append((strategy, dataset, model))
        elif status == 'pending':
            total_pending += 1
            per_dataset_stats[dataset]['pending'] += 1
            pending_configs.append((strategy, dataset, model))
        elif status == 'buggy':
            total_buggy += 1
            per_dataset_stats[dataset]['buggy'] += 1
            buggy_configs.append((strategy, dataset, model, miou))
        elif status == 'stale':
            total_stale += 1
            per_dataset_stats[dataset]['stale'] += 1
            stale_configs.append((strategy, dataset, model, info.get('checkpoint_path', 'unknown')))
        elif status == 'no_weights':
            total_no_weights += 1
            per_dataset_stats[dataset]['no_weights'] += 1
        else:  # missing, empty, no_json, error
            total_missing += 1
            per_dataset_stats[dataset]['missing'] += 1
            missing_configs.append((strategy, dataset, model, status))
    
    total = total_complete + total_running + total_pending + total_buggy + total_missing + total_stale
    
    # Generate markdown
    lines = []
    lines.append("# Testing Coverage Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    
    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append("| Status | Count | Percentage |")
    lines.append("|--------|------:|----------:|")
    if total > 0:
        lines.append(f"| ✅ Complete (valid mIoU) | {total_complete} | {100*total_complete/total:.1f}% |")
        lines.append(f"| 🔄 Running | {total_running} | {100*total_running/total:.1f}% |")
        lines.append(f"| ⏳ Pending (in queue) | {total_pending} | {100*total_pending/total:.1f}% |")
        lines.append(f"| ⚠️ Buggy (mIoU < 5%) | {total_buggy} | {100*total_buggy/total:.1f}% |")
        lines.append(f"| 🔃 Stale (wrong checkpoint) | {total_stale} | {100*total_stale/total:.1f}% |")
        lines.append(f"| ❌ Missing (no results) | {total_missing} | {100*total_missing/total:.1f}% |")
        lines.append(f"| **Total** | **{total}** | **100%** |")
    else:
        lines.append(f"| ✅ Complete (valid mIoU) | {total_complete} | N/A |")
        lines.append(f"| 🔄 Running | {total_running} | N/A |")
        lines.append(f"| ⏳ Pending (in queue) | {total_pending} | N/A |")
        lines.append(f"| ⚠️ Buggy (mIoU < 5%) | {total_buggy} | N/A |")
        lines.append(f"| ❌ Missing (no results) | {total_missing} | N/A |")
        lines.append(f"| ℹ️ No weights trained yet | {total_no_weights} | N/A |")
        lines.append(f"| **Total** | **0** | **N/A** |")
    lines.append("")
    
    # Per-dataset breakdown
    lines.append("## Per-Dataset Breakdown")
    lines.append("")
    
    for ds in DATASETS:
        stats = per_dataset_stats[ds]
        ds_total = stats['total']
        if ds_total == 0:
            continue
        
        lines.append(f"### {DATASET_DISPLAY[ds]}")
        lines.append(f"- Complete: {stats['complete']}/{ds_total} ({100*stats['complete']/ds_total:.1f}%)")
        lines.append(f"- Running: {stats['running']}")
        lines.append(f"- Pending (in queue): {stats['pending']}")
        lines.append(f"- Buggy (mIoU < 5%): {stats['buggy']}")
        lines.append(f"- Missing (no results): {stats['missing']}")
        lines.append("")
    
    # Running configurations
    lines.append("## Running Configurations")
    lines.append("")
    if running_configs:
        lines.append("| Strategy | Dataset | Model |")
        lines.append("|----------|---------|-------|")
        for strategy, dataset, model in sorted(running_configs):
            model_display = MODEL_DISPLAY.get(model.replace('_ratio0p50', ''), model)
            lines.append(f"| {strategy} | {dataset} | {model_display} |")
    else:
        lines.append("*No test jobs currently running.*")
    lines.append("")
    
    # Pending configurations
    lines.append("## Pending Configurations (in queue)")
    lines.append("")
    if pending_configs:
        lines.append("| Strategy | Dataset | Model |")
        lines.append("|----------|---------|-------|")
        for strategy, dataset, model in sorted(pending_configs)[:30]:  # Limit to 30
            model_display = MODEL_DISPLAY.get(model.replace('_ratio0p50', ''), model)
            lines.append(f"| {strategy} | {dataset} | {model_display} |")
        if len(pending_configs) > 30:
            lines.append(f"| ... | ... | ... ({len(pending_configs)-30} more) |")
    else:
        lines.append("*No configurations pending in queue.*")
    lines.append("")
    
    # Buggy configurations (need retesting)
    lines.append("## Buggy Configurations (need retesting)")
    lines.append("")
    if buggy_configs:
        lines.append("These configurations have test results with mIoU < 5%, indicating a bug in the test.")
        lines.append("")
        lines.append("| Strategy | Dataset | Model | mIoU |")
        lines.append("|----------|---------|-------|-----:|")
        for strategy, dataset, model, miou in sorted(buggy_configs):
            model_display = MODEL_DISPLAY.get(model.replace('_ratio0p50', ''), model)
            miou_str = f"{miou:.2f}%" if miou else "N/A"
            lines.append(f"| {strategy} | {dataset} | {model_display} | {miou_str} |")
    else:
        lines.append("*No buggy configurations - all tests have valid mIoU.*")
    lines.append("")
    
    # Stale configurations (tested with wrong checkpoint)
    lines.append("## Stale Configurations (need retesting)")
    lines.append("")
    if stale_configs:
        lines.append("These tests were run against checkpoints from a **different stage**. They need to be re-run against the correct checkpoints.")
        lines.append("")
        lines.append("| Strategy | Dataset | Model | Checkpoint Used |")
        lines.append("|----------|---------|-------|-----------------|")
        for strategy, dataset, model, ckpt_path in sorted(stale_configs)[:50]:
            model_display = MODEL_DISPLAY.get(model.replace('_ratio0p50', ''), model)
            # Shorten checkpoint path for display
            if 'WEIGHTS_STAGE_2' in ckpt_path:
                ckpt_short = "Stage 2 ✅"
            elif 'WEIGHTS/' in ckpt_path:
                ckpt_short = "Stage 1 ⚠️"
            else:
                ckpt_short = ckpt_path[-50:] if len(ckpt_path) > 50 else ckpt_path
            lines.append(f"| {strategy} | {dataset} | {model_display} | {ckpt_short} |")
        if len(stale_configs) > 50:
            lines.append(f"| ... | ... | ... | ({len(stale_configs)-50} more) |")
    else:
        lines.append("*No stale configurations - all tests use correct checkpoints.*")
    lines.append("")
    
    # Missing configurations
    lines.append("## Missing Configurations (no test results)")
    lines.append("")
    if missing_configs:
        lines.append("| Strategy | Dataset | Model | Issue |")
        lines.append("|----------|---------|-------|-------|")
        for strategy, dataset, model, issue in sorted(missing_configs)[:30]:
            model_display = MODEL_DISPLAY.get(model.replace('_ratio0p50', ''), model)
            lines.append(f"| {strategy} | {dataset} | {model_display} | {issue} |")
        if len(missing_configs) > 30:
            lines.append(f"| ... | ... | ... | ({len(missing_configs)-30} more) |")
    else:
        lines.append("*No missing configurations.*")
    lines.append("")
    
    # Complete configurations matrix
    lines.append("## Complete Configurations")
    lines.append("")
    lines.append("| Strategy | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k |")
    lines.append("|----------|--------|--------|-----------------|------------|")
    
    for strategy in ALL_STRATEGIES:
        row = [strategy]
        for ds in DATASETS:
            models_complete = []
            for (s, d, m), info in per_model_status.items():
                if s == strategy and d == ds and info['status'] == 'complete':
                    # Get short model name
                    if 'deeplabv3plus' in m:
                        models_complete.append('DLV3+')
                    elif 'pspnet' in m:
                        models_complete.append('PSP')
                    elif 'segformer' in m:
                        models_complete.append('SF')
            
            if models_complete:
                row.append('✅ ' + ', '.join(sorted(set(models_complete))))
            else:
                # Check if running/pending
                has_running = any(s == strategy and d == ds and info['status'] == 'running' 
                                 for (s, d, m), info in per_model_status.items())
                if has_running:
                    row.append('🔄')
                else:
                    row.append('⏳')
        
        lines.append('| ' + ' | '.join(row) + ' |')
    
    lines.append("")
    
    # Write to file using the passed coverage_path
    content = '\n'.join(lines)
    coverage_path.write_text(content)
    print(f"Coverage report updated: {coverage_path}")
    
    return {
        'complete': total_complete,
        'running': total_running,
        'pending': total_pending,
        'buggy': total_buggy,
        'stale': total_stale,
        'missing': total_missing,
        'total': total
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Update testing progress tracker')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed status')
    parser.add_argument('--coverage-only', action='store_true', help='Only generate TESTING_COVERAGE.md')
    parser.add_argument('--stage', type=str, choices=['1', '2', 'cityscapes-gen', 'all'], default='1',
                       help='Stage to check (1=clear_day training, 2=all_domains training, cityscapes-gen, all=run all stages)')
    args = parser.parse_args()
    
    all_stages = ['1', '2', 'cityscapes-gen']
    stages = all_stages if args.stage == 'all' else [args.stage]
    
    for stage in stages:
        if len(stages) > 1:
            print(f"\n{'#' * 70}")
            print(f"# STAGE {stage.upper()}")
            print(f"{'#' * 70}")
        run_stage(stage, verbose=args.verbose, coverage_only=args.coverage_only)


def run_stage(stage, verbose=False, coverage_only=False):
    """Run testing tracker for a single stage."""
    # Declare global variables at the start of the function
    global WEIGHTS_ROOT, TRACKER_PATH, DATASETS, CURRENT_STAGE
    global ALL_STRATEGIES, GENERATIVE_STRATEGIES, STANDARD_STRATEGIES
    
    # Set current stage for strategy filtering
    CURRENT_STAGE = stage
    
    # Apply stage-specific strategy exclusions
    if stage == 'cityscapes-gen':
        GENERATIVE_STRATEGIES = [s for s in GENERATIVE_STRATEGIES if s not in CG_EXCLUDED_STRATEGIES]
        STANDARD_STRATEGIES = [s for s in STANDARD_STRATEGIES if s not in CG_EXCLUDED_STRATEGIES]
        ALL_STRATEGIES = GENERATIVE_STRATEGIES + STANDARD_STRATEGIES
    
    # Select paths based on stage
    if stage == 'cityscapes-gen':
        weights_root = WEIGHTS_ROOT_CITYSCAPES_GEN
        tracker_path = TRACKER_PATH_CITYSCAPES_GEN
        coverage_path = COVERAGE_PATH_CITYSCAPES_GEN
        datasets = DATASETS_CITYSCAPES_GEN
        print(f"Cityscapes-gen mode: Using {weights_root}")
    elif stage == '2':
        weights_root = WEIGHTS_ROOT_STAGE2
        tracker_path = TRACKER_PATH_STAGE2
        coverage_path = COVERAGE_PATH_STAGE2
        datasets = DATASETS_DEFAULT
        print(f"Stage 2 mode: Using {weights_root}")
    else:
        weights_root = Path(os.environ.get('PROVE_WEIGHTS_ROOT', '${AWARE_DATA_ROOT}/WEIGHTS'))
        tracker_path = PROJECT_ROOT / 'docs' / 'TESTING_TRACKER.md'
        coverage_path = COVERAGE_PATH
        datasets = DATASETS_DEFAULT
        print(f"Stage 1 mode: Using {weights_root}")
    
    # Override global variables for functions that use them
    WEIGHTS_ROOT = weights_root
    TRACKER_PATH = tracker_path
    DATASETS = datasets
    
    print("\nCollecting test status...")
    status_matrix, summary, retest_jobs, job_counts = collect_test_status(verbose=verbose)
    
    if not coverage_only:
        print(f"\nUpdating {tracker_path.name}...")
        update_tracker(status_matrix, summary, retest_jobs, job_counts)
    
    # Generate detailed coverage report
    print(f"\nGenerating {coverage_path.name}...")
    coverage_stats = generate_testing_coverage(coverage_path=coverage_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TESTING COVERAGE SUMMARY")
    print("=" * 60)
    
    print(f"\n✅ Complete (valid mIoU): {coverage_stats['complete']}")
    print(f"🔄 Running: {coverage_stats['running']}")
    print(f"⏳ Pending: {coverage_stats['pending']}")
    print(f"⚠️ Buggy (mIoU < 5%): {coverage_stats['buggy']}")
    print(f"🔃 Stale (wrong checkpoint): {coverage_stats['stale']}")
    print(f"❌ Missing: {coverage_stats['missing']}")
    print(f"\nTotal: {coverage_stats['total']}")
    
    # Print job counts from queue
    total_jobs = sum(sum(v.values()) for v in job_counts.values())
    if total_jobs > 0:
        print(f"\nRetest Jobs in Queue: {total_jobs}")
        for ds in DATASETS:
            if ds in job_counts:
                running = job_counts[ds].get('RUN', 0)
                pending = job_counts[ds].get('PEND', 0)
                if running + pending > 0:
                    print(f"  {DATASET_DISPLAY[ds]}: {running} running, {pending} pending")
    
    print("\nTest Result Status (by strategy/dataset combination):")
    for ds in DATASETS:
        complete = summary[ds].get('complete', 0) + summary[ds].get('complete_detailed', 0)
        running = summary[ds].get('running', 0)
        pending = summary[ds].get('pending', 0)
        print(f"\n{DATASET_DISPLAY[ds]}:")
        print(f"  Complete: {complete}")
        print(f"  Running:  {running}")
        print(f"  Pending:  {pending}")


if __name__ == '__main__':
    main()
