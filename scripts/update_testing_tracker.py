#!/usr/bin/env python3
"""
Update the testing progress tracker based on current job status and test results.

Reads results directly from the weights folder (test_results_detailed_fixed/*/results.json)
to get the latest mIoU values without needing to run the test_result_analyzer first.

Usage:
    python scripts/update_testing_tracker.py
    python scripts/update_testing_tracker.py --verbose   # Show all status details
"""

import os
import subprocess
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WEIGHTS_ROOT = Path(os.environ.get('PROVE_WEIGHTS_ROOT', '/scratch/aaa_exchange/AWARE/WEIGHTS'))
TRACKER_PATH = PROJECT_ROOT / 'docs' / 'TESTING_TRACKER.md'
TEST_RESULTS_CSV = PROJECT_ROOT / 'test_results_summary.csv'

DATASETS = ['bdd10k', 'idd-aw', 'mapillaryvistas', 'outside15k']
DATASET_DISPLAY = {
    'bdd10k': 'BDD10k',
    'idd-aw': 'IDD-AW',
    'mapillaryvistas': 'MapillaryVistas',
    'outside15k': 'OUTSIDE15k',
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
    'photometric_distort',
    'std_autoaugment',
    'std_cutmix',
    'std_mixup',
    'std_randaugment',
]

ALL_STRATEGIES = GENERATIVE_STRATEGIES + STANDARD_STRATEGIES

# Skip combinations (no data available)
# NOTE: This list should be empty now that most strategies have full 4/4 coverage
# Only add combinations here if training data is truly unavailable
SKIP_COMBOS = set()  # All strategies now have full dataset coverage

# Models for per-model breakdown
BASE_MODELS = ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5']
MODEL_DISPLAY = {
    'deeplabv3plus_r50': 'DeepLabV3+',
    'pspnet_r50': 'PSPNet',
    'segformer_mit-b5': 'SegFormer',
}


def load_per_model_results():
    """Load per-model mIoU results from test_results_summary.csv.
    
    Returns:
        dict: {dataset: {model: {'avg': float, 'count': int}}}
    """
    import pandas as pd
    
    csv_path = TEST_RESULTS_CSV
    if not csv_path.exists():
        return {}
    
    try:
        df = pd.read_csv(csv_path)
        
        # Filter to clear_day and fixed results
        df = df[df['dataset'].str.endswith('_cd')]
        df = df[df['test_type'] == 'test_results_detailed_fixed']
        df = df.dropna(subset=['mIoU'])
        
        # Normalize model names (remove ratio suffix)
        df['base_model'] = df['model'].str.replace('_ratio0p50', '', regex=False)
        
        results = {}
        for ds in DATASETS:
            ds_cd = f"{ds}_cd"
            ds_data = df[df['dataset'] == ds_cd]
            
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


def load_miou_results():
    """Load mIoU results directly from the weights folder.
    
    Scans all strategy/dataset/model directories for test_results_detailed_fixed
    and extracts mIoU from results.json files.
    
    Returns:
        dict: {(strategy, dataset): {'best_miou': float, 'best_model': str, 'models': {model: miou}}}
    """
    import json
    
    results = {}
    
    # Scan all strategies
    for strategy in ALL_STRATEGIES:
        for dataset in DATASETS:
            dataset_dir = f"{dataset}_cd"  # Add _cd suffix for directory
            strategy_path = WEIGHTS_ROOT / strategy / dataset_dir
            
            if not strategy_path.exists():
                continue
            
            # Get mIoU values per model
            models = {}
            for model_dir in strategy_path.iterdir():
                if not model_dir.is_dir() or model_dir.name.endswith('_backup'):
                    continue
                
                # Look for test_results_detailed_fixed
                fixed_path = model_dir / 'test_results_detailed_fixed'
                if not fixed_path.exists():
                    continue
                
                # Find the latest result directory (by timestamp)
                result_dirs = [d for d in fixed_path.iterdir() 
                              if d.is_dir() and d.name.startswith('2026')]
                if not result_dirs:
                    continue
                
                latest = sorted(result_dirs, key=lambda x: x.name)[-1]
                results_json = latest / 'results.json'
                
                if not results_json.exists():
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
            ['bjobs', '-u', 'mima2416', '-a', '-o', 'JOBID JOB_NAME STAT'],
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
                    # Parse job name: retest_<strategy>_<dataset>_cd_<model>
                    # Find dataset
                    dataset = None
                    strategy = None
                    
                    for ds in DATASETS:
                        ds_pattern = f'_{ds}_cd' if ds != 'idd-aw' else '_idd-aw_cd'
                        if ds_pattern in job_name:
                            dataset = ds
                            # Extract strategy
                            start_idx = job_name.find('retest_') + len('retest_')
                            end_idx = job_name.find(ds_pattern)
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


def check_test_results(strategy, dataset, test_dir='test_results_detailed_fixed'):
    """Check if test results exist for a strategy/dataset combination."""
    # Dataset directory pattern (with _cd suffix)
    dataset_dir = f"{dataset}_cd"
    
    # Check for any model's test results
    strategy_path = WEIGHTS_ROOT / strategy / dataset_dir
    if not strategy_path.exists():
        return False, False
    
    has_results = False
    has_detailed = False
    
    for model_dir in strategy_path.iterdir():
        if model_dir.is_dir():
            test_path = model_dir / test_dir
            if test_path.exists():
                # Check for any timestamped results
                result_dirs = list(test_path.glob('*/'))
                if result_dirs:
                    has_results = True
                    # Check for detailed results
                    for result_dir in result_dirs:
                        if (result_dir / 'per_domain_results.json').exists():
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
    
    status_matrix = {}
    summary = defaultdict(lambda: defaultdict(int))  # summary[dataset][status] = count
    
    for strategy in ALL_STRATEGIES:
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
    lines.append("bjobs -u mima2416 | grep retest")
    lines.append("")
    lines.append("# Count by status")
    lines.append("bjobs -u mima2416 -o \"JOB_NAME STAT\" | grep retest | awk '{print $2}' | sort | uniq -c")
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Update testing progress tracker')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed status')
    args = parser.parse_args()
    
    print("Collecting test status...")
    status_matrix, summary, retest_jobs, job_counts = collect_test_status(verbose=args.verbose)
    
    print("\nUpdating tracker...")
    update_tracker(status_matrix, summary, retest_jobs, job_counts)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TESTING PROGRESS SUMMARY")
    print("=" * 60)
    
    # Print job counts
    total_jobs = sum(sum(v.values()) for v in job_counts.values())
    if total_jobs > 0:
        print(f"\nRetest Jobs in Queue: {total_jobs}")
        for ds in DATASETS:
            if ds in job_counts:
                running = job_counts[ds].get('RUN', 0)
                pending = job_counts[ds].get('PEND', 0)
                if running + pending > 0:
                    print(f"  {DATASET_DISPLAY[ds]}: {running} running, {pending} pending")
    
    total_complete = 0
    total_running = 0
    total_pending = 0
    
    print("\nTest Result Status (by strategy/dataset combination):")
    for ds in DATASETS:
        complete = summary[ds].get('complete', 0) + summary[ds].get('complete_detailed', 0)
        running = summary[ds].get('running', 0)
        pending = summary[ds].get('pending', 0)
        total_complete += complete
        total_running += running
        total_pending += pending
        print(f"\n{DATASET_DISPLAY[ds]}:")
        print(f"  Complete: {complete}")
        print(f"  Running:  {running}")
        print(f"  Pending:  {pending}")
    
    print(f"\nTOTAL:")
    print(f"  Complete: {total_complete}")
    print(f"  Running:  {total_running}")
    print(f"  Pending:  {total_pending}")


if __name__ == '__main__':
    main()
