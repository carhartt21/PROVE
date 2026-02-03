#!/usr/bin/env python3
"""
Generate an overview of available baseline training checkpoints and test results.

This script scans the WEIGHTS directories to find:
- Completed training checkpoints (iter_15000.pth, iter_10000.pth, iter_80000.pth)
- Test results (results.json files)
- Per-model and per-dataset status

Usage:
    python scripts/generate_baseline_overview.py
    python scripts/generate_baseline_overview.py --stage 1
    python scripts/generate_baseline_overview.py --stage 2
    python scripts/generate_baseline_overview.py --output overview.md
    python scripts/generate_baseline_overview.py --json  # Output as JSON
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Configuration
WEIGHTS_ROOT = Path(os.environ.get('PROVE_WEIGHTS_ROOT', '/scratch/aaa_exchange/AWARE/WEIGHTS'))
WEIGHTS_ROOT_STAGE2 = Path(os.environ.get('PROVE_WEIGHTS_ROOT_STAGE2', '/scratch/aaa_exchange/AWARE/WEIGHTS_STAGE_2'))

# Models and their expected checkpoint iterations
MODELS = [
    'deeplabv3plus_r50',
    'pspnet_r50',
    'segformer_mit-b3',
    'segnext_mscan-b',
    'hrnet_hr48',
    'mask2former_swin-b',
]

# Model display names for pretty printing
MODEL_DISPLAY = {
    'deeplabv3plus_r50': 'DeepLabV3+',
    'pspnet_r50': 'PSPNet',
    'segformer_mit-b3': 'SegFormer',
    'segnext_mscan-b': 'SegNeXt',
    'hrnet_hr48': 'HRNet',
    'mask2former_swin-b': 'Mask2Former',
}

# Datasets
DATASETS = ['bdd10k', 'iddaw', 'mapillaryvistas', 'outside15k']

# Dataset display names
DATASET_DISPLAY = {
    'bdd10k': 'BDD10k',
    'iddaw': 'IDD-AW',
    'mapillaryvistas': 'MapillaryVistas',
    'outside15k': 'OUTSIDE15k',
}

# Valid checkpoint iterations (in order of preference)
VALID_CHECKPOINTS = [15000, 10000, 80000, 20000, 40000]


def find_checkpoint(weights_dir):
    """Find the best available checkpoint in a directory.
    
    Returns:
        tuple: (checkpoint_path, iteration_number) or (None, None) if not found
    """
    if not weights_dir.exists():
        return None, None
    
    for iter_num in VALID_CHECKPOINTS:
        ckpt = weights_dir / f"iter_{iter_num}.pth"
        if ckpt.exists() and ckpt.stat().st_size > 1000:
            return ckpt, iter_num
    
    return None, None


def find_test_results(weights_dir):
    """Find test results in a weights directory.
    
    Returns:
        dict: Test results data or None if not found
    """
    # Check for test_results_detailed directory
    test_dirs = list(weights_dir.glob("test_results_detailed/*/results.json"))
    if test_dirs:
        # Get the most recent results
        most_recent = max(test_dirs, key=lambda p: p.stat().st_mtime)
        try:
            with open(most_recent) as f:
                return json.load(f)
        except:
            pass
    
    # Check for results.json directly
    results_file = weights_dir / "results.json"
    if results_file.exists():
        try:
            with open(results_file) as f:
                return json.load(f)
        except:
            pass
    
    return None


def get_checkpoint_status(weights_root, strategy, dataset, model):
    """Get the training and testing status for a specific configuration.
    
    Returns:
        dict: {
            'checkpoint': bool,
            'checkpoint_iter': int or None,
            'checkpoint_path': str or None,
            'has_results': bool,
            'mIoU': float or None,
            'status': str  # 'complete', 'trained', 'pending', 'running', 'failed'
        }
    """
    weights_dir = weights_root / strategy / dataset / model
    
    result = {
        'checkpoint': False,
        'checkpoint_iter': None,
        'checkpoint_path': None,
        'has_results': False,
        'mIoU': None,
        'per_domain_mIoU': None,
        'status': 'pending',
    }
    
    # Check for lock file (running)
    lock_file = weights_dir / ".training.lock"
    if lock_file.exists():
        result['status'] = 'running'
        # Still check if there's a partial checkpoint
        ckpt_path, ckpt_iter = find_checkpoint(weights_dir)
        if ckpt_path:
            result['checkpoint'] = True
            result['checkpoint_iter'] = ckpt_iter
            result['checkpoint_path'] = str(ckpt_path)
            result['status'] = 'running (partial)'
        return result
    
    # Check for checkpoint
    ckpt_path, ckpt_iter = find_checkpoint(weights_dir)
    if ckpt_path:
        result['checkpoint'] = True
        result['checkpoint_iter'] = ckpt_iter
        result['checkpoint_path'] = str(ckpt_path)
        result['status'] = 'trained'
    elif weights_dir.exists():
        # Directory exists but no checkpoint
        result['status'] = 'failed'
    
    # Check for test results
    test_results = find_test_results(weights_dir)
    if test_results:
        result['has_results'] = True
        result['status'] = 'complete' if result['checkpoint'] else 'tested_only'
        
        # Extract mIoU
        if 'overall' in test_results and 'mIoU' in test_results['overall']:
            result['mIoU'] = test_results['overall']['mIoU']
        elif 'mIoU' in test_results:
            result['mIoU'] = test_results['mIoU']
        
        # Extract per-domain mIoU if available
        if 'per_domain' in test_results:
            result['per_domain_mIoU'] = {
                domain: data.get('mIoU') 
                for domain, data in test_results['per_domain'].items()
                if isinstance(data, dict) and 'mIoU' in data
            }
    
    return result


def generate_overview(stage=None, output_format='markdown'):
    """Generate overview of baseline checkpoints and results.
    
    Args:
        stage: 1, 2, or None for both
        output_format: 'markdown' or 'json'
    
    Returns:
        str or dict: Overview in requested format
    """
    results = defaultdict(lambda: defaultdict(dict))
    summary = {
        'total': 0,
        'complete': 0,  # Has checkpoint AND test results
        'trained': 0,   # Has checkpoint but no test results
        'tested_only': 0,  # Has test results but no checkpoint (unlikely)
        'running': 0,
        'failed': 0,
        'pending': 0,
    }
    
    # Determine which stages to check
    stages_to_check = []
    if stage is None or stage == 1:
        stages_to_check.append(('Stage 1', WEIGHTS_ROOT))
    if stage is None or stage == 2:
        stages_to_check.append(('Stage 2', WEIGHTS_ROOT_STAGE2))
    
    for stage_name, weights_root in stages_to_check:
        for dataset in DATASETS:
            for model in MODELS:
                status = get_checkpoint_status(weights_root, 'baseline', dataset, model)
                results[stage_name][dataset][model] = status
                
                summary['total'] += 1
                if status['status'].startswith('running'):
                    summary['running'] += 1
                elif status['status'] == 'complete':
                    summary['complete'] += 1
                elif status['status'] == 'trained':
                    summary['trained'] += 1
                elif status['status'] == 'tested_only':
                    summary['tested_only'] += 1
                elif status['status'] == 'failed':
                    summary['failed'] += 1
                else:
                    summary['pending'] += 1
    
    if output_format == 'json':
        return {
            'summary': summary,
            'results': dict(results),
            'generated': datetime.now().isoformat(),
        }
    
    # Generate markdown
    lines = []
    lines.append("# Baseline Training & Testing Overview")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    
    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append("| Status | Count | Percentage |")
    lines.append("|--------|------:|----------:|")
    lines.append(f"| ✅ Complete (trained + tested) | {summary['complete']} | {100*summary['complete']/summary['total']:.1f}% |")
    lines.append(f"| 🔶 Trained (no test results) | {summary['trained']} | {100*summary['trained']/summary['total']:.1f}% |")
    lines.append(f"| 🔄 Running | {summary['running']} | {100*summary['running']/summary['total']:.1f}% |")
    lines.append(f"| ❌ Failed | {summary['failed']} | {100*summary['failed']/summary['total']:.1f}% |")
    lines.append(f"| ⏳ Pending | {summary['pending']} | {100*summary['pending']/summary['total']:.1f}% |")
    lines.append(f"| **Total** | **{summary['total']}** | **100%** |")
    lines.append("")
    
    # Per-stage breakdown
    for stage_name, stage_results in results.items():
        lines.append(f"## {stage_name} Baseline Status")
        lines.append("")
        
        # Header row
        header = "| Dataset |"
        divider = "|---------|"
        for model in MODELS:
            header += f" {MODEL_DISPLAY[model]} |"
            divider += "----------|"
        lines.append(header)
        lines.append(divider)
        
        # Data rows
        for dataset in DATASETS:
            row = f"| {DATASET_DISPLAY[dataset]} |"
            for model in MODELS:
                status = stage_results[dataset].get(model, {})
                if status.get('status') == 'complete':
                    mIoU = status.get('mIoU')
                    if mIoU:
                        row += f" ✅ {mIoU:.1f}% |"
                    else:
                        row += " ✅ |"
                elif status.get('status') == 'trained':
                    row += f" 🔶 {status.get('checkpoint_iter', '?')}k |"
                elif status.get('status', '').startswith('running'):
                    row += " 🔄 |"
                elif status.get('status') == 'failed':
                    row += " ❌ |"
                else:
                    row += " ⏳ |"
            lines.append(row)
        
        lines.append("")
        
        # Best mIoU per model
        lines.append("### Best mIoU per Model")
        lines.append("")
        for model in MODELS:
            best_miou = None
            best_dataset = None
            for dataset in DATASETS:
                status = stage_results[dataset].get(model, {})
                miou = status.get('mIoU')
                if miou and (best_miou is None or miou > best_miou):
                    best_miou = miou
                    best_dataset = dataset
            
            if best_miou:
                lines.append(f"- **{MODEL_DISPLAY[model]}**: {best_miou:.2f}% ({DATASET_DISPLAY[best_dataset]})")
            else:
                lines.append(f"- **{MODEL_DISPLAY[model]}**: No results yet")
        
        lines.append("")
    
    # Missing configurations
    lines.append("## Missing Baseline Configurations")
    lines.append("")
    missing = []
    for stage_name, stage_results in results.items():
        for dataset in DATASETS:
            for model in MODELS:
                status = stage_results[dataset].get(model, {})
                if status.get('status') in ['pending', 'failed']:
                    missing.append({
                        'stage': stage_name,
                        'dataset': DATASET_DISPLAY[dataset],
                        'model': MODEL_DISPLAY[model],
                        'status': status.get('status'),
                    })
    
    if missing:
        lines.append("| Stage | Dataset | Model | Status |")
        lines.append("|-------|---------|-------|--------|")
        for m in missing:
            status_emoji = '❌' if m['status'] == 'failed' else '⏳'
            lines.append(f"| {m['stage']} | {m['dataset']} | {m['model']} | {status_emoji} {m['status']} |")
    else:
        lines.append("*All baseline configurations are complete or running.*")
    
    lines.append("")
    
    # Training recommendations
    lines.append("## Recommendations")
    lines.append("")
    
    # Find models missing from all datasets
    missing_models = defaultdict(list)
    for stage_name, stage_results in results.items():
        for dataset in DATASETS:
            for model in MODELS:
                status = stage_results[dataset].get(model, {})
                if status.get('status') == 'pending':
                    missing_models[model].append(f"{stage_name}/{DATASET_DISPLAY[dataset]}")
    
    if missing_models:
        lines.append("### Priority Training Jobs")
        lines.append("")
        for model, locations in sorted(missing_models.items(), key=lambda x: -len(x[1])):
            lines.append(f"**{MODEL_DISPLAY[model]}** - Missing from {len(locations)} configurations:")
            for loc in locations[:5]:
                lines.append(f"  - {loc}")
            if len(locations) > 5:
                lines.append(f"  - ... and {len(locations) - 5} more")
            lines.append("")
    
    # Find trained but not tested
    need_testing = []
    for stage_name, stage_results in results.items():
        for dataset in DATASETS:
            for model in MODELS:
                status = stage_results[dataset].get(model, {})
                if status.get('status') == 'trained':
                    need_testing.append(f"{stage_name}/{DATASET_DISPLAY[dataset]}/{MODEL_DISPLAY[model]}")
    
    if need_testing:
        lines.append("### Need Testing")
        lines.append("")
        lines.append("The following configurations have checkpoints but no test results:")
        lines.append("")
        for item in need_testing:
            lines.append(f"- {item}")
        lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate baseline overview")
    parser.add_argument('--stage', type=int, choices=[1, 2], 
                        help='Stage to check (1 or 2, default: both)')
    parser.add_argument('--output', '-o', type=str,
                        help='Output file path (default: stdout)')
    parser.add_argument('--json', action='store_true',
                        help='Output as JSON instead of markdown')
    args = parser.parse_args()
    
    output_format = 'json' if args.json else 'markdown'
    result = generate_overview(stage=args.stage, output_format=output_format)
    
    if args.json:
        output = json.dumps(result, indent=2, default=str)
    else:
        output = result
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Overview saved to: {args.output}")
    else:
        print(output)


if __name__ == '__main__':
    main()
