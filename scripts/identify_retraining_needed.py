#!/usr/bin/env python3
"""Identify models that need retraining due to incorrect validation configuration.

This script scans all MapillaryVistas models in WEIGHTS/ and WEIGHTS_STAGE_2/
to identify which ones were trained with CityscapesDataset (wrong 19-class validation)
vs MapillaryDataset_v1 (correct 66-class validation).

Models using CityscapesDataset need retraining because:
1. The model has 66 output classes (MapillaryVistas native)
2. But validation was computed using CityscapesDataset's 19-class METAINFO
3. This means training validation metrics (mIoU, etc.) were meaningless

Usage:
    python scripts/identify_retraining_needed.py [--stage 1|2|both] [--output FILE]
"""

import argparse
import os
from pathlib import Path
from datetime import datetime


def find_training_configs(weights_dir: Path) -> list:
    """Find all training_config.py files for MapillaryVistas models."""
    configs = []
    
    # Find all mapillaryvistas directories
    for config_path in weights_dir.glob("**/mapillaryvistas/*/training_config.py"):
        # Skip configs directory
        if '/configs/' in str(config_path):
            continue
        configs.append(config_path)
    
    return configs


def check_config_dataset_type(config_path: Path) -> dict:
    """Check what dataset type a training config uses.
    
    Returns:
        dict with keys: path, dataset_type, needs_retraining, strategy, model
    """
    result = {
        'path': str(config_path),
        'dataset_type': None,
        'needs_retraining': False,
        'strategy': None,
        'model': None,
    }
    
    # Parse path to get strategy and model
    parts = config_path.parts
    for i, part in enumerate(parts):
        if part == 'mapillaryvistas' and i >= 2:
            result['strategy'] = parts[i-1]
            result['model'] = parts[i+1]
            break
    
    # Read config file
    try:
        with open(config_path) as f:
            content = f.read()
        
        # Check for dataset type
        if "dataset_type = 'MapillaryDataset_v1'" in content:
            result['dataset_type'] = 'MapillaryDataset_v1'
            result['needs_retraining'] = False
        elif "dataset_type = 'MapillaryDataset_v2'" in content:
            result['dataset_type'] = 'MapillaryDataset_v2'
            result['needs_retraining'] = False  # v2 has 124 classes, different issue
        elif "dataset_type = 'CityscapesDataset'" in content:
            result['dataset_type'] = 'CityscapesDataset'
            result['needs_retraining'] = True
        else:
            result['dataset_type'] = 'Unknown'
            result['needs_retraining'] = True  # Assume needs checking
    
    except Exception as e:
        result['dataset_type'] = f'Error: {e}'
        result['needs_retraining'] = True
    
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['1', '2', 'both'], default='both',
                       help='Which stage to check (default: both)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results (default: print to stdout)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show details for all models, not just those needing retraining')
    args = parser.parse_args()
    
    # Determine directories to scan
    weights_dirs = []
    if args.stage in ['1', 'both']:
        weights_dirs.append(Path('/scratch/aaa_exchange/AWARE/WEIGHTS'))
    if args.stage in ['2', 'both']:
        weights_dirs.append(Path('/scratch/aaa_exchange/AWARE/WEIGHTS_STAGE_2'))
    
    # Find and check all configs
    all_results = []
    needs_retraining = []
    ok_models = []
    
    for weights_dir in weights_dirs:
        stage = '1' if 'WEIGHTS_STAGE_2' not in str(weights_dir) else '2'
        configs = find_training_configs(weights_dir)
        
        for config_path in configs:
            result = check_config_dataset_type(config_path)
            result['stage'] = stage
            all_results.append(result)
            
            if result['needs_retraining']:
                needs_retraining.append(result)
            else:
                ok_models.append(result)
    
    # Generate report
    lines = []
    lines.append("=" * 80)
    lines.append("MapillaryVistas Model Retraining Assessment")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")
    
    lines.append("## Summary")
    lines.append(f"- Total MapillaryVistas models: {len(all_results)}")
    lines.append(f"- Models needing retraining: {len(needs_retraining)} (CityscapesDataset)")
    lines.append(f"- Models OK: {len(ok_models)} (MapillaryDataset_v1)")
    lines.append("")
    
    if needs_retraining:
        lines.append("## Models Needing Retraining")
        lines.append("")
        lines.append("These models used CityscapesDataset type, which means:")
        lines.append("- Training validation metrics (mIoU) were computed using 19 Cityscapes classes")
        lines.append("- But the model has 66 MapillaryVistas output classes")
        lines.append("- The training loss was correct, only validation metrics were wrong")
        lines.append("")
        
        # Group by stage
        for stage in ['1', '2']:
            stage_models = [r for r in needs_retraining if r['stage'] == stage]
            if stage_models:
                lines.append(f"### Stage {stage}")
                lines.append("")
                
                # Group by strategy
                strategies = sorted(set(r['strategy'] for r in stage_models))
                for strategy in strategies:
                    strategy_models = [r for r in stage_models if r['strategy'] == strategy]
                    models = sorted(set(r['model'] for r in strategy_models))
                    lines.append(f"- **{strategy}**: {', '.join(models)}")
                lines.append("")
    
    if args.verbose and ok_models:
        lines.append("## Models OK (No Retraining Needed)")
        lines.append("")
        for stage in ['1', '2']:
            stage_models = [r for r in ok_models if r['stage'] == stage]
            if stage_models:
                lines.append(f"### Stage {stage}")
                strategies = sorted(set(r['strategy'] for r in stage_models))
                for strategy in strategies:
                    strategy_models = [r for r in stage_models if r['strategy'] == strategy]
                    models = sorted(set(r['model'] for r in strategy_models))
                    lines.append(f"- {strategy}: {', '.join(models)}")
                lines.append("")
    
    # Output
    report = '\n'.join(lines)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report written to: {args.output}")
    else:
        print(report)
    
    # Return exit code based on whether retraining is needed
    return 1 if needs_retraining else 0


if __name__ == '__main__':
    exit(main())
