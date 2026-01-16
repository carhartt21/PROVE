#!/usr/bin/env python3
"""
Find configurations where weights are newer than test results.

This script scans the WEIGHTS directory to find configurations where:
1. iter_80000.pth exists (training complete)
2. Test results exist but are older than the weights
3. Or test results are missing entirely

Usage:
    python scripts/find_outdated_tests.py
    python scripts/find_outdated_tests.py --missing-only  # Only show missing tests
    python scripts/find_outdated_tests.py --outdated-only  # Only show outdated tests
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import argparse

# Configuration
WEIGHTS_ROOT = Path(os.environ.get('PROVE_WEIGHTS_ROOT', '/scratch/aaa_exchange/AWARE/WEIGHTS'))

# Strategies to check
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
    'std_minimal',
    'std_autoaugment',
    'std_cutmix',
    'std_mixup',
    'std_randaugment',
]

ALL_STRATEGIES = GENERATIVE_STRATEGIES + STANDARD_STRATEGIES

DATASETS = ['bdd10k', 'idd-aw', 'mapillaryvistas', 'outside15k']
# Also check alternate naming
DATASET_ALIASES = {
    'idd-aw': ['idd-aw', 'iddaw'],
}

MODELS = {
    'gen': ['deeplabv3plus_r50_ratio0p50', 'pspnet_r50_ratio0p50', 'segformer_mit-b5_ratio0p50'],
    'std': ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5'],
}

def get_latest_test_time(weights_path):
    """Get the modification time of the latest test results."""
    test_dirs = ['test_results_detailed_fixed', 'test_results_detailed']
    
    latest_time = None
    latest_dir = None
    
    for test_dir_name in test_dirs:
        test_dir = weights_path / test_dir_name
        if not test_dir.exists():
            continue
            
        # Find all timestamped subdirectories
        for subdir in test_dir.iterdir():
            if subdir.is_dir():
                results_json = subdir / 'results.json'
                if results_json.exists():
                    mtime = results_json.stat().st_mtime
                    if latest_time is None or mtime > latest_time:
                        latest_time = mtime
                        latest_dir = subdir
    
    return latest_time, latest_dir


def check_configuration(strategy, dataset, model, domain_suffix='_cd'):
    """Check if a configuration has outdated or missing tests."""
    # Try different dataset naming conventions
    dataset_names = DATASET_ALIASES.get(dataset, [dataset])
    
    for ds_name in dataset_names:
        dataset_dir = f'{ds_name}{domain_suffix}'
        weights_path = WEIGHTS_ROOT / strategy / dataset_dir / model
        checkpoint = weights_path / 'iter_80000.pth'
        
        if checkpoint.exists():
            checkpoint_time = checkpoint.stat().st_mtime
            test_time, test_dir = get_latest_test_time(weights_path)
            
            if test_time is None:
                return {
                    'status': 'missing',
                    'strategy': strategy,
                    'dataset': dataset,
                    'model': model,
                    'weights_path': str(weights_path),
                    'checkpoint_time': datetime.fromtimestamp(checkpoint_time),
                    'test_time': None,
                }
            elif checkpoint_time > test_time:
                return {
                    'status': 'outdated',
                    'strategy': strategy,
                    'dataset': dataset,
                    'model': model,
                    'weights_path': str(weights_path),
                    'checkpoint_time': datetime.fromtimestamp(checkpoint_time),
                    'test_time': datetime.fromtimestamp(test_time),
                    'test_dir': str(test_dir),
                }
            else:
                return {
                    'status': 'up-to-date',
                    'strategy': strategy,
                    'dataset': dataset,
                    'model': model,
                }
    
    return None  # Weights don't exist


def main():
    parser = argparse.ArgumentParser(description='Find configurations with outdated tests')
    parser.add_argument('--missing-only', action='store_true', help='Only show missing tests')
    parser.add_argument('--outdated-only', action='store_true', help='Only show outdated tests')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    args = parser.parse_args()
    
    outdated = []
    missing = []
    up_to_date = 0
    no_weights = 0
    
    for strategy in ALL_STRATEGIES:
        # Determine models based on strategy type
        if strategy.startswith('gen_'):
            models = MODELS['gen']
        else:
            models = MODELS['std']
        
        for dataset in DATASETS:
            for model in models:
                result = check_configuration(strategy, dataset, model)
                
                if result is None:
                    no_weights += 1
                elif result['status'] == 'outdated':
                    outdated.append(result)
                elif result['status'] == 'missing':
                    missing.append(result)
                else:
                    up_to_date += 1
    
    # Print results
    if args.json:
        import json
        print(json.dumps({
            'outdated': outdated,
            'missing': missing,
            'summary': {
                'outdated': len(outdated),
                'missing': len(missing),
                'up_to_date': up_to_date,
                'no_weights': no_weights,
            }
        }, indent=2, default=str))
        return
    
    print("=" * 80)
    print("OUTDATED TEST RESULTS ANALYSIS")
    print("=" * 80)
    
    if not args.missing_only and outdated:
        print(f"\n🔄 OUTDATED TESTS ({len(outdated)} configurations):")
        print("-" * 80)
        for config in outdated:
            print(f"  {config['strategy']}/{config['dataset']}/{config['model']}")
            print(f"    Weights: {config['checkpoint_time']}")
            print(f"    Test:    {config['test_time']}")
            print()
    
    if not args.outdated_only and missing:
        print(f"\n⚠️  MISSING TESTS ({len(missing)} configurations):")
        print("-" * 80)
        for config in missing:
            print(f"  {config['strategy']}/{config['dataset']}/{config['model']}")
            print(f"    Weights: {config['checkpoint_time']}")
            print()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  ✅ Up-to-date:    {up_to_date}")
    print(f"  🔄 Outdated:      {len(outdated)}")
    print(f"  ⚠️  Missing tests: {len(missing)}")
    print(f"  ❌ No weights:    {no_weights}")
    print(f"  Total checked:   {up_to_date + len(outdated) + len(missing) + no_weights}")
    
    # Return count for script usage
    return len(outdated) + len(missing)


if __name__ == '__main__':
    sys.exit(main() if main() else 0)
