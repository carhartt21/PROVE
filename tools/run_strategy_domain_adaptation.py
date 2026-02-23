#!/usr/bin/env python3
"""
Run domain adaptation evaluation for a specific strategy.

This script evaluates all models for a given strategy against 
Cityscapes (clear_day) and ACDC adverse weather conditions.

Usage:
    python run_strategy_domain_adaptation.py --strategy gen_NST
    python run_strategy_domain_adaptation.py --strategy gen_NST --include-clearday
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_domain_adaptation import run_evaluation


def main():
    parser = argparse.ArgumentParser(
        description='Run domain adaptation evaluation for a strategy'
    )
    parser.add_argument('--strategy', required=True, 
                        help='Strategy name (e.g., gen_NST, gen_cycleGAN)')
    parser.add_argument('--weights-root', default='${AWARE_DATA_ROOT}/WEIGHTS',
                        help='Root directory for model weights')
    parser.add_argument('--include-clearday', action='store_true',
                        help='Also evaluate _clear_day variants if available')
    args = parser.parse_args()
    
    STRATEGY = args.strategy
    WEIGHTS_ROOT = Path(args.weights_root)
    SOURCE_DATASETS = ['BDD10k', 'IDD-AW', 'MapillaryVistas']
    BASE_MODELS = ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5']
    
    # Determine variants
    VARIANTS = ['']  # Always include full model
    if args.include_clearday:
        VARIANTS.append('_clear_day')
    
    print()
    print('=' * 70)
    print(f'Strategy Domain Adaptation Evaluation: {STRATEGY}')
    print('=' * 70)
    print('Evaluating on Cityscapes (clear_day) + ACDC (foggy, night, rainy, snowy)')
    print('=' * 70)
    print()
    
    results_count = 0
    errors = []
    
    for source in SOURCE_DATASETS:
        for model in BASE_MODELS:
            for variant in VARIANTS:
                full_model = model + variant
                
                # Build checkpoint path for strategy
                ckpt_dir = WEIGHTS_ROOT / STRATEGY / source.lower() / full_model
                ckpt_path = ckpt_dir / 'iter_80000.pth'
                
                if not ckpt_path.exists():
                    print(f'SKIP: No checkpoint for {STRATEGY}/{source}/{full_model}')
                    continue
                
                print()
                print('=' * 70)
                print(f'Evaluating: {STRATEGY} / {source} / {full_model}')
                print('=' * 70)
                
                try:
                    result = run_evaluation(
                        source_dataset=source,
                        model=model,
                        checkpoint_path=str(ckpt_path),
                        variant=variant
                    )
                    
                    if result:
                        results_count += 1
                        # Save result to strategy-specific location
                        output_dir = WEIGHTS_ROOT / 'domain_adaptation_ablation' / STRATEGY / source.lower() / full_model
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        result_file = output_dir / 'domain_adaptation_evaluation.json'
                        with open(result_file, 'w') as f:
                            json.dump(result, f, indent=2)
                        print(f'Saved to: {result_file}')
                        
                except Exception as e:
                    print(f'ERROR: {e}')
                    errors.append(f'{source}/{full_model}: {e}')
    
    print()
    print('=' * 70)
    print('EVALUATION COMPLETE')
    print('=' * 70)
    print(f'Strategy: {STRATEGY}')
    print(f'Successful evaluations: {results_count}')
    print(f'Errors: {len(errors)}')
    if errors:
        print('Error details:')
        for err in errors:
            print(f'  - {err}')
    
    return 0 if not errors else 1


if __name__ == '__main__':
    sys.exit(main())
