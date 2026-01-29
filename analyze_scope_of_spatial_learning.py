#!/usr/bin/env python3
"""
Check if spatial pattern learning issue persists across:
1. All models (deeplabv3plus, pspnet, segformer)
2. All datasets (IDD-AW, BDD10k, ACDC, MapillaryVistas, OUTSIDE15k)
3. All strategies (baseline, augmentation, generative)

This determines scope of the issue.
"""

import json
from pathlib import Path
from datetime import datetime

def extract_mIoU_from_results_json(results_dir):
    """Extract mIoU from test results."""
    results_file = results_dir / 'test_results_detailed'
    if results_file.exists():
        for subdir in results_file.iterdir():
            if subdir.is_dir():
                results_json = subdir / 'results.json'
                if results_json.exists():
                    try:
                        with open(results_json) as f:
                            data = json.load(f)
                            return data.get('overall', {}).get('mIoU', None)
                    except:
                        pass
    return None


def analyze_scope_of_issue():
    """Analyze if spatial pattern issue is systematic."""
    
    print("\n" + "="*80)
    print("SCOPE ANALYSIS: Does spatial pattern learning affect all models/datasets?")
    print("="*80)
    
    root = Path('/scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION')
    stage1_root = Path('/scratch/aaa_exchange/AWARE/WEIGHTS')
    
    # Collect all mIoU results
    results_by_strategy = {}
    results_by_dataset = {}
    results_by_model = {}
    
    print("\nScanning trained models for results...\n")
    
    for strategy_dir in stage1_root.iterdir():
        if not strategy_dir.is_dir():
            continue
        
        strategy_name = strategy_dir.name
        results_by_strategy[strategy_name] = []
        
        for dataset_dir in strategy_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            dataset_name = dataset_dir.name
            if dataset_name not in results_by_dataset:
                results_by_dataset[dataset_name] = []
            
            for model_dir in dataset_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                
                model_name = model_dir.name
                if model_name not in results_by_model:
                    results_by_model[model_name] = []
                
                mIoU = extract_mIoU_from_results_json(model_dir)
                if mIoU:
                    print(f"✓ {strategy_name:20s} × {dataset_name:15s} × {model_name:20s} = {mIoU:6.2f}%")
                    results_by_strategy[strategy_name].append(mIoU)
                    results_by_dataset[dataset_name].append(mIoU)
                    results_by_model[model_name].append(mIoU)
    
    # Analysis by Strategy
    print("\n" + "-"*80)
    print("ANALYSIS BY STRATEGY (If all strategies have similar mIoU → systematic issue)")
    print("-"*80)
    
    for strategy_name in sorted(results_by_strategy.keys()):
        values = results_by_strategy[strategy_name]
        if values:
            mean_mIoU = sum(values) / len(values)
            min_mIoU = min(values)
            max_mIoU = max(values)
            print(f"{strategy_name:20s}: mean={mean_mIoU:6.2f}  range=[{min_mIoU:6.2f}, {max_mIoU:6.2f}]  n={len(values)}")
    
    # Analysis by Dataset
    print("\n" + "-"*80)
    print("ANALYSIS BY DATASET (If all datasets similar → not dataset-specific)")
    print("-"*80)
    
    for dataset_name in sorted(results_by_dataset.keys()):
        values = results_by_dataset[dataset_name]
        if values:
            mean_mIoU = sum(values) / len(values)
            min_mIoU = min(values)
            max_mIoU = max(values)
            print(f"{dataset_name:15s}: mean={mean_mIoU:6.2f}  range=[{min_mIoU:6.2f}, {max_mIoU:6.2f}]  n={len(values)}")
    
    # Analysis by Model
    print("\n" + "-"*80)
    print("ANALYSIS BY MODEL (If all models similar → not model-specific)")
    print("-"*80)
    
    for model_name in sorted(results_by_model.keys()):
        values = results_by_model[model_name]
        if values:
            mean_mIoU = sum(values) / len(values)
            min_mIoU = min(values)
            max_mIoU = max(values)
            print(f"{model_name:20s}: mean={mean_mIoU:6.2f}  range=[{min_mIoU:6.2f}, {max_mIoU:6.2f}]  n={len(values)}")
    
    # Cross-dataset consistency check
    print("\n" + "-"*80)
    print("CROSS-DATASET CONSISTENCY (If mIoU similar across datasets → overfitting to spatial patterns)")
    print("-"*80)
    
    stage1_baseline = stage1_root / 'baseline'
    if stage1_baseline.exists():
        dataset_scores = {}
        for dataset_dir in stage1_baseline.iterdir():
            if dataset_dir.is_dir():
                dataset_name = dataset_dir.name
                mIoU_values = []
                for model_dir in dataset_dir.iterdir():
                    if model_dir.is_dir():
                        mIoU = extract_mIoU_from_results_json(model_dir)
                        if mIoU:
                            mIoU_values.append(mIoU)
                
                if mIoU_values:
                    mean_mIoU = sum(mIoU_values) / len(mIoU_values)
                    dataset_scores[dataset_name] = mean_mIoU
        
        if dataset_scores:
            print("\nBaseline Strategy Across Datasets:")
            for dataset_name, mIoU in sorted(dataset_scores.items()):
                print(f"  {dataset_name:15s}: {mIoU:6.2f}%")
            
            # Check variance
            mean_score = sum(dataset_scores.values()) / len(dataset_scores)
            variance = sum((v - mean_score)**2 for v in dataset_scores.values()) / len(dataset_scores)
            std_dev = variance ** 0.5
            
            print(f"\n  Mean mIoU: {mean_score:6.2f}%")
            print(f"  Std Dev:   {std_dev:6.2f}%")
            print(f"  Range:     [{min(dataset_scores.values()):6.2f}%, {max(dataset_scores.values()):6.2f}%]")
            
            if std_dev < 5:
                print("\n  ⚠️  WARNING: Low variance across datasets!")
                print("      This suggests the model may NOT be learning dataset-specific semantics")
                print("      High probability: learning spatial patterns that happen to work across datasets")
            elif std_dev > 15:
                print("\n  ✓ GOOD: High variance across datasets!")
                print("    This suggests the model learns dataset-specific information")
                print("    Lower risk of spatial-pattern-only learning")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    all_values = []
    for values in results_by_strategy.values():
        all_values.extend(values)
    
    if all_values:
        overall_mean = sum(all_values) / len(all_values)
        overall_min = min(all_values)
        overall_max = max(all_values)
        overall_range = overall_max - overall_min
        
        print(f"\nOverall Statistics:")
        print(f"  Total evaluated: {len(all_values)} configurations")
        print(f"  Mean mIoU: {overall_mean:6.2f}%")
        print(f"  Range: [{overall_min:6.2f}%, {overall_max:6.2f}%]")
        print(f"  Spread: {overall_range:6.2f}%")
        
        if overall_range < 10:
            print(f"\n  ⚠️  CRITICAL: Very tight range ({overall_range:.2f}%) across all strategies/datasets/models")
            print("      This STRONGLY suggests spatial pattern learning (not semantic)")
        elif overall_range < 20:
            print(f"\n  ⚠️  WARNING: Modest range ({overall_range:.2f}%) - some variation but limited")
            print("      Suspicious if all different strategies perform similarly")
        else:
            print(f"\n  ✓ Good: Wide range ({overall_range:.2f}%) - models differentiate well")
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    analyze_scope_of_issue()
