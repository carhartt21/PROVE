#!/usr/bin/env python3
"""
Analyze the mask2former paradox: hurt by augmentation in S1, helped in CG.

This script:
1. Collects all mask2former test results (S1 + CG) per-class
2. Compares per-class IoU between baseline and augmented strategies
3. Identifies which classes drive the S1 degradation and CG improvement
4. Groups classes by category (road infra, objects, vegetation, etc.)
"""

import json
import glob
import os
import sys
from collections import defaultdict
import numpy as np

# Class definitions for 19-class Cityscapes format
CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence',
    'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck',
    'bus', 'train', 'motorcycle', 'bicycle'
]

# Group classes by semantic category
CLASS_GROUPS = {
    'Flat': ['road', 'sidewalk', 'terrain'],
    'Construction': ['building', 'wall', 'fence'],
    'Object': ['pole', 'traffic light', 'traffic sign'],
    'Nature': ['vegetation', 'sky'],
    'Human': ['person', 'rider'],
    'Vehicle': ['car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'],
}

# Inverse mapping: class -> group
CLASS_TO_GROUP = {}
for group, classes in CLASS_GROUPS.items():
    for cls in classes:
        CLASS_TO_GROUP[cls] = group


def find_latest_results(base_path):
    """Find the latest results.json in test_results_detailed/."""
    pattern = os.path.join(base_path, 'test_results_detailed', '*', 'results.json')
    results = sorted(glob.glob(pattern))
    if results:
        return results[-1]
    return None


def collect_s1_results():
    """Collect all mask2former S1 results across strategies and datasets."""
    base = '${AWARE_DATA_ROOT}/WEIGHTS'
    datasets = ['bdd10k', 'iddaw']  # Only 19-class datasets for fair class comparison
    
    results = {}  # {strategy: {dataset: {class: IoU}}}
    overall = {}  # {strategy: {dataset: mIoU}}
    
    # Find all strategies
    strategies = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])
    
    for strategy in strategies:
        for dataset in datasets:
            # Try both naming conventions
            model_candidates = [
                os.path.join(base, strategy, dataset, 'mask2former_swin-b'),
                os.path.join(base, strategy, dataset, 'mask2former_swin-b_ratio0p50'),
            ]
            
            model_path = None
            for candidate in model_candidates:
                if os.path.isdir(candidate):
                    model_path = candidate
                    break
            
            if not model_path:
                continue
            
            result_file = find_latest_results(model_path)
            if not result_file:
                continue
            
            try:
                with open(result_file) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
            
            if strategy not in results:
                results[strategy] = {}
                overall[strategy] = {}
            
            # Per-class IoU
            if 'per_class' in data:
                results[strategy][dataset] = {}
                for cls_name, cls_data in data['per_class'].items():
                    if 'IoU' in cls_data:
                        results[strategy][dataset][cls_name] = cls_data['IoU']
            
            # Overall mIoU
            if 'overall' in data and 'mIoU' in data['overall']:
                overall[strategy][dataset] = data['overall']['mIoU']
            
            # Per-domain per-class
            if 'per_domain' in data:
                for domain, domain_data in data['per_domain'].items():
                    key = f"{dataset}_{domain}"
                    if 'per_class' in domain_data:
                        results[strategy][key] = {}
                        for cls_name, cls_data in domain_data['per_class'].items():
                            if 'IoU' in cls_data:
                                results[strategy][key][cls_name] = cls_data['IoU']
    
    return results, overall


def collect_cg_results():
    """Collect all mask2former CG results from WEIGHTS_CITYSCAPES_GEN."""
    base = '${AWARE_DATA_ROOT}/WEIGHTS_CITYSCAPES_GEN'
    
    results = {}  # {strategy: {test_dataset: {class: IoU}}}
    overall = {}  # {strategy: {test_dataset: mIoU}}
    
    strategies = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])
    
    for strategy in strategies:
        # Try both naming conventions: mask2former_swin-b and mask2former_swin-b_ratio0p50
        model_candidates = [
            os.path.join(base, strategy, 'cityscapes', 'mask2former_swin-b'),
            os.path.join(base, strategy, 'cityscapes', 'mask2former_swin-b_ratio0p50'),
        ]
        
        model_path = None
        for candidate in model_candidates:
            if os.path.isdir(candidate):
                model_path = candidate
                break
        
        if not model_path:
            continue
        
        # For CG, check both test_results_acdc and test_results_detailed
        # ACDC cross-domain results (primary interest for mask2former paradox)
        acdc_pattern = os.path.join(model_path, 'test_results_acdc', '*', 'results.json')
        acdc_files = sorted(glob.glob(acdc_pattern))
        
        # In-domain Cityscapes test results
        detailed_pattern = os.path.join(model_path, 'test_results_detailed', '*', 'results.json')
        detailed_files = sorted(glob.glob(detailed_pattern))
        
        if not acdc_files and not detailed_files:
            continue
        
        if strategy not in results:
            results[strategy] = {}
            overall[strategy] = {}
        
        # Process ACDC cross-domain results
        if acdc_files:
            try:
                with open(acdc_files[-1]) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                data = {}
            
            if 'overall' in data and 'mIoU' in data['overall']:
                overall[strategy]['acdc'] = data['overall']['mIoU']
            
            if 'per_class' in data:
                results[strategy]['acdc'] = {}
                for cls_name, cls_data in data['per_class'].items():
                    if 'IoU' in cls_data:
                        results[strategy]['acdc'][cls_name] = cls_data['IoU']
            
            if 'per_domain' in data:
                for domain, domain_data in data['per_domain'].items():
                    if 'per_class' in domain_data:
                        results[strategy][f"acdc_{domain}"] = {}
                        for cls_name, cls_data in domain_data['per_class'].items():
                            if 'IoU' in cls_data:
                                results[strategy][f"acdc_{domain}"][cls_name] = cls_data['IoU']
        
        # Process in-domain Cityscapes results
        if detailed_files:
            try:
                with open(detailed_files[-1]) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                data = {}
            
            if 'overall' in data and 'mIoU' in data['overall']:
                overall[strategy]['cityscapes'] = data['overall']['mIoU']
            
            if 'per_class' in data:
                results[strategy]['cityscapes'] = {}
                for cls_name, cls_data in data['per_class'].items():
                    if 'IoU' in cls_data:
                        results[strategy]['cityscapes'][cls_name] = cls_data['IoU']
    
    return results, overall


def analyze_per_class_deltas(results, overall, stage_name):
    """Analyze per-class deltas between baseline and each strategy."""
    if 'baseline' not in results:
        print(f"  WARNING: No baseline found for {stage_name}")
        return None
    
    baseline_classes = results['baseline']
    baseline_overall = overall.get('baseline', {})
    
    strategies = sorted([s for s in results if s != 'baseline'])
    
    # Determine which datasets we have baseline for
    datasets = sorted(baseline_classes.keys())
    print(f"\n{'='*80}")
    print(f"  {stage_name} mask2former Analysis")
    print(f"{'='*80}")
    print(f"  Baseline datasets: {datasets}")
    print(f"  Strategies with results: {len(strategies)}")
    
    # Collect per-class deltas averaged across datasets
    # For each strategy, compute: delta[class] = mean over datasets of (strat_IoU - baseline_IoU)
    all_deltas = {}  # {strategy: {class: delta}}
    miou_deltas = {}  # {strategy: miou_delta}
    
    for strategy in strategies:
        if strategy not in results:
            continue
        
        strat_classes = results[strategy]
        strat_overall = overall.get(strategy, {})
        
        class_deltas = defaultdict(list)
        miou_delta_list = []
        
        for dataset in datasets:
            if dataset not in baseline_classes or dataset not in strat_classes:
                continue
            
            # Skip domain-specific keys for top-level analysis
            if '_' in dataset and dataset.split('_')[0] in ['bdd10k', 'iddaw', 'acdc']:
                continue
            
            bl = baseline_classes[dataset]
            st = strat_classes[dataset]
            
            for cls in bl:
                if cls in st:
                    class_deltas[cls].append(st[cls] - bl[cls])
            
            if dataset in baseline_overall and dataset in strat_overall:
                miou_delta_list.append(strat_overall[dataset] - baseline_overall[dataset])
        
        if class_deltas:
            all_deltas[strategy] = {cls: np.mean(vals) for cls, vals in class_deltas.items()}
            if miou_delta_list:
                miou_deltas[strategy] = np.mean(miou_delta_list)
    
    if not all_deltas:
        print("  No strategy deltas computed")
        return None
    
    # Average delta across ALL strategies for each class
    class_avg_delta = {}
    class_all_deltas = defaultdict(list)
    for strategy, deltas in all_deltas.items():
        for cls, delta in deltas.items():
            class_all_deltas[cls].append(delta)
    
    for cls in CITYSCAPES_CLASSES:
        if cls in class_all_deltas:
            class_avg_delta[cls] = {
                'mean': np.mean(class_all_deltas[cls]),
                'std': np.std(class_all_deltas[cls]),
                'min': np.min(class_all_deltas[cls]),
                'max': np.max(class_all_deltas[cls]),
                'n_positive': sum(1 for d in class_all_deltas[cls] if d > 0),
                'n_total': len(class_all_deltas[cls]),
                'group': CLASS_TO_GROUP.get(cls, 'Unknown'),
            }
    
    # Print per-class summary sorted by mean delta
    print(f"\n  Per-Class Mean Delta (averaged across strategies and datasets):")
    print(f"  {'Class':<16} {'Group':<14} {'Mean Δ':>8} {'Std':>7} {'Min':>8} {'Max':>8} {'#Pos':>6}")
    print(f"  {'-'*14}   {'-'*12}   {'-'*6}   {'-'*5}   {'-'*6}   {'-'*6}   {'-'*4}")
    
    sorted_classes = sorted(class_avg_delta.items(), key=lambda x: x[1]['mean'])
    for cls, stats in sorted_classes:
        pos_str = f"{stats['n_positive']}/{stats['n_total']}"
        print(f"  {cls:<16} {stats['group']:<14} {stats['mean']:>+7.2f}  {stats['std']:>6.2f}  {stats['min']:>+7.2f}  {stats['max']:>+7.2f}  {pos_str:>6}")
    
    # Print per-group summary
    print(f"\n  Per-Group Mean Delta:")
    group_deltas = defaultdict(list)
    for cls, stats in class_avg_delta.items():
        group_deltas[stats['group']].append(stats['mean'])
    
    print(f"  {'Group':<14} {'Mean Δ':>8} {'Classes':>10}")
    print(f"  {'-'*12}   {'-'*6}   {'-'*8}")
    for group in ['Flat', 'Construction', 'Object', 'Nature', 'Human', 'Vehicle']:
        if group in group_deltas:
            mean = np.mean(group_deltas[group])
            n = len(group_deltas[group])
            print(f"  {group:<14} {mean:>+7.2f}  {n:>8}")
    
    # Print overall mIoU delta distribution
    if miou_deltas:
        deltas_list = list(miou_deltas.values())
        print(f"\n  Overall mIoU delta distribution:")
        print(f"    Mean: {np.mean(deltas_list):+.2f} pp")
        print(f"    Std:  {np.std(deltas_list):.2f} pp")
        print(f"    Min:  {np.min(deltas_list):+.2f} pp  ({min(miou_deltas, key=miou_deltas.get)})")
        print(f"    Max:  {np.max(deltas_list):+.2f} pp  ({max(miou_deltas, key=miou_deltas.get)})")
        n_pos = sum(1 for d in deltas_list if d > 0)
        print(f"    Positive: {n_pos}/{len(deltas_list)}")
    
    return all_deltas, class_avg_delta, miou_deltas


def analyze_per_domain_classes(results, stage_name):
    """Analyze per-class deltas broken down by domain."""
    if 'baseline' not in results:
        return
    
    baseline = results['baseline']
    strategies = sorted([s for s in results if s != 'baseline'])
    
    # Find domain-specific keys
    domain_keys = [k for k in baseline.keys() if '_' in k]
    if not domain_keys:
        return
    
    # Group by domain
    domains = set()
    for k in domain_keys:
        parts = k.split('_', 1)
        if len(parts) == 2:
            domains.add(parts[1])
    
    print(f"\n  Per-Domain Per-Class Analysis ({stage_name}):")
    
    for domain in sorted(domains):
        # Find matching keys
        domain_datasets = [k for k in domain_keys if k.endswith(f"_{domain}")]
        
        # Average over strategies
        class_deltas = defaultdict(list)
        for strategy in strategies:
            if strategy not in results:
                continue
            strat = results[strategy]
            
            for dk in domain_datasets:
                if dk not in baseline or dk not in strat:
                    continue
                bl = baseline[dk]
                st = strat[dk]
                for cls in bl:
                    if cls in st:
                        class_deltas[cls].append(st[cls] - bl[cls])
        
        if not class_deltas:
            continue
        
        avg = {cls: np.mean(vals) for cls, vals in class_deltas.items()}
        worst_cls = min(avg, key=avg.get)
        best_cls = max(avg, key=avg.get)
        overall_mean = np.mean(list(avg.values()))
        n_negative = sum(1 for v in avg.values() if v < 0)
        
        print(f"\n    {domain:<12} Overall: {overall_mean:+.2f}pp  Neg classes: {n_negative}/{len(avg)}")
        print(f"      Worst: {worst_cls} ({avg[worst_cls]:+.2f}pp)  Best: {best_cls} ({avg[best_cls]:+.2f}pp)")


def cross_stage_class_comparison(s1_class_avg, cg_class_avg):
    """Compare per-class deltas between S1 and CG."""
    if not s1_class_avg or not cg_class_avg:
        return
    
    print(f"\n{'='*80}")
    print(f"  Cross-Stage Per-Class Comparison (S1 vs CG)")
    print(f"{'='*80}")
    
    common_classes = sorted(set(s1_class_avg.keys()) & set(cg_class_avg.keys()))
    
    print(f"\n  {'Class':<16} {'Group':<14} {'S1 Δ':>8} {'CG Δ':>8} {'Gap':>8} {'Pattern':>12}")
    print(f"  {'-'*14}   {'-'*12}   {'-'*6}   {'-'*6}   {'-'*6}   {'-'*10}")
    
    reversals = []
    for cls in common_classes:
        s1_d = s1_class_avg[cls]['mean']
        cg_d = cg_class_avg[cls]['mean']
        gap = cg_d - s1_d
        group = CLASS_TO_GROUP.get(cls, 'Unknown')
        
        # Determine pattern
        if s1_d < 0 and cg_d > 0:
            pattern = "REVERSAL ↑"
            reversals.append(cls)
        elif s1_d > 0 and cg_d < 0:
            pattern = "REVERSAL ↓"
            reversals.append(cls)
        elif s1_d < 0 and cg_d < 0:
            pattern = "Both ↓"
        else:
            pattern = "Both ↑"
        
        print(f"  {cls:<16} {group:<14} {s1_d:>+7.2f}  {cg_d:>+7.2f}  {gap:>+7.2f}  {pattern:>12}")
    
    # Group-level summary
    print(f"\n  Group-Level Cross-Stage:")
    for group in ['Flat', 'Construction', 'Object', 'Nature', 'Human', 'Vehicle']:
        group_classes = [c for c in common_classes if CLASS_TO_GROUP.get(c) == group]
        if not group_classes:
            continue
        s1_mean = np.mean([s1_class_avg[c]['mean'] for c in group_classes])
        cg_mean = np.mean([cg_class_avg[c]['mean'] for c in group_classes])
        print(f"    {group:<14} S1: {s1_mean:+.2f}pp  CG: {cg_mean:+.2f}pp  Gap: {cg_mean - s1_mean:+.2f}pp")
    
    if reversals:
        print(f"\n  Reversal classes: {', '.join(reversals)}")


def main():
    print("Collecting S1 mask2former results...")
    s1_results, s1_overall = collect_s1_results()
    print(f"  Found {len(s1_results)} strategies with S1 results")
    
    print("Collecting CG mask2former results...")
    cg_results, cg_overall = collect_cg_results()
    print(f"  Found {len(cg_results)} strategies with CG results")
    
    # Analyze S1
    s1_analysis = analyze_per_class_deltas(s1_results, s1_overall, "Stage 1 (S1)")
    
    # Analyze CG
    cg_analysis = analyze_per_class_deltas(cg_results, cg_overall, "Cityscapes-Gen (CG)")
    
    # Per-domain breakdown for S1
    analyze_per_domain_classes(s1_results, "S1")
    
    # Per-domain breakdown for CG
    analyze_per_domain_classes(cg_results, "CG")
    
    # Cross-stage comparison
    if s1_analysis and cg_analysis:
        _, s1_class_avg, _ = s1_analysis
        _, cg_class_avg, _ = cg_analysis
        cross_stage_class_comparison(s1_class_avg, cg_class_avg)
    
    print(f"\n{'='*80}")
    print("  Analysis complete.")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
