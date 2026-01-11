#!/usr/bin/env python3
"""
Comprehensive sanity check of valid test results.
Reviews per-domain, per-class, per-model, and per-dataset metrics.
"""

import pandas as pd
import json
import ast
from tabulate import tabulate
import numpy as np

def safe_parse_dict(s):
    """Safely parse dictionary strings from CSV."""
    if pd.isna(s) or s == '':
        return {}
    try:
        return ast.literal_eval(s)
    except:
        return {}

def main():
    df = pd.read_csv('/home/mima2416/repositories/PROVE/downstream_results.csv')
    
    # Filter to good tests only
    good = df[df['mIoU'] > 10.0].copy()
    
    print("=" * 90)
    print("COMPREHENSIVE SANITY CHECK OF VALID TEST RESULTS")
    print("=" * 90)
    print(f"Valid tests: {len(good)} / {len(df)}")
    
    # ==================== PER-DATASET ANALYSIS ====================
    print("\n" + "=" * 90)
    print("1. PER-DATASET ANALYSIS")
    print("=" * 90)
    
    for ds in sorted(good['dataset'].unique()):
        ds_data = good[good['dataset'] == ds]
        print(f"\n{'='*30} {ds} {'='*30}")
        print(f"Tests: {len(ds_data)}, Strategies: {ds_data['strategy'].nunique()}")
        
        # Summary stats
        print(f"\nMetric Summary:")
        print(f"  mIoU:  min={ds_data['mIoU'].min():.2f}%, max={ds_data['mIoU'].max():.2f}%, mean={ds_data['mIoU'].mean():.2f}%")
        print(f"  mAcc:  min={ds_data['mAcc'].min():.2f}%, max={ds_data['mAcc'].max():.2f}%, mean={ds_data['mAcc'].mean():.2f}%")
        print(f"  aAcc:  min={ds_data['aAcc'].min():.2f}%, max={ds_data['aAcc'].max():.2f}%, mean={ds_data['aAcc'].mean():.2f}%")
        
        # Baseline comparison
        baseline = ds_data[ds_data['strategy'] == 'baseline']
        if len(baseline) > 0:
            baseline_miou = baseline['mIoU'].mean()
            print(f"\n  Baseline mIoU: {baseline_miou:.2f}%")
            better = ds_data[ds_data['mIoU'] > baseline_miou]
            print(f"  Strategies > baseline: {len(better)} tests")
    
    # ==================== PER-MODEL ANALYSIS ====================
    print("\n" + "=" * 90)
    print("2. PER-MODEL ANALYSIS")
    print("=" * 90)
    
    # Normalize model names for comparison
    def normalize_model(m):
        if 'deeplabv3plus_r50' in m:
            return 'DeepLabV3+'
        elif 'pspnet_r50' in m:
            return 'PSPNet'
        elif 'segformer' in m:
            return 'SegFormer'
        return m
    
    good['model_type'] = good['model'].apply(normalize_model)
    
    model_stats = good.groupby('model_type').agg({
        'mIoU': ['mean', 'std', 'count'],
        'mAcc': 'mean',
        'aAcc': 'mean',
    }).round(2)
    model_stats.columns = ['mIoU', 'mIoU_std', 'count', 'mAcc', 'aAcc']
    model_stats = model_stats.sort_values('mIoU', ascending=False)
    
    print("\nModel Performance Comparison:")
    print(tabulate(
        model_stats.reset_index(),
        headers=['Model', 'mIoU (%)', 'Std', 'N', 'mAcc (%)', 'aAcc (%)'],
        tablefmt='simple',
        floatfmt='.2f'
    ))
    
    # Per-dataset model comparison
    print("\nPer-Dataset Model Performance:")
    pivot = good.pivot_table(values='mIoU', index='model_type', columns='dataset', aggfunc='mean').round(2)
    print(tabulate(pivot.reset_index(), headers=['Model'] + list(pivot.columns), tablefmt='simple', floatfmt='.2f'))
    
    # ==================== PER-DOMAIN ANALYSIS ====================
    print("\n" + "=" * 90)
    print("3. PER-DOMAIN ANALYSIS (from BDD10K)")
    print("=" * 90)
    
    # Parse per_domain_metrics for BDD10K tests
    bdd10k = good[good['dataset'].isin(['bdd10k', 'bdd10k_cd'])]
    
    if len(bdd10k) > 0 and 'per_domain_metrics' in bdd10k.columns:
        domains = ['clear_day', 'cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy']
        domain_data = []
        
        for _, row in bdd10k.iterrows():
            per_domain = safe_parse_dict(row['per_domain_metrics'])
            if per_domain:
                for domain in domains:
                    if domain in per_domain:
                        domain_data.append({
                            'strategy': row['strategy'],
                            'dataset': row['dataset'],
                            'domain': domain,
                            'mIoU': per_domain[domain].get('mIoU', 0),
                            'mAcc': per_domain[domain].get('mAcc', 0),
                        })
        
        if domain_data:
            domain_df = pd.DataFrame(domain_data)
            
            print("\nDomain mIoU by Strategy (BDD10K):")
            pivot = domain_df.pivot_table(values='mIoU', index='strategy', columns='domain', aggfunc='mean').round(2)
            if not pivot.empty:
                # Reorder columns
                cols = [c for c in domains if c in pivot.columns]
                pivot = pivot[cols]
                print(tabulate(pivot.reset_index(), headers=['Strategy'] + cols, tablefmt='simple', floatfmt='.2f'))
            
            print("\nDomain Average Performance:")
            domain_avg = domain_df.groupby('domain')['mIoU'].agg(['mean', 'std', 'count']).round(2)
            domain_avg = domain_avg.reindex([d for d in domains if d in domain_avg.index])
            print(tabulate(domain_avg.reset_index(), headers=['Domain', 'mIoU (%)', 'Std', 'N'], tablefmt='simple', floatfmt='.2f'))
    
    # ==================== PER-CLASS ANALYSIS ====================
    print("\n" + "=" * 90)
    print("4. PER-CLASS ANALYSIS")
    print("=" * 90)
    
    if 'per_class_metrics' in good.columns:
        cityscapes_classes = [
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
            'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
        ]
        
        class_data = []
        for _, row in good.iterrows():
            per_class = safe_parse_dict(row['per_class_metrics'])
            if per_class:
                for cls in cityscapes_classes:
                    if cls in per_class:
                        class_data.append({
                            'strategy': row['strategy'],
                            'dataset': row['dataset'],
                            'class': cls,
                            'IoU': per_class[cls].get('IoU', 0),
                            'Acc': per_class[cls].get('Acc', 0),
                            'area_label': per_class[cls].get('area_label', 0),
                        })
        
        if class_data:
            class_df = pd.DataFrame(class_data)
            
            print("\nPer-Class IoU (averaged across all valid tests):")
            class_avg = class_df.groupby('class')['IoU'].agg(['mean', 'std']).round(2)
            # Reorder by cityscapes order
            class_avg = class_avg.reindex([c for c in cityscapes_classes if c in class_avg.index])
            print(tabulate(class_avg.reset_index(), headers=['Class', 'IoU (%)', 'Std'], tablefmt='simple', floatfmt='.2f'))
            
            print("\nPer-Class IoU by Dataset:")
            pivot = class_df.pivot_table(values='IoU', index='class', columns='dataset', aggfunc='mean').round(2)
            pivot = pivot.reindex([c for c in cityscapes_classes if c in pivot.index])
            print(tabulate(pivot.reset_index(), headers=['Class'] + list(pivot.columns), tablefmt='simple', floatfmt='.2f'))
            
            # Sanity check: area_label should be consistent across strategies for same dataset
            print("\n\nSANITY CHECK: Class area_label consistency")
            for ds in class_df['dataset'].unique():
                ds_class = class_df[class_df['dataset'] == ds]
                print(f"\n{ds}:")
                
                # Check if area_label is consistent for each class
                for cls in ['road', 'vegetation', 'sky', 'car']:
                    if cls in ds_class['class'].values:
                        cls_areas = ds_class[ds_class['class'] == cls]['area_label'].unique()
                        if len(cls_areas) > 1:
                            # Some variation expected due to different test runs
                            min_area = cls_areas.min()
                            max_area = cls_areas.max()
                            variation = (max_area - min_area) / max(min_area, 1) * 100
                            status = "⚠️ VARIES" if variation > 5 else "✓"
                            print(f"  {cls}: {len(cls_areas)} unique values, variation: {variation:.1f}% {status}")
                        else:
                            print(f"  {cls}: ✓ consistent (area={cls_areas[0]:,.0f})")
    
    # ==================== SANITY CHECKS ====================
    print("\n" + "=" * 90)
    print("5. ADDITIONAL SANITY CHECKS")
    print("=" * 90)
    
    # Check for suspicious values
    print("\nChecking for anomalies:")
    
    # mIoU should be reasonable (not 0, not 100)
    suspicious_low = good[good['mIoU'] < 15]
    suspicious_high = good[good['mIoU'] > 80]
    print(f"  Very low mIoU (<15%): {len(suspicious_low)} tests")
    if len(suspicious_low) > 0:
        for _, row in suspicious_low.iterrows():
            print(f"    - {row['strategy']}/{row['dataset']}/{row['model']}: {row['mIoU']:.2f}%")
    
    print(f"  Very high mIoU (>80%): {len(suspicious_high)} tests")
    if len(suspicious_high) > 0:
        for _, row in suspicious_high.iterrows():
            print(f"    - {row['strategy']}/{row['dataset']}/{row['model']}: {row['mIoU']:.2f}%")
    
    # Check aAcc should be higher than mIoU generally
    weird_acc = good[good['aAcc'] < good['mIoU']]
    print(f"  aAcc < mIoU (unusual): {len(weird_acc)} tests")
    
    # Check num_images is consistent
    print("\nImage count by dataset:")
    for ds in good['dataset'].unique():
        ds_images = good[good['dataset'] == ds]['num_images'].unique()
        print(f"  {ds}: {ds_images}")
    
    print("\n" + "=" * 90)
    print("SANITY CHECK COMPLETE")
    print("=" * 90)


if __name__ == '__main__':
    main()
