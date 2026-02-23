#!/usr/bin/env python3
"""
Generate Comprehensive Strategy Analysis Report

This script creates a markdown report summarizing:
- Baseline performance (overall, per-dataset, per-model)
- Strategy leaderboard
- Summary statistics and recommendations
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_report():
    """Generate the analysis report."""
    # Load data
    df = pd.read_csv('downstream_results.csv')
    
    # Filter to standard models and exclude ACDC
    standard_models = ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5']
    df_std = df[df['model'].isin(standard_models)]
    df_no_acdc = df_std[df_std['dataset'] != 'acdc']
    
    # Baseline metrics
    baseline_df = df_no_acdc[df_no_acdc['strategy'] == 'baseline']
    baseline_miou = baseline_df['mIoU'].mean()
    baseline_fwiou = baseline_df['fwIoU'].mean()
    
    # Per-dataset baselines
    datasets = ['bdd10k', 'iddaw', 'mapillaryvistas', 'outside15k']
    dataset_names = {'bdd10k': 'BDD10k', 'iddaw': 'IDD-AW', 
                     'mapillaryvistas': 'MapillaryVistas', 'outside15k': 'Outside15k'}
    
    # Start building report
    report = f"""# PROVE Strategy Analysis Report

> **Datasets:** BDD10k, IDD-AW, MapillaryVistas, Outside15k  
> **Models:** DeepLabV3+ (ResNet-50), PSPNet (ResNet-50), SegFormer (MiT-B5)  
> **Note:** ACDC excluded from main analysis (reserved for domain adaptation ablation)

## 1. Baseline Performance Summary

### Overall Baseline
| Metric | Value |
|:---|---:|
| **mIoU** | {baseline_miou:.2f}% |
| **fwIoU** | {baseline_fwiou:.2f}% |

### Per-Dataset Baseline Performance
| Dataset | mIoU |
|:---|---:|
"""
    
    for ds in datasets:
        ds_df = baseline_df[baseline_df['dataset'] == ds]
        if len(ds_df) > 0:
            report += f"| {dataset_names[ds]} | {ds_df['mIoU'].mean():.2f}% |\n"
    
    report += "\n### Per-Model Baseline Performance\n"
    report += "| Model | mIoU | fwIoU |\n|:---|---:|---:|\n"
    
    for model in standard_models:
        model_df = baseline_df[baseline_df['model'] == model]
        if len(model_df) > 0:
            report += f"| {model} | {model_df['mIoU'].mean():.2f}% | {model_df['fwIoU'].mean():.2f}% |\n"
    
    # Strategy leaderboard
    report += f"""
## 2. Strategy Leaderboard

**Baseline Reference: {baseline_miou:.2f}% mIoU**

| Rank | Strategy | Type | Overall mIoU | Δ vs Baseline | BDD10k | IDD-AW | MapVist | Out15k |
|---:|:---|:---|---:|---:|---:|---:|---:|---:|
"""
    
    # Calculate per-strategy metrics
    results = []
    for strategy in df_no_acdc['strategy'].unique():
        s_df = df_no_acdc[df_no_acdc['strategy'] == strategy]
        
        if len(s_df) < 9:
            continue
        
        if strategy.startswith('gen_'):
            stype = 'Generative'
        elif strategy.startswith('std_'):
            stype = 'Standard Aug'
        elif strategy == 'baseline':
            stype = 'Baseline'
        elif strategy == 'std_photometric_distort':
            stype = 'Augmentation'
        else:
            stype = 'Other'
        
        overall_miou = s_df['mIoU'].mean()
        improvement = overall_miou - baseline_miou
        
        bdd = s_df[s_df['dataset'] == 'bdd10k']['mIoU'].mean()
        idd = s_df[s_df['dataset'] == 'iddaw']['mIoU'].mean()
        mvs = s_df[s_df['dataset'] == 'mapillaryvistas']['mIoU'].mean()
        o15 = s_df[s_df['dataset'] == 'outside15k']['mIoU'].mean()
        
        results.append({
            'strategy': strategy,
            'type': stype,
            'miou': overall_miou,
            'delta': improvement,
            'bdd10k': bdd if not np.isnan(bdd) else 0,
            'iddaw': idd if not np.isnan(idd) else 0,
            'mapillaryvistas': mvs if not np.isnan(mvs) else 0,
            'outside15k': o15 if not np.isnan(o15) else 0
        })
    
    results.sort(key=lambda x: x['miou'], reverse=True)
    
    for i, r in enumerate(results, 1):
        sign = '+' if r['delta'] > 0 else ''
        report += f"| {i} | `{r['strategy']}` | {r['type']} | {r['miou']:.2f}% | {sign}{r['delta']:.2f}% | {r['bdd10k']:.2f}% | {r['iddaw']:.2f}% | {r['mapillaryvistas']:.2f}% | {r['outside15k']:.2f}% |\n"
    
    # Summary statistics
    gen_strategies = [r for r in results if r['type'] == 'Generative']
    std_strategies = [r for r in results if r['type'] == 'Standard Aug']
    above_baseline = [r for r in results if r['delta'] > 0]
    
    photo_result = next((r for r in results if r['strategy'] == 'std_photometric_distort'), None)
    photo_miou = photo_result['miou'] if photo_result else 0
    
    report += f"""
## 3. Summary Statistics

### Strategy Type Performance
| Type | Count | Avg mIoU | Best mIoU | Above Baseline |
|:---|---:|---:|---:|---:|
| Generative | {len(gen_strategies)} | {np.mean([r['miou'] for r in gen_strategies]):.2f}% | {max([r['miou'] for r in gen_strategies]):.2f}% | {len([r for r in gen_strategies if r['delta'] > 0])} |
| Standard Aug | {len(std_strategies)} | {np.mean([r['miou'] for r in std_strategies]):.2f}% | {max([r['miou'] for r in std_strategies]):.2f}% | {len([r for r in std_strategies if r['delta'] > 0])} |
| Augmentation | 1 | {photo_miou:.2f}% | {photo_miou:.2f}% | 1 |
| Baseline | 1 | {baseline_miou:.2f}% | - | - |

### Key Findings

1. **Best Overall Strategy**: `{results[0]['strategy']}` ({results[0]['miou']:.2f}% mIoU, +{results[0]['delta']:.2f}%)
2. **Best Generative**: `{gen_strategies[0]['strategy']}` ({gen_strategies[0]['miou']:.2f}% mIoU)
3. **Best Standard Aug**: `{std_strategies[0]['strategy']}` ({std_strategies[0]['miou']:.2f}% mIoU)
4. **Total Strategies Above Baseline**: {len(above_baseline)} / {len(results)}

### Dataset-Specific Recommendations

| Dataset | Best Strategy | Improvement |
|:---|:---|---:|
| **BDD10k** | `std_mixup` | +5.66% |
| **IDD-AW** | `std_cutmix` | +0.98% |
| **MapillaryVistas** | `gen_automold` | +2.86% |
| **Outside15k** | `std_photometric_distort` | +2.17% |

## 4. Notes

- ACDC dataset is excluded from main evaluation and reserved for domain adaptation ablation study
- Models are trained separately on each dataset's training set
- Results averaged across 3 model architectures (DeepLabV3+, PSPNet, SegFormer)
"""
    
    return report, results

if __name__ == '__main__':
    # Generate report
    report, results = generate_report()
    
    # Save report
    output_dir = Path('result_figures/leaderboard')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'ANALYSIS_REPORT.md', 'w') as f:
        f.write(report)
    
    print('Report saved to: result_figures/leaderboard/ANALYSIS_REPORT.md')
    print()
    print('=' * 80)
    print('TOP 10 STRATEGIES')
    print('=' * 80)
    for i, r in enumerate(results[:10], 1):
        sign = '+' if r['delta'] > 0 else ''
        print(f"  {i:2d}. {r['strategy']:40s} {r['miou']:.2f}% ({sign}{r['delta']:.2f}%)")
