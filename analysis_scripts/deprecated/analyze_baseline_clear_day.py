#!/usr/bin/env python3
"""
Comprehensive Analysis of Baseline Clear_Day Results

This script analyzes the baseline_clear_day results from /scratch/aaa_exchange/AWARE/WEIGHTS/
to understand how models trained only on clear weather conditions perform across different
weather domains.

Output includes:
- Per-dataset results (ACDC, BDD10k, IDD-AW, MapillaryVistas, Outside15k)
- Per-domain results (clear_day, foggy, rainy, snowy, night, cloudy, dawn_dusk)
- Per-model results (deeplabv3plus_r50, pspnet_r50, segformer_mit-b5)
- Average values across all combinations
- Publication-ready figures
"""

import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuration
WEIGHTS_DIR = "/scratch/aaa_exchange/AWARE/WEIGHTS/baseline"
OUTPUT_DIR = "result_figures/baseline_clear_day_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define domains and their categories
ADVERSE_DOMAINS = ['foggy', 'rainy', 'snowy', 'night', 'dawn_dusk']
NORMAL_DOMAINS = ['clear_day', 'cloudy']
ALL_DOMAINS = ['clear_day', 'cloudy', 'foggy', 'rainy', 'snowy', 'night', 'dawn_dusk']

# Define datasets
DATASETS = ['acdc', 'bdd10k', 'idd-aw', 'mapillaryvistas', 'outside15k']

# Define models
MODELS = ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b5']


def parse_test_report(filepath):
    """Parse a test_report.txt file and extract all metrics."""
    results = {
        'overall': {},
        'domains': {},
        'classes': {}
    }
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract overall metrics
    overall_match = re.search(
        r'OVERALL METRICS.*?aAcc:\s*([\d.]+).*?mIoU:\s*([\d.]+).*?mAcc:\s*([\d.]+).*?fwIoU:\s*([\d.]+).*?num_images:\s*(\d+)',
        content, re.DOTALL
    )
    if overall_match:
        results['overall'] = {
            'aAcc': float(overall_match.group(1)),
            'mIoU': float(overall_match.group(2)),
            'mAcc': float(overall_match.group(3)),
            'fwIoU': float(overall_match.group(4)),
            'num_images': int(overall_match.group(5))
        }
    
    # Extract per-domain metrics
    domain_pattern = r'--- (\w+) ---\s*\n\s*aAcc:\s*([\d.]+)\s*\n\s*mIoU:\s*([\d.]+)\s*\n\s*mAcc:\s*([\d.]+)\s*\n\s*fwIoU:\s*([\d.]+)\s*\n\s*num_images:\s*(\d+)'
    for match in re.finditer(domain_pattern, content):
        domain = match.group(1)
        results['domains'][domain] = {
            'aAcc': float(match.group(2)),
            'mIoU': float(match.group(3)),
            'mAcc': float(match.group(4)),
            'fwIoU': float(match.group(5)),
            'num_images': int(match.group(6))
        }
    
    # Extract per-class metrics
    class_pattern = r'^(\w+(?:\s+\w+)*)\s+([\d.]+)\s+([\d.]+)\s*$'
    in_class_section = False
    for line in content.split('\n'):
        if 'PER-CLASS METRICS' in line:
            in_class_section = True
            continue
        if in_class_section:
            match = re.match(class_pattern, line.strip())
            if match:
                class_name = match.group(1)
                results['classes'][class_name] = {
                    'IoU': float(match.group(2)),
                    'Acc': float(match.group(3))
                }
    
    return results


def find_test_reports(base_dir, model_suffix='_clear_day'):
    """Find all test_report.txt files for clear_day models."""
    reports = []
    
    for dataset in DATASETS:
        dataset_dir = os.path.join(base_dir, dataset)
        if not os.path.exists(dataset_dir):
            continue
            
        for model_dir in os.listdir(dataset_dir):
            if model_suffix in model_dir:
                model_path = os.path.join(dataset_dir, model_dir)
                
                # Look for test_results_detailed directory
                detailed_dir = os.path.join(model_path, 'test_results_detailed')
                if os.path.exists(detailed_dir):
                    for timestamp_dir in sorted(os.listdir(detailed_dir), reverse=True):
                        report_path = os.path.join(detailed_dir, timestamp_dir, 'test_report.txt')
                        if os.path.exists(report_path):
                            # Extract model name (without _clear_day suffix)
                            model_name = model_dir.replace('_clear_day', '')
                            reports.append({
                                'dataset': dataset,
                                'model': model_name,
                                'path': report_path
                            })
                            break  # Use most recent report
    
    return reports


def load_all_results(base_dir):
    """Load all clear_day baseline results."""
    reports = find_test_reports(base_dir)
    
    all_results = []
    
    for report_info in reports:
        try:
            parsed = parse_test_report(report_info['path'])
            
            # Add overall metrics
            result = {
                'dataset': report_info['dataset'],
                'model': report_info['model'],
                'type': 'overall',
                **parsed['overall']
            }
            all_results.append(result)
            
            # Add per-domain metrics
            for domain, metrics in parsed['domains'].items():
                result = {
                    'dataset': report_info['dataset'],
                    'model': report_info['model'],
                    'domain': domain,
                    'type': 'domain',
                    **metrics
                }
                all_results.append(result)
            
        except Exception as e:
            print(f"Error parsing {report_info['path']}: {e}")
    
    return pd.DataFrame(all_results)


def compute_averages(df):
    """Compute various average metrics."""
    averages = {}
    
    # Overall average across all datasets and models
    overall_df = df[df['type'] == 'overall']
    averages['overall'] = {
        'mIoU': overall_df['mIoU'].mean(),
        'fwIoU': overall_df['fwIoU'].mean(),
        'aAcc': overall_df['aAcc'].mean(),
        'mAcc': overall_df['mAcc'].mean(),
        'count': len(overall_df)
    }
    
    # Per-dataset averages
    averages['per_dataset'] = {}
    for dataset in DATASETS:
        dataset_df = overall_df[overall_df['dataset'] == dataset]
        if len(dataset_df) > 0:
            averages['per_dataset'][dataset] = {
                'mIoU': dataset_df['mIoU'].mean(),
                'fwIoU': dataset_df['fwIoU'].mean(),
                'aAcc': dataset_df['aAcc'].mean(),
                'mAcc': dataset_df['mAcc'].mean(),
                'count': len(dataset_df)
            }
    
    # Per-model averages
    averages['per_model'] = {}
    for model in MODELS:
        model_df = overall_df[overall_df['model'] == model]
        if len(model_df) > 0:
            averages['per_model'][model] = {
                'mIoU': model_df['mIoU'].mean(),
                'fwIoU': model_df['fwIoU'].mean(),
                'aAcc': model_df['aAcc'].mean(),
                'mAcc': model_df['mAcc'].mean(),
                'count': len(model_df)
            }
    
    # Per-domain averages
    domain_df = df[df['type'] == 'domain']
    averages['per_domain'] = {}
    for domain in ALL_DOMAINS:
        domain_subset = domain_df[domain_df['domain'] == domain]
        if len(domain_subset) > 0:
            averages['per_domain'][domain] = {
                'mIoU': domain_subset['mIoU'].mean(),
                'fwIoU': domain_subset['fwIoU'].mean(),
                'aAcc': domain_subset['aAcc'].mean(),
                'mAcc': domain_subset['mAcc'].mean(),
                'num_images': domain_subset['num_images'].sum(),
                'count': len(domain_subset)
            }
    
    # Adverse vs Normal condition averages
    adverse_df = domain_df[domain_df['domain'].isin(ADVERSE_DOMAINS)]
    normal_df = domain_df[domain_df['domain'].isin(NORMAL_DOMAINS)]
    
    averages['adverse_conditions'] = {
        'mIoU': adverse_df['mIoU'].mean() if len(adverse_df) > 0 else None,
        'fwIoU': adverse_df['fwIoU'].mean() if len(adverse_df) > 0 else None,
        'aAcc': adverse_df['aAcc'].mean() if len(adverse_df) > 0 else None,
        'count': len(adverse_df)
    }
    
    averages['normal_conditions'] = {
        'mIoU': normal_df['mIoU'].mean() if len(normal_df) > 0 else None,
        'fwIoU': normal_df['fwIoU'].mean() if len(normal_df) > 0 else None,
        'aAcc': normal_df['aAcc'].mean() if len(normal_df) > 0 else None,
        'count': len(normal_df)
    }
    
    return averages


def create_comprehensive_report(df, averages):
    """Create a comprehensive text report."""
    report = []
    report.append("=" * 80)
    report.append("COMPREHENSIVE BASELINE CLEAR_DAY ANALYSIS")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")
    
    # Overall summary
    report.append("OVERALL SUMMARY")
    report.append("-" * 40)
    report.append(f"Total models analyzed: {averages['overall']['count']}")
    report.append(f"Overall mIoU:  {averages['overall']['mIoU']:.2f}%")
    report.append(f"Overall fwIoU: {averages['overall']['fwIoU']:.2f}%")
    report.append(f"Overall aAcc:  {averages['overall']['aAcc']:.2f}%")
    report.append(f"Overall mAcc:  {averages['overall']['mAcc']:.2f}%")
    report.append("")
    
    # Per-dataset results
    report.append("PER-DATASET RESULTS")
    report.append("-" * 40)
    report.append(f"{'Dataset':<20} {'mIoU':>8} {'fwIoU':>8} {'aAcc':>8} {'mAcc':>8} {'Count':>6}")
    report.append("-" * 60)
    for dataset in DATASETS:
        if dataset in averages['per_dataset']:
            d = averages['per_dataset'][dataset]
            report.append(f"{dataset:<20} {d['mIoU']:>8.2f} {d['fwIoU']:>8.2f} {d['aAcc']:>8.2f} {d['mAcc']:>8.2f} {d['count']:>6}")
    report.append("")
    
    # Per-model results
    report.append("PER-MODEL RESULTS")
    report.append("-" * 40)
    report.append(f"{'Model':<20} {'mIoU':>8} {'fwIoU':>8} {'aAcc':>8} {'mAcc':>8} {'Count':>6}")
    report.append("-" * 60)
    for model in MODELS:
        if model in averages['per_model']:
            m = averages['per_model'][model]
            report.append(f"{model:<20} {m['mIoU']:>8.2f} {m['fwIoU']:>8.2f} {m['aAcc']:>8.2f} {m['mAcc']:>8.2f} {m['count']:>6}")
    report.append("")
    
    # Per-domain results
    report.append("PER-DOMAIN RESULTS")
    report.append("-" * 40)
    report.append(f"{'Domain':<15} {'mIoU':>8} {'fwIoU':>8} {'aAcc':>8} {'Images':>8} {'Count':>6}")
    report.append("-" * 60)
    for domain in ALL_DOMAINS:
        if domain in averages['per_domain']:
            d = averages['per_domain'][domain]
            report.append(f"{domain:<15} {d['mIoU']:>8.2f} {d['fwIoU']:>8.2f} {d['aAcc']:>8.2f} {d['num_images']:>8} {d['count']:>6}")
    report.append("")
    
    # Adverse vs Normal comparison
    report.append("ADVERSE vs NORMAL CONDITIONS")
    report.append("-" * 40)
    adv = averages['adverse_conditions']
    norm = averages['normal_conditions']
    if adv['mIoU'] and norm['mIoU']:
        report.append(f"{'Condition':<15} {'mIoU':>8} {'fwIoU':>8} {'aAcc':>8} {'Count':>6}")
        report.append("-" * 50)
        report.append(f"{'Normal':<15} {norm['mIoU']:>8.2f} {norm['fwIoU']:>8.2f} {norm['aAcc']:>8.2f} {norm['count']:>6}")
        report.append(f"{'Adverse':<15} {adv['mIoU']:>8.2f} {adv['fwIoU']:>8.2f} {adv['aAcc']:>8.2f} {adv['count']:>6}")
        report.append(f"{'Difference':<15} {adv['mIoU']-norm['mIoU']:>8.2f} {adv['fwIoU']-norm['fwIoU']:>8.2f} {adv['aAcc']-norm['aAcc']:>8.2f}")
    report.append("")
    
    # Detailed breakdown: Dataset x Domain
    report.append("DETAILED BREAKDOWN: DATASET x DOMAIN (mIoU)")
    report.append("-" * 80)
    domain_df = df[df['type'] == 'domain']
    
    # Create pivot table
    header = f"{'Dataset':<18}"
    for domain in ALL_DOMAINS:
        header += f" {domain[:8]:>8}"
    report.append(header)
    report.append("-" * 80)
    
    for dataset in DATASETS:
        row = f"{dataset:<18}"
        for domain in ALL_DOMAINS:
            subset = domain_df[(domain_df['dataset'] == dataset) & (domain_df['domain'] == domain)]
            if len(subset) > 0:
                row += f" {subset['mIoU'].mean():>8.2f}"
            else:
                row += f" {'N/A':>8}"
        report.append(row)
    report.append("")
    
    # Detailed breakdown: Dataset x Domain (fwIoU)
    report.append("DETAILED BREAKDOWN: DATASET x DOMAIN (fwIoU)")
    report.append("-" * 80)
    
    header = f"{'Dataset':<18}"
    for domain in ALL_DOMAINS:
        header += f" {domain[:8]:>8}"
    report.append(header)
    report.append("-" * 80)
    
    for dataset in DATASETS:
        row = f"{dataset:<18}"
        for domain in ALL_DOMAINS:
            subset = domain_df[(domain_df['dataset'] == dataset) & (domain_df['domain'] == domain)]
            if len(subset) > 0:
                row += f" {subset['fwIoU'].mean():>8.2f}"
            else:
                row += f" {'N/A':>8}"
        report.append(row)
    report.append("")
    
    # Detailed breakdown: Model x Domain
    report.append("DETAILED BREAKDOWN: MODEL x DOMAIN (mIoU)")
    report.append("-" * 80)
    
    header = f"{'Model':<18}"
    for domain in ALL_DOMAINS:
        header += f" {domain[:8]:>8}"
    report.append(header)
    report.append("-" * 80)
    
    for model in MODELS:
        row = f"{model:<18}"
        for domain in ALL_DOMAINS:
            subset = domain_df[(domain_df['model'] == model) & (domain_df['domain'] == domain)]
            if len(subset) > 0:
                row += f" {subset['mIoU'].mean():>8.2f}"
            else:
                row += f" {'N/A':>8}"
        report.append(row)
    report.append("")
    
    # Number of images per domain
    report.append("IMAGE COUNTS PER DOMAIN BY DATASET")
    report.append("-" * 80)
    
    header = f"{'Dataset':<18}"
    for domain in ALL_DOMAINS:
        header += f" {domain[:8]:>8}"
    report.append(header)
    report.append("-" * 80)
    
    for dataset in DATASETS:
        row = f"{dataset:<18}"
        for domain in ALL_DOMAINS:
            subset = domain_df[(domain_df['dataset'] == dataset) & (domain_df['domain'] == domain)]
            if len(subset) > 0:
                # Take the first value since num_images is the same across models
                row += f" {int(subset['num_images'].iloc[0]):>8}"
            else:
                row += f" {'N/A':>8}"
        report.append(row)
    
    return '\n'.join(report)


def create_visualizations(df, averages):
    """Create comprehensive visualizations."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    domain_df = df[df['type'] == 'domain']
    overall_df = df[df['type'] == 'overall']
    
    # 1. Per-domain bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    domain_means = domain_df.groupby('domain')[['mIoU', 'fwIoU']].mean().reindex(ALL_DOMAINS)
    
    colors = ['#2ecc71' if d in NORMAL_DOMAINS else '#e74c3c' for d in domain_means.index]
    
    ax = axes[0]
    x = np.arange(len(domain_means))
    ax.bar(x, domain_means['mIoU'], color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(domain_means.index, rotation=45, ha='right')
    ax.set_ylabel('mIoU (%)')
    ax.set_title('Mean IoU by Weather Domain\n(Clear_Day Trained Models)')
    ax.axhline(y=domain_means['mIoU'].mean(), color='gray', linestyle='--', label='Average')
    ax.legend()
    
    ax = axes[1]
    ax.bar(x, domain_means['fwIoU'], color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(domain_means.index, rotation=45, ha='right')
    ax.set_ylabel('fwIoU (%)')
    ax.set_title('Frequency-Weighted IoU by Weather Domain\n(Clear_Day Trained Models)')
    ax.axhline(y=domain_means['fwIoU'].mean(), color='gray', linestyle='--', label='Average')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'domain_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Dataset x Domain heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # mIoU heatmap
    pivot_miou = domain_df.pivot_table(values='mIoU', index='dataset', columns='domain', aggfunc='mean')
    pivot_miou = pivot_miou.reindex(columns=ALL_DOMAINS, index=DATASETS)
    
    ax = axes[0]
    sns.heatmap(pivot_miou, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax,
                vmin=0, vmax=50, cbar_kws={'label': 'mIoU (%)'})
    ax.set_title('mIoU: Dataset × Domain')
    ax.set_xlabel('Weather Domain')
    ax.set_ylabel('Dataset')
    
    # fwIoU heatmap
    pivot_fwiou = domain_df.pivot_table(values='fwIoU', index='dataset', columns='domain', aggfunc='mean')
    pivot_fwiou = pivot_fwiou.reindex(columns=ALL_DOMAINS, index=DATASETS)
    
    ax = axes[1]
    sns.heatmap(pivot_fwiou, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax,
                vmin=50, vmax=100, cbar_kws={'label': 'fwIoU (%)'})
    ax.set_title('fwIoU: Dataset × Domain')
    ax.set_xlabel('Weather Domain')
    ax.set_ylabel('Dataset')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dataset_domain_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Model comparison across domains
    fig, ax = plt.subplots(figsize=(12, 6))
    
    model_domain_df = domain_df.groupby(['model', 'domain'])['mIoU'].mean().unstack()
    model_domain_df = model_domain_df.reindex(columns=ALL_DOMAINS)
    
    x = np.arange(len(ALL_DOMAINS))
    width = 0.25
    
    for i, model in enumerate(MODELS):
        if model in model_domain_df.index:
            offset = (i - 1) * width
            ax.bar(x + offset, model_domain_df.loc[model], width, label=model, alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(ALL_DOMAINS, rotation=45, ha='right')
    ax.set_ylabel('mIoU (%)')
    ax.set_title('Model Comparison Across Weather Domains\n(Clear_Day Trained Models)')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_domain_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Per-dataset comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    dataset_means = overall_df.groupby('dataset')[['mIoU', 'fwIoU']].mean().reindex(DATASETS)
    
    x = np.arange(len(dataset_means))
    width = 0.35
    
    ax.bar(x - width/2, dataset_means['mIoU'], width, label='mIoU', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, dataset_means['fwIoU'], width, label='fwIoU', color='#e74c3c', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_means.index, rotation=45, ha='right')
    ax.set_ylabel('IoU (%)')
    ax.set_title('Per-Dataset Performance\n(Clear_Day Trained Models)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (miou, fwiou) in enumerate(zip(dataset_means['mIoU'], dataset_means['fwIoU'])):
        ax.annotate(f'{miou:.1f}', (i - width/2, miou + 1), ha='center', fontsize=9)
        ax.annotate(f'{fwiou:.1f}', (i + width/2, fwiou + 1), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dataset_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Adverse vs Normal conditions radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    categories = list(averages['per_domain'].keys())
    N = len(categories)
    
    values_miou = [averages['per_domain'][d]['mIoU'] for d in categories]
    values_miou += values_miou[:1]  # Close the loop
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    ax.plot(angles, values_miou, 'o-', linewidth=2, color='#3498db', label='mIoU')
    ax.fill(angles, values_miou, alpha=0.25, color='#3498db')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title('mIoU by Weather Domain (Radar)\n(Clear_Day Trained Models)', y=1.1)
    
    # Color adverse domains differently
    for i, cat in enumerate(categories):
        color = '#e74c3c' if cat in ADVERSE_DOMAINS else '#2ecc71'
        ax.get_xticklabels()[i].set_color(color)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'domain_radar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Publication-ready summary figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Top-left: Overall summary
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['mIoU', 'fwIoU', 'aAcc', 'mAcc']
    values = [averages['overall'][m] for m in metrics]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    ax1.barh(metrics, values, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xlim(0, 100)
    ax1.set_xlabel('Percentage (%)')
    ax1.set_title('Overall Performance Summary')
    for i, v in enumerate(values):
        ax1.annotate(f'{v:.1f}%', (v + 1, i), va='center')
    
    # Top-right: Adverse vs Normal
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(3)
    width = 0.35
    normal_vals = [averages['normal_conditions']['mIoU'], 
                   averages['normal_conditions']['fwIoU'],
                   averages['normal_conditions']['aAcc']]
    adverse_vals = [averages['adverse_conditions']['mIoU'],
                    averages['adverse_conditions']['fwIoU'],
                    averages['adverse_conditions']['aAcc']]
    ax2.bar(x - width/2, normal_vals, width, label='Normal', color='#2ecc71', alpha=0.8)
    ax2.bar(x + width/2, adverse_vals, width, label='Adverse', color='#e74c3c', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['mIoU', 'fwIoU', 'aAcc'])
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Normal vs Adverse Conditions')
    ax2.legend()
    
    # Middle-left: Dataset comparison
    ax3 = fig.add_subplot(gs[1, 0])
    dataset_means = overall_df.groupby('dataset')['mIoU'].mean().reindex(DATASETS)
    ax3.barh(DATASETS, dataset_means, color='#3498db', alpha=0.8, edgecolor='black')
    ax3.set_xlabel('mIoU (%)')
    ax3.set_title('Performance by Dataset')
    for i, v in enumerate(dataset_means):
        ax3.annotate(f'{v:.1f}', (v + 0.5, i), va='center')
    
    # Middle-right: Model comparison
    ax4 = fig.add_subplot(gs[1, 1])
    model_means = overall_df.groupby('model')['mIoU'].mean().reindex(MODELS)
    ax4.barh(MODELS, model_means, color='#9b59b6', alpha=0.8, edgecolor='black')
    ax4.set_xlabel('mIoU (%)')
    ax4.set_title('Performance by Model')
    for i, v in enumerate(model_means):
        ax4.annotate(f'{v:.1f}', (v + 0.5, i), va='center')
    
    # Bottom: Domain heatmap
    ax5 = fig.add_subplot(gs[2, :])
    pivot_miou = domain_df.pivot_table(values='mIoU', index='dataset', columns='domain', aggfunc='mean')
    pivot_miou = pivot_miou.reindex(columns=ALL_DOMAINS, index=DATASETS)
    sns.heatmap(pivot_miou, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax5,
                vmin=0, vmax=50, cbar_kws={'label': 'mIoU (%)'})
    ax5.set_title('mIoU: Dataset × Domain')
    ax5.set_xlabel('Weather Domain')
    ax5.set_ylabel('Dataset')
    
    fig.suptitle('Baseline Clear_Day Models: Comprehensive Analysis', fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'publication_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {OUTPUT_DIR}/")


def main():
    print("=" * 60)
    print("Comprehensive Baseline Clear_Day Analysis")
    print("=" * 60)
    print()
    
    # Load all results
    print("Loading results from WEIGHTS directory...")
    df = load_all_results(WEIGHTS_DIR)
    
    if len(df) == 0:
        print("No results found!")
        return
    
    print(f"Loaded {len(df)} results")
    print(f"  - Overall results: {len(df[df['type'] == 'overall'])}")
    print(f"  - Domain results: {len(df[df['type'] == 'domain'])}")
    print()
    
    # Compute averages
    print("Computing averages...")
    averages = compute_averages(df)
    
    # Generate report
    print("Generating comprehensive report...")
    report = create_comprehensive_report(df, averages)
    
    # Save report
    report_path = os.path.join(OUTPUT_DIR, 'comprehensive_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to {report_path}")
    
    # Print report
    print()
    print(report)
    print()
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(df, averages)
    
    # Save raw data
    df.to_csv(os.path.join(OUTPUT_DIR, 'baseline_clear_day_results.csv'), index=False)
    print(f"Raw data saved to {OUTPUT_DIR}/baseline_clear_day_results.csv")
    
    # Save averages as JSON
    with open(os.path.join(OUTPUT_DIR, 'averages.json'), 'w') as f:
        json.dump(averages, f, indent=2, default=str)
    print(f"Averages saved to {OUTPUT_DIR}/averages.json")


if __name__ == '__main__':
    main()
