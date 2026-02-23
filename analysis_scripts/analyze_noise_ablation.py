#!/usr/bin/env python3
"""
PROVE Noise Ablation Study Analyzer

Analyzes results from the noise ablation study to evaluate whether
augmentation gains come from meaningful weather-domain content or just
additional training samples (regularization effect).

The noise ablation replaces generated images with uniform random noise
while keeping the same label maps. If noise helps as much as real generated
images, gains are from regularization. If generated images help more, 
the visual content provides genuine additional training signal.

Usage:
    python analyze_noise_ablation.py
    python analyze_noise_ablation.py --export-csv noise_results.csv
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
import csv

NOISE_WEIGHTS_ROOT = Path('${AWARE_DATA_ROOT}/WEIGHTS_NOISE_ABLATION')
S1_WEIGHTS_ROOT = Path('${AWARE_DATA_ROOT}/WEIGHTS')
FIGURES_DIR = Path('result_figures/noise_ablation')

DATASETS = ['bdd10k', 'iddaw', 'mapillaryvistas', 'outside15k']
MODELS = ['deeplabv3plus_r50', 'pspnet_r50', 'segformer_mit-b3', 'segnext_mscan-b', 'mask2former_swin-b']
# Note: hrnet_hr48 excluded — S1 baseline results suspiciously low (15-21% mIoU),
# inflating noise gains (+12-13pp) due to broken/undertrained baseline.


def find_results_json(base_dir):
    """Find the most recent results.json in a test_results_detailed directory."""
    test_dir = base_dir / 'test_results_detailed'
    if not test_dir.exists():
        return None
    results = list(test_dir.rglob('results.json'))
    if not results:
        return None
    # Return the most recent
    return max(results, key=lambda p: p.stat().st_mtime)


def load_miou(results_path):
    """Load mIoU from a results.json file."""
    if results_path is None:
        return None
    try:
        with open(results_path) as f:
            data = json.load(f)
        return data.get('overall', {}).get('mIoU', None)
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def load_per_domain(results_path):
    """Load per-domain mIoU from a results.json file."""
    if results_path is None:
        return {}
    try:
        with open(results_path) as f:
            data = json.load(f)
        return data.get('per_domain', {})
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def collect_noise_results():
    """Collect all noise ablation results."""
    results = []
    
    for ds in DATASETS:
        for model in MODELS:
            # Noise result
            noise_dir = NOISE_WEIGHTS_ROOT / 'gen_random_noise' / ds / f'{model}_ratio0p50'
            noise_results = find_results_json(noise_dir)
            noise_miou = load_miou(noise_results)
            
            # Baseline result
            base_dir = S1_WEIGHTS_ROOT / 'baseline' / ds / model
            base_results = find_results_json(base_dir)
            base_miou = load_miou(base_results)
            
            results.append({
                'dataset': ds,
                'model': model,
                'noise_miou': noise_miou,
                'baseline_miou': base_miou,
                'diff': (noise_miou - base_miou) if (noise_miou is not None and base_miou is not None) else None,
            })
    
    return results


def collect_gen_comparison(strategy='gen_cycleGAN'):
    """Collect gen_* strategy results for comparison with noise."""
    results = []
    for ds in DATASETS:
        for model in MODELS:
            gen_dir = S1_WEIGHTS_ROOT / strategy / ds / f'{model}_ratio0p50'
            gen_results = find_results_json(gen_dir)
            gen_miou = load_miou(gen_results)
            
            base_dir = S1_WEIGHTS_ROOT / 'baseline' / ds / model
            base_miou = load_miou(find_results_json(base_dir))
            
            noise_dir = NOISE_WEIGHTS_ROOT / 'gen_random_noise' / ds / f'{model}_ratio0p50'
            noise_miou = load_miou(find_results_json(noise_dir))
            
            results.append({
                'dataset': ds,
                'model': model,
                'gen_miou': gen_miou,
                'noise_miou': noise_miou,
                'baseline_miou': base_miou,
            })
    return results


def print_noise_results(results):
    """Print formatted noise ablation results."""
    print("=" * 90)
    print("NOISE ABLATION STUDY — Random Noise vs Baseline")
    print("=" * 90)
    print(f"{'Dataset/Model':<40} {'Baseline':>10} {'Noise':>10} {'Diff':>10}")
    print("-" * 90)
    
    diffs = []
    positive = 0
    negative = 0
    neutral = 0
    
    for ds in DATASETS:
        ds_results = [r for r in results if r['dataset'] == ds]
        for r in ds_results:
            base_str = f"{r['baseline_miou']:.2f}" if r['baseline_miou'] is not None else "—"
            noise_str = f"{r['noise_miou']:.2f}" if r['noise_miou'] is not None else "—"
            if r['diff'] is not None:
                diff_str = f"{r['diff']:+.2f}"
                diffs.append(r['diff'])
                if r['diff'] > 0.1:
                    positive += 1
                elif r['diff'] < -0.1:
                    negative += 1
                else:
                    neutral += 1
            else:
                diff_str = "—"
            print(f"  {r['dataset']}/{r['model']:<36} {base_str:>10} {noise_str:>10} {diff_str:>10}")
        print()
    
    print("-" * 90)
    if diffs:
        avg = sum(diffs) / len(diffs)
        median = sorted(diffs)[len(diffs) // 2]
        print(f"  {'Average':>38} {'':>10} {'':>10} {avg:+.2f}")
        print(f"  {'Median':>38} {'':>10} {'':>10} {median:+.2f}")
        print(f"\n  Compared: {len(diffs)} | Positive: {positive} | Negative: {negative} | Neutral: {neutral}")
    print()


def print_gen_vs_noise(results, strategy):
    """Print gen_* vs noise comparison to separate regularization from content effect."""
    print("=" * 100)
    print(f"GEN_* vs NOISE — Separating Content from Regularization ({strategy})")
    print("=" * 100)
    print(f"{'Dataset/Model':<35} {'Baseline':>8} {'Noise':>8} {'Gen':>8} {'Noise-Base':>11} {'Gen-Base':>9} {'Gen-Noise':>10} {'% Content':>10}")
    print("-" * 100)
    
    content_effects = []
    
    for r in results:
        if all(v is not None for v in [r['baseline_miou'], r['noise_miou'], r['gen_miou']]):
            noise_gain = r['noise_miou'] - r['baseline_miou']
            gen_gain = r['gen_miou'] - r['baseline_miou']
            content_effect = r['gen_miou'] - r['noise_miou']
            # % of gen gain attributable to content (vs regularization)
            if gen_gain > 0.01:
                pct_content = max(0, content_effect / gen_gain * 100)
            else:
                pct_content = None
            
            content_effects.append({
                'dataset': r['dataset'],
                'model': r['model'],
                'noise_gain': noise_gain,
                'gen_gain': gen_gain,
                'content_effect': content_effect,
                'pct_content': pct_content,
            })
            
            pct_str = f"{pct_content:.0f}%" if pct_content is not None else "—"
            print(f"  {r['dataset']}/{r['model']:<31} {r['baseline_miou']:>8.2f} {r['noise_miou']:>8.2f} {r['gen_miou']:>8.2f} {noise_gain:>+11.2f} {gen_gain:>+9.2f} {content_effect:>+10.2f} {pct_str:>10}")
    
    print("-" * 100)
    if content_effects:
        avg_noise = sum(c['noise_gain'] for c in content_effects) / len(content_effects)
        avg_gen = sum(c['gen_gain'] for c in content_effects) / len(content_effects)
        avg_content = sum(c['content_effect'] for c in content_effects) / len(content_effects)
        valid_pcts = [c['pct_content'] for c in content_effects if c['pct_content'] is not None]
        avg_pct = sum(valid_pcts) / len(valid_pcts) if valid_pcts else None
        pct_str = f"{avg_pct:.0f}%" if avg_pct is not None else "—"
        print(f"  {'Average':>33} {'':>8} {'':>8} {'':>8} {avg_noise:>+11.2f} {avg_gen:>+9.2f} {avg_content:>+10.2f} {pct_str:>10}")
    print()
    
    return content_effects


def print_per_dataset_summary(results):
    """Print per-dataset summary of noise ablation."""
    print("=" * 70)
    print("PER-DATASET NOISE ABLATION SUMMARY")
    print("=" * 70)
    
    for ds in DATASETS:
        ds_results = [r for r in results if r['dataset'] == ds and r['diff'] is not None]
        if not ds_results:
            continue
        diffs = [r['diff'] for r in ds_results]
        avg = sum(diffs) / len(diffs)
        positive = sum(1 for d in diffs if d > 0.1)
        negative = sum(1 for d in diffs if d < -0.1)
        print(f"  {ds:<20} avg={avg:+.2f} pp  ({len(diffs)} models, {positive} positive, {negative} negative)")
    print()


def print_per_model_summary(results):
    """Print per-model summary of noise ablation."""
    print("=" * 70)
    print("PER-MODEL NOISE ABLATION SUMMARY")
    print("=" * 70)
    
    for model in MODELS:
        model_results = [r for r in results if r['model'] == model and r['diff'] is not None]
        if not model_results:
            continue
        diffs = [r['diff'] for r in model_results]
        avg = sum(diffs) / len(diffs)
        positive = sum(1 for d in diffs if d > 0.1)
        negative = sum(1 for d in diffs if d < -0.1)
        print(f"  {model:<25} avg={avg:+.2f} pp  ({len(diffs)} datasets, {positive} positive, {negative} negative)")
    print()


def export_csv(results, filepath):
    """Export results to CSV."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'model', 'baseline_miou', 'noise_miou', 'diff'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"Exported to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Analyze noise ablation study results")
    parser.add_argument('--export-csv', type=str, help='Export results to CSV file')
    parser.add_argument('--gen-strategy', type=str, default='gen_cycleGAN',
                        help='Strategy to compare gen vs noise (default: gen_cycleGAN)')
    parser.add_argument('--compare-strategies', nargs='+', 
                        default=['gen_cycleGAN', 'gen_Img2Img', 'gen_augmenters'],
                        help='List of gen strategies to compare against noise')
    args = parser.parse_args()
    
    print("\nCollecting noise ablation results...")
    results = collect_noise_results()
    
    # Basic noise vs baseline
    print_noise_results(results)
    print_per_dataset_summary(results)
    print_per_model_summary(results)
    
    # Noise vs gen_* comparison for each strategy
    for strategy in args.compare_strategies:
        gen_results = collect_gen_comparison(strategy)
        content_effects = print_gen_vs_noise(gen_results, strategy)
    
    # Summary interpretation
    paired = [r for r in results if r['diff'] is not None]
    if paired:
        diffs = [r['diff'] for r in paired]
        avg_noise = sum(diffs) / len(diffs)
        print("=" * 70)
        print("INTERPRETATION SUMMARY")
        print("=" * 70)
        print(f"  Noise vs baseline: avg {avg_noise:+.2f} pp ({sum(1 for d in diffs if d > 0.1)}/{len(diffs)} positive)")
        print()
        print("  Key findings:")
        if avg_noise > 1.0:
            print("  1. Random noise provides SIGNIFICANT regularization benefit")
            print("     → Part of gen_* gains likely from data volume, not content")
            print("  2. Models were undertrained at 15k iters with limited data")
            print("  3. Gen_* vs Noise comparison needed to quantify content contribution")
        else:
            print("  1. Noise provides minimal regularization benefit")
            print("  2. Gen_* gains are primarily from meaningful content")
        
        # Check which models benefited most
        large_gains = [(r['dataset'], r['model'], r['diff']) for r in paired if r['diff'] > 5.0]
        if large_gains:
            print(f"\n  Models with large noise gains (>5 pp): {len(large_gains)}")
            for ds, m, d in sorted(large_gains, key=lambda x: -x[2]):
                print(f"    {ds}/{m}: {d:+.2f} pp — likely severely undertrained baseline")
    
    if args.export_csv:
        export_csv(results, args.export_csv)
    
    print()


if __name__ == '__main__':
    main()
