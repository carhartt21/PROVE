#!/usr/bin/env python3
"""
PROVE Extended Training Ablation Analyzer (New Regime)

Analyzes results from WEIGHTS_EXTENDED_ABLATION/ which contains:
- stage1/: S1 models extended from 15k → 45k (3× standard)
- cityscapes_gen/: CG models extended from 20k → 60k (3× standard)

Compares performance at standard vs extended iterations to determine
whether augmentation gains diminish, persist, or grow with more training.

Usage:
    python analyze_extended_training_ablation.py
    python analyze_extended_training_ablation.py --stage s1
    python analyze_extended_training_ablation.py --stage cg
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

EXTENDED_ROOT = Path('${AWARE_DATA_ROOT}/WEIGHTS_EXTENDED_ABLATION')
S1_WEIGHTS = Path('${AWARE_DATA_ROOT}/WEIGHTS')
CG_WEIGHTS = Path('${AWARE_DATA_ROOT}/WEIGHTS_CITYSCAPES_GEN')


def find_results_json(base_dir):
    """Find the most recent results.json in a test_results_detailed directory."""
    test_dir = base_dir / 'test_results_detailed'
    if not test_dir.exists():
        return None
    results = list(test_dir.rglob('results.json'))
    if not results:
        return None
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


def analyze_stage1():
    """Analyze extended Stage 1 training (15k → 45k)."""
    print("=" * 90)
    print("EXTENDED TRAINING ANALYSIS — Stage 1 (15k → 45k)")
    print("=" * 90)
    
    stage1_dir = EXTENDED_ROOT / 'stage1'
    if not stage1_dir.exists():
        print("  Stage 1 extended directory not found!")
        return
    
    strategies = ['baseline', 'gen_Img2Img', 'gen_augmenters', 'gen_cycleGAN', 'std_randaugment']
    datasets = ['bdd10k', 'iddaw']
    models = ['pspnet_r50', 'segformer_mit-b3']
    
    results = []
    
    for strat in strategies:
        for ds in datasets:
            for model in models:
                # Extended (45k) result
                if strat in ('baseline', 'std_randaugment'):
                    ext_dir = stage1_dir / strat / ds / model
                    s1_dir = S1_WEIGHTS / strat / ds / model
                else:
                    ext_dir = stage1_dir / strat / ds / f'{model}_ratio0p50'
                    s1_dir = S1_WEIGHTS / strat / ds / f'{model}_ratio0p50'
                
                ext_miou = load_miou(find_results_json(ext_dir))
                s1_miou = load_miou(find_results_json(s1_dir))
                
                # Also get baseline at both points for gap analysis
                base_s1_dir = S1_WEIGHTS / 'baseline' / ds / model
                base_ext_dir = stage1_dir / 'baseline' / ds / model
                base_s1 = load_miou(find_results_json(base_s1_dir))
                base_ext = load_miou(find_results_json(base_ext_dir))
                
                results.append({
                    'strategy': strat,
                    'dataset': ds,
                    'model': model,
                    'miou_15k': s1_miou,
                    'miou_45k': ext_miou,
                    'diff': (ext_miou - s1_miou) if (ext_miou and s1_miou) else None,
                    'base_15k': base_s1,
                    'base_45k': base_ext,
                })
    
    # Print results table
    print(f"\n{'Strategy/Dataset/Model':<45} {'15k':>8} {'45k':>8} {'Diff':>8} {'Gap@15k':>9} {'Gap@45k':>9}")
    print("-" * 90)
    
    for strat in strategies:
        strat_results = [r for r in results if r['strategy'] == strat]
        for r in strat_results:
            s1_str = f"{r['miou_15k']:.2f}" if r['miou_15k'] else "—"
            ext_str = f"{r['miou_45k']:.2f}" if r['miou_45k'] else "—"
            diff_str = f"{r['diff']:+.2f}" if r['diff'] else "—"
            
            # Gap over baseline
            if r['strategy'] != 'baseline' and r['miou_15k'] and r['base_15k']:
                gap_15k = r['miou_15k'] - r['base_15k']
                gap_15k_str = f"{gap_15k:+.2f}"
            else:
                gap_15k_str = "—"
            
            if r['strategy'] != 'baseline' and r['miou_45k'] and r['base_45k']:
                gap_45k = r['miou_45k'] - r['base_45k']
                gap_45k_str = f"{gap_45k:+.2f}"
            else:
                gap_45k_str = "—"
            
            print(f"  {r['strategy']}/{r['dataset']}/{r['model']:<39} {s1_str:>8} {ext_str:>8} {diff_str:>8} {gap_15k_str:>9} {gap_45k_str:>9}")
        print()
    
    # Augmentation gap analysis
    print("=" * 90)
    print("AUGMENTATION GAP ANALYSIS — Does the gap close with 3× training?")
    print("=" * 90)
    
    gap_changes = []
    for r in results:
        if r['strategy'] == 'baseline':
            continue
        if all(v is not None for v in [r['miou_15k'], r['miou_45k'], r['base_15k'], r['base_45k']]):
            gap_15k = r['miou_15k'] - r['base_15k']
            gap_45k = r['miou_45k'] - r['base_45k']
            gap_change = gap_45k - gap_15k
            gap_changes.append({
                'strategy': r['strategy'],
                'dataset': r['dataset'],
                'model': r['model'],
                'gap_15k': gap_15k,
                'gap_45k': gap_45k,
                'gap_change': gap_change,
            })
    
    print(f"\n{'Config':<45} {'Gap@15k':>9} {'Gap@45k':>9} {'Change':>9} {'Verdict':<15}")
    print("-" * 90)
    
    for gc in gap_changes:
        if gc['gap_change'] < -0.5:
            verdict = "GAP CLOSING"
        elif gc['gap_change'] > 0.5:
            verdict = "GAP WIDENING"
        else:
            verdict = "GAP STABLE"
        
        print(f"  {gc['strategy']}/{gc['dataset']}/{gc['model']:<39} {gc['gap_15k']:>+9.2f} {gc['gap_45k']:>+9.2f} {gc['gap_change']:>+9.2f} {verdict:<15}")
    
    if gap_changes:
        avg_change = sum(gc['gap_change'] for gc in gap_changes) / len(gap_changes)
        avg_15k = sum(gc['gap_15k'] for gc in gap_changes) / len(gap_changes)
        avg_45k = sum(gc['gap_45k'] for gc in gap_changes) / len(gap_changes)
        closing = sum(1 for gc in gap_changes if gc['gap_change'] < -0.5)
        stable = sum(1 for gc in gap_changes if -0.5 <= gc['gap_change'] <= 0.5)
        widening = sum(1 for gc in gap_changes if gc['gap_change'] > 0.5)
        
        print(f"\n  Average gap: {avg_15k:+.2f} pp @15k → {avg_45k:+.2f} pp @45k (change: {avg_change:+.2f})")
        print(f"  Gap closing: {closing} | Stable: {stable} | Widening: {widening}")
        
        print("\n  INTERPRETATION:")
        if avg_change < -0.5:
            print("  → Augmentation gap CLOSES with extended training")
            print("  → Augmentation mainly helps convergence speed, not final quality")
        elif avg_change > 0.5:
            print("  → Augmentation gap WIDENS with extended training")
            print("  → Augmentation provides genuine additional training signal")
        else:
            print("  → Augmentation gap is STABLE at 3× training")
            print("  → Augmentation gains are real and independent of training duration")
    
    # Per-strategy summary
    print("\n" + "=" * 90)
    print("PER-STRATEGY SUMMARY")
    print("=" * 90)
    
    for strat in strategies:
        if strat == 'baseline':
            continue
        strat_gc = [gc for gc in gap_changes if gc['strategy'] == strat]
        if strat_gc:
            avg_gap_15k = sum(gc['gap_15k'] for gc in strat_gc) / len(strat_gc)
            avg_gap_45k = sum(gc['gap_45k'] for gc in strat_gc) / len(strat_gc)
            avg_change = sum(gc['gap_change'] for gc in strat_gc) / len(strat_gc)
            print(f"  {strat:<25} gap@15k={avg_gap_15k:+.2f}  gap@45k={avg_gap_45k:+.2f}  change={avg_change:+.2f}")
    
    print()
    return results


def analyze_cityscapes_gen():
    """Analyze extended Cityscapes-Gen training (20k → 60k)."""
    print("=" * 90)
    print("EXTENDED TRAINING ANALYSIS — Cityscapes-Gen (20k → 60k)")
    print("=" * 90)
    
    cg_dir = EXTENDED_ROOT / 'cityscapes_gen'
    if not cg_dir.exists():
        print("  Cityscapes-Gen extended directory not found!")
        return
    
    strategies = ['baseline', 'gen_augmenters', 'gen_Img2Img', 'gen_CUT', 'std_randaugment']
    models = ['pspnet_r50', 'segformer_mit-b3']
    
    results = []
    
    for strat in strategies:
        for model in models:
            if strat in ('baseline', 'std_randaugment'):
                ext_dir = cg_dir / strat / 'cityscapes' / model
                cg_s_dir = CG_WEIGHTS / strat / 'cityscapes' / model
            else:
                ext_dir = cg_dir / strat / 'cityscapes' / f'{model}_ratio0p50'
                cg_s_dir = CG_WEIGHTS / strat / 'cityscapes' / f'{model}_ratio0p50'
            
            # Check for 60k checkpoint
            ckpt_60k = ext_dir / 'iter_60000.pth'
            ext_miou = load_miou(find_results_json(ext_dir)) if ckpt_60k.exists() else None
            cg_miou = load_miou(find_results_json(cg_s_dir))
            
            results.append({
                'strategy': strat,
                'model': model,
                'miou_20k': cg_miou,
                'miou_60k': ext_miou,
                'diff': (ext_miou - cg_miou) if (ext_miou and cg_miou) else None,
                'has_60k': ckpt_60k.exists(),
            })
    
    print(f"\n{'Strategy/Model':<45} {'20k':>8} {'60k':>8} {'Diff':>8} {'Status':<10}")
    print("-" * 80)
    
    for r in results:
        s20_str = f"{r['miou_20k']:.2f}" if r['miou_20k'] else "—"
        s60_str = f"{r['miou_60k']:.2f}" if r['miou_60k'] else "—"
        diff_str = f"{r['diff']:+.2f}" if r['diff'] else "—"
        status = "✅" if r['has_60k'] else "🔄 running"
        print(f"  {r['strategy']}/{r['model']:<41} {s20_str:>8} {s60_str:>8} {diff_str:>8} {status:<10}")
    
    completed = [r for r in results if r['diff'] is not None]
    if completed:
        avg = sum(r['diff'] for r in completed) / len(completed)
        print(f"\n  Average diff (completed): {avg:+.2f} pp ({len(completed)}/{len(results)})")
    
    print()
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze extended training ablation results")
    parser.add_argument('--stage', choices=['s1', 'cg', 'all'], default='all',
                        help='Which stage to analyze (default: all)')
    args = parser.parse_args()
    
    if args.stage in ('s1', 'all'):
        analyze_stage1()
    
    if args.stage in ('cg', 'all'):
        analyze_cityscapes_gen()


if __name__ == '__main__':
    main()
