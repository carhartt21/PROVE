#!/usr/bin/env python3
"""
PROVE Extended Training Convergence Analysis

Collects per-iteration test results from tExt_* jobs across the full training 
trajectory (iter_2000 → iter_60000) for:
- Stage 1 (S1): baseline, gen_Img2Img, gen_augmenters, gen_cycleGAN, std_randaugment
  on bdd10k, iddaw with pspnet_r50, segformer_mit-b3
- Cityscapes-Gen (CG): baseline, gen_augmenters, gen_Img2Img, gen_CUT, std_randaugment
  on cityscapes with pspnet_r50, segformer_mit-b3

Result directories span:
- WEIGHTS/{strat}/{ds}/{model}/test_results_detailed/iter_{N}/  (early S1 ckpts)
- WEIGHTS_EXTENDED_ABLATION/stage1/{strat}/{ds}/{model}/test_results_detailed/iter_{N}/
- WEIGHTS_CITYSCAPES_GEN/{strat}/cityscapes/{model}/test_results_detailed/iter_{N}/

Outputs:
- CSV: analysis_scripts/result_figures/extended_convergence/extended_convergence_results.csv
- IEEE CSV: {ieee_repo}/analysis/data/ablation/extended_convergence_results.csv
- Figures: convergence curves, augmentation gap evolution, diminishing returns
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# --- Configuration ---
WEIGHTS_ROOT = Path('${AWARE_DATA_ROOT}/WEIGHTS')
EXT_ROOT = Path('${AWARE_DATA_ROOT}/WEIGHTS_EXTENDED_ABLATION')
CG_ROOT = Path('${AWARE_DATA_ROOT}/WEIGHTS_CITYSCAPES_GEN')

IEEE_REPO = Path('${HOME}/repositories/-IEEE-Access-01-26-Data-Augmentation')
IEEE_DATA = IEEE_REPO / 'analysis' / 'data' / 'ablation'

OUTPUT_DIR = Path(__file__).parent / 'result_figures' / 'extended_convergence'

# S1 configs: tested in tExt jobs
S1_STRATEGIES = ['baseline', 'gen_Img2Img', 'gen_augmenters', 'gen_cycleGAN', 'std_randaugment']
S1_DATASETS = ['bdd10k', 'iddaw']
S1_MODELS = ['pspnet_r50', 'segformer_mit-b3']

# CG configs: tested in tExt jobs
CG_STRATEGIES = ['baseline', 'gen_augmenters', 'gen_Img2Img', 'gen_CUT', 'std_randaugment']
CG_DATASETS = ['cityscapes']
CG_MODELS = ['pspnet_r50', 'segformer_mit-b3']


def model_subdir(strategy, model):
    """Return the model subdirectory name (with or without _ratio0p50)."""
    if strategy in ('baseline', 'std_randaugment', 'std_autoaugment', 'std_cutmix', 'std_mixup', 'std_photometric_distort'):
        return model
    return f'{model}_ratio0p50'


def find_iter_results(base_dir, iteration):
    """Find results.json for a specific iteration inside test_results_detailed/iter_{N}/."""
    test_dir = base_dir / 'test_results_detailed' / f'iter_{iteration}'
    if not test_dir.exists():
        return None
    # Find the most recent timestamp subdir
    results = list(test_dir.glob('*/results.json'))
    if not results:
        return None
    return max(results, key=lambda p: p.stat().st_mtime)


def find_standard_results(base_dir):
    """Find the standard (non-iter-specific) results.json — used for final checkpoint."""
    test_dir = base_dir / 'test_results_detailed'
    if not test_dir.exists():
        return None
    # Look for results.json NOT in iter_* subdirs (standard test)
    results = []
    for p in test_dir.rglob('results.json'):
        # Skip iter_XXXXX subdirs
        rel = p.relative_to(test_dir)
        if not str(rel.parts[0]).startswith('iter_'):
            results.append(p)
    if not results:
        return None
    return max(results, key=lambda p: p.stat().st_mtime)


def load_result(results_path):
    """Load mIoU and per-domain data from a results.json."""
    if results_path is None:
        return None
    try:
        with open(results_path) as f:
            data = json.load(f)
        overall = data.get('overall', {})
        return {
            'mIoU': overall.get('mIoU'),
            'aAcc': overall.get('aAcc'),
            'per_domain': data.get('per_domain', {}),
        }
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def collect_s1_results():
    """Collect all S1 per-iteration results."""
    rows = []
    
    for strat in S1_STRATEGIES:
        for ds in S1_DATASETS:
            for model in S1_MODELS:
                subdir = model_subdir(strat, model)
                
                # Early checkpoints are in WEIGHTS/ (S1 base training at 15k)
                s1_dir = WEIGHTS_ROOT / strat / ds / subdir
                # Extended checkpoints in WEIGHTS_EXTENDED_ABLATION/stage1/
                ext_dir = EXT_ROOT / 'stage1' / strat / ds / subdir
                
                # Collect from both directories
                checked_iters = set()
                
                # Check S1 base for early checkpoints 
                for d in [s1_dir, ext_dir]:
                    td = d / 'test_results_detailed'
                    if not td.exists():
                        continue
                    for entry in td.iterdir():
                        if entry.is_dir() and entry.name.startswith('iter_'):
                            try:
                                it = int(entry.name.replace('iter_', ''))
                            except ValueError:
                                continue
                            if it in checked_iters:
                                continue
                            checked_iters.add(it)
                            
                            res_path = find_iter_results(d, it)
                            if res_path is None:
                                # Try the other directory
                                other = ext_dir if d == s1_dir else s1_dir
                                res_path = find_iter_results(other, it)
                            
                            res = load_result(res_path)
                            if res and res['mIoU'] is not None:
                                rows.append({
                                    'stage': 'S1',
                                    'strategy': strat,
                                    'dataset': ds,
                                    'model': model,
                                    'iteration': it,
                                    'mIoU': res['mIoU'],
                                    'aAcc': res.get('aAcc'),
                                })
                
                # Also check for standard (final) test results at 15k
                std_res_path = find_standard_results(s1_dir)
                std_res = load_result(std_res_path)
                if std_res and std_res['mIoU'] is not None and 15000 not in checked_iters:
                    rows.append({
                        'stage': 'S1',
                        'strategy': strat,
                        'dataset': ds,
                        'model': model,
                        'iteration': 15000,
                        'mIoU': std_res['mIoU'],
                        'aAcc': std_res.get('aAcc'),
                    })
    
    return rows


def collect_cg_results():
    """Collect all CG per-iteration results."""
    rows = []
    
    for strat in CG_STRATEGIES:
        for model in CG_MODELS:
            subdir = model_subdir(strat, model)
            cg_dir = CG_ROOT / strat / 'cityscapes' / subdir
            ext_dir = EXT_ROOT / 'cityscapes_gen' / strat / 'cityscapes' / subdir
            
            checked_iters = set()
            
            for d in [cg_dir, ext_dir]:
                td = d / 'test_results_detailed'
                if not td.exists():
                    continue
                for entry in td.iterdir():
                    if entry.is_dir() and entry.name.startswith('iter_'):
                        try:
                            it = int(entry.name.replace('iter_', ''))
                        except ValueError:
                            continue
                        if it in checked_iters:
                            continue
                        checked_iters.add(it)
                        
                        res_path = find_iter_results(d, it)
                        if res_path is None:
                            other = ext_dir if d == cg_dir else cg_dir
                            res_path = find_iter_results(other, it)
                        
                        res = load_result(res_path)
                        if res and res['mIoU'] is not None:
                            rows.append({
                                'stage': 'CG',
                                'strategy': strat,
                                'dataset': 'cityscapes',
                                'model': model,
                                'iteration': it,
                                'mIoU': res['mIoU'],
                                'aAcc': res.get('aAcc'),
                            })
            
            # Standard final result at 20k
            std_res_path = find_standard_results(cg_dir)
            std_res = load_result(std_res_path)
            if std_res and std_res['mIoU'] is not None and 20000 not in checked_iters:
                rows.append({
                    'stage': 'CG',
                    'strategy': strat,
                    'dataset': 'cityscapes',
                    'model': model,
                    'iteration': 20000,
                    'mIoU': std_res['mIoU'],
                    'aAcc': std_res.get('aAcc'),
                })
    
    return rows


def print_summary(rows):
    """Print a summary table of all results."""
    if not rows:
        print("No results found!")
        return
    
    print(f"\n{'='*100}")
    print(f"EXTENDED CONVERGENCE RESULTS — {len(rows)} data points")
    print(f"{'='*100}")
    
    # Group by stage
    for stage in ['S1', 'CG']:
        stage_rows = [r for r in rows if r['stage'] == stage]
        if not stage_rows:
            continue
        
        iters = sorted(set(r['iteration'] for r in stage_rows))
        strategies = sorted(set(r['strategy'] for r in stage_rows))
        datasets = sorted(set(r['dataset'] for r in stage_rows))
        models = sorted(set(r['model'] for r in stage_rows))
        
        print(f"\n--- {stage} ---")
        print(f"  Iterations: {iters}")
        print(f"  Strategies: {strategies}")
        print(f"  Datasets: {datasets}")
        print(f"  Models: {models}")
        print(f"  Data points: {len(stage_rows)}")
        
        # Per strategy × dataset × model: show convergence
        for strat in strategies:
            for ds in datasets:
                for model in models:
                    pts = sorted(
                        [r for r in stage_rows if r['strategy'] == strat and r['dataset'] == ds and r['model'] == model],
                        key=lambda x: x['iteration']
                    )
                    if not pts:
                        continue
                    vals = ' → '.join(f"{p['iteration']//1000}k:{p['mIoU']:.1f}" for p in pts)
                    print(f"  {strat}/{ds}/{model}: {vals}")


def _interpolate_configs(strat_rows, all_iters):
    """Interpolate missing checkpoint values per config using linear interpolation.
    
    Returns dict: iteration -> list of mIoU values (one per config, all configs present).
    """
    # Group by config (dataset/model)
    config_data = defaultdict(dict)
    for r in strat_rows:
        key = (r['dataset'], r['model'])
        config_data[key][r['iteration']] = r['mIoU']
    
    all_iters = sorted(all_iters)
    result = defaultdict(list)
    
    for config_key, iter_miou in config_data.items():
        known_iters = sorted(iter_miou.keys())
        known_vals = [iter_miou[it] for it in known_iters]
        
        if len(known_iters) < 2:
            # Can't interpolate with single point — use it for all
            for it in all_iters:
                result[it].append(known_vals[0])
            continue
        
        # Linear interpolation using numpy
        interpolated = np.interp(
            all_iters,
            known_iters,
            known_vals,
        )
        for it, val in zip(all_iters, interpolated):
            result[it].append(val)
    
    return result


def generate_convergence_curves(rows, output_dir):
    """Generate convergence curve figures."""
    if not HAS_MPL:
        print("matplotlib not available — skipping figures")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # IEEE formatting
    plt.rcParams.update({
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'figure.dpi': 150,
        'savefig.dpi': 300,
    })
    
    STRATEGY_COLORS = {
        'baseline': '#333333',
        'gen_Img2Img': '#e41a1c',
        'gen_augmenters': '#377eb8',
        'gen_cycleGAN': '#4daf4a',
        'gen_CUT': '#984ea3',
        'std_randaugment': '#ff7f00',
    }
    
    STRATEGY_MARKERS = {
        'baseline': 'o',
        'gen_Img2Img': 's',
        'gen_augmenters': '^',
        'gen_cycleGAN': 'D',
        'gen_CUT': 'v',
        'std_randaugment': 'P',
    }
    
    STRATEGY_LABELS = {
        'baseline': 'Baseline',
        'gen_Img2Img': 'Img2Img',
        'gen_augmenters': 'Augmenters',
        'gen_cycleGAN': 'CycleGAN',
        'gen_CUT': 'CUT',
        'std_randaugment': 'RandAugment',
    }
    
    # --- Figure 1: S1 convergence curves per dataset×model ---
    for stage in ['S1', 'CG']:
        stage_rows = [r for r in rows if r['stage'] == stage]
        if not stage_rows:
            continue
        
        datasets = sorted(set(r['dataset'] for r in stage_rows))
        models = sorted(set(r['model'] for r in stage_rows))
        strategies = sorted(set(r['strategy'] for r in stage_rows))
        
        n_cols = len(datasets)
        n_rows = len(models)
        
        fig, axes = plt.subplots(n_rows, max(n_cols, 1), figsize=(4.5 * max(n_cols, 1), 3.5 * n_rows), 
                                  squeeze=False)
        
        for row_idx, model in enumerate(models):
            for col_idx, ds in enumerate(datasets):
                ax = axes[row_idx][col_idx]
                
                for strat in strategies:
                    pts = sorted(
                        [r for r in stage_rows if r['strategy'] == strat and r['dataset'] == ds and r['model'] == model],
                        key=lambda x: x['iteration']
                    )
                    if not pts:
                        continue
                    
                    x = [p['iteration'] / 1000 for p in pts]
                    y = [p['mIoU'] for p in pts]
                    
                    color = STRATEGY_COLORS.get(strat, '#aaaaaa')
                    marker = STRATEGY_MARKERS.get(strat, 'x')
                    label = STRATEGY_LABELS.get(strat, strat)
                    lw = 2.0 if strat == 'baseline' else 1.2
                    ls = '-' if strat == 'baseline' else '--'
                    
                    ax.plot(x, y, color=color, marker=marker, markersize=3, 
                            linewidth=lw, linestyle=ls, label=label, alpha=0.9)
                
                model_short = model.replace('segformer_mit-b3', 'SegFormer-B3').replace('pspnet_r50', 'PSPNet-R50')
                ax.set_title(f'{ds} / {model_short}', fontweight='bold')
                ax.set_xlabel('Iterations (k)')
                ax.set_ylabel('mIoU (%)')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='lower right', framealpha=0.8, ncol=1)
        
        fig.suptitle(f'{stage} — mIoU Convergence Curves', fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        for ext in ['png', 'pdf']:
            fig.savefig(output_dir / f'{stage.lower()}_convergence_curves.{ext}', bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {stage}_convergence_curves.png/pdf")
    
    # --- Figure 2: Augmentation gap evolution ---
    for stage in ['S1', 'CG']:
        stage_rows = [r for r in rows if r['stage'] == stage]
        if not stage_rows:
            continue
        
        datasets = sorted(set(r['dataset'] for r in stage_rows))
        models = sorted(set(r['model'] for r in stage_rows))
        strategies = sorted(set(r['strategy'] for r in stage_rows))
        non_base = [s for s in strategies if s != 'baseline']
        
        n_cols = len(datasets)
        n_rows = len(models)
        
        fig, axes = plt.subplots(n_rows, max(n_cols, 1), figsize=(4.5 * max(n_cols, 1), 3.5 * n_rows),
                                  squeeze=False)
        
        for row_idx, model in enumerate(models):
            for col_idx, ds in enumerate(datasets):
                ax = axes[row_idx][col_idx]
                
                # Get baseline trajectory
                base_pts = sorted(
                    [r for r in stage_rows if r['strategy'] == 'baseline' and r['dataset'] == ds and r['model'] == model],
                    key=lambda x: x['iteration']
                )
                if not base_pts:
                    continue
                base_dict = {p['iteration']: p['mIoU'] for p in base_pts}
                
                for strat in non_base:
                    pts = sorted(
                        [r for r in stage_rows if r['strategy'] == strat and r['dataset'] == ds and r['model'] == model],
                        key=lambda x: x['iteration']
                    )
                    if not pts:
                        continue
                    
                    # Compute gap at each shared iteration
                    x = []
                    y_gap = []
                    for p in pts:
                        if p['iteration'] in base_dict:
                            gap = p['mIoU'] - base_dict[p['iteration']]
                            x.append(p['iteration'] / 1000)
                            y_gap.append(gap)
                    
                    color = STRATEGY_COLORS.get(strat, '#aaaaaa')
                    marker = STRATEGY_MARKERS.get(strat, 'x')
                    label = STRATEGY_LABELS.get(strat, strat)
                    
                    ax.plot(x, y_gap, color=color, marker=marker, markersize=3,
                            linewidth=1.2, linestyle='--', label=label, alpha=0.9)
                
                ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)
                model_short = model.replace('segformer_mit-b3', 'SegFormer-B3').replace('pspnet_r50', 'PSPNet-R50')
                ax.set_title(f'{ds} / {model_short}', fontweight='bold')
                ax.set_xlabel('Iterations (k)')
                ax.set_ylabel('mIoU Gap vs Baseline (pp)')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best', framealpha=0.8, ncol=1)
        
        fig.suptitle(f'{stage} — Augmentation Gap Evolution', fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        for ext in ['png', 'pdf']:
            fig.savefig(output_dir / f'{stage.lower()}_augmentation_gap.{ext}', bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {stage}_augmentation_gap.png/pdf")
    
    # --- Figure 3: Averaged convergence (all configs per strategy, interpolated) ---
    for stage in ['S1', 'CG']:
        stage_rows = [r for r in rows if r['stage'] == stage]
        if not stage_rows:
            continue
        
        strategies = sorted(set(r['strategy'] for r in stage_rows))
        # Get union of all iterations for this stage
        all_stage_iters = sorted(set(r['iteration'] for r in stage_rows))
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        for strat in strategies:
            strat_rows = [r for r in stage_rows if r['strategy'] == strat]
            
            # Interpolate missing checkpoints per config, then average
            interp_vals = _interpolate_configs(strat_rows, all_stage_iters)
            
            iters = sorted(interp_vals.keys())
            x = [it / 1000 for it in iters]
            y = [np.mean(interp_vals[it]) for it in iters]
            y_std = [np.std(interp_vals[it]) for it in iters]
            
            color = STRATEGY_COLORS.get(strat, '#aaaaaa')
            marker = STRATEGY_MARKERS.get(strat, 'x')
            label = STRATEGY_LABELS.get(strat, strat)
            lw = 2.0 if strat == 'baseline' else 1.2
            ls = '-' if strat == 'baseline' else '--'
            
            ax.plot(x, y, color=color, marker=marker, markersize=4,
                    linewidth=lw, linestyle=ls, label=label, alpha=0.9)
            ax.fill_between(x, [yi - si for yi, si in zip(y, y_std)],
                           [yi + si for yi, si in zip(y, y_std)],
                           color=color, alpha=0.1)
        
        ax.set_xlabel('Iterations (k)')
        ax.set_ylabel('Mean mIoU (%)')
        ax.set_title(f'{stage} — Averaged Convergence (±1σ)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', framealpha=0.8)
        plt.tight_layout()
        
        for ext in ['png', 'pdf']:
            fig.savefig(output_dir / f'{stage.lower()}_averaged_convergence.{ext}', bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {stage}_averaged_convergence.png/pdf")
    
    # --- Figure 4: Diminishing returns (marginal improvement, interpolated) ---
    for stage in ['S1', 'CG']:
        stage_rows = [r for r in rows if r['stage'] == stage]
        if not stage_rows:
            continue
        
        strategies = sorted(set(r['strategy'] for r in stage_rows))
        all_stage_iters = sorted(set(r['iteration'] for r in stage_rows))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        for strat in strategies:
            strat_rows = [r for r in stage_rows if r['strategy'] == strat]
            interp_vals = _interpolate_configs(strat_rows, all_stage_iters)
            
            iters = sorted(interp_vals.keys())
            y_mean = [np.mean(interp_vals[it]) for it in iters]
            
            if len(iters) < 2:
                continue
            
            color = STRATEGY_COLORS.get(strat, '#aaaaaa')
            marker = STRATEGY_MARKERS.get(strat, 'x')
            label = STRATEGY_LABELS.get(strat, strat)
            
            # Absolute improvement from first checkpoint
            x = [it / 1000 for it in iters]
            y_abs = [y - y_mean[0] for y in y_mean]
            ax1.plot(x, y_abs, color=color, marker=marker, markersize=3,
                    linewidth=1.2, label=label, alpha=0.9)
            
            # Marginal improvement (per 5k iters)
            x_marg = [(iters[i] + iters[i-1]) / 2000 for i in range(1, len(iters))]
            y_marg = [y_mean[i] - y_mean[i-1] for i in range(1, len(iters))]
            ax2.plot(x_marg, y_marg, color=color, marker=marker, markersize=3,
                    linewidth=1.2, label=label, alpha=0.9)
        
        ax1.set_xlabel('Iterations (k)')
        ax1.set_ylabel('Absolute Improvement (pp)')
        ax1.set_title('Cumulative Improvement', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=7, framealpha=0.8)
        
        ax2.axhline(y=0, color='black', linewidth=0.8, linestyle='-', alpha=0.3)
        ax2.set_xlabel('Iterations (k)')
        ax2.set_ylabel('Marginal Improvement (pp / step)')
        ax2.set_title('Marginal Returns', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=7, framealpha=0.8)
        
        fig.suptitle(f'{stage} — Diminishing Returns Analysis', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        for ext in ['png', 'pdf']:
            fig.savefig(output_dir / f'{stage.lower()}_diminishing_returns.{ext}', bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {stage}_diminishing_returns.png/pdf")


def export_csv(rows, output_dir, ieee_dir=None):
    """Export results to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort rows
    rows_sorted = sorted(rows, key=lambda r: (r['stage'], r['strategy'], r['dataset'], r['model'], r['iteration']))
    
    csv_path = output_dir / 'extended_convergence_results.csv'
    with open(csv_path, 'w') as f:
        f.write('stage,strategy,dataset,model,iteration,mIoU,aAcc\n')
        for r in rows_sorted:
            f.write(f"{r['stage']},{r['strategy']},{r['dataset']},{r['model']},"
                    f"{r['iteration']},{r['mIoU']:.4f},{r.get('aAcc', '') if r.get('aAcc') else ''}\n")
    
    print(f"  Saved {csv_path} ({len(rows_sorted)} rows)")
    
    if ieee_dir:
        ieee_dir = Path(ieee_dir)
        ieee_dir.mkdir(parents=True, exist_ok=True)
        ieee_path = ieee_dir / 'extended_convergence_results.csv'
        import shutil
        shutil.copy2(csv_path, ieee_path)
        print(f"  Copied to {ieee_path}")
    
    return csv_path


def print_convergence_analysis(rows):
    """Print key convergence insights."""
    print(f"\n{'='*100}")
    print("CONVERGENCE ANALYSIS INSIGHTS")
    print(f"{'='*100}")
    
    for stage in ['S1', 'CG']:
        stage_rows = [r for r in rows if r['stage'] == stage]
        if not stage_rows:
            continue
        
        strategies = sorted(set(r['strategy'] for r in stage_rows))
        non_base = [s for s in strategies if s != 'baseline']
        
        print(f"\n--- {stage} ---")
        
        # Get baseline at each iteration (averaged)
        base_rows = [r for r in stage_rows if r['strategy'] == 'baseline']
        base_by_iter = defaultdict(list)
        for r in base_rows:
            base_by_iter[r['iteration']].append(r['mIoU'])
        
        # Check if augmentation gap widens or narrows
        print(f"\n  Augmentation Gap Evolution (strategy - baseline, averaged across configs):")
        print(f"  {'Strategy':<20}", end='')
        
        # Get common iterations
        common_iters = sorted(set.intersection(*[
            set(r['iteration'] for r in stage_rows if r['strategy'] == s)
            for s in strategies
        ])) if strategies else []
        
        for it in common_iters:
            print(f"  {it//1000}k", end='')
        print(f"  {'Trend':<12}")
        print(f"  {'-'*20}", end='')
        for _ in common_iters:
            print(f"  {'---':>5}", end='')
        print(f"  {'-'*12}")
        
        for strat in non_base:
            strat_rows_s = [r for r in stage_rows if r['strategy'] == strat]
            iter_vals = defaultdict(list)
            for r in strat_rows_s:
                iter_vals[r['iteration']].append(r['mIoU'])
            
            print(f"  {STRATEGY_LABELS.get(strat, strat):<20}", end='')
            
            gaps = []
            for it in common_iters:
                if it in iter_vals and it in base_by_iter:
                    gap = np.mean(iter_vals[it]) - np.mean(base_by_iter[it])
                    gaps.append(gap)
                    print(f"  {gap:>+5.1f}", end='')
                else:
                    print(f"  {'—':>5}", end='')
            
            # Trend analysis
            if len(gaps) >= 3:
                early = np.mean(gaps[:len(gaps)//3])
                late = np.mean(gaps[-len(gaps)//3:])
                if late - early > 0.5:
                    trend = "↑ WIDENING"
                elif late - early < -0.5:
                    trend = "↓ CLOSING"
                else:
                    trend = "→ STABLE"
            else:
                trend = "?"
            print(f"  {trend}")
        
        # Marginal improvement analysis
        print(f"\n  Marginal Improvement (mean mIoU gain per step, all configs):")
        for strat in strategies:
            strat_rows_s = [r for r in stage_rows if r['strategy'] == strat]
            iter_vals = defaultdict(list)
            for r in strat_rows_s:
                iter_vals[r['iteration']].append(r['mIoU'])
            
            iters = sorted(iter_vals.keys())
            y_mean = [np.mean(iter_vals[it]) for it in iters]
            
            if len(iters) >= 2:
                early_gain = y_mean[min(3, len(iters)-1)] - y_mean[0]
                late_gain = y_mean[-1] - y_mean[max(0, len(iters)-4)]
                total_gain = y_mean[-1] - y_mean[0]
                
                label = STRATEGY_LABELS.get(strat, strat)
                print(f"  {label:<20} total: {total_gain:>+5.1f}pp | first-third: {early_gain:>+5.1f}pp | last-third: {late_gain:>+5.1f}pp")

STRATEGY_LABELS = {
    'baseline': 'Baseline',
    'gen_Img2Img': 'Img2Img',
    'gen_augmenters': 'Augmenters',
    'gen_cycleGAN': 'CycleGAN',
    'gen_CUT': 'CUT',
    'std_randaugment': 'RandAugment',
}


def main():
    parser = argparse.ArgumentParser(description='Extended training convergence analysis')
    parser.add_argument('--no-figures', action='store_true', help='Skip figure generation')
    parser.add_argument('--no-export', action='store_true', help='Skip IEEE export')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR),
                        help=f'Output directory (default: {OUTPUT_DIR})')
    args = parser.parse_args()
    
    print("=" * 100)
    print("PROVE Extended Training Convergence Analysis")
    print("=" * 100)
    
    # Collect data
    print("\nCollecting S1 results...")
    s1_rows = collect_s1_results()
    print(f"  {len(s1_rows)} S1 data points")
    
    print("\nCollecting CG results...")
    cg_rows = collect_cg_results()
    print(f"  {len(cg_rows)} CG data points")
    
    all_rows = s1_rows + cg_rows
    print(f"\nTotal: {len(all_rows)} data points")
    
    if not all_rows:
        print("ERROR: No results found!")
        return
    
    # Print summary
    print_summary(all_rows)
    
    # Convergence analysis
    print_convergence_analysis(all_rows)
    
    # Export CSV
    print(f"\nExporting CSV...")
    ieee_dir = IEEE_DATA if not args.no_export and IEEE_DATA.exists() else None
    export_csv(all_rows, args.output_dir, ieee_dir=ieee_dir)
    
    # Generate figures
    if not args.no_figures:
        print(f"\nGenerating figures...")
        generate_convergence_curves(all_rows, args.output_dir)
    
    print(f"\nDone! All outputs in {args.output_dir}")


if __name__ == '__main__':
    main()
