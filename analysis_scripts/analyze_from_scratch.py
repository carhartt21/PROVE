#!/usr/bin/env python3
"""
PROVE From-Scratch Experiment Analyzer

Analyzes from-scratch training results (no pretrained backbone) to determine
whether augmentation gains are genuine or masked by pretrained features.

Compares:
  - ratio=0.00 (100% generated images, no real data)
  - ratio=0.50 (50% generated, 50% real) [reference]
  - baseline (real images only, no augmentation)
  - std_* (standard augmentation strategies on real images)

Key questions:
  1. Does generative augmentation help when training from scratch?
  2. Which strategies benefit most without pretrained features?
  3. Is the pretrained-backbone "strategy inversion" confirmed?

Usage:
    # Full analysis with tables
    python analysis_scripts/analyze_from_scratch.py

    # Export CSV for IEEE repo
    python analysis_scripts/analyze_from_scratch.py --export-csv

    # Only show complete dataset-model combinations
    python analysis_scripts/analyze_from_scratch.py --complete-only

    # Compare with Stage 1 pretrained results
    python analysis_scripts/analyze_from_scratch.py --compare-pretrained
"""

import os
import sys
import json
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass, field

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# Configuration
# ============================================================================

WEIGHTS_FROM_SCRATCH = Path('${AWARE_DATA_ROOT}/WEIGHTS_FROM_SCRATCH')
WEIGHTS_STAGE1 = Path('${AWARE_DATA_ROOT}/WEIGHTS')
IEEE_REPO = Path('${HOME}/repositories/-IEEE-Access-01-26-Data-Augmentation')

DATASETS = ['bdd10k', 'mapillaryvistas', 'outside15k']
DATASET_DISPLAY = {
    'bdd10k': 'BDD10k',
    'mapillaryvistas': 'MapillaryVistas',
    'outside15k': 'OUTSIDE15k',
}

# All gen strategies
GEN_STRATEGIES = [
    'gen_albumentations_weather', 'gen_Attribute_Hallucination', 'gen_augmenters',
    'gen_automold', 'gen_CNetSeg', 'gen_CUT', 'gen_cyclediffusion',
    'gen_cycleGAN', 'gen_flux_kontext', 'gen_Img2Img', 'gen_IP2P',
    'gen_Qwen_Image_Edit', 'gen_stargan_v2', 'gen_step1x_new',
    'gen_step1x_v1p2', 'gen_SUSTechGAN', 'gen_TSIT', 'gen_UniControl',
    'gen_VisualCloze', 'gen_Weather_Effect_Generator',
]

STD_STRATEGIES = [
    'std_autoaugment', 'std_cutmix', 'std_mixup',
    'std_photometric_distort', 'std_randaugment',
]

ALL_STRATEGIES = ['baseline'] + STD_STRATEGIES + GEN_STRATEGIES

RATIOS = [0.0, 0.5]  # ratio=0.0 (100% gen) and ratio=0.5 (50/50)


# ============================================================================
# Data Collection
# ============================================================================

@dataclass
class Result:
    strategy: str
    dataset: str
    model: str
    ratio: float
    mIoU: float
    mAcc: float
    aAcc: float
    fwIoU: float
    num_images: int
    result_dir: str
    timestamp: str
    per_domain: Dict = field(default_factory=dict)
    per_class: Dict = field(default_factory=dict)


def find_results_json(weights_dir: Path) -> Optional[Path]:
    """Find the latest results.json in a weights directory."""
    test_dir = weights_dir / 'test_results_detailed'
    if not test_dir.exists():
        return None
    
    results_files = sorted(test_dir.glob('*/results.json'))
    if not results_files:
        return None
    
    return results_files[-1]  # Latest timestamp


def load_result(results_path: Path) -> Optional[Dict]:
    """Load and parse results.json."""
    try:
        with open(results_path) as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, IOError):
        return None


def collect_from_scratch_results(
    ratios: List[float] = None,
    datasets: List[str] = None,
) -> List[Result]:
    """Collect all from-scratch results across strategies, datasets, and ratios."""
    ratios = ratios or RATIOS
    datasets = datasets or DATASETS
    results = []
    
    for strategy in ALL_STRATEGIES:
        for dataset in datasets:
            # Only gen_* strategies have ratio variants (0.0, 0.5, etc.)
            # baseline and std_* always use real data only
            if not strategy.startswith('gen_'):
                effective_ratios = [1.0]  # real-only, no ratio concept
            else:
                effective_ratios = ratios
            
            for ratio in effective_ratios:
                # Determine model directory name
                if strategy.startswith('gen_') and ratio != 1.0:
                    model_dir = f'segformer_mit-b3_ratio{ratio:.2f}'.replace('.', 'p')
                else:
                    model_dir = 'segformer_mit-b3'
                
                weights_dir = WEIGHTS_FROM_SCRATCH / strategy / dataset / model_dir
                
                if not weights_dir.exists():
                    continue
                
                results_path = find_results_json(weights_dir)
                if results_path is None:
                    continue
                
                data = load_result(results_path)
                if data is None or 'overall' not in data:
                    continue
                
                overall = data['overall']
                result = Result(
                    strategy=strategy,
                    dataset=dataset,
                    model='segformer_mit-b3',
                    ratio=ratio if strategy.startswith('gen_') else 1.0,
                    mIoU=overall.get('mIoU', 0.0),
                    mAcc=overall.get('mAcc', 0.0),
                    aAcc=overall.get('aAcc', 0.0),
                    fwIoU=overall.get('fwIoU', 0.0),
                    num_images=overall.get('num_images', 0),
                    result_dir=str(results_path.parent),
                    timestamp=results_path.parent.name,
                    per_domain=data.get('per_domain', {}),
                    per_class=data.get('per_class', {}),
                )
                results.append(result)
    
    return results


def collect_stage1_results(datasets: List[str] = None) -> List[Result]:
    """Collect Stage 1 pretrained results for comparison."""
    datasets = datasets or DATASETS
    results = []
    
    for strategy in ALL_STRATEGIES:
        for dataset in datasets:
            # Stage 1 uses different model dirs
            if strategy.startswith('gen_'):
                model_dir = 'segformer_mit-b3_ratio0p50'
            else:
                model_dir = 'segformer_mit-b3'
            
            weights_dir = WEIGHTS_STAGE1 / strategy / dataset / model_dir
            
            if not weights_dir.exists():
                continue
            
            results_path = find_results_json(weights_dir)
            if results_path is None:
                continue
            
            data = load_result(results_path)
            if data is None or 'overall' not in data:
                continue
            
            overall = data['overall']
            result = Result(
                strategy=strategy,
                dataset=dataset,
                model='segformer_mit-b3',
                ratio=0.5 if strategy.startswith('gen_') else 1.0,
                mIoU=overall.get('mIoU', 0.0),
                mAcc=overall.get('mAcc', 0.0),
                aAcc=overall.get('aAcc', 0.0),
                fwIoU=overall.get('fwIoU', 0.0),
                num_images=overall.get('num_images', 0),
                result_dir=str(results_path.parent),
                timestamp=results_path.parent.name,
                per_domain=data.get('per_domain', {}),
                per_class=data.get('per_class', {}),
            )
            results.append(result)
    
    return results


# ============================================================================
# Analysis Functions
# ============================================================================

def get_baseline_miou(results: List[Result], dataset: str, ratio: float = 1.0) -> Optional[float]:
    """Get baseline mIoU for a dataset."""
    for r in results:
        if r.strategy == 'baseline' and r.dataset == dataset:
            return r.mIoU
    return None


def analyze_strategy_gains(results: List[Result]) -> Dict:
    """Compute mIoU gains over baseline for each strategy × dataset."""
    gains = {}
    
    for dataset in DATASETS:
        baseline_miou = get_baseline_miou(results, dataset)
        if baseline_miou is None:
            continue
        
        gains[dataset] = {}
        for r in results:
            if r.dataset == dataset and r.strategy != 'baseline':
                key = (r.strategy, r.ratio)
                gains[dataset][key] = {
                    'mIoU': r.mIoU,
                    'gain': r.mIoU - baseline_miou,
                    'ratio': r.ratio,
                }
    
    return gains


def print_summary_table(results: List[Result], title: str = "From-Scratch Results"):
    """Print a summary table of results."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    
    # Group by ratio
    by_ratio = defaultdict(list)
    for r in results:
        by_ratio[r.ratio].append(r)
    
    for ratio in sorted(by_ratio.keys()):
        ratio_results = by_ratio[ratio]
        ratio_label = f"ratio={ratio:.2f}" if ratio < 1.0 else "real-only (baseline + std)"
        
        # For ratio < 1.0, only show gen_* strategies
        if ratio < 1.0:
            ratio_results = [r for r in ratio_results if r.strategy.startswith('gen_')]
        
        if not ratio_results:
            continue
        
        print(f"\n--- {ratio_label} ---")
        
        # Build table
        strategies_seen = set()
        data_by_strategy = {}
        for r in ratio_results:
            strategies_seen.add(r.strategy)
            if r.strategy not in data_by_strategy:
                data_by_strategy[r.strategy] = {}
            data_by_strategy[r.strategy][r.dataset] = r.mIoU
        
        # Get baselines
        baselines = {}
        for dataset in DATASETS:
            bl = get_baseline_miou(results, dataset)
            if bl is not None:
                baselines[dataset] = bl
        
        # Print header
        ds_headers = [DATASET_DISPLAY.get(d, d) for d in DATASETS if d in baselines]
        header = f"{'Strategy':<35}" + "".join(f"{'  ' + h:>16}" for h in ds_headers)
        print(header)
        print("-" * len(header))
        
        # Sort: baseline first, then std_*, then gen_*
        strategy_order = []
        for s in sorted(strategies_seen):
            if s == 'baseline':
                strategy_order.insert(0, s)
            elif s.startswith('std_'):
                strategy_order.append(s)
            else:
                strategy_order.append(s)
        
        for strategy in strategy_order:
            row = f"{strategy:<35}"
            for dataset in DATASETS:
                if dataset not in baselines:
                    continue
                miou = data_by_strategy.get(strategy, {}).get(dataset)
                if miou is not None:
                    gain = miou - baselines[dataset]
                    sign = '+' if gain >= 0 else ''
                    row += f"  {miou:6.2f} ({sign}{gain:.2f})"
                else:
                    row += f"  {'—':>14}"
            print(row)
    
    # Summary statistics
    print(f"\n--- Summary (ratio=0.0 only: gen strategies trained on 100% generated data) ---")
    for dataset in DATASETS:
        ds_label = DATASET_DISPLAY.get(dataset, dataset)
        
        # For ratio=0.0: only gen strategies are relevant
        gen_results_r0 = [r for r in results if r.dataset == dataset 
                          and r.strategy.startswith('gen_') and r.ratio == 0.0]
        gen_results_r5 = [r for r in results if r.dataset == dataset 
                          and r.strategy.startswith('gen_') and r.ratio == 0.5]
        
        if not gen_results_r0:
            print(f"\n  {ds_label}: No ratio=0.0 results yet")
            continue
        
        avg_miou = sum(r.mIoU for r in gen_results_r0) / len(gen_results_r0)
        best_gen = max(gen_results_r0, key=lambda r: r.mIoU)
        worst_gen = min(gen_results_r0, key=lambda r: r.mIoU)
        
        print(f"\n  {ds_label} ({len(gen_results_r0)} gen strategies at ratio=0.0):")
        print(f"    Avg mIoU: {avg_miou:.2f}")
        print(f"    Best:  {best_gen.strategy} ({best_gen.mIoU:.2f})")
        print(f"    Worst: {worst_gen.strategy} ({worst_gen.mIoU:.2f})")
        print(f"    Range: {worst_gen.mIoU:.2f} – {best_gen.mIoU:.2f} (Δ={best_gen.mIoU - worst_gen.mIoU:.2f})")
        
        if gen_results_r5:
            avg_r5 = sum(r.mIoU for r in gen_results_r5) / len(gen_results_r5)
            print(f"    vs ratio=0.5 avg: {avg_r5:.2f} (Δ={avg_miou - avg_r5:+.2f})")


def compare_with_pretrained(scratch_results: List[Result], pretrained_results: List[Result]):
    """Compare from-scratch vs pretrained results."""
    print(f"\n{'='*80}")
    print(f"  From-Scratch vs Pretrained Comparison")
    print(f"{'='*80}")
    
    for dataset in DATASETS:
        ds_label = DATASET_DISPLAY.get(dataset, dataset)
        
        # Get baselines
        scratch_bl = get_baseline_miou(scratch_results, dataset)
        pretrained_bl = get_baseline_miou(pretrained_results, dataset)
        
        if scratch_bl is None or pretrained_bl is None:
            continue
        
        print(f"\n--- {ds_label} ---")
        print(f"  Baseline: scratch={scratch_bl:.2f}  pretrained={pretrained_bl:.2f}  (Δ={pretrained_bl - scratch_bl:.2f})")
        
        # Compare best gen strategy
        scratch_gen = [r for r in scratch_results if r.dataset == dataset 
                       and r.strategy.startswith('gen_') and r.ratio == 0.0]
        pretrained_gen = [r for r in pretrained_results if r.dataset == dataset 
                          and r.strategy.startswith('gen_')]
        
        scratch_std = [r for r in scratch_results if r.dataset == dataset 
                       and r.strategy.startswith('std_')]
        pretrained_std = [r for r in pretrained_results if r.dataset == dataset 
                          and r.strategy.startswith('std_')]
        
        if scratch_gen:
            best_scratch_gen = max(scratch_gen, key=lambda r: r.mIoU)
            print(f"  Best gen (scratch):     {best_scratch_gen.strategy} "
                  f"({best_scratch_gen.mIoU:.2f}, {best_scratch_gen.mIoU - scratch_bl:+.2f} over baseline)")
        
        if pretrained_gen:
            best_pretrained_gen = max(pretrained_gen, key=lambda r: r.mIoU)
            print(f"  Best gen (pretrained):  {best_pretrained_gen.strategy} "
                  f"({best_pretrained_gen.mIoU:.2f}, {best_pretrained_gen.mIoU - pretrained_bl:+.2f} over baseline)")
        
        if scratch_std:
            best_scratch_std = max(scratch_std, key=lambda r: r.mIoU)
            gen_beats_std = (scratch_gen and scratch_std and 
                           max(r.mIoU for r in scratch_gen) > max(r.mIoU for r in scratch_std))
            print(f"  Best std (scratch):     {best_scratch_std.strategy} "
                  f"({best_scratch_std.mIoU:.2f}, {best_scratch_std.mIoU - scratch_bl:+.2f} over baseline)")
            print(f"  Gen beats std (scratch)?  {'YES ✓' if gen_beats_std else 'NO ✗'}")
        
        if pretrained_std:
            best_pretrained_std = max(pretrained_std, key=lambda r: r.mIoU)
            std_beats_gen = (pretrained_gen and pretrained_std and 
                            max(r.mIoU for r in pretrained_std) > max(r.mIoU for r in pretrained_gen))
            print(f"  Best std (pretrained):  {best_pretrained_std.strategy} "
                  f"({best_pretrained_std.mIoU:.2f}, {best_pretrained_std.mIoU - pretrained_bl:+.2f} over baseline)")
            print(f"  Std beats gen (pretrained)? {'YES ✓' if std_beats_gen else 'NO ✗'}")
        
        # Strategy inversion check
        if scratch_gen and pretrained_std and scratch_std and pretrained_gen:
            scratch_gen_better = max(r.mIoU for r in scratch_gen) > max(r.mIoU for r in scratch_std)
            pretrained_std_better = max(r.mIoU for r in pretrained_std) > max(r.mIoU for r in pretrained_gen)
            inversion = scratch_gen_better and pretrained_std_better
            print(f"  STRATEGY INVERSION: {'CONFIRMED ✓' if inversion else 'NOT CONFIRMED ✗'}")


# ============================================================================
# Export Functions
# ============================================================================

def export_csv(results: List[Result], output_path: Path):
    """Export results to CSV in the same format as downstream_results_stage1.csv."""
    fieldnames = [
        'strategy', 'dataset', 'model', 'ratio', 'test_type', 'result_type',
        'result_dir', 'timestamp', 'mIoU', 'mAcc', 'aAcc', 'fwIoU',
        'num_images', 'has_per_domain', 'has_per_class',
        'per_domain_metrics', 'per_class_metrics',
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in sorted(results, key=lambda x: (x.strategy, x.dataset, x.ratio)):
            writer.writerow({
                'strategy': r.strategy,
                'dataset': r.dataset,
                'model': r.model,
                'ratio': r.ratio,
                'test_type': 'test_results_detailed',
                'result_type': 'detailed',
                'result_dir': r.result_dir,
                'timestamp': r.timestamp,
                'mIoU': r.mIoU,
                'mAcc': r.mAcc,
                'aAcc': r.aAcc,
                'fwIoU': r.fwIoU,
                'num_images': r.num_images,
                'has_per_domain': bool(r.per_domain),
                'has_per_class': bool(r.per_class),
                'per_domain_metrics': str(r.per_domain) if r.per_domain else '',
                'per_class_metrics': str(r.per_class) if r.per_class else '',
            })
    
    print(f"Exported {len(results)} results to {output_path}")


def export_summary_csv(results: List[Result], output_path: Path):
    """Export a compact summary CSV with just strategy × dataset × ratio → mIoU."""
    fieldnames = ['strategy', 'strategy_type', 'dataset', 'ratio', 'mIoU', 'mAcc', 'aAcc']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in sorted(results, key=lambda x: (x.dataset, x.strategy, x.ratio)):
            strat_type = 'baseline' if r.strategy == 'baseline' else \
                         'std' if r.strategy.startswith('std_') else 'gen'
            writer.writerow({
                'strategy': r.strategy,
                'strategy_type': strat_type,
                'dataset': r.dataset,
                'ratio': r.ratio,
                'mIoU': round(r.mIoU, 4),
                'mAcc': round(r.mAcc, 4),
                'aAcc': round(r.aAcc, 4),
            })
    
    print(f"Exported {len(results)} results to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze from-scratch training results')
    parser.add_argument('--export-csv', action='store_true',
                       help='Export results to CSV files (PROVE repo + IEEE repo)')
    parser.add_argument('--complete-only', action='store_true',
                       help='Only show dataset-model combos with all results')
    parser.add_argument('--compare-pretrained', action='store_true',
                       help='Compare with Stage 1 pretrained results')
    parser.add_argument('--ratio', type=float, nargs='+', default=[0.0, 0.5],
                       help='Ratios to analyze (default: 0.0 0.5)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV path (default: auto)')
    args = parser.parse_args()
    
    print("Collecting from-scratch results...")
    results = collect_from_scratch_results(ratios=args.ratio)
    print(f"  Found {len(results)} results")
    
    if not results:
        print("ERROR: No results found!")
        sys.exit(1)
    
    # Count by type
    gen_count = sum(1 for r in results if r.strategy.startswith('gen_'))
    std_count = sum(1 for r in results if r.strategy.startswith('std_'))
    bl_count = sum(1 for r in results if r.strategy == 'baseline')
    print(f"  Breakdown: {bl_count} baseline, {std_count} std, {gen_count} gen")
    
    # Show completion status
    print(f"\n--- Completion Status ---")
    for dataset in DATASETS:
        ds_label = DATASET_DISPLAY.get(dataset, dataset)
        ds_results = [r for r in results if r.dataset == dataset]
        gen_0 = [r for r in ds_results if r.strategy.startswith('gen_') and r.ratio == 0.0]
        gen_5 = [r for r in ds_results if r.strategy.startswith('gen_') and r.ratio == 0.5]
        std_r = [r for r in ds_results if r.strategy.startswith('std_')]
        bl_r = [r for r in ds_results if r.strategy == 'baseline']
        print(f"  {ds_label}: baseline={'✅' if bl_r else '❌'} "
              f"std={len(std_r)}/{len(STD_STRATEGIES)} "
              f"gen_r0.0={len(gen_0)}/{len(GEN_STRATEGIES)} "
              f"gen_r0.5={len(gen_5)}/{len(GEN_STRATEGIES)}")
    
    # Print results table
    print_summary_table(results)
    
    # Compare with pretrained if requested
    if args.compare_pretrained:
        print("\nCollecting Stage 1 pretrained results...")
        pretrained_results = collect_stage1_results()
        print(f"  Found {len(pretrained_results)} pretrained results")
        if pretrained_results:
            compare_with_pretrained(results, pretrained_results)
    
    # Export CSV
    if args.export_csv:
        # Filter: ratio=0.0 only includes gen_* strategies
        export_results = [r for r in results if r.strategy.startswith('gen_')]
        
        # Full CSV to PROVE repo
        prove_csv = Path(__file__).parent.parent / 'downstream_results_from_scratch.csv'
        export_csv(export_results, prove_csv)
        
        # Full CSV to IEEE repo
        ieee_csv = IEEE_REPO / 'analysis' / 'data' / 'PROVE' / 'downstream_results_from_scratch.csv'
        if ieee_csv.parent.exists():
            export_csv(export_results, ieee_csv)
        else:
            print(f"WARNING: IEEE repo path not found: {ieee_csv.parent}")
        
        # Summary CSV (compact)
        summary_csv = Path(__file__).parent.parent / 'from_scratch_summary.csv'
        export_summary_csv(export_results, summary_csv)
        
        # Also copy to IEEE repo
        ieee_summary = IEEE_REPO / 'analysis' / 'data' / 'PROVE' / 'from_scratch_summary.csv'
        if ieee_summary.parent.exists():
            export_summary_csv(export_results, ieee_summary)


if __name__ == '__main__':
    main()
