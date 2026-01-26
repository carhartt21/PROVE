#!/usr/bin/env python3
"""
PROVE Ratio Ablation Study Analyzer

Analyzes results from the ratio ablation study to understand
the impact of different real/generated image ratios on model performance.

Features:
- Collect results across all ratio values (0.125 - 1.0)
- Compare performance by ratio, strategy, dataset, and model
- Identify optimal ratios per configuration
- Generate summary statistics and tables

Usage:
    # Analyze all ratio ablation results
    python analyze_ratio_ablation.py

    # Specify custom weights root
    python analyze_ratio_ablation.py --weights-root /path/to/weights

    # Export to CSV
    python analyze_ratio_ablation.py --output results.csv --format csv

    # Show detailed breakdown
    python analyze_ratio_ablation.py --detailed
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass, field

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


# Default configuration
DEFAULT_WEIGHTS_ROOT = "/scratch/aaa_exchange/AWARE/WEIGHTS_RATIO_ABLATION"
DEFAULT_REGULAR_WEIGHTS_ROOT = "/scratch/aaa_exchange/AWARE/WEIGHTS"  # For baseline (0) and standard (0.5)
RATIOS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
TOP_5_STRATEGIES = [
    "gen_LANIT",
    "gen_step1x_new", 
    "gen_automold",
    "gen_TSIT",
    "gen_NST"
]


@dataclass
class RatioResult:
    """Container for a single ratio ablation result."""
    strategy: str
    dataset: str
    model: str
    ratio: float
    miou: float
    macc: float
    aacc: float
    fwiou: float
    timestamp: float = 0.0
    checkpoint_iter: int = 0
    stage: int = 0  # 1 = clear_day, 2 = all domains, 0 = unknown
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy': self.strategy,
            'dataset': self.dataset,
            'model': self.model,
            'ratio': self.ratio,
            'mIoU': self.miou,
            'mAcc': self.macc,
            'aAcc': self.aacc,
            'fwIoU': self.fwiou,
            'timestamp': self.timestamp,
            'checkpoint_iter': self.checkpoint_iter,
            'stage': self.stage
        }


class RatioAblationAnalyzer:
    """Analyzer for ratio ablation study results."""
    
    def __init__(self, weights_root: str = DEFAULT_WEIGHTS_ROOT,
                 regular_weights_root: str = DEFAULT_REGULAR_WEIGHTS_ROOT):
        self.weights_root = Path(weights_root)
        self.regular_weights_root = Path(regular_weights_root)
        self.results: List[RatioResult] = []
        self.ratio_pattern = re.compile(r'_ratio(\d+)p(\d+)')
        
    def scan_results(self, verbose: bool = False, include_regular: bool = True) -> int:
        """
        Scan the weights directory for ratio ablation results.
        
        Args:
            verbose: Print detailed progress information
            include_regular: Also scan regular WEIGHTS folder for baseline (ratio=0)
                            and standard training (ratio=0.5)
        
        Returns:
            Number of results found
        """
        self.results = []
        
        # Scan ablation weights directory
        if self.weights_root.exists():
            # Check for new stage-based structure (stage1/, stage2/)
            stage1_dir = self.weights_root / "stage1"
            stage2_dir = self.weights_root / "stage2"
            
            if stage1_dir.exists():
                if verbose:
                    print(f"Scanning Stage 1 directory: {stage1_dir}")
                self._scan_directory(stage1_dir, verbose=verbose, stage=1)
            
            if stage2_dir.exists():
                if verbose:
                    print(f"Scanning Stage 2 directory: {stage2_dir}")
                self._scan_directory(stage2_dir, verbose=verbose, stage=2)
            
            # Also scan for legacy flat structure (strategy/dataset/model)
            # if neither stage1 nor stage2 exist
            if not stage1_dir.exists() and not stage2_dir.exists():
                self._scan_directory(self.weights_root, verbose=verbose, stage=None)
        else:
            print(f"Warning: Weights root not found: {self.weights_root}")
        
        # Scan regular weights for baseline (ratio=0) and standard training (ratio=0.5)
        if include_regular and self.regular_weights_root.exists():
            self._scan_regular_weights(verbose=verbose)
        
        return len(self.results)
    
    def _scan_directory(self, weights_root: Path, verbose: bool = False, stage: Optional[int] = None):
        """Scan a weights directory for ratio ablation results."""
        # Walk through directory structure: strategy/dataset/model_ratio*/test_results/
        for strategy_dir in sorted(weights_root.iterdir()):
            if not strategy_dir.is_dir():
                continue
            
            strategy = strategy_dir.name
            
            # Skip non-strategy directories
            if strategy.startswith('_') or strategy in ['stage1', 'stage2', 'configs']:
                continue
            
            for dataset_dir in sorted(strategy_dir.iterdir()):
                if not dataset_dir.is_dir():
                    continue
                
                dataset = dataset_dir.name
                
                for model_dir in sorted(dataset_dir.iterdir()):
                    if not model_dir.is_dir():
                        continue
                    
                    model_name = model_dir.name
                    
                    # Extract ratio from model directory name
                    ratio = self._extract_ratio(model_name)
                    if ratio is None:
                        # Model without ratio suffix (ratio=1.0)
                        ratio = 1.0
                        base_model = model_name
                    else:
                        # Remove ratio suffix from model name
                        base_model = self.ratio_pattern.sub('', model_name)
                    
                    # Look for test results
                    result = self._find_latest_test_result(model_dir)
                    if result:
                        result.strategy = strategy
                        result.dataset = dataset
                        result.model = base_model
                        result.ratio = ratio
                        result.stage = stage if stage else 0
                        self.results.append(result)
                        
                        if verbose:
                            stage_str = f" [Stage {stage}]" if stage else ""
                            print(f"Found: {strategy}/{dataset}/{base_model} ratio={ratio:.3f} mIoU={result.miou:.2f}{stage_str}")
        
        return len(self.results)
    
    def _scan_regular_weights(self, verbose: bool = False):
        """
        Scan regular WEIGHTS folder for baseline and standard training results.
        
        - 'baseline' folder -> ratio=0.0 (no generated images, only real)
        - 'gen_*' folders -> ratio=0.5 (standard gen training with 50/50 split)
        
        Only scans for the top 5 strategies used in ratio ablation.
        """
        for strategy_dir in sorted(self.regular_weights_root.iterdir()):
            if not strategy_dir.is_dir():
                continue
            
            strategy = strategy_dir.name
            
            # Determine ratio based on strategy type
            if strategy == 'baseline':
                ratio = 0.0
            elif strategy in TOP_5_STRATEGIES:
                ratio = 0.5  # Standard gen training is 50/50
            else:
                continue  # Skip strategies not in our ablation study
            
            for dataset_dir in sorted(strategy_dir.iterdir()):
                if not dataset_dir.is_dir():
                    continue
                
                dataset = dataset_dir.name
                
                for model_dir in sorted(dataset_dir.iterdir()):
                    if not model_dir.is_dir():
                        continue
                    
                    model_name = model_dir.name
                    
                    # Look for test results
                    result = self._find_latest_test_result(model_dir)
                    if result:
                        result.strategy = strategy if strategy != 'baseline' else 'baseline'
                        result.dataset = dataset
                        result.model = model_name
                        result.ratio = ratio
                        
                        # Only add baseline results for the gen strategies we're studying
                        if strategy == 'baseline':
                            # Add a result entry for each gen strategy (as their baseline)
                            for gen_strategy in TOP_5_STRATEGIES:
                                baseline_result = RatioResult(
                                    strategy=gen_strategy,
                                    dataset=dataset,
                                    model=model_name,
                                    ratio=0.0,  # No generated images
                                    miou=result.miou,
                                    macc=result.macc,
                                    aacc=result.aacc,
                                    fwiou=result.fwiou,
                                    timestamp=result.timestamp,
                                    checkpoint_iter=result.checkpoint_iter
                                )
                                self.results.append(baseline_result)
                                
                                if verbose:
                                    print(f"Found baseline for {gen_strategy}/{dataset}/{model_name} ratio=0.0 mIoU={result.miou:.2f}")
                        else:
                            self.results.append(result)
                            
                            if verbose:
                                print(f"Found: {strategy}/{dataset}/{model_name} ratio=0.5 mIoU={result.miou:.2f}")
    
    def _extract_ratio(self, model_name: str) -> Optional[float]:
        """Extract ratio value from model directory name."""
        match = self.ratio_pattern.search(model_name)
        if match:
            integer_part = int(match.group(1))
            decimal_part = int(match.group(2))
            return float(f"{integer_part}.{decimal_part}")
        return None
    
    def _find_latest_test_result(self, model_dir: Path) -> Optional[RatioResult]:
        """Find the latest test result in a model directory."""
        # Pattern 1: test_results_detailed/TIMESTAMP/results.json (fine_grained_test.py)
        test_detailed_dir = model_dir / "test_results_detailed"
        if test_detailed_dir.exists():
            result = self._find_result_in_timestamped_dir(test_detailed_dir)
            if result:
                return result
        
        # Pattern 2: test_results/test/TIMESTAMP/TIMESTAMP.json (MMEngine)
        test_dir = model_dir / "test_results" / "test"
        if test_dir.exists():
            result = self._find_result_in_timestamped_dir(test_dir, mmengine_format=True)
            if result:
                return result
        
        return None
    
    def _find_result_in_timestamped_dir(self, base_dir: Path, mmengine_format: bool = False) -> Optional[RatioResult]:
        """Find the latest test result in a directory with timestamp subdirs."""
        # Find latest timestamped directory
        timestamp_dirs = []
        for item in base_dir.iterdir():
            if item.is_dir() and re.match(r'\d{8}_\d{6}', item.name):
                timestamp_dirs.append(item)
        
        if not timestamp_dirs:
            return None
        
        latest_dir = max(timestamp_dirs, key=lambda x: x.name)
        
        # Look for results JSON based on format
        if mmengine_format:
            # MMEngine: TIMESTAMP/TIMESTAMP.json
            json_files = list(latest_dir.glob("*.json"))
        else:
            # fine_grained_test.py: TIMESTAMP/results.json
            json_files = [latest_dir / "results.json"]
        
        for json_file in json_files:
            if not json_file.exists():
                continue
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Handle fine_grained_test.py format with 'overall' key
                if 'overall' in data:
                    overall = data['overall']
                    miou = overall.get('mIoU', 0.0)
                    macc = overall.get('mAcc', 0.0)
                    aacc = overall.get('aAcc', 0.0)
                    fwiou = overall.get('fwIoU', 0.0)
                else:
                    # Extract metrics (handle both 'mIoU' and 'test/mIoU' formats)
                    miou = data.get('mIoU') or data.get('test/mIoU', 0.0)
                    macc = data.get('mAcc') or data.get('test/mAcc', 0.0)
                    aacc = data.get('aAcc') or data.get('test/aAcc', 0.0)
                    fwiou = data.get('fwIoU') or data.get('test/fwIoU', 0.0)
                
                if miou is not None and miou > 0:
                    return RatioResult(
                        strategy='',
                        dataset='',
                        model='',
                        ratio=0.0,
                        miou=float(miou),
                        macc=float(macc) if macc else 0.0,
                        aacc=float(aacc) if aacc else 0.0,
                        fwiou=float(fwiou) if fwiou else 0.0,
                        timestamp=latest_dir.stat().st_mtime,
                        checkpoint_iter=self._get_checkpoint_iter(latest_dir)
                    )
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                continue
        
        return None
    
    def _get_checkpoint_iter(self, test_dir: Path) -> int:
        """Extract checkpoint iteration from test log."""
        for log_file in test_dir.glob("*.log"):
            try:
                with open(log_file, 'r', errors='ignore') as f:
                    content = f.read()
                    match = re.search(r'iter_(\d+)\.pth', content)
                    if match:
                        return int(match.group(1))
            except:
                pass
        return 0
    
    def get_summary_by_ratio(self) -> Dict[float, Dict[str, float]]:
        """Get average metrics for each ratio value."""
        ratio_metrics = defaultdict(lambda: {'miou': [], 'macc': [], 'aacc': [], 'fwiou': []})
        
        for result in self.results:
            ratio_metrics[result.ratio]['miou'].append(result.miou)
            ratio_metrics[result.ratio]['macc'].append(result.macc)
            ratio_metrics[result.ratio]['aacc'].append(result.aacc)
            ratio_metrics[result.ratio]['fwiou'].append(result.fwiou)
        
        summary = {}
        for ratio, metrics in sorted(ratio_metrics.items()):
            summary[ratio] = {
                'mIoU': sum(metrics['miou']) / len(metrics['miou']),
                'mAcc': sum(metrics['macc']) / len(metrics['macc']),
                'aAcc': sum(metrics['aacc']) / len(metrics['aacc']),
                'fwIoU': sum(metrics['fwiou']) / len(metrics['fwiou']),
                'count': len(metrics['miou'])
            }
        
        return summary
    
    def get_summary_by_strategy(self) -> Dict[str, Dict[float, float]]:
        """Get mIoU for each strategy at each ratio."""
        strategy_ratios = defaultdict(lambda: defaultdict(list))
        
        for result in self.results:
            strategy_ratios[result.strategy][result.ratio].append(result.miou)
        
        summary = {}
        for strategy, ratios in sorted(strategy_ratios.items()):
            summary[strategy] = {}
            for ratio, mious in sorted(ratios.items()):
                summary[strategy][ratio] = sum(mious) / len(mious)
        
        return summary
    
    def get_optimal_ratios(self) -> Dict[Tuple[str, str, str], Tuple[float, float]]:
        """Find optimal ratio for each strategy/dataset/model combination."""
        config_results = defaultdict(list)
        
        for result in self.results:
            key = (result.strategy, result.dataset, result.model)
            config_results[key].append((result.ratio, result.miou))
        
        optimal = {}
        for key, ratio_mious in config_results.items():
            best_ratio, best_miou = max(ratio_mious, key=lambda x: x[1])
            optimal[key] = (best_ratio, best_miou)
        
        return optimal
    
    def get_dataframe(self) -> 'pd.DataFrame':
        """Convert results to pandas DataFrame."""
        if not HAS_PANDAS:
            raise ImportError("pandas required for DataFrame export")
        
        return pd.DataFrame([r.to_dict() for r in self.results])
    
    def print_summary(self, detailed: bool = False):
        """Print analysis summary."""
        print("\n" + "=" * 70)
        print("PROVE Ratio Ablation Study - Analysis Summary")
        print("=" * 70)
        
        print(f"\nWeights Root: {self.weights_root}")
        print(f"Total Results: {len(self.results)}")
        
        # Summary by ratio
        ratio_summary = self.get_summary_by_ratio()
        print("\n" + "-" * 50)
        print("Average Metrics by Ratio")
        print("-" * 50)
        
        if HAS_TABULATE:
            table_data = []
            for ratio, metrics in sorted(ratio_summary.items()):
                table_data.append([
                    f"{ratio:.3f}",
                    f"{metrics['mIoU']:.2f}",
                    f"{metrics['mAcc']:.2f}",
                    f"{metrics['aAcc']:.2f}",
                    f"{metrics['fwIoU']:.2f}",
                    metrics['count']
                ])
            print(tabulate(table_data, 
                          headers=['Ratio', 'mIoU', 'mAcc', 'aAcc', 'fwIoU', 'Count'],
                          tablefmt='grid'))
        else:
            print(f"{'Ratio':<10} {'mIoU':<8} {'mAcc':<8} {'aAcc':<8} {'fwIoU':<8} {'Count':<6}")
            print("-" * 50)
            for ratio, metrics in sorted(ratio_summary.items()):
                print(f"{ratio:<10.3f} {metrics['mIoU']:<8.2f} {metrics['mAcc']:<8.2f} "
                      f"{metrics['aAcc']:<8.2f} {metrics['fwIoU']:<8.2f} {metrics['count']:<6}")
        
        # Find best ratio overall
        if ratio_summary:
            best_ratio = max(ratio_summary.items(), key=lambda x: x[1]['mIoU'])
            print(f"\nBest Overall Ratio: {best_ratio[0]:.3f} (mIoU: {best_ratio[1]['mIoU']:.2f})")
        
        # Summary by strategy
        strategy_summary = self.get_summary_by_strategy()
        print("\n" + "-" * 50)
        print("mIoU by Strategy and Ratio")
        print("-" * 50)
        
        if HAS_TABULATE and strategy_summary:
            # Get all ratios
            all_ratios = sorted(set(r for s in strategy_summary.values() for r in s.keys()))
            
            table_data = []
            for strategy, ratios in sorted(strategy_summary.items()):
                row = [strategy]
                for ratio in all_ratios:
                    row.append(f"{ratios.get(ratio, 0):.2f}" if ratio in ratios else "-")
                table_data.append(row)
            
            headers = ['Strategy'] + [f"{r:.3f}" for r in all_ratios]
            print(tabulate(table_data, headers=headers, tablefmt='grid'))
        
        # Optimal ratios
        optimal = self.get_optimal_ratios()
        if optimal and detailed:
            print("\n" + "-" * 50)
            print("Optimal Ratio per Configuration")
            print("-" * 50)
            
            optimal_counts = defaultdict(int)
            for (strategy, dataset, model), (ratio, miou) in optimal.items():
                optimal_counts[ratio] += 1
                if detailed:
                    print(f"  {strategy}/{dataset}/{model}: ratio={ratio:.3f} mIoU={miou:.2f}")
            
            print("\nOptimal Ratio Distribution:")
            for ratio, count in sorted(optimal_counts.items()):
                pct = count / len(optimal) * 100
                print(f"  {ratio:.3f}: {count} configs ({pct:.1f}%)")
    
    def export_csv(self, output_path: str):
        """Export results to CSV."""
        if not HAS_PANDAS:
            print("Error: pandas required for CSV export")
            return
        
        df = self.get_dataframe()
        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} results to {output_path}")
    
    def export_json(self, output_path: str):
        """Export results to JSON."""
        data = {
            'metadata': {
                'weights_root': str(self.weights_root),
                'timestamp': datetime.now().isoformat(),
                'total_results': len(self.results)
            },
            'results': [r.to_dict() for r in self.results],
            'summary_by_ratio': {str(k): v for k, v in self.get_summary_by_ratio().items()},
            'summary_by_strategy': self.get_summary_by_strategy()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Exported results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze PROVE ratio ablation study results'
    )
    parser.add_argument('--weights-root', type=str, default=DEFAULT_WEIGHTS_ROOT,
                       help=f'Weights root directory (default: {DEFAULT_WEIGHTS_ROOT})')
    parser.add_argument('--regular-weights-root', type=str, default=DEFAULT_REGULAR_WEIGHTS_ROOT,
                       help=f'Regular weights root for baseline/0.5 results (default: {DEFAULT_REGULAR_WEIGHTS_ROOT})')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--format', choices=['csv', 'json'], default='csv',
                       help='Output format (default: csv)')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed breakdown')
    parser.add_argument('--no-regular', action='store_true',
                       help='Do not include baseline/standard (ratio 0/0.5) from regular WEIGHTS folder')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output during scanning')
    
    args = parser.parse_args()
    
    # Create analyzer and scan results
    analyzer = RatioAblationAnalyzer(args.weights_root, args.regular_weights_root)
    count = analyzer.scan_results(verbose=args.verbose, include_regular=not args.no_regular)
    
    if count == 0:
        print("No results found.")
        return 1
    
    # Print summary
    analyzer.print_summary(detailed=args.detailed)
    
    # Export if requested
    if args.output:
        if args.format == 'csv':
            analyzer.export_csv(args.output)
        else:
            analyzer.export_json(args.output)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
