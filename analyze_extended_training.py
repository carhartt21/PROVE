#!/usr/bin/env python3
"""
PROVE Extended Training Ablation Study Analyzer

Analyzes results from extended training experiments to evaluate the impact
of training duration on model performance.

Features:
- Scans for results at different iteration checkpoints
- Compares performance across training stages
- Finds convergence points and optimal training length
- Tracks improvement over extended training
- Supports filtering by strategy, dataset, model

Usage:
    # Analyze all extended training results
    python analyze_extended_training.py

    # Specify custom weights root
    python analyze_extended_training.py --weights-root /path/to/weights

    # Filter by specific strategy
    python analyze_extended_training.py --strategy gen_LANIT

    # Export results to CSV
    python analyze_extended_training.py --export-csv results.csv
"""

import os
import sys
import json
import re
import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

# Default weights root for extended training
DEFAULT_WEIGHTS_ROOT = os.environ.get(
    'PROVE_WEIGHTS_ROOT', 
    '/scratch/aaa_exchange/AWARE/WEIGHTS_EXTENDED'
)


@dataclass
class ExtendedTrainingResult:
    """Single test result from extended training experiment."""
    strategy: str
    dataset: str
    model: str
    iteration: int  # The iteration checkpoint (e.g., 80000, 120000, 160000)
    miou: float
    pixel_acc: float = 0.0
    result_file: str = ""
    
    def __hash__(self):
        return hash((self.strategy, self.dataset, self.model, self.iteration))


@dataclass
class ConvergenceInfo:
    """Information about training convergence for a configuration."""
    strategy: str
    dataset: str
    model: str
    best_iteration: int
    best_miou: float
    baseline_miou: float  # mIoU at 80000 iterations (standard)
    improvement: float  # Absolute improvement over baseline
    relative_improvement: float  # Percentage improvement over baseline
    iterations_tested: List[int] = field(default_factory=list)


class ExtendedTrainingAnalyzer:
    """Analyzer for extended training ablation study results."""
    
    def __init__(self, weights_root: str = DEFAULT_WEIGHTS_ROOT):
        self.weights_root = Path(weights_root)
        self.results: List[ExtendedTrainingResult] = []
    
    def scan_results(self, verbose: bool = False) -> int:
        """
        Scan the weights directory for extended training results.
        
        Returns the number of results found.
        """
        self.results = []
        
        if not self.weights_root.exists():
            print(f"Warning: Weights root does not exist: {self.weights_root}")
            return 0
        
        # Expected structure: {strategy}/{dataset}/{model}/
        # Results in: {model}/test_results_iter_{N}.json or eval.json
        
        for strategy_dir in self.weights_root.iterdir():
            if not strategy_dir.is_dir():
                continue
            strategy = strategy_dir.name
            
            for dataset_dir in strategy_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                dataset = dataset_dir.name
                
                for model_dir in dataset_dir.iterdir():
                    if not model_dir.is_dir():
                        continue
                    model = model_dir.name
                    
                    # Look for test results at different iterations
                    results_found = self._scan_model_results(
                        model_dir, strategy, dataset, model, verbose
                    )
                    
                    if verbose and results_found:
                        print(f"Found {results_found} results in {strategy}/{dataset}/{model}")
        
        return len(self.results)
    
    def _scan_model_results(self, model_dir: Path, strategy: str, 
                           dataset: str, model: str, verbose: bool) -> int:
        """Scan a model directory for test results at different iterations."""
        count = 0
        
        # Pattern 1: test_results_iter_{N}.json
        for result_file in model_dir.glob("test_results_iter_*.json"):
            iteration = self._extract_iteration_from_filename(result_file.name)
            if iteration is not None:
                result = self._parse_result_file(
                    result_file, strategy, dataset, model, iteration
                )
                if result:
                    self.results.append(result)
                    count += 1
        
        # Pattern 2: eval_{N}.json
        for result_file in model_dir.glob("eval_*.json"):
            iteration = self._extract_iteration_from_filename(result_file.name)
            if iteration is not None:
                result = self._parse_result_file(
                    result_file, strategy, dataset, model, iteration
                )
                if result:
                    self.results.append(result)
                    count += 1
        
        # Pattern 3: Standard eval.json (assumed to be at 80000)
        eval_file = model_dir / "eval.json"
        if eval_file.exists():
            # Check if this might be from extended training
            # Look for nearby checkpoint info
            iteration = self._infer_iteration_from_context(model_dir)
            result = self._parse_result_file(
                eval_file, strategy, dataset, model, iteration
            )
            if result:
                self.results.append(result)
                count += 1
        
        return count
    
    def _extract_iteration_from_filename(self, filename: str) -> Optional[int]:
        """Extract iteration number from result filename."""
        # Match patterns like: test_results_iter_80000.json, eval_160000.json
        patterns = [
            r'iter_(\d+)',
            r'eval_(\d+)',
            r'_(\d{5,})\.json',  # 5+ digit numbers before .json
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return int(match.group(1))
        
        return None
    
    def _infer_iteration_from_context(self, model_dir: Path) -> int:
        """
        Try to infer iteration count from checkpoint files or config.
        Default to 80000 (standard training length).
        """
        # Look for checkpoint files
        for ckpt_file in model_dir.glob("iter_*.pth"):
            match = re.search(r'iter_(\d+)', ckpt_file.name)
            if match:
                return int(match.group(1))
        
        # Check config file
        config_file = model_dir / "config.json"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = json.load(f)
                if 'max_iters' in config:
                    return config['max_iters']
            except (json.JSONDecodeError, KeyError):
                pass
        
        # Default to standard training length
        return 80000
    
    def _parse_result_file(self, result_file: Path, strategy: str,
                          dataset: str, model: str, iteration: int) -> Optional[ExtendedTrainingResult]:
        """Parse a result JSON file and extract metrics."""
        try:
            with open(result_file) as f:
                data = json.load(f)
            
            miou = None
            pixel_acc = 0.0
            
            # Try different JSON structures
            if isinstance(data, dict):
                # Standard format: {"mIoU": X, ...} or {"test/mIoU": X, ...}
                if 'mIoU' in data:
                    miou = float(data['mIoU'])
                elif 'test/mIoU' in data:
                    miou = float(data['test/mIoU'])
                elif 'miou' in data:
                    miou = float(data['miou'])
                elif 'metric' in data:
                    if isinstance(data['metric'], dict):
                        miou = data['metric'].get('mIoU', data['metric'].get('miou'))
                    else:
                        miou = float(data['metric'])
                
                # Extract pixel accuracy if available
                pixel_acc = float(data.get('aAcc', data.get('test/aAcc', data.get('pixel_acc', 0.0))))
            
            if miou is not None:
                return ExtendedTrainingResult(
                    strategy=strategy,
                    dataset=dataset,
                    model=model,
                    iteration=iteration,
                    miou=miou,
                    pixel_acc=pixel_acc,
                    result_file=str(result_file)
                )
        
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Warning: Could not parse {result_file}: {e}")
        
        return None
    
    def get_summary_by_iteration(self) -> Dict[int, Dict[str, float]]:
        """
        Get average metrics for each iteration checkpoint.
        
        Returns:
            Dict mapping iteration -> {mIoU, pixel_acc, count}
        """
        iteration_metrics = defaultdict(lambda: {'sum_miou': 0.0, 'sum_acc': 0.0, 'count': 0})
        
        for result in self.results:
            iteration_metrics[result.iteration]['sum_miou'] += result.miou
            iteration_metrics[result.iteration]['sum_acc'] += result.pixel_acc
            iteration_metrics[result.iteration]['count'] += 1
        
        summary = {}
        for iteration, data in sorted(iteration_metrics.items()):
            count = data['count']
            summary[iteration] = {
                'mIoU': data['sum_miou'] / count,
                'pixel_acc': data['sum_acc'] / count,
                'count': count
            }
        
        return summary
    
    def get_summary_by_strategy(self) -> Dict[str, Dict[int, float]]:
        """
        Get mIoU for each strategy at each iteration.
        
        Returns:
            Dict mapping strategy -> {iteration -> mIoU}
        """
        strategy_data = defaultdict(lambda: defaultdict(list))
        
        for result in self.results:
            strategy_data[result.strategy][result.iteration].append(result.miou)
        
        summary = {}
        for strategy, iterations in strategy_data.items():
            summary[strategy] = {
                iter_: sum(mious) / len(mious) 
                for iter_, mious in iterations.items()
            }
        
        return summary
    
    def get_convergence_analysis(self) -> List[ConvergenceInfo]:
        """
        Analyze convergence for each strategy-dataset-model configuration.
        
        Returns:
            List of ConvergenceInfo objects with convergence details.
        """
        # Group results by configuration
        config_results: Dict[Tuple[str, str, str], Dict[int, float]] = defaultdict(dict)
        
        for result in self.results:
            key = (result.strategy, result.dataset, result.model)
            config_results[key][result.iteration] = result.miou
        
        convergence_list = []
        
        for (strategy, dataset, model), iter_mious in config_results.items():
            if not iter_mious:
                continue
            
            iterations = sorted(iter_mious.keys())
            
            # Find best performance
            best_iter = max(iter_mious, key=iter_mious.get)
            best_miou = iter_mious[best_iter]
            
            # Get baseline (80000 iterations, or minimum available)
            baseline_iter = 80000 if 80000 in iter_mious else min(iterations)
            baseline_miou = iter_mious[baseline_iter]
            
            # Calculate improvement
            improvement = best_miou - baseline_miou
            rel_improvement = (improvement / baseline_miou * 100) if baseline_miou > 0 else 0.0
            
            convergence_list.append(ConvergenceInfo(
                strategy=strategy,
                dataset=dataset,
                model=model,
                best_iteration=best_iter,
                best_miou=best_miou,
                baseline_miou=baseline_miou,
                improvement=improvement,
                relative_improvement=rel_improvement,
                iterations_tested=iterations
            ))
        
        return convergence_list
    
    def get_improvement_stats(self) -> Dict[str, Any]:
        """
        Calculate overall statistics about improvement from extended training.
        
        Returns:
            Dict with overall statistics.
        """
        convergence_list = self.get_convergence_analysis()
        
        if not convergence_list:
            return {}
        
        improvements = [c.improvement for c in convergence_list]
        rel_improvements = [c.relative_improvement for c in convergence_list]
        best_iters = [c.best_iteration for c in convergence_list]
        
        # Count configurations where extended training helped
        improved_count = sum(1 for imp in improvements if imp > 0)
        
        return {
            'total_configs': len(convergence_list),
            'improved_configs': improved_count,
            'improved_percentage': improved_count / len(convergence_list) * 100,
            'mean_improvement': sum(improvements) / len(improvements),
            'max_improvement': max(improvements),
            'min_improvement': min(improvements),
            'mean_rel_improvement': sum(rel_improvements) / len(rel_improvements),
            'most_common_best_iter': max(set(best_iters), key=best_iters.count) if best_iters else None,
            'best_iter_distribution': dict(sorted(
                {iter_: best_iters.count(iter_) for iter_ in set(best_iters)}.items()
            ))
        }
    
    def filter_results(self, strategy: Optional[str] = None,
                      dataset: Optional[str] = None,
                      model: Optional[str] = None,
                      min_iteration: Optional[int] = None,
                      max_iteration: Optional[int] = None) -> 'ExtendedTrainingAnalyzer':
        """
        Create a new analyzer with filtered results.
        
        Returns:
            New ExtendedTrainingAnalyzer with filtered results.
        """
        filtered = ExtendedTrainingAnalyzer(str(self.weights_root))
        
        for result in self.results:
            if strategy and result.strategy != strategy:
                continue
            if dataset and result.dataset.lower() != dataset.lower():
                continue
            if model and result.model != model:
                continue
            if min_iteration and result.iteration < min_iteration:
                continue
            if max_iteration and result.iteration > max_iteration:
                continue
            
            filtered.results.append(result)
        
        return filtered
    
    def print_summary(self):
        """Print a formatted summary of the analysis."""
        try:
            from tabulate import tabulate
            HAS_TABULATE = True
        except ImportError:
            HAS_TABULATE = False
        
        print("\n" + "=" * 70)
        print("PROVE Extended Training Ablation Study - Analysis Summary")
        print("=" * 70)
        
        print(f"\nWeights root: {self.weights_root}")
        print(f"Total results: {len(self.results)}")
        
        # Summary by iteration
        iter_summary = self.get_summary_by_iteration()
        if iter_summary:
            print("\n" + "-" * 50)
            print("Average Performance by Iteration:")
            print("-" * 50)
            
            if HAS_TABULATE:
                table = [[iter_, f"{data['mIoU']:.2f}", data['count']] 
                        for iter_, data in sorted(iter_summary.items())]
                print(tabulate(table, headers=['Iteration', 'Avg mIoU', 'Count'], 
                             tablefmt='simple'))
            else:
                for iter_, data in sorted(iter_summary.items()):
                    print(f"  {iter_:>8}: mIoU={data['mIoU']:.2f}, n={data['count']}")
        
        # Improvement stats
        stats = self.get_improvement_stats()
        if stats:
            print("\n" + "-" * 50)
            print("Extended Training Impact:")
            print("-" * 50)
            print(f"  Configurations analyzed: {stats['total_configs']}")
            print(f"  Configurations improved: {stats['improved_configs']} ({stats['improved_percentage']:.1f}%)")
            print(f"  Mean improvement: {stats['mean_improvement']:+.2f} mIoU")
            print(f"  Max improvement: {stats['max_improvement']:+.2f} mIoU")
            print(f"  Mean relative improvement: {stats['mean_rel_improvement']:+.2f}%")
            
            if stats['best_iter_distribution']:
                print(f"\n  Best iteration distribution:")
                for iter_, count in stats['best_iter_distribution'].items():
                    print(f"    {iter_}: {count} configs")
        
        # Strategy summary
        strategy_summary = self.get_summary_by_strategy()
        if strategy_summary:
            print("\n" + "-" * 50)
            print("Performance by Strategy:")
            print("-" * 50)
            
            for strategy, iterations in sorted(strategy_summary.items()):
                sorted_iters = sorted(iterations.items())
                iter_str = ", ".join([f"{i}={m:.2f}" for i, m in sorted_iters])
                print(f"  {strategy}: {iter_str}")
        
        print("\n" + "=" * 70)
    
    def export_csv(self, output_path: str):
        """Export results to CSV format."""
        import csv
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['strategy', 'dataset', 'model', 'iteration', 'mIoU', 'pixel_acc'])
            
            for result in sorted(self.results, key=lambda r: (r.strategy, r.dataset, r.model, r.iteration)):
                writer.writerow([
                    result.strategy,
                    result.dataset,
                    result.model,
                    result.iteration,
                    f"{result.miou:.4f}",
                    f"{result.pixel_acc:.4f}"
                ])
        
        print(f"Exported to: {output_path}")
    
    def export_json(self, output_path: str):
        """Export results to JSON format."""
        data = {
            'weights_root': str(self.weights_root),
            'total_results': len(self.results),
            'results': [asdict(r) for r in self.results],
            'summary_by_iteration': self.get_summary_by_iteration(),
            'summary_by_strategy': self.get_summary_by_strategy(),
            'improvement_stats': self.get_improvement_stats()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze PROVE extended training ablation study results'
    )
    parser.add_argument('--weights-root', type=str, default=DEFAULT_WEIGHTS_ROOT,
                       help=f'Weights root directory (default: {DEFAULT_WEIGHTS_ROOT})')
    parser.add_argument('--strategy', type=str,
                       help='Filter by specific strategy')
    parser.add_argument('--dataset', type=str,
                       help='Filter by specific dataset')
    parser.add_argument('--model', type=str,
                       help='Filter by specific model')
    parser.add_argument('--export-csv', type=str,
                       help='Export results to CSV file')
    parser.add_argument('--export-json', type=str,
                       help='Export results to JSON file')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output during scanning')
    
    args = parser.parse_args()
    
    # Create analyzer and scan
    analyzer = ExtendedTrainingAnalyzer(args.weights_root)
    count = analyzer.scan_results(verbose=args.verbose)
    
    if count == 0:
        print(f"No results found in {args.weights_root}")
        return 1
    
    # Apply filters if specified
    if args.strategy or args.dataset or args.model:
        analyzer = analyzer.filter_results(
            strategy=args.strategy,
            dataset=args.dataset,
            model=args.model
        )
        print(f"After filtering: {len(analyzer.results)} results")
    
    # Print summary
    analyzer.print_summary()
    
    # Export if requested
    if args.export_csv:
        analyzer.export_csv(args.export_csv)
    
    if args.export_json:
        analyzer.export_json(args.export_json)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
