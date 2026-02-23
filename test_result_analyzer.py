#!/usr/bin/env python3
"""
PROVE Test Results Analyzer

This script analyzes test results stored in the PROVE weights directory
and generates comprehensive summaries of model performance across different
configurations, datasets, and domains.

Usage:
    python test_result_analyzer.py [--root PATH] [--format {table,json,csv}] [--verbose]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import re


# Strategy classification patterns
COMBINED_STRATEGY_PATTERN = re.compile(r'^(.+)\+(.+)$')  # e.g., gen_cycleGAN+std_cutmix
GEN_STRATEGY_PATTERN = re.compile(r'^gen_.+$')  # e.g., gen_cycleGAN
STD_STRATEGY_PATTERN = re.compile(r'^std_.+$')  # e.g., std_cutmix


def parse_strategy(strategy: str) -> Dict[str, str]:
    """
    Parse a strategy name and return its components.
    
    Args:
        strategy: Strategy name (e.g., 'gen_cycleGAN+std_cutmix', 'gen_cycleGAN', 'baseline')
    
    Returns:
        Dictionary with keys:
        - 'full': The full strategy name
        - 'type': 'combined', 'gen', 'std', or 'baseline'
        - 'gen_component': The generative component (if any)
        - 'std_component': The standard augmentation component (if any)
    """
    result = {
        'full': strategy,
        'type': 'unknown',
        'gen_component': None,
        'std_component': None
    }
    
    # Check for combined strategy (gen_*+std_* or baseline+std_*)
    combined_match = COMBINED_STRATEGY_PATTERN.match(strategy)
    if combined_match:
        first_part, second_part = combined_match.groups()
        result['type'] = 'combined'
        
        # Determine which part is gen and which is std
        if GEN_STRATEGY_PATTERN.match(first_part):
            result['gen_component'] = first_part
        elif first_part == 'baseline':
            result['gen_component'] = 'baseline'
        else:
            result['gen_component'] = first_part
            
        if STD_STRATEGY_PATTERN.match(second_part):
            result['std_component'] = second_part
        else:
            result['std_component'] = second_part
        return result
    
    # Check for gen_* strategy
    if GEN_STRATEGY_PATTERN.match(strategy):
        result['type'] = 'gen'
        result['gen_component'] = strategy
        return result
    
    # Check for std_* strategy
    if STD_STRATEGY_PATTERN.match(strategy):
        result['type'] = 'std'
        result['std_component'] = strategy
        return result
    
    # Check for baseline
    if strategy == 'baseline':
        result['type'] = 'baseline'
        return result
    
    # Unknown strategy type
    result['type'] = 'other'
    return result


def get_strategy_type(strategy: str) -> str:
    """Get the type of a strategy ('combined', 'gen', 'std', 'baseline', or 'other')."""
    return parse_strategy(strategy)['type']


def is_combined_strategy(strategy: str) -> bool:
    """Check if a strategy is a combined gen+std strategy."""
    return get_strategy_type(strategy) == 'combined'


def get_gen_component(strategy: str) -> Optional[str]:
    """Get the generative component of a strategy."""
    return parse_strategy(strategy)['gen_component']


def get_std_component(strategy: str) -> Optional[str]:
    """Get the standard augmentation component of a strategy."""
    return parse_strategy(strategy)['std_component']


class TestResultAnalyzer:
    """Analyze PROVE test results and generate performance summaries."""
    
    def __init__(self, root_dir: str = "${AWARE_DATA_ROOT}/WEIGHTS/"):
        self.root_dir = Path(root_dir)
        self.test_results = []
        
    def scan_directory(self, verbose: bool = False) -> List[Dict]:
        """
        Scan the weights directory for test results.
        
        Returns:
            List of dictionaries containing test result data
        """
        if not self.root_dir.exists():
            print(f"Error: Directory {self.root_dir} does not exist")
            return []
        
        print(f"Scanning {self.root_dir} for test results...")
        
        # Iterate through strategy directories
        for strategy_dir in sorted(self.root_dir.iterdir()):
            if not strategy_dir.is_dir():
                continue
            
            strategy = strategy_dir.name
            
            # Iterate through dataset directories
            for dataset_dir in sorted(strategy_dir.iterdir()):
                if not dataset_dir.is_dir():
                    continue
                
                dataset = dataset_dir.name
                
                # Iterate through model directories
                for model_dir in sorted(dataset_dir.iterdir()):
                    if not model_dir.is_dir():
                        continue
                    
                    model = model_dir.name
                    
                    # Look for test results directories
                    results = self._analyze_test_results(
                        model_dir, strategy, dataset, model
                    )
                    
                    if results:
                        self.test_results.extend(results)
                        
                        if verbose:
                            print(f"  Found: {strategy}/{dataset}/{model} - "
                                  f"{len(results)} test result(s)")
        
        print(f"Found {len(self.test_results)} test result configurations")
        return self.test_results
    
    def deduplicate_results(self) -> None:
        """
        Deduplicate test results by keeping only the most recent result
        for each unique (strategy, dataset, model) combination.
        
        This is useful when multiple test runs exist for the same configuration.
        """
        if not self.test_results:
            return
        
        # Group results by (strategy, dataset, model)
        grouped = {}
        for result in self.test_results:
            key = (result['strategy'], result['dataset'], result['model'])
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)
        
        # Keep only the most recent result for each group
        deduplicated = []
        duplicates_removed = 0
        
        def get_timestamp(r):
            """Get timestamp as float, handling string timestamps from directory names."""
            ts = r.get('timestamp', 0)
            if isinstance(ts, str):
                # Handle directory name format like "20251210_092854"
                try:
                    from datetime import datetime
                    return datetime.strptime(ts, "%Y%m%d_%H%M%S").timestamp()
                except:
                    return 0
            return ts if ts else 0
        
        for key, results in grouped.items():
            if len(results) > 1:
                # Sort by timestamp (most recent first), fall back to result_type preference
                sorted_results = sorted(
                    results,
                    key=lambda r: (
                        get_timestamp(r),
                        1 if r.get('result_type') == 'detailed' else 0  # Prefer detailed results
                    ),
                    reverse=True
                )
                deduplicated.append(sorted_results[0])
                duplicates_removed += len(results) - 1
            else:
                deduplicated.append(results[0])
        
        self.test_results = deduplicated
        print(f"Deduplicated: Kept {len(deduplicated)} unique configurations, "
              f"removed {duplicates_removed} duplicate test runs")
    
    def _analyze_test_results(
        self, 
        model_dir: Path, 
        strategy: str, 
        dataset: str, 
        model: str
    ) -> List[Dict]:
        """
        Analyze test results in a model directory.
        
        Returns:
            List of test result dictionaries
        """
        results = []
        
        # Look for test_results and test_results_detailed directories
        test_dirs = list(model_dir.glob("test_results*"))
        
        for test_dir in test_dirs:
            if not test_dir.is_dir():
                continue
            
            test_type = test_dir.name  # test_results or test_results_detailed
            
            # Check for nested test directory
            test_subdir = test_dir / "test"
            if test_subdir.exists():
                # Parse metrics.json
                metrics_file = test_subdir / "metrics.json"
                if metrics_file.exists():
                    result = self._parse_metrics_json(
                        metrics_file, strategy, dataset, model, test_type
                    )
                    if result:
                        results.append(result)
            
            # Check for detailed test results (timestamped directories)
            for timestamp_dir in sorted(test_dir.iterdir()):
                if not timestamp_dir.is_dir():
                    continue
                
                # Parse detailed metrics
                result = self._parse_detailed_metrics(
                    timestamp_dir, strategy, dataset, model, test_type
                )
                if result:
                    results.append(result)
        
        return results
    
    def _parse_metrics_json(
        self,
        metrics_file: Path,
        strategy: str,
        dataset: str,
        model: str,
        test_type: str
    ) -> Optional[Dict]:
        """Parse a metrics.json file."""
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Extract metric values
            return {
                'strategy': strategy,
                'dataset': dataset,
                'model': model,
                'test_type': test_type,
                'result_type': 'basic',
                'metrics_file': str(metrics_file),
                'mIoU': metrics.get('test/mIoU'),
                'mAcc': metrics.get('test/mAcc'),
                'aAcc': metrics.get('test/aAcc'),
                'fwIoU': metrics.get('test/fwIoU'),
                'timestamp': metrics_file.stat().st_mtime,
                'has_per_domain': False,
                'has_per_class': False
            }
        except Exception as e:
            print(f"Warning: Could not parse {metrics_file}: {e}")
            return None
    
    def _parse_detailed_metrics(
        self,
        result_dir: Path,
        strategy: str,
        dataset: str,
        model: str,
        test_type: str
    ) -> Optional[Dict]:
        """Parse detailed test results from a timestamp directory.
        
        Supports both the new unified format (results.json) and legacy format.
        """
        # Try new unified format first (results.json)
        unified_file = result_dir / "results.json"
        if unified_file.exists():
            return self._parse_unified_results(
                unified_file, strategy, dataset, model, test_type, result_dir
            )
        
        # Fall back to legacy format
        summary_file = result_dir / "metrics_summary.json"
        domain_file = result_dir / "metrics_per_domain.json"
        full_file = result_dir / "metrics_full.json"
        
        if not summary_file.exists():
            return None
        
        try:
            # Parse summary metrics
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            # Extract basic info
            result = {
                'strategy': strategy,
                'dataset': dataset,
                'model': model,
                'test_type': test_type,
                'result_type': 'detailed',
                'result_dir': str(result_dir),
                'timestamp': result_dir.name,
                'has_per_domain': domain_file.exists(),
                'has_per_class': full_file.exists()
            }
            
            # Extract metrics if present
            if 'metrics' in summary and summary['metrics']:
                metrics = summary['metrics']
                result.update({
                    'mIoU': metrics.get('test/mIoU'),
                    'mAcc': metrics.get('test/mAcc'),
                    'aAcc': metrics.get('test/aAcc'),
                    'fwIoU': metrics.get('test/fwIoU')
                })
            elif 'overall' in summary and summary['overall']:
                # Alternative format with 'overall' key
                overall = summary['overall']
                result.update({
                    'mIoU': overall.get('mIoU'),
                    'mAcc': overall.get('mAcc'),
                    'aAcc': overall.get('aAcc'),
                    'fwIoU': overall.get('fwIoU')
                })
            else:
                result.update({
                    'mIoU': None,
                    'mAcc': None,
                    'aAcc': None,
                    'fwIoU': None
                })
            
            # Parse per-domain metrics if available
            if domain_file.exists():
                result['per_domain_metrics'] = self._parse_per_domain(domain_file)
            
            return result
            
        except Exception as e:
            print(f"Warning: Could not parse detailed metrics in {result_dir}: {e}")
            return None
    
    def _parse_unified_results(
        self,
        unified_file: Path,
        strategy: str,
        dataset: str,
        model: str,
        test_type: str,
        result_dir: Path
    ) -> Optional[Dict]:
        """Parse the new unified results.json format."""
        try:
            with open(unified_file, 'r') as f:
                data = json.load(f)
            
            # Extract overall metrics
            overall = data.get('overall', {})
            
            result = {
                'strategy': strategy,
                'dataset': dataset,
                'model': model,
                'test_type': test_type,
                'result_type': 'detailed',
                'result_dir': str(result_dir),
                'timestamp': result_dir.name,
                'mIoU': overall.get('mIoU'),
                'mAcc': overall.get('mAcc'),
                'aAcc': overall.get('aAcc'),
                'fwIoU': overall.get('fwIoU'),
                'num_images': overall.get('num_images', 0),
                'has_per_domain': bool(data.get('per_domain')),
                'has_per_class': bool(data.get('per_class'))
            }
            
            # Parse per-domain metrics from unified format
            if data.get('per_domain'):
                domain_metrics = {}
                for domain, domain_data in data['per_domain'].items():
                    # Handle nested structure with 'summary' key
                    metrics = domain_data.get('summary', domain_data) if isinstance(domain_data, dict) else {}
                    domain_metrics[domain] = {
                        'mIoU': metrics.get('mIoU'),
                        'mAcc': metrics.get('mAcc'),
                        'aAcc': metrics.get('aAcc'),
                        'fwIoU': metrics.get('fwIoU')
                    }
                result['per_domain_metrics'] = domain_metrics
            
            # Store per-class data for top class analysis
            if data.get('per_class'):
                result['per_class_metrics'] = data['per_class']
            
            return result
            
        except Exception as e:
            print(f"Warning: Could not parse unified results in {unified_file}: {e}")
            return None
    
    def _parse_per_domain(self, domain_file: Path) -> Dict:
        """Parse per-domain metrics file."""
        try:
            with open(domain_file, 'r') as f:
                data = json.load(f)
            
            domains = {}
            if 'per_domain' in data:
                for domain, metrics in data['per_domain'].items():
                    if 'error' not in metrics:
                        # Handle both formats: with and without 'test/' prefix
                        domains[domain] = {
                            'mIoU': metrics.get('test/mIoU') or metrics.get('mIoU'),
                            'mAcc': metrics.get('test/mAcc') or metrics.get('mAcc'),
                            'aAcc': metrics.get('test/aAcc') or metrics.get('aAcc'),
                            'fwIoU': metrics.get('test/fwIoU') or metrics.get('fwIoU')
                        }
            return domains
        except Exception as e:
            print(f"Warning: Could not parse per-domain metrics: {e}")
            return {}
    
    def format_table(self, show_domains: bool = False) -> str:
        """
        Format the collected data as a nicely formatted ASCII table.
        
        Args:
            show_domains: Include per-domain metrics in table
        
        Returns:
            Formatted table string
        """
        if not self.test_results:
            return "No test results found"
        
        # Filter to valid results with metrics
        valid_results = [
            r for r in self.test_results 
            if r.get('mIoU') is not None
        ]
        
        if not valid_results:
            return "No valid test results with metrics found"
        
        # Build table rows
        rows = []
        
        # Header
        if show_domains:
            header = [
                "Strategy", "Dataset", "Model", "Type", 
                "mIoU", "mAcc", "aAcc", "fwIoU", "Domains"
            ]
        else:
            header = [
                "Strategy", "Dataset", "Model", "Type",
                "mIoU", "mAcc", "aAcc", "fwIoU"
            ]
        
        # Data rows
        for item in sorted(valid_results, key=lambda x: (x['strategy'], x['dataset'], x['model'])):
            row = [
                item['strategy'],
                item['dataset'],
                item['model'],
                item['result_type'],
                f"{item['mIoU']:.2f}" if item['mIoU'] else "—",
                f"{item['mAcc']:.2f}" if item['mAcc'] else "—",
                f"{item['aAcc']:.2f}" if item['aAcc'] else "—",
                f"{item['fwIoU']:.2f}" if item['fwIoU'] else "—"
            ]
            
            if show_domains:
                domain_count = len(item.get('per_domain_metrics', {}))
                domain_str = f"{domain_count} domains" if domain_count > 0 else "—"
                row.append(domain_str)
            
            rows.append(row)
        
        # Calculate column widths
        col_widths = [len(h) for h in header]
        
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Build formatted table
        lines = []
        
        # Top border
        lines.append("┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐")
        
        # Header row
        header_row = "│"
        for i, cell in enumerate(header):
            header_row += f" {cell:<{col_widths[i]}} │"
        lines.append(header_row)
        
        # Header separator
        lines.append("├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤")
        
        # Data rows
        for row in rows:
            data_row = "│"
            for i, cell in enumerate(row):
                data_row += f" {str(cell):<{col_widths[i]}} │"
            lines.append(data_row)
        
        # Bottom border
        lines.append("└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘")
        
        return "\n".join(lines)
    
    def format_summary(self) -> str:
        """
        Generate a summary statistics section.
        
        Returns:
            Formatted summary string
        """
        if not self.test_results:
            return "No test results available"
        
        # Calculate statistics
        total_results = len(self.test_results)
        valid_results = [r for r in self.test_results if r.get('mIoU') is not None]
        
        # Count by type
        basic_results = [r for r in self.test_results if r['result_type'] == 'basic']
        detailed_results = [r for r in self.test_results if r['result_type'] == 'detailed']
        
        # Count with per-domain metrics
        with_domains = sum(1 for r in self.test_results if r.get('has_per_domain'))
        
        # Count by strategy
        strategies = defaultdict(int)
        for item in valid_results:
            strategies[item['strategy']] += 1
        
        # Count by dataset
        datasets = defaultdict(int)
        for item in valid_results:
            datasets[item['dataset']] += 1
        
        # Calculate average metrics
        if valid_results:
            avg_mIoU = sum(r['mIoU'] for r in valid_results) / len(valid_results)
            avg_mAcc = sum(r['mAcc'] for r in valid_results if r.get('mAcc')) / len([r for r in valid_results if r.get('mAcc')])
            avg_aAcc = sum(r['aAcc'] for r in valid_results if r.get('aAcc')) / len([r for r in valid_results if r.get('aAcc')])
        else:
            avg_mIoU = avg_mAcc = avg_aAcc = 0
        
        summary = []
        summary.append("=" * 70)
        summary.append("TEST RESULTS SUMMARY")
        summary.append("=" * 70)
        summary.append(f"Total Test Results: {total_results}")
        summary.append(f"Valid Results with Metrics: {len(valid_results)}")
        summary.append(f"Basic Results: {len(basic_results)}")
        summary.append(f"Detailed Results: {len(detailed_results)}")
        summary.append(f"Results with Per-Domain Metrics: {with_domains}")
        summary.append("")
        summary.append("Average Performance:")
        summary.append(f"  mIoU: {avg_mIoU:.2f}%")
        summary.append(f"  mAcc: {avg_mAcc:.2f}%")
        summary.append(f"  aAcc: {avg_aAcc:.2f}%")
        summary.append("")
        summary.append("Results by Strategy:")
        for strategy, count in sorted(strategies.items()):
            summary.append(f"  - {strategy}: {count} results")
        
        # Categorize strategies by type
        summary.append("")
        summary.append("Results by Strategy Type:")
        strategy_types = defaultdict(int)
        combined_strategies = []
        for strategy in strategies.keys():
            stype = get_strategy_type(strategy)
            strategy_types[stype] += strategies[strategy]
            if stype == 'combined':
                combined_strategies.append(strategy)
        
        for stype, count in sorted(strategy_types.items()):
            emoji = {'combined': '🔗', 'gen': '🎨', 'std': '📊', 'baseline': '📌', 'other': '❓'}.get(stype, '•')
            summary.append(f"  {emoji} {stype}: {count} results")
        
        # List combined strategies if any
        if combined_strategies:
            summary.append("")
            summary.append("Combined Strategies (gen+std):")
            for strategy in sorted(combined_strategies):
                parsed = parse_strategy(strategy)
                summary.append(f"  - {strategy}")
                summary.append(f"      Gen: {parsed['gen_component']}, Std: {parsed['std_component']}")
        
        summary.append("")
        summary.append("Results by Dataset:")
        for dataset, count in sorted(datasets.items()):
            summary.append(f"  - {dataset}: {count} results")
        summary.append("=" * 70)
        
        return "\n".join(summary)
    
    def format_comprehensive_summary(self, top_n: int = 5) -> str:
        """
        Generate a comprehensive summary with top performing configurations.
        
        Args:
            top_n: Number of top configurations to show
        
        Returns:
            Formatted comprehensive summary string
        """
        if not self.test_results:
            return "No test results available for comprehensive summary"
        
        # Filter to valid results with metrics
        valid_results = [
            r for r in self.test_results 
            if r.get('mIoU') is not None
        ]
        
        if not valid_results:
            return "No valid test results with metrics for comprehensive summary"
        
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("COMPREHENSIVE PERFORMANCE SUMMARY")
        lines.append("=" * 80)
        
        # === TOP OVERALL CONFIGURATIONS ===
        sorted_by_miou = sorted(valid_results, key=lambda x: x['mIoU'] or 0, reverse=True)
        
        lines.append(f"\n🏆 TOP {top_n} CONFIGURATIONS BY mIoU")
        lines.append("-" * 80)
        lines.append(f"{'Rank':<5} {'Strategy':<20} {'Dataset':<20} {'Model':<25} {'mIoU':>8}")
        lines.append("-" * 80)
        
        for i, result in enumerate(sorted_by_miou[:top_n], 1):
            lines.append(f"{i:<5} {result['strategy']:<20} {result['dataset']:<20} "
                        f"{result['model']:<25} {result['mIoU']:>7.2f}%")
        
        # === BEST PER DATASET ===
        lines.append(f"\n📊 BEST CONFIGURATION PER DATASET")
        lines.append("-" * 80)
        lines.append(f"{'Dataset':<20} {'Strategy':<20} {'Model':<25} {'mIoU':>8}")
        lines.append("-" * 80)
        
        datasets = set(r['dataset'] for r in valid_results)
        for dataset in sorted(datasets):
            dataset_results = [r for r in valid_results if r['dataset'] == dataset]
            if dataset_results:
                best = max(dataset_results, key=lambda x: x['mIoU'] or 0)
                lines.append(f"{dataset:<20} {best['strategy']:<20} "
                            f"{best['model']:<25} {best['mIoU']:>7.2f}%")
        
        # === BEST PER STRATEGY ===
        lines.append(f"\n🎯 BEST CONFIGURATION PER STRATEGY")
        lines.append("-" * 80)
        lines.append(f"{'Strategy':<20} {'Dataset':<20} {'Model':<25} {'mIoU':>8}")
        lines.append("-" * 80)
        
        strategies = set(r['strategy'] for r in valid_results)
        strategy_bests = []
        for strategy in sorted(strategies):
            strategy_results = [r for r in valid_results if r['strategy'] == strategy]
            if strategy_results:
                best = max(strategy_results, key=lambda x: x['mIoU'] or 0)
                strategy_bests.append((strategy, best))
                lines.append(f"{strategy:<20} {best['dataset']:<20} "
                            f"{best['model']:<25} {best['mIoU']:>7.2f}%")
        
        # === STRATEGY COMPARISON ===
        lines.append(f"\n📈 STRATEGY PERFORMANCE COMPARISON (Average mIoU)")
        lines.append("-" * 80)
        
        strategy_stats = {}
        for strategy in strategies:
            strategy_results = [r for r in valid_results if r['strategy'] == strategy]
            if strategy_results:
                avg_miou = sum(r['mIoU'] for r in strategy_results) / len(strategy_results)
                max_miou = max(r['mIoU'] for r in strategy_results)
                min_miou = min(r['mIoU'] for r in strategy_results)
                strategy_stats[strategy] = {
                    'avg': avg_miou,
                    'max': max_miou,
                    'min': min_miou,
                    'count': len(strategy_results)
                }
        
        # Sort by average mIoU
        sorted_strategies = sorted(strategy_stats.items(), key=lambda x: x[1]['avg'], reverse=True)
        
        lines.append(f"{'Strategy':<25} {'Avg mIoU':>10} {'Max mIoU':>10} {'Min mIoU':>10} {'Count':>8}")
        lines.append("-" * 80)
        
        for strategy, stats in sorted_strategies:
            lines.append(f"{strategy:<25} {stats['avg']:>9.2f}% {stats['max']:>9.2f}% "
                        f"{stats['min']:>9.2f}% {stats['count']:>8}")
        
        # === BASELINE CALCULATION ===
        # Use baseline strategy as reference
        # Note: Since directory restructuring, dataset names no longer have _cd suffix
        # All models in WEIGHTS are Stage 1 (clear_day training)
        baseline_results = [
            r for r in valid_results 
            if r['strategy'] == 'baseline'
        ]
        
        if baseline_results:
            baseline_avg = sum(r['mIoU'] for r in baseline_results) / len(baseline_results)
            baseline_label = "Baseline"
        else:
            baseline_avg = 0
            baseline_label = "Baseline"
        
        # === PERFORMANCE GAINS VS BASELINE (clear_day) ===
        if baseline_avg > 0:
            lines.append(f"\n📊 PERFORMANCE GAINS VS {baseline_label.upper()}")
            lines.append("-" * 80)
            lines.append(f"Reference: {baseline_label} with avg mIoU = {baseline_avg:.2f}%")
            lines.append(f"(Using models trained on clear_day domain as baseline reference)")
            lines.append("")
            lines.append(f"{'Strategy':<25} {'Baseline':>10} {'Strategy':>10} {'Gain':>10} {'%Improvement':>12}")
            lines.append("-" * 80)
            
            gains = []
            for strategy, stats in sorted_strategies:
                if strategy != 'baseline':
                    gain = stats['avg'] - baseline_avg
                    pct_gain = (gain / baseline_avg) * 100 if baseline_avg > 0 else 0
                    gains.append((strategy, baseline_avg, stats['avg'], gain, pct_gain))
            
            # Sort by gain
            gains.sort(key=lambda x: x[3], reverse=True)
            
            for strategy, base, strat_avg, gain, pct in gains:
                gain_str = f"+{gain:.2f}" if gain >= 0 else f"{gain:.2f}"
                pct_str = f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%"
                lines.append(f"{strategy:<25} {base:>9.2f}% {strat_avg:>9.2f}% "
                            f"{gain_str:>10} {pct_str:>12}")
        
        # === MODEL COMPARISON ===
        lines.append(f"\n🔧 MODEL PERFORMANCE COMPARISON (Average mIoU)")
        lines.append("-" * 80)
        
        models = set(r['model'] for r in valid_results)
        # Models are now stored without suffixes; no need to clean names
        model_base_names = models.copy()
        
        model_stats = {}
        for model in sorted(model_base_names):
            # Get results for this model
            model_results = [r for r in valid_results if r['model'] == model]
            if model_results:
                avg_miou = sum(r['mIoU'] for r in model_results) / len(model_results)
                model_stats[model] = {
                    'avg': avg_miou,
                    'count': len(model_results)
                }
        
        sorted_models = sorted(model_stats.items(), key=lambda x: x[1]['avg'], reverse=True)
        
        lines.append(f"{'Model':<35} {'Avg mIoU':>10} {'Count':>8}")
        lines.append("-" * 80)
        for model, stats in sorted_models:
            lines.append(f"{model:<35} {stats['avg']:>9.2f}% {stats['count']:>8}")
        
        # === KEY INSIGHTS ===
        lines.append(f"\n💡 KEY INSIGHTS")
        lines.append("-" * 80)
        
        if sorted_by_miou:
            best = sorted_by_miou[0]
            lines.append(f"• Best Overall: {best['strategy']}/{best['dataset']}/{best['model']} "
                        f"with {best['mIoU']:.2f}% mIoU")
        
        if sorted_strategies:
            best_strategy = sorted_strategies[0][0]
            lines.append(f"• Best Strategy (by avg): {best_strategy} "
                        f"with {sorted_strategies[0][1]['avg']:.2f}% average mIoU")
        
        if sorted_models:
            best_model = sorted_models[0][0]
            lines.append(f"• Best Model (by avg): {best_model} "
                        f"with {sorted_models[0][1]['avg']:.2f}% average mIoU")
        
        if baseline_avg > 0 and gains:
            best_gain = gains[0]
            lines.append(f"• Largest Improvement over {baseline_label}: {best_gain[0]} "
                        f"with +{best_gain[3]:.2f}% mIoU ({best_gain[4]:.1f}% relative improvement)")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)
    
    def format_per_domain_table(self) -> str:
        """
        Generate a per-domain performance table.
        
        Returns:
            Formatted domain performance table
        """
        # Filter results with per-domain metrics
        domain_results = [
            r for r in self.test_results 
            if r.get('per_domain_metrics') and len(r['per_domain_metrics']) > 0
        ]
        
        if not domain_results:
            return "No per-domain test results available"
        
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("PER-DOMAIN PERFORMANCE BREAKDOWN")
        lines.append("=" * 70)
        
        for result in domain_results:
            lines.append(f"\n{result['strategy']}/{result['dataset']}/{result['model']}")
            lines.append("-" * 60)
            
            # Build domain table
            domains = result['per_domain_metrics']
            if not domains:
                lines.append("  No valid domain metrics")
                continue
            
            # Header
            lines.append(f"{'Domain':<15} {'mIoU':>8} {'mAcc':>8} {'aAcc':>8} {'fwIoU':>8}")
            lines.append("-" * 60)
            
            # Domain rows
            for domain, metrics in sorted(domains.items()):
                miou = f"{metrics['mIoU']:.2f}" if metrics.get('mIoU') else "—"
                macc = f"{metrics['mAcc']:.2f}" if metrics.get('mAcc') else "—"
                aacc = f"{metrics['aAcc']:.2f}" if metrics.get('aAcc') else "—"
                fwiou = f"{metrics['fwIoU']:.2f}" if metrics.get('fwIoU') else "—"
                
                lines.append(f"{domain:<15} {miou:>8} {macc:>8} {aacc:>8} {fwiou:>8}")
        
        return "\n".join(lines)
    
    def format_dataset_insights(self) -> str:
        """
        Generate high-level insights per dataset.
        
        Returns:
            Formatted per-dataset insights string
        """
        if not self.test_results:
            return "No test results available for dataset insights"
        
        valid_results = [r for r in self.test_results if r.get('mIoU') is not None]
        if not valid_results:
            return "No valid test results with metrics for dataset insights"
        
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("📊 PER-DATASET INSIGHTS")
        lines.append("=" * 80)
        
        datasets = sorted(set(r['dataset'] for r in valid_results))
        
        for dataset in datasets:
            dataset_results = [r for r in valid_results if r['dataset'] == dataset]
            if not dataset_results:
                continue
            
            lines.append(f"\n{'─' * 80}")
            lines.append(f"📁 DATASET: {dataset.upper()}")
            lines.append(f"{'─' * 80}")
            
            # Calculate statistics
            mious = [r['mIoU'] for r in dataset_results]
            avg_miou = sum(mious) / len(mious)
            max_miou = max(mious)
            min_miou = min(mious)
            std_miou = (sum((x - avg_miou) ** 2 for x in mious) / len(mious)) ** 0.5
            
            lines.append(f"\n📈 Overall Performance:")
            lines.append(f"   • Configurations tested: {len(dataset_results)}")
            lines.append(f"   • Average mIoU: {avg_miou:.2f}%")
            lines.append(f"   • Best mIoU: {max_miou:.2f}%")
            lines.append(f"   • Worst mIoU: {min_miou:.2f}%")
            lines.append(f"   • Std. Deviation: {std_miou:.2f}%")
            lines.append(f"   • Performance Spread: {max_miou - min_miou:.2f}%")
            
            # Best configuration
            best = max(dataset_results, key=lambda x: x['mIoU'])
            worst = min(dataset_results, key=lambda x: x['mIoU'])
            
            lines.append(f"\n🏆 Best Configuration:")
            lines.append(f"   • Strategy: {best['strategy']}")
            lines.append(f"   • Model: {best['model']}")
            lines.append(f"   • mIoU: {best['mIoU']:.2f}%")
            
            lines.append(f"\n⚠️  Worst Configuration:")
            lines.append(f"   • Strategy: {worst['strategy']}")
            lines.append(f"   • Model: {worst['model']}")
            lines.append(f"   • mIoU: {worst['mIoU']:.2f}%")
            
            # Strategy breakdown for this dataset
            lines.append(f"\n📋 Strategy Performance on {dataset}:")
            strategy_stats = {}
            for r in dataset_results:
                strategy = r['strategy']
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = []
                strategy_stats[strategy].append(r['mIoU'])
            
            sorted_strategies = sorted(
                strategy_stats.items(), 
                key=lambda x: sum(x[1])/len(x[1]), 
                reverse=True
            )
            
            for strategy, mious in sorted_strategies:
                avg = sum(mious) / len(mious)
                lines.append(f"   • {strategy:<25} avg: {avg:.2f}% ({len(mious)} configs)")
            
            # Model breakdown for this dataset
            lines.append(f"\n🔧 Model Performance on {dataset}:")
            model_stats = {}
            for r in dataset_results:
                # Group by model name (no suffix to remove with new organization)
                model_name = r['model']
                if model_name not in model_stats:
                    model_stats[model_name] = []
                model_stats[model_name].append(r['mIoU'])
            
            sorted_models = sorted(
                model_stats.items(),
                key=lambda x: sum(x[1])/len(x[1]),
                reverse=True
            )
            
            for model, mious in sorted_models:
                avg = sum(mious) / len(mious)
                lines.append(f"   • {model:<30} avg: {avg:.2f}% ({len(mious)} configs)")
            
            # Insight: best strategy-model combination
            lines.append(f"\n💡 Key Insight for {dataset}:")
            best_strategy = sorted_strategies[0][0] if sorted_strategies else "N/A"
            best_model = sorted_models[0][0] if sorted_models else "N/A"
            lines.append(f"   Best strategy: {best_strategy}")
            lines.append(f"   Best model architecture: {best_model}")
            
            # Check for baseline comparison
            baseline_results = [r for r in dataset_results if r['strategy'] == 'baseline']
            if baseline_results:
                baseline_avg = sum(r['mIoU'] for r in baseline_results) / len(baseline_results)
                improvement = avg_miou - baseline_avg
                if improvement > 0:
                    lines.append(f"   Average improvement over baseline: +{improvement:.2f}%")
                else:
                    lines.append(f"   Performance vs baseline: {improvement:.2f}%")
        
        lines.append("\n" + "=" * 80)
        return "\n".join(lines)
    
    def format_domain_insights(self) -> str:
        """
        Generate high-level insights per weather domain.
        
        Returns:
            Formatted per-domain insights string
        """
        # Filter results with per-domain metrics
        domain_results = [
            r for r in self.test_results 
            if r.get('per_domain_metrics') and len(r['per_domain_metrics']) > 0
        ]
        
        if not domain_results:
            return "No per-domain test results available for domain insights"
        
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("🌤️  PER-DOMAIN INSIGHTS (WEATHER CONDITIONS)")
        lines.append("=" * 80)
        
        # Aggregate domain metrics across all configurations
        domain_aggregates = defaultdict(list)
        
        for result in domain_results:
            for domain, metrics in result['per_domain_metrics'].items():
                if metrics.get('mIoU') is not None:
                    domain_aggregates[domain].append({
                        'mIoU': metrics['mIoU'],
                        'strategy': result['strategy'],
                        'dataset': result['dataset'],
                        'model': result['model'],
                        'mAcc': metrics.get('mAcc'),
                        'aAcc': metrics.get('aAcc'),
                        'fwIoU': metrics.get('fwIoU')
                    })
        
        if not domain_aggregates:
            return "No valid per-domain metrics found"
        
        # Calculate overall domain statistics
        lines.append(f"\n📊 DOMAIN DIFFICULTY RANKING (by average mIoU)")
        lines.append("-" * 80)
        
        domain_stats = {}
        for domain, results in domain_aggregates.items():
            mious = [r['mIoU'] for r in results]
            domain_stats[domain] = {
                'avg': sum(mious) / len(mious),
                'max': max(mious),
                'min': min(mious),
                'count': len(results),
                'std': (sum((x - sum(mious)/len(mious)) ** 2 for x in mious) / len(mious)) ** 0.5
            }
        
        # Sort by average mIoU (descending = easiest first)
        sorted_domains = sorted(domain_stats.items(), key=lambda x: x[1]['avg'], reverse=True)
        
        lines.append(f"{'Rank':<5} {'Domain':<20} {'Avg mIoU':>10} {'Best':>10} {'Worst':>10} {'Std':>8} {'Configs':>8}")
        lines.append("-" * 80)
        
        for i, (domain, stats) in enumerate(sorted_domains, 1):
            difficulty = "🟢 Easy" if stats['avg'] > 70 else "🟡 Medium" if stats['avg'] > 50 else "🔴 Hard"
            lines.append(f"{i:<5} {domain:<20} {stats['avg']:>9.2f}% {stats['max']:>9.2f}% "
                        f"{stats['min']:>9.2f}% {stats['std']:>7.2f} {stats['count']:>8}")
        
        lines.append("")
        
        # Identify easiest and hardest domains
        easiest = sorted_domains[0]
        hardest = sorted_domains[-1]
        
        lines.append(f"✅ Easiest Domain: {easiest[0]} (avg mIoU: {easiest[1]['avg']:.2f}%)")
        lines.append(f"❌ Hardest Domain: {hardest[0]} (avg mIoU: {hardest[1]['avg']:.2f}%)")
        
        performance_gap = easiest[1]['avg'] - hardest[1]['avg']
        lines.append(f"📏 Domain Performance Gap: {performance_gap:.2f}% mIoU")
        
        # Best configuration per domain
        lines.append(f"\n🏆 BEST CONFIGURATION PER DOMAIN")
        lines.append("-" * 80)
        
        for domain in sorted(domain_aggregates.keys()):
            results = domain_aggregates[domain]
            best = max(results, key=lambda x: x['mIoU'])
            lines.append(f"\n  {domain.upper()}:")
            lines.append(f"    • Strategy: {best['strategy']}")
            lines.append(f"    • Model: {best['model']}")
            lines.append(f"    • Dataset: {best['dataset']}")
            lines.append(f"    • mIoU: {best['mIoU']:.2f}%")
        
        # Strategy effectiveness per domain
        lines.append(f"\n📈 STRATEGY EFFECTIVENESS BY DOMAIN")
        lines.append("-" * 80)
        lines.append("(Average mIoU per strategy for each domain)")
        lines.append("")
        
        # Get unique strategies
        strategies = sorted(set(r['strategy'] for d in domain_aggregates.values() for r in d))
        
        # Build header
        header = f"{'Strategy':<25}"
        for domain, _ in sorted_domains[:6]:  # Top 6 domains
            header += f" {domain[:8]:>10}"
        lines.append(header)
        lines.append("-" * 80)
        
        # Build rows
        strategy_domain_avg = {}
        for strategy in strategies:
            row = f"{strategy:<25}"
            for domain, _ in sorted_domains[:6]:
                domain_strategy_results = [
                    r for r in domain_aggregates[domain] 
                    if r['strategy'] == strategy
                ]
                if domain_strategy_results:
                    avg = sum(r['mIoU'] for r in domain_strategy_results) / len(domain_strategy_results)
                    row += f" {avg:>9.2f}%"
                    if strategy not in strategy_domain_avg:
                        strategy_domain_avg[strategy] = {}
                    strategy_domain_avg[strategy][domain] = avg
                else:
                    row += f" {'—':>10}"
            lines.append(row)
        
        # Key insights per domain
        lines.append(f"\n💡 KEY DOMAIN INSIGHTS")
        lines.append("-" * 80)
        
        for domain, stats in sorted_domains:
            # Find best strategy for this domain
            domain_by_strategy = {}
            for r in domain_aggregates[domain]:
                if r['strategy'] not in domain_by_strategy:
                    domain_by_strategy[r['strategy']] = []
                domain_by_strategy[r['strategy']].append(r['mIoU'])
            
            if domain_by_strategy:
                best_strategy = max(
                    domain_by_strategy.items(),
                    key=lambda x: sum(x[1])/len(x[1])
                )
                lines.append(f"\n  {domain}:")
                lines.append(f"    • Best strategy: {best_strategy[0]} "
                           f"(avg {sum(best_strategy[1])/len(best_strategy[1]):.2f}%)")
                
                # Calculate improvement potential (gap to best)
                gap_to_best = stats['max'] - stats['avg']
                lines.append(f"    • Improvement potential: +{gap_to_best:.2f}% (gap to best config)")
                
                # Variability insight
                if stats['std'] > 5:
                    lines.append(f"    • High variability (std={stats['std']:.1f}%): Strategy choice matters significantly")
                elif stats['std'] > 2:
                    lines.append(f"    • Moderate variability (std={stats['std']:.1f}%): Some strategy impact")
                else:
                    lines.append(f"    • Low variability (std={stats['std']:.1f}%): Consistent across strategies")
        
        lines.append("\n" + "=" * 80)
        return "\n".join(lines)
    
    def format_combined_strategy_insights(self) -> str:
        """
        Generate insights specifically for combined gen+std strategies.
        
        This method analyzes how combining generative augmentation (gen_*)
        with standard augmentation (std_*) affects performance compared to
        using either strategy alone.
        
        Returns:
            Formatted combined strategy insights string
        """
        if not self.test_results:
            return "No test results available for combined strategy insights"
        
        valid_results = [r for r in self.test_results if r.get('mIoU') is not None]
        if not valid_results:
            return "No valid test results with metrics for combined strategy insights"
        
        # Separate results by strategy type
        combined_results = []
        gen_only_results = []
        std_only_results = []
        baseline_results = []
        
        for r in valid_results:
            stype = get_strategy_type(r['strategy'])
            if stype == 'combined':
                combined_results.append(r)
            elif stype == 'gen':
                gen_only_results.append(r)
            elif stype == 'std':
                std_only_results.append(r)
            elif stype == 'baseline':
                baseline_results.append(r)
        
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("🔗 COMBINED STRATEGY INSIGHTS (Gen + Std Augmentation)")
        lines.append("=" * 80)
        
        # Summary counts
        lines.append(f"\n📊 STRATEGY COUNTS")
        lines.append("-" * 80)
        lines.append(f"  Combined (gen+std): {len(combined_results)} configurations")
        lines.append(f"  Gen-only:           {len(gen_only_results)} configurations")
        lines.append(f"  Std-only:           {len(std_only_results)} configurations")
        lines.append(f"  Baseline:           {len(baseline_results)} configurations")
        
        if not combined_results:
            lines.append("\n⚠️  No combined strategy results found.")
            lines.append("   To test combined strategies, use:")
            lines.append("   ./scripts/train_unified.sh single --strategy gen_cycleGAN --std-strategy std_cutmix ...")
            lines.append("\n" + "=" * 80)
            return "\n".join(lines)
        
        # === COMBINED STRATEGY PERFORMANCE OVERVIEW ===
        lines.append(f"\n🏆 COMBINED STRATEGY PERFORMANCE")
        lines.append("-" * 80)
        
        # Group combined results by strategy
        combined_by_strategy = defaultdict(list)
        for r in combined_results:
            combined_by_strategy[r['strategy']].append(r)
        
        lines.append(f"{'Combined Strategy':<40} {'Avg mIoU':>10} {'Max':>8} {'Min':>8} {'Count':>6}")
        lines.append("-" * 80)
        
        sorted_combined = sorted(
            combined_by_strategy.items(),
            key=lambda x: sum(r['mIoU'] for r in x[1]) / len(x[1]),
            reverse=True
        )
        
        for strategy, results in sorted_combined:
            mious = [r['mIoU'] for r in results]
            avg = sum(mious) / len(mious)
            lines.append(f"{strategy:<40} {avg:>9.2f}% {max(mious):>7.2f}% {min(mious):>7.2f}% {len(results):>6}")
        
        # === COMPONENT ANALYSIS ===
        lines.append(f"\n📈 COMPONENT ANALYSIS")
        lines.append("-" * 80)
        lines.append("Analyzing how each gen component performs with different std components:")
        lines.append("")
        
        # Group by gen component
        gen_component_stats = defaultdict(lambda: defaultdict(list))
        for r in combined_results:
            parsed = parse_strategy(r['strategy'])
            gen_comp = parsed['gen_component']
            std_comp = parsed['std_component']
            gen_component_stats[gen_comp][std_comp].append(r['mIoU'])
        
        # Find all std components
        all_std_components = set()
        for gen_comp, std_dict in gen_component_stats.items():
            all_std_components.update(std_dict.keys())
        all_std_components = sorted(all_std_components)
        
        # Header for component matrix
        header = f"{'Gen Component':<20}"
        for std in all_std_components:
            # Abbreviate std component names
            short_std = std.replace('std_', '')[:8]
            header += f" {short_std:>10}"
        header += " {'Avg':>8}"
        lines.append(header)
        lines.append("-" * 80)
        
        # Rows for each gen component
        gen_avgs = []
        for gen_comp in sorted(gen_component_stats.keys()):
            row = f"{gen_comp:<20}"
            row_values = []
            for std in all_std_components:
                if std in gen_component_stats[gen_comp]:
                    mious = gen_component_stats[gen_comp][std]
                    avg = sum(mious) / len(mious)
                    row += f" {avg:>9.2f}%"
                    row_values.append(avg)
                else:
                    row += f" {'—':>10}"
            if row_values:
                gen_avg = sum(row_values) / len(row_values)
                row += f" {gen_avg:>7.2f}%"
                gen_avgs.append((gen_comp, gen_avg))
            else:
                row += f" {'—':>8}"
            lines.append(row)
        
        # === SYNERGY ANALYSIS ===
        lines.append(f"\n🔬 SYNERGY ANALYSIS")
        lines.append("-" * 80)
        lines.append("Comparing combined performance vs individual components:")
        lines.append("")
        
        # For each combined strategy, compare to its components
        synergy_data = []
        for strategy, results in combined_by_strategy.items():
            parsed = parse_strategy(strategy)
            gen_comp = parsed['gen_component']
            std_comp = parsed['std_component']
            
            combined_avg = sum(r['mIoU'] for r in results) / len(results)
            
            # Find gen-only performance
            gen_only_avg = None
            for r in gen_only_results:
                if r['strategy'] == gen_comp:
                    # Match by dataset and model for fair comparison
                    matching = [rr for rr in gen_only_results 
                               if rr['strategy'] == gen_comp]
                    if matching:
                        gen_only_avg = sum(rr['mIoU'] for rr in matching) / len(matching)
                    break
            
            # Find std-only performance
            std_only_avg = None
            for r in std_only_results:
                if r['strategy'] == std_comp:
                    matching = [rr for rr in std_only_results 
                               if rr['strategy'] == std_comp]
                    if matching:
                        std_only_avg = sum(rr['mIoU'] for rr in matching) / len(matching)
                    break
            
            # Find baseline performance
            baseline_avg = None
            if baseline_results:
                baseline_avg = sum(r['mIoU'] for r in baseline_results) / len(baseline_results)
            
            synergy_data.append({
                'strategy': strategy,
                'combined_avg': combined_avg,
                'gen_only_avg': gen_only_avg,
                'std_only_avg': std_only_avg,
                'baseline_avg': baseline_avg,
                'gen_comp': gen_comp,
                'std_comp': std_comp
            })
        
        # Display synergy table
        lines.append(f"{'Combined Strategy':<35} {'Combined':>9} {'Gen Only':>9} {'Std Only':>9} {'Baseline':>9} {'vs Gen':>8} {'vs Base':>8}")
        lines.append("-" * 100)
        
        for data in sorted(synergy_data, key=lambda x: x['combined_avg'], reverse=True):
            row = f"{data['strategy']:<35}"
            row += f" {data['combined_avg']:>8.2f}%"
            
            if data['gen_only_avg'] is not None:
                row += f" {data['gen_only_avg']:>8.2f}%"
                synergy_vs_gen = data['combined_avg'] - data['gen_only_avg']
            else:
                row += f" {'—':>9}"
                synergy_vs_gen = None
            
            if data['std_only_avg'] is not None:
                row += f" {data['std_only_avg']:>8.2f}%"
            else:
                row += f" {'—':>9}"
            
            if data['baseline_avg'] is not None:
                row += f" {data['baseline_avg']:>8.2f}%"
                synergy_vs_baseline = data['combined_avg'] - data['baseline_avg']
            else:
                row += f" {'—':>9}"
                synergy_vs_baseline = None
            
            # Synergy indicators
            if synergy_vs_gen is not None:
                sign = '+' if synergy_vs_gen >= 0 else ''
                row += f" {sign}{synergy_vs_gen:>6.2f}%"
            else:
                row += f" {'—':>8}"
            
            if synergy_vs_baseline is not None:
                sign = '+' if synergy_vs_baseline >= 0 else ''
                row += f" {sign}{synergy_vs_baseline:>6.2f}%"
            else:
                row += f" {'—':>8}"
            
            lines.append(row)
        
        # === BEST COMBINATIONS ===
        lines.append(f"\n💡 KEY INSIGHTS")
        lines.append("-" * 80)
        
        if sorted_combined:
            best = sorted_combined[0]
            best_avg = sum(r['mIoU'] for r in best[1]) / len(best[1])
            parsed = parse_strategy(best[0])
            lines.append(f"• Best Combined Strategy: {best[0]}")
            lines.append(f"  - Average mIoU: {best_avg:.2f}%")
            lines.append(f"  - Gen component: {parsed['gen_component']}")
            lines.append(f"  - Std component: {parsed['std_component']}")
        
        # Find best gen component overall
        if gen_avgs:
            best_gen = max(gen_avgs, key=lambda x: x[1])
            lines.append(f"\n• Best Gen Component in Combined Strategies: {best_gen[0]}")
            lines.append(f"  - Average across all std combinations: {best_gen[1]:.2f}%")
        
        # Find best std component overall
        std_avgs = defaultdict(list)
        for gen_comp, std_dict in gen_component_stats.items():
            for std_comp, mious in std_dict.items():
                std_avgs[std_comp].extend(mious)
        
        if std_avgs:
            best_std = max(std_avgs.items(), key=lambda x: sum(x[1])/len(x[1]))
            best_std_avg = sum(best_std[1]) / len(best_std[1])
            lines.append(f"\n• Best Std Component in Combined Strategies: {best_std[0]}")
            lines.append(f"  - Average across all gen combinations: {best_std_avg:.2f}%")
        
        # Recommendations
        lines.append(f"\n📋 RECOMMENDATIONS")
        lines.append("-" * 80)
        
        # Check for positive synergies
        positive_synergies = [d for d in synergy_data 
                            if d['gen_only_avg'] is not None 
                            and d['combined_avg'] > d['gen_only_avg']]
        
        if positive_synergies:
            lines.append(f"• {len(positive_synergies)}/{len(synergy_data)} combined strategies show positive synergy over gen-only")
            best_synergy = max(positive_synergies, key=lambda x: x['combined_avg'] - x['gen_only_avg'])
            gain = best_synergy['combined_avg'] - best_synergy['gen_only_avg']
            lines.append(f"• Highest synergy: {best_synergy['strategy']} (+{gain:.2f}% over gen-only)")
        else:
            lines.append("• No clear positive synergies detected between gen and std augmentations")
            lines.append("  Consider testing more combinations or different hyperparameters")
        
        lines.append("\n" + "=" * 80)
        return "\n".join(lines)
    
    def export_json(self, output_path: str):
        """Export data to JSON file with strategy type information."""
        # Convert to serializable format with enriched strategy info
        export_data = []
        for result in self.test_results:
            item = dict(result)
            # Add strategy parsing info
            parsed = parse_strategy(result['strategy'])
            item['strategy_type'] = parsed['type']
            item['gen_component'] = parsed['gen_component']
            item['std_component'] = parsed['std_component']
            # Remove non-serializable fields
            if 'per_domain_metrics' in item:
                item['per_domain_metrics'] = dict(item['per_domain_metrics'])
            export_data.append(item)
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        print(f"Exported JSON to {output_path}")
    
    def export_csv(self, output_path: str):
        """Export data to CSV file with strategy type information."""
        import csv
        import json
        
        if not self.test_results:
            print("No data to export")
            return
        
        # Define CSV columns (including strategy type info and per-domain metrics)
        columns = [
            'strategy', 'strategy_type', 'gen_component', 'std_component',
            'dataset', 'model', 'test_type', 'result_type',
            'mIoU', 'mAcc', 'aAcc', 'fwIoU', 
            'has_per_domain', 'has_per_class', 'per_domain_metrics', 'timestamp'
        ]
        
        # Enrich results with strategy parsing info
        enriched_results = []
        for r in self.test_results:
            item = dict(r)
            parsed = parse_strategy(r['strategy'])
            item['strategy_type'] = parsed['type']
            item['gen_component'] = parsed['gen_component'] or ''
            item['std_component'] = parsed['std_component'] or ''
            # Serialize per_domain_metrics to JSON string for CSV
            if 'per_domain_metrics' in item and item['per_domain_metrics']:
                item['per_domain_metrics'] = json.dumps(item['per_domain_metrics'])
            else:
                item['per_domain_metrics'] = ''
            enriched_results.append(item)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(enriched_results)
        
        print(f"Exported CSV to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze PROVE test results and generate summary"
    )
    parser.add_argument(
        '--root',
        default="${AWARE_DATA_ROOT}/WEIGHTS/",
        help="Root directory to scan (default: ${AWARE_DATA_ROOT}/WEIGHTS/)"
    )
    parser.add_argument(
        '--format',
        choices=['table', 'json', 'csv'],
        default='table',
        help="Output format (default: table)"
    )
    parser.add_argument(
        '--output',
        help="Output file path (for json/csv formats)"
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Verbose output during scanning"
    )
    parser.add_argument(
        '--show-domains',
        action='store_true',
        help="Show per-domain metrics in table"
    )
    parser.add_argument(
        '--domain-breakdown',
        action='store_true',
        help="Show detailed per-domain performance breakdown"
    )
    parser.add_argument(
        '--all-runs',
        action='store_false',
        dest='latest_only',
        help="Include all test runs instead of only the most recent one (default: only latest)"
    )
    parser.set_defaults(latest_only=True)
    parser.add_argument(
        '--comprehensive',
        action='store_true',
        help="Show comprehensive summary with top performers and strategy comparisons"
    )
    parser.add_argument(
        '--dataset-insights',
        action='store_true',
        help="Show high-level insights per dataset"
    )
    parser.add_argument(
        '--domain-insights',
        action='store_true',
        help="Show high-level insights per weather domain"
    )
    parser.add_argument(
        '--combined-insights',
        action='store_true',
        help="Show insights for combined gen+std strategies"
    )
    parser.add_argument(
        '--all-insights',
        action='store_true',
        help="Show all insights (comprehensive + dataset + domain + combined)"
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=5,
        help="Number of top configurations to show (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = TestResultAnalyzer(args.root)
    
    # Scan directory
    analyzer.scan_directory(verbose=args.verbose)
    
    # Deduplicate results if requested
    if args.latest_only:
        analyzer.deduplicate_results()
    
    # Generate output
    if args.format == 'table':
        print("\n" + analyzer.format_table(show_domains=args.show_domains))
        print("\n" + analyzer.format_summary())
        
        # Always show comprehensive summary for better insight
        print(analyzer.format_comprehensive_summary(top_n=args.top_n))
        
        if args.domain_breakdown:
            print(analyzer.format_per_domain_table())
        
        # Show dataset insights if requested or all-insights
        if args.dataset_insights or args.all_insights:
            print(analyzer.format_dataset_insights())
        
        # Show domain insights if requested or all-insights
        if args.domain_insights or args.all_insights:
            print(analyzer.format_domain_insights())
        
        # Show combined strategy insights if requested or all-insights
        if args.combined_insights or args.all_insights:
            print(analyzer.format_combined_strategy_insights())
    elif args.format == 'json':
        output_path = args.output or 'test_results_summary.json'
        analyzer.export_json(output_path)
    elif args.format == 'csv':
        output_path = args.output or 'test_results_summary.csv'
        analyzer.export_csv(output_path)


if __name__ == '__main__':
    main()
