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


class TestResultAnalyzer:
    """Analyze PROVE test results and generate performance summaries."""
    
    def __init__(self, root_dir: str = "/scratch/aaa_exchange/AWARE/WEIGHTS/"):
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
        # Use baseline strategy with clear_day training data as reference
        baseline_clear_day_results = [
            r for r in valid_results 
            if r['strategy'] == 'baseline' and '_clear_day' in r['model']
        ]
        
        if baseline_clear_day_results:
            baseline_avg = sum(r['mIoU'] for r in baseline_clear_day_results) / len(baseline_clear_day_results)
            baseline_label = "Baseline (clear_day training)"
        else:
            # Fall back to any baseline if no clear_day variants exist
            baseline_results = [r for r in valid_results if r['strategy'] == 'baseline']
            baseline_avg = sum(r['mIoU'] for r in baseline_results) / len(baseline_results) if baseline_results else 0
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
        # Clean model names (remove _clear_day suffix for grouping)
        model_base_names = set()
        for model in models:
            base_name = model.replace('_clear_day', '')
            model_base_names.add(base_name)
        
        model_stats = {}
        for model in sorted(model_base_names):
            # Include both base model and _clear_day variant
            model_results = [r for r in valid_results 
                           if r['model'] == model or r['model'] == f"{model}_clear_day"]
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
                # Group by base model (remove _clear_day suffix)
                model_base = r['model'].replace('_clear_day', '')
                if model_base not in model_stats:
                    model_stats[model_base] = []
                model_stats[model_base].append(r['mIoU'])
            
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
    
    def export_json(self, output_path: str):
        """Export data to JSON file."""
        # Convert to serializable format
        export_data = []
        for result in self.test_results:
            item = dict(result)
            # Remove non-serializable fields
            if 'per_domain_metrics' in item:
                item['per_domain_metrics'] = dict(item['per_domain_metrics'])
            export_data.append(item)
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        print(f"Exported JSON to {output_path}")
    
    def export_csv(self, output_path: str):
        """Export data to CSV file."""
        import csv
        
        if not self.test_results:
            print("No data to export")
            return
        
        # Define CSV columns
        columns = [
            'strategy', 'dataset', 'model', 'test_type', 'result_type',
            'mIoU', 'mAcc', 'aAcc', 'fwIoU', 
            'has_per_domain', 'has_per_class', 'timestamp'
        ]
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.test_results)
        
        print(f"Exported CSV to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze PROVE test results and generate summary"
    )
    parser.add_argument(
        '--root',
        default="/scratch/aaa_exchange/AWARE/WEIGHTS/",
        help="Root directory to scan (default: /scratch/aaa_exchange/AWARE/WEIGHTS/)"
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
        '--all-insights',
        action='store_true',
        help="Show all insights (comprehensive + dataset + domain)"
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
    elif args.format == 'json':
        output_path = args.output or 'test_results_summary.json'
        analyzer.export_json(output_path)
    elif args.format == 'csv':
        output_path = args.output or 'test_results_summary.csv'
        analyzer.export_csv(output_path)


if __name__ == '__main__':
    main()
