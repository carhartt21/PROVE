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
        """Parse detailed test results from a timestamp directory."""
        # Look for metrics files
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
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = TestResultAnalyzer(args.root)
    
    # Scan directory
    analyzer.scan_directory(verbose=args.verbose)
    
    # Generate output
    if args.format == 'table':
        print("\n" + analyzer.format_table(show_domains=args.show_domains))
        print("\n" + analyzer.format_summary())
        
        if args.domain_breakdown:
            print(analyzer.format_per_domain_table())
    elif args.format == 'json':
        output_path = args.output or 'test_results_summary.json'
        analyzer.export_json(output_path)
    elif args.format == 'csv':
        output_path = args.output or 'test_results_summary.csv'
        analyzer.export_csv(output_path)


if __name__ == '__main__':
    main()
