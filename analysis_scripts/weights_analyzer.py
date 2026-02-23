#!/usr/bin/env python3
"""
PROVE Weights Directory Analyzer

This script analyzes the generated weights in the PROVE project and generates
a nicely formatted summary table of all training configurations, checkpoints, and metadata.

Supports both Stage 1 and Stage 2:
- Stage 1 (WEIGHTS/): Models trained only on clear_day
- Stage 2 (WEIGHTS_STAGE_2/): Models trained on all domains

Usage:
    python weights_analyzer.py                        # Stage 1 (default)
    python weights_analyzer.py --stage 2             # Stage 2
    python weights_analyzer.py --root /path/to/weights
    python weights_analyzer.py --format {table,json,csv} [--verbose]
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

# Stage-specific weights directories
WEIGHTS_ROOT_STAGE1 = "${AWARE_DATA_ROOT}/WEIGHTS/"
WEIGHTS_ROOT_STAGE2 = "${AWARE_DATA_ROOT}/WEIGHTS_STAGE_2/"


class WeightsAnalyzer:
    """Analyze PROVE weights directory structure and generate summaries."""
    
    def __init__(self, root_dir: str = "${AWARE_DATA_ROOT}/WEIGHTS/"):
        self.root_dir = Path(root_dir)
        self.weights_data = []
        
    def scan_directory(self, verbose: bool = False) -> List[Dict]:
        """
        Scan the weights directory recursively and collect information.
        
        Returns:
            List of dictionaries containing weight configuration data
        """
        if not self.root_dir.exists():
            print(f"Error: Directory {self.root_dir} does not exist")
            return []
        
        print(f"Scanning {self.root_dir}...")
        
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
                    
                    # Collect checkpoint information
                    checkpoint_info = self._analyze_model_dir(
                        model_dir, strategy, dataset, model
                    )
                    
                    if checkpoint_info:
                        self.weights_data.append(checkpoint_info)
                        
                        if verbose:
                            print(f"  Found: {strategy}/{dataset}/{model} - "
                                  f"{checkpoint_info['num_checkpoints']} checkpoints")
        
        print(f"Found {len(self.weights_data)} training configurations")
        return self.weights_data
    
    def _analyze_model_dir(
        self, 
        model_dir: Path, 
        strategy: str, 
        dataset: str, 
        model: str
    ) -> Optional[Dict]:
        """
        Analyze a single model directory.
        
        Returns:
            Dictionary with checkpoint information or None if no checkpoints found
        """
        # Find all .pth checkpoint files
        checkpoints = list(model_dir.glob("*.pth"))
        
        if not checkpoints:
            return None
        
        # Get checkpoint details
        checkpoint_files = []
        total_size = 0
        latest_time = None
        
        for ckpt in checkpoints:
            try:
                stat = ckpt.stat()
                size = stat.st_size
                mtime = datetime.fromtimestamp(stat.st_mtime)
                
                checkpoint_files.append({
                    'name': ckpt.name,
                    'size': size,
                    'modified': mtime
                })
                
                total_size += size
                
                if latest_time is None or mtime > latest_time:
                    latest_time = mtime
                    
            except Exception as e:
                print(f"Warning: Could not stat {ckpt}: {e}")
                continue
        
        # Try to read training config
        config_path = model_dir / "training_config.py"
        config_info = {}
        
        if config_path.exists():
            config_info = self._parse_training_config(config_path)
        
        # Check for test results
        test_results_dirs = list(model_dir.glob("test_results*"))
        # Filter out test_results_detailed from basic test results count
        basic_test_dirs = [d for d in test_results_dirs if d.name != "test_results_detailed"]
        has_test_results = len(basic_test_dirs) > 0
        
        # Check for detailed test results
        detailed_results_dir = model_dir / "test_results_detailed"
        has_detailed_test_results = (
            detailed_results_dir.exists() and 
            detailed_results_dir.is_dir() and
            len(list(detailed_results_dir.iterdir())) > 0  # Has at least one timestamped subfolder
        )
        
        return {
            'strategy': strategy,
            'dataset': dataset,
            'model': model,
            'path': str(model_dir),
            'num_checkpoints': len(checkpoints),
            'checkpoint_files': sorted(
                checkpoint_files, 
                key=lambda x: x['name']
            ),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'total_size_gb': total_size / (1024 * 1024 * 1024),
            'latest_checkpoint': latest_time.isoformat() if latest_time else None,
            'has_config': config_path.exists(),
            'has_test_results': has_test_results,
            'has_detailed_test_results': has_detailed_test_results,
            'num_test_result_dirs': len(basic_test_dirs),
            **config_info
        }
    
    def _parse_training_config(self, config_path: Path) -> Dict:
        """
        Parse training_config.py to extract metadata.
        
        Returns:
            Dictionary with extracted config information
        """
        info = {}
        
        try:
            with open(config_path, 'r') as f:
                content = f.read()
            
            # Extract max_iters
            max_iters_match = re.search(r"max_iters\s*=\s*(\d+)", content)
            if max_iters_match:
                info['max_iters'] = int(max_iters_match.group(1))
            
            # Extract batch_size
            batch_size_match = re.search(r"batch_size\s*=\s*(\d+)", content)
            if batch_size_match:
                info['batch_size'] = int(batch_size_match.group(1))
            
            # Extract num_classes
            num_classes_match = re.search(r"num_classes\s*=\s*(\d+)", content)
            if num_classes_match:
                info['num_classes'] = int(num_classes_match.group(1))
                
        except Exception as e:
            print(f"Warning: Could not parse {config_path}: {e}")
        
        return info
    
    def format_table(self) -> str:
        """
        Format the collected data as a nicely formatted ASCII table.
        
        Returns:
            Formatted table string
        """
        if not self.weights_data:
            return "No weights data collected"
        
        # Build table rows
        rows = []
        
        # Header
        header = [
            "Strategy",
            "Dataset",
            "Model",
            "Checkpoints",
            "Total Size",
            "Latest",
            "Test Results",
            "Detailed"
        ]
        
        # Data rows
        for item in self.weights_data:
            # Format size
            if item['total_size_gb'] >= 1.0:
                size_str = f"{item['total_size_gb']:.2f} GB"
            else:
                size_str = f"{item['total_size_mb']:.1f} MB"
            
            # Format date
            if item['latest_checkpoint']:
                date_str = item['latest_checkpoint'].split('T')[0]
            else:
                date_str = "N/A"
            
            # Test results indicator
            if item['has_test_results']:
                test_str = f"✓ ({item['num_test_result_dirs']})"
            else:
                test_str = "—"
            
            # Detailed test results indicator
            detailed_str = "✓" if item.get('has_detailed_test_results', False) else "—"
            
            rows.append([
                item['strategy'],
                item['dataset'],
                item['model'],
                str(item['num_checkpoints']),
                size_str,
                date_str,
                test_str,
                detailed_str
            ])
        
        # Calculate column widths
        col_widths = [len(h) for h in header]
        
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))
        
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
                data_row += f" {cell:<{col_widths[i]}} │"
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
        if not self.weights_data:
            return "No data available"
        
        # Calculate statistics
        total_configs = len(self.weights_data)
        total_checkpoints = sum(item['num_checkpoints'] for item in self.weights_data)
        total_size_gb = sum(item['total_size_gb'] for item in self.weights_data)
        
        # Count by strategy
        strategies = defaultdict(int)
        for item in self.weights_data:
            strategies[item['strategy']] += 1
        
        # Count by dataset
        datasets = defaultdict(int)
        for item in self.weights_data:
            datasets[item['dataset']] += 1
        
        # Count with test results
        with_tests = sum(1 for item in self.weights_data if item['has_test_results'])
        with_detailed = sum(1 for item in self.weights_data if item.get('has_detailed_test_results', False))
        
        summary = []
        summary.append("=" * 70)
        summary.append("WEIGHTS DIRECTORY SUMMARY")
        summary.append("=" * 70)
        summary.append(f"Total Configurations: {total_configs}")
        summary.append(f"Total Checkpoints: {total_checkpoints}")
        summary.append(f"Total Storage: {total_size_gb:.2f} GB")
        summary.append(f"Configurations with Test Results: {with_tests}")
        summary.append(f"Configurations with Detailed Test Results: {with_detailed}")
        summary.append("")
        summary.append("Strategies:")
        for strategy, count in sorted(strategies.items()):
            summary.append(f"  - {strategy}: {count} configurations")
        summary.append("")
        summary.append("Datasets:")
        for dataset, count in sorted(datasets.items()):
            summary.append(f"  - {dataset}: {count} configurations")
        summary.append("=" * 70)
        
        return "\n".join(summary)
    
    def export_json(self, output_path: str):
        """Export data to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.weights_data, f, indent=2, default=str)
        print(f"Exported JSON to {output_path}")
    
    def export_csv(self, output_path: str):
        """Export data to CSV file."""
        import csv
        
        if not self.weights_data:
            print("No data to export")
            return
        
        # Define CSV columns
        columns = [
            'strategy', 'dataset', 'model', 'num_checkpoints', 
            'total_size_mb', 'latest_checkpoint', 'has_test_results',
            'max_iters', 'batch_size', 'num_classes'
        ]
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.weights_data)
        
        print(f"Exported CSV to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze PROVE weights directory and generate summary"
    )
    parser.add_argument(
        '--stage',
        type=int,
        choices=[1, 2],
        default=1,
        help="Stage to analyze (1=clear_day training, 2=all domains training)"
    )
    parser.add_argument(
        '--root',
        default=None,
        help="Override root directory to scan"
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
        '--summary-only',
        action='store_true',
        help="Show only summary statistics"
    )
    
    args = parser.parse_args()
    
    # Determine root directory
    if args.root:
        root_dir = args.root
    elif args.stage == 1:
        root_dir = WEIGHTS_ROOT_STAGE1
    else:
        root_dir = WEIGHTS_ROOT_STAGE2
    
    stage_name = f"Stage {args.stage}"
    stage_desc = "Clear Day Training" if args.stage == 1 else "All Domains Training"
    print(f"=== PROVE Weights Analyzer - {stage_name} ({stage_desc}) ===")
    
    # Create analyzer
    analyzer = WeightsAnalyzer(root_dir)
    
    # Scan directory
    analyzer.scan_directory(verbose=args.verbose)
    
    # Generate output
    if args.format == 'table':
        if not args.summary_only:
            print("\n" + analyzer.format_table())
        print("\n" + analyzer.format_summary())
    elif args.format == 'json':
        output_path = args.output or f'weights_summary_stage{args.stage}.json'
        analyzer.export_json(output_path)
    elif args.format == 'csv':
        output_path = args.output or f'weights_summary_stage{args.stage}.csv'
        analyzer.export_csv(output_path)


if __name__ == '__main__':
    main()
