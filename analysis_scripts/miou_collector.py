#!/usr/bin/env python3
"""
mIoU Collector - Collects mIoU values from WEIGHTS and WEIGHTS_STAGE_2 directories.

Extracts mIoU values from:
1. test_results_detailed/*/results.json files (preferred - test metrics)
2. Training log files (*.log) (fallback - validation metrics with iteration number)

Results are grouped by: stage, augmentation strategy, model, and dataset.

Usage:
    python analysis_scripts/miou_collector.py [options]
    
Examples:
    # Basic collection with table output
    python analysis_scripts/miou_collector.py
    
    # JSON output
    python analysis_scripts/miou_collector.py --format json
    
    # CSV output for analysis
    python analysis_scripts/miou_collector.py --format csv --output miou_results.csv
    
    # Include per-domain mIoU from test results
    python analysis_scripts/miou_collector.py --include-domains
    
    # Only collect from log files (validation metrics)
    python analysis_scripts/miou_collector.py --log-only
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
from dataclasses import dataclass, field, asdict
import csv


# Default paths
DEFAULT_WEIGHTS_DIR = "${AWARE_DATA_ROOT}/WEIGHTS"
DEFAULT_WEIGHTS_STAGE_2_DIR = "${AWARE_DATA_ROOT}/WEIGHTS_STAGE_2"


@dataclass
class MIoUResult:
    """Represents a single mIoU measurement."""
    stage: int
    strategy: str
    dataset: str
    model: str
    miou: float
    source: str  # 'test_results' or 'log'
    iteration: Optional[int] = None  # Only for log sources
    timestamp: Optional[str] = None
    checkpoint: Optional[str] = None
    per_domain_miou: Dict[str, float] = field(default_factory=dict)
    aAcc: Optional[float] = None
    fwIoU: Optional[float] = None
    num_images: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        # Remove empty per_domain_miou
        if not result['per_domain_miou']:
            del result['per_domain_miou']
        # Remove None values
        return {k: v for k, v in result.items() if v is not None}


def parse_log_file_miou(log_path: Path) -> List[Dict[str, Any]]:
    """
    Parse mIoU values from MMSegmentation log file.
    
    Returns list of dicts with keys: iteration, miou, aAcc, mAcc, fwIoU
    """
    results = []
    
    # Pattern for validation results:
    # Iter(val) [1857/1857]    val/aAcc: 89.68  val/mIoU: 39.92  val/mAcc: 46.42  val/fwIoU: 82.16
    val_pattern = re.compile(
        r'Iter\(val\)\s+\[\d+/\d+\]\s+'
        r'val/aAcc:\s*([0-9.]+)\s+'
        r'val/mIoU:\s*([0-9.]+)\s+'
        r'val/mAcc:\s*([0-9.]+)\s+'
        r'val/fwIoU:\s*([0-9.]+)'
    )
    
    # Pattern for checkpoint save:
    # Saving checkpoint at 5000 iterations
    checkpoint_pattern = re.compile(r'Saving checkpoint at (\d+) iterations')
    
    current_iteration = None
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Check for checkpoint save
                ckpt_match = checkpoint_pattern.search(line)
                if ckpt_match:
                    current_iteration = int(ckpt_match.group(1))
                    continue
                
                # Check for validation results
                val_match = val_pattern.search(line)
                if val_match:
                    results.append({
                        'iteration': current_iteration,
                        'aAcc': float(val_match.group(1)),
                        'miou': float(val_match.group(2)),
                        'mAcc': float(val_match.group(3)),
                        'fwIoU': float(val_match.group(4)),
                    })
    except Exception as e:
        print(f"Warning: Could not parse log file {log_path}: {e}", file=sys.stderr)
    
    return results


def parse_test_results(results_json_path: Path, include_domains: bool = False) -> Optional[Dict[str, Any]]:
    """
    Parse test results from results.json file.
    
    Returns dict with keys: miou, aAcc, fwIoU, num_images, per_domain_miou, checkpoint
    """
    try:
        with open(results_json_path, 'r') as f:
            data = json.load(f)
        
        result = {}
        
        # Extract overall metrics
        if 'overall' in data:
            overall = data['overall']
            result['miou'] = overall.get('mIoU')
            result['aAcc'] = overall.get('aAcc')
            result['fwIoU'] = overall.get('fwIoU')
            result['num_images'] = overall.get('num_images')
        
        # Extract checkpoint path from config
        if 'config' in data:
            config = data['config']
            result['checkpoint'] = config.get('checkpoint_path')
            result['timestamp'] = config.get('timestamp')
        
        # Extract per-domain mIoU if requested
        if include_domains and 'per_domain' in data:
            per_domain_miou = {}
            for domain, domain_data in data['per_domain'].items():
                if 'summary' in domain_data and 'mIoU' in domain_data['summary']:
                    per_domain_miou[domain] = domain_data['summary']['mIoU']
            if per_domain_miou:
                result['per_domain_miou'] = per_domain_miou
        
        return result
    except Exception as e:
        print(f"Warning: Could not parse test results {results_json_path}: {e}", file=sys.stderr)
        return None


def find_model_dirs(base_dir: Path) -> List[Tuple[str, str, Path]]:
    """
    Find all model directories under a base directory.
    
    Expected structure: base_dir/strategy/dataset/model/
    
    Returns list of (strategy, dataset, model_path) tuples.
    """
    results = []
    
    if not base_dir.exists():
        return results
    
    for strategy_dir in base_dir.iterdir():
        if not strategy_dir.is_dir() or strategy_dir.name.startswith('.'):
            continue
        
        strategy = strategy_dir.name
        
        for dataset_dir in strategy_dir.iterdir():
            if not dataset_dir.is_dir() or dataset_dir.name.startswith('.'):
                continue
            
            dataset = dataset_dir.name
            
            for model_dir in dataset_dir.iterdir():
                if not model_dir.is_dir() or model_dir.name.startswith('.'):
                    continue
                
                results.append((strategy, dataset, model_dir))
    
    return results


def collect_miou_from_model_dir(
    model_dir: Path,
    stage: int,
    strategy: str,
    dataset: str,
    include_domains: bool = False,
    log_only: bool = False
) -> List[MIoUResult]:
    """
    Collect mIoU results from a model directory.
    
    Prioritizes test results over log files unless log_only is True.
    """
    results = []
    model_name = model_dir.name
    
    # Try test_results_detailed first (unless log_only)
    if not log_only:
        test_results_dir = model_dir / "test_results_detailed"
        if test_results_dir.exists():
            # Find the most recent results.json
            for timestamp_dir in sorted(test_results_dir.iterdir(), reverse=True):
                if not timestamp_dir.is_dir():
                    continue
                
                results_json = timestamp_dir / "results.json"
                if results_json.exists():
                    parsed = parse_test_results(results_json, include_domains)
                    if parsed and parsed.get('miou') is not None:
                        # Extract iteration from checkpoint path if available
                        iteration = None
                        if parsed.get('checkpoint'):
                            iter_match = re.search(r'iter_(\d+)\.pth', parsed['checkpoint'])
                            if iter_match:
                                iteration = int(iter_match.group(1))
                        
                        result = MIoUResult(
                            stage=stage,
                            strategy=strategy,
                            dataset=dataset,
                            model=model_name,
                            miou=parsed['miou'],
                            source='test_results',
                            iteration=iteration,
                            timestamp=parsed.get('timestamp'),
                            checkpoint=parsed.get('checkpoint'),
                            aAcc=parsed.get('aAcc'),
                            fwIoU=parsed.get('fwIoU'),
                            num_images=parsed.get('num_images'),
                            per_domain_miou=parsed.get('per_domain_miou', {})
                        )
                        results.append(result)
                        # Only take the most recent test result
                        break
    
    # If no test results found (or log_only), try log files
    if not results or log_only:
        # Find log files in timestamp directories
        for item in model_dir.iterdir():
            if item.is_dir() and re.match(r'\d{8}_\d{6}', item.name):
                # This is a timestamp directory
                for log_file in item.glob('*.log'):
                    log_results = parse_log_file_miou(log_file)
                    if log_results:
                        # Get the last (most recent) validation result
                        last_result = log_results[-1]
                        result = MIoUResult(
                            stage=stage,
                            strategy=strategy,
                            dataset=dataset,
                            model=model_name,
                            miou=last_result['miou'],
                            source='log',
                            iteration=last_result.get('iteration'),
                            aAcc=last_result.get('aAcc'),
                            fwIoU=last_result.get('fwIoU'),
                        )
                        results.append(result)
                        break  # Only take first log file found
    
    return results


def collect_all_miou(
    weights_dir: str = DEFAULT_WEIGHTS_DIR,
    weights_stage_2_dir: str = DEFAULT_WEIGHTS_STAGE_2_DIR,
    include_domains: bool = False,
    log_only: bool = False,
    stage_filter: Optional[int] = None
) -> List[MIoUResult]:
    """
    Collect all mIoU values from both WEIGHTS directories.
    
    Args:
        weights_dir: Path to Stage 1 weights directory
        weights_stage_2_dir: Path to Stage 2 weights directory
        include_domains: Include per-domain mIoU values
        log_only: Only collect from log files
        stage_filter: Only collect from specific stage (1 or 2)
    
    Returns:
        List of MIoUResult objects
    """
    all_results = []
    
    # Stage 1
    if stage_filter is None or stage_filter == 1:
        stage1_path = Path(weights_dir)
        if stage1_path.exists():
            for strategy, dataset, model_dir in find_model_dirs(stage1_path):
                results = collect_miou_from_model_dir(
                    model_dir, 1, strategy, dataset, include_domains, log_only
                )
                all_results.extend(results)
    
    # Stage 2
    if stage_filter is None or stage_filter == 2:
        stage2_path = Path(weights_stage_2_dir)
        if stage2_path.exists():
            for strategy, dataset, model_dir in find_model_dirs(stage2_path):
                results = collect_miou_from_model_dir(
                    model_dir, 2, strategy, dataset, include_domains, log_only
                )
                all_results.extend(results)
    
    return all_results


def group_results(
    results: List[MIoUResult]
) -> Dict[int, Dict[str, Dict[str, Dict[str, MIoUResult]]]]:
    """
    Group results by stage -> strategy -> dataset -> model.
    
    Returns nested dict structure.
    """
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for result in results:
        grouped[result.stage][result.strategy][result.dataset][result.model] = result
    
    return grouped


def format_table(results: List[MIoUResult], verbose: bool = False) -> str:
    """Format results as a table."""
    if not results:
        return "No results found."
    
    lines = []
    grouped = group_results(results)
    
    for stage in sorted(grouped.keys()):
        lines.append(f"\n{'='*80}")
        lines.append(f"STAGE {stage}")
        lines.append(f"{'='*80}")
        
        stage_data = grouped[stage]
        
        for strategy in sorted(stage_data.keys()):
            lines.append(f"\n  Strategy: {strategy}")
            lines.append(f"  {'-'*70}")
            
            strategy_data = stage_data[strategy]
            
            # Create header
            if verbose:
                header = f"  {'Dataset':<20} {'Model':<30} {'mIoU':>8} {'Source':>12} {'Iter':>8}"
            else:
                header = f"  {'Dataset':<20} {'Model':<30} {'mIoU':>8}"
            lines.append(header)
            lines.append(f"  {'-'*70}")
            
            for dataset in sorted(strategy_data.keys()):
                dataset_data = strategy_data[dataset]
                
                for model in sorted(dataset_data.keys()):
                    result = dataset_data[model]
                    
                    if verbose:
                        iter_str = str(result.iteration) if result.iteration else '-'
                        line = f"  {dataset:<20} {model:<30} {result.miou:>8.2f} {result.source:>12} {iter_str:>8}"
                    else:
                        line = f"  {dataset:<20} {model:<30} {result.miou:>8.2f}"
                    lines.append(line)
    
    # Summary statistics
    lines.append(f"\n{'='*80}")
    lines.append("SUMMARY")
    lines.append(f"{'='*80}")
    lines.append(f"Total results collected: {len(results)}")
    
    # Count by source
    test_count = sum(1 for r in results if r.source == 'test_results')
    log_count = sum(1 for r in results if r.source == 'log')
    lines.append(f"  From test results: {test_count}")
    lines.append(f"  From log files: {log_count}")
    
    # Count by stage
    stage1_count = sum(1 for r in results if r.stage == 1)
    stage2_count = sum(1 for r in results if r.stage == 2)
    lines.append(f"  Stage 1: {stage1_count}")
    lines.append(f"  Stage 2: {stage2_count}")
    
    return '\n'.join(lines)


def format_csv(results: List[MIoUResult], include_domains: bool = False) -> str:
    """Format results as CSV."""
    if not results:
        return ""
    
    # Base columns
    columns = ['stage', 'strategy', 'dataset', 'model', 'miou', 'source', 'iteration', 
               'aAcc', 'fwIoU', 'num_images', 'checkpoint', 'timestamp']
    
    # Add domain columns if requested
    domain_columns = set()
    if include_domains:
        for result in results:
            domain_columns.update(result.per_domain_miou.keys())
        domain_columns = sorted(domain_columns)
        columns.extend([f"miou_{d}" for d in domain_columns])
    
    lines = [','.join(columns)]
    
    for result in results:
        row = [
            str(result.stage),
            result.strategy,
            result.dataset,
            result.model,
            f"{result.miou:.4f}",
            result.source,
            str(result.iteration) if result.iteration else '',
            f"{result.aAcc:.2f}" if result.aAcc else '',
            f"{result.fwIoU:.2f}" if result.fwIoU else '',
            str(result.num_images) if result.num_images else '',
            result.checkpoint or '',
            result.timestamp or '',
        ]
        
        if include_domains:
            for domain in domain_columns:
                val = result.per_domain_miou.get(domain)
                row.append(f"{val:.4f}" if val else '')
        
        lines.append(','.join(row))
    
    return '\n'.join(lines)


def format_json(results: List[MIoUResult]) -> str:
    """Format results as JSON."""
    return json.dumps([r.to_dict() for r in results], indent=2)


def format_grouped_json(results: List[MIoUResult]) -> str:
    """Format results as grouped JSON (stage -> strategy -> dataset -> model)."""
    grouped = group_results(results)
    
    # Convert MIoUResult objects to dicts
    output = {}
    for stage, stage_data in grouped.items():
        output[f"stage_{stage}"] = {}
        for strategy, strat_data in stage_data.items():
            output[f"stage_{stage}"][strategy] = {}
            for dataset, ds_data in strat_data.items():
                output[f"stage_{stage}"][strategy][dataset] = {}
                for model, result in ds_data.items():
                    output[f"stage_{stage}"][strategy][dataset][model] = result.to_dict()
    
    return json.dumps(output, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Collect mIoU values from WEIGHTS and WEIGHTS_STAGE_2 directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--weights-dir', '-w',
        default=DEFAULT_WEIGHTS_DIR,
        help=f"Stage 1 weights directory (default: {DEFAULT_WEIGHTS_DIR})"
    )
    parser.add_argument(
        '--weights-stage-2-dir', '-w2',
        default=DEFAULT_WEIGHTS_STAGE_2_DIR,
        help=f"Stage 2 weights directory (default: {DEFAULT_WEIGHTS_STAGE_2_DIR})"
    )
    parser.add_argument(
        '--format', '-f',
        choices=['table', 'csv', 'json', 'json-grouped'],
        default='table',
        help="Output format (default: table)"
    )
    parser.add_argument(
        '--output', '-o',
        help="Output file path (default: stdout)"
    )
    parser.add_argument(
        '--include-domains', '-d',
        action='store_true',
        help="Include per-domain mIoU values in output"
    )
    parser.add_argument(
        '--log-only',
        action='store_true',
        help="Only collect from log files (validation metrics), ignore test results"
    )
    parser.add_argument(
        '--stage', '-s',
        type=int,
        choices=[1, 2],
        help="Only collect from specific stage (1 or 2)"
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Show additional details in table output"
    )
    parser.add_argument(
        '--filter-strategy',
        help="Filter by strategy name (supports partial match)"
    )
    parser.add_argument(
        '--filter-dataset',
        help="Filter by dataset name (supports partial match)"
    )
    parser.add_argument(
        '--filter-model',
        help="Filter by model name (supports partial match)"
    )
    
    args = parser.parse_args()
    
    # Collect results
    results = collect_all_miou(
        weights_dir=args.weights_dir,
        weights_stage_2_dir=args.weights_stage_2_dir,
        include_domains=args.include_domains,
        log_only=args.log_only,
        stage_filter=args.stage
    )
    
    # Apply filters
    if args.filter_strategy:
        results = [r for r in results if args.filter_strategy.lower() in r.strategy.lower()]
    if args.filter_dataset:
        results = [r for r in results if args.filter_dataset.lower() in r.dataset.lower()]
    if args.filter_model:
        results = [r for r in results if args.filter_model.lower() in r.model.lower()]
    
    # Sort results
    results.sort(key=lambda r: (r.stage, r.strategy, r.dataset, r.model))
    
    # Format output
    if args.format == 'table':
        output = format_table(results, args.verbose)
    elif args.format == 'csv':
        output = format_csv(results, args.include_domains)
    elif args.format == 'json':
        output = format_json(results)
    elif args.format == 'json-grouped':
        output = format_grouped_json(results)
    else:
        output = format_table(results, args.verbose)
    
    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Results written to {args.output}")
    else:
        print(output)


if __name__ == '__main__':
    main()
