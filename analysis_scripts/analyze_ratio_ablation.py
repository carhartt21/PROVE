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
DEFAULT_WEIGHTS_ROOT = "${AWARE_DATA_ROOT}/WEIGHTS_RATIO_ABLATION"
DEFAULT_REGULAR_WEIGHTS_ROOT = "${AWARE_DATA_ROOT}/WEIGHTS"  # For baseline (0) and standard (0.5)
RATIOS = [0.0, 0.12, 0.25, 0.38, 0.5, 0.62, 0.75, 0.88, 1.0]
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
    
    # Dataset name normalization map
    DATASET_ALIASES = {
        'idd-aw': 'iddaw',
        'IDD-AW': 'iddaw',
        'bdd10k': 'bdd10k',
        'BDD10k': 'bdd10k',
        'outside15k': 'outside15k',
        'OUTSIDE15k': 'outside15k',
        'mapillaryvistas': 'mapillaryvistas',
        'MapillaryVistas': 'mapillaryvistas',
    }
    
    # Datasets to EXCLUDE from ratio comparison due to inconsistent configurations
    # (e.g., MapillaryVistas uses different num_classes between WEIGHTS and WEIGHTS_RATIO_ABLATION)
    # outside15k is excluded to focus on the two main datasets (BDD10k and IDD-AW)
    INCONSISTENT_DATASETS = {'mapillaryvistas', 'outside15k'}
    
    # Models that exist in the ratio ablation study (WEIGHTS_RATIO_ABLATION)
    # deeplabv3plus_r50 only exists in WEIGHTS/ at ratio 0.5 and 1.0, not in ablation
    CONSISTENT_MODELS = {'pspnet_r50', 'segformer_mit-b5'}
    
    # Strategies with stage mismatch: ratio ablation is stage 2 (all domains),
    # but ratio 0.50 and 1.00 from WEIGHTS/ are stage 1 (clear_day only)
    # These MUST be excluded from ratio comparison for valid apples-to-apples comparison
    STAGE_MISMATCH_STRATEGIES = {'gen_step1x_new', 'gen_step1x_v1p2'}
    
    def __init__(self, weights_root: str = DEFAULT_WEIGHTS_ROOT,
                 regular_weights_root: str = DEFAULT_REGULAR_WEIGHTS_ROOT):
        self.weights_root = Path(weights_root)
        self.regular_weights_root = Path(regular_weights_root)
        self.results: List[RatioResult] = []
        self.ratio_pattern = re.compile(r'_ratio(\d+)p(\d+)')
    
    def _normalize_dataset(self, dataset: str) -> str:
        """Normalize dataset name for consistent comparison."""
        return self.DATASET_ALIASES.get(dataset, dataset)
    
    def get_consistent_results(self) -> List[RatioResult]:
        """Get results filtered to only include datasets and models with consistent configurations.
        
        Also excludes strategies with stage mismatch (e.g., step1x_* where ablation is stage 2
        but ratio 0.50/1.00 are stage 1).
        """
        return [r for r in self.results 
                if r.dataset not in self.INCONSISTENT_DATASETS 
                and r.model in self.CONSISTENT_MODELS
                and r.strategy not in self.STAGE_MISMATCH_STRATEGIES]
    
    def get_common_config_results(self) -> List[RatioResult]:
        """Get results filtered to only include configurations that exist across ALL ratios.
        
        This ensures true apples-to-apples comparison by only including (strategy, dataset, model)
        combinations that have results for every ratio value in the study.
        """
        # First filter by consistent datasets and models
        consistent = self.get_consistent_results()
        
        # Group by ratio to find configs per ratio
        from collections import defaultdict
        ratio_configs: Dict[float, set] = defaultdict(set)
        config_results: Dict[Tuple[float, str, str, str], RatioResult] = {}
        
        for result in consistent:
            config = (result.strategy, result.dataset, result.model)
            ratio_configs[result.ratio].add(config)
            config_results[(result.ratio, result.strategy, result.dataset, result.model)] = result
        
        if not ratio_configs:
            return []
        
        # Find intersection of all configs across all ratios
        all_ratios = sorted(ratio_configs.keys())
        common_configs = ratio_configs[all_ratios[0]].copy()
        for ratio in all_ratios[1:]:
            common_configs &= ratio_configs[ratio]
        
        # Return only results for common configs
        common_results = []
        for result in consistent:
            config = (result.strategy, result.dataset, result.model)
            if config in common_configs:
                common_results.append(result)
        
        return common_results
    
    def get_globally_common_config_results(self) -> List[RatioResult]:
        """Get results filtered to only include (dataset, model) pairs common to ALL strategies.
        
        This ensures truly fair comparison where the baseline (ratio 1.0) is the same
        across all strategies, since they share the same (dataset, model) combinations.
        """
        # First get common configs per strategy
        common = self.get_common_config_results()
        
        if not common:
            return []
        
        # Group by strategy to find (dataset, model) pairs per strategy
        from collections import defaultdict
        strat_configs: Dict[str, set] = defaultdict(set)
        for r in common:
            strat_configs[r.strategy].add((r.dataset, r.model))
        
        # Find intersection of (dataset, model) across ALL strategies
        strategies = list(strat_configs.keys())
        global_common = strat_configs[strategies[0]].copy()
        for strategy in strategies[1:]:
            global_common &= strat_configs[strategy]
        
        # Filter to only results with globally common (dataset, model)
        return [r for r in common if (r.dataset, r.model) in global_common]
        
    def scan_results(self, verbose: bool = False, include_regular: bool = True) -> int:
        """
        Scan the weights directory for ratio ablation results.
        
        Args:
            verbose: Print detailed progress information
            include_regular: Also scan regular WEIGHTS folder for baseline (ratio=1.0 = 100% real)
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
        
        # Scan regular weights for baseline (ratio=1.0) and standard training (ratio=0.5)
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
            
            try:
                dataset_dirs = list(strategy_dir.iterdir())
            except PermissionError:
                if verbose:
                    print(f"Warning: Permission denied for {strategy_dir}, skipping")
                continue
            
            for dataset_dir in sorted(dataset_dirs):
                if not dataset_dir.is_dir():
                    continue
                
                dataset = dataset_dir.name
                
                try:
                    model_dirs = list(dataset_dir.iterdir())
                except PermissionError:
                    if verbose:
                        print(f"Warning: Permission denied for {dataset_dir}, skipping")
                    continue
                
                for model_dir in sorted(model_dirs):
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
                        result.dataset = self._normalize_dataset(dataset)
                        result.model = base_model
                        result.ratio = ratio
                        result.stage = stage if stage else 0
                        self.results.append(result)
                        
                        if verbose:
                            stage_str = f" [Stage {stage}]" if stage else ""
                            print(f"Found: {strategy}/{result.dataset}/{base_model} ratio={ratio:.3f} mIoU={result.miou:.2f}{stage_str}")
        
        return len(self.results)
    
    def _get_ablation_strategies(self) -> set:
        """Get all strategies present in the ratio ablation directory."""
        strategies = set()
        for stage_dir in ['stage1', 'stage2']:
            stage_path = self.weights_root / stage_dir
            if stage_path.exists():
                for strategy_dir in stage_path.iterdir():
                    if strategy_dir.is_dir() and not strategy_dir.name.startswith('_'):
                        strategies.add(strategy_dir.name)
        # Also check flat structure (legacy)
        if not (self.weights_root / 'stage1').exists() and not (self.weights_root / 'stage2').exists():
            for strategy_dir in self.weights_root.iterdir():
                if strategy_dir.is_dir() and strategy_dir.name.startswith('gen_'):
                    strategies.add(strategy_dir.name)
        return strategies

    def _scan_regular_weights(self, verbose: bool = False):
        """
        Scan regular WEIGHTS folder for baseline and standard training results.
        
        Ratio convention (real_gen_ratio = proportion of REAL images):
        - 'baseline' folder -> ratio=1.0 (100% real images, 0% generated)
        - 'gen_*' folders -> ratio=0.5 (standard gen training with 50/50 split)
          OR extracts ratio from model name suffix (e.g., _ratio0p50 -> 0.5)
        
        Scans all strategies present in WEIGHTS_RATIO_ABLATION, not just TOP_5.
        """
        # Get all strategies from the ablation study
        ablation_strategies = self._get_ablation_strategies()
        if verbose:
            print(f"Strategies in ablation study: {sorted(ablation_strategies)}")
        
        for strategy_dir in sorted(self.regular_weights_root.iterdir()):
            if not strategy_dir.is_dir():
                continue
            
            strategy = strategy_dir.name
            
            # Skip strategies not in our ablation study (except baseline)
            if strategy != 'baseline' and strategy not in ablation_strategies:
                continue
            
            for dataset_dir in sorted(strategy_dir.iterdir()):
                if not dataset_dir.is_dir():
                    continue
                
                dataset = dataset_dir.name
                
                for model_dir in sorted(dataset_dir.iterdir()):
                    if not model_dir.is_dir():
                        continue
                    
                    model_name = model_dir.name
                    
                    # Extract ratio from model name if present
                    extracted_ratio = self._extract_ratio(model_name)
                    if extracted_ratio is not None:
                        ratio = extracted_ratio
                        base_model = self.ratio_pattern.sub('', model_name)
                    else:
                        # Default ratio based on strategy type
                        if strategy == 'baseline':
                            ratio = 1.0  # Baseline = 100% real
                        else:
                            ratio = 0.5  # Standard gen training = 50/50
                        base_model = model_name
                    
                    # Normalize dataset name
                    normalized_dataset = self._normalize_dataset(dataset)
                    
                    # Look for test results
                    result = self._find_latest_test_result(model_dir)
                    if result:
                        result.dataset = normalized_dataset
                        result.model = base_model
                        result.ratio = ratio
                        
                        if strategy == 'baseline':
                            # Add baseline result for each gen strategy in the ablation study
                            for gen_strategy in ablation_strategies:
                                baseline_result = RatioResult(
                                    strategy=gen_strategy,
                                    dataset=normalized_dataset,
                                    model=base_model,
                                    ratio=1.0,  # 100% real images (baseline)
                                    miou=result.miou,
                                    macc=result.macc,
                                    aacc=result.aacc,
                                    fwiou=result.fwiou,
                                    timestamp=result.timestamp,
                                    checkpoint_iter=result.checkpoint_iter
                                )
                                self.results.append(baseline_result)
                                
                                if verbose:
                                    print(f"Found baseline for {gen_strategy}/{normalized_dataset}/{base_model} ratio=1.0 mIoU={result.miou:.2f}")
                        else:
                            result.strategy = strategy
                            self.results.append(result)
                            
                            if verbose:
                                print(f"Found: {strategy}/{normalized_dataset}/{base_model} ratio={ratio:.2f} mIoU={result.miou:.2f}")
    
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
        try:
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
        except PermissionError:
            # Skip directories we don't have permission to read
            pass
        
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
    
    def get_summary_by_ratio(self, consistent_only: bool = True, common_only: bool = True, 
                              globally_common: bool = False) -> Dict[float, Dict[str, float]]:
        """Get average metrics for each ratio value.
        
        Args:
            consistent_only: If True, exclude datasets with inconsistent configurations
                           (e.g., MapillaryVistas with different num_classes)
            common_only: If True, only include configurations present across ALL ratios
            globally_common: If True, only include (dataset, model) pairs common to ALL strategies
                           (ensures baseline is identical across strategies)
        """
        if globally_common:
            results = self.get_globally_common_config_results()
        elif common_only:
            results = self.get_common_config_results()
        elif consistent_only:
            results = self.get_consistent_results()
        else:
            results = self.results
            
        ratio_metrics = defaultdict(lambda: {'miou': [], 'macc': [], 'aacc': [], 'fwiou': []})
        
        for result in results:
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
    
    def get_summary_by_strategy(self, consistent_only: bool = True, common_only: bool = True,
                                 globally_common: bool = False) -> Dict[str, Dict[float, float]]:
        """Get mIoU for each strategy at each ratio.
        
        Args:
            consistent_only: If True, exclude datasets with inconsistent configurations
            common_only: If True, only include configurations present across ALL ratios
            globally_common: If True, only include (dataset, model) pairs common to ALL strategies
                           (ensures baseline is identical across strategies)
        """
        if globally_common:
            results = self.get_globally_common_config_results()
        elif common_only:
            results = self.get_common_config_results()
        elif consistent_only:
            results = self.get_consistent_results()
        else:
            results = self.results
        strategy_ratios = defaultdict(lambda: defaultdict(list))
        
        for result in results:
            strategy_ratios[result.strategy][result.ratio].append(result.miou)
        
        summary = {}
        for strategy, ratios in sorted(strategy_ratios.items()):
            summary[strategy] = {}
            for ratio, mious in sorted(ratios.items()):
                summary[strategy][ratio] = sum(mious) / len(mious)
        
        return summary
    
    def get_optimal_ratios(self, consistent_only: bool = True) -> Dict[Tuple[str, str, str], Tuple[float, float]]:
        """Find optimal ratio for each strategy/dataset/model combination.
        
        Args:
            consistent_only: If True, exclude datasets with inconsistent configurations
        """
        results = self.get_consistent_results() if consistent_only else self.results
        config_results = defaultdict(list)
        
        for result in results:
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
        
        # Note about consistent results
        consistent_results = self.get_consistent_results()
        common_results = self.get_common_config_results()
        global_common_results = self.get_globally_common_config_results()
        print(f"Consistent Results: {len(consistent_results)} (excluding {self.INCONSISTENT_DATASETS}, models not in {self.CONSISTENT_MODELS}, and {self.STAGE_MISMATCH_STRATEGIES})")
        print(f"Common Config Results: {len(common_results)} (configs present across ALL ratios)")
        print(f"Globally Common Results: {len(global_common_results)} (same (dataset,model) across ALL strategies)")
        
        # Calculate common configs count
        configs_per_ratio = len(common_results) // len(set(r.ratio for r in common_results)) if common_results else 0
        global_configs_per_ratio = len(global_common_results) // len(set(r.ratio for r in global_common_results)) if global_common_results else 0
        print(f"Common Configurations: {configs_per_ratio} (strategy, dataset, model) tuples")
        print(f"Globally Common: {global_configs_per_ratio // 6} (dataset, model) pairs × 6 strategies")
        
        # Summary by ratio (using globally common configs)
        ratio_summary = self.get_summary_by_ratio(globally_common=True)
        print("\n" + "-" * 50)
        print("Average Metrics by Ratio (Globally Common Configs)")
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
        strategy_summary = self.get_summary_by_strategy(globally_common=True)
        print("\n" + "-" * 50)
        print("mIoU by Strategy and Ratio (Globally Common)")
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
                       help=f'Regular weights root for baseline (ratio 1.0) and standard (ratio 0.5) results (default: {DEFAULT_REGULAR_WEIGHTS_ROOT})')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--format', choices=['csv', 'json'], default='csv',
                       help='Output format (default: csv)')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed breakdown')
    parser.add_argument('--no-regular', action='store_true',
                       help='Do not include baseline/standard (ratio 1.0/0.5) from regular WEIGHTS folder')
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
