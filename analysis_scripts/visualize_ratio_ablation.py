#!/usr/bin/env python3
"""
PROVE Ratio Ablation Study Visualizer

Generates visualizations for the ratio ablation study results.

Features:
- Line plots showing mIoU vs ratio for each strategy
- Heatmaps of mIoU across strategies and ratios
- Bar charts comparing optimal ratios
- Box plots showing variance at each ratio
- Dataset-specific analysis plots
- Model comparison plots

Usage:
    # Generate all visualizations
    python visualize_ratio_ablation.py

    # Specify custom weights root
    python visualize_ratio_ablation.py --weights-root /path/to/weights

    # Generate specific plot types
    python visualize_ratio_ablation.py --plots line heatmap bar

    # Save to specific directory
    python visualize_ratio_ablation.py --output-dir ./figures/ratio_ablation
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_ratio_ablation import RatioAblationAnalyzer, DEFAULT_WEIGHTS_ROOT, DEFAULT_REGULAR_WEIGHTS_ROOT, RATIOS

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError as e:
    HAS_PLOTTING = False
    np = None
    pd = None
    plt = None
    sns = None
    GridSpec = None
    mpatches = None
    print(f"Warning: Required plotting libraries not installed: {e}")
    print("Install with: pip install matplotlib seaborn pandas numpy")

# Use TYPE_CHECKING for type hints that require matplotlib
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from matplotlib.figure import Figure


# Color palette for strategies
STRATEGY_COLORS = {
    'gen_LANIT': '#1f77b4',
    'gen_step1x_new': '#ff7f0e',
    'gen_automold': '#2ca02c',
    'gen_TSIT': '#d62728',
    'gen_NST': '#9467bd',
}

# Color palette for datasets
DATASET_COLORS = {
    'acdc': '#e41a1c',
    'bdd10k': '#377eb8',
    'idd-aw': '#4daf4a',
    'mapillaryvistas': '#984ea3',
    'outside15k': '#ff7f00',
}

# Color palette for models
MODEL_COLORS = {
    'deeplabv3plus_r50': '#66c2a5',
    'pspnet_r50': '#fc8d62',
    'segformer_mit-b5': '#8da0cb',
}


class RatioAblationVisualizer:
    """Visualizer for ratio ablation study results."""
    
    def __init__(self, analyzer: RatioAblationAnalyzer, output_dir: str = './figures/ratio_ablation'):
        self.analyzer = analyzer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def _save_figure(self, fig: 'Figure', name: str, dpi: int = 150):
        """Save figure to output directory."""
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Saved: {path}")
        
        # Also save PDF version
        pdf_path = self.output_dir / f"{name}.pdf"
        fig.savefig(pdf_path, bbox_inches='tight')
    
    def plot_miou_vs_ratio_by_strategy(self):
        """
        Line plot showing mIoU vs ratio for each strategy.
        Aggregated across all datasets and models.
        Uses globally common configs to ensure fair comparison with identical baseline.
        """
        strategy_summary = self.analyzer.get_summary_by_strategy(globally_common=True)
        
        if not strategy_summary:
            print("No data for mIoU vs ratio plot")
            return
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for strategy, ratios in strategy_summary.items():
            sorted_ratios = sorted(ratios.items())
            x = [r[0] for r in sorted_ratios]
            y = [r[1] for r in sorted_ratios]
            
            color = STRATEGY_COLORS.get(strategy, None)
            ax.plot(x, y, 'o-', label=strategy, color=color, markersize=8, linewidth=2)
        
        ax.set_xlabel('Real/Generated Ratio', fontsize=12)
        ax.set_ylabel('Mean IoU (%)', fontsize=12)
        ax.set_title('mIoU vs Real/Generated Ratio by Strategy\n(Globally Common Configs)', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.set_xticks(RATIOS)
        ax.set_xlim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        
        self._save_figure(fig, 'miou_vs_ratio_by_strategy')
    
    def plot_heatmap_strategy_ratio(self):
        """
        Heatmap showing mIoU GAIN (vs baseline) for each strategy-ratio combination.
        Uses globally common configs to ensure baseline is identical across strategies.
        Excludes ratio 1.0 (baseline) column and highlights best ratio per strategy.
        """
        strategy_summary = self.analyzer.get_summary_by_strategy(globally_common=True)
        
        if not strategy_summary:
            print("No data for heatmap")
            return
        
        # Build matrix of mIoU GAINS (vs baseline at ratio 1.0)
        strategies = sorted(strategy_summary.keys())
        # Exclude ratio 1.0 from the visualization
        ratios = sorted([r for r in set(r for s in strategy_summary.values() for r in s.keys()) if r != 1.0])
        
        matrix = np.zeros((len(strategies), len(ratios)))
        best_ratio_idx = []  # Track best ratio index for each strategy
        
        for i, strategy in enumerate(strategies):
            baseline_miou = strategy_summary[strategy].get(1.0, 0)
            best_gain = -float('inf')
            best_j = 0
            for j, ratio in enumerate(ratios):
                miou = strategy_summary[strategy].get(ratio, np.nan)
                gain = miou - baseline_miou if not np.isnan(miou) else np.nan
                matrix[i, j] = gain
                if not np.isnan(gain) and gain > best_gain:
                    best_gain = gain
                    best_j = j
            best_ratio_idx.append(best_j)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(matrix, 
                   xticklabels=[f"{r:.2f}" for r in ratios],
                   yticklabels=[s.replace('gen_', '') for s in strategies],
                   annot=True, 
                   fmt='+.2f',
                   cmap='RdYlGn',
                   center=0,
                   vmin=-1.5,
                   vmax=2.5,
                   ax=ax)
        
        # Highlight best ratio for each strategy with a rectangle
        for i, best_j in enumerate(best_ratio_idx):
            ax.add_patch(plt.Rectangle((best_j, i), 1, 1, fill=False, 
                                       edgecolor='blue', linewidth=3))
        
        ax.set_xlabel('Real/Generated Ratio', fontsize=12)
        ax.set_ylabel('Strategy', fontsize=12)
        ax.set_title('mIoU Gain vs Baseline (ratio=1.0)\nBlue box = Best ratio per strategy', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        self._save_figure(fig, 'heatmap_strategy_ratio')
    
    def plot_optimal_ratio_distribution(self):
        """
        Bar chart showing distribution of optimal ratios across configurations.
        """
        optimal = self.analyzer.get_optimal_ratios()
        
        if not optimal:
            print("No data for optimal ratio distribution")
            return
        
        # Count optimal ratios
        ratio_counts = defaultdict(int)
        for (strategy, dataset, model), (ratio, miou) in optimal.items():
            ratio_counts[ratio] += 1
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ratios = sorted(ratio_counts.keys())
        counts = [ratio_counts[r] for r in ratios]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(ratios)))
        
        bars = ax.bar([f"{r:.3f}" for r in ratios], counts, color=colors, edgecolor='black')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.annotate(f'{count}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Optimal Ratio', fontsize=12)
        ax.set_ylabel('Number of Configurations', fontsize=12)
        ax.set_title('Distribution of Optimal Ratios', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        self._save_figure(fig, 'optimal_ratio_distribution')
    
    def plot_miou_boxplot_by_ratio(self):
        """
        Box plot showing mIoU distribution at each ratio.
        """
        if not self.analyzer.results:
            print("No data for boxplot")
            return
        
        # Organize data by ratio
        ratio_mious = defaultdict(list)
        for result in self.analyzer.results:
            ratio_mious[result.ratio].append(result.miou)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ratios = sorted(ratio_mious.keys())
        data = [ratio_mious[r] for r in ratios]
        
        bp = ax.boxplot(data, labels=[f"{r:.3f}" for r in ratios], patch_artist=True)
        
        # Color boxes
        colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(ratios)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Real/Generated Ratio', fontsize=12)
        ax.set_ylabel('Mean IoU (%)', fontsize=12)
        ax.set_title('mIoU Distribution by Ratio', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        self._save_figure(fig, 'miou_boxplot_by_ratio')
    
    def plot_miou_vs_ratio_by_dataset(self):
        """
        Line plots showing mIoU vs ratio, separated by dataset.
        """
        if not self.analyzer.results:
            print("No data for dataset plots")
            return
        
        # Organize data by dataset and ratio
        dataset_ratio_mious = defaultdict(lambda: defaultdict(list))
        for result in self.analyzer.results:
            dataset_ratio_mious[result.dataset][result.ratio].append(result.miou)
        
        datasets = sorted(dataset_ratio_mious.keys())
        n_datasets = len(datasets)
        
        if n_datasets == 0:
            return
        
        # Create subplot grid
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, dataset in enumerate(datasets):
            if i >= len(axes):
                break
            
            ax = axes[i]
            ratio_data = dataset_ratio_mious[dataset]
            
            ratios = sorted(ratio_data.keys())
            means = [np.mean(ratio_data[r]) for r in ratios]
            stds = [np.std(ratio_data[r]) for r in ratios]
            
            color = DATASET_COLORS.get(dataset.lower(), 'blue')
            ax.errorbar(ratios, means, yerr=stds, fmt='o-', color=color, 
                       capsize=4, markersize=6, linewidth=2)
            
            ax.set_xlabel('Ratio', fontsize=10)
            ax.set_ylabel('mIoU (%)', fontsize=10)
            ax.set_title(f'{dataset.upper()}', fontsize=12, fontweight='bold')
            ax.set_xticks(RATIOS)
            ax.set_xticklabels(['0', '.125', '.25', '.375', '.5', '.625', '.75', '.875', '1.0'], fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_datasets, len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle('mIoU vs Ratio by Dataset', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure(fig, 'miou_vs_ratio_by_dataset')
    
    def plot_miou_vs_ratio_by_model(self):
        """
        Line plots showing mIoU vs ratio, separated by model architecture.
        """
        if not self.analyzer.results:
            print("No data for model plots")
            return
        
        # Organize data by model and ratio
        model_ratio_mious = defaultdict(lambda: defaultdict(list))
        for result in self.analyzer.results:
            model_ratio_mious[result.model][result.ratio].append(result.miou)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for model, ratio_data in sorted(model_ratio_mious.items()):
            ratios = sorted(ratio_data.keys())
            means = [np.mean(ratio_data[r]) for r in ratios]
            
            color = MODEL_COLORS.get(model, None)
            ax.plot(ratios, means, 'o-', label=model, color=color, markersize=8, linewidth=2)
        
        ax.set_xlabel('Real/Generated Ratio', fontsize=12)
        ax.set_ylabel('Mean IoU (%)', fontsize=12)
        ax.set_title('mIoU vs Ratio by Model Architecture', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.set_xticks(RATIOS)
        ax.grid(True, alpha=0.3)
        
        self._save_figure(fig, 'miou_vs_ratio_by_model')
    
    def plot_relative_improvement(self):
        """
        Bar chart showing relative improvement of best ratio vs baseline (ratio=1.0).
        """
        optimal = self.analyzer.get_optimal_ratios()
        
        if not optimal:
            print("No data for relative improvement plot")
            return
        
        # Calculate improvement per strategy
        strategy_improvements = defaultdict(list)
        
        for (strategy, dataset, model), (best_ratio, best_miou) in optimal.items():
            # Find baseline (ratio=1.0) for this config
            baseline_miou = None
            for result in self.analyzer.results:
                if (result.strategy == strategy and 
                    result.dataset == dataset and 
                    result.model == model and 
                    result.ratio == 1.0):
                    baseline_miou = result.miou
                    break
            
            if baseline_miou and baseline_miou > 0:
                improvement = ((best_miou - baseline_miou) / baseline_miou) * 100
                strategy_improvements[strategy].append(improvement)
        
        if not strategy_improvements:
            print("No data for relative improvement calculation")
            return
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        strategies = sorted(strategy_improvements.keys())
        mean_improvements = [np.mean(strategy_improvements[s]) for s in strategies]
        std_improvements = [np.std(strategy_improvements[s]) for s in strategies]
        
        colors = [STRATEGY_COLORS.get(s, 'gray') for s in strategies]
        
        bars = ax.bar(strategies, mean_improvements, yerr=std_improvements, 
                     color=colors, edgecolor='black', capsize=5, alpha=0.8)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Strategy', fontsize=12)
        ax.set_ylabel('Relative Improvement (%)', fontsize=12)
        ax.set_title('Relative Improvement: Best Ratio vs Baseline (ratio=1.0)', 
                    fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, 'relative_improvement_vs_baseline')
    
    def plot_summary_dashboard(self):
        """
        Create a comprehensive dashboard with multiple plots.
        Uses globally common configs for fair comparison.
        """
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        strategy_summary = self.analyzer.get_summary_by_strategy(globally_common=True)
        ratio_summary = self.analyzer.get_summary_by_ratio(globally_common=True)
        
        # 1. mIoU vs Ratio (main plot)
        ax1 = fig.add_subplot(gs[0, :2])
        for strategy, ratios in strategy_summary.items():
            sorted_ratios = sorted(ratios.items())
            x = [r[0] for r in sorted_ratios]
            y = [r[1] for r in sorted_ratios]
            color = STRATEGY_COLORS.get(strategy, None)
            ax1.plot(x, y, 'o-', label=strategy, color=color, markersize=6, linewidth=2)
        ax1.set_xlabel('Ratio')
        ax1.set_ylabel('mIoU (%)')
        ax1.set_title('mIoU vs Ratio by Strategy (Globally Common)', fontweight='bold')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Average mIoU by ratio (bar chart)
        ax2 = fig.add_subplot(gs[0, 2])
        ratios = sorted(ratio_summary.keys())
        mious = [ratio_summary[r]['mIoU'] for r in ratios]
        colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(ratios)))
        ax2.bar([f"{r:.2f}" for r in ratios], mious, color=colors)
        ax2.set_xlabel('Ratio')
        ax2.set_ylabel('Avg mIoU (%)')
        ax2.set_title('Average mIoU by Ratio', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Heatmap
        ax3 = fig.add_subplot(gs[1, :2])
        strategies = sorted(strategy_summary.keys())
        all_ratios = sorted(set(r for s in strategy_summary.values() for r in s.keys()))
        matrix = np.zeros((len(strategies), len(all_ratios)))
        for i, strategy in enumerate(strategies):
            for j, ratio in enumerate(all_ratios):
                matrix[i, j] = strategy_summary[strategy].get(ratio, np.nan)
        sns.heatmap(matrix, xticklabels=[f"{r:.2f}" for r in all_ratios],
                   yticklabels=strategies, annot=True, fmt='.1f',
                   cmap='RdYlGn', ax=ax3, cbar_kws={'shrink': 0.8})
        ax3.set_title('mIoU Heatmap', fontweight='bold')
        
        # 4. Optimal ratio distribution
        ax4 = fig.add_subplot(gs[1, 2])
        optimal = self.analyzer.get_optimal_ratios()
        ratio_counts = defaultdict(int)
        for (strategy, dataset, model), (ratio, miou) in optimal.items():
            ratio_counts[ratio] += 1
        opt_ratios = sorted(ratio_counts.keys())
        counts = [ratio_counts[r] for r in opt_ratios]
        ax4.pie(counts, labels=[f"{r:.2f}" for r in opt_ratios], autopct='%1.1f%%',
               colors=plt.cm.Set3(np.linspace(0, 1, len(opt_ratios))))
        ax4.set_title('Optimal Ratio Distribution', fontweight='bold')
        
        # 5. Box plot
        ax5 = fig.add_subplot(gs[2, :])
        ratio_mious = defaultdict(list)
        for result in self.analyzer.results:
            ratio_mious[result.ratio].append(result.miou)
        ratios_sorted = sorted(ratio_mious.keys())
        data = [ratio_mious[r] for r in ratios_sorted]
        bp = ax5.boxplot(data, labels=[f"{r:.3f}" for r in ratios_sorted], patch_artist=True)
        colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(ratios_sorted)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax5.set_xlabel('Ratio')
        ax5.set_ylabel('mIoU (%)')
        ax5.set_title('mIoU Distribution by Ratio', fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)
        
        fig.suptitle('PROVE Ratio Ablation Study - Summary Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        self._save_figure(fig, 'ratio_ablation_dashboard')
    
    def generate_all_plots(self):
        """Generate all visualization plots."""
        print(f"\nGenerating ratio ablation visualizations...")
        print(f"Output directory: {self.output_dir}")
        print("-" * 50)
        
        self.plot_miou_vs_ratio_by_strategy()
        self.plot_heatmap_strategy_ratio()
        self.plot_optimal_ratio_distribution()
        self.plot_miou_boxplot_by_ratio()
        self.plot_miou_vs_ratio_by_dataset()
        self.plot_miou_vs_ratio_by_model()
        self.plot_relative_improvement()
        self.plot_summary_dashboard()
        
        print("-" * 50)
        print(f"All visualizations saved to: {self.output_dir}")


def main():
    if not HAS_PLOTTING:
        print("Error: Required plotting libraries not available.")
        print("Install with: pip install matplotlib seaborn pandas numpy")
        return 1
    
    parser = argparse.ArgumentParser(
        description='Visualize PROVE ratio ablation study results'
    )
    parser.add_argument('--weights-root', type=str, default=DEFAULT_WEIGHTS_ROOT,
                       help=f'Weights root directory (default: {DEFAULT_WEIGHTS_ROOT})')
    parser.add_argument('--regular-weights-root', type=str, default=DEFAULT_REGULAR_WEIGHTS_ROOT,
                       help=f'Regular weights root for baseline/0.5 results (default: {DEFAULT_REGULAR_WEIGHTS_ROOT})')
    parser.add_argument('--output-dir', type=str, default='./figures/ratio_ablation',
                       help='Output directory for figures')
    parser.add_argument('--plots', type=str, nargs='+', 
                       choices=['line', 'heatmap', 'bar', 'box', 'dataset', 'model', 'improvement', 'dashboard', 'all'],
                       default=['all'],
                       help='Plot types to generate')
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
    
    print(f"Found {count} results")
    
    # Create visualizer
    visualizer = RatioAblationVisualizer(analyzer, args.output_dir)
    
    # Generate plots
    if 'all' in args.plots:
        visualizer.generate_all_plots()
    else:
        if 'line' in args.plots:
            visualizer.plot_miou_vs_ratio_by_strategy()
        if 'heatmap' in args.plots:
            visualizer.plot_heatmap_strategy_ratio()
        if 'bar' in args.plots:
            visualizer.plot_optimal_ratio_distribution()
        if 'box' in args.plots:
            visualizer.plot_miou_boxplot_by_ratio()
        if 'dataset' in args.plots:
            visualizer.plot_miou_vs_ratio_by_dataset()
        if 'model' in args.plots:
            visualizer.plot_miou_vs_ratio_by_model()
        if 'improvement' in args.plots:
            visualizer.plot_relative_improvement()
        if 'dashboard' in args.plots:
            visualizer.plot_summary_dashboard()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
