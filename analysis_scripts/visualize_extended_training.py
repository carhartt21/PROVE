#!/usr/bin/env python3
"""
PROVE Extended Training Ablation Study Visualizer

Generates visualizations for the extended training ablation study results.

Features:
- Learning curves showing mIoU vs iterations
- Convergence analysis plots
- Improvement distribution histograms
- Strategy comparison across training lengths
- Dataset-specific training curves
- Model comparison at different checkpoints

Usage:
    # Generate all visualizations
    python visualize_extended_training.py

    # Specify custom weights root
    python visualize_extended_training.py --weights-root /path/to/weights

    # Generate specific plot types
    python visualize_extended_training.py --plots learning convergence improvement

    # Save to specific directory
    python visualize_extended_training.py --output-dir ./figures/extended_training
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_extended_training import ExtendedTrainingAnalyzer, DEFAULT_WEIGHTS_ROOT

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


# Color palette for strategies (extended to include all current strategies)
STRATEGY_COLORS = {
    # Baseline reference
    'baseline': '#000000',  # Black for baseline
    # Generative strategies (blues/purples)
    'gen_LANIT': '#1f77b4',
    'gen_step1x_new': '#ff7f0e',
    'gen_automold': '#2ca02c',
    'gen_TSIT': '#d62728',
    'gen_NST': '#9467bd',
    'gen_CNetSeg': '#8c564b',
    'gen_CUT': '#e377c2',
    'gen_cycleGAN': '#7f7f7f',
    'gen_ControlNetSeg': '#bcbd22',
    'gen_HRDA': '#17becf',
    'gen_cyclediffusion': '#636efa',  # Blue
    'gen_albumentations_weather': '#00cc96',  # Teal
    'gen_flux_kontext': '#ef553b',  # Red-orange
    'gen_UniControl': '#ab63fa',  # Purple
    'gen_IP2P': '#ffa15a',  # Orange
    'gen_stargan_v2': '#19d3f3',  # Cyan
    # Standard augmentation strategies
    'std_randaugment': '#aec7e8',
    'std_mixup': '#ffbb78',
    'std_cutout': '#98df8a',
    'std_randaugment+std_mixup': '#ff9896',
    'std_randaugment+std_cutout': '#c5b0d5',
    'std_photometric_distort': '#c49c94',  # Brown
}

# Color palette for datasets
DATASET_COLORS = {
    'acdc': '#e41a1c',
    'bdd10k': '#377eb8',
    'idd-aw': '#4daf4a',
    'iddaw': '#4daf4a',  # Also support lowercase without hyphen
    'mapillaryvistas': '#984ea3',
    'outside15k': '#ff7f00',
}

# Color palette for models
MODEL_COLORS = {
    'deeplabv3plus_r50': '#66c2a5',
    'pspnet_r50': '#fc8d62',
    'segformer_mit-b5': '#8da0cb',
}


class ExtendedTrainingVisualizer:
    """Visualizer for extended training ablation study results."""
    
    def __init__(self, analyzer: ExtendedTrainingAnalyzer, output_dir: str = './figures/extended_training'):
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
    
    def plot_learning_curves_by_strategy(self):
        """
        Line plot showing mIoU vs iteration for each strategy.
        Aggregated across all datasets and models.
        """
        strategy_summary = self.analyzer.get_summary_by_strategy()
        
        if not strategy_summary:
            print("No data for learning curves plot")
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for strategy, iterations in strategy_summary.items():
            sorted_iters = sorted(iterations.items())
            x = [r[0] for r in sorted_iters]
            y = [r[1] for r in sorted_iters]
            
            color = STRATEGY_COLORS.get(strategy, None)
            ax.plot(x, y, 'o-', label=strategy, color=color, markersize=8, linewidth=2)
        
        ax.set_xlabel('Training Iterations', fontsize=12)
        ax.set_ylabel('Mean IoU (%)', fontsize=12)
        ax.set_title('Learning Curves: mIoU vs Training Iterations by Strategy', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis for large numbers
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        self._save_figure(fig, 'learning_curves_by_strategy')
    
    def plot_improvement_histogram(self):
        """
        Histogram showing distribution of improvement from extended training.
        """
        convergence_list = self.analyzer.get_convergence_analysis()
        
        if not convergence_list:
            print("No data for improvement histogram")
            return
        
        improvements = [c.improvement for c in convergence_list]
        rel_improvements = [c.relative_improvement for c in convergence_list]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Absolute improvement
        ax1 = axes[0]
        n_bins = 20
        ax1.hist(improvements, bins=n_bins, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No improvement')
        ax1.axvline(x=np.mean(improvements), color='green', linestyle='-', linewidth=2, 
                   label=f'Mean: {np.mean(improvements):.2f}')
        ax1.set_xlabel('mIoU Improvement', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Distribution of Absolute Improvement', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        
        # Relative improvement
        ax2 = axes[1]
        ax2.hist(rel_improvements, bins=n_bins, color='coral', edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No improvement')
        ax2.axvline(x=np.mean(rel_improvements), color='green', linestyle='-', linewidth=2,
                   label=f'Mean: {np.mean(rel_improvements):.2f}%')
        ax2.set_xlabel('Relative Improvement (%)', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Distribution of Relative Improvement', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Impact of Extended Training', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_figure(fig, 'improvement_histogram')
    
    def plot_convergence_analysis(self):
        """
        Bar chart showing best iteration for each configuration.
        """
        convergence_list = self.analyzer.get_convergence_analysis()
        
        if not convergence_list:
            print("No data for convergence analysis plot")
            return
        
        # Count best iterations
        best_iter_counts = defaultdict(int)
        for c in convergence_list:
            best_iter_counts[c.best_iteration] += 1
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart of best iteration distribution
        ax1 = axes[0]
        iters = sorted(best_iter_counts.keys())
        counts = [best_iter_counts[i] for i in iters]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(iters)))
        
        bars = ax1.bar([format(i, ',') for i in iters], counts, color=colors, edgecolor='black')
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.annotate(f'{count}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
        
        ax1.set_xlabel('Best Iteration', fontsize=12)
        ax1.set_ylabel('Number of Configurations', fontsize=12)
        ax1.set_title('When Does Training Peak?', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Pie chart
        ax2 = axes[1]
        ax2.pie(counts, labels=[format(i, ',') for i in iters], autopct='%1.1f%%',
               colors=colors, explode=[0.05] * len(iters))
        ax2.set_title('Distribution of Optimal Training Length', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        self._save_figure(fig, 'convergence_analysis')
    
    def plot_improvement_by_strategy(self):
        """
        Bar chart showing average improvement per strategy.
        """
        convergence_list = self.analyzer.get_convergence_analysis()
        
        if not convergence_list:
            print("No data for improvement by strategy plot")
            return
        
        # Group by strategy
        strategy_improvements = defaultdict(list)
        for c in convergence_list:
            strategy_improvements[c.strategy].append(c.improvement)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        strategies = sorted(strategy_improvements.keys())
        means = [np.mean(strategy_improvements[s]) for s in strategies]
        stds = [np.std(strategy_improvements[s]) for s in strategies]
        
        colors = [STRATEGY_COLORS.get(s, 'gray') for s in strategies]
        
        bars = ax.bar(range(len(strategies)), means, yerr=stds,
                     color=colors, edgecolor='black', capsize=5, alpha=0.8)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=9)
        ax.set_xlabel('Strategy', fontsize=12)
        ax.set_ylabel('mIoU Improvement', fontsize=12)
        ax.set_title('Average Improvement from Extended Training by Strategy', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Highlight positive/negative
        for bar, mean in zip(bars, means):
            bar.set_color('green' if mean > 0 else 'red')
            bar.set_alpha(0.7)
        
        plt.tight_layout()
        self._save_figure(fig, 'improvement_by_strategy')
    
    def plot_learning_curves_by_dataset(self):
        """
        Learning curves separated by dataset.
        """
        if not self.analyzer.results:
            print("No data for dataset learning curves")
            return
        
        # Organize data by dataset
        dataset_data = defaultdict(lambda: defaultdict(list))
        for result in self.analyzer.results:
            dataset_data[result.dataset][result.iteration].append(result.miou)
        
        datasets = sorted(dataset_data.keys())
        n_datasets = len(datasets)
        
        if n_datasets == 0:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, dataset in enumerate(datasets):
            if i >= len(axes):
                break
            
            ax = axes[i]
            iter_data = dataset_data[dataset]
            
            iterations = sorted(iter_data.keys())
            means = [np.mean(iter_data[it]) for it in iterations]
            stds = [np.std(iter_data[it]) for it in iterations]
            
            color = DATASET_COLORS.get(dataset.lower(), 'blue')
            ax.errorbar(iterations, means, yerr=stds, fmt='o-', color=color,
                       capsize=4, markersize=6, linewidth=2)
            
            ax.set_xlabel('Iterations', fontsize=10)
            ax.set_ylabel('mIoU (%)', fontsize=10)
            ax.set_title(f'{dataset.upper()}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}k'))
        
        # Hide unused subplots
        for i in range(n_datasets, len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle('Learning Curves by Dataset', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure(fig, 'learning_curves_by_dataset')
    
    def plot_learning_curves_by_model(self):
        """
        Learning curves separated by model architecture.
        """
        if not self.analyzer.results:
            print("No data for model learning curves")
            return
        
        # Organize data by model
        model_data = defaultdict(lambda: defaultdict(list))
        for result in self.analyzer.results:
            model_data[result.model][result.iteration].append(result.miou)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for model, iter_data in sorted(model_data.items()):
            iterations = sorted(iter_data.keys())
            means = [np.mean(iter_data[it]) for it in iterations]
            
            color = MODEL_COLORS.get(model, None)
            ax.plot(iterations, means, 'o-', label=model, color=color, 
                   markersize=8, linewidth=2)
        
        ax.set_xlabel('Training Iterations', fontsize=12)
        ax.set_ylabel('Mean IoU (%)', fontsize=12)
        ax.set_title('Learning Curves by Model Architecture', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        self._save_figure(fig, 'learning_curves_by_model')
    
    def plot_heatmap_strategy_iteration(self):
        """
        Heatmap showing mIoU for each strategy-iteration combination.
        """
        strategy_summary = self.analyzer.get_summary_by_strategy()
        
        if not strategy_summary:
            print("No data for heatmap")
            return
        
        # Build matrix
        strategies = sorted(strategy_summary.keys())
        all_iters = sorted(set(it for s in strategy_summary.values() for it in s.keys()))
        
        matrix = np.zeros((len(strategies), len(all_iters)))
        for i, strategy in enumerate(strategies):
            for j, iteration in enumerate(all_iters):
                matrix[i, j] = strategy_summary[strategy].get(iteration, np.nan)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(matrix,
                   xticklabels=[f"{it//1000}k" for it in all_iters],
                   yticklabels=strategies,
                   annot=True,
                   fmt='.2f',
                   cmap='RdYlGn',
                   center=matrix[~np.isnan(matrix)].mean() if not np.all(np.isnan(matrix)) else 50,
                   ax=ax)
        
        ax.set_xlabel('Iterations', fontsize=12)
        ax.set_ylabel('Strategy', fontsize=12)
        ax.set_title('mIoU Heatmap: Strategy × Training Iterations', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        self._save_figure(fig, 'heatmap_strategy_iteration')
    
    def plot_diminishing_returns(self):
        """
        Line plot showing diminishing returns from extended training.
        Plots marginal improvement at each iteration increment.
        """
        strategy_summary = self.analyzer.get_summary_by_strategy()
        
        if not strategy_summary:
            print("No data for diminishing returns plot")
            return
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for strategy, iterations in strategy_summary.items():
            sorted_iters = sorted(iterations.items())
            if len(sorted_iters) < 2:
                continue
            
            # Calculate marginal improvements
            x = [sorted_iters[i][0] for i in range(1, len(sorted_iters))]
            y = [sorted_iters[i][1] - sorted_iters[i-1][1] for i in range(1, len(sorted_iters))]
            
            color = STRATEGY_COLORS.get(strategy, None)
            ax.plot(x, y, 'o--', label=strategy, color=color, markersize=6, linewidth=1.5, alpha=0.8)
        
        ax.axhline(y=0, color='red', linestyle='-', linewidth=1.5, label='No gain')
        ax.set_xlabel('Training Iterations', fontsize=12)
        ax.set_ylabel('Marginal mIoU Improvement', fontsize=12)
        ax.set_title('Diminishing Returns: Marginal Improvement per Training Step', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        self._save_figure(fig, 'diminishing_returns')
    
    def plot_summary_dashboard(self):
        """
        Create a comprehensive dashboard with multiple plots.
        """
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        strategy_summary = self.analyzer.get_summary_by_strategy()
        iter_summary = self.analyzer.get_summary_by_iteration()
        convergence_list = self.analyzer.get_convergence_analysis()
        stats = self.analyzer.get_improvement_stats()
        
        # 1. Learning curves (main plot)
        ax1 = fig.add_subplot(gs[0, :2])
        for strategy, iterations in strategy_summary.items():
            sorted_iters = sorted(iterations.items())
            x = [r[0] for r in sorted_iters]
            y = [r[1] for r in sorted_iters]
            color = STRATEGY_COLORS.get(strategy, None)
            ax1.plot(x, y, 'o-', label=strategy, color=color, markersize=5, linewidth=1.5)
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('mIoU (%)')
        ax1.set_title('Learning Curves by Strategy', fontweight='bold')
        ax1.legend(loc='best', fontsize=7, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}k'))
        
        # 2. Average by iteration (bar chart)
        ax2 = fig.add_subplot(gs[0, 2])
        iters = sorted(iter_summary.keys())
        mious = [iter_summary[i]['mIoU'] for i in iters]
        colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(iters)))
        ax2.bar([f"{i//1000}k" for i in iters], mious, color=colors)
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Avg mIoU (%)')
        ax2.set_title('Average mIoU by Iteration', fontweight='bold')
        
        # 3. Improvement histogram
        ax3 = fig.add_subplot(gs[1, 0])
        improvements = [c.improvement for c in convergence_list]
        ax3.hist(improvements, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax3.axvline(x=np.mean(improvements), color='green', linestyle='-', linewidth=2)
        ax3.set_xlabel('mIoU Improvement')
        ax3.set_ylabel('Count')
        ax3.set_title('Improvement Distribution', fontweight='bold')
        
        # 4. Best iteration distribution (pie)
        ax4 = fig.add_subplot(gs[1, 1])
        if stats.get('best_iter_distribution'):
            best_iters = stats['best_iter_distribution']
            ax4.pie(list(best_iters.values()), 
                   labels=[f"{k//1000}k" for k in best_iters.keys()],
                   autopct='%1.1f%%',
                   colors=plt.cm.Set3(np.linspace(0, 1, len(best_iters))))
            ax4.set_title('Optimal Training Length', fontweight='bold')
        
        # 5. Heatmap
        ax5 = fig.add_subplot(gs[1, 2])
        strategies = sorted(strategy_summary.keys())[:8]  # Top 8 for readability
        all_iters = sorted(set(it for s in strategy_summary.values() for it in s.keys()))
        matrix = np.zeros((len(strategies), len(all_iters)))
        for i, strategy in enumerate(strategies):
            for j, iteration in enumerate(all_iters):
                matrix[i, j] = strategy_summary[strategy].get(iteration, np.nan)
        sns.heatmap(matrix, xticklabels=[f"{it//1000}k" for it in all_iters],
                   yticklabels=strategies, annot=True, fmt='.1f',
                   cmap='RdYlGn', ax=ax5, cbar_kws={'shrink': 0.8})
        ax5.set_title('Strategy × Iteration Heatmap', fontweight='bold')
        
        # 6. Improvement by strategy
        ax6 = fig.add_subplot(gs[2, :])
        strategy_imps = defaultdict(list)
        for c in convergence_list:
            strategy_imps[c.strategy].append(c.improvement)
        strategies_sorted = sorted(strategy_imps.keys())
        means = [np.mean(strategy_imps[s]) for s in strategies_sorted]
        stds = [np.std(strategy_imps[s]) for s in strategies_sorted]
        colors = ['green' if m > 0 else 'red' for m in means]
        ax6.bar(range(len(strategies_sorted)), means, yerr=stds,
               color=colors, edgecolor='black', capsize=3, alpha=0.7)
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax6.set_xticks(range(len(strategies_sorted)))
        ax6.set_xticklabels(strategies_sorted, rotation=45, ha='right', fontsize=8)
        ax6.set_ylabel('mIoU Improvement')
        ax6.set_title('Improvement from Extended Training by Strategy', fontweight='bold')
        ax6.grid(axis='y', alpha=0.3)
        
        # Summary statistics text box
        stats_text = (
            f"Total Configs: {stats.get('total_configs', 'N/A')}\n"
            f"Improved: {stats.get('improved_configs', 'N/A')} ({stats.get('improved_percentage', 0):.1f}%)\n"
            f"Mean Improvement: {stats.get('mean_improvement', 0):+.2f} mIoU\n"
            f"Max Improvement: {stats.get('max_improvement', 0):+.2f} mIoU"
        )
        fig.text(0.02, 0.02, stats_text, fontsize=10, fontfamily='monospace',
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle('PROVE Extended Training Ablation Study - Summary Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        self._save_figure(fig, 'extended_training_dashboard')
    
    def plot_baseline_comparison(self):
        """
        Compare baseline performance against all strategies at each iteration.
        Shows delta from baseline to highlight strategy effectiveness.
        """
        strategy_summary = self.analyzer.get_summary_by_strategy()
        
        if not strategy_summary:
            print("No data for baseline comparison plot")
            return
        
        # Check if baseline exists
        if 'baseline' not in strategy_summary:
            print("No baseline data found for comparison")
            return
        
        baseline_iters = strategy_summary['baseline']
        other_strategies = {k: v for k, v in strategy_summary.items() if k != 'baseline'}
        
        if not other_strategies:
            print("No non-baseline strategies for comparison")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot 1: All strategies vs baseline
        ax1 = axes[0]
        baseline_sorted = sorted(baseline_iters.items())
        base_x = [r[0] for r in baseline_sorted]
        base_y = [r[1] for r in baseline_sorted]
        
        ax1.plot(base_x, base_y, 'o-', label='baseline', color='black', 
                markersize=10, linewidth=3, zorder=10)
        
        for strategy, iterations in other_strategies.items():
            sorted_iters = sorted(iterations.items())
            x = [r[0] for r in sorted_iters]
            y = [r[1] for r in sorted_iters]
            color = STRATEGY_COLORS.get(strategy, None)
            ax1.plot(x, y, 'o--', label=strategy, color=color, 
                    markersize=5, linewidth=1.5, alpha=0.7)
        
        ax1.set_xlabel('Training Iterations', fontsize=12)
        ax1.set_ylabel('Mean IoU (%)', fontsize=12)
        ax1.set_title('Baseline vs All Strategies', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}k'))
        
        # Plot 2: Delta from baseline
        ax2 = axes[1]
        for strategy, iterations in other_strategies.items():
            # Calculate delta from baseline at each iteration
            deltas = []
            iter_points = []
            for it, miou in sorted(iterations.items()):
                if it in baseline_iters:
                    delta = miou - baseline_iters[it]
                    deltas.append(delta)
                    iter_points.append(it)
            
            if deltas:
                color = STRATEGY_COLORS.get(strategy, None)
                ax2.plot(iter_points, deltas, 'o-', label=strategy, color=color,
                        markersize=6, linewidth=2, alpha=0.8)
        
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=2, label='Baseline')
        ax2.set_xlabel('Training Iterations', fontsize=12)
        ax2.set_ylabel('Δ mIoU vs Baseline (%)', fontsize=12)
        ax2.set_title('Strategy Advantage Over Baseline', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=8, ncol=2)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}k'))
        
        plt.tight_layout()
        self._save_figure(fig, 'baseline_comparison')
    
    def plot_strategy_ranking_by_iteration(self):
        """
        Bar chart showing strategy rankings at different iteration checkpoints.
        """
        strategy_summary = self.analyzer.get_summary_by_strategy()
        
        if not strategy_summary:
            print("No data for strategy ranking plot")
            return
        
        # Select key iterations to show
        all_iters = sorted(set(it for s in strategy_summary.values() for it in s.keys()))
        key_iters = [it for it in all_iters if it in [80000, 160000, 240000, 320000] or it == max(all_iters)]
        
        if len(key_iters) < 2:
            key_iters = all_iters[:4] if len(all_iters) >= 4 else all_iters
        
        n_plots = len(key_iters)
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 8))
        if n_plots == 1:
            axes = [axes]
        
        for ax, iteration in zip(axes, key_iters):
            # Get mIoU for each strategy at this iteration
            strategy_miou = {}
            for strategy, iters in strategy_summary.items():
                if iteration in iters:
                    strategy_miou[strategy] = iters[iteration]
            
            if not strategy_miou:
                continue
            
            # Sort by mIoU
            sorted_strategies = sorted(strategy_miou.items(), key=lambda x: x[1], reverse=True)
            strategies = [s[0] for s in sorted_strategies]
            mious = [s[1] for s in sorted_strategies]
            
            colors = [STRATEGY_COLORS.get(s, 'gray') for s in strategies]
            y_pos = range(len(strategies))
            
            bars = ax.barh(y_pos, mious, color=colors, edgecolor='black', alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(strategies, fontsize=9)
            ax.set_xlabel('mIoU (%)')
            ax.set_title(f'{iteration//1000}k Iterations', fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars, mious):
                ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, 
                       f'{val:.1f}', va='center', fontsize=8)
        
        fig.suptitle('Strategy Ranking at Different Training Lengths', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure(fig, 'strategy_ranking_by_iteration')
    
    def generate_all_plots(self):
        """Generate all visualization plots."""
        print(f"\nGenerating extended training visualizations...")
        print(f"Output directory: {self.output_dir}")
        print("-" * 50)
        
        self.plot_learning_curves_by_strategy()
        self.plot_improvement_histogram()
        self.plot_convergence_analysis()
        self.plot_improvement_by_strategy()
        self.plot_learning_curves_by_dataset()
        self.plot_learning_curves_by_model()
        self.plot_heatmap_strategy_iteration()
        self.plot_diminishing_returns()
        self.plot_baseline_comparison()  # New: baseline vs strategies
        self.plot_strategy_ranking_by_iteration()  # New: rankings at different iterations
        self.plot_summary_dashboard()
        
        print("-" * 50)
        print(f"All visualizations saved to: {self.output_dir}")


def main():
    if not HAS_PLOTTING:
        print("Error: Required plotting libraries not available.")
        print("Install with: pip install matplotlib seaborn pandas numpy")
        return 1
    
    parser = argparse.ArgumentParser(
        description='Visualize PROVE extended training ablation study results'
    )
    parser.add_argument('--weights-root', type=str, default=DEFAULT_WEIGHTS_ROOT,
                       help=f'Weights root directory (default: {DEFAULT_WEIGHTS_ROOT})')
    parser.add_argument('--output-dir', type=str, default='result_figures/extended_training',
                       help='Output directory for figures (default: result_figures/extended_training)')
    parser.add_argument('--plots', type=str, nargs='+',
                       choices=['learning', 'improvement', 'convergence', 'strategy', 
                               'dataset', 'model', 'heatmap', 'diminishing', 'dashboard', 
                               'baseline', 'ranking', 'all'],
                       default=['all'],
                       help='Plot types to generate')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output during scanning')
    
    args = parser.parse_args()
    
    # Create analyzer and scan results
    analyzer = ExtendedTrainingAnalyzer(args.weights_root)
    count = analyzer.scan_results(verbose=args.verbose)
    
    if count == 0:
        print("No results found.")
        return 1
    
    print(f"Found {count} results")
    
    # Create visualizer
    visualizer = ExtendedTrainingVisualizer(analyzer, args.output_dir)
    
    # Generate plots
    if 'all' in args.plots:
        visualizer.generate_all_plots()
    else:
        if 'learning' in args.plots:
            visualizer.plot_learning_curves_by_strategy()
        if 'improvement' in args.plots:
            visualizer.plot_improvement_histogram()
        if 'convergence' in args.plots:
            visualizer.plot_convergence_analysis()
        if 'strategy' in args.plots:
            visualizer.plot_improvement_by_strategy()
        if 'dataset' in args.plots:
            visualizer.plot_learning_curves_by_dataset()
        if 'model' in args.plots:
            visualizer.plot_learning_curves_by_model()
        if 'heatmap' in args.plots:
            visualizer.plot_heatmap_strategy_iteration()
        if 'diminishing' in args.plots:
            visualizer.plot_diminishing_returns()
        if 'baseline' in args.plots:
            visualizer.plot_baseline_comparison()
        if 'ranking' in args.plots:
            visualizer.plot_strategy_ranking_by_iteration()
        if 'dashboard' in args.plots:
            visualizer.plot_summary_dashboard()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
