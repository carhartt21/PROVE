#!/usr/bin/env python3
"""
PROVE Extended Training Study - IEEE Publication Figure Generator

Generates publication-ready figures for the extended training ablation study.
All figures follow IEEE conference/journal formatting guidelines:
- Single-column: 3.5 inches (8.89 cm) width
- Double-column: 7.16 inches (18.19 cm) width  
- Font sizes appropriate for readability at these sizes
- Vector output (PDF) for quality, PNG for preview

Output Structure:
    result_figures/extended_training/
    ├── ieee/                           # Publication-ready figures
    │   ├── fig_learning_curves.pdf
    │   ├── fig_convergence.pdf
    │   ├── fig_improvement_heatmap.pdf
    │   ├── fig_best_iteration_distribution.pdf
    │   ├── fig_strategy_comparison.pdf
    │   └── fig_model_comparison.pdf
    ├── preview/                        # PNG previews
    │   └── *.png
    └── data/                           # Underlying data
        ├── extended_training_results.csv
        └── extended_training_summary.json

Usage:
    # Generate all IEEE figures
    python generate_ieee_figures_extended_training.py

    # Specify custom weights root
    python generate_ieee_figures_extended_training.py --weights-root /path/to/weights

    # Generate specific figure
    python generate_ieee_figures_extended_training.py --figures learning convergence
"""

import os
import sys
import json
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
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import matplotlib.ticker as ticker
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError as e:
    HAS_PLOTTING = False
    print(f"Warning: Required plotting libraries not installed: {e}")
    print("Install with: pip install matplotlib seaborn pandas numpy")
    sys.exit(1)


# ============================================================================
# IEEE Style Configuration
# ============================================================================

# Figure sizes (inches)
SINGLE_COL_WIDTH = 3.5
DOUBLE_COL_WIDTH = 7.16

# Font sizes for IEEE publications
FONT_SIZE_TITLE = 10
FONT_SIZE_LABEL = 9
FONT_SIZE_TICK = 8
FONT_SIZE_LEGEND = 8

# Color palettes designed for accessibility and B&W printing
STRATEGY_COLORS = {
    'gen_albumentations_weather': '#1f77b4',
    'gen_automold': '#ff7f0e',
    'gen_cyclediffusion': '#2ca02c',
    'gen_cycleGAN': '#d62728',
    'gen_flux_kontext': '#9467bd',
    'gen_step1x_new': '#8c564b',
    'gen_TSIT': '#e377c2',
    'gen_UniControl': '#7f7f7f',
    'std_randaugment': '#bcbd22',
    'baseline': '#17becf',
}

STRATEGY_MARKERS = {
    'gen_albumentations_weather': 'o',
    'gen_automold': 's',
    'gen_cyclediffusion': '^',
    'gen_cycleGAN': 'v',
    'gen_flux_kontext': 'D',
    'gen_step1x_new': 'p',
    'gen_TSIT': 'h',
    'gen_UniControl': '*',
    'std_randaugment': 'X',
    'baseline': '+',
}

DATASET_COLORS = {
    'bdd10k': '#377eb8',
    'bdd10k_ad': '#377eb8',
    'idd-aw': '#4daf4a',
    'iddaw_ad': '#4daf4a',
    'mapillaryvistas': '#984ea3',
    'mapillaryvistas_ad': '#984ea3',
    'outside15k': '#ff7f00',
    'outside15k_ad': '#ff7f00',
}

MODEL_COLORS = {
    'pspnet_r50_ratio0p50': '#66c2a5',
    'segformer_mit-b5_ratio0p50': '#fc8d62',
}

# Clean names for publication
CLEAN_STRATEGY_NAMES = {
    'gen_albumentations_weather': 'Albumentations',
    'gen_automold': 'AutoMold',
    'gen_cyclediffusion': 'CycleDiff',
    'gen_cycleGAN': 'CycleGAN',
    'gen_flux_kontext': 'Flux Kontext',
    'gen_step1x_new': 'Step1X',
    'gen_TSIT': 'TSIT',
    'gen_UniControl': 'UniControl',
    'std_randaugment': 'RandAugment',
    'baseline': 'Baseline',
}

CLEAN_DATASET_NAMES = {
    'bdd10k': 'BDD10K',
    'bdd10k_ad': 'BDD10K',
    'idd-aw': 'IDD-AW',
    'iddaw_ad': 'IDD-AW',
    'mapillaryvistas': 'Mapillary',
    'mapillaryvistas_ad': 'Mapillary',
    'outside15k': 'OUTSIDE15K',
    'outside15k_ad': 'OUTSIDE15K',
}

CLEAN_MODEL_NAMES = {
    'pspnet_r50_ratio0p50': 'PSPNet-R50',
    'segformer_mit-b5_ratio0p50': 'SegFormer-B5',
}


def setup_ieee_style():
    """Configure matplotlib for IEEE publication style."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': FONT_SIZE_TICK,
        'axes.titlesize': FONT_SIZE_TITLE,
        'axes.labelsize': FONT_SIZE_LABEL,
        'xtick.labelsize': FONT_SIZE_TICK,
        'ytick.labelsize': FONT_SIZE_TICK,
        'legend.fontsize': FONT_SIZE_LEGEND,
        'figure.titlesize': FONT_SIZE_TITLE,
        'axes.linewidth': 0.5,
        'grid.linewidth': 0.3,
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })


class IEEEFigureGenerator:
    """Generator for IEEE publication-ready figures."""
    
    def __init__(self, analyzer: ExtendedTrainingAnalyzer, output_dir: str):
        self.analyzer = analyzer
        self.output_dir = Path(output_dir)
        self.ieee_dir = self.output_dir / 'ieee'
        self.preview_dir = self.output_dir / 'preview'
        self.data_dir = self.output_dir / 'data'
        
        # Create directories
        for d in [self.ieee_dir, self.preview_dir, self.data_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        setup_ieee_style()
    
    def _save_figure(self, fig: plt.Figure, name: str):
        """Save figure in both PDF and PNG formats."""
        # PDF for publication
        pdf_path = self.ieee_dir / f'{name}.pdf'
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"  Saved: {pdf_path}")
        
        # PNG for preview
        png_path = self.preview_dir / f'{name}.png'
        fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
        
        plt.close(fig)
    
    def export_data(self):
        """Export underlying data to CSV and JSON."""
        # CSV with all results
        csv_path = self.data_dir / 'extended_training_results.csv'
        self.analyzer.export_csv(str(csv_path))
        
        # JSON with summary
        json_path = self.data_dir / 'extended_training_summary.json'
        self.analyzer.export_json(str(json_path))
    
    def fig_learning_curves(self):
        """
        Learning curves showing mIoU vs iterations for each strategy.
        Double-column figure with line plots.
        """
        print("Generating: Learning curves...")
        
        strategy_summary = self.analyzer.get_summary_by_strategy()
        if not strategy_summary:
            print("  No data available")
            return
        
        fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 3.5))
        
        for strategy in sorted(strategy_summary.keys()):
            iterations = strategy_summary[strategy]
            sorted_iters = sorted(iterations.items())
            x = np.array([i[0] for i in sorted_iters]) / 1000  # Convert to K
            y = np.array([i[1] for i in sorted_iters])
            
            color = STRATEGY_COLORS.get(strategy, '#333333')
            marker = STRATEGY_MARKERS.get(strategy, 'o')
            label = CLEAN_STRATEGY_NAMES.get(strategy, strategy)
            
            ax.plot(x, y, marker=marker, label=label, color=color,
                   markersize=3, linewidth=1.0, markeredgewidth=0.5)
        
        ax.set_xlabel('Training Iterations (×1000)')
        ax.set_ylabel('Mean IoU (%)')
        ax.set_title('Extended Training Learning Curves')
        
        # Format x-axis
        ax.xaxis.set_major_locator(ticker.MultipleLocator(40))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(20))
        
        # Legend outside plot
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                 ncol=1, frameon=True, fontsize=FONT_SIZE_LEGEND)
        
        plt.tight_layout()
        self._save_figure(fig, 'fig_learning_curves')
    
    def fig_convergence_heatmap(self):
        """
        Heatmap showing mIoU for each strategy at each iteration.
        Double-column figure.
        """
        print("Generating: Convergence heatmap...")
        
        strategy_summary = self.analyzer.get_summary_by_strategy()
        if not strategy_summary:
            print("  No data available")
            return
        
        # Collect all iterations
        all_iterations = set()
        for iters in strategy_summary.values():
            all_iterations.update(iters.keys())
        all_iterations = sorted(all_iterations)
        
        # Build matrix
        strategies = sorted(strategy_summary.keys())
        data = np.full((len(strategies), len(all_iterations)), np.nan)
        
        for i, strategy in enumerate(strategies):
            for j, iteration in enumerate(all_iterations):
                if iteration in strategy_summary[strategy]:
                    data[i, j] = strategy_summary[strategy][iteration]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 4))
        
        # Clean names
        clean_strategies = [CLEAN_STRATEGY_NAMES.get(s, s) for s in strategies]
        iter_labels = [f'{int(i/1000)}K' for i in all_iterations]
        
        # Heatmap
        im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=40, vmax=55)
        
        # Ticks
        ax.set_xticks(np.arange(len(all_iterations)))
        ax.set_yticks(np.arange(len(strategies)))
        ax.set_xticklabels(iter_labels, rotation=45, ha='right')
        ax.set_yticklabels(clean_strategies)
        
        ax.set_xlabel('Training Iterations')
        ax.set_ylabel('Strategy')
        ax.set_title('mIoU (%) vs Training Duration')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('mIoU (%)', fontsize=FONT_SIZE_LABEL)
        
        # Annotate cells
        for i in range(len(strategies)):
            for j in range(len(all_iterations)):
                if not np.isnan(data[i, j]):
                    text_color = 'white' if data[i, j] < 47 else 'black'
                    ax.text(j, i, f'{data[i, j]:.1f}', ha='center', va='center',
                           fontsize=6, color=text_color)
        
        plt.tight_layout()
        self._save_figure(fig, 'fig_convergence_heatmap')
    
    def fig_improvement_distribution(self):
        """
        Distribution of improvement from extended training.
        Single-column figure with histogram.
        """
        print("Generating: Improvement distribution...")
        
        convergence_list = self.analyzer.get_convergence_analysis()
        if not convergence_list:
            print("  No data available")
            return
        
        improvements = [c.improvement for c in convergence_list]
        
        fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 2.5))
        
        # Histogram
        n, bins, patches = ax.hist(improvements, bins=15, color='steelblue',
                                   edgecolor='black', linewidth=0.5, alpha=0.7)
        
        # Mean line
        mean_imp = np.mean(improvements)
        ax.axvline(mean_imp, color='red', linestyle='--', linewidth=1.5,
                  label=f'Mean: {mean_imp:.2f}')
        
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
        
        ax.set_xlabel('mIoU Improvement (pp)')
        ax.set_ylabel('Count')
        ax.set_title('Improvement from Extended Training')
        ax.legend(fontsize=FONT_SIZE_LEGEND)
        
        plt.tight_layout()
        self._save_figure(fig, 'fig_improvement_distribution')
    
    def fig_best_iteration_distribution(self):
        """
        Bar chart showing at which iteration each configuration achieves best performance.
        Single-column figure.
        """
        print("Generating: Best iteration distribution...")
        
        convergence_list = self.analyzer.get_convergence_analysis()
        if not convergence_list:
            print("  No data available")
            return
        
        # Count best iterations
        best_iter_counts = defaultdict(int)
        for c in convergence_list:
            best_iter_counts[c.best_iteration] += 1
        
        fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 2.5))
        
        iterations = sorted(best_iter_counts.keys())
        counts = [best_iter_counts[i] for i in iterations]
        iter_labels = [f'{int(i/1000)}K' for i in iterations]
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(iterations)))
        
        bars = ax.bar(range(len(iterations)), counts, color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_xticks(range(len(iterations)))
        ax.set_xticklabels(iter_labels, rotation=45, ha='right')
        ax.set_xlabel('Best Iteration')
        ax.set_ylabel('Number of Configurations')
        ax.set_title('Optimal Training Duration')
        
        # Annotate bars
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.annotate(str(count), xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 2), textcoords='offset points',
                           ha='center', va='bottom', fontsize=FONT_SIZE_TICK)
        
        plt.tight_layout()
        self._save_figure(fig, 'fig_best_iteration')
    
    def fig_strategy_comparison(self):
        """
        Bar chart comparing average improvement per strategy.
        Double-column figure.
        """
        print("Generating: Strategy comparison...")
        
        convergence_list = self.analyzer.get_convergence_analysis()
        if not convergence_list:
            print("  No data available")
            return
        
        # Group by strategy
        strategy_improvements = defaultdict(list)
        for c in convergence_list:
            strategy_improvements[c.strategy].append(c.improvement)
        
        fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 3))
        
        strategies = sorted(strategy_improvements.keys())
        means = [np.mean(strategy_improvements[s]) for s in strategies]
        stds = [np.std(strategy_improvements[s]) for s in strategies]
        clean_names = [CLEAN_STRATEGY_NAMES.get(s, s) for s in strategies]
        colors = [STRATEGY_COLORS.get(s, '#333333') for s in strategies]
        
        x = np.arange(len(strategies))
        bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor='black',
                     linewidth=0.5, capsize=3, error_kw={'linewidth': 0.5})
        
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(clean_names, rotation=45, ha='right')
        ax.set_xlabel('Strategy')
        ax.set_ylabel('mIoU Improvement (pp)')
        ax.set_title('Extended Training Improvement by Strategy')
        
        plt.tight_layout()
        self._save_figure(fig, 'fig_strategy_comparison')
    
    def fig_model_dataset_comparison(self):
        """
        Grouped bar chart comparing improvement across models and datasets.
        Double-column figure.
        """
        print("Generating: Model/Dataset comparison...")
        
        convergence_list = self.analyzer.get_convergence_analysis()
        if not convergence_list:
            print("  No data available")
            return
        
        # Group by dataset and model
        data = defaultdict(lambda: defaultdict(list))
        for c in convergence_list:
            data[c.dataset][c.model].append(c.improvement)
        
        if not data:
            print("  No data available")
            return
        
        fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 3))
        
        datasets = sorted(data.keys())
        models = sorted(set(m for d in data.values() for m in d.keys()))
        
        x = np.arange(len(datasets))
        width = 0.35
        
        for i, model in enumerate(models):
            means = []
            stds = []
            for dataset in datasets:
                if model in data[dataset] and data[dataset][model]:
                    means.append(np.mean(data[dataset][model]))
                    stds.append(np.std(data[dataset][model]))
                else:
                    means.append(0)
                    stds.append(0)
            
            color = MODEL_COLORS.get(model, '#333333')
            label = CLEAN_MODEL_NAMES.get(model, model)
            offset = (i - len(models)/2 + 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds, label=label, color=color,
                  edgecolor='black', linewidth=0.5, capsize=2, error_kw={'linewidth': 0.5})
        
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        
        clean_datasets = [CLEAN_DATASET_NAMES.get(d, d) for d in datasets]
        ax.set_xticks(x)
        ax.set_xticklabels(clean_datasets)
        ax.set_xlabel('Dataset')
        ax.set_ylabel('mIoU Improvement (pp)')
        ax.set_title('Extended Training Improvement by Dataset and Model')
        ax.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
        
        plt.tight_layout()
        self._save_figure(fig, 'fig_model_dataset_comparison')
    
    def fig_diminishing_returns(self):
        """
        Line plot showing marginal improvement at each iteration checkpoint.
        Double-column figure.
        """
        print("Generating: Diminishing returns analysis...")
        
        iteration_summary = self.analyzer.get_summary_by_iteration()
        if len(iteration_summary) < 2:
            print("  Not enough data points")
            return
        
        iterations = sorted(iteration_summary.keys())
        miou_values = [iteration_summary[i]['mIoU'] for i in iterations]
        
        # Calculate marginal improvement
        marginal = [0]
        for i in range(1, len(miou_values)):
            marginal.append(miou_values[i] - miou_values[i-1])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 2.5))
        
        x = np.array(iterations) / 1000
        
        # Absolute mIoU
        ax1.plot(x, miou_values, 'o-', color='steelblue', markersize=4, linewidth=1.0)
        ax1.set_xlabel('Iterations (×1000)')
        ax1.set_ylabel('Mean IoU (%)')
        ax1.set_title('(a) Absolute Performance')
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(40))
        
        # Marginal improvement
        colors = ['green' if m > 0 else 'red' for m in marginal]
        ax2.bar(x, marginal, width=8, color=colors, edgecolor='black', linewidth=0.3)
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Iterations (×1000)')
        ax2.set_ylabel('Marginal Improvement (pp)')
        ax2.set_title('(b) Marginal Gains')
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(40))
        
        plt.tight_layout()
        self._save_figure(fig, 'fig_diminishing_returns')
    
    def generate_all(self):
        """Generate all IEEE figures."""
        print("\n" + "="*60)
        print("Generating IEEE Publication Figures")
        print("="*60)
        
        # Export data first
        print("\nExporting data...")
        self.export_data()
        
        # Generate figures
        print("\nGenerating figures...")
        self.fig_learning_curves()
        self.fig_convergence_heatmap()
        self.fig_improvement_distribution()
        self.fig_best_iteration_distribution()
        self.fig_strategy_comparison()
        self.fig_model_dataset_comparison()
        self.fig_diminishing_returns()
        
        print("\n" + "="*60)
        print(f"All figures saved to: {self.output_dir}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Generate IEEE publication figures for PROVE extended training study'
    )
    parser.add_argument('--weights-root', type=str, default=DEFAULT_WEIGHTS_ROOT,
                       help=f'Weights root directory (default: {DEFAULT_WEIGHTS_ROOT})')
    parser.add_argument('--output-dir', type=str, default='result_figures/extended_training',
                       help='Output directory for figures (default: result_figures/extended_training)')
    parser.add_argument('--figures', type=str, nargs='+',
                       choices=['learning', 'heatmap', 'improvement', 'best_iter', 
                               'strategy', 'model_dataset', 'diminishing', 'all'],
                       default=['all'],
                       help='Which figures to generate (default: all)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output during scanning')
    
    args = parser.parse_args()
    
    # Create analyzer and scan
    print(f"Scanning results in: {args.weights_root}")
    analyzer = ExtendedTrainingAnalyzer(args.weights_root)
    count = analyzer.scan_results(verbose=args.verbose)
    
    if count == 0:
        print("No results found.")
        return 1
    
    print(f"Found {count} results")
    
    # Create figure generator
    generator = IEEEFigureGenerator(analyzer, args.output_dir)
    
    # Generate figures
    if 'all' in args.figures:
        generator.generate_all()
    else:
        generator.export_data()
        if 'learning' in args.figures:
            generator.fig_learning_curves()
        if 'heatmap' in args.figures:
            generator.fig_convergence_heatmap()
        if 'improvement' in args.figures:
            generator.fig_improvement_distribution()
        if 'best_iter' in args.figures:
            generator.fig_best_iteration_distribution()
        if 'strategy' in args.figures:
            generator.fig_strategy_comparison()
        if 'model_dataset' in args.figures:
            generator.fig_model_dataset_comparison()
        if 'diminishing' in args.figures:
            generator.fig_diminishing_returns()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
