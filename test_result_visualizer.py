#!/usr/bin/env python3
"""
PROVE Test Result Visualizer

Generates diagrams and figures from test results using matplotlib and seaborn.

Features:
- Per-domain bar charts (mIoU, fwIoU by weather condition)
- Per-class horizontal bar charts (IoU by semantic class)
- Radar plots for cross-domain comparison
- Heatmaps for per-domain per-class breakdown
- Summary dashboard with multiple subplots
- Model/strategy comparison plots
- Class frequency vs performance correlation
- Domain performance gap analysis
- Box plot distributions
- Training curve visualization
- CSV export for further analysis

Usage:
    # Visualize single test result
    python test_result_visualizer.py --results-dir /path/to/test_results_detailed/timestamp/

    # Compare multiple models
    python test_result_visualizer.py --compare --results-dirs dir1 dir2 dir3

    # Generate specific plot types
    python test_result_visualizer.py --results-dir /path/to/results --plots domain class radar

    # Export to specific format
    python test_result_visualizer.py --results-dir /path/to/results --format pdf
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not installed. Install with: pip install matplotlib seaborn")


# Color palettes
DOMAIN_COLORS = {
    'clear_day': '#FFD700',      # Gold
    'cloudy': '#A9A9A9',         # Dark Gray
    'dawn_dusk': '#FF8C00',      # Dark Orange
    'foggy': '#B0C4DE',          # Light Steel Blue
    'night': '#191970',          # Midnight Blue
    'rainy': '#4682B4',          # Steel Blue
    'snowy': '#F0F8FF',          # Alice Blue
    'clear': '#87CEEB',          # Sky Blue
    'overcast': '#778899',       # Light Slate Gray
    'partly_cloudy': '#ADD8E6',  # Light Blue
}

CLASS_COLORS = sns.color_palette("husl", 19) if HAS_PLOTTING else None

# Cityscapes class names
CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]


class TestResultVisualizer:
    """Visualization engine for PROVE test results."""
    
    def __init__(
        self,
        results_dir: str,
        output_dir: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 150,
        style: str = 'seaborn-v0_8-whitegrid',
    ):
        """
        Initialize visualizer.
        
        Args:
            results_dir: Path to test results directory (timestamped folder)
            output_dir: Output directory for figures (default: results_dir/figures)
            figsize: Default figure size
            dpi: Figure resolution
            style: Matplotlib style
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir) if output_dir else self.results_dir / 'figures'
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.data = self._load_results()
        
        # Set style
        if HAS_PLOTTING:
            try:
                plt.style.use(style)
            except:
                plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
            sns.set_palette("husl")
    
    def _load_results(self) -> Dict[str, Any]:
        """Load all result files from the results directory."""
        data = {}
        
        # Load summary
        summary_path = self.results_dir / 'metrics_summary.json'
        if summary_path.exists():
            with open(summary_path) as f:
                data['summary'] = json.load(f)
        
        # Load per-domain metrics
        domain_path = self.results_dir / 'metrics_per_domain.json'
        if domain_path.exists():
            with open(domain_path) as f:
                data['per_domain'] = json.load(f)
        
        # Load per-class metrics
        class_path = self.results_dir / 'metrics_per_class.json'
        if class_path.exists():
            with open(class_path) as f:
                data['per_class'] = json.load(f)
        
        # Load full metrics
        full_path = self.results_dir / 'metrics_full.json'
        if full_path.exists():
            with open(full_path) as f:
                data['full'] = json.load(f)
        
        # Load CSV files for easier DataFrame access
        domain_csv = self.results_dir / 'per_domain_metrics.csv'
        if domain_csv.exists():
            data['domain_df'] = pd.read_csv(domain_csv)
        
        class_csv = self.results_dir / 'per_class_metrics.csv'
        if class_csv.exists():
            data['class_df'] = pd.read_csv(class_csv)
        
        return data
    
    def plot_domain_metrics(
        self,
        metrics: List[str] = ['mIoU', 'fwIoU'],
        save: bool = True,
        show: bool = False,
    ) -> Optional[plt.Figure]:
        """
        Create bar chart of metrics by domain.
        
        Args:
            metrics: Metrics to plot
            save: Whether to save the figure
            show: Whether to display the figure
        """
        if 'per_domain' not in self.data:
            print("No per-domain data available")
            return None
        
        per_domain = self.data['per_domain'].get('per_domain', {})
        if not per_domain:
            print("Empty per-domain data")
            return None
        
        # Prepare data
        domains = list(per_domain.keys())
        n_domains = len(domains)
        n_metrics = len(metrics)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x = np.arange(n_domains)
        width = 0.8 / n_metrics
        
        for i, metric in enumerate(metrics):
            values = [per_domain[d].get(metric, 0) for d in domains]
            offset = (i - n_metrics/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=metric, alpha=0.8)
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.annotate(f'{val:.1f}',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3),
                           textcoords='offset points',
                           ha='center', va='bottom', fontsize=8)
        
        # Customize plot
        ax.set_xlabel('Domain', fontsize=12)
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_title('Performance by Weather Domain', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(domains, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        # Add config info
        config = self.data.get('per_domain', {}).get('config', {})
        if config:
            info_text = f"Dataset: {config.get('dataset', 'N/A')}"
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'domain_metrics.png'
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_class_metrics(
        self,
        metric: str = 'IoU',
        sort_by: str = 'value',
        save: bool = True,
        show: bool = False,
    ) -> Optional[plt.Figure]:
        """
        Create horizontal bar chart of per-class metrics.
        
        Args:
            metric: Metric to plot ('IoU' or 'Acc')
            sort_by: Sort order ('value', 'name', or 'none')
            save: Whether to save the figure
            show: Whether to display the figure
        """
        if 'per_class' not in self.data:
            print("No per-class data available")
            return None
        
        per_class = self.data['per_class'].get('per_class', {})
        if not per_class:
            print("Empty per-class data")
            return None
        
        # Prepare data
        classes = list(per_class.keys())
        values = [per_class[c].get(metric, 0) for c in classes]
        
        # Sort if requested
        if sort_by == 'value':
            sorted_pairs = sorted(zip(classes, values), key=lambda x: x[1], reverse=True)
            classes, values = zip(*sorted_pairs)
        elif sort_by == 'name':
            sorted_pairs = sorted(zip(classes, values), key=lambda x: x[0])
            classes, values = zip(*sorted_pairs)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(self.figsize[0], max(8, len(classes) * 0.4)))
        
        # Color bars by value
        colors = plt.cm.RdYlGn(np.array(values) / 100)
        
        y = np.arange(len(classes))
        bars = ax.barh(y, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.1f}%',
                       xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                       xytext=(5, 0),
                       textcoords='offset points',
                       ha='left', va='center', fontsize=9)
        
        # Customize plot
        ax.set_xlabel(f'{metric} (%)', fontsize=12)
        ax.set_ylabel('Class', fontsize=12)
        ax.set_title(f'Per-Class {metric}', fontsize=14, fontweight='bold')
        ax.set_yticks(y)
        ax.set_yticklabels(classes)
        ax.set_xlim(0, 105)
        ax.grid(axis='x', alpha=0.3)
        
        # Add overall metric
        overall = self.data['per_class'].get('overall', {})
        if overall:
            mean_val = overall.get(f'm{metric}', np.mean(values))
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}%')
            ax.legend(loc='lower right')
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f'class_{metric.lower()}.png'
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_radar_chart(
        self,
        metrics: List[str] = ['mIoU', 'fwIoU', 'mAcc', 'aAcc'],
        save: bool = True,
        show: bool = False,
    ) -> Optional[plt.Figure]:
        """
        Create radar chart comparing domains across metrics.
        
        Args:
            metrics: Metrics to include in radar
            save: Whether to save the figure
            show: Whether to display the figure
        """
        if 'per_domain' not in self.data:
            print("No per-domain data available")
            return None
        
        per_domain = self.data['per_domain'].get('per_domain', {})
        if not per_domain:
            print("Empty per-domain data")
            return None
        
        domains = list(per_domain.keys())
        n_metrics = len(metrics)
        
        # Create angles for radar chart
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Plot each domain
        for domain in domains:
            values = [per_domain[domain].get(m, 0) for m in metrics]
            values += values[:1]  # Complete the circle
            
            color = DOMAIN_COLORS.get(domain, None)
            ax.plot(angles, values, 'o-', linewidth=2, label=domain, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        # Customize plot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylim(0, 100)
        ax.set_title('Domain Comparison Radar Chart', fontsize=14, fontweight='bold', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'domain_radar.png'
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_heatmap(
        self,
        metric: str = 'IoU',
        save: bool = True,
        show: bool = False,
    ) -> Optional[plt.Figure]:
        """
        Create heatmap of per-domain per-class metrics.
        
        Args:
            metric: Metric to visualize
            save: Whether to save the figure
            show: Whether to display the figure
        """
        if 'full' not in self.data:
            print("No full metrics data available (per-domain per-class)")
            return None
        
        full_data = self.data['full']
        per_domain_class = full_data.get('per_domain_per_class', {})
        
        if not per_domain_class:
            print("No per-domain per-class data available")
            return None
        
        # Build matrix
        domains = list(per_domain_class.keys())
        classes = list(per_domain_class[domains[0]].keys()) if domains else []
        
        if not classes:
            print("No class data in per-domain breakdown")
            return None
        
        matrix = np.zeros((len(domains), len(classes)))
        for i, domain in enumerate(domains):
            for j, cls in enumerate(classes):
                matrix[i, j] = per_domain_class[domain][cls].get(metric, 0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(12, len(classes) * 0.6), max(6, len(domains) * 0.8)))
        
        # Create heatmap
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label(f'{metric} (%)', fontsize=11)
        
        # Set ticks
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(domains)))
        ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(domains, fontsize=10)
        
        # Add value annotations
        for i in range(len(domains)):
            for j in range(len(classes)):
                val = matrix[i, j]
                color = 'white' if val < 50 else 'black'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center', 
                       color=color, fontsize=7)
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Domain', fontsize=12)
        ax.set_title(f'Per-Domain Per-Class {metric} Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f'heatmap_{metric.lower()}.png'
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_class_frequency_performance(
        self,
        save: bool = True,
        show: bool = False,
    ) -> Optional[plt.Figure]:
        """
        Create scatter plot showing correlation between class frequency and performance.
        
        This helps identify if rare classes have lower performance.
        
        Args:
            save: Whether to save the figure
            show: Whether to display the figure
        """
        if 'per_class' not in self.data:
            print("No per-class data available")
            return None
        
        per_class = self.data['per_class'].get('per_class', {})
        if not per_class:
            print("Empty per-class data")
            return None
        
        # Extract data
        classes = []
        iou_values = []
        frequencies = []  # Use pixel count or acc as proxy for frequency
        
        for cls_name, cls_data in per_class.items():
            classes.append(cls_name)
            iou_values.append(cls_data.get('IoU', 0))
            # Use Acc as proxy for frequency impact (higher acc often means more samples)
            frequencies.append(cls_data.get('Acc', 0))
        
        if not classes:
            print("No class data")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Scatter plot with size based on IoU
        scatter = ax.scatter(frequencies, iou_values, 
                            s=np.array(iou_values) * 3 + 50,  # Size based on IoU
                            c=iou_values, cmap='RdYlGn', 
                            alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add class labels
        for i, cls in enumerate(classes):
            ax.annotate(cls, (frequencies[i], iou_values[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8)
        
        # Add trend line
        if len(frequencies) > 2:
            z = np.polyfit(frequencies, iou_values, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(frequencies), max(frequencies), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        
        # Calculate correlation
        correlation = np.corrcoef(frequencies, iou_values)[0, 1]
        
        # Customize plot
        ax.set_xlabel('Class Accuracy (%)', fontsize=12)
        ax.set_ylabel('Class IoU (%)', fontsize=12)
        ax.set_title(f'Class Accuracy vs IoU Correlation (r={correlation:.3f})', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('IoU (%)', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'class_freq_performance.png'
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_domain_gap_analysis(
        self,
        reference_domain: Optional[str] = None,
        metric: str = 'mIoU',
        save: bool = True,
        show: bool = False,
    ) -> Optional[plt.Figure]:
        """
        Create domain gap analysis showing performance drop from reference domain.
        
        Args:
            reference_domain: Reference domain (default: domain with highest mIoU)
            metric: Metric to analyze
            save: Whether to save the figure
            show: Whether to display the figure
        """
        if 'per_domain' not in self.data:
            print("No per-domain data available")
            return None
        
        per_domain = self.data['per_domain'].get('per_domain', {})
        if not per_domain:
            print("Empty per-domain data")
            return None
        
        # Find reference domain (best performing)
        if reference_domain is None:
            reference_domain = max(per_domain.keys(), 
                                  key=lambda d: per_domain[d].get(metric, 0))
        
        ref_value = per_domain[reference_domain].get(metric, 0)
        
        # Calculate gaps
        domains = []
        values = []
        gaps = []
        
        for domain, data in per_domain.items():
            domains.append(domain)
            val = data.get(metric, 0)
            values.append(val)
            gaps.append(ref_value - val)
        
        # Sort by gap
        sorted_indices = np.argsort(gaps)
        domains = [domains[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        gaps = [gaps[i] for i in sorted_indices]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Performance values
        colors = ['green' if d == reference_domain else 
                 ('orange' if g < 20 else 'red') for d, g in zip(domains, gaps)]
        bars1 = ax1.barh(domains, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add reference line
        ax1.axvline(ref_value, color='green', linestyle='--', linewidth=2, 
                   label=f'Reference ({reference_domain}): {ref_value:.1f}%')
        
        for bar, val in zip(bars1, values):
            ax1.annotate(f'{val:.1f}%', xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                        xytext=(5, 0), textcoords='offset points', ha='left', va='center', fontsize=9)
        
        ax1.set_xlabel(f'{metric} (%)', fontsize=12)
        ax1.set_ylabel('Domain', fontsize=12)
        ax1.set_title(f'Domain {metric} Performance', fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.set_xlim(0, 100)
        ax1.grid(axis='x', alpha=0.3)
        
        # Right: Performance gap (drop from reference)
        gap_colors = ['gray' if g == 0 else ('orange' if g < 20 else 'red') for g in gaps]
        bars2 = ax2.barh(domains, gaps, color=gap_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        for bar, g in zip(bars2, gaps):
            if g > 0:
                ax2.annotate(f'-{g:.1f}%', xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                            xytext=(5, 0), textcoords='offset points', ha='left', va='center', fontsize=9)
        
        ax2.set_xlabel('Performance Gap (%)', fontsize=12)
        ax2.set_ylabel('Domain', fontsize=12)
        ax2.set_title(f'Performance Drop from {reference_domain}', fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # Add summary text
        avg_gap = np.mean([g for g in gaps if g > 0])
        max_gap = max(gaps)
        worst_domain = domains[gaps.index(max_gap)]
        
        summary_text = f'Average Gap: {avg_gap:.1f}%\nMax Gap: {max_gap:.1f}% ({worst_domain})'
        ax2.text(0.95, 0.05, summary_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'domain_gap_analysis.png'
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_domain_boxplot(
        self,
        metric: str = 'IoU',
        save: bool = True,
        show: bool = False,
    ) -> Optional[plt.Figure]:
        """
        Create box plot showing distribution of class metrics per domain.
        
        Args:
            metric: Metric to visualize
            save: Whether to save the figure
            show: Whether to display the figure
        """
        if 'full' not in self.data:
            print("No full metrics data available")
            return None
        
        full_data = self.data['full']
        per_domain_class = full_data.get('per_domain_per_class', {})
        
        if not per_domain_class:
            print("No per-domain per-class data available")
            return None
        
        # Prepare data for boxplot
        data_for_plot = []
        domain_labels = []
        
        for domain, class_data in per_domain_class.items():
            values = [class_data[cls].get(metric, 0) for cls in class_data.keys()]
            data_for_plot.append(values)
            domain_labels.append(domain)
        
        # Sort by median
        medians = [np.median(d) for d in data_for_plot]
        sorted_indices = np.argsort(medians)[::-1]
        data_for_plot = [data_for_plot[i] for i in sorted_indices]
        domain_labels = [domain_labels[i] for i in sorted_indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create boxplot
        bp = ax.boxplot(data_for_plot, labels=domain_labels, patch_artist=True)
        
        # Color the boxes
        colors = [DOMAIN_COLORS.get(d, '#ADD8E6') for d in domain_labels]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add individual points with jitter
        for i, (domain, data) in enumerate(zip(domain_labels, data_for_plot)):
            x = np.random.normal(i + 1, 0.04, size=len(data))
            ax.scatter(x, data, alpha=0.5, color='black', s=20, zorder=3)
        
        # Add mean markers
        means = [np.mean(d) for d in data_for_plot]
        ax.scatter(range(1, len(domain_labels) + 1), means, 
                  color='red', marker='D', s=50, zorder=4, label='Mean')
        
        ax.set_xlabel('Domain', fontsize=12)
        ax.set_ylabel(f'{metric} (%)', fontsize=12)
        ax.set_title(f'Distribution of Per-Class {metric} by Domain', fontsize=14, fontweight='bold')
        ax.set_xticklabels(domain_labels, rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f'domain_boxplot_{metric.lower()}.png'
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_training_curves(
        self,
        scalars_path: Optional[str] = None,
        save: bool = True,
        show: bool = False,
    ) -> Optional[plt.Figure]:
        """
        Plot training curves from scalars.json if available.
        
        Args:
            scalars_path: Path to scalars.json (default: look in parent directories)
            save: Whether to save the figure
            show: Whether to display the figure
        """
        # Try to find scalars.json
        if scalars_path is None:
            # Look in results directory and parent directories
            search_paths = [
                self.results_dir / 'scalars.json',
                self.results_dir.parent / 'scalars.json',
                self.results_dir.parent.parent / 'scalars.json',
            ]
            
            # Also look for vis_data directories (MMEngine logging)
            for vis_path in self.results_dir.parent.glob('**/vis_data/scalars.json'):
                search_paths.append(vis_path)
            
            scalars_path = None
            for path in search_paths:
                if path.exists():
                    scalars_path = path
                    break
        else:
            scalars_path = Path(scalars_path)
        
        if scalars_path is None or not scalars_path.exists():
            print("No scalars.json found for training curves")
            return None
        
        # Load scalars
        scalars = []
        with open(scalars_path) as f:
            for line in f:
                try:
                    scalars.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        
        if not scalars:
            print("Empty scalars.json")
            return None
        
        # Extract metrics
        iterations = []
        losses = []
        learning_rates = []
        
        for entry in scalars:
            if 'loss' in entry:
                iterations.append(entry.get('iter', entry.get('step', len(iterations))))
                losses.append(entry.get('loss', 0))
                if 'lr' in entry:
                    learning_rates.append(entry.get('lr', 0))
        
        if not losses:
            print("No loss data found in scalars")
            return None
        
        # Create figure
        fig, axes = plt.subplots(1, 2 if learning_rates else 1, figsize=(14 if learning_rates else 10, 6))
        
        if not learning_rates:
            axes = [axes]
        
        # Loss curve
        ax1 = axes[0]
        ax1.plot(iterations, losses, 'b-', alpha=0.7, linewidth=1)
        
        # Add smoothed curve
        if len(losses) > 10:
            window = min(50, len(losses) // 10)
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax1.plot(iterations[window-1:], smoothed, 'r-', linewidth=2, label='Smoothed')
        
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Learning rate curve
        if learning_rates:
            ax2 = axes[1]
            ax2.plot(iterations[:len(learning_rates)], learning_rates, 'g-', linewidth=1.5)
            ax2.set_xlabel('Iteration', fontsize=12)
            ax2.set_ylabel('Learning Rate', fontsize=12)
            ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'training_curves.png'
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_improvement_comparison(
        self,
        baseline_dir: str,
        metric: str = 'mIoU',
        save: bool = True,
        show: bool = False,
    ) -> Optional[plt.Figure]:
        """
        Create delta/improvement chart comparing current results to baseline.
        
        Args:
            baseline_dir: Path to baseline results directory
            metric: Metric to compare
            save: Whether to save the figure
            show: Whether to display the figure
        """
        baseline_path = Path(baseline_dir)
        
        # Load baseline domain metrics
        baseline_domain_path = baseline_path / 'metrics_per_domain.json'
        if not baseline_domain_path.exists():
            print(f"No baseline metrics found at {baseline_domain_path}")
            return None
        
        with open(baseline_domain_path) as f:
            baseline_data = json.load(f)
        
        if 'per_domain' not in self.data:
            print("No per-domain data in current results")
            return None
        
        current_data = self.data['per_domain'].get('per_domain', {})
        baseline_domains = baseline_data.get('per_domain', {})
        
        # Calculate improvements
        domains = []
        current_values = []
        baseline_values = []
        improvements = []
        
        for domain in current_data.keys():
            if domain in baseline_domains:
                domains.append(domain)
                curr = current_data[domain].get(metric, 0)
                base = baseline_domains[domain].get(metric, 0)
                current_values.append(curr)
                baseline_values.append(base)
                improvements.append(curr - base)
        
        if not domains:
            print("No matching domains found between current and baseline")
            return None
        
        # Sort by improvement
        sorted_indices = np.argsort(improvements)[::-1]
        domains = [domains[i] for i in sorted_indices]
        current_values = [current_values[i] for i in sorted_indices]
        baseline_values = [baseline_values[i] for i in sorted_indices]
        improvements = [improvements[i] for i in sorted_indices]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Side by side comparison
        x = np.arange(len(domains))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, baseline_values, width, label='Baseline', color='coral', alpha=0.8)
        bars2 = ax1.bar(x + width/2, current_values, width, label='Current', color='steelblue', alpha=0.8)
        
        ax1.set_xlabel('Domain', fontsize=12)
        ax1.set_ylabel(f'{metric} (%)', fontsize=12)
        ax1.set_title(f'Baseline vs Current {metric}', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(domains, rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim(0, 105)
        ax1.grid(axis='y', alpha=0.3)
        
        # Right: Improvement delta
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars3 = ax2.barh(domains, improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        for bar, imp in zip(bars3, improvements):
            sign = '+' if imp > 0 else ''
            ax2.annotate(f'{sign}{imp:.1f}%', 
                        xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                        xytext=(5 if imp >= 0 else -5, 0), 
                        textcoords='offset points', 
                        ha='left' if imp >= 0 else 'right', 
                        va='center', fontsize=9)
        
        ax2.axvline(0, color='black', linewidth=1)
        ax2.set_xlabel(f'{metric} Change (%)', fontsize=12)
        ax2.set_ylabel('Domain', fontsize=12)
        ax2.set_title('Improvement from Baseline', fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # Summary
        avg_improvement = np.mean(improvements)
        improved_count = sum(1 for i in improvements if i > 0)
        summary_text = f'Avg: {avg_improvement:+.1f}%\nImproved: {improved_count}/{len(domains)}'
        ax2.text(0.95, 0.05, summary_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', 
                         facecolor='lightgreen' if avg_improvement > 0 else 'lightsalmon', 
                         alpha=0.5))
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'improvement_comparison.png'
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_summary_dashboard(
        self,
        save: bool = True,
        show: bool = False,
    ) -> Optional[plt.Figure]:
        """
        Create comprehensive dashboard with multiple plots.
        
        Args:
            save: Whether to save the figure
            show: Whether to display the figure
        """
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title
        config = self.data.get('summary', {}).get('config', {})
        dataset = config.get('dataset', 'Unknown')
        timestamp = config.get('timestamp', '')
        fig.suptitle(f'Test Results Dashboard - {dataset}\n{timestamp}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Overall metrics (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        overall = self.data.get('summary', {}).get('overall', {})
        if overall:
            metrics = ['aAcc', 'mIoU', 'mAcc', 'fwIoU']
            values = [overall.get(m, 0) for m in metrics]
            colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(metrics)))
            bars = ax1.bar(metrics, values, color=colors, edgecolor='black', linewidth=0.5)
            for bar, val in zip(bars, values):
                ax1.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
            ax1.set_ylabel('Score (%)')
            ax1.set_title('Overall Metrics', fontweight='bold')
            ax1.set_ylim(0, 105)
            ax1.grid(axis='y', alpha=0.3)
        
        # 2. Per-domain mIoU (top middle and right)
        ax2 = fig.add_subplot(gs[0, 1:])
        per_domain = self.data.get('per_domain', {}).get('per_domain', {})
        if per_domain:
            domains = list(per_domain.keys())
            miou_values = [per_domain[d].get('mIoU', 0) for d in domains]
            fwiou_values = [per_domain[d].get('fwIoU', 0) for d in domains]
            
            x = np.arange(len(domains))
            width = 0.35
            bars1 = ax2.bar(x - width/2, miou_values, width, label='mIoU', color='steelblue', alpha=0.8)
            bars2 = ax2.bar(x + width/2, fwiou_values, width, label='fwIoU', color='coral', alpha=0.8)
            
            ax2.set_xlabel('Domain')
            ax2.set_ylabel('Score (%)')
            ax2.set_title('Per-Domain Performance', fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(domains, rotation=45, ha='right')
            ax2.legend()
            ax2.set_ylim(0, 105)
            ax2.grid(axis='y', alpha=0.3)
        
        # 3. Per-class IoU (middle row, spanning full width)
        ax3 = fig.add_subplot(gs[1, :])
        per_class = self.data.get('per_class', {}).get('per_class', {})
        if per_class:
            classes = list(per_class.keys())
            iou_values = [per_class[c].get('IoU', 0) for c in classes]
            
            # Sort by value
            sorted_pairs = sorted(zip(classes, iou_values), key=lambda x: x[1], reverse=True)
            classes, iou_values = zip(*sorted_pairs)
            
            colors = plt.cm.RdYlGn(np.array(iou_values) / 100)
            bars = ax3.bar(classes, iou_values, color=colors, edgecolor='black', linewidth=0.5)
            
            ax3.set_xlabel('Class')
            ax3.set_ylabel('IoU (%)')
            ax3.set_title('Per-Class IoU (sorted)', fontweight='bold')
            ax3.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
            ax3.set_ylim(0, 105)
            ax3.axhline(np.mean(iou_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(iou_values):.1f}%')
            ax3.legend()
            ax3.grid(axis='y', alpha=0.3)
        
        # 4. Domain radar (bottom left)
        ax4 = fig.add_subplot(gs[2, 0], polar=True)
        if per_domain:
            metrics = ['mIoU', 'fwIoU', 'mAcc', 'aAcc']
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]
            
            for domain in list(per_domain.keys())[:5]:  # Limit to 5 domains for readability
                values = [per_domain[domain].get(m, 0) for m in metrics]
                values += values[:1]
                ax4.plot(angles, values, 'o-', linewidth=2, label=domain)
                ax4.fill(angles, values, alpha=0.1)
            
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(metrics, fontsize=9)
            ax4.set_ylim(0, 100)
            ax4.set_title('Domain Radar', fontweight='bold', y=1.1)
            ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
        
        # 5. Image count distribution (bottom middle)
        ax5 = fig.add_subplot(gs[2, 1])
        if per_domain:
            domains = list(per_domain.keys())
            img_counts = [per_domain[d].get('num_images', 0) for d in domains]
            
            colors = [DOMAIN_COLORS.get(d, 'gray') for d in domains]
            wedges, texts, autotexts = ax5.pie(img_counts, labels=domains, autopct='%1.0f%%',
                                               colors=colors, startangle=90)
            ax5.set_title('Image Distribution', fontweight='bold')
        
        # 6. Best/Worst classes (bottom right)
        ax6 = fig.add_subplot(gs[2, 2])
        if per_class:
            classes = list(per_class.keys())
            iou_values = [per_class[c].get('IoU', 0) for c in classes]
            sorted_pairs = sorted(zip(classes, iou_values), key=lambda x: x[1])
            
            # Get worst 5 and best 5
            worst_5 = sorted_pairs[:5]
            best_5 = sorted_pairs[-5:][::-1]
            
            labels = [f"{c} ({v:.1f}%)" for c, v in best_5 + worst_5]
            values_display = [v for _, v in best_5 + worst_5]
            colors_display = ['green'] * 5 + ['red'] * 5
            
            y_pos = np.arange(len(labels))
            ax6.barh(y_pos, values_display, color=colors_display, alpha=0.7)
            ax6.set_yticks(y_pos)
            ax6.set_yticklabels(labels, fontsize=9)
            ax6.set_xlabel('IoU (%)')
            ax6.set_title('Best/Worst Classes', fontweight='bold')
            ax6.axhline(4.5, color='black', linestyle='-', linewidth=1)
            ax6.text(50, 2, 'Best 5', fontsize=10, ha='center', color='green', fontweight='bold')
            ax6.text(50, 7, 'Worst 5', fontsize=10, ha='center', color='red', fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save:
            filepath = self.output_dir / 'dashboard.png'
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def generate_all_plots(self, show: bool = False):
        """Generate all available plots."""
        print(f"\nGenerating visualizations for: {self.results_dir}")
        print(f"Output directory: {self.output_dir}")
        print("-" * 50)
        
        # Domain metrics
        if 'per_domain' in self.data:
            self.plot_domain_metrics(show=show)
            self.plot_radar_chart(show=show)
            self.plot_domain_gap_analysis(show=show)
        
        # Class metrics
        if 'per_class' in self.data:
            self.plot_class_metrics(metric='IoU', show=show)
            self.plot_class_metrics(metric='Acc', show=show)
            self.plot_class_frequency_performance(show=show)
        
        # Heatmap and boxplot (requires full data)
        if 'full' in self.data:
            self.plot_heatmap(metric='IoU', show=show)
            self.plot_domain_boxplot(metric='IoU', show=show)
        
        # Training curves (if available)
        self.plot_training_curves(show=show)
        
        # Dashboard
        self.plot_summary_dashboard(show=show)
        
        print("-" * 50)
        print(f"All plots saved to: {self.output_dir}")


def compare_results(
    results_dirs: List[str],
    output_dir: str,
    labels: Optional[List[str]] = None,
    metric: str = 'mIoU',
) -> Optional[plt.Figure]:
    """
    Compare results from multiple test runs.
    
    Args:
        results_dirs: List of paths to result directories
        output_dir: Output directory for comparison plots
        labels: Labels for each result (default: directory names)
        metric: Metric to compare
    """
    if not HAS_PLOTTING:
        print("Plotting libraries not available")
        return None
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load all results
    all_data = []
    if labels is None:
        labels = [Path(d).name for d in results_dirs]
    
    for results_dir in results_dirs:
        summary_path = Path(results_dir) / 'metrics_summary.json'
        if summary_path.exists():
            with open(summary_path) as f:
                all_data.append(json.load(f))
        else:
            all_data.append(None)
    
    # Filter valid results
    valid_data = [(l, d) for l, d in zip(labels, all_data) if d is not None]
    if not valid_data:
        print("No valid results found")
        return None
    
    labels, all_data = zip(*valid_data)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Overall metrics comparison
    ax1 = axes[0]
    metrics = ['aAcc', 'mIoU', 'mAcc', 'fwIoU']
    x = np.arange(len(metrics))
    width = 0.8 / len(labels)
    
    for i, (label, data) in enumerate(zip(labels, all_data)):
        overall = data.get('overall', {})
        values = [overall.get(m, 0) for m in metrics]
        offset = (i - len(labels)/2 + 0.5) * width
        ax1.bar(x + offset, values, width, label=label, alpha=0.8)
    
    ax1.set_xlabel('Metric')
    ax1.set_ylabel('Score (%)')
    ax1.set_title('Overall Metrics Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.3)
    
    # mIoU comparison bar chart
    ax2 = axes[1]
    miou_values = [d.get('overall', {}).get('mIoU', 0) for d in all_data]
    colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(labels)))
    bars = ax2.bar(labels, miou_values, color=colors, edgecolor='black', linewidth=0.5)
    
    for bar, val in zip(bars, miou_values):
        ax2.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
    
    ax2.set_xlabel('Model/Configuration')
    ax2.set_ylabel('mIoU (%)')
    ax2.set_title('mIoU Comparison', fontweight='bold')
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    filepath = output_path / 'comparison.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Saved comparison: {filepath}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='PROVE Test Result Visualizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Visualize single test result
    python test_result_visualizer.py --results-dir /path/to/test_results_detailed/timestamp/

    # Generate specific plots
    python test_result_visualizer.py --results-dir /path/to/results --plots domain class

    # Compare multiple results
    python test_result_visualizer.py --compare --results-dirs dir1 dir2 --labels "Model A" "Model B"

    # Change output format
    python test_result_visualizer.py --results-dir /path/to/results --format pdf --dpi 300

    # Generate domain gap analysis
    python test_result_visualizer.py --results-dir /path/to/results --plots gap boxplot

    # Compare with baseline
    python test_result_visualizer.py --results-dir /path/to/results --baseline /path/to/baseline
        """
    )
    
    # Input options
    parser.add_argument('--results-dir', type=str, help='Path to test results directory')
    parser.add_argument('--output-dir', type=str, help='Output directory for figures')
    
    # Comparison mode
    parser.add_argument('--compare', action='store_true', help='Compare multiple results')
    parser.add_argument('--results-dirs', type=str, nargs='+', help='Paths to result directories for comparison')
    parser.add_argument('--labels', type=str, nargs='+', help='Labels for comparison')
    parser.add_argument('--baseline', type=str, help='Baseline results directory for improvement comparison')
    
    # Plot selection
    parser.add_argument('--plots', type=str, nargs='+', 
                       choices=['domain', 'class', 'radar', 'heatmap', 'dashboard', 
                               'gap', 'boxplot', 'freq', 'training', 'all'],
                       default=['all'], help='Types of plots to generate')
    
    # Output options
    parser.add_argument('--format', type=str, default='png', choices=['png', 'pdf', 'svg'],
                       help='Output format')
    parser.add_argument('--dpi', type=int, default=150, help='Figure DPI')
    parser.add_argument('--show', action='store_true', help='Display plots')
    
    args = parser.parse_args()
    
    if not HAS_PLOTTING:
        print("Error: matplotlib and seaborn are required.")
        print("Install with: pip install matplotlib seaborn")
        sys.exit(1)
    
    # Comparison mode
    if args.compare:
        if not args.results_dirs:
            print("Error: --results-dirs required for comparison mode")
            sys.exit(1)
        output_dir = args.output_dir or 'comparison_results'
        compare_results(args.results_dirs, output_dir, args.labels)
        return
    
    # Single result visualization
    if not args.results_dir:
        parser.print_help()
        print("\nError: --results-dir is required")
        sys.exit(1)
    
    visualizer = TestResultVisualizer(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )
    
    if 'all' in args.plots:
        visualizer.generate_all_plots(show=args.show)
    else:
        if 'domain' in args.plots:
            visualizer.plot_domain_metrics(show=args.show)
        if 'class' in args.plots:
            visualizer.plot_class_metrics(show=args.show)
        if 'radar' in args.plots:
            visualizer.plot_radar_chart(show=args.show)
        if 'heatmap' in args.plots:
            visualizer.plot_heatmap(show=args.show)
        if 'dashboard' in args.plots:
            visualizer.plot_summary_dashboard(show=args.show)
        if 'gap' in args.plots:
            visualizer.plot_domain_gap_analysis(show=args.show)
        if 'boxplot' in args.plots:
            visualizer.plot_domain_boxplot(show=args.show)
        if 'freq' in args.plots:
            visualizer.plot_class_frequency_performance(show=args.show)
        if 'training' in args.plots:
            visualizer.plot_training_curves(show=args.show)
    
    # Baseline comparison if provided
    if args.baseline:
        visualizer.plot_improvement_comparison(args.baseline, show=args.show)


if __name__ == '__main__':
    main()
