#!/usr/bin/env python3
"""
Extended Weather Domain Analysis

Generates additional visualizations:
1. Per-dataset strategy comparison
2. Per-model architecture comparison
3. Strategy correlation analysis
4. Publication-ready summary figure

Usage:
    python analyze_weather_extended.py [--input ./result_figures/weather_analysis/weather_domain_results.csv]
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import pearsonr

# Configuration
OUTPUT_DIR = "./result_figures/weather_analysis"
ADVERSE_DOMAINS = ["foggy", "rainy", "snowy", "night", "dawn_dusk"]
NORMAL_DOMAINS = ["clear_day", "cloudy"]

# Color palettes
STRATEGY_COLORS = {
    "gen": "#E74C3C",      # Red for generative
    "std": "#3498DB",      # Blue for standard
    "baseline": "#2ECC71", # Green for baseline
    "other": "#95A5A6"     # Gray for other
}

DOMAIN_COLORS = {
    "foggy": "#9B59B6",
    "rainy": "#3498DB",
    "snowy": "#ECF0F1",
    "night": "#2C3E50",
    "dawn_dusk": "#E67E22",
    "clear_day": "#F1C40F",
    "cloudy": "#BDC3C7"
}


def load_results(csv_path: str) -> pd.DataFrame:
    """Load weather domain results CSV."""
    df = pd.read_csv(csv_path)
    
    # Ensure proper types
    df["improvement"] = pd.to_numeric(df["improvement"], errors="coerce")
    df["mIoU"] = pd.to_numeric(df["mIoU"], errors="coerce")
    
    return df


def categorize_strategy(strategy: str) -> str:
    """Categorize strategy into gen/std/baseline/other."""
    if strategy == "baseline":
        return "baseline"
    elif strategy.startswith("gen_"):
        return "gen"
    elif strategy.startswith("std_"):
        return "std"
    else:
        return "other"


def create_dataset_comparison(df: pd.DataFrame, output_dir: str):
    """Create bar chart comparing strategy improvements across datasets."""
    # Filter to non-baseline strategies
    df_filt = df[df["strategy"] != "baseline"].copy()
    df_filt["strategy_type"] = df_filt["strategy"].apply(categorize_strategy)
    
    # Get mean improvement per strategy and dataset
    pivot = df_filt.pivot_table(
        values="improvement",
        index="strategy",
        columns="dataset",
        aggfunc="mean"
    )
    
    # Sort by overall improvement
    pivot["mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("mean", ascending=False).drop("mean", axis=1)
    
    # Get strategy types for coloring
    strategy_types = [categorize_strategy(s) for s in pivot.index]
    colors = [STRATEGY_COLORS[t] for t in strategy_types]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    pivot.plot(kind="bar", ax=ax, width=0.8, alpha=0.8)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title("Strategy Performance Across Datasets (mIoU Improvement)", fontsize=14)
    ax.set_xlabel("Strategy", fontsize=12)
    ax.set_ylabel("mIoU Improvement over Baseline", fontsize=12)
    ax.legend(title="Dataset", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dataset_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {output_dir}/dataset_comparison.png")


def create_model_comparison(df: pd.DataFrame, output_dir: str):
    """Create comparison of strategy performance across model architectures."""
    # Filter to non-baseline strategies
    df_filt = df[df["strategy"] != "baseline"].copy()
    df_filt["strategy_type"] = df_filt["strategy"].apply(categorize_strategy)
    
    # Simplify model names
    model_map = {
        "deeplabv3plus_r50": "DeepLabv3+",
        "pspnet_r50": "PSPNet",
        "segformer_mit-b5": "SegFormer"
    }
    df_filt["model_clean"] = df_filt["model"].map(model_map).fillna(df_filt["model"])
    
    # Get mean improvement per strategy and model
    pivot = df_filt.pivot_table(
        values="improvement",
        index="strategy",
        columns="model_clean",
        aggfunc="mean"
    )
    
    # Sort by overall improvement
    pivot["mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("mean", ascending=False).drop("mean", axis=1)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    pivot.plot(kind="bar", ax=ax, width=0.8, alpha=0.85)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title("Strategy Performance Across Model Architectures", fontsize=14)
    ax.set_xlabel("Strategy", fontsize=12)
    ax.set_ylabel("mIoU Improvement over Baseline", fontsize=12)
    ax.legend(title="Model", loc='upper right')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {output_dir}/model_comparison.png")


def create_strategy_correlation(df: pd.DataFrame, output_dir: str):
    """Create correlation heatmap between strategies across domains."""
    # Pivot: rows are domain-dataset-model combinations, columns are strategies
    df_filt = df[df["strategy"] != "baseline"].copy()
    
    pivot = df_filt.pivot_table(
        values="mIoU",
        index=["dataset", "model", "domain"],
        columns="strategy",
        aggfunc="mean"
    )
    
    # Calculate correlation matrix
    corr = pivot.corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        square=True
    )
    
    ax.set_title("Strategy Correlation Matrix (mIoU)", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "strategy_correlation.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {output_dir}/strategy_correlation.png")


def create_domain_strategy_matrix(df: pd.DataFrame, output_dir: str):
    """Create comprehensive domain x strategy heatmap."""
    # Filter to non-baseline strategies
    df_filt = df[df["strategy"] != "baseline"].copy()
    df_filt["strategy_type"] = df_filt["strategy"].apply(categorize_strategy)
    
    # Pivot: domains vs strategies
    pivot = df_filt.pivot_table(
        values="improvement",
        index="domain",
        columns="strategy",
        aggfunc="mean"
    )
    
    # Reorder domains (adverse first)
    domain_order = [d for d in ADVERSE_DOMAINS if d in pivot.index] + \
                   [d for d in NORMAL_DOMAINS if d in pivot.index]
    pivot = pivot.reindex(domain_order)
    
    # Reorder strategies by type then average improvement
    strategies_gen = [s for s in pivot.columns if categorize_strategy(s) == "gen"]
    strategies_std = [s for s in pivot.columns if categorize_strategy(s) == "std"]
    strategies_other = [s for s in pivot.columns if categorize_strategy(s) == "other"]
    
    strategy_order = sorted(strategies_gen, key=lambda s: pivot[s].mean(), reverse=True) + \
                     sorted(strategies_std, key=lambda s: pivot[s].mean(), reverse=True) + \
                     sorted(strategies_other, key=lambda s: pivot[s].mean(), reverse=True)
    pivot = pivot[strategy_order]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-1.5,
        vmax=1.5,
        ax=ax,
        cbar_kws={"label": "mIoU Improvement"}
    )
    
    ax.set_title("Strategy Performance Matrix: Domain × Strategy", fontsize=14)
    ax.set_xlabel("Strategy", fontsize=12)
    ax.set_ylabel("Weather Domain", fontsize=12)
    
    # Add type annotations on x-axis
    xtick_colors = [STRATEGY_COLORS[categorize_strategy(s)] for s in strategy_order]
    for tick, color in zip(ax.get_xticklabels(), xtick_colors):
        tick.set_color(color)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "domain_strategy_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {output_dir}/domain_strategy_matrix.png")


def create_publication_figure(df: pd.DataFrame, output_dir: str):
    """Create a publication-ready multi-panel summary figure."""
    fig = plt.figure(figsize=(16, 12))
    
    # Filter data
    df_filt = df[df["strategy"] != "baseline"].copy()
    df_filt["strategy_type"] = df_filt["strategy"].apply(categorize_strategy)
    df_filt["domain_type"] = df_filt["domain"].apply(
        lambda x: "adverse" if x in ADVERSE_DOMAINS else "normal"
    )
    
    # Panel 1: Overall improvement by strategy type
    ax1 = fig.add_subplot(2, 2, 1)
    type_means = df_filt.groupby("strategy_type")["improvement"].mean().sort_values(ascending=False)
    type_stds = df_filt.groupby("strategy_type")["improvement"].std()
    
    bars = ax1.bar(type_means.index, type_means.values, 
                   color=[STRATEGY_COLORS[t] for t in type_means.index],
                   yerr=type_stds.values, capsize=5, alpha=0.8)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_title("(A) Average Improvement by Strategy Type", fontsize=12, fontweight='bold')
    ax1.set_ylabel("mIoU Improvement", fontsize=10)
    ax1.set_xlabel("Strategy Type", fontsize=10)
    
    # Panel 2: Adverse vs Normal conditions
    ax2 = fig.add_subplot(2, 2, 2)
    domain_type_means = df_filt.groupby(["strategy", "domain_type"])["improvement"].mean().unstack()
    
    x = np.arange(len(domain_type_means))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, domain_type_means.get("adverse", [0]*len(domain_type_means)), 
                    width, label="Adverse", color="#E74C3C", alpha=0.8)
    bars2 = ax2.bar(x + width/2, domain_type_means.get("normal", [0]*len(domain_type_means)), 
                    width, label="Normal", color="#2ECC71", alpha=0.8)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title("(B) Performance in Adverse vs Normal Conditions", fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(domain_type_means.index, rotation=45, ha='right')
    ax2.set_ylabel("mIoU Improvement", fontsize=10)
    ax2.legend(loc='upper right')
    
    # Panel 3: Top performers per domain
    ax3 = fig.add_subplot(2, 2, 3)
    
    top_per_domain = df_filt.groupby("domain")["improvement"].apply(
        lambda x: x.nlargest(1).values[0] if len(x) > 0 else 0
    )
    domain_order = [d for d in ADVERSE_DOMAINS + NORMAL_DOMAINS if d in top_per_domain.index]
    top_per_domain = top_per_domain.reindex(domain_order)
    
    colors = [DOMAIN_COLORS.get(d, "#888888") for d in top_per_domain.index]
    bars = ax3.barh(top_per_domain.index, top_per_domain.values, color=colors, alpha=0.8)
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_title("(C) Best Strategy Improvement per Domain", fontsize=12, fontweight='bold')
    ax3.set_xlabel("mIoU Improvement", fontsize=10)
    
    # Annotate with strategy names
    best_strategies = df_filt.loc[df_filt.groupby("domain")["improvement"].idxmax()].set_index("domain")["strategy"]
    for i, (domain, imp) in enumerate(top_per_domain.items()):
        if domain in best_strategies.index:
            ax3.annotate(f' {best_strategies[domain]}', (imp, i), va='center', fontsize=8)
    
    # Panel 4: Strategy ranking
    ax4 = fig.add_subplot(2, 2, 4)
    strategy_means = df_filt.groupby("strategy")["improvement"].mean().sort_values(ascending=True)
    strategy_stds = df_filt.groupby("strategy")["improvement"].std()
    
    colors = [STRATEGY_COLORS[categorize_strategy(s)] for s in strategy_means.index]
    bars = ax4.barh(strategy_means.index, strategy_means.values, 
                    xerr=strategy_stds.reindex(strategy_means.index).values, 
                    color=colors, capsize=3, alpha=0.8)
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_title("(D) Overall Strategy Ranking", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Mean mIoU Improvement", fontsize=10)
    
    # Add legend for strategy types
    legend_patches = [
        mpatches.Patch(color=STRATEGY_COLORS["gen"], label="Generative (gen_)", alpha=0.8),
        mpatches.Patch(color=STRATEGY_COLORS["std"], label="Standard (std_)", alpha=0.8),
        mpatches.Patch(color=STRATEGY_COLORS["other"], label="Other", alpha=0.8),
    ]
    ax4.legend(handles=legend_patches, loc='lower right', fontsize=8)
    
    plt.suptitle("Weather Domain Segmentation: Augmentation Strategy Analysis", 
                 fontsize=14, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "publication_summary.png"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "publication_summary.pdf"), bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {output_dir}/publication_summary.png")
    print(f"Saved: {output_dir}/publication_summary.pdf")


def create_improvement_distribution(df: pd.DataFrame, output_dir: str):
    """Create violin plot showing improvement distributions."""
    df_filt = df[df["strategy"] != "baseline"].copy()
    df_filt["strategy_type"] = df_filt["strategy"].apply(categorize_strategy)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: By strategy type
    ax1 = axes[0]
    sns.violinplot(data=df_filt, x="strategy_type", y="improvement", ax=ax1,
                   palette=STRATEGY_COLORS, inner="box")
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax1.set_title("Improvement Distribution by Strategy Type", fontsize=12)
    ax1.set_xlabel("Strategy Type", fontsize=10)
    ax1.set_ylabel("mIoU Improvement", fontsize=10)
    
    # Panel 2: By domain type
    df_filt["domain_type"] = df_filt["domain"].apply(
        lambda x: "adverse" if x in ADVERSE_DOMAINS else "normal"
    )
    ax2 = axes[1]
    sns.violinplot(data=df_filt, x="domain_type", y="improvement", hue="strategy_type",
                   ax=ax2, palette=STRATEGY_COLORS, inner="box", split=True)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax2.set_title("Improvement Distribution by Domain & Strategy Type", fontsize=12)
    ax2.set_xlabel("Domain Type", fontsize=10)
    ax2.set_ylabel("mIoU Improvement", fontsize=10)
    ax2.legend(title="Strategy Type", loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "improvement_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {output_dir}/improvement_distributions.png")


def main():
    parser = argparse.ArgumentParser(description="Extended Weather Domain Analysis")
    parser.add_argument("--input", type=str, 
                       default="./result_figures/weather_analysis/weather_domain_results.csv",
                       help="Input CSV file")
    parser.add_argument("--output", type=str,
                       default="./result_figures/weather_analysis",
                       help="Output directory")
    args = parser.parse_args()
    
    print("=" * 60)
    print("EXTENDED WEATHER DOMAIN ANALYSIS")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from: {args.input}")
    df = load_results(args.input)
    print(f"Loaded {len(df)} records")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    create_dataset_comparison(df, args.output)
    create_model_comparison(df, args.output)
    create_strategy_correlation(df, args.output)
    create_domain_strategy_matrix(df, args.output)
    create_publication_figure(df, args.output)
    create_improvement_distribution(df, args.output)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
