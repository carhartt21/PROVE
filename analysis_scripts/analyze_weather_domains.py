#!/usr/bin/env python3
"""
Weather Domain Analysis Script

Analyzes segmentation performance across different weather conditions for all
augmentation strategies. Generates insights about which strategies improve
performance in specific weather domains (fog, rain, snow, night, etc.).

Supports both Stage 1 and Stage 2:
- Stage 1 (WEIGHTS/): Models trained only on clear_day
- Stage 2 (WEIGHTS_STAGE_2/): Models trained on all domains

Usage:
    python analyze_weather_domains.py                  # Stage 1 (default)
    python analyze_weather_domains.py --stage 2       # Stage 2
    python analyze_weather_domains.py --output-dir ./figures
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration - Stage-specific weights directories
WEIGHTS_ROOT_STAGE1 = "${AWARE_DATA_ROOT}/WEIGHTS"
WEIGHTS_ROOT_STAGE2 = "${AWARE_DATA_ROOT}/WEIGHTS_STAGE_2"

DATASETS = ["acdc", "bdd10k", "iddaw", "mapillaryvistas", "outside15k"]
MODELS = ["deeplabv3plus_r50", "pspnet_r50", "segformer_mit-b5"]

# Weather domains by dataset
WEATHER_DOMAINS = {
    "acdc": ["clear_day", "cloudy", "dawn_dusk", "foggy", "night", "rainy", "snowy"],
    "bdd10k": ["clear_day", "cloudy", "dawn_dusk", "foggy", "night", "rainy", "snowy"],
    "iddaw": ["clear_day", "cloudy", "dawn_dusk", "foggy", "night", "rainy"],
    "mapillaryvistas": ["clear_day", "cloudy", "dawn_dusk", "foggy", "night", "rainy", "snowy"],
    "outside15k": ["clear_day", "cloudy", "dawn_dusk", "foggy", "night", "rainy", "snowy"],
}

# Domain categories for analysis
ADVERSE_DOMAINS = ["foggy", "rainy", "snowy", "night", "dawn_dusk"]
NORMAL_DOMAINS = ["clear_day", "cloudy"]


def parse_test_report(filepath: Path) -> Optional[Dict]:
    """Parse a test_report.txt file and extract per-domain metrics."""
    try:
        with open(filepath) as f:
            content = f.read()
        
        # Extract per-domain metrics
        per_domain = {}
        current_domain = None
        
        lines = content.split('\n')
        in_per_domain = False
        
        for line in lines:
            if 'PER-DOMAIN METRICS' in line:
                in_per_domain = True
                continue
            if in_per_domain and 'PER-CLASS METRICS' in line:
                break
            if in_per_domain:
                if line.startswith('---') and line.endswith('---'):
                    current_domain = line.strip('- ')
                    per_domain[current_domain] = {}
                elif current_domain and ':' in line:
                    key, value = line.strip().split(':')
                    key = key.strip()
                    try:
                        per_domain[current_domain][key] = float(value.strip())
                    except ValueError:
                        pass
        
        return per_domain if per_domain else None
    except Exception as e:
        return None


def load_detailed_results(weights_dir: str) -> List[Dict]:
    """Load all detailed test results with per-domain breakdowns."""
    results = []
    loaded_count = 0
    skipped_count = 0
    
    # Find all strategy directories
    for strategy_dir in Path(weights_dir).iterdir():
        if not strategy_dir.is_dir():
            continue
        
        strategy = strategy_dir.name
        
        # Find all dataset directories
        for dataset_dir in strategy_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            dataset = dataset_dir.name.lower()
            
            # Find all model directories
            for model_dir in dataset_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                
                model = model_dir.name
                
                # Skip clear_day models
                if "clear_day" in model:
                    continue
                
                # Find detailed results
                detailed_dir = model_dir / "test_results_detailed"
                if not detailed_dir.exists():
                    continue
                
                # Get the latest results
                result_dirs = sorted([d for d in detailed_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
                if not result_dirs:
                    continue
                
                latest_result = result_dirs[-1]
                per_domain = None
                
                # Try multiple file formats in priority order
                for fname in ["metrics_per_domain.json", "results.json", "test_report.txt"]:
                    potential_file = latest_result / fname
                    if not potential_file.exists():
                        continue
                    
                    try:
                        if fname.endswith('.json'):
                            with open(potential_file) as f:
                                data = json.load(f)
                            if "per_domain" in data:
                                per_domain = data["per_domain"]
                                break
                        elif fname == 'test_report.txt':
                            per_domain = parse_test_report(potential_file)
                            if per_domain:
                                break
                    except Exception as e:
                        continue
                
                if not per_domain:
                    skipped_count += 1
                    continue
                
                for domain, domain_data in per_domain.items():
                    # Handle metrics_per_domain.json format (flat structure)
                    if "mIoU" in domain_data:
                        results.append({
                            "strategy": strategy,
                            "dataset": dataset,
                            "model": model,
                            "domain": domain,
                            "mIoU": domain_data.get("mIoU", 0),
                            "mAcc": domain_data.get("mAcc", 0),
                            "aAcc": domain_data.get("aAcc", 0),
                            "fwIoU": domain_data.get("fwIoU", 0),
                            "num_images": domain_data.get("num_images", 0),
                        })
                        loaded_count += 1
                    # Handle results.json format (nested summary structure)
                    elif "summary" in domain_data:
                        summary = domain_data["summary"]
                        results.append({
                            "strategy": strategy,
                            "dataset": dataset,
                            "model": model,
                            "domain": domain,
                            "mIoU": summary.get("mIoU", 0),
                            "mAcc": summary.get("mAcc", 0),
                            "aAcc": summary.get("aAcc", 0),
                            "fwIoU": summary.get("fwIoU", 0),
                            "num_images": summary.get("num_images", 0),
                        })
                        loaded_count += 1
    
    print(f"Loaded {loaded_count} domain results from {len(set(r['strategy'] for r in results))} strategies")
    print(f"Skipped {skipped_count} model directories without per-domain results")
    print(f"Skipped {skipped_count} model directories without per-domain results")
    
    return results


def categorize_strategy(strategy: str) -> Tuple[str, str, str]:
    """Categorize a strategy into type, gen component, and std component."""
    if strategy == "baseline":
        return "baseline", "", ""
    
    parts = strategy.split("+")
    gen_parts = [p for p in parts if p.startswith("gen_")]
    std_parts = [p for p in parts if p.startswith("std_")]
    
    if gen_parts and std_parts:
        strategy_type = "combined"
    elif gen_parts:
        strategy_type = "gen"
    elif std_parts:
        strategy_type = "std"
    else:
        strategy_type = "other"
    
    gen_component = "+".join(gen_parts) if gen_parts else ""
    std_component = "+".join(std_parts) if std_parts else ""
    
    return strategy_type, gen_component, std_component


def normalize_model_name(model: str) -> str:
    """Normalize model name by removing ratio suffix for comparison."""
    # Remove _ratio0pXX suffix if present
    import re
    return re.sub(r'_ratio\dp\d+', '', model)


def analyze_domain_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance improvement per domain relative to baseline."""
    # First normalize model names for comparison
    df = df.copy()
    df["model_normalized"] = df["model"].apply(normalize_model_name)
    
    # Get baseline performance per domain (using normalized model names)
    baseline_df = df[df["strategy"] == "baseline"]
    baseline_perf = baseline_df.groupby(["dataset", "model_normalized", "domain"])["mIoU"].mean().to_dict()
    
    # Calculate improvement for each entry (using normalized model names)
    improvements = []
    for idx, row in df.iterrows():
        key = (row["dataset"], row["model_normalized"], row["domain"])
        baseline_val = baseline_perf.get(key, 0)
        improvement = row["mIoU"] - baseline_val if baseline_val > 0 else 0
        improvements.append(improvement)
    
    df["improvement"] = improvements
    
    return df


def create_domain_heatmap(df: pd.DataFrame, output_dir: str):
    """Create heatmap showing strategy performance across weather domains."""
    # Filter to gen strategies only
    gen_df = df[df["strategy_type"] == "gen"]
    
    # Pivot table: strategy vs domain
    pivot = gen_df.pivot_table(
        values="improvement", 
        index="strategy", 
        columns="domain",
        aggfunc="mean"
    )
    
    # Sort by average improvement
    pivot["avg"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("avg", ascending=False).drop("avg", axis=1)
    
    # Reorder columns: adverse first, then normal
    col_order = [c for c in ADVERSE_DOMAINS if c in pivot.columns] + \
                [c for c in NORMAL_DOMAINS if c in pivot.columns]
    pivot = pivot[[c for c in col_order if c in pivot.columns]]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create heatmap
    sns.heatmap(
        pivot.head(20),  # Top 20 strategies
        annot=True, 
        fmt=".2f", 
        cmap="RdYlGn",
        center=0,
        vmin=-3,
        vmax=3,
        ax=ax,
        cbar_kws={"label": "mIoU Improvement over Baseline"}
    )
    
    ax.set_title("Strategy Performance by Weather Domain (mIoU Improvement)", fontsize=14)
    ax.set_xlabel("Weather Domain", fontsize=12)
    ax.set_ylabel("Strategy", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "domain_heatmap_gen.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {output_dir}/domain_heatmap_gen.png")


def create_domain_comparison_plot(df: pd.DataFrame, output_dir: str):
    """Create bar chart comparing performance in adverse vs normal domains."""
    # Classify domains
    df = df.copy()
    df["domain_type"] = df["domain"].apply(
        lambda x: "adverse" if x in ADVERSE_DOMAINS else "normal"
    )
    
    # Calculate average improvement by strategy and domain type
    agg = df.groupby(["strategy", "strategy_type", "domain_type"])["improvement"].mean().reset_index()
    
    # Pivot for plotting
    pivot = agg.pivot_table(
        values="improvement", 
        index="strategy", 
        columns="domain_type",
        aggfunc="mean"
    ).reset_index()
    
    # Merge with strategy type
    strategy_types = df[["strategy", "strategy_type"]].drop_duplicates()
    pivot = pivot.merge(strategy_types, on="strategy")
    
    # Sort by adverse domain improvement
    pivot = pivot.sort_values("adverse", ascending=False)
    
    # Take top 15 strategies (excluding baseline)
    plot_df = pivot[pivot["strategy"] != "baseline"].head(15)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(plot_df))
    width = 0.35
    
    # Color by strategy type
    colors_adverse = []
    colors_normal = []
    for _, row in plot_df.iterrows():
        if row["strategy_type"] == "gen":
            colors_adverse.append("#2196F3")
            colors_normal.append("#90CAF9")
        elif row["strategy_type"] == "std":
            colors_adverse.append("#4CAF50")
            colors_normal.append("#A5D6A7")
        else:
            colors_adverse.append("#FF9800")
            colors_normal.append("#FFCC80")
    
    bars1 = ax.bar(x - width/2, plot_df["adverse"], width, color=colors_adverse, 
                   label="Adverse Conditions")
    bars2 = ax.bar(x + width/2, plot_df["normal"], width, color=colors_normal,
                   label="Normal Conditions")
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Strategy", fontsize=12)
    ax.set_ylabel("mIoU Improvement over Baseline", fontsize=12)
    ax.set_title("Performance Improvement: Adverse vs Normal Weather Conditions", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["strategy"], rotation=45, ha='right')
    ax.legend()
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "adverse_vs_normal_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {output_dir}/adverse_vs_normal_comparison.png")


def create_per_domain_boxplot(df: pd.DataFrame, output_dir: str):
    """Create boxplot showing distribution of improvements per domain."""
    # Filter to non-baseline strategies
    plot_df = df[df["strategy"] != "baseline"]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Order domains
    domain_order = ADVERSE_DOMAINS + NORMAL_DOMAINS
    domain_order = [d for d in domain_order if d in plot_df["domain"].unique()]
    
    # Create boxplot
    sns.boxplot(
        data=plot_df,
        x="domain",
        y="improvement",
        hue="strategy_type",
        order=domain_order,
        ax=ax
    )
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Weather Domain", fontsize=12)
    ax.set_ylabel("mIoU Improvement over Baseline", fontsize=12)
    ax.set_title("Distribution of Performance Improvements by Weather Domain", fontsize=14)
    ax.legend(title="Strategy Type")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "domain_improvement_boxplot.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {output_dir}/domain_improvement_boxplot.png")


def create_top_strategies_per_domain(df: pd.DataFrame, output_dir: str):
    """Find and visualize top performing strategies per domain."""
    # Group by domain and find top strategies
    results = []
    
    for domain in df["domain"].unique():
        domain_df = df[df["domain"] == domain]
        domain_df = domain_df[domain_df["strategy"] != "baseline"]
        
        # Get average improvement per strategy
        avg_improvement = domain_df.groupby("strategy")["improvement"].mean().sort_values(ascending=False)
        
        for rank, (strategy, improvement) in enumerate(avg_improvement.head(5).items(), 1):
            strategy_type = domain_df[domain_df["strategy"] == strategy]["strategy_type"].iloc[0]
            results.append({
                "domain": domain,
                "rank": rank,
                "strategy": strategy,
                "strategy_type": strategy_type,
                "improvement": improvement
            })
    
    results_df = pd.DataFrame(results)
    
    # Create pivot for display
    pivot = results_df[results_df["rank"] == 1].pivot_table(
        values="improvement",
        index="domain",
        columns="strategy",
        aggfunc="first"
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("TOP STRATEGIES PER WEATHER DOMAIN")
    print("=" * 80)
    
    domain_order = ADVERSE_DOMAINS + NORMAL_DOMAINS
    for domain in domain_order:
        if domain not in results_df["domain"].values:
            continue
        
        domain_results = results_df[results_df["domain"] == domain].sort_values("rank")
        print(f"\n{domain.upper()}:")
        for _, row in domain_results.iterrows():
            print(f"  {row['rank']}. {row['strategy']}: {row['improvement']:+.2f} mIoU ({row['strategy_type']})")
    
    # Save to CSV
    results_df.to_csv(os.path.join(output_dir, "top_strategies_per_domain.csv"), index=False)
    print(f"\nSaved: {output_dir}/top_strategies_per_domain.csv")
    
    return results_df


def create_worst_strategies_per_domain(df: pd.DataFrame, output_dir: str):
    """Find and visualize worst performing strategies per domain."""
    results = []
    
    for domain in df["domain"].unique():
        domain_df = df[df["domain"] == domain]
        domain_df = domain_df[domain_df["strategy"] != "baseline"]
        
        # Get average improvement per strategy
        avg_improvement = domain_df.groupby("strategy")["improvement"].mean().sort_values(ascending=True)
        
        for rank, (strategy, improvement) in enumerate(avg_improvement.head(5).items(), 1):
            strategy_type = domain_df[domain_df["strategy"] == strategy]["strategy_type"].iloc[0]
            results.append({
                "domain": domain,
                "rank": rank,
                "strategy": strategy,
                "strategy_type": strategy_type,
                "improvement": improvement
            })
    
    results_df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "=" * 80)
    print("WORST STRATEGIES PER WEATHER DOMAIN")
    print("=" * 80)
    
    domain_order = ADVERSE_DOMAINS + NORMAL_DOMAINS
    for domain in domain_order:
        if domain not in results_df["domain"].values:
            continue
        
        domain_results = results_df[results_df["domain"] == domain].sort_values("rank")
        print(f"\n{domain.upper()}:")
        for _, row in domain_results.iterrows():
            print(f"  {row['rank']}. {row['strategy']}: {row['improvement']:+.2f} mIoU ({row['strategy_type']})")
    
    return results_df


def create_domain_radar_plot(df: pd.DataFrame, strategies: List[str], output_dir: str):
    """Create radar plot comparing top strategies across domains."""
    from math import pi
    
    # Get domains
    domains = ADVERSE_DOMAINS + NORMAL_DOMAINS
    domains = [d for d in domains if d in df["domain"].unique()]
    
    # Calculate average improvement per strategy and domain
    plot_data = {}
    for strategy in strategies:
        strategy_df = df[df["strategy"] == strategy]
        avg_by_domain = strategy_df.groupby("domain")["improvement"].mean()
        plot_data[strategy] = [avg_by_domain.get(d, 0) for d in domains]
    
    # Create radar plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Number of variables
    N = len(domains)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
    
    for idx, (strategy, values) in enumerate(plot_data.items()):
        values += values[:1]  # Complete the circle
        ax.plot(angles, values, 'o-', linewidth=2, label=strategy, color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(domains, fontsize=10)
    ax.set_title("Strategy Performance Across Weather Domains", fontsize=14, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "domain_radar_plot.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {output_dir}/domain_radar_plot.png")


def generate_summary_statistics(df: pd.DataFrame, output_dir: str):
    """Generate and save summary statistics."""
    # Overall statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # By strategy type
    print("\nAverage Improvement by Strategy Type:")
    type_stats = df.groupby("strategy_type")["improvement"].agg(["mean", "std", "count"])
    print(type_stats.to_string())
    
    # By domain category
    df_copy = df.copy()
    df_copy["domain_category"] = df_copy["domain"].apply(
        lambda x: "adverse" if x in ADVERSE_DOMAINS else "normal"
    )
    
    print("\nAverage Improvement by Domain Category:")
    cat_stats = df_copy.groupby(["strategy_type", "domain_category"])["improvement"].mean().unstack()
    print(cat_stats.to_string())
    
    # Best overall strategies
    print("\nTop 10 Overall Strategies (average across all domains):")
    best_overall = df[df["strategy"] != "baseline"].groupby("strategy")["improvement"].mean().sort_values(ascending=False).head(10)
    for rank, (strategy, improvement) in enumerate(best_overall.items(), 1):
        print(f"  {rank}. {strategy}: {improvement:+.2f} mIoU")
    
    # Save to file
    summary_path = os.path.join(output_dir, "weather_domain_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("WEATHER DOMAIN ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("AVERAGE IMPROVEMENT BY STRATEGY TYPE:\n")
        f.write(type_stats.to_string() + "\n\n")
        
        f.write("AVERAGE IMPROVEMENT BY DOMAIN CATEGORY:\n")
        f.write(cat_stats.to_string() + "\n\n")
        
        f.write("TOP 10 OVERALL STRATEGIES:\n")
        for rank, (strategy, improvement) in enumerate(best_overall.items(), 1):
            f.write(f"  {rank}. {strategy}: {improvement:+.2f} mIoU\n")
    
    print(f"\nSaved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze segmentation performance across weather domains")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1,
                        help="Stage to analyze (1=clear_day training, 2=all domains training)")
    parser.add_argument("--weights-dir", type=str, default=None, help="Override weights directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    # Determine weights directory
    if args.weights_dir:
        weights_dir = args.weights_dir
    elif args.stage == 1:
        weights_dir = WEIGHTS_ROOT_STAGE1
    else:
        weights_dir = WEIGHTS_ROOT_STAGE2
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"./result_figures/weather_analysis_stage{args.stage}"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    stage_name = f"Stage {args.stage}"
    stage_desc = "Clear Day Training" if args.stage == 1 else "All Domains Training"
    
    print("=" * 80)
    print(f"WEATHER DOMAIN ANALYSIS - {stage_name}")
    print(f"Training: {stage_desc}")
    print(f"Weights: {weights_dir}")
    print("=" * 80)
    
    # Load results
    print("\nLoading detailed test results...")
    results = load_detailed_results(weights_dir)
    
    if not results:
        print("ERROR: No detailed results found. Run tests with detailed output first.")
        return
    
    df = pd.DataFrame(results)
    print(f"Loaded {len(df)} results from {df['strategy'].nunique()} strategies")
    print(f"Domains: {df['domain'].unique().tolist()}")
    
    # Add strategy categorization
    categorized = df["strategy"].apply(categorize_strategy)
    df["strategy_type"] = categorized.apply(lambda x: x[0])
    df["gen_component"] = categorized.apply(lambda x: x[1])
    df["std_component"] = categorized.apply(lambda x: x[2])
    
    # Calculate improvements
    print("\nCalculating improvements over baseline...")
    df = analyze_domain_performance(df)
    
    # Save raw data
    df.to_csv(os.path.join(output_dir, "weather_domain_results.csv"), index=False)
    print(f"Saved: {output_dir}/weather_domain_results.csv")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    create_domain_heatmap(df, output_dir)
    create_domain_comparison_plot(df, output_dir)
    create_per_domain_boxplot(df, output_dir)
    
    # Find top/worst strategies per domain
    top_df = create_top_strategies_per_domain(df, output_dir)
    worst_df = create_worst_strategies_per_domain(df, output_dir)
    
    # Radar plot with top 5 strategies
    top_strategies = df[df["strategy"] != "baseline"].groupby("strategy")["improvement"].mean().nlargest(5).index.tolist()
    create_domain_radar_plot(df, top_strategies + ["baseline"], output_dir)
    
    # Generate summary
    generate_summary_statistics(df, output_dir)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
