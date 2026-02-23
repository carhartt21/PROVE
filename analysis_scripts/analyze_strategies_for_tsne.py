#!/usr/bin/env python3
"""Analyze downstream results to identify most relevant strategies for t-SNE visualization."""

import pandas as pd
from pathlib import Path

def main():
    # Load results
    df = pd.read_csv("downstream_results.csv")
    
    # Filter to basic results (exclude per-domain splits like _clear_day)
    df = df[~df["model"].str.contains("_clear_day")]
    df = df[df["result_type"] == "basic"]
    
    # Get unique strategies
    strategies = df["strategy"].unique()
    print(f"Total strategies: {len(strategies)}")
    
    # Calculate average mIoU improvement over baseline
    baseline_avg = df[df["strategy"] == "baseline"]["mIoU"].mean()
    print(f"\nBaseline average mIoU: {baseline_avg:.2f}")
    
    # Calculate improvement for each strategy
    results = []
    for strategy in strategies:
        strategy_df = df[df["strategy"] == strategy]
        avg_miou = strategy_df["mIoU"].mean()
        improvement = avg_miou - baseline_avg
        count = len(strategy_df)
        strategy_type = strategy_df["strategy_type"].iloc[0] if len(strategy_df) > 0 else "unknown"
        results.append({
            "strategy": strategy,
            "type": strategy_type,
            "avg_mIoU": avg_miou,
            "improvement": improvement,
            "count": count
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("avg_mIoU", ascending=False)
    
    print("\n=== Top 20 Strategies by Average mIoU ===")
    header = f"{'Strategy':<40} {'Type':<8} {'Avg mIoU':>10} {'Improvement':>12} {'Samples':>8}"
    print(header)
    print("-" * len(header))
    for _, row in results_df.head(20).iterrows():
        print(f"{row['strategy']:<40} {row['type']:<8} {row['avg_mIoU']:>10.2f} {row['improvement']:>+12.2f} {row['count']:>8}")
    
    print("\n=== Gen Strategies Only (Top 15) ===")
    gen_results = results_df[results_df["type"] == "gen"]
    header = f"{'Strategy':<40} {'Avg mIoU':>10} {'Improvement':>12} {'Samples':>8}"
    print(header)
    print("-" * len(header))
    for _, row in gen_results.head(15).iterrows():
        print(f"{row['strategy']:<40} {row['avg_mIoU']:>10.2f} {row['improvement']:>+12.2f} {row['count']:>8}")
    
    print("\n=== Std Strategies Only ===")
    std_results = results_df[results_df["type"] == "std"]
    header = f"{'Strategy':<40} {'Avg mIoU':>10} {'Improvement':>12} {'Samples':>8}"
    print(header)
    print("-" * len(header))
    for _, row in std_results.iterrows():
        print(f"{row['strategy']:<40} {row['avg_mIoU']:>10.2f} {row['improvement']:>+12.2f} {row['count']:>8}")
    
    # Identify recommended strategies for t-SNE
    print("\n" + "=" * 70)
    print("RECOMMENDED STRATEGIES FOR t-SNE DOMAIN GAP VISUALIZATION")
    print("=" * 70)
    
    # Top 3 gen strategies
    top_gen = gen_results.head(3)["strategy"].tolist()
    print(f"\nTop 3 Gen Strategies: {top_gen}")
    
    # Top 2 std strategies
    top_std = std_results.head(2)["strategy"].tolist()
    print(f"Top 2 Std Strategies: {top_std}")
    
    # Bottom gen strategy (worst performing)
    bottom_gen = gen_results.tail(1)["strategy"].tolist()
    print(f"Bottom Gen Strategy: {bottom_gen}")
    
    print("\nRecommended strategies for t-SNE:")
    recommended = ["baseline"] + top_gen + top_std + bottom_gen
    for i, s in enumerate(recommended, 1):
        print(f"  {i}. {s}")
    
    return recommended

if __name__ == "__main__":
    main()
