#!/usr/bin/env python3
"""
Generate a preliminary strategy leaderboard from existing valid results.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    df = pd.read_csv('/home/mima2416/repositories/PROVE/downstream_results.csv')

    print("=" * 80)
    print("PRELIMINARY STRATEGY LEADERBOARD (181 valid results)")
    print("=" * 80)

    # Group by strategy and calculate mean mIoU
    strategy_stats = df.groupby('strategy').agg({
        'mIoU': ['mean', 'std', 'count']
    }).round(2)
    strategy_stats.columns = ['Mean_mIoU', 'Std', 'Count']
    strategy_stats = strategy_stats.sort_values('Mean_mIoU', ascending=False)

    print("\n## Overall Strategy Performance (mIoU %)")
    print("-" * 60)
    print(f"{'Strategy':<35} {'Mean mIoU':>10} {'Std':>8} {'Count':>8}")
    print("-" * 60)
    for strategy, row in strategy_stats.iterrows():
        print(f"{strategy:<35} {row['Mean_mIoU']:>10.2f} {row['Std']:>8.2f} {int(row['Count']):>8}")
    print("-" * 60)

    print("\n\n## Per-Dataset Breakdown")
    for dataset in ['bdd10k_cd', 'outside15k_cd', 'idd-aw_cd', 'bdd10k']:
        df_ds = df[df['dataset'] == dataset]
        if len(df_ds) == 0:
            continue
        print(f"\n### {dataset} ({len(df_ds)} results)")
        print("-" * 50)
        ds_stats = df_ds.groupby('strategy').agg({'mIoU': 'mean'}).sort_values('mIoU', ascending=False)
        for i, (strat, row) in enumerate(ds_stats.iterrows()):
            if i < 10:  # Top 10 only
                print(f"  {i+1}. {strat:<30} {row['mIoU']:.2f}%")

    # Save to markdown
    output_path = Path('/home/mima2416/repositories/PROVE/result_figures/preliminary_leaderboard.md')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("# PRELIMINARY STRATEGY LEADERBOARD\n\n")
        f.write(f"**Total Valid Results: {len(df)}**\n\n")
        f.write("## Overall Strategy Performance (mIoU %)\n\n")
        f.write("| Strategy | Mean mIoU | Std | Count |\n")
        f.write("|----------|-----------|-----|-------|\n")
        for strategy, row in strategy_stats.iterrows():
            f.write(f"| {strategy} | {row['Mean_mIoU']:.2f} | {row['Std']:.2f} | {int(row['Count'])} |\n")
        
        f.write("\n## Notes\n\n")
        f.write("- 115 retest jobs submitted (will update these results)\n")
        f.write("- 81 MapillaryVistas retraining jobs submitted\n")
        f.write("- MapillaryVistas models not included (were trained incorrectly)\n")
    
    print(f"\n\nLeaderboard saved to: {output_path}")
    print("\n## Notes:")
    print("- 115 retest jobs submitted (will update these results)")
    print("- 81 MapillaryVistas retraining jobs submitted")
    print("- MapillaryVistas models not included (were trained incorrectly)")


if __name__ == '__main__':
    main()
