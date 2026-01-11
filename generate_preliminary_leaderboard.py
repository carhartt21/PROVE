#!/usr/bin/env python3
"""
Generate a preliminary strategy leaderboard from valid test results.

Uses only tests with mIoU > 10% (properly evaluated results).
"""

import pandas as pd
from pathlib import Path
from tabulate import tabulate

def main():
    df = pd.read_csv('/home/mima2416/repositories/PROVE/downstream_results.csv')
    
    # Filter to good tests only (mIoU > 10%)
    good_tests = df[df['mIoU'] > 10.0].copy()
    
    print("=" * 80)
    print("PRELIMINARY STRATEGY LEADERBOARD")
    print("Based on valid test results (mIoU > 10%)")
    print("=" * 80)
    
    print(f"\nTotal valid tests: {len(good_tests)}")
    print(f"Total strategies: {good_tests['strategy'].nunique()}")
    print(f"Total datasets: {good_tests['dataset'].nunique()}")
    
    # Summary by dataset
    print("\n" + "=" * 80)
    print("VALID TESTS BY DATASET")
    print("=" * 80)
    for ds in sorted(good_tests['dataset'].unique()):
        ds_tests = good_tests[good_tests['dataset'] == ds]
        strategies = ds_tests['strategy'].nunique()
        avg_miou = ds_tests['mIoU'].mean()
        print(f"  {ds:25s}: {len(ds_tests):3d} tests, {strategies:2d} strategies, avg mIoU: {avg_miou:.2f}%")
    
    # Overall leaderboard (aggregate across all datasets)
    print("\n" + "=" * 80)
    print("OVERALL STRATEGY RANKING (all valid datasets)")
    print("=" * 80)
    
    strategy_stats = good_tests.groupby('strategy').agg({
        'mIoU': ['mean', 'std', 'count'],
        'mAcc': 'mean',
        'aAcc': 'mean',
    }).round(2)
    strategy_stats.columns = ['mIoU_mean', 'mIoU_std', 'num_tests', 'mAcc', 'aAcc']
    strategy_stats = strategy_stats.sort_values('mIoU_mean', ascending=False)
    
    print(tabulate(
        strategy_stats.reset_index()[['strategy', 'mIoU_mean', 'mIoU_std', 'num_tests', 'mAcc', 'aAcc']],
        headers=['Strategy', 'mIoU (%)', 'Std', 'N', 'mAcc (%)', 'aAcc (%)'],
        tablefmt='simple',
        floatfmt='.2f'
    ))
    
    # BDD10K leaderboard (most complete dataset)
    print("\n" + "=" * 80)
    print("BDD10K LEADERBOARD (most complete dataset)")
    print("=" * 80)
    
    bdd10k = good_tests[good_tests['dataset'] == 'bdd10k']
    if len(bdd10k) > 0:
        bdd10k_stats = bdd10k.groupby('strategy').agg({
            'mIoU': ['mean', 'count'],
            'mAcc': 'mean',
        }).round(2)
        bdd10k_stats.columns = ['mIoU', 'num_tests', 'mAcc']
        bdd10k_stats = bdd10k_stats.sort_values('mIoU', ascending=False)
        
        # Add gain over baseline
        baseline_miou = bdd10k_stats.loc['baseline', 'mIoU'] if 'baseline' in bdd10k_stats.index else 0
        bdd10k_stats['gain'] = bdd10k_stats['mIoU'] - baseline_miou
        
        print(f"Baseline mIoU: {baseline_miou:.2f}%\n")
        print(tabulate(
            bdd10k_stats.reset_index()[['strategy', 'mIoU', 'gain', 'num_tests', 'mAcc']],
            headers=['Strategy', 'mIoU (%)', 'Gain', 'N', 'mAcc (%)'],
            tablefmt='simple',
            floatfmt='.2f'
        ))
    
    # Per-model performance on BDD10K
    print("\n" + "=" * 80)
    print("BDD10K PER-MODEL PERFORMANCE")
    print("=" * 80)
    
    if len(bdd10k) > 0:
        model_pivot = bdd10k.pivot_table(
            values='mIoU',
            index='strategy',
            columns='model',
            aggfunc='mean'
        ).round(2)
        
        if not model_pivot.empty:
            print(tabulate(
                model_pivot.reset_index(),
                headers=['Strategy'] + list(model_pivot.columns),
                tablefmt='simple',
                floatfmt='.2f'
            ))
    
    # Top strategies by category
    print("\n" + "=" * 80)
    print("TOP STRATEGIES BY CATEGORY")
    print("=" * 80)
    
    categories = {
        'Generative (diffusion)': ['gen_EDICT', 'gen_IP2P', 'gen_cyclediffusion', 'gen_UniControl', 
                                   'gen_VisualCloze', 'gen_Img2Img', 'gen_Attribute_Hallucination',
                                   'gen_flux_kontext', 'gen_step1x_new', 'gen_step1x_v1p2',
                                   'gen_Qwen_Image_Edit', 'gen_CNetSeg'],
        'Generative (GAN)': ['gen_CUT', 'gen_cycleGAN', 'gen_stargan_v2', 'gen_TSIT',
                            'gen_SUSTechGAN', 'gen_LANIT', 'gen_Weather_Effect_Generator'],
        'Classical augmentation': ['gen_augmenters', 'gen_automold', 'gen_albumentations_weather',
                                   'photometric_distort'],
        'Standard augmentation': ['std_autoaugment', 'std_randaugment', 'std_cutmix', 'std_mixup'],
        'Baseline': ['baseline'],
    }
    
    for cat_name, strategies in categories.items():
        cat_tests = good_tests[good_tests['strategy'].isin(strategies)]
        if len(cat_tests) > 0:
            top = cat_tests.groupby('strategy')['mIoU'].mean().sort_values(ascending=False).head(3)
            print(f"\n{cat_name}:")
            for strat, miou in top.items():
                print(f"  {strat:35s}: {miou:.2f}%")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if 'baseline' in bdd10k_stats.index and len(bdd10k_stats) > 1:
        baseline_miou = bdd10k_stats.loc['baseline', 'mIoU']
        better_than_baseline = bdd10k_stats[bdd10k_stats['mIoU'] > baseline_miou]
        
        print(f"\nBaseline mIoU on BDD10K: {baseline_miou:.2f}%")
        print(f"Strategies better than baseline: {len(better_than_baseline)} / {len(bdd10k_stats)}")
        
        if len(better_than_baseline) > 0:
            best = bdd10k_stats['mIoU'].idxmax()
            best_miou = bdd10k_stats.loc[best, 'mIoU']
            print(f"Best strategy: {best} ({best_miou:.2f}%, +{best_miou - baseline_miou:.2f}%)")
    
    print("\n" + "=" * 80)
    print("NOTE: 170 tests need to be re-run after label processing fix")
    print("Run: scripts/retest_jobs/submit_all_retests.sh on a SLURM login node")
    print("=" * 80)


if __name__ == '__main__':
    main()
